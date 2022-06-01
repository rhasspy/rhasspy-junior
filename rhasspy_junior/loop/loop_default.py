# Copyright 2022 Michael Hansen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import logging
import threading
import typing
from enum import Enum, auto
from queue import Queue

from ..handle import IntentHandler, IntentHandleRequest
from ..hotword import Hotword
from ..intent import IntentRecognizer, IntentRequest
from ..mic import Microphone
from ..stt import SpeechToText, SpeechToTextRequest
from ..utils import load_class
from ..vad import VoiceActivityDetector
from .const import VoiceLoop

_LOGGER = logging.getLogger(__package__)


class State(str, Enum):
    """Voice loop state"""

    DETECTING_HOTWORD = auto()
    """Waiting for hotword to be detected"""

    RECORDING_COMMAND = auto()
    """User is speaking voice command"""

    HANDLING_INTENT = auto()
    """Handling intent from voice command"""


class DefaultVoiceLoop(VoiceLoop):
    """Runs a standard hotword -> stt -> intent voice loop"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        self.config = config

        self.mic: typing.Optional[Microphone] = None
        self._mic_thread: typing.Optional[threading.Thread] = None
        self._mic_queue: "typing.Optional[Queue[typing.Optional[bytes]]]" = None

        self.hotword: typing.Optional[Hotword] = None
        self.vad: typing.Optional[VoiceActivityDetector] = None
        self.stt: typing.Optional[SpeechToText] = None
        self.intent: typing.Optional[IntentRecognizer] = None
        self.handle: typing.Optional[IntentHandler] = None

        self._state = State.DETECTING_HOTWORD

    def start(self):
        """Initialize voice loop"""

        self.mic = self.load_microphone()
        self._mic_queue = Queue()
        self._mic_thread = threading.Thread(
            target=self._mic_proc, args=(self.mic, self._mic_queue), daemon=True
        )
        self._mic_thread.start()

        self.hotword = self.load_hotword()
        self.hotword.start()

        self.vad = self.load_vad()
        self.vad.start()

        self.stt = self.load_stt()
        self.stt.start()

        self.intent = self.load_intent()
        self.intent.start()

        self.handle = self.load_handle()
        self.handle.start()

    def stop(self):
        """Uninitialize voice loop"""

        self.mic.stop()
        self._mic_thread = None
        self._mic_queue = None

        self.hotword.stop()
        self.hotword = None

        self.vad.stop()
        self.vad = None

        self.stt.stop()
        self.stt = None

        self.intent.stop()
        self.intent = None

        self.handle.stop()
        self.handle = None

    def run(self):
        """Run voice loop"""
        state = State.DETECTING_HOTWORD

        while True:
            chunk = self._mic_queue.get()
            if not chunk:
                _LOGGER.debug("Empty audio chunk from microphone. Exiting")
                break

            try:
                if state == State.DETECTING_HOTWORD:
                    hotword_result = self.hotword.process_chunk(chunk)
                    if hotword_result.is_detected:
                        _LOGGER.debug("Hotword detected")
                        state = State.RECORDING_COMMAND
                        self.vad.begin_command()
                        self.stt.begin_speech(SpeechToTextRequest())
                elif state == State.RECORDING_COMMAND:
                    self.stt.process_chunk(chunk)
                    vad_result = self.vad.process_chunk(chunk)
                    if vad_result.is_end_of_command:
                        _LOGGER.debug("Recording ended")
                        stt_result = self.stt.end_speech()
                        if stt_result is not None:
                            intent_result = self.intent.recognize(
                                IntentRequest(text=stt_result.text)
                            )
                            if intent_result is not None:
                                # handle
                                state = State.HANDLING_INTENT
                                _LOGGER.debug(intent_result)

                                handle_result = self.handle.run(
                                    IntentHandleRequest(intent_result=intent_result)
                                )
                                _LOGGER.debug(handle_result)
                            else:
                                _LOGGER.warning("No intent recognized")
                        else:
                            _LOGGER.warning("No speech transcribed")

                        self._drain_mic_queue()
                        state = State.DETECTING_HOTWORD
            except Exception:
                _LOGGER.exception("Unexpected error while handling audio chunk")

                # Clean up and go back to listening for hotword
                self.stt.end_speech()
                self._drain_mic_queue()
                state = State.DETECTING_HOTWORD

    def load_microphone(self) -> Microphone:
        mic_config = self.config["mic"]
        mic_class = load_class(mic_config["type"])

        _LOGGER.debug("Loading microphone (%s)", mic_class)
        mic = typing.cast(Microphone, mic_class(self.config))
        _LOGGER.info("Microphone loaded (%s)", mic_class)

        return mic

    def load_hotword(self) -> Hotword:
        hotword_config = self.config["hotword"]
        hotword_class = load_class(hotword_config["type"])

        _LOGGER.debug("Loading hotword (%s)", hotword_class)
        hotword = typing.cast(Hotword, hotword_class(self.config))
        _LOGGER.info("Hotword loaded (%s)", hotword_class)

        return hotword

    def load_vad(self) -> VoiceActivityDetector:
        vad_config = self.config["vad"]
        vad_class = load_class(vad_config["type"])

        _LOGGER.debug("Loading voice activity detector (%s)", vad_class)
        vad = typing.cast(VoiceActivityDetector, vad_class(self.config))
        _LOGGER.info("Voice activity detector loaded (%s)", vad_class)

        return vad

    def load_stt(self) -> SpeechToText:
        stt_config = self.config["stt"]
        stt_class = load_class(stt_config["type"])

        _LOGGER.debug("Loading speech to text (%s)", stt_class)
        stt = typing.cast(SpeechToText, stt_class(self.config))
        _LOGGER.info("Speech to text loaded (%s)", stt_class)

        return stt

    def load_intent(self) -> IntentRecognizer:
        intent_config = self.config["intent"]
        intent_class = load_class(intent_config["type"])

        _LOGGER.debug("Loading intent recognizer (%s)", intent_class)
        intent = typing.cast(IntentRecognizer, intent_class(self.config))
        _LOGGER.info("Intent recognizer loaded (%s)", intent_class)

        return intent

    def load_handle(self) -> IntentHandler:
        handle_config = self.config["handle"]
        handle_class = load_class(handle_config["type"])

        _LOGGER.debug("Loading intent handler (%s)", handle_class)
        handle = typing.cast(IntentHandler, handle_class(self.config))
        _LOGGER.info("Intent handler loaded (%s)", handle_class)

        return handle

    # -------------------------------------------------------------------------

    def _mic_proc(self, mic: Microphone, mic_queue: "Queue[typing.Optional[bytes]]"):
        try:
            max_mic_queue_chunks = self.config["mic"]["max_queue_chunks"]
            mic.start()

            while True:
                chunk = mic.get_chunk()

                # Drop extra chunks
                while mic_queue.qsize() >= max_mic_queue_chunks:
                    mic_queue.get()

                mic_queue.put_nowait(chunk)

                if not chunk:
                    break
        except Exception:
            _LOGGER.exception("Unexpected error in microphone thread")

    def _drain_mic_queue(self):
        while not self._mic_queue.empty():
            self._mic_queue.get()
