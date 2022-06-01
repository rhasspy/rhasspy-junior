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
#

import logging
import platform
import threading
import typing
from queue import Queue
from pathlib import Path

from ..const import SpeechToText, SpeechToTextRequest, SpeechToTextResult

from .transcribe import KaldiCommandLineTranscriber

_LOGGER = logging.getLogger(__package__)


class FsticuffsSpeechToText(SpeechToText):
    """Recognize speech using fsticuffs"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)

        self.config = config["stt"]["fsticuffs"]
        self._running = True
        self._input_queue: "Queue[typing.Optional[SpeechToTextRequest]]" = Queue()
        self._result: typing.Optional[SpeechToTextResult] = None
        self._audio_queue: "Queue[typing.Optional[bytes]]" = Queue()

        self._thread: typing.Optional[threading.Thread] = None
        self._result_event = threading.Event()

    def begin_speech(self, request: SpeechToTextRequest):
        """Start speech to text phrase"""
        self._drain_audio_queue()

        self._result_event.clear()
        self._input_queue.put_nowait(request)

    def process_chunk(self, chunk: bytes):
        """Process audio chunk"""
        self._audio_queue.put_nowait(chunk)

    def end_speech(self) -> typing.Optional[SpeechToTextResult]:
        """End speech to text phrase"""
        self._drain_audio_queue()

        # Signal end of audio
        self._audio_queue.put(None)

        end_speech_timeout_sec = self.config.get("end_speech_timeout_sec")
        self._result_event.wait(timeout=end_speech_timeout_sec)

        return self._result

    def start(self):
        self.stop()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._running = False

            # Drain queues
            self._drain_input_queue()
            self._drain_audio_queue()

            # Signal termination
            self._input_queue.put(None)
            self._audio_queue.put(None)

            self._thread.join()
            self._thread = None

    def _drain_input_queue(self):
        while not self._input_queue.empty():
            self._input_queue.get()

    def _drain_audio_queue(self):
        while not self._audio_queue.empty():
            self._audio_queue.get()

    def _run(self):
        try:
            kaldi_dir = Path(str(self.config["kaldi_dir"])) / platform.machine()
            sample_rate = int(self.config["sample_rate"])
            sample_width = int(self.config["sample_width"])
            channels = int(self.config["channels"])

            transcriber = KaldiCommandLineTranscriber(
                model_dir=str(self.config["model_dir"]),
                graph_dir=str(self.config["graph_dir"]),
                kaldi_dir=kaldi_dir,
            )

            transcriber.start_decode()

            def audio_stream():
                while True:
                    chunk = self._audio_queue.get()
                    if chunk is None:
                        break

                    yield chunk

            try:
                while True:
                    request = self._input_queue.get()

                    if (not self._running) or (request is None):
                        break

                    self._result = None

                    try:
                        _LOGGER.debug("Processing speech to text")
                        transcribe_result = transcriber.transcribe_stream(
                            audio_stream=audio_stream(),
                            sample_rate=sample_rate,
                            sample_width=sample_width,
                            channels=channels,
                        )

                        _LOGGER.debug(transcribe_result)

                        if transcribe_result:
                            self._result = SpeechToTextResult(
                                text=transcribe_result.text
                            )
                    except Exception:
                        _LOGGER.exception(
                            "Unexpected error processing request: %s", request
                        )
                    finally:
                        self._result_event.set()
            finally:
                transcriber.stop()
        except Exception:
            _LOGGER.exception("Unexpected error in speech to text thread")
