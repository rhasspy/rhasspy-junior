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

import typing

from rhasspy_junior.vad.const import (
    VoiceActivityDetector,
    VoiceActivityResult,
    VoiceCommandState,
)

from .silence import SilenceDetector, SilenceResultType


class SileroVoiceActivityDetector(VoiceActivityDetector):
    """Voice activity detection using Silero VAD"""

    def __init__(
        self,
        root_config: typing.Dict[str, typing.Any],
        config_extra_path: typing.Optional[str] = None,
    ):
        super().__init__(root_config, config_extra_path=config_extra_path)
        self.detector: typing.Optional[SilenceDetector] = None

        self._command_state: VoiceCommandState = VoiceCommandState.NOT_STARTED

    @classmethod
    def config_path(cls) -> str:
        return "vad.silero"

    def begin_command(self):
        assert self.detector is not None
        self._command_state = VoiceCommandState.NOT_STARTED
        self.detector.start()

    def process_chunk(self, chunk: bytes) -> VoiceActivityResult:
        """Process audio chunk"""
        assert self.detector is not None
        result = self.detector.process(chunk)
        is_speech = False

        if result.type == SilenceResultType.PHRASE_START:
            is_speech = True
            self._command_state = VoiceCommandState.STARTED
        elif result.type == SilenceResultType.PHRASE_END:
            self._command_state = VoiceCommandState.ENDED
        elif result.type == SilenceResultType.TIMEOUT:
            self._command_state = VoiceCommandState.TIMEOUT
        else:
            is_speech = result.type == SilenceResultType.SPEECH

        return VoiceActivityResult(
            is_speech=is_speech, command_state=self._command_state
        )

    def start(self):
        """Initialize VAD"""
        self.detector = SilenceDetector(
            vad_model=str(self.config["model"]),
            vad_threshold=float(self.config["threshold"]),
            sample_rate=int(self.config["sample_rate"]),
            chunk_size=int(self.config["chunk_size"]),
            skip_seconds=float(self.config["skip_seconds"]),
            min_seconds=float(self.config["min_seconds"]),
            max_seconds=float(self.config["max_seconds"]),
            speech_seconds=float(self.config["speech_seconds"]),
            silence_seconds=float(self.config["silence_seconds"]),
        )

    def stop(self):
        """Uninitialize VAD"""
        self.detector = None
