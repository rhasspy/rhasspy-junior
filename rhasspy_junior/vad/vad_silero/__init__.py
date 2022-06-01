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

from ..const import VoiceActivityDetector, VoiceActivityResult
from .silence import SilenceDetector, SilenceResultType


class SileroVoiceActivityDetector(VoiceActivityDetector):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)
        self.config = config["vad"]["silero"]
        self.detector: typing.Optional[SilenceDetector] = None

        self._speech = VoiceActivityResult(is_speech=True)
        self._silence = VoiceActivityResult(is_speech=False)

    def begin_command(self):
        assert self.detector is not None
        self.detector.start()

    def process_chunk(self, chunk: bytes) -> VoiceActivityResult:
        """Process audio chunk"""
        assert self.detector is not None
        result = self.detector.process(chunk)

        if result.type == SilenceResultType.PHRASE_END:
            is_speech = True
            is_end_of_command = True
        else:
            is_speech = result.type == SilenceResultType.SPEECH
            is_end_of_command = False

        return VoiceActivityResult(
            is_speech=is_speech, is_end_of_command=is_end_of_command
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
            before_seconds=float(self.config["before_seconds"]),
        )

    def stop(self):
        """Uninitialize VAD"""
        self.detector = None
