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
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VoiceActivityResult:
    is_speech: bool
    is_end_of_command: bool = False


class VoiceActivityDetector(ABC):
    """Base class for voice activity detection (VAD)"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        pass

    @abstractmethod
    def begin_command(self):
        """Start new voice command"""

    @abstractmethod
    def process_chunk(self, chunk: bytes) -> VoiceActivityResult:
        """Process audio chunk"""

    def start(self):
        """Initialize VAD"""

    def stop(self):
        """Uninitialize VAD"""
