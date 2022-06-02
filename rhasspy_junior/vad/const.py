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
from abc import abstractmethod
from dataclasses import dataclass

from rhasspy_junior.const import ConfigurableComponent


@dataclass
class VoiceActivityResult:
    is_speech: bool
    is_end_of_command: bool = False


class VoiceActivityDetector(ConfigurableComponent):
    """Base class for voice activity detection (VAD)"""

    def __init__(
        self,
        root_config: typing.Dict[str, typing.Any],
        config_extra_path: typing.Optional[str] = None,
    ):
        super().__init__(root_config, config_extra_path=config_extra_path)

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
