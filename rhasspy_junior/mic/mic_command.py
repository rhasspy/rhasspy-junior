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

import shlex
import subprocess
import typing

from .const import Microphone


class CommandMicrophone(Microphone):
    """Microphone that reads audio from a subprocess"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)

        self.config = config["mic"]["command"]
        self._proc: typing.Optional[subprocess.Popen] = None

        self.command = shlex.split(str(self.config["command"]))
        self.chunk_bytes = int(self.config["chunk_bytes"])
        self.stop_timeout_sec = float(self.config["stop_timeout_sec"])

    def start(self):
        self.stop()

        # pylint: disable=consider-using-with
        self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE)

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.communicate(timeout=self.stop_timeout_sec)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            finally:
                self._proc = None

    def get_chunk(self) -> typing.Optional[bytes]:
        """Get chunk of raw audio data"""
        if (self._proc is not None) and (self._proc.poll() is None):
            assert self._proc.stdout is not None

            return self._proc.stdout.read(self.chunk_bytes)

        return None
