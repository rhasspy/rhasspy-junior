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


class ConfigurableComponent(ABC):
    """Base class for all voice loop components"""

    def __init__(
        self,
        root_config: typing.Dict[str, typing.Any],
        config_extra_path: typing.Optional[str] = None,
    ):
        self.root_config = root_config

        # "x.y.z" -> ["x", "y", "z"]
        config_path_parts = self.config_path().split(".")
        if config_extra_path:
            config_path_parts.extend(config_extra_path.split("."))

        # Locate config section from root
        self.config = self.root_config
        for path_part in config_path_parts:
            self.config = self.config[path_part]

    @classmethod
    @abstractmethod
    def config_path(cls) -> str:
        """Dotted path in config object where "x.y" means config["x"]["y"]"""
