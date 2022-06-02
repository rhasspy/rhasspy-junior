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

from rhasspy_junior.utils import load_class

from .const import IntentHandler, IntentHandleRequest, IntentHandleResult


class MultiIntentHandler(IntentHandler):
    """Run multiple intent handlers in series until an intent is handled"""

    def __init__(
        self,
        root_config: typing.Dict[str, typing.Any],
        config_extra_path: typing.Optional[str] = None,
    ):
        super().__init__(root_config, config_extra_path=config_extra_path)

    def run(self, request: IntentHandleRequest) -> IntentHandleResult:
        """Run trainer"""
        handler_types = self.config["types"]
        for handler_type in handler_types:

            # Path to Python class
            # Maybe be <type> or <type>#<path> where <path> is appended to the config path
            config_extra_path: typing.Optional[str] = None
            if "#" in handler_type:
                handler_type, config_extra_path = handler_type.split("#", maxsplit=1)

            handler_class = load_class(handler_type)
            handler = typing.cast(
                IntentHandler,
                handler_class(self.root_config, config_extra_path=config_extra_path),
            )
            result = handler.run(request)

            if result.handled:
                break

        return result
