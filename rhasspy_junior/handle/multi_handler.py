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

from ..utils import load_class
from .const import IntentHandler, IntentHandleRequest, IntentHandleResult


class MultiIntentHandler(IntentHandler):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)
        self._root_config = config
        self.config = config["handle"]["multi"]

    def run(self, request: IntentHandleRequest) -> IntentHandleResult:
        """Run trainer"""
        handler_types = self.config["types"]
        for handler_type in handler_types:
            handler_class = load_class(handler_type)
            handler = typing.cast(IntentHandler, handler_class(self._root_config))
            result = handler.run(request)

            if result.handled:
                break

        return result
