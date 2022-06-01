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

import json
import typing

import networkx as nx

from ..const import IntentRecognizer, IntentEntity, IntentResult, IntentRequest

from .fsticuffs import recognize
from .jsgf_graph import json_to_graph


class FsticuffsIntentRecognizer(IntentRecognizer):
    """Recognize intents using fsticuffs"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)

        self.config = config["intent"]["fsticuffs"]
        self.graph: typing.Optional[nx.DiGraph] = None

    def recognize(self, request: IntentRequest) -> typing.Optional[IntentResult]:
        """Recognize an intent"""
        assert self.graph is not None

        results = recognize(request.text, self.graph)
        if results:
            result = results[0]
            if result.intent is not None:
                return IntentResult(
                    intent_name=result.intent.name,
                    entities=[
                        IntentEntity(name=e.entity, value=e.value)
                        for e in result.entities
                    ],
                )

        return None

    def start(self):
        graph_path = str(self.config["graph"])
        with open(graph_path, "r", encoding="utf-8") as graph_file:
            graph_dict = json.load(graph_file)
            self.graph = json_to_graph(graph_dict)

    def stop(self):
        self.graph = None
