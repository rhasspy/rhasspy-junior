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
from dataclasses import dataclass, field


@dataclass
class IntentEntity:
    """Entity from intent recognition"""

    name: str
    """Name of entity"""

    value: typing.Any
    """Value of entitiy"""


@dataclass
class IntentRequest:
    """Request for intent recognition"""

    text: str
    """Text to recognize intent from"""


@dataclass
class IntentResult:
    """Result of intent recognition"""

    intent_name: str
    """Name of recognized intent"""

    entities: typing.List[IntentEntity] = field(default_factory=list)
    """Recognized named entities"""


class IntentRecognizer(ABC):
    """Base class for intent recognizers"""

    def __init__(self, config: typing.Dict[str, typing.Any]):
        pass

    @abstractmethod
    def recognize(self, request: IntentRequest) -> typing.Optional[IntentResult]:
        """Recognize an intent"""

    def start(self):
        """Initialize recognizer"""

    def stop(self):
        """Uninitialize recognizer"""
