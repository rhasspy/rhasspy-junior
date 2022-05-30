"""
Data structures for intent recognition.
"""
import dataclasses
import datetime
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Number

from . import utils


@dataclass
class Entity:
    """Named entity from intent."""

    entity: str
    value: typing.Any
    raw_value: str = ""
    source: str = ""
    start: int = 0
    raw_start: int = 0
    end: int = 0
    raw_end: int = 0
    tokens: typing.List[typing.Any] = field(default_factory=list)
    raw_tokens: typing.List[str] = field(default_factory=list)

    @property
    def value_dict(self):
        """Get dictionary representation of value."""
        if isinstance(self.value, Mapping):
            return self.value

        kind = "Unknown"

        if isinstance(self.value, Number):
            kind = "Number"
        elif isinstance(self.value, datetime.date):
            kind = "Date"
        elif isinstance(self.value, datetime.time):
            kind = "Time"
        elif isinstance(self.value, datetime.datetime):
            kind = "Datetime"
        elif isinstance(self.value, datetime.timedelta):
            kind = "Duration"

        return {"kind": kind, "value": self.value}

    @classmethod
    def from_dict(cls, entity_dict: typing.Dict[str, typing.Any]) -> "Entity":
        """Create Entity from dictionary."""
        return Entity(**utils.only_fields(cls, entity_dict))


@dataclass
class Intent:
    """Named intention with entities and slots."""

    name: str
    confidence: float = 0

    @classmethod
    def from_dict(cls, intent_dict: typing.Dict[str, typing.Any]) -> "Intent":
        """Create Intent from dictionary."""
        return Intent(**utils.only_fields(cls, intent_dict))


@dataclass
class TagInfo:
    """Information used to process FST tags."""

    tag: str
    start_index: int = 0
    raw_start_index: int = 0
    symbols: typing.List[str] = field(default_factory=list)
    raw_symbols: typing.List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, tag_dict: typing.Dict[str, typing.Any]) -> "TagInfo":
        """Create TagInfo from dictionary."""
        return TagInfo(**utils.only_fields(cls, tag_dict))


class RecognitionResult(str, Enum):
    """Result of a recognition."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Recognition:
    """Output of intent recognition."""

    intent: typing.Optional[Intent] = None
    entities: typing.List[Entity] = field(default_factory=list)
    text: str = ""
    raw_text: str = ""
    recognize_seconds: float = 0
    tokens: typing.List[typing.Any] = field(default_factory=list)
    raw_tokens: typing.List[str] = field(default_factory=list)

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def empty(cls) -> "Recognition":
        """Return an empty recognition."""
        return Recognition(intent=Intent(name=""))

    @classmethod
    def from_dict(cls, recognition_dict: typing.Dict[str, typing.Any]) -> "Recognition":
        """Create Recognition from dictionary."""

        intent_dict = recognition_dict.pop("intent", None)
        entity_dicts = recognition_dict.pop("entities", None)
        slots_dict = recognition_dict.pop("slots", None)
        recognition = Recognition(**utils.only_fields(cls, recognition_dict))

        if intent_dict:
            recognition.intent = Intent.from_dict(intent_dict)

        if entity_dicts:
            recognition.entities = [Entity.from_dict(e) for e in entity_dicts]

        if slots_dict:
            recognition.entities = [
                Entity(entity=key, value=value) for key, value in slots_dict.items()
            ]

        return recognition
