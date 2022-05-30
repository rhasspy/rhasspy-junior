"""Types and constants."""
import typing

from .jsgf import Expression, Rule, Sentence

IntentsType = typing.Dict[str, typing.MutableSequence[typing.Union[Sentence, Rule]]]
SentencesType = typing.Dict[str, typing.MutableSequence[Sentence]]
ReplacementsType = typing.Dict[str, typing.MutableSequence[Expression]]
