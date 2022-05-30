"""Number range expansion."""
import logging
import re
import typing

from lingua_franca.format import pronounce_number

from .jsgf import Expression, Sequence, SequenceType, SlotReference, Word

_LOGGER = logging.getLogger(__name__)

# 0..100, -100..100
NUMBER_RANGE_PATTERN = re.compile(r"^(-?[0-9]+)\.\.(-?[0-9]+)(,[0-9]+)?$")
NUMBER_PATTERN = re.compile(r"^(-?[0-9]+)$")

# -----------------------------------------------------------------------------


def number_to_words(number: int) -> typing.List[str]:
    """Convert number to list of words (75 -> seventy five)"""
    number_text = pronounce_number(number)
    return number_text.split()


def replace_numbers(words: typing.Iterable[str]) -> typing.Iterable[str]:
    """Replace numbers with words in a sentence (75 hats -> seventy five hats)"""
    for word in words:
        if NUMBER_PATTERN.match(word):
            n = int(word)
            for number_word in number_to_words(n):
                yield number_word
        else:
            yield word


def number_range_transform(word: Expression, slot_name="mycroft/number"):
    """Automatically transform number ranges to slot reference (e.g., 0..100)"""
    if not isinstance(word, Word):
        # Skip anything besides words
        return

    match = NUMBER_RANGE_PATTERN.match(word.text)

    if not match:
        return

    try:
        lower_bound = int(match.group(1))
        upper_bound = int(match.group(2))
        step = 1

        if len(match.groups()) > 3:
            # Exclude ,
            step = int(match.group(3)[1:])

        # Transform to $mycroft/number
        return SlotReference(
            text=word.text,
            slot_name=f"{slot_name},{lower_bound},{upper_bound},{step}",
            converters=["int"],
        )
    except ValueError:
        # Not a number
        pass
    except Exception:
        _LOGGER.exception("number_range_transform")


def number_transform(word: Expression):
    """Automatically transform numbers to words (e.g., 75)"""
    if not isinstance(word, Word):
        # Skip anything besides words
        return

    visited = getattr(word, "_number_transform_visited", False)
    if visited:
        return

    match = NUMBER_PATTERN.match(word.text)

    if not match:
        return

    try:
        n = int(match.group(1))

        # 75 -> (75 | seventy five):75!int
        number_words = number_to_words(n)

        # Mark word as visited
        setattr(word, "_number_transform_visited", True)

        if len(number_words) == 1:
            # Easy case, single word
            text_word = Word(
                text=number_words[0], substitution=[str(n)], converters=["int"]
            )
            return Sequence(
                text=text_word.text,
                type=SequenceType.ALTERNATIVE,
                substitution=[str(n)],
                converters=["int"],
                items=[text_word],
            )

        # Hard case, split into multiple Words
        number_text = " ".join(number_words)

        return Sequence(
            text=number_text,
            type=SequenceType.ALTERNATIVE,
            substitution=[str(n)],
            converters=["int"],
            items=[
                Sequence(
                    text=number_text,
                    type=SequenceType.GROUP,
                    items=[Word(w) for w in number_words],
                ),
            ],
        )

    except ValueError:
        # Not a number
        pass
    except Exception:
        _LOGGER.exception("number_transform")
