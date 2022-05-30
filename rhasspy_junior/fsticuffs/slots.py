"""Slot load/parsing utility methods."""
import logging
import typing

from .const import IntentsType, ReplacementsType
from .jsgf import Expression, Rule, Sentence, Sequence, SlotReference, walk_expression

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


SlotGenerator = typing.Callable[..., typing.Iterable[str]]


# -----------------------------------------------------------------------------


def add_slot_replacements(
    replacements: ReplacementsType,
    sentences: IntentsType,
    slot_generators: typing.Dict[str, SlotGenerator],
    slot_visitor: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, Expression]]
    ] = None,
):
    """Create replacement dictionary for referenced slots."""

    # Gather used slot names
    slot_names: typing.Set[str] = set()
    for intent_name in sentences:
        for item in sentences[intent_name]:
            for slot_name in get_slot_names(item):
                slot_names.add(slot_name)

    # Load slot values
    for slot_key in slot_names:
        if slot_key in replacements:
            # Skip already loaded slot
            continue

        slot_name, slot_args = split_slot_args(slot_key)

        if slot_args is None:
            slot_args = []

        slot_gen = slot_generators.get(slot_name)
        assert slot_gen is not None, f"No generator for slot: {slot_name}"

        # Generate values in place
        slot_values = []
        has_output = False
        for line in slot_gen(*slot_args):
            line = line.strip()
            if line:
                has_output = True
                sentence = Sentence.parse(line)
                if slot_visitor:
                    walk_expression(sentence, slot_visitor)

                slot_values.append(sentence)

        assert has_output, f"No output from {slot_key}"

        # Replace $slot with sentences
        replacements[slot_key] = slot_values


# -----------------------------------------------------------------------------


def get_slot_names(item: typing.Union[Expression, Rule]) -> typing.Iterable[str]:
    """Yield referenced slot names from an expression."""
    if isinstance(item, SlotReference):
        yield f"${item.slot_name}"
    elif isinstance(item, Sequence):
        for sub_item in item.items:
            for slot_name in get_slot_names(sub_item):
                yield slot_name
    elif isinstance(item, Rule):
        for slot_name in get_slot_names(item.rule_body):
            yield slot_name


def split_slot_args(
    slot_name: str,
) -> typing.Tuple[str, typing.Optional[typing.List[str]]]:
    """Split slot name and arguments out (slot,arg1,arg2,...)"""
    # Check for arguments.
    slot_args: typing.Optional[typing.List[str]] = None

    # Slot name retains argument(s).
    if "," in slot_name:
        slot_name, *slot_args = slot_name.split(",")

    return slot_name, slot_args
