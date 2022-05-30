"""Utility methods for fsticuffs"""
import dataclasses
import itertools
import typing


def pairwise(iterable: typing.Iterable[typing.Any]):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    return zip(a, itertools.islice(b, 1, None))


def only_fields(
    cls, message_dict: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
    """Return dict with only valid fields."""
    if dataclasses.is_dataclass(cls):
        field_names = set(f.name for f in dataclasses.fields(cls))
        valid_fields = set(message_dict.keys()).intersection(field_names)
        return {key: message_dict[key] for key in valid_fields}

    return message_dict
