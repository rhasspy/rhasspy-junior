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
"""Generates JSON intent graph from .ini files"""

import argparse
import datetime
import io
import json
import logging
import sys
import typing
from pathlib import Path

import lingua_franca

from fsticuffs.ini_jsgf import Expression, Word, parse_ini, split_rules
from fsticuffs.jsgf import walk_expression
from fsticuffs.jsgf_graph import graph_to_json, sentences_to_graph
from fsticuffs.number_utils import number_range_transform, number_transform
from fsticuffs.slots import add_slot_replacements

_LOGGER = logging.getLogger(__name__)


def generate_graph(
    intents_paths: typing.Iterable[typing.Union[str, Path]],
    graph_path: typing.Union[str, Path, typing.TextIO],
    number_language: str = "en",
    casing: str = "lower",
    no_replace_numbers: bool = False,
):
    if isinstance(graph_path, (str, Path)):
        graph_path = Path(graph_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_file = open(graph_path, "w", encoding="utf-8")
    else:
        graph_file = graph_path

    lingua_franca.load_language(number_language)

    # Read sentences
    with io.StringIO() as ini_file:
        for sentences_file_or_dir in intents_paths:
            sentences_file_or_dir = Path(sentences_file_or_dir)

            if sentences_file_or_dir.is_dir():
                sentences_paths = sentences_file_or_dir.rglob("*.ini")
            else:
                sentences_paths = [sentences_file_or_dir]

            for sentences_path in sentences_paths:
                _LOGGER.debug("Reading %s", sentences_path)
                with open(sentences_path, "r", encoding="utf-8") as sentences_file:
                    ini_file.write(sentences_file.read())
                    print("", file=ini_file)

        ini_text = ini_file.getvalue()

    _LOGGER.debug("Parsing sentences")
    intents = parse_ini(ini_text)

    _LOGGER.debug("Processing sentences")
    sentences, replacements = split_rules(intents)

    # Transform words
    word_transform: typing.Optional[typing.Callable[[str], str]] = None

    if casing == "lower":
        word_transform = str.lower
    elif casing == "upper":
        word_transform = str.upper

    word_visitor: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, Expression]]
    ] = None

    if word_transform:
        # Apply transformation to words

        def transform_visitor(word: Expression):
            if isinstance(word, Word):
                assert word_transform
                new_text = word_transform(word.text)

                # Preserve case by using original text as substition
                if (word.substitution is None) and (new_text != word.text):
                    word.substitution = word.text

                word.text = new_text

            return word

        word_visitor = transform_visitor

    # Apply case/number transforms
    if word_visitor or (not no_replace_numbers):
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                if not no_replace_numbers:
                    # Replace number ranges with slot references
                    # type: ignore
                    walk_expression(sentence, number_range_transform, replacements)

                if word_visitor:
                    # Do case transformation
                    # type: ignore
                    walk_expression(sentence, word_visitor, replacements)

    def number_range(*args):
        if len(args) == 3:
            start, stop, step = map(int, args)
        elif len(args) == 2:
            start, stop = map(int, args)
            step = 1
        else:
            raise ValueError(f"Invalid number range args {args}")

        for num in range(start, stop + 1, step):
            yield str(num)

    # Load slot values
    add_slot_replacements(
        replacements,
        intents,
        slot_generators={"$mycroft/number": number_range},
        slot_visitor=word_visitor,
    )

    if not no_replace_numbers:
        # Do single number transformations
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                walk_expression(
                    sentence,
                    number_transform,
                    replacements,
                )

    # Convert to directed graph
    _LOGGER.debug("Converting to graph")
    graph = sentences_to_graph(sentences, replacements=replacements)

    # Write JSON graph
    json.dump(graph_to_json(graph), graph_file)


def json_converter(obj):
    if isinstance(obj, datetime.datetime):
        return str(obj)


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("intents", nargs="+")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    generate_graph(args.intents, sys.stdout)


if __name__ == "__main__":
    main()
