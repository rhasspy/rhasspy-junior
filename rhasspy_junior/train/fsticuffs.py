#!/usr/bin/env python3
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

import configparser
import hashlib
import io
import json
import logging
import platform
import shutil
import sqlite3
import typing
from collections import defaultdict
from pathlib import Path

import lingua_franca

from rhasspy_junior.intent.intent_fsticuffs.ini_jsgf import (
    Expression,
    Word,
    parse_ini,
    split_rules,
)
from rhasspy_junior.intent.intent_fsticuffs.jsgf import walk_expression
from rhasspy_junior.intent.intent_fsticuffs.jsgf_graph import (
    graph_to_json,
    json_to_graph,
    sentences_to_graph,
)
from rhasspy_junior.intent.intent_fsticuffs.number_utils import (
    number_range_transform,
    number_transform,
)
from rhasspy_junior.intent.intent_fsticuffs.slots import add_slot_replacements
from rhasspy_junior.stt.stt_fsticuffs.g2p import PronunciationsType
from rhasspy_junior.stt.stt_fsticuffs.train import train
from rhasspy_junior.train.const import Trainer, TrainingContext

_LOGGER = logging.getLogger(__package__)


class FsticuffsTrainer(Trainer):
    """Train speech to text and intent recognizer with fsticuffs"""

    @classmethod
    def config_path(cls) -> str:
        return "train.fsticuffs"

    def run(self, context: TrainingContext) -> TrainingContext:
        """Run trainer"""
        if self.train_intent():
            self.train_stt()

        return context

    def train_intent(self) -> bool:
        casing = str(self.config["casing"])
        number_language = str(self.config["number_language"])
        replace_numbers = bool(self.config["replace_numbers"])
        force_retrain = bool(self.config["force_retrain"])

        input_files = [Path(v) for v in self.config["input_files"]]

        output_skip_hash_path = Path(str(self.config["output_skip_hash"]))
        actual_hash = self.compute_ini_hash(input_files)

        if (not force_retrain) and output_skip_hash_path.is_file():
            # Check if training can be skipped
            expected_hash = output_skip_hash_path.read_text(encoding="utf-8").strip()

            if expected_hash == actual_hash:
                _LOGGER.info(
                    "Training can be skipped due to matching hash in %s",
                    output_skip_hash_path,
                )
                return False

            _LOGGER.debug(
                "Training hashes do not match. Got '%s', expected '%s'",
                actual_hash,
                expected_hash,
            )

        output_skip_hash_path.parent.mkdir(parents=True, exist_ok=True)

        output_graph_path = Path(str(self.config["output_graph"]))
        output_graph_path.parent.mkdir(parents=True, exist_ok=True)

        lingua_franca.load_language(number_language)

        # Read sentences
        with io.StringIO() as ini_file:
            for sentences_file_or_dir in input_files:
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
        if word_visitor or replace_numbers:
            for intent_sentences in sentences.values():
                for sentence in intent_sentences:
                    if replace_numbers:
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

        if replace_numbers:
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
        with open(output_graph_path, "w", encoding="utf-8") as graph_file:
            json.dump(graph_to_json(graph), graph_file)

        # Write hash so training can be skipped next time if there are no changes
        output_skip_hash_path.write_text(actual_hash, encoding="utf-8")

        return True

    def train_stt(self) -> bool:
        language = str(self.config["language"])
        data_dir = Path(str(self.config["data_dir"]))
        train_dir = Path(str(self.config["train_dir"]))
        train_dir.mkdir(parents=True, exist_ok=True)

        graph_path = Path(str(self.config["output_graph"]))

        with open(graph_path, "r", encoding="utf-8") as graph_file:
            graph_dict = json.load(graph_file)
            graph = json_to_graph(graph_dict)

        data_dir = Path(data_dir).absolute()
        train_dir = Path(train_dir).absolute()

        # Load pronunciations
        pronunciations: PronunciationsType = defaultdict(list)
        lexicon_db_path = data_dir / "en-us" / "lexicon.db"

        _LOGGER.debug("Loading pronunciations from %s", lexicon_db_path)
        with sqlite3.connect(lexicon_db_path) as lexicon_db:
            cursor = lexicon_db.execute(
                "SELECT word, phonemes from word_phonemes ORDER BY word, pron_order",
            )

            for row in cursor:
                word, phonemes = row[0], row[1].split()
                pronunciations[word].append(phonemes)

        # Copy Kaldi model
        model_dir = data_dir / language / "model"
        model_dir = model_dir.absolute()

        _LOGGER.debug("Copying Kaldi model from %s to %s", model_dir, train_dir)
        if train_dir.exists():
            shutil.rmtree(train_dir)

        shutil.copytree(model_dir, train_dir)

        # Train
        _LOGGER.debug("Training")

        kaldi_dir = data_dir / "kaldi" / platform.machine()
        kaldi_dir = kaldi_dir.absolute()

        g2p_path = data_dir / language / "g2p.fst"
        graph_dir = train_dir / "graph"
        train(
            graph=graph,
            pronunciations=pronunciations,
            model_dir=train_dir,
            graph_dir=graph_dir,
            kaldi_dir=kaldi_dir,
            g2p_model=g2p_path,
            g2p_word_transform=str.lower,
        )

        return True

    def compute_ini_hash(self, input_files: typing.Iterable[Path]) -> str:

        # Write combined ini file
        with io.StringIO() as ini_file:
            for input_path in input_files:
                with open(input_path, "r", encoding="utf-8") as input_file:
                    for line in input_file:
                        line = line.strip()
                        print(line, file=ini_file)

            # Load ini file
            config = configparser.ConfigParser(
                allow_no_value=True, strict=False, delimiters=["="]
            )

            # case sensitive
            config.optionxform = str  # type: ignore
            config.read_string(ini_file.getvalue())

            # Compute hash
            ini_hash = hashlib.sha256()
            for sorted_section in sorted(config.sections()):
                for sorted_key, sorted_value in sorted(config[sorted_section].items()):
                    sorted_value = sorted_value or ""
                    sorted_line = f"{sorted_key} = ${sorted_value}"
                    ini_hash.update(sorted_line.encode())

            actual_hash = ini_hash.hexdigest()

        return actual_hash
