# Copyright 2022 Mycroft AI Inc.
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
"""Command-line utility for fsticuffs"""
import argparse
import csv
import json
import logging
import platform
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

from .g2p import PronunciationsType
from .train import train
from .transcribe import KaldiCommandLineTranscriber

_LOGGER = logging.getLogger(__package__)


def main():
    """Main method"""
    main_parser = argparse.ArgumentParser("fsticuffs-stt")
    sub_parsers = main_parser.add_subparsers(dest="command", required=True)

    train_parser = sub_parsers.add_parser("train")
    train_parser.add_argument(
        "--graph", required=True, help="Path to JSON fsticuffs graph"
    )
    train_parser.add_argument(
        "--data-dir", required=True, help="Path to data directory with models/kaldi"
    )
    train_parser.add_argument(
        "--train-dir", required=True, help="Directory to write training artifacts"
    )
    train_parser.add_argument("--language", default="en-us", help="Language model name")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument(
        "--data-dir", required=True, help="Path to data directory with models/kaldi"
    )
    transcribe_parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory to write transcribeing artifacts",
    )
    transcribe_parser.add_argument(
        "--language", default="en-us", help="Language model name"
    )

    for parser in [train_parser, transcribe_parser]:
        parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    args = main_parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.command == "train":
        do_train(args)
    elif args.command == "transcribe":
        do_transcribe(args)


# -----------------------------------------------------------------------------


def do_train(args: argparse.Namespace):

    # Load graph
    args.graph = Path(args.graph)

    _LOGGER.debug("Loading graph from %s", args.graph)
    with open(args.graph, "r", encoding="utf-8") as graph_file:
        graph_dict = json.load(graph_file)
        graph = nx.readwrite.json_graph.node_link_graph(graph_dict)

    args.data_dir = Path(args.data_dir)
    args.train_dir = Path(args.train_dir)

    # Load pronunciations
    pronunciations: PronunciationsType = defaultdict(list)
    lexicon_db_path = args.data_dir / "en-us" / "lexicon.db"

    _LOGGER.debug("Loading pronunciations from %s", lexicon_db_path)
    with sqlite3.connect(lexicon_db_path) as lexicon_db:
        cursor = lexicon_db.execute(
            "SELECT word, phonemes from word_phonemes ORDER BY word, pron_order",
        )

        for row in cursor:
            word, phonemes = row[0], row[1].split()
            pronunciations[word].append(phonemes)

    # Copy Kaldi model
    model_dir = args.data_dir / args.language / "model"
    model_dir = model_dir.absolute()

    _LOGGER.debug("Copying Kaldi model from %s to %s", model_dir, args.train_dir)
    if args.train_dir.exists():
        shutil.rmtree(args.train_dir)

    shutil.copytree(model_dir, args.train_dir)

    # Train
    _LOGGER.debug("Training")

    kaldi_dir = args.data_dir / "kaldi" / platform.machine()
    kaldi_dir = kaldi_dir.absolute()

    g2p_path = args.data_dir / args.language / "g2p.fst"
    graph_dir = args.train_dir / "graph"
    train(
        graph=graph,
        pronunciations=pronunciations,
        model_dir=args.train_dir,
        graph_dir=graph_dir,
        kaldi_dir=kaldi_dir,
        g2p_model=g2p_path,
        g2p_word_transform=str.lower,
    )


# -----------------------------------------------------------------------------


def do_transcribe(args: argparse.Namespace):

    args.data_dir = Path(args.data_dir)
    args.train_dir = Path(args.train_dir)

    kaldi_dir = args.data_dir / "kaldi" / platform.machine()
    kaldi_dir = kaldi_dir.absolute()

    model_dir = args.data_dir / args.language / "model"
    model_dir = model_dir.absolute()

    graph_dir = args.train_dir / "graph"

    transcriber = KaldiCommandLineTranscriber(
        model_dir=model_dir, graph_dir=graph_dir, kaldi_dir=kaldi_dir
    )

    writer = csv.writer(sys.stdout)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        wav_path = Path(line)
        result = transcriber.transcribe_wav_file(wav_path)
        if result:
            text = result.text
        else:
            text = ""

        writer.writerow((wav_path, text))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
