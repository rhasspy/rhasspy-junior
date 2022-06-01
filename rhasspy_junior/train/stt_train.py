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
import json
import logging
import platform
import shutil
import sqlite3
import sys
import typing
from collections import defaultdict
from pathlib import Path

import networkx as nx

from fsticuffs_stt.g2p import PronunciationsType
from fsticuffs_stt.train import train

_LOGGER = logging.getLogger(__name__)


def train_model(
    graph_path: typing.Union[str, Path],
    data_dir: typing.Union[str, Path],
    train_dir: typing.Union[str, Path],
    language: str = "en-us",
):
    _LOGGER.debug("Loading graph from %s", graph_path)
    with open(graph_path, "r", encoding="utf-8") as graph_file:
        graph_dict = json.load(graph_file)
        graph = nx.readwrite.json_graph.node_link_graph(graph_dict)

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


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("data_dir")
    parser.add_argument("train_dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    train_model(args.graph, args.data_dir, args.train_dir)


if __name__ == "__main__":
    main()
