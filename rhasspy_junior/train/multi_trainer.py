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

import typing

from .const import Trainer, TrainingContext
from ..utils import load_class


class MultiTrainer(Trainer):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)
        self._root_config = config
        self.config = config["train"]["multi"]

    def run(self, context: TrainingContext) -> TrainingContext:
        """Run trainer"""
        context = TrainingContext()
        trainer_types = self.config["types"]
        for trainer_type in trainer_types:
            trainer_class = load_class(trainer_type)
            trainer = typing.cast(Trainer, trainer_class(self._root_config))
            context = trainer.run(context)

        return context
