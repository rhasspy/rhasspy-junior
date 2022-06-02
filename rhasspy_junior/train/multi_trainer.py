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

from rhasspy_junior.utils import load_class

from .const import Trainer, TrainingContext


class MultiTrainer(Trainer):
    """Run multiple trainers in series"""

    @classmethod
    def config_path(cls) -> str:
        return "train.multi"

    def run(self, context: TrainingContext) -> TrainingContext:
        """Run trainer"""
        context = TrainingContext()
        trainer_types = self.config["types"]
        for trainer_type in trainer_types:

            # Path to Python class
            # Maybe be <type> or <type>#<path> where <path> is appended to the config path
            config_extra_path: typing.Optional[str] = None
            if "#" in trainer_type:
                trainer_type, config_extra_path = trainer_type.split("#", maxsplit=1)

            trainer_class = load_class(trainer_type)
            trainer = typing.cast(
                Trainer,
                trainer_class(self.root_config, config_extra_path=config_extra_path),
            )
            context = trainer.run(context)

        return context
