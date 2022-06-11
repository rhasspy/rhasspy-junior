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
import logging
import typing

from rhasspy_junior.config import load_configs
from rhasspy_junior.const import DEFAULT_CONFIG_PATH
from rhasspy_junior.train import Trainer, TrainingContext
from rhasspy_junior.utils import load_class

from .args import get_args

_LOGGER = logging.getLogger(__package__)


def main():
    """Main entry point"""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Load default config first
    args.config.insert(0, DEFAULT_CONFIG_PATH)

    config = load_configs(
        args.config,
        system_data_dir=args.system_data_dir,
        user_data_dir=args.user_data_dir,
        user_train_dir=args.user_train_dir,
    )
    _LOGGER.debug(config)

    train_config = config["train"]
    train_class = load_class(train_config["type"])

    _LOGGER.debug("Loading trainer (%s)", train_class)
    train = typing.cast(Trainer, train_class(config))
    _LOGGER.info("Trainer loaded (%s)", train_class)

    context = TrainingContext()
    train.run(context)


if __name__ == "__main__":
    main()
