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

from .args import get_args
from .config import load_configs
from .const import DEFAULT_CONFIG_PATH
from .loop import VoiceLoop
from .utils import load_class

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

    config = load_configs(args.config)
    _LOGGER.debug(config)

    loop_config = config["loop"]
    loop_class = load_class(loop_config["type"])

    _LOGGER.debug("Loading voice loop (%s)", loop_class)
    loop = typing.cast(VoiceLoop, loop_class(config))
    _LOGGER.info("Voice loop loaded (%s)", loop_class)

    loop.start()

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()


if __name__ == "__main__":
    main()
