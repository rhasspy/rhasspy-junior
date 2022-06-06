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

import collections
import logging
import typing
from pathlib import Path

import toml

_LOGGER = logging.getLogger(__package__)


def load_configs(
    config_paths: typing.Iterable[typing.Union[str, Path]]
) -> typing.Dict[str, typing.Any]:
    config: typing.Dict[str, typing.Any] = {}

    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.is_file():
            _LOGGER.warning("Skipping missing config %s", config_path)
            continue

        _LOGGER.debug("Loading config %s", config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            new_config = toml.load(config_file)
            recursive_update(config, new_config)

    return config


def recursive_update(
    base_dict: typing.Dict[typing.Any, typing.Any],
    new_dict: typing.Mapping[typing.Any, typing.Any],
) -> None:
    """Recursively overwrites values in base dictionary with values from new dictionary"""
    for k, v in new_dict.items():
        if isinstance(v, collections.abc.Mapping) and (k in base_dict):
            recursive_update(base_dict[k], v)
        else:
            base_dict[k] = v
