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
import platform
import typing
from pathlib import Path

import toml
from jinja2 import Environment, FileSystemLoader

_LOGGER = logging.getLogger(__package__)


def load_configs(
    config_paths: typing.Iterable[typing.Union[str, Path]],
    system_data_dir: typing.Union[str, Path],
    user_data_dir: typing.Union[str, Path],
    user_train_dir: typing.Union[str, Path],
) -> typing.Dict[str, typing.Any]:
    config: typing.Dict[str, typing.Any] = {}
    system_data_dir = Path(system_data_dir).absolute()
    user_data_dir = Path(user_data_dir).absolute()
    user_train_dir = Path(user_train_dir).absolute()

    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.is_file():
            _LOGGER.warning("Skipping missing config %s", config_path)
            continue

        _LOGGER.debug("Loading config %s", config_path)

        # Pre-process with jinja2
        template_env = Environment(
            loader=FileSystemLoader(config_path.parent),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = template_env.get_template(config_path.name)

        new_config = toml.loads(
            template.render(
                system_data_dir=system_data_dir,
                user_data_dir=user_data_dir,
                user_train_dir=user_train_dir,
                platform_machine=platform.machine(),
            )
        )
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
