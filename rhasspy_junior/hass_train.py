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
"""Loads entities from Home Assistant API and generates intents"""

import argparse
import logging
import sys
import typing
from pathlib import Path
from dataclasses import dataclass

import requests

_LOGGER = logging.getLogger(__package__)


@dataclass
class Entity:
    id: str
    domain: str
    name: str
    spoken_name: str
    state: typing.Dict[str, typing.Any]


def generate_intents(
    api_url: str,
    token: str,
    intents_path: typing.Union[str, Path, typing.TextIO],
    entities_to_ignore: typing.Optional[typing.Set[str]] = None,
):
    if isinstance(intents_path, (str, Path)):
        intents_path = Path(intents_path)
        intents_path.parent.mkdir(parents=True, exist_ok=True)
        intents_file = open(intents_path, "w", encoding="utf-8")
    else:
        intents_file = intents_path

    if entities_to_ignore is None:
        entities_to_ignore = set()

    if api_url.endswith("/"):
        api_url = api_url[:-1]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    response = requests.get(f"{api_url}/states", headers=headers)
    assert response.ok, response

    states = response.json()

    for entity_state in states:
        entity_id = entity_state.get("entity_id", "")

        if entity_id in entities_to_ignore:
            _LOGGER.debug("Ignoring entity: %s", entity_id)
            continue

        if "." not in entity_id:
            # No domain
            _LOGGER.warning("Skipping entity %s: no domain", entity_id)
            continue

        friendly_name = entity_state.get("attributes", {}).get("friendly_name", "")
        if not friendly_name:
            # No friendly name
            _LOGGER.warning("Skipping entity %s: no friendly name")
            continue

        entity = Entity(
            id=entity_id,
            domain=entity_id.split(".", maxsplit=1)[0],
            name=friendly_name,
            spoken_name=preprocess_name(friendly_name),
            state=entity_state,
        )

        write_entity(entity, intents_file)


# -----------------------------------------------------------------------------


def write_entity(entity: Entity, intents_file: typing.TextIO):
    domain = entity.domain

    if domain == "cover":
        write_cover(entity, intents_file)
    elif domain == "switch":
        write_switch(entity, intents_file)
    elif domain == "light":
        write_light(entity, intents_file)
    elif domain == "lock":
        write_lock(entity, intents_file)
    elif domain == "camera":
        write_camera(entity, intents_file)
    elif domain == "climate":
        write_climate(entity, intents_file)
    elif domain == "fan":
        write_fan(entity, intents_file)
    elif domain == "humidifier":
        write_humidifier(entity, intents_file)


def write_verb(verb: str, entity: Entity, intents_file: typing.TextIO):
    print(
        f"{verb} [the] ({entity.spoken_name}){{name:{entity.name}}} :{{entity_id:{entity.id}}}",
        file=intents_file,
    )


def write_cover(entity: Entity, intents_file: typing.TextIO):
    print("[HassOpenCover]")
    write_verb("open", entity, intents_file)
    print("")

    print("[HassCloseCover]")
    write_verb("close", entity, intents_file)
    print("")


def write_switch(entity: Entity, intents_file: typing.TextIO):
    print("[HassTurnOn]")
    write_verb("turn on", entity, intents_file)
    print("")

    print("[HassTurnOff]")
    write_verb("turn off", entity, intents_file)
    print("")

    print("[HassToggle]")
    write_verb("toggle", entity, intents_file)
    print("")


def write_light(entity: Entity, intents_file: typing.TextIO):
    write_switch(entity, intents_file)

    print("[HassLightSet]")
    print(
        f"set [the] ({entity.spoken_name}){{name:{entity.name}}} brightness to (0..100){{brightness}} :{{entity_id:{entity.id}}}"
    )
    print(
        f"set [the] ({entity.spoken_name}){{name:{entity.name}}} [color] to (white | red | orange | green | blue | yellow | purple | brown){{color}}  :{{entity_id:{entity.id}}}"
    )
    print("")


def write_lock(entity: Entity, intents_file: typing.TextIO):
    print("[HassLock]")
    write_verb("lock", entity, intents_file)
    print("")

    print("[HassUnlock]")
    write_verb("unlock", entity, intents_file)
    print("")


def write_camera(entity: Entity, intents_file: typing.TextIO):
    print("[HassTurnOn]")
    write_verb("turn on", entity, intents_file)
    print("")

    print("[HassTurnOff]")
    write_verb("turn off", entity, intents_file)
    print("")


def write_climate(entity: Entity, intents_file: typing.TextIO):
    print("[HassTurnOn]")
    write_verb("turn on", entity, intents_file)
    print("")

    print("[HassTurnOff]")
    write_verb("turn off", entity, intents_file)
    print("")

    # Temperature
    print("[HassClimateTemperature]")
    print(
        f"set [the] ({entity.spoken_name}){{name:{entity.name}}} temperature to (0..100){{temperature}} [degrees] :{{entity_id:{entity.id}}}"
    )

    attributes = entity.state.get("attributes", {})

    # HVAC modes
    # NOTE: Won't work without also setting a temperature
    # hvac_modes = attributes.get("hvac_modes", [])
    # if hvac_modes:
    #     print("[HassClimateHvacMode]")
    #     hvac_mode_names = [f"({preprocess_name(mode)}):({mode})" for mode in hvac_modes]
    #     hvac_mode_names_str = " | ".join(hvac_mode_names)
    #     print(
    #         f"set [the] ({entity.spoken_name}){{name:{entity.name}}} mode to ({hvac_mode_names_str}){{mode}} :{{entity_id:{entity_id}}}"
    #     )

    # Preset modes
    preset_modes = attributes.get("preset_modes", [])
    if preset_modes:
        print("[HassClimatePresetMode]")
        preset_mode_names = [
            f"({preprocess_name(mode)}):({mode})" for mode in preset_modes
        ]
        preset_mode_names_str = " | ".join(preset_mode_names)
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} preset [mode] to ({preset_mode_names_str}){{mode}} :{{entity_id:{entity.id}}}"
        )


def write_fan(entity: Entity, intents_file: typing.TextIO):
    print("[HassTurnOn]")
    write_verb("turn on", entity, intents_file)
    print("")

    print("[HassTurnOff]")
    write_verb("turn off", entity, intents_file)
    print("")

    print("[HassFanSpeed]")
    print(
        f"set [the] ({entity.spoken_name}){{name:{entity.name}}} speed to (0..100){{percentage}} [percent] :{{entity_id:{entity.id}}}"
    )


def write_humidifier(entity: Entity, intents_file: typing.TextIO):
    print("[HassTurnOn]")
    write_verb("turn on", entity, intents_file)
    print("")

    print("[HassTurnOff]")
    write_verb("turn off", entity, intents_file)
    print("")

    attributes = entity.state.get("attributes", {})

    # Preset modes
    modes = attributes.get("available_modes", [])
    if modes:
        print("[HassHumidifierMode]")
        mode_names = [f"({preprocess_name(mode)}):({mode})" for mode in modes]
        mode_names_str = " | ".join(mode_names)
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} mode to ({mode_names_str}){{mode}} :{{entity_id:{entity.id}}}"
        )


def preprocess_name(name: str) -> str:
    """Ensure name can be spoken"""
    name = name.strip()

    # Split apart initialisms
    # ABC -> A B C
    if (len(name) > 1) and name.isupper():
        name = " ".join(list(name))

    name = name.replace("_", " ")

    return name


# -----------------------------------------------------------------------------


def main():
    """Print generated intents to the console"""
    parser = argparse.ArgumentParser()
    parser.add_argument("api_url")
    parser.add_argument("token")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    generate_intents(args.api_url, args.token, sys.stdout)


if __name__ == "__main__":
    main()
