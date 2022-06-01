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

import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import requests

from .const import Trainer, TrainingContext

_LOGGER = logging.getLogger(__package__)


@dataclass
class Entity:
    id: str
    domain: str
    name: str
    spoken_name: str
    state: typing.Dict[str, typing.Any]


class HomeAssistantTrainer(Trainer):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)
        self.config = config["train"]["home_assistant"]

    def run(self, context: TrainingContext) -> TrainingContext:
        """Run trainer"""
        api_url = str(self.config["api_url"])
        api_token = str(self.config["api_token"])
        output_sentences_path = Path(self.config["output_sentences"])
        entities_to_ignore = self.config.get("entities_to_ignore")

        output_sentences_path.parent.mkdir(parents=True, exist_ok=True)

        if entities_to_ignore is None:
            entities_to_ignore = set()

        if api_url.endswith("/"):
            api_url = api_url[:-1]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}",
        }

        response = requests.get(f"{api_url}/states", headers=headers)
        assert response.ok, response

        states = response.json()

        with open(
            output_sentences_path, "w", encoding="utf-8"
        ) as output_sentences_file:
            for entity_state in states:
                entity_id = entity_state.get("entity_id", "")

                if entity_id in entities_to_ignore:
                    _LOGGER.debug("Ignoring entity: %s", entity_id)
                    continue

                if "." not in entity_id:
                    # No domain
                    _LOGGER.warning("Skipping entity %s: no domain", entity_id)
                    continue

                friendly_name = entity_state.get("attributes", {}).get(
                    "friendly_name", ""
                )
                if not friendly_name:
                    # No friendly name
                    _LOGGER.warning("Skipping entity %s: no friendly name")
                    continue

                entity = Entity(
                    id=entity_id,
                    domain=entity_id.split(".", maxsplit=1)[0],
                    name=friendly_name,
                    spoken_name=self.preprocess_name(friendly_name),
                    state=entity_state,
                )

                self.write_entity(entity, output_sentences_file)

        return context

    def write_entity(self, entity: Entity, intents_file: typing.TextIO):
        domain = entity.domain

        if domain == "cover":
            self.write_cover(entity, intents_file)
        elif domain == "switch":
            self.write_switch(entity, intents_file)
        elif domain == "light":
            self.write_light(entity, intents_file)
        elif domain == "lock":
            self.write_lock(entity, intents_file)
        elif domain == "camera":
            self.write_camera(entity, intents_file)
        elif domain == "climate":
            self.write_climate(entity, intents_file)
        elif domain == "fan":
            self.write_fan(entity, intents_file)
        elif domain == "humidifier":
            self.write_humidifier(entity, intents_file)

    def write_verb(self, verb: str, entity: Entity, intents_file: typing.TextIO):
        print(
            f"{verb} [the] ({entity.spoken_name}){{name:{entity.name}}} :{{entity_id:{entity.id}}}",
            file=intents_file,
        )

    def write_cover(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassOpenCover]", file=intents_file)
        self.write_verb("open", entity, intents_file)
        print("", file=intents_file)

        print("[HassCloseCover]", file=intents_file)
        self.write_verb("close", entity, intents_file)
        print("", file=intents_file)

    def write_switch(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassTurnOn]", file=intents_file)
        self.write_verb("turn on", entity, intents_file)
        print("", file=intents_file)

        print("[HassTurnOff]", file=intents_file)
        self.write_verb("turn off", entity, intents_file)
        print("", file=intents_file)

        print("[HassToggle]", file=intents_file)
        self.write_verb("toggle", entity, intents_file)
        print("", file=intents_file)

    def write_light(self, entity: Entity, intents_file: typing.TextIO):
        self.write_switch(entity, intents_file)

        print("[HassLightSet]", file=intents_file)
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} brightness to (0..100){{brightness}} :{{entity_id:{entity.id}}}",
            file=intents_file,
        )
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} [color] to (white | red | orange | green | blue | yellow | purple | brown){{color}}  :{{entity_id:{entity.id}}}",
            file=intents_file,
        )
        print("", file=intents_file)

    def write_lock(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassLock]", file=intents_file)
        self.write_verb("lock", entity, intents_file)
        print("", file=intents_file)

        print("[HassUnlock]", file=intents_file)
        self.write_verb("unlock", entity, intents_file)
        print("", file=intents_file)

    def write_camera(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassTurnOn]", file=intents_file)
        self.write_verb("turn on", entity, intents_file)
        print("", file=intents_file)

        print("[HassTurnOff]", file=intents_file)
        self.write_verb("turn off", entity, intents_file)
        print("", file=intents_file)

    def write_climate(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassTurnOn]", file=intents_file)
        self.write_verb("turn on", entity, intents_file)
        print("", file=intents_file)

        print("[HassTurnOff]", file=intents_file)
        self.write_verb("turn off", entity, intents_file)
        print("", file=intents_file)

        # Temperature
        print("[HassClimateTemperature]", file=intents_file)
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} temperature to (0..100){{temperature}} [degrees] :{{entity_id:{entity.id}}}",
            file=intents_file,
        )

        attributes = entity.state.get("attributes", {})

        # HVAC modes
        # NOTE: Won't work without also setting a temperature
        # hvac_modes = attributes.get("hvac_modes", [])
        # if hvac_modes:
        #     print("[HassClimateHvacMode]", file=intents_file)
        #     hvac_mode_names = [f"({preprocess_name(mode)}):({mode})" for mode in hvac_modes]
        #     hvac_mode_names_str = " | ".join(hvac_mode_names)
        #     print(
        #         f"set [the] ({entity.spoken_name}){{name:{entity.name}}} mode to ({hvac_mode_names_str}){{mode}} :{{entity_id:{entity_id}}}"
        #     )

        # Preset modes
        preset_modes = attributes.get("preset_modes", [])
        if preset_modes:
            print("[HassClimatePresetMode]", file=intents_file)
            preset_mode_names = [
                f"({self.preprocess_name(mode)}):({mode})" for mode in preset_modes
            ]
            preset_mode_names_str = " | ".join(preset_mode_names)
            print(
                f"set [the] ({entity.spoken_name}){{name:{entity.name}}} preset [mode] to ({preset_mode_names_str}){{mode}} :{{entity_id:{entity.id}}}",
                file=intents_file,
            )

    def write_fan(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassTurnOn]", file=intents_file)
        self.write_verb("turn on", entity, intents_file)
        print("", file=intents_file)

        print("[HassTurnOff]", file=intents_file)
        self.write_verb("turn off", entity, intents_file)
        print("", file=intents_file)

        print("[HassFanSpeed]", file=intents_file)
        print(
            f"set [the] ({entity.spoken_name}){{name:{entity.name}}} speed to (0..100){{percentage}} [percent] :{{entity_id:{entity.id}}}",
            file=intents_file,
        )

    def write_humidifier(self, entity: Entity, intents_file: typing.TextIO):
        print("[HassTurnOn]", file=intents_file)
        self.write_verb("turn on", entity, intents_file)
        print("", file=intents_file)

        print("[HassTurnOff]", file=intents_file)
        self.write_verb("turn off", entity, intents_file)
        print("", file=intents_file)

        attributes = entity.state.get("attributes", {})

        # Preset modes
        modes = attributes.get("available_modes", [])
        if modes:
            print("[HassHumidifierMode]", file=intents_file)
            mode_names = [f"({self.preprocess_name(mode)}):({mode})" for mode in modes]
            mode_names_str = " | ".join(mode_names)
            print(
                f"set [the] ({entity.spoken_name}){{name:{entity.name}}} mode to ({mode_names_str}){{mode}} :{{entity_id:{entity.id}}}",
                file=intents_file,
            )

    def preprocess_name(self, name: str) -> str:
        """Ensure name can be spoken"""
        name = name.strip()

        # Split apart initialisms
        # ABC -> A B C
        if (len(name) > 1) and name.isupper():
            name = " ".join(list(name))

        name = name.replace("_", " ")

        return name
