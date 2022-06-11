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
from jinja2 import Environment, FileSystemLoader

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
    """Pulls entities from Home Assistant and writes sentences to control them"""

    @classmethod
    def config_path(cls) -> str:
        return "train.home_assistant"

    def run(self, context: TrainingContext) -> TrainingContext:
        """Run trainer"""
        api_url = str(self.config["api_url"])
        api_token = str(self.config["api_token"])
        input_template_paths = [Path(p) for p in self.config["input_templates"]]
        output_sentences_path = Path(self.config["output_sentences"])
        entities_to_ignore = self.config.get("entities_to_ignore")

        templates = []
        for template_path in input_template_paths:
            template_env = Environment(
                loader=FileSystemLoader(template_path.parent),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = template_env.get_template(template_path.name)
            templates.append(template)

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
            entities = []

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
                    spoken_name=self.clean_name(friendly_name),
                    state=entity_state,
                )

                entities.append(entity)

            for template in templates:
                output_sentences_file.write(
                    template.render(
                        entities=entities,
                        verb=self.verb,
                        clean_name=self.clean_name,
                        map=map,
                    )
                )

        return context

    def verb(self, verb: str, entity: Entity) -> str:
        return f"{verb} [the] ({entity.spoken_name}){{name:{entity.name}}} :{{entity_id:{entity.id}}}"

    def clean_name(self, name: str) -> str:
        """Ensure name can be spoken"""
        name = name.strip()

        # Split apart initialisms
        # ABC -> A B C
        if (len(name) > 1) and name.isupper():
            name = " ".join(list(name))

        name = name.replace("_", " ")

        return name
