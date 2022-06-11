#!/usr/bin/env python3
import collections.abc
import logging
import typing

import requests
import toml

from .const import IntentHandler, IntentHandleRequest, IntentHandleResult

_LOGGER = logging.getLogger(__package__)

_BUILTIN_INTENTS = {
    "HassTurnOn",
    "HassTurnOff",
    "HassToggle",
    "HassOpenCover",
    "HassCloseCover",
    "HassLightSet",
}


class HomeAssistantIntentHandler(IntentHandler):
    """Handle intents using Home Assistant"""

    def __init__(
        self,
        root_config: typing.Dict[str, typing.Any],
        config_extra_path: typing.Optional[str] = None,
    ):
        super().__init__(root_config, config_extra_path=config_extra_path)

        self.api_url = self.config["api_url"]
        self.api_token = self.config["api_token"]

        # Load mapping from intents to Home Assistant services
        intent_service_map_path = str(self.config["intent_service_map"])
        with open(
            intent_service_map_path, "r", encoding="utf-8"
        ) as intent_service_map_file:
            self.intent_service_map = toml.load(intent_service_map_file)

        self._handled = IntentHandleResult(handled=True)
        self._not_handled = IntentHandleResult(handled=False)

    @classmethod
    def config_path(cls) -> str:
        return "handle.home_assistant"

    def run(self, request: IntentHandleRequest) -> IntentHandleResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

        intent_name = request.intent_result.intent_name
        service_info = self.intent_service_map.get(intent_name)
        if service_info is None:
            _LOGGER.debug(
                "Cannot handle intent with Home Assistant: %s", request.intent_result
            )
            return self._not_handled

        entity_map = {"entity_id": "entity_id"}
        service_entities = service_info.get("entities", {})
        if not isinstance(service_entities, collections.abc.Mapping):
            service_entities = {e: e for e in service_entities}

        entity_map.update(service_entities)

        service_data: typing.Dict[str, str] = {}
        for entity in request.intent_result.entities:
            mapped_name = entity_map.get(entity.name)
            if mapped_name is not None:
                service_data[mapped_name] = entity.value

        service_name = service_info.get("service")

        if service_name:
            # Call service
            url = f"{self.api_url}/services/{service_name}"

            _LOGGER.debug("Calling service at %s: %s", url, service_data)
            response = requests.post(
                url,
                headers=headers,
                json=service_data,
            )

            if not response.ok:
                _LOGGER.error("Error from %s: %s", url, response)
                return self._not_handled
        else:
            # Handle as intent
            url = f"{self.api_url}/intent/handle"
            intent_data = {"name": intent_name, "data": service_data}

            _LOGGER.debug("Posting intent to %s: %s", url, intent_data)
            response = requests.post(
                url,
                headers=headers,
                json=intent_data,
            )

            if not response.ok:
                _LOGGER.error("Error from %s: %s", url, response)
                return self._not_handled

            response_dict = response.json()
            _LOGGER.debug(response_dict)

            # Handle TTS response
            response_speech = (
                response_dict.get("speech", {}).get("plain", {}).get("speech")
            )
            if response_speech:
                tts_service_info = self.config.get("tts", {})
                tts_service = tts_service_info.get("service")
                if tts_service:
                    tts_data = tts_service_info.get("entities", {})
                    tts_data["message"] = response_speech

                    tts_url = f"{self.api_url}/services/{tts_service}"
                    _LOGGER.debug("Posting speech to %s: %s", tts_url, tts_data)
                    requests.post(tts_url, headers=headers, json=tts_data)

        return self._handled
