#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import typing

import requests

from ..intent import IntentResult
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
    def __init__(self, config: typing.Dict[str, typing.Any]):
        self.config = config["handle"]["home_assistant"]

        self.api_url = self.config["api_url"]
        self.api_token = self.config["api_token"]

        self._handled = IntentHandleResult(handled=True)
        self._not_handled = IntentHandleResult(handled=False)

    def run(self, request: IntentHandleRequest) -> IntentHandleResult:
        url = f"{self.api_url}/intent/handle"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

        data: typing.Dict[str, str] = {}

        intent_name = request.intent_result.intent_name

        for entity in request.intent_result.entities:
            data[entity.name] = entity.value

        if intent_name in _BUILTIN_INTENTS:
            requests.post(
                url, headers=headers, json={"name": intent_name, "data": data}
            )

            return self._handled

        return self.handle_custom_intent(intent_name, data, headers)

    def handle_custom_intent(
        self, intent_name: str, data: typing.Dict[str, str], headers
    ) -> IntentHandleResult:
        service_name = ""
        service_data = {"entity_id": data["entity_id"]}

        if intent_name == "HassLock":
            service_name = "lock/lock"
        elif intent_name == "HassUnlock":
            service_name = "lock/unlock"
        elif intent_name == "HassClimateTemperature":
            service_name = "climate/set_temperature"
            service_data["temperature"] = data["temperature"]
        # elif intent_name == "HassClimateHvacMode":
        #     service_name = "climate/set_temperature"
        #     service_data["hvac_mode"] = data["mode"]
        elif intent_name == "HassClimatePresetMode":
            service_name = "climate/set_preset_mode"
            service_data["preset_mode"] = data["mode"]
        elif intent_name == "HassFanSpeed":
            service_name = "fan/set_percentage"
            service_data["percentage"] = data["percentage"]
        elif intent_name == "HassHumidifierMode":
            service_name = "humidifier/set_mode"
            service_data["mode"] = data["mode"]

        if service_name:
            requests.post(
                f"{self.api_url}/services/{service_name}",
                headers=headers,
                json=service_data,
            )

            return self._handled

        return self._not_handled
