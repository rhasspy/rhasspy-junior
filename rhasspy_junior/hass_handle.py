#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import typing

import requests

_LOGGER = logging.getLogger(__package__)

_BUILTIN_INTENTS = {
    "HassTurnOn",
    "HassTurnOff",
    "HassToggle",
    "HassOpenCover",
    "HassCloseCover",
    "HassLightSet",
}


def handle_intent(
    api_url: str, token: str, intent_object: typing.Dict[str, typing.Any]
):

    url = f"{api_url}/intent/handle"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    intent = intent_object.get("intent", {})
    intent_name = intent.get("name", "")

    if not intent_name:
        _LOGGER.warning("Skipping intent: %s", intent_object)
        return

    data: typing.Dict[str, str] = {"_intent": intent}

    entities = intent_object.get("entities", [])
    for entity in entities:
        data[entity["entity"]] = entity["value"]

    if intent_name in _BUILTIN_INTENTS:
        requests.post(url, headers=headers, json={"name": intent_name, "data": data})
    else:
        handle_custom_intent(api_url, headers, intent_name, data)


# -----------------------------------------------------------------------------


def handle_custom_intent(
    api_url, headers, intent_name: str, data: typing.Dict[str, str]
):
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
            f"{api_url}/services/{service_name}", headers=headers, json=service_data
        )
