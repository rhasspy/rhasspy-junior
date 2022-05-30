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
import json
import shlex
import subprocess
import typing
from enum import Enum, auto
from pathlib import Path

import networkx as nx

from .hass_handle import handle_intent
from .hotword_precise_lite.mycroft_hotword import TFLiteHotWordEngine
from .fsticuffs import recognize, json_to_graph
from .stt import STTEngine
from .vad_silero.silence import SilenceDetector, SilenceResultType

_LOGGER = logging.getLogger(__package__)


class State(str, Enum):
    DETECTING_HOTWORD = auto()
    BEFORE_COMMAND = auto()
    IN_COMMAND = auto()
    AFTER_COMMAND = auto()


def run_junior(config: typing.Dict[str, typing.Any]):
    state = State.DETECTING_HOTWORD

    hass = config["home_assistant"]
    api_url = hass["api_url"]
    api_token = hass["api_token"]

    hotword_engine = load_hotword_engine(config)
    vad_engine = load_vad_engine(config)
    stt_engine = load_stt_engine(config)
    intent_graph = load_intent_graph(config)

    # Start mic
    mic_run = config["microphone"]["run"]
    mic_chunk_bytes = int(config["microphone"]["chunk_bytes"])
    mic_cmd = shlex.split(mic_run)
    _LOGGER.debug(mic_cmd)

    try:
        with subprocess.Popen(mic_cmd, stdout=subprocess.PIPE) as mic_proc:
            assert mic_proc.stdout is not None

            while True:
                chunk = mic_proc.stdout.read(mic_chunk_bytes)
                if not chunk:
                    break

                try:
                    if state == State.DETECTING_HOTWORD:
                        hotword_engine.update(chunk)
                        if hotword_engine.found_wake_word(None):
                            _LOGGER.debug("Hotword detected")
                            state = State.BEFORE_COMMAND
                            vad_engine.start()
                            stt_engine.start_phrase()
                    elif state == State.BEFORE_COMMAND:
                        stt_engine.process(chunk)
                        result = vad_engine.process(chunk)
                        if result.type == SilenceResultType.PHRASE_START:
                            _LOGGER.debug(result)
                            state = State.IN_COMMAND
                    elif state == State.IN_COMMAND:
                        stt_engine.process(chunk)
                        result = vad_engine.process(chunk)
                        if result.type == SilenceResultType.PHRASE_END:
                            _LOGGER.debug("Recording ended")
                            state = State.AFTER_COMMAND
                            stt_engine.stop_phrase()
                            vad_engine.stop()
                    elif state == State.AFTER_COMMAND:
                        if stt_engine.text is not None:
                            intent_results = recognize(stt_engine.text, intent_graph)
                            if intent_results:
                                intent_result = intent_results[0]
                                _LOGGER.debug(intent_result)

                                # Send to Home Assistant
                                handle_intent(api_url, api_token, intent_result.asdict())
                            else:
                                _LOGGER.debug(
                                    "No intent recognized (text: %s)", stt_engine.text
                                )

                            state = State.DETECTING_HOTWORD
                except Exception:
                    _LOGGER.exception("Unexpected error while handling audio chunk")

                    # Clean up and go back to listening for hotword
                    vad_engine.stop()
                    stt_engine.stop_phrase()
                    state = State.DETECTING_HOTWORD
    finally:
        stt_engine.stop_engine()


def load_hotword_engine(config: typing.Dict[str, typing.Any]) -> TFLiteHotWordEngine:
    hotword = config["hotword"]

    return TFLiteHotWordEngine(
        local_model_file=str(hotword["model"]),
        sensitivity=float(hotword["sensitivity"]),
        trigger_level=int(hotword["trigger_level"]),
        chunk_size=int(hotword["chunk_size"]),
    )


def load_vad_engine(config: typing.Dict[str, typing.Any]) -> SilenceDetector:
    vad = config["vad"]

    return SilenceDetector(
        vad_model=str(vad["model"]),
        vad_threshold=float(vad["threshold"]),
        sample_rate=int(vad["sample_rate"]),
        chunk_size=int(vad["chunk_size"]),
        skip_seconds=float(vad["skip_seconds"]),
        min_seconds=float(vad["min_seconds"]),
        max_seconds=float(vad["max_seconds"]),
        speech_seconds=float(vad["speech_seconds"]),
        silence_seconds=float(vad["silence_seconds"]),
        before_seconds=float(vad["before_seconds"]),
    )


def load_stt_engine(config: typing.Dict[str, typing.Any]) -> STTEngine:
    return STTEngine(config)


def load_intent_graph(config: typing.Dict[str, typing.Any]) -> nx.DiGraph:
    intent = config["intent"]

    with open(intent["graph"], "r", encoding="utf-8") as graph_file:
        graph_dict = json.load(graph_file)
        return json_to_graph(graph_dict)
