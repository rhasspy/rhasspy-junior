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
from pathlib import Path

import numpy as np
import tflite_runtime.interpreter as tflite
from sonopy import mfcc_spec

from ..const import Hotword, HotwordProcessResult
from .params import ListenerParams
from .util import buffer_to_audio

_LOGGER = logging.getLogger(__package__)


class PreciseHotword(Hotword):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__(config)
        self.config = config["hotword"]["precise"]

        self.model_path = Path(str(self.config["model"])).absolute()
        self.sensitivity = float(self.config["sensitivity"])
        self.trigger_level = int(self.config["trigger_level"])
        self.chunk_bytes = int(self.config["chunk_bytes"])

        self._interpreter: typing.Optional[tflite.Interpreter] = None
        self._params: typing.Optional[ListenerParams] = None
        self._input_details: typing.Optional[typing.Any] = None
        self._output_details: typing.Optional[typing.Any] = None

        # Rolling window of MFCCs (fixed sized)
        self._inputs: typing.Optional[np.ndarray] = None

        # Current MFCC timestep
        self._inputs_idx: int = 0

        # Bytes for one window of audio
        self._window_bytes: int = 0

        # Bytes for one MFCC hop
        self._hop_bytes: int = 0

        # Raw audio
        self._chunk_buffer = bytes()

        # Activation level (> trigger_level = wake word found)
        self._activation: int = 0

        self._found = HotwordProcessResult(is_detected=True)
        self._not_found = HotwordProcessResult(is_detected=False)

    def start(self):
        """Start detector"""
        _LOGGER.debug("Loading model from %s", self.model_path)

        self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # TODO: Load these from adjacent file
        self._params = ListenerParams()

        self._window_bytes = self._params.window_samples * self._params.sample_depth
        self._hop_bytes = self._params.hop_samples * self._params.sample_depth

        # Rolling window of MFCCs (fixed sized)
        self._inputs = np.zeros(
            (1, self._params.n_features, self._params.n_mfcc), dtype=np.float32
        )

        self._chunk_buffer = bytes()

    def stop(self):
        """Stop detector"""
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._params = None
        self._inputs = None
        self._chunk_buffer = bytes()

    def process_chunk(self, chunk: bytes) -> HotwordProcessResult:
        """Process chunk of raw audio data."""
        assert self._interpreter is not None
        assert self._params is not None
        assert self._inputs is not None
        assert self._input_details is not None
        assert self._output_details is not None

        self._chunk_buffer += chunk

        if len(self._chunk_buffer) < self._window_bytes:
            # Need a full window of audio first
            return self._not_found

        # Process current audio
        audio = buffer_to_audio(self._chunk_buffer)

        # TODO: Implement different MFCC algorithms
        mfccs = mfcc_spec(
            audio,
            self._params.sample_rate,
            (self._params.window_samples, self._params.hop_samples),
            num_filt=self._params.n_filt,
            fft_size=self._params.n_fft,
            num_coeffs=self._params.n_mfcc,
        )

        # Number of timesteps processed
        num_features = mfccs.shape[0]

        # Remove processed audio
        self._chunk_buffer = self._chunk_buffer[num_features * self._hop_bytes :]

        inputs_end_idx = self._inputs_idx + num_features
        if inputs_end_idx > self._inputs.shape[1]:
            # Roll mfccs array backwards along time dimension
            self._inputs = np.roll(self._inputs, -num_features, axis=1)
            self._inputs_idx -= num_features
            inputs_end_idx -= num_features

        # Append to end of rolling window
        self._inputs[0, self._inputs_idx : inputs_end_idx, :] = mfccs
        self._inputs_idx = inputs_end_idx

        if self._inputs_idx < self._inputs.shape[1]:
            return self._not_found

        # TODO: Add deltas

        # raw_output
        self._interpreter.set_tensor(self._input_details[0]["index"], self._inputs)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(self._output_details[0]["index"])
        prob = raw_output[0][0]

        if (prob < 0.0) or (prob > 1.0):
            # TODO: Handle out of range.
            # Not seeing these currently, so ignoring.
            return self._not_found

        # Decode
        activated = prob > 1.0 - self.sensitivity
        triggered = False
        if activated or (self._activation < 0):
            # Increase activation
            self._activation += 1

            triggered = self._activation > self.trigger_level
            if triggered or (activated and (self._activation < 0)):
                # Push activation down far to avoid an accidental re-activation
                self._activation = -(8 * 2048) // self.chunk_bytes
        elif self._activation > 0:
            # Decrease activation
            self._activation -= 1

        if triggered:
            return self._found

        return self._not_found
