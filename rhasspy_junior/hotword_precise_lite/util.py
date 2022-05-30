# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Miscellaneous utility functions for things like audio loading
"""
import typing

import numpy as np

MAX_WAV_VALUE = 32768


def chunk_audio(
    audio: np.ndarray, chunk_size: int
) -> typing.Generator[np.ndarray, None, None]:
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size : i]


def buffer_to_audio(audio_buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.frombuffer(audio_buffer, dtype="<i2").astype(
        np.float32, order="C"
    ) / float(MAX_WAV_VALUE)


def audio_to_buffer(audio: np.ndarray) -> bytes:
    """Convert a numpy array of floats to raw mono audio"""
    return (audio * MAX_WAV_VALUE).astype("<i2").tobytes()
