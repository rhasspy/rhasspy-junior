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
Parameters used in the audio pipeline
These configure the following stages:
 - Conversion from audio to input vectors
 - Interpretation of the network output to a confidence value
"""
import typing
from dataclasses import dataclass
from enum import IntEnum
from math import floor


class Vectorizer(IntEnum):
    """
    Chooses which function to call to vectorize audio

    Options:
        mels: Convert to a compressed Mel spectrogram
        mfccs: Convert to a MFCC spectrogram
        speechpy_mfccs: Legacy option to convert to MFCCs using old library
    """

    mels = 1
    mfccs = 2
    speechpy_mfccs = 3


@dataclass
class ListenerParams:
    """
    General pipeline information:
     - Audio goes through a series of transformations to convert raw audio into machine readable data
     - These transformations are as follows:
       - Raw audio -> chopped audio
         - buffer_t, sample_depth: Input audio loaded and truncated using these value
         - window_t, hop_t: Linear audio chopped into overlapping frames using a sliding window
       - Chopped audio -> FFT spectrogram
         - n_fft, sample_rate: Each audio frame is converted to n_fft frequency intensities
       - FFT spectrogram -> Mel spectrogram (compressed)
         - n_filt: Each fft frame is compressed to n_filt summarized mel frequency bins/bands
       - Mel spectrogram -> MFCC
         - n_mfcc: Each mel frame is converted to MFCCs and the first n_mfcc values are taken
       - Disabled by default: Last phase -> Delta vectors
         - use_delta: If this value is true, the difference between consecutive vectors is concatenated to each frame

    Parameters for audio pipeline:
     - buffer_t: Input size of audio. Wakeword must fit within this time
     - window_t: Time of the window used to calculate a single spectrogram frame
     - hop_t: Time the window advances forward to calculate the next spectrogram frame
     - sample_rate: Input audio sample rate
     - sample_depth: Bytes per input audio sample
     - n_fft: Size of FFT to generate from audio frame
     - n_filt: Number of filters to compress FFT to
     - n_mfcc: Number of MFCC coefficients to use
     - use_delta: If True, generates "delta vectors" before sending to network
     - vectorizer: The type of input fed into the network. Options listed in class Vectorizer
     - threshold_config: Output distribution configuration automatically generated from precise-calc-threshold
     - threshold_center: Output distribution center automatically generated from precise-calc-threshold
    """

    buffer_t: float = 1.5
    window_t: float = 0.1
    hop_t: float = 0.05
    sample_rate: int = 16000
    sample_depth: int = 2
    n_fft: int = 512
    n_filt: int = 20
    n_mfcc: int = 13
    use_delta: bool = False
    vectorizer: int = Vectorizer.mfccs
    threshold_config: typing.Tuple[typing.Tuple[int, ...], ...] = ((6, 4),)
    threshold_center: float = 0.2

    @property
    def buffer_samples(self):
        """buffer_t converted to samples, truncating partial frames"""
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        """Number of timesteps in one input to the network"""
        return 1 + int(
            floor((self.buffer_samples - self.window_samples) / self.hop_samples)
        )

    @property
    def window_samples(self):
        """window_t converted to samples"""
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        """hop_t converted to samples"""
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        """The input size converted to audio samples"""
        return int(self.buffer_t * self.sample_rate)

    @property
    def feature_size(self):
        """The size of an input vector generated with these parameters"""
        num_features = {
            Vectorizer.mfccs: self.n_mfcc,
            Vectorizer.mels: self.n_filt,
            Vectorizer.speechpy_mfccs: self.n_mfcc,
        }[self.vectorizer]
        if self.use_delta:
            num_features *= 2
        return num_features
