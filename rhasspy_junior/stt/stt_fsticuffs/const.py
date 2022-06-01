"""Classes for automated speech recognition."""
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionToken:
    """Token from transcription.

    Attributes
    ----------
    token: str
        Token value

    start_time: float
        Seconds in utterance that token starts

    end_time: float
        Seconds in utterance that token ends

    likelihood: float
        Likelihood of transcription 0-1, 1 being sure
    """

    token: str
    start_time: float
    end_time: float
    likelihood: float


@dataclass
class Transcription:
    """Result of speech to text.

    Attributes
    ----------
    text: str
        Final transcription text

    likelihood: float
        Likelihood of transcription 0-1, 1 being sure

    transcribe_seconds: float
        Seconds it took to do transcription

    wav_seconds: float
        Duration of the transcribed WAV audio

    tokens: Optional[List[TranscriptionToken]] = None
        Optional list of tokens with times
    """

    text: str
    likelihood: float
    transcribe_seconds: float
    wav_seconds: float
    tokens: typing.Optional[typing.List[TranscriptionToken]] = None

    @classmethod
    def empty(cls) -> "Transcription":
        """Returns an empty transcription."""
        return Transcription(text="", likelihood=0, transcribe_seconds=0, wav_seconds=0)


class Transcriber(ABC):
    """Base class for speech to text transcribers."""

    @abstractmethod
    def transcribe_wav(self, wav_bytes: bytes) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""

    @abstractmethod
    def transcribe_stream(
        self,
        audio_stream: typing.Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> typing.Optional[Transcription]:
        """Speech to text from an audio stream."""

    @abstractmethod
    def stop(self):
        """Stop the transcriber."""
