"""Automated speech recognition in Rhasspy using Kaldi."""
import io
import itertools
import logging
import os
import subprocess
import tempfile
import time
import typing
import wave
from pathlib import Path

from .const import Transcription, TranscriptionToken
from .train import prune_intents

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class KaldiCommandLineTranscriber:
    """Speech to text with external Kaldi scripts."""

    def __init__(
        self,
        model_dir: typing.Union[str, Path],
        graph_dir: typing.Union[str, Path],
        kaldi_dir: typing.Union[str, Path],
        port_num: typing.Optional[int] = None,
        kaldi_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)
        self.kaldi_dir = Path(kaldi_dir)
        self.scopes: typing.Optional[typing.Set[str]] = None
        self.decode_proc = None
        self.port_num = 5050 if port_num is None else port_num

        self.timeout_seconds = 20

        # Additional arguments passed to Kaldi process
        self.kaldi_args = kaldi_args

        self.temp_dir = None
        self.chunk_fifo_path = None
        self.chunk_fifo_file = None

        _LOGGER.debug("Using kaldi at %s", str(self.kaldi_dir))

    def set_scope(
        self,
        scopes: typing.Optional[typing.Set[str]] = None,
    ):
        if scopes != self.scopes:
            prune_intents(self.graph_dir.parent, self.graph_dir, self.kaldi_dir, scopes)

    def transcribe_wav_file(
        self, wav_path: typing.Union[str, Path]
    ) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""
        start_time = time.perf_counter()

        wav_path_str = str(wav_path)
        text = self._transcribe_wav_nnet3(wav_path_str)

        if text:
            # Success
            end_time = time.perf_counter()

            with wave.open(wav_path_str, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                wav_seconds = frames / float(rate)

            return Transcription(
                text=text.strip(),
                likelihood=1,
                transcribe_seconds=(end_time - start_time),
                wav_seconds=wav_seconds,
            )

        # Failure
        return None

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""
        start_time = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb") as wav_file:
            wav_file.write(wav_bytes)
            wav_file.seek(0)

            text = self._transcribe_wav_nnet3(wav_file.name)

        if text:
            # Success
            end_time = time.perf_counter()

            return Transcription(
                text=text.strip(),
                likelihood=1,
                transcribe_seconds=(end_time - start_time),
                wav_seconds=get_wav_duration(wav_bytes),
            )

        # Failure
        return None

    def _transcribe_wav_nnet3(self, wav_path: str) -> str:
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "online" / "conf" / "online.conf"
        kaldi_cmd = [
            str(self.kaldi_dir / "online2-wav-nnet3-latgen-faster"),
            "--online=false",
            "--do-endpointing=false",
            "--max-active=7000",
            "--lattice-beam=8.0",
            "--acoustic-scale=1.0",
            "--beam=24.0",
            f"--word-symbol-table={words_txt}",
            f"--config={online_conf}",
            str(self.model_dir / "model" / "final.mdl"),
            str(self.graph_dir / "HCLG.fst"),
            "ark:echo utt1 utt1|",
            f"scp:echo utt1 {wav_path}|",
            "ark:/dev/null",
        ]

        # Add custom arguments
        if self.kaldi_args:
            for arg_name, arg_value in self.kaldi_args.items():
                kaldi_cmd.append(f"--{arg_name}={arg_value}")

        _LOGGER.debug(kaldi_cmd)

        try:
            lines = subprocess.check_output(
                kaldi_cmd, stderr=subprocess.STDOUT, universal_newlines=True
            ).splitlines()
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("_transcribe_wav_nnet3")
            _LOGGER.error(e.output)
            lines = []

        text = ""
        for line in lines:
            if line.startswith("utt1 "):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    text = parts[1]
                break

        return text

    # -------------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio_stream: typing.Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> typing.Optional[Transcription]:
        """Speech to text from an audio stream."""

        # Use online2-tcp-nnet3-decode-faster
        if not self.decode_proc:
            self.start_decode()

        assert self.decode_proc, "No decode process"

        start_time = time.perf_counter()
        num_frames = 0
        for chunk in audio_stream:
            if chunk:
                num_samples = len(chunk) // sample_width

                # Write sample count to process stdin
                print(num_samples, file=self.decode_proc.stdin)
                self.decode_proc.stdin.flush()

                # Write chunk to FIFO.
                # Make sure that we write exactly the right number of bytes.
                self.chunk_fifo_file.write(chunk[: num_samples * sample_width])
                self.chunk_fifo_file.flush()
                num_frames += num_samples

        # Finish utterance
        print("0", file=self.decode_proc.stdin)
        self.decode_proc.stdin.flush()

        _LOGGER.debug("Finished stream. Getting transcription.")

        confidence_and_text = ""
        for line in self.decode_proc.stdout:
            line = line.strip()
            if line.lower() == "ready":
                continue

            confidence_and_text = line
            break

        _LOGGER.debug(confidence_and_text)

        if confidence_and_text:
            # Success
            end_time = time.perf_counter()

            # <mbr_wer> <word> <word_confidence> <word_start_time> <word_end_time> ...
            wer_str, *words = confidence_and_text.split()
            confidence = 0.0

            try:
                # Try to parse minimum bayes risk (MBR) word error rate (WER)
                confidence = max(0, 1 - float(wer_str))
            except ValueError:
                _LOGGER.exception(wer_str)

            tokens = []
            for word, word_confidence, word_start_time, word_end_time in grouper(
                words, n=4
            ):
                tokens.append(
                    TranscriptionToken(
                        token=word,
                        start_time=float(word_start_time),
                        end_time=float(word_end_time),
                        likelihood=float(word_confidence),
                    )
                )

            text = " ".join(t.token for t in tokens)
            return Transcription(
                text=text,
                likelihood=confidence,
                transcribe_seconds=(end_time - start_time),
                wav_seconds=(num_frames / sample_rate),
                tokens=tokens,
            )

        # Failure
        return None

    def stop(self):
        """Stop the transcriber."""
        if self.decode_proc:
            self.decode_proc.terminate()
            self.decode_proc.wait()
            self.decode_proc = None

        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None

        if self.chunk_fifo_file:
            self.chunk_fifo_file.close()
            self.chunk_fifo_file = None

        self.chunk_fifo_path = None

    def start_decode(self):
        """Starts online2-tcp-nnet3-decode-faster process."""
        if self.temp_dir is None:
            # pylint: disable=consider-using-with
            self.temp_dir = tempfile.TemporaryDirectory()

        if self.chunk_fifo_path is None:
            self.chunk_fifo_path = os.path.join(self.temp_dir.name, "chunks.fifo")
            _LOGGER.debug("Creating FIFO at %s", self.chunk_fifo_path)
            os.mkfifo(self.chunk_fifo_path)

        online_conf = self.model_dir / "online" / "conf" / "online.conf"

        kaldi_cmd = [
            str(self.kaldi_dir / "online2-cli-nnet3-decode-faster-confidence"),
            f"--config={online_conf}",
            "--max-active=7000",
            "--lattice-beam=8.0",
            "--acoustic-scale=1.0",
            "--beam=24.0",
            str(self.model_dir / "model" / "final.mdl"),
            str(self.graph_dir / "HCLG.fst"),
            str(self.graph_dir / "words.txt"),
            str(self.chunk_fifo_path),
        ]

        # Add custom arguments
        if self.kaldi_args:
            for arg_name, arg_value in self.kaldi_args.items():
                kaldi_cmd.append(f"--{arg_name}={arg_value}")

        _LOGGER.debug(kaldi_cmd)

        # pylint: disable=consider-using-with
        self.decode_proc = subprocess.Popen(
            kaldi_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        # NOTE: The placement of this open is absolutely critical
        #
        # At this point, the decode process will block waiting for the other
        # side of the pipe.
        #
        # We won't reach the "ready" stage if we open this earlier or later.
        if self.chunk_fifo_file is None:
            # pylint: disable=consider-using-with
            self.chunk_fifo_file = open(self.chunk_fifo_path, mode="wb")

        # Read until started
        line = self.decode_proc.stdout.readline().lower().strip()
        if line:
            _LOGGER.debug(line)

        while "ready" not in line:
            line = self.decode_proc.stdout.readline().lower().strip()
            if line:
                _LOGGER.debug(line)

        _LOGGER.debug("Decoder started")

    def __repr__(self) -> str:
        return (
            "KaldiCommandLineTranscriber("
            f", model_dir={self.model_dir}"
            f", graph_dir={self.graph_dir}"
            ")"
        )


# -----------------------------------------------------------------------------


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return the real-time duration of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
