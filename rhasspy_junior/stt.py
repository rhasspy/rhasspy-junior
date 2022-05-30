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
import platform
import threading
import typing
from pathlib import Path
from queue import Queue

from .fsticuffs_stt.transcribe import KaldiCommandLineTranscriber

_LOGGER = logging.getLogger(__package__)


class STTEngine:
    def __init__(self, config: typing.Dict[str, typing.Any]):
        self.config = config["stt"]

        self._running = True
        self._input_queue = Queue()
        self._text: typing.Optional[str] = None
        self._audio_queue = Queue()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def text(self) -> typing.Optional[str]:
        return self._text

    def start_phrase(self):
        while not self._audio_queue.empty():
            self._audio_queue.get()

        self._text = None
        self._input_queue.put(True)

    def process(self, chunk: bytes):
        self._audio_queue.put_nowait(chunk)

    def stop_phrase(self):
        self._audio_queue.put_nowait(None)

    def stop_engine(self):
        self._running = False
        self._input_queue.put(None)
        self._thread.join()

    def _run(self):
        try:
            kaldi_dir = Path(str(self.config["kaldi_dir"])) / platform.machine()
            sample_rate = int(self.config["sample_rate"])
            sample_width = int(self.config["sample_width"])
            channels = int(self.config["channels"])

            transcriber = KaldiCommandLineTranscriber(
                model_dir=str(self.config["model_dir"]),
                graph_dir=str(self.config["graph_dir"]),
                kaldi_dir=kaldi_dir,
            )

            transcriber.start_decode()

            def audio_stream():
                while True:
                    chunk = self._audio_queue.get()
                    if chunk is None:
                        break

                    yield chunk

            while True:
                todo = self._input_queue.get()

                if (not self._running) or (todo is None):
                    break

                _LOGGER.debug("Processing speech to text")
                result = transcriber.transcribe_stream(
                    audio_stream=audio_stream(),
                    sample_rate=sample_rate,
                    sample_width=sample_width,
                    channels=channels,
                )

                _LOGGER.debug(result)

                if result:
                    self._text = result.text
                else:
                    self._text = ""

        except Exception:
            _LOGGER.exception("Unexpected error in speech to text thread")
