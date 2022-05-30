#!/usr/bin/env python3
# Copyright 2021 Mycroft AI Inc.
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
import argparse
import logging
import sys

from universal_text_protocol import read_event

from .silence import SilenceDetector, SilenceResultType

_LOGGER = logging.getLogger("vad")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-text",
        default="start",
        help="Text to print when vad has detected the start of a voice command",
    )
    parser.add_argument(
        "--end-text",
        default="end",
        help="Text to print when vad has detected the end of a voice command",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    detector = SilenceDetector()

    try:
        recording = False
        last_silence = True

        while True:
            event = read_event(sys.stdin.buffer)

            # TODO: Remove
            if event.topic == "mic/audio":
                is_silence = detector.is_silence(event.payload)
                if last_silence and (not is_silence):
                    print("speech", flush=True)
                elif (not last_silence) and is_silence:
                    print("silence", flush=True)

                last_silence = is_silence

            if (not recording) and (event.topic == "wake/detected"):
                detector.start()
                recording = True
            elif recording and (event.topic == "mic/audio"):
                result = detector.process(event.payload)
                if result.type == SilenceResultType.PHRASE_START:
                    print(args.start_text, flush=True)
                elif result.type == SilenceResultType.PHRASE_END:
                    detector.stop()
                    recording = False
                    print(args.end_text, flush=True)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
