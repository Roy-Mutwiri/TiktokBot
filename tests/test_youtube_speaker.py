import os
import sys
import unittest


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import youtube_speaker


class YouTubeSpeakerTests(unittest.TestCase):
    def test_parse_vtt_removes_timestamps_and_dedupes(self):
        raw = """WEBVTT

00:00:01.000 --> 00:00:02.000
hello <c>world</c>

00:00:02.000 --> 00:00:03.000
hello world

00:00:03.000 --> 00:00:04.000
[Music]
next line
"""
        self.assertEqual(
            youtube_speaker._parse_vtt(raw),
            "hello world next line",
        )

    def test_parse_json3_events(self):
        raw = '{"events":[{"tStartMs":1000,"segs":[{"utf8":"hello "},{"utf8":"there"}]},{"tStartMs":3000,"segs":[{"utf8":"again"}]}]}'
        self.assertEqual(youtube_speaker._parse_json3(raw), "hello there again")

    def test_parse_json3_filters_time_range(self):
        raw = '{"events":[{"tStartMs":1000,"segs":[{"utf8":"first"}]},{"tStartMs":70000,"segs":[{"utf8":"second"}]}]}'
        self.assertEqual(
            youtube_speaker._parse_json3(raw, start_seconds=60, end_seconds=80),
            "second",
        )

    def test_chunk_for_speech_respects_size(self):
        text = "One short sentence. Two short sentence. Three short sentence."
        chunks = youtube_speaker.chunk_for_speech(text, chunk_chars=25)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 25 for c in chunks))


if __name__ == "__main__":
    unittest.main()
