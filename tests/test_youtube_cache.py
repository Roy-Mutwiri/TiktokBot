import os
import sys
import tempfile
import unittest


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import youtube_cache


class YouTubeCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cache_dir = youtube_cache.CACHE_DIR
        self.old_db_path = youtube_cache.DB_PATH
        youtube_cache.CACHE_DIR = self.tmp.name
        youtube_cache.DB_PATH = os.path.join(self.tmp.name, "cache.sqlite")

    def tearDown(self):
        youtube_cache.CACHE_DIR = self.old_cache_dir
        youtube_cache.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def test_video_id_from_short_url(self):
        self.assertEqual(
            youtube_cache.video_id_from_url("https://youtu.be/7-cyM9maRR0"),
            "7-cyM9maRR0",
        )

    def test_transcript_round_trip(self):
        url = "https://youtu.be/7-cyM9maRR0"
        youtube_cache.save_transcript(url, "Title", 123, "vtt", "WEBVTT\nhello")

        cached = youtube_cache.get_cached_transcript(url)

        self.assertEqual(cached["title"], "Title")
        self.assertEqual(cached["duration"], 123)
        self.assertEqual(cached["ext"], "vtt")
        self.assertIn("hello", cached["raw"])

    def test_audio_round_trip_requires_existing_file(self):
        url = "https://youtu.be/7-cyM9maRR0"
        path = os.path.join(self.tmp.name, "audio.webm")
        with open(path, "wb") as f:
            f.write(b"audio")
        youtube_cache.save_audio(url, "Title", 456, path)

        cached = youtube_cache.get_cached_audio(url)

        self.assertEqual(cached["title"], "Title")
        self.assertEqual(cached["duration"], 456)
        self.assertEqual(cached["audio_path"], path)


if __name__ == "__main__":
    unittest.main()
