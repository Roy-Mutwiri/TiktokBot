import hashlib
import json
import os
import random
import threading
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from engines import bg_music


MANIFEST = os.path.join(bg_music.MUSIC_DIR, "manifest.json")


@unittest.skipUnless(os.path.exists(MANIFEST), "real music library not installed")
class RealMusicLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MANIFEST, encoding="utf-8") as source:
            cls.manifest = json.load(source)
        cls.tracks = cls.manifest["tracks"]

    def test_library_has_at_least_100_unique_real_tracks(self):
        files = bg_music._discover_music_files()
        self.assertGreaterEqual(len(files), 100)
        self.assertEqual(len(files), len(set(map(os.path.basename, files))))

    def test_every_manifest_hash_matches_and_file_decodes(self):
        hashes = set()
        for item in self.tracks:
            path = os.path.join(bg_music.MUSIC_DIR, item["file"])
            with open(path, "rb") as source:
                digest = hashlib.sha256(source.read()).hexdigest()
            self.assertEqual(digest, item["sha256"], item["file"])
            self.assertNotIn(digest, hashes, item["file"])
            hashes.add(digest)
            info = sf.info(path)
            self.assertGreater(info.duration, 60.0, item["file"])
            self.assertGreaterEqual(info.samplerate, 22050, item["file"])

    def test_sampled_tracks_are_acoustically_distinct(self):
        sample = random.Random(42).sample(self.tracks, 16)
        fingerprints = []
        for item in sample:
            path = os.path.join(bg_music.MUSIC_DIR, item["file"])
            audio, rate = sf.read(
                path, frames=rate_or_default(path) * 20,
                dtype="float32", always_2d=True,
            )
            mono = audio.mean(axis=1)
            windows = np.array_split(mono, 80)
            fingerprints.append(np.array([
                np.sqrt(np.mean(window * window)) for window in windows
            ]))
        similarities = []
        for index, first in enumerate(fingerprints):
            for second in fingerprints[index + 1:]:
                similarities.append(float(np.corrcoef(first, second)[0, 1]))
        self.assertLess(float(np.median(similarities)), 0.85)

    def test_file_selector_is_random_and_respects_cooldown(self):
        music = bg_music.BackgroundMusic.__new__(bg_music.BackgroundMusic)
        music._library = bg_music._discover_music_files()[:20]
        music._hist = {}
        music._history_lock = threading.Lock()
        music._rng = random.Random(91)
        now = 2_000_000.0
        picks = []
        with mock.patch.object(bg_music.time, "time", return_value=now):
            for _ in range(12):
                path = music._pick_file(exclude=tuple(map(bg_music._file_track_id, picks)))
                self.assertIsNotNone(path)
                picks.append(path)
                music._hist[bg_music._file_track_id(path)] = now
        self.assertEqual(len(picks), len(set(picks)))
        self.assertNotEqual(picks, sorted(picks))

    def test_loaded_file_uses_its_full_duration(self):
        path = bg_music._discover_music_files()[0]
        audio, meta = bg_music._load_music_file(path)
        music = bg_music.BackgroundMusic.__new__(bg_music.BackgroundMusic)
        music._library = [path]
        music._cur = audio
        music.meta = meta
        self.assertAlmostEqual(music._current_song_seconds(), meta["seconds"], places=3)
        self.assertGreater(meta["seconds"], 60.0)

    def test_exhausted_real_library_does_not_replay_early(self):
        music = bg_music.BackgroundMusic.__new__(bg_music.BackgroundMusic)
        music._library = bg_music._discover_music_files()[:3]
        music._hist = {
            bg_music._file_track_id(path): 2_000_000.0
            for path in music._library
        }
        music._history_lock = threading.Lock()
        music._rng = random.Random(5)
        with mock.patch.object(bg_music.time, "time", return_value=2_000_001.0):
            self.assertIsNone(music._pick_file())
            self.assertIsNotNone(music._pick_seed())


def rate_or_default(path):
    return sf.info(path).samplerate


if __name__ == "__main__":
    unittest.main()
