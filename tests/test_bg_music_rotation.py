import random
import threading
import unittest
from unittest import mock

import numpy as np

from engines import bg_music


def bare_music(history=None):
    music = bg_music.BackgroundMusic.__new__(bg_music.BackgroundMusic)
    music._hist = {} if history is None else dict(history)
    music._history_lock = threading.Lock()
    music._state_lock = threading.Lock()
    music._rng = random.Random(17)
    music._library = []
    return music


def audio_fingerprint(track, bins=96):
    """Low-resolution energy shape captures rhythm and arrangement differences."""
    usable = track[: (len(track) // bins) * bins]
    energy = np.sqrt(np.mean(usable.reshape(bins, -1) ** 2, axis=1))
    peak = float(np.max(energy)) or 1.0
    return energy / peak


class BackgroundMusicRotationTests(unittest.TestCase):
    def test_recent_tracks_are_not_eligible(self):
        now = 1_000_000.0
        music = bare_music({
            "1": now - bg_music.REPEAT_GAP + 1,
            "2": now - bg_music.REPEAT_GAP,
        })
        with (
            mock.patch.object(bg_music, "NUM_SONGS", 2),
            mock.patch.object(bg_music.time, "time", return_value=now),
        ):
            self.assertEqual(music._pick_seed(), 2)

    def test_cooldown_is_strict_when_every_track_is_recent(self):
        now = 1_000_000.0
        music = bare_music({"1": now, "2": now})
        with (
            mock.patch.object(bg_music, "NUM_SONGS", 2),
            mock.patch.object(bg_music.time, "time", return_value=now),
        ):
            self.assertIsNone(music._pick_seed())

    def test_selection_is_random_not_playlist_order(self):
        music = bare_music()
        with (
            mock.patch.object(bg_music, "NUM_SONGS", 20),
            mock.patch.object(bg_music.time, "time", return_value=1_000_000.0),
        ):
            picks = []
            for _ in range(8):
                seed = music._pick_seed(exclude=picks)
                picks.append(seed)
        self.assertEqual(len(set(picks)), len(picks))
        self.assertNotEqual(picks, sorted(picks))

    def test_startup_can_avoid_the_previous_style(self):
        music = bare_music()
        with (
            mock.patch.object(bg_music, "NUM_SONGS", 20),
            mock.patch.object(bg_music.time, "time", return_value=1_000_000.0),
        ):
            seed = music._pick_seed(avoid_styles=("melodic",))
        self.assertNotEqual(bg_music._style_for_seed(seed), "melodic")

    def test_recent_styles_are_unique_and_newest_first(self):
        music = bare_music({"1": 100.0, "3": 300.0, "4": 200.0, "7": 400.0})
        expected = []
        for seed in (7, 3, 4, 1):
            style = bg_music._style_for_seed(seed)
            if style not in expected:
                expected.append(style)
        self.assertEqual(music._recent_styles(4), tuple(expected))

    def test_random_start_position_uses_song_sections(self):
        music = bare_music()
        positions = {music._random_start_position(np.zeros(800)) for _ in range(8)}
        self.assertTrue(positions)
        self.assertTrue(all(position % 100 == 0 for position in positions))

    def test_genres_have_distinct_compositions_and_audio_shapes(self):
        seeds_by_style = {}
        seed = 1
        while len(seeds_by_style) < len(bg_music.STYLES):
            style = bg_music._style_for_seed(seed)
            seeds_by_style.setdefault(style, seed)
            seed += 1

        fingerprints = {}
        identities = set()
        for style, style_seed in seeds_by_style.items():
            track, meta = bg_music._make_track(style_seed)
            fingerprints[style] = audio_fingerprint(track)
            identities.add((meta["groove"], meta["voice"]))

        self.assertGreaterEqual(len(identities), 8)
        pairs = []
        styles = list(fingerprints)
        for index, first in enumerate(styles):
            for second in styles[index + 1:]:
                distance = float(np.mean(np.abs(
                    fingerprints[first] - fingerprints[second]
                )))
                pairs.append(distance)
        self.assertGreater(float(np.median(pairs)), 0.08)

    def test_expired_deadline_switches_on_the_next_audio_block(self):
        music = bare_music()
        music._cur = np.full(16, 0.1, np.float32)
        music.meta = {"style": "dark", "bpm": 120}
        music._cur_seed = 1
        music._next = (
            np.full(16, 0.7, np.float32),
            {"style": "house", "bpm": 126},
        )
        music._next_seed = 2
        music._played_pending = None
        music._song_until = 99.0
        music._pos = 0
        music.base_vol = 1.0
        music.active = True
        music._speaking = False
        music._gain = 1.0
        out = np.zeros((4, 1), np.float32)

        with (
            mock.patch.object(bg_music.time, "monotonic", return_value=100.0),
            mock.patch.object(bg_music.time, "time", return_value=2_000_000.0),
        ):
            music._callback(out, 4, None, None)

        self.assertEqual(music._cur_seed, 2)
        self.assertEqual(music._played_pending, (2, 2_000_000.0))
        np.testing.assert_allclose(out[:, 0], 0.7)

    def test_skip_expires_the_current_deadline(self):
        music = bare_music()
        with mock.patch.object(bg_music.time, "monotonic", return_value=50.0):
            music.skip()
        self.assertLess(music._song_until, 50.0)


if __name__ == "__main__":
    unittest.main()
