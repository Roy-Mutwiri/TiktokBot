import os
import sys
import tempfile
import unittest
from unittest import mock


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import youtube_cache_janitor as janitor  # noqa: E402


def _write(path, megabytes, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\0" * int(megabytes * 1024 * 1024))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class CacheJanitorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_budget_defaults_and_can_be_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AVATAR_YOUTUBE_CACHE_GB", None)
            self.assertEqual(
                janitor.budget_bytes(),
                int(janitor.DEFAULT_BUDGET_GB * 1024 ** 3))
        with mock.patch.dict(os.environ, {"AVATAR_YOUTUBE_CACHE_GB": "0"}):
            self.assertEqual(janitor.budget_bytes(), 0)
        with mock.patch.dict(os.environ, {"AVATAR_YOUTUBE_CACHE_GB": "nonsense"}):
            self.assertEqual(
                janitor.budget_bytes(),
                int(janitor.DEFAULT_BUDGET_GB * 1024 ** 3))

    def test_purge_removes_only_unplayable_leftovers(self):
        keep = _write(os.path.join(self.cache, "abc", "preview-hd.mp4"), 1)
        audio = _write(os.path.join(self.cache, "abc", "audio-1-2-3.webm"), 1)
        vocals = _write(os.path.join(self.cache, "abc", "vocals.flac"), 1)
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4.part"), 1)
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4.bad-20260101"), 1)
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4.old-20260101"), 1)
        # Unreachable: the scene only globs "preview-hd.*" and "preview.*".
        _write(os.path.join(self.cache, "abc", "preview-low.mp4"), 1)
        _write(os.path.join(self.cache, "abc", "demucs_stems", "htdemucs",
                            "x", "vocals.wav"), 2)

        files, freed = janitor.purge_junk(self.cache)

        self.assertEqual(files, 5)
        self.assertEqual(freed, 6 * 1024 * 1024)
        self.assertTrue(os.path.exists(keep))
        self.assertTrue(os.path.exists(audio))
        self.assertTrue(os.path.exists(vocals))
        self.assertFalse(os.path.exists(os.path.join(self.cache, "abc", "demucs_stems")))

    def test_eviction_removes_the_least_recently_used_videos_first(self):
        _write(os.path.join(self.cache, "old", "preview-hd.mp4"), 4, mtime=1000)
        _write(os.path.join(self.cache, "middle", "preview-hd.mp4"), 4, mtime=2000)
        fresh = _write(
            os.path.join(self.cache, "fresh", "preview-hd.mp4"), 4, mtime=3000)

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}), \
                mock.patch("youtube_cache_janitor._forget_video"):
            removed, freed, names = janitor.enforce_budget(
                max_bytes=5 * 1024 * 1024, cache_dir=self.cache)

        self.assertEqual(removed, 2)
        self.assertEqual(freed, 8 * 1024 * 1024)
        self.assertEqual(names, ["old", "middle"])
        self.assertTrue(os.path.exists(fresh))

    def test_eviction_never_touches_a_video_in_use(self):
        in_use = _write(
            os.path.join(self.cache, "playing", "preview-hd.mp4"), 4, mtime=1000)
        _write(os.path.join(self.cache, "idle", "preview-hd.mp4"), 4, mtime=2000)

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}), \
                mock.patch("youtube_cache_janitor._forget_video"):
            removed, _freed, names = janitor.enforce_budget(
                max_bytes=5 * 1024 * 1024, cache_dir=self.cache,
                keep_ids=["playing"])

        self.assertEqual(removed, 1)
        self.assertEqual(names, ["idle"])
        self.assertTrue(os.path.exists(in_use))

    def test_eviction_is_skipped_when_the_cache_already_fits(self):
        _write(os.path.join(self.cache, "one", "preview-hd.mp4"), 2)

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}):
            removed, freed, names = janitor.enforce_budget(
                max_bytes=64 * 1024 * 1024, cache_dir=self.cache)

        self.assertEqual((removed, freed, names), (0, 0, []))

    def test_eviction_disabled_by_a_zero_budget(self):
        _write(os.path.join(self.cache, "one", "preview-hd.mp4"), 4)

        with mock.patch.dict(os.environ, {"AVATAR_YOUTUBE_CACHE_GB": "0"}):
            removed, freed, names = janitor.enforce_budget(cache_dir=self.cache)

        self.assertEqual((removed, freed, names), (0, 0, []))

    def test_dry_run_reports_without_deleting(self):
        part = _write(os.path.join(self.cache, "abc", "preview-hd.mp4.part"), 1)
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4"), 4, mtime=1000)
        _write(os.path.join(self.cache, "zzz", "preview-hd.mp4"), 4, mtime=3000)

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}), \
                mock.patch("youtube_cache_janitor.budget_bytes",
                           return_value=5 * 1024 * 1024):
            result = janitor.sweep(cache_dir=self.cache, dry_run=True)

        self.assertEqual(result["junk_files"], 1)
        self.assertEqual(result["evicted"], 1)
        self.assertTrue(os.path.exists(part))
        self.assertTrue(
            os.path.exists(os.path.join(self.cache, "abc", "preview-hd.mp4")))

    def test_sweep_reports_the_space_it_reclaimed(self):
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4.part"), 3)
        _write(os.path.join(self.cache, "abc", "preview-hd.mp4"), 4, mtime=1000)
        _write(os.path.join(self.cache, "zzz", "preview-hd.mp4"), 4, mtime=3000)

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}), \
                mock.patch("youtube_cache_janitor._forget_video"), \
                mock.patch("youtube_cache_janitor.budget_bytes",
                           return_value=5 * 1024 * 1024):
            result = janitor.sweep(cache_dir=self.cache)

        self.assertEqual(result["freed_bytes"], 7 * 1024 * 1024)
        self.assertEqual(result["after_bytes"], 4 * 1024 * 1024)
        self.assertIn("freed", janitor.summary_line(result))

    def test_partial_removal_still_counts_what_it_reclaimed(self):
        locked_dir = os.path.join(self.cache, "locked")
        _write(os.path.join(locked_dir, "preview-hd.mp4"), 4, mtime=1000)
        _write(os.path.join(self.cache, "fresh", "preview-hd.mp4"), 4, mtime=3000)

        def _half_delete(path):
            # Mimic Windows leaving an open file behind after rmtree fails.
            if os.path.basename(path) == "locked":
                return False
            return True

        with mock.patch("youtube_cache_janitor._db_last_used", return_value={}), \
                mock.patch("youtube_cache_janitor._forget_video") as forget, \
                mock.patch("youtube_cache_janitor._remove_tree",
                           side_effect=_half_delete):
            removed, freed, _names = janitor.enforce_budget(
                max_bytes=5 * 1024 * 1024, cache_dir=self.cache)

        # Nothing actually left the disk, so nothing is reported as freed.
        self.assertEqual((removed, freed), (0, 0))
        forget.assert_not_called()

    def test_human_bytes_reads_naturally(self):
        self.assertEqual(janitor.human_bytes(0), "0 B")
        self.assertEqual(janitor.human_bytes(1536), "2 KB")
        self.assertEqual(janitor.human_bytes(5 * 1024 ** 3), "5.0 GB")


if __name__ == "__main__":
    unittest.main()
