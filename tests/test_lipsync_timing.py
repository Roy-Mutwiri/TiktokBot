import collections
import threading
import unittest
from unittest import mock

import numpy as np

from engines import musetalk_engine
from engines import wav2lip_engine


def _bare_wav2lip():
    engine = wav2lip_engine.Wav2LipEngine.__new__(wav2lip_engine.Wav2LipEngine)
    engine.audio_buffer = collections.deque(maxlen=wav2lip_engine.SAMPLE_RATE * 2)
    engine._audio_lock = threading.Lock()
    engine._timeline_audio = None
    engine._timeline_start = 0.0
    engine._last_audio_feed = 0.0
    engine._silence_frames = 0
    engine.is_speaking = False
    return engine


def _bare_musetalk():
    engine = musetalk_engine.MuseTalkEngine.__new__(musetalk_engine.MuseTalkEngine)
    engine.audio_buffer = collections.deque(maxlen=musetalk_engine.SAMPLE_RATE * 2)
    engine._audio_lock = threading.Lock()
    engine._timeline_audio = None
    engine._timeline_start = 0.0
    engine._last_audio_feed = 0.0
    engine._silence = 0
    engine.is_speaking = False
    engine._w2l = None
    return engine


class LipSyncTimingTests(unittest.TestCase):
    def test_wav2lip_window_tracks_elapsed_utterance_time(self):
        engine = _bare_wav2lip()
        pcm = np.arange(wav2lip_engine.SAMPLE_RATE, dtype=np.float32)
        engine.start_audio(pcm, start_time=100.0)

        early = engine._current_audio_window(now=100.25)
        self.assertEqual(len(early), wav2lip_engine.AUDIO_WINDOW)
        np.testing.assert_array_equal(early[-4000:], pcm[:4000])

        later = engine._current_audio_window(now=100.50)
        np.testing.assert_array_equal(later, pcm[1600:8000])

    def test_streaming_audio_expires_when_no_new_chunks_arrive(self):
        engine = _bare_wav2lip()
        chunk = np.ones(wav2lip_engine.MIN_AUDIO, dtype=np.float32)
        with mock.patch.object(wav2lip_engine.time, "monotonic", return_value=10.0):
            engine.feed_audio(chunk)
        with mock.patch.object(wav2lip_engine.time, "monotonic", return_value=10.1):
            self.assertTrue(engine._audio_active())
        with mock.patch.object(wav2lip_engine.time, "monotonic", return_value=11.0):
            self.assertFalse(engine._audio_active())

    def test_new_utterance_resets_mouth_temporal_history(self):
        engine = _bare_musetalk()
        engine._prev_mouth = np.ones((2, 2, 3), dtype=np.uint8)
        engine._neutral = np.ones((2, 2, 3), dtype=np.float32)
        engine._mt_last = np.ones((2, 2, 3), dtype=np.uint8)

        engine.start_audio(np.ones(3200, dtype=np.float32), start_time=50.0)

        self.assertIsNone(engine._prev_mouth)
        self.assertIsNone(engine._neutral)
        self.assertIsNone(engine._mt_last)
        engine.end_audio()
        self.assertFalse(engine.is_speaking)
        self.assertEqual(len(engine.audio_buffer), 0)


if __name__ == "__main__":
    unittest.main()
