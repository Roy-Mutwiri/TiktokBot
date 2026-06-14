import asyncio
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np

from engines import tts_stream_engine
from engines import reactions


class FakeMouth:
    def __init__(self):
        self.starts = []
        self.ends = 0

    def start_audio(self, pcm, start_time=None):
        self.starts.append((np.asarray(pcm).copy(), start_time))

    def end_audio(self):
        self.ends += 1


def bare_engine():
    engine = tts_stream_engine.TTSStreamEngine.__new__(
        tts_stream_engine.TTSStreamEngine
    )
    engine.mouth = FakeMouth()
    engine.behavior = None
    engine.muted = True
    engine._running = True
    engine._hifi = {}
    engine._prepared_audio = {}
    engine._prepared_lock = threading.Lock()
    engine._interrupt_serial = 0
    engine._audio_stream = None
    engine._audio_lock = threading.Lock()
    return engine


class TTSFluencyTests(unittest.TestCase):
    def test_interrupt_current_aborts_audio_and_ends_mouth(self):
        engine = bare_engine()

        class FakeStream:
            def __init__(self):
                self.aborted = 0
                self.closed = 0

            def abort(self):
                self.aborted += 1

            def close(self):
                self.closed += 1

        stream = FakeStream()
        engine._audio_stream = stream
        engine.synthesizing = True
        engine.speaking = True
        engine.interrupt_current()
        self.assertEqual(stream.aborted, 1)
        self.assertEqual(stream.closed, 1)
        self.assertEqual(engine.mouth.ends, 1)
        self.assertFalse(engine.speaking)

    def test_supporter_response_bank_has_no_dynamic_placeholders(self):
        lines = reactions.ready_lines()
        self.assertGreaterEqual(len(lines), 220)
        self.assertTrue(all("{" not in line and "}" not in line for line in lines))
        self.assertIn("Thank you", lines)
        self.assertIn("Maya", reactions.ready_follow("Maya"))
        self.assertIn("Ahmed", reactions.ready_gift("Ahmed", "Rose", 5000))
        lengths = [len(line.split()) for line in lines]
        self.assertLessEqual(min(lengths), 3)
        self.assertGreaterEqual(max(lengths), 15)

    def test_ready_reactions_shuffle_without_immediate_repeats(self):
        reactions.reload()
        follow_lines = [reactions.ready_follow("Maya") for _ in range(100)]
        share_lines = [reactions.ready_share("Maya") for _ in range(100)]

        self.assertEqual(len(follow_lines), len(set(follow_lines)))
        self.assertEqual(len(share_lines), len(set(share_lines)))

    def test_generated_event_templates_do_not_repeat_until_exhausted(self):
        reactions.reload()
        follow_lines = [reactions.follow("Maya") for _ in range(120)]
        gift_lines = [reactions.gift("Ahmed", "Rose", 1, 100) for _ in range(50)]

        self.assertEqual(len(follow_lines), len(set(follow_lines)))
        self.assertEqual(len(gift_lines), len(set(gift_lines)))

    def test_mass_follow_batch_mentions_extra_followers_once(self):
        line = reactions.follow_many(
            ["Maya", "Ahmed", "Sara", "Omar", "Nora", "Ali"])

        self.assertIn("Maya", line)
        self.assertIn("plus 2 more", line)

    def test_personalized_reaction_forces_name_into_its_own_chunk(self):
        engine = bare_engine()
        chunks = engine._split_for_tts(reactions.ready_follow("Muhammad Fajar"))
        self.assertEqual(chunks[1], "Muhammad Fajar.")
        self.assertEqual(len(chunks), 3)

    def test_is_prepared_requires_every_playback_chunk(self):
        engine = bare_engine()
        engine._split_for_tts = lambda text: ["one", "two"]
        engine._cache_load = lambda text: engine._prepared_audio.get(text)
        engine._prepared_audio["one"] = np.ones(32, np.float32)
        self.assertFalse(engine.is_prepared("line"))
        engine._prepared_audio["two"] = np.ones(32, np.float32)
        self.assertTrue(engine.is_prepared("line"))

    def test_prepare_renders_the_same_chunks_used_for_playback(self):
        engine = bare_engine()
        engine._loop = None
        rendered = []
        engine._split_for_tts = lambda text: ["first chunk", "second chunk"]
        engine._cache_load = lambda text: engine._prepared_audio.get(text)

        def synth(text):
            rendered.append(text)
            pcm = np.ones(320, np.float32)
            engine._prepared_audio[text] = pcm
            return pcm

        engine._synthesize_blocking = synth
        loop_ready = threading.Event()

        def run_loop():
            engine._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(engine._loop)
            loop_ready.set()
            engine._loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        loop_ready.wait(timeout=2)
        try:
            self.assertTrue(engine.prepare("full response", timeout=2))
            self.assertEqual(rendered, ["first chunk", "second chunk"])
        finally:
            engine._loop.call_soon_threadsafe(engine._loop.stop)
            thread.join(timeout=2)
            engine._loop.close()

    def test_conditioner_removes_dc_invalid_values_and_clipping(self):
        pcm = np.array([np.nan, np.inf, -np.inf, 2.0, -2.0, 0.5], np.float32)
        clean = tts_stream_engine.TTSStreamEngine._condition_playback(pcm)
        self.assertTrue(np.all(np.isfinite(clean)))
        self.assertLessEqual(float(np.max(np.abs(clean))), 0.94001)
        self.assertAlmostEqual(float(np.mean(clean)), 0.0, places=5)

    def test_nonfinal_chunk_has_no_tail_pause_or_mouth_end(self):
        engine = bare_engine()
        pcm = np.ones(1280, np.float32) * 0.1
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with (
            mock.patch.object(tts_stream_engine.asyncio, "sleep", fake_sleep),
            mock.patch.object(tts_stream_engine.time, "monotonic", return_value=10.0),
        ):
            asyncio.run(engine._play_and_feed(pcm, final_chunk=False))

        self.assertEqual(engine.mouth.ends, 0)
        self.assertNotIn(tts_stream_engine.TAIL_SILENCE, sleeps)

    def test_final_chunk_ends_mouth_once_with_short_tail(self):
        engine = bare_engine()
        pcm = np.ones(640, np.float32) * 0.1
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with (
            mock.patch.object(tts_stream_engine.asyncio, "sleep", fake_sleep),
            mock.patch.object(tts_stream_engine.time, "monotonic", return_value=10.0),
        ):
            asyncio.run(engine._play_and_feed(pcm, final_chunk=True))

        self.assertEqual(engine.mouth.ends, 1)
        self.assertIn(tts_stream_engine.TAIL_SILENCE, sleeps)
        self.assertLessEqual(tts_stream_engine.TAIL_SILENCE, 0.10)

    def test_direct_playback_reuses_one_float_stream(self):
        created = []

        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = 0
                self.writes = []
                created.append(self)

            def start(self):
                self.started += 1

            def write(self, audio):
                self.writes.append(np.asarray(audio).copy())

        fake_sd = types.SimpleNamespace(OutputStream=FakeStream)
        engine = bare_engine()
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            engine._play_direct(np.ones(1600, np.float32) * 0.1, 16000)
            engine._play_direct(np.ones(1600, np.float32) * 0.2, 16000)

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].started, 1)
        self.assertEqual(len(created[0].writes), 2)
        self.assertEqual(created[0].kwargs["dtype"], "float32")

    def test_tts_chunks_are_bounded_for_stable_voice_generation(self):
        engine = bare_engine()
        text = (
            "Gold is testing resistance while buyers defend the intraday trend. "
            "Watch the next candle closely because momentum remains constructive. "
            "Risk management still matters, so wait for confirmation before entry."
        )
        chunks = engine._split_for_tts(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= engine._TTS_CHUNK_CHARS for chunk in chunks))

    def test_worker_keeps_one_speaking_session_across_three_chunks(self):
        engine = bare_engine()
        engine.speech_queue = None
        engine.synthesizing = False
        engine.speaking = False
        engine.behavior = None
        engine._split_for_tts = lambda text: ["one", "two", "three"]
        states = []
        played = []

        def synth(text):
            return np.ones(640, np.float32) * (len(text) / 10.0)

        async def play(pcm, final_chunk=True):
            played.append((final_chunk, engine.speaking))

        original_set = engine._set_speaking

        def set_speaking(value):
            original_set(value)
            states.append(bool(value))

        engine._synthesize_blocking = synth
        engine._synthesize = lambda text: None
        engine._play_and_feed = play
        engine._set_speaking = set_speaking

        async def run_worker():
            engine.speech_queue = asyncio.Queue()
            await engine.speech_queue.put((0, "line"))
            await engine.speech_queue.put(None)
            await engine._stream_worker()

        asyncio.run(run_worker())
        self.assertEqual(played, [(False, True), (False, True), (True, True)])
        self.assertEqual(states, [True, False])

    def test_speak_reports_when_queue_accepts_the_line(self):
        engine = bare_engine()
        engine.behavior = None
        engine.words_spoken = 0
        engine.lines_spoken = 0
        loop_ready = threading.Event()

        def run_loop():
            engine._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(engine._loop)
            engine.speech_queue = asyncio.Queue()
            loop_ready.set()
            engine._loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        loop_ready.wait(timeout=2)
        try:
            self.assertTrue(engine.speak("ready line", priority=2))
            queued = asyncio.run_coroutine_threadsafe(
                engine.speech_queue.get(), engine._loop
            ).result(timeout=2)
            self.assertEqual(queued, (2, "ready line"))
        finally:
            engine._loop.call_soon_threadsafe(engine._loop.stop)
            thread.join(timeout=2)
            engine._loop.close()


if __name__ == "__main__":
    unittest.main()
