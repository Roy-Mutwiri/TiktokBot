# =============================================================================
# tests_hard.py — aggressive correctness/edge-case/stress tests for the
# webcam-avatar code. Designed to actually try to BREAK things:
#   * malformed / empty / huge / unicode text
#   * cache integrity (miss/hit/corruption/cross-backend keys)
#   * thread-safety of runtime backend switching (the race we fixed)
#   * Maya1 SNAC token unpacking on degenerate inputs (pure, no 7GB load)
#   * face-restore on no-face / black / tiny / non-square frames + interval cache
#   * Real-ESRGAN, enhance, compositor, 1€ filter on edge inputs
#
#   python tests_hard.py            # everything that doesn't need the 7GB Maya1
#   python tests_hard.py --heavy    # also load Maya1 + Chatterbox and synth
# =============================================================================
import os
import sys
import time
import threading
import asyncio
import traceback

sys.path.insert(0, "engines")
sys.path.insert(0, ".")
import numpy as np
import cv2

HEAVY = "--heavy" in sys.argv
PASS = 0
FAIL = 0
FAILED = []


def t(name, fn):
    """Run a test fn; PASS if it returns truthy and doesn't raise."""
    global PASS, FAIL
    try:
        r = fn()
        ok = (r is None) or bool(r)
    except Exception as exc:
        ok = False
        print(f"  [FAIL] {name}\n         {type(exc).__name__}: {str(exc)[:160]}")
        FAILED.append(name)
        FAIL += 1
        return
    if ok:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  (returned falsey)")
        FAILED.append(name)
        FAIL += 1


def section(s):
    print("\n" + "=" * 64 + f"\n{s}\n" + "=" * 64)


# =============================================================================
section("A) COMPILE + IMPORT (webcam path; autonomous must be gone)")
import py_compile
WEBCAM_FILES = ["avatar_studio.py", "realtime_avatar.py", "control_gui.py",
                "build_character.py", "preview_avatar.py", "setup_models.py",
                "engines/tts_stream_engine.py", "engines/maya1_tts.py",
                "engines/chatterbox_tts.py", "engines/face_restore_engine.py",
                "engines/upscale_engine.py", "engines/musetalk_engine.py",
                "engines/compositor.py", "engines/one_euro.py",
                "engines/enhance_engine.py", "engines/wav2lip_engine.py"]


def _compile_all():
    for f in WEBCAM_FILES:
        py_compile.compile(f, doraise=True)
    return True
t("all webcam source files compile", _compile_all)

t("autonomous track is GONE", lambda: not any(os.path.exists(p) for p in
  ["autonomous_avatar.py", "engines/behavior_engine.py",
   "engines/motion_library.py", "harvester", "motion_clips"]))

import tts_stream_engine as TTS
t("tts_stream_engine imports", lambda: hasattr(TTS, "TTSStreamEngine"))
import maya1_tts
t("maya1_tts imports (no model load)", lambda: hasattr(maya1_tts, "Maya1TTS"))
import face_restore_engine
import upscale_engine
import enhance_engine
import one_euro
t("realtime/studio import (incl MuseTalk wav2lip-fallback chain)",
  lambda: __import__("avatar_studio") and __import__("realtime_avatar"))

# =============================================================================
section("B) Maya1 SNAC unpacking — pure functions, degenerate inputs")
M = maya1_tts.Maya1TTS
OFF = maya1_tts.CODE_TOKEN_OFFSET
t("_unpack([]) -> three empty levels",
  lambda: M._unpack([]) == [[], [], []])
t("_unpack(<7 tokens) -> empty (no full frame)",
  lambda: M._unpack([OFF] * 6) == [[], [], []])
t("_unpack(exactly 7) -> 1+2+4 codes",
  lambda: [len(x) for x in M._unpack([OFF + i for i in range(7)])] == [1, 2, 4])
t("_unpack(10 = 1 frame + 3 leftover) -> drops partial frame",
  lambda: [len(x) for x in M._unpack([OFF] * 10)] == [1, 2, 4])
t("_unpack codes are in [0,4096)",
  lambda: all(0 <= c < 4096 for lvl in M._unpack([OFF + 99999] * 7) for c in lvl))
t("_extract stops at CODE_END",
  lambda: M._extract([OFF, OFF + 1, maya1_tts.CODE_END_TOKEN_ID, OFF + 2]) == [OFF, OFF + 1])
t("_extract filters out-of-range tokens",
  lambda: M._extract([5, OFF, 999999, OFF + 3]) == [OFF, OFF + 3])

# =============================================================================
section("C) TTS — emotion-tag stripping (must never read tags aloud)")
strip = TTS.TTSStreamEngine._strip_emotion_tags
t("strips a single tag", lambda: "<" not in strip("hi <laugh> there"))
t("strips multiple tags",
  lambda: strip("a <laugh> b <sigh> c").replace(" ", "") == "abc")
t("text with NO tags is unchanged-ish",
  lambda: strip("plain trading talk").strip() == "plain trading talk")
t("only-tags collapses to empty/space",
  lambda: strip("<laugh><sigh>").strip() == "")
t("does NOT strip math/comparison '2 < 3 and 5 > 1'",
  lambda: "<" in strip("2 < 3 and 5 > 1") or ">" in strip("2 < 3 and 5 > 1"))
t("over-long pseudo-tag is left alone",
  lambda: "<" in strip("a <thisisaverylongnotag12345> b"))

# =============================================================================
section("D) TTS — resample correctness (edge sample rates)")
rs = TTS.TTSStreamEngine._resample
t("24k->16k length ratio ~2/3",
  lambda: abs(len(rs(np.zeros(2400, np.float32), 24000, 16000)) - 1600) <= 4)
t("48k->16k length ratio ~1/3",
  lambda: abs(len(rs(np.zeros(4800, np.float32), 48000, 16000)) - 1600) <= 4)
t("8k->16k upsamples ~2x",
  lambda: abs(len(rs(np.zeros(800, np.float32), 8000, 16000)) - 1600) <= 4)
t("resample preserves a sine roughly (no NaN/inf)",
  lambda: np.all(np.isfinite(rs(np.sin(np.linspace(0, 50, 2400)).astype(np.float32), 24000, 16000))))
t("resample 1-sample doesn't crash",
  lambda: rs(np.zeros(1, np.float32), 24000, 16000) is not None)

# =============================================================================
section("E) TTS — engine: backends, cache integrity, weird inputs, threads")


class Mouth:
    def __init__(self):
        self.fed = 0
    def feed_audio(self, x):
        self.fed += 1


eng = TTS.TTSStreamEngine(Mouth())
eng.set_backend("kokoro")
eng.warm_backend()


def synth(text):
    fut = asyncio.run_coroutine_threadsafe(eng._synthesize(text), eng._loop)
    return fut.result(60)


def _fresh(text):
    k = eng._cache_key(text)
    if os.path.exists(k):
        os.remove(k)
    return text

t("set_backend rejects junk, keeps current",
  lambda: eng.set_backend("garbage")[0] is False and eng.tts_backend == "kokoro")
t("set_backend accepts all valid",
  lambda: all(eng.set_backend(b)[0] for b in ("kokoro", "edge", "maya1", "chatterbox", "auto")))
eng.set_backend("kokoro")

ln = _fresh("Cache integrity test line one.")
t("kokoro fresh synth produces audio", lambda: len(synth(ln)) > 1600)
t("cache file now exists", lambda: os.path.exists(eng._cache_key(ln)))


def _cache_fast():
    a = time.time(); synth(ln); return (time.time() - a) < 0.2
t("cached synth < 200ms", _cache_fast)

t("corrupted cache file -> graceful re-synth (no crash)",
  lambda: (open(eng._cache_key(ln), "wb").write(b"NOT A WAV") and False) or
          len(synth(ln)) > 1600)

t("empty string -> None (no crash)", lambda: synth("") is None)
t("whitespace only -> None", lambda: synth("   \n  ") is None)
t("only-emotion-tags (kokoro) doesn't crash",
  lambda: synth(_fresh("<laugh> <sigh>")) is not None or True)
t("very long text (1500 chars) synths",
  lambda: len(synth(_fresh("Gold is moving. " * 100))) > 1600)
t("unicode/emoji text synths",
  lambda: len(synth(_fresh("Café résumé naïve 你好 — gold up 📈 today"))) > 800)
t("punctuation-heavy text synths",
  lambda: len(synth(_fresh("Wait... really?! $2,050.50 — up +3.2% (huge)."))) > 800)

# cross-backend cache keys must differ for same text
keys = set()
for b in ("kokoro", "edge", "maya1", "chatterbox"):
    eng.set_backend(b); keys.add(eng._cache_key("same text"))
t("cache key unique per backend (4 distinct)", lambda: len(keys) == 4)
eng.set_backend("kokoro")

# thread-safety: hammer set_backend from many threads while synthesizing
def _hammer():
    import random
    stop = time.time() + 1.5
    i = 0
    while time.time() < stop:
        eng.set_backend(("kokoro", "edge", "auto")[i % 3]); i += 1
def _race():
    threads = [threading.Thread(target=_hammer) for _ in range(6)]
    [th.start() for th in threads]
    eng.set_backend("kokoro")
    out = synth(_fresh("Thread safety probe line."))
    [th.join() for th in threads]
    return out is not None and len(out) > 800
t("rapid concurrent set_backend during synth (no crash/deadlock)", _race)

# load lock present (the race fix)
t("load lock exists", lambda: hasattr(eng, "_load_lock"))
t("synthesizing flag is False at rest", lambda: eng.synthesizing is False)
eng.shutdown()

# fresh-process cache persistence: a NEW engine instance reads the old cache
eng2 = TTS.TTSStreamEngine(Mouth()); eng2.set_backend("kokoro")
t("cache persists across engine instances",
  lambda: eng2._cache_load("Cache integrity test line one.") is not None)
eng2.shutdown()

# =============================================================================
section("F) FACE RESTORE (CodeFormer) — degenerate frames + interval cache")
fr = face_restore_engine.FaceRestoreEngine()
if fr.ready:
    t("no-face frame (random noise) returns same-shape, no crash",
      lambda: fr.restore((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).shape == (512, 512, 3))
    t("all-black frame survives",
      lambda: fr.restore(np.zeros((512, 512, 3), np.uint8)).shape == (512, 512, 3))
    t("non-square frame is handled",
      lambda: fr.restore((np.random.rand(480, 640, 3) * 255).astype(np.uint8)) is not None)
    t("tiny frame survives",
      lambda: fr.restore((np.random.rand(32, 32, 3) * 255).astype(np.uint8)) is not None)
    char = "ai-face/character.jpg" if os.path.exists("ai-face/character.jpg") else "character.jpg"
    img = cv2.resize(cv2.imread(char), (512, 512))
    t("real face restore returns 512x512",
      lambda: fr.restore(img).shape == (512, 512, 3))
    # interval cache: with RESTORE_INTERVAL>1, some frames return the cache
    fr._frame_counter = 0
    t("interval caching returns frames each call",
      lambda: all(fr.restore(img) is not None for _ in range(5)))
else:
    print("  [skip] CodeFormer not ready")

# =============================================================================
section("G) UPSCALE (Real-ESRGAN) — sizes + off-state safety")
up = upscale_engine.UpscaleEngine()
if up.ready:
    t("upscale 512 frame -> 512",
      lambda: up.process_frame((np.random.rand(512, 512, 3) * 255).astype(np.uint8)).shape == (512, 512, 3))
    t("upscale non-512 handled",
      lambda: up.process_frame((np.random.rand(300, 400, 3) * 255).astype(np.uint8)) is not None)
else:
    print("  [skip] Real-ESRGAN not ready")

# =============================================================================
section("H) ENHANCE — sizes, speaking flag, never raises")
t("enhance 512 speaking",
  lambda: enhance_engine.enhance_frame(np.full((512, 512, 3), 90, np.uint8), True).shape == (512, 512, 3))
t("enhance 512 idle",
  lambda: enhance_engine.enhance_frame(np.full((512, 512, 3), 90, np.uint8), False) is not None)
t("enhance non-512 resized to 512",
  lambda: enhance_engine.enhance_frame(np.full((300, 300, 3), 90, np.uint8)).shape[:2] == (512, 512))
t("enhance black frame",
  lambda: enhance_engine.enhance_frame(np.zeros((512, 512, 3), np.uint8)) is not None)

# =============================================================================
section("I) 1€ FILTER — smooths noise, tracks steps, finite")
of = one_euro.OneEuroFilter(min_cutoff=1.0, beta=0.0)


def _smooths():
    of.reset()
    rng = np.random.RandomState(0)
    raw = 100.0 + rng.randn(200) * 5.0      # noisy constant
    out = np.array([of(float(v), i / 30.0) for i, v in enumerate(raw)])
    # filtered signal should have smaller variance than the raw noise
    return out[20:].std() < raw[20:].std()
t("reduces variance of noisy constant", _smooths)


def _tracks_step():
    of.reset()
    for i in range(60):
        y = of(0.0 if i < 30 else 10.0, i / 30.0)
    return abs(y - 10.0) < 2.0              # converged near the new level
t("tracks a step change", _tracks_step)
t("output always finite",
  lambda: np.isfinite(one_euro.OneEuroFilter()(123.4, 0.0)))

# =============================================================================
section("J) COMPOSITOR — bbox detect + blend keep frame valid")
try:
    from compositor import Compositor
    comp = Compositor()
    char = "ai-face/character.jpg" if os.path.exists("ai-face/character.jpg") else "character.jpg"
    face = cv2.resize(cv2.imread(char), (512, 512))
    bbox = comp.detect_mouth_bbox(face)
    t("detect_mouth_bbox returns a 4-tuple or None",
      lambda: bbox is None or (len(bbox) == 4))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = face[y1:y2, x1:x2].copy()
        t("blend_mouth returns same-shape frame",
          lambda: comp.blend_mouth(face, crop, bbox).shape == face.shape)
    t("detect on black frame doesn't crash",
      lambda: comp.detect_mouth_bbox(np.zeros((512, 512, 3), np.uint8)) is None or True)
except Exception as exc:
    print(f"  [skip] compositor: {type(exc).__name__}: {str(exc)[:80]}")

# =============================================================================
section("K) HEAVY (only with --heavy): Maya1 + Chatterbox real synth")
if HEAVY:
    e = TTS.TTSStreamEngine(Mouth())
    e.set_backend("maya1")
    e.warm_backend()

    def _esynth(txt):
        f = asyncio.run_coroutine_threadsafe(e._synthesize(txt), e._loop)
        return f.result(120)
    if e._maya1 is not None:
        k = e._cache_key("Right <laugh> gold just broke out.")
        if os.path.exists(k):
            os.remove(k)
        t("Maya1 synth w/ <laugh> -> audio", lambda: len(_esynth("Right <laugh> gold just broke out.")) > 1600)
        t("Maya1 long line doesn't OOM-crash (returns audio or graceful None)",
          lambda: _esynth("Gold is moving. " * 40) is not None or True)
    else:
        print("  [skip] Maya1 did not load")
    e.set_backend("chatterbox"); e.warm_backend()
    if e._chatter is not None:
        t("Chatterbox synth -> audio",
          lambda: len(_esynth(_fresh("Welcome back to the stream."))) > 1600)
    else:
        print("  [skip] Chatterbox did not load")
    e.shutdown()
else:
    print("  [skip] run with --heavy to load Maya1/Chatterbox")

# =============================================================================
print("\n" + "#" * 64)
print(f"RESULT: {PASS} passed, {FAIL} failed  (of {PASS + FAIL})")
for n in FAILED:
    print("   FAILED:", n)
print("#" * 64)
sys.exit(1 if FAIL else 0)
