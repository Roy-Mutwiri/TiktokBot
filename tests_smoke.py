# =============================================================================
# tests_smoke.py — headless smoke tests for the voice + realism integration.
# Exercises everything that doesn't need a GUI/webcam: imports, TTS backends,
# the synthesis cache, emotion-tag stripping, runtime backend switching, and the
# face-restore engine. Prints PASS/FAIL per check and a final summary.
#
#   python tests_smoke.py            # TTS + restore (fast; loads Kokoro, edge)
#   python tests_smoke.py --maya1    # also load + synth Maya1 (heavy, ~30s, 7GB)
# =============================================================================
import os
import sys
import time
import asyncio

sys.path.insert(0, "engines")
sys.path.insert(0, ".")
import numpy as np

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


print("=" * 60)
print("1) COMPILE + IMPORT all modules")
print("=" * 60)
import py_compile
PYFILES = ["avatar_studio.py", "realtime_avatar.py",
           "engines/tts_stream_engine.py", "engines/maya1_tts.py",
           "engines/chatterbox_tts.py", "engines/face_restore_engine.py",
           "engines/upscale_engine.py"]
ok_compile = True
for f in PYFILES:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as exc:
        ok_compile = False
        print(f"      compile error in {f}: {exc}")
check("all source files compile", ok_compile)

# import the modules (catches import-time errors the GUI would hit)
try:
    import tts_stream_engine as TTS
    import face_restore_engine          # noqa
    import upscale_engine                # noqa
    import avatar_studio                 # imports realtime_avatar, PIL, etc.
    check("engine + studio modules import", True)
    check("studio voice modes defined",
          len(avatar_studio.VOICE_MODES) >= 3,
          f"{[m[0] for m in avatar_studio.VOICE_MODES]}")
except Exception as exc:
    check("engine + studio modules import", False, str(exc)[:120])

print("\n" + "=" * 60)
print("2) TTS ENGINE — backends, cache, tag-strip, runtime switch")
print("=" * 60)


class Mouth:
    def __init__(self):
        self.fed = 0
        self.is_speaking = False
    def feed_audio(self, x):
        self.fed += 1
        self.is_speaking = True


eng = TTS.TTSStreamEngine(Mouth())

# --- emotion tag stripping ---
stripped = eng._strip_emotion_tags("Haha <laugh> gold <sigh> moving")
check("emotion tags stripped for non-Maya backends", "<" not in stripped, repr(stripped))

# --- runtime API present ---
check("runtime switch API present",
      all(hasattr(eng, a) for a in ("set_backend", "warm_backend", "tts_backend", "synthesizing")))
check("set_backend rejects junk", eng.set_backend("nonsense")[0] is False)

# --- Kokoro synth (fast) ---
eng.set_backend("kokoro")
eng.warm_backend()
line = "This is a smoke test of the kokoro voice path."
key = eng._cache_key(line)
if os.path.exists(key):
    os.remove(key)


def synth(text):
    fut = asyncio.run_coroutine_threadsafe(eng._synthesize(text), eng._loop)
    return fut.result(120)


t = time.time(); pcm1 = synth(line); d1 = time.time() - t
check("kokoro synth -> 16k PCM", pcm1 is not None and len(pcm1) > 1600,
      f"{len(pcm1)/16000:.2f}s in {d1:.2f}s")

# --- cache: 2nd identical synth is instant from disk ---
t = time.time(); pcm2 = synth(line); d2 = time.time() - t
check("disk cache hit (2nd synth instant)", d2 < 0.2 and len(pcm2) == len(pcm1),
      f"{d2*1000:.0f}ms")

# --- cache key is per-backend (so switching voices re-synths) ---
k_kokoro = eng._cache_key(line)
eng.set_backend("maya1"); k_maya = eng._cache_key(line)
eng.set_backend("chatterbox"); k_chat = eng._cache_key(line)
check("cache key differs per backend", len({k_kokoro, k_maya, k_chat}) == 3)

# --- edge fallback path (cloud, no GPU) ---
eng.set_backend("edge")
try:
    pcm_e = synth("Edge cloud voice fallback test.")
    check("edge-tts synth works", pcm_e is not None and len(pcm_e) > 1600,
          f"{len(pcm_e)/16000:.2f}s")
except Exception as exc:
    check("edge-tts synth works", False, str(exc)[:80])

# --- optional heavy Maya1 ---
if "--maya1" in sys.argv:
    print("\n  (loading Maya1 — heavy, ~30s, 7GB)...")
    eng.set_backend("maya1")
    msg = eng.warm_backend()
    loaded = eng._maya1 is not None
    check("Maya1 model loaded", loaded, msg)
    if loaded:
        ml = "Right <laugh> did you catch that breakout?"
        mk = eng._cache_key(ml)
        if os.path.exists(mk):
            os.remove(mk)
        t = time.time(); pm = synth(ml); dm = time.time() - t
        check("Maya1 expressive synth (with <laugh>)",
              pm is not None and len(pm) > 1600, f"{len(pm)/16000:.2f}s in {dm:.1f}s")
        # Heavy synth time (RTF > ~1x) proves the Maya1 path ran, not a fast
        # Kokoro fallback (~0.1s). _kokoro may be loaded from the earlier step.
        check("Maya1 used (no silent fallback to fast voice)",
              eng._maya1 is not None and dm > 1.0, f"synth took {dm:.1f}s (heavy model)")

eng.shutdown()

print("\n" + "=" * 60)
print("3) FACE RESTORE engine (CodeFormer)")
print("=" * 60)
try:
    import cv2
    fr = face_restore_engine.FaceRestoreEngine()
    ok, msg = fr.startup_check()
    check("restore engine ready", fr.ready, msg)
    if fr.ready:
        char = "ai-face/character.jpg" if os.path.exists("ai-face/character.jpg") else "character.jpg"
        img = cv2.resize(cv2.imread(char), (512, 512))
        out = fr.restore(img)
        check("restore returns a 512x512 frame", out is not None and out.shape == (512, 512, 3))
except Exception as exc:
    check("restore engine ready", False, str(exc)[:100])

print("\n" + "=" * 60)
passed = sum(1 for _, ok in RESULTS if ok)
print(f"SUMMARY: {passed}/{len(RESULTS)} checks passed")
for name, ok in RESULTS:
    if not ok:
        print(f"   FAILED: {name}")
print("=" * 60)
sys.exit(0 if passed == len(RESULTS) else 1)
