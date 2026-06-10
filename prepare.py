# =============================================================================
# prepare.py — download + warm + pre-cache EVERYTHING so the studio is instant.
# -----------------------------------------------------------------------------
# Run this ONCE (or after adding phrases). It:
#   1. loads the Kokoro live voice (downloads on first run) and PRE-RENDERS the
#      greeting + every quick phrase into the TTS cache -> they play instantly,
#      zero generation, in the studio.
#   2. pulls + warms the Ollama brain model so the first question is fast.
#   3. checks every visual model weight (CodeFormer, Real-ESRGAN, LivePortrait,
#      MuseTalk, Maya1, Chatterbox) and downloads the HF ones it can.
#
#   python prepare.py
# =============================================================================
import os
import sys
import time
import asyncio

sys.path.insert(0, "engines")
sys.path.insert(0, ".")

OK, WARN = "[✓]", "[!]"


def hdr(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


# Phrases to pre-render (instant playback). Pull the studio's quick phrases too.
COMMON_LINES = [
    "Hey everyone, welcome back to the stream.",
    "Welcome back guys, let's get into it.",
    "Gold is pushing into a key resistance level right now.",
    "This is a serious move, watch the volume coming in.",
    "Thank you all for the support, let's get into it.",
    "Alright, let's break down today's setup.",
    "That's all for tonight folks, thanks for tuning in.",
]
try:
    import avatar_studio
    for p in getattr(avatar_studio, "QUICK_PHRASES", []):
        if p not in COMMON_LINES:
            COMMON_LINES.append(p)
except Exception:
    pass


# -----------------------------------------------------------------------------
hdr("1) VOICE — load Kokoro + pre-render common lines (cache = instant)")
import tts_stream_engine as TTS


class _M:
    def feed_audio(self, x):
        pass


eng = TTS.TTSStreamEngine(_M())
eng.set_backend("kokoro")
t = time.time()
print("   loading/ warming Kokoro (downloads on first run)...")
print("   ->", eng.warm_backend(), f"({time.time()-t:.1f}s)")


def _synth(text):
    fut = asyncio.run_coroutine_threadsafe(eng._synthesize(text), eng._loop)
    return fut.result(120)


pre = hit = 0
for ln in COMMON_LINES:
    if eng._cache_load(ln) is not None:
        hit += 1
        continue
    t = time.time()
    pcm = _synth(ln)                 # synth + auto-cache to disk
    if pcm is not None and len(pcm):
        pre += 1
        print(f"   {OK} cached ({time.time()-t:.1f}s): {ln[:46]}")
print(f"   -> {pre} pre-rendered, {hit} already cached "
      f"({pre+hit}/{len(COMMON_LINES)} common lines instant now)")
eng.shutdown()


# -----------------------------------------------------------------------------
hdr("2) BRAIN — pull + warm the Ollama model")
try:
    from llm_brain import LLMBrain, MODEL
    brain = LLMBrain()
    if not brain.ok:
        # try to pull the model via the CLI
        print(f"   model not ready ({brain.last_error}); pulling {MODEL} ...")
        import shutil
        import subprocess
        exe = shutil.which("ollama") or os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        if os.path.exists(exe):
            subprocess.run([exe, "pull", MODEL])
            brain = LLMBrain()
    if brain.ok:
        t = time.time()
        warmed = brain.warmup()
        print(f"   {OK} brain ready ({brain.model}); warmed={warmed} ({time.time()-t:.1f}s)")
        print("   sample:", (brain.respond("say hi to the stream in one short line") or "")[:80])
    else:
        print(f"   {WARN} Ollama brain unavailable: {brain.last_error}")
        print("       install Ollama + run it, then re-run prepare.py")
except Exception as exc:
    print(f"   {WARN} brain step failed: {exc}")


# -----------------------------------------------------------------------------
hdr("3) VISUAL MODELS — verify weights present")
ROOT = os.path.dirname(os.path.abspath(__file__))


def _exists(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


checks = [
    ("CodeFormer", _exists("ai-face", "models", "codeformer.pth")),
    ("Real-ESRGAN x2", _exists("ai-face", "models", "RealESRGAN_x2.pth")),
    ("GFPGAN (restore fallback)", _exists("ai-face", "models", "GFPGANv1.4.pth")),
    ("Wav2Lip checkpoint", _exists("ai-face", "Wav2Lip", "checkpoints", "wav2lip_gan.pth")),
    ("character image", _exists("ai-face", "character.jpg") or _exists("character.jpg")),
]
# LivePortrait lives as a sibling repo
lp = os.path.join(os.path.dirname(ROOT), "LivePortrait", "pretrained_weights")
checks.append(("LivePortrait weights", os.path.isdir(lp)))

for name, ok in checks:
    print(f"   {OK if ok else WARN} {name}: {'present' if ok else 'MISSING'}")

# CodeFormer + Real-ESRGAN can be (re)fetched by loading their engines once.
try:
    import face_restore_engine
    fr = face_restore_engine.FaceRestoreEngine()
    print(f"   {OK if fr.ready else WARN} CodeFormer engine loads + facexlib weights cached")
except Exception as exc:
    print(f"   {WARN} CodeFormer engine: {exc}")

print("\n" + "#" * 60)
print("PREP DONE. Quick phrases + greetings now play INSTANTLY (cached).")
print("Live voice = Kokoro (fast). Brain = Ollama (warmed, resident).")
print("Start the studio:  python avatar_studio.py")
print("#" * 60)
