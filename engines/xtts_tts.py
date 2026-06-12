# =============================================================================
# engines/xtts_tts.py
# -----------------------------------------------------------------------------
# Coqui XTTS-v2 voice backend (https://github.com/coqui-ai/TTS, idiap fork
# `coqui-tts`). XTTS-v2 is a multilingual zero-shot voice CLONER that speaks
# ARABIC and ENGLISH natively in a cloned voice — so the avatar is ONE Arabic
# male who code-switches Arabic/English cleanly (pure Arabic, not accented).
#
# It clones from a short reference clip (voice_refs/arabic_accent.wav by default,
# override with AVATAR_XTTS_REF) and the speaker latents are computed ONCE at load
# so each line synthesises fast.
#
# COMPAT SHIMS (this env: torch 2.11+cu128, transformers 5.x):
#   * transformers 5.x removed `isin_mps_friendly` that XTTS imports -> shim it.
#   * torch 2.9+ routes torchaudio.load through torchcodec, which needs FFmpeg 4-7
#     but the box has FFmpeg 8 -> patch torchaudio.load to use soundfile (the
#     references are WAV, read fine via libsndfile, no ffmpeg needed).
# These are applied BEFORE importing TTS.
# =============================================================================

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# ---- compat shims (must run before `import TTS`) ---------------------------
import torch as _torch
import transformers.pytorch_utils as _ptu
if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = lambda elements, test_elements: _torch.isin(elements, test_elements)

import torchaudio as _ta
import soundfile as _sf


def _sf_load(path, *args, **kwargs):
    """soundfile-backed torchaudio.load (bypasses torchcodec/ffmpeg for WAV)."""
    audio, sr = _sf.read(str(path), dtype="float32", always_2d=True)   # (N, ch)
    return _torch.from_numpy(audio.T.copy()), sr                        # (ch, N), sr


_ta.load = _sf_load     # XTTS's load_audio() calls torchaudio.load internally
# ----------------------------------------------------------------------------

import glob as _glob
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFDIR = os.path.join(_PROJECT, "voice_refs")


def _default_refs():
    """Reference clips for cloning, best-first:
      1. arabic_master_*.wav  — curated best clips from the training videos
      2. arabic_trained_*.wav — the 45s windows
      3. arabic_accent.wav    — original fallback sample."""
    for pat in ("arabic_master_*.wav", "arabic_trained_*.wav"):
        hits = sorted(_glob.glob(os.path.join(_REFDIR, pat)))
        if hits:
            return hits
    one = os.path.join(_REFDIR, "arabic_trained.wav")
    if os.path.exists(one):
        return [one]
    return [os.path.join(_REFDIR, "arabic_accent.wav")]


# AVATAR_XTTS_REF may be a single path or comma-separated list; else auto.
_env_ref = os.environ.get("AVATAR_XTTS_REF", "").strip()
XTTS_REFS = [p.strip() for p in _env_ref.split(",") if p.strip()] or _default_refs()
XTTS_REF = XTTS_REFS[0]
XTTS_TEMP = float(os.environ.get("AVATAR_XTTS_TEMP", "0.7"))
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def detect_lang(text):
    """Arabic script present -> 'ar', else 'en' (XTTS takes one language/call)."""
    return "ar" if _ARABIC.search(text or "") else "en"


class XTTSBackend:
    """Resident XTTS-v2. synthesize(text) -> (float32 mono audio, sample_rate)."""

    def __init__(self, ref_path=None):
        self.ready = False
        if ref_path:
            self.refs = ref_path if isinstance(ref_path, list) else [ref_path]
        else:
            self.refs = XTTS_REFS
        self.refs = [p for p in self.refs if os.path.exists(p)]
        self.ref_path = self.refs[0] if self.refs else None
        self.sr = 24000                       # XTTS native output rate
        self.device = "cpu"
        self._model = None
        self._gpt_latent = None
        self._spk_emb = None
        try:
            self._load()
        except Exception as exc:
            print(f"[XTTS] load failed ({exc}) — XTTS voice unavailable.")

    @property
    def ok(self):
        return self.ready

    def startup_check(self):
        if not self.ready:
            return False, "XTTS unavailable (load failed)."
        ref = os.path.basename(self.ref_path) if self.ref_path else "built-in"
        return True, f"XTTS-v2 Arabic+English clone ready ({self.device}, ref: {ref})."

    def _load(self):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        print("[XTTS] loading XTTS-v2 (first run downloads ~1.8GB)...")
        api = TTS(MODEL_NAME).to(self.device)
        self._model = api.synthesizer.tts_model        # low-level Xtts for speed
        refs = self.refs or _default_refs()
        refs = [p for p in refs if os.path.exists(p)]
        if not refs:
            raise RuntimeError("no XTTS reference audio found")
        print(f"[XTTS] cloning from {len(refs)} reference clip(s): "
              + ", ".join(os.path.basename(p) for p in refs))
        # compute the speaker latents ONCE from ALL refs -> richer, fast per-line
        self._gpt_latent, self._spk_emb = self._model.get_conditioning_latents(
            audio_path=refs, gpt_cond_len=30, max_ref_length=60)
        self._warmup()
        self.ready = True
        if self.device == "cuda":
            print(f"[XTTS] GPU memory: {_torch.cuda.memory_allocated()/1e9:.1f} GB")
        print("[XTTS] Ready — Arabic + English voice clone enabled.")

    def _warmup(self):
        try:
            self.synthesize("مرحبا")
        except Exception:
            pass

    def synthesize(self, text):
        """Synthesise `text` (auto ar/en) in the cloned voice -> (float32, sr)."""
        if self._model is None or not (text or "").strip():
            return None, self.sr
        lang = detect_lang(text)
        out = self._model.inference(
            text, lang, self._gpt_latent, self._spk_emb,
            temperature=XTTS_TEMP, enable_text_splitting=True)
        wav = out["wav"] if isinstance(out, dict) else out
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).flatten(), self.sr


if __name__ == "__main__":
    eng = XTTSBackend()
    print("[XTTS] startup_check:", eng.startup_check())
    for txt, name in [("مرحبا بكم في البث المباشر، الذهب يتحرك بقوة اليوم", "_xtts_ar.wav"),
                      ("Welcome back to the stream, gold is moving fast today", "_xtts_en.wav")]:
        wav, sr = eng.synthesize(txt)
        if wav is not None and len(wav):
            _sf.write(os.path.join(_PROJECT, name), wav, sr)
            print(f"[XTTS] wrote {name} ({len(wav)/sr:.2f}s @ {sr}Hz, lang={detect_lang(txt)})")
