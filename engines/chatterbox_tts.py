# =============================================================================
# engines/chatterbox_tts.py
# -----------------------------------------------------------------------------
# Chatterbox (Resemble AI) voice CLONING backend. Give it a short (~7-20s) clean
# reference clip of a REAL person and it speaks new text in that person's voice —
# so the avatar sounds like a specific real human, not a generic AI voice. An
# `exaggeration` control dials emotional intensity.
#
#   AVATAR_CLONE_REF=C:\path\to\voice_sample.wav   # the voice to clone
#   AVATAR_CLONE_EXAGGERATION=0.6                   # 0.3 calm .. 1.0 very emotive
#
# Installed deliberately with --no-deps so it can't downgrade the cu128 torch
# (its setup pins torch==2.6.0, which would break the Blackwell GPU). Runs fine
# on the resident torch 2.11+cu128.
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# Reference voice clip to clone (None -> Chatterbox's built-in default voice).
CLONE_REF = os.environ.get("AVATAR_CLONE_REF", "").strip() or None
EXAGGERATION = float(os.environ.get("AVATAR_CLONE_EXAGGERATION", "0.6"))
CFG_WEIGHT = float(os.environ.get("AVATAR_CLONE_CFG", "0.5"))


class ChatterboxTTSBackend:
    """Resident Chatterbox model. synthesize() -> (float32 audio, sample_rate)."""

    def __init__(self, ref_path=None):
        self.ready = False
        self.ref_path = ref_path if ref_path is not None else CLONE_REF
        self.sr = 24000
        self._model = None
        self._torch = None
        self.device = "cpu"
        try:
            self._load()
        except Exception as exc:
            print(f"[CHATTERBOX] load failed ({exc}) — voice clone unavailable.")

    # -------------------------------------------------------------------------
    @property
    def ok(self):
        return self.ready

    def startup_check(self):
        if not self.ready:
            return False, "Chatterbox unavailable (load failed)."
        ref = os.path.basename(self.ref_path) if self.ref_path else "built-in voice"
        return True, f"Chatterbox clone ready ({self.device}, ref: {ref})."

    def _load(self):
        import torch
        from chatterbox.tts import ChatterboxTTS
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[CHATTERBOX] loading model (first run downloads ~2GB)...")
        self._model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = getattr(self._model, "sr", 24000)
        if self.ref_path and not os.path.exists(self.ref_path):
            print(f"[CHATTERBOX] ref clip not found: {self.ref_path} "
                  "— using built-in voice.")
            self.ref_path = None
        self._warmup()
        self.ready = True
        if self.device == "cuda":
            print(f"[CHATTERBOX] GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print("[CHATTERBOX] Ready — real-voice cloning enabled.")

    def _warmup(self):
        try:
            self.synthesize("Warming up the cloned voice.")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    def synthesize(self, text):
        """Synthesize `text` in the cloned voice. Returns (float32 mono, sr)."""
        if self._model is None:
            return None, self.sr
        kw = {"exaggeration": EXAGGERATION, "cfg_weight": CFG_WEIGHT}
        if self.ref_path:
            kw["audio_prompt_path"] = self.ref_path
        wav = self._model.generate(text, **kw)
        # Chatterbox returns a torch tensor (1, N) at self.sr.
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).flatten()
        return wav, self.sr


if __name__ == "__main__":
    import soundfile as sf
    eng = ChatterboxTTSBackend()
    print("[CHATTERBOX] startup_check:", eng.startup_check())
    wav, sr = eng.synthesize(
        "Hey everyone, welcome back to the stream. Gold is on the move today.")
    if wav is not None:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "_chatterbox_test.wav")
        sf.write(out, wav, sr)
        print(f"[CHATTERBOX] wrote _chatterbox_test.wav ({len(wav)/sr:.2f}s @ {sr}Hz)")
    else:
        print("[CHATTERBOX] synthesis returned None.")
