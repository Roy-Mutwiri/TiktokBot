# =============================================================================
# engines/maya1_tts.py
# -----------------------------------------------------------------------------
# Maya1 (maya-research/maya1) — a 3B speech-LLM that produces genuinely human,
# EXPRESSIVE voice: it speaks with emotion and performs inline non-verbal sounds
# via tags written right in the text:
#
#   "Haha that candle was wild <laugh> did you all catch that breakout?"
#   "<sigh> alright, real talk, risk management is everything."
#
# Supported tags include <laugh> <giggle> <chuckle> <sigh> <gasp> <whisper>
# <cry> <gasp> <groan> <yawn> ... and a voice is *designed* with a natural-
# language description ("40-year-old male, warm, confident, conversational").
#
# Architecture: a Llama-style LM emits SNAC audio tokens (7 per frame across 3
# hierarchical codebooks) which the SNAC 24kHz neural codec decodes to a wave.
# This is heavy (~7GB VRAM, ~1-3s/line) — meant for the autonomous/OBS avatar,
# not the latency-sensitive realtime webcam studio.
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# --- Maya1 / SNAC token protocol (from the official model card) --------------
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

MODEL_ID = os.environ.get("AVATAR_MAYA_MODEL", "maya-research/maya1")
SNAC_ID = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000

# Default voice design — a warm, confident male trading-stream host. Override
# with AVATAR_MAYA_DESC to redesign the voice in plain English.
DEFAULT_DESCRIPTION = os.environ.get(
    "AVATAR_MAYA_DESC",
    "Male voice in his early thirties, warm, confident and energetic, "
    "clear American accent, conversational live-stream host tone.")


class Maya1TTS:
    """Resident Maya1 speech-LLM + SNAC decoder. synthesize() -> 24kHz float32."""

    def __init__(self, description=None):
        self.ready = False
        self.description = description or DEFAULT_DESCRIPTION
        self._model = None
        self._tok = None
        self._snac = None
        self._torch = None
        self.device = "cpu"
        try:
            self._load()
        except Exception as exc:
            print(f"[MAYA1] load failed ({exc}) — expressive voice unavailable.")

    # -------------------------------------------------------------------------
    @property
    def ok(self):
        return self.ready

    def startup_check(self):
        if not self.ready:
            return False, "Maya1 unavailable (load failed / model missing)."
        return True, f"Maya1 expressive voice ready ({self.device}, bf16)."

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[MAYA1] loading {MODEL_ID} (~7GB, first run downloads)...")
        self._tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device).eval()
        self._snac = SNAC.from_pretrained(SNAC_ID).eval().to(self.device)
        self._warmup()
        self.ready = True
        if self.device == "cuda":
            print(f"[MAYA1] GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print("[MAYA1] Ready — expressive human voice with emotion tags enabled.")

    def _warmup(self):
        # Short warm (few tokens) just to trigger cuDNN autotune without paying a
        # full ~3s synth at load — keeps the load wait shorter.
        try:
            self.synthesize("Hello.", max_new_tokens=48)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    def _build_prompt(self, text):
        tok = self._tok
        soh = tok.decode([SOH_ID]); eoh = tok.decode([EOH_ID])
        soa = tok.decode([SOA_ID]); sos = tok.decode([CODE_START_TOKEN_ID])
        eot = tok.decode([TEXT_EOT_ID]); bos = tok.bos_token
        formatted = f'<description="{self.description}"> {text}'
        return soh + bos + formatted + eot + eoh + soa + sos

    def synthesize(self, text, max_new_tokens=2048):
        """Generate expressive 24kHz mono float32 audio for `text` (which may
        contain inline emotion tags like <laugh>). Returns ndarray or None."""
        if not self.ready and self._model is None:
            return None
        torch = self._torch
        prompt = self._build_prompt(text)
        inputs = self._tok(prompt, return_tensors="pt",
                           add_special_tokens=False).to(self.device)
        try:
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=28,
                    temperature=0.4, top_p=0.9, repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID,
                    pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            # Sharing a 16GB GPU with MuseTalk + LivePortrait can run out of VRAM
            # on a long line. Free the cache and signal a graceful fallback
            # instead of leaving a poisoned CUDA context.
            torch.cuda.empty_cache()
            print("[MAYA1] CUDA OOM during generate — falling back for this line.")
            return None
        gen = out[0, inputs["input_ids"].shape[1]:].tolist()
        del out, inputs
        torch.cuda.empty_cache()         # release the KV cache promptly
        levels = self._unpack(self._extract(gen))
        if not levels[0]:
            return None
        codes = [torch.tensor(l, dtype=torch.long, device=self.device).unsqueeze(0)
                 for l in levels]
        with torch.inference_mode():
            z_q = self._snac.quantizer.from_codes(codes)
            audio = self._snac.decoder(z_q)[0, 0].float().cpu().numpy()
        if len(audio) > 2048:            # trim the codec's warmup ramp
            audio = audio[2048:]
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _extract(token_ids):
        try:
            eos = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos = len(token_ids)
        return [t for t in token_ids[:eos] if SNAC_MIN_ID <= t <= SNAC_MAX_ID]

    @staticmethod
    def _unpack(snac_tokens):
        frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
        l1, l2, l3 = [], [], []
        off = CODE_TOKEN_OFFSET
        for i in range(frames):
            s = snac_tokens[i * 7:(i + 1) * 7]
            l1.append((s[0] - off) % 4096)
            l2.extend([(s[1] - off) % 4096, (s[4] - off) % 4096])
            l3.extend([(s[2] - off) % 4096, (s[3] - off) % 4096,
                       (s[5] - off) % 4096, (s[6] - off) % 4096])
        return [l1, l2, l3]


if __name__ == "__main__":
    import soundfile as sf
    eng = Maya1TTS()
    print("[MAYA1] startup_check:", eng.startup_check())
    line = ("Haha, that candle was absolutely wild <laugh> did you all catch "
            "that breakout? <chuckle> Gold is on the move today.")
    wav = eng.synthesize(line)
    if wav is not None:
        sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "_maya1_test.wav"), wav, SAMPLE_RATE)
        print(f"[MAYA1] wrote _maya1_test.wav  ({len(wav)/SAMPLE_RATE:.2f}s)")
    else:
        print("[MAYA1] synthesis returned None.")
