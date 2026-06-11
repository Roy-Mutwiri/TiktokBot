# =============================================================================
# engines/multilingual_tts.py
# -----------------------------------------------------------------------------
# Multilingual voice for the avatar — Arabic + English (+ 21 more) in ONE human,
# clonable voice, powered by Chatterbox Multilingual (Resemble AI).
#
# Why this and not "training a model": a from-scratch TTS needs weeks of compute
# and huge corpora. Chatterbox Multilingual already speaks Arabic/English at
# SOTA naturalness and lets you CLONE any voice zero-shot from a short clip — so
# we get a limitless, human, multilingual voice today and (optionally) fine-tune
# later when there's a dataset.
#
# Key features:
#   * language routing  — picks the right language per UTTERANCE, and for
#     code-switched text ("welcome, مرحبا بك") splits by script and speaks each
#     run in its own language, then stitches them seamlessly.
#   * voice cloning     — AVATAR_CLONE_REF=<wav> clones that voice across all
#     languages (default: voice_refs/arabic_accent.wav if present).
#   * drop-in           — synthesize(text) matches chatterbox_tts so it slots
#     straight into tts_stream_engine.
#
# Env:
#   AVATAR_TTS_LANG=auto|ar|en|...   force a language (default auto-detect)
#   AVATAR_CLONE_REF=<wav>           voice to clone (all languages)
#   AVATAR_CLONE_EXAGGERATION=0.6    emotional intensity 0.3 calm .. 1.0 emotive
#   AVATAR_CLONE_CFG=0.5             pacing/adherence
# =============================================================================

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACCENT_REF = os.path.join(_PROJECT, "voice_refs", "arabic_accent.wav")
CLONE_REF = (os.environ.get("AVATAR_CLONE_REF", "").strip()
             or (_ACCENT_REF if os.path.exists(_ACCENT_REF) else None))
EXAGGERATION = float(os.environ.get("AVATAR_CLONE_EXAGGERATION", "0.6"))
CFG_WEIGHT = float(os.environ.get("AVATAR_CLONE_CFG", "0.5"))
DEFAULT_LANG = os.environ.get("AVATAR_TTS_LANG", "auto").strip().lower()

# Unicode ranges that mean "this is Arabic script".
_ARABIC_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_lang(text):
    """Coarse language guess for a whole string: 'ar' if Arabic script dominates."""
    ar = len(_ARABIC_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    return "ar" if ar > lat else "en"


_NUM2WORDS_LANGS = ("en", "ar", "fr", "es", "de", "it", "ru", "pt", "tr",
                    "pl", "nl", "fi", "da", "no", "sv", "he", "hi")
_CLAUSE_RE = re.compile(r"[^.!?؟،,;:\n]+[.!?؟،,;:\n]*")


def normalize_numbers(text, lang):
    """Speak digits as words in `lang` (4517 -> 'four thousand...'; ٤ -> ...).

    Covers ordinals (3rd), clock times (9:45), decimals (3.5) and integers, so
    the TTS never has to guess how to pronounce a numeral. Falls back silently
    (leaves the digits) if num2words can't handle the language.
    """
    try:
        from num2words import num2words
    except Exception:
        return text
    code = lang if lang in _NUM2WORDS_LANGS else "en"

    def _n(value, **kw):
        try:
            return num2words(value, lang=code, **kw)
        except Exception:
            try:
                return num2words(value, lang="en", **kw)
            except Exception:
                return str(value)

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b",
                  lambda m: _n(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: _n(int(m.group(1))) + ("" if m.group(2) == "00"
                                                   else " " + _n(int(m.group(2)))), text)
    text = re.sub(r"\b\d+\.\d+\b", lambda m: _n(float(m.group(0))), text)
    text = re.sub(r"\b\d+\b", lambda m: _n(int(m.group(0))), text)
    return text


def split_clauses(text, max_chars=180):
    """Break long text into <=max_chars clause chunks (bounds generation length,
    which sharply cuts Chatterbox's runaway-repetition failures)."""
    parts = [p.strip() for p in _CLAUSE_RE.findall(text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out or [text]


def segment_by_script(text):
    """Split mixed text into [(lang, chunk), ...] runs by script.

    Words in Arabic script -> 'ar', Latin words -> 'en'. Whitespace, digits and
    punctuation inherit the surrounding run so we never make a choppy switch on a
    comma or a number. Pure-single-language text comes back as one segment.
    """
    tokens = re.findall(r"\s+|\S+", text)
    segs = []  # list of [lang, text]
    for tok in tokens:
        if tok.isspace():
            if segs:
                segs[-1][1] += tok
            continue
        has_ar = bool(_ARABIC_RE.search(tok))
        has_lat = bool(_LATIN_RE.search(tok))
        if has_ar and not has_lat:
            lang = "ar"
        elif has_lat and not has_ar:
            lang = "en"
        else:
            lang = segs[-1][0] if segs else detect_lang(text)  # digits/punct
        if segs and segs[-1][0] == lang:
            segs[-1][1] += tok
        else:
            segs.append([lang, tok])
    return [(l, t.strip()) for l, t in segs if t.strip()]


class MultilingualTTSBackend:
    """Resident Chatterbox Multilingual model. synthesize() -> (float32, sr)."""

    def __init__(self, ref_path=None):
        self.ready = False
        self.ref_path = ref_path if ref_path is not None else CLONE_REF
        self.sr = 24000
        self._model = None
        self._torch = None
        self.device = "cpu"
        self.supported = {}
        try:
            self._load()
        except Exception as exc:
            print(f"[MLTTS] load failed ({exc}) — multilingual voice unavailable.")

    @property
    def ok(self):
        return self.ready

    def startup_check(self):
        if not self.ready:
            return False, "Multilingual TTS unavailable (load failed)."
        ref = os.path.basename(self.ref_path) if self.ref_path else "built-in voice"
        return True, (f"Chatterbox Multilingual ready ({self.device}, "
                      f"{len(self.supported)} langs incl. Arabic+English, ref: {ref}).")

    def _load(self):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
        self._torch = torch
        self.supported = dict(SUPPORTED_LANGUAGES)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[MLTTS] loading Chatterbox Multilingual (first run downloads ~2GB)...")
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sr = getattr(self._model, "sr", 24000)
        if self.ref_path and not os.path.exists(self.ref_path):
            print(f"[MLTTS] ref clip not found: {self.ref_path} — built-in voice.")
            self.ref_path = None
        self._warmup()
        self.ready = True
        if self.device == "cuda":
            print(f"[MLTTS] GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"[MLTTS] Ready — Arabic + English + {len(self.supported) - 2} more, "
              f"voice {'cloned' if self.ref_path else 'built-in'}.")

    def _warmup(self):
        for lang, txt in (("en", "Warming up."), ("ar", "تجهيز الصوت.")):
            try:
                self._gen(txt, lang)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    def _gen(self, text, lang):
        """One generate() call for `text` in a single `language_id`."""
        if lang not in self.supported:
            lang = "en"
        kw = {"language_id": lang, "exaggeration": EXAGGERATION,
              "cfg_weight": CFG_WEIGHT}
        if self.ref_path:
            kw["audio_prompt_path"] = self.ref_path
        wav = self._model.generate(text, **kw)
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).flatten()

    def _gen_safe(self, text, lang):
        """_gen with a runaway-repetition guard: if the audio comes out far
        longer than the text warrants (Chatterbox looped), retry once with
        steadier sampling and keep whichever clip is shorter/cleaner."""
        wav = self._gen(text, lang)
        nchar = max(1, len(re.sub(r"\s", "", text)))
        if len(wav) / self.sr > max(2.2, nchar * 0.20) and len(text) > 12:
            try:
                kw = {"language_id": lang if lang in self.supported else "en",
                      "exaggeration": min(EXAGGERATION, 0.5), "cfg_weight": 0.6,
                      "temperature": 0.5, "repetition_penalty": 2.5}
                if self.ref_path:
                    kw["audio_prompt_path"] = self.ref_path
                w2 = self._model.generate(text, **kw)
                if hasattr(w2, "detach"):
                    w2 = w2.detach().float().cpu().numpy()
                w2 = np.asarray(w2, dtype=np.float32).flatten()
                if len(w2) and len(w2) < len(wav):
                    return w2
            except Exception:
                pass
        return wav

    def _render(self, text, lang):
        """Full pipeline for one language: number-normalize -> clause-split ->
        generate each clause (with retry) -> concatenate."""
        text = normalize_numbers(text, lang)
        clauses = split_clauses(text)
        gap = np.zeros(int(0.04 * self.sr), dtype=np.float32)
        pieces = []
        for i, c in enumerate(clauses):
            pieces.append(self._gen_safe(c, lang))
            if i < len(clauses) - 1:
                pieces.append(gap)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def synthesize(self, text, lang=None):
        """Synthesize `text`. Returns (float32 mono, sr).

        lang=None -> use AVATAR_TTS_LANG (default 'auto', which routes each
        script-run to its own language for code-switched text). Pass an explicit
        code ('ar'/'en'/...) to force one language for the whole string.
        """
        if self._model is None:
            return None, self.sr
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.sr

        mode = (lang or DEFAULT_LANG or "auto").lower()
        if mode != "auto":
            return self._render(text, mode), self.sr

        segs = segment_by_script(text)
        if len(segs) <= 1:
            only = segs[0][0] if segs else detect_lang(text)
            return self._render(text, only), self.sr

        # code-switched: speak each run in its language, stitch with a short gap
        gap = np.zeros(int(0.06 * self.sr), dtype=np.float32)
        pieces = []
        for i, (l, chunk) in enumerate(segs):
            try:
                pieces.append(self._render(chunk, l))
            except Exception as exc:
                print(f"[MLTTS] segment ({l}) failed: {exc}")
            if i < len(segs) - 1:
                pieces.append(gap)
        if not pieces:
            return np.zeros(0, dtype=np.float32), self.sr
        return np.concatenate(pieces), self.sr


if __name__ == "__main__":
    import soundfile as sf
    eng = MultilingualTTSBackend()
    print("[MLTTS] startup_check:", eng.startup_check())
    samples = {
        "english": "Welcome back to the stream, everyone. Gold is on the move today.",
        "arabic": "مرحباً بكم في البث المباشر. الذهب يتحرك بقوة اليوم.",
        "mixed": "Welcome everyone, أهلاً وسهلاً بكم, let's get started.",
    }
    out_dir = os.path.join(_PROJECT, "tts_samples")
    os.makedirs(out_dir, exist_ok=True)
    for name, txt in samples.items():
        wav, sr = eng.synthesize(txt)
        if wav is not None and len(wav):
            p = os.path.join(out_dir, f"sample_{name}.wav")
            sf.write(p, wav, sr)
            print(f"[MLTTS] {name}: {len(wav)/sr:.2f}s -> {p}")
