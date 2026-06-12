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
# Expressive but STABLE sampling. Temperature too high (0.85+) made XTTS distort
# on longer speech, so keep it lively-yet-stable (0.75) and rely on per-sentence
# chunking (below) to keep every generation short. repetition_penalty prevents the
# autoregressive repeat/garble loop. All env-tunable.
XTTS_TEMP = float(os.environ.get("AVATAR_XTTS_TEMP", "0.65"))      # stable sweet spot
XTTS_REP_PEN = float(os.environ.get("AVATAR_XTTS_REP_PENALTY", "2.0"))  # high = artifacts
XTTS_TOP_K = int(os.environ.get("AVATAR_XTTS_TOP_K", "50"))
XTTS_TOP_P = float(os.environ.get("AVATAR_XTTS_TOP_P", "0.80"))
XTTS_LEN_PEN = float(os.environ.get("AVATAR_XTTS_LEN_PENALTY", "1.0"))
# A calm, natural man speaks measured — XTTS at 1.0 rushes and reads "AI". 0.88 lands
# a relaxed human pace; dynamic pacing (below) nudges it faster when excited / slower
# when analysing. Env-tunable.
XTTS_SPEED = float(os.environ.get("AVATAR_XTTS_SPEED", "0.80"))
PACE_DYNAMIC = os.environ.get("AVATAR_XTTS_DYNAMIC_PACE", "1") == "1"
PACE_FAST = float(os.environ.get("AVATAR_XTTS_PACE_FAST", "1.10"))   # excited multiplier
PACE_SLOW = float(os.environ.get("AVATAR_XTTS_PACE_SLOW", "0.95"))   # calm/analytical multiplier
XTTS_GAP = float(os.environ.get("AVATAR_XTTS_GAP", "0.09"))          # pause between segments (s)
# Per-call cap: keep WHOLE SENTENCES together (natural human speech), only split a
# sentence if it's longer than this (rare). A full sentence is stable in one call;
# it's MULTI-sentence runs that distort, so we split at sentence boundaries only.
XTTS_MAX_CHARS = int(os.environ.get("AVATAR_XTTS_MAX_CHARS", "250"))
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\:؟…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,،;:\-—])\s+")

# TikTok names/comments are full of emojis, flags, arrows and zero-width chars that
# make XTTS error or distort; hype words like "gooooo"/"yaaaa" make it drift. Strip
# the junk and tame long same-letter runs BEFORE synthesis.
_JUNK = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0001F900-\U0001F9FF︀-️←-⇿"
    "⌀-⏿⬀-⯿■-◿✀-➿]+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁠﻿]")
_RUN = re.compile(r"(\w)\1{2,}")            # 3+ of the same letter in a row
# LaTeX / math / markdown artifacts that smarter LLMs (qwen) sometimes emit around
# numbers — they would be read aloud as garbage, so strip them.
_MATH = re.compile(r"\\[\(\)\[\]]|\\[a-zA-Z]+|[\\$*_`#{}]|\^|~")


def clean_for_tts(text):
    """Strip emojis/symbols/control/math-markup chars and tame repeated-letter runs
    so names, hype words and LLM number-formatting don't make XTTS error or distort.
    Keeps letters (incl Arabic), digits and normal punctuation."""
    t = _JUNK.sub(" ", text or "")
    t = _CTRL.sub("", t)
    t = _MATH.sub(" ", t)                   # \( \) $ * _ ` # etc -> gone
    t = t.replace("_", " ")                 # handles like "Ahmed_2010" read naturally
    t = _RUN.sub(r"\1\1", t)                # "gooooo" -> "goo", "yaaaa" -> "yaa"
    return re.sub(r"\s+", " ", t).strip()


def chunk_text(text, max_chars=None):
    """Split into short, sentence-sized chunks so each XTTS generation stays
    short and stable (long single generations are what distort/drift)."""
    max_chars = max_chars or XTTS_MAX_CHARS
    out = []
    for sent in _SENT_SPLIT.split((text or "").strip()):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        cur = ""                                  # long sentence -> split at clauses
        for piece in _CLAUSE_SPLIT.split(sent):
            if len(cur) + len(piece) + 1 <= max_chars:
                cur = (cur + " " + piece).strip()
            else:
                if cur:
                    out.append(cur)
                # piece itself too long -> hard-split on words
                while len(piece) > max_chars:
                    cut = piece.rfind(" ", 0, max_chars)
                    cut = cut if cut > 0 else max_chars
                    out.append(piece[:cut].strip()); piece = piece[cut:].strip()
                cur = piece
        if cur:
            out.append(cur)
    return out

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def detect_lang(text):
    """Arabic script present -> 'ar', else 'en' (XTTS takes one language/call)."""
    return "ar" if _ARABIC.search(text or "") else "en"


# The persona writes Arabic words in LATIN letters (wallahi, habibi...). XTTS then
# reads them with ENGLISH phonetics = wrong, robotic Arabic. Convert the common ones
# to ARABIC SCRIPT before synthesis so the code-switcher routes them to the Arabic
# voice and they sound like a real Arab speaker. Add more freely.
ARABIC_WORDS = {
    "wallahi": "والله", "wallah": "والله", "wallahy": "والله", "wallahi!": "والله",
    "habibi": "حبيبي", "habibti": "حبيبتي", "habayebi": "حبايبي", "habayb": "حبايب",
    "mashallah": "ما شاء الله", "masha'allah": "ما شاء الله",
    "inshallah": "إن شاء الله", "insha'allah": "إن شاء الله",
    "alhamdulillah": "الحمد لله", "alhamdullah": "الحمد لله", "hamdulillah": "الحمد لله",
    "yalla": "يلا", "yallah": "يلا",
    "akhi": "أخي", "ya": "يا", "yaani": "يعني", "yani": "يعني",
    "khalas": "خلاص", "sahbi": "صاحبي", "shukran": "شكراً",
    "salam": "سلام", "salaam": "سلام", "habibo": "حبيبو", "tamam": "تمام",
    "bismillah": "بسم الله", "subhanallah": "سبحان الله", "wala": "ولا",
}
_AR_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(ARABIC_WORDS, key=len, reverse=True))
    + r")\b", re.I)


def arabicize(text):
    """Swap transliterated Arabic words for Arabic script so they're pronounced
    correctly (whole-word, case-insensitive)."""
    if not text:
        return text
    return _AR_WORD_RE.sub(lambda m: ARABIC_WORDS.get(m.group(0).lower(), m.group(0)), text)


_HYPE_RE = re.compile(
    r"\b(yalla|let'?s\s*go+|smash|goo+|come on|right now|wow|insane|huge|legend|"
    r"crazy|let'?s\s*gooo|pump|moon|fire|lets go)\b|!!", re.I)
_CALM_RE = re.compile(
    r"\b(support|resistance|level|levels|trend|rsi|analysis|patient|careful|"
    r"carefully|wait|watching|notice|remember|steady|slowly|consider|honestly|"
    r"the key|keep in mind)\b", re.I)


def pace_for(text):
    """A calm man's pace by default, a touch FASTER when excited and SLOWER when
    analysing — so it's never a flat, uniform (robotic) speed."""
    if not PACE_DYNAMIC:
        return XTTS_SPEED
    t = text or ""
    if t.count("!") >= 2 or _HYPE_RE.search(t):
        s = XTTS_SPEED * PACE_FAST
    elif _CALM_RE.search(t) or len(t) > 170:
        s = XTTS_SPEED * PACE_SLOW
    else:
        s = XTTS_SPEED
    return max(0.78, min(1.08, s))


def codeswitch_segments(text):
    """Split mixed text into consecutive runs of Arabic-script vs Latin so each
    run is synthesised in its OWN language (XTTS takes one language per call).
    Without this, one Arabic word in an English line makes the WHOLE line get
    spoken as Arabic (garbled). Returns [(segment_text, lang), ...]."""
    segs = []
    for tok in (text or "").split():
        lang = "ar" if _ARABIC.search(tok) else "en"
        if segs and segs[-1][1] == lang:
            segs[-1][0].append(tok)
        else:
            segs.append(([tok], lang))
    return [(" ".join(ws), lang) for ws, lang in segs]


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

    def _synth_one(self, text, lang, speed=None):
        # already-short chunk -> own splitting OFF (we control chunk size)
        out = self._model.inference(
            text, lang, self._gpt_latent, self._spk_emb,
            temperature=XTTS_TEMP, length_penalty=XTTS_LEN_PEN,
            repetition_penalty=XTTS_REP_PEN, top_k=XTTS_TOP_K, top_p=XTTS_TOP_P,
            speed=speed if speed is not None else XTTS_SPEED,
            enable_text_splitting=False)
        wav = out["wav"] if isinstance(out, dict) else out
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).flatten()

    def synthesize(self, text):
        """Synthesise `text` in the cloned voice -> (float32, sr). Two safeguards
        against long-speech distortion: (1) code-switch into Arabic/Latin runs;
        (2) chunk each run into short sentence-sized pieces so no single XTTS
        generation runs long enough to drift/distort. Pieces are concatenated."""
        text = clean_for_tts(text)            # strip emojis/symbols, tame letter runs
        text = arabicize(text)                # transliterated Arabic -> Arabic script
        if self._model is None or not text:
            return None, self.sr
        speed = pace_for(text)                # calm/fast per line (not a flat robotic pace)
        # a SHORT natural pause between sentences (like a real person breathing/
        # pacing) — small enough that the speech still flows, not choppy.
        gap = np.zeros(int(0.12 * self.sr), dtype=np.float32)
        parts = []
        for seg_text, lang in codeswitch_segments(text):
            for chunk in chunk_text(seg_text):
                chunk = chunk.strip()
                # an Arabic segment with no actual Arabic letters left (just stray
                # punctuation) -> skip so XTTS isn't handed junk it distorts on
                if not chunk or not re.search(r"\w", chunk):
                    continue
                try:
                    w = self._synth_one(chunk, lang, speed)
                    if w is not None and len(w):
                        parts.append(w); parts.append(gap)
                except Exception:
                    continue
        if not parts:
            return None, self.sr
        return np.concatenate(parts).astype(np.float32), self.sr


if __name__ == "__main__":
    eng = XTTSBackend()
    print("[XTTS] startup_check:", eng.startup_check())
    for txt, name in [("مرحبا بكم في البث المباشر، الذهب يتحرك بقوة اليوم", "_xtts_ar.wav"),
                      ("Welcome back to the stream, gold is moving fast today", "_xtts_en.wav")]:
        wav, sr = eng.synthesize(txt)
        if wav is not None and len(wav):
            _sf.write(os.path.join(_PROJECT, name), wav, sr)
            print(f"[XTTS] wrote {name} ({len(wav)/sr:.2f}s @ {sr}Hz, lang={detect_lang(txt)})")
