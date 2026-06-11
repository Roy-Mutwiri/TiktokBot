# =============================================================================
# tts_hardtest.py  —  hard, objective test of the multilingual avatar voice
# -----------------------------------------------------------------------------
# We cannot "hear" the voice in an automated run, so we test it OBJECTIVELY:
#
#   text  --(Chatterbox Multilingual)-->  speech  --(Whisper ASR)-->  text'
#
# and measure how well text' matches text. Low error == intelligible, fluent,
# correct-language speech. We test English, Modern Standard Arabic, Arabic
# dialect, code-switching, and number/punctuation-heavy lines, and for each:
#   * CER / WER vs the reference (intelligibility & fluency)
#   * a cross-language check (transcribe forcing the WRONG language; a big error
#     gap there proves the audio is genuinely in the right language/accent)
#   * synthesis real-time-factor (RTF) and audio duration
# Every clip is saved under tts_samples/ so a human can also just listen.
#
#   python tts_hardtest.py            # full suite
#   python tts_hardtest.py --smoke    # 1 EN + 1 AR (quick pipeline check)
#
# ASR model override:  TTS_ASR_MODEL=openai/whisper-medium
# =============================================================================

import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines"))
import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(PROJECT, "tts_samples")
ASR_MODEL = os.environ.get("TTS_ASR_MODEL", "openai/whisper-large-v3-turbo")
WHISPER_LANG = {"ar": "arabic", "en": "english"}

# -----------------------------------------------------------------------------
# test corpus
# -----------------------------------------------------------------------------
CORPUS = {
    "english": [
        "Welcome back to the stream, everyone. Let's get into it.",
        "The market opened higher today, up about three and a half percent.",
        "I genuinely can't believe how fast this year has gone by.",
        "She sells seashells by the seashore on a sunny Saturday.",
    ],
    "arabic_msa": [
        "مرحباً بكم في البث المباشر، أتمنى أن تكونوا بخير.",
        "افتتح السوق اليوم على ارتفاع بنسبة ثلاثة بالمئة تقريباً.",
        "هل يمكنك أن تشرح لي هذه الفكرة مرة أخرى من فضلك؟",
        "العلم نورٌ يضيء طريق الإنسان نحو المستقبل.",
    ],
    "arabic_dialect": [
        "إزيكم يا شباب، عاملين إيه النهاردة؟",
        "والله العظيم الكلام ده مظبوط مية في المية.",
    ],
    "mixed": [
        "Welcome everyone, أهلاً وسهلاً بكم, let's get started.",
        "النتيجة كانت رائعة, that's amazing, شكراً لكم.",
    ],
    "hard": [
        "Order 4517 ships on March 3rd at 9:45 a.m. sharp — don't be late!",
        "اتصل بالرقم سبعة تسعة اثنين خمسة، الساعة الثامنة مساءً.",
    ],
}
LANG_OF = {"english": "en", "arabic_msa": "ar", "arabic_dialect": "ar",
           "mixed": "en", "hard": "en"}   # 'mixed'/'hard' scored on dominant lang


# -----------------------------------------------------------------------------
# text normalization + edit-distance metrics
# -----------------------------------------------------------------------------
_AR_DIAC = re.compile(r"[ؗ-ًؚ-ْٰـ]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_ar(s):
    s = s.translate(_AR_DIGITS)
    s = _AR_DIAC.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي")):
        s = s.replace(a, b)
    s = re.sub(r"[^؀-ۿ0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_en(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_combined(s):
    """Keep BOTH Latin and Arabic (for scoring code-switched clips)."""
    s = s.translate(_AR_DIGITS).lower()
    s = _AR_DIAC.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z؀-ۿ0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize(s, lang):
    return norm_ar(s) if lang == "ar" else norm_en(s)


def _lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def cer(ref, hyp):
    r = ref.replace(" ", "")
    h = hyp.replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return min(1.0, _lev(list(r), list(h)) / len(r))   # capped: 1.0 = fully wrong


def wer(ref, hyp):
    r = ref.split()
    if not r:
        return 0.0 if not hyp.split() else 1.0
    return min(1.0, _lev(r, hyp.split()) / len(r))


# -----------------------------------------------------------------------------
def load_asr():
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[ASR] loading {ASR_MODEL} (first run downloads ~1.6GB)...")
    return pipeline("automatic-speech-recognition", model=ASR_MODEL,
                    device=dev, torch_dtype=dt)


def transcribe(asr, wav16, lang):
    out = asr({"array": wav16, "sampling_rate": 16000},
              generate_kwargs={"language": WHISPER_LANG[lang], "task": "transcribe"})
    return (out.get("text") or "").strip()


def transcribe_auto(asr, wav16):
    """Let Whisper auto-detect the language (used for code-switched clips)."""
    out = asr({"array": wav16, "sampling_rate": 16000},
              generate_kwargs={"task": "transcribe"})
    return (out.get("text") or "").strip()


def to16k(wav, sr):
    if sr == 16000:
        return wav.astype(np.float32)
    import librosa
    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)


def run(smoke=False):
    import soundfile as sf
    from multilingual_tts import MultilingualTTSBackend

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print("=" * 70)
    print("  MULTILINGUAL VOICE — HARD TEST  (Chatterbox ML  ->  Whisper ASR)")
    print("=" * 70)

    tts = MultilingualTTSBackend()
    ok, msg = tts.startup_check()
    print("[TTS]", msg)
    if not ok:
        print("[TTS] engine failed to load — aborting."); return 1
    asr = load_asr()

    corpus = CORPUS
    if smoke:
        corpus = {"english": CORPUS["english"][:1], "arabic_msa": CORPUS["arabic_msa"][:1]}

    rows = []
    for cat, lines in corpus.items():
        lang = LANG_OF[cat]
        other = "en" if lang == "ar" else "ar"
        for i, text in enumerate(lines):
            t0 = time.perf_counter()
            wav, sr = tts.synthesize(text)
            synth_s = time.perf_counter() - t0
            if wav is None or not len(wav):
                print(f"  [{cat}#{i}] SYNTH FAILED"); continue
            dur = len(wav) / sr
            path = os.path.join(SAMPLE_DIR, f"{cat}_{i}.wav")
            sf.write(path, wav, sr)

            wav16 = to16k(wav, sr)
            xc = None
            if cat == "mixed":
                # code-switched: auto-detect language, score keeping both scripts
                hyp = transcribe_auto(asr, wav16)
                ref_n, hyp_n = norm_combined(text), norm_combined(hyp)
            else:
                hyp = transcribe(asr, wav16, lang)
                ref_n, hyp_n = normalize(text, lang), normalize(hyp, lang)
                hyp_x = normalize(transcribe(asr, wav16, other), lang)
                xc = cer(ref_n, hyp_x)   # wrong-language CER (should be HIGH)
            c, w = cer(ref_n, hyp_n), wer(ref_n, hyp_n)

            rows.append(dict(cat=cat, lang=lang, dur=dur, rtf=synth_s / max(dur, 1e-6),
                             cer=c, wer=w, cer_wrong_lang=xc, text=text, asr=hyp,
                             file=os.path.basename(path)))
            xtag = f" xCER={xc:.2f}" if xc is not None else ""
            print(f"  [{cat}#{i}] {dur:4.1f}s RTF={synth_s/max(dur,1e-6):.2f} "
                  f"CER={c:.2f} WER={w:.2f}{xtag}")
            print(f"        ref: {text}")
            print(f"        asr: {hyp}")

    # ---- aggregate report ----
    print("\n" + "=" * 70)
    print("  RESULTS BY CATEGORY  (lower CER/WER = clearer; xCER should be HIGH)")
    print("=" * 70)
    summary = {}
    for cat in corpus:
        cr = [r for r in rows if r["cat"] == cat]
        if not cr:
            continue
        mc = np.mean([r["cer"] for r in cr])
        mw = np.mean([r["wer"] for r in cr])
        mr = np.mean([r["rtf"] for r in cr])
        xs = [r["cer_wrong_lang"] for r in cr if r["cer_wrong_lang"] is not None]
        mx = np.mean(xs) if xs else float("nan")
        summary[cat] = dict(cer=mc, wer=mw, rtf=mr, xcer=mx, n=len(cr))
        print(f"  {cat:14s} n={len(cr)}  CER={mc:.3f}  WER={mw:.3f}  "
              f"RTF={mr:.2f}  xCER(wrong-lang)={mx:.2f}")

    overall = dict(
        cer=float(np.mean([r["cer"] for r in rows])) if rows else None,
        wer=float(np.mean([r["wer"] for r in rows])) if rows else None,
        rtf=float(np.mean([r["rtf"] for r in rows])) if rows else None,
        n=len(rows))
    print("-" * 70)
    print(f"  OVERALL  n={overall['n']}  CER={overall['cer']:.3f}  "
          f"WER={overall['wer']:.3f}  RTF={overall['rtf']:.2f}")
    print(f"  samples saved in: {SAMPLE_DIR}")

    rep = os.path.join(SAMPLE_DIR, "hardtest_report.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump({"asr_model": ASR_MODEL, "summary": summary,
                   "overall": overall, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"  report: {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(smoke="--smoke" in sys.argv))
