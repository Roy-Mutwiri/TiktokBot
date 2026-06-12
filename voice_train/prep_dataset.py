# Build an XTTS fine-tune dataset from the extracted training audio:
#   * Whisper (large-v3) transcribes each file with segment timestamps (ar/en).
#   * cut clean 2-11s clips at segment boundaries, resample 22.05k mono.
#   * write LJSpeech-format metadata (audio_file|text|speaker) + train/eval split.
#   * also rank clips by quality -> a curated multi-reference for the CLONE.
import os, sys, glob, csv
import numpy as np
import soundfile as sf

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw")
DS = os.path.join(HERE, "dataset")
WAVS = os.path.join(DS, "wavs")
os.makedirs(WAVS, exist_ok=True)
SR = 22050
MIN_S, MAX_S = 2.0, 11.0
SPEAKER = "arab_host"

import torch
_tl = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_tl):
    try: os.add_dll_directory(_tl)
    except Exception: pass
    os.environ["PATH"] = _tl + os.pathsep + os.environ["PATH"]
from faster_whisper import WhisperModel

MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
print(f"[PREP] loading Whisper {MODEL} ...")
asr = WhisperModel(MODEL, device="cuda", compute_type="float16")

rows = []          # (clip_rel, text, lang, rms)
idx = 0
for wavp in sorted(glob.glob(os.path.join(RAW, "*.wav"))):
    name = os.path.splitext(os.path.basename(wavp))[0]
    audio, sr = sf.read(wavp, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"[PREP] transcribing {name} ({len(audio)/sr:.0f}s) ...")
    segs, info = asr.transcribe(wavp, vad_filter=True, word_timestamps=False,
                                beam_size=5,
                                vad_parameters=dict(min_silence_duration_ms=400))
    kept = 0
    for s in segs:
        dur = s.end - s.start
        text = (s.text or "").strip()
        if dur < MIN_S or dur > MAX_S or len(text) < 4:
            continue
        a = audio[int(s.start*sr):int(s.end*sr)]
        if len(a) < int(MIN_S*sr):
            continue
        rms = float(np.sqrt((a**2).mean() + 1e-9))
        if rms < 0.01:                       # skip near-silent / very quiet
            continue
        # resample to 22.05k
        if sr != SR:
            import math
            n = int(round(len(a) * SR / sr))
            a = np.interp(np.linspace(0, len(a)-1, n), np.arange(len(a)), a).astype(np.float32)
        a = a - a.mean()
        peak = float(np.max(np.abs(a))) or 1.0
        a = (a / peak * 0.97).astype(np.float32)
        rel = f"wavs/clip_{idx:05d}.wav"
        sf.write(os.path.join(DS, rel), a, SR)
        lang = getattr(info, "language", "") or ""
        rows.append((rel, text, lang, rms))
        idx += 1; kept += 1
    print(f"[PREP]   {name}: kept {kept} clips")

print(f"[PREP] total clips: {len(rows)}")

# train/eval split (95/5)
import random
random.seed(0)
random.shuffle(rows)
n_eval = max(2, len(rows)//20)
ev, tr = rows[:n_eval], rows[n_eval:]

def write_meta(path, items):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for rel, text, lang, rms in items:
            w.writerow([rel, text, SPEAKER])

write_meta(os.path.join(DS, "metadata_train.csv"), tr)
write_meta(os.path.join(DS, "metadata_eval.csv"), ev)
print(f"[PREP] wrote metadata_train ({len(tr)}) + metadata_eval ({len(ev)})")

# curated multi-reference for the CLONE: top clips by loudness+duration, 6-10s each
ref_pool = [r for r in rows if 6.0 <= (sf.info(os.path.join(DS, r[0])).frames/SR) <= 11.0]
ref_pool.sort(key=lambda r: r[3], reverse=True)
REFDIR = os.path.join(os.path.dirname(HERE), "voice_refs")
for i, (rel, text, lang, rms) in enumerate(ref_pool[:6]):
    a, _ = sf.read(os.path.join(DS, rel), dtype="float32")
    sf.write(os.path.join(REFDIR, f"arabic_master_{i}.wav"), a, SR)
print(f"[PREP] wrote {min(6,len(ref_pool))} curated reference clips (arabic_master_*.wav)")
print("[PREP] DONE")
