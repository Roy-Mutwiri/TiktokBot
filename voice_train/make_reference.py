# Build the best XTTS reference clip(s) from the extracted training audio.
# Energy-VAD finds clean, continuous, single-speaker speech windows (high speech
# ratio, steady level) and exports a ~45s reference + a few short backups.
import os, sys
import numpy as np
import soundfile as sf

RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_refs")
os.makedirs(OUT, exist_ok=True)
SR = 24000
WIN = 0.025                      # 25ms frames
REF_SECONDS = 45.0


def frames_rms(x, sr):
    n = int(WIN * sr)
    m = len(x) // n
    f = x[:m * n].reshape(m, n)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-9), n


def best_window(x, sr, seconds):
    rms, n = frames_rms(x, sr)
    thr = np.median(rms) * 0.6                # speech vs silence threshold
    speech = (rms > thr).astype(np.float32)
    fpw = int(seconds / WIN)                  # frames per window
    if fpw >= len(speech):
        return 0, len(x)
    # speech ratio over a sliding window (cumsum) — pick the densest, steadiest
    csum = np.concatenate([[0], np.cumsum(speech)])
    ratio = (csum[fpw:] - csum[:-fpw]) / fpw
    # prefer high speech ratio AND consistent loudness (low RMS variance)
    best_i, best_score = 0, -1
    step = max(1, fpw // 20)
    for i in range(0, len(ratio), step):
        seg = rms[i:i + fpw]
        score = ratio[i] - 0.4 * (seg.std() / (seg.mean() + 1e-9))
        if score > best_score:
            best_score, best_i = score, i
    s = best_i * n
    return s, s + int(seconds * sr)


def norm(x):
    x = x - np.mean(x)
    p = np.percentile(np.abs(x), 99.5) or 1.0
    return np.clip(x / p * 0.95, -1.0, 1.0).astype(np.float32)


# Prefer the Arabic+English file for the main reference (the bot code-switches).
priority = ["ArabEnglish", "Arab", "GoodOpeningSpeech"]
made = []
for name in priority:
    p = os.path.join(RAW, name + ".wav")
    if not os.path.exists(p):
        continue
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    s, e = best_window(x, sr, REF_SECONDS)
    clip = norm(x[s:e])
    out = os.path.join(OUT, f"arabic_trained_{name}.wav")
    sf.write(out, clip, sr)
    secs = len(clip) / sr
    print(f"[REF] {name}: window {s/sr:.0f}-{e/sr:.0f}s -> {os.path.basename(out)} ({secs:.1f}s)")
    made.append(out)

# the primary reference = the Arabic+English window
if made:
    primary = made[0]
    dst = os.path.join(OUT, "arabic_trained.wav")
    import shutil
    shutil.copyfile(primary, dst)
    print(f"[REF] primary reference -> {dst}")
