# Pick the best clips from the segmented dataset as a curated multi-reference for
# the XTTS clone (no transcripts needed). Scores by ideal duration (6-10s), clean
# loudness, and low clipping; exports the top N as voice_refs/arabic_master_*.wav.
import os, glob
import numpy as np
import soundfile as sf

HERE = os.path.dirname(__file__)
WAVS = os.path.join(HERE, "dataset", "wavs")
REFDIR = os.path.join(os.path.dirname(HERE), "voice_refs")
SR = 22050
N = int(os.environ.get("MASTER_REFS", "8"))

scored = []
for p in glob.glob(os.path.join(WAVS, "*.wav")):
    try:
        a, sr = sf.read(p, dtype="float32")
    except Exception:
        continue
    if a.ndim > 1:
        a = a.mean(axis=1)
    dur = len(a) / sr
    if dur < 4 or dur > 11:
        continue
    rms = float(np.sqrt((a ** 2).mean() + 1e-9))
    peak = float(np.max(np.abs(a)))
    clip_ratio = float(np.mean(np.abs(a) > 0.98))          # clipping penalty
    # consistency: low variance of frame energy = steady single-speaker speech
    n = int(0.025 * sr); m = len(a) // n
    fe = np.sqrt((a[:m*n].reshape(m, n) ** 2).mean(axis=1) + 1e-9)
    steady = 1.0 / (1.0 + fe.std() / (fe.mean() + 1e-9))
    dur_fit = 1.0 - abs(dur - 8.0) / 8.0                   # prefer ~8s
    score = (0.4 * steady + 0.3 * dur_fit + 0.3 * min(rms / 0.12, 1.0)
             - 2.0 * clip_ratio)
    scored.append((score, p, dur, rms))

scored.sort(reverse=True)
# clear any prior master refs
for old in glob.glob(os.path.join(REFDIR, "arabic_master_*.wav")):
    os.remove(old)
picked = scored[:N]
for i, (sc, p, dur, rms) in enumerate(picked):
    a, sr = sf.read(p, dtype="float32")
    a = a - a.mean()
    pk = float(np.max(np.abs(a))) or 1.0
    a = (a / pk * 0.97).astype(np.float32)
    sf.write(os.path.join(REFDIR, f"arabic_master_{i}.wav"), a, SR)
    print(f"[CURATE] #{i} score={sc:.3f} dur={dur:.1f}s rms={rms:.3f}  {os.path.basename(p)}")
print(f"[CURATE] wrote {len(picked)} master reference clips from {len(scored)} candidates.")
