# Mine the speaker's REAL non-speech human sounds from the training audio:
#   * breaths / lip-clicks  — short, low-but-present-energy gaps between phrases
#   * laughs                — high-energy bursts with strong ~4-8 Hz amplitude
#                             modulation (the rhythmic "ha-ha-ha" envelope)
# Exports them to voice_refs/human_sounds/ so the TTS can splice REAL human
# sounds (in the speaker's own voice) into the avatar's speech.
import os, glob
import numpy as np
import soundfile as sf

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(os.path.dirname(HERE), "voice_refs", "human_sounds")
os.makedirs(OUT, exist_ok=True)
SR = 24000
FR = 0.02                                   # 20ms frames


def frames(x, sr):
    n = int(FR * sr); m = len(x) // n
    return x[:m * n].reshape(m, n), n


def env_rms(x, sr):
    f, n = frames(x, sr)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-9), n


breaths, laughs = [], []
for p in sorted(glob.glob(os.path.join(RAW, "*.wav"))):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms, n = env_rms(x, sr)
    sp = rms.mean()
    speech = rms > sp * 0.5
    # ---- breaths / clicks: short non-speech runs with some energy ----
    i = 0
    while i < len(speech):
        if not speech[i]:
            j = i
            while j < len(speech) and not speech[j]:
                j += 1
            dur = (j - i) * FR
            seg_rms = rms[i:j].mean()
            if 0.12 <= dur <= 0.7 and sp * 0.08 < seg_rms < sp * 0.5:
                a = x[i * n:j * n]
                breaths.append((float(seg_rms), a.copy()))
            i = j
        else:
            i += 1
    # ---- laughs: ~0.8-2.5s high-energy windows with strong 4-8Hz modulation ----
    win = int(1.2 / FR)
    for s in range(0, len(rms) - win, win // 2):
        seg = rms[s:s + win]
        if seg.mean() < sp * 0.7:
            continue
        e = seg - seg.mean()
        sp_fft = np.abs(np.fft.rfft(e))
        freqs = np.fft.rfftfreq(len(e), FR)
        band = (freqs >= 3.5) & (freqs <= 9.0)
        mod = sp_fft[band].max() / (sp_fft.sum() + 1e-9)
        if mod > 0.18:                       # strong rhythmic envelope = laughter-like
            a = x[s * n:(s + win) * n]
            laughs.append((float(mod), a.copy()))

breaths.sort(reverse=True)
laughs.sort(reverse=True)


def save(items, prefix, k, fade=0.01):
    out = []
    for i, (score, a) in enumerate(items[:k]):
        a = a - a.mean()
        pk = float(np.max(np.abs(a))) or 1.0
        a = (a / pk * 0.9).astype(np.float32)
        f = int(fade * SR)
        if len(a) > 2 * f:
            a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f)
        path = os.path.join(OUT, f"{prefix}_{i}.wav")
        sf.write(path, a, SR)
        out.append(path)
    return out


b = save(breaths, "breath", 12)
l = save(laughs, "laugh", 10)
print(f"[HUMAN] breaths: {len(b)} (of {len(breaths)} candidates)")
print(f"[HUMAN] laughs:  {len(l)} (of {len(laughs)} candidates)")
print("[HUMAN] -> voice_refs/human_sounds/  (review/keep the good ones)")
