# =============================================================================
# engines/bg_music.py
# -----------------------------------------------------------------------------
# A GENERATIVE, never-repeating PHONK playlist for the trading stream — the dark,
# hard, hypnotic "hustle/grind" sound (metallic cowbell melodies, distorted 808
# sub-bass, punchy kicks, tight triplet hats). It synthesises a library of 50+
# DISTINCT phonk tracks (each with its own key, tempo, dark scale, cowbell riff
# and drum pattern) and auto-advances to a new one every ~26s in a shuffled,
# no-repeat order — fresh for hours.
#
# All numpy (no audio files, no licensing). Plays on its own sounddevice stream
# (Windows mixes it with the winsound voice) and DUCKS under the AI's voice,
# swelling back up the moment it pauses.
#
# Instrumentals only — sung lyrics need a music-gen model (Suno); the avatar's
# Auto-host already speaks trading bars over the beat.
#
#   m = BackgroundMusic(); m.start(); m.set_active(True)
#   m.set_speaking(True/False)     # duck under voice / swell on pause
#   m.skip()                       # jump to the next unique track
# =============================================================================

import os
import sys
import time
import json
import random
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

SR = 44100
BASE_VOL = float(os.environ.get("AVATAR_MUSIC_VOL", "0.17"))   # idle (pause) bed level
# While the bot talks the music drops to this fraction so the VOICE clearly sits
# on top (0.15 -> bed ~5-6x quieter than the voice). Lower = voice more dominant.
DUCK = float(os.environ.get("AVATAR_MUSIC_DUCK", "0.15"))
SONG_SECONDS = float(os.environ.get("AVATAR_MUSIC_SONG_SECONDS", "26"))  # per track
# HUGE pool of procedurally-unique tracks so we can go a long time with no repeat.
# 5000 tracks x ~26s = ~36 hours of music before the pool could even be exhausted.
NUM_SONGS = int(os.environ.get("AVATAR_MUSIC_SONGS", "5000"))
# Don't replay a track until this many DAYS have passed (persisted across sessions).
REPEAT_GAP = float(os.environ.get("AVATAR_MUSIC_REPEAT_DAYS", "2")) * 86400.0
# Persistent play history (seed -> last-played unix time) so "heard today, not
# again for ~2 days" survives restarts.
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(_PROJECT, "_music_history.json")

# Dark scales only — phonk lives in minor/phrygian/harmonic-minor.
DARK_SCALES = {
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],     # darkest / tense
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],     # exotic
    "minor_pent":     [0, 3, 5, 7, 10],           # hooky
}
PROGRESSIONS = [          # chord roots as scale-degree indices (loop of 4)
    [0, 0, 5, 3], [0, 3, 4, 0], [0, 6, 5, 0],
    [0, 5, 6, 4], [0, 0, 3, 4], [0, 4, 0, 5],
]


def _semi(root, n):
    return root * (2.0 ** (n / 12.0))


def _env(n, a, d, s, r, sus=0.6):
    e = np.zeros(n, np.float32)
    ai, di, ri = int(n * a), int(n * d), int(n * r)
    si = max(0, n - ai - di - ri); i = 0
    if ai: e[i:i+ai] = np.linspace(0, 1, ai); i += ai
    if di: e[i:i+di] = np.linspace(1, sus, di); i += di
    if si: e[i:i+si] = sus; i += si
    if ri: e[i:i+ri] = np.linspace(sus, 0, ri)
    return e


def _osc(f, t, kind):
    if kind == "saw":
        return 2.0 * (t * f - np.floor(0.5 + t * f))
    if kind == "square":
        return np.sign(np.sin(2 * np.pi * f * t))
    if kind == "tri":
        return 2.0 * np.abs(2.0 * (t * f - np.floor(0.5 + t * f))) - 1.0
    return np.sin(2 * np.pi * f * t)


# Distinct phonk "vibes" so the 50 tracks don't all sound the same.
STYLES = ["aggressive", "melodic", "classic", "dark"]
LEAD_KINDS = ["bell", "saw", "square", "tri"]


def _make_track(seed):
    """Synthesise ONE unique seamless PHONK loop (mono float32) from `seed`.
    Each track picks a STYLE + key + tempo + chord prog + cowbell & lead motifs
    (A/B halves) so it has its own character — not a clone of the others."""
    rng = np.random.RandomState(seed * 2654435761 % (2**31))
    style = STYLES[rng.randint(len(STYLES))]
    bpm = float(rng.choice([128, 132, 138, 140, 145, 150, 155, 160]))
    beat = 60.0 / bpm
    beats = 16                                     # longer loop = far less repetitive
    n = int(SR * beat * beats)
    out = np.zeros(n, np.float32)
    root = float(rng.choice([48.99, 51.91, 55.0, 58.27, 61.74, 65.41]))   # G1..C2
    sc_name = rng.choice(["phrygian", "harmonic_minor"]) if style == "dark" \
        else rng.choice(list(DARK_SCALES.keys()))
    scale = DARK_SCALES[sc_name]
    prog = PROGRESSIONS[rng.randint(len(PROGRESSIONS))]
    # per-style character
    distort = {"aggressive": 2.2, "melodic": 1.0, "classic": 1.5, "dark": 1.2}[style]
    hat_div = {"aggressive": 4, "melodic": 2, "classic": 2, "dark": 2}[style]   # subdivisions/beat
    cow_gain = {"aggressive": 0.11, "melodic": 0.12, "classic": 0.17, "dark": 0.10}[style]
    has_lead = style != "aggressive" and rng.rand() < 0.85
    lead_kind = LEAD_KINDS[rng.randint(len(LEAD_KINDS))]
    has_pad = style in ("dark", "melodic") or rng.rand() < 0.5

    def at(sec):
        return int(sec * SR)

    def add(start, sig, gain):
        s = int(start); ln = len(sig)
        if s >= n:
            return
        if s + ln > n:
            sig = sig[:n - s]; ln = n - s
        out[s:s+ln] += sig.astype(np.float32) * gain

    # --- PHONK voices --------------------------------------------------------
    def kick(ln):
        tk = np.arange(ln) / SR
        k = np.sin(2*np.pi*(115*np.exp(-tk*42)+50)*tk) * np.exp(-tk*11)
        return np.tanh(k * 1.7)

    def b808(f, ln):
        tk = np.arange(ln) / SR
        gl = f * (1.0 + 0.45*np.exp(-tk*26))
        s = np.tanh(np.sin(2*np.pi*gl*tk) * 2.4 * distort)
        return s * _env(ln, 0.004, 0.25, 0.78, 0.16, 0.78)

    def cowbell(f, ln):
        tk = np.arange(ln) / SR
        s = np.sign(np.sin(2*np.pi*f*tk)) + np.sign(np.sin(2*np.pi*f*1.48*tk))
        return s * np.exp(-tk*15)

    def lead(f, ln, kind):
        tk = np.arange(ln) / SR
        if kind == "bell":
            s = np.sin(2*np.pi*f*tk) + 0.5*np.sin(2*np.pi*f*2.01*tk) + 0.25*np.sin(2*np.pi*f*3.0*tk)
            env = _env(ln, 0.004, 0.18, 0.0, 0.6, 0.0)
        else:
            s = _osc(f, tk, kind)
            env = _env(ln, 0.01, 0.2, 0.4, 0.4, 0.4)
        return s * env

    def snare(ln):
        tk = np.arange(ln) / SR
        return rng.randn(ln)*np.exp(-tk*24) + 0.35*np.sin(2*np.pi*180*tk)*np.exp(-tk*28)

    def hat(ln):
        tk = np.arange(ln) / SR
        return rng.randn(ln) * np.exp(-tk*150)

    # --- A/B melodic motifs (scale-degree offsets) so the two halves differ --
    motifA = [int(rng.choice([0, 0, 2, 3, 4, 5, 6, 7])) for _ in range(8)]
    motifB = [int(rng.choice([0, 0, 2, 3, 4, 5, 6, 7])) for _ in range(8)]
    half = beats // 2

    # --- drums: kick pattern (repeats each 8-beat half, fill in the 2nd) ------
    base_kick = [[0, 2, 3, 5, 6], [0, 3, 4, 6, 7], [0, 2, 4, 6], [0, 1.5, 4, 5.5]][rng.randint(4)]
    for hh in (0, 8):
        for b in base_kick:
            add(at((hh + b) * beat), kick(at(0.2)), 0.7)
    for bar in range(beats // 4):                  # snare on beats 2 & 4 of each bar
        add(at((bar*4 + 2) * beat), snare(at(0.14)), 0.30)
    for k in range(beats * hat_div):               # hats (style density)
        add(at(k * beat / hat_div), hat(at(0.03)), 0.06 if k % 2 else 0.09)
    for j in range(3):                             # one triplet hat roll near the end
        add(at((beats - 0.5) * beat + j*beat/3.0), hat(at(0.03)), 0.07)

    # --- cowbell melody (the hook) — A motif first half, B motif second -------
    step = beat / 2.0
    nsteps = beats * 2
    for k in range(nsteps):
        if rng.rand() < 0.28:
            continue
        m = motifA if k < nsteps // 2 else motifB
        deg = (prog[(k // 4) % len(prog)] + m[k % len(m)]) % len(scale)
        f = _semi(root, scale[deg] + int(rng.choice([24, 24, 26])))
        add(at(k*step), cowbell(f, at(step*0.9)), cow_gain)

    # --- optional lead melody (quarter notes) — gives melodic tracks identity -
    if has_lead:
        for k in range(beats):
            if rng.rand() < 0.45:
                continue
            m = motifA if k < half else motifB
            deg = (prog[(k // 4) % len(prog)] + m[k % len(m)]) % len(scale)
            f = _semi(root, scale[deg] + int(rng.choice([12, 12, 24])))
            add(at(k*beat), lead(f, at(beat*0.85), lead_kind), 0.07)

    # --- 808 bass: chord roots, long gliding notes ---------------------------
    for b in range(0, beats, 2):
        deg = prog[(b // 2) % len(prog)]
        f = _semi(root, scale[deg % len(scale)])
        add(at(b*beat), b808(f, at(beat*1.85)), 0.5)

    # --- dark atmosphere pad -------------------------------------------------
    if has_pad:
        t = np.arange(n) / SR
        pad = np.zeros(n, np.float32)
        for iv in (0, 3, 7):
            pad += np.sin(2*np.pi*_semi(root*2, scale[iv % len(scale)])*t).astype(np.float32)
        pad *= (0.5 + 0.5*np.sin(2*np.pi*0.2*t)).astype(np.float32)
        out += pad * 0.03

    # --- master: tape saturation glue + normalise + click-free loop seam -----
    out = np.tanh(out * 1.15)
    peak = float(np.max(np.abs(out))) or 1.0
    out = (out / peak) * 0.9
    xf = at(0.012)
    out[:xf] *= np.linspace(0, 1, xf)
    out[-xf:] *= np.linspace(1, 0, xf)
    return out.astype(np.float32), dict(bpm=int(bpm), style=style, scale=sc_name)


class BackgroundMusic:
    """Generative, non-repeating PHONK playlist (own sounddevice stream)."""

    def __init__(self, volume=BASE_VOL):
        self.base_vol = float(volume)
        self.active = False
        self._speaking = False
        self._gain = 0.0
        self._pos = 0
        self._stream = None
        self._alive = True
        # Persistent play history -> pick a track NOT heard in the last ~2 days, so
        # every session opens on a different song and repeats are spaced far apart.
        self._hist = self._load_history()
        try:
            seed = self._pick_seed()
            self._cur, self.meta = _make_track(seed)
            self._cur_seed = seed
            self._next = None
            self._next_seed = None
            self._song_until = None
            self._gen = threading.Thread(target=self._gen_loop, daemon=True)
            self._gen.start()
        except Exception as exc:
            print(f"[MUSIC] track synth failed ({exc}); music disabled.")
            self._cur = None

    # ---- persistent no-repeat history --------------------------------------
    def _load_history(self):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._hist, f)
        except Exception:
            pass

    def _pick_seed(self):
        """Pick a track seed not played in REPEAT_GAP; if all were (heavy use),
        pick the LEAST-recently played. Marks it played + persists. Thread-safe-ish
        (called only from __init__ and the generator thread, never the audio cb)."""
        now = time.time()
        eligible = [s for s in range(1, NUM_SONGS + 1)
                    if now - self._hist.get(str(s), 0.0) > REPEAT_GAP]
        if eligible:
            seed = random.choice(eligible)
        else:
            seed = min(range(1, NUM_SONGS + 1),
                       key=lambda s: self._hist.get(str(s), 0.0))
        self._hist[str(seed)] = now
        # prune ancient entries so the file stays small
        if len(self._hist) > NUM_SONGS:
            cutoff = now - REPEAT_GAP * 2
            self._hist = {k: v for k, v in self._hist.items() if v > cutoff}
        self._save_history()
        return seed

    def startup_check(self):
        if self._cur is None:
            return False, "music unavailable (synth failed)."
        return True, (f"generative PHONK playlist — {NUM_SONGS} unique tracks, "
                      f"~{SONG_SECONDS:.0f}s each, ducks under voice.")

    # -------------------------------------------------------------------------
    def _gen_loop(self):
        while self._alive:
            if self._cur is not None and self._next is None:
                ni = (self._oi + 1) % len(self._order)
                try:
                    self._next = _make_track(self._order[ni])
                except Exception:
                    self._next = None
            time.sleep(0.25)

    def start(self):
        if self._cur is None or self._stream is not None:
            return
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1,
                                           blocksize=1024, dtype="float32",
                                           callback=self._callback)
            self._stream.start()
            self._song_until = time.monotonic() + SONG_SECONDS
        except Exception as exc:
            print(f"[MUSIC] could not open audio stream ({exc}).")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        cur = self._cur
        if cur is None:
            outdata.fill(0); return
        target = self.base_vol * (DUCK if self._speaking else 1.0) if self.active else 0.0
        # Duck FAST (so the voice is never masked at the start of a line) but let
        # the music swell back up GENTLY in pauses (no jarring jump).
        ramp = 0.5 if target < self._gain else 0.12
        g_end = self._gain + (target - self._gain) * ramp
        gains = np.linspace(self._gain, g_end, frames, dtype=np.float32)
        self._gain = g_end
        end = self._pos + frames
        if end <= len(cur):
            buf = cur[self._pos:end]
            self._pos = end
        else:
            buf = np.concatenate([cur[self._pos:], cur[:end - len(cur)]])
            self._pos = end - len(cur)
            if (self._song_until and time.monotonic() >= self._song_until
                    and self._next is not None):
                self._cur, self.meta = self._next
                self._next = None
                self._oi = (self._oi + 1) % len(self._order)
                self._pos = 0
                self._song_until = time.monotonic() + SONG_SECONDS
                buf = self._cur[:frames]
        outdata[:, 0] = buf * gains

    def skip(self):
        self._song_until = 0.0

    # -------------------------------------------------------------------------
    def set_active(self, on):
        self.active = bool(on)
        if self.active and self._stream is None:
            self.start()

    def set_speaking(self, speaking):
        self._speaking = bool(speaking)

    def set_volume(self, v):
        self.base_vol = max(0.0, min(1.0, float(v)))

    def stop(self):
        self._alive = False
        try:
            if self._stream is not None:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None


if __name__ == "__main__":
    import soundfile as sf
    m = BackgroundMusic()
    print("[MUSIC]", m.startup_check()[1])
    if m._cur is not None:
        clips = []
        for s in m._order[:6]:
            trk, meta = _make_track(s)
            print(f"  phonk track seed={s}: {meta['bpm']} BPM")
            clips.append(trk)
            clips.append(np.zeros(int(SR*0.4), np.float32))
        sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "_bg_music_demo.wav"), np.concatenate(clips), SR)
        print("[MUSIC] wrote _bg_music_demo.wav (6 unique phonk tracks)")
