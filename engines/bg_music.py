# =============================================================================
# engines/bg_music.py
# -----------------------------------------------------------------------------
# A GENERATIVE, never-repeating "trading mood" music playlist for the stream.
# Instead of one loop, it procedurally synthesises a large library of DISTINCT
# tracks (each with its own key, tempo, scale/mood, chord progression, drum
# pattern, bassline, lead and timbre) and auto-advances to a new one every ~25s,
# in a shuffled order with no repeats — so the soundtrack stays fresh for hours.
#
# Everything is synthesised with numpy (no audio files, no licensing). It plays
# on its own sounddevice stream (Windows mixes it with the winsound voice) and
# DUCKS under the AI's voice, swelling back up the moment it pauses.
#
# Note: these are INSTRUMENTALS. Actual SUNG lyrics need a music-generation model
# (e.g. Suno) — but the avatar's Auto-host already speaks trading bars over them.
#
#   m = BackgroundMusic(); m.start(); m.set_active(True)
#   m.set_speaking(True/False)     # duck under voice / swell on pause
#   m.skip()                       # jump to the next unique track
# =============================================================================

import os
import sys
import time
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

SR = 44100
BASE_VOL = float(os.environ.get("AVATAR_MUSIC_VOL", "0.16"))   # idle (pause) level
DUCK = float(os.environ.get("AVATAR_MUSIC_DUCK", "0.35"))      # multiplier while talking
SONG_SECONDS = float(os.environ.get("AVATAR_MUSIC_SONG_SECONDS", "26"))  # per track
NUM_SONGS = int(os.environ.get("AVATAR_MUSIC_SONGS", "50"))    # unique tracks before any repeat

# Scales (semitone steps from the root) — each gives a different mood.
SCALES = {
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],     # dark / tense
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],     # exotic
    "minor_pent":     [0, 3, 5, 7, 10],           # safe / hooky
}
PROGRESSIONS = [          # chord roots as scale-degree indices (loop of 4)
    [0, 5, 3, 4], [0, 3, 4, 0], [0, 6, 5, 4],
    [0, 4, 5, 3], [0, 2, 5, 4], [0, 5, 6, 4],
]
DRUM_STYLES = ["four", "trap", "halftime", "breaks"]


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
    return np.sin(2 * np.pi * f * t)            # sine


def _make_track(seed):
    """Synthesise ONE unique seamless loop (mono float32) determined by `seed`."""
    rng = np.random.RandomState(seed * 2654435761 % (2**31))
    bpm = float(rng.choice([90, 100, 108, 116, 120, 128, 136, 144]))
    beat = 60.0 / bpm
    beats = 8
    n = int(SR * beat * beats)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    root = float(rng.choice([55.0, 58.27, 61.74, 65.41, 69.30, 73.42]))   # A1..D2
    scale = SCALES[rng.choice(list(SCALES.keys()))]
    prog = PROGRESSIONS[rng.randint(len(PROGRESSIONS))]
    bass_kind = rng.choice(["sine", "saw", "tri", "square"])
    lead_kind = rng.choice(["sine", "saw", "tri", "square"])
    drum = rng.choice(DRUM_STYLES)
    has_lead = rng.rand() < 0.7
    has_pad = rng.rand() < 0.8

    def at(sec):
        return int(sec * SR)

    def add(buf, start, sig, gain):
        s = int(start); ln = len(sig)
        if s >= n:
            return
        if s + ln > n:
            sig = sig[:n - s]; ln = n - s
        buf[s:s+ln] += sig.astype(np.float32) * gain

    # --- drums ---------------------------------------------------------------
    def kick(ln):
        tk = np.arange(ln) / SR
        return np.sin(2*np.pi*(95*np.exp(-tk*30)+45)*tk) * np.exp(-tk*15)

    def snare(ln):
        tk = np.arange(ln) / SR
        return (rng.randn(ln)*np.exp(-tk*22) + 0.4*np.sin(2*np.pi*190*tk)*np.exp(-tk*26))

    def hat(ln):
        tk = np.arange(ln) / SR
        return rng.randn(ln)*np.exp(-tk*120)

    if drum == "four":
        for b in range(beats):
            add(out, at(b*beat), kick(at(0.18)), 0.6)
            add(out, at(b*beat+beat*0.5), hat(at(0.04)), 0.10)
        for b in (1, 3, 5, 7):
            add(out, at(b*beat), snare(at(0.14)), 0.28)
    elif drum == "trap":
        for b in (0, 2, 3, 5, 7):
            add(out, at(b*beat), kick(at(0.18)), 0.6)
        for b in (1, 5):
            add(out, at(b*beat), snare(at(0.13)), 0.30)
        for k in range(beats*4):                 # fast/rolling hats
            g = 0.09 if k % 2 == 0 else 0.05
            add(out, at(k*beat/2), hat(at(0.03)), g)
    elif drum == "halftime":
        for b in (0, 4):
            add(out, at(b*beat), kick(at(0.2)), 0.65)
        add(out, at(2*beat), snare(at(0.18)), 0.32)
        add(out, at(6*beat), snare(at(0.18)), 0.32)
        for b in range(beats):
            add(out, at(b*beat+beat*0.5), hat(at(0.04)), 0.08)
    else:  # breaks
        for b in (0, 3, 4, 6):
            add(out, at(b*beat), kick(at(0.17)), 0.55)
        for b in (2, 6):
            add(out, at(b*beat), snare(at(0.14)), 0.3)
        for k in range(beats*2):
            add(out, at(k*beat/1.0 + beat*0.5), hat(at(0.035)), 0.07)

    # --- bassline (root of each chord, plucked) ------------------------------
    for b in range(beats):
        deg = prog[(b // 2) % len(prog)]
        f = _semi(root, scale[deg % len(scale)])
        ln = at(beat*0.92)
        tb = np.arange(ln) / SR
        sig = _osc(f, tb, bass_kind) + 0.25*_osc(2*f, tb, bass_kind)
        add(out, at(b*beat), sig * _env(ln, 0.01, 0.25, 0.5, 0.35, 0.5), 0.30)

    # --- pad chord (triad of the bar's chord), soft + tremolo ----------------
    if has_pad:
        pad = np.zeros(n, np.float32)
        for bar in range(beats // 2):
            deg = prog[bar % len(prog)]
            s = at(bar*2*beat); ln = at(2*beat)
            tb = np.arange(ln) / SR
            for iv in (0, 2, 4):                 # 1-3-5 of the scale from deg
                f = _semi(root*4, scale[(deg+iv) % len(scale)])
                pad[s:s+ln] += np.sin(2*np.pi*f*tb).astype(np.float32)
        pad *= (0.6 + 0.4*np.sin(2*np.pi*0.3*t)).astype(np.float32)
        out += pad * 0.045

    # --- lead arp/melody -----------------------------------------------------
    if has_lead:
        step = beat / 2.0
        nsteps = int(beats / 0.5)
        for k in range(nsteps):
            if rng.rand() < 0.35:                # leave space
                continue
            deg = (prog[(k//4) % len(prog)] + rng.choice([0, 2, 4, 6])) % len(scale)
            octave = rng.choice([8, 12])         # up 1-1.5 oct from root
            f = _semi(root*4, scale[deg] + octave)
            ln = at(step*0.9)
            tb = np.arange(ln) / SR
            sig = _osc(f, tb, lead_kind)
            add(out, at(k*step), sig * _env(ln, 0.01, 0.15, 0.4, 0.5, 0.4), 0.07)

    # normalise + click-free loop seam
    peak = float(np.max(np.abs(out))) or 1.0
    out = (out / peak) * 0.9
    xf = at(0.012)
    out[:xf] *= np.linspace(0, 1, xf)
    out[-xf:] *= np.linspace(1, 0, xf)
    meta = dict(bpm=int(bpm), drum=drum)
    return out.astype(np.float32), meta


class BackgroundMusic:
    """Generative, non-repeating trading-mood playlist (own sounddevice stream)."""

    def __init__(self, volume=BASE_VOL):
        self.base_vol = float(volume)
        self.active = False
        self._speaking = False
        self._gain = 0.0
        self._pos = 0
        self._stream = None
        self._alive = True
        # shuffled play order over NUM_SONGS unique seeds (no repeat per cycle)
        self._order = list(range(1, NUM_SONGS + 1))
        np.random.RandomState(12345).shuffle(self._order)
        self._oi = 0
        try:
            self._cur, self.meta = _make_track(self._order[0])
            self._next = None
            self._song_until = None
            self._gen = threading.Thread(target=self._gen_loop, daemon=True)
            self._gen.start()
        except Exception as exc:
            print(f"[MUSIC] track synth failed ({exc}); music disabled.")
            self._cur = None

    def startup_check(self):
        if self._cur is None:
            return False, "music unavailable (synth failed)."
        return True, (f"generative playlist ready — {NUM_SONGS} unique tracks, "
                      f"~{SONG_SECONDS:.0f}s each, ducks under voice.")

    # -------------------------------------------------------------------------
    def _gen_loop(self):
        """Keep the NEXT track pre-rendered so song changes are seamless."""
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
        g_end = self._gain + (target - self._gain) * 0.25
        gains = np.linspace(self._gain, g_end, frames, dtype=np.float32)
        self._gain = g_end
        end = self._pos + frames
        if end <= len(cur):
            buf = cur[self._pos:end]
            self._pos = end
        else:
            buf = np.concatenate([cur[self._pos:], cur[:end - len(cur)]])
            self._pos = end - len(cur)
            # at the loop seam, advance to the NEXT unique track if it's time
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
        """Jump to the next track now (next loop boundary)."""
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
    # render 6 distinct tracks to one file so you can hear the variety
    if m._cur is not None:
        clips = []
        for s in m._order[:6]:
            trk, meta = _make_track(s)
            print(f"  track seed={s}: {meta['bpm']} BPM, {meta['drum']}")
            clips.append(trk)
            clips.append(np.zeros(int(SR*0.4), np.float32))
        sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "_bg_music_demo.wav"), np.concatenate(clips), SR)
        print("[MUSIC] wrote _bg_music_demo.wav (6 unique tracks back-to-back)")
