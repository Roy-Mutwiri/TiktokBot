# =============================================================================
# engines/bg_music.py
# -----------------------------------------------------------------------------
# Subtle "trading mood" background music bed for the avatar stream. It loops a
# soft electronic groove continuously and DUCKS (drops in volume) while the AI
# is talking, then swells back up the moment the AI pauses — so dead air never
# kills the energy. Plays on its own sounddevice stream, which Windows mixes with
# the winsound voice playback (two independent streams to the default device).
#
# The loop is SYNTHESISED with numpy (no audio file, no licensing) — a soft kick,
# a plucked sub-bass on a minor riff, a quiet pad chord, and an offbeat hat.
#
#   m = BackgroundMusic(); m.start(); m.set_active(True)
#   m.set_speaking(True)   # duck under the voice
#   m.set_speaking(False)  # swell back up during a pause
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

SR = 44100
BASE_VOL = float(os.environ.get("AVATAR_MUSIC_VOL", "0.16"))   # idle (pause) level
DUCK = float(os.environ.get("AVATAR_MUSIC_DUCK", "0.35"))      # multiplier while talking
BPM = float(os.environ.get("AVATAR_MUSIC_BPM", "120"))


def _adsr(n, a, d, s, r, sus=0.7):
    """Simple ADSR envelope of length n samples (fractions of n)."""
    env = np.zeros(n, np.float32)
    ai, di, ri = int(n * a), int(n * d), int(n * r)
    si = max(0, n - ai - di - ri)
    i = 0
    if ai:
        env[i:i + ai] = np.linspace(0, 1, ai); i += ai
    if di:
        env[i:i + di] = np.linspace(1, sus, di); i += di
    if si:
        env[i:i + si] = sus; i += si
    if ri:
        env[i:i + ri] = np.linspace(sus, 0, ri)
    return env


def _make_loop():
    """Synthesise a seamless ~4s trading-mood loop (mono float32)."""
    beat = 60.0 / BPM
    bars = 2
    beats = bars * 4                       # 8 beats
    total = beat * beats
    n = int(SR * total)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    rng = np.random.RandomState(7)

    def at(sec):
        return int(sec * SR)

    # --- soft kick on every beat (sine thump, fast pitch + amp decay) ---
    for b in range(beats):
        s = at(b * beat)
        ln = at(0.18)
        if s + ln > n:
            ln = n - s
        tk = np.arange(ln) / SR
        pitch = 90 * np.exp(-tk * 28) + 45
        kick = np.sin(2 * np.pi * pitch * tk) * np.exp(-tk * 16)
        out[s:s + ln] += kick.astype(np.float32) * 0.6

    # --- plucked sub-bass: a small minor riff (root, b3, 5, b7) ---
    root = 65.41                           # C2
    riff = [root, root, root * 1.5, root * 1.189]   # C C G Eb-ish
    for b in range(beats):
        if b % 2 != 0:                     # on beats 1,3,5,7
            continue
        f = riff[(b // 2) % len(riff)]
        s = at(b * beat)
        ln = at(beat * 0.9)
        if s + ln > n:
            ln = n - s
        tb = np.arange(ln) / SR
        wave = (np.sin(2 * np.pi * f * tb) + 0.3 * np.sin(2 * np.pi * 2 * f * tb))
        out[s:s + ln] += (wave * _adsr(ln, 0.01, 0.2, 0.5, 0.4, 0.5)).astype(np.float32) * 0.32

    # --- quiet sustained pad (C minor triad), slow tremolo for movement ---
    chord = [130.81, 155.56, 196.00]       # C Eb G
    pad = np.zeros(n, np.float32)
    for f in chord:
        pad += np.sin(2 * np.pi * f * t).astype(np.float32)
    pad *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32)  # tremolo
    out += pad * 0.05

    # --- offbeat closed hat (filtered noise burst) ---
    for b in range(beats):
        s = at(b * beat + beat * 0.5)
        ln = at(0.05)
        if s + ln > n:
            continue
        hat = rng.randn(ln).astype(np.float32) * np.exp(-np.arange(ln) / SR * 90)
        out[s:s + ln] += hat * 0.10

    # normalise headroom + tiny fade at the seam for a click-free loop
    peak = float(np.max(np.abs(out))) or 1.0
    out = (out / peak) * 0.9
    xf = at(0.01)
    out[:xf] *= np.linspace(0, 1, xf)
    out[-xf:] *= np.linspace(1, 0, xf)
    return out.astype(np.float32)


class BackgroundMusic:
    """Looping, duck-able trading-mood music bed (own sounddevice stream)."""

    def __init__(self, volume=BASE_VOL):
        self.base_vol = float(volume)
        self.active = False
        self._speaking = False
        self._gain = 0.0                   # smoothed current gain
        self._pos = 0
        self._stream = None
        self._sd = None
        try:
            self.loop = _make_loop()
        except Exception as exc:
            print(f"[MUSIC] loop synth failed ({exc}); music disabled.")
            self.loop = None

    def startup_check(self):
        if self.loop is None:
            return False, "background music unavailable (synth failed)."
        return True, f"background music ready ({len(self.loop)/SR:.1f}s loop, ducks under voice)."

    # -------------------------------------------------------------------------
    def start(self):
        """Open the audio stream (silent until set_active(True))."""
        if self.loop is None or self._stream is not None:
            return
        try:
            import sounddevice as sd
            self._sd = sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1,
                                           blocksize=1024, dtype="float32",
                                           callback=self._callback)
            self._stream.start()
        except Exception as exc:
            print(f"[MUSIC] could not open audio stream ({exc}).")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        if self.loop is None:
            outdata.fill(0); return
        target = self.base_vol * (DUCK if self._speaking else 1.0) if self.active else 0.0
        # per-block linear ramp toward target -> smooth fade/duck (no clicks)
        g_end = self._gain + (target - self._gain) * 0.25
        gains = np.linspace(self._gain, g_end, frames, dtype=np.float32)
        self._gain = g_end
        end = self._pos + frames
        if end <= len(self.loop):
            buf = self.loop[self._pos:end]
        else:
            buf = np.concatenate([self.loop[self._pos:], self.loop[:end - len(self.loop)]])
        self._pos = end % len(self.loop)
        outdata[:, 0] = buf * gains

    # -------------------------------------------------------------------------
    def set_active(self, on):
        """Turn the bed on/off (smoothly fades via the callback ramp)."""
        self.active = bool(on)
        if self.active and self._stream is None:
            self.start()

    def set_speaking(self, speaking):
        """Duck while the AI talks; swell back up when it pauses."""
        self._speaking = bool(speaking)

    def set_volume(self, v):
        self.base_vol = max(0.0, min(1.0, float(v)))

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None


if __name__ == "__main__":
    import time
    import soundfile as sf
    m = BackgroundMusic()
    print("[MUSIC]", m.startup_check()[1])
    if m.loop is not None:
        sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "_bg_music_loop.wav"), m.loop, SR)
        print("[MUSIC] wrote _bg_music_loop.wav — playing 6s (duck demo)...")
        m.start(); m.set_active(True)
        time.sleep(2)
        print("  ducking (talking)..."); m.set_speaking(True); time.sleep(2)
        print("  pause (swell up)...");  m.set_speaking(False); time.sleep(2)
        m.stop()
