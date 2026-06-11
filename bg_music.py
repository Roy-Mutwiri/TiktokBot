# =============================================================================
# bg_music.py — looping background-music bed for the live stream.
#
# Plays a short music loop continuously via sounddevice, and DUCKS its volume
# under the bot's voice (swells back the instant the AI pauses) so commentary is
# always clear. Interface used by avatar_studio.py:
#     m = BackgroundMusic()
#     m.set_active(True/False)     # play while LIVE / stop
#     m.set_speaking(True/False)   # duck under the voice / swell back
# Never raises — if audio can't start, it's a silent no-op.
# =============================================================================
import os
import threading

import numpy as np

_PROJECT = os.path.dirname(os.path.abspath(__file__))
# First existing file wins. Drop your own track in as music/loop.wav to replace it.
_CANDIDATES = [
    os.environ.get("AVATAR_MUSIC_FILE", ""),
    os.path.join(_PROJECT, "music", "loop.wav"),
    os.path.join(_PROJECT, "bg_music.wav"),
    os.path.join(_PROJECT, "_bg_music_loop.wav"),
]
BASE_VOL = float(os.environ.get("AVATAR_MUSIC_VOL", "0.16"))    # music level when idle
DUCK_VOL = float(os.environ.get("AVATAR_MUSIC_DUCK", "0.045"))  # music level under voice
_RAMP = float(os.environ.get("AVATAR_MUSIC_RAMP", "0.12"))      # gain smoothing per block


class BackgroundMusic:
    """Continuous music loop with voice ducking (sounddevice output stream)."""

    def __init__(self):
        self._audio = None
        self._pos = 0
        self._gain = 0.0
        self._active = False
        self._speaking = False
        self._stream = None
        try:
            import sounddevice as sd
            import soundfile as sf
            path = next((p for p in _CANDIDATES if p and os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError("no background-music file found")
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            self._audio = data
            self._ch = data.shape[1]
            self._stream = sd.OutputStream(samplerate=sr, channels=self._ch,
                                           blocksize=1024, callback=self._callback)
            self._stream.start()
            print(f"[MUSIC] background bed ready ({os.path.basename(path)}, "
                  f"{sr} Hz, {len(data)/sr:.1f}s loop)")
        except Exception as exc:
            print(f"[MUSIC] background music unavailable ({exc}) — silent.")
            self._audio = None

    def _callback(self, outdata, frames, time_info, status):
        a = self._audio
        target = (DUCK_VOL if self._speaking else BASE_VOL) if self._active else 0.0
        if a is None or (target <= 0.0 and self._gain < 1e-4):
            outdata.fill(0.0)
            return
        end = self._pos + frames
        if end <= len(a):
            chunk = a[self._pos:end]
        else:                                   # wrap the loop seamlessly
            chunk = np.concatenate([a[self._pos:], a[:end - len(a)]])
        self._pos = end % len(a)
        g1 = self._gain + (target - self._gain) * _RAMP      # smooth gain ramp
        ramp = np.linspace(self._gain, g1, frames, dtype=np.float32).reshape(-1, 1)
        self._gain = g1
        outdata[:] = chunk * ramp

    def set_active(self, on):
        self._active = bool(on)

    def set_speaking(self, speaking):
        self._speaking = bool(speaking)

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
