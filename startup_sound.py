# =============================================================================
# startup_sound.py  —  the "JARVIS boot" sound the avatar plays on startup
# -----------------------------------------------------------------------------
# When the bot comes online we play a short sci-fi power-up cue (a rising sweep
# + a soft "systems online" chime), like a HUD booting up. It is:
#   * non-blocking (plays on a daemon thread; never delays startup)
#   * crash-proof (any audio error is swallowed; the bot still boots)
#   * overridable: drop your OWN clip at  assets/jarvis_startup.wav  (or .mp3)
#     and that plays instead of the synthesized cue.
#
#   from startup_sound import play_startup_sound
#   play_startup_sound()
#
# Disable with the env var  AVATAR_BOOT_SOUND=0 .
# =============================================================================

import os
import threading

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 44100

# Where a user-supplied boot clip can live (first match wins).
_CUSTOM_CANDIDATES = [
    os.path.join(PROJECT_DIR, "assets", "jarvis_startup.wav"),
    os.path.join(PROJECT_DIR, "assets", "jarvis_startup.mp3"),
    os.path.join(PROJECT_DIR, "assets", "boot.wav"),
]


def _synth_boot():
    """Generate a ~1.7s JARVIS-style boot cue as float32 mono @ 44.1k."""
    import numpy as np

    total = int(1.7 * SR)
    out = np.zeros(total, dtype=np.float64)

    # 1) low "power on" sub thump at t=0
    sub_n = int(0.45 * SR)
    st = np.arange(sub_n) / SR
    out[:sub_n] += 0.6 * np.sin(2 * np.pi * 52 * st) * np.exp(-5.0 * st)

    # 2) rising power-up sweep (exponential 150 -> 760 Hz) with shimmer layers
    sw_dur = 0.95
    sw_n = int(sw_dur * SR)
    t = np.arange(sw_n) / SR
    f0, f1 = 150.0, 760.0
    inst_f = f0 * (f1 / f0) ** (t / sw_dur)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    sweep = (0.5 * np.sin(phase)
             + 0.22 * np.sin(phase * 1.004)      # slight detune = shimmer
             + 0.14 * np.sin(2 * phase))          # octave harmonic
    attack = np.minimum(1.0, t / 0.04)
    decay = np.exp(-1.6 * np.maximum(0.0, t - 0.55))
    out[:sw_n] += sweep * attack * decay * 0.9

    # 3) soft "systems online" double chime after the sweep peaks
    chime_start = int(0.72 * SR)
    for freq, off in ((880.0, 0.0), (1318.5, 0.14)):   # A5 then E6
        cn = int(0.40 * SR)
        ct = np.arange(cn) / SR
        blip = (np.sin(2 * np.pi * freq * ct)
                + 0.35 * np.sin(2 * np.pi * 2 * freq * ct)) * np.exp(-6.5 * ct)
        s = chime_start + int(off * SR)
        out[s:s + cn] += 0.35 * blip

    # gentle soft-clip + normalize to a comfortable level
    out = np.tanh(out * 1.15)
    peak = float(np.max(np.abs(out))) or 1.0
    out = (out / peak) * 0.55
    return out.astype(np.float32), SR


def _load_clip(path):
    """Load a user clip (wav via soundfile, mp3 via ffmpeg) -> (float32, sr)."""
    import numpy as np
    if path.lower().endswith(".mp3"):
        import subprocess, tempfile, shutil
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg needed for mp3 boot clip")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run([ff, "-y", "-i", path, "-ar", str(SR), "-ac", "1", tmp.name],
                       capture_output=True, check=True)
        path = tmp.name
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1).astype("float32")
    return data, sr


def _play(data, sr):
    """Play a float32 mono array. Prefer sounddevice; fall back to winsound."""
    try:
        import sounddevice as sd
        sd.play(data, sr)
        sd.wait()
        return
    except Exception:
        pass
    # fallback: write a temp PCM16 wav and use winsound (Windows)
    try:
        import numpy as np
        import tempfile
        import soundfile as sf
        import winsound
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        pcm = np.clip(data, -1.0, 1.0)
        sf.write(tmp.name, pcm, sr, subtype="PCM_16")
        winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
    except Exception:
        pass


def _run():
    try:
        clip = next((p for p in _CUSTOM_CANDIDATES if os.path.exists(p)), None)
        if clip:
            data, sr = _load_clip(clip)
        else:
            data, sr = _synth_boot()
        _play(data, sr)
    except Exception:
        pass   # never let a boot sound break startup


def play_startup_sound(blocking=False):
    """Play the JARVIS boot cue. Returns immediately unless blocking=True."""
    if os.environ.get("AVATAR_BOOT_SOUND", "1") == "0":
        return
    if blocking:
        _run()
    else:
        threading.Thread(target=_run, name="jarvis-boot", daemon=True).start()


if __name__ == "__main__":
    # quick manual test:  python startup_sound.py
    print("playing JARVIS boot cue...")
    play_startup_sound(blocking=True)
    print("done.")
