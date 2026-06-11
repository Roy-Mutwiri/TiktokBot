# =============================================================================
# startup_sound.py  —  the audio cues the avatar plays at each lifecycle stage
# -----------------------------------------------------------------------------
# Three distinct, JARVIS-style cues mark the boot sequence:
#   1. BOOT    (play_startup_sound) - deep power-up sweep + "systems online"
#                                     chime, the instant the bot starts.
#   2. CAMERA  (play_camera_sound)  - crisp scanner "lock" blips when the
#                                     virtual camera is detected / ready.
#   3. SCENE   (play_scene_sound)   - bright rising triad "go live" swell when
#                                     the avatar scene starts streaming.
#
# Each cue is non-blocking (daemon thread), crash-proof (errors swallowed), and
# overridable: drop your own clip in assets/ as jarvis_<cue>.wav (or .mp3):
#   assets/jarvis_startup.wav   assets/jarvis_camera.wav   assets/jarvis_scene.wav
#
# Env toggles:
#   AVATAR_SOUNDS=0       - disable ALL cues
#   AVATAR_BOOT_SOUND=0   - disable just the boot cue (back-compat)
# =============================================================================

import os
import threading

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 44100


# ---------------------------------------------------------------------------
# synthesizers (each returns float32 mono @ 44.1k) - kept sonically distinct
# ---------------------------------------------------------------------------
def _finish(out):
    import numpy as np
    out = np.tanh(out * 1.15)
    peak = float(np.max(np.abs(out))) or 1.0
    return ((out / peak) * 0.55).astype(np.float32), SR


def _synth_boot():
    """~1.7s deep power-up: sub thump + rising sweep + online chime."""
    import numpy as np
    total = int(1.7 * SR)
    out = np.zeros(total, dtype=np.float64)

    sub_n = int(0.45 * SR)
    st = np.arange(sub_n) / SR
    out[:sub_n] += 0.6 * np.sin(2 * np.pi * 52 * st) * np.exp(-5.0 * st)

    sw_dur = 0.95
    sw_n = int(sw_dur * SR)
    t = np.arange(sw_n) / SR
    f0, f1 = 150.0, 760.0
    inst_f = f0 * (f1 / f0) ** (t / sw_dur)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    sweep = (0.5 * np.sin(phase) + 0.22 * np.sin(phase * 1.004)
             + 0.14 * np.sin(2 * phase))
    attack = np.minimum(1.0, t / 0.04)
    decay = np.exp(-1.6 * np.maximum(0.0, t - 0.55))
    out[:sw_n] += sweep * attack * decay * 0.9

    chime_start = int(0.72 * SR)
    for freq, off in ((880.0, 0.0), (1318.5, 0.14)):
        cn = int(0.40 * SR)
        ct = np.arange(cn) / SR
        blip = (np.sin(2 * np.pi * freq * ct)
                + 0.35 * np.sin(2 * np.pi * 2 * freq * ct)) * np.exp(-6.5 * ct)
        s = chime_start + int(off * SR)
        out[s:s + cn] += 0.35 * blip
    return _finish(out)


def _synth_camera():
    """~0.55s crisp 'detected/lock' cue: a tiny click + two high rising pings."""
    import numpy as np
    total = int(0.55 * SR)
    out = np.zeros(total, dtype=np.float64)

    # short "click" (shutter-ish) at the very start - deterministic pseudo-noise
    cn = int(0.02 * SR)
    rng = np.linspace(1.0, 0.0, cn)
    idx = np.arange(cn)
    click = np.sin(idx * 12.9898) * 43758.5453
    click = (click - np.floor(click)) * 2 - 1
    out[:cn] += 0.25 * click * rng

    # two quick ascending pings (scanner lock): 1175 Hz then 1760 Hz
    for freq, start in ((1174.7, 0.06), (1760.0, 0.20)):
        pn = int(0.22 * SR)
        pt = np.arange(pn) / SR
        ping = (np.sin(2 * np.pi * freq * pt)
                + 0.3 * np.sin(2 * np.pi * 2 * freq * pt)) * np.exp(-14.0 * pt)
        s = int(start * SR)
        out[s:s + pn] += 0.5 * ping
    return _finish(out)


def _synth_scene():
    """~1.0s bright 'go live' rising major triad with a swelling pad."""
    import numpy as np
    total = int(1.05 * SR)
    out = np.zeros(total, dtype=np.float64)

    # arpeggio C5-E5-G5-C6, each note short and bright, stacking into a chord
    notes = [(523.25, 0.00), (659.25, 0.10), (783.99, 0.20), (1046.5, 0.30)]
    for freq, start in notes:
        nn = int(0.7 * SR)
        nt = np.arange(nn) / SR
        tone = (np.sin(2 * np.pi * freq * nt)
                + 0.25 * np.sin(2 * np.pi * 2 * freq * nt))
        env = np.minimum(1.0, nt / 0.01) * np.exp(-2.2 * nt)
        s = int(start * SR)
        end = min(total, s + nn)
        out[s:end] += 0.32 * (tone * env)[:end - s]

    # soft swelling pad underneath (held C major) for a "live!" body
    pn = int(0.9 * SR)
    pt = np.arange(pn) / SR
    pad = (np.sin(2 * np.pi * 261.63 * pt) + np.sin(2 * np.pi * 329.63 * pt)
           + np.sin(2 * np.pi * 392.0 * pt))
    swell = np.minimum(1.0, pt / 0.25) * np.exp(-1.0 * np.maximum(0.0, pt - 0.4))
    out[:pn] += 0.12 * pad * swell
    return _finish(out)


# ---------------------------------------------------------------------------
# playback / dispatch
# ---------------------------------------------------------------------------
def _load_clip(path):
    import numpy as np
    if path.lower().endswith(".mp3"):
        import subprocess, tempfile, shutil
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg needed for mp3 cue")
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
    try:
        import sounddevice as sd
        sd.play(data, sr)
        sd.wait()
        return
    except Exception:
        pass
    try:
        import numpy as np
        import tempfile
        import soundfile as sf
        import winsound
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
        winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
    except Exception:
        pass


def _candidates(cue):
    names = [f"jarvis_{cue}.wav", f"jarvis_{cue}.mp3", f"{cue}.wav"]
    return [os.path.join(PROJECT_DIR, "assets", n) for n in names]


def _run(cue, synth):
    try:
        clip = next((p for p in _candidates(cue) if os.path.exists(p)), None)
        data, sr = _load_clip(clip) if clip else synth()
        _play(data, sr)
    except Exception:
        pass   # never let a cue break the bot


def _cue(cue, synth, blocking=False, extra_off_env=None):
    if os.environ.get("AVATAR_SOUNDS", "1") == "0":
        return
    if extra_off_env and os.environ.get(extra_off_env, "1") == "0":
        return
    if blocking:
        _run(cue, synth)
    else:
        threading.Thread(target=_run, args=(cue, synth),
                         name=f"cue-{cue}", daemon=True).start()


def play_startup_sound(blocking=False):
    """BOOT cue - the instant the bot starts."""
    _cue("startup", _synth_boot, blocking, extra_off_env="AVATAR_BOOT_SOUND")


def play_camera_sound(blocking=False):
    """CAMERA cue - when the virtual camera is detected / ready."""
    _cue("camera", _synth_camera, blocking)


def play_scene_sound(blocking=False):
    """SCENE cue - when the avatar scene starts streaming (going live)."""
    _cue("scene", _synth_scene, blocking)


if __name__ == "__main__":
    import sys, time
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"boot": play_startup_sound, "camera": play_camera_sound,
           "scene": play_scene_sound}
    if which == "all":
        for name, fn in fns.items():
            print(f"playing {name}..."); fn(blocking=True); time.sleep(0.35)
    else:
        print(f"playing {which}..."); fns[which](blocking=True)
    print("done.")
