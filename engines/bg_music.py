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
DUCK = float(os.environ.get("AVATAR_MUSIC_DUCK", "0.08"))
SONG_SECONDS = float(os.environ.get("AVATAR_MUSIC_SONG_SECONDS", "90"))  # per track
# HUGE pool of procedurally-unique tracks so we can go a long time with no repeat.
# 5000 tracks x ~26s = ~36 hours of music before the pool could even be exhausted.
NUM_SONGS = int(os.environ.get("AVATAR_MUSIC_SONGS", "20000"))
# Don't replay a track until this many DAYS have passed (persisted across sessions).
REPEAT_GAP = float(os.environ.get("AVATAR_MUSIC_REPEAT_DAYS", "2")) * 86400.0
# Persistent play history (seed -> last-played unix time) so "heard today, not
# again for ~2 days" survives restarts.
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(_PROJECT, "_music_history.json")
MUSIC_DIR = os.path.join(_PROJECT, "music", "streambeats")

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
STYLE_PROFILES = {
    "aggressive": dict(bpms=[145, 150, 155, 160], distort=2.2, hats=4, cow=0.12,
                       drums=1.15, lead=0.30, pad=0.20, groove="phonk",
                       voice="cow", bass=0.58),
    "melodic": dict(bpms=[118, 124, 128, 132], distort=1.0, hats=2, cow=0.07,
                    drums=0.85, lead=0.95, pad=0.90, groove="trap",
                    voice="bell", bass=0.42),
    "classic": dict(bpms=[128, 132, 138, 140], distort=1.5, hats=2, cow=0.17,
                    drums=1.00, lead=0.55, pad=0.45, groove="phonk",
                    voice="cow", bass=0.50),
    "dark": dict(bpms=[108, 116, 124, 132], distort=1.2, hats=2, cow=0.05,
                 drums=0.75, lead=0.55, pad=1.00, groove="halftime",
                 voice="pluck", bass=0.48),
    "synthwave": dict(bpms=[96, 104, 110, 118], distort=0.8, hats=2, cow=0.0,
                      drums=0.72, lead=1.00, pad=1.00, groove="four",
                      voice="arp", bass=0.34),
    "drill": dict(bpms=[138, 142, 146, 150], distort=1.8, hats=4, cow=0.0,
                  drums=1.10, lead=0.38, pad=0.35, groove="drill",
                  voice="pluck", bass=0.60),
    "lofi": dict(bpms=[78, 84, 90, 96], distort=0.55, hats=2, cow=0.0,
                 drums=0.52, lead=0.90, pad=0.90, groove="boombap",
                 voice="keys", bass=0.28),
    "cyber": dict(bpms=[148, 156, 164, 172], distort=1.7, hats=4, cow=0.0,
                  drums=1.00, lead=0.95, pad=0.55, groove="breakbeat",
                  voice="arp", bass=0.48),
    "house": dict(bpms=[120, 124, 126, 130], distort=0.75, hats=4, cow=0.0,
                  drums=0.90, lead=0.70, pad=0.75, groove="four",
                  voice="chord", bass=0.32),
    "ambient": dict(bpms=[72, 78, 84, 90], distort=0.45, hats=1, cow=0.0,
                    drums=0.12, lead=0.65, pad=1.35, groove="ambient",
                    voice="air", bass=0.16),
}
STYLES = list(STYLE_PROFILES)
LEAD_KINDS = ["bell", "saw", "square", "tri"]


def _style_for_seed(seed):
    rng = np.random.RandomState(seed * 2654435761 % (2**31))
    return STYLES[rng.randint(len(STYLES))]


def _discover_music_files():
    if not os.path.isdir(MUSIC_DIR):
        return []
    extensions = (".mp3", ".wav", ".flac", ".ogg")
    return sorted(
        os.path.join(MUSIC_DIR, name)
        for name in os.listdir(MUSIC_DIR)
        if name.lower().endswith(extensions)
    )


def _file_track_id(path):
    return "file:" + os.path.basename(path)


def _load_music_file(path):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SR:
        from scipy.signal import resample_poly
        divisor = np.gcd(sample_rate, SR)
        mono = resample_poly(mono, SR // divisor, sample_rate // divisor)
    peak = float(np.max(np.abs(mono))) or 1.0
    if peak > 0.98:
        mono = mono * (0.98 / peak)
    title = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    return mono.astype(np.float32), {
        "title": title,
        "style": "StreamBeats",
        "source": "file",
        "seconds": len(mono) / SR,
    }


def _make_track(seed):
    """Synthesise ONE unique seamless PHONK loop (mono float32) from `seed`.
    Each track picks a STYLE + key + tempo + chord prog + cowbell & lead motifs
    (A/B halves) so it has its own character — not a clone of the others."""
    rng = np.random.RandomState(seed * 2654435761 % (2**31))
    style = STYLES[rng.randint(len(STYLES))]
    profile = STYLE_PROFILES[style]
    bpm = float(rng.choice(profile["bpms"]))
    beat = 60.0 / bpm
    beats = 48
    n = int(SR * beat * beats)
    out = np.zeros(n, np.float32)
    root = float(rng.choice([48.99, 51.91, 55.0, 58.27, 61.74, 65.41]))   # G1..C2
    sc_name = rng.choice(["phrygian", "harmonic_minor"]) if style == "dark" \
        else rng.choice(list(DARK_SCALES.keys()))
    scale = DARK_SCALES[sc_name]
    prog = PROGRESSIONS[rng.randint(len(PROGRESSIONS))]
    # per-style character
    distort = profile["distort"]
    hat_div = profile["hats"]
    cow_gain = profile["cow"]
    drum_gain = profile["drums"]
    has_lead = rng.rand() < profile["lead"]
    lead_kind = LEAD_KINDS[rng.randint(len(LEAD_KINDS))]
    has_pad = rng.rand() < min(1.0, profile["pad"])

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

    def pluck(f, ln):
        tk = np.arange(ln) / SR
        s = _osc(f, tk, "tri") + 0.25 * np.sin(2*np.pi*f*2.0*tk)
        return s * np.exp(-tk * 7.0)

    def keys(f, ln):
        tk = np.arange(ln) / SR
        s = np.sin(2*np.pi*f*tk) + 0.28*np.sin(2*np.pi*f*2.0*tk)
        return s * _env(ln, 0.02, 0.18, 0.55, 0.38, 0.48)

    def chord(f, ln):
        tk = np.arange(ln) / SR
        s = sum(_osc(f * (2.0 ** (iv / 12.0)), tk, "saw") for iv in (0, 3, 7))
        return (s / 3.0) * _env(ln, 0.01, 0.10, 0.45, 0.24, 0.42)

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
    motifs = [
        [int(rng.choice([0, 0, 2, 3, 4, 5, 6, 7])) for _ in range(8)]
        for _ in range(4)
    ]

    # --- drums: kick pattern (repeats each 8-beat half, fill in the 2nd) ------
    groove = profile["groove"]
    kick_patterns = {
        "phonk": [0, 2, 3, 5, 6],
        "trap": [0, 2.5, 4, 6.75],
        "halftime": [0, 3.5, 6],
        "drill": [0, 1.75, 4.5, 6.25],
        "boombap": [0, 2.75, 4, 6.5],
        "breakbeat": [0, 1.5, 3, 4.75, 6.5],
        "four": list(range(8)),
        "ambient": [0],
    }
    snare_beats = {
        "phonk": [2, 6], "trap": [2, 6], "halftime": [4],
        "drill": [3, 7], "boombap": [2, 6], "breakbeat": [2, 5.5],
        "four": [2, 6], "ambient": [],
    }
    for block in range(0, beats, 8):
        for b in kick_patterns[groove]:
            add(at((block + b) * beat), kick(at(0.2)), 0.7 * drum_gain)
        for b in snare_beats[groove]:
            add(at((block + b) * beat), snare(at(0.14)), 0.30 * drum_gain)
    if groove == "four":
        hat_times = [k + 0.5 for k in range(beats)]
    elif groove == "ambient":
        hat_times = [k for k in range(0, beats, 4)]
    else:
        hat_times = [k / hat_div for k in range(beats * hat_div)]
    for index, b in enumerate(hat_times):
        gain = (0.045 if index % 2 else 0.075) * drum_gain
        add(at(b * beat), hat(at(0.03)), gain)
    if groove in ("phonk", "drill", "breakbeat"):
        for j in range(3):
            add(at((beats - 0.5) * beat + j*beat/3.0), hat(at(0.03)), 0.07)

    # --- cowbell melody (the hook) — A motif first half, B motif second -------
    step = beat / 2.0
    nsteps = beats * 2
    for k in range(nsteps):
        voice = profile["voice"]
        skip_chance = 0.62 if voice in ("air", "chord", "keys") else 0.28
        if rng.rand() < skip_chance:
            continue
        m = motifs[min(3, (k * 4) // nsteps)]
        deg = (prog[(k // 4) % len(prog)] + m[k % len(m)]) % len(scale)
        f = _semi(root, scale[deg] + int(rng.choice([24, 24, 26])))
        if voice == "cow":
            hook, hook_gain = cowbell(f, at(step*0.9)), cow_gain
        elif voice in ("pluck", "arp"):
            hook, hook_gain = pluck(f, at(step*0.9)), 0.10
        elif voice == "keys":
            hook, hook_gain = keys(f / 2.0, at(beat*1.8)), 0.11
        elif voice == "chord":
            hook, hook_gain = chord(f / 2.0, at(beat*0.9)), 0.09
        elif voice == "air":
            hook, hook_gain = keys(f / 2.0, at(beat*3.5)), 0.055
        else:
            hook, hook_gain = lead(f, at(step*0.9), "bell"), 0.08
        add(at(k*step), hook, hook_gain)

    # --- optional lead melody (quarter notes) — gives melodic tracks identity -
    if has_lead:
        for k in range(beats):
            if rng.rand() < 0.45:
                continue
            m = motifs[min(3, (k * 4) // beats)]
            deg = (prog[(k // 4) % len(prog)] + m[k % len(m)]) % len(scale)
            f = _semi(root, scale[deg] + int(rng.choice([12, 12, 24])))
            add(at(k*beat), lead(f, at(beat*0.85), lead_kind), 0.07)

    # --- 808 bass: chord roots, long gliding notes ---------------------------
    for b in range(0, beats, 2):
        deg = prog[(b // 2) % len(prog)]
        f = _semi(root, scale[deg % len(scale)])
        add(at(b*beat), b808(f, at(beat*1.85)), profile["bass"])

    # --- dark atmosphere pad -------------------------------------------------
    if has_pad:
        t = np.arange(n) / SR
        pad = np.zeros(n, np.float32)
        for iv in (0, 3, 7):
            pad += np.sin(2*np.pi*_semi(root*2, scale[iv % len(scale)])*t).astype(np.float32)
        pad *= (0.5 + 0.5*np.sin(2*np.pi*0.2*t)).astype(np.float32)
        out += pad * (0.03 * profile["pad"])

    # --- master: tape saturation glue + normalise + click-free loop seam -----
    out = np.tanh(out * 1.15)
    peak = float(np.max(np.abs(out))) or 1.0
    out = (out / peak) * 0.9
    xf = at(0.012)
    out[:xf] *= np.linspace(0, 1, xf)
    out[-xf:] *= np.linspace(1, 0, xf)
    return out.astype(np.float32), dict(
        bpm=int(bpm),
        style=style,
        scale=sc_name,
        groove=profile["groove"],
        voice=profile["voice"],
    )


class BackgroundMusic:
    """Generative, non-repeating PHONK playlist (own sounddevice stream)."""

    def __init__(self, volume=BASE_VOL):
        self.base_vol = float(volume)
        self.active = False
        self._speaking = False
        self._gain = 0.0
        self.audio_level = 0.0
        self._pos = 0
        self._stream = None
        self._alive = True
        self._state_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._rng = random.SystemRandom()
        self._library = _discover_music_files()
        # Persistent play history -> pick a track NOT heard in the last ~2 days, so
        # every session opens on a different song and repeats are spaced far apart.
        self._hist = self._load_history()
        try:
            if self._library:
                path = self._pick_file()
                if path is not None:
                    self._cur, self.meta = _load_music_file(path)
                    self._cur_seed = _file_track_id(path)
                    self._pos = 0
                else:
                    seed = self._pick_seed()
                    if seed is None:
                        raise RuntimeError("all music is inside the repeat cooldown")
                    self._cur, self.meta = _make_track(seed)
                    self._cur_seed = seed
                    self._pos = self._random_start_position(self._cur)
            else:
                recent_styles = self._recent_styles(limit=4)
                previous_seed = self._most_recent_seed()
                seed = self._pick_seed(
                    exclude=(() if previous_seed is None else (previous_seed,)),
                    avoid_styles=recent_styles,
                )
                if seed is None:
                    raise RuntimeError("all music seeds are inside the repeat cooldown")
                self._cur, self.meta = _make_track(seed)
                self._cur_seed = seed
                self._pos = self._random_start_position(self._cur)
            self._mark_played(self._cur_seed)
            self._next = None
            self._next_seed = None
            self._played_pending = None
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
            temp_file = HISTORY_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._hist, f)
            os.replace(temp_file, HISTORY_FILE)
        except Exception:
            pass

    def _most_recent_seed(self):
        with self._history_lock:
            if not self._hist:
                return None
            key = max(self._hist, key=lambda item: float(self._hist[item]))
        try:
            return int(key)
        except (TypeError, ValueError):
            return None

    def _recent_styles(self, limit=4):
        with self._history_lock:
            recent = sorted(
                self._hist.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        styles = []
        for key, _ in recent:
            try:
                style = _style_for_seed(int(key))
            except (TypeError, ValueError):
                continue
            if style not in styles:
                styles.append(style)
            if len(styles) >= limit:
                break
        return tuple(styles)

    def _pick_seed(self, exclude=(), avoid_styles=()):
        """Choose randomly without marking a queued track as played."""
        now = time.time()
        excluded = set(exclude)
        avoided = set(avoid_styles)
        with self._history_lock:
            eligible = [
                seed for seed in range(1, NUM_SONGS + 1)
                if seed not in excluded
                and now - float(self._hist.get(str(seed), 0.0)) >= REPEAT_GAP
            ]
        if avoided:
            different_style = [
                seed for seed in eligible if _style_for_seed(seed) not in avoided
            ]
            if different_style:
                eligible = different_style
        return self._rng.choice(eligible) if eligible else None

    def _pick_file(self, exclude=()):
        now = time.time()
        excluded = set(exclude)
        with self._history_lock:
            eligible = [
                path for path in self._library
                if _file_track_id(path) not in excluded
                and now - float(self._hist.get(_file_track_id(path), 0.0)) >= REPEAT_GAP
            ]
        return self._rng.choice(eligible) if eligible else None

    def _random_start_position(self, track):
        """Open at a random musical section so every app launch sounds fresh."""
        if track is None or len(track) < 8:
            return 0
        section = len(track) // 8
        return self._rng.randrange(8) * section

    def _mark_played(self, seed, played_at=None):
        """Persist a play only when the track becomes audible."""
        now = time.time() if played_at is None else float(played_at)
        with self._history_lock:
            self._hist[str(seed)] = now
            cutoff = now - REPEAT_GAP * 2
            self._hist = {
                key: value for key, value in self._hist.items()
                if float(value) >= cutoff
            }
            self._save_history()

    def startup_check(self):
        if self._cur is None:
            return False, "music unavailable (synth failed)."
        if self._library:
            return True, (
                f"real StreamBeats playlist - {len(self._library)} local songs, "
                f"random order, {REPEAT_GAP/3600:.0f}-hour cooldown."
            )
        return True, (f"generative PHONK playlist — {NUM_SONGS} unique tracks, no "
                      f"repeat for ~{REPEAT_GAP/86400:.0f} days (persisted), ducks under voice.")

    # -------------------------------------------------------------------------
    def _gen_loop(self):
        while self._alive:
            with self._state_lock:
                played_pending = self._played_pending
                self._played_pending = None
                needs_track = self._cur is not None and self._next is None
                current_seed = self._cur_seed
            if played_pending is not None:
                seed, played_at = played_pending
                self._mark_played(seed, played_at)
            if needs_track:
                try:
                    if self._library:
                        path = self._pick_file(exclude=(current_seed,))
                        if path is not None:
                            track_id = _file_track_id(path)
                            generated = _load_music_file(path)
                        else:
                            track_id = self._pick_seed(exclude=(current_seed,))
                            generated = (
                                None if track_id is None else _make_track(track_id)
                            )
                    else:
                        track_id = self._pick_seed(
                            exclude=(current_seed,),
                            avoid_styles=(self.meta["style"],),
                        )
                        generated = None if track_id is None else _make_track(track_id)
                    if generated is not None:
                        with self._state_lock:
                            if self._next is None:
                                self._next = generated
                                self._next_seed = track_id
                except Exception:
                    pass
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
            self._song_until = time.monotonic() + self._current_song_seconds()
        except Exception as exc:
            print(f"[MUSIC] could not open audio stream ({exc}).")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        now = time.monotonic()
        switched_seed = None
        with self._state_lock:
            if (self._song_until is not None and now >= self._song_until
                    and self._next is not None):
                self._cur, self.meta = self._next
                self._cur_seed = self._next_seed
                switched_seed = self._cur_seed
                self._next = None
                self._next_seed = None
                self._played_pending = (switched_seed, time.time())
                self._pos = (
                    0 if self.meta.get("source") == "file"
                    else self._random_start_position(self._cur)
                )
                self._song_until = now + self._current_song_seconds()
        cur = self._cur
        if switched_seed is not None:
            label = self.meta.get("title", switched_seed)
            details = self.meta.get("style", "unknown")
            if "bpm" in self.meta:
                details += f" {self.meta['bpm']} bpm"
            print(f"[MUSIC] now playing {label} ({details})")
        if cur is None:
            self.audio_level = 0.0
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
        output = buf * gains
        outdata[:, 0] = output
        self.audio_level = float(
            np.sqrt(np.mean(output * output)) + 1e-9)

    def skip(self):
        self._song_until = time.monotonic() - 1.0

    def _current_song_seconds(self):
        meta = getattr(self, "meta", {})
        if meta.get("source") == "file" and self._cur is not None:
            return max(1.0, len(self._cur) / SR)
        return SONG_SECONDS

    # -------------------------------------------------------------------------
    def set_active(self, on):
        self.active = bool(on)
        if not self.active:
            # A mute control must be immediate; do not apply the normal musical
            # fade used when ducking underneath speech.
            self._gain = 0.0
            self.audio_level = 0.0
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
        seeds = random.SystemRandom().sample(range(1, NUM_SONGS + 1), 6)
        for s in seeds:
            trk, meta = _make_track(s)
            print(f"  track seed={s}: {meta['style']} / {meta['bpm']} BPM")
            clips.append(trk)
            clips.append(np.zeros(int(SR*0.4), np.float32))
        sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "_bg_music_demo.wav"), np.concatenate(clips), SR)
        print("[MUSIC] wrote _bg_music_demo.wav (6 unique phonk tracks)")
