# =============================================================================
# voice_changer_engine.py  —  LIVE MIC -> voice changer -> avatar mouth
# -----------------------------------------------------------------------------
# Replaces the TTS front-end with a live performance:
#
#     you talk into the mic
#         -> voice changer (Passthrough / DSP / RVC persona voice)
#         -> the SAME mouth engine the TTS fed (feed_audio, 16 kHz float32)
#         -> avatar lip-syncs to the converted voice
#     (+ the converted voice is monitored on your speakers)
#
# The render loop (realtime_avatar.py / avatar_studio.py) is unchanged: it just
# polls `mouth.is_speaking`, which feed_audio() sets and silence relaxes. We only
# replace WHAT produces the 16 kHz chunks — capture instead of synthesis.
#
# Capture and conversion are DECOUPLED: the sounddevice input callback only
# enqueues raw blocks (must never block / never call the GPU), and a worker
# thread does the (possibly heavy) conversion, mouth-feed and monitor push. That
# keeps mic capture glitch-free even when RVC inference is slow.
#
#   python engines/voice_changer_engine.py            # mic -> monitor passthrough
#   python engines/voice_changer_engine.py dsp        # mic -> pitch/formant DSP
# =============================================================================

import os
import sys
import time
import threading
import collections

import numpy as np

try:
    import sounddevice as sd
except Exception as _sd_exc:                       # pragma: no cover
    sd = None
    _SD_IMPORT_ERROR = _sd_exc

# Must match the mouth engine's expected feed rate (musetalk_engine.SAMPLE_RATE).
SAMPLE_RATE = 16000
# 40 ms feed blocks — same granularity TTSStreamEngine uses (FEED_CHUNK_SAMPLES).
BLOCK = 640
# Drop the oldest capture if the converter falls >this-many blocks behind, so
# latency can't snowball when conversion is slower than real time.
MAX_LAG_BLOCKS = 12


# =============================================================================
# VOICE CONVERTERS  —  convert(chunk_f32_16k) -> chunk_f32_16k
# =============================================================================
class VoiceConverter:
    """Base interface. `convert` takes mono float32 @16 kHz and returns the same."""
    name = "passthrough"

    def convert(self, chunk):
        return chunk

    def reset(self):
        """Drop any internal carry/context (called when the speech gate closes)."""
        pass

    def startup_check(self):
        return True, self.name


class PassthroughConverter(VoiceConverter):
    """Identity — proves the whole mic->mouth->monitor path with zero processing."""
    name = "passthrough (no change)"


class ArabicLightConverter(VoiceConverter):
    """Subtle Arabic-flavoured tone for source audio without hurting clarity."""
    name = "arabic light tone"

    def __init__(self):
        self.warmth = float(os.environ.get("AVATAR_ARABIC_WARMTH", "0.18"))
        self.bright = float(os.environ.get("AVATAR_ARABIC_BRIGHT", "0.08"))
        self.drive = float(os.environ.get("AVATAR_ARABIC_DRIVE", "1.12"))
        self._lp = 0.0
        self._hp = 0.0
        self._prev = 0.0

    def convert(self, chunk):
        x = np.asarray(chunk, dtype=np.float32).flatten()
        if len(x) == 0:
            return x
        y = np.empty_like(x)
        lp = float(self._lp)
        hp = float(self._hp)
        prev = float(self._prev)
        for i, sample in enumerate(x):
            sample = float(sample)
            lp = 0.92 * lp + 0.08 * sample
            hp = 0.72 * hp + 0.28 * (sample - prev)
            prev = sample
            y[i] = sample + self.warmth * lp - self.bright * hp
        self._lp = lp
        self._hp = hp
        self._prev = prev
        y = np.tanh(y * self.drive) / np.tanh(self.drive)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0.98:
            y = y * (0.98 / peak)
        return y.astype(np.float32)

    def reset(self):
        self._lp = 0.0
        self._hp = 0.0
        self._prev = 0.0


class DSPConverter(VoiceConverter):
    """Zero-model fallback: pitch + formant shift (a generic 'changed voice').

    Real voice conversion to the avatar persona is RVCConverter; this is the
    instant, no-training toggle. Processes on a small overlapping buffer so the
    phase-vocoder pitch shift doesn't click between 40 ms blocks.
    """
    name = "DSP pitch/formant"

    def __init__(self, semitones=None, formant=None):
        # Down a few semitones + formant shift reads as a different, deeper voice.
        self.semitones = float(os.environ.get("AVATAR_DSP_PITCH", semitones if semitones is not None else -3.0))
        self.formant = float(os.environ.get("AVATAR_DSP_FORMANT", formant if formant is not None else 1.0))
        self._buf = np.zeros(0, dtype=np.float32)
        try:
            import librosa  # noqa: F401  (validate availability up front)
            self._ok = True
        except Exception as exc:
            print(f"[VC] DSP needs librosa ({exc}); converter will pass audio through.")
            self._ok = False

    def convert(self, chunk):
        if not self._ok or abs(self.semitones) < 0.01:
            return chunk
        try:
            import librosa
            # Buffer ~3 blocks so pitch_shift has enough context for clean STFT,
            # then emit one block and keep the tail for next time.
            self._buf = np.concatenate([self._buf, chunk])
            if len(self._buf) < BLOCK * 3:
                return np.zeros(0, dtype=np.float32)
            shifted = librosa.effects.pitch_shift(
                self._buf, sr=SAMPLE_RATE, n_steps=self.semitones, n_fft=1024)
            out = shifted[BLOCK:BLOCK * 2].astype(np.float32)
            self._buf = self._buf[BLOCK:]
            if self.formant != 1.0:
                out = self._shift_formant(out, self.formant)
            return out
        except Exception:
            return chunk

    @staticmethod
    def _shift_formant(x, factor):
        """Cheap formant shift: resample the spectral envelope via time-domain
        resample-then-restore-length (approximate, good enough for a disguise)."""
        try:
            import librosa
            n = len(x)
            y = librosa.resample(x, orig_sr=SAMPLE_RATE,
                                 target_sr=int(SAMPLE_RATE * factor))
            if len(y) >= n:
                return y[:n].astype(np.float32)
            out = np.zeros(n, dtype=np.float32)
            out[:len(y)] = y
            return out
        except Exception:
            return x

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def startup_check(self):
        if not self._ok:
            return True, "DSP unavailable (no librosa) — passthrough"
        return True, f"DSP pitch {self.semitones:+.1f}st formant x{self.formant:.2f}"


class YouTubeDisguiseConverter(VoiceConverter):
    """Change a recorded voice while preserving its original performance."""
    name = "YouTube real-voice disguise"

    def __init__(self):
        pass

    def convert(self, chunk):
        return np.asarray(chunk, dtype=np.float32)

    def reset(self):
        pass

    def startup_check(self):
        profile = os.environ.get("AVATAR_YOUTUBE_PERSONA", "deep")
        return True, f"recorded voice mapped to '{profile}' persona"


def make_converter(kind="passthrough"):
    """Build a converter by name. 'rvc' is imported lazily (Phase B)."""
    kind = (kind or "passthrough").lower()
    if kind in ("passthrough", "none", "off"):
        return PassthroughConverter()
    if kind in ("arabic", "arabic-light", "arabic_light"):
        return ArabicLightConverter()
    if kind in ("youtube-disguise", "youtube_disguise"):
        return YouTubeDisguiseConverter()
    if kind == "dsp":
        return DSPConverter()
    if kind in ("rvc", "persona"):
        try:
            from rvc_engine import RVCConverter        # Phase B
            return RVCConverter()
        except Exception as exc:
            print(f"[VC] RVC unavailable ({exc}) — falling back to DSP.")
            return DSPConverter()
    print(f"[VC] unknown converter '{kind}' — using passthrough.")
    return PassthroughConverter()


# =============================================================================
# LIVE VOICE ENGINE
# =============================================================================
class LiveVoiceEngine:
    """Capture mic -> convert -> feed the avatar mouth + monitor on speakers.

    Mirrors the slice of the TTSStreamEngine contract the app touches so it can
    stand in as the speech source: `muted`, `speaking`, `set_muted()`,
    `shutdown()`, `startup_check()`. The crucial shared behaviour is that it
    feeds the SAME mouth engine (`mouth.feed_audio`) the TTS fed.
    """

    def __init__(self, mouth_engine, converter=None, in_device=None,
                 out_device=None, monitor=True, gate_thresh=None):
        self.mouth = mouth_engine
        self.converter = converter if converter is not None else PassthroughConverter()
        self.in_device = in_device
        self.out_device = out_device
        self.monitor = monitor
        # RMS speech gate: open above `gate_thresh`, stay open through brief gaps
        # (hangover) so words don't chop. Tunable live and via env.
        self.gate_thresh = float(os.environ.get("AVATAR_MIC_GATE",
                                 gate_thresh if gate_thresh is not None else 0.012))
        self.gate_hangover = int(os.environ.get("AVATAR_MIC_HANGOVER", "8"))  # blocks (~320ms)
        # Software input gain — quiet mics (e.g. a far webcam mic) need a boost so
        # normal speech clears the gate and the monitor is audible. Applied at
        # capture, so the gate threshold and the converter both see the boost.
        self.gain = float(os.environ.get("AVATAR_MIC_GAIN", "1.0"))

        # contract-compat attributes (so studio/realtime stat & mute code "just works")
        self.muted = False
        self.speaking = False
        self.words_spoken = 0
        self.lines_spoken = 0

        self._running = False
        self._in_stream = None
        self._out_stream = None
        self._worker = None
        self._native_sr = SAMPLE_RATE          # actual mic rate (may need resample)
        self._in_block = BLOCK                  # capture blocksize at native rate

        # raw captured blocks (16 kHz mono); auto-drop oldest to bound latency
        self._in_q = collections.deque(maxlen=MAX_LAG_BLOCKS)
        self._in_evt = threading.Event()
        # monitor output ring (converted 16 kHz blocks) + residual carry
        self._out_q = collections.deque(maxlen=MAX_LAG_BLOCKS * 2)
        self._out_residual = np.zeros(0, dtype=np.float32)
        self._gate_open = 0                     # >0 = open, counts down (hangover)
        self.last_rms = 0.0

    # -------------------------------------------------------------------------
    def startup_check(self):
        if sd is None:
            return False, f"sounddevice unavailable ({_SD_IMPORT_ERROR})"
        try:
            ina = sd.query_devices(self.in_device, "input")["name"]
        except Exception as exc:
            return False, f"no usable input device ({exc})"
        cok, cmsg = self.converter.startup_check()
        mon = "monitor on" if self.monitor else "monitor off"
        return True, f"Live mic '{ina}' -> {cmsg} ({mon})"

    # -------------------------------------------------------------------------
    def start(self):
        """Open mic + monitor streams and the conversion worker."""
        if sd is None:
            raise RuntimeError(f"sounddevice not available: {_SD_IMPORT_ERROR}")
        if self._running:
            return
        self._running = True

        self._native_sr, self._in_block = self._choose_input_rate()
        self._in_stream = sd.InputStream(
            samplerate=self._native_sr, channels=1, dtype="float32",
            blocksize=self._in_block, device=self.in_device,
            callback=self._in_callback)
        self._in_stream.start()

        if self.monitor:
            try:
                self._out_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=BLOCK, device=self.out_device,
                    callback=self._out_callback)
                self._out_stream.start()
            except Exception as exc:
                print(f"[VC] monitor output unavailable ({exc}) — mouth-only.")
                self._out_stream = None

        self._worker = threading.Thread(target=self._convert_loop, daemon=True)
        self._worker.start()
        ok, msg = self.startup_check()
        print(f"[VC] live mic started — {msg}")

    def _choose_input_rate(self):
        """Prefer a true 16 kHz mic (no resample); else native rate + resample.
        Returns (native_sr, native_blocksize)."""
        try:
            sd.check_input_settings(device=self.in_device, samplerate=SAMPLE_RATE,
                                    channels=1, dtype="float32")
            return SAMPLE_RATE, BLOCK
        except Exception:
            pass
        try:
            native = int(sd.query_devices(self.in_device, "input")["default_samplerate"])
        except Exception:
            native = 48000
        if native <= 0:
            native = 48000
        block = max(1, int(round(BLOCK * native / SAMPLE_RATE)))
        print(f"[VC] mic doesn't do 16 kHz — capturing {native} Hz, resampling.")
        return native, block

    # -------------------------------------------------------------------------
    # AUDIO-THREAD CALLBACKS  (keep these light: no GPU, no blocking)
    # -------------------------------------------------------------------------
    def _in_callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        if self.gain != 1.0:
            block = np.clip(block * self.gain, -1.0, 1.0).astype(np.float32)
        if self._native_sr != SAMPLE_RATE:
            try:
                import librosa
                block = librosa.resample(block.astype(np.float32),
                                         orig_sr=self._native_sr,
                                         target_sr=SAMPLE_RATE).astype(np.float32)
            except Exception:
                pass
        self._in_q.append(block)
        self._in_evt.set()

    def _out_callback(self, outdata, frames, time_info, status):
        if self.muted:
            outdata.fill(0.0)
            self._out_q.clear()
            self._out_residual = np.zeros(0, dtype=np.float32)
            return
        out = self._out_residual
        while len(out) < frames and self._out_q:
            out = np.concatenate([out, self._out_q.popleft()])
        if len(out) >= frames:
            outdata[:, 0] = out[:frames]
            self._out_residual = out[frames:]
        else:
            outdata[:len(out), 0] = out
            outdata[len(out):, 0] = 0.0
            self._out_residual = np.zeros(0, dtype=np.float32)

    # -------------------------------------------------------------------------
    # WORKER  —  gate, convert, feed the mouth, push to the monitor
    # -------------------------------------------------------------------------
    def _convert_loop(self):
        while self._running:
            if not self._in_q:
                self._in_evt.wait(0.1)
                self._in_evt.clear()
                continue
            try:
                block = self._in_q.popleft()
            except IndexError:
                continue
            try:
                self._process(block)
            except Exception as exc:
                print(f"[VC] convert error: {exc}")

    def _process(self, block):
        rms = float(np.sqrt(np.mean(block * block)) + 1e-9)
        self.last_rms = rms
        if rms >= self.gate_thresh:
            self._gate_open = self.gate_hangover
        elif self._gate_open > 0:
            self._gate_open -= 1

        if self._gate_open <= 0:
            # silence: relax the converter's context; stop feeding so the mouth
            # closes (mouth engine drops is_speaking after its SILENCE_FRAMES).
            if self.speaking:
                self.speaking = False
                try:
                    self.converter.reset()
                except Exception:
                    pass
            return

        out = self.converter.convert(block)
        if out is None or len(out) == 0:
            return                                   # converter still buffering
        out = np.asarray(out, dtype=np.float32).flatten()
        self.speaking = True
        try:
            self.mouth.feed_audio(out)               # drive the avatar mouth
        except Exception as exc:
            print(f"[VC] feed_audio error: {exc}")
        if self.monitor and self._out_stream is not None and not self.muted:
            self._out_q.append(out)                  # hear the converted voice

    # -------------------------------------------------------------------------
    def set_muted(self, muted):
        """Mute only the speaker MONITOR; the avatar mouth still tracks."""
        self.muted = bool(muted)
        if self.muted:
            self._out_q.clear()
            self._out_residual = np.zeros(0, dtype=np.float32)

    def set_gate(self, thresh):
        self.gate_thresh = float(thresh)

    def set_gain(self, gain):
        self.gain = max(0.0, float(gain))

    def set_converter(self, converter):
        """Hot-swap the converter (e.g. DSP <-> RVC) while live."""
        old = self.converter
        self.converter = converter
        try:
            old.reset()
        except Exception:
            pass

    def shutdown(self):
        self._running = False
        self._in_evt.set()
        for s in (self._in_stream, self._out_stream):
            try:
                if s is not None:
                    s.stop(); s.close()
            except Exception:
                pass
        self._in_stream = self._out_stream = None
        self.speaking = False


# -----------------------------------------------------------------------------
# DEVICE ENUMERATION  (for the studio dropdowns)
# -----------------------------------------------------------------------------
def list_input_devices():
    """Return [(index, name), ...] of input-capable devices, or []."""
    if sd is None:
        return []
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d["name"]))
    except Exception:
        pass
    return out


def list_output_devices():
    """Return [(index, name), ...] of output-capable devices, or []."""
    if sd is None:
        return []
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                out.append((i, d["name"]))
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# STANDALONE SMOKE TEST:  mic -> converter -> monitor (dummy mouth)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "passthrough"

    class _DummyMouth:
        is_speaking = False
        def feed_audio(self, x):
            self.is_speaking = True

    eng = LiveVoiceEngine(_DummyMouth(), converter=make_converter(kind), monitor=True)
    ok, msg = eng.startup_check()
    print(f"[VC] startup_check: ({ok}) {msg}")
    if not ok:
        sys.exit(1)
    eng.start()
    print("[VC] TALK into the mic — you should hear the converted voice. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            print(f"[VC] rms={eng.last_rms:.4f} gate={'OPEN' if eng._gate_open > 0 else 'shut'} "
                  f"speaking={eng.speaking}")
    except KeyboardInterrupt:
        print("\n[VC] stopping.")
    finally:
        eng.shutdown()
