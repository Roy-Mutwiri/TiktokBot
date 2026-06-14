import os
import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from tts_stream_engine import FFMPEG
from voice_changer_engine import BLOCK, SAMPLE_RATE, make_converter
from youtube_cache import audio_dir, get_cached_audio, save_audio

MONITOR_RATE = 48000
MONITOR_BLOCK = BLOCK * (MONITOR_RATE // SAMPLE_RATE)
DEMUCS_MODEL_FILENAME = "955717e8-8726e21a.th"
DEMUCS_MODEL_HASH_PREFIX = "8726e21a"
DEMUCS_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
    + DEMUCS_MODEL_FILENAME
)

try:
    import sounddevice as sd
except Exception as _sd_exc:  # pragma: no cover
    sd = None
    _SD_IMPORT_ERROR = _sd_exc


class YouTubeAudioError(RuntimeError):
    pass


class YouTubeAudioPlayer:
    """Play YouTube audio through the avatar mouth, optionally style-converted.

    This keeps the original timing/cadence from the video. It does not transcribe
    or regenerate speech with TTS. The selected converter changes the audio color
    toward the local avatar style where a converter is available.
    """

    def __init__(self, mouth_engine, converter_kind="youtube-disguise", monitor=True,
                 status_callback=None, out_device=None, persona="deep_male",
                 smooth_transition=True):
        self.mouth = mouth_engine
        self.converter_kind = converter_kind or "youtube-disguise"
        self.converter = make_converter(self.converter_kind)
        self.monitor = monitor
        self.out_device = out_device
        self.persona = persona or "deep_male"
        self.smooth_transition = bool(smooth_transition)
        self.status_callback = status_callback
        self.muted = False
        self.speaking = False
        self.audio_level = 0.0
        self.title = ""
        self.status = "idle"
        self.last_error = ""
        self.position_blocks = 0
        self.duration = 0.0
        self.gain = float(os.environ.get("AVATAR_YOUTUBE_GAIN", "1.10"))
        self.fade_blocks = max(1, int(os.environ.get("AVATAR_YOUTUBE_FADE_BLOCKS", "8")))
        self.noise_floor = float(os.environ.get("AVATAR_YOUTUBE_NOISE_FLOOR", "0.00018"))

        self._running = False
        self._paused = threading.Event()
        self._paused.set()
        self._thread = None
        self._proc = None
        self._out = None
        self._fade_pos = 0
        self._last_styled = None
        self._pause_release_pending = False

    def start(self, url, start_seconds=None, end_seconds=None):
        if self._running:
            self.stop()
        audio_url, title, duration, cache_hit = _resolve_audio_source(
            url, self._set_status)
        if os.environ.get("AVATAR_YOUTUBE_VOCALS_ONLY", "1") == "1":
            audio_url = _isolate_vocals(audio_url, self._set_status)
        self.title = title
        self.duration = float(duration or 0.0)
        self.start_seconds = float(start_seconds or 0.0)
        self.end_seconds = float(end_seconds) if end_seconds is not None else None
        if self.end_seconds is not None and self.end_seconds > self.start_seconds:
            self.duration = self.end_seconds
        self.position_blocks = 0
        self.converter = make_converter(self.converter_kind)
        _ok, converter_status = self.converter.startup_check()
        self._set_status(f"voice transform: {converter_status}")
        self._running = True
        self._paused.clear()
        self._fade_pos = 0
        self._last_styled = None
        self._pause_release_pending = False
        if cache_hit:
            self._set_status("found audio in local db/cache")
        if self.start_seconds > 0:
            self._set_status(f"seeking exact time {format_time(self.start_seconds)}")
        else:
            self._set_status("buffering youtube audio")
        self._thread = threading.Thread(
            target=self._run, args=(audio_url,), daemon=True)
        self._thread.start()

    def pause(self):
        self._paused.set()
        self._pause_release_pending = True
        self.speaking = False
        self.audio_level = 0.0
        self._set_status("paused")

    def resume(self):
        self._paused.clear()
        self._fade_pos = 0
        self._pause_release_pending = False
        if self._running:
            self._set_status("resuming")

    def stop(self):
        self._running = False
        self._paused.clear()
        try:
            if self._proc is not None:
                self._proc.kill()
        except Exception:
            pass
        try:
            if self._out is not None:
                self._out.stop()
                self._out.close()
        except Exception:
            pass
        self._proc = None
        self._out = None
        self._last_styled = None
        self._pause_release_pending = False
        self.speaking = False
        self.audio_level = 0.0
        if self.status not in ("ended", "failed"):
            self._set_status("stopped")

    def set_muted(self, muted):
        self.muted = bool(muted)
        self.audio_level = 0.0 if self.muted else self.audio_level
        stream = self._out
        if stream is None:
            return
        try:
            if self.muted:
                # Abort flushes any samples already buffered by PortAudio.
                stream.abort()
            else:
                stream.start()
        except Exception:
            pass

    @property
    def position_seconds(self):
        return self.start_seconds + self.position_blocks * BLOCK / float(SAMPLE_RATE)

    def _run(self, audio_url):
        try:
            cmd = _build_ffmpeg_cmd(
                audio_url, self.start_seconds, self.end_seconds,
                voice_disguise=self.converter_kind in (
                    "youtube-disguise", "youtube_disguise"),
                persona=self.persona)
            self._set_status("starting ffmpeg")
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if self.monitor and sd is not None:
                try:
                    self._out = sd.OutputStream(
                        samplerate=MONITOR_RATE, channels=1, dtype="float32",
                        blocksize=MONITOR_BLOCK, device=self.out_device)
                    self._out.start()
                except Exception as exc:
                    raise YouTubeAudioError(
                        f"audio output unavailable ({exc})") from exc
            elif self.monitor:
                raise YouTubeAudioError(
                    f"sounddevice unavailable ({_SD_IMPORT_ERROR})")

            nbytes = MONITOR_BLOCK * 2
            while self._running:
                if self._paused.is_set():
                    if self._pause_release_pending:
                        self._pause_release_pending = False
                        self._emit_release_tail()
                    self.speaking = False
                    time.sleep(0.05)
                    continue
                raw = self._proc.stdout.read(nbytes) if self._proc.stdout else b""
                if len(raw) < nbytes:
                    return_code = self._proc.wait()
                    error_text = ""
                    if self._proc.stderr is not None:
                        error_text = self._proc.stderr.read().decode(
                            "utf-8", "replace").strip()
                    if return_code != 0:
                        raise YouTubeAudioError(
                            error_text or f"ffmpeg exited with code {return_code}")
                    break
                block = pcm16_bytes_to_float(raw)
                if self.gain != 1.0:
                    block = np.clip(block * self.gain, -1.0, 1.0).astype(np.float32)
                styled = self.converter.convert(block)
                if styled is None or len(styled) == 0:
                    continue
                styled = np.asarray(styled, dtype=np.float32).flatten()
                styled = self._smooth_block(styled)
                self._last_styled = styled
                self.position_blocks += 1
                self.speaking = True
                self.audio_level = float(
                    np.sqrt(np.mean(styled * styled)) + 1e-9)
                if self.position_blocks == 1:
                    self._set_status("playing youtube voice")
                try:
                    # The monitor stays full-bandwidth at 48 kHz. MuseTalk only
                    # needs a 16 kHz amplitude stream, so decimate its private copy.
                    self.mouth.feed_audio(
                        np.ascontiguousarray(styled[::3], dtype=np.float32))
                except Exception:
                    pass
                if self._out is not None and not self.muted:
                    try:
                        self._out.write(styled.reshape(-1, 1))
                    except Exception:
                        pass
            self._set_status("ended")
        except Exception as exc:
            self.last_error = str(exc)
            self._set_status(f"failed: {exc}")
        finally:
            self.stop()

    def _set_status(self, status):
        self.status = status
        cb = self.status_callback
        if cb is not None:
            try:
                cb(status)
            except Exception:
                pass

    def _smooth_block(self, samples):
        if not self.smooth_transition or samples.size == 0:
            return samples
        out = np.asarray(samples, dtype=np.float32).copy()
        fade_len = max(1, self.fade_blocks * max(1, out.size))
        start = self._fade_pos
        stop = start + out.size
        if start < fade_len:
            env = np.linspace(
                start / fade_len,
                min(1.0, stop / fade_len),
                out.size,
                endpoint=False,
                dtype=np.float32,
            )
            out *= env
            self._fade_pos = stop
        if self.noise_floor > 0.0:
            rms = float(np.sqrt(np.mean(out * out)) + 1e-9)
            if rms < self.noise_floor * 2.0:
                out += np.random.normal(
                    0.0, self.noise_floor, out.size).astype(np.float32)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _emit_release_tail(self):
        if not self.smooth_transition:
            return
        last = self._last_styled
        if last is None or len(last) == 0:
            return
        base = np.asarray(last, dtype=np.float32).flatten()
        tails = min(4, self.fade_blocks)
        for i in range(tails):
            if not self._running:
                break
            gain = max(0.0, 1.0 - (i + 1) / float(tails))
            tail = (base * gain).astype(np.float32)
            try:
                self.mouth.feed_audio(
                    np.ascontiguousarray(tail[::3], dtype=np.float32))
            except Exception:
                pass
            if self._out is not None and not self.muted:
                try:
                    self._out.write(tail.reshape(-1, 1))
                except Exception:
                    pass


def _resolve_audio_source(url, status_callback=None):
    cached = get_cached_audio(url)
    if cached is not None:
        _status(status_callback, "found audio in local db/cache")
        return cached["audio_path"], cached["title"], cached["duration"], True

    _status(status_callback, "new link - downloading youtube audio to cache")
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeAudioError(f"yt-dlp is not installed ({exc})")
    out_dir = audio_dir(url)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "audio.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise YouTubeAudioError(f"could not read YouTube audio ({exc})")
    title = (info or {}).get("title") or "YouTube audio"
    duration = float((info or {}).get("duration") or 0.0)
    audio_path = _find_downloaded_audio(out_dir)
    if not audio_path:
        audio_path = (info or {}).get("requested_downloads", [{}])[0].get("filepath")
    if not audio_path or not os.path.exists(audio_path):
        raise YouTubeAudioError("audio download finished but no cached file was found")
    save_audio(url, title, duration, audio_path)
    _status(status_callback, "saved audio in local db/cache")
    return audio_path, title, duration, False


def _find_downloaded_audio(out_dir):
    files = [
        p for p in glob.glob(os.path.join(out_dir, "audio.*"))
        if os.path.isfile(p) and not p.endswith(".part")
    ]
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _isolate_vocals(audio_path, status_callback=None):
    """Return a cached Demucs vocal stem; never fall back to music-mixed audio."""
    source = os.path.abspath(audio_path)
    cache_dir = os.path.dirname(source)
    vocals_path = os.path.join(cache_dir, "vocals.wav")
    if os.path.exists(vocals_path) and os.path.getsize(vocals_path) > 4096:
        _status(status_callback, "found isolated vocals in local cache")
        return vocals_path

    _status(status_callback, "isolating voice and removing background music")
    _prepare_demucs_checkpoint(status_callback)
    work_dir = os.path.join(cache_dir, "demucs_stems")
    os.makedirs(work_dir, exist_ok=True)
    if _low_commit_headroom():
        _status(
            status_callback,
            "voice isolation 5% - using low-memory live mode")
        return _isolate_dialogue_low_memory(
            source, vocals_path, status_callback)
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "--shifts", "0",
        "--overlap", "0.10",
        "--segment", "7",
        "-j", "1",
        "-d", device,
        "-o", work_dir,
        source,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = []
    last_pct = -1
    if proc.stdout is not None:
        # tqdm redraws with carriage returns, so preserve those updates instead
        # of waiting for Demucs to exit and returning one large captured string.
        pending = ""
        while True:
            char = proc.stdout.read(1)
            if not char:
                break
            if char in "\r\n":
                line = pending.strip()
                pending = ""
                if line:
                    output.append(line)
                    output = output[-80:]
                    matches = re.findall(r"(?<!\d)(\d{1,3})%", line)
                    if matches:
                        pct = max(0, min(100, int(matches[-1])))
                        if pct >= last_pct + 2 or pct == 100:
                            last_pct = pct
                            _status(
                                status_callback,
                                f"voice isolation {pct}% - removing background music")
                continue
            pending += char
        if pending.strip():
            output.append(pending.strip())
    return_code = proc.wait()
    candidates = glob.glob(
        os.path.join(work_dir, "**", "vocals.wav"), recursive=True)
    if return_code != 0 or not candidates:
        detail = "\n".join(output).strip() or "no vocals stem produced"
        _status(
            status_callback,
            "voice isolation 5% - Demucs unavailable, using low-memory live mode")
        try:
            return _isolate_dialogue_low_memory(
                source, vocals_path, status_callback)
        except Exception as fallback_exc:
            raise YouTubeAudioError(
                "voice isolation failed; original music-mixed audio was not "
                f"played ({detail[-300:]}; fallback: {fallback_exc})"
            ) from fallback_exc
    candidates.sort(key=os.path.getmtime, reverse=True)
    shutil.copy2(candidates[0], vocals_path)
    if not os.path.exists(vocals_path) or os.path.getsize(vocals_path) <= 4096:
        raise YouTubeAudioError("voice isolation produced an empty vocal stem")
    _status(status_callback, "vocals ready - background music removed")
    return vocals_path


def _low_commit_headroom(minimum_bytes=4 * 1024 ** 3):
    """Return True when Windows cannot safely map a second PyTorch runtime."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)):
            return False
        return status.ullAvailPageFile < int(minimum_bytes)
    except Exception:
        return False


def _media_duration(path):
    ffprobe = os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe") or ""
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-of", "json", path,
                ],
                capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return float(json.loads(proc.stdout)["format"]["duration"])
        except Exception:
            pass

    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", path],
            capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            proc.stderr or "")
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        pass
    return 0.0


def _isolate_dialogue_low_memory(source, vocals_path, status_callback=None):
    """Extract the dialogue center with FFmpeg when Demucs cannot coexist live."""
    duration = _media_duration(source)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", source,
        "-af",
        (
            "dialoguenhance=original=0.15:enhance=3:voice=12,"
            "pan=mono|c0=c2,highpass=f=80,lowpass=f=12000,afftdn=nf=-25"
        ),
        "-ar", str(MONITOR_RATE),
        "-c:a", "pcm_s16le",
        "-progress", "pipe:1", "-nostats",
        vocals_path,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    last_pct = -1
    if proc.stdout is not None:
        for line in proc.stdout:
            key, _, raw = line.strip().partition("=")
            if key not in ("out_time_ms", "out_time_us") or duration <= 0:
                continue
            try:
                elapsed = float(raw) / 1_000_000.0
                pct = max(0, min(99, int(elapsed / duration * 100)))
            except (TypeError, ValueError):
                continue
            if pct >= last_pct + 3:
                last_pct = pct
                _status(
                    status_callback,
                    f"voice isolation {pct}% - low-memory dialogue mode")
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise YouTubeAudioError(
            (stderr or "FFmpeg dialogue isolation failed").strip()[-500:])
    if not os.path.exists(vocals_path) or os.path.getsize(vocals_path) <= 4096:
        raise YouTubeAudioError("low-memory isolation produced an empty track")
    _status(status_callback, "voice isolation 100% - dialogue track ready")
    _status(status_callback, "vocals ready - background music reduced")
    return vocals_path


def _prepare_demucs_checkpoint(status_callback=None, attempts=8):
    """Download the default Demucs model with resume support and verify it."""
    try:
        import torch
    except Exception as exc:
        raise YouTubeAudioError(f"PyTorch is unavailable for voice isolation ({exc})")

    checkpoint_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, DEMUCS_MODEL_FILENAME)
    if _has_hash_prefix(checkpoint_path, DEMUCS_MODEL_HASH_PREFIX):
        return checkpoint_path

    os.makedirs(checkpoint_dir, exist_ok=True)
    partial_path = checkpoint_path + ".part"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    _status(status_callback, "downloading voice-isolation model (resumable)")

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise YouTubeAudioError(
            "voice-isolation model download needs curl for resumable transfers")

    last_detail = ""
    for _ in range(max(1, int(attempts))):
        proc = subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--retry", "5",
                "--retry-delay", "1",
                "--retry-all-errors",
                "--continue-at", "-",
                "--output", partial_path,
                DEMUCS_MODEL_URL,
            ],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and _has_hash_prefix(
                partial_path, DEMUCS_MODEL_HASH_PREFIX):
            os.replace(partial_path, checkpoint_path)
            _status(status_callback, "voice-isolation model ready")
            return checkpoint_path
        last_detail = (proc.stderr or proc.stdout or "").strip()

    raise YouTubeAudioError(
        "voice-isolation model download was interrupted repeatedly"
        + (f" ({last_detail[-300:]})" if last_detail else "")
    )


def _has_hash_prefix(path, expected_prefix):
    if not os.path.isfile(path):
        return False
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest().startswith(expected_prefix)


def _status(callback, msg):
    if callback is not None:
        try:
            callback(msg)
        except Exception:
            pass


def _build_ffmpeg_cmd(audio_url, start_seconds=0.0, end_seconds=None,
                      voice_disguise=False, persona="deep_male"):
    start_seconds = float(start_seconds or 0.0)
    fast_seek = max(0.0, start_seconds - 120.0) if start_seconds > 120.0 else 0.0
    fine_seek = start_seconds - fast_seek
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if fast_seek > 0:
        cmd += ["-ss", str(fast_seek)]
    cmd += ["-i", audio_url]
    # Hybrid seek: input-side seek gets close quickly on remote YouTube streams,
    # then output-side seek trims the final part accurately before any PCM is
    # emitted to the mouth/speakers.
    if fine_seek > 0:
        cmd += ["-ss", str(fine_seek)]
    if end_seconds is not None and float(end_seconds) > start_seconds:
        cmd += ["-t", str(float(end_seconds) - start_seconds)]
    if voice_disguise:
        profiles = {
            "deep_male": {
                "f0": 0.72,
                "formant": 0.74,
                "filters": (
                    "highpass=f=45,lowpass=f=12000,"
                    "equalizer=f=105:t=q:w=0.8:g=5.0,"
                    "equalizer=f=650:t=q:w=1.0:g=2.5,"
                    "equalizer=f=2400:t=q:w=1.1:g=-4.0,"
                    "acompressor=threshold=0.14:ratio=2.0:attack=16:release=200:makeup=1.15"
                ),
            },
            "warm_male": {
                "f0": 0.88,
                "formant": 0.82,
                "filters": (
                    "highpass=f=50,lowpass=f=14500,"
                    "equalizer=f=145:t=q:w=0.9:g=4.0,"
                    "equalizer=f=900:t=q:w=1.0:g=2.0,"
                    "equalizer=f=3000:t=q:w=1.2:g=-2.0,"
                    "acompressor=threshold=0.16:ratio=1.7:attack=14:release=180:makeup=1.12"
                ),
            },
            "young_male": {
                "f0": 1.10,
                "formant": 1.03,
                "filters": (
                    "highpass=f=85,lowpass=f=17000,"
                    "equalizer=f=180:t=q:w=1.0:g=-3.0,"
                    "equalizer=f=1200:t=q:w=1.0:g=-1.5,"
                    "equalizer=f=3900:t=q:w=0.9:g=3.5,"
                    "acompressor=threshold=0.16:ratio=1.5:attack=10:release=150:makeup=1.08"
                ),
            },
            "broadcast_male": {
                "f0": 0.95,
                "formant": 0.91,
                "filters": (
                    "highpass=f=65,lowpass=f=13500,"
                    "equalizer=f=120:t=q:w=0.8:g=3.5,"
                    "equalizer=f=700:t=q:w=1.0:g=-2.0,"
                    "equalizer=f=2100:t=q:w=1.0:g=3.0,"
                    "deesser=i=0.16:m=0.35:f=0.55:s=o,"
                    "acompressor=threshold=0.11:ratio=3.0:attack=6:release=130:makeup=1.25"
                ),
            },
            "natural_woman": {
                "f0": 1.46,
                "formant": 1.25,
                "filters": (
                    "highpass=f=110,lowpass=f=17500,"
                    "equalizer=f=190:t=q:w=1.0:g=-4.0,"
                    "equalizer=f=1050:t=q:w=1.0:g=2.5,"
                    "equalizer=f=4300:t=q:w=0.9:g=4.0,"
                    "deesser=i=0.14:m=0.30:f=0.58:s=o,"
                    "acompressor=threshold=0.17:ratio=1.5:attack=10:release=155:makeup=1.08"
                ),
            },
            "warm_woman": {
                "f0": 1.32,
                "formant": 1.18,
                "filters": (
                    "highpass=f=90,lowpass=f=16500,"
                    "equalizer=f=220:t=q:w=0.9:g=2.5,"
                    "equalizer=f=950:t=q:w=1.0:g=3.0,"
                    "equalizer=f=3600:t=q:w=1.1:g=1.5,"
                    "deesser=i=0.15:m=0.32:f=0.56:s=o,"
                    "acompressor=threshold=0.16:ratio=1.5:attack=12:release=170:makeup=1.10"
                ),
            },
            "bright_woman": {
                "f0": 1.62,
                "formant": 1.34,
                "filters": (
                    "highpass=f=140,lowpass=f=18000,"
                    "equalizer=f=260:t=q:w=1.0:g=-5.0,"
                    "equalizer=f=1350:t=q:w=0.9:g=2.5,"
                    "equalizer=f=5200:t=q:w=0.9:g=5.0,"
                    "deesser=i=0.13:m=0.38:f=0.58:s=o,"
                    "acompressor=threshold=0.17:ratio=1.4:attack=9:release=145:makeup=1.06"
                ),
            },
            "low_woman": {
                "f0": 1.22,
                "formant": 1.13,
                "filters": (
                    "highpass=f=75,lowpass=f=15000,"
                    "equalizer=f=175:t=q:w=0.9:g=3.0,"
                    "equalizer=f=850:t=q:w=1.0:g=2.0,"
                    "equalizer=f=3300:t=q:w=1.1:g=2.0,"
                    "deesser=i=0.15:m=0.28:f=0.56:s=o,"
                    "acompressor=threshold=0.14:ratio=2.0:attack=11:release=175:makeup=1.15"
                ),
            },
        }
        profile = (persona or os.environ.get(
            "AVATAR_YOUTUBE_PERSONA", "deep_male")).strip().lower()
        settings = profiles.get(profile, profiles["deep_male"])
        formant = settings["formant"]
        pitch_only = settings["f0"] / formant
        tempo_restore = 1.0 / formant
        cmd += [
            "-af",
            (
                # Extract center dialogue before changing vocal identity. Most
                # source music lives in the stereo sides; speech is center-panned.
                "dialoguenhance=original=0.15:enhance=3.0:voice=12,"
                "pan=mono|c0=FC,"
                "afftdn=nr=14:nf=-34:tn=1,"
                f"rubberband=tempo=1.0:pitch={pitch_only:.6f}:"
                "transients=crisp:detector=compound:phase=laminar:"
                "window=standard:smoothing=off:formant=preserved:pitchq=quality,"
                f"asetrate={MONITOR_RATE}*{formant:.6f},"
                f"aresample={MONITOR_RATE},atempo={tempo_restore:.6f},"
                f"{settings['filters']}"
            ),
        ]
    cmd += [
        "-vn", "-ac", "1", "-ar", str(MONITOR_RATE),
        "-f", "s16le", "pipe:1",
    ]
    return cmd


def format_time(seconds):
    seconds = max(0, int(float(seconds or 0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def pcm16_bytes_to_float(raw):
    if not raw:
        return np.zeros(0, dtype=np.float32)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return np.clip(data / 32768.0, -1.0, 1.0)
