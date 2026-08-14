import os
import collections
import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from tts_stream_engine import FFMPEG
from voice_changer_engine import BLOCK, SAMPLE_RATE, make_converter
from youtube_cache import audio_dir, get_cached_audio, save_audio, video_id_from_url
from youtube_cache_janitor import enforce_budget, human_bytes
from youtube_dlp_options import extract_info_with_retries, one_line

MONITOR_RATE = 48000
MONITOR_BLOCK = BLOCK * (MONITOR_RATE // SAMPLE_RATE)

# Jitter buffer between ffmpeg and the speaker. See _PcmReader for the
# measurements. The pre-roll grows each time the buffer runs dry, because a
# broadcast that stalls once tends to stall again. PCM is 96 KB/s, so even the
# 30 s cap is under 3 MB.
#
# The deepest stalls come at the start, while ffmpeg is still filling its HLS
# window: a 2.5 s pre-roll was measured being wiped out by a 10.4 s stall five
# seconds into playback. Waiting a few seconds longer up front is far less
# noticeable than a hole in the voice, so the cushion starts where it can
# actually survive that.
#
# This is chosen for the voice alone, and deliberately so. It was briefly held
# equal to the picture's pre-roll, on the reasoning that each side settles
# behind the broadcast edge by its own cushion and the difference is lip sync
# error. That is true, but it is the wrong lever: the picture now measures both
# lags and holds itself back to match (see youtube_video._picture_is_ahead_by),
# so it corrects any difference on its own. Matching the pre-rolls bought no
# sync that was not already there and cost the voice its depth - at 4 s the
# buffer starved within a minute of a live broadcast, and because the picture
# follows this player, the starve became a six second freeze in the picture
# too. A stall here is now expensive for both halves, which is a reason to
# make it rarer, not a reason to run shallow.
AUDIO_PREROLL_SECONDS = 6.0
AUDIO_MAX_PREROLL_SECONDS = 12.0
AUDIO_BUFFER_SECONDS = 30.0
# A live broadcast is braked exactly like a recording, and the brake is load
# bearing. It was once removed on the theory that a braked ffmpeg stops fetching
# segments and drifts off the live edge - the voice was starving 27 to 29 s into
# every live playback against a 30 s buffer draining at 1x, which looked like
# the brake causing it. Unbraked, ffmpeg instead raced through the segments the
# playlist had, reached the end of them, and exited cleanly: the reader saw EOF
# and the voice ended outright 25 s in. Pacing ffmpeg to the speed the audio is
# consumed is what keeps it following a live playlist, so the brake stays. The
# starve is real but it is a hole, not a stop, and the cushion is what covers
# it - see AUDIO_PREROLL_SECONDS.
AUDIO_LIVE_BUFFER_SECONDS = AUDIO_BUFFER_SECONDS
# Bitrate ceiling, in kbps, for the live rendition the voice is pulled from.
# See _select_live_audio_url: the widest stream available is the one most
# likely to starve, and speech does not need it.
LIVE_AUDIO_ABR_CAP = 96.0
# How long the voice may receive nothing at all before its decoder is replaced.
# ffmpeg staying alive proves nothing - a connection can die without ever
# raising - and the playback loop had no timeout on its refill, so a stream
# that stopped delivering became permanent silence with no error and no log
# line while the picture carried on. Segments are a few seconds, so nothing
# this long is a stream that is still working.
AUDIO_STALL_RESTART_SECONDS = 12.0
# Consecutive reconnects that deliver no audio before the voice is declared
# failed. Reconnecting forever would hide a dead link as effectively as the
# original silent wait did.
AUDIO_STALL_RESTART_LIMIT = 6
DEMUCS_MODEL_FILENAME = "955717e8-8726e21a.th"
DEMUCS_MODEL_HASH_PREFIX = "8726e21a"
DEMUCS_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
    + DEMUCS_MODEL_FILENAME
)
_AUDIO_DOWNLOAD_LOCKS = {}
_AUDIO_DOWNLOAD_LOCKS_GUARD = threading.Lock()

try:
    import sounddevice as sd
except Exception as _sd_exc:  # pragma: no cover
    sd = None
    _SD_IMPORT_ERROR = _sd_exc


FFMPEG_ERROR_LINES = 40


class YouTubeAudioError(RuntimeError):
    pass


def _drain_stderr(proc):
    """Keep reading ffmpeg's stderr so it can never block on a full pipe.

    Same reasoning as the video scene: nothing read this pipe during normal
    playback, so on a long stream ffmpeg's warnings eventually filled the 64 KB
    buffer and it stopped emitting audio while still looking alive. The drained
    text is kept in a small rolling buffer so failures can be reported with
    detail.
    """
    stream = getattr(proc, "stderr", None)
    if stream is None:
        return
    proc._error_tail = collections.deque(maxlen=FFMPEG_ERROR_LINES)

    def _pump():
        try:
            for line in iter(stream.readline, b""):
                text = line.decode("utf-8", "replace").strip()
                if text:
                    proc._error_tail.append(text)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    threading.Thread(target=_pump, daemon=True).start()


class _PcmReader:
    """Reads PCM blocks off ffmpeg's stdout in its own thread.

    The playback loop used to read one block straight from the pipe and then
    block writing it to the speaker, so the network and the sound card were
    chained together: nothing was read while a block was playing, and nothing
    played while a block was being fetched. A live broadcast does not deliver
    on that rhythm. Measured over 60 s of one, ffmpeg supplied audio at 1.09x
    realtime overall - fast enough - but in bursts separated by 21 stalls, the
    longest 9.6 s. Every one of those was a hole in the sound.

    Averaging 1.09x is exactly the case a buffer fixes: there is enough audio,
    it just arrives early or late. This thread keeps reading regardless of what
    playback is doing, and parks whole blocks until the speaker wants them.

    The buffer is bounded, and a full buffer makes this thread wait rather than
    discard: dropping a block would be a skip in the audio, and back-pressure on
    a paused player is the correct behaviour anyway. PCM is cheap - 96 KB/s - so
    the cap can be generous.
    """

    def __init__(self, proc, block_bytes, max_blocks):
        self.proc = proc
        self.block_bytes = int(block_bytes)
        self.max_blocks = max(1, int(max_blocks))
        self.blocks = collections.deque()
        self.eof = False
        self.short_tail = b""
        # When audio last arrived. ffmpeg can stop delivering without exiting -
        # a dead connection that never raises - and the playback loop then
        # waits for a refill that never comes. Alive says nothing; this is the
        # only signal that the stream has actually stopped.
        self.last_progress_at = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        stdout = getattr(self.proc, "stdout", None)
        if stdout is None:
            self.eof = True
            return
        try:
            while not self._stop.is_set():
                while (not self._stop.is_set()
                       and self.depth() >= self.max_blocks):
                    # Full: let ffmpeg block on the pipe instead of losing audio.
                    # Progress is marked while braking - a throttled reader is
                    # healthy, and letting the clock run here would read as a
                    # dead stream to the watchdog.
                    self.last_progress_at = time.monotonic()
                    self._stop.wait(0.02)
                if self._stop.is_set():
                    break
                raw = stdout.read(self.block_bytes)
                if len(raw) < self.block_bytes:
                    self.short_tail = raw
                    break
                with self._lock:
                    self.blocks.append(raw)
                self.last_progress_at = time.monotonic()
        except Exception:
            pass
        finally:
            self.eof = True

    def pop(self):
        with self._lock:
            return self.blocks.popleft() if self.blocks else None

    def depth(self):
        with self._lock:
            return len(self.blocks)

    def drained(self):
        return self.eof and self.depth() == 0

    def close(self):
        self._stop.set()
        with self._lock:
            self.blocks.clear()


def _env_float(name, default, minimum=0.0):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _audio_preroll_seconds():
    return _env_float(
        "AVATAR_YOUTUBE_AUDIO_PREROLL", AUDIO_PREROLL_SECONDS, minimum=0.05)


def _audio_max_preroll_seconds():
    return max(
        _audio_preroll_seconds(),
        _env_float("AVATAR_YOUTUBE_AUDIO_MAX_PREROLL",
                   AUDIO_MAX_PREROLL_SECONDS, minimum=0.05))


def _audio_buffer_seconds(live=False):
    """Buffer cap, never shallower than the deepest pre-roll it must hold.

    A broadcast gets a cap it will not reach, so the reader never brakes the
    ffmpeg that has to keep fetching to hold the live edge.
    """
    if live:
        return max(
            _audio_max_preroll_seconds(),
            _env_float("AVATAR_YOUTUBE_AUDIO_LIVE_BUFFER",
                       AUDIO_LIVE_BUFFER_SECONDS, minimum=0.2))
    return max(
        _audio_max_preroll_seconds(),
        _env_float("AVATAR_YOUTUBE_AUDIO_BUFFER", AUDIO_BUFFER_SECONDS,
                   minimum=0.2))


def _stderr_tail(proc):
    """Last few lines of ffmpeg's stderr, for error messages."""
    try:
        lines = list(getattr(proc, "_error_tail", ()) or ())
        if not lines:
            return ""
        return " ".join(lines).replace("\r", " ")[-500:]
    except Exception:
        return ""


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
        self.duck_gain = float(os.environ.get("AVATAR_YOUTUBE_DUCK_GAIN", "0.22"))
        self._output_gain = 1.0
        self._target_output_gain = 1.0

        self._running = False
        self._paused = threading.Event()
        self._paused.set()
        self._thread = None
        self._proc = None
        self._out = None
        self._fade_pos = 0
        self._last_styled = None
        self._pause_release_pending = False
        self._output_gain = 1.0
        self._target_output_gain = 1.0
        self._playback_anchor_t = None
        self._last_output_warning_t = 0.0
        self._reader = None
        self._audio_stalls = 0
        self._last_stall_warning_t = 0.0
        self._edge_open_t = None
        self.is_live = False
        self._source_url = ""

    def start(self, url, start_seconds=None, end_seconds=None):
        if self._running:
            self.stop()
        # Kept so a reconnect can resolve a fresh stream URL. A broadcast's
        # googlevideo URL expires, so reopening ffmpeg on the same one just
        # reconnects to something already dead.
        self._source_url = url
        resolved = {}
        audio_url, title, duration, cache_hit = _resolve_audio_source(
            url, self._set_status, out_info=resolved)
        # A downloaded track is a local file; a broadcast stays a remote URL.
        # Not the same question as "is this a broadcast" any more, though: a
        # recording whose download was refused is streamed from a URL too, and
        # calling that live would throw away its FROM/TO range and its end.
        is_live = bool(resolved.get("is_live", not os.path.isfile(audio_url)))
        if (os.environ.get("AVATAR_YOUTUBE_VOCALS_ONLY", "0") == "1"
                and not is_live):
            audio_url = _isolate_vocals(audio_url, self._set_status)
        elif is_live:
            self._set_status(
                "live stream connected - voice isolation skipped for low latency")
        self.is_live = is_live
        self.title = title
        self.duration = float(duration or 0.0)
        self.start_seconds = float(start_seconds or 0.0)
        self.end_seconds = float(end_seconds) if end_seconds is not None else None
        if is_live and (self.start_seconds > 0 or self.end_seconds is not None):
            # A broadcast in progress has no timeline to seek in. Handing ffmpeg
            # "-ss 8400" against a rolling playlist starves it to a trickle, so
            # a FROM/TO left over from the previous video silences the voice.
            self._set_status(
                "live stream - ignoring the FROM/TO time range and joining now")
            self.start_seconds = 0.0
            self.end_seconds = None
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
        self._playback_anchor_t = None
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
            reader = getattr(self, "_reader", None)
            if reader is not None:
                reader.close()
        except Exception:
            pass
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

    def set_ducked(self, ducked, gain=None):
        """Lower monitor output under bot speech without pausing playback."""
        if gain is not None:
            self.duck_gain = max(0.0, min(1.0, float(gain)))
        self._target_output_gain = self.duck_gain if ducked else 1.0
        self._set_status("ducking youtube voice" if ducked else "restoring youtube voice")

    @property
    def position_seconds(self):
        return self.start_seconds + self.position_blocks * BLOCK / float(SAMPLE_RATE)

    def _reconnect_source(self, audio_url, block_bytes, max_blocks):
        """Replace a decoder that went quiet without exiting.

        A recording resumes where playback got to, so the reconnect is
        inaudible. A broadcast has no timeline to resume into and rejoins at
        the edge, which is the same thing it did when it first connected.
        """
        try:
            if self._proc is not None:
                self._proc.kill()
        except Exception:
            pass
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        if self.is_live and self._source_url:
            # Resolve a fresh stream URL rather than reopening the old one. A
            # broadcast's googlevideo URL expires, and reconnecting to an
            # expired URL just reproduces the silence that triggered this -
            # measured reconnecting four times in a row against a dead URL
            # without ever recovering the voice.
            try:
                fresh, _title, _duration, _hit = _resolve_audio_source(
                    self._source_url, self._set_status)
                if fresh:
                    audio_url = fresh
            except Exception as exc:
                self._set_status(f"could not refresh voice link ({one_line(exc)})")
        resume_at = 0.0 if self.is_live else self.position_seconds
        cmd = _build_ffmpeg_cmd(
            audio_url, resume_at, self.end_seconds,
            voice_disguise=self.converter_kind in (
                "youtube-disguise", "youtube_disguise"),
            persona=self.persona)
        self._edge_open_t = time.monotonic()
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _drain_stderr(self._proc)
        self._reader = _PcmReader(self._proc, block_bytes, max_blocks)

    def _expected_end_seconds(self):
        """Where playback should reach, or None for a broadcast (no end)."""
        if self.is_live:
            return None
        if self.end_seconds is not None:
            return float(self.end_seconds)
        if self.duration and self.duration > 0:
            return float(self.duration)
        return None

    @property
    def live_lag_seconds(self):
        """How far behind the broadcast edge this voice is playing, in seconds.

        Wall time since ffmpeg opened at the edge, minus the audio actually
        played in that time. A picture decoded from the same broadcast can
        measure its own lag the same way and hold itself back to match, which
        is what puts the lips on the sound: both sides are then showing the
        same instant of the broadcast, whatever each of them spent getting
        there. None until playback has produced something to measure.
        """
        opened = self._edge_open_t
        if opened is None or self.position_blocks <= 0:
            return None
        played = self.position_seconds - self.start_seconds
        return max(0.0, (time.monotonic() - opened) - played)

    def _run(self, audio_url):
        try:
            cmd = _build_ffmpeg_cmd(
                audio_url, self.start_seconds, self.end_seconds,
                voice_disguise=self.converter_kind in (
                    "youtube-disguise", "youtube_disguise"),
                persona=self.persona)
            self._set_status("starting ffmpeg")
            # A live playlist has no timeline to seek, so ffmpeg starts from
            # wherever the broadcast edge is at this instant. That makes this
            # moment the reference for live_lag_seconds: everything spent
            # afterwards not playing - startup, pre-roll, stalls - is time the
            # broadcast moved on without this player, and is exactly how far
            # behind the edge the sound has settled.
            self._edge_open_t = time.monotonic()
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Nothing reads this pipe until playback ends. On a long live
            # stream ffmpeg's reconnect warnings eventually fill the 64 KB
            # buffer, and it then blocks writing to stderr and stops emitting
            # audio - the voice simply stops with the process still alive.
            _drain_stderr(self._proc)
            if self.monitor and sd is not None:
                try:
                    from audio_output import ManagedOutputStream, output_health
                    if self.out_device is None:
                        # Managed: survives the playback device being swapped or
                        # unplugged mid-video instead of going silently deaf.
                        self._out = ManagedOutputStream(
                            samplerate=MONITOR_RATE, channels=1,
                            dtype="float32", blocksize=MONITOR_BLOCK,
                            status_callback=self._set_status)
                        # Open now: an unusable speaker must fail here with a
                        # clear message, the way it did before, instead of
                        # turning into a silent per-block retry during playback.
                        self._out.open()
                        self._set_status(output_health(self._set_status))
                    else:
                        self._out = sd.OutputStream(
                            samplerate=MONITOR_RATE, channels=1,
                            dtype="float32", blocksize=MONITOR_BLOCK,
                            device=self.out_device)
                        self._out.start()
                except Exception as exc:
                    raise YouTubeAudioError(
                        f"audio output unavailable ({exc})") from exc
            elif self.monitor:
                raise YouTubeAudioError(
                    f"sounddevice unavailable ({_SD_IMPORT_ERROR})")

            nbytes = MONITOR_BLOCK * 2
            block_seconds = MONITOR_BLOCK / float(MONITOR_RATE)
            preroll = _audio_preroll_seconds()
            max_preroll = _audio_max_preroll_seconds()
            max_blocks = max(2, int(round(
                _audio_buffer_seconds(live=bool(self.is_live))
                / block_seconds)))
            self._reader = _PcmReader(self._proc, nbytes, max_blocks)
            primed = False
            reconnects = 0
            self._playback_anchor_t = time.monotonic()
            while self._running:
                if self._paused.is_set():
                    if self._pause_release_pending:
                        self._pause_release_pending = False
                        self._emit_release_tail()
                    self.speaking = False
                    self._reset_realtime_anchor()
                    time.sleep(0.05)
                    continue
                if not primed:
                    want = max(1, int(round(preroll / block_seconds)))
                    if self._reader.depth() < want and not self._reader.eof:
                        quiet = (time.monotonic()
                                 - self._reader.last_progress_at)
                        if quiet >= AUDIO_STALL_RESTART_SECONDS:
                            # ffmpeg is alive and delivering nothing, which it
                            # can do indefinitely on a connection that died
                            # without erroring. Waiting here was an unbounded
                            # silence: the voice simply stopped, with no error,
                            # no log line and no recovery, while the picture
                            # carried on. Replace the decoder instead.
                            reconnects += 1
                            if reconnects > AUDIO_STALL_RESTART_LIMIT:
                                raise YouTubeAudioError(
                                    "voice stream stopped delivering audio "
                                    f"and did not recover after {reconnects} "
                                    "reconnects")
                            self._set_status(
                                f"voice stream silent for {quiet:.0f}s; "
                                f"reconnecting ({reconnects})")
                            self._reconnect_source(audio_url, nbytes, max_blocks)
                        else:
                            time.sleep(0.02)
                        continue
                    primed = True
                    reconnects = 0
                    self._reset_realtime_anchor()
                raw = self._reader.pop()
                if raw is None:
                    if not self._reader.eof:
                        # The broadcast went quiet mid-stream. Hold and refill
                        # rather than feed the speaker a hole, and earn a deeper
                        # cushion so the next stall is covered too.
                        primed = False
                        preroll = min(max_preroll, preroll + 1.0)
                        self._note_audio_stall(preroll)
                        continue
                    return_code = self._proc.wait()
                    error_text = _stderr_tail(self._proc)
                    if return_code != 0:
                        raise YouTubeAudioError(
                            error_text or f"ffmpeg exited with code {return_code}")
                    # ffmpeg exiting 0 is only good news if it reached the end
                    # it was asked for. Short of that the voice has been lost,
                    # and "ended" on its own is indistinguishable from a track
                    # finishing normally - which is how a voice that stopped 8 s
                    # into a 9 minute track read as a clean finish in the log.
                    played = self.position_seconds
                    expected = self._expected_end_seconds()
                    if expected is not None and played < expected - 1.0:
                        self._set_status(
                            f"voice ended early at {played:.0f}s of "
                            f"{expected:.0f}s (ffmpeg exited {return_code})"
                            + (f": {error_text}" if error_text else ""))
                    elif expected is None:
                        # A broadcast has no end to reach at all, so any exit
                        # is early: it ran out of playlist and stopped.
                        self._set_status(
                            "live voice ended early after "
                            f"{played - self.start_seconds:.0f}s "
                            f"(ffmpeg exited {return_code})"
                            + (f": {error_text}" if error_text else ""))
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
                wrote_monitor = False
                if self._out is not None and not self.muted:
                    try:
                        monitor = self._apply_output_duck(styled)
                        self._out.write(monitor.reshape(-1, 1))
                        wrote_monitor = True
                    except Exception as exc:
                        # This used to swallow everything, so a speaker that had
                        # gone away looked identical to normal playback: no
                        # sound, nothing in the log, no way to tell why.
                        self._note_output_failure(exc)
                self._pace_realtime_if_needed(wrote_monitor)
            self._set_status("ended")
        except Exception as exc:
            self.last_error = str(exc)
            self._set_status(f"failed: {exc}")
        finally:
            self.stop()

    def _note_audio_stall(self, preroll):
        """Report a starved buffer at most once every few seconds."""
        self._audio_stalls = getattr(self, "_audio_stalls", 0) + 1
        now = time.monotonic()
        if now - getattr(self, "_last_stall_warning_t", 0.0) < 5.0:
            return
        self._last_stall_warning_t = now
        self._set_status(
            f"youtube audio buffering ({self._audio_stalls} so far); "
            f"cushion now {preroll:.1f}s")

    def _note_output_failure(self, exc):
        """Report a dead speaker once, not once per 21 ms audio block."""
        now = time.monotonic()
        if now - getattr(self, "_last_output_warning_t", 0.0) < 5.0:
            return
        self._last_output_warning_t = now
        self.last_error = str(exc)
        self._set_status(f"NO SOUND - audio output failing: {exc}")

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

    def _apply_output_duck(self, samples):
        out = np.asarray(samples, dtype=np.float32).copy()
        if out.size == 0:
            return out
        target = float(getattr(self, "_target_output_gain", 1.0))
        current = float(getattr(self, "_output_gain", 1.0))
        if abs(target - current) < 0.002:
            self._output_gain = target
            return (out * target).astype(np.float32)
        # Ramp over about 180 ms so the listener hears a blend, not a cut.
        step = max(0.02, min(0.22, out.size / float(MONITOR_RATE) / 0.18))
        next_gain = current + (target - current) * step
        env = np.linspace(current, next_gain, out.size, dtype=np.float32)
        self._output_gain = float(next_gain)
        return (out * env).astype(np.float32)

    def _reset_realtime_anchor(self):
        block_seconds = MONITOR_BLOCK / float(MONITOR_RATE)
        self._playback_anchor_t = (
            time.monotonic() - self.position_blocks * block_seconds)

    def _pace_realtime_if_needed(self, wrote_monitor):
        """Keep ffmpeg reads realtime when no output device write is blocking."""
        if wrote_monitor:
            return
        block_seconds = MONITOR_BLOCK / float(MONITOR_RATE)
        if self._playback_anchor_t is None:
            self._reset_realtime_anchor()
        deadline = self._playback_anchor_t + self.position_blocks * block_seconds
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, block_seconds))

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


def _resolve_audio_source(url, status_callback=None, out_info=None):
    cached = get_cached_audio(url)
    if cached is not None:
        _status(status_callback, "found audio in local db/cache")
        return cached["audio_path"], cached["title"], cached["duration"], True

    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeAudioError(f"yt-dlp is not installed ({exc})")
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
    }
    _status(status_callback, "checking youtube link")
    try:
        info = extract_info_with_retries(
            yt_dlp, url, probe_opts, download=False,
            status_callback=status_callback, status_prefix="real audio")
    except Exception as exc:
        raise YouTubeAudioError(f"could not read YouTube audio ({exc})")

    probe_info = info
    title = (info or {}).get("title") or "YouTube audio"
    duration = float((info or {}).get("duration") or 0.0)
    if out_info is not None:
        out_info["is_live"] = _is_live_info(info)
        out_info["streamed"] = False
    if _is_live_info(info):
        audio_url = _select_live_audio_url(info)
        if not audio_url:
            raise YouTubeAudioError(
                "the live video is online but no playable audio stream was found")
        _status(status_callback, "live stream found - connecting without download")
        if out_info is not None:
            out_info["streamed"] = True
        return audio_url, title, duration, False

    _status(status_callback, "new link - downloading youtube audio to cache")
    out_dir = audio_dir(url)
    lock = _audio_download_lock(out_dir)
    with lock:
        cached = get_cached_audio(url)
        if cached is not None:
            _status(status_callback, "found audio in local db/cache")
            return cached["audio_path"], cached["title"], cached["duration"], True
        _cleanup_partial_audio_files(out_dir)
        download_id = f"audio-{os.getpid()}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            # Plain "bestaudio" sometimes lands on a 390 kbps AAC track, which
            # is several times the bytes of the transparent ~120 kbps Opus one
            # for sound that ends up mono 48 kHz either way. Naming Opus and
            # capping the bitrate keeps the quality that is actually audible and
            # drops only the overshoot. AVATAR_YOUTUBE_AUDIO_ABR moves the cap:
            # lower saves more disk, higher goes back to the biggest track.
            "format": (
                f"bestaudio[acodec=opus][abr<=?{_audio_abr_cap()}]/"
                f"bestaudio[abr<=?{_audio_abr_cap()}]/"
                "bestaudio/best"
            ),
            "continuedl": False,
            "outtmpl": os.path.join(out_dir, f"{download_id}.%(ext)s"),
        }
        try:
            from youtube_video import _download_tuning_opts
            opts.update(_download_tuning_opts())
        except Exception:
            pass
        try:
            info = extract_info_with_retries(
                yt_dlp, url, opts, download=True,
                status_callback=status_callback, status_prefix="real audio")
        except Exception as exc:
            # Downloading is refused far more often than playing is: YouTube
            # answers the download with 403 while the very same media URLs
            # stream fine, which is visible whenever the picture keeps playing
            # and only the voice dies. A voice that streams is worth more than
            # a cached one, so fall back to the stream the live path already
            # uses rather than losing the voice altogether.
            stream_url = _select_live_audio_url(probe_info)
            if not stream_url:
                raise YouTubeAudioError(f"could not read YouTube audio ({exc})")
            _status(
                status_callback,
                "download refused ("
                f"{one_line(exc)[:90]}); streaming the voice instead")
            if out_info is not None:
                out_info["is_live"] = False
                out_info["streamed"] = True
            return stream_url, title, duration, False
    title = (info or {}).get("title") or "YouTube audio"
    duration = float((info or {}).get("duration") or 0.0)
    audio_path = _find_downloaded_audio(out_dir, prefix=download_id)
    if not audio_path:
        audio_path = (info or {}).get("requested_downloads", [{}])[0].get("filepath")
    if not audio_path or not os.path.exists(audio_path):
        raise YouTubeAudioError("audio download finished but no cached file was found")
    save_audio(url, title, duration, audio_path)
    _status(status_callback, "saved audio in local db/cache")
    _trim_cache_after_download(url, status_callback)
    return audio_path, title, duration, False


def _live_audio_abr_cap():
    """Bitrate ceiling for a live rendition, in kbps.

    Low enough to keep the stream fetchable on a slow link, high enough that
    speech is unaffected - a voice at 96 kbps is indistinguishable from the
    same voice at 160 once it has been pitch shifted and rebroadcast.
    """
    return _env_float(
        "AVATAR_YOUTUBE_LIVE_AUDIO_ABR", LIVE_AUDIO_ABR_CAP, minimum=8.0)


def _audio_abr_cap():
    raw = os.environ.get("AVATAR_YOUTUBE_AUDIO_ABR", "").strip()
    try:
        cap = int(float(raw)) if raw else 160
    except ValueError:
        cap = 160
    return max(48, cap)


def _trim_cache_after_download(url, status_callback=None):
    """Evict the oldest cached videos so this download stays inside the budget."""
    try:
        removed, freed, _names = enforce_budget(keep_ids=[video_id_from_url(url)])
    except Exception as exc:
        _status(status_callback, f"cache cleanup skipped ({exc})")
        return
    if removed:
        _status(
            status_callback,
            f"cache trimmed: {removed} old video(s) removed, "
            f"{human_bytes(freed)} freed")


def _is_live_info(info):
    info = info or {}
    return bool(
        info.get("is_live")
        or info.get("live_status") == "is_live"
    )


def _select_live_audio_url(info):
    info = info or {}
    requested = info.get("requested_formats") or []
    candidates = list(requested) + list(info.get("formats") or [])
    candidates.append(info)
    audio = [
        fmt for fmt in candidates
        if fmt.get("url") and fmt.get("acodec") not in (None, "none")
    ]
    if not audio:
        return ""
    cap = _live_audio_abr_cap()

    def rank(fmt):
        # Audio-only first - pulling a video rendition to get its sound wastes
        # the bandwidth this is trying to protect.
        #
        # Then the best stream that fits under the cap, rather than the best
        # stream outright. Taking the fattest rendition available maximises the
        # bytes that have to arrive on time, which on a slow link is precisely
        # what starves the buffer and puts holes in the voice. The voice is
        # pitch shifted and rebroadcast, so the top rendition buys nothing
        # audible and costs the reliability that matters. If nothing fits,
        # the smallest overshoot wins for the same reason.
        abr = float(fmt.get("abr") or fmt.get("tbr") or 0.0)
        within = abr <= cap
        return (fmt.get("vcodec") == "none", within, abr if within else -abr)

    audio.sort(key=rank, reverse=True)
    return audio[0]["url"]


def _audio_download_lock(out_dir):
    key = os.path.abspath(out_dir)
    with _AUDIO_DOWNLOAD_LOCKS_GUARD:
        lock = _AUDIO_DOWNLOAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _AUDIO_DOWNLOAD_LOCKS[key] = lock
        return lock


def _cleanup_partial_audio_files(out_dir):
    for path in glob.glob(os.path.join(out_dir, "*.part")):
        try:
            os.remove(path)
        except OSError:
            # A currently running downloader may still own this file. New
            # attempts use a unique name, so a locked stale part will not block.
            pass


def _find_downloaded_audio(out_dir, prefix="audio"):
    files = [
        p for p in glob.glob(os.path.join(out_dir, f"{prefix}.*"))
        if os.path.isfile(p) and not p.endswith(".part")
    ]
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _cached_vocals_path(cache_dir):
    """Return an already-isolated stem, preferring the compact FLAC copy."""
    for name in ("vocals.flac", "vocals.wav"):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 4096:
            return path
    return ""


def _compress_vocals(wav_path, status_callback=None):
    """Store the stem as FLAC. Lossless audio, roughly a third of the bytes.

    The player still feeds ffmpeg a stereo file, so dialoguenhance and the
    centre-channel pan behave exactly as they did with the WAV.
    """
    if os.environ.get("AVATAR_YOUTUBE_VOCALS_FLAC", "1") == "0":
        return wav_path
    flac_path = os.path.splitext(wav_path)[0] + ".flac"
    try:
        proc = subprocess.run(
            [
                FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-i", wav_path,
                "-c:a", "flac", "-compression_level", "5",
                "-sample_fmt", "s16",
                flac_path,
            ],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return wav_path
    if proc.returncode != 0 or not os.path.exists(flac_path) \
            or os.path.getsize(flac_path) <= 4096:
        try:
            os.remove(flac_path)
        except OSError:
            pass
        return wav_path
    try:
        saved = os.path.getsize(wav_path) - os.path.getsize(flac_path)
        os.remove(wav_path)
        _status(
            status_callback,
            f"vocals stored losslessly ({saved / float(1024 ** 2):.0f} MB saved)")
    except OSError:
        pass
    return flac_path


def _isolate_vocals(audio_path, status_callback=None):
    """Return a cached Demucs vocal stem; never fall back to music-mixed audio."""
    source = os.path.abspath(audio_path)
    cache_dir = os.path.dirname(source)
    vocals_path = os.path.join(cache_dir, "vocals.wav")
    cached_vocals = _cached_vocals_path(cache_dir)
    if cached_vocals:
        _status(status_callback, "found isolated vocals in local cache")
        return cached_vocals

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
    # The raw Demucs stems are only an intermediate; the copy above is the stem
    # that gets played, so the multi-gigabyte work directory can go now.
    _remove_tree(work_dir)
    vocals_path = _compress_vocals(vocals_path, status_callback)
    _status(status_callback, "vocals ready - background music removed")
    return vocals_path


def _remove_tree(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


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
    vocals_path = _compress_vocals(vocals_path, status_callback)
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
    if str(audio_url).lower().startswith(("http://", "https://")):
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]
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
