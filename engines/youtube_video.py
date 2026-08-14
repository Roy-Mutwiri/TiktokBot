import collections
import glob
import io
import os
import subprocess
import threading
import time
import urllib.request

import numpy as np

from tts_stream_engine import FFMPEG
from youtube_cache import audio_dir, video_id_from_url
from youtube_cache_janitor import enforce_budget, human_bytes
from youtube_dlp_options import extract_info_with_retries, one_line


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 15.0
MAX_SOURCE_HEIGHT = 720
MAX_FORWARD_DRIFT = 30.0
DECODER_READ_TIMEOUT_US = 5_000_000
LIVE_READ_TIMEOUT_US = 15_000_000
DECODER_RESTART_BACKOFF = 0.35
DECODER_ERROR_LINES = 40
DECODER_FIRST_FRAME_TIMEOUT = 8.0
# A network source has to fetch and parse the index before it can decode
# anything, and a seek makes it range-request from a new offset. Measured on a
# 5 h recording: 8.5 s to the first frame at the start and 11.3 s after seeking
# to 05:00 - both over the local-file budget, so every seek was declared a dead
# decoder and retried, which is why a seek showed one frame and then stopped.
REMOTE_FIRST_FRAME_TIMEOUT = 25.0
LIVE_FIRST_FRAME_TIMEOUT = 25.0
LIVE_SOURCE_REFRESH_SECONDS = 45.0
# How long a live decoder may deliver nothing before it counts as dead. ffmpeg
# goes quiet for a few seconds each time it reconnects a dropped TLS connection.
LIVE_PARTIAL_STALL_LIMIT = 25.0

# Jitter buffer. ffmpeg writes 2.7 MB per frame into a 64 KB pipe, so when the
# scene loop read that pipe only at the moment it wanted to show a frame,
# ffmpeg spent nearly all of its time blocked on the write and could never
# fetch ahead. Every network hiccup on a live playlist then became a frozen
# picture, and the stream had no headroom left to recover with. Measured over
# 45 s of a live news broadcast, that path ran at 12.8 fps with eleven freezes
# of up to 5.1 s; draining the pipe into this buffer instead gave a flat
# 15.0 fps with no gap longer than 94 ms on the same stream.
#
# A frame costs 2.7 MB, so these seconds are also the memory bill: the 12 s cap
# is about 500 MB, and it is only reached while ffmpeg is running ahead of real
# time to catch up.
#
# The pre-roll starts short so the picture appears quickly, and grows by half a
# second each time the buffer runs dry. A live playlist was measured going
# quiet for up to 3.1 s at a time while it fetched a segment, and no fixed
# pre-roll both starts fast and covers that; letting a stream that keeps
# stalling earn a deeper cushion does.
#
# The pre-roll is also how far behind the broadcast edge this side settles: a
# decoder opened at the edge shows its first frame one pre-roll later and runs
# at 1x from there. The voice is a second decoder doing the same thing on its
# own cushion, and the difference between the two lags is the lip sync error.
# It is tempting to fix that by holding the two cushions equal, and that was
# tried; it starves whichever side needs more depth. The two are independent
# instead, and _picture_is_ahead_by measures both lags and holds the picture
# back to match the voice - which corrects the difference whatever each side
# chose, including the depth a stalling voice earns for itself mid-broadcast.
LIVE_PREROLL_SECONDS = 4.0
LIVE_MAX_PREROLL_SECONDS = 6.0
# Deep enough to hold both: this side's own cushion at its deepest, plus the
# longest the frame meter will park the picture waiting for a stalled voice.
# Anything shallower and ffmpeg jams on the pipe mid-hold and drops off the
# broadcast edge, which is a resync - the failure the buffer exists to prevent.
LIVE_BUFFER_SECONDS = 12.0
# How long the picture may be metered against a voice position that has stopped
# advancing before it gives up and paces itself. A stalled voice should freeze
# the picture with it - that is the whole point - but a scene whose audio has
# ended, or never started, must not freeze the broadcast forever.
AUDIO_CLOCK_STALL_LIMIT = 6.0
# How long a freshly opened live decoder waits for the voice to start playing
# before showing frame 0 unmetered. Covers the voice's own ffmpeg startup plus
# its pre-roll, and has to cover the slow case rather than the typical one: at
# 20 s this expired three times in a row on a link where the voice needed 22 to
# 23 s, losing the frame 0 alignment by two or three seconds each time - the
# one thing this wait exists to buy. The picture only ever waits when a voice
# actually exists, so a budget this generous costs nothing when there is none.
AUDIO_CLOCK_START_TIMEOUT = 40.0
# A voice position cannot advance faster than realtime, so a step larger than
# this is a different clock being read, not sound that played. Generous next to
# the ~0.02 s a real step covers, and well under the multi-second jumps that
# swapping voices produces.
CLOCK_JUMP_LIMIT = 1.0
# How the picture corrects itself against the voice. Frames used to be gated
# directly on the voice position - shown only once the voice had reached them -
# which is exact but brutal: a stalled voice stopped the picture dead, and on a
# slow connection that was a six second freeze every time the voice starved.
#
# Nudging the frame rate instead absorbs the same error invisibly. The picture
# runs a few percent slow or fast until it draws level, which nobody sees,
# where a freeze is the most visible artefact a video can have. The gain sets
# full correction authority at one second of error; the clamps keep the change
# under human notice - beyond about 15% a rate change starts to read as wrong.
SYNC_RATE_GAIN = 0.15
SYNC_MIN_RATE = 0.85
SYNC_MAX_RATE = 1.15
# A local file never stalls on the network, but it does stall on the CPU. This
# studio runs face swap and restoration across most of the machine, and a
# descheduled ffmpeg was measured going quiet for 3 s mid-file with the decoder
# alive, nothing dropped, and the buffer empty - which reads as a dead decoder
# and costs a resync and a visible jump.
#
# One second of slack was sized for a decoder that ran free and kept the buffer
# full by racing ahead of realtime, discarding what would not fit. Now that a
# recording is correctly braked to the rate frames are consumed, this buffer is
# the only thing between a starved ffmpeg and a hole in the picture, so it has
# to cover the pauses the CPU actually imposes. A frame is 2.7 MB, so this is
# about 165 MB.
LOCAL_BUFFER_SECONDS = 4.0
# How long the reader thread waits on one read before checking whether the
# decoder is still alive. It blocks in its own thread, so this costs nothing.
READER_READ_TIMEOUT = 2.0

# How often a live scene writes a smoothness line to the status log. Stutter
# used to be invisible in the log - it only recorded state changes, so a scene
# that was bursting and freezing its way through a broadcast looked exactly
# like a healthy one.
HEALTH_REPORT_SECONDS = 30.0
END_OF_VIDEO_MARGIN = 0.75
LOCAL_CACHE_FIRST_FRAME_RETRIES = 2
LOCAL_CACHE_DECODE_RETRIES = 3
# Consecutive failures on a remote source before re-asking yt-dlp for a fresh
# URL. googlevideo links are short-lived and start answering 403.
REMOTE_SOURCE_REFRESH_RETRIES = 2

# Resolution first, then the smallest rendition of that resolution. The scene
# renders 1280x720 at 15 fps, so a taller or higher-bitrate source is bytes on
# disk that never reach the screen.
DOWNLOAD_FORMAT_SORT = ("res:720", "+size", "+br")

# Caching a multi-hour video costs a gigabyte or more for a file the scene
# usually plays a few minutes of, and one such download evicts most of the rest
# of the cache. Ended livestreams are the common case: they report
# live_status="was_live" with is_live False, so the live check above lets them
# through. Anything past this length streams instead.
MAX_CACHE_DURATION_SECONDS = 3 * 3600

# yt-dlp asks ONE YouTube "player client" for the format list, and a client
# YouTube has just tightened answers with an empty list instead of an error a
# viewer would ever see. Measured against a live broadcast on 2026-08-13:
# "android" returned 6 video formats while "tv", "ios" and "web_safari" each
# raised "No video formats found!" - and an hour earlier "android" itself had
# failed the same way. Which client is healthy moves week to week, so pinning
# any single one turns a routine YouTube change into a dead scene.
#
# A LIVE link pays for that twice: the studio reads is_live off the resolved
# video, so a format hiccup does not just lose the picture, it makes a
# broadcast in progress look like a recording and sends it to the caption
# reader, which has nothing to read.
#
# android stays first because it measured fastest here, so the common case
# costs nothing; the rest are only touched after it comes back empty. None
# means "let yt-dlp pick", which is the documented advice for the download
# path (see AVATAR_README).
PLAYER_CLIENT_LADDER = (["android"], None, ["web"], ["tv"], ["ios"])

# yt-dlp reports a client that returned nothing as a hard error. These are not
# "this video is gone", they are "ask someone else".
FORMAT_MISSING_MARKERS = (
    "no video formats found",
    "no formats found",
    "requested format is not available",
)

# Download tuning. The old 1 MB chunk size turned a 68 MB preview into 68
# sequential ranged GETs, each paying a fresh round-trip and TCP ramp-up:
# measured 1.5 MB/s against 2.8-3.2 MB/s with big chunks and parallel
# fragments on the same link and connection.
DOWNLOAD_CHUNK_BYTES = 10 * 1024 * 1024
DOWNLOAD_FRAGMENT_THREADS = 8

_VIDEO_DOWNLOAD_LOCKS = {}
_VIDEO_DOWNLOAD_LOCKS_GUARD = threading.Lock()
_BACKGROUND_PREVIEW_JOBS = set()
_BACKGROUND_PREVIEW_GUARD = threading.Lock()


def _env_int(name, default, minimum=0):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(float(raw)))
    except ValueError:
        return default


def _env_float(name, default, minimum=0.0):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _frame_bytes():
    return VIDEO_WIDTH * VIDEO_HEIGHT * 3


def _live_preroll_seconds():
    return _env_float(
        "AVATAR_YOUTUBE_LIVE_PREROLL", LIVE_PREROLL_SECONDS, minimum=0.1)


def _live_max_preroll_seconds():
    return max(
        _live_preroll_seconds(),
        _env_float("AVATAR_YOUTUBE_LIVE_MAX_PREROLL",
                   LIVE_MAX_PREROLL_SECONDS, minimum=0.1))


def _video_buffer_seconds():
    """Jitter buffer depth for a network source, never below the pre-roll."""
    return max(
        _live_max_preroll_seconds(),
        _env_float("AVATAR_YOUTUBE_VIDEO_BUFFER", LIVE_BUFFER_SECONDS,
                   minimum=0.2))


class _DecoderStream:
    """An ffmpeg decoder plus the thread that keeps its stdout pipe drained.

    The scene loop shows one frame every 1/VIDEO_FPS s, but ffmpeg does not
    produce frames on that rhythm: it arrives in bursts between HLS segment
    fetches. Reading the pipe on the display rhythm coupled the two, so a slow
    segment froze the picture and a full pipe stopped ffmpeg from fetching the
    next one. This thread reads as fast as ffmpeg writes and parks whole frames
    in a bounded buffer; the scene loop then paces itself out of the buffer and
    never touches the pipe.
    """

    def __init__(self, proc, buffer_frames, block_when_full=False):
        self.proc = proc
        self.max_frames = max(1, int(buffer_frames))
        self.frames = collections.deque()
        # What to do when the buffer is full. A live decoder drops its oldest
        # frame: it must stay at the broadcast edge, and braking ffmpeg would
        # cost it the segments it has to fetch ahead.
        #
        # A recording is the opposite case and must brake. ffmpeg decodes a
        # local file about ten times faster than realtime, so it overruns the
        # buffer continuously and the dropping deque throws away almost every
        # frame it decodes - leaving the newest handful. The scene loop then
        # showed 15 of those a second, each far ahead of the last, and the
        # picture raced through the video: smooth, correct frame rate, and
        # several times too fast. Nothing measured it either, because
        # stream_position counts frames shown rather than reading the frame's
        # real timestamp, so playback reported a confident 1.00x throughout.
        # Blocking makes ffmpeg decode at the rate frames are consumed, which
        # is what "play this file at 1x" means.
        self.block_when_full = bool(block_when_full)
        self.produced = 0
        self.dropped = 0
        self.last_progress_at = time.monotonic()
        self.eof = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        frame_bytes = _frame_bytes()
        stdout = getattr(self.proc, "stdout", None)
        if stdout is None:
            self.eof = True
            return
        pending = b""
        try:
            while not self._stop.is_set():
                chunk = _read_exact(
                    stdout, frame_bytes - len(pending),
                    timeout=READER_READ_TIMEOUT)
                if chunk:
                    pending += chunk
                    self.last_progress_at = time.monotonic()
                if len(pending) < frame_bytes:
                    # A half-read frame is normal: ffmpeg pauses for a few
                    # seconds whenever it reconnects a dropped connection. Keep
                    # the partial frame and wait; only an exited decoder is EOF.
                    if self.proc.poll() is not None:
                        break
                    if not chunk:
                        # Closed pipe, or a mocked stream with nothing left.
                        # Back off so this can never become a hot loop.
                        self._stop.wait(0.02)
                    continue
                frame = pending[:frame_bytes]
                pending = pending[frame_bytes:]
                while (self.block_when_full and not self._stop.is_set()
                       and self.depth() >= self.max_frames):
                    # Full and braking: hold the frame and let ffmpeg block on
                    # the pipe behind us. Progress is marked while waiting -
                    # the decoder is healthy and deliberately throttled, and
                    # leaving the clock running here would read as a stalled
                    # decoder and trigger a resync every time the buffer filled.
                    self.last_progress_at = time.monotonic()
                    self._stop.wait(0.005)
                if self._stop.is_set():
                    break
                with self._lock:
                    if len(self.frames) >= self.max_frames:
                        # Running ahead of realtime with nowhere to put this.
                        # Drop the oldest rather than grow without bound.
                        self.frames.popleft()
                        self.dropped += 1
                    self.frames.append(frame)
                    self.produced += 1
        except Exception:
            pass
        finally:
            self.eof = True

    def pop(self):
        with self._lock:
            return self.frames.popleft() if self.frames else None

    def depth(self):
        with self._lock:
            return len(self.frames)

    def alive(self):
        if self.eof:
            return False
        try:
            return self.proc.poll() is None
        except Exception:
            return False

    def returncode(self):
        try:
            return self.proc.poll()
        except Exception:
            return None

    def stalled_for(self):
        return time.monotonic() - float(self.last_progress_at or 0.0)

    def error_tail(self):
        return _decoder_error_tail(self.proc)

    def close(self):
        self._stop.set()
        try:
            self.proc.kill()
        except Exception:
            pass
        with self._lock:
            self.frames.clear()


def _download_tuning_opts():
    """Throughput options shared by foreground and background cache downloads."""
    opts = {
        "concurrent_fragment_downloads": _env_int(
            "AVATAR_YOUTUBE_DL_THREADS", DOWNLOAD_FRAGMENT_THREADS, minimum=1),
    }
    chunk = _env_int(
        "AVATAR_YOUTUBE_DL_CHUNK_MB",
        DOWNLOAD_CHUNK_BYTES // (1024 * 1024), minimum=0)
    # 0 disables chunking entirely and lets yt-dlp stream the whole format in
    # one request, which is the fastest path when the connection is stable.
    if chunk:
        opts["http_chunk_size"] = chunk * 1024 * 1024
    return opts


def _stream_first_enabled():
    """Play the direct stream at once and fill the disk cache in the background.

    Waiting for the full 720p file before the first frame is what made a new
    link feel slow: nothing is on screen for the whole download. ffmpeg reads
    the googlevideo URL directly just as well - that is already the fallback
    used when a cached file turns out to be broken.
    """
    return os.environ.get("AVATAR_YOUTUBE_VIDEO_STREAM_FIRST", "1") != "0"


class YouTubeVideoError(RuntimeError):
    pass


def _decoder_needs_restart(target, stream_position):
    drift = float(target) - float(stream_position)
    return drift < -1.0 or drift > MAX_FORWARD_DRIFT


def _is_live_info(info):
    info = info or {}
    return bool(info.get("is_live") or info.get("live_status") == "is_live")


def _cache_duration_limit():
    """Longest video worth caching, in seconds. 0 disables the limit."""
    minutes = _env_int(
        "AVATAR_YOUTUBE_CACHE_MAX_MINUTES",
        MAX_CACHE_DURATION_SECONDS // 60, minimum=0)
    return minutes * 60


def _too_long_to_cache(info):
    limit = _cache_duration_limit()
    if limit <= 0:
        return False
    try:
        duration = float((info or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return False
    return duration > limit


def _select_video_source(info):
    selected = _select_video_format(info)
    return selected.get("url", "") if selected else ""


def _is_hls_format(fmt):
    protocol = str((fmt or {}).get("protocol") or "").lower()
    return "m3u8" in protocol or "hls" in protocol


def _select_video_format(info):
    info = info or {}
    requested = info.get("requested_formats") or []
    candidates = list(requested) + list(info.get("formats") or [])
    candidates.append(info)
    video = [
        fmt for fmt in candidates
        if fmt.get("url") and fmt.get("vcodec") not in (None, "none")
    ]
    if not video:
        return ""
    bounded = [
        fmt for fmt in video
        if 0 < int(fmt.get("height") or 0) <= MAX_SOURCE_HEIGHT
    ]
    if bounded:
        video = bounded
    live = _is_live_info(info)
    video.sort(
        key=lambda fmt: (
            # A live broadcast only plays cleanly off its rolling HLS playlist;
            # a plain https segment URL runs dry the moment the buffer catches up.
            _is_hls_format(fmt) if live else (
                str(fmt.get("ext") or "").lower() == "mp4"),
            str(fmt.get("protocol") or "").lower() in ("https", "http")
            if not live else True,
            str(fmt.get("vcodec") or "").startswith(("avc1", "h264")),
            int(fmt.get("height") or 0),
            int(fmt.get("width") or 0),
            float(fmt.get("fps") or 0.0),
            float(fmt.get("tbr") or 0.0),
            fmt.get("acodec") not in (None, "none"),
        ),
        reverse=True,
    )
    return video[0]


def _player_client_ladder():
    """Clients to try, in order. AVATAR_YOUTUBE_PLAYER_CLIENTS overrides it."""
    raw = os.environ.get("AVATAR_YOUTUBE_PLAYER_CLIENTS", "").strip()
    if not raw:
        return PLAYER_CLIENT_LADDER
    ladder = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        ladder.append(None if name.lower() == "default" else [name])
    return tuple(ladder) or PLAYER_CLIENT_LADDER


def _looks_like_missing_formats(exc):
    text = one_line(exc).lower()
    return any(marker in text for marker in FORMAT_MISSING_MARKERS)


def _extract_video_info(yt_dlp, url, base_opts, status_callback=None):
    """Ask yt-dlp for the formats, walking past clients that answer empty.

    Only an empty answer moves to the next client. A real failure - the video
    is private, YouTube wants a sign-in, the network is down - is raised at
    once, because no other client would answer that differently.
    """
    first_error = None
    for index, clients in enumerate(_player_client_ladder()):
        opts = dict(base_opts)
        if clients:
            opts["extractor_args"] = {"youtube": {"player_client": list(clients)}}
        else:
            opts.pop("extractor_args", None)
        label = "/".join(clients) if clients else "default client"
        if index:
            _status(
                status_callback,
                f"video: no formats from the last client; retrying with {label}")
        try:
            info = extract_info_with_retries(
                yt_dlp, url, opts, download=False,
                status_callback=status_callback, status_prefix="video")
        except Exception as exc:
            if not _looks_like_missing_formats(exc):
                raise
            first_error = first_error or exc
            continue
        if _select_video_format(info):
            return info
        # A client can answer without error and still carry nothing playable.
        _status(status_callback, f"video: {label} returned no playable stream")
        first_error = first_error or YouTubeVideoError(
            f"{label} returned no playable video stream")
    raise first_error or YouTubeVideoError("no playable video stream was found")


def probe_youtube_live(url):
    """True/False if YouTube says this link is a broadcast in progress.

    Deliberately independent of the video scene and of any format filter or
    pinned client: the studio has to know whether a link is live even when the
    picture cannot be resolved, because a live link sent to the caption reader
    finds no captions and the button does nothing at all. None means the
    question could not be answered.
    """
    url = (url or "").strip()
    if not url:
        return None
    try:
        import yt_dlp
    except Exception:
        return None
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    try:
        info = extract_info_with_retries(yt_dlp, url, opts, download=False)
    except Exception:
        return None
    if not info:
        return None
    return _is_live_info(info)


def resolve_youtube_video(url, status_callback=None, force_refresh_cache=False,
                          wait_for_cache=False):
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeVideoError(f"yt-dlp is not installed ({exc})")

    _status(status_callback, "checking youtube video")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # "<=?" keeps the filter advisory: a live broadcast that only publishes
        # HLS renditions still resolves instead of failing "format not available".
        "format": (
            f"bestvideo[height<=?{MAX_SOURCE_HEIGHT}]/"
            f"best[height<=?{MAX_SOURCE_HEIGHT}]/"
            "best/"
            "worst"
        ),
        "format_sort": list(DOWNLOAD_FORMAT_SORT),
    }
    try:
        info = _extract_video_info(yt_dlp, url, opts, status_callback)
    except YouTubeVideoError:
        raise
    except Exception as exc:
        raise YouTubeVideoError(f"could not read YouTube video ({exc})")

    selected = _select_video_format(info)
    direct_source = selected.get("url", "") if selected else ""
    if not direct_source:
        raise YouTubeVideoError("no playable video stream was found")
    headers = dict((info or {}).get("http_headers") or {})
    headers.update(dict((selected or {}).get("http_headers") or {}))
    direct_headers = dict(headers)
    source = direct_source
    is_live = _is_live_info(info)
    too_long = _too_long_to_cache(info)
    if too_long:
        _status(
            status_callback,
            "video is longer than "
            f"{_cache_duration_limit() // 60} min; streaming without caching")
    if (not is_live and not too_long
            and os.environ.get("AVATAR_YOUTUBE_VIDEO_CACHE", "1") != "0"):
        cached_source = ""
        if _stream_first_enabled() and not force_refresh_cache and not wait_for_cache:
            cached_source = _existing_preview_video(url)
            if not cached_source:
                _start_background_preview_download(
                    yt_dlp, url, info, status_callback)
                _status(
                    status_callback,
                    "streaming video now; caching it in the background")
            else:
                _status(status_callback, "video preview found in local cache")
        else:
            cached_source = _cached_or_download_preview_video(
                yt_dlp, url, info, status_callback,
                force_refresh=bool(force_refresh_cache))
        if cached_source:
            source = cached_source
            headers = {}
    return {
        "source": source,
        "direct_source": direct_source,
        "title": (info or {}).get("title") or "YouTube video",
        "duration": float((info or {}).get("duration") or 0.0),
        "headers": headers,
        "direct_headers": direct_headers,
        "is_live": is_live,
        "thumbnail": (info or {}).get("thumbnail") or "",
    }


def _cache_preview_video_blocking(url, status_callback=None):
    """Fully cache a link's preview video before returning.

    Queue preloading wants the file on disk, not a stream URL: the whole point
    of preloading is that the next video is already local when it starts.
    """
    cached = _existing_preview_video(url)
    if cached:
        _status(status_callback, "video preview found in local cache")
        return cached
    info = resolve_youtube_video(
        url, status_callback=status_callback, wait_for_cache=True)
    source = info.get("source") or ""
    return source if not _is_remote_source(source) else ""


def _existing_preview_video(url):
    """Path of an already cached preview for this link, or "" if there is none."""
    try:
        for path in _preview_video_candidates(audio_dir(url)):
            return path
    except Exception:
        pass
    return ""


def _start_background_preview_download(yt_dlp, url, info, status_callback=None):
    """Download the cache copy off-thread so playback does not wait for it."""
    key = os.path.abspath(audio_dir(url))
    with _BACKGROUND_PREVIEW_GUARD:
        if key in _BACKGROUND_PREVIEW_JOBS:
            return False
        _BACKGROUND_PREVIEW_JOBS.add(key)

    def _worker():
        try:
            _cached_or_download_preview_video(
                yt_dlp, url, info, status_callback, force_refresh=False)
        except Exception as exc:
            _status(status_callback, f"background video cache failed ({exc})")
        finally:
            with _BACKGROUND_PREVIEW_GUARD:
                _BACKGROUND_PREVIEW_JOBS.discard(key)

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _cached_or_download_preview_video(
        yt_dlp, url, info, status_callback=None, force_refresh=False):
    out_dir = audio_dir(url)
    if not force_refresh:
        for path in _preview_video_candidates(out_dir):
            _status(status_callback, "video preview found in local cache")
            return path
    with _preview_download_lock(out_dir):
        if force_refresh:
            _quarantine_preview_files(out_dir, status_callback)
        else:
            for path in _preview_video_candidates(out_dir):
                _status(status_callback, "video preview found in local cache")
                return path
        _status(status_callback, "downloading HD video preview to cache")
        download_id = "preview-hd"
        dl_info = None
        last_exc = None
        for attempt in range(1, 5):
            opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                # Video-only and 720p. The scene decodes with -an and renders at
                # 1280x720, so an audio track or a taller rendition is downloaded
                # and then thrown away.
                "format": (
                    "bestvideo[height<=?720]/"
                    "best[height<=?720]/"
                    "best[ext=mp4]/"
                    "worst"
                ),
                "format_sort": list(DOWNLOAD_FORMAT_SORT),
                "continuedl": True,
                "retries": 8,
                "fragment_retries": 8,
                "file_access_retries": 5,
                "outtmpl": os.path.join(out_dir, f"{download_id}.%(ext)s"),
                "progress_hooks": [_preview_progress_hook(status_callback)],
            }
            opts.update(_download_tuning_opts())
            try:
                _status(
                    status_callback,
                    f"video preview download attempt {attempt}/4")
                dl_info = extract_info_with_retries(
                    yt_dlp, url, opts, download=True,
                    status_callback=status_callback,
                    status_prefix="video preview")
                break
            except Exception as exc:
                last_exc = exc
                _status(
                    status_callback,
                    f"video preview download interrupted; retrying ({exc})")
                time.sleep(0.5)
        if dl_info is None:
            _status(
                status_callback,
                f"video preview cache failed; direct stream fallback ({last_exc})")
            return ""
        path = _find_downloaded_preview(out_dir, download_id)
        if not path:
            path = (dl_info or {}).get("requested_downloads", [{}])[0].get(
                "filepath")
        if not path or not os.path.exists(path):
            _status(
                status_callback,
                "video preview cache missing; direct stream fallback")
            return ""
        _status(
            status_callback,
            "video preview saved in local cache "
            f"({human_bytes(os.path.getsize(path))})")
        _trim_cache_after_download(url, status_callback)
        return path


def _trim_cache_after_download(url, status_callback=None):
    """Evict the oldest cached videos so this download stays inside the budget."""
    try:
        removed, freed, _names = enforce_budget(
            keep_ids=[video_id_from_url(url)], status_callback=status_callback)
    except Exception as exc:
        _status(status_callback, f"cache cleanup skipped ({exc})")
        return
    if removed:
        _status(
            status_callback,
            f"cache trimmed: {removed} old video(s) removed, "
            f"{human_bytes(freed)} freed")


def _preview_download_lock(out_dir):
    key = os.path.abspath(out_dir)
    with _VIDEO_DOWNLOAD_LOCKS_GUARD:
        lock = _VIDEO_DOWNLOAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _VIDEO_DOWNLOAD_LOCKS[key] = lock
    return lock


def _quarantine_preview_files(out_dir, status_callback=None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = 0
    for path in _preview_video_candidates(out_dir):
        try:
            os.rename(path, f"{path}.old-{stamp}")
            moved += 1
        except Exception as exc:
            _status(
                status_callback,
                f"video preview refresh skipped locked cache ({exc})")
            return False
    if moved:
        _status(status_callback, "old video preview cache archived for refresh")
    return True


def _preview_video_candidates(out_dir):
    for pattern in ("preview-hd.*", "preview.*"):
        candidates = []
        for path in glob.glob(os.path.join(out_dir, pattern)):
            if not _is_usable_preview_file(path):
                continue
            candidates.append(path)
        if candidates:
            return sorted(candidates, key=lambda p: os.path.getmtime(p),
                          reverse=True)
    return []


def _find_downloaded_preview(out_dir, prefix):
    for path in glob.glob(os.path.join(out_dir, f"{prefix}.*")):
        if not _is_usable_preview_file(path):
            continue
        return path
    return ""


def _is_usable_preview_file(path):
    if not path or not os.path.isfile(path):
        return False
    name = os.path.basename(path).lower()
    if (name.endswith(".part") or ".bad" in name
            or ".old" in name or ".tmp" in name):
        return False
    return os.path.splitext(name)[1] in {
        ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts"
    }


def _preview_progress_hook(callback):
    last = {"value": -1}

    def _hook(data):
        try:
            status = data.get("status")
            if status == "downloading":
                total = (
                    data.get("total_bytes")
                    or data.get("total_bytes_estimate")
                    or 0)
                done = data.get("downloaded_bytes") or 0
                if total:
                    pct = int(max(0, min(100, done * 100.0 / total)))
                    if pct >= last["value"] + 2 or pct == 100:
                        last["value"] = pct
                        _status(callback, f"video preview download {pct}%")
                else:
                    mb = done / (1024.0 * 1024.0)
                    whole = int(mb)
                    if whole > last["value"]:
                        last["value"] = whole
                        _status(callback, f"video preview download {mb:.1f} MB")
            elif status == "finished":
                _status(callback, "video preview download 100%")
        except Exception:
            pass

    return _hook


def normalized_crop(frame, crop):
    if frame is None:
        return None
    if not crop:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in crop]
    left = max(0, min(w - 1, int(round(x1 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    right = max(left + 1, min(w, int(round(x2 * w))))
    bottom = max(top + 1, min(h, int(round(y2 * h))))
    return frame[top:bottom, left:right].copy()


def _placeholder_frame(label="LOADING VIDEO"):
    frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (8, 12, 16),
                    dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        image = Image.fromarray(frame, "RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, VIDEO_WIDTH - 1, VIDEO_HEIGHT - 1),
                       outline=(24, 210, 230), width=2)
        text = str(label or "LOADING VIDEO")
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((VIDEO_WIDTH - tw) // 2, (VIDEO_HEIGHT - th) // 2),
                  text, fill=(95, 245, 255), font=font)
        return np.asarray(image.convert("RGB"))
    except Exception:
        frame[::16, :, :] = (18, 60, 70)
        frame[:, ::16, :] = (18, 60, 70)
        return frame


def _thumbnail_frame(url, headers=None):
    url = (url or "").strip()
    if not url:
        return None
    try:
        from PIL import Image
        req_headers = {
            "User-Agent": (headers or {}).get(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
        }
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read(2_000_000)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (6, 8, 10))
        canvas.paste(
            image,
            ((VIDEO_WIDTH - image.width) // 2,
             (VIDEO_HEIGHT - image.height) // 2))
        return np.asarray(canvas)
    except Exception:
        return None


class _AudioClock:
    """The voice's playback position, as a time reference for live frames.

    A recorded video is decoded by seeking to this position, so it is in sync
    with the voice by construction. A live broadcast cannot be seeked, and its
    frames used to be metered to the wall clock instead - which left the two
    ffmpeg processes running open-loop against each other, with separate jitter
    buffers, separate stalls, and nothing to correct the gap that opened up
    between them. Metering against this position instead makes the sound the
    reference: while the voice is stalled its position stops advancing and the
    picture waits with it, so a hole in the audio costs a shared freeze rather
    than lip sync error that never comes back.

    The clock is only trusted while it is actually moving. It sits still during
    the voice's own pre-roll, and it stops for good when the audio ends or was
    never started, so `started` and `idle_for` let the caller pace itself
    instead of freezing a broadcast against a clock that will never tick again.
    """

    def __init__(self, getter):
        self._getter = getter
        self._value = None
        self._moved_at = 0.0
        self._started = False

    def read(self):
        """Current position in seconds, or None when there is no usable clock."""
        getter = self._getter
        if getter is None:
            return None
        try:
            value = getter()
        except Exception:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        now = time.monotonic()
        delta = None if self._value is None else value - self._value
        if delta is None:
            self._value = value
            self._moved_at = now
        elif delta < 0.0 or delta > CLOCK_JUMP_LIMIT:
            # A jump, not playback. Swapping to a new voice reads as the old
            # position replaced by the new one's starting point, and a scene
            # losing its voice reads as the studio's fallback position taking
            # over - both change the number without a single sample having
            # played. Counted as movement, that released the picture the
            # instant a voice's ffmpeg launched, before its pre-roll had
            # produced any sound, and the head start was back. A discontinuity
            # makes the clock unstarted again: it has to earn it by advancing.
            self._value = value
            self._moved_at = now
            self._started = False
        elif delta > 1e-6:
            # Movement, not merely a reading: a position parked at 0 while the
            # voice pre-rolls must not count as a clock that has started, or
            # the picture would anchor to it and run ahead exactly as before.
            self._value = value
            self._moved_at = now
            self._started = True
        return value

    def started(self):
        return self._started

    def idle_for(self):
        if not self._moved_at:
            return 0.0
        return max(0.0, time.monotonic() - self._moved_at)


class YouTubeVideoScene:
    """Decode a YouTube video against an external playback-position clock."""

    def __init__(
            self, position_getter, status_callback=None,
            force_refresh_cache=False, voice_active=None, voice_lag=None):
        self.position_getter = position_getter
        # How far behind the broadcast edge the voice is playing, in seconds.
        # The picture measures its own lag the same way and holds itself back
        # to match, so the two show the same instant of the broadcast. Without
        # it a picture already running when the voice starts keeps whatever
        # head start it had, and the frame meter below would lock that head
        # start in permanently instead of closing it.
        self.voice_lag = voice_lag
        # Whether a voice is playing this link, so a live scene knows if there
        # is a clock worth waiting for before it shows its first frame. A scene
        # with no voice must never hold the picture waiting for one: without
        # this it can only tell "not started yet" from "never coming" by
        # running out the clock, which is a black screen for that whole wait.
        self.voice_active = voice_active
        self.status_callback = status_callback
        self.title = ""
        self.duration = 0.0
        self.status = "idle"
        self.last_error = ""
        self.url = ""
        self.latest_frame = None
        self.frame_serial = 0
        self.video_ready = False
        # Frames are decoded and waiting, whether or not one has been shown
        # yet. A live picture holds frame 0 until the voice starts, and the
        # studio holds the voice until the picture is ready - so "ready" had to
        # stop meaning "a frame is on screen" or the two would wait on each
        # other until one of them timed out. This is the signal that says the
        # picture has done everything it can without the voice.
        self.buffered_ready = False
        self.is_live = False
        self.force_refresh_cache = bool(force_refresh_cache)

        self._source = ""
        self._direct_source = ""
        self._headers = {}
        self._direct_headers = {}
        self._running = False
        self._thread = None
        self._proc = None
        self._lock = threading.Lock()
        self._live_resolved_at = 0.0
        self._cache_rejected = False

    def start(self, url):
        self.stop()
        self.url = (url or "").strip()
        self.is_live = False
        with self._lock:
            self.latest_frame = _placeholder_frame("LOADING VIDEO")
            self.frame_serial = 1
            self.video_ready = False
            self.buffered_ready = False
        self.last_error = ""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def frame(self):
        with self._lock:
            return self.latest_frame

    def frame_snapshot(self):
        with self._lock:
            return self.frame_serial, self.latest_frame

    def _run(self):
        try:
            info = resolve_youtube_video(
                self.url, self._scene_status_callback(self.url),
                force_refresh_cache=self.force_refresh_cache)
            thumb = _thumbnail_frame(info.get("thumbnail"), info.get("headers"))
            if thumb is not None:
                with self._lock:
                    self.latest_frame = thumb
                    self.frame_serial += 1
            self._source = info["source"]
            self._headers = info["headers"]
            self._direct_source = info.get("direct_source") or self._source
            self._direct_headers = info.get("direct_headers") or {}
            self.title = info["title"]
            self.duration = info["duration"]
            self.is_live = bool(info.get("is_live"))
            self._live_resolved_at = time.monotonic()
            self._set_status(
                "live video scene ready" if self.is_live else "video scene ready")

            stream = None
            base_position = 0.0
            frame_index = 0
            frame_interval = 1.0 / VIDEO_FPS
            decoder_opened_at = 0.0
            pace_anchor = 0.0
            last_frame_at = 0.0
            pace_index = 0
            primed = False
            audio_clock = _AudioClock(self.position_getter)
            audio_anchor = None
            anchor_position = None
            paced_on_audio = False
            audio_wait_started = 0.0
            preroll = _live_preroll_seconds()
            buffering_announced = False
            health_at = 0.0
            health_base_position = 0.0
            health_frames = 0
            health_max_gap = 0.0
            health_stalls = 0
            last_shown_at = 0.0
            first_frame_failures = 0
            cache_decode_failures = 0
            stream_failures = 0
            ended_announced = False
            opened_once = False
            produced_frames = False
            while self._running:
                live = bool(self.is_live)
                first_frame_timeout = self._first_frame_timeout(live)
                # A live broadcast has no seekable timeline: it is always "now",
                # so the playback clock can never be a seek position for it.
                target = (
                    0.0 if live
                    else max(0.0, float(self.position_getter() or 0.0)))
                if self._at_end_of_video(target, live):
                    if self.video_ready:
                        # The clock has run past the last frame. Hold the picture
                        # instead of reopening a decoder that has nothing to
                        # read: that loop used to burn CPU and eventually delete
                        # the cached download as if it were broken.
                        if stream is not None:
                            stream.close()
                            stream = None
                            self._proc = None
                        if not ended_announced:
                            ended_announced = True
                            self._set_status(
                                "video reached the end; holding the last frame")
                        time.sleep(0.2)
                        continue
                    # Nothing has been shown yet, so the clock is stale from an
                    # earlier, longer video. Decode the closing seconds rather
                    # than hold a blank frame forever.
                    target = max(0.0, float(self.duration) - 1.0)
                stream_position = base_position + frame_index * frame_interval
                if stream is None or (
                        not live
                        and _decoder_needs_restart(target, stream_position)):
                    if stream is not None:
                        stream.close()
                    if live and opened_once and not produced_frames:
                        # A decoder that opened but never delivered a frame is
                        # what an expired playlist URL looks like. A mid-stream
                        # drop still delivered frames, so reopen straight away
                        # rather than pay for another five-second resolve.
                        self._refresh_live_source()
                    opened_once = True
                    produced_frames = False
                    primed = False
                    buffering_announced = False
                    # A stream-first start plays the googlevideo URL while the
                    # cache copy downloads. Every decoder reopen is a free
                    # chance to move onto the local file once it has landed:
                    # seeking a file beats re-ranging a remote URL.
                    self._adopt_cached_source_if_ready()
                    stream = self._open_stream(target)
                    decoder_opened_at = time.monotonic()
                    self._set_status(
                        "live video decoder opened at the broadcast edge" if live
                        else f"video decoder opened at {_fmt_time(target)}")
                    base_position = target
                    frame_index = 0
                    stream_position = target
                    # A reopen moves the playback position discontinuously, and
                    # the rate report measures how far it travelled during the
                    # window. Without re-basing here a seek is counted as
                    # playback: one reopen produced "109.63x, 0.0 fps", which is
                    # a number that cannot happen and would send anyone reading
                    # it after a speed complaint chasing the wrong thing.
                    health_at = 0.0
                    health_base_position = target

                if not live and frame_index > 0 and target <= stream_position + 0.01:
                    time.sleep(0.02)
                    continue

                # A live broadcast has no clock of its own to seek against, so
                # its frames are metered against the voice's playback position
                # instead - see _AudioClock - and against the wall clock only
                # when there is no voice to follow. Either meter needs
                # something buffered to meter: hold the picture until the
                # jitter buffer has enough frames to ride out a slow segment
                # fetch, then run at exactly 1x from there.
                if live:
                    if not primed:
                        want = max(1, int(round(preroll * VIDEO_FPS)))
                        patient = (
                            stream.alive()
                            and stream.stalled_for() < LIVE_PARTIAL_STALL_LIMIT
                            and (frame_index > 0
                                 or time.monotonic() - decoder_opened_at
                                 < first_frame_timeout))
                        if stream.depth() < want and patient:
                            if not buffering_announced and frame_index > 0:
                                buffering_announced = True
                                self._set_status("live video buffering")
                            time.sleep(0.02)
                            continue
                        if stream.depth() > 0:
                            # Decoded and ready. Said before the wait below, so
                            # a studio holding the voice back until the picture
                            # is ready can start it now - and be the thing that
                            # releases this wait - instead of the two sitting
                            # waiting on each other.
                            self.buffered_ready = True
                            # Frame 0 of a freshly opened decoder waits for the
                            # voice to actually start. The voice pre-rolls its
                            # own cushion before its first sample, and showing
                            # the picture during that wait is what put the
                            # picture ahead of the sound on every open.
                            if (frame_index == 0 and not audio_clock.started()
                                    and self._voice_is_playing()):
                                if not audio_wait_started:
                                    audio_wait_started = time.monotonic()
                                waited = time.monotonic() - audio_wait_started
                                if (audio_clock.read() is not None
                                        and waited < AUDIO_CLOCK_START_TIMEOUT):
                                    if not buffering_announced:
                                        buffering_announced = True
                                        self._set_status(
                                            "live video waiting for the voice")
                                    time.sleep(0.02)
                                    continue
                            primed = True
                            buffering_announced = False
                            audio_wait_started = 0.0
                            if frame_index == 0:
                                # A decoder that just opened starts at the
                                # broadcast edge, so any mapping from frame
                                # number to voice position is stale.
                                audio_anchor = None
                                anchor_position = None
                                pace_anchor = time.monotonic()
                                last_frame_at = pace_anchor
                                pace_index = 0
                            elif audio_anchor is None:
                                # Riding out a hiccup with no voice to steer by.
                                # Anchor on the moment playback actually
                                # resumes: anchoring on the decoder opening
                                # banked every second ffmpeg spent starting up
                                # as pacing credit, which the loop then spent
                                # fast-forwarding. With a voice to steer by the
                                # anchor is deliberately kept, so the frames the
                                # stall held back are overdue and the picture
                                # catches back up to the sound.
                                pace_anchor = time.monotonic()
                                last_frame_at = pace_anchor
                                pace_index = 0
                    if primed:
                        position = audio_clock.read()
                        use_audio = (
                            position is not None
                            and audio_clock.started()
                            # A scene with no voice still has a position to
                            # read: the studio falls back to a clock that just
                            # follows wall time, which is indistinguishable
                            # from a voice playing perfectly. Metering against
                            # that claims a sync that does not exist, and hands
                            # the picture to a clock the studio is free to stop
                            # - which would freeze a live broadcast for reasons
                            # that have nothing to do with the broadcast.
                            and self._voice_is_playing()
                            and audio_clock.idle_for() < AUDIO_CLOCK_STALL_LIMIT)
                        if (use_audio and anchor_position is not None
                                and position < anchor_position - 1.0):
                            # The voice restarted or was seeked backwards. Drop
                            # the mapping rather than wait for a clock that will
                            # never reach the old mark.
                            #
                            # Compared against where the voice was when this
                            # anchored, not against the anchor itself: holding
                            # the picture back deliberately puts the anchor
                            # ahead of the voice, and testing that looked
                            # exactly like the voice jumping backwards - so
                            # every hold re-anchored itself on the next pass,
                            # hundreds of times a second, until the hold decayed
                            # below this threshold on its own.
                            audio_anchor = None
                        if (use_audio != paced_on_audio
                                or (use_audio and audio_anchor is None)):
                            # Never inherit a deadline measured against the
                            # other clock: re-anchor whenever the reference the
                            # meter runs on changes.
                            paced_on_audio = use_audio
                            audio_anchor = position if use_audio else None
                            # Where the voice actually was at this anchor, kept
                            # separately because audio_anchor gets offset below.
                            anchor_position = position if use_audio else None
                            pace_anchor = time.monotonic()
                            last_frame_at = pace_anchor
                            pace_index = 0
                            if use_audio:
                                # Anchoring alone only stops the gap growing -
                                # it pins whatever head start the picture had
                                # when the voice started, which on this scene
                                # is the whole lip sync error. Close it: hold
                                # the picture by however much less of the
                                # broadcast it has missed than the voice has,
                                # and from here the meter keeps them level.
                                ahead = self._picture_is_ahead_by(
                                    decoder_opened_at, frame_index,
                                    frame_interval)
                                if ahead:
                                    audio_anchor += ahead
                                    self._set_status(
                                        "live video following the voice; "
                                        f"holding {ahead:.1f}s to match it")
                                else:
                                    self._set_status(
                                        "live video following the voice")
                        if use_audio:
                            # Rate control rather than a gate. How far the
                            # picture is behind the voice, in seconds of
                            # broadcast: positive means the voice is ahead and
                            # the picture should hurry, negative means it is
                            # running ahead and should ease off. Either way it
                            # keeps moving, which is the whole point - the gate
                            # this replaces stopped the picture dead for as
                            # long as the voice was stalled.
                            error = ((position - audio_anchor)
                                     - pace_index * frame_interval)
                            rate = max(SYNC_MIN_RATE,
                                       min(SYNC_MAX_RATE,
                                           1.0 + SYNC_RATE_GAIN * error))
                            due_at = last_frame_at + frame_interval / rate
                        else:
                            due_at = pace_anchor + pace_index * frame_interval
                        behind = due_at - time.monotonic()
                        if behind > 0:
                            time.sleep(min(behind, 0.02))
                            continue

                raw = stream.pop()
                if raw is None:
                    # Nothing buffered. On a live stream that is a hiccup to
                    # ride out, not a failure: hold the last picture, refill,
                    # and re-anchor the meter so playback resumes at 1x instead
                    # of racing to make up the lost time.
                    patience = (
                        first_frame_timeout if frame_index == 0
                        else (LIVE_PARTIAL_STALL_LIMIT if live else 1.5))
                    if stream.alive() and stream.stalled_for() < patience:
                        if live and primed:
                            primed = False
                            health_stalls += 1
                            preroll = min(
                                _live_max_preroll_seconds(), preroll + 0.5)
                        else:
                            time.sleep(0.01)
                        continue
                    if not self._running:
                        # stop() kills the decoder when a scene is replaced.
                        # Reporting that as a stream failure sent every scene
                        # switch to the log as a decoder hiccup.
                        break
                    return_code = stream.returncode()
                    waiting = time.monotonic() - decoder_opened_at
                    first_frame_failed = frame_index == 0
                    if first_frame_failed:
                        first_frame_failures += 1
                    if not _is_remote_source(self._source):
                        cache_decode_failures += 1
                    if return_code not in (None, 0):
                        detail = stream.error_tail()
                        self._set_status(
                            f"video decoder hiccup ({return_code}); resyncing"
                            + (f": {detail}" if detail else ""))
                    elif frame_index == 0 and waiting >= first_frame_timeout:
                        detail = stream.error_tail()
                        self._set_status(
                            "video decoder produced no first frame; resyncing"
                            + (f": {detail}" if detail else ""))
                    else:
                        # The generic stall used to say nothing about itself,
                        # which made it the one failure here that could only be
                        # guessed at. These are the terms the branch was chosen
                        # on: whether the reader is still alive, how long the
                        # decoder has been quiet, and what it has produced.
                        detail = stream.error_tail()
                        self._set_status(
                            "video decoder stalled; resyncing "
                            f"(alive={stream.alive()}, "
                            f"quiet {stream.stalled_for():.1f}s, "
                            f"depth {stream.depth()}, "
                            f"produced {stream.produced}, "
                            f"dropped {stream.dropped}, rc={return_code})"
                            + (f": {detail}" if detail else ""))
                    stream.close()
                    stream = None
                    self._proc = None
                    if _is_remote_source(self._source):
                        stream_failures += 1
                    should_bypass_cache = (
                        first_frame_failed
                        and first_frame_failures >= LOCAL_CACHE_FIRST_FRAME_RETRIES
                    ) or cache_decode_failures >= LOCAL_CACHE_DECODE_RETRIES
                    if should_bypass_cache and self._fallback_from_bad_cache():
                        first_frame_failures = 0
                        cache_decode_failures = 0
                    # A stream-first scene plays a googlevideo URL, and those
                    # expire and start returning 403. _fallback_from_bad_cache
                    # cannot help - it only rescues a local file - so without
                    # this the loop retried a dead URL forever and the video
                    # simply never appeared.
                    elif stream_failures >= REMOTE_SOURCE_REFRESH_RETRIES:
                        if self._refresh_direct_source():
                            stream_failures = 0
                            first_frame_failures = 0
                    time.sleep(DECODER_RESTART_BACKOFF)
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    VIDEO_HEIGHT, VIDEO_WIDTH, 3)
                produced_frames = True
                with self._lock:
                    self.latest_frame = frame
                    self.frame_serial += 1
                    self.video_ready = True
                if frame_index == 0:
                    self._set_status("video first frame ready")
                    first_frame_failures = 0
                    cache_decode_failures = 0
                    ended_announced = False
                frame_index += 1
                pace_index += 1
                now = time.monotonic()
                if live:
                    # The frame clock the rate control paces against. Stepped
                    # by the interval it was due rather than set to now, so a
                    # late frame does not quietly become the new baseline and
                    # shorten every interval that follows. Only the live path
                    # defines due_at; a recording is paced by seeking.
                    last_frame_at = max(due_at, now - frame_interval)
                if last_shown_at:
                    health_max_gap = max(health_max_gap, now - last_shown_at)
                last_shown_at = now
                health_frames += 1
                if not health_at:
                    health_at = now
                    health_base_position = stream_position
                elif now - health_at >= HEALTH_REPORT_SECONDS:
                    span = now - health_at
                    if live:
                        self._report_playback_health(
                            span, health_frames, health_max_gap,
                            health_stalls, stream.depth())
                    else:
                        # Recorded video is metered by seeking to the playback
                        # clock, so the number that matters is how much video
                        # went by per second of real time. Only live playback
                        # used to report anything, which left "the picture is
                        # running fast" as something nobody could measure.
                        played = stream_position - health_base_position
                        self._set_status(
                            f"video playback {played / span:.2f}x, "
                            f"{health_frames / span:.1f} fps of {VIDEO_FPS:.0f}, "
                            f"clock {_fmt_time(target)}")
                    health_at = now
                    health_base_position = stream_position
                    health_frames = 0
                    health_max_gap = 0.0
                    health_stalls = 0
        except Exception as exc:
            self.last_error = str(exc)
            with self._lock:
                if self.latest_frame is None:
                    self.latest_frame = _placeholder_frame("VIDEO RECONNECTING")
                    self.frame_serial += 1
            self._set_status(f"video scene failed: {exc}")
        finally:
            self.stop()

    def _open_stream(self, position):
        """Open a decoder and start draining its pipe into a jitter buffer."""
        proc = self._open_decoder(position)
        self._proc = proc
        # Only a live decoder may drop frames. A recording is seekable and is
        # metered by the playback clock, so every frame it decodes is one the
        # viewer is owed, in order.
        return _DecoderStream(
            proc, self._buffer_frames(), block_when_full=not self.is_live)

    def _report_playback_health(self, span, frames, max_gap, stalls, depth):
        """Log how smooth the last stretch of live playback actually was.

        Deliberately avoids the word "ready": the studio rebuilds its scene
        panel on any status containing it.
        """
        if span <= 0.0 or frames <= 0:
            return
        self._set_status(
            f"live video playback {frames / span:.1f} fps of {VIDEO_FPS:.0f}, "
            f"longest gap {max_gap * 1000:.0f} ms, {stalls} stall(s), "
            f"buffer {depth / VIDEO_FPS:.1f}s")

    def _picture_is_ahead_by(self, decoder_opened_at, frame_index,
                             frame_interval):
        """Seconds of broadcast the picture has ahead of the voice.

        Each side measures how much the broadcast moved on while it was not
        playing - startup, pre-roll, every stall - which is its distance behind
        the live edge. The difference is the lip sync error, and it is positive
        when the picture is the one running ahead.

        Both decoders open at the same place relative to the edge, so what one
        spent getting started cancels against the other and only the difference
        survives. Zero when there is nothing to measure or the picture is the
        side that is behind, since the meter closes that case by itself: those
        frames are already overdue and play out until they catch up.
        """
        voice = self._voice_lag_seconds()
        if voice is None or not decoder_opened_at:
            return 0.0
        shown = max(0, int(frame_index)) * float(frame_interval)
        picture = max(0.0, (time.monotonic() - decoder_opened_at) - shown)
        ahead = voice - picture
        if ahead <= 0.05:
            return 0.0
        # A correction beyond the jitter buffer cannot be held anyway: the
        # decoder would jam on the pipe and drop off the edge mid-hold.
        return min(ahead, max(0.0, _video_buffer_seconds() - _live_preroll_seconds()))

    def _voice_lag_seconds(self):
        """The voice's lag behind the broadcast edge, or None if unknown."""
        getter = self.voice_lag
        if getter is None:
            return None
        try:
            value = getter()
        except Exception:
            return None
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _voice_is_playing(self):
        """Is there a voice whose clock this picture should wait for?"""
        getter = self.voice_active
        if getter is None:
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _first_frame_timeout(self, live):
        """How long this source may take to produce its first frame.

        A cached file on disk starts decoding almost at once, so a long wait
        there really is a broken download. A network source is slow by nature
        and must not be judged by the same clock.
        """
        if live:
            return LIVE_FIRST_FRAME_TIMEOUT
        if _is_remote_source(self._source):
            return REMOTE_FIRST_FRAME_TIMEOUT
        return DECODER_FIRST_FRAME_TIMEOUT

    def _buffer_frames(self):
        """How many frames of slack this source needs.

        A live playlist and a remote recording both stall on segment fetches; a
        cached file on disk does not, and buffering it only costs memory.
        """
        seconds = (
            _video_buffer_seconds() if (
                self.is_live or _is_remote_source(self._source))
            else _env_float(
                "AVATAR_YOUTUBE_LOCAL_BUFFER", LOCAL_BUFFER_SECONDS,
                minimum=0.1))
        return max(2, int(round(seconds * VIDEO_FPS)))

    def _open_decoder(self, position):
        live = bool(self.is_live)
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        user_agent = self._headers.get("User-Agent")
        if user_agent:
            cmd += ["-user_agent", user_agent]
        header_text = _ffmpeg_header_text(self._headers)
        if header_text:
            cmd += ["-headers", header_text]
        if _is_remote_source(self._source):
            cmd += [
                "-rw_timeout", str(
                    LIVE_READ_TIMEOUT_US if live else DECODER_READ_TIMEOUT_US),
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "2",
            ]
            if not live:
                # On a live HLS playlist this makes ffmpeg reconnect forever on
                # the normal end-of-segment EOF and never emit a single frame.
                cmd += ["-reconnect_at_eof", "1"]
        position = 0.0 if live else max(0.0, float(position))
        fast_seek = max(0.0, position - 5.0)
        fine_seek = position - fast_seek
        if fast_seek > 0:
            cmd += ["-ss", str(fast_seek)]
        cmd += ["-i", self._source]
        if fine_seek > 0:
            cmd += ["-ss", str(fine_seek)]
        cmd += [
            "-an",
            "-vf",
            (
                f"fps={VIDEO_FPS},"
                f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
            ),
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _drain_stderr(proc)
        return proc

    def _at_end_of_video(self, position, live=False):
        """True when the clock has run past the last frame of a finished video."""
        if live:
            return False
        duration = float(self.duration or 0.0)
        if duration <= 0.0:
            return False
        return float(position) >= duration - END_OF_VIDEO_MARGIN

    def _refresh_live_source(self):
        """Re-ask yt-dlp for the playlist URL; live manifests expire as they age."""
        if not self.is_live or not self.url:
            return False
        age = time.monotonic() - float(self._live_resolved_at or 0.0)
        if age < LIVE_SOURCE_REFRESH_SECONDS:
            return False
        try:
            info = resolve_youtube_video(self.url)
        except Exception as exc:
            self._set_status(f"live link refresh failed ({exc})")
            self._live_resolved_at = time.monotonic()
            return False
        self._source = info["source"]
        self._headers = info["headers"]
        self._direct_source = info.get("direct_source") or self._source
        self._direct_headers = info.get("direct_headers") or {}
        self.is_live = bool(info.get("is_live"))
        self._live_resolved_at = time.monotonic()
        self._set_status("live link refreshed")
        return True

    def _scene_status_callback(self, url):
        """Status sink that goes quiet once this scene is no longer the one shown.

        The cache download outlives the scene that started it, so without this
        an abandoned download keeps driving the progress bar for a video the
        user already moved off.
        """
        def _forward(message):
            if not self._running or self.url != url:
                return
            self._set_status(message)

        return _forward

    def _adopt_cached_source_if_ready(self):
        """Switch from the direct stream to the cached file once it exists."""
        if self.is_live or self._cache_rejected or not self.url:
            return False
        if not _is_remote_source(self._source):
            return False
        cached = _existing_preview_video(self.url)
        if not cached:
            return False
        self._source = cached
        self._headers = {}
        self._set_status("video cache ready; playing from local file")
        return True

    def _refresh_direct_source(self):
        """Re-resolve an expired googlevideo URL for a finished (non-live) video.

        Prefers the cached file if the background download has landed by now;
        otherwise asks yt-dlp for a fresh stream URL.
        """
        if self.is_live or not self.url:
            return False
        if self._adopt_cached_source_if_ready():
            return True
        try:
            info = resolve_youtube_video(self.url)
        except Exception as exc:
            self._set_status(f"video link refresh failed ({exc})")
            return False
        source = info.get("direct_source") or info.get("source") or ""
        if not source:
            return False
        self._source = source
        self._headers = dict(info.get("direct_headers") or {})
        self._direct_source = source
        self._direct_headers = dict(self._headers)
        self._set_status("video link refreshed; resuming stream")
        return True

    def _fallback_from_bad_cache(self):
        if _is_remote_source(self._source):
            return False
        if not _is_remote_source(self._direct_source):
            return False
        bad_path = self._source
        # Never re-adopt a cache copy this scene has already judged broken.
        self._cache_rejected = True
        self._set_status("cached video failed to decode; using direct stream")
        try:
            if bad_path and os.path.isfile(bad_path):
                os.remove(bad_path)
                self._set_status("removed broken youtube video cache")
        except Exception as exc:
            self._set_status(f"could not remove broken video cache ({exc})")
        self._source = self._direct_source
        self._headers = dict(self._direct_headers or {})
        return True

    def _set_status(self, status):
        self.status = status
        _status(self.status_callback, status)


def _read_exact(stream, size, timeout=1.5):
    """Read up to `size` bytes, returning early on EOF or when time runs out.

    This deliberately uses a plain blocking read. The previous version asked for
    a non-blocking pipe first, but os.set_blocking does not exist on Windows, so
    the call raised, was swallowed, and every read blocked anyway - the timeout
    was decorative. It is now called only from _DecoderStream's own thread,
    where blocking is what we want: no busy-wait, and nothing else is held up.
    """
    chunks = []
    remaining = int(size)
    deadline = time.monotonic() + max(0.05, float(timeout))
    while remaining > 0 and time.monotonic() < deadline:
        try:
            chunk = os.read(stream.fileno(), remaining)
        except Exception:
            try:
                chunk = stream.read(remaining)
            except Exception:
                chunk = b""
        if not chunk:
            # An empty read on a blocking pipe means the writer closed it.
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain_stderr(proc):
    """Keep reading ffmpeg's stderr so it can never block on a full pipe.

    Nothing read this pipe during normal playback. A long live session emits a
    reconnect warning every so often, and once the 64 KB pipe buffer filled,
    ffmpeg blocked writing to it and stopped producing video while still
    appearing alive - the "decoder stalled" freezes. The drained text is kept in
    a small rolling buffer so failures can still be reported with detail.
    """
    stream = getattr(proc, "stderr", None)
    if stream is None:
        return
    proc._error_tail = collections.deque(maxlen=DECODER_ERROR_LINES)

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

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()


def _decoder_error_tail(proc):
    try:
        lines = list(getattr(proc, "_error_tail", ()) or ())
        if not lines:
            return ""
        return " ".join(lines).replace("\r", " ")[-500:]
    except Exception:
        return ""


def _ffmpeg_header_text(headers):
    allowed = {
        "Accept",
        "Accept-Language",
        "Cookie",
        "Origin",
        "Referer",
    }
    lines = []
    for key, value in (headers or {}).items():
        key = str(key).strip()
        if key not in allowed:
            continue
        value = str(value).replace("\r", " ").replace("\n", " ").strip()
        if value:
            lines.append(f"{key}: {value}")
    return "\r\n".join(lines) + ("\r\n" if lines else "")


def _is_remote_source(source):
    return str(source or "").lower().startswith(("http://", "https://"))


def _status(callback, message):
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _fmt_time(seconds):
    seconds = max(0, int(float(seconds or 0.0)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
