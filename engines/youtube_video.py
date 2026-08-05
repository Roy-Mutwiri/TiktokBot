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
from youtube_dlp_options import extract_info_with_retries


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
LIVE_FIRST_FRAME_TIMEOUT = 20.0
LIVE_SOURCE_REFRESH_SECONDS = 45.0
# How long a live decoder may deliver nothing before it counts as dead. ffmpeg
# goes quiet for a few seconds each time it reconnects a dropped TLS connection.
LIVE_PARTIAL_STALL_LIMIT = 25.0
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
        "extractor_args": {
            "youtube": {"player_client": ["android"]},
        },
    }
    try:
        info = extract_info_with_retries(
            yt_dlp, url, opts, download=False,
            status_callback=status_callback, status_prefix="video")
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
    if not is_live and os.environ.get("AVATAR_YOUTUBE_VIDEO_CACHE", "1") != "0":
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


class YouTubeVideoScene:
    """Decode a YouTube video against an external playback-position clock."""

    def __init__(
            self, position_getter, status_callback=None,
            force_refresh_cache=False):
        self.position_getter = position_getter
        self.status_callback = status_callback
        self.title = ""
        self.duration = 0.0
        self.status = "idle"
        self.last_error = ""
        self.url = ""
        self.latest_frame = None
        self.frame_serial = 0
        self.video_ready = False
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

            proc = None
            base_position = 0.0
            frame_index = 0
            frame_interval = 1.0 / VIDEO_FPS
            decoder_opened_at = 0.0
            first_frame_failures = 0
            cache_decode_failures = 0
            stream_failures = 0
            ended_announced = False
            opened_once = False
            produced_frames = False
            pending = b""
            last_progress_at = time.monotonic()
            while self._running:
                live = bool(self.is_live)
                first_frame_timeout = (
                    LIVE_FIRST_FRAME_TIMEOUT if live
                    else DECODER_FIRST_FRAME_TIMEOUT)
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
                        if proc is not None:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            proc = None
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
                if proc is None or (
                        not live
                        and _decoder_needs_restart(target, stream_position)):
                    if proc is not None:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    if live and opened_once and not produced_frames:
                        # A decoder that opened but never delivered a frame is
                        # what an expired playlist URL looks like. A mid-stream
                        # drop still delivered frames, so reopen straight away
                        # rather than pay for another five-second resolve.
                        self._refresh_live_source()
                    opened_once = True
                    produced_frames = False
                    pending = b""
                    # A stream-first start plays the googlevideo URL while the
                    # cache copy downloads. Every decoder reopen is a free
                    # chance to move onto the local file once it has landed:
                    # seeking a file beats re-ranging a remote URL.
                    self._adopt_cached_source_if_ready()
                    proc = self._open_decoder(target)
                    self._proc = proc
                    decoder_opened_at = time.monotonic()
                    last_progress_at = decoder_opened_at
                    self._set_status(
                        "live video decoder opened at the broadcast edge" if live
                        else f"video decoder opened at {_fmt_time(target)}")
                    base_position = target
                    frame_index = 0
                    stream_position = target

                if not live and frame_index > 0 and target <= stream_position + 0.01:
                    time.sleep(0.02)
                    continue
                # ffmpeg hands over the segments it already buffered as fast as
                # it can decode them, then follows the broadcast in real time.
                # Showing that opening burst at whatever speed it arrives makes
                # the picture race, and leaves it minutes ahead of the voice,
                # which is paced to real time. Meter live frames to the wall
                # clock so both run at 1x from the same moment.
                if live and frame_index > 0:
                    due_at = decoder_opened_at + frame_index * frame_interval
                    behind = due_at - time.monotonic()
                    if behind > 0:
                        time.sleep(min(behind, 0.05))
                        continue

                frame_bytes = VIDEO_WIDTH * VIDEO_HEIGHT * 3
                chunk = _read_exact(
                    proc.stdout, frame_bytes - len(pending),
                    timeout=(
                        first_frame_timeout if frame_index == 0
                        else (6.0 if live else 1.5))
                ) if proc.stdout is not None else b""
                if chunk:
                    pending += chunk
                    last_progress_at = time.monotonic()
                if len(pending) < frame_bytes:
                    # A live broadcast pauses its output for a few seconds while
                    # ffmpeg reconnects after a dropped TLS connection, and it
                    # recovers on its own. Keep the half-read frame and wait
                    # rather than killing a decoder that is still alive.
                    if (live and proc.poll() is None
                            and time.monotonic() - last_progress_at
                            < LIVE_PARTIAL_STALL_LIMIT):
                        continue
                    pending = b""
                    if not self._running:
                        # stop() kills the decoder when a scene is replaced.
                        # Reporting that as a stream failure sent every scene
                        # switch to the log as a decoder hiccup.
                        break
                    return_code = proc.poll()
                    waiting = time.monotonic() - decoder_opened_at
                    first_frame_failed = frame_index == 0
                    if first_frame_failed:
                        first_frame_failures += 1
                    if not _is_remote_source(self._source):
                        cache_decode_failures += 1
                    if return_code not in (None, 0):
                        detail = _decoder_error_tail(proc)
                        self._set_status(
                            f"video decoder hiccup ({return_code}); resyncing"
                            + (f": {detail}" if detail else ""))
                    elif frame_index == 0 and waiting >= first_frame_timeout:
                        detail = _decoder_error_tail(proc)
                        self._set_status(
                            "video decoder produced no first frame; resyncing"
                            + (f": {detail}" if detail else ""))
                    else:
                        self._set_status("video decoder stalled; resyncing")
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc = None
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
                raw = pending
                pending = b""
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
        except Exception as exc:
            self.last_error = str(exc)
            with self._lock:
                if self.latest_frame is None:
                    self.latest_frame = _placeholder_frame("VIDEO RECONNECTING")
                    self.frame_serial += 1
            self._set_status(f"video scene failed: {exc}")
        finally:
            self.stop()

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
    try:
        os.set_blocking(stream.fileno(), False)
    except Exception:
        pass
    chunks = []
    remaining = int(size)
    deadline = time.monotonic() + max(0.05, float(timeout))
    while remaining > 0 and time.monotonic() < deadline:
        try:
            chunk = os.read(stream.fileno(), remaining)
        except BlockingIOError:
            time.sleep(0.01)
            continue
        except Exception:
            try:
                chunk = stream.read(remaining)
            except Exception:
                chunk = b""
        if not chunk:
            time.sleep(0.01)
            continue
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
