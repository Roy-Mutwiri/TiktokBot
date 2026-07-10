import glob
import io
import os
import subprocess
import threading
import time
import urllib.request

import numpy as np

from tts_stream_engine import FFMPEG
from youtube_cache import audio_dir
from youtube_dlp_options import extract_info_with_retries


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 15.0
MAX_SOURCE_HEIGHT = 720
MAX_FORWARD_DRIFT = 30.0
DECODER_READ_TIMEOUT_US = 5_000_000
DECODER_RESTART_BACKOFF = 0.35
DECODER_ERROR_BYTES = 4000
DECODER_FIRST_FRAME_TIMEOUT = 8.0
LOCAL_CACHE_FIRST_FRAME_RETRIES = 2
LOCAL_CACHE_DECODE_RETRIES = 3

_VIDEO_DOWNLOAD_LOCKS = {}
_VIDEO_DOWNLOAD_LOCKS_GUARD = threading.Lock()


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
    video.sort(
        key=lambda fmt: (
            str(fmt.get("ext") or "").lower() == "mp4",
            str(fmt.get("protocol") or "").lower() in ("https", "http"),
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


def resolve_youtube_video(url, status_callback=None, force_refresh_cache=False):
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeVideoError(f"yt-dlp is not installed ({exc})")

    _status(status_callback, "checking youtube video")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": (
            f"bestvideo[height<={MAX_SOURCE_HEIGHT}][ext=mp4]/"
            f"best[height<={MAX_SOURCE_HEIGHT}][ext=mp4]/"
            "best[height<=720]/"
            "worst"
        ),
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
                "format": (
                    "bestvideo[height<=720][ext=mp4]/"
                    "best[height<=720][ext=mp4]/"
                    "best[height<=720]/"
                    "best[ext=mp4]/"
                    "worst"
                ),
                "continuedl": True,
                "retries": 8,
                "fragment_retries": 8,
                "file_access_retries": 5,
                "http_chunk_size": 1024 * 1024,
                "outtmpl": os.path.join(out_dir, f"{download_id}.%(ext)s"),
                "progress_hooks": [_preview_progress_hook(status_callback)],
            }
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
        _status(status_callback, "video preview saved in local cache")
        return path


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
        self.force_refresh_cache = bool(force_refresh_cache)

        self._source = ""
        self._direct_source = ""
        self._headers = {}
        self._direct_headers = {}
        self._running = False
        self._thread = None
        self._proc = None
        self._lock = threading.Lock()

    def start(self, url):
        self.stop()
        self.url = (url or "").strip()
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
                self.url, self._set_status,
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
            self._set_status("video scene ready")

            proc = None
            base_position = 0.0
            frame_index = 0
            frame_interval = 1.0 / VIDEO_FPS
            decoder_opened_at = 0.0
            first_frame_failures = 0
            cache_decode_failures = 0
            while self._running:
                target = max(0.0, float(self.position_getter() or 0.0))
                stream_position = base_position + frame_index * frame_interval
                if proc is None or _decoder_needs_restart(
                        target, stream_position):
                    if proc is not None:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    proc = self._open_decoder(target)
                    self._proc = proc
                    decoder_opened_at = time.monotonic()
                    self._set_status(f"video decoder opened at {_fmt_time(target)}")
                    base_position = target
                    frame_index = 0
                    stream_position = target

                if frame_index > 0 and target <= stream_position + 0.01:
                    time.sleep(0.02)
                    continue

                raw = _read_exact(
                    proc.stdout, VIDEO_WIDTH * VIDEO_HEIGHT * 3,
                    timeout=(
                        DECODER_FIRST_FRAME_TIMEOUT
                        if frame_index == 0 else 1.5)
                ) if proc.stdout is not None else b""
                if len(raw) != VIDEO_WIDTH * VIDEO_HEIGHT * 3:
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
                    elif frame_index == 0 and waiting >= DECODER_FIRST_FRAME_TIMEOUT:
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
                    should_bypass_cache = (
                        first_frame_failed
                        and first_frame_failures >= LOCAL_CACHE_FIRST_FRAME_RETRIES
                    ) or cache_decode_failures >= LOCAL_CACHE_DECODE_RETRIES
                    if should_bypass_cache and self._fallback_from_bad_cache():
                        first_frame_failures = 0
                        cache_decode_failures = 0
                    time.sleep(DECODER_RESTART_BACKOFF)
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    VIDEO_HEIGHT, VIDEO_WIDTH, 3)
                with self._lock:
                    self.latest_frame = frame
                    self.frame_serial += 1
                    self.video_ready = True
                if frame_index == 0:
                    self._set_status("video first frame ready")
                    first_frame_failures = 0
                    cache_decode_failures = 0
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
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        user_agent = self._headers.get("User-Agent")
        if user_agent:
            cmd += ["-user_agent", user_agent]
        header_text = _ffmpeg_header_text(self._headers)
        if header_text:
            cmd += ["-headers", header_text]
        if _is_remote_source(self._source):
            cmd += [
                "-rw_timeout", str(DECODER_READ_TIMEOUT_US),
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_delay_max", "2",
            ]
        position = max(0.0, float(position))
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
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _fallback_from_bad_cache(self):
        if _is_remote_source(self._source):
            return False
        if not _is_remote_source(self._direct_source):
            return False
        bad_path = self._source
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


def _decoder_error_tail(proc):
    try:
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return ""
        try:
            os.set_blocking(stream.fileno(), False)
        except Exception:
            pass
        data = b""
        while True:
            chunk = stream.read(DECODER_ERROR_BYTES)
            if not chunk:
                break
            data += chunk
            if len(data) > DECODER_ERROR_BYTES:
                data = data[-DECODER_ERROR_BYTES:]
        return data.decode("utf-8", "replace").strip().replace(
            "\r", " ").replace("\n", " ")[-500:]
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
