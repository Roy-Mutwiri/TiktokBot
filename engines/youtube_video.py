import subprocess
import threading
import time

import numpy as np

from tts_stream_engine import FFMPEG


VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_FPS = 24.0
MAX_FORWARD_DRIFT = 30.0


class YouTubeVideoError(RuntimeError):
    pass


def _decoder_needs_restart(target, stream_position):
    drift = float(target) - float(stream_position)
    return drift < -1.0 or drift > MAX_FORWARD_DRIFT


def _is_live_info(info):
    info = info or {}
    return bool(info.get("is_live") or info.get("live_status") == "is_live")


def _select_video_source(info):
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
    progressive = [
        fmt for fmt in video
        if fmt.get("acodec") not in (None, "none")
        and int(fmt.get("height") or 0) <= 480
        and str(fmt.get("vcodec") or "").startswith(("avc1", "h264"))
    ]
    if progressive:
        video = progressive
    video.sort(
        key=lambda fmt: (
            int(fmt.get("height") or 0) <= 720,
            int(fmt.get("height") or 0),
            str(fmt.get("vcodec") or "").startswith(("avc1", "h264")),
            str(fmt.get("vcodec") or "").startswith(("vp8", "vp9")),
            float(fmt.get("fps") or 0.0),
            float(fmt.get("tbr") or 0.0),
        ),
        reverse=True,
    )
    return video[0]["url"]


def resolve_youtube_video(url, status_callback=None):
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeVideoError(f"yt-dlp is not installed ({exc})")

    _status(status_callback, "checking youtube video")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "extractor_args": {
            "youtube": {"player_client": ["android"]},
        },
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise YouTubeVideoError(f"could not read YouTube video ({exc})")

    source = _select_video_source(info)
    if not source:
        raise YouTubeVideoError("no playable video stream was found")
    return {
        "source": source,
        "title": (info or {}).get("title") or "YouTube video",
        "duration": float((info or {}).get("duration") or 0.0),
        "headers": dict((info or {}).get("http_headers") or {}),
        "is_live": _is_live_info(info),
    }


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


class YouTubeVideoScene:
    """Decode a YouTube video against an external playback-position clock."""

    def __init__(self, position_getter, status_callback=None):
        self.position_getter = position_getter
        self.status_callback = status_callback
        self.title = ""
        self.duration = 0.0
        self.status = "idle"
        self.last_error = ""
        self.url = ""
        self.latest_frame = None
        self.frame_serial = 0

        self._source = ""
        self._headers = {}
        self._running = False
        self._thread = None
        self._proc = None
        self._lock = threading.Lock()

    def start(self, url):
        self.stop()
        self.url = (url or "").strip()
        self.latest_frame = None
        self.frame_serial = 0
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
            # Frames are replaced atomically and never mutated after publish.
            # Returning the current array avoids another 1.5 MB copy per frame.
            return self.latest_frame

    def frame_snapshot(self):
        with self._lock:
            return self.frame_serial, self.latest_frame

    def _run(self):
        try:
            info = resolve_youtube_video(self.url, self._set_status)
            self._source = info["source"]
            self._headers = info["headers"]
            self.title = info["title"]
            self.duration = info["duration"]
            self._set_status("video scene ready")

            proc = None
            base_position = 0.0
            frame_index = 0
            frame_interval = 1.0 / VIDEO_FPS
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
                    base_position = target
                    frame_index = 0
                    stream_position = target

                if frame_index > 0 and target <= stream_position + 0.01:
                    time.sleep(0.02)
                    continue

                raw = _read_exact(
                    proc.stdout, VIDEO_WIDTH * VIDEO_HEIGHT * 3
                ) if proc.stdout is not None else b""
                if len(raw) != VIDEO_WIDTH * VIDEO_HEIGHT * 3:
                    return_code = proc.poll()
                    if return_code not in (None, 0):
                        raise YouTubeVideoError(
                            f"video decoder exited with code {return_code}")
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    VIDEO_HEIGHT, VIDEO_WIDTH, 3)
                with self._lock:
                    self.latest_frame = frame
                    self.frame_serial += 1
                frame_index += 1
        except Exception as exc:
            self.last_error = str(exc)
            self._set_status(f"video scene failed: {exc}")
        finally:
            self.stop()

    def _open_decoder(self, position):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        user_agent = self._headers.get("User-Agent")
        if user_agent:
            cmd += ["-user_agent", user_agent]
        position = max(0.0, float(position))
        fast_seek = max(0.0, position - 5.0)
        fine_seek = position - fast_seek
        if fast_seek > 0:
            cmd += ["-ss", str(fast_seek)]
        cmd += ["-i", self._source]
        # Use an HTTP range seek to avoid decoding a large remote video from
        # the beginning, then trim only the final few seconds for exact sync.
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
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _set_status(self, status):
        self.status = status
        _status(self.status_callback, status)


def _read_exact(stream, size):
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _status(callback, message):
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass
