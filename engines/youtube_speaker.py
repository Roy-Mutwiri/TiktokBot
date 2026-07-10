import json
import os
import re
import urllib.request
from html import unescape

from youtube_cache import get_cached_transcript, save_transcript
from youtube_dlp_options import extract_info_with_retries


# Long videos are expected. 250k chars is roughly several hours of captions while
# still bounding memory and accidental playlist-sized inputs. Set 0 for no cap.
MAX_TRANSCRIPT_CHARS = int(os.environ.get("AVATAR_YOUTUBE_MAX_CHARS", "250000"))
CHUNK_CHARS = int(os.environ.get("AVATAR_YOUTUBE_CHUNK_CHARS", "420"))
LANG_PREFS = tuple(
    x.strip() for x in os.environ.get("AVATAR_YOUTUBE_LANGS", "en,en-US,en-GB,ar").split(",")
    if x.strip()
)


class YouTubeTranscriptError(RuntimeError):
    pass


def fetch_youtube_transcript(url, max_chars=MAX_TRANSCRIPT_CHARS,
                             start_seconds=None, end_seconds=None,
                             status_callback=None):
    """Return (title, transcript_text) for a YouTube video using captions.

    This intentionally re-speaks captions through the avatar's current TTS voice.
    It does not clone the voice from the source video.
    """
    cached = get_cached_transcript(url)
    if cached is not None:
        _status(status_callback, "found captions in local db/cache")
        text = _parse_cached_caption(cached, start_seconds, end_seconds)
        text = clean_transcript(text)
        if not text:
            raise YouTubeTranscriptError("cached captions were empty for this range")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        return cached["title"], text

    _status(status_callback, "new link - downloading captions to local db")
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeTranscriptError(f"yt-dlp is not installed ({exc})")

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        info = extract_info_with_retries(
            yt_dlp, url, opts, download=False,
            status_callback=status_callback, status_prefix="captions")
    except Exception as exc:
        raise YouTubeTranscriptError(f"could not read YouTube video ({exc})")

    title = (info or {}).get("title") or "YouTube video"
    duration = float((info or {}).get("duration") or 0.0)
    track = _select_caption_track(info or {})
    if track is None:
        raise YouTubeTranscriptError("no captions or auto-captions found for this video")

    raw = _download_text(track["url"])
    ext = (track.get("ext") or "").lower()
    save_transcript(url, title, duration, ext, raw)
    _status(status_callback, "saved captions in local db/cache")
    text = _parse_json3(raw, start_seconds, end_seconds) if ext == "json3" \
        else _parse_vtt(raw, start_seconds, end_seconds)
    text = clean_transcript(text)
    if not text:
        raise YouTubeTranscriptError("captions were empty after cleanup")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return title, text


def _parse_cached_caption(cached, start_seconds=None, end_seconds=None):
    raw = cached.get("raw") or ""
    ext = (cached.get("ext") or "").lower()
    return _parse_json3(raw, start_seconds, end_seconds) if ext == "json3" \
        else _parse_vtt(raw, start_seconds, end_seconds)


def _status(callback, msg):
    if callback is not None:
        try:
            callback(msg)
        except Exception:
            pass


def chunk_for_speech(text, chunk_chars=CHUNK_CHARS):
    """Split transcript into TTS-friendly chunks without requiring an LLM."""
    text = clean_transcript(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?؟])\s+", text)
    chunks = []
    cur = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > chunk_chars:
            for piece in _hard_wrap(part, chunk_chars):
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.append(piece)
            continue
        if cur and len(cur) + 1 + len(part) > chunk_chars:
            chunks.append(cur)
            cur = part
        else:
            cur = part if not cur else cur + " " + part
    if cur:
        chunks.append(cur)
    return chunks


def clean_transcript(text):
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^)]*(music|applause|laughter|silence)[^)]*\)", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _select_caption_track(info):
    captions = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for source in (captions, auto):
        track = _pick_from_source(source)
        if track is not None:
            return track
    return None


def _pick_from_source(source):
    if not source:
        return None
    keys = list(source.keys())
    ordered = []
    for pref in LANG_PREFS:
        ordered.extend([k for k in keys if k == pref or k.lower().startswith(pref.lower() + "-")])
    ordered.extend([k for k in keys if k not in ordered])
    for key in ordered:
        formats = source.get(key) or []
        for ext in ("json3", "vtt", "srv3", "ttml"):
            for fmt in formats:
                if (fmt.get("ext") or "").lower() == ext and fmt.get("url"):
                    return fmt
    return None


def _download_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_json3(raw, start_seconds=None, end_seconds=None):
    data = json.loads(raw)
    lines = []
    for ev in data.get("events", []):
        t = float(ev.get("tStartMs", 0) or 0) / 1000.0
        if not _inside_range(t, start_seconds, end_seconds):
            continue
        segs = ev.get("segs") or []
        line = "".join(seg.get("utf8", "") for seg in segs)
        line = clean_transcript(line)
        if line:
            lines.append(line)
    return _dedupe_lines(lines)


def _parse_vtt(raw, start_seconds=None, end_seconds=None):
    lines = []
    cur_time = None
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if "-->" in s:
            cur_time = _parse_vtt_time(s.split("-->", 1)[0].strip())
            continue
        if re.match(r"^\d+$", s):
            continue
        if not _inside_range(cur_time, start_seconds, end_seconds):
            continue
        s = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", s)
        s = clean_transcript(s)
        if s:
            lines.append(s)
    return _dedupe_lines(lines)


def _inside_range(t, start_seconds=None, end_seconds=None):
    if t is None:
        return True
    if start_seconds is not None and t < float(start_seconds):
        return False
    if end_seconds is not None and t >= float(end_seconds):
        return False
    return True


def _parse_vtt_time(s):
    try:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        return None
    return None


def _dedupe_lines(lines):
    out = []
    prev = ""
    for line in lines:
        if line != prev:
            out.append(line)
        prev = line
    return " ".join(out)


def _hard_wrap(text, chunk_chars):
    words = text.split()
    cur = []
    n = 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur and n + add > chunk_chars:
            yield " ".join(cur)
            cur = [word]
            n = len(word)
        else:
            cur.append(word)
            n += add
    if cur:
        yield " ".join(cur)
