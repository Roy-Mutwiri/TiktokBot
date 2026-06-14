import hashlib
import json
import os
import re
import sqlite3
import time
from urllib.parse import parse_qs, urlparse


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "youtube_cache")
DB_PATH = os.path.join(CACHE_DIR, "youtube_cache.sqlite")


def video_id_from_url(url):
    url = (url or "").strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        vid = parsed.path.strip("/").split("/", 1)[0]
        return _clean_video_id(vid)
    qs = parse_qs(parsed.query or "")
    if "v" in qs and qs["v"]:
        return _clean_video_id(qs["v"][0])
    m = re.search(r"(?:shorts|embed|live)/([A-Za-z0-9_-]{6,})", parsed.path)
    if m:
        return _clean_video_id(m.group(1))
    return hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:16]


def get_cached_transcript(url):
    vid = video_id_from_url(url)
    row = _fetch_one(
        "select title, duration, ext, raw from transcripts where video_id=?",
        (vid,),
    )
    if row is None:
        return None
    _touch("transcripts", vid)
    return {
        "video_id": vid,
        "title": row[0] or "YouTube video",
        "duration": float(row[1] or 0.0),
        "ext": row[2] or "",
        "raw": row[3] or "",
    }


def save_transcript(url, title, duration, ext, raw):
    vid = video_id_from_url(url)
    _execute(
        """insert into transcripts(video_id, url, title, duration, ext, raw, updated_at, last_used_at)
           values(?,?,?,?,?,?,?,?)
           on conflict(video_id) do update set
             url=excluded.url,
             title=excluded.title,
             duration=excluded.duration,
             ext=excluded.ext,
             raw=excluded.raw,
             updated_at=excluded.updated_at,
             last_used_at=excluded.last_used_at""",
        (vid, url, title, float(duration or 0.0), ext or "", raw or "", time.time(), time.time()),
    )
    return vid


def get_cached_audio(url):
    vid = video_id_from_url(url)
    row = _fetch_one(
        "select title, duration, audio_path from videos where video_id=?",
        (vid,),
    )
    if row is None:
        return None
    path = row[2] or ""
    if not path or not os.path.exists(path):
        return None
    _touch("videos", vid)
    return {
        "video_id": vid,
        "title": row[0] or "YouTube audio",
        "duration": float(row[1] or 0.0),
        "audio_path": path,
    }


def save_audio(url, title, duration, audio_path):
    vid = video_id_from_url(url)
    _execute(
        """insert into videos(video_id, url, title, duration, audio_path, updated_at, last_used_at)
           values(?,?,?,?,?,?,?)
           on conflict(video_id) do update set
             url=excluded.url,
             title=excluded.title,
             duration=excluded.duration,
             audio_path=excluded.audio_path,
             updated_at=excluded.updated_at,
             last_used_at=excluded.last_used_at""",
        (vid, url, title, float(duration or 0.0), audio_path, time.time(), time.time()),
    )
    return vid


def audio_dir(url):
    path = os.path.join(CACHE_DIR, video_id_from_url(url))
    os.makedirs(path, exist_ok=True)
    return path


def cache_summary(url):
    return {
        "video_id": video_id_from_url(url),
        "has_transcript": get_cached_transcript(url) is not None,
        "has_audio": get_cached_audio(url) is not None,
    }


def _clean_video_id(value):
    value = (value or "").strip()
    m = re.match(r"^[A-Za-z0-9_-]{6,}$", value)
    return value if m else hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _connect():
    os.makedirs(CACHE_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.execute(
        """create table if not exists transcripts(
             video_id text primary key,
             url text,
             title text,
             duration real,
             ext text,
             raw text,
             updated_at real,
             last_used_at real
           )"""
    )
    con.execute(
        """create table if not exists videos(
             video_id text primary key,
             url text,
             title text,
             duration real,
             audio_path text,
             updated_at real,
             last_used_at real
           )"""
    )
    return con


def _fetch_one(sql, args):
    con = _connect()
    try:
        return con.execute(sql, args).fetchone()
    finally:
        con.close()


def _execute(sql, args):
    con = _connect()
    try:
        con.execute(sql, args)
        con.commit()
    finally:
        con.close()


def _touch(table, video_id):
    if table not in ("videos", "transcripts"):
        return
    con = _connect()
    try:
        con.execute(
            f"update {table} set last_used_at=? where video_id=?",
            (time.time(), video_id),
        )
        con.commit()
    finally:
        con.close()
