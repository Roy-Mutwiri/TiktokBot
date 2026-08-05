"""Keep data/youtube_cache inside a disk budget without lowering quality.

The cache used to grow forever: every YouTube link left a full-length video
preview, a bestaudio download and an isolated vocal stem behind, plus the
half-finished ``.part``/``.bad-``/``.old-`` files a retried download drops. This
module reclaims that space in two passes:

* ``purge_junk`` removes files that can never be played again.
* ``enforce_budget`` evicts whole video folders, least-recently-used first,
  until the cache fits ``AVATAR_YOUTUBE_CACHE_GB``.

Eviction only ever removes re-downloadable copies, so picture and sound quality
are untouched; a evicted link simply downloads again the next time it is used.
"""

import os
import re
import shutil
import sqlite3
import time

from youtube_cache import CACHE_DIR, DB_PATH, video_id_from_url


DEFAULT_BUDGET_GB = 12.0
JUNK_SUFFIX_RE = re.compile(r"\.(part|tmp|ytdl|temp)$", re.IGNORECASE)
JUNK_MARKER_RE = re.compile(r"\.(bad|old)-", re.IGNORECASE)
# The scene only ever looks up "preview-hd.*" and "preview.*", so a file left
# over from the old low-resolution naming can never be played again.
DEAD_PREFIX_RE = re.compile(r"^preview-low\.", re.IGNORECASE)
JUNK_DIR_NAMES = ("demucs_stems",)


def budget_bytes():
    """Cache size cap in bytes; 0 disables eviction entirely."""
    raw = os.environ.get("AVATAR_YOUTUBE_CACHE_GB", "")
    try:
        gigabytes = float(raw) if str(raw).strip() else DEFAULT_BUDGET_GB
    except (TypeError, ValueError):
        gigabytes = DEFAULT_BUDGET_GB
    if gigabytes <= 0:
        return 0
    return int(gigabytes * 1024 ** 3)


def human_bytes(count):
    value = float(max(0, int(count or 0)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def is_junk_file(path):
    name = os.path.basename(path or "")
    if not name:
        return False
    return bool(
        JUNK_SUFFIX_RE.search(name)
        or JUNK_MARKER_RE.search(name)
        or DEAD_PREFIX_RE.match(name))


def purge_junk(cache_dir=None, status_callback=None, dry_run=False):
    """Delete unplayable leftovers. Returns (file_count, freed_bytes)."""
    cache_dir = cache_dir or CACHE_DIR
    if not os.path.isdir(cache_dir):
        return 0, 0
    files = 0
    freed = 0
    for root, dirnames, filenames in os.walk(cache_dir, topdown=True):
        for junk_dir in [d for d in dirnames if d in JUNK_DIR_NAMES]:
            dirnames.remove(junk_dir)
            path = os.path.join(root, junk_dir)
            size = _tree_bytes(path)
            if dry_run or _remove_tree(path):
                files += 1
                freed += size
        for name in filenames:
            path = os.path.join(root, name)
            if not is_junk_file(path):
                continue
            size = _file_bytes(path)
            if dry_run or _remove_file(path):
                files += 1
                freed += size
    if files and status_callback is not None:
        _status(
            status_callback,
            f"cache cleanup removed {files} dead file(s), {human_bytes(freed)} freed")
    return files, freed


def cache_entries(cache_dir=None):
    """One record per cached video folder, newest use first."""
    cache_dir = cache_dir or CACHE_DIR
    if not os.path.isdir(cache_dir):
        return []
    used_at = _db_last_used()
    entries = []
    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        if not os.path.isdir(path):
            continue
        size, newest = _tree_stats(path)
        if size <= 0:
            continue
        entries.append({
            "video_id": name,
            "path": path,
            "bytes": size,
            "used_at": max(newest, float(used_at.get(name, 0.0))),
        })
    entries.sort(key=lambda item: item["used_at"], reverse=True)
    return entries


def cache_bytes(cache_dir=None):
    return sum(entry["bytes"] for entry in cache_entries(cache_dir))


def enforce_budget(max_bytes=None, keep_ids=(), cache_dir=None,
                   status_callback=None, dry_run=False):
    """Evict least-recently-used video folders until the cache fits the cap."""
    cache_dir = cache_dir or CACHE_DIR
    limit = budget_bytes() if max_bytes is None else int(max_bytes)
    if limit <= 0:
        return 0, 0, []
    protected = {str(item) for item in (keep_ids or ()) if item}
    entries = cache_entries(cache_dir)
    total = sum(entry["bytes"] for entry in entries)
    if total <= limit:
        return 0, 0, []
    removed = 0
    freed = 0
    names = []
    # entries are newest-first, so evict from the tail (oldest) upward.
    for entry in reversed(entries):
        if total - freed <= limit:
            break
        if entry["video_id"] in protected:
            continue
        if dry_run:
            removed += 1
            freed += entry["bytes"]
            names.append(entry["video_id"])
            continue
        _remove_tree(entry["path"])
        # A file held open by a running player leaves part of the folder behind.
        # Whatever went is still reclaimed, and the database row has to go too so
        # the gap is treated as a cache miss instead of a broken hit.
        leftover = _tree_bytes(entry["path"]) if os.path.isdir(entry["path"]) else 0
        gained = max(0, entry["bytes"] - leftover)
        if gained <= 0:
            continue
        _forget_video(entry["video_id"])
        removed += 1
        freed += gained
        names.append(entry["video_id"])
    if removed and status_callback is not None:
        _status(
            status_callback,
            f"cache over {human_bytes(limit)}: removed {removed} old video(s), "
            f"{human_bytes(freed)} freed")
    return removed, freed, names


def sweep(keep_urls=(), cache_dir=None, status_callback=None, dry_run=False):
    """Full maintenance pass. Safe to call on startup and after downloads."""
    cache_dir = cache_dir or CACHE_DIR
    before = _tree_bytes(cache_dir) if os.path.isdir(cache_dir) else 0
    junk_files, junk_freed = purge_junk(
        cache_dir, status_callback=status_callback, dry_run=dry_run)
    keep_ids = [video_id_from_url(url) for url in (keep_urls or ()) if url]
    removed, evict_freed, names = enforce_budget(
        keep_ids=keep_ids, cache_dir=cache_dir,
        status_callback=status_callback, dry_run=dry_run)
    return {
        "junk_files": junk_files,
        "junk_bytes": junk_freed,
        "evicted": removed,
        "evicted_bytes": evict_freed,
        "evicted_ids": names,
        "freed_bytes": junk_freed + evict_freed,
        "before_bytes": before,
        "after_bytes": max(0, before - junk_freed - evict_freed),
        "limit_bytes": budget_bytes(),
    }


def summary_line(result):
    result = result or {}
    return (
        f"youtube cache {human_bytes(result.get('after_bytes', 0))}"
        f" / {human_bytes(result.get('limit_bytes', 0))} cap"
        f" (freed {human_bytes(result.get('freed_bytes', 0))}:"
        f" {result.get('junk_files', 0)} dead file(s),"
        f" {result.get('evicted', 0)} old video(s))"
    )


def _tree_stats(path):
    size = 0
    newest = 0.0
    for root, _dirs, filenames in os.walk(path):
        for name in filenames:
            full = os.path.join(root, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            size += stat.st_size
            newest = max(newest, stat.st_mtime, stat.st_atime)
    return size, newest


def _tree_bytes(path):
    return _tree_stats(path)[0]


def _file_bytes(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _remove_file(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _remove_tree(path):
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        return False


def _db_last_used():
    """video_id -> newest last_used_at across the cache tables."""
    if not os.path.exists(DB_PATH):
        return {}
    used = {}
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
    except sqlite3.Error:
        return {}
    try:
        for table in ("videos", "transcripts"):
            try:
                rows = con.execute(
                    f"select video_id, last_used_at, updated_at from {table}"
                ).fetchall()
            except sqlite3.Error:
                continue
            for video_id, last_used, updated in rows:
                stamp = float(last_used or updated or 0.0)
                if stamp > used.get(video_id, 0.0):
                    used[video_id] = stamp
    finally:
        con.close()
    return used


def _forget_video(video_id):
    if not os.path.exists(DB_PATH):
        return
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
    except sqlite3.Error:
        return
    try:
        for table in ("videos", "transcripts"):
            try:
                con.execute(f"delete from {table} where video_id=?", (video_id,))
            except sqlite3.Error:
                continue
        con.commit()
    finally:
        con.close()


def _status(callback, message):
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube cache maintenance")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be deleted, delete nothing")
    parser.add_argument("--gb", type=float, default=None,
                        help="cache size cap in GB for this run")
    options = parser.parse_args()
    if options.gb is not None:
        os.environ["AVATAR_YOUTUBE_CACHE_GB"] = str(options.gb)
    started = time.monotonic()
    outcome = sweep(status_callback=print, dry_run=options.dry_run)
    print(summary_line(outcome))
    if outcome["evicted_ids"]:
        print("evicted:", ", ".join(outcome["evicted_ids"]))
    print(f"done in {time.monotonic() - started:.1f}s"
          + (" (dry run, nothing deleted)" if options.dry_run else ""))
