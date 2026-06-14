"""Install a local, licensed StreamBeats playlist from official Bandcamp streams."""

import hashlib
import html
import http.client
import json
import os
import re
import time
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "music", "streambeats")
MANIFEST = os.path.join(OUTPUT_DIR, "manifest.json")
ALBUMS = ("groovy", "midnight-1", "pulse", "electric")
ALBUM_URL = "https://streambeats.bandcamp.com/album/{}"
LICENSE_URL = "https://www.streambeats.com/licensing"
USER_AGENT = "Mozilla/5.0 TiktokBot StreamBeats installer"


def fetch(url):
    error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, http.client.IncompleteRead) as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after retries: {url}") from error


def album_tracks(slug):
    page_url = ALBUM_URL.format(slug)
    page = fetch(page_url).decode("utf-8", errors="replace")
    match = re.search(r'data-tralbum="([^"]+)"', page)
    if not match:
        raise RuntimeError(f"Bandcamp metadata missing for {slug}")
    payload = json.loads(html.unescape(match.group(1)))
    album = payload["current"]["title"]
    tracks = []
    for item in payload.get("trackinfo", []):
        stream_url = (item.get("file") or {}).get("mp3-128")
        if stream_url:
            tracks.append({
                "album": album,
                "album_slug": slug,
                "title": item.get("title") or f"Track {item.get('track_num', 0)}",
                "track_num": int(item.get("track_num") or 0),
                "duration": float(item.get("duration") or 0),
                "source": page_url,
                "stream_url": stream_url,
            })
    return tracks


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._ -]+", "", value).strip()
    return re.sub(r"\s+", "_", value)[:80] or "track"


def install():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    installed = []
    hashes = set()
    for slug in ALBUMS:
        tracks = album_tracks(slug)
        print(f"[{slug}] {len(tracks)} tracks")
        for item in tracks:
            filename = (
                f"{safe_name(item['album'])}_{item['track_num']:02d}_"
                f"{safe_name(item['title'])}.mp3"
            )
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(path) and os.path.getsize(path) > 64_000:
                with open(path, "rb") as existing:
                    data = existing.read()
            else:
                try:
                    data = fetch(item["stream_url"])
                except RuntimeError as exc:
                    print(f"  skipped after retries: {filename} ({exc})")
                    continue
                with open(path, "wb") as output:
                    output.write(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                os.remove(path)
                print(f"  duplicate removed: {filename}")
                continue
            hashes.add(digest)
            installed.append({
                **item,
                "file": filename,
                "bytes": len(data),
                "sha256": digest,
                "license": LICENSE_URL,
            })
            print(f"  {len(installed):03d} {filename} ({len(data) // 1024} KiB)")

    with open(MANIFEST, "w", encoding="utf-8") as output:
        json.dump({
            "provider": "StreamBeats",
            "catalog": "https://www.streambeats.com/",
            "license": LICENSE_URL,
            "tracks": installed,
        }, output, indent=2)
    print(f"Installed {len(installed)} unique real tracks in {OUTPUT_DIR}")
    if len(installed) < 100:
        raise RuntimeError("Expected at least 100 unique tracks")


if __name__ == "__main__":
    install()
