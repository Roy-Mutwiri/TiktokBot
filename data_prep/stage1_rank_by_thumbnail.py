# =============================================================================
# data_prep/stage1_rank_by_thumbnail.py
# -----------------------------------------------------------------------------
# STAGE 1 of the staged face harvester (cheap -> expensive, stop early).
#
# Goal: rank ALL ~2700 channel videos by how likely they contain a clear,
# frontal face -- WITHOUT downloading a single video. We only pull the flat
# metadata list (yt-dlp) and each video's tiny thumbnail, run MediaPipe face
# detection on the thumbnail, and combine that with weak title hints.
#
# Output: data_prep/candidates_ranked.json (all videos, best face-likelihood
# first). The top TAKE_TOP become the candidate shortlist for stage 2.
#
# Nothing here downloads full video. Thumbnails + metadata only.
# =============================================================================

import os
import sys
import json
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

try:
    import yt_dlp
except ImportError:
    print("[STAGE1] yt-dlp not installed. Run:  pip install yt-dlp")
    raise

try:
    import mediapipe as mp
except ImportError:
    print("[STAGE1] mediapipe not installed. Run:  pip install mediapipe")
    raise

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
THUMB_DIR = os.path.join(DATA_PREP_DIR, "thumbnails")
CACHE_DIR = os.path.join(DATA_PREP_DIR, "cache")
THUMB_SCORE_CACHE = os.path.join(CACHE_DIR, "thumb_scores.json")
RANKED_OUT = os.path.join(DATA_PREP_DIR, "candidates_ranked.json")

CHANNEL_URL = "https://www.youtube.com/@ghaithabohlal/videos"

TAKE_TOP = 150                 # shortlist size handed to stage 2
DETECT_CONFIDENCE = 0.3        # thumbnails are small + compressed; be lenient
AREA_FULL_FRAC = 0.10          # bbox this fraction of thumbnail -> full area score
DOWNLOAD_DELAY = 0.05          # polite tiny delay between thumbnail fetches

# Weak title signals. Trading channels mix talking-head ("live", "analysis")
# with pure chart/screen-share uploads ("gold price update"). Treat as a nudge
# only -- the thumbnail face detector is the real signal.
TITLE_POSITIVE = [
    "live", "analysis", "q&a", "qa", "interview", "vlog", "talk", "webinar",
    "session", "lesson", "course", "explain", "story", "podcast", "review",
    "تحليل", "مباشر", "بث", "لقاء",          # ar: analysis / live / stream / meeting
]
TITLE_NEGATIVE = [
    "price update", "gold price", "signal", "chart", "forecast", "target",
    "buy now", "sell now", "scalp", "setup", "levels", "update", "xau",
    "تحديث", "اشارة", "سعر",                  # ar: update / signal / price
]


# -----------------------------------------------------------------------------
# METADATA: flat channel listing (no downloads)
# -----------------------------------------------------------------------------
def fetch_video_list():
    """Return a list of {id, title, url, duration, view_count} for every video.

    Uses yt-dlp flat extraction -- one network call for the whole channel, no
    per-video page loads and definitely no media download.
    """
    ydl_opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    print(f"[STAGE1] Fetching flat video list for {CHANNEL_URL} ...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)

    entries = []
    _collect_entries(info, entries)

    videos = []
    seen = set()
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        videos.append({
            "id": vid,
            "title": e.get("title") or "",
            "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
            "duration": e.get("duration"),
            "view_count": e.get("view_count"),
        })
    print(f"[STAGE1] Channel lists {len(videos)} video(s).")
    return videos


def _collect_entries(info, out):
    """Recursively flatten nested playlist/tab structures into a flat list."""
    if not info:
        return
    entries = info.get("entries")
    if entries is None:
        out.append(info)
        return
    for e in entries:
        if e and e.get("entries") is not None:
            _collect_entries(e, out)
        elif e:
            out.append(e)


# -----------------------------------------------------------------------------
# THUMBNAILS
# -----------------------------------------------------------------------------
def thumbnail_url(video_id):
    """High-quality default thumbnail URL, derivable from the id alone."""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def download_thumbnail(video_id):
    """Download one thumbnail to THUMB_DIR (cached). Returns local path or None."""
    os.makedirs(THUMB_DIR, exist_ok=True)
    path = os.path.join(THUMB_DIR, f"{video_id}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(
        thumbnail_url(video_id), headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if not data:
            return None
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


# -----------------------------------------------------------------------------
# FACE SCORING ON A THUMBNAIL
# -----------------------------------------------------------------------------
def score_thumbnail(img_path, detector):
    """Score one thumbnail for face likelihood in [0, 1].

    Combines detector confidence, bbox area, and a frontal-ness proxy built
    from the 6 detection keypoints (eyes / nose symmetry). Returns
    (face_score, detail_dict). 0.0 when no face is found.
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0, {"face": False}
    h, w = img.shape[:2]
    res = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return 0.0, {"face": False}

    # Best detection by score.
    best = max(res.detections,
               key=lambda d: d.score[0] if d.score else 0.0)
    conf = float(best.score[0]) if best.score else 0.0
    box = best.location_data.relative_bounding_box
    area = max(0.0, float(box.width)) * max(0.0, float(box.height))
    area_score = min(area / AREA_FULL_FRAC, 1.0)

    frontal = _frontal_score(best, w, h)

    face_score = 0.40 * conf + 0.40 * area_score + 0.20 * frontal
    detail = {
        "face": True,
        "conf": round(conf, 3),
        "bbox_area_frac": round(area, 4),
        "frontal": round(frontal, 3),
        "n_faces": len(res.detections),
    }
    return float(face_score), detail


def _frontal_score(detection, w, h):
    """Frontal-ness in [0, 1] from MediaPipe's 6 keypoints.

    Keypoint order: right eye, left eye, nose tip, mouth, right ear, left ear.
    A frontal face has the nose roughly midway between the eyes and the eyes at
    a similar height. Turned heads push the nose toward one eye.
    """
    kps = detection.location_data.relative_keypoints
    if not kps or len(kps) < 4:
        return 0.5
    r_eye, l_eye, nose = kps[0], kps[1], kps[2]
    eye_dx = abs(l_eye.x - r_eye.x)
    if eye_dx < 1e-4:
        return 0.0
    eye_mid_x = (l_eye.x + r_eye.x) / 2.0
    # Horizontal yaw proxy: 0 frontal, ~0.5+ strongly turned.
    yaw_proxy = abs(nose.x - eye_mid_x) / eye_dx
    yaw_term = max(0.0, 1.0 - yaw_proxy / 0.5)
    # Eye-height symmetry (roll/tilt) proxy.
    eye_dy = abs(l_eye.y - r_eye.y) / eye_dx
    roll_term = max(0.0, 1.0 - eye_dy / 0.5)
    return 0.7 * yaw_term + 0.3 * roll_term


def title_bonus(title):
    """Weak multiplicative nudge in [-0.3, +0.3] from title keywords."""
    t = title.lower()
    bonus = 0.0
    for kw in TITLE_POSITIVE:
        if kw in t:
            bonus += 0.10
    for kw in TITLE_NEGATIVE:
        if kw in t:
            bonus -= 0.10
    return max(-0.30, min(0.30, bonus))


# -----------------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------------
def load_cache():
    if os.path.exists(THUMB_SCORE_CACHE):
        try:
            with open(THUMB_SCORE_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache):
    with open(THUMB_SCORE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# DRIVER
# -----------------------------------------------------------------------------
def run():
    os.makedirs(THUMB_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    videos = fetch_video_list()
    if not videos:
        print("[STAGE1] No videos found. Check the channel URL / yt-dlp version "
              "(`pip install -U yt-dlp`).")
        return []

    cache = load_cache()
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=DETECT_CONFIDENCE)

    ranked = []
    face_count = 0
    try:
        for i, v in enumerate(videos, start=1):
            vid = v["id"]
            cached = cache.get(vid)
            if cached is not None:
                face_score = cached["face_score"]
                detail = cached["detail"]
            else:
                path = download_thumbnail(vid)
                if path is None:
                    face_score, detail = 0.0, {"face": False, "thumb": "missing"}
                else:
                    face_score, detail = score_thumbnail(path, detector)
                cache[vid] = {"face_score": face_score, "detail": detail}
                time.sleep(DOWNLOAD_DELAY)

            tb = title_bonus(v["title"])
            likelihood = face_score * (1.0 + tb)
            if detail.get("face"):
                face_count += 1

            ranked.append({
                **v,
                "face_score": round(face_score, 4),
                "title_bonus": round(tb, 3),
                "face_likelihood": round(likelihood, 4),
                "detail": detail,
            })

            if i % 100 == 0 or i == len(videos):
                print(f"[STAGE1] scored {i}/{len(videos)} thumbnails "
                      f"({face_count} with a face so far)")
                save_cache(cache)
    finally:
        detector.close()
        save_cache(cache)

    ranked.sort(key=lambda r: r["face_likelihood"], reverse=True)
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
        r["shortlisted"] = rank <= TAKE_TOP

    payload = {
        "channel": CHANNEL_URL,
        "total_videos": len(ranked),
        "faces_in_thumbnail": face_count,
        "take_top": TAKE_TOP,
        "videos": ranked,
    }
    with open(RANKED_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[STAGE1] Of {len(ranked)} videos, ~{face_count} show a clear "
          f"face in the thumbnail.")
    print(f"[STAGE1] Top candidate: {ranked[0]['title'][:60]!r} "
          f"(likelihood {ranked[0]['face_likelihood']})")
    print(f"[STAGE1] Wrote ranked list -> {RANKED_OUT}")
    print(f"[STAGE1] Shortlisted top {min(TAKE_TOP, len(ranked))} for stage 2.")
    return ranked


if __name__ == "__main__":
    run()
