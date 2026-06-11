# =============================================================================
# data_prep/stage2_scan_candidates.py
# -----------------------------------------------------------------------------
# STAGE 2 of the staged face harvester.
#
# Input: the top candidates from stage 1 (data_prep/candidates_ranked.json).
# For each candidate, in ranked order:
#   - download at 720p (enough to scan; cached in data_prep/scan_videos/)
#   - sample frames at 1 fps
#   - score each sampled frame for face presence/quality
#       (bbox area > 12% of frame, yaw within +/-30 deg, sharpness, brightness)
#   - group good frames into SEGMENTS (>=2s, merge gaps <1s)
#   - score each segment: avg face quality x duration
#   - save a representative face crop per segment for stage 3 embeddings
#
# Output: data_prep/segments_scored.json  (segments ranked, NOT cut yet).
# Per-video scan results are cached so re-runs are cheap. We stop early once we
# have gathered enough good footage (TARGET_MINUTES x SCAN_MULTIPLIER), which
# leaves stage 3 a comfortable buffer for its consistency filtering.
# =============================================================================

import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

try:
    import yt_dlp
except ImportError:
    print("[STAGE2] yt-dlp not installed. Run:  pip install yt-dlp")
    raise

try:
    import mediapipe as mp
except ImportError:
    print("[STAGE2] mediapipe not installed. Run:  pip install mediapipe")
    raise

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
RANKED_IN = os.path.join(DATA_PREP_DIR, "candidates_ranked.json")
SCAN_VIDEO_DIR = os.path.join(DATA_PREP_DIR, "scan_videos")
SEG_REP_DIR = os.path.join(DATA_PREP_DIR, "seg_reps")
CACHE_DIR = os.path.join(DATA_PREP_DIR, "cache")
SEGMENTS_OUT = os.path.join(DATA_PREP_DIR, "segments_scored.json")
SCAN_STATE_OUT = os.path.join(DATA_PREP_DIR, "scan_state.json")

TARGET_MINUTES = 25.0          # final dataset target (stage 3 enforces it)
SCAN_MULTIPLIER = 2.0          # gather ~2x so stage 3 can drop inconsistent looks
SCAN_TARGET_MINUTES = TARGET_MINUTES * SCAN_MULTIPLIER

SAMPLE_FPS = 1.0               # frames sampled per second of video
MIN_FACE_AREA = 0.03           # bbox must be > 3% of the frame. This channel is
                               # chart screen-share with a SMALL webcam inset in
                               # the corner -- real faces sit around 3-4% (12%
                               # only fits full-screen close-ups, which are rare).
MAX_YAW_DEG = 30.0             # accept frontal-ish heads within +/-30 deg
MIN_SHARPNESS = 20.0           # Laplacian variance on the face crop (small insets
                               # upscale soft; 40 was tuned for full-screen faces)
BRIGHTNESS_RANGE = (40, 225)   # mean gray on the face crop
DETECT_CONFIDENCE = 0.5

MIN_SEG_SECONDS = 2.0          # a segment must span at least 2s
MERGE_GAP_SECONDS = 1.0        # bridge dropouts of up to 1s of missing footage

DOWNLOAD_FORMAT = "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]/best"


def ffmpeg_exe():
    """Resolve an ffmpeg binary: PATH first, else the imageio_ffmpeg bundle.

    yt-dlp needs ffmpeg to merge separate video+audio streams; on this box
    ffmpeg is not on PATH, so we fall back to the binary imageio ships.
    """
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# DOWNLOAD (720p, cached)
# -----------------------------------------------------------------------------
def download_720p(video):
    """Download one candidate at <=720p into SCAN_VIDEO_DIR. Returns path|None."""
    vid = video["id"]
    existing = _find_local(vid)
    if existing:
        return existing

    outtmpl = os.path.join(SCAN_VIDEO_DIR, f"{vid}.%(ext)s")
    ydl_opts = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    ff = ffmpeg_exe()
    if ff:
        ydl_opts["ffmpeg_location"] = ff
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])
    except Exception as exc:
        print(f"[STAGE2] download failed for {vid}: {exc}")
        return None
    return _find_local(vid)


def _find_local(video_id):
    """Return the already-downloaded scan file for a video id, if any."""
    if not os.path.isdir(SCAN_VIDEO_DIR):
        return None
    for f in os.listdir(SCAN_VIDEO_DIR):
        name, ext = os.path.splitext(f)
        if name == video_id and ext.lower() in (".mp4", ".mkv", ".webm"):
            p = os.path.join(SCAN_VIDEO_DIR, f)
            if os.path.getsize(p) > 0:
                return p
    return None


# -----------------------------------------------------------------------------
# FRAME QUALITY
# -----------------------------------------------------------------------------
def score_frame(frame, detector):
    """Quality score in [0, 1] for one frame, or 0.0 if it fails any gate.

    Gates: exactly-ish one usable face, bbox area > MIN_FACE_AREA, |yaw| <=
    MAX_YAW_DEG, sharpness >= MIN_SHARPNESS, brightness in range. Also returns
    the face crop (for the representative image) and the measured yaw.
    """
    h, w = frame.shape[:2]
    res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return 0.0, None, None
    best = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    box = best.location_data.relative_bounding_box
    area = max(0.0, float(box.width)) * max(0.0, float(box.height))
    if area < MIN_FACE_AREA:
        return 0.0, None, None

    yaw = _yaw_degrees(best)
    if abs(yaw) > MAX_YAW_DEG:
        return 0.0, None, None

    crop = _crop_face(frame, box, w, h)
    if crop is None or crop.size == 0:
        return 0.0, None, None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharp < MIN_SHARPNESS:
        return 0.0, None, None
    bright = float(np.mean(gray))
    if bright < BRIGHTNESS_RANGE[0] or bright > BRIGHTNESS_RANGE[1]:
        return 0.0, None, None

    conf = float(best.score[0]) if best.score else 0.5
    area_term = min(area / 0.08, 1.0)              # 8% bbox -> full marks (insets
                                                   # are small; don't zero them out)
    yaw_term = max(0.0, 1.0 - abs(yaw) / MAX_YAW_DEG)
    sharp_term = min(sharp / 200.0, 1.0)
    quality = 0.30 * conf + 0.25 * area_term + 0.25 * yaw_term + 0.20 * sharp_term
    metrics = {"area": area, "yaw": yaw, "sharp": sharp, "bright": bright}
    return float(quality), crop, metrics


def _yaw_degrees(detection):
    """Approximate yaw in degrees from the eye/nose keypoints."""
    kps = detection.location_data.relative_keypoints
    if not kps or len(kps) < 3:
        return 0.0
    r_eye, l_eye, nose = kps[0], kps[1], kps[2]
    eye_dx = abs(l_eye.x - r_eye.x)
    if eye_dx < 1e-4:
        return 90.0
    eye_mid_x = (l_eye.x + r_eye.x) / 2.0
    yaw_proxy = (nose.x - eye_mid_x) / eye_dx      # 0 frontal, +/-0.5 turned
    return float(yaw_proxy * 90.0)                 # rough proxy -> degrees


def _crop_face(frame, box, w, h):
    """Crop the face bbox with a small margin; clamped to frame bounds."""
    x1 = int(max(0, (box.xmin - 0.05) * w))
    y1 = int(max(0, (box.ymin - 0.05) * h))
    x2 = int(min(w, (box.xmin + box.width + 0.05) * w))
    y2 = int(min(h, (box.ymin + box.height + 0.05) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# -----------------------------------------------------------------------------
# SCAN ONE VIDEO -> SEGMENTS
# -----------------------------------------------------------------------------
def scan_video(video, detector):
    """Sample at 1fps, score frames, group into segments. Returns segment list.

    Each segment: {start, end, duration, quality, score, rep_frame, mean_*}.
    A representative (best-quality) face crop is written to SEG_REP_DIR for
    stage 3's appearance embedding.
    """
    vid = video["id"]
    path = download_720p(video)
    if path is None:
        return None                                # signal: unavailable, skip

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / SAMPLE_FPS)))

    samples = []        # (t_seconds, quality, metrics, crop)
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        quality, crop, metrics = score_frame(frame, detector)
        if quality > 0.0:
            samples.append((t, quality, metrics, crop))
        idx += step
        if total and idx >= total:
            break
    cap.release()

    return _group_segments(vid, samples)


def _group_segments(video_id, samples):
    """Group good 1fps samples into segments, bridging short gaps."""
    if not samples:
        return []
    step_s = 1.0 / SAMPLE_FPS
    segments = []
    run = [samples[0]]
    for s in samples[1:]:
        gap_footage = (s[0] - run[-1][0]) - step_s
        if gap_footage <= MERGE_GAP_SECONDS:
            run.append(s)
        else:
            seg = _finalize_segment(video_id, run, step_s)
            if seg:
                segments.append(seg)
            run = [s]
    seg = _finalize_segment(video_id, run, step_s)
    if seg:
        segments.append(seg)
    return segments


def _finalize_segment(video_id, run, step_s):
    """Build one segment dict from a run of good samples (or None if too short)."""
    start = run[0][0]
    end = run[-1][0] + step_s                      # include the last sampled second
    duration = end - start
    if duration < MIN_SEG_SECONDS:
        return None
    qualities = [r[1] for r in run]
    avg_q = float(np.mean(qualities))
    metrics = [r[2] for r in run]

    # Representative = the highest-quality sampled frame in this run.
    best = max(run, key=lambda r: r[1])
    rep_name = f"{video_id}_{int(round(start))}_{int(round(end))}.jpg"
    rep_path = os.path.join(SEG_REP_DIR, rep_name)
    if best[3] is not None:
        cv2.imwrite(rep_path, best[3])

    return {
        "video_id": video_id,
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(duration, 2),
        "quality": round(avg_q, 4),
        "score": round(avg_q * duration, 4),
        "rep_frame": rep_path,
        "mean_area": round(float(np.mean([m["area"] for m in metrics])), 4),
        "mean_yaw": round(float(np.mean([m["yaw"] for m in metrics])), 2),
        "mean_sharp": round(float(np.mean([m["sharp"] for m in metrics])), 1),
        "mean_bright": round(float(np.mean([m["bright"] for m in metrics])), 1),
    }


# -----------------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------------
def _cache_path(video_id):
    return os.path.join(CACHE_DIR, f"scan_{video_id}.json")


def load_scan_cache(video_id):
    p = _cache_path(video_id)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_scan_cache(video_id, segments):
    with open(_cache_path(video_id), "w", encoding="utf-8") as f:
        json.dump({"video_id": video_id, "segments": segments},
                  f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# DRIVER
# -----------------------------------------------------------------------------
def run():
    if not os.path.exists(RANKED_IN):
        print(f"[STAGE2] {RANKED_IN} not found. Run stage 1 first.")
        return []
    with open(RANKED_IN, encoding="utf-8") as f:
        ranked = json.load(f)

    take_top = ranked.get("take_top", 150)
    candidates = [v for v in ranked["videos"] if v.get("rank", 1e9) <= take_top]
    print(f"[STAGE2] {len(candidates)} candidate(s) from stage 1; "
          f"scan target {SCAN_TARGET_MINUTES:.0f} min, then early-stop.")

    for d in (SCAN_VIDEO_DIR, SEG_REP_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)

    # model_selection=1 = full-range detector: catches the small corner-inset
    # webcam faces this channel uses (short-range misses faces under ~5% area).
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=DETECT_CONFIDENCE)

    all_segments = []
    scanned_ids = []
    good_seconds = 0.0
    stopped_early = False
    try:
        for i, video in enumerate(candidates, start=1):
            vid = video["id"]
            cached = load_scan_cache(vid)
            if cached is not None:
                segments = cached["segments"]
                print(f"[STAGE2] ({i}/{len(candidates)}) cache hit {vid} "
                      f"-> {len(segments)} segment(s)")
            else:
                print(f"[STAGE2] ({i}/{len(candidates)}) scanning {vid} "
                      f"{video['title'][:48]!r}")
                segments = scan_video(video, detector)
                if segments is None:
                    print(f"[STAGE2] {vid} unavailable/private/deleted -> skip")
                    continue
                save_scan_cache(vid, segments)

            scanned_ids.append(vid)
            for seg in segments:
                seg = dict(seg)
                seg["title"] = video["title"]
                seg["url"] = video["url"]
                all_segments.append(seg)
                good_seconds += seg["duration"]

            print(f"[STAGE2]   running total: {good_seconds/60.0:.1f} min of "
                  f"good footage across {len(scanned_ids)} video(s)")
            if good_seconds >= SCAN_TARGET_MINUTES * 60.0:
                stopped_early = True
                print(f"[STAGE2] Reached scan target "
                      f"({good_seconds/60.0:.1f} min). Stopping early.")
                break
    finally:
        detector.close()

    all_segments.sort(key=lambda s: s["score"], reverse=True)
    for i, seg in enumerate(all_segments, start=1):
        seg["seg_id"] = i

    payload = {
        "target_minutes": TARGET_MINUTES,
        "scan_target_minutes": SCAN_TARGET_MINUTES,
        "videos_scanned": len(scanned_ids),
        "total_candidates": len(candidates),
        "good_minutes": round(good_seconds / 60.0, 2),
        "stopped_early": stopped_early,
        "segments": all_segments,
    }
    with open(SEGMENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(SCAN_STATE_OUT, "w", encoding="utf-8") as f:
        json.dump({"scanned_ids": scanned_ids,
                   "total_channel_videos": ranked.get("total_videos")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n[STAGE2] {len(all_segments)} segment(s), "
          f"{good_seconds/60.0:.1f} min good footage, "
          f"from {len(scanned_ids)} scanned video(s).")
    print(f"[STAGE2] Wrote -> {SEGMENTS_OUT}")
    return all_segments


if __name__ == "__main__":
    run()
