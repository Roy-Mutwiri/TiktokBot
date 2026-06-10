# =============================================================================
# build_character.py
# -----------------------------------------------------------------------------
# Build a MULTI-ANGLE character from a video of a real person. Scans the video,
# measures each frame's head yaw with LivePortrait's own pose estimator, and
# saves the SHARPEST, best-framed face crop at each angle bucket as a reference
# view. The realtime engine then drives whichever real view matches your head
# turn — clean turning at every angle the video actually covers (real side data,
# no hallucination, no training).
#
#   python build_character.py source_vids/char.mp4
# =============================================================================

import os
import sys
import glob

sys.path.insert(0, "engines")
import numpy as np
import cv2

OUT_DIR = "character_views"
FRAME_SIZE = 512
SAMPLE_EVERY = 2                 # dense sampling (smooth profile transitions)
MAX_SAMPLES_PER_VIDEO = 800      # spread this many samples evenly across each video
CROP_PAD = 2.0                   # square crop = this * face box (portrait framing)
MIN_FACE_FRAC = 0.07             # ignore tiny faces (want sharp, large faces)
# fine yaw buckets (deg) across the FULL profile range — a dedicated turn clip
# has real data all the way to ±90, so capture it.
YAW_BUCKETS = [-85, -72, -60, -52, -44, -36, -28, -21, -14, -7, 0,
               7, 14, 21, 28, 36, 44, 52, 60, 72, 85]   # ~7deg steps = smooth
BUCKET_HALF = 5.0                # a frame falls in a bucket if within this many deg
MAX_PITCH = 22.0                 # reject strongly up/down frames for the main set
MIN_SHARPNESS = 4.0              # phone turn-clips read "soft" on Laplacian; keep them


def main(videos):
    if isinstance(videos, str):
        videos = [videos]
    videos = [v for v in videos if os.path.exists(v)]
    if not videos:
        print("[BUILD] no videos found.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    from liveportrait_engine import LivePortraitEngine
    char = os.path.join("ai-face", "character.jpg")
    if not os.path.exists(char):
        char = "character.jpg"
    eng = LivePortraitEngine(char)
    if eng.fallback_mode:
        print("[BUILD] LivePortrait unavailable — cannot measure pose.")
        return
    W = eng.wrapper
    mesh = eng._get_mesh()
    import mediapipe as mp
    eye_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                               max_num_faces=1, refine_landmarks=False,
                                               min_detection_confidence=0.4)

    # best[bucket] = (score, crop_bgr, yaw, pitch)  — pooled across ALL videos
    best = {}
    analysed = 0
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        n = min(MAX_SAMPLES_PER_VIDEO, total // SAMPLE_EVERY) if total else 0
        positions = ([int(k * total / n) for k in range(n)] if n > 0 else None)
        print(f"[BUILD] scanning {os.path.basename(video_path)} ({total} frames, "
              f"{n} samples by seek)...")
        # SEEK to evenly spaced positions instead of decoding every frame (fast
        # on long videos). Falls back to sequential read if seeking is unsupported.
        for pos in (positions or []):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop = _face_crop(frame, mesh)
            if crop is None:
                continue
            sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                  cv2.CV_64F).var()
            if sharp < MIN_SHARPNESS:
                continue
            analysed += 1
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                info = W.get_kp_info(W.prepare_source(rgb))
                yaw = float(info["yaw"]); pitch = float(info["pitch"])
            except Exception:
                continue
            if abs(pitch) > MAX_PITCH:
                continue
            b = min(YAW_BUCKETS, key=lambda c: abs(c - yaw))
            if abs(b - yaw) > BUCKET_HALF:
                continue
            # prefer EYES-OPEN frames (reject blinks); profiles can't be measured
            # (one eye hidden) so they're not penalised.
            score = sharp * _eye_open(crop, eye_mesh)
            if b not in best or score > best[b][0]:
                best[b] = (score, crop.copy(), yaw, pitch, os.path.basename(video_path))
        cap.release()

    if not best:
        print("[BUILD] no usable face frames found.")
        return

    # save references
    for old in glob.glob(os.path.join(OUT_DIR, "*.jpg")):
        os.remove(old)
    covered = []
    from collections import Counter
    vid_per_bucket = {}
    for b in sorted(best):
        score, crop, yaw, pitch, vid = best[b]
        path = os.path.join(OUT_DIR, f"yaw_{b:+03d}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        covered.append((b, yaw, score, vid))
        vid_per_bucket[b] = vid
        if b == 0 or (0 not in best and abs(b) == min(abs(x) for x in best)):
            cv2.imwrite(os.path.join(OUT_DIR, "character.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[BUILD] analysed {analysed} face frames; built {len(best)} angle views:")
    for b, yaw, score, vid in covered:
        print(f"   bucket {b:+4d}deg  (measured {yaw:+5.1f}, sharp {score:.0f})  <- {vid}")
    ys = sorted(best)
    print(f"[BUILD] yaw coverage: {min(ys):+d} to {max(ys):+d} deg")
    # Which single video gives the widest CONSISTENT span (same body/background)?
    by_vid = {}
    for b, _, _, vid in covered:
        by_vid.setdefault(vid, []).append(b)
    best_vid = max(by_vid, key=lambda v: max(by_vid[v]) - min(by_vid[v]))
    span = by_vid[best_vid]
    print(f"[BUILD] widest SINGLE-video span: {best_vid} covers "
          f"{min(span):+d}..{max(span):+d} deg ({len(span)} views) — a consistent "
          f"multi-view set comes from ONE video.")
    print(f"[BUILD] (wide views from DIFFERENT videos = body/background jump on "
          f"turn; same-video views are seamless.)")
    yaws = [b for b in best]
    print(f"[BUILD] yaw coverage: {min(yaws):+d} to {max(yaws):+d} deg")
    print(f"[BUILD] saved to {OUT_DIR}/  -> the engine will load these as the "
          f"multi-angle character.")


def _eye_open(crop, eye_mesh):
    """EAR-based eyes-open score: 1.0 open, ~0.25 closed (blink), 1.0 if the eyes
    can't be measured (profile) so we never penalise valid profile frames."""
    try:
        h, w = crop.shape[:2]
        res = eye_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return 1.0
        lm = res.multi_face_landmarks[0].landmark
        def ear(top, bot, l, r):
            import math
            v = abs(lm[top].y - lm[bot].y) * h
            hor = math.hypot((lm[l].x - lm[r].x) * w, (lm[l].y - lm[r].y) * h) + 1e-6
            return v / hor
        e = max(ear(159, 145, 33, 133), ear(386, 374, 362, 263))   # best of both eyes
        return 1.0 if e > 0.16 else 0.25
    except Exception:
        return 1.0


_LAST_BOX = {}      # persistent (cx, cy, side) so profile frames (no detection)
#                     still crop at the head's last known location


def _face_crop(frame, mesh):
    """Padded square crop around the face (512). At FULL profile the detector
    fails, so we reuse the last good box (the head stays put, only rotates) — that
    is exactly what lets us capture the ±90 profile frames."""
    h, w = frame.shape[:2]
    box = None
    try:
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            r = max((d.location_data.relative_bounding_box for d in res.detections),
                    key=lambda b: b.width * b.height)
            fw, fh = r.width * w, r.height * h
            if max(fw, fh) >= MIN_FACE_FRAC * max(w, h):
                cx = (r.xmin + r.width / 2) * w
                cy = (r.ymin + r.height / 2) * h
                box = (cx, cy, max(fw, fh) * CROP_PAD)
                _LAST_BOX["b"] = box
    except Exception:
        pass
    if box is None:
        box = _LAST_BOX.get("b")              # profile: reuse last head location
    if box is None:
        box = (w / 2.0, h * 0.40, min(w, h) * 0.9)   # first frames: center crop
    cx, cy, side = box
    x1 = int(cx - side / 2); y1 = int(cy - side / 2)
    x2 = int(cx + side / 2); y2 = int(cy + side / 2)
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    top = max(0, -y1); left = max(0, -x1)
    bottom = max(0, y2 - h); right = max(0, x2 - w)
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vids = sys.argv[1:]
    else:
        vids = sorted(glob.glob("source_vids/*.mp4")) or ["source_vids/char.mp4"]
    main(vids)
