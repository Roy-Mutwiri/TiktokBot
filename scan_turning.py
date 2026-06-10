# Fast yaw-coverage scan: estimate head yaw from face_detection keypoints (eyes +
# nose) across ALL videos to find which ones contain TURNING (profile) footage —
# no LivePortrait needed, so it's ~20x faster than the full extractor.
import os, sys, glob
import numpy as np, cv2, mediapipe as mp

fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
SAMPLES = 120                  # frames per video (by seek)


def yaw_estimate(frame):
    """Rough yaw (deg) from eye/nose keypoints. + = turned one way, - the other."""
    h, w = frame.shape[:2]
    res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    d = max(res.detections, key=lambda x: x.location_data.relative_bounding_box.width)
    kp = d.location_data.relative_keypoints       # 0=Reye 1=Leye 2=nose 3=mouth 4=Rear 5=Lear
    if len(kp) < 6:
        return None
    reye = np.array([kp[0].x, kp[0].y]); leye = np.array([kp[1].x, kp[1].y])
    nose = np.array([kp[2].x, kp[2].y])
    eye_c = (reye + leye) / 2.0
    eye_d = np.linalg.norm(reye - leye) + 1e-6
    # nose horizontal offset from eye-center, in eye-widths -> ~yaw
    off = (nose[0] - eye_c[0]) / eye_d
    return float(off * 90.0)                      # scale to ~deg (approx)


vids = sorted(glob.glob("source_vids/*.mp4"))
print(f"[PREP] scanning {len(vids)} videos for turning footage...\n", flush=True)
STRIDE = 12          # process every Nth frame (sequential read — fast, no seek)
MAX_PROC = 200       # stop after this many processed frames per video
rows = []
for v in vids:
    cap = cv2.VideoCapture(v)
    yaws = []; i = 0; proc = 0
    while i < 500:                    # hard decode cap (first ~20s) — keeps it fast
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i % STRIDE:
            continue
        proc += 1
        y = yaw_estimate(cv2.resize(f, (480, 480)))
        if y is not None:
            yaws.append(y)
    cap.release()
    if not yaws:
        continue
    yaws = np.array(yaws)
    rng = float(yaws.max() - yaws.min())
    rows.append((rng, float(yaws.min()), float(yaws.max()),
                 int((np.abs(yaws) > 25).sum()), os.path.basename(v)))

rows.sort(reverse=True)
print(f"{'turn-range':>10} {'min':>6} {'max':>6} {'#>25deg':>8}  video")
for rng, lo, hi, nbig, name in rows:
    flag = "  <== TURNS" if rng > 40 or nbig > 8 else ""
    print(f"{rng:10.0f} {lo:6.0f} {hi:6.0f} {nbig:8d}  {name}{flag}")
print(f"\n[PREP] videos with the WIDEST head-turn range listed first.")
print(f"[PREP] '<== TURNS' = good profile/turning footage to extract real side views from.")
