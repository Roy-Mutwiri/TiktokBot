# Drive the FULL engine with REAL frames from the user's videos, capturing
# webcam|avatar montages at frames with different head yaw, to SEE turning errors.
import os, sys, glob
sys.path.insert(0, "engines")
import numpy as np, cv2, torch
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine

char = r._character_path()
lp = LivePortraitEngine(char)
W = lp.wrapper

vids = sorted(glob.glob("source_vids/*.mp4"))
cap = cv2.VideoCapture(vids[0]); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# sample many frames, drive engine, record (yaw_delta, webcam, avatar)
lp.recenter()
samples = []   # (yaw_delta_deg, webcam, avatar)
step = max(1, tot // 250)
ref_yaw = None
for k in range(250):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k * step); ok, f = cap.read()
    if not ok:
        continue
    wc = cv2.resize(f, (512, 512))
    ai = lp.process_frame(wc)
    if not lp._face_found or lp._ref_kp_info is None:
        continue
    yaw = float(lp._ref_kp_info["yaw"])      # ref drifts; use current driving via crop
    # measure the driving yaw of THIS frame
    try:
        crop = lp._crop_driving_face(wc)
        dyaw = float(W.get_kp_info(W.prepare_source(cv2.cvtColor(cv2.resize(crop,(512,512)), cv2.COLOR_BGR2RGB)))["yaw"]) - yaw
    except Exception:
        dyaw = 0.0
    samples.append((dyaw, wc.copy(), ai.copy()))
cap.release()

# pick frames at increasing turn magnitude
samples.sort(key=lambda s: abs(s[0]))
picks = []
for target in (3, 12, 20, 28, 38):
    best = min(samples, key=lambda s: abs(abs(s[0]) - target))
    picks.append(best)
tiles = []
for dy, wc, ai in picks:
    m = np.vstack([cv2.resize(wc, (224, 224)), cv2.resize(ai, (224, 224))])
    cv2.putText(m, f"{dy:+.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(m)
cv2.imwrite("_turn_diag.jpg", np.hstack(tiles))
print("turn samples captured; yaw deltas:", [round(p[0]) for p in picks])
print("saved _turn_diag.jpg (top=webcam you, bottom=avatar; columns by turn deg)")
