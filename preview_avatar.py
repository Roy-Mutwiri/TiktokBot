# Headless visual test: capture real webcam frames, run LivePortrait + enhance,
# and save side-by-side montages so we can SEE the webcam -> AI face transform.
import os, sys, time
sys.path.insert(0, "engines")
import numpy as np
import cv2

from liveportrait_engine import LivePortraitEngine
import enhance_engine

CHAR = os.path.join("ai-face", "character.jpg")
if not os.path.exists(CHAR):
    CHAR = "character.jpg"

print("loading LivePortrait...")
lp = LivePortraitEngine(CHAR)
print("startup:", lp.startup_check()[1])

src = cv2.resize(cv2.imread(CHAR), (512, 512))   # the AI character (source)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("NO WEBCAM"); sys.exit(1)

# discard first few frames (camera auto-exposure warmup)
for _ in range(8):
    cap.read()
    time.sleep(0.05)

print("capturing ~4s — MOVE YOUR HEAD (nod, turn, blink, smile)...")
saved = 0
shots = []
t_end = time.time() + 4.0
i = 0
while time.time() < t_end:
    ok, frame = cap.read()
    if not ok:
        continue
    webcam = cv2.resize(frame, (512, 512))
    ai = lp.process_frame(webcam)          # webcam pose -> AI character
    final = enhance_engine.enhance_frame(ai.copy(), is_speaking=False)
    i += 1
    # save 3 snapshots spread across the capture (skip the very first = neutral ref)
    if i in (12, 30, 55):
        montage = np.hstack([webcam, ai, final])
        cv2.putText(montage, "WEBCAM (you)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(montage, "AI FACE (LivePortrait)", (522, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(montage, "FINAL (-> OBS)", (1034, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        path = f"_shot_{saved}.jpg"
        cv2.imwrite(path, montage)
        shots.append(path)
        saved += 1
        print(f"  saved {path}")

cap.release()
print("frames processed:", i, "| shots:", shots)
