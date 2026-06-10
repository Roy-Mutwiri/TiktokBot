# Drive the FULL process_frame path (soft-limit knee + auto-center + mouth-guard +
# gaze + stabilization) with an INJECTED yaw sweep, to see exactly what a live
# head turn renders with the current Safe settings.
import os, sys
sys.path.insert(0, "engines")
import numpy as np, cv2, torch
import realtime_avatar as r
import liveportrait_engine as lpe
from liveportrait_engine import LivePortraitEngine

char = r._character_path()
lp = LivePortraitEngine(char)
print(f"YAW_CAP={lpe.YAW_CAP} PITCH_CAP={lpe.PITCH_CAP} MOUTH_GUARD={lpe.MOUTH_GUARD} "
      f"AUTOCENTER={lpe.AUTOCENTER_PITCH}")
frontal = cv2.resize(cv2.imread(char), (512, 512))

# establish a frontal reference
lp.recenter()
lp.process_frame(frontal)

orig = lp._kp_info_from_crop
inj = {"yaw": 0.0}
def patched(cropped):
    info = orig(cropped)
    info["yaw"] = info["yaw"] + inj["yaw"]      # simulate turning your head
    return info
lp._kp_info_from_crop = patched

tiles = []
for y in (0, 10, 20, 30, 45):
    inj["yaw"] = float(y)
    out = None
    for _ in range(4):                          # let 1€/auto-center settle
        out = lp.process_frame(frontal)
    cv2.putText(out, f"turn {y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    tiles.append(cv2.resize(out, (256, 256)))
cv2.imwrite("_turn_diag.jpg", np.hstack(tiles))
print("saved _turn_diag.jpg (avatar at injected turn 0/10/20/30/45 through full live path)")
