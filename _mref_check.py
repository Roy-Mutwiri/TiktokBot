import sys
sys.path.insert(0, "engines")
import numpy as np, cv2
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine
lp = LivePortraitEngine(r._character_path())
print(f"multi={lp._multi} refs={sorted(round(x['base_yaw']) for x in lp._refs)} cap={round(lp._multi_yaw_cap)}")
frontal = cv2.resize(cv2.imread(r._character_path()), (512, 512))
lp.recenter(); lp.process_frame(frontal)
orig = lp._kp_info_from_crop
inj = {"y": 0.0}
lp._kp_info_from_crop = lambda c: (lambda i: (i.__setitem__("yaw", i["yaw"] + inj["y"]) or i))(orig(c))
tiles = []
for y in (0, 20, 40, 55, 70):
    inj["y"] = float(y); out = None
    for _ in range(3): out = lp.process_frame(frontal)
    cv2.putText(out, f"{y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    tiles.append(cv2.resize(out, (256, 256)))
cv2.imwrite("_mref_check.jpg", np.hstack(tiles))
print("saved _mref_check.jpg (REAL-view turns 0/20/40/55/70)")
