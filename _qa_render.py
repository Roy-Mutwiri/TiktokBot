import sys
sys.path.insert(0, "engines")
import numpy as np, cv2
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine
lp = LivePortraitEngine(r._character_path())
frontal = cv2.resize(cv2.imread(r._character_path()), (512, 512))
lp.recenter(); lp.process_frame(frontal)
orig = lp._kp_info_from_crop
inj = {"y": 0.0}
lp._kp_info_from_crop = lambda c: (lambda i: (i.__setitem__("yaw", i["yaw"] + inj["y"]) or i))(orig(c))
def shot(y):
    inj["y"] = float(y); out = None
    for _ in range(3): out = lp.process_frame(frontal)
    cv2.putText(out, f"{y:+d}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return cv2.resize(out, (200, 200))
top = np.hstack([shot(y) for y in (-55, -40, -25, -10, 0)])
bot = np.hstack([shot(y) for y in (10, 25, 40, 55, 0)])
cv2.imwrite("_qa_turns.jpg", np.vstack([top, bot]))
print("saved _qa_turns.jpg")
