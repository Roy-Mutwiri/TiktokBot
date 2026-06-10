# Verify GFPGAN restoration improves the LivePortrait output + measure ms.
import sys, time
sys.path.insert(0, "engines")
import numpy as np, cv2
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine
from restore_engine import RestoreEngine

char = r._character_path()
lp = LivePortraitEngine(char)
drv = cv2.resize(cv2.imread(char), (512, 512))
for _ in range(4):
    lp.process_frame(drv)
ai = lp.process_frame(drv)          # the plastic LP output
re = RestoreEngine()
re.skin_detail = 0.70

# time it (every-frame, so force restore each call)
re.every_n = 1
for _ in range(3):
    re.restore(ai.copy())           # warm
N = 20; t = time.perf_counter()
for _ in range(N):
    out = re.restore(ai.copy())
import torch; torch.cuda.synchronize()
ms = (time.perf_counter() - t) / N * 1000
print(f"[GFPGAN] restore: {ms:.1f}ms/frame  (every 2nd -> ~{ms/2:.0f}ms avg)")

restored = re.restore(ai.copy())
cv2.imwrite("_restore_check.jpg", np.hstack([cv2.resize(ai,(400,400)),
                                             cv2.resize(restored,(400,400))]))
print("saved _restore_check.jpg (left LP plastic, right GFPGAN restored)")
