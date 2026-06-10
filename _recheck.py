# Re-measure per-stage timing AFTER the optimizations (light enhance + body
# pose-interval). Projects fps at each LP interval.
import sys, time
sys.path.insert(0, "engines")
import numpy as np, cv2, torch
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine
import enhance_engine
from body_motion import BodyMotionEngine

char = r._character_path()
lp = LivePortraitEngine(char)
body = BodyMotionEngine()
enhance_engine.set_level("light")          # Balanced/Smooth use light
drv = cv2.resize(cv2.imread(char), (512, 512))
for _ in range(6):
    lp.process_frame(drv)

N = 40
t_lp = t_body = t_enh = 0.0
for i in range(N):
    a = time.perf_counter(); lp.process_frame(drv); torch.cuda.synchronize(); b = time.perf_counter()
    body._fc = i                            # exercise pose-interval path
    body.process(drv, drv.copy()); c = time.perf_counter()
    enhance_engine.enhance_frame(drv.copy(), is_speaking=False); d = time.perf_counter()
    t_lp += b - a; t_body += c - b; t_enh += d - c

lpm = t_lp/N*1000; bd = t_body/N*1000; en = t_enh/N*1000
print(f"[RECHECK] LP {lpm:.0f}ms | body {bd:.0f}ms (pose every 2) | enh-light {en:.0f}ms")
for itv in (1, 2, 3):
    tot = lpm/itv + bd + en + 4   # +4 read/send
    print(f"   LP interval {itv}: ~{tot:.0f}ms -> {1000/tot:.1f} fps")
