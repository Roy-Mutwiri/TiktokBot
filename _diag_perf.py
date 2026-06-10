# Per-stage performance diagnostic for the avatar pipeline.
# Reports: CUDA status, model device + dtype (FP16?), source-cache check, and
# per-stage ms (liveportrait / body / enhance) + total + fps. No webcam needed
# (drives with a real face image so LivePortrait actually runs).
import os, sys, time
sys.path.insert(0, "engines")
import numpy as np, cv2, torch

print("=" * 64)
print(" CUDA:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine, USE_HALF
from realtime_avatar import LP_INTERVAL
import enhance_engine
from body_motion import BodyMotionEngine

char = r._character_path()
lp = LivePortraitEngine(char)
body = BodyMotionEngine()

# --- model device + dtype (is FP16 actually applied?) ---
print(" USE_HALF flag:", USE_HALF, "| LP_INTERVAL:", LP_INTERVAL)
W = lp.wrapper
for name in ("appearance_feature_extractor", "motion_extractor", "warping_module",
             "spade_generator"):
    m = getattr(W, name, None)
    if m is not None:
        try:
            p = next(m.parameters())
            print(f"   {name:28s} device={p.device} dtype={p.dtype}")
        except Exception as e:
            print(f"   {name}: {e}")
# source cache check: these must be precomputed (not None)
print(" source cached:", all(getattr(lp, a, None) is not None
                              for a in ("_f_s", "_x_s", "_x_s_info", "_R_s")))

# driving frame = the character itself (has a face so LP runs full path)
drv = cv2.resize(cv2.imread(char), (512, 512))
for _ in range(6):
    lp.process_frame(drv)            # warm up

N = 40
t_lp = t_body = t_enh = 0.0
for _ in range(N):
    a = time.perf_counter()
    ai = lp.process_frame(drv)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    b = time.perf_counter()
    ai = body.process(drv, ai)
    c = time.perf_counter()
    enhance_engine.enhance_frame(ai, is_speaking=False)
    d = time.perf_counter()
    t_lp += (b - a); t_body += (c - b); t_enh += (d - c)

lp_ms = t_lp / N * 1000; body_ms = t_body / N * 1000; enh_ms = t_enh / N * 1000
total = lp_ms + body_ms + enh_ms
print("-" * 64)
print(f"[DIAG] liveportrait: {lp_ms:5.1f}ms | body: {body_ms:5.1f}ms | "
      f"enhance: {enh_ms:5.1f}ms | TOTAL: {total:5.1f}ms | fps: {1000/total:4.1f}")
# what fps would be at LP_INTERVAL=2 (LP every other frame)
half = lp_ms / 2 + body_ms + enh_ms
print(f"[DIAG] at LP_INTERVAL=2: ~{1000/half:4.1f} fps  (LP amortized)")
print("=" * 64)
