import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.getcwd(), "engines"))
VID = r"C:/Users/user/Downloads/Telegram Desktop/video_2026-06-10_14-45-12.mp4"
cap = cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, fr = cap.read(); cap.release()
fr = cv2.resize(fr, (512,512))

import face_restore_engine as R
eng = R.get_engine()
print("backend:", eng.backend, "ready:", eng.ready, "device:", eng.device)
print("RESTORE_INTERVAL:", R.RESTORE_INTERVAL, "DETECT_INTERVAL:", R.DETECT_INTERVAL)

if not eng.ready:
    print("NOT READY - cannot time"); sys.exit()

# Time refresh_alignment (detect) alone
eng._refresh_alignment(fr)  # warm
best=1e9;tot=0
for _ in range(8):
    s=time.perf_counter(); eng._refresh_alignment(fr); e=time.perf_counter()-s
    tot+=e;best=min(best,e)
print(f"_refresh_alignment (detect+parse) avg {tot/8*1000:7.2f} best {best*1000:7.2f}")

# Time the CodeFormer crop forward alone
crop = cv2.warpAffine(fr, eng._affine, (512,512))
eng._run_codeformer_crop(crop)
import torch; torch.cuda.synchronize() if eng.device=="cuda" else None
best=1e9;tot=0
for _ in range(15):
    s=time.perf_counter(); eng._run_codeformer_crop(crop)
    if eng.device=="cuda": torch.cuda.synchronize()
    e=time.perf_counter()-s; tot+=e;best=min(best,e)
print(f"_run_codeformer_crop (forward)    avg {tot/15*1000:7.2f} best {best*1000:7.2f}")

# Full _restore_codeformer (warp+forward+warp+blend), no detect (cached affine)
eng._restore_calls = 1  # avoid hitting DETECT_INTERVAL
eng._restore_codeformer(fr)
best=1e9;tot=0
for i in range(15):
    eng._restore_calls = 1
    s=time.perf_counter(); eng._restore_codeformer(fr)
    if eng.device=="cuda": torch.cuda.synchronize()
    e=time.perf_counter()-s; tot+=e;best=min(best,e)
print(f"_restore_codeformer (cached align) avg {tot/15*1000:7.2f} best {best*1000:7.2f}")

# process_frame amortized over RESTORE_INTERVAL
eng._cached=None; eng._frame_counter=0
import time as _t
s=_t.perf_counter()
N=30
for _ in range(N):
    eng.process_frame(fr)
    if eng.device=="cuda": torch.cuda.synchronize()
print(f"process_frame amortized/frame      avg {(_t.perf_counter()-s)/N*1000:7.2f} (interval={R.RESTORE_INTERVAL})")
