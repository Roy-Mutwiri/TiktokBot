# One-time STRONG upscale of the character source + all angle views:
# CodeFormer (identity-preserving face restoration) -> Real-ESRGAN x2 (crisp
# micro-detail). Bakes HD quality into the avatar's source images so every warp
# starts from a sharp base instead of the soft phone-video crop.
import os, sys, glob
sys.path.insert(0, "engines")
import cv2
from face_restore_engine import FaceRestoreEngine
from upscale_engine import UpscaleEngine

os.environ["AVATAR_UPSCALE_INTERVAL"] = "1"      # always process (no cache skip)
cf = FaceRestoreEngine()
up = UpscaleEngine()

paths = ["character_views/character.jpg"] + sorted(glob.glob("character_views/yaw_*.jpg"))
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    h0, w0 = img.shape[:2]
    work = cv2.resize(img, (512, 512))
    try:
        cf._counter = 0                          # force a fresh restore each image
    except Exception:
        pass
    restored = cf.process_frame(work)            # CodeFormer face restore
    try:
        up._counter = 0
    except Exception:
        pass
    sharp = up.process_frame(restored)           # Real-ESRGAN x2 micro-detail
    out = cv2.resize(sharp, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"upscaled {os.path.basename(p)}")
print("DONE — character + views upscaled (CodeFormer + Real-ESRGAN).")
