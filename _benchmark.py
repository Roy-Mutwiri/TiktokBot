# Hard benchmark suite for the avatar pipeline.
#   BENCH 1: zoom-on-turn — output face SIZE vs head yaw (should stay ~constant)
#   BENCH 2: per-stage timing + fps across quality presets (+/- GFPGAN)
#   BENCH 3: real-video drive — fps + identity/size stability on REAL frames
import os, sys, glob, time
sys.path.insert(0, "engines")
import numpy as np, cv2, torch, mediapipe as mp
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine
import enhance_engine
from body_motion import BodyMotionEngine
from restore_engine import RestoreEngine

char = r._character_path()
lp = LivePortraitEngine(char)
body = BodyMotionEngine()
restore = RestoreEngine()
fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
drv = cv2.resize(cv2.imread(char), (512, 512))
for _ in range(5):
    lp.process_frame(drv)


def face_h(frame):
    res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    b = max((d.location_data.relative_bounding_box for d in res.detections),
            key=lambda x: x.width * x.height)
    return b.height * frame.shape[0]


def render_yaw(dy):
    """Drive the avatar at a synthetic yaw delta and return the output frame."""
    W = lp.wrapper; getR = lp._get_rotation_matrix; info = lp._x_s_info
    z = torch.zeros_like(info["yaw"])
    R_new = getR(z, z + float(dy), z) @ lp._R_s
    x_d = info["scale"] * (info["kp"] @ R_new + info["exp"]) + info["t"]
    try: x_d = W.stitching(lp._x_s, x_d)
    except Exception: pass
    out = W.warp_decode(lp._f_s, lp._x_s, x_d)
    return cv2.cvtColor(W.parse_output(out["out"])[0], cv2.COLOR_RGB2BGR)


print("\n===== BENCH 1: ZOOM-ON-TURN (output face height vs yaw) =====")
hs = []
for dy in (0, 10, 20, 30, 40):
    h = face_h(render_yaw(dy))
    hs.append(h)
    print(f"  yaw {dy:2d}deg -> face height {h:.0f}px" if h else f"  yaw {dy}: no face")
hs = [h for h in hs if h]
if hs:
    spread = (max(hs) - min(hs)) / np.mean(hs) * 100
    print(f"  >> face-size spread across turns: {spread:.1f}%  "
          f"({'GOOD (no zoom)' if spread < 8 else 'ZOOMING'})")

print("\n===== BENCH 2: PER-STAGE TIMING / FPS =====")
def time_stage(fn, n=20):
    for _ in range(3): fn()
    t = time.perf_counter()
    for _ in range(n): fn(); torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000
ai = lp.process_frame(drv)
lp_ms = time_stage(lambda: lp.process_frame(drv))
restore.every_n = 1; restore.skin_detail = 0.7
gf_ms = time_stage(lambda: restore.restore(ai.copy()))
bd_ms = time_stage(lambda: body.process(drv, ai.copy()))
enhance_engine.set_level("light"); enl = time_stage(lambda: enhance_engine.enhance_frame(ai.copy()))
enhance_engine.set_level("full");  enf = time_stage(lambda: enhance_engine.enhance_frame(ai.copy()))
print(f"  LP {lp_ms:.0f} | GFPGAN {gf_ms:.0f} | body {bd_ms:.0f} | enh-light {enl:.0f} | enh-full {enf:.0f} ms")
for name, lpi, en, gf_on, bd_on in [("Smooth", 3, enl, False, False),
                                    ("Balanced", 2, enl, True, True),
                                    ("Sharp", 1, enf, True, True)]:
    tot = lp_ms/lpi + (gf_ms/2 if gf_on else 0) + (bd_ms/2 if bd_on else 0) + en + 4
    tot_g = lp_ms/lpi + gf_ms/2 + (bd_ms/2 if bd_on else 0) + en + 4
    print(f"  {name:9s}: ~{1000/tot:4.1f} fps (no GFPGAN) | ~{1000/tot_g:4.1f} fps (GFPGAN on)")

print("\n===== BENCH 3: REAL-VIDEO DRIVE (fps + size stability) =====")
vids = sorted(glob.glob("source_vids/*.mp4"))
if vids:
    cap = cv2.VideoCapture(vids[0]); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, tot // 60); frames = []
    for k in range(60):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k*step); ok, f = cap.read()
        if ok: frames.append(cv2.resize(f, (512, 512)))
    cap.release()
    lp.recenter(); sizes = []; t = time.perf_counter(); nproc = 0
    for f in frames:
        out = lp.process_frame(f); nproc += 1
        if lp._face_found:
            h = face_h(out)
            if h: sizes.append(h)
    dt = time.perf_counter() - t
    print(f"  drove {nproc} REAL frames from {os.path.basename(vids[0])} in {dt:.1f}s "
          f"-> {nproc/dt:.1f} fps (LP every frame)")
    if sizes:
        st = np.std(sizes)/np.mean(sizes)*100
        print(f"  output face-size stability over real motion: std {st:.1f}% "
              f"({'STABLE' if st < 6 else 'DRIFTING'})")
else:
    print("  (no source_vids found)")
print("\n===== DONE =====")
