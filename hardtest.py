# Hard test suite for the avatar model. Stresses the FULL pipeline on REAL video
# frames and checks every quality dimension. Prints PASS/FAIL per test.
import os, sys, glob, time
sys.path.insert(0, "engines")
import numpy as np, cv2, torch, mediapipe as mp
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine, PITCH_CAP
import enhance_engine
from body_motion import BodyMotionEngine
from restore_engine import RestoreEngine

char = r._character_path()
lp = LivePortraitEngine(char)
body = BodyMotionEngine()
restore = RestoreEngine()
fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
drv = cv2.resize(cv2.imread(char), (512, 512))
W = lp.wrapper
results = []


def out_pitch(frame):
    info = W.get_kp_info(W.prepare_source(cv2.cvtColor(cv2.resize(frame, (512, 512)),
                                                        cv2.COLOR_BGR2RGB)))
    return float(info["pitch"])


def face_h(frame):
    res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    b = max((d.location_data.relative_bounding_box for d in res.detections),
            key=lambda x: x.width * x.height)
    return b.height * frame.shape[0]


def rec(name, ok, detail):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


for _ in range(5):
    lp.process_frame(drv)

# ---- TEST A: pitch auto-center removes a baked-in offset ----
print("\n== TEST A: pitch auto-center convergence ==")
lp.recenter(); lp.process_frame(drv)
drive_pitch = float(lp._ref_kp_info["pitch"])
lp._ref_kp_info["pitch"] = lp._ref_kp_info["pitch"] + 10.0   # simulate looking-down neutral
off0 = abs(float(lp._ref_kp_info["pitch"]) - drive_pitch)
for _ in range(120):
    lp.process_frame(drv)
off1 = abs(float(lp._ref_kp_info["pitch"]) - drive_pitch)
rec("auto-center", off1 < off0 * 0.3,
    f"offset {off0:.1f}deg -> {off1:.1f}deg after ~120 frames (decays to level)")

# ---- TEST B: pitch FOLLOWS movement (auto-center doesn't kill it) ----
print("\n== TEST B: pitch following ==")
lp.recenter(); base = out_pitch(lp.process_frame(drv))
# inject a downward look into the driving kp by monkey-patching one frame
orig = lp._kp_info_from_crop
def patched(cropped):
    info = orig(cropped); info["pitch"] = info["pitch"] + 12.0; return info
lp._kp_info_from_crop = patched
moved = out_pitch(lp.process_frame(drv))
lp._kp_info_from_crop = orig
rec("pitch-follows", abs(moved - base) > 3.0,
    f"output pitch moved {moved - base:+.1f}deg when you tilted (responds to webcam)")

# ---- TEST C: zoom-on-turn ----
print("\n== TEST C: zoom-on-turn ==")
def render_yaw(dy):
    getR = lp._get_rotation_matrix; info = lp._x_s_info
    z = torch.zeros_like(info["yaw"])
    R_new = getR(z, z + float(dy), z) @ lp._R_s
    x_d = info["scale"] * (info["kp"] @ R_new + info["exp"]) + info["t"]
    try: x_d = W.stitching(lp._x_s, x_d)
    except Exception: pass
    return cv2.cvtColor(W.parse_output(W.warp_decode(lp._f_s, lp._x_s, x_d)["out"])[0],
                        cv2.COLOR_RGB2BGR)
hs = [h for h in (face_h(render_yaw(d)) for d in (0, 15, 30, 40)) if h]
spread = (max(hs) - min(hs)) / np.mean(hs) * 100 if hs else 99
rec("no-zoom-on-turn", spread < 9, f"face-size spread {spread:.1f}% across 0-40deg yaw")

# ---- TEST D: real-video stress (errors / face-found / stability / jitter) ----
print("\n== TEST D: real-video stress (200 frames) ==")
vids = sorted(glob.glob("source_vids/*.mp4"))
if vids:
    cap = cv2.VideoCapture(vids[0]); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, tot // 200); frames = []
    for k in range(200):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k * step); ok, f = cap.read()
        if ok: frames.append(cv2.resize(f, (512, 512)))
    cap.release()
    lp.recenter(); restore.every_n = 3; sizes = []; diffs = []; prev = None; errs = 0
    for f in frames:
        try:
            ai = lp.process_frame(f)
            ai = body.process(f, ai)
            ai = restore.restore(ai)
            ai = enhance_engine.enhance_frame(ai)
        except Exception:
            errs += 1; continue
        if lp._face_found:
            h = face_h(ai)
            if h: sizes.append(h)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(ai.astype(np.int16) - prev))))
        prev = ai.astype(np.int16)
    rec("no-crashes", errs == 0, f"{errs} pipeline errors over {len(frames)} real frames")
    if sizes:
        st = np.std(sizes) / np.mean(sizes) * 100
        rec("identity-stable", st < 7, f"output face-size std {st:.1f}% over real motion")
    if diffs:
        rec("low-jitter", np.median(diffs) < 14,
            f"median frame-to-frame diff {np.median(diffs):.1f} (lower=smoother)")
else:
    print("  (no source_vids)")

# ---- TEST E: fps with FULL Delulu pipeline ----
print("\n== TEST E: fps (full Delulu pipeline) ==")
enhance_engine.set_level("light"); restore.every_n = 3
ai = lp.process_frame(drv)
def full():
    a = lp.process_frame(drv); a = body.process(drv, a); a = restore.restore(a)
    return enhance_engine.enhance_frame(a)
for _ in range(3): full()
t = time.perf_counter()
for _ in range(30): full(); torch.cuda.synchronize()
per = (time.perf_counter() - t) / 30 * 1000
amort = per - lp_only if False else None
# amortized at LP interval 3: LP runs 1/3 of frames
lp_ms = 0.0
t2 = time.perf_counter()
for _ in range(10): lp.process_frame(drv); torch.cuda.synchronize()
lp_ms = (time.perf_counter() - t2) / 10 * 1000
amort_ms = per - lp_ms * (2/3)        # LP skipped 2 of 3 frames at interval 3
rec("fps-delulu", 1000/amort_ms >= 15,
    f"~{1000/amort_ms:.1f} fps full pipeline @ Delulu preset (LP int 3)")

# ---- TEST F: robustness on degenerate inputs ----
print("\n== TEST F: robustness (no crash on bad input) ==")
okrob = True
for bad in [np.zeros((512,512,3),np.uint8), np.full((512,512,3),255,np.uint8),
            (np.random.rand(512,512,3)*255).astype(np.uint8)]:
    try:
        lp.process_frame(bad); body.process(bad, drv.copy()); restore.restore(drv.copy())
    except Exception as e:
        okrob = False; print("   exception:", e)
rec("robust-degenerate", okrob, "no crash on black/white/noise frames")

print("\n===== SUMMARY =====")
p = sum(1 for _, ok in results if ok)
print(f"  {p}/{len(results)} PASS")
for n, ok in results:
    print(f"   {'PASS' if ok else 'FAIL'}  {n}")
