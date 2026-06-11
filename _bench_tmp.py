import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.getcwd(), "engines"))

VID = r"C:/Users/user/Downloads/Telegram Desktop/video_2026-06-10_14-45-12.mp4"
cap = cv2.VideoCapture(VID)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, fr = cap.read()
cap.release()
assert ok, "no frame"
fr = cv2.resize(fr, (512, 512))
print("frame", fr.shape)

import enhance_engine as E
E._ensure_assets()

def t(fn, n=50, *a, **k):
    fn(fr.copy(), *a, **k)  # warm
    best = 1e9; tot = 0
    for _ in range(n):
        f = fr.copy()
        s = time.perf_counter()
        fn(f, *a, **k)
        e = time.perf_counter() - s
        tot += e; best = min(best, e)
    return tot/n*1000, best*1000

print("\n=== ENHANCE STAGES (ms avg / best) ===")
for name, fn in [
    ("background_composite", E.background_composite),
    ("apply_clarity", E.apply_clarity),
    ("apply_color_grade", E.apply_color_grade),
    ("apply_vignette", E.apply_vignette),
    ("apply_camera_shake", E.apply_camera_shake),
    ("apply_chromatic_aberration", E.apply_chromatic_aberration),
    ("apply_grain", E.apply_grain),
    ("draw_ticker", E.draw_ticker),
    ("draw_live_badge", E.draw_live_badge),
]:
    a, b = t(fn)
    print(f"  {name:28s} avg {a:6.2f}  best {b:6.2f}")

# full enhance_frame
a, b = t(lambda f, **k: E.enhance_frame(f, **k), is_speaking=True)
print(f"  {'enhance_frame FULL':28s} avg {a:6.2f}  best {b:6.2f}")
E.set_level("light")
a, b = t(lambda f, **k: E.enhance_frame(f, **k), is_speaking=True)
print(f"  {'enhance_frame LIGHT':28s} avg {a:6.2f}  best {b:6.2f}")
E.set_level("full")

# selfie-seg raw cost (the model forward only)
seg = E._get_segmenter()
if seg is not None:
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    seg.process(rgb)
    best=1e9; tot=0
    for _ in range(40):
        s=time.perf_counter(); seg.process(rgb); e=time.perf_counter()-s
        tot+=e; best=min(best,e)
    print(f"\n  selfie-seg forward ONLY       avg {tot/40*1000:6.2f}  best {best*1000:6.2f}")
