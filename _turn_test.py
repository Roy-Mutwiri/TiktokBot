# Hard turning test: render the character at increasing RAW yaw (no clamp) to
# show exactly where single-image LivePortrait breaks. Saves a montage + an
# artifact metric (edge energy spikes as the profile hallucinates).
import os, sys
sys.path.insert(0, "engines")
import numpy as np, cv2, torch
import realtime_avatar as r
from liveportrait_engine import LivePortraitEngine

char = r._character_path()
lp = LivePortraitEngine(char)
W = lp.wrapper; getR = lp._get_rotation_matrix; info = lp._x_s_info
for _ in range(4): lp.process_frame(cv2.resize(cv2.imread(char), (512, 512)))

def render_yaw(dy):
    z = torch.zeros_like(info["yaw"])
    R_new = getR(z, z + float(dy), z) @ lp._R_s
    x_d = info["scale"] * (info["kp"] @ R_new + info["exp"]) + info["t"]
    try: x_d = W.stitching(lp._x_s, x_d)
    except Exception: pass
    return cv2.cvtColor(W.parse_output(W.warp_decode(lp._f_s, lp._x_s, x_d)["out"])[0],
                        cv2.COLOR_RGB2BGR)

angles = [0, 15, 30, 45, 60, 75, 90]
tiles = []
print("yaw  edge-energy(artifact proxy)")
base = None
for a in angles:
    img = render_yaw(a)
    e = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if base is None: base = e
    cv2.putText(img, f"{a}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    tiles.append(cv2.resize(img, (256, 256)))
    print(f"  {a:3d}   {e:7.0f}  ({e/base*100:4.0f}% of frontal)")

row1 = np.hstack(tiles[:4]); row2 = np.hstack(tiles[4:] + [np.zeros((256,256,3),np.uint8)])
cv2.imwrite("_turn_sweep.jpg", np.vstack([row1, row2]))
print("saved _turn_sweep.jpg (top 0/15/30/45  bottom 60/75/90)")
