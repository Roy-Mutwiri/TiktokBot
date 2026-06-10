# Does torch.compile speed up LivePortrait on this Windows/torch build?
import os, sys, time
sys.path.insert(0, "engines")
import numpy as np, cv2, torch
import realtime_avatar as r

# monkey-patch the engine to enable torch.compile in the wrapper config
import liveportrait_engine as lpe
_orig_init = lpe.LivePortraitEngine._init_pipeline

def patched(self, lp_path):
    import sys as _s
    if lp_path not in _s.path:
        _s.path.insert(0, lp_path)
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.camera import get_rotation_matrix
    self._torch = torch
    self._get_rotation_matrix = get_rotation_matrix
    cfg = InferenceConfig()
    if hasattr(cfg, "flag_do_torch_compile"):
        cfg.flag_do_torch_compile = True
    self.wrapper = LivePortraitWrapper(inference_cfg=cfg)
    src_rgb = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
    I_s = self.wrapper.prepare_source(src_rgb)
    self._x_s_info = self.wrapper.get_kp_info(I_s)
    self._R_s = get_rotation_matrix(self._x_s_info["pitch"], self._x_s_info["yaw"], self._x_s_info["roll"])
    self._f_s = self.wrapper.extract_feature_3d(I_s)
    self._x_s = self.wrapper.transform_keypoint(self._x_s_info)
    self._frontal_yaw = float(self._x_s_info["yaw"])
    self._refs = [dict(f=self._f_s, xs=self._x_s, kp=self._x_s_info["kp"],
                       exp=self._x_s_info["exp"], scale=self._x_s_info["scale"],
                       t=self._x_s_info["t"], R=self._R_s, base_yaw=0.0)]
    self._multi = False; self._cur_ref = 0; self._multi_yaw_cap = 30.0

lpe.LivePortraitEngine._init_pipeline = patched

char = r._character_path()
eng = lpe.LivePortraitEngine(char)
drv = cv2.resize(cv2.imread(char), (512, 512))
print("warming up (torch.compile autotune — may take 1-3 min)...")
t0 = time.perf_counter()
for _ in range(8):
    eng.process_frame(drv)
torch.cuda.synchronize()
print(f"warmup done in {time.perf_counter()-t0:.1f}s")
N = 30
t = time.perf_counter()
for _ in range(N):
    eng.process_frame(drv)
    torch.cuda.synchronize()
ms = (time.perf_counter() - t) / N * 1000
print(f"[COMPILE TEST] LivePortrait with torch.compile: {ms:.1f}ms  ({1000/ms:.1f} fps)")
print("(compare to 89.5ms eager)")
