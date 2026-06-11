# =============================================================================
# engines/upscale_engine.py
# -----------------------------------------------------------------------------
# Optional final-frame detail pass: Real-ESRGAN (x2) adds crisp micro-texture so
# the picture reads as a real high-bitrate camera feed instead of a slightly soft
# render. The 512 frame is super-resolved to 1024 then resampled back to 512
# (INTER_AREA), which bakes in fine detail while keeping the virtual-camera size.
#
# Weights: Real-ESRGAN x2 (RRDBNet) pulled from HF (ai-forever/Real-ESRGAN). The
# RRDBNet architecture ships inside the already-installed `basicsr`, so no extra
# package is needed.
#
# This is the lowest-value / highest-cost of the realism stages (the face is
# already restored by CodeFormer), so it is OFF by default. Enable with
# AVATAR_UPSCALE=1. Like restore, it runs every Nth frame and caches between.
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_DIR, "ai-face", "models", "RealESRGAN_x2.pth")

# Run the (costly) RRDBNet forward every Nth frame, reuse the cached result.
UPSCALE_INTERVAL = int(os.environ.get("AVATAR_UPSCALE_INTERVAL", "2"))
# Blend the upscaled detail back at this strength (1.0 = full Real-ESRGAN,
# lower = subtler so it doesn't look over-processed).
UPSCALE_STRENGTH = float(os.environ.get("AVATAR_UPSCALE_STRENGTH", "0.85"))
USE_BF16 = os.environ.get("AVATAR_UPSCALE_BF16", "1") == "1"
OUT_SIZE = 512


class UpscaleEngine:
    """Resident Real-ESRGAN x2 detail enhancer applied to the final frame."""

    def __init__(self):
        self.ready = False
        self._net = None
        self._torch = None
        self.device = "cpu"
        self._frame_counter = 0
        self._cached = None
        self._err_printed = False
        try:
            self._load()
        except Exception as exc:
            print(f"[UPSCALE] init failed ({exc}) — upscaling disabled.")

    # -------------------------------------------------------------------------
    @property
    def ok(self):
        return self.ready

    def startup_check(self):
        if not self.ready:
            return True, "FALLBACK — no upscale (weights/model missing)."
        return True, (f"Real-ESRGAN x2 active (every {UPSCALE_INTERVAL} frame(s), "
                      f"strength {UPSCALE_STRENGTH}, "
                      f"{'bf16' if USE_BF16 else 'fp32'}).")

    def _load(self):
        if not os.path.exists(WEIGHTS_PATH):
            print(f"[UPSCALE] weights not found at {WEIGHTS_PATH}.")
            return
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True       # fixed 512 input -> tunes once

        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                      num_grow_ch=32, scale=2)
        ck = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        sd = ck.get("params_ema", ck.get("params", ck)) if isinstance(ck, dict) else ck
        net.load_state_dict(sd, strict=True)
        self._net = net.to(self.device).eval()
        self._warmup()
        self.ready = True
        if self.device == "cuda":
            print(f"[UPSCALE] GPU memory: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
        print("[UPSCALE] Ready — Real-ESRGAN x2 detail pass enabled.")

    def _warmup(self):
        dummy = (np.random.rand(OUT_SIZE, OUT_SIZE, 3) * 255).astype(np.uint8)
        for _ in range(2):
            try:
                self._run(dummy)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        """Return the frame with Real-ESRGAN detail blended in. Runs every
        UPSCALE_INTERVAL frames, reuses the cache between; never raises."""
        if not self.ready:
            return frame
        self._frame_counter += 1
        if self._cached is not None and (self._frame_counter % UPSCALE_INTERVAL) != 0:
            return self._cached
        try:
            sr = self._run(frame)                          # 512 BGR uint8, detailed
            if UPSCALE_STRENGTH < 1.0:
                sr = cv2.addWeighted(sr, UPSCALE_STRENGTH, frame, 1.0 - UPSCALE_STRENGTH, 0)
            self._cached = sr
            return sr
        except Exception as exc:
            if not self._err_printed:
                print(f"[UPSCALE] frame error ({exc}) — passing frame through.")
                self._err_printed = True
        return self._cached if self._cached is not None else frame

    # alias matching the loop's stage convention
    def upscale(self, frame):
        return self.process_frame(frame)

    def enhance_crop(self, crop, strength=0.7):
        """Real-ESRGAN detail on a SMALL crop at its NATIVE size (no forced 512), so
        it's fast (~15ms on a mouth crop) — used to de-blur the SPEAKING mouth every
        frame. Returns the detail-blended crop, same size. Never raises."""
        if not self.ready or crop is None:
            return crop
        try:
            torch = self._torch
            h, w = crop.shape[:2]
            if h < 8 or w < 8:
                return crop
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[np.newaxis]).to(self.device)
            with torch.no_grad():
                if USE_BF16 and self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = self._net(t)
                    out = out.float()
                else:
                    out = self._net(t)
            out = out[0].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
            big = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)   # 2x
            sr = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
            return cv2.addWeighted(sr, strength, crop, 1.0 - strength, 0)
        except Exception:
            return crop

    def _run(self, frame_bgr):
        """RRDBNet x2 forward on a 512 BGR frame; resample back to 512."""
        torch = self._torch
        if frame_bgr.shape[0] != OUT_SIZE or frame_bgr.shape[1] != OUT_SIZE:
            frame_bgr = cv2.resize(frame_bgr, (OUT_SIZE, OUT_SIZE))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[np.newaxis]).to(self.device)
        with torch.no_grad():
            if USE_BF16 and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self._net(t)
                out = out.float()
            else:
                out = self._net(t)
        out = out[0].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        big = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)  # 1024
        return cv2.resize(big, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = UpscaleEngine()
    return _engine


def startup_check():
    return get_engine().startup_check()


def upscale_frame(frame):
    return get_engine().process_frame(frame)


if __name__ == "__main__":
    eng = UpscaleEngine()
    print("[UPSCALE] startup_check:", eng.startup_check())
    for p in (os.path.join(PROJECT_DIR, "ai-face", "character.jpg"),
              os.path.join(PROJECT_DIR, "character.jpg")):
        if os.path.exists(p):
            img = cv2.resize(cv2.imread(p), (OUT_SIZE, OUT_SIZE))
            soft = cv2.GaussianBlur(img, (0, 0), 1.2)     # simulate the soft render
            out = eng.process_frame(soft)
            cv2.imwrite(os.path.join(PROJECT_DIR, "_upscale_before.jpg"), soft)
            cv2.imwrite(os.path.join(PROJECT_DIR, "_upscale_after.jpg"), out)
            print("[UPSCALE] wrote _upscale_before.jpg / _upscale_after.jpg")
            break
