# =============================================================================
# engines/gpu_fx.py — run the HEAVY per-frame image filters on the GPU (torch +
# kornia), instead of OpenCV on the CPU.
#
# Why: OpenCV here is a CPU-only build (0 CUDA devices), so the enhance pass
# (bilateral denoise + CLAHE local-contrast + multi-scale clarity unsharp) pegged
# the CPU at ~96% while the GPU napped at 20-40%. These ops are exactly what a GPU
# is fast at. This module does them as torch tensors on CUDA in ONE upload/​download
# per frame, rebalancing the load — same look, far less CPU.
#
#   out = gpu_fx.enhance(bgr)        # None if GPU path unavailable -> caller uses CPU
#
# Never raises; returns None on any failure so enhance_engine falls back to OpenCV.
# =============================================================================
import os

import numpy as np
import cv2

_OK = False
_DEV = "cpu"
try:
    import torch
    import kornia.filters as _kf
    import kornia.enhance as _ke
    import kornia.color as _kc
    if torch.cuda.is_available():
        _DEV = "cuda"
        _OK = True
except Exception as _exc:                       # torch/kornia missing -> CPU path
    print(f"[GPUFX] GPU filters unavailable ({_exc}) — using CPU enhance.")
    _OK = False

# match the CPU enhance config (same env vars so the look is identical)
_CLAHE = float(os.environ.get("AVATAR_CLAHE", "0.55"))
_CLARITY = float(os.environ.get("AVATAR_CLARITY", "0.34"))
_FINE = float(os.environ.get("AVATAR_FINE_SHARP", "0.7"))
_DENOISE = float(os.environ.get("AVATAR_POLISH_DENOISE", "1.0"))


def available():
    return _OK


def _ks(sigma):
    k = int(round(sigma * 6)) | 1
    return max(3, k)


def enhance(bgr, clahe=None, clarity=None, fine=None, denoise=None):
    """GPU bilateral-denoise + CLAHE(L) + 2-scale clarity, in one pass. Returns a BGR
    uint8 frame, or None if the GPU path isn't usable (caller falls back to CPU)."""
    if not _OK or bgr is None:
        return None
    clahe = _CLAHE if clahe is None else clahe
    clarity = _CLARITY if clarity is None else clarity
    fine = _FINE if fine is None else fine
    denoise = _DENOISE if denoise is None else denoise
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(_DEV, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)   # 1,3,H,W [0,1]
        with torch.no_grad():
            if denoise > 0:                       # edge-preserving skin denoise
                t = _kf.bilateral_blur(t, (5, 5), 0.11 * denoise, (2.0, 2.0))
            if clahe > 0:                         # local contrast on LUMINANCE only
                lab = _kc.rgb_to_lab(t)
                L = (lab[:, 0:1] / 100.0).clamp(0, 1)
                Lc = _ke.equalize_clahe(L, clip_limit=2.2, grid_size=(8, 8))
                lab = lab.clone()
                lab[:, 0:1] = (L * (1 - clahe) + Lc * clahe).clamp(0, 1) * 100.0
                t = _kc.lab_to_rgb(lab).clamp(0, 1)
            if clarity > 0:                       # large-radius clarity
                k = _ks(3.0)
                b = _kf.gaussian_blur2d(t, (k, k), (3.0, 3.0))
                t = (t * (1.0 + clarity) - b * clarity).clamp(0, 1)
            if fine > 0:                          # fine detail / edge crispness
                k = _ks(0.9)
                b = _kf.gaussian_blur2d(t, (k, k), (0.9, 0.9))
                t = (t * (1.0 + fine) - b * fine).clamp(0, 1)
        out = (t.squeeze(0).permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f"[GPUFX] enhance error ({exc}) — CPU fallback.")
        return None
