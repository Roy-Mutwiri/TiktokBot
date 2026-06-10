# =============================================================================
# engines/faceswap_engine.py
# -----------------------------------------------------------------------------
# Real-time FACE SWAP (inswapper) — technique borrowed from Deep-Live-Cam /
# iRoopDeepFaceCam. Unlike LivePortrait (which animates a static image and
# struggles at profiles), this takes the OPERATOR's REAL webcam head — real pose,
# real profile, real lighting, real expression — and swaps the CHARACTER's facial
# identity onto it with insightface's inswapper_128. Because the head geometry is
# real, turns to a full 90deg profile "just work".
#
# Runs on GPU: onnxruntime's CUDA EP only loads its DLLs if the CUDA 12 + cuDNN
# runtime is on the DLL path — torch's bundled libs provide exactly that, so we
# add torch/lib before importing onnxruntime (the fix that makes ORT use the
# Blackwell GPU instead of silently falling back to CPU).
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- make onnxruntime's CUDA EP find CUDA12 + cuDNN9 (bundled with torch) ------
try:
    import torch
    _tlib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_tlib):
        os.add_dll_directory(_tlib)
except Exception:
    pass
import glob
try:
    import site
    for _base in site.getsitepackages():
        for _sub in ("nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "nvidia/cublas/bin"):
            _p = os.path.join(_base, _sub)
            if os.path.isdir(_p):
                try:
                    os.add_dll_directory(_p)
                except Exception:
                    pass
except Exception:
    pass

import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWAPPER_PATHS = [
    os.path.join(PROJECT_DIR, "models", "inswapper_128_fp16.onnx"),
    os.path.join(PROJECT_DIR, "models", "inswapper_128.onnx"),
    os.path.join(PROJECT_DIR, "ai-face", "models", "inswapper_128.onnx"),
]
DET_SIZE = int(os.environ.get("AVATAR_SWAP_DET", "320"))   # smaller = faster detect
DET_EVERY = int(os.environ.get("AVATAR_SWAP_DET_EVERY", "1"))  # reuse bbox N frames
# match the swapped face's colour to the target head (LAB Reinhard) so it blends
# into the operator's real lighting instead of carrying the source clip's tone.
COLOR_MATCH = os.environ.get("AVATAR_SWAP_COLORMATCH", "1") == "1"


def _color_transfer(source, target):
    """LAB mean/std colour transfer (Deep-Live-Cam technique): recolour `source`
    to match `target`'s lighting. Both BGR uint8."""
    s = source.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    s_lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
    t_lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB)
    sm, ss = cv2.meanStdDev(s_lab)
    tm, ts = cv2.meanStdDev(t_lab)
    sm = sm.reshape(1, 1, 3); tm = tm.reshape(1, 1, 3)
    ss = np.maximum(ss.reshape(1, 1, 3), 1e-6); ts = ts.reshape(1, 1, 3)
    out = (s_lab - sm) * (ts / ss) + tm
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class FaceSwapEngine:
    """inswapper-based real-time face swap (operator head + character identity)."""

    def __init__(self, source_image_path=None):
        self.ready = False
        self.app = None
        self.swapper = None
        self.source_face = None
        self._provider = "?"
        self._n = 0
        self._cached_target = None
        self.last_found = False        # did the last swap() see a face? (chart logic)
        try:
            import insightface
            from insightface.app.common import Face
            self._Face = Face
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # only detection (per-frame target needs bbox+kps) + recognition (source
            # embedding, computed once) — skip landmark_2d_106 + genderage.
            self.app = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=providers,
                allowed_modules=["detection", "recognition"])
            self.app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
            path = next((p for p in SWAPPER_PATHS if os.path.exists(p)), None)
            if path is None:
                print("[SWAP] inswapper model not found — face swap disabled.")
                return
            self.swapper = insightface.model_zoo.get_model(path, providers=providers)
            self._provider = self.app.models["detection"].session.get_providers()[0]
            self.ready = True
            print(f"[SWAP] FaceSwapEngine ready | provider={self._provider} | model={os.path.basename(path)}")
            if source_image_path:
                self.set_source(source_image_path)
        except Exception as exc:
            print(f"[SWAP] init failed ({exc}) — face swap disabled.")

    def set_source(self, image_path):
        """Pick the CHARACTER face whose identity we paste onto the operator."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"[SWAP] source image unreadable: {image_path}")
            return False
        faces = self.app.get(img)
        if not faces:                       # tight crop -> pad and retry
            img = cv2.copyMakeBorder(img, 120, 120, 120, 120, cv2.BORDER_REPLICATE)
            faces = self.app.get(img)
        if not faces:
            print("[SWAP] no face found in source image.")
            return False
        self.source_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        print(f"[SWAP] source face set from {os.path.basename(image_path)}")
        return True

    def swap(self, frame):
        """Swap the character face onto the largest face in `frame` (the operator's
        real webcam head). Returns the composited BGR frame; passes through on no
        face so the loop never stalls."""
        if not self.ready or self.source_face is None:
            self.last_found = False
            return frame
        try:
            # per-frame TARGET needs only detection (bbox + 5 kps) — far faster than
            # full app.get(). Reuse the box for DET_EVERY frames (head barely moves).
            self._n += 1
            target = self._cached_target
            if target is None or self._n % DET_EVERY == 0:
                bboxes, kpss = self.app.det_model.detect(frame, max_num=1, metric="default")
                if bboxes is None or len(bboxes) == 0:
                    self._cached_target = None
                    return frame
                i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
                target = self._Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
                self._cached_target = target
            out = self.swapper.get(frame, target, self.source_face, paste_back=True)
            if COLOR_MATCH:
                # recolour only the swapped face box to the original lighting
                x1, y1, x2, y2 = [int(v) for v in target.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    out[y1:y2, x1:x2] = _color_transfer(out[y1:y2, x1:x2], frame[y1:y2, x1:x2])
            return out
        except Exception:
            return frame
