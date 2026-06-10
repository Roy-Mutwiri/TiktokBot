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
    sm = sm.reshape(1, 1, 3).astype(np.float32); tm = tm.reshape(1, 1, 3).astype(np.float32)
    ss = np.maximum(ss.reshape(1, 1, 3), 1e-6).astype(np.float32)
    ts = ts.reshape(1, 1, 3).astype(np.float32)
    out = ((s_lab - sm) * (ts / ss) + tm).astype(np.float32)   # keep float32 for cvtColor
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
        self._kps_euro = None          # temporal smoothing of target keypoints
        self.enhancer = None           # CodeFormer HD face enhancement (lazy)
        try:
            from one_euro import OneEuroFilter
            # one filter per kps coordinate (5 pts x 2) — kills swap jitter on move
            self._kps_euro = [[OneEuroFilter(min_cutoff=1.2, beta=0.02) for _ in range(2)]
                              for _ in range(5)]
        except Exception:
            self._kps_euro = None
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

    def set_source_from_folder(self, folder):
        """Build the CHARACTER identity from EVERY photo in `folder` (all angles).
        Robust: stores each photo's arcface embedding, drops OUTLIERS (wrong face /
        bad detection — low cosine-sim to the consensus) and weights the rest by
        detection confidence, so the averaged identity is accurate, not diluted.
        Incremental: caches per-photo embeddings so daily-added photos only cost
        their own detection. Returns the number of photos kept in the identity."""
        if not self.ready or not os.path.isdir(folder):
            return 0
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
        cache_path = os.path.join(folder, "_character_embeddings.npz")
        embs = []        # list of (512,) raw embeddings
        scores = []      # detection confidence per photo
        done = []        # filenames processed
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=True)
                embs = list(z["embs"].astype(np.float32))
                scores = list(z["scores"].astype(np.float32))
                done = list(z["files"].tolist())
            except Exception:
                embs, scores, done = [], [], []
        done_set = set(done)
        template = self.source_face
        new = 0
        for fn in files:
            if fn in done_set:
                continue
            img = cv2.imread(os.path.join(folder, fn))
            done.append(fn); done_set.add(fn)
            if img is None:
                continue
            faces = self.app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embs.append(face.embedding.astype(np.float32))
            scores.append(float(getattr(face, "det_score", 1.0)))
            new += 1
            template = face
        if not embs:
            print(f"[SWAP] no faces found in {folder}.")
            return 0
        E = np.stack(embs).astype(np.float32)            # (N,512)
        S = np.array(scores, dtype=np.float32)
        try:
            np.savez(cache_path, embs=E, scores=S,
                     files=np.array(done, dtype=object))
        except Exception:
            pass
        # robust consensus: L2-normalise, drop low-similarity outliers, weight by score
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        mean = En.mean(axis=0); mean /= (np.linalg.norm(mean) + 1e-9)
        sims = En @ mean
        keep = sims >= max(0.25, float(np.median(sims)) - 0.25)   # drop clear outliers
        if keep.sum() < 1:
            keep = np.ones(len(En), dtype=bool)
        w = (S[keep] * np.clip(sims[keep], 0, 1)).reshape(-1, 1)
        ident = (E[keep] * w).sum(axis=0) / (w.sum() + 1e-9)
        if template is not None:
            template.embedding = ident.astype(np.float32)
            self.source_face = template
        kept = int(keep.sum())
        print(f"[SWAP] character identity = {kept}/{len(embs)} photos "
              f"(dropped {len(embs)-kept} outliers, {new} new) from {os.path.basename(folder)}")
        return kept

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
                    self.last_found = False
                    return frame
                i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
                target = self._Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
                self._cached_target = target
            self.last_found = True
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
