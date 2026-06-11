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

# --- make onnxruntime's CUDA + TensorRT EPs find their DLLs (CUDA12/cuDNN9 from
# torch, nvinfer from tensorrt_libs). MUST run before onnxruntime is imported, and
# the TensorRT provider DLL resolves nvinfer via PATH (not add_dll_directory), so
# we set BOTH. ------------------------------------------------------------------
import glob
_dll_dirs = []
try:
    import torch
    _dll_dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    pass
try:
    import site
    for _base in site.getsitepackages():
        for _sub in ("nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "nvidia/cublas/bin",
                     "tensorrt_libs"):
            _dll_dirs.append(os.path.join(_base, _sub))
except Exception:
    pass
_dll_dirs = [d for d in _dll_dirs if os.path.isdir(d)]
if _dll_dirs:
    os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")
    for _d in _dll_dirs:
        try:
            os.add_dll_directory(_d)
        except Exception:
            pass

import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# HQ = ReSwapper-256 (highest fidelity). Now the DEFAULT because TensorRT fp16
# makes the 256 forward ~20ms (real-time). Falls back to 128 if the 256 model is
# absent. Set AVATAR_SWAP_HQ=0 to force the lighter 128 model.
_HQ = os.environ.get("AVATAR_SWAP_HQ", "1") == "1"
SWAPPER_PATHS = ([os.path.join(PROJECT_DIR, "models", "reswapper_256.onnx")] if _HQ else []) + [
    os.path.join(PROJECT_DIR, "models", "inswapper_128_fp16.onnx"),
    os.path.join(PROJECT_DIR, "models", "inswapper_128.onnx"),
    os.path.join(PROJECT_DIR, "ai-face", "models", "inswapper_128.onnx"),
    os.path.join(PROJECT_DIR, "models", "reswapper_256.onnx"),
]
DET_SIZE = int(os.environ.get("AVATAR_SWAP_DET", "320"))   # smaller = faster detect
DET_EVERY = int(os.environ.get("AVATAR_SWAP_DET_EVERY", "1"))  # reuse bbox N frames
# match the swapped face's colour to the target head (LAB Reinhard) so it blends
# into the operator's real lighting instead of carrying the source clip's tone.
COLOR_MATCH = os.environ.get("AVATAR_SWAP_COLORMATCH", "1") == "1"
# CodeFormer HD enhancement of the swapped face (inswapper is only 128px).
ENHANCE_SWAP = os.environ.get("AVATAR_SWAP_ENHANCE", "1") == "1"
# How much CodeFormer to blend in (0 = raw swap/most real, 1 = full CodeFormer/
# smoothest). Low keeps the eyes/mouth real while still adding crispness.
ENHANCE_BLEND = float(os.environ.get("AVATAR_SWAP_ENHANCE_BLEND", "0.8"))
COLOR_STRENGTH = float(os.environ.get("AVATAR_SWAP_COLORSTR", "0.45"))  # gentle = keep Haddan skin
AUTO_CENTER = os.environ.get("AVATAR_SWAP_CENTER", "1") == "1"          # auto-framing
CENTER_Y = float(os.environ.get("AVATAR_SWAP_CENTER_Y", "0.46"))       # target face y (0..1)


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
        self._enh_tried = False
        try:
            from one_euro import OneEuroFilter
            # one filter per kps coordinate (5 pts x 2) — kills swap jitter on move
            # light smoothing only — enough to de-jitter, NOT enough to lag the
            # eyes/turns (over-smoothing made tracking feel dead).
            self._kps_euro = [[OneEuroFilter(min_cutoff=4.0, beta=0.10) for _ in range(2)]
                              for _ in range(5)]
            self._shift_euro = [OneEuroFilter(min_cutoff=0.8, beta=0.02) for _ in range(2)]
        except Exception:
            self._kps_euro = None
            self._shift_euro = None
        self._amask = None
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
            # Force the INSwapper class with a GPU session (the model_zoo router
            # mis-detects some ReSwapper exports as a recognition model). For the
            # heavy 256 model, run it on TensorRT (fp16) — ~4x faster than CUDA and,
            # unlike naive fp16 conversion, mathematically correct (no garbage).
            try:
                import onnxruntime
                from insightface.model_zoo.inswapper import INSwapper
                sw_providers = providers
                if "256" in os.path.basename(path) and \
                        "TensorrtExecutionProvider" in onnxruntime.get_available_providers():
                    _trt_cache = os.path.join(PROJECT_DIR, "models", "trt_cache")
                    os.makedirs(_trt_cache, exist_ok=True)
                    sw_providers = [("TensorrtExecutionProvider",
                                     {"trt_fp16_enable": True,
                                      "trt_engine_cache_enable": True,
                                      "trt_engine_cache_path": _trt_cache})] + providers
                sess = onnxruntime.InferenceSession(path, providers=sw_providers)
                self.swapper = INSwapper(model_file=path, session=sess)
                self._swap_provider = sess.get_providers()[0]
            except Exception:
                self.swapper = insightface.model_zoo.get_model(path, providers=providers)
                self._swap_provider = "?"
            self._provider = self.app.models["detection"].session.get_providers()[0]
            self.ready = True
            sz = getattr(self.swapper, "input_size", ("?",))[0]
            print(f"[SWAP] FaceSwapEngine ready | detect={self._provider} | "
                  f"swap={getattr(self, '_swap_provider', '?')} | "
                  f"model={os.path.basename(path)} ({sz}px)")
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

    def _aligned_mask(self, S):
        """Cached feathered ellipse mask in the aligned (SxS) face space, extended
        UP toward the forehead so the swap covers more of the hairline (pushes the
        boundary above the brows) and blends softly into the real hair."""
        if getattr(self, "_amask", None) is not None and self._amask.shape[0] == S:
            return self._amask
        m = np.zeros((S, S), np.float32)
        # natural face oval — only a little forehead extension (over-extending
        # clipped the swap onto hair/background at turns = the hairline+turn errors).
        cv2.ellipse(m, (S // 2, int(S * 0.52)),
                    (int(S * 0.40), int(S * 0.48)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), S * 0.06)          # soft edge / hairline feather
        self._amask = m
        return m

    def swap(self, frame):
        """Swap the character face onto the largest face in `frame` (your real
        webcam head). Auto-centers the face, smooths keypoints, custom forehead
        paste, gentle skin-tone match, CodeFormer HD. Passes through on no face."""
        if not self.ready or self.source_face is None:
            self.last_found = False
            return frame
        try:
            H, W = frame.shape[:2]
            self._n += 1
            bboxes, kpss = self.app.det_model.detect(frame, max_num=1, metric="default")
            if bboxes is None or len(bboxes) == 0:
                self.last_found = False
                return frame
            i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
            kps = kpss[i].astype(np.float32)
            bbox = bboxes[i, :4].astype(np.float32)
            # TEMPORAL SMOOTHING of the 5 keypoints — no jitter/slide on movement.
            if self._kps_euro is not None:
                t = self._n / 30.0
                for j in range(5):
                    kps[j, 0] = self._kps_euro[j][0](float(kps[j, 0]), t)
                    kps[j, 1] = self._kps_euro[j][1](float(kps[j, 1]), t)
            # AUTO-CENTER (auto-framing): shift the head to frame centre.
            if AUTO_CENTER:
                cx = (bbox[0] + bbox[2]) / 2.0; cy = (bbox[1] + bbox[3]) / 2.0
                dx = float(np.clip(W * 0.5 - cx, -W * 0.3, W * 0.3))
                dy = float(np.clip(H * CENTER_Y - cy, -H * 0.25, H * 0.25))
                if self._shift_euro is not None:           # smooth the framing
                    dx = self._shift_euro[0](dx, t); dy = self._shift_euro[1](dy, t)
                Msh = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
                frame = cv2.warpAffine(frame, Msh, (W, H), borderMode=cv2.BORDER_REPLICATE)
                kps = kps + np.array([dx, dy], np.float32)
            self.last_found = True
            target = self._Face(bbox=bbox, kps=kps, det_score=bboxes[i, 4])
            # SWAP (aligned) + CUSTOM forehead paste-back -------------------------
            bgr_fake, M = self.swapper.get(frame, target, self.source_face, paste_back=False)
            S = bgr_fake.shape[0]
            if COLOR_MATCH:                                # GENTLE skin-tone match
                aimg = cv2.warpAffine(frame, M, (S, S))
                matched = _color_transfer(bgr_fake, aimg)
                bgr_fake = cv2.addWeighted(bgr_fake, 1.0 - COLOR_STRENGTH,
                                           matched, COLOR_STRENGTH, 0)
            mask = self._aligned_mask(S)
            IM = cv2.invertAffineTransform(M)
            fake_full = cv2.warpAffine(bgr_fake, IM, (W, H))
            mask_full = cv2.warpAffine(mask, IM, (W, H))[:, :, None]
            out = (frame.astype(np.float32) * (1 - mask_full)
                   + fake_full.astype(np.float32) * mask_full).astype(np.uint8)
            # CodeFormer HD enhancement (inswapper is only 128px -> crisp + exact).
            if ENHANCE_SWAP:
                if self.enhancer is None and not self._enh_tried:
                    self._enh_tried = True
                    try:
                        from face_restore_engine import FaceRestoreEngine
                        self.enhancer = FaceRestoreEngine()
                        print("[SWAP] CodeFormer face enhancement ON")
                    except Exception as exc:
                        print(f"[SWAP] enhancer unavailable ({exc})")
                if self.enhancer is not None and getattr(self.enhancer, "ready", False):
                    enh = self.enhancer.process_frame(out)
                    # BLEND back, don't replace — full CodeFormer over-smooths the
                    # eyes/mouth (dead/fake look). Keep the swap's real expression
                    # detail and just add sharpness.
                    out = cv2.addWeighted(out, 1.0 - ENHANCE_BLEND, enh, ENHANCE_BLEND, 0)
            return out
        except Exception:
            return frame
