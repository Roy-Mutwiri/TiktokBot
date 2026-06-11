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
DET_SIZE = int(os.environ.get("AVATAR_SWAP_DET", "320"))   # bigger = more reliable detect (320 = faster)
# Poisson seamlessClone gives the best forehead tone-match but costs ~25-65ms/frame
# (single-threaded CPU). Default OFF = fast feathered alpha blend (COLOR_MATCH keeps
# the tone close). Set AVATAR_SWAP_SEAMLESS=1 for the slow high-quality clone.
SEAMLESS = os.environ.get("AVATAR_SWAP_SEAMLESS", "0") == "1"
# The 256 swap is soft (looks like a flat 'mask'). A targeted unsharp on the swapped
# face restores feature/skin crispness for ~2ms (vs ~100ms for CodeFormer).
SWAP_SHARPEN = float(os.environ.get("AVATAR_SWAP_SHARPEN", "0.85"))
DET_EVERY = int(os.environ.get("AVATAR_SWAP_DET_EVERY", "1"))  # reuse bbox N frames
# ignore faces smaller than this fraction of the frame (background people/objects)
MIN_FACE_FRAC = float(os.environ.get("AVATAR_SWAP_MINFACE", "0.05"))
# gentle lighting lift: pull the face up to ~this mean brightness if it's dark
FACE_LIGHT = float(os.environ.get("AVATAR_SWAP_FACELIGHT", "0"))  # OFF: adaptive whole-frame gain flickered
# match the swapped face's colour to the target head (LAB Reinhard) so it blends
# into the operator's real lighting instead of carrying the source clip's tone.
COLOR_MATCH = os.environ.get("AVATAR_SWAP_COLORMATCH", "1") == "1"
# CodeFormer HD enhancement of the swapped face (inswapper is only 128px).
ENHANCE_SWAP = os.environ.get("AVATAR_SWAP_ENHANCE", "0") == "1"  # OFF for speed (no lags); AVATAR_SWAP_ENHANCE=1 for CodeFormer (~9fps)
# How much CodeFormer to blend in (0 = raw swap/most real, 1 = full CodeFormer/
# smoothest). Low keeps the eyes/mouth real while still adding crispness.
ENHANCE_BLEND = float(os.environ.get("AVATAR_SWAP_ENHANCE_BLEND", "0.92"))
# CLEAN BASELINE: the swap uses the inswapper model's OWN proven paste-back. All
# the experimental layers below are OFF by default — opt in one at a time.
COLOR_STRENGTH = float(os.environ.get("AVATAR_SWAP_COLORSTR", "0.0"))
AUTO_CENTER = os.environ.get("AVATAR_SWAP_CENTER", "0") == "1"          # auto-framing (off)
CENTER_Y = float(os.environ.get("AVATAR_SWAP_CENTER_Y", "0.46"))
EYE_PRESERVE = float(os.environ.get("AVATAR_SWAP_EYES", "0.62"))        # real-eye blend (livelier)
HAIR_MATCH = os.environ.get("AVATAR_SWAP_HAIRMATCH", "1") == "1"        # hair/beard recolour
HAIR_DARKEN = float(os.environ.get("AVATAR_SWAP_HAIRDARKEN", "0.55"))
# Hair/beard colour presets (recolours gray hair+beard so they match the
# character — fixes the "lost beard" + lets you pick a hair colour).
HAIR_COLORS = {
    "none":   None,
    "brown":  dict(hue=12, sat=95,  valf=0.55),
    "black":  dict(hue=0,  sat=25,  valf=0.32),
    "blonde": dict(hue=23, sat=120, valf=1.18),
    "gray":   dict(hue=0,  sat=8,   valf=0.92),
    "darkbeard": dict(hue=13, sat=75, valf=0.45),   # consistent dark beard (white-Haddan)
}
HAIR_COLOR = os.environ.get("AVATAR_SWAP_HAIRCOLOR", "brown")
BEARD_COLOR = os.environ.get("AVATAR_SWAP_BEARDCOLOR", "darkbeard")
# EYE COLOUR — recolour the iris (MediaPipe iris landmarks) to a target hue while
# KEEPING luminance (pupil stays dark, highlights bright, iris shading natural).
EYE_COLORS = {       # (HSV hue 0-180, saturation)
    "off":   None,
    "blue":  (113, 165),
    "green": (62, 150),
    "hazel": (22, 120),
    "brown": (12, 155),
    "amber": (16, 195),
    "gray":  (0, 12),
}
EYE_COLOR = os.environ.get("AVATAR_SWAP_EYECOLOR", "off")
EYE_STRENGTH = float(os.environ.get("AVATAR_SWAP_EYESTR", "0.7"))
CUSTOM_PASTE = os.environ.get("AVATAR_SWAP_CUSTOMPASTE", "0") == "1"    # custom forehead mask (off)
# Lighten the face skin toward a Caucasian tone (0 = off, ~0.6 = clearly lighter).
SKIN_LIGHTEN = float(os.environ.get("AVATAR_SWAP_SKINLIGHTEN", "0"))  # OFF: white-man SOURCE handles skin, no color hacks
# PREDICTIVE TRACKING: extrapolate face keypoints this many frames forward to
# cancel pipeline latency (the turn lag). ~1 frame; 0 = off. Too high overshoots.
PREDICT_LEAD = float(os.environ.get("AVATAR_SWAP_PREDICT", "1.25"))
# FACE LOCK: lock onto ONE identity (the operator, captured on first frame). When
# another face enters, it's ignored — only the locked face is tracked/swapped.
FACE_LOCK = os.environ.get("AVATAR_SWAP_LOCK", "1") == "1"
# AUTO-FRAME: crop/zoom to the locked face (head-and-shoulders) so the background
# and any other people are cropped OUT — focused, low-noise shot that follows you.
AUTO_FRAME = os.environ.get("AVATAR_SWAP_FRAME", "1") == "1"
FRAME_ZOOM = float(os.environ.get("AVATAR_SWAP_FRAMEZOOM", "2.2"))   # crop = this * face
LOCK_SIM = float(os.environ.get("AVATAR_SWAP_LOCKSIM", "0.25"))      # min match to keep lock


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
        self._seg = None
        self._seg_tried = False
        self._last_center = None       # last frame's face center (lock-on tracking)
        self._skin_prev = None         # previous skin mask (temporal smoothing)
        self._prev_kps = None          # previous keypoints (velocity prediction)
        self._kps_vel = None           # smoothed keypoint velocity
        self._lock_emb = None          # locked operator face embedding (identity lock)
        self._frame_box = None         # smoothed auto-frame crop box [cx,cy,side]
        self._hair_color = HAIR_COLOR  # hair recolour target (live-settable)
        self._beard_color = BEARD_COLOR  # beard recolour target (consistent dark)
        self._hairm_prev = None          # previous hair/beard mask (temporal stick)
        self._eye_color = EYE_COLOR     # iris recolour target (live-settable)
        self._eyemesh = None
        self._eyemesh_tried = False
        self._stab = 0.4                # stabilization level (0..1), live-settable
        self.last_head = None           # (cx, cy, eye_w) of the swapped head (output coords)
        self.last_mouth = None          # (cx, cy, mouth_w) of the swapped mouth (exact)
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
            except Exception as e:
                print(f"[SWAP] TRT/CUDA session build failed ({e}) — "
                      f"falling back to model_zoo.get_model (may be slower)")
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

    def _get_seg(self):
        """Lazy mediapipe selfie segmenter (person mask for hair recolour)."""
        if self._seg_tried:
            return self._seg
        self._seg_tried = True
        try:
            import mediapipe as mp
            self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._seg = None
        return self._seg

    def _match_hair(self, out, bbox):
        """Recolour GRAY hair/beard toward the character's dark hair so the result
        looks like the (dark-haired) source photos. Person-masked + gray-gated so
        it never touches skin (saturated) or the background (not the person)."""
        seg = self._get_seg()
        if seg is None:
            return out
        try:
            # the selfie-seg forward is the cost — run it every 3rd frame, reuse the
            # person mask between (hair/head moves slowly). Big speed win.
            self._hair_fn = getattr(self, "_hair_fn", 0) + 1
            if getattr(self, "_person_cache", None) is None or self._hair_fn % 2 == 0:
                res = seg.process(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                if res.segmentation_mask is None:
                    return out
                self._person_cache = res.segmentation_mask > 0.6
            person = self._person_cache
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            sat, val = hsv[:, :, 1], hsv[:, :, 2]
            # gray/white hair: very low saturation, fairly bright, on the person
            c = HAIR_COLORS.get(getattr(self, "_hair_color", HAIR_COLOR))
            if c is None:
                return out
            # CONTINUOUS gray/beard weight (0..1) — NO hard threshold, so it doesn't
            # flip pixels in/out frame-to-frame (that binary flicker = "not consistent").
            gw = np.clip((68.0 - sat) / 48.0, 0, 1)            # low saturation = gray hair/beard
            gw *= np.clip((val - 38.0) / 55.0, 0, 1)           # not the deepest shadow
            gw *= person.astype(np.float32)
            # HEAD band: hair above + beard well below the face box (catch the jaw).
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh = max(1, y2 - y1); fw = max(1, x2 - x1)
            band = np.zeros(out.shape[:2], np.float32)
            band[max(0, y1 - int(fh * 1.1)):min(out.shape[0], y2 + int(fh * 0.7)),
                 max(0, x1 - int(fw * 0.45)):min(out.shape[1], x2 + int(fw * 0.45))] = 1.0
            m = cv2.GaussianBlur(gw * band, (0, 0), 3.0)
            # TEMPORAL smoothing of the recolour mask — stops the recoloured region
            # flickering/shifting as you move, so the beard tint STICKS to the beard.
            if getattr(self, "_hairm_prev", None) is not None and self._hairm_prev.shape == m.shape:
                m = 0.45 * m + 0.55 * self._hairm_prev
            self._hairm_prev = m
            # split: HAIR (top) keeps the chosen hair colour; BEARD (lower face/jaw)
            # is recoloured to ONE consistent dark tone matching the swap's beard, so
            # the gray operator-beard at the edges no longer two-tones with it.
            beard_y = max(0, y1 + int(fh * 0.55))
            zb = np.zeros_like(m); zb[beard_y:, :] = 1.0
            mh, mb = m * (1.0 - zb), m * zb
            bd = HAIR_COLORS.get(getattr(self, "_beard_color", "darkbeard"), c)

            def _tint(col, mk):
                hsv[:, :, 0] = hsv[:, :, 0] * (1 - mk) + col["hue"] * mk
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - mk * 0.5) + col["sat"] * mk * 0.5, 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 - mk * (1 - col["valf"])), 0, 255)
            _tint(c, mh)
            _tint(bd or c, mb)
            return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        except Exception:
            return out

    def set_source_blend(self, folder_a, folder_b, w):
        """Set the source identity to a BLEND of two folders' averaged embeddings
        (w*A + (1-w)*B), in arcface identity space — e.g. Haddan blended toward the
        white-man identity makes a 'white Haddan' (model-driven, no colour hacks)."""
        if not self.set_source_from_folder(folder_a):
            return 0
        ea = self.source_face.embedding.astype(np.float32).copy()
        if not self.set_source_from_folder(folder_b):   # source_face now = B's template
            return 0
        eb = self.source_face.embedding.astype(np.float32).copy()
        self.source_face.embedding = (w * ea + (1.0 - w) * eb).astype(np.float32)
        print(f"[SWAP] blended identity {os.path.basename(folder_a)}*{w:.2f} + "
              f"{os.path.basename(folder_b)}*{1-w:.2f}")
        return 1

    def restore_face(self, frame):
        """No-op now: CodeFormer runs IN-SWAP again (cached, ~9fps) instead of an
        every-frame after-mouth pass (~6fps, too laggy). Kept so the loop call is
        harmless. Set AVATAR_SWAP_ENHANCE=0 + this back on for the after-mouth path."""
        return frame

    def set_stabilization(self, level):
        """0 = responsive, 1 = max stability. Heavily smooths the face keypoints +
        framing and eases off prediction so the swapped face is rock-steady."""
        level = max(0.0, min(1.0, float(level)))
        self._stab = level
        mc = 4.0 - 3.6 * level          # 1€ min_cutoff 4.0 (snappy) -> 0.4 (very smooth)
        bt = 0.10 - 0.088 * level       # 1€ beta 0.10 -> 0.012
        if self._kps_euro is not None:
            for j in range(5):
                for k in range(2):
                    self._kps_euro[j][k].min_cutoff = mc
                    self._kps_euro[j][k].beta = bt

    def _get_eyemesh(self):
        if self._eyemesh_tried:
            return self._eyemesh
        self._eyemesh_tried = True
        try:
            import mediapipe as mp
            self._eyemesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)
        except Exception:
            self._eyemesh = None
        return self._eyemesh

    def _recolor_eyes(self, out):
        """Recolour the IRIS to the chosen eye colour. MediaPipe iris landmarks give
        the iris circle; we change hue/sat but KEEP value, so the pupil stays dark,
        catch-light stays bright and the iris shading looks natural."""
        c = EYE_COLORS.get(getattr(self, "_eye_color", "off"))
        if c is None:
            return out
        mesh = self._get_eyemesh()
        if mesh is None:
            return out
        try:
            H, W = out.shape[:2]
            # FaceMesh (refine) is the cost — run it every 2nd frame, reuse the iris
            # mask between (eyes move little frame-to-frame).
            self._eye_fn = getattr(self, "_eye_fn", 0) + 1
            if getattr(self, "_iris_cache", None) is None or self._eye_fn % 2 == 0 \
                    or self._iris_cache.shape[:2] != (H, W):
                res = mesh.process(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks:
                    return out
                lm = res.multi_face_landmarks[0].landmark
                mk = np.zeros((H, W), np.float32)
                for ring in ([469, 470, 471, 472], [474, 475, 476, 477]):
                    pts = np.array([[lm[i].x * W, lm[i].y * H] for i in ring], np.float32)
                    (cx, cy), r = cv2.minEnclosingCircle(pts)
                    if r > 1:
                        cv2.circle(mk, (int(cx), int(cy)), int(r * 1.05), 1.0, -1)
                self._iris_cache = cv2.GaussianBlur(mk, (0, 0), 1.5)
            mask = self._iris_cache
            if mask.max() < 0.5:
                return out
            mask = mask * EYE_STRENGTH
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            th, ts = c
            hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask) + th * mask          # target hue
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask) + ts * mask          # target saturation
            # value untouched -> pupil dark, highlights bright, natural shading
            return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        except Exception:
            return out

    def _embed(self, frame, kps):
        """Arcface embedding for a face (by its 5 kps) — used to identity-lock onto
        the operator and ignore any other face that enters the frame."""
        try:
            f = self._Face(bbox=None, kps=kps.astype(np.float32), det_score=1.0)
            e = self.app.models["recognition"].get(frame, f)
            n = np.linalg.norm(e)
            return e / n if n > 1e-6 else None
        except Exception:
            return None

    def _lighten_skin(self, out):
        """Shift ALL visible skin (face, NECK, HANDS) toward a lighter / less-warm
        Caucasian tone — consistent body, no white-face/dark-neck mismatch. YCrCb
        skin-gated + saturation gate so the red shirt, beard, eyes, hair and
        background are left untouched."""
        try:
            f = out.astype(np.float32)
            ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            cr, cb = ycc[:, :, 1], ycc[:, :, 2]
            sat, val = hsv[:, :, 1], hsv[:, :, 2]
            # skin: classic YCrCb range, moderate saturation (the bright RED SHIRT is
            # far more saturated), reasonably bright (not the dark beard/shadows).
            # WIDE gate so even darker / dim-webcam skin is caught (val down to 35).
            skin = ((cr > 130) & (cr < 182) & (cb > 75) & (cb < 135) &
                    (sat > 18) & (sat < 165) & (val > 35)).astype(np.float32)
            skin = cv2.GaussianBlur(skin, (0, 0), 3.0)
            # TEMPORAL smoothing — stops gate-boundary pixels flickering in/out
            # frame-to-frame (the "not consistent" tone wobble).
            if getattr(self, "_skin_prev", None) is not None and self._skin_prev.shape == skin.shape:
                skin = 0.5 * skin + 0.5 * self._skin_prev
            self._skin_prev = skin
            skin = skin[:, :, None] * SKIN_LIGHTEN
            # LAB tone shift toward a natural LIGHT-Caucasian skin (peachy, NOT the
            # gray/blue cast the old de-warm gave): strong lift of L, pull a/b toward
            # a light warm-pink target. Keeps texture; consistent face/neck/hands.
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:, :, 0] = lab[:, :, 0] + (238.0 - lab[:, :, 0]) * 0.45   # strong lighten
            lab[:, :, 1] = lab[:, :, 1] * 0.55 + 143.0 * 0.45             # a -> light red
            lab[:, :, 2] = lab[:, :, 2] * 0.55 + 150.0 * 0.45             # b -> light yellow
            toned = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            return np.clip(f * (1 - skin) + toned.astype(np.float32) * skin, 0, 255).astype(np.uint8)
        except Exception:
            return out

    def _paste_seamless(self, frame, target):
        """Paste the swapped face with Poisson seamless cloning so the boundary
        (esp. the FOREHEAD) matches the surrounding skin tone/lighting — no visible
        'mask line'. Falls back to a feathered alpha blend if seamlessClone fails."""
        H, W = frame.shape[:2]
        bgr_fake, M = self.swapper.get(frame, target, self.source_face, paste_back=False)
        S = bgr_fake.shape[0]
        IM = cv2.invertAffineTransform(M)
        fake_full = cv2.warpAffine(bgr_fake, IM, (W, H))
        # TALL face-oval mask that reaches UP to the hairline so MORE of the
        # character's forehead is pasted (boundary hidden near the hair), while the
        # sides stay inside the face (no ears/bg). Replaces the all-sides erosion
        # that was cutting the forehead.
        white = np.zeros((S, S), np.uint8)
        cv2.ellipse(white, (S // 2, int(S * 0.49)),
                    (int(S * 0.40), int(S * 0.49)), 0, 0, 360, 255, -1)  # forehead, inside crop
        mask = cv2.warpAffine(white, IM, (W, H))
        # VALID swap region: warpAffine leaves BLACK outside the crop quad. Clip the
        # mask (and its feather) to the eroded valid region so that black border can
        # NEVER blend onto the face (it was bleeding in as black lines, esp. on turns).
        valid = cv2.warpAffine(np.full((S, S), 255, np.uint8), IM, (W, H))
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        mask = cv2.bitwise_and(mask, valid)
        ys, xs = np.where(mask > 10)
        if len(xs) == 0:
            return None
        if SEAMLESS:                                       # slow, best tone match (opt-in)
            cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
            mask[:3, :] = 0; mask[-3:, :] = 0; mask[:, :3] = 0; mask[:, -3:] = 0
            try:
                return cv2.seamlessClone(fake_full, frame, mask, (cx, cy), cv2.NORMAL_CLONE)
            except Exception:
                pass
        # FAST default: feathered alpha blend, cropped to the face bbox so the blur +
        # composite only touch the masked region (~1-2ms vs ~25-65ms for Poisson).
        pad = max(16, (xs.max() - xs.min()) // 8)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        sub = mask[y0:y1, x0:x1].astype(np.float32)
        # erode then heavily blur -> the alpha transition spreads over many pixels so
        # the boundary (esp. the forehead) feathers away with no visible mask line.
        sub = cv2.erode(sub, np.ones((max(3, sub.shape[0] // 22),) * 2, np.uint8))
        m = cv2.GaussianBlur(sub, (0, 0), max(8, sub.shape[0] // 9)) / 255.0
        # clip the FEATHERED alpha to valid swap pixels too (the blur spreads outward
        # past the crop edge -> would re-introduce the black border). Belt and braces.
        m = (m * (valid[y0:y1, x0:x1] / 255.0))[:, :, None]
        fake_roi = fake_full[y0:y1, x0:x1].astype(np.float32)
        frame_roi = frame[y0:y1, x0:x1].astype(np.float32)
        # CLARITY: unsharp-mask the soft 256 swap so the face has crisp features/skin
        # texture instead of the flat 'mask' look (cheap stand-in for CodeFormer).
        if SWAP_SHARPEN > 0:
            blur = cv2.GaussianBlur(fake_roi, (0, 0), 1.4)
            fake_roi = np.clip(fake_roi * (1 + SWAP_SHARPEN) - blur * SWAP_SHARPEN, 0, 255)
        # CHEAP TONE MATCH (kills the forehead skin-tone seam without Poisson): shift
        # the swap's mean toward the surrounding real skin's mean, clipped so the
        # character stays light. ~1ms vs ~65ms for seamlessClone.
        if COLOR_MATCH:
            core = sub > 200
            if int(core.sum()) > 80:
                off = np.clip(frame_roi[core].mean(0) - fake_roi[core].mean(0), -24, 24)
                fake_roi = np.clip(fake_roi + off, 0, 255)
        out = frame.copy()
        out[y0:y1, x0:x1] = (frame_roi * (1 - m) + fake_roi * m).astype(np.uint8)
        return out

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
            # detect ALL faces, then lock onto YOURS: largest + most central, with
            # tiny background faces gated out + a bias toward last frame's position
            # (so it never jumps to a face/object in the background).
            bboxes, kpss = self.app.det_model.detect(frame, max_num=0, metric="default")
            if bboxes is None or len(bboxes) == 0:
                self.last_found = False
                self._last_center = None
                return frame
            cx0, cy0 = W / 2.0, H / 2.0
            cands = [k for k in range(len(bboxes))
                     if max(bboxes[k, 2] - bboxes[k, 0],
                            bboxes[k, 3] - bboxes[k, 1]) >= W * MIN_FACE_FRAC]
            if not cands:
                self.last_found = False
                self._last_center = None
                return frame
            i = -1
            # IDENTITY LOCK: if a 2nd+ face is present, pick the one matching the
            # LOCKED operator embedding (so an intruder face is ignored, not swapped).
            if FACE_LOCK and self._lock_emb is not None and len(cands) > 1:
                best_sim = -1.0
                for k in cands:
                    e = self._embed(frame, kpss[k])
                    if e is None:
                        continue
                    sim = float(e @ self._lock_emb)
                    if sim > best_sim:
                        best_sim, i = sim, k
                if best_sim < LOCK_SIM:          # lock lost -> fall back to spatial
                    i = -1
            if i < 0:                            # spatial pick (largest+central+last)
                best = -1.0
                for k in cands:
                    x1, y1, x2, y2 = bboxes[k, :4]
                    area = (x2 - x1) * (y2 - y1)
                    fcx, fcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    dist = ((fcx - cx0) ** 2 + (fcy - cy0) ** 2) ** 0.5 / cx0
                    sc = area * (1.0 - 0.4 * min(dist, 1.0))
                    if self._last_center is not None:
                        ld = ((fcx - self._last_center[0]) ** 2 +
                              (fcy - self._last_center[1]) ** 2) ** 0.5 / cx0
                        sc *= (1.0 - 0.5 * min(ld, 1.0))
                    if sc > best:
                        best, i = sc, k
            kps = kpss[i].astype(np.float32)
            bbox = bboxes[i, :4].astype(np.float32)
            self._last_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            # capture the operator's identity once (lock onto the first face seen)
            if FACE_LOCK and self._lock_emb is None:
                self._lock_emb = self._embed(frame, kps)
            # AUTO-FRAME: crop/zoom to the locked face (head & shoulders) so the
            # background AND any other person are cropped OUT. Smoothed framing.
            if AUTO_FRAME:
                fw = bbox[2] - bbox[0]; fh = bbox[3] - bbox[1]
                fcx = (bbox[0] + bbox[2]) / 2.0
                fcy = (bbox[1] + bbox[3]) / 2.0 + fh * 0.15      # bias down for shoulders
                side = min(max(fw, fh) * FRAME_ZOOM, float(min(W, H)))
                box = np.array([fcx, fcy, side], np.float32)
                a = 0.18 * (1.0 - 0.7 * self._stab)          # more stab = steadier framing
                self._frame_box = (box if self._frame_box is None
                                   else (1.0 - a) * self._frame_box + a * box)
                fcx, fcy, side = self._frame_box
                x1c = float(np.clip(fcx - side / 2, 0, W - side))
                y1c = float(np.clip(fcy - side / 2, 0, H - side))
                crop = frame[int(y1c):int(y1c + side), int(x1c):int(x1c + side)]
                if crop.size > 0:
                    frame = cv2.resize(crop, (W, H))
                    s = W / side
                    kps = (kps - np.array([x1c, y1c], np.float32)) * s
                    bbox = (bbox - np.array([x1c, y1c, x1c, y1c], np.float32)) * s
            # TEMPORAL SMOOTHING of the 5 keypoints — no jitter/slide on movement.
            t = self._n / 30.0
            if self._kps_euro is not None:
                for j in range(5):
                    kps[j, 0] = self._kps_euro[j][0](float(kps[j, 0]), t)
                    kps[j, 1] = self._kps_euro[j][1](float(kps[j, 1]), t)
            # PREDICTIVE TRACKING: the swap appears ~1 frame behind your head because
            # of detect+swap+render latency (the "lag on turn"). Extrapolate the
            # keypoints FORWARD by the latency using velocity, so the swapped face
            # lands where your head IS, not where it WAS. Clamped (no overshoot).
            if PREDICT_LEAD > 0.0:
                if self._prev_kps is not None and self._prev_kps.shape == kps.shape:
                    vel = kps - self._prev_kps
                    self._kps_vel = (vel if self._kps_vel is None
                                     else 0.6 * self._kps_vel + 0.4 * vel)
                    eye_d = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
                    # ease off prediction at high stabilization (steadier face)
                    lead = self._kps_vel * PREDICT_LEAD * (1.0 - 0.6 * self._stab)
                    n = np.linalg.norm(lead, axis=1, keepdims=True)
                    lead = lead * np.minimum(1.0, (eye_d * 0.5) / (n + 1e-6))  # cap
                    self._prev_kps = kps.copy()
                    kps = kps + lead
                else:
                    self._prev_kps = kps.copy()
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
                bbox = bbox + np.array([dx, dy, dx, dy], np.float32)
            self.last_found = True
            # publish the head location (output coords) so the background composite
            # can PROTECT it — the bg cut never intrudes on the face.
            ec = kps.mean(axis=0)
            ew = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
            self.last_head = (float(ec[0]), float(ec[1]), ew)
            # publish the EXACT mouth location (kps[3]/[4] = mouth corners) so the
            # mouth-sync bbox uses the swap's own geometry, not a re-detection that
            # mis-places it on this character (causing the doubled-mouth look).
            mc = (kps[3] + kps[4]) / 2.0
            mw = float(np.linalg.norm(kps[3] - kps[4])) + 1e-6
            self.last_mouth = (float(mc[0]), float(mc[1]), mw)
            target = self._Face(bbox=bbox, kps=kps, det_score=bboxes[i, 4])
            # SEAMLESS paste (Poisson) — boundary matches skin tone, no forehead line.
            out = self._paste_seamless(frame, target)
            if out is None:
                out = self.swapper.get(frame, target, self.source_face, paste_back=True)
            # GENTLE LIGHTING: if the face is dark, lift it toward FACE_LIGHT (soft,
            # face-region only) so you're not under-lit. Never darkens.
            if FACE_LIGHT > 0:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    reg = out[y1:y2, x1:x2]
                    mean = float(reg.reshape(-1, 3).mean())
                    if 5 < mean < FACE_LIGHT:
                        gain = min(1.6, FACE_LIGHT / mean)
                        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            # lighten ALL skin (face, neck, hands) toward a Caucasian tone
            if SKIN_LIGHTEN > 0.0:
                out = self._lighten_skin(out)
            # CodeFormer HD enhancement IN-SWAP (cached every 2 frames = ~9fps, no
            # whole-face stutter since the composite is fresh each frame). The mouth
            # (added later) gets a fast per-frame sharpen instead of CodeFormer — the
            # every-frame after-mouth restore was sharp but only ~6fps (too laggy).
            if ENHANCE_SWAP:
                if self.enhancer is None and not self._enh_tried:
                    self._enh_tried = True
                    try:
                        from face_restore_engine import FaceRestoreEngine
                        self.enhancer = FaceRestoreEngine()
                        print("[SWAP] CodeFormer face enhancement ON (in-swap)")
                    except Exception as exc:
                        print(f"[SWAP] enhancer unavailable ({exc})")
                if self.enhancer is not None and getattr(self.enhancer, "ready", False):
                    enh = self.enhancer.process_frame(out)
                    out = cv2.addWeighted(out, 1.0 - ENHANCE_BLEND, enh, ENHANCE_BLEND, 0)
            # EYE PRESERVATION: paste YOUR real eyes (real catchlights, gaze, blinks)
            # back over Haddan's face. AI source photos give dead eyes; your live
            # eyes are what make them look human. kps[0]/kps[1] = right/left eye.
            if EYE_PRESERVE > 0.0:
                try:
                    eye_d = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
                    rad = max(6, int(eye_d * 0.40))
                    # work in a small ROI around the two eyes (was a full-512 blur+blend ~27ms)
                    pad = rad * 2
                    xs = [kps[0][0], kps[1][0]]; ys = [kps[0][1], kps[1][1]]
                    x0 = max(0, int(min(xs) - pad)); x1 = min(W, int(max(xs) + pad))
                    y0 = max(0, int(min(ys) - pad)); y1 = min(H, int(max(ys) + pad))
                    if x1 > x0 and y1 > y0:
                        em = np.zeros((y1 - y0, x1 - x0), np.float32)
                        for ex, ey in (kps[0], kps[1]):
                            cv2.circle(em, (int(ex) - x0, int(ey) - y0), rad, 1.0, -1)
                        em = cv2.GaussianBlur(em, (0, 0), rad * 0.5)[:, :, None] * EYE_PRESERVE
                        roi = (out[y0:y1, x0:x1].astype(np.float32) * (1 - em)
                               + frame[y0:y1, x0:x1].astype(np.float32) * em)
                        out[y0:y1, x0:x1] = roi.astype(np.uint8)
                except Exception:
                    pass
            # recolour gray hair/beard -> match the character
            if HAIR_MATCH:
                out = self._match_hair(out, bbox)
            # recolour the iris to the chosen eye colour
            if getattr(self, "_eye_color", "off") != "off":
                out = self._recolor_eyes(out)
            return out
        except Exception as e:
            self._swap_errs = getattr(self, "_swap_errs", 0) + 1
            if self._swap_errs <= 3:
                import traceback
                print(f"[SWAP] per-frame swap error ({e}) — passing raw frame")
                traceback.print_exc()
            return frame
