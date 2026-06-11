# =============================================================================
# engines/face_restore_engine.py
# -----------------------------------------------------------------------------
# Photoreal face restoration that runs AFTER Wav2Lip. Wav2Lip synthesises the
# mouth at 96x96 and upscales it, so the lower face comes out soft/blurry — the
# single biggest "this is AI" tell. This engine detects the face, restores it to
# crisp, photoreal detail with a HuggingFace model, and pastes it back over the
# blurred-edge background so only the head is sharpened.
#
# Primary model:  CodeFormer  (weights pulled from HF: zhangbo2008/codeformer)
#                 - identity-preserving, controllable fidelity (w), less of the
#                   plastic/over-smoothed look GFPGAN can give.
# Fallback model: GFPGAN v1.4 (already on disk) if CodeFormer fails to load.
#
# Detection / alignment / parsing-mask paste-back are handled by facexlib's
# FaceRestoreHelper (the same machinery CodeFormer's own demo uses).
#
# Perf: the CodeFormer forward on a fixed 512x512 aligned crop is the cost
# centre (~30-50ms on the RTX 5060 Ti). It is run every RESTORE_INTERVAL frames
# and the result cached/reused between, matching the LP_INTERVAL pattern in the
# main loop. RESTORE_INTERVAL=1 = sharpest lip-sync; higher = more fps.
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CF_ARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codeformer_arch")

# CodeFormer weights (downloaded from HF into ai-face/models/).
CODEFORMER_PATH = os.path.join(PROJECT_DIR, "ai-face", "models", "codeformer.pth")
GFPGAN_PATH = os.path.join(PROJECT_DIR, "ai-face", "models", "GFPGANv1.4.pth")

# Restoration fidelity (CodeFormer's `w`): 0 = max quality/most invented detail,
# 1 = max fidelity to the (blurry) input. ~0.7 keeps identity while still
# cleaning up the Wav2Lip mouth. Tunable via env AVATAR_RESTORE_FIDELITY.
FIDELITY = float(os.environ.get("AVATAR_RESTORE_FIDELITY", "0.5"))

# Run the (costly) restore every Nth frame, reuse the cached result between.
# 1 = every frame (sharpest, slowest). Tunable via env AVATAR_RESTORE_INTERVAL.
RESTORE_INTERVAL = int(os.environ.get("AVATAR_RESTORE_INTERVAL", "1"))  # after-mouth restore: every frame (no stutter)

# Re-run face DETECTION (retinaface, ~29ms) only every Nth restore — the AI head
# is near-static, so the alignment transform barely moves. Between detects we
# re-use the cached affine matrix + paste mask (warp is ~1ms), which removes the
# detector and parse-net cost from most frames.
DETECT_INTERVAL = 12

# bf16 autocast: ~1.5x faster, and (unlike fp16) its exponent range does NOT
# overflow the StyleGAN/VQGAN decoder into garbage. Set AVATAR_RESTORE_BF16=0
# to force fp32 if a card misbehaves.
USE_BF16 = os.environ.get("AVATAR_RESTORE_BF16", "1") == "1"


class FaceRestoreEngine:
    """Resident face restorer (CodeFormer, GFPGAN fallback) applied per frame."""

    def __init__(self):
        self.ready = False
        self.fallback = True          # True until a model loads
        self.backend = "none"
        self._net = None              # CodeFormer net (or GFPGANer)
        self._helper = None           # facexlib FaceRestoreHelper (CodeFormer path)
        self._torch = None
        self.device = "cpu"
        self._frame_counter = 0
        self._cached = None
        self._err_printed = False
        # Cached alignment geometry (refreshed every DETECT_INTERVAL restores).
        self._affine = None          # 2x3 frame -> 512 aligned crop
        self._inv_affine = None      # 2x3 aligned crop -> frame
        self._paste_mask = None      # HxWx1 feathered blend mask in frame space
        self._restore_calls = 0

        try:
            self._load()
        except Exception as exc:
            print(f"[RESTORE] init failed ({exc}) — restoration disabled (face passes through).")

    # -------------------------------------------------------------------------
    @property
    def ok(self):
        """True when a restoration model is loaded and usable."""
        return self.ready

    def restore(self, frame):
        """Alias for process_frame (matches the main loop's restorer API)."""
        return self.process_frame(frame)

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Report restoration readiness. Returns (ok, message)."""
        if not self.ready:
            return True, "FALLBACK — no face restoration (model missing)."
        return True, (f"{self.backend} restore active "
                      f"(fidelity {FIDELITY}, every {RESTORE_INTERVAL} frame(s), "
                      f"{'bf16' if USE_BF16 else 'fp32'}).")

    # -------------------------------------------------------------------------
    def _load(self):
        """Load CodeFormer (preferred) or fall back to GFPGAN."""
        import torch
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CodeFormer's aligned crop is a FIXED 512x512 shape, so cudnn autotune
        # runs once and never re-tunes.
        torch.backends.cudnn.benchmark = True

        if self._try_load_codeformer():
            self.backend = "CodeFormer"
        elif self._try_load_gfpgan():
            self.backend = "GFPGAN"
        else:
            self.ready = False
            self.fallback = True
            print("[RESTORE] No restoration model could be loaded.")
            return

        self.ready = True
        self.fallback = False
        if self.device == "cuda":
            print(f"[RESTORE] GPU memory: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
        print(f"[RESTORE] Ready — {self.backend} photoreal restoration enabled.")

    def _try_load_codeformer(self):
        """Load the CodeFormer net + facexlib helper. Returns True on success."""
        if not os.path.exists(CODEFORMER_PATH):
            print(f"[RESTORE] CodeFormer weights not found at {CODEFORMER_PATH}.")
            return False
        try:
            torch = self._torch
            if _CF_ARCH_DIR not in sys.path:
                sys.path.insert(0, _CF_ARCH_DIR)
            from codeformer_arch import CodeFormer
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                             connect_list=["32", "64", "128", "256"]).to(self.device)
            ck = torch.load(CODEFORMER_PATH, map_location="cpu", weights_only=False)
            sd = ck["params_ema"] if "params_ema" in ck else ck
            net.load_state_dict(sd)
            net.eval()
            self._net = net

            self._helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=True, device=self.device)
            self._warmup_codeformer()
            return True
        except Exception as exc:
            print(f"[RESTORE] CodeFormer load failed ({exc}); trying GFPGAN.")
            self._net = None
            self._helper = None
            return False

    def _try_load_gfpgan(self):
        """Fallback: load GFPGAN v1.4 (already on disk). Returns True on success."""
        if not os.path.exists(GFPGAN_PATH):
            return False
        try:
            from gfpgan import GFPGANer
            self._net = GFPGANer(
                model_path=GFPGAN_PATH, upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None, device=self.device)
            return True
        except Exception as exc:
            print(f"[RESTORE] GFPGAN load failed ({exc}).")
            return False

    def _warmup_codeformer(self):
        """Run one forward on the character image so cuDNN autotunes the fixed
        512 shape at startup, not on the first live frame."""
        face = None
        for p in (os.path.join(PROJECT_DIR, "ai-face", "character.jpg"),
                  os.path.join(PROJECT_DIR, "character.jpg")):
            if os.path.exists(p):
                face = cv2.imread(p)
                break
        if face is None:
            return
        face = cv2.resize(face, (512, 512))
        try:
            for _ in range(2):
                self._restore_codeformer(face)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # FRAME PROCESSING
    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        """Return frame with the face restored. Restores every RESTORE_INTERVAL
        frames and reuses the cached result between; never raises."""
        if not self.ready:
            return frame
        self._frame_counter += 1
        if self._cached is not None and (self._frame_counter % RESTORE_INTERVAL) != 0:
            return self._cached
        try:
            if self.backend == "CodeFormer":
                out = self._restore_codeformer(frame)
            else:
                out = self._restore_gfpgan(frame)
            if out is not None:
                self._cached = out
                return out
        except Exception as exc:
            if not self._err_printed:
                print(f"[RESTORE] frame error ({exc}) — passing face through.")
                self._err_printed = True
        return self._cached if self._cached is not None else frame

    def _restore_codeformer(self, frame):
        """CodeFormer path with a cached fast paste.

        Detection (retinaface) + the alignment transform are refreshed only
        every DETECT_INTERVAL restores; in between, the frame is warped to the
        512 aligned crop with the cached affine, restored, and warped back with
        a cached feathered mask — bypassing facexlib's per-frame detector and
        parse-net (the two biggest costs).
        """
        self._restore_calls += 1
        need_detect = (self._affine is None or
                       (self._restore_calls % DETECT_INTERVAL) == 0)
        if need_detect and not self._refresh_alignment(frame):
            # detection failed: keep using the last good geometry if we have it,
            # otherwise pass the frame through unchanged.
            if self._affine is None:
                return frame

        H, W = frame.shape[:2]
        crop = cv2.warpAffine(frame, self._affine, (512, 512),
                              flags=cv2.INTER_LINEAR)
        restored = self._run_codeformer_crop(crop)                # 512 BGR uint8
        restored_full = cv2.warpAffine(restored, self._inv_affine, (W, H),
                                       flags=cv2.INTER_LINEAR)
        m = self._paste_mask
        out = frame.astype(np.float32) * (1.0 - m) + restored_full.astype(np.float32) * m
        return np.clip(out, 0, 255).astype(np.uint8)

    def _refresh_alignment(self, frame):
        """Detect the face, cache the affine matrices + feathered paste mask.
        Returns True if a face was found."""
        h = self._helper
        h.clean_all()
        h.read_image(frame)
        n = h.get_face_landmarks_5(only_center_face=True, resize=640,
                                   eye_dist_threshold=5)
        if n == 0:
            return False
        h.align_warp_face()
        self._affine = np.asarray(h.affine_matrices[0], dtype=np.float32)
        self._inv_affine = cv2.invertAffineTransform(self._affine)
        # Feathered oval mask in the 512 aligned space (covers the face, fades at
        # the edges so the sharpened head melts into the soft background instead
        # of a hard rectangle), warped back into frame space and cached.
        H, W = frame.shape[:2]
        oval = np.zeros((512, 512), np.float32)
        cv2.ellipse(oval, (256, 262), (190, 240), 0, 0, 360, 1.0, -1)
        oval = cv2.GaussianBlur(oval, (0, 0), 22)
        mask_full = cv2.warpAffine(oval, self._inv_affine, (W, H),
                                   flags=cv2.INTER_LINEAR)
        self._paste_mask = np.clip(mask_full, 0, 1)[:, :, None]
        return True

    def _run_codeformer_crop(self, crop):
        """Run CodeFormer on a 512 aligned BGR crop; return restored BGR uint8."""
        torch = self._torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        t = img2tensor(crop / 255.0, bgr2rgb=True, float32=True)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        t = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if USE_BF16 and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self._net(t, w=FIDELITY, adain=True)[0]
                out = out.float()
            else:
                out = self._net(t, w=FIDELITY, adain=True)[0]
        return tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)

    def _restore_gfpgan(self, frame):
        """GFPGAN fallback path."""
        _, _, restored = self._net.enhance(
            frame, has_aligned=False, only_center_face=True, paste_back=True,
            weight=FIDELITY)
        return restored if restored is not None else frame


# Module-level singleton so the main loop can `import face_restore_engine`
# and call helpers, mirroring enhance_engine's style.
_engine = None


def get_engine():
    """Lazily build and return the shared FaceRestoreEngine."""
    global _engine
    if _engine is None:
        _engine = FaceRestoreEngine()
    return _engine


def startup_check():
    return get_engine().startup_check()


def restore_frame(frame):
    return get_engine().process_frame(frame)


if __name__ == "__main__":
    eng = FaceRestoreEngine()
    print("[RESTORE] startup_check:", eng.startup_check())
    # Quick visual smoke test on the character image.
    for p in (os.path.join(PROJECT_DIR, "ai-face", "character.jpg"),
              os.path.join(PROJECT_DIR, "character.jpg")):
        if os.path.exists(p):
            img = cv2.imread(p)
            # Simulate the Wav2Lip blur on the lower face.
            blurred = img.copy()
            blurred[256:, :] = cv2.GaussianBlur(blurred[256:, :], (0, 0), 3)
            out = eng.process_frame(blurred)
            cv2.imwrite(os.path.join(PROJECT_DIR, "_restore_before.jpg"), blurred)
            cv2.imwrite(os.path.join(PROJECT_DIR, "_restore_after.jpg"), out)
            print("[RESTORE] wrote _restore_before.jpg / _restore_after.jpg", out.shape)
            break
