# =============================================================================
# engines/restore_engine.py
# -----------------------------------------------------------------------------
# Real-time GFPGAN face restoration on the LivePortrait OUTPUT, FACE-CROP ONLY.
# LivePortrait's one-shot output looks slightly plastic with glassy eyes; GFPGAN
# restores skin texture, sharpens the eyes (adds natural catch-lights) and removes
# the synthetic look. To stay real-time:
#   * restore ONLY the face bounding box (not the whole frame)
#   * run every RESTORE_EVERY_N frames, hold the restored crop between
#   * feather-blend the restored crop back (no seam), at "skin detail" strength
#
# CRITICAL: GFPGAN's StyleGAN decoder OVERFLOWS in pure fp16 (-> flat brown blob).
# We run it under bf16 autocast (fp32 range, ~2x faster than fp32). [learned]
#
#   r = RestoreEngine()
#   frame = r.restore(frame)        # face restored in place (every Nth frame)
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
GFPGAN_CANDIDATES = [
    os.path.join(PROJECT_DIR, "ai-face", "models", "GFPGANv1.4.pth"),
    os.path.join(PROJECT_DIR, "models", "GFPGANv1.4.pth"),
]

NET_SIZE = 512                 # GFPGAN works at 512x512 aligned faces
RESTORE_EVERY_N = 2            # restore every Nth frame; hold result between
FACE_REDETECT = 15             # re-find the face box every N frames (it barely moves)
FACE_PAD = 1.25                # pad the detected face box (include forehead/jaw)
FIDELITY = 0.5                 # GFPGAN identity-vs-quality weight
FEATHER = 0.16                 # feather fraction of the crop for the blend mask


class RestoreEngine:
    """Crop-only GFPGAN restoration of the LivePortrait face (real-time)."""

    def __init__(self):
        self.ready = False
        self.skin_detail = 0.70        # blend strength of the restored face
        self.every_n = RESTORE_EVERY_N
        self._net = None
        self._device = "cpu"
        self._bf16 = False
        self._facedet = None
        self._facedet_tried = False
        self._bbox = None              # cached face box
        self._restored = None          # cached restored crop (for hold-between)
        self._fc = 0
        self._err = False
        self._import()

    def _import(self):
        model = next((p for p in GFPGAN_CANDIDATES if os.path.exists(p)), None)
        if model is None:
            print("[GFPGAN] weights not found (GFPGANv1.4.pth) — restoration off.")
            return
        try:
            import torch
            from gfpgan import GFPGANer
            from basicsr.utils import img2tensor, tensor2img
            from torchvision.transforms.functional import normalize
            self._torch = torch
            self._img2tensor = img2tensor
            self._tensor2img = tensor2img
            self._normalize = normalize
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._bf16 = self._device == "cuda"
            restorer = GFPGANer(model_path=model, upscale=1, arch="clean",
                                channel_multiplier=2, bg_upsampler=None,
                                device=self._device)
            self._net = restorer.gfpgan
            self._warmup()
            self.ready = True
            self._print_gpu()
            print(f"[GFPGAN] Ready — crop-only restoration ({self._device}, "
                  f"{'bf16' if self._bf16 else 'fp32'}).")
        except Exception as exc:
            print(f"[GFPGAN] init failed ({exc}) — restoration off.")
            self._net = None

    def startup_check(self):
        if self.ready:
            return True, "GFPGAN restoration active (face crop)."
        return True, "GFPGAN restoration OFF (weights/deps missing)."

    # -------------------------------------------------------------------------
    def _autocast(self):
        import contextlib
        if self._bf16:
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def _run_net(self, face_bgr):
        """Restore a single 512 face crop (BGR uint8) -> restored BGR uint8."""
        torch = self._torch
        crop = cv2.resize(face_bgr, (NET_SIZE, NET_SIZE))
        t = self._img2tensor(crop / 255.0, bgr2rgb=True, float32=True)
        self._normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        t = t.unsqueeze(0).to(self._device)
        with torch.no_grad(), self._autocast():
            out = self._net(t, return_rgb=False, weight=FIDELITY)[0]
        out = out[0].float()
        return self._tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)

    def _warmup(self):
        dummy = (np.random.rand(NET_SIZE, NET_SIZE, 3) * 255).astype(np.uint8)
        for _ in range(2):
            self._run_net(dummy)

    # -------------------------------------------------------------------------
    def _get_facedet(self):
        if self._facedet_tried:
            return self._facedet
        self._facedet_tried = True
        try:
            import mediapipe as mp
            self._facedet = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
        except Exception:
            self._facedet = None
        return self._facedet

    def _face_box(self, frame):
        """Padded square face box (cached, re-detected periodically)."""
        if self._bbox is not None and self._fc % FACE_REDETECT != 0:
            return self._bbox
        det = self._get_facedet()
        h, w = frame.shape[:2]
        if det is None:
            return self._bbox
        try:
            res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.detections:
                return self._bbox
            r = max((d.location_data.relative_bounding_box for d in res.detections),
                    key=lambda b: b.width * b.height)
            cx = (r.xmin + r.width / 2) * w
            cy = (r.ymin + r.height / 2) * h
            side = max(r.width * w, r.height * h) * FACE_PAD
            x1 = max(0, int(cx - side / 2)); y1 = max(0, int(cy - side / 2))
            x2 = min(w, int(cx + side / 2)); y2 = min(h, int(cy + side / 2))
            if x2 - x1 > 24 and y2 - y1 > 24:
                self._bbox = (x1, y1, x2, y2)
        except Exception:
            pass
        return self._bbox

    # -------------------------------------------------------------------------
    def restore(self, frame):
        """Restore the face region of frame (real-time, hold-between). In place-safe."""
        if not self.ready or self.skin_detail <= 0.01:
            return frame
        try:
            self._fc += 1
            box = self._face_box(frame)
            if box is None:
                return frame
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return frame
            bw, bh = x2 - x1, y2 - y1

            # restore every Nth frame; reuse the cached restored crop between
            if self._restored is None or (self._fc % self.every_n == 0):
                self._restored = self._run_net(crop)
            restored = cv2.resize(self._restored, (bw, bh))

            # feathered blend at skin_detail strength
            mask = np.ones((bh, bw), np.float32)
            b = max(2, int(min(bw, bh) * FEATHER))
            mask[:b, :] *= np.linspace(0, 1, b)[:, None]
            mask[-b:, :] *= np.linspace(1, 0, b)[:, None]
            mask[:, :b] *= np.linspace(0, 1, b)[None, :]
            mask[:, -b:] *= np.linspace(1, 0, b)[None, :]
            mask = (mask * self.skin_detail)[:, :, None]

            blended = restored.astype(np.float32) * mask + crop.astype(np.float32) * (1 - mask)
            frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            return frame
        except Exception as exc:
            if not self._err:
                print(f"[GFPGAN] restore error ({exc}) — passing frame through.")
                self._err = True
            return frame

    def _print_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[GFPGAN] GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB")
        except Exception:
            pass


if __name__ == "__main__":
    r = RestoreEngine()
    print("[GFPGAN]", r.startup_check()[1])
    test = np.full((512, 512, 3), 120, np.uint8)
    cv2.circle(test, (256, 256), 140, (170, 160, 150), -1)
    out = r.restore(test)
    print("[GFPGAN] output:", out.shape)
