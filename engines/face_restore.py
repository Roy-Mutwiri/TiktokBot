# =============================================================================
# engines/face_restore.py
# -----------------------------------------------------------------------------
# Per-frame GFPGAN face restoration for the avatar — adds real skin texture /
# pore detail to the LivePortrait-animated face so it reads as camera footage
# rather than a smooth synthetic render.
#
# Reuses the GFPGAN model already downloaded for the ai-face pipeline. Tuned for
# speed: upscale=1 (restore at native 512, cheaper paste), square paste mask
# (no per-frame face-parse net), cuDNN autotuning, and warmed up at init.
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
FIDELITY = 0.5                 # GFPGAN weight: lower = stronger restore, higher = closer to input
UPSCALE = 1                    # 1 = restore only (fast); output stays 512


class FaceRestorer:
    """Wraps GFPGANer for fast single-frame face restoration."""

    def __init__(self):
        """Load GFPGAN (or disable gracefully) and warm it up."""
        self.ok = False
        self._err_printed = False
        model = next((p for p in GFPGAN_CANDIDATES if os.path.exists(p)), None)
        if model is None:
            print("[RESTORE] GFPGANv1.4.pth not found — face restoration disabled.")
            return
        try:
            import torch
            from gfpgan import GFPGANer
            torch.backends.cudnn.benchmark = True
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.restorer = GFPGANer(model_path=model, upscale=UPSCALE, arch="clean",
                                     channel_multiplier=2, bg_upsampler=None, device=device)
            # Square paste mask instead of the per-frame face-parse net (faster).
            try:
                self.restorer.face_helper.use_parse = False
            except Exception:
                pass
            self._warmup()
            self.ok = True
            print(f"[RESTORE] GFPGAN ready on {device} — adding skin texture to the face.")
        except Exception as exc:
            print(f"[RESTORE] could not init GFPGAN ({exc}) — face restoration disabled.")

    def startup_check(self):
        """Return (ok, message)."""
        return True, ("GFPGAN skin restoration active" if self.ok
                      else "FALLBACK — no GFPGAN (face stays as-is)")

    def _warmup(self):
        """Run a couple of restores so cuDNN autotunes before the live loop."""
        face = np.full((512, 512, 3), 110, np.uint8)
        cv2.circle(face, (256, 250), 150, (170, 160, 150), -1)
        cv2.circle(face, (210, 220), 16, (90, 90, 90), -1)
        cv2.circle(face, (300, 220), 16, (90, 90, 90), -1)
        for _ in range(3):
            self.restore(face)

    def restore(self, frame_bgr):
        """Restore the face in a 512 BGR frame. Returns same-size BGR.

        The LivePortrait output is a centered face filling the frame, so we pass
        has_aligned=True: GFPGAN treats the whole frame as a pre-aligned face and
        runs ONLY the restoration network — no detection, no warp, no paste-back
        (≈2x faster). Never raises — returns the input unchanged on failure.
        """
        if not self.ok:
            return frame_bgr
        try:
            h, w = frame_bgr.shape[:2]
            _, restored, _ = self.restorer.enhance(
                frame_bgr, has_aligned=True, paste_back=False, weight=FIDELITY)
            if not restored:
                return frame_bgr
            out = restored[0]
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w, h))
            return out
        except Exception as exc:
            if not self._err_printed:
                print(f"[RESTORE] frame error ({exc}) — passing face through.")
                self._err_printed = True
            return frame_bgr


if __name__ == "__main__":
    import time
    r = FaceRestorer()
    print("startup:", r.startup_check())
    if r.ok:
        char = os.path.join(PROJECT_DIR, "ai-face", "character.jpg")
        face = cv2.resize(cv2.imread(char), (512, 512))
        for _ in range(3):
            r.restore(face)
        ts = []
        for _ in range(20):
            t = time.perf_counter(); r.restore(face); ts.append((time.perf_counter() - t) * 1000)
        print(f"[RESTORE] steady: median {sorted(ts)[len(ts)//2]:.0f}ms  min {min(ts):.0f}ms")
        cv2.imwrite(os.path.join(PROJECT_DIR, "_restore_test.jpg"), r.restore(face))
