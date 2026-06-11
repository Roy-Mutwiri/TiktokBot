# =============================================================================
# engines/compositor.py
# -----------------------------------------------------------------------------
# Mouth region detection + seamless compositing.
#
# In the per-frame pipeline this sits BETWEEN LivePortrait and the final enhance
# pass: LivePortrait owns the whole animated AI face, MuseTalk produces a
# mouth-synced crop, and the Compositor stitches that crop back onto the
# LivePortrait frame with a feathered, colour-matched blend so there is no seam.
#
#   bbox  = compositor.detect_mouth_bbox(lp_face)         # MediaPipe FaceMesh
#   mouth = musetalk.process_mouth(lp_face, bbox)         # synced crop (bbox-sized)
#   final = compositor.blend_mouth(lp_face, mouth, bbox)  # feathered + colour-matched
#
# MediaPipe FaceMesh is CPU-only on Windows (~2-4 ms/frame) and the mouth box is
# only needed while speaking, so the cost is paid only during speech.
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
# FaceMesh lip landmark indices (468-point mesh) spanning the full outer+inner
# lip contour and the philtrum/chin edge, so the bbox encloses the whole mouth.
MOUTH_LANDMARKS = [
    61, 291, 0, 17, 39, 269, 405, 181, 84, 314,        # outer lip + corners
    78, 308, 13, 14, 87, 317, 37, 267, 82, 312,        # inner lip
    91, 321, 146, 375,                                  # lower outer
]
PAD_X = float(os.environ.get("AVATAR_MOUTH_PADX", "0.36"))   # cover lip corners
PAD_Y_TOP = float(os.environ.get("AVATAR_MOUTH_PADTOP", "0.30"))
# more room below = the jaw/chin drop on an open mouth is captured -> the mouth
# OPENING reads clearly instead of being clipped.
PAD_Y_BOTTOM = float(os.environ.get("AVATAR_MOUTH_PADBOT", "0.56"))
FEATHER_KERNEL = 31   # GaussianBlur kernel for the alpha edge feather (odd) - softer, no seam
COLOR_MATCH_STRENGTH = 0.6   # 0..1 how strongly to match crop colour to skin
# Unsharp-mask strength on the soft Wav2Lip mouth (0 = off). A little helps real
# speech, but too much amplifies Wav2Lip's colour artifacts, so default OFF and
# leave it as an opt-in tunable. AVATAR_MOUTH_SHARPEN.
MOUTH_SHARPEN = float(os.environ.get("AVATAR_MOUTH_SHARPEN", "0.0"))
MIN_BBOX = 12         # reject degenerate detections smaller than this (px)


class Compositor:
    """Detects the mouth region and blends a synced mouth crop back seamlessly."""

    def __init__(self):
        """Create a persistent FaceMesh instance (None if MediaPipe unavailable)."""
        self._mesh = None
        self._mesh_tried = False
        self._last_bbox = None        # keep last good box if a frame loses the face
        self._err_printed = False

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Report mouth-detection readiness. Returns (ok, message)."""
        mesh = self._get_mesh()
        if mesh is None:
            return True, "compositor: FaceMesh unavailable — center-box fallback."
        return True, "compositor: FaceMesh mouth tracking active."

    # -------------------------------------------------------------------------
    def _get_mesh(self):
        """Lazily build the MediaPipe FaceMesh (video mode, single face)."""
        if self._mesh_tried:
            return self._mesh
        self._mesh_tried = True
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        except Exception as exc:
            print(f"[COMPOSITE] FaceMesh unavailable ({exc}) — using center box.")
            self._mesh = None
        return self._mesh

    # -------------------------------------------------------------------------
    def detect_mouth_bbox(self, face_frame, mouth_hint=None):
        """Return (x1, y1, x2, y2) around the mouth, padded for jaw movement.

        If `mouth_hint` (cx, cy, mouth_width) is given — the face-swap's OWN exact
        mouth keypoints — build the box from that (no re-detection that can mis-place
        it and cause a doubled mouth). Else fall back to FaceMesh / center box.
        """
        h, w = face_frame.shape[:2]
        if mouth_hint is not None:
            try:
                cx, cy, mw = mouth_hint
                mw = max(12.0, float(mw))
                bw = mw * (1.0 + 2 * PAD_X)             # mouth width + corners
                bh = mw * 0.75                          # mouth height ~ 0.75 * width
                x1 = int(cx - bw / 2); x2 = int(cx + bw / 2)
                y1 = int(cy - bh * (0.5 + PAD_Y_TOP))   # room above the upper lip
                y2 = int(cy + bh * (0.5 + PAD_Y_BOTTOM))  # room below for jaw drop
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                if x2 - x1 >= MIN_BBOX and y2 - y1 >= MIN_BBOX:
                    self._last_bbox = (x1, y1, x2, y2)
                    return self._last_bbox
            except Exception:
                pass
        mesh = self._get_mesh()
        if mesh is None:
            return self._center_box(w, h)

        try:
            res = mesh.process(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return self._last_bbox if self._last_bbox else self._center_box(w, h)

            lms = res.multi_face_landmarks[0].landmark
            xs, ys = [], []
            for idx in MOUTH_LANDMARKS:
                if idx < len(lms):
                    xs.append(lms[idx].x * w)
                    ys.append(lms[idx].y * h)
            if not xs:
                return self._last_bbox if self._last_bbox else self._center_box(w, h)

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bw = x_max - x_min
            bh = y_max - y_min

            x1 = int(x_min - bw * PAD_X)
            x2 = int(x_max + bw * PAD_X)
            y1 = int(y_min - bh * PAD_Y_TOP)
            y2 = int(y_max + bh * PAD_Y_BOTTOM)

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 - x1 < MIN_BBOX or y2 - y1 < MIN_BBOX:
                return self._last_bbox if self._last_bbox else self._center_box(w, h)

            self._last_bbox = (x1, y1, x2, y2)
            return self._last_bbox
        except Exception as exc:
            if not self._err_printed:
                print(f"[COMPOSITE] detect error ({exc}) — using last/center box.")
                self._err_printed = True
            return self._last_bbox if self._last_bbox else self._center_box(w, h)

    def _center_box(self, w, h):
        """Lower-center mouth box used when no landmarks are available."""
        x1 = int(w * 0.34); x2 = int(w * 0.66)
        y1 = int(h * 0.62); y2 = int(h * 0.86)
        return (x1, y1, x2, y2)

    # -------------------------------------------------------------------------
    def blend_mouth(self, lp_frame, synced_mouth_crop, bbox):
        """Composite the synced mouth crop onto lp_frame within bbox.

        Uses a feathered alpha mask (so the bbox edges melt into the surrounding
        face) and a mean-shift colour match (so the synced crop's skin tone
        matches the LivePortrait face it is dropped into). Never raises.
        """
        try:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or synced_mouth_crop is None:
                return lp_frame

            result = lp_frame.copy()
            region = result[y1:y2, x1:x2]
            if region.size == 0:
                return lp_frame

            # Size the crop to the destination region.
            crop = synced_mouth_crop
            if crop.shape[:2] != (bh, bw):
                crop = cv2.resize(crop, (bw, bh))
            # SHARPEN the soft 96px Wav2Lip mouth (it's the blurriest, most-watched
            # region, and the CodeFormer restore ran BEFORE the mouth so it never
            # touched it). An unsharp mask adds crisp lip/teeth detail so it reads
            # as real camera footage. Tunable via AVATAR_MOUTH_SHARPEN.
            if MOUTH_SHARPEN > 0.0:
                blur = cv2.GaussianBlur(crop, (0, 0), 1.5)
                crop = cv2.addWeighted(crop, 1.0 + MOUTH_SHARPEN, blur, -MOUTH_SHARPEN, 0)
            crop = crop.astype(np.float32)
            region_f = region.astype(np.float32)

            # --- colour match: shift crop mean toward the destination mean ---
            # Sample a thin border ring of the destination as the skin reference
            # (the center is mouth, which legitimately differs in colour).
            ref_mean = self._border_mean(region_f)
            crop_mean = crop.reshape(-1, 3).mean(axis=0)
            shift = (ref_mean - crop_mean) * COLOR_MATCH_STRENGTH
            crop = np.clip(crop + shift, 0, 255)

            # --- feathered alpha mask (1 inside, ramping to 0 at the edges) ---
            # Border kept SMALL (10%) so the whole mouth sits in the fully-replaced
            # centre — otherwise the operator's lip corners / jaw motion leak in
            # through the edges (the "my mouth still syncs" bug). The box is padded
            # generously (PAD_*), so this 10% feather lands on cheek/chin SKIN.
            mask = np.ones((bh, bw), dtype=np.float32)
            border = max(2, int(min(bw, bh) * 0.15))
            mask[:border, :] = 0; mask[-border:, :] = 0
            mask[:, :border] = 0; mask[:, -border:] = 0
            k = FEATHER_KERNEL | 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            mask = np.clip(mask, 0.0, 1.0)[:, :, None]

            blended = crop * mask + region_f * (1.0 - mask)
            result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            return result
        except Exception as exc:
            if not self._err_printed:
                print(f"[COMPOSITE] blend error ({exc}) — passing frame through.")
                self._err_printed = True
            return lp_frame

    def _border_mean(self, region_f):
        """Mean BGR of a thin ring around the region (skin reference)."""
        h, w = region_f.shape[:2]
        b = max(1, int(min(h, w) * 0.12))
        ring = np.concatenate([
            region_f[:b, :].reshape(-1, 3),
            region_f[-b:, :].reshape(-1, 3),
            region_f[:, :b].reshape(-1, 3),
            region_f[:, -b:].reshape(-1, 3),
        ], axis=0)
        return ring.mean(axis=0)


if __name__ == "__main__":
    comp = Compositor()
    print("[COMPOSITE]", comp.startup_check()[1])
    test = np.full((512, 512, 3), 120, dtype=np.uint8)
    cv2.circle(test, (256, 256), 150, (170, 160, 150), -1)   # face-ish blob
    bbox = comp.detect_mouth_bbox(test)
    print("[COMPOSITE] bbox:", bbox)
    mouth = np.full((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), (40, 40, 160), np.uint8)
    out = comp.blend_mouth(test, mouth, bbox)
    print("[COMPOSITE] output shape:", out.shape)
