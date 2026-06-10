# =============================================================================
# engines/body_motion.py
# -----------------------------------------------------------------------------
# Adds believable UPPER-BODY motion to the avatar. LivePortrait animates the head
# but leaves the torso frozen, so head turns look detached. This engine tracks
# the OPERATOR's shoulders/torso from the webcam (MediaPipe Pose) and applies a
# gentle, feathered warp to the avatar's body region so the shoulders sway, lean
# and rotate WITH you — while the head (top of frame) stays driven by LivePortrait.
#
#   body = BodyMotionEngine()
#   frame = body.process(webcam_frame, avatar_frame)   # avatar with live torso
#
# Motion is relative to a neutral captured on the first good frame (recenter()
# to reset), 1€-smoothed, and gain-limited so it never goes rubbery.
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

_ENGINES_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINES_DIR not in sys.path:
    sys.path.insert(0, _ENGINES_DIR)

# -----------------------------------------------------------------------------
# CONFIGURATION  (gentle gains — believable, not rubbery)
# -----------------------------------------------------------------------------
FRAME = 512
# The warp NEVER touches the face/neck — the boundary is placed just BELOW the
# detected chin, so head+neck stay locked to the LivePortrait head (no neck
# seam). Only the shoulders/chest below it move. Rotation pivots at that line so
# the neck stays put and the shoulders swing under it.
CHIN_MARGIN = 0.05          # keep this much (of frame) below the chin static
FEATHER = 0.14              # feather depth below the keep line (of frame)

ROLL_GAIN = 0.35            # gentle shoulder tilt (rotation tears necks — keep low)
SHIFT_GAIN = 0.9            # horizontal sway
LEAN_GAIN = 0.55            # vertical lean
SCALE_GAIN = 0.45           # lean in/out
MAX_ROLL = 6.0              # clamp degrees
MAX_SHIFT = 24.0            # clamp px
MAX_SCALE = 0.07            # clamp +/- scale
FACE_REDETECT = 12          # re-find the avatar face every N frames (it barely moves)
POSE_INTERVAL = 2           # run MediaPipe Pose every N frames (reuse params between)

EURO = dict(min_cutoff=1.0, beta=0.02)   # smooth the body signal


class BodyMotionEngine:
    """Warps the avatar's torso to follow the operator's shoulders (webcam)."""

    def __init__(self):
        self._pose = None
        self._pose_tried = False
        self._facedet = None
        self._facedet_tried = False
        self._ref = None                 # neutral (cx, cy, width, roll)
        self._params = None              # last (angle, tx, ty, scale) — reused between pose runs
        self._keep_y = int(FRAME * 0.72)  # below-chin line (updated by detection)
        self._pivot_y = self._keep_y
        self._mask = None                # (re)built when _keep_y changes
        self._fc = 0                     # frame counter for face redetection
        self._err = False
        from one_euro import OneEuroFilter
        self._f_roll = OneEuroFilter(**EURO)
        self._f_dx = OneEuroFilter(**EURO)
        self._f_dy = OneEuroFilter(**EURO)
        self._f_sc = OneEuroFilter(**EURO)

    def startup_check(self):
        if self._get_pose() is None:
            return True, "body motion: MediaPipe Pose unavailable — static torso."
        return True, "body motion: webcam-driven torso active."

    def recenter(self):
        """Reset the neutral body pose (call when sitting upright/centered)."""
        self._ref = None
        self._params = None
        for f in (self._f_roll, self._f_dx, self._f_dy, self._f_sc):
            f.reset()

    # -------------------------------------------------------------------------
    def _get_pose(self):
        if self._pose_tried:
            return self._pose
        self._pose_tried = True
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as exc:
            print(f"[BODY] MediaPipe Pose unavailable ({exc}) — static torso.")
            self._pose = None
        return self._pose

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

    def _update_keep_line(self, avatar_frame):
        """Find the avatar's chin and keep everything above it (face+neck) static."""
        det = self._get_facedet()
        if det is None:
            return
        try:
            h, w = avatar_frame.shape[:2]
            res = det.process(cv2.cvtColor(avatar_frame, cv2.COLOR_BGR2RGB))
            if not res.detections:
                return
            r = max((d.location_data.relative_bounding_box for d in res.detections),
                    key=lambda b: b.width * b.height)
            chin = (r.ymin + r.height) * h
            keep = int(min(FRAME - 4, chin + CHIN_MARGIN * FRAME))
            if abs(keep - self._keep_y) > 2:
                self._keep_y = keep
                self._pivot_y = keep
                self._mask = None        # rebuild
        except Exception:
            pass

    def _build_mask(self):
        """0 over head+neck (above keep line), ramping to 1 over the torso below."""
        m = np.ones((FRAME, 1), np.float32)
        keep = self._keep_y
        full = min(FRAME, keep + int(FEATHER * FRAME))
        m[:keep, 0] = 0.0
        ramp = max(1, full - keep)
        for y in range(keep, full):
            m[y, 0] = (y - keep) / ramp
        self._mask = m[:, :, None]
        return self._mask

    # -------------------------------------------------------------------------
    def process(self, webcam_frame, avatar_frame):
        """Return avatar_frame with the torso warped to follow your shoulders."""
        pose = self._get_pose()
        if pose is None or webcam_frame is None or avatar_frame is None:
            return avatar_frame
        try:
            # keep the warp strictly BELOW the chin (face+neck never move)
            self._fc += 1
            if self._mask is None or self._fc % FACE_REDETECT == 0:
                self._update_keep_line(avatar_frame)
                if self._mask is None:
                    self._build_mask()
            h, w = webcam_frame.shape[:2]
            # Pose detection is the cost — run it every POSE_INTERVAL frames and
            # reuse the last warp params between (the warp+blend below is cheap and
            # still runs every frame, so the body stays smooth).
            if self._fc % POSE_INTERVAL == 0 or self._params is None:
                res = pose.process(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB))
                lm = getattr(res, "pose_landmarks", None)
                if lm is not None:
                    ls = lm.landmark[11]; rs = lm.landmark[12]   # shoulders
                    if ls.visibility >= 0.4 and rs.visibility >= 0.4:
                        lx, ly = ls.x * w, ls.y * h
                        rx, ry = rs.x * w, rs.y * h
                        cx = (lx + rx) / 2.0; cy = (ly + ry) / 2.0
                        width = float(np.hypot(lx - rx, ly - ry)) + 1e-3
                        roll = float(np.degrees(np.arctan2(ly - ry, lx - rx)))
                        if self._ref is None:
                            self._ref = (cx, cy, width, roll)
                        else:
                            ref_cx, ref_cy, ref_w, ref_roll = self._ref
                            d_roll = self._f_roll(-(roll - ref_roll))
                            d_x = self._f_dx(-(cx - ref_cx) / ref_w)
                            d_y = self._f_dy((cy - ref_cy) / ref_w)
                            d_sc = self._f_sc(width / ref_w - 1.0)
                            angle = float(np.clip(d_roll * ROLL_GAIN, -MAX_ROLL, MAX_ROLL))
                            tx = float(np.clip(d_x * ref_w * SHIFT_GAIN * (FRAME / w),
                                               -MAX_SHIFT, MAX_SHIFT))
                            ty = float(np.clip(d_y * ref_w * LEAN_GAIN * (FRAME / h),
                                               -MAX_SHIFT, MAX_SHIFT))
                            scale = 1.0 + float(np.clip(d_sc * SCALE_GAIN,
                                                        -MAX_SCALE, MAX_SCALE))
                            self._params = (angle, tx, ty, scale)
            if self._params is None:
                return avatar_frame
            angle, tx, ty, scale = self._params

            M = cv2.getRotationMatrix2D((FRAME / 2.0, float(self._pivot_y)), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            warped = cv2.warpAffine(avatar_frame, M, (FRAME, FRAME),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask = self._mask if self._mask is not None else self._build_mask()
            if mask.shape[0] != avatar_frame.shape[0]:
                mask = cv2.resize(self._mask[:, :, 0], (1, avatar_frame.shape[0]))[:, :, None]
            out = avatar_frame * (1.0 - mask) + warped * mask
            return out.astype(np.uint8)
        except Exception as exc:
            if not self._err:
                print(f"[BODY] frame error ({exc}) — torso left static.")
                self._err = True
            return avatar_frame


if __name__ == "__main__":
    eng = BodyMotionEngine()
    print("[BODY]", eng.startup_check()[1])
    wc = np.full((480, 640, 3), 100, np.uint8)
    av = np.full((512, 512, 3), 80, np.uint8)
    out = eng.process(wc, av)
    print("[BODY] output:", out.shape)
