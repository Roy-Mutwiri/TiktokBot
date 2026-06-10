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
NECK_Y = 330                # pivot row for shoulder rotation (below the face)
HEAD_KEEP_Y = 250           # above this row = head, untouched (LivePortrait owns it)
BODY_FULL_Y = 360           # below this row = full body warp; feather in between

ROLL_GAIN = 0.7             # avatar shoulder tilt per deg of your shoulder tilt
SHIFT_GAIN = 0.9            # horizontal sway (px avatar per px you shift, norm'd)
LEAN_GAIN = 0.6             # vertical lean
SCALE_GAIN = 0.5            # lean in/out (shoulder-width change -> scale)
MAX_ROLL = 10.0             # clamp degrees
MAX_SHIFT = 26.0            # clamp px
MAX_SCALE = 0.08            # clamp +/- scale

EURO = dict(min_cutoff=1.0, beta=0.02)   # smooth the body signal


class BodyMotionEngine:
    """Warps the avatar's torso to follow the operator's shoulders (webcam)."""

    def __init__(self):
        self._pose = None
        self._pose_tried = False
        self._ref = None                 # neutral (cx, cy, width, roll)
        self._mask = self._build_mask()
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

    def _build_mask(self):
        """Vertical feather: 0 over the head, ramping to 1 over the torso."""
        m = np.ones((FRAME, 1), np.float32)
        m[:HEAD_KEEP_Y, 0] = 0.0
        ramp = max(1, BODY_FULL_Y - HEAD_KEEP_Y)
        for y in range(HEAD_KEEP_Y, min(FRAME, BODY_FULL_Y)):
            m[y, 0] = (y - HEAD_KEEP_Y) / ramp
        return m[:, :, None]             # (H,1,1) broadcasts over width+channels

    # -------------------------------------------------------------------------
    def process(self, webcam_frame, avatar_frame):
        """Return avatar_frame with the torso warped to follow your shoulders."""
        pose = self._get_pose()
        if pose is None or webcam_frame is None or avatar_frame is None:
            return avatar_frame
        try:
            h, w = webcam_frame.shape[:2]
            res = pose.process(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB))
            lm = getattr(res, "pose_landmarks", None)
            if lm is None:
                return avatar_frame
            ls = lm.landmark[11]; rs = lm.landmark[12]   # left / right shoulder
            if ls.visibility < 0.4 or rs.visibility < 0.4:
                return avatar_frame

            lx, ly = ls.x * w, ls.y * h
            rx, ry = rs.x * w, rs.y * h
            cx = (lx + rx) / 2.0; cy = (ly + ry) / 2.0
            width = float(np.hypot(lx - rx, ly - ry)) + 1e-3
            roll = float(np.degrees(np.arctan2(ly - ry, lx - rx)))   # shoulder tilt

            if self._ref is None:
                self._ref = (cx, cy, width, roll)
                return avatar_frame
            ref_cx, ref_cy, ref_w, ref_roll = self._ref

            # relative, normalized motion (mirror x: webcam is mirrored vs avatar)
            d_roll = self._f_roll(-(roll - ref_roll))
            d_x = self._f_dx(-(cx - ref_cx) / ref_w)     # in shoulder-widths
            d_y = self._f_dy((cy - ref_cy) / ref_w)
            d_sc = self._f_sc(width / ref_w - 1.0)

            angle = float(np.clip(d_roll * ROLL_GAIN, -MAX_ROLL, MAX_ROLL))
            tx = float(np.clip(d_x * ref_w * SHIFT_GAIN * (FRAME / w), -MAX_SHIFT, MAX_SHIFT))
            ty = float(np.clip(d_y * ref_w * LEAN_GAIN * (FRAME / h), -MAX_SHIFT, MAX_SHIFT))
            scale = 1.0 + float(np.clip(d_sc * SCALE_GAIN, -MAX_SCALE, MAX_SCALE))

            M = cv2.getRotationMatrix2D((FRAME / 2.0, NECK_Y), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            warped = cv2.warpAffine(avatar_frame, M, (FRAME, FRAME),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask = self._mask
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
