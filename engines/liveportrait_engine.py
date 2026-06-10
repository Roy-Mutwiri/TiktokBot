# =============================================================================
# engines/liveportrait_engine.py
# -----------------------------------------------------------------------------
# Real-time LivePortrait inference: animates the AI character (source image)
# with the head/expression motion of a driving frame (a real streamer face from
# the behaviour engine).
#
# LivePortrait is NOT bundled with this repo. This engine locates it, loads the
# wrapper, encodes the source ONCE, then per frame transfers RELATIVE driving
# motion onto the source (the standard LivePortrait reenactment math).
#
# If LivePortrait cannot be loaded it runs in FALLBACK mode: it simply returns
# the source image unchanged (the avatar is visible and lip-syncs, but the head
# does not move). Install LivePortrait to enable motion:
#     git clone https://github.com/KwaiVGI/LivePortrait  (as ../LivePortrait)
#     download its pretrained weights per its README
# =============================================================================

import os
import sys
import time
import glob

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
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Candidate LivePortrait locations, tried in order.
LIVEPORTRAIT_CANDIDATES = [
    os.path.join(os.path.dirname(PROJECT_DIR), "LivePortrait"),   # sibling of TiktokBot
    os.path.join(PROJECT_DIR, "LivePortrait"),                    # inside TiktokBot
    os.environ.get("LIVEPORTRAIT_PATH", ""),
]

FRAME_SIZE = 512
USE_HALF = True                # FP16 for speed

# Real multi-angle views extracted from a video (build_character.py) live here.
CHARACTER_VIEWS_DIR = os.path.join(PROJECT_DIR, "character_views")

# LivePortrait's motion extractor expects a CENTERED, framed, FRONTAL-ish face.
# A raw webcam frame (face off-center / small / lots of background) yields weak,
# unreliable motion. So each driving frame is tracked with MediaPipe FaceMesh
# (temporal tracking — robust to head turn/tilt, unlike the single-shot face
# detector) and cropped to a padded square around the face landmarks. The box is
# EMA-smoothed so the crop doesn't jitter. When NO face is visible (operator
# looks away / down / out of frame) we HOLD the last good animated face instead
# of feeding garbage (hair, background) to LivePortrait — which otherwise renders
# a contorted grimace.
DRIVING_CROP_PAD = 1.8         # square side = this * face box size
DRIVING_MIN_FACE = 0.04        # below this the detection is ignored entirely
# QUALITY GATE: a face smaller than this fraction of the frame is too far away —
# its crop upscales to a low-detail, noisy driving image and LivePortrait renders
# a degraded/distorted face. Below this we treat it like a face-loss (hold the
# last good frame, then switch to charts) so a BAD face is never shown. Tune via
# AVATAR_MIN_FACE; raise it to require the operator closer, lower to be lenient.
MIN_GOOD_FACE = float(os.environ.get("AVATAR_MIN_FACE", "0.09"))
HOLD_ON_FACE_LOSS = True       # hold last good face when no (good) face is detected

# 1€-filter tuning per driving signal (jitter-vs-lag). Higher min_cutoff / beta
# = more responsive (less smoothing). These are deliberately light: kill shimmer
# without making the avatar feel laggy. Set SMOOTH_MOTION=False to disable.
SMOOTH_MOTION = os.environ.get("AVATAR_SMOOTH", "1") == "1"
EURO_POSE = dict(min_cutoff=1.5, beta=0.07)    # pitch/yaw/roll — high beta = snappy on
#                                                FAST turns (low lag), smooth when still
POSE_BETA = 0.07                               # kept responsive regardless of stabilization
EURO_TRANS = dict(min_cutoff=1.5, beta=0.03)   # translation t
EURO_SCALE = dict(min_cutoff=1.0, beta=0.01)   # scale (lean in/out)
EURO_EXP = dict(min_cutoff=2.5, beta=0.05)     # expression (keep blinks crisp)
EURO_BOX = dict(min_cutoff=1.0, beta=0.02)     # face-crop box — higher beta so the
#                                                crop SNAPS to fast head turns (no lag/trail)
MISS_GRACE = 6                                 # keep driving this many frames after a
#                                                detector miss before holding (no freeze flicker)

# POSE-AWARE SOFT LIMITING — caps the head-pose DELTA (degrees) with a tanh knee
# so turns stay inside the clean single-image range and never hit the distortion
# zone (measured: clean to ~30deg yaw, breaks 40deg+). Tune via env.
POSE_LIMIT = os.environ.get("AVATAR_POSE_LIMIT", "1") == "1"
# Turns FOLLOW you 1:1 up to KNEE_FRAC*cap, then ease smoothly to the cap (so the
# avatar tracks your head naturally and only the extreme is clamped to stay clean).
KNEE_FRAC = 0.7
# Clamp the driving->avatar scale ratio: the scale estimate jumps on head turns
# and zoomed the face. A talking head holds a near-constant size, so we allow
# only tiny variation (breathing) and never the big zoom.
SCALE_BAND = float(os.environ.get("AVATAR_SCALE_BAND", "0.05"))
# Per-frame leak of the pitch/roll neutral toward your current pose, so a baked-in
# up/down/tilt offset decays to level over a few seconds (your movement still
# shows as the transient delta). 0 disables. ~0.02 ≈ correct in ~4-5s.
AUTOCENTER_PITCH = float(os.environ.get("AVATAR_AUTOCENTER_PITCH", "0.02"))
# WIDE-MOUTH GUARD: per-element soft-clamp on the expression delta (LP exp values
# are ~0.003 typical, 0.027 max). A wide-open mouth/phoneme drives a big jaw exp
# delta that balloons the lower face; tanh-clamping each element to ±this keeps a
# natural max jaw drop (GFPGAN then sharpens teeth). 0 disables.
MOUTH_GUARD = float(os.environ.get("AVATAR_MOUTH_GUARD", "0.04"))
# SAFE-ZONE defaults: one-shot reenactment stays clean inside ~30deg yaw / ~10deg
# pitch; beyond that the unseen geometry stretches/melts. Knee-eased (not hard
# stop), and the Studio "Safe/Cinematic/Free" pose preset can widen this.
YAW_CAP = float(os.environ.get("AVATAR_YAW_CAP", "30"))     # left/right turn
PITCH_CAP = float(os.environ.get("AVATAR_PITCH_CAP", "10"))  # up/down (melts worst)
ROLL_CAP = float(os.environ.get("AVATAR_ROLL_CAP", "24"))   # head tilt

# MULTI-REFERENCE reenactment: the single character image has no side-of-head
# data, so hard turns hallucinate. We generate extra VIEWS of the SAME character
# (turned left/right via LivePortrait — identity-exact, no diffusion drift) once
# at startup, then each frame drive whichever view is nearest the operator's head
# angle. This pushes the clean turn range out to ~YAW_CAP_MULTI. Verified to beat
# single-image driving at 40-50deg. Disable with AVATAR_MULTIREF=0.
# Default OFF: the multi-view path can introduce profile artifacts on some live
# motion. The single-image path + soft pose-limit is bulletproof (never bends).
# Opt in with AVATAR_MULTIREF=1 (or the Studio "Extended turning" checkbox).
MULTI_REF = os.environ.get("AVATAR_MULTIREF", "0") == "1"
MULTI_REF_YAWS = [-40.0, -22.0, 22.0, 40.0]   # generated view yaw offsets (deg)
YAW_CAP_MULTI = float(os.environ.get("AVATAR_YAW_CAP_MULTI", "52"))  # cap when multi-ref on
REF_SWITCH_HYST = 5.0          # deg hysteresis so the active view doesn't flicker


class LivePortraitEngine:
    """Animates a cached source face with per-frame driving motion."""

    def __init__(self, source_image_path):
        """Locate + load LivePortrait, encode the source once, warm up."""
        self.fallback_mode = False
        self._error_printed = False
        self._ref_kp_info = None       # first driving frame (relative-motion ref)
        self._mesh = None              # MediaPipe FaceMesh tracker (lazy)
        self._mesh_tried = False
        self._drive_box = None         # EMA-smoothed driving crop box (cx,cy,side)
        self._face_found = False       # did THIS frame have a good (large enough) face?
        self._face_size = 0.0          # detected face size as a fraction of frame
        self._miss = 0                 # consecutive detector misses (grace window)
        self.min_good_face = MIN_GOOD_FACE   # live-tunable quality gate (see setter)
        self._last_output = None       # last good animated face (for hold-on-loss)

        # 1€ filters that stabilise the driving signal (no jitter) + crop box.
        from one_euro import OneEuroFilter
        self._euro = {
            "pitch": OneEuroFilter(**EURO_POSE), "yaw": OneEuroFilter(**EURO_POSE),
            "roll": OneEuroFilter(**EURO_POSE), "t": OneEuroFilter(**EURO_TRANS),
            "scale": OneEuroFilter(**EURO_SCALE), "exp": OneEuroFilter(**EURO_EXP),
        }
        self._euro_box = OneEuroFilter(**EURO_BOX)
        # live-tunable quality knobs (set by the Studio)
        self.gaze_lock = True
        self.gaze_strength = 0.8     # 0..1 how strongly to re-center eyes to camera
        self._stab = 0.4
        self.set_stabilization(0.4)
        self.wrapper = None

        # Load + cache the source image regardless of mode.
        src = cv2.imread(source_image_path)
        if src is None:
            print(f"[LP] could not read source image: {source_image_path}")
            src = np.full((FRAME_SIZE, FRAME_SIZE, 3), 60, dtype=np.uint8)
        self.source_image = cv2.resize(src, (FRAME_SIZE, FRAME_SIZE))

        lp_path = self._find_liveportrait()
        if lp_path is None:
            self.fallback_mode = True
            print("[LP] LivePortrait not found — FALLBACK mode (static source, "
                  "lip-sync still works). See header for install steps.")
            return

        try:
            self._init_pipeline(lp_path)
            self._warmup()
            self._print_gpu()
            print("[LP] Ready — motion driving enabled.")
        except Exception as exc:
            self.fallback_mode = True
            print(f"[LP] init failed ({exc}) — FALLBACK mode (static source).")

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Report whether motion driving is active. Returns (ok, message)."""
        if self.fallback_mode:
            return True, "FALLBACK (no LivePortrait) — static face, lip-sync only."
        return True, "LivePortrait active — real motion driving."

    def set_stabilization(self, level):
        """0 = raw/responsive, 1 = very smooth. Lowers the 1€ cutoff (more
        temporal smoothing) on pose/translation/scale/box; expression kept a bit
        more responsive so blinks survive."""
        self._stab = max(0.0, min(1.0, float(level)))
        cut = 3.0 - 2.6 * self._stab          # 3.0 (raw) -> 0.4 (very smooth)
        for k in ("pitch", "yaw", "roll", "t", "scale"):
            self._euro[k].min_cutoff = cut
        # keep pose beta HIGH so fast turns stay snappy (low lag) at ANY
        # stabilization level — stabilization only smooths SLOW/still motion.
        for k in ("pitch", "yaw", "roll"):
            self._euro[k].beta = POSE_BETA
        self._euro["exp"].min_cutoff = max(1.0, cut + 1.2)   # keep blinks crisp-ish
        self._euro_box.min_cutoff = max(0.6, cut)

    def set_gaze(self, on, strength=None):
        """Enable/disable gaze re-centering and set its strength (0..1)."""
        self.gaze_lock = bool(on)
        if strength is not None:
            self.gaze_strength = max(0.0, min(1.0, float(strength)))

    def recenter(self):
        """Drop the neutral reference so the NEXT driving frame becomes the new
        baseline. Call this while facing the camera upright/relaxed — it fixes a
        tilted head, off-axis gaze, or baked-in expression caused by the original
        reference frame being captured mid-motion."""
        self._ref_kp_info = None
        self._drive_box = None
        self._last_output = None
        self._cur_ref = 0
        self._miss = 0
        self._reset_filters()

    def _reset_filters(self):
        """Clear all 1€ filter history (avoids a jump after recenter / re-acquire)."""
        for f in self._euro.values():
            f.reset()
        self._euro_box.reset()

    def _knee_limit(self, delta, cap_deg):
        """Soft-clip with a KNEE: pass the angle through 1:1 up to KNEE_FRAC*cap
        (so normal turns follow you exactly), then smoothly ease to the cap. This
        avoids the tanh's habit of shrinking even moderate turns. Tensor-safe."""
        torch = self._torch
        knee = KNEE_FRAC * cap_deg
        span = max(1e-3, cap_deg - knee)
        ad = delta.abs()
        sgn = torch.sign(delta)
        excess = (ad - knee).clamp(min=0.0)
        mag = torch.where(ad <= knee, ad, knee + span * torch.tanh(excess / span))
        return sgn * mag

    def _soft_limit(self, angle_d, angle_ref, cap_deg):
        """Knee-limited absolute angle: follows 1:1 until ~KNEE_FRAC*cap, eases to cap."""
        return angle_ref + self._knee_limit(angle_d - angle_ref, cap_deg)

    def _limited_delta(self, angle_d, angle_ref, cap_deg):
        """Knee-limited DELTA (driving - reference)."""
        return self._knee_limit(angle_d - angle_ref, cap_deg)

    def _select_ref(self, dyaw):
        """Pick the reference view whose base yaw is nearest the desired turn,
        with hysteresis so it doesn't flicker between views at a boundary."""
        cur = self._refs[self._cur_ref]
        best = min(range(len(self._refs)),
                   key=lambda i: abs(self._refs[i]["base_yaw"] - dyaw))
        # only switch if the new view is meaningfully closer than the current one
        if abs(self._refs[best]["base_yaw"] - dyaw) + REF_SWITCH_HYST \
                < abs(cur["base_yaw"] - dyaw):
            self._cur_ref = best
        return self._refs[self._cur_ref]

    # -------------------------------------------------------------------------
    def _find_liveportrait(self):
        """Return the first valid LivePortrait path, or None."""
        for cand in LIVEPORTRAIT_CANDIDATES:
            if cand and os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "src")):
                return cand
        return None

    def _init_pipeline(self, lp_path):
        """Import the LivePortrait wrapper, build config, encode the source."""
        import torch
        if lp_path not in sys.path:
            sys.path.insert(0, lp_path)

        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix

        self._torch = torch
        self._get_rotation_matrix = get_rotation_matrix

        cfg = InferenceConfig()
        # Enable FP16 if the config exposes the flag (name varies by version).
        for attr in ("flag_use_half_precision", "flag_use_half"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, USE_HALF)
        self.wrapper = LivePortraitWrapper(inference_cfg=cfg)

        # --- encode source ONCE (the expensive part) ---
        src_rgb = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
        I_s = self.wrapper.prepare_source(src_rgb)
        self._x_s_info = self.wrapper.get_kp_info(I_s)
        self._R_s = get_rotation_matrix(self._x_s_info["pitch"],
                                        self._x_s_info["yaw"],
                                        self._x_s_info["roll"])
        self._f_s = self.wrapper.extract_feature_3d(I_s)
        self._x_s = self.wrapper.transform_keypoint(self._x_s_info)

        # --- confirm FP16/CUDA + source caching (per perf checklist) ---
        try:
            dev = next(self.wrapper.warping_module.parameters()).device
            fp16 = bool(getattr(self.wrapper.inference_cfg,
                                "flag_use_half_precision", False))
            cached = all(getattr(self, a, None) is not None
                         for a in ("_f_s", "_x_s", "_x_s_info", "_R_s"))
            print(f"[LP] device={dev} | FP16(autocast)={fp16} | "
                  f"source features cached ONCE={cached} (not per-frame)")
        except Exception:
            pass

        # --- build the multi-reference view set (frontal + generated turns) ---
        self._frontal_yaw = float(self._x_s_info["yaw"])
        self._refs = [dict(f=self._f_s, xs=self._x_s, kp=self._x_s_info["kp"],
                           exp=self._x_s_info["exp"], scale=self._x_s_info["scale"],
                           t=self._x_s_info["t"], R=self._R_s, base_yaw=0.0)]
        self._multi = False
        self._cur_ref = 0
        self._multi_yaw_cap = YAW_CAP_MULTI    # clamped to real coverage below

        # 1) REAL multi-angle views extracted from a video (character_views/yaw_*.jpg)
        #    take priority — real side data, identity-exact, no hallucination.
        real_views = sorted(glob.glob(os.path.join(CHARACTER_VIEWS_DIR, "yaw_*.jpg")))
        if real_views:
            for path in real_views:
                try:
                    bgr = cv2.imread(path)
                    if bgr is None:
                        continue
                    ref = self._encode_source_bgr(bgr)
                    ref["base_yaw"] = float(ref["_yaw"]) - self._frontal_yaw
                    self._refs.append(ref)
                except Exception as exc:
                    print(f"[LP] real view {os.path.basename(path)} failed ({exc}).")
            self._refs.sort(key=lambda r: r["base_yaw"])
            bases = ", ".join(f"{r['base_yaw']:+.0f}" for r in self._refs)
            cov = max(abs(r["base_yaw"]) for r in self._refs)
            self._multi_yaw_cap = min(YAW_CAP_MULTI, cov + 8.0)
            # IMPORTANT: views pooled from DIFFERENT videos have different body /
            # clothing / background, so switching them mid-turn makes the body jump
            # ("face vs body" mismatch). So multi-view is OFF by default — single
            # consistent source = body never changes. Enable only when the views
            # come from ONE session (env AVATAR_MULTIREF=1 or the Studio checkbox).
            self._multi = MULTI_REF and len(self._refs) > 1
            mode = "ON" if self._multi else "OFF (single consistent source)"
            print(f"[LP] {len(real_views)} real views loaded (yaw {bases}); "
                  f"multi-view {mode}. Cap ~{self._multi_yaw_cap:.0f}deg.")
        # 2) otherwise, optionally generate views (off by default; can bend).
        elif MULTI_REF:
            for off in MULTI_REF_YAWS:
                try:
                    view = self._render_pose_offset(off)        # character turned 'off'
                    ref = self._encode_source_bgr(view)
                    ref["base_yaw"] = float(ref["_yaw"]) - self._frontal_yaw
                    self._refs.append(ref)
                except Exception as exc:
                    print(f"[LP] multi-ref view {off:+.0f} failed ({exc}) — skipped.")
            self._refs.sort(key=lambda r: r["base_yaw"])
            self._multi = len(self._refs) > 1
            bases = ", ".join(f"{r['base_yaw']:+.0f}" for r in self._refs)
            print(f"[LP] multi-reference: {len(self._refs)} generated views (yaw {bases}).")

    def _render_pose_offset(self, yaw_off_deg, pitch_off_deg=0.0):
        """Render the frontal source turned by an absolute pose offset (BGR).

        Uses the same warp the live path uses (clean within +/-~30deg). The result
        is re-encoded as an additional reference view of the SAME identity."""
        torch = self._torch
        z = torch.zeros_like(self._x_s_info["yaw"])
        R_rel = self._get_rotation_matrix(z + float(pitch_off_deg),
                                          z + float(yaw_off_deg), z)
        R_new = R_rel @ self._R_s
        x_d = self._x_s_info["scale"] * (self._x_s_info["kp"] @ R_new
                                         + self._x_s_info["exp"]) + self._x_s_info["t"]
        try:
            x_d = self.wrapper.stitching(self._x_s, x_d)
        except Exception:
            pass
        out = self.wrapper.warp_decode(self._f_s, self._x_s, x_d)
        rgb = self.wrapper.parse_output(out["out"])[0]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _encode_source_bgr(self, bgr):
        """Encode a BGR image as a LivePortrait source (returns a ref dict)."""
        rgb = cv2.cvtColor(cv2.resize(bgr, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB)
        I = self.wrapper.prepare_source(rgb)
        info = self.wrapper.get_kp_info(I)
        R = self._get_rotation_matrix(info["pitch"], info["yaw"], info["roll"])
        f = self.wrapper.extract_feature_3d(I)
        xs = self.wrapper.transform_keypoint(info)
        return dict(f=f, xs=xs, kp=info["kp"], exp=info["exp"], scale=info["scale"],
                    t=info["t"], R=R, base_yaw=0.0, _yaw=info["yaw"])

    def _kp_info_from_crop(self, cropped_bgr):
        """Run the LivePortrait encoder on an (already-cropped) frame -> kp info."""
        rgb = cv2.cvtColor(cv2.resize(cropped_bgr, (FRAME_SIZE, FRAME_SIZE)),
                           cv2.COLOR_BGR2RGB)
        I_d = self.wrapper.prepare_source(rgb)
        return self.wrapper.get_kp_info(I_d)

    # -------------------------------------------------------------------------
    def _get_mesh(self):
        """Lazily create a MediaPipe face detector (None if unavailable).

        Empirically, the short-range face_detection model (model_selection=0)
        tracks a webcam operator at ~100% when they face the camera — far better
        than FaceMesh (~39%) or the full-range model (~51%). It is also the
        fastest. Named _mesh for historical reasons.
        """
        if self._mesh_tried:
            return self._mesh
        self._mesh_tried = True
        try:
            import mediapipe as mp
            # lower confidence catches more frames (fewer track drops) — the
            # grace window in _crop_driving_face handles the occasional miss.
            self._mesh = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3)
        except Exception as exc:
            print(f"[LP] face detector unavailable ({exc}) — driving uncropped.")
            self._mesh = None
        return self._mesh

    def _crop_driving_face(self, frame):
        """Return a centered square crop around the detected face (EMA-smoothed).

        Sets self._face_found = True only when a face is detected AND large enough
        (>= MIN_GOOD_FACE) to drive cleanly. A detected-but-too-small face (operator
        far away) updates the crop box for tracking continuity but leaves
        _face_found False, so process_frame HOLDS the last good output rather than
        rendering a degraded face from an upscaled low-detail crop. self._face_size
        holds the detected size fraction (0 = none) for callers/diagnostics.
        """
        self._face_found = False
        mesh = self._get_mesh()
        h, w = frame.shape[:2]
        if mesh is None:
            return frame
        try:
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box = None
            if res.detections:
                # Largest face = the foreground operator (ignore background people).
                r = max((d.location_data.relative_bounding_box
                         for d in res.detections),
                        key=lambda b: b.width * b.height)
                fw, fh = r.width * w, r.height * h
                self._face_size = max(fw, fh) / max(w, h)
                if self._face_size >= DRIVING_MIN_FACE:
                    cx = (r.xmin + r.width / 2) * w
                    cy = (r.ymin + r.height / 2) * h
                    side = max(fw, fh) * DRIVING_CROP_PAD
                    box = np.array([cx, cy, side], dtype=np.float32)
                    self._face_found = self._face_size >= self.min_good_face
                    self._miss = 0                     # detected -> reset grace
            else:
                # GRACE WINDOW: a detector blip shouldn't freeze the avatar. For a
                # few frames after a miss, keep DRIVING with the last crop box (your
                # head barely moves in ~5 frames) instead of holding. Only after
                # sustained loss do we drop _face_found (-> hold / charts).
                self._miss = getattr(self, "_miss", 0) + 1
                if self._miss <= MISS_GRACE and self._drive_box is not None:
                    self._face_found = True            # still tracking (last box)
                else:
                    self._face_size = 0.0              # truly lost

            if box is None:
                if self._drive_box is None:
                    return frame                       # no face yet -> full frame
                box = self._drive_box                  # reuse last box for the crop

            # 1€-filter the crop box so the AI head framing doesn't jitter
            # (adaptive: snaps on fast moves, steady when you hold still).
            if SMOOTH_MOTION and self._face_found:
                self._drive_box = self._euro_box(box)
            else:
                self._drive_box = box

            cx, cy, side = self._drive_box
            half = side / 2.0
            x1 = int(cx - half); y1 = int(cy - half)
            x2 = int(cx + half); y2 = int(cy + half)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                return frame
            top = max(0, -y1); left = max(0, -x1)
            bottom = max(0, y2 - h); right = max(0, x2 - w)
            if top or bottom or left or right:
                crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                          cv2.BORDER_REPLICATE)
            return crop
        except Exception:
            return frame

    # -------------------------------------------------------------------------
    def process_frame(self, driving_frame):
        """Animate the source with the driving frame's motion (512x512 BGR)."""
        if self.fallback_mode:
            return self.source_image.copy()

        try:
            # Track + crop the face first. If no face is visible this frame, HOLD
            # the last good animated face rather than feed garbage (hair / empty
            # background) to the motion extractor — which renders a contorted
            # grimace and makes tracking look "broken".
            crop = self._crop_driving_face(driving_frame)
            if not self._face_found and HOLD_ON_FACE_LOSS \
                    and self._last_output is not None:
                return self._last_output.copy()

            x_d_info = self._kp_info_from_crop(crop)

            # 1€-filter the driving motion so the avatar tracks smoothly with no
            # frame-to-frame shimmer (the jitter that read as "tracking errors").
            if SMOOTH_MOTION:
                ts = time.monotonic()
                for k in ("pitch", "yaw", "roll", "t", "scale", "exp"):
                    try:
                        x_d_info[k] = self._euro[k](x_d_info[k], ts)
                    except Exception:
                        pass

            # Establish the relative-motion reference on the first frame that
            # actually has a face (never on a faceless frame).
            if self._ref_kp_info is None:
                if not self._face_found:
                    return (self._last_output.copy() if self._last_output is not None
                            else self.source_image.copy())
                self._ref_kp_info = x_d_info

            ref = self._ref_kp_info
            # PITCH/ROLL AUTO-CENTER: slowly drift the neutral toward your average
            # head pitch/roll so a baked-in offset (e.g. neutral captured while you
            # looked DOWN at the screen -> avatar stuck looking UP) self-corrects to
            # level over a few seconds. Your actual up/down/tilt MOVEMENT still
            # comes through as the delta; only the constant bias decays. Yaw is left
            # fully responsive (no auto-center) so turns aren't pulled back.
            if AUTOCENTER_PITCH > 0.0:
                try:
                    for k in ("pitch", "roll"):
                        ref[k] = ref[k] * (1.0 - AUTOCENTER_PITCH) + x_d_info[k] * AUTOCENTER_PITCH
                except Exception:
                    pass
            # GAZE LOCK: pull the expression delta toward the source (camera-
            # facing) so the eyes/face don't wander off-camera. Strength scales
            # how much of YOUR expression deviation is kept (blinks largely
            # survive as they're fast & symmetric; slow gaze drift is suppressed).
            if self.gaze_lock and self.gaze_strength > 0.01:
                g = min(0.92, self.gaze_strength * 0.9)
                try:
                    x_d_info["exp"] = ref["exp"] + (x_d_info["exp"] - ref["exp"]) * (1.0 - g)
                except Exception:
                    pass
            # WIDE-MOUTH GUARD: soft-clamp the per-element expression delta so a
            # wide phoneme / open mouth can't balloon the lower face.
            if MOUTH_GUARD > 0.0:
                try:
                    d = x_d_info["exp"] - ref["exp"]
                    x_d_info["exp"] = ref["exp"] + MOUTH_GUARD * self._torch.tanh(d / MOUTH_GUARD)
                except Exception:
                    pass
            if self._multi:
                bgr = self._drive_multiref(x_d_info, ref)
            else:
                bgr = self._drive_single(x_d_info, ref)
            self._last_output = bgr                              # for hold-on-loss
            return bgr
        except Exception as exc:
            if not self._error_printed:
                print(f"[LP] frame error ({exc}) — returning source for remaining errors.")
                self._error_printed = True
            if self._last_output is not None:
                return self._last_output.copy()
            return self.source_image.copy()

    def _drive_single(self, x_d_info, ref):
        """Single-source relative reenactment with pose-aware soft limiting."""
        if POSE_LIMIT:
            pit = self._soft_limit(x_d_info["pitch"], ref["pitch"], PITCH_CAP)
            yaw = self._soft_limit(x_d_info["yaw"], ref["yaw"], YAW_CAP)
            rol = self._soft_limit(x_d_info["roll"], ref["roll"], ROLL_CAP)
        else:
            pit, yaw, rol = x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]
        R_d = self._get_rotation_matrix(pit, yaw, rol)
        R_d0 = self._get_rotation_matrix(ref["pitch"], ref["yaw"], ref["roll"])
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self._R_s
        delta_new = self._x_s_info["exp"] + (x_d_info["exp"] - ref["exp"])
        # FREEZE scale (clamped to a tiny band): the driving scale estimate jumps
        # when you turn your head, which used to ZOOM the avatar's face. A talking
        # head shouldn't zoom on rotation, so we clamp the scale ratio tight.
        ratio = (x_d_info["scale"] / ref["scale"]).clamp(1.0 - SCALE_BAND, 1.0 + SCALE_BAND)
        scale_new = self._x_s_info["scale"] * ratio
        t_new = self._x_s_info["t"] + (x_d_info["t"] - ref["t"])
        t_new[..., 2] = 0
        x_d_new = scale_new * (self._x_s_info["kp"] @ R_new + delta_new) + t_new
        try:
            x_d_new = self.wrapper.stitching(self._x_s, x_d_new)
        except Exception:
            pass
        out = self.wrapper.warp_decode(self._f_s, self._x_s, x_d_new)
        result = self.wrapper.parse_output(out["out"])[0]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _drive_multiref(self, x_d_info, ref):
        """Multi-reference reenactment. The TARGET head pose is computed exactly
        like the single-source path (so combined yaw+pitch+roll never skews); only
        the APPEARANCE swaps to the generated side-view nearest the turn, so the
        warp from that view to the target is small and renders real side geometry.
        """
        # absolute target rotation — identical formulation to single source, with
        # the extended multi-ref yaw cap (others unchanged).
        pit = self._soft_limit(x_d_info["pitch"], ref["pitch"], PITCH_CAP)
        yaw = self._soft_limit(x_d_info["yaw"], ref["yaw"], self._multi_yaw_cap)
        rol = self._soft_limit(x_d_info["roll"], ref["roll"], ROLL_CAP)
        R_d = self._get_rotation_matrix(pit, yaw, rol)
        R_d0 = self._get_rotation_matrix(ref["pitch"], ref["yaw"], ref["roll"])
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self._R_s     # canonical -> target pose

        dyaw = float(yaw - ref["yaw"])                  # soft-limited yaw delta (deg)
        view = self._select_ref(dyaw)                   # nearest side-view (hysteresis)

        delta_new = view["exp"] + (x_d_info["exp"] - ref["exp"])
        ratio = (x_d_info["scale"] / ref["scale"]).clamp(1.0 - SCALE_BAND, 1.0 + SCALE_BAND)
        scale_new = view["scale"] * ratio
        t_new = view["t"] + (x_d_info["t"] - ref["t"])
        t_new[..., 2] = 0
        # canonical kp of the chosen view, posed to the SAME absolute target; the
        # warp then runs from the view's base pose to the target (small residual).
        x_d_new = scale_new * (view["kp"] @ R_new + delta_new) + t_new
        try:
            x_d_new = self.wrapper.stitching(view["xs"], x_d_new)
        except Exception:
            pass
        out = self.wrapper.warp_decode(view["f"], view["xs"], x_d_new)
        result = self.wrapper.parse_output(out["out"])[0]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------------------------
    def _warmup(self):
        """Run a few dummy frames so CUDA kernels are tuned before the loop."""
        dummy = self.source_image.copy()
        for _ in range(5):
            self.process_frame(dummy)
        self._ref_kp_info = None         # reset reference after warm-up
        self._drive_box = None           # reset crop smoothing after warm-up
        self._last_output = None         # reset hold-on-loss cache after warm-up
        self._reset_filters()            # reset 1€ filters after warm-up

    def _print_gpu(self):
        """Print GPU memory currently allocated."""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[LP] GPU memory: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
        except Exception:
            pass


if __name__ == "__main__":
    char = os.path.join(PROJECT_DIR, "ai-face", "character.jpg")
    if not os.path.exists(char):
        char = os.path.join(PROJECT_DIR, "character.jpg")
    eng = LivePortraitEngine(char)
    print("[LP] startup_check:", eng.startup_check())
    out = eng.process_frame(np.full((512, 512, 3), 90, dtype=np.uint8))
    print("[LP] output shape:", out.shape)
