# =============================================================================
# engines/enhance_engine.py
# -----------------------------------------------------------------------------
# Visual polish for the final avatar frame: studio background compositing, color
# grade, vignette, subtle camera shake, scrolling ticker, LIVE badge, and a
# speaking indicator. Also generates the studio background programmatically.
#
# All heavy assets (background image, vignette mask, segmenter) are built once
# and cached at module level — nothing is recomputed per frame beyond the draw.
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
STUDIO_BG_PATH = os.path.join(PROJECT_DIR, "studio_background.jpg")

TICKER_TEXT = "XAUUSD   -   GOLD SIGNALS   -   @xauusa2   -   LIVE TRADING"
TICKER_SPEED = 2
TICKER_COLOR = (200, 200, 200)
TICKER_BG = (15, 15, 15)
BADGE_COLOR_GOLD = (0, 215, 255)
LIVE_DOT_COLOR = (0, 0, 220)
VIGNETTE_STRENGTH = 0.18       # softer vignette (less edge darkening)
CAMERA_SHAKE_AMOUNT = 1.5
GRAIN_STRENGTH = 8.0            # luminance sensor grain (real cameras aren't clean)
SOFTEN = 0.06                  # minimal — heavy soften made the swap look too smooth
EDGE_FEATHER = 4                # selfie-seg edge softness (px) — tight, no halo
FRAME_SIZE = 512

# -----------------------------------------------------------------------------
# MODULE STATE (persists between frames)
# -----------------------------------------------------------------------------
_ticker_offset = 0
_live_dot_phase = 0
_shake_state = np.array([0.0, 0.0])
_bg_image = None
_vignette_mask = None
_segmenter = None
_segmenter_tried = False
_seg_mask_cache = None
_seg_counter = 0
SEG_INTERVAL = 2               # every 2 frames: tracks head, less added latency


# -----------------------------------------------------------------------------
# STUDIO BACKGROUND GENERATION
# -----------------------------------------------------------------------------
def generate_studio_background(output_path=STUDIO_BG_PATH, size=(FRAME_SIZE, FRAME_SIZE)):
    """Generate a photoreal, heavily out-of-focus BOKEH background.

    Real streamer backgrounds are a blurred room with soft circular highlights
    (out-of-focus lights) — once blurred enough this is indistinguishable from a
    real shallow-depth-of-field photo, and reads as "real camera" not "CG".
    """
    w, h = size
    rng = np.random.RandomState(7)

    # Warm-to-cool vertical gradient base (dim room).
    top = np.array([22, 18, 16], np.float32)        # BGR, cool-dark top
    bot = np.array([30, 34, 46], np.float32)        # warmer toward the desk
    grad = (top[None, None] + (bot - top)[None, None] *
            (np.linspace(0, 1, h)[:, None, None])).astype(np.float32)
    img = np.repeat(grad, w, axis=1)

    # Scatter soft bokeh discs (out-of-focus highlights), warm + cool.
    palette = [(60, 110, 200), (40, 90, 170), (150, 140, 90),
               (120, 160, 210), (70, 70, 110), (180, 170, 120)]
    for _ in range(26):
        cx, cy = rng.randint(0, w), rng.randint(0, int(h * 0.95))
        r = rng.randint(18, 70)
        color = palette[rng.randint(len(palette))]
        alpha = rng.uniform(0.12, 0.45)
        disc = np.zeros((h, w, 3), np.float32)
        cv2.circle(disc, (cx, cy), r, color, -1)
        cv2.circle(disc, (cx, cy), r, tuple(int(c * 1.3) for c in color), 3)  # bright rim
        img += disc * alpha

    # Heavy blur = shallow depth of field (the whole bg is out of focus).
    img = cv2.GaussianBlur(img, (0, 0), 18)
    # A couple of brighter, larger soft glows for depth.
    for cx, cy, r, col, a in [(int(w*0.75), int(h*0.3), 150, (50, 95, 175), 0.5),
                              (int(w*0.18), int(h*0.6), 130, (130, 150, 90), 0.35)]:
        g = np.zeros((h, w, 3), np.float32)
        cv2.circle(g, (cx, cy), r, col, -1)
        img += cv2.GaussianBlur(g, (0, 0), 60) * a

    img = np.clip(img, 0, 255).astype(np.uint8)
    # Fine sensor grain so the bg itself isn't a clean CG gradient.
    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.5, (h, w, 3)), 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return img


# -----------------------------------------------------------------------------
# STARTUP / ASSET LOADING
# -----------------------------------------------------------------------------
def startup_check():
    """Ensure background + vignette exist. Returns (ok, message)."""
    _ensure_assets()
    seg = "selfie-seg on" if _get_segmenter() is not None else "selfie-seg off (no bg cut)"
    return True, f"enhance ready ({seg})"


def _ensure_assets():
    """Load/generate the background image and vignette mask once."""
    global _bg_image, _vignette_mask
    if _bg_image is None:
        if os.path.exists(STUDIO_BG_PATH):
            bg = cv2.imread(STUDIO_BG_PATH)
            _bg_image = cv2.resize(bg, (FRAME_SIZE, FRAME_SIZE)) if bg is not None \
                else generate_studio_background()
        else:
            _bg_image = generate_studio_background()
    if _vignette_mask is None:
        _vignette_mask = _build_vignette()


def _build_vignette():
    """Elliptical vignette mask: 1.0 center -> 0.65 edges."""
    h = w = FRAME_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = 1.0 - np.clip(d, 0, 1) * VIGNETTE_STRENGTH
    return np.clip(mask, 0.65, 1.0).astype(np.float32)[:, :, None]


def _get_segmenter():
    """Lazily create the MediaPipe selfie segmenter (None if unavailable)."""
    global _segmenter, _segmenter_tried
    if _segmenter_tried:
        return _segmenter
    _segmenter_tried = True
    try:
        import mediapipe as mp
        _segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    except Exception:
        _segmenter = None
    return _segmenter


# -----------------------------------------------------------------------------
# PIPELINE STEPS
# -----------------------------------------------------------------------------
def background_composite(frame):
    """Cut the person out and composite onto the studio background.

    The segmentation mask is recomputed only every SEG_INTERVAL frames (the head
    moves slowly), which roughly thirds the per-frame cost of this stage.
    """
    global _seg_mask_cache, _seg_counter
    seg = _get_segmenter()
    if seg is None:
        return frame
    try:
        _seg_counter += 1
        # recompute EVERY frame so the cut tracks your head — a stale/lagged mask
        # let the background cut into your moving face.
        if _seg_mask_cache is None or (_seg_counter % SEG_INTERVAL) == 0:
            res = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.segmentation_mask is not None:
                # DILATE slightly (margin) so a fast-moving head stays INSIDE the
                # person cut — better to keep a sliver of real edge than to slice
                # the head with the background.
                m = (res.segmentation_mask > 0.5).astype(np.float32)
                m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
                m = cv2.GaussianBlur(m, (0, 0), EDGE_FEATHER)
                # light temporal blend only (responsive, not lagging the head)
                if _seg_mask_cache is not None and _seg_mask_cache.shape[:2] == m.shape[:2]:
                    m = 0.85 * m + 0.15 * _seg_mask_cache[:, :, 0]
                _seg_mask_cache = m[:, :, None]
        if _seg_mask_cache is None or _seg_mask_cache.shape[:2] != frame.shape[:2]:
            return frame
        bg = _bg_image
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        return (frame * _seg_mask_cache + bg * (1.0 - _seg_mask_cache)).astype(np.uint8)
    except Exception:
        return frame


def apply_color_grade(frame):
    """Bright studio-lit streamer grade: exposure + SHADOW LIFT (so the face is
    never dark), gentle warm key, mild contrast. Soft, flattering — not crushed."""
    f = frame.astype(np.float32) / 255.0
    f = np.clip(f * 1.12, 0, 1)                             # exposure lift (brighter)
    f = np.power(f, 0.86)                                   # gamma <1 = lift shadows/mids
    luma = (0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2])[:, :, None]
    hi = np.clip((luma - 0.5) * 2.0, 0, 1)
    f[:, :, 2] = np.clip(f[:, :, 2] + hi[:, :, 0] * 0.030 + 0.015, 0, 1)  # warm highlights
    f[:, :, 1] = np.clip(f[:, :, 1] + hi[:, :, 0] * 0.008 + 0.004, 0, 1)
    f = np.clip((f - 0.5) * 1.05 + 0.5, 0, 1)              # mild contrast (keeps it bright)
    return (f * 255).astype(np.uint8)


def apply_soften(frame):
    """Blend in a tiny blur to take the edge off the over-sharp 'AI' look."""
    if SOFTEN <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 - SOFTEN, blur, SOFTEN, 0)


def apply_clarity(frame, amount=0.30):
    """Local-contrast (clarity) — large-radius unsharp mask. Brings back skin
    micro-texture/pores that enhancement smooths away, so it reads as real skin
    through a lens, not a clean AI render."""
    blur = cv2.GaussianBlur(frame, (0, 0), 3.0)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def _scale_channel(ch, s, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), 0, s)
    return cv2.warpAffine(ch, M, (ch.shape[1], ch.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def apply_chromatic_aberration(frame, amt=0.0016):
    """Subtle lens chromatic aberration: scale R out / B in around centre so the
    edges get a faint colour fringe — a real-lens characteristic that sells it as
    camera footage rather than a perfect digital composite."""
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    b, g, r = cv2.split(frame)
    r = _scale_channel(r, 1.0 + amt, cx, cy)
    b = _scale_channel(b, 1.0 - amt, cx, cy)
    return cv2.merge([b, g, r])


def apply_grain(frame):
    """Add monochromatic luminance grain — real sensors are never perfectly clean."""
    if GRAIN_STRENGTH <= 0:
        return frame
    h, w = frame.shape[:2]
    noise = (np.random.randn(h, w, 1) * GRAIN_STRENGTH).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_vignette(frame):
    """Multiply by the cached vignette mask."""
    mask = _vignette_mask
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask[:, :, 0], (frame.shape[1], frame.shape[0]))[:, :, None]
    return (frame.astype(np.float32) * mask).astype(np.uint8)


def apply_camera_shake(frame):
    """Subtle ±1-2px random-walk translation to feel alive."""
    global _shake_state
    _shake_state = _shake_state * 0.9 + np.random.uniform(
        -CAMERA_SHAKE_AMOUNT, CAMERA_SHAKE_AMOUNT, 2) * 0.1
    dx, dy = float(_shake_state[0]), float(_shake_state[1])
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def draw_ticker(frame):
    """Scrolling bottom news bar."""
    global _ticker_offset
    h, w = frame.shape[:2]
    bar_h = 36
    y0 = h - bar_h
    cv2.rectangle(frame, (0, y0), (w, h), TICKER_BG, -1)
    text = (TICKER_TEXT + "     ") * 3
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    _ticker_offset = (_ticker_offset - TICKER_SPEED) % tw
    x = -_ticker_offset
    while x < w:
        cv2.putText(frame, text, (x, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    TICKER_COLOR, 1, cv2.LINE_AA)
        x += tw
    return frame


def draw_live_badge(frame):
    """Top-left LIVE badge with a pulsing red dot and gold accents."""
    global _live_dot_phase
    _live_dot_phase = (_live_dot_phase + 1) % 60
    pulse = 0.5 + 0.5 * abs(30 - _live_dot_phase) / 30.0
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (150, 46), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (12, 12), (150, 46), BADGE_COLOR_GOLD, 1)
    dot = tuple(int(c * pulse) for c in LIVE_DOT_COLOR)
    cv2.circle(frame, (28, 29), 6, dot, -1)
    cv2.putText(frame, "LIVE", (42, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 60, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, "XAUUSD", (90, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                BADGE_COLOR_GOLD, 1, cv2.LINE_AA)
    return frame


def draw_speaking_indicator(frame, is_speaking):
    """Top-right circle: green + AI HOST when speaking, gray when idle."""
    w = frame.shape[1]
    color = (0, 220, 0) if is_speaking else (110, 110, 110)
    cv2.circle(frame, (w - 24, 28), 8, color, -1)
    cv2.circle(frame, (w - 24, 28), 8, (255, 255, 255), 1)
    if is_speaking:
        cv2.putText(frame, "AI HOST", (w - 96, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


# -----------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -----------------------------------------------------------------------------
# Performance level (set by the Quality preset). "full" = everything;
# "light" = skip the costly background segmentation + grain/soften (saves ~20ms);
# overlays (ticker/badges) always stay since they're cheap.
ENHANCE_LEVEL = "full"          # "full" | "light"


def set_level(level):
    global ENHANCE_LEVEL
    ENHANCE_LEVEL = level if level in ("full", "light") else "full"


def enhance_frame(frame, is_speaking=False):
    """Run the polish pipeline. Never raises — returns a frame regardless.

    'light' level skips background segmentation + grain/soften (the ~20ms CPU
    cost) for higher fps, keeping the cheap overlays and grade.
    """
    try:
        _ensure_assets()
        if frame.shape[0] != FRAME_SIZE or frame.shape[1] != FRAME_SIZE:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        full = ENHANCE_LEVEL == "full"
        if full:
            frame = background_composite(frame)
            frame = apply_clarity(frame)          # skin micro-texture (real lens look)
        frame = apply_color_grade(frame)
        frame = apply_vignette(frame)
        frame = apply_camera_shake(frame)
        if full:
            frame = apply_chromatic_aberration(frame)   # subtle lens fringe
            frame = apply_grain(frame)            # sensor grain (before UI overlays)
        frame = draw_ticker(frame)
        frame = draw_live_badge(frame)
        frame = draw_speaking_indicator(frame, is_speaking)
        return frame
    except Exception as exc:
        print(f"[ENHANCE] frame error ({exc}) — returning unmodified frame.")
        return frame


if __name__ == "__main__":
    print("[ENHANCE]", startup_check()[1])
    generate_studio_background()
    test = np.full((FRAME_SIZE, FRAME_SIZE, 3), 90, dtype=np.uint8)
    out = enhance_frame(test, is_speaking=True)
    cv2.imwrite(os.path.join(PROJECT_DIR, "_enhance_test.jpg"), out)
    print("[ENHANCE] wrote _enhance_test.jpg", out.shape)
