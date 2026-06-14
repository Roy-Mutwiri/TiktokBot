"""Procedural trading-studio backgrounds and lighting themes."""

import hashlib
import math

import cv2
import numpy as np


SCENES = [
    "Wall Street LED", "Institutional Desk", "Crypto Command Center",
    "Gold Market Studio", "Futures Control Room", "Forex Newsroom",
    "Quant Lab", "Glass Chart Wall", "Market Data Tunnel", "Exchange Floor",
    "Risk Desk", "Analyst Loft",
]

LIGHTS = [
    "Midnight Blue", "Cyan Edge", "Amber Key", "Teal Magenta",
    "Emerald Pulse", "Red Blue", "Gold Luxury", "Ice White",
    "Violet Futures", "Tokyo Neon",
]

BACKGROUND_PRESETS = ["No Background"] + [
    f"{scene} / {light}" for scene in SCENES for light in LIGHTS
]

_LIGHT_BGR = {
    "Midnight Blue": ((38, 22, 10), (100, 58, 24)),
    "Cyan Edge": ((58, 34, 8), (180, 110, 20)),
    "Amber Key": ((20, 62, 110), (42, 125, 230)),
    "Teal Magenta": ((90, 48, 8), (105, 25, 150)),
    "Emerald Pulse": ((42, 72, 10), (55, 155, 45)),
    "Red Blue": ((85, 20, 12), (24, 25, 150)),
    "Gold Luxury": ((14, 55, 95), (20, 135, 230)),
    "Ice White": ((60, 58, 54), (150, 145, 135)),
    "Violet Futures": ((75, 24, 42), (130, 45, 135)),
    "Tokyo Neon": ((95, 35, 18), (145, 40, 110)),
}

_cache = {}


def split_preset(name):
    if not name or name == "No Background":
        return None, None
    parts = name.split(" / ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (SCENES[0], LIGHTS[0])


def _seed(name):
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def _chart_panel(img, rect, rng, accent, dense=False):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (13, 18, 27), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), accent, 1, cv2.LINE_AA)
    for i in range(1, 5):
        y = y1 + int((y2 - y1) * i / 5)
        cv2.line(img, (x1, y), (x2, y), (27, 38, 50), 1)
    for i in range(1, 7):
        x = x1 + int((x2 - x1) * i / 7)
        cv2.line(img, (x, y1), (x, y2), (23, 33, 45), 1)

    n = 28 if dense else 18
    xs = np.linspace(x1 + 8, x2 - 8, n).astype(int)
    price = (y1 + y2) / 2
    scale = max(3.0, (y2 - y1) * 0.045)
    for x in xs:
        o = price
        c = np.clip(o + rng.normal(0, scale), y1 + 10, y2 - 10)
        hi = max(o, c) + rng.uniform(2, scale)
        lo = min(o, c) - rng.uniform(2, scale)
        col = (80, 220, 120) if c < o else (75, 80, 235)
        cv2.line(img, (x, int(lo)), (x, int(hi)), col, 1)
        cv2.rectangle(img, (x - 2, int(min(o, c))), (x + 2, int(max(o, c)) + 1), col, -1)
        price = c


def render_background(name, size=(512, 512), phase=0.0):
    """Render a deterministic trading set. `phase` animates subtle light sweeps."""
    if name == "No Background":
        return None
    w, h = size
    scene, light = split_preset(name)
    cache_key = (name, w, h)
    base = _cache.get(cache_key)
    if base is None:
        rng = np.random.default_rng(_seed(name))
        c1, c2 = _LIGHT_BGR.get(light, _LIGHT_BGR["Midnight Blue"])
        yy, xx = np.mgrid[0:h, 0:w]
        vertical = (yy / max(1, h - 1))[:, :, None]
        radial = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.42) / h) ** 2)
        img = np.zeros((h, w, 3), np.float32)
        img[:] = np.array((8, 11, 17), np.float32)
        img += vertical * np.array(c1, np.float32) * 0.20
        img += np.clip(0.55 - radial, 0, 0.55)[:, :, None] * np.array(c2, np.float32) * 0.38

        img = np.clip(img, 0, 255).astype(np.uint8)
        accent = tuple(int(v) for v in c2)
        layout = SCENES.index(scene) if scene in SCENES else 0
        if layout % 4 == 0:
            _chart_panel(img, (24, 55, 210, 235), rng, accent)
            _chart_panel(img, (302, 48, 490, 238), rng, accent, True)
        elif layout % 4 == 1:
            _chart_panel(img, (12, 70, 180, 260), rng, accent)
            _chart_panel(img, (332, 70, 500, 260), rng, accent)
        elif layout % 4 == 2:
            _chart_panel(img, (35, 38, 477, 180), rng, accent, True)
        else:
            _chart_panel(img, (20, 45, 245, 210), rng, accent)
            _chart_panel(img, (267, 45, 492, 210), rng, accent)

        # Desk and architectural depth stay behind the subject.
        cv2.rectangle(img, (0, int(h * 0.78)), (w, h), (10, 13, 18), -1)
        cv2.line(img, (0, int(h * 0.78)), (w, int(h * 0.78)), accent, 2, cv2.LINE_AA)
        for x in range(-w, w * 2, 70):
            cv2.line(img, (w // 2, int(h * 0.78)), (x, h), (20, 28, 38), 1)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        _cache[cache_key] = img
        base = img

    out = base.copy()
    scene, light = split_preset(name)
    c1, c2 = _LIGHT_BGR.get(light, _LIGHT_BGR["Midnight Blue"])
    sweep_x = int((0.5 + 0.45 * math.sin(phase)) * w)
    glow = np.zeros_like(out, dtype=np.float32)
    cv2.circle(glow, (sweep_x, int(h * 0.18)), int(w * 0.28), c2, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), w * 0.12)
    return np.clip(out.astype(np.float32) + glow * 0.10, 0, 255).astype(np.uint8)


def apply_subject_lighting(frame, mask, preset, strength=0.22):
    """Apply theme light only to foreground pixels; background cannot leak in."""
    if preset == "No Background" or strength <= 0:
        return frame
    _, light = split_preset(preset)
    c1, c2 = _LIGHT_BGR.get(light, _LIGHT_BGR["Midnight Blue"])
    h, w = frame.shape[:2]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    tint = np.array(c1, np.float32)[None, None, :] * (1.0 - x)
    tint += np.array(c2, np.float32)[None, None, :] * x
    alpha = np.clip(mask, 0, 1) * float(strength)
    lit = frame.astype(np.float32) * (1.0 - alpha) + np.clip(
        frame.astype(np.float32) * 0.78 + tint, 0, 255) * alpha
    return np.clip(lit, 0, 255).astype(np.uint8)
