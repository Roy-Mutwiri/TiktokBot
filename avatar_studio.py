# =============================================================================
# avatar_studio.py  —  all-in-one test studio with LIVE preview
# -----------------------------------------------------------------------------
# One window to test the whole avatar without OBS:
#   * START / STOP the pipeline
#   * SEE the final composited frame live (webcam -> AI face -> mouth sync ->
#     studio overlays) right in the window
#   * type text + SPEAK so you can watch the mouth sync to the AI voice
#   * mute, pick a voice, optionally also push to the OBS virtual camera
#
#   python avatar_studio.py
#
# The heavy engines run in a worker thread; the Tk main thread only draws the
# latest frame (pulled under a lock) and reads a status/log queue.
# =============================================================================

import os
import sys
import math
import time
import queue
import threading

# Reduce CUDA fragmentation OOM when a heavy voice (Maya1) and the video models
# (LivePortrait + MuseTalk) share the GPU. Must be set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = os.path.join(PROJECT_DIR, "engines")
for p in (ENGINES_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from realtime_avatar import _character_path, _open_webcam, FRAME_SIZE, FPS
from tts_stream_engine import MALE_VOICES

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PREVIEW_SIZE = 512
TARGET_FRAME_TIME = 1.0 / FPS

# Face-loss -> trading-chart scene
NO_FACE_SECONDS = 1.5          # no face for this long -> switch to charts
CHART_FADE_STEP = 0.12         # crossfade speed per frame (~0.5s transition)

MOTION_THRESH = 4.0            # mean 64x64 gray-diff above which LP runs every frame
#                                (lower = turns reliably trigger full-rate LP -> less lag)

# Quality presets -> (lp_interval, enhance level, body motion, restore cadence).
# "Delulu" is the tuned default: restoration every 3rd frame (crisp filmed look)
# + smooth head (motion-adaptive bumps to every-frame on movement) + subtle body.
QUALITY_PRESETS = {
    "Delulu (recommended)":  dict(lp=3, enhance="light", body=True,  restore_every=3),
    "Smooth (max fps)":      dict(lp=3, enhance="light", body=False, restore_every=4),
    "Sharp (max detail)":    dict(lp=1, enhance="full",  body=True,  restore_every=2),
}
QUALITY_LABELS = list(QUALITY_PRESETS.keys())

# Pose presets -> (max turn deg, max tilt deg). Safe never melts (best for
# streaming); Free is the wide-range testing mode.
POSE_PRESETS = {
    "Safe (no melt)":   dict(turn=30, tilt=10),
    "Cinematic":        dict(turn=40, tilt=14),
    "Free (testing)":   dict(turn=62, tilt=22),
}
POSE_LABELS = list(POSE_PRESETS.keys())

QUICK_PHRASES = [
    "Hey everyone, welcome back to the stream.",
    "Gold is pushing into a key resistance level right now.",
    "This is a serious move, watch the volume coming in.",
    "Thank you all for the support, let's get into it.",
]

# Voice modes shown in the dropdown -> TTS backend key. Labels carry the honest
# per-NEW-line latency so the live tradeoff is clear (all CACHE, so a repeated
# line is instant whatever the voice):
#   Kokoro  ~0.1s  smooth live
#   Chatterbox ~3s clones a real human voice — best expressive option for live
#   Maya1  ~10s + 25s load — laughs/emotion tags, but too slow for smooth live;
#          best for pre-rendering / repeated (cached) lines
VOICE_MODES = [
    ("Fast — instant (Kokoro)",            "kokoro"),
    ("Real human voice ~3s (Chatterbox)",  "chatterbox"),
    ("Laughs/emotion — SLOW ~10s (Maya1)", "maya1"),
    ("Cloud (edge)",                       "edge"),
]
VOICE_MODE_LABELS = [m[0] for m in VOICE_MODES]
VOICE_MODE_KEY = dict(VOICE_MODES)

# -----------------------------------------------------------------------------
# DESIGN TOKENS — a futuristic "HUD / neon" palette on a deep-space canvas.
#   BG       near-black navy app canvas
#   SURFACE  panel faces (drawn as glowing rounded rects on a Canvas)
#   SURFACE2 insets / fields / entries
#   CYAN     primary neon  ·  MAG secondary neon  ·  MINT live/go  ·  AMBER warn
#   FG/MUTED/FAINT  cool-white text hierarchy
# Legacy names (ACCENT*/GREEN*/BG2/ENTRY_BG) are aliases so the rest of the
# class — and the runtime engine logic — stay byte-for-byte unchanged.
# -----------------------------------------------------------------------------
BG       = "#05070e"      # deep-space canvas
SURFACE  = "#0a0f1a"      # panel fill
SURFACE2 = "#070b14"      # fields / entries / insets
BORDER   = "#16243a"      # hairline borders
FG       = "#e3ecf7"      # cool-white primary text
MUTED    = "#6f87a0"      # secondary text / control labels
FAINT    = "#3b4f66"      # captions / dim chrome

CYAN     = "#26e8ff"      # primary neon accent
CYAN_HI  = "#7af2ff"
CYAN_INK = "#02181f"
MAG      = "#ff2f9e"      # secondary neon accent
MAG_HI   = "#ff74bf"
MINT     = "#27ffb0"      # live / go
MINT_HI  = "#73ffcb"
MINT_INK = "#02160e"
AMBER    = "#ffb43d"      # warnings / recenter
RED      = "#ff3b5c"      # stopped / error

# Back-compat aliases (referenced elsewhere in this module + engine glue).
ACCENT     = CYAN
ACCENT_HI  = CYAN_HI
ACCENT_INK = CYAN_INK
GREEN      = MINT
GREEN_HI   = MINT_HI
GREEN_INK  = MINT_INK
BG2        = "#0d1626"    # ghost-button / mute base surface
ENTRY_BG   = SURFACE2


class AvatarStudio:
    """Tk window that runs the avatar pipeline and previews the final frame."""

    def __init__(self, root):
        self.root = root
        self.running = False
        self.booting = False
        self.engines = None
        self.swap_engine = None              # lazy inswapper face-swap (real head)
        self.tts = None
        self.brain = None                    # Ollama LLM brain (answers in character)
        self._thinking = False               # True while the brain is generating
        self.cap = None
        self.obs_cam = None
        self.lp_interval = 2
        self._char_path = None               # chosen character image (any face)

        self._latest = None                 # latest final frame (BGR ndarray)
        self._frame_lock = threading.Lock()
        self._log_q = queue.Queue()
        self._fps = 0.0
        self._diag = ""                      # per-stage ms readout
        self._speaking = False
        self._worker = None

        root.title("AVATAR STUDIO ◆ neural pipeline")
        root.configure(bg=BG)
        root.geometry("1240x900")
        root.minsize(1040, 660)

        self._init_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()                     # start the UI refresh loop
        self._animate()                     # start the HUD animation loop

    # -------------------------------------------------------------------------
    # STYLING + SMALL UI BUILDERS
    # -------------------------------------------------------------------------
    def _init_style(self):
        """Theme every ttk widget (combobox/spinbox/scale/scrollbar) to match the
        dark studio palette — the default clam look is too light otherwise."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Studio.TCombobox",
                        fieldbackground=SURFACE2, background=SURFACE2, foreground=FG,
                        arrowcolor=MUTED, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, relief="flat", padding=5)
        style.map("Studio.TCombobox",
                  fieldbackground=[("readonly", SURFACE2)],
                  foreground=[("readonly", FG), ("disabled", FAINT)],
                  selectbackground=[("readonly", SURFACE2)],
                  selectforeground=[("readonly", FG)],
                  bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)], arrowcolor=[("active", ACCENT)])
        # popdown list (the dropdown itself is a classic Tk Listbox)
        self.root.option_add("*TCombobox*Listbox.background", SURFACE2)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_INK)
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)

        style.configure("Studio.TSpinbox",
                        fieldbackground=SURFACE2, background=SURFACE, foreground=FG,
                        arrowcolor=MUTED, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, relief="flat", padding=4)
        style.map("Studio.TSpinbox",
                  bordercolor=[("focus", ACCENT)], arrowcolor=[("active", ACCENT)])

        style.configure("Studio.Horizontal.TScale",
                        background=ACCENT, troughcolor=SURFACE2, bordercolor=SURFACE2,
                        lightcolor=ACCENT, darkcolor=ACCENT)

        style.configure("Studio.Vertical.TScrollbar",
                        background=BG2, troughcolor=BG, bordercolor=BG,
                        arrowcolor=MUTED, relief="flat")
        style.map("Studio.Vertical.TScrollbar", background=[("active", BORDER)])

    # ---- low-level drawing helpers (neon HUD chrome on Canvas) -------------
    @staticmethod
    def _mix(c1, c2, t):
        """Blend two #rrggbb colors (t=0 -> c1, t=1 -> c2)."""
        a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(
            max(0, min(255, int(round(a[k] + (b[k] - a[k]) * t)))) for k in range(3))

    def _round_rect(self, cv, x1, y1, x2, y2, r, **kw):
        pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
               x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return cv.create_polygon(pts, smooth=True, **kw)

    def _glow_text(self, cv, x, y, text, color, font, anchor="w", tags="chrome"):
        """Fake a neon bloom: dim 1px-offset copies beneath a bright top layer."""
        dim = self._mix(BG, color, 0.42)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv.create_text(x+dx, y+dy, text=text, fill=dim, font=font,
                           anchor=anchor, tags=tags)
        return cv.create_text(x, y, text=text, fill=color, font=font,
                              anchor=anchor, tags=tags)

    def _draw_panel(self, cv, x1, y1, x2, y2, accent, title, code):
        """Render one glowing HUD panel: halo + body + corner brackets + header."""
        R = 14
        cv.delete("chrome")
        self._round_rect(cv, x1-1, y1-1, x2+1, y2+1, R+1, fill="",
                         outline=self._mix(BG, accent, 0.20), width=1, tags="chrome")
        self._round_rect(cv, x1, y1, x2, y2, R, fill=SURFACE,
                         outline=self._mix(BG, accent, 0.55), width=1, tags="chrome")
        L, o = 13, 10
        for cx, cy, dx, dy in ((x1+o, y1+o, 1, 1), (x2-o, y1+o, -1, 1),
                               (x1+o, y2-o, 1, -1), (x2-o, y2-o, -1, -1)):
            cv.create_line(cx, cy, cx+dx*L, cy, fill=accent, width=2,
                           capstyle="round", tags="chrome")
            cv.create_line(cx, cy, cx, cy+dy*L, fill=accent, width=2,
                           capstyle="round", tags="chrome")
        if title:
            cv.create_rectangle(x1+18, y1+17, x1+21, y1+30, fill=accent,
                                outline="", tags="chrome")
            self._glow_text(cv, x1+30, y1+24, title, accent, ("Consolas", 10, "bold"))
            if code:
                cv.create_text(x2-18, y1+24, text=code, anchor="e",
                               fill=self._mix(accent, BG, 0.35),
                               font=("Consolas", 9), tags="chrome")
            cv.create_line(x1+18, y1+39, x2-18, y1+39,
                           fill=self._mix(SURFACE, accent, 0.22), width=1, tags="chrome")

    def _panel(self, parent, title=None, accent=None, code=""):
        """A glowing HUD panel drawn on a Canvas; returns the inner content Frame.
        The canvas auto-resizes to the content's requested height and redraws the
        chrome, so panels grow/shrink with their widgets."""
        accent = accent or CYAN
        holder = tk.Frame(parent, bg=BG); holder.pack(fill="x", pady=(0, 14))
        cv = tk.Canvas(holder, bg=BG, highlightthickness=0, bd=0, height=64)
        cv.pack(fill="x")
        content = tk.Frame(cv, bg=SURFACE)
        PAD = 18
        TOP = 48 if title else 18
        win = cv.create_window(PAD, TOP, anchor="nw", window=content)
        st = {"h": 0}

        def redraw(_=None):
            w = cv.winfo_width()
            if w <= 1:
                return
            content.update_idletasks()
            H = content.winfo_reqheight() + TOP + PAD + 8     # extra bottom breathing room
            if abs(H - st["h"]) > 1:
                st["h"] = H
                cv.configure(height=H)
            cv.itemconfigure(win, width=w - 2*PAD)
            self._draw_panel(cv, 3, 3, w - 3, H - 3, accent, title, code)
        content.bind("<Configure>", redraw)
        cv.bind("<Configure>", redraw)
        return content

    def _card(self, parent, title=None):
        """Back-compat shim: the old flat card is now a glowing HUD panel. Each
        section gets its own neon accent + console code so the rail reads like a
        cockpit."""
        accent, code = CYAN, ""
        theme = {
            "SESSION":        (CYAN,  "SYS·00"),
            "PERFORMANCE":    (CYAN,  "PERF·01"),
            "REALISM":        (MAG,   "RND·02"),
            "SCENE & OUTPUT": (MINT,  "OUT·03"),
            "VOICE":          (MAG,   "TTS·04"),
            "SPEAK":          (MINT,  "MSG·05"),
            "ACTIVITY LOG":   (self._mix(CYAN, BG, 0.35), "LOG·06"),
        }.get(title)
        if theme:
            accent, code = theme
        return self._panel(parent, title, accent=accent, code=code)

    # ---- neon controls -----------------------------------------------------
    def _btn(self, parent, text, cmd, *, bg, fg, hover, border=None,
             hover_border=None, font=("Consolas", 11, "bold"), state="normal"):
        """Flat button with a 1px neon outline that lights up on hover."""
        border = border or fg
        hover_border = hover_border or self._mix(border, "#ffffff", 0.3)
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font,
                      relief="flat", bd=0, cursor="hand2", activebackground=hover,
                      activeforeground=fg, state=state, highlightthickness=1,
                      highlightbackground=border, highlightcolor=border,
                      disabledforeground=FAINT)

        def en(_):
            if str(b["state"]) != "disabled":
                b.configure(bg=hover, highlightbackground=hover_border)

        def lv(_):
            if str(b["state"]) != "disabled":
                b.configure(bg=bg, highlightbackground=border)
        b.bind("<Enter>", en); b.bind("<Leave>", lv)
        return b

    def _chip(self, parent, text, cmd, full=False):
        base = SURFACE2
        brd = self._mix(SURFACE2, CYAN, 0.30)
        hov = self._mix(SURFACE2, CYAN, 0.16)
        b = tk.Button(parent, text=text, command=cmd, bg=base, fg=MUTED,
                      font=("Segoe UI", 8) if full else ("Consolas", 8),
                      relief="flat", bd=0, cursor="hand2", padx=8, pady=4,
                      activebackground=hov, activeforeground=CYAN,
                      highlightthickness=1, highlightbackground=brd,
                      highlightcolor=brd, anchor="w" if full else "center")

        def en(_): b.configure(bg=hov, fg=FG, highlightbackground=CYAN)
        def lv(_): b.configure(bg=base, fg=MUTED, highlightbackground=brd)
        b.bind("<Enter>", en); b.bind("<Leave>", lv)
        return b

    def _check(self, parent, text, var, cmd=None):
        return tk.Checkbutton(parent, text=text, variable=var, command=cmd,
                              bg=SURFACE, fg=FG, selectcolor=SURFACE2,
                              activebackground=SURFACE, activeforeground=CYAN,
                              font=("Segoe UI", 9), anchor="w", justify="left",
                              highlightthickness=0, bd=0, padx=0, cursor="hand2")

    def _row(self, parent, label):
        r = tk.Frame(parent, bg=SURFACE); r.pack(fill="x", pady=5)
        tk.Label(r, text=label, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        return r

    def _animate(self):
        """Lightweight ~16fps loop: breathing status ring + header sweep dot."""
        self._anim = getattr(self, "_anim", 0) + 1
        p = abs((self._anim % 24) / 12.0 - 1.0)          # 0..1 triangle wave
        try:
            if getattr(self, "running", False):
                glow = self._mix(SURFACE, MINT, 0.25 + 0.55 * (1 - p))
            else:
                glow = self._mix(SURFACE, RED, 0.12 + 0.28 * (1 - p))
            self.status_canvas.itemconfig(self.status_glow, outline=glow)
        except Exception:
            pass
        try:
            cvt = self._topcv
            w = cvt.winfo_width()
            if w > 160:
                x = 26 + ((self._anim * 7) % (w - 200))
                cvt.coords(self._sweep, x-2, self._sweep_y-2, x+2, self._sweep_y+2)
                cvt.itemconfig(self._sweep, fill=self._mix(CYAN, BG, 0.2 + 0.6*(1-p)))
        except Exception:
            pass
        self.root.after(60, self._animate)

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # ===== TOP APP BAR (full-width Canvas — glowing brand + telemetry) ==
        topcv = tk.Canvas(self.root, bg=BG, height=70, highlightthickness=0, bd=0)
        topcv.pack(side="top", fill="x")
        self._topcv = topcv
        self._sweep_y = 56
        self._sweep = topcv.create_oval(0, 0, 0, 0, fill=CYAN, outline="")

        def _topdraw(_=None):
            w = topcv.winfo_width()
            if w <= 1:
                return
            topcv.delete("tb")
            # brand hexagon
            hx, hy, r = 30, 32, 12
            pts = []
            for k in range(6):
                ang = math.pi / 3 * k - math.pi / 6
                pts += [hx + r * math.cos(ang), hy + r * math.sin(ang)]
            topcv.create_polygon(pts, outline=CYAN, fill=self._mix(BG, CYAN, 0.12),
                                 width=2, tags="tb")
            topcv.create_oval(hx-3, hy-3, hx+3, hy+3, fill=CYAN, outline="", tags="tb")
            # wordmark + neon slash
            self._glow_text(topcv, 56, 32, "AVATAR", FG, ("Consolas", 18, "bold"),
                            tags="tb")
            self._glow_text(topcv, 56 + 112, 32, "// STUDIO", CYAN,
                            ("Consolas", 18, "bold"), tags="tb")
            # right-side telemetry
            topcv.create_text(w-26, 24, text="SYS > ONLINE    v2.0", anchor="e",
                              fill=self._mix(FG, BG, 0.3), font=("Consolas", 9), tags="tb")
            topcv.create_text(w-26, 42, text="NEURAL AVATAR PIPELINE", anchor="e",
                              fill=self._mix(CYAN, BG, 0.35), font=("Consolas", 8), tags="tb")
            # underline rail (fading neon ticks)
            y = self._sweep_y
            span = w - 52
            for i in range(0, span, 7):
                topcv.create_line(26+i, y, 26+i+4, y,
                                  fill=self._mix(CYAN, BG, 0.55 + 0.4*(i/float(span))),
                                  width=1, tags="tb")
            topcv.tag_raise(self._sweep)
        topcv.bind("<Configure>", _topdraw)
        tk.Frame(self.root, bg=self._mix(BG, CYAN, 0.18), height=1).pack(
            side="top", fill="x")

        # ===== BODY: preview (left) + control rail (right) ==================
        bodyf = tk.Frame(self.root, bg=BG)
        bodyf.pack(side="top", fill="both", expand=True)

        # ---- LEFT: glowing HUD feed panel (status ring + fps + diagnostics)
        left = tk.Frame(bodyf, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(18, 9), pady=14)
        pv = self._panel(left, "LIVE FEED", accent=CYAN, code="CAM·00")

        # header: breathing status ring + state label (left), fps (right)
        ph = tk.Frame(pv, bg=SURFACE); ph.pack(fill="x", pady=(0, 8))
        self.status_canvas = tk.Canvas(ph, width=22, height=22, bg=SURFACE,
                                       highlightthickness=0)
        self.status_canvas.pack(side="left")
        self.status_glow = self.status_canvas.create_oval(3, 3, 19, 19,
                                                          outline=SURFACE, width=2)
        self.status_dot = self.status_canvas.create_oval(7, 7, 15, 15,
                                                        fill=RED, outline="")
        self.status_lbl = tk.Label(ph, text="OFFLINE", bg=SURFACE, fg=FG,
                                   font=("Consolas", 10, "bold"))
        self.status_lbl.pack(side="left", padx=8)
        self.fps_lbl = tk.Label(ph, text="", bg=SURFACE, fg=CYAN,
                                font=("Consolas", 10))
        self.fps_lbl.pack(side="right")

        # the composited frame (black stage with a neon hairline frame)
        stageb = tk.Frame(pv, bg=self._mix(BG, CYAN, 0.22))
        stageb.pack(fill="both", expand=True)
        stage = tk.Frame(stageb, bg="#000000")
        stage.pack(fill="both", expand=True, padx=1, pady=1)
        self.preview = tk.Label(stage, bg="#000000", bd=0)
        self.preview.pack(fill="both", expand=True)

        # footer: per-stage timing readout
        pf = tk.Frame(pv, bg=SURFACE); pf.pack(fill="x", pady=(8, 0))
        self.diag_lbl = tk.Label(pf, text="// ready", bg=SURFACE,
                                 fg=self._mix(CYAN, BG, 0.3), font=("Consolas", 9))
        self.diag_lbl.pack(side="left")

        self._show_placeholder()

        # ---- RIGHT: scrollable control rail --------------------------------
        right_outer = tk.Frame(bodyf, bg=BG, width=412)
        right_outer.pack(side="right", fill="y", padx=(9, 12), pady=16)
        right_outer.pack_propagate(False)
        _canvas = tk.Canvas(right_outer, bg=BG, highlightthickness=0)
        _vsb = ttk.Scrollbar(right_outer, orient="vertical", command=_canvas.yview,
                             style="Studio.Vertical.TScrollbar")
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side="right", fill="y")
        _canvas.pack(side="left", fill="both", expand=True)
        right = tk.Frame(_canvas, bg=BG)          # inner frame holds ALL controls
        _win = _canvas.create_window((0, 0), window=right, anchor="nw")

        def _sync_scroll(_=None):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
            _canvas.itemconfigure(_win, width=_canvas.winfo_width())
        right.bind("<Configure>", _sync_scroll)
        _canvas.bind("<Configure>", _sync_scroll)

        def _wheel(e):                            # mouse-wheel scrolls the panel
            _canvas.yview_scroll(int(-(e.delta or 0) / 120), "units")
        _canvas.bind_all("<MouseWheel>", _wheel)
        self._ctrl_canvas = _canvas

        # ---- SESSION -------------------------------------------------------
        c = self._card(right, "SESSION")
        self.start_btn = self._btn(
            c, "START", self.start, bg=MINT, fg=MINT_INK,
            hover=self._mix(MINT, "#ffffff", 0.18), border=MINT, hover_border="#ffffff",
            font=("Consolas", 12, "bold"))
        self.start_btn.pack(fill="x", ipady=9, pady=(0, 7))
        self.stop_btn = self._btn(
            c, "STOP", self.stop, bg=self._mix(SURFACE2, RED, 0.06), fg=RED,
            hover=self._mix(SURFACE2, RED, 0.16), border=self._mix(RED, BG, 0.35),
            hover_border=RED, font=("Consolas", 12, "bold"), state="disabled")
        self.stop_btn.pack(fill="x", ipady=9, pady=(0, 10))
        self.char_btn = self._btn(
            c, "LOAD CHARACTER", self._load_character, bg=SURFACE2, fg=CYAN,
            hover=self._mix(SURFACE2, CYAN, 0.14), border=self._mix(CYAN, BG, 0.35),
            hover_border=CYAN, font=("Consolas", 9, "bold"))
        self.char_btn.pack(fill="x", ipady=6, pady=(0, 4))
        tk.Label(c, text="any face image — celebrity, AI render, cartoon",
                 bg=SURFACE, fg=FAINT, font=("Consolas", 8)).pack(anchor="w", pady=(0, 9))
        self.recenter_btn = self._btn(
            c, "RECENTER POSE", self.recenter, bg=SURFACE2, fg=AMBER,
            hover=self._mix(SURFACE2, AMBER, 0.14), border=self._mix(AMBER, BG, 0.35),
            hover_border=AMBER, font=("Consolas", 9, "bold"), state="disabled")
        self.recenter_btn.pack(fill="x", ipady=6)
        tk.Label(c, text="sit upright facing the camera, then click",
                 bg=SURFACE, fg=FAINT, font=("Consolas", 8)).pack(anchor="w", pady=(4, 0))

        # ---- PERFORMANCE ---------------------------------------------------
        c = self._card(right, "PERFORMANCE")
        r = self._row(c, "Quality preset")
        self.quality_var = tk.StringVar(value="Delulu (recommended)")
        ttk.Combobox(r, textvariable=self.quality_var, values=QUALITY_LABELS,
                     state="readonly", width=18,
                     style="Studio.TCombobox").pack(side="right")
        self.quality_var.trace_add("write", self._on_quality)

        r = self._row(c, "Head update · LP every N")
        self.interval_var = tk.IntVar(value=2)
        ttk.Spinbox(r, from_=1, to=4, width=5, textvariable=self.interval_var,
                    command=self._on_interval, style="Studio.TSpinbox").pack(side="right")

        r = self._row(c, "Stabilization")
        self.stab_var = tk.IntVar(value=40)
        ttk.Scale(r, from_=0, to=100, variable=self.stab_var, length=150,
                  style="Studio.Horizontal.TScale",
                  command=lambda e: self._on_stab()).pack(side="right")

        r = self._row(c, "Min face size %")
        self.minface_var = tk.IntVar(value=9)
        ttk.Spinbox(r, from_=6, to=40, increment=2, width=5,
                    textvariable=self.minface_var, command=self._on_minface,
                    style="Studio.TSpinbox").pack(side="right")

        r = self._row(c, "Pose preset")
        self.pose_var = tk.StringVar(value="Safe (no melt)")
        ttk.Combobox(r, textvariable=self.pose_var, values=POSE_LABELS,
                     state="readonly", width=16,
                     style="Studio.TCombobox").pack(side="right")
        self.pose_var.trace_add("write", self._on_pose)

        r = self._row(c, "Max turn °  (Safe 30 = no melt)")
        self.turncap_var = tk.IntVar(value=30)
        ttk.Spinbox(r, from_=20, to=90, increment=5, width=5,
                    textvariable=self.turncap_var, command=self._on_turncap,
                    style="Studio.TSpinbox").pack(side="right")

        r = self._row(c, "Max tilt °  (Safe 10 = no melt)")
        self.tilt_var = tk.IntVar(value=10)
        ttk.Spinbox(r, from_=8, to=30, increment=1, width=5,
                    textvariable=self.tilt_var, command=self._on_tilt,
                    style="Studio.TSpinbox").pack(side="right")

        # ---- REALISM -------------------------------------------------------
        c = self._card(right, "REALISM")
        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", pady=3)
        self.gaze_var = tk.BooleanVar(value=True)
        self._check(r, "Lock gaze", self.gaze_var, self._on_gaze).pack(side="left")
        self.gaze_var2 = tk.IntVar(value=55)   # gentler default — keeps iris life
        ttk.Scale(r, from_=0, to=100, variable=self.gaze_var2, length=120,
                  style="Studio.Horizontal.TScale",
                  command=lambda e: self._on_gaze()).pack(side="right")

        self.restore_var = tk.BooleanVar(value=True)
        self._check(c, "Face restoration  ·  GFPGAN (fixes plastic look)",
                    self.restore_var).pack(fill="x", pady=3)
        r = self._row(c, "Skin detail")
        self.skin_var = tk.IntVar(value=70)
        ttk.Scale(r, from_=0, to=100, variable=self.skin_var, length=150,
                  style="Studio.Horizontal.TScale",
                  command=lambda e: self._on_skin()).pack(side="right")

        self.body_var = tk.BooleanVar(value=True)
        self._check(c, "Live body motion  ·  torso follows you",
                    self.body_var).pack(fill="x", pady=3)
        self.multiref_var = tk.BooleanVar(value=False)
        self._check(c, "Extended turning  ·  multi-view (wider, less stable)",
                    self.multiref_var, self._on_multiref).pack(fill="x", pady=3)
        self.swap_var = tk.BooleanVar(value=False)
        self._check(c, "FACE-SWAP mode  ·  real head, perfect 90° turns (GPU)",
                    self.swap_var).pack(fill="x", pady=3)

        # ---- SCENE & OUTPUT ------------------------------------------------
        c = self._card(right, "SCENE & OUTPUT")
        self.chart_var = tk.BooleanVar(value=True)
        self._check(c, "Show live charts when face is lost",
                    self.chart_var).pack(fill="x", pady=3)
        self.obs_var = tk.BooleanVar(value=False)
        self._check(c, "Also send to OBS virtual camera",
                    self.obs_var).pack(fill="x", pady=3)

        # ---- VOICE ---------------------------------------------------------
        c = self._card(right, "VOICE")
        tk.Label(c, text="Mode", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.voicemode_var = tk.StringVar(value=VOICE_MODE_LABELS[0])
        ttk.Combobox(c, textvariable=self.voicemode_var, values=VOICE_MODE_LABELS,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(3, 9))
        self.voicemode_var.trace_add("write", self._on_voice_mode)

        tk.Label(c, text="Speaker", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.voice_var = tk.StringVar(value=MALE_VOICES[0])
        ttk.Combobox(c, textvariable=self.voice_var, values=MALE_VOICES,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(3, 9))
        self.voice_var.trace_add("write", self._on_voice)

        tk.Label(c, text="Insert emotion tag  (Maya1 performs these)", bg=SURFACE,
                 fg=FAINT, font=("Segoe UI", 8)).pack(anchor="w")
        erow = tk.Frame(c, bg=SURFACE); erow.pack(fill="x", pady=(4, 0))
        for tag in ("<laugh>", "<sigh>", "<chuckle>", "<gasp>", "<whisper>"):
            self._chip(erow, tag, lambda t=tag: self._insert_tag(t)).pack(
                side="left", padx=(0, 4))

        # ---- ASK THE AVATAR (Ollama answers, avatar speaks it) -------------
        c = self._card(right, "ASK THE AVATAR")
        tk.Label(c, text="Type a question — the AI answers in character and speaks it",
                 bg=SURFACE, fg=FAINT, font=("Segoe UI", 8),
                 wraplength=300, justify="left").pack(anchor="w", pady=(0, 4))
        self.ask_entry = tk.Text(c, height=2, bg=SURFACE2, fg=FG, insertbackground=MAG,
                                 font=("Segoe UI", 11), relief="flat", wrap="word",
                                 padx=9, pady=7, highlightthickness=1,
                                 highlightbackground=BORDER, highlightcolor=MAG)
        self.ask_entry.pack(fill="x", pady=(0, 7))
        self.ask_entry.bind("<Return>", self._on_ask_enter)
        self.ask_btn = self._btn(
            c, "ASK  ▸  the avatar answers", self.ask, bg=MAG, fg=CYAN_INK,
            hover=self._mix(MAG, "#ffffff", 0.18), border=MAG, hover_border="#ffffff",
            font=("Consolas", 11, "bold"), state="disabled")
        self.ask_btn.pack(fill="x", ipady=6)

        # ---- SPEAK (verbatim — the avatar says exactly this) ---------------
        c = self._card(right, "SPEAK")
        tk.Label(c, text="Make the avatar say this text exactly",
                 bg=SURFACE, fg=FAINT, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))
        self.entry = tk.Text(c, height=3, bg=SURFACE2, fg=FG, insertbackground=ACCENT,
                             font=("Segoe UI", 11), relief="flat", wrap="word",
                             padx=9, pady=7, highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=ACCENT)
        self.entry.pack(fill="x", pady=(0, 7))
        self.entry.bind("<Return>", self._on_enter)

        brow = tk.Frame(c, bg=SURFACE); brow.pack(fill="x")
        self.speak_btn = self._btn(
            brow, "SPEAK", self.speak, bg=CYAN, fg=CYAN_INK,
            hover=self._mix(CYAN, "#ffffff", 0.18), border=CYAN, hover_border="#ffffff",
            font=("Consolas", 11, "bold"), state="disabled")
        self.speak_btn.pack(side="left", fill="x", expand=True, ipady=6)
        # plain Button (no hover binding) so toggle_mute's color change persists
        self.mute_btn = tk.Button(brow, text="MUTE", command=self.toggle_mute,
                                  bg=SURFACE2, fg=MUTED, font=("Consolas", 11, "bold"),
                                  relief="flat", bd=0, width=8, cursor="hand2",
                                  activebackground=self._mix(SURFACE2, RED, 0.16),
                                  state="disabled", highlightthickness=1,
                                  highlightbackground=self._mix(MUTED, BG, 0.45),
                                  highlightcolor=RED)
        self.mute_btn.pack(side="left", padx=(7, 0), ipady=6)

        tk.Label(c, text="Quick phrases", bg=SURFACE, fg=FAINT,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(10, 3))
        for t in QUICK_PHRASES:
            self._chip(c, t[:34] + ("…" if len(t) > 34 else ""),
                       lambda x=t: self._speak_text(x), full=True).pack(fill="x", pady=2)

        # ---- ACTIVITY LOG --------------------------------------------------
        c = self._card(right, "ACTIVITY LOG")
        self.log = tk.Text(c, height=8, bg=SURFACE2, fg=MUTED, relief="flat",
                           font=("Consolas", 8), wrap="word", state="disabled",
                           padx=8, pady=6, highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=BORDER)
        self.log.pack(fill="both", expand=True)

    # -------------------------------------------------------------------------
    # PREVIEW / UI REFRESH (Tk main thread only)
    # -------------------------------------------------------------------------
    def _show_placeholder(self):
        # A sci-fi HUD "standby" feed: faint grid, corner brackets, a targeting
        # reticle and telemetry — so the idle stage reads like a cockpit display.
        S = PREVIEW_SIZE
        img = np.full((S, S, 3), 8, np.uint8)                 # near-black
        grid = (30, 24, 14)
        for x in range(0, S, 32):
            cv2.line(img, (x, 0), (x, S), grid, 1, cv2.LINE_AA)
        for y in range(0, S, 32):
            cv2.line(img, (0, y), (S, y), grid, 1, cv2.LINE_AA)
        cyan = (255, 232, 38)            # BGR of #26e8ff
        dim = (110, 86, 28)
        m, L = 18, 42                    # corner brackets
        for px, py, dx, dy in ((m, m, 1, 1), (S-m, m, -1, 1),
                               (m, S-m, 1, -1), (S-m, S-m, -1, -1)):
            cv2.line(img, (px, py), (px+dx*L, py), cyan, 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py+dy*L), cyan, 2, cv2.LINE_AA)
        c = S // 2                       # targeting reticle
        cv2.circle(img, (c, c-12), 46, dim, 1, cv2.LINE_AA)
        cv2.circle(img, (c, c-12), 62, (60, 48, 18), 1, cv2.LINE_AA)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(img, (c+dx*58, c-12+dy*58), (c+dx*78, c-12+dy*78),
                     cyan, 1, cv2.LINE_AA)
        cv2.circle(img, (c, c-12), 3, cyan, -1, cv2.LINE_AA)
        cv2.putText(img, "FEED // STANDBY", (24, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, cyan, 1, cv2.LINE_AA)
        cv2.putText(img, "00:00:00", (S-120, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, dim, 1, cv2.LINE_AA)
        for txt, y, sc, col, th in (("AWAITING SIGNAL", c+104, 0.72, cyan, 2),
                                    ("press START to initialise avatar", c+134, 0.45,
                                     (150, 140, 120), 1)):
            (w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
            cv2.putText(img, txt, (c - w // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                        sc, col, th, cv2.LINE_AA)
        self._draw(img)

    def _draw(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        if im.size != (PREVIEW_SIZE, PREVIEW_SIZE):
            im = im.resize((PREVIEW_SIZE, PREVIEW_SIZE))
        tkimg = ImageTk.PhotoImage(im)
        self.preview.configure(image=tkimg)
        self.preview.image = tkimg          # keep a reference

    def _poll_ui(self):
        # drain log queue
        try:
            while True:
                msg = self._log_q.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        # draw latest frame
        if self.running:
            with self._frame_lock:
                frame = None if self._latest is None else self._latest.copy()
            if frame is not None:
                self._draw(frame)
            self.fps_lbl.configure(text=f"{self._fps:4.1f} fps")
            if self._diag:
                self.diag_lbl.configure(text=self._diag)
            # While a heavy voice generates a NEW line the GPU is busy and the
            # preview briefly stalls — show why so it doesn't look frozen.
            if self._thinking:
                self._set_status("thinking...", "#cc9933")
            elif self.tts is not None and getattr(self.tts, "synthesizing", False):
                self._set_status("generating voice...", "#cc9933")
            elif getattr(self, "_speaking", False):
                self._set_status("speaking", GREEN)
            elif self.status_lbl.cget("text") in ("thinking...", "generating voice...", "speaking"):
                self._set_status("LIVE", GREEN)
        self.root.after(33, self._poll_ui)   # ~30 Hz UI refresh

    def _append_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_msg(self, msg):
        self._log_q.put(msg)

    def _set_status(self, text, color):
        # safe: called from Tk thread via _poll? we call from worker -> use after
        def _apply():
            self.status_lbl.configure(text=text)
            self.status_canvas.itemconfig(self.status_dot, fill=color)
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # START / STOP
    # -------------------------------------------------------------------------
    def start(self):
        if self.running or self.booting:
            return
        self.booting = True
        self.start_btn.configure(state="disabled", text="STARTING...")
        self.lp_interval = max(1, int(self.interval_var.get()))
        self._set_status("starting...", "#cc9933")
        self._log_msg("[studio] building engines (LivePortrait + Wav2Lip warmup ~60-90s)...")
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        try:
            from liveportrait_engine import LivePortraitEngine
            from musetalk_engine import MuseTalkEngine
            from compositor import Compositor
            from tts_stream_engine import TTSStreamEngine
            from trading_view import TradingView
            import enhance_engine

            char = self._char_path or _character_path()
            self._log_msg(f"[studio] character: {os.path.basename(char)}")
            self._log_msg("[studio] LivePortrait...")
            lp = LivePortraitEngine(char)
            try:
                lp.min_good_face = max(0.04, self.minface_var.get() / 100.0)
                lp._multi = bool(self.multiref_var.get()) and len(getattr(lp, "_refs", [])) > 1
                lp.set_stabilization(self.stab_var.get() / 100.0)
                lp.set_gaze(self.gaze_var.get(), self.gaze_var2.get() / 100.0)
            except Exception:
                pass
            self._log_msg("   -> " + lp.startup_check()[1])
            self._log_msg("[studio] MuseTalk / mouth sync...")
            mt = MuseTalkEngine(char)
            self._log_msg("   -> " + mt.startup_check()[1])
            comp = Compositor()
            self._log_msg("[studio] TTS (loading voice model)...")
            tts = TTSStreamEngine(mt)
            tts.set_voice(self.voice_var.get())
            # Honor the voice-mode dropdown's current selection (default Kokoro).
            tts.set_backend(VOICE_MODE_KEY.get(self.voicemode_var.get(), "kokoro"))
            # Pre-load/warm the selected backend NOW. Without this it would load
            # lazily on the first SPEAK press — freezing the live loop and making
            # SPEAK feel broken.
            self._log_msg("   -> " + tts.startup_check()[1])
            # AI brain (Ollama) — optional; the avatar answers in character.
            try:
                from llm_brain import LLMBrain
                self.brain = LLMBrain()
                self._log_msg("   -> brain: " + self.brain.startup_check()[1])
                # Pre-load the model into VRAM in the background so the first
                # question isn't a ~45s cold-load. Keeps it resident after.
                if self.brain.ok:
                    def _warm_brain():
                        if self.brain.warmup():
                            self._log_msg("[studio] AI brain warmed (resident, fast now).")
                    threading.Thread(target=_warm_brain, daemon=True).start()
            except Exception as exc:
                self.brain = None
                self._log_msg(f"[studio] brain unavailable ({exc}).")
            self._log_msg("[studio] webcam...")
            cap = _open_webcam()
            if cap is None:
                self._log_msg("[studio] NO WEBCAM — driving with a static frame.")
            obs = None
            if self.obs_var.get():
                try:
                    import pyvirtualcam
                    obs = pyvirtualcam.Camera(width=FRAME_SIZE, height=FRAME_SIZE,
                                              fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR)
                    self._log_msg(f"[studio] OBS cam: {obs.device}")
                except Exception as exc:
                    self._log_msg(f"[studio] OBS cam unavailable ({exc}) — preview only.")
                    obs = None

            from body_motion import BodyMotionEngine
            from restore_engine import RestoreEngine
            self._log_msg("[studio] GFPGAN restoration...")
            restore = RestoreEngine()
            self._log_msg("   -> " + restore.startup_check()[1])
            restore.skin_detail = self.skin_var.get() / 100.0
            self.engines = {"lp": lp, "mt": mt, "comp": comp, "enh": enhance_engine,
                            "chart": TradingView("XAUUSD"), "body": BodyMotionEngine(),
                            "restore": restore}
            self._on_quality()        # apply the selected quality preset at boot
            self._on_tilt()           # apply max-tilt (pitch) cap at boot
            self._on_turncap()        # apply max-turn (yaw) cap at boot
            self.tts = tts
            self.cap = cap
            self.obs_cam = obs

            self.running = True
            self.booting = False
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()

            def _enable():
                self.start_btn.configure(text="START")
                self.stop_btn.configure(state="normal")
                for b in (self.speak_btn, self.ask_btn, self.mute_btn, self.recenter_btn):
                    b.configure(state="normal")
            self.root.after(0, _enable)
            self._set_status("LIVE", GREEN)
            self._log_msg("[studio] LIVE — auto-centering pose in ~2s; sit upright "
                          "facing the camera. Use RECENTER anytime it looks tilted.")
        except Exception as exc:
            self.booting = False
            self._log_msg(f"[studio] startup FAILED: {exc}")
            self._set_status("error", RED)
            self.root.after(0, lambda: self.start_btn.configure(
                state="normal", text="START"))

    def _loop(self):
        lp = self.engines["lp"]; mt = self.engines["mt"]
        comp = self.engines["comp"]; enh = self.engines["enh"]
        chart = self.engines["chart"]
        blank = np.full((FRAME_SIZE, FRAME_SIZE, 3), 60, np.uint8)
        last_frame = blank.copy()
        last_final = blank.copy()         # last fully-composed output frame
        cached_face = None; cached_bbox = None
        frame_count = 0
        errs = 0
        recentered = False
        noface = 0                       # consecutive frames with no face
        chart_fade = 0.0                 # 0 = avatar, 1 = trading chart
        in_chart = False                 # for edge-triggered logging
        fps_t = time.perf_counter()
        next_tick = time.monotonic()
        # per-stage timing accumulators (for the [DIAG] readout)
        t_read = t_lp = t_body = t_enh = t_gfp = 0.0
        prev_small = None                # for motion-adaptive LP scheduling

        while self.running:
            # No real camera? Show a clear message instead of running the
            # pipeline on a blank frame (which would just sit on charts).
            if self.cap is None:
                msg = np.full((FRAME_SIZE, FRAME_SIZE, 3), 24, np.uint8)
                cv2.putText(msg, "NO WEBCAM", (120, 230), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (60, 60, 230), 2, cv2.LINE_AA)
                cv2.putText(msg, "Your camera is busy in another app.", (70, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(msg, "Close the browser tab / video call, then STOP+START.",
                            (40, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1,
                            cv2.LINE_AA)
                with self._frame_lock:
                    self._latest = msg
                time.sleep(0.1)
                continue

            _t = time.perf_counter()
            driving = last_frame
            if self.cap is not None:
                ok, fr = self.cap.read()
                if ok and fr is not None:
                    driving = cv2.resize(fr, (FRAME_SIZE, FRAME_SIZE))
                    last_frame = driving
            t_read += time.perf_counter() - _t

            # motion-adaptive LP: measure how much the frame changed (cheap 64x64
            # gray diff). Big movement -> run LP THIS frame (no smear); still ->
            # let the interval amortize it.
            small = cv2.cvtColor(cv2.resize(driving, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            motion = float(np.mean(np.abs(small - prev_small))) if prev_small is not None else 99.0
            prev_small = small

            # While a HEAVY expressive voice (Maya1/Chatterbox) is GENERATING a
            # new line, give it the whole GPU: skip LivePortrait + MuseTalk this
            # frame. Running them concurrently with the 3B TTS thrashes the GPU
            # and can OOM (which poisons the CUDA context -> everything freezes).
            # Hold the last frame with a clear overlay so it reads as "working".
            _busy = None
            if self._thinking:
                _busy = "thinking..."
            elif self.tts is not None and getattr(self.tts, "synthesizing", False):
                _busy = "generating voice..."
            if _busy is not None:
                hold = last_final.copy()
                ov = hold.copy()
                cv2.rectangle(ov, (0, FRAME_SIZE // 2 - 26),
                              (FRAME_SIZE, FRAME_SIZE // 2 + 26), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.55, hold, 0.45, 0, hold)
                cv2.putText(hold, _busy,
                            (FRAME_SIZE // 2 - 150, FRAME_SIZE // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                with self._frame_lock:
                    self._latest = hold
                time.sleep(0.05)
                continue

            # one automatic recenter ~2s in, once the operator has settled, so
            # the neutral baseline isn't the (often mid-motion) very first frame.
            if not recentered and frame_count == int(FPS * 2):
                try:
                    lp.recenter()
                    self.engines["body"].recenter()
                    self._log_msg("[studio] neutral pose set (auto). "
                                  "Press RECENTER if it still looks tilted.")
                except Exception:
                    pass
                recentered = True
                cached_face = None

            lp_fresh = False
            did_swap = False
            # --- FACE-SWAP MODE (inswapper): swap the character's face onto your
            # REAL webcam head. Perfect profiles/turns because the head is real.
            if self.swap_var.get():
                if self.swap_engine is None:
                    try:
                        from faceswap_engine import FaceSwapEngine
                        self._log_msg("[studio] loading face-swap (insightface + inswapper)...")
                        self.swap_engine = FaceSwapEngine(self._char_path or _character_path())
                    except Exception as exc:
                        self._log_msg(f"[studio] face-swap load failed: {exc}")
                        self.swap_var.set(False)
                if self.swap_engine is not None and self.swap_engine.ready:
                    _t = time.perf_counter()
                    ai = self.swap_engine.swap(driving)
                    cached_face = ai; lp_fresh = True; did_swap = True
                    lp._face_found = self.swap_engine.last_found   # chart/loss logic
                    t_lp += time.perf_counter() - _t

            lp_due = (cached_face is None or (frame_count % self.lp_interval) == 0
                      or motion > MOTION_THRESH)        # run every frame on big motion
            if not did_swap:
                _t = time.perf_counter()
                try:
                    if getattr(lp, "fallback_mode", False):
                        ai = lp.process_frame(driving); lp_fresh = True
                    elif lp_due:
                        ai = lp.process_frame(driving); cached_face = ai; lp_fresh = True
                    else:
                        ai = cached_face
                except Exception as exc:
                    ai = driving
                    errs += 1
                    if errs <= 3:
                        self._log_msg(f"[studio] LP frame error: {exc}")
                t_lp += time.perf_counter() - _t

            # --- upper-body motion: warp the torso to follow YOUR shoulders ----
            # Runs every frame (full webcam rate) so the body stays alive even on
            # cached-LP frames. Only when a face is present (else chart/hold).
            _t = time.perf_counter()
            if self.body_var.get() and not did_swap and getattr(lp, "_face_found", False):
                try:
                    ai = self.engines["body"].process(driving, ai)
                except Exception:
                    pass
            t_body += time.perf_counter() - _t

            # --- GFPGAN restoration: fix the plastic look on the FACE crop ----
            _t = time.perf_counter()
            if self.restore_var.get() and getattr(lp, "_face_found", False):
                try:
                    ai = self.engines["restore"].restore(ai)
                except Exception:
                    pass
            t_gfp += time.perf_counter() - _t

            # --- face-loss -> trading chart scene -----------------------------
            # When the webcam can't see the face (operator looks away/down) the
            # output crossfades to a live-moving trading chart, then back when the
            # face returns. Disabled in LP fallback (no real face tracking).
            # Charts only when there is NO face at all (you left / looked away).
            # A small/far face still shows the avatar (held) — not charts.
            face_ok = (not getattr(lp, "fallback_mode", False)) \
                and getattr(lp, "_face_size", 0.0) > 0.0
            noface = 0 if face_ok else noface + 1
            want_chart = (self.chart_var.get()
                          and not getattr(lp, "fallback_mode", False)
                          and noface >= int(FPS * NO_FACE_SECONDS))
            target = 1.0 if want_chart else 0.0
            if target > chart_fade:
                chart_fade = min(1.0, chart_fade + CHART_FADE_STEP)
            elif target < chart_fade:
                chart_fade = max(0.0, chart_fade - CHART_FADE_STEP)
            if want_chart and not in_chart:
                in_chart = True
                chart.reset_price_drift()
                self._log_msg("[studio] no face — switching to live charts.")
            elif not want_chart and in_chart and chart_fade <= 0.0:
                in_chart = False
                self._log_msg("[studio] face back — avatar resumed.")

            self._speaking = bool(getattr(mt, "is_speaking", False))

            if chart_fade >= 1.0:
                # fully on charts — skip the (now hidden) avatar mouth/enhance work
                final = chart.render(speaking=self._speaking)
            else:
                if self._speaking:
                    try:
                        if lp_fresh or cached_bbox is None:
                            cached_bbox = comp.detect_mouth_bbox(ai)
                        mouth = mt.process_mouth(ai, cached_bbox)
                        ai = comp.blend_mouth(ai, mouth, cached_bbox)
                    except Exception as exc:
                        errs += 1
                        if errs <= 3:
                            self._log_msg(f"[studio] mouth error: {exc}")
                _t = time.perf_counter()
                try:
                    # FACE-SWAP streamer look: force FULL enhance so the person is
                    # cut from their room and composited onto the trading studio
                    # background, with the lighting grade + ticker + LIVE badge.
                    if did_swap:
                        enh.set_level("full")
                    final = enh.enhance_frame(ai, is_speaking=self._speaking)
                except Exception:
                    final = ai
                t_enh += time.perf_counter() - _t
                if chart_fade > 0.0:      # crossfade avatar <-> chart
                    cf = chart.render(speaking=self._speaking)
                    final = cv2.addWeighted(final, 1.0 - chart_fade, cf, chart_fade, 0)

            last_final = final            # remember for the "generating" hold
            with self._frame_lock:
                self._latest = final
            if self.obs_cam is not None:
                try:
                    self.obs_cam.send(np.ascontiguousarray(
                        cv2.resize(final, (FRAME_SIZE, FRAME_SIZE))))
                except Exception:
                    pass

            frame_count += 1
            if frame_count % 15 == 0:
                now = time.perf_counter()
                self._fps = 15.0 / (now - fps_t)
                fps_t = now
                rd, lpm, gf, bd, en = (x / 15 * 1000 for x in
                                       (t_read, t_lp, t_gfp, t_body, t_enh))
                self._diag = (f"{self._fps:.1f}fps | read {rd:.0f} | LP {lpm:.0f} | "
                              f"gfpgan {gf:.0f} | body {bd:.0f} | enh {en:.0f} ms")
                print("[DIAG] " + self._diag)
                t_read = t_lp = t_body = t_enh = t_gfp = 0.0

            next_tick += TARGET_FRAME_TIME
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def stop(self):
        if not self.running:
            return
        self._log_msg("[studio] stopping...")
        self.running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        for fn in (lambda: self.cap.release() if self.cap else None,
                   lambda: self.obs_cam.close() if self.obs_cam else None):
            try:
                fn()
            except Exception:
                pass
        self.cap = None; self.obs_cam = None
        self._latest = None
        self.stop_btn.configure(state="disabled")
        for b in (self.speak_btn, self.ask_btn, self.mute_btn, self.recenter_btn):
            b.configure(state="disabled")
        self.start_btn.configure(state="normal", text="START")
        self._set_status("stopped", RED)
        self.fps_lbl.configure(text="")
        self._show_placeholder()
        self._log_msg("[studio] stopped (engines kept warm; START to resume).")

    # -------------------------------------------------------------------------
    # SPEAK / CONTROLS
    # -------------------------------------------------------------------------
    def _on_enter(self, event):
        self.speak()
        return "break"

    def speak(self):
        txt = self.entry.get("1.0", "end").strip()
        if not txt:
            return
        self.entry.delete("1.0", "end")
        self._speak_text(txt)

    def _speak_text(self, txt):
        """SPEAK box / quick phrases: the avatar says EXACTLY this text."""
        if self.tts is None or not self.running:
            self._log_msg("[studio] press START first.")
            return
        self.tts.speak(txt)
        self._log_msg("> " + txt)

    # ---- ASK (Ollama answers, avatar speaks the reply) ----------------------
    def _on_ask_enter(self, event):
        self.ask()
        return "break"

    def ask(self):
        txt = self.ask_entry.get("1.0", "end").strip()
        if not txt:
            return
        self.ask_entry.delete("1.0", "end")
        self._ask_text(txt)

    def _ask_text(self, txt):
        """Send a question to the Ollama brain; the avatar speaks the answer."""
        if self.tts is None or not self.running:
            self._log_msg("[studio] press START first.")
            return
        if self.brain is None or not self.brain.ok:
            why = self.brain.startup_check()[1] if self.brain else "brain not started"
            self._log_msg(f"[studio] AI brain unavailable ({why}) — speaking as-is.")
            self.tts.speak(txt)
            self._log_msg("> " + txt)
            return
        self._brain_answer(txt)

    def _brain_answer(self, txt):
        """Generate the in-character answer on the GPU (loop pauses via
        self._thinking, 'thinking...' overlay), then speak it. Runs in a thread
        so the UI stays responsive."""
        self._log_msg("you> " + txt)

        def _think():
            self._thinking = True
            reply = None
            try:
                reply = self.brain.respond(txt)
            except Exception as exc:
                self._log_msg(f"[studio] brain error: {exc}")
            self._thinking = False
            if reply:
                self._log_msg("avatar> " + reply)
                self.tts.speak(reply)
            else:
                self._log_msg("[studio] no answer — speaking your text as-is.")
                self.tts.speak(txt)
        threading.Thread(target=_think, daemon=True).start()

    def _load_character(self):
        """Pick ANY face image as the avatar character (no training needed).

        LivePortrait animates any source face, so the avatar is not tied to one
        character — swap to a celebrity, an AI-generated face, a cartoon, anyone.
        The source is encoded at engine start, so this restarts the pipeline.
        """
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Choose a character face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All", "*.*")])
        if not path:
            return
        self._char_path = path
        self._log_msg(f"[studio] character set: {os.path.basename(path)}")
        if self.running:
            self._log_msg("[studio] restarting with new character…")
            self.stop()
            self.root.after(600, self.start)
        else:
            self._log_msg("[studio] press START to bring this character to life.")

    def recenter(self):
        if self.engines and self.running:
            try:
                self.engines["lp"].recenter()
                self.engines["body"].recenter()
                self._log_msg("[studio] pose recentered — hold still, facing forward.")
            except Exception as exc:
                self._log_msg(f"[studio] recenter failed: {exc}")

    def toggle_mute(self):
        if self.tts is None:
            return
        self.tts.set_muted(not self.tts.muted)
        muted = self.tts.muted
        self.mute_btn.configure(text="UNMUTE" if muted else "MUTE",
                                bg=RED if muted else BG2,
                                fg="#ffffff" if muted else FG)
        self._log_msg("[studio] muted" if muted else "[studio] unmuted")

    def _on_voice(self, *args):
        if self.tts is not None:
            self.tts.set_voice(self.voice_var.get())

    def _on_voice_mode(self, *args):
        """Switch the TTS backend live, then warm its model in the background.
        SPEAK is disabled until the model is ready, so a line fired mid-load
        can't slip through on the wrong (fallback) voice."""
        if self.tts is None:
            return
        key = VOICE_MODE_KEY.get(self.voicemode_var.get(), "kokoro")
        self.tts.set_backend(key)
        heavy = key in ("maya1", "chatterbox")
        self._log_msg(f"[studio] voice mode -> {self.voicemode_var.get()}"
                      + (" (loading model, ~15-30s, please wait...)" if heavy else ""))
        if key == "maya1":
            self._log_msg("[studio] NOTE: Maya1 generates each NEW line in ~8-14s "
                          "(preview holds while it works). It's too heavy for smooth "
                          "live talking on one GPU — best for short/repeated lines "
                          "(repeats are cached = instant). For smooth live use "
                          "'Real human voice (Chatterbox)' or 'Fast (Kokoro)'.")
        if heavy:
            self.root.after(0, lambda: self.speak_btn.configure(
                state="disabled", text="LOADING VOICE..."))
            self._set_status("loading voice...", "#cc9933")

        def _warm():
            try:
                msg = self.tts.warm_backend()
                self._log_msg("[studio] voice ready: " + msg)
                # If a heavy voice was asked for but the banner says Kokoro, it
                # failed to load (e.g. VRAM) — make that obvious, not silent.
                if heavy and "Maya1" not in msg and "Chatterbox" not in msg:
                    self._log_msg("[studio] ⚠ expressive voice did NOT load — "
                                  "fell back. Check VRAM / console for the error.")
            except Exception as exc:
                self._log_msg(f"[studio] voice load failed: {exc}")
            finally:
                if self.running:
                    self.root.after(0, lambda: self.speak_btn.configure(
                        state="normal", text="SPEAK"))
                    self._set_status("LIVE", GREEN)
        threading.Thread(target=_warm, daemon=True).start()

    def _insert_tag(self, tag):
        """Insert an emotion tag at the cursor in the speak box (Maya1 performs
        it; other voices strip it)."""
        try:
            self.entry.insert("insert", " " + tag + " ")
            self.entry.focus_set()
        except Exception:
            pass

    def _on_interval(self):
        self.lp_interval = max(1, int(self.interval_var.get()))

    def _on_quality(self, *args):
        p = QUALITY_PRESETS.get(self.quality_var.get())
        if not p:
            return
        self.lp_interval = p["lp"]
        self.interval_var.set(p["lp"])
        self.body_var.set(p["body"])
        try:
            import enhance_engine as ee
            ee.set_level(p["enhance"])
        except Exception:
            pass
        if self.engines and "restore" in self.engines:
            try:
                self.engines["restore"].every_n = p.get("restore_every", 2)
            except Exception:
                pass
        self._log_msg(f"[studio] quality: {self.quality_var.get()} "
                      f"(LP every {p['lp']}, enhance {p['enhance']}, "
                      f"body {'on' if p['body'] else 'off'}, "
                      f"restore every {p.get('restore_every', 2)})")

    def _on_stab(self):
        if self.engines:
            try:
                self.engines["lp"].set_stabilization(self.stab_var.get() / 100.0)
            except Exception:
                pass

    def _on_gaze(self):
        if self.engines:
            try:
                self.engines["lp"].set_gaze(self.gaze_var.get(),
                                            self.gaze_var2.get() / 100.0)
            except Exception:
                pass

    def _on_minface(self):
        if self.engines:
            try:
                self.engines["lp"].min_good_face = max(0.04, self.minface_var.get() / 100.0)
            except Exception:
                pass

    def _on_pose(self, *args):
        """Safe / Cinematic / Free — sets the turn + tilt caps together."""
        p = POSE_PRESETS.get(self.pose_var.get())
        if not p:
            return
        self.turncap_var.set(p["turn"])
        self.tilt_var.set(p["tilt"])
        self._on_turncap()
        self._on_tilt()
        self._log_msg(f"[studio] pose preset: {self.pose_var.get()} "
                      f"(turn {p['turn']}deg / tilt {p['tilt']}deg)")

    def _on_turncap(self):
        """Live-set the yaw (turn) + roll caps. Pitch is owned by Max tilt."""
        try:
            import liveportrait_engine as lpe
            v = float(self.turncap_var.get())
            lpe.YAW_CAP = v
            lpe.ROLL_CAP = max(10.0, v * 0.9)
            self._log_msg(f"[studio] max turn -> {v:.0f}deg (cleaner if smaller)")
        except Exception:
            pass

    def _on_tilt(self):
        """Live-set the pitch (up/down tilt) cap — stops the uncanny stretch."""
        try:
            import liveportrait_engine as lpe
            lpe.PITCH_CAP = float(self.tilt_var.get())
            self._log_msg(f"[studio] max tilt -> {self.tilt_var.get()}deg")
        except Exception:
            pass

    def _on_skin(self):
        """Live-set GFPGAN restoration blend strength (skin detail)."""
        if self.engines:
            try:
                self.engines["restore"].skin_detail = self.skin_var.get() / 100.0
            except Exception:
                pass

    def _on_multiref(self):
        """Live A/B: extended multi-view turning vs safe single-image (capped)."""
        if not self.engines:
            return
        try:
            lp = self.engines["lp"]
            want = bool(self.multiref_var.get())
            if want and len(getattr(lp, "_refs", [])) <= 1:
                self._log_msg("[studio] multi-view not loaded this session. To enable, "
                              "restart with env AVATAR_MULTIREF=1 (slower boot).")
                self.multiref_var.set(False)
                return
            lp._multi = want
            self._log_msg("[studio] turning: "
                          + ("EXTENDED multi-view" if lp._multi else "SAFE single-image (clean)"))
        except Exception:
            pass

    def _on_close(self):
        self.running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        for fn in (lambda: self.cap.release() if self.cap else None,
                   lambda: self.obs_cam.close() if self.obs_cam else None,
                   lambda: self.tts.shutdown() if self.tts else None):
            try:
                fn()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    AvatarStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
