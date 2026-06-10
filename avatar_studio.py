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

MOTION_THRESH = 7.0            # mean 64x64 gray-diff above which LP runs every frame

# Quality presets -> (lp_interval, enhance level, body motion, restore cadence).
# "Delulu" is the tuned default: restoration every 3rd frame (crisp filmed look)
# + smooth head (motion-adaptive bumps to every-frame on movement) + subtle body.
QUALITY_PRESETS = {
    "Delulu (recommended)":  dict(lp=2, enhance="light", body=True,  restore_every=3),
    "Smooth (max fps)":      dict(lp=3, enhance="light", body=False, restore_every=4),
    "Sharp (max detail)":    dict(lp=1, enhance="full",  body=True,  restore_every=2),
}
QUALITY_LABELS = list(QUALITY_PRESETS.keys())

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
# DESIGN TOKENS — a small, cohesive dark "studio" palette.
#   BG       app canvas (near-black, faint blue)
#   SURFACE  floating card faces            SURFACE2  insets / fields / entries
#   BORDER   hairline card + field edges
#   FG/MUTED/FAINT  text hierarchy (primary / secondary / labels)
#   ACCENT   single brand accent (cyan) + matching hover; semantic green/red/amber
# Legacy names (BG2/ENTRY_BG) kept as aliases so the rest of the class is unchanged.
# -----------------------------------------------------------------------------
BG       = "#0c0e13"      # app canvas
SURFACE  = "#161b24"      # cards
SURFACE2 = "#0f131a"      # fields / entries / insets
BORDER   = "#262e3b"      # hairline borders
FG       = "#e9edf4"      # primary text
MUTED    = "#8b95a8"      # secondary text / control labels
FAINT    = "#5a6677"      # section headers / faint captions

ACCENT     = "#2dd4ff"    # brand accent (cyan)
ACCENT_HI  = "#62e2ff"    # accent hover
ACCENT_INK = "#04222c"    # text on accent
GREEN      = "#2bd576"    # live / success
GREEN_HI   = "#46e189"
GREEN_INK  = "#04240f"    # text on green
RED        = "#f0556a"    # stopped / error
AMBER      = "#f5b13d"    # warnings / recenter

# Back-compat aliases (referenced elsewhere in this module).
BG2      = "#1b212c"      # ghost-button / mute base surface
ENTRY_BG = SURFACE2


class AvatarStudio:
    """Tk window that runs the avatar pipeline and previews the final frame."""

    def __init__(self, root):
        self.root = root
        self.running = False
        self.booting = False
        self.engines = None
        self.tts = None
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

        root.title("Avatar Studio")
        root.configure(bg=BG)
        root.geometry("1200x880")
        root.minsize(1000, 640)

        self._init_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()                     # start the UI refresh loop

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

    def _add_hover(self, btn, base, hover):
        """Lighten a flat button on hover (skips disabled state)."""
        def on_enter(_):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=hover)
        def on_leave(_):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=base)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def _btn(self, parent, text, cmd, *, bg, fg, hover,
             font=("Segoe UI", 10, "bold"), state="normal"):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font,
                      relief="flat", bd=0, cursor="hand2", activebackground=hover,
                      activeforeground=fg, state=state, highlightthickness=0,
                      disabledforeground=FAINT)
        self._add_hover(b, bg, hover)
        return b

    def _chip(self, parent, text, cmd, full=False):
        b = tk.Button(parent, text=text, command=cmd, bg=SURFACE2, fg=MUTED,
                      font=("Segoe UI", 8) if full else ("Consolas", 8),
                      relief="flat", bd=0, cursor="hand2", padx=8, pady=4,
                      activebackground=BORDER, activeforeground=FG,
                      highlightthickness=0, anchor="w" if full else "center")
        self._add_hover(b, SURFACE2, BORDER)
        return b

    def _check(self, parent, text, var, cmd=None):
        return tk.Checkbutton(parent, text=text, variable=var, command=cmd,
                              bg=SURFACE, fg=FG, selectcolor=SURFACE2,
                              activebackground=SURFACE, activeforeground=FG,
                              font=("Segoe UI", 9), anchor="w", justify="left",
                              highlightthickness=0, bd=0, padx=0, cursor="hand2")

    def _card(self, parent, title=None):
        """A floating SURFACE card with a 1px border and an accent-dotted title."""
        border = tk.Frame(parent, bg=BORDER)
        border.pack(fill="x", pady=(0, 11))
        inner = tk.Frame(border, bg=SURFACE)
        inner.pack(fill="both", padx=1, pady=1)
        body = tk.Frame(inner, bg=SURFACE)
        body.pack(fill="both", padx=14, pady=(11, 13))
        if title:
            head = tk.Frame(body, bg=SURFACE); head.pack(fill="x", pady=(0, 9))
            tk.Label(head, text="●", bg=SURFACE, fg=ACCENT,
                     font=("Segoe UI", 7)).pack(side="left", padx=(0, 6))
            tk.Label(head, text=title, bg=SURFACE, fg=FAINT,
                     font=("Segoe UI", 8, "bold")).pack(side="left")
        return body

    def _row(self, parent, label):
        r = tk.Frame(parent, bg=SURFACE); r.pack(fill="x", pady=5)
        tk.Label(r, text=label, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        return r

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # ===== TOP APP BAR (full width) =====================================
        bar = tk.Frame(self.root, bg=BG, height=58)
        bar.pack(side="top", fill="x", padx=18, pady=(14, 0))
        bar.pack_propagate(False)
        tk.Label(bar, text="◆", bg=BG, fg=ACCENT, font=("Segoe UI", 16)).pack(side="left")
        tk.Label(bar, text="  AVATAR  STUDIO", bg=BG, fg=FG,
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        tk.Label(bar, text="LIVE TEST", bg=SURFACE2, fg=MUTED,
                 font=("Segoe UI", 8, "bold"), padx=9, pady=3).pack(side="left", padx=12)
        tk.Label(bar, text="real-time AI avatar pipeline", bg=BG, fg=FAINT,
                 font=("Segoe UI", 9)).pack(side="right")
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="top", fill="x")

        # ===== BODY: preview (left) + control rail (right) ==================
        bodyf = tk.Frame(self.root, bg=BG)
        bodyf.pack(side="top", fill="both", expand=True)

        # ---- LEFT: framed live preview with status + fps + diagnostics -----
        left = tk.Frame(bodyf, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(18, 9), pady=16)

        pv_border = tk.Frame(left, bg=BORDER)
        pv_border.pack(fill="both", expand=True)
        pv = tk.Frame(pv_border, bg=SURFACE)
        pv.pack(fill="both", expand=True, padx=1, pady=1)

        # preview header: live status dot + label (left), fps (right)
        ph = tk.Frame(pv, bg=SURFACE, height=40); ph.pack(side="top", fill="x")
        ph.pack_propagate(False)
        self.status_canvas = tk.Canvas(ph, width=14, height=14, bg=SURFACE,
                                       highlightthickness=0)
        self.status_canvas.pack(side="left", padx=(15, 0))
        self.status_dot = self.status_canvas.create_oval(2, 2, 12, 12,
                                                         fill=RED, outline="")
        self.status_lbl = tk.Label(ph, text="stopped", bg=SURFACE, fg=FG,
                                   font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=8)
        self.fps_lbl = tk.Label(ph, text="", bg=SURFACE, fg=MUTED,
                                font=("Consolas", 10))
        self.fps_lbl.pack(side="right", padx=15)
        tk.Frame(pv, bg=BORDER, height=1).pack(side="top", fill="x")

        # the composited frame (black stage)
        stage = tk.Frame(pv, bg="#000000")
        stage.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.preview = tk.Label(stage, bg="#000000", bd=0)
        self.preview.pack(fill="both", expand=True)

        # footer: per-stage timing readout
        tk.Frame(pv, bg=BORDER, height=1).pack(side="top", fill="x")
        pf = tk.Frame(pv, bg=SURFACE, height=30); pf.pack(side="top", fill="x")
        pf.pack_propagate(False)
        self.diag_lbl = tk.Label(pf, text="ready", bg=SURFACE, fg=FAINT,
                                 font=("Consolas", 9))
        self.diag_lbl.pack(side="left", padx=15)

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
        self.start_btn = self._btn(c, "START", self.start, bg=GREEN, fg=GREEN_INK,
                                   hover=GREEN_HI, font=("Segoe UI", 12, "bold"))
        self.start_btn.pack(fill="x", ipady=9, pady=(0, 6))
        self.stop_btn = self._btn(c, "STOP", self.stop, bg=BG2, fg=FG, hover=BORDER,
                                  font=("Segoe UI", 12, "bold"), state="disabled")
        self.stop_btn.pack(fill="x", ipady=9, pady=(0, 9))
        self.char_btn = self._btn(c, "LOAD CHARACTER", self._load_character,
                                  bg=SURFACE2, fg=ACCENT, hover=BORDER,
                                  font=("Segoe UI", 9, "bold"))
        self.char_btn.pack(fill="x", ipady=6, pady=(0, 4))
        tk.Label(c, text="any face image — celebrity, AI render, cartoon",
                 bg=SURFACE, fg=FAINT, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 9))
        self.recenter_btn = self._btn(c, "RECENTER POSE", self.recenter,
                                      bg=SURFACE2, fg=AMBER, hover=BORDER,
                                      font=("Segoe UI", 9, "bold"), state="disabled")
        self.recenter_btn.pack(fill="x", ipady=6)
        tk.Label(c, text="sit upright facing the camera, then click",
                 bg=SURFACE, fg=FAINT, font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        # ---- PERFORMANCE ---------------------------------------------------
        c = self._card(right, "PERFORMANCE")
        r = self._row(c, "Quality preset")
        self.quality_var = tk.StringVar(value="Balanced")
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

        r = self._row(c, "Max turn °  (smaller = cleaner)")
        self.turncap_var = tk.IntVar(value=30)
        ttk.Spinbox(r, from_=10, to=44, increment=2, width=5,
                    textvariable=self.turncap_var, command=self._on_turncap,
                    style="Studio.TSpinbox").pack(side="right")

        r = self._row(c, "Max tilt °  (up / down)")
        self.tilt_var = tk.IntVar(value=15)
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
        self._check(c, "Extended turning  ·  multi-view  [experimental]",
                    self.multiref_var, self._on_multiref).pack(fill="x", pady=3)

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

        # ---- SPEAK ---------------------------------------------------------
        c = self._card(right, "SPEAK")
        self.entry = tk.Text(c, height=3, bg=SURFACE2, fg=FG, insertbackground=ACCENT,
                             font=("Segoe UI", 11), relief="flat", wrap="word",
                             padx=9, pady=7, highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=ACCENT)
        self.entry.pack(fill="x", pady=(0, 7))
        self.entry.bind("<Return>", self._on_enter)

        brow = tk.Frame(c, bg=SURFACE); brow.pack(fill="x")
        self.speak_btn = self._btn(brow, "SPEAK", self.speak, bg=ACCENT, fg=ACCENT_INK,
                                   hover=ACCENT_HI, font=("Segoe UI", 11, "bold"),
                                   state="disabled")
        self.speak_btn.pack(side="left", fill="x", expand=True, ipady=6)
        self.mute_btn = tk.Button(brow, text="MUTE", command=self.toggle_mute,
                                  bg=BG2, fg=FG, font=("Segoe UI", 11, "bold"),
                                  relief="flat", bd=0, width=8, cursor="hand2",
                                  activebackground=BORDER, state="disabled",
                                  highlightthickness=0)
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
        # A soft vertical gradient (BGR) so the idle stage reads as "designed",
        # not just a blank black box — with a centered call-to-action.
        top = np.array((26, 19, 14), np.float32)      # ~ #0e131a
        bot = np.array((42, 33, 24), np.float32)      # ~ #18212a
        ramp = np.linspace(0.0, 1.0, PREVIEW_SIZE, dtype=np.float32)[:, None, None]
        img = (top * (1 - ramp) + bot * ramp).astype(np.uint8)
        img = np.repeat(img, PREVIEW_SIZE, axis=1)
        cx = PREVIEW_SIZE // 2
        cv2.circle(img, (cx, 232), 30, (255, 212, 45), 2, cv2.LINE_AA)   # accent ring
        cv2.circle(img, (cx, 232), 4, (255, 212, 45), -1, cv2.LINE_AA)
        for txt, y, sc, col, th in (("PRESS  START", 300, 1.0, (235, 237, 233), 2),
                                    ("to bring the avatar to life", 336, 0.5,
                                     (140, 130, 118), 1)):
            (w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
            cv2.putText(img, txt, (cx - w // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
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
            if self.tts is not None and getattr(self.tts, "synthesizing", False):
                self._set_status("generating voice...", "#cc9933")
            elif getattr(self, "_speaking", False):
                self._set_status("speaking", GREEN)
            elif self.status_lbl.cget("text") in ("generating voice...", "speaking"):
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
                for b in (self.speak_btn, self.mute_btn, self.recenter_btn):
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
            if self.tts is not None and getattr(self.tts, "synthesizing", False):
                hold = last_final.copy()
                ov = hold.copy()
                cv2.rectangle(ov, (0, FRAME_SIZE // 2 - 26),
                              (FRAME_SIZE, FRAME_SIZE // 2 + 26), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.55, hold, 0.45, 0, hold)
                cv2.putText(hold, "generating voice...",
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
            lp_due = (cached_face is None or (frame_count % self.lp_interval) == 0
                      or motion > MOTION_THRESH)        # run every frame on big motion
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
            if self.body_var.get() and getattr(lp, "_face_found", False):
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
        for b in (self.speak_btn, self.mute_btn, self.recenter_btn):
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
        if self.tts is not None and self.running:
            self.tts.speak(txt)
            self._log_msg("> " + txt)
        else:
            self._log_msg("[studio] press START first.")

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
        self._log_msg(f"[studio] quality: {self.quality_var.get()} "
                      f"(LP every {p['lp']}, enhance {p['enhance']}, "
                      f"body {'on' if p['body'] else 'off'})")

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
