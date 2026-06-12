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
# Insert PROJECT first then ENGINES so ENGINES ends up FIRST in sys.path — the
# engine modules in engines/ must win over any stale duplicate in the project root
# (a root copy of bg_music.py was shadowing engines/bg_music.py, so the studio ran
# the OLD single-loop music and ignored the 50-track playlist edits).
for p in (PROJECT_DIR, ENGINES_DIR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# AUTO-CONFIG: probe the machine (GPU/VRAM/CPU + benchmark) and pick the voice /
# brain / restore / cadence that best fit. Run ASYNC (after the window shows) so
# the user sees a "benchmarking..." loading bar; START is disabled until it's done
# (the env it sets must be in place before the engines load). Your own AVATAR_*
# env vars still win (setdefault). Disable with AVATAR_AUTOCONFIG=0.
AUTO_PROFILE = None

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
# The bot speaks ARABIC + ENGLISH ONLY, via the Coqui XTTS-v2 cloned voice — one
# Arabic male who code-switches cleanly. No other voice options by request.
VOICE_MODES = [
    ("Arabic + English · XTTS-v2 (cloned)", "xtts"),
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
        # ALWAYS-ON resource monitor: live CPU/GPU/VRAM -> adaptive load routing so the
        # avatar never lags (movable filter work goes to whoever's free; heavy optional
        # passes drop when both are saturated). Starts now, runs the whole session.
        self.monitor = None
        try:
            from resource_monitor import ResourceMonitor
            self.monitor = ResourceMonitor()
        except Exception as _mexc:
            print(f"[monitor] resource monitor unavailable ({_mexc})")
        self.brain = None                    # Ollama LLM brain (answers in character)
        self.tiktok = None                   # LIVE TikTok comment reader
        self.responder = None                # comment filter + answerer (web research)
        self._comment_q = queue.Queue(maxsize=80)
        self._event_q = queue.Queue(maxsize=40)   # gifts / follows / shares / like-milestones
        self._next_like_ms = 500                  # next likes milestone to celebrate
        # SESSION STATS + gift goal (on-screen bar + CTAs)
        self._sess_likes = 0
        self._sess_coins = 0
        self._sess_follows = 0
        self._coin_goal = int(os.environ.get("AVATAR_COIN_GOAL", "200"))
        self._poll = None                         # active buy/sell poll {buy,sell,end}
        self._poll_last = 0.0                     # last poll start (monotonic)
        self.market = None                   # LIVE gold price feed (Binance PAXG)
        self._thinking = False               # True while the brain is generating
        self.cap = None
        self.obs_cam = None
        self._tv_proc = None                # the AI-driven TradingView browser
        self.lp_interval = 2
        self._char_path = None               # chosen character image (any face)

        self._latest = None                 # latest final frame (BGR ndarray)
        self._frame_lock = threading.Lock()
        self._log_q = queue.Queue()
        self._fps = 0.0
        self._diag = ""                      # per-stage ms readout
        self._speaking = False
        self._worker = None
        if not hasattr(self, "_thinking"):
            self._thinking = False
        # ONE coherent speech pipeline: every speech source (ASK, SPEAK, quick
        # phrases, Auto-host) shares ONE brain lock so only one generation runs at
        # a time (no GPU clash / no jumbled conversation history), and Auto-host
        # YIELDS for a cooldown after you interact so it never talks over you.
        self._brain_lock = threading.Lock()
        self._user_active_until = 0.0        # auto-host pauses until this time
        try:
            from bg_music import BackgroundMusic
            self.music = BackgroundMusic()       # trading-mood bed, ducks under voice
        except Exception:
            self.music = None

        root.title("AVATAR STUDIO ◆ neural pipeline")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        _wx = max(0, (sw - 1240) // 2); _wy = max(0, (sh - 900) // 3)
        root.geometry(f"1240x900+{_wx}+{_wy}")
        root.minsize(1040, 660)
        # frameless window -> our own futuristic title bar with custom controls
        self._drag = None
        self._tb_buttons = {}
        self._tb_hover = None
        try:
            root.overrideredirect(True)
            root.bind("<Map>", self._restore_override)
            root.after(80, lambda: (root.lift(), root.focus_force()))
        except Exception:
            pass

        self._init_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()                     # start the UI refresh loop
        self._animate()                     # start the HUD animation loop
        # ALWAYS-ON live-status monitor: from the moment the app opens it keeps
        # checking whether the entered @handle is LIVE on TikTok and drives the
        # green/red light by the SPEECH button. No START needed, never sleeps long.
        self._live_stop = False
        threading.Thread(target=self._live_status_loop, daemon=True).start()
        # ASYNC auto-config: benchmark the GPU AFTER the window paints, with a
        # loading bar; START stays disabled until the chosen model is known (its
        # env must be set before the engines load).
        if os.environ.get("AVATAR_AUTOCONFIG", "1") == "1":
            try:
                self.start_btn.configure(state="disabled")
            except Exception:
                pass
            threading.Thread(target=self._run_autoconfig, daemon=True).start()
        else:
            self._autoconfig_done()         # no benchmark — just show env defaults
        # Open the real TradingView and let the AI operate it (env AVATAR_TV=0
        # to disable). Delayed so the Studio window paints first.
        self.root.after(1500, self._launch_tradingview)

    def _launch_tradingview(self):
        """Open TradingView.com in a browser the AI drives (scroll/zoom/draw)."""
        import subprocess
        if os.environ.get("AVATAR_TV", "1") == "0":
            return
        proj = os.path.dirname(os.path.abspath(__file__))
        sym = os.environ.get("AVATAR_TV_SYMBOL", "OANDA:XAUUSD")
        lang = os.environ.get("AVATAR_TTS_LANG", "en")
        args = [sys.executable, os.path.join(proj, "tradingview_pilot.py"),
                "--symbol", sym, "--lang", lang if lang in ("en", "ar") else "en"]
        if os.environ.get("AVATAR_TV_SPEAK", "0") != "1":
            args.append("--no-speak")        # avoid double TTS load by default
        try:
            self._tv_proc = subprocess.Popen(args, cwd=proj)
            self._log_msg(f"[studio] opening TradingView ({sym}) - AI taking control...")
        except Exception as exc:
            self._log_msg(f"[studio] TradingView launch failed: {exc}")

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
    # FRAMELESS TITLE BAR — Mercedes brand, window controls, drag-to-move
    # -------------------------------------------------------------------------
    def _draw_mercedes(self, cv, cx, cy, r, tags="tb"):
        """Draw a chrome Mercedes-Benz three-pointed star with a faint glow."""
        SIL = "#d2dcea"
        glow = self._mix(CYAN, BG, 0.5)
        cv.create_oval(cx-r-2, cy-r-2, cx+r+2, cy+r+2, outline=self._mix(glow, BG, 0.6),
                       width=1, tags=tags)                       # outer halo
        cv.create_oval(cx-r, cy-r, cx+r, cy+r, outline=self._mix(SIL, BG, 0.45),
                       width=3, tags=tags)                       # ring (thick, dim)
        cv.create_oval(cx-r, cy-r, cx+r, cy+r, outline=SIL, width=1, tags=tags)  # ring rim
        for ang in (-math.pi / 2, math.pi / 6, math.pi * 5 / 6):  # up, dn-right, dn-left
            ex = cx + (r - 2) * math.cos(ang)
            ey = cy + (r - 2) * math.sin(ang)
            cv.create_line(cx, cy, ex, ey, fill=self._mix(SIL, BG, 0.5), width=3, tags=tags)
            cv.create_line(cx, cy, ex, ey, fill=SIL, width=1, tags=tags)
        cv.create_oval(cx-2, cy-2, cx+2, cy+2, fill=SIL, outline="", tags=tags)

    def _draw_winbtn(self, cv, kind, x, y, w, h):
        """Draw a minimise / exit control; lights up its accent on hover."""
        hover = (self._tb_hover == kind)
        accent = RED if kind == "exit" else CYAN
        self._round_rect(cv, x, y, x + w, y + h, 6,
                         outline=accent if hover else self._mix(BORDER, accent, 0.35),
                         fill=self._mix(BG, accent, 0.18 if hover else 0.05),
                         width=1, tags="tb")
        cxm, cym = x + w // 2, y + h // 2
        col = accent if hover else self._mix(FG, BG, 0.4)
        if kind == "min":
            cv.create_line(cxm - 6, cym + 4, cxm + 6, cym + 4, fill=col, width=2, tags="tb")
        else:
            cv.create_line(cxm - 5, cym - 5, cxm + 5, cym + 5, fill=col, width=2, tags="tb")
            cv.create_line(cxm - 5, cym + 5, cxm + 5, cym - 5, fill=col, width=2, tags="tb")

    def _tb_hit(self, x, y):
        for k, (x1, y1, x2, y2) in self._tb_buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return k
        return None

    def _tb_motion(self, e):
        h = self._tb_hit(e.x, e.y)
        if h != self._tb_hover:
            self._tb_hover = h
            try:
                self._topdraw()
                self._topcv.config(cursor="hand2" if h else "fleur")
            except Exception:
                pass

    def _tb_press(self, e):
        h = self._tb_hit(e.x, e.y)
        if h == "min":
            self._minimise(); return
        if h == "exit":
            self._on_close(); return
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _tb_drag(self, e):
        if self._drag:
            self.root.geometry(f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _tb_release(self, e):
        self._drag = None

    def _minimise(self):
        """Minimise a frameless window (drop override so it can iconify)."""
        try:
            self.root.overrideredirect(False)
            self.root.iconify()
        except Exception:
            pass

    def _restore_override(self, e=None):
        try:
            if e is not None and getattr(e, "widget", None) is not self.root:
                return
            if self.root.state() == "normal":
                self.root.overrideredirect(True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # ===== FRAMELESS TITLE BAR (Mercedes star + wordmark + win controls) ==
        TBH = 68
        topcv = tk.Canvas(self.root, bg=BG, height=TBH, highlightthickness=0, bd=0)
        topcv.pack(side="top", fill="x")
        self._topcv = topcv
        self._tbh = TBH
        self._sweep_y = TBH - 12
        self._sweep = topcv.create_oval(0, 0, 0, 0, fill=CYAN, outline="")

        def _topdraw(_=None):
            w = topcv.winfo_width()
            if w <= 1:
                return
            topcv.delete("tb")
            cy = TBH // 2 - 2
            # --- Mercedes three-pointed star, top-left ---
            self._draw_mercedes(topcv, 32, cy, 17, tags="tb")
            # --- wordmark + neon slash ---
            self._glow_text(topcv, 64, cy - 6, "AVATAR", FG, ("Consolas", 17, "bold"), tags="tb")
            self._glow_text(topcv, 64 + 104, cy - 6, "// STUDIO", CYAN,
                            ("Consolas", 17, "bold"), tags="tb")
            # subtitle: live dot + spaced caps
            topcv.create_oval(66, cy + 9, 72, cy + 15, fill=MINT, outline="", tags="tb")
            topcv.create_oval(64, cy + 7, 74, cy + 17, outline=self._mix(MINT, BG, 0.4),
                              width=1, tags="tb")
            topcv.create_text(80, cy + 12, text="N E U R A L   P I P E L I N E", anchor="w",
                              fill=self._mix(CYAN, BG, 0.45), font=("Consolas", 8), tags="tb")
            # --- window controls (minimise + exit), top-right ---
            bw, bh, gap = 36, 26, 8
            ex_x = w - 12 - bw
            mn_x = ex_x - gap - bw
            by = cy - bh // 2
            self._tb_buttons = {"min": (mn_x, by, mn_x + bw, by + bh),
                                "exit": (ex_x, by, ex_x + bw, by + bh)}
            self._draw_winbtn(topcv, "min", mn_x, by, bw, bh)
            self._draw_winbtn(topcv, "exit", ex_x, by, bw, bh)
            # telemetry to the left of the buttons
            topcv.create_text(mn_x - 16, cy - 5, text="SYS // ONLINE", anchor="e",
                              fill=self._mix(FG, BG, 0.32), font=("Consolas", 9), tags="tb")
            topcv.create_text(mn_x - 16, cy + 9, text="v2.0  •  GPU READY", anchor="e",
                              fill=self._mix(CYAN, BG, 0.4), font=("Consolas", 8), tags="tb")
            # underline rail (fading neon ticks)
            y = self._sweep_y
            span = w - 52
            for i in range(0, max(1, span), 7):
                topcv.create_line(26 + i, y, 26 + i + 4, y,
                                  fill=self._mix(CYAN, BG, 0.5 + 0.45 * (i / float(max(1, span)))),
                                  width=1, tags="tb")
            topcv.tag_raise(self._sweep)
        self._topdraw = _topdraw
        topcv.bind("<Configure>", _topdraw)
        topcv.bind("<Motion>", self._tb_motion)
        topcv.bind("<Button-1>", self._tb_press)
        topcv.bind("<B1-Motion>", self._tb_drag)
        topcv.bind("<ButtonRelease-1>", self._tb_release)
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
        # TOP music mute toggle — one click to silence / restore the bg music.
        self.music_btn = tk.Button(ph, text="♪ MUSIC", command=self._toggle_music,
                                   bg=SURFACE2, fg=CYAN, font=("Consolas", 10, "bold"),
                                   relief="flat", bd=0, padx=10, cursor="hand2",
                                   activebackground=self._mix(SURFACE2, CYAN, 0.2),
                                   activeforeground=CYAN, highlightthickness=1,
                                   highlightbackground=self._mix(CYAN, BG, 0.5))
        self.music_btn.pack(side="right", padx=(0, 10))
        # TOP bot-speech mute — silences the VOICE (lips still move).
        self.speech_btn = tk.Button(ph, text="🎤 SPEECH", command=self._toggle_speech,
                                    bg=SURFACE2, fg=MAG, font=("Consolas", 10, "bold"),
                                    relief="flat", bd=0, padx=10, cursor="hand2",
                                    activebackground=self._mix(SURFACE2, MAG, 0.2),
                                    activeforeground=MAG, highlightthickness=1,
                                    highlightbackground=self._mix(MAG, BG, 0.5))
        self.speech_btn.pack(side="right", padx=(0, 10))
        # LIVE light: GREEN = the entered @handle is LIVE on TikTok, RED = offline,
        # grey = no handle. Checked continuously by _live_status_loop (always active).
        self.live_light = tk.Canvas(ph, width=22, height=22, bg=BG, highlightthickness=0)
        self._live_glow = self.live_light.create_oval(2, 2, 20, 20, fill="", outline="")
        self._live_dot = self.live_light.create_oval(6, 6, 16, 16, fill="#3a3f4a", outline="")
        self.live_light.pack(side="right", padx=(0, 2))
        self._handle_live = False
        # tooltip-ish label under it via the log; the dot is the at-a-glance signal.
        # auto-config readout: shows a "benchmarking..." loading bar first, then
        # resolves to the chosen LLM brain + GPU benchmark once it completes.
        self.info_lbl = tk.Label(ph, text="⏳ benchmarking GPU…", bg=SURFACE, fg=AMBER,
                                 font=("Consolas", 9))
        self.info_lbl.pack(side="left", padx=(10, 6))
        self.bench_bar = ttk.Progressbar(ph, mode="indeterminate", length=120)
        self.bench_bar.pack(side="left", pady=2)
        self.bench_bar.start(14)

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

        # ===== LIVE TIKTOK COMMENTS — docked BELOW the avatar ================
        # Moved here from the right rail so the comment feed sits right under the
        # face, like a real TikTok live: @handle + toggle + a scrolling feed of
        # incoming comments and the avatar's spoken replies.
        cm = tk.Frame(left, bg=SURFACE, highlightthickness=1,
                      highlightbackground=self._mix(MAG, BG, 0.5))
        cm.pack(side="top", fill="both", expand=True, pady=(10, 0))
        ch = tk.Frame(cm, bg=SURFACE); ch.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ch, text="\U0001f4ac LIVE TIKTOK COMMENTS", bg=SURFACE, fg=MAG,
                 font=("Consolas", 10, "bold")).pack(side="left")
        # green/red live dot + word, mirrored from the one by the SPEECH button
        self.feed_light = tk.Canvas(ch, width=16, height=16, bg=SURFACE,
                                    highlightthickness=0)
        self._feed_dot = self.feed_light.create_oval(3, 3, 13, 13,
                                                     fill="#3a3f4a", outline="")
        self.feed_light.pack(side="left", padx=(10, 0))
        self.feed_status = tk.Label(ch, text="no handle", bg=SURFACE, fg=MUTED,
                                    font=("Consolas", 9))
        self.feed_status.pack(side="left", padx=(5, 0))
        # @handle entry + Answer toggle on the right
        self.comments_var = tk.BooleanVar(value=False)
        self._check(ch, "Answer", self.comments_var,
                    self._on_comments).pack(side="right")
        self.handle_var = tk.StringVar(value=os.environ.get("AVATAR_TIKTOK_USER", ""))
        tk.Entry(ch, textvariable=self.handle_var, width=14, bg=BG, fg=FG,
                 insertbackground=CYAN, relief="flat",
                 font=("Segoe UI", 10)).pack(side="right", padx=(0, 8), ipady=2)
        tk.Label(ch, text="@handle", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="right", padx=(0, 4))
        # "NOW ANSWERING" — the comment the AI has committed to and is researching
        # / answering right now (genuine answers only, not filtered spam).
        anbg = self._mix(SURFACE, MAG, 0.22)
        self._answer_bar = tk.Frame(cm, bg=anbg)
        self._answer_bar.pack(fill="x", padx=10, pady=(0, 5))
        self.answering_lbl = tk.Label(self._answer_bar, text="○  idle — waiting for a question",
                                      bg=anbg, fg=MUTED, font=("Consolas", 9),
                                      anchor="w", justify="left", wraplength=560)
        self.answering_lbl.pack(fill="x", padx=9, pady=5)
        # scrolling read-only feed
        fb = tk.Frame(cm, bg=SURFACE); fb.pack(fill="both", expand=True, padx=10, pady=(0, 9))
        fsb = tk.Scrollbar(fb); fsb.pack(side="right", fill="y")
        self.feed = tk.Text(fb, height=6, bg=BG, fg=FG, relief="flat", bd=0,
                            font=("Consolas", 9), wrap="word", padx=8, pady=6,
                            state="disabled", yscrollcommand=fsb.set)
        self.feed.pack(side="left", fill="both", expand=True)
        fsb.config(command=self.feed.yview)
        self.feed.tag_config("q", foreground=CYAN)            # viewer comment
        self.feed.tag_config("a", foreground="#27ff9e")       # avatar reply
        self.feed.tag_config("ev", foreground=AMBER)          # gift / follow
        self.feed.tag_config("sys", foreground=MUTED)         # system note
        self._feed_msg("enter your @handle and go live — comments appear here.", "sys")

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
        self.stab_var = tk.IntVar(value=20)
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

        self.liplock_var = tk.BooleanVar(value=True)
        self._check(c, "Lips from bot voice only  ·  ignore my real mouth",
                    self.liplock_var, self._on_liplock).pack(fill="x", pady=3)
        # AUTO-TALK: the brain writes + speaks gold commentary on its own (no typing).
        self.autotalk_var = tk.BooleanVar(value=True)
        self._check(c, "Auto-talk  ·  bot hosts the stream by itself (AI commentary)",
                    self.autotalk_var, self._on_autotalk).pack(fill="x", pady=3)

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
        self.music_var = tk.BooleanVar(value=True)
        self._check(c, "Background music  ·  trading mood (ducks under voice)",
                    self.music_var, self._on_music).pack(fill="x", pady=3)
        self.multiref_var = tk.BooleanVar(value=False)
        self._check(c, "Extended turning  ·  multi-view (wider, less stable)",
                    self.multiref_var, self._on_multiref).pack(fill="x", pady=3)
        # ON by default: face-swap (White-Haddan onto your real head). Bot-only lip-lock
        # keeps the mouth driven by the voice, not your webcam mouth.
        self.swap_var = tk.BooleanVar(value=True)
        self._check(c, "FACE-SWAP mode  ·  YOUR real head (your real mouth)",
                    self.swap_var, self._on_swap).pack(fill="x", pady=3)
        # CHARACTER picker — switch identity live (white man / Haddan / any folder).
        r = self._row(c, "Character")
        self.char_var = tk.StringVar(value="White Haddan")
        ttk.Combobox(r, textvariable=self.char_var,
                     values=["White Haddan", "Haddan", "White man"],
                     state="readonly", width=14,
                     style="Studio.TCombobox").pack(side="right")
        self.char_var.trace_add("write", self._on_character)
        # Hair / beard COLOUR — recolours gray hair+beard to match the character.
        r = self._row(c, "Hair colour")
        self.hair_var = tk.StringVar(value="gray")
        ttk.Combobox(r, textvariable=self.hair_var,
                     values=["brown", "black", "blonde", "gray", "none"],
                     state="readonly", width=14,
                     style="Studio.TCombobox").pack(side="right")
        self.hair_var.trace_add("write", self._on_hair)
        # Eye COLOUR — recolour the iris (off keeps the swapped source's eyes).
        r = self._row(c, "Eye colour")
        self.eye_var = tk.StringVar(value="gray")
        ttk.Combobox(r, textvariable=self.eye_var,
                     values=["off", "blue", "green", "hazel", "brown", "amber", "gray"],
                     state="readonly", width=14,
                     style="Studio.TCombobox").pack(side="right")
        self.eye_var.trace_add("write", self._on_eye)

        # ---- SCENE & OUTPUT ------------------------------------------------
        c = self._card(right, "SCENE & OUTPUT")
        self.chart_var = tk.BooleanVar(value=False)
        self._check(c, "Show live charts when face is lost",
                    self.chart_var).pack(fill="x", pady=3)
        # TRADER SCENE: live chart full-frame + avatar host in a PiP corner (one app).
        self.trader_var = tk.BooleanVar(value=False)
        self._check(c, "Trader scene  ·  chart + avatar PiP (AI trading host)",
                    self.trader_var).pack(fill="x", pady=3)
        # BROADCAST framing: avatar at natural size on a soft self-blur = SHARP mouth
        # (the 96px lip-sync isn't stretched across a full-screen face) + cleaner look.
        self.broadcast_var = tk.BooleanVar(value=True)
        self._check(c, "Broadcast framing  ·  sharper mouth (no full-screen stretch)",
                    self.broadcast_var).pack(fill="x", pady=3)
        # Live CPU/GPU/VRAM readout (resource governor) — corner overlay.
        self.perf_var = tk.BooleanVar(value=True)
        self._check(c, "Show CPU/GPU monitor  ·  live load + auto-balancing",
                    self.perf_var).pack(fill="x", pady=3)
        self.obs_var = tk.BooleanVar(value=False)
        self._check(c, "Also send to OBS virtual camera",
                    self.obs_var).pack(fill="x", pady=3)

        # (LIVE TIKTOK COMMENTS moved to the docked panel below the avatar.)

        # ---- VOICE ---------------------------------------------------------
        c = self._card(right, "VOICE")
        tk.Label(c, text="Mode", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        # default the dropdown to the AUTO-CONFIG-picked voice (AVATAR_TTS)
        _auto_tts = os.environ.get("AVATAR_TTS", "")
        _def_label = next((lbl for lbl, key in VOICE_MODES if key == _auto_tts),
                          VOICE_MODE_LABELS[0])
        self.voicemode_var = tk.StringVar(value=_def_label)
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
        # Background music: on while LIVE + toggled, ducks under the voice and
        # swells back up the instant the AI pauses.
        if self.music is not None:
            try:
                want = bool(self.running and getattr(self, "music_var", None)
                            and self.music_var.get())
                self.music.set_active(want)
                self.music.set_speaking(bool(getattr(self, "_speaking", False)))
            except Exception:
                pass
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
        # JARVIS-style boot cue the moment START is pressed (non-blocking).
        try:
            from startup_sound import play_startup_sound
            play_startup_sound()
        except Exception:
            pass
        self.start_btn.configure(state="disabled", text="STARTING...")
        self.lp_interval = max(1, int(self.interval_var.get()))
        self._set_status("starting...", "#cc9933")
        self._log_msg("[studio] building engines (LivePortrait + Wav2Lip warmup ~60-90s)...")
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        try:
            if AUTO_PROFILE:
                r = AUTO_PROFILE["res"]
                self._log_msg(f"[auto-config] {r['gpu']} · {r['vram_free']:.1f}GB free "
                              f"· {r['tflops']:.0f} TFLOP/s")
                self._log_msg("[auto-config] -> " + AUTO_PROFILE["cfg"]["why"])
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
                lp.set_lip_lock(self.liplock_var.get())
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
                self.root.after(0, self._update_info)   # show the ACTUAL model up top
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

            # CAMERA cue: input/virtual camera is now detected / ready.
            try:
                from startup_sound import play_camera_sound
                play_camera_sound()
            except Exception:
                pass

            from body_motion import BodyMotionEngine
            from restore_engine import RestoreEngine
            self._log_msg("[studio] GFPGAN restoration...")
            restore = RestoreEngine()
            self._log_msg("   -> " + restore.startup_check()[1])
            restore.skin_detail = self.skin_var.get() / 100.0
            self.engines = {"lp": lp, "mt": mt, "comp": comp, "enh": enhance_engine,
                            "chart": TradingView("XAUUSD"), "body": BodyMotionEngine(),
                            "restore": restore}
            # LIVE market feed so the host talks about the REAL current gold price
            # (not the simulated chart's drifting fake number). Background poller.
            try:
                from market_data import MarketData
                self.market = MarketData("PAXGUSDT", "1m")
                self.market.start()
                self._log_msg("   -> " + self.market.startup_check()[1])
                p = self.market.price
                if p > 0:                       # match the visual chart to reality
                    self.engines["chart"].price = p
                    self.engines["chart"].day_open = p
            except Exception as exc:
                self.market = None
                self._log_msg(f"[studio] live market data unavailable ({exc}).")
            self._on_quality()        # apply the selected quality preset at boot
            self._on_tilt()           # apply max-tilt (pitch) cap at boot
            self._on_turncap()        # apply max-turn (yaw) cap at boot
            # FACE-SWAP is our focus mode: load it eagerly at boot so it's ready
            # the instant the loop starts (no first-frame stall).
            if self.swap_var.get():
                try:
                    from faceswap_engine import FaceSwapEngine
                    self._log_msg("[studio] loading face-swap (ReSwapper-256 + insightface)...")
                    self.swap_engine = FaceSwapEngine(self._char_path or _character_path())
                    if self.swap_engine.ready:
                        # load the DEFAULT character (from the Character dropdown),
                        # falling back to whichever folder exists.
                        _pref = {"White Haddan": "haddan_white", "Haddan": "Haddan",
                                 "White man": "character_src"}.get(self.char_var.get(), "haddan_white")
                        order = [_pref] + [d for d in ("haddan_white", "Haddan", "character_src")
                                           if d != _pref]
                        for _cand in order:
                            _d = os.path.join(PROJECT_DIR, _cand)
                            if os.path.isdir(_d) and self.swap_engine.set_source_from_folder(_d):
                                self._log_msg(f"[studio] character: {_cand}")
                                break
                        # apply default hair/eye colour + stabilization to the engine
                        self.swap_engine._hair_color = self.hair_var.get()
                        self.swap_engine._eye_color = self.eye_var.get()
                        if hasattr(self.swap_engine, "set_stabilization"):
                            self.swap_engine.set_stabilization(self.stab_var.get() / 100.0)
                except Exception as exc:
                    self._log_msg(f"[studio] face-swap load failed: {exc}")
            self.tts = tts
            self.cap = cap
            self.obs_cam = obs

            self.running = True
            self.booting = False
            # SCENE cue: the avatar scene is going live now.
            try:
                from startup_sound import play_scene_sound
                play_scene_sound()
            except Exception:
                pass
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
            # AUTO-TALK: the brain writes + speaks gold commentary on its own, PIPELINED
            # (generates the next line while the current one plays — voice gen never
            # pauses the LLM).
            self._autotalk_thread = threading.Thread(target=self._autotalk_loop, daemon=True)
            self._autotalk_thread.start()
            # LIVE MARKET ALERTS: watch the real gold price, react to big moves/levels.
            self._market_thread = threading.Thread(target=self._market_monitor, daemon=True)
            self._market_thread.start()
            # WATCHDOG: auto-recover the render thread if it dies (unattended streaming).
            self._wd_thread = threading.Thread(target=self._watchdog, daemon=True)
            self._wd_thread.start()

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

            # The avatar KEEPS RENDERING while the brain thinks and the TTS generates
            # voice — those run on separate threads/process (Ollama + Chatterbox fit
            # alongside the swap in 16GB), so the face must NEVER freeze on stream.
            # (The old code held a "thinking..." frame here — that read as the avatar
            # breaking, which is exactly what we're removing.)
            self._busy_gen = bool(self._thinking
                                  or (self.tts is not None
                                      and getattr(self.tts, "synthesizing", False)))

            # WHILE THE BOT IS TALKING: keep the frame budget low so the lip-sync
            # stays smooth (no lag). The head and skin are basically static during
            # speech and only the mouth moves (owned by the mouth-sync), so we run
            # LivePortrait + the costly face restore HALF as often and reuse the
            # held result between — freeing the GPU for per-frame mouth-sync.
            speaking_now = bool(self.tts is not None and getattr(self.tts, "speaking", False))
            _speed = 2 if speaking_now else 1

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
                        # CHARACTER identity from the training folder (all angles),
                        # averaged + incremental so daily-added photos fold in.
                        tdir = next((os.path.join(PROJECT_DIR, d) for d in ("character_src", "Haddan") if os.path.isdir(os.path.join(PROJECT_DIR, d))), os.path.join(PROJECT_DIR, "Haddan"))
                        if self.swap_engine.ready and os.path.isdir(tdir):
                            n = self.swap_engine.set_source_from_folder(tdir)
                            if n:
                                self._log_msg(f"[studio] character trained from {n} photos (Haddan)")
                    except Exception as exc:
                        self._log_msg(f"[studio] face-swap load failed: {exc}")
                        self.swap_var.set(False)
                if self.swap_engine is not None and self.swap_engine.ready:
                    _t = time.perf_counter()
                    ai = self.swap_engine.swap(driving)
                    cached_face = ai; lp_fresh = True; did_swap = True
                    lp._face_found = self.swap_engine.last_found   # chart/loss logic
                    t_lp += time.perf_counter() - _t

            # head updates less often while the bot talks (it's near-static then)
            eff_lp_interval = max(1, self.lp_interval * _speed)
            lp_due = (cached_face is None or (frame_count % eff_lp_interval) == 0
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
            if self.restore_var.get() and not did_swap and getattr(lp, "_face_found", False):
                try:
                    re = self.engines["restore"]
                    # restore less often while talking (fewer 85ms spikes) — the
                    # restored skin/eyes are reused; the mouth gets overwritten by
                    # the mouth-sync anyway, so there's nothing lost during speech.
                    re.every_n = max(1, getattr(self, "_restore_every_base", 3) * _speed)
                    ai = re.restore(ai)
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

            # let the mouth de-blur run only when the GPU has room (skips during the
            # voice-synthesis GPU spike = sharp mouth with NO speech lag).
            if mt is not None:
                mt.allow_deblur = (self.monitor is None or self.monitor.gpu_free(78))

            # "the bot is ACTUALLY talking" — from the TTS, NOT the mouth engine
            # (which we may keep alive with idle silence below).
            self._speaking = bool(self.tts is not None and getattr(self.tts, "speaking", False))
            lips_from_bot = bool(self.liplock_var.get())
            try:                       # mouth = BOT only (closed when silent, never webcam)
                mt.bot_only = lips_from_bot
            except Exception:
                pass

            if chart_fade >= 1.0:
                # fully on charts — skip the (now hidden) avatar mouth/enhance work
                final = chart.render(speaking=self._speaking)
            else:
                # LIPS FROM BOT: the mouth is driven by the TTS EVERY frame —
                # talking when the bot speaks, CLOSED (neutral) when it's silent —
                # so the operator's real mouth NEVER shows. Works in face-swap mode
                # too (where LivePortrait/lip-lock is bypassed). When lip-lock is
                # off, fall back to old behaviour (mouth-sync only while speaking).
                if lips_from_bot and not self._speaking:
                    try:                       # trickle silence -> closed-mouth render
                        mt.feed_audio((np.random.randn(640).astype(np.float32)) * 1e-3)
                    except Exception:
                        pass
                if self._speaking or lips_from_bot:
                    try:
                        if lp_fresh or cached_bbox is None:
                            _mh = (getattr(self.swap_engine, "last_mouth", None)
                                   if did_swap and self.swap_engine is not None else None)
                            cached_bbox = comp.detect_mouth_bbox(ai, _mh)
                        mouth = mt.process_mouth(ai, cached_bbox)
                        ai = comp.blend_mouth(ai, mouth, cached_bbox)
                    except Exception as exc:
                        errs += 1
                        if errs <= 3:
                            self._log_msg(f"[studio] mouth error: {exc}")
                # HD-restore the FINAL face (swap + bot mouth + real eyes) together so
                # the MOUTH gets the same CodeFormer sharpness as the eyes/skin.
                if did_swap and self.swap_engine is not None:
                    try:
                        ai = self.swap_engine.restore_face(ai)
                    except Exception:
                        pass
                _t = time.perf_counter()
                try:
                    # FACE-SWAP streamer look: force FULL enhance so the person is
                    # cut from their room and composited onto the trading studio
                    # background, with the lighting grade + ticker + LIVE badge.
                    # ADAPTIVE: the monitor picks the enhance level (drop to 'light' if
                    # CPU+GPU are both saturated = never stutter) and the device the
                    # movable filter work runs on (whichever is freer).
                    _dev, _lvl = "cpu", "full"
                    if self.monitor is not None:
                        _dev = self.monitor.route_filters()
                        _lvl = self.monitor.quality()
                    if did_swap:
                        enh.set_level(_lvl)
                    final = enh.enhance_frame(ai, is_speaking=self._speaking, device=_dev)
                except Exception:
                    final = ai
                t_enh += time.perf_counter() - _t
                # TRADER SCENE (merged AI-trader): the live CHART is the main view and
                # the avatar host sits in a picture-in-picture corner, narrating the
                # market. When the face is lost it falls back to the full chart.
                if getattr(self, "trader_var", None) and self.trader_var.get():
                    final = self._trader_scene(final, chart, self._speaking)
                elif chart_fade > 0.0:    # (classic) crossfade avatar <-> chart
                    cf = chart.render(speaking=self._speaking)
                    final = cv2.addWeighted(final, 1.0 - chart_fade, cf, chart_fade, 0)
                elif getattr(self, "broadcast_var", None) and self.broadcast_var.get():
                    final = self._broadcast_frame(final)   # sharper mouth, no stretch
                final = self._stats_overlay(final)         # likes/coins/goal bar (if live)
                final = self._perf_overlay(final)          # live CPU/GPU/VRAM readout

            last_final = final            # remember for the "generating" hold
            self._last_frame_t = time.monotonic()   # heartbeat for the watchdog
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
                _res = ""
                if self.monitor is not None:
                    _res = (f" | {self.monitor.summary()}"
                            f" | fx:{self.monitor.route_filters()}"
                            + ("  [LIGHT]" if self.monitor.saturated() else ""))
                self._diag = (f"{self._fps:.1f}fps | read {rd:.0f} | LP {lpm:.0f} | "
                              f"gfpgan {gf:.0f} | body {bd:.0f} | enh {en:.0f} ms" + _res)
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

    USER_COOLDOWN = 6.0       # seconds Auto-host yields after you ASK/SPEAK

    def _user_priority(self):
        """Mark that YOU just spoke: pause auto-host + clear its queued backlog so
        your line plays next, not behind auto-host commentary."""
        import time as _t
        self._user_active_until = _t.monotonic() + self.USER_COOLDOWN
        try:
            self.tts.clear_pending()
        except Exception:
            pass

    def _generate(self, prompt):
        """The ONE brain entry point — serialized so two speech sources can never
        generate at once (no GPU clash, no interleaved conversation history)."""
        with self._brain_lock:
            self._thinking = True
            try:
                return self.brain.respond(prompt)
            finally:
                self._thinking = False

    def _speak_text(self, txt):
        """SPEAK box / quick phrases: the avatar says EXACTLY this text."""
        if self.tts is None or not self.running:
            self._log_msg("[studio] press START first.")
            return
        self._user_priority()                 # you take precedence over auto-host
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
        """Generate the in-character answer (serialized via _generate, loop pauses
        via self._thinking), then speak it. Your question takes priority over the
        Auto-host. Runs in a thread so the UI stays responsive."""
        self._log_msg("you> " + txt)
        self._user_priority()                 # pause auto-host + clear its backlog

        def _think():
            reply = None
            try:
                # give the brain the LIVE price so answers reflect the real market
                prompt = (self._live_market_ctx() + " The viewer asks: " + txt).strip()
                reply = self._generate(prompt)   # one-at-a-time brain access
            except Exception as exc:
                self._log_msg(f"[studio] brain error: {exc}")
            if reply:
                self._log_msg("avatar> " + reply)
                self.tts.speak(reply)
            else:
                self._log_msg("[studio] no answer — speaking your text as-is.")
                self.tts.speak(txt)
        threading.Thread(target=_think, daemon=True).start()

    # ----- AUTO-TALK: continuous self-generated gold commentary -----------------
    # MARKET-focused beats — used when gold is actually moving (lean into analysis,
    # levels, reactions). Mixed length so pacing stays unpredictable/human.
    _MARKET_BEATS = [
        "Snap a quick one-line hyped reaction to gold's latest move. Keep it SHORT, one breath.",
        "Drop a single quick gut-reaction word or phrase about gold right now, like you just glanced at the chart.",
        "Quick short call: bullish or bearish on gold right now, in one snappy line.",
        "Give a live update on where gold is trading and what you're watching.",
        "Call out a key support or resistance level on gold and what you'd do around it.",
        "Take your time and go DEEP — read the real technicals out loud (trend, RSI, the exact support and resistance) and walk the chat through your full thinking like a sharp analyst, ramble a bit.",
        "React in the moment to gold's move and tell the chat exactly what level you're watching next and why.",
        "Tell a little story or tangent about trading gold — a lesson, a past move — then bring it back to THIS move today.",
    ]
    # ENGAGEMENT-focused beats — used when the market is quiet (work the chat, CTAs,
    # questions, hype) so there's never dead air.
    _ENGAGE_BEATS = [
        "Fire off a punchy short line telling the chat to smash like. One sentence max.",
        "Push the gift goal — tell viewers to send a rose to unlock the next gold signal.",
        "Really hype the room for a while — build the energy up, talk to the chat, react, go on a passionate rant.",
        "Ask the chat a fun question about gold or their trades, then IMMEDIATELY answer it yourself and keep rolling — never leave a silent pause waiting.",
        "Welcome the room warmly, call out that you see new people coming in, and tell them to hit follow and smash the like.",
        "Tease that a big gold signal is coming up soon and tell them to send a rose to unlock it — build anticipation so nobody leaves.",
        "Banter with the chat — answer the vibe of the room, shout people out, keep it warm and fun.",
    ]
    _AUTOTALK_BEATS = _MARKET_BEATS + _ENGAGE_BEATS    # union (compat)

    def _live_market_ctx(self):
        """Build a REAL-TIME market context string for the brain — the actual live
        gold price + recent % move, fetched at THIS moment. Falls back to the
        simulated chart only if the live feed is down. Also keeps the on-screen
        chart price synced to reality so the visuals match the commentary."""
        md = getattr(self, "market", None)
        price = pct = None
        if md is not None:
            try:
                p = md.price
                if p and p > 0:
                    price = p
                    snap = md.snapshot()
                    if snap is not None and len(snap) > 12:
                        ref = float(snap[-12, 4])
                        pct = (price - ref) / ref * 100.0 if ref else None
                    chart = self.engines.get("chart") if self.engines else None
                    if chart is not None:        # sync the visual chart to reality
                        chart.price = price
            except Exception:
                price = None
        if price is None:                        # live feed down -> simulated chart
            chart = self.engines.get("chart") if self.engines else None
            price = getattr(chart, "price", None)
        if not price:
            return ""
        move = ""
        if pct is not None:
            move = f", {'up' if pct >= 0 else 'down'} {abs(pct):.2f}% in the last few minutes"
        ta = ""
        try:
            import market_ta
            if md is not None:
                ta = market_ta.ta_summary(md.snapshot())
        except Exception:
            ta = ""
        return (f" LIVE RIGHT NOW: gold (XAUUSD) is ${price:,.0f}{move}.{ta} "
                f"Talk about THIS exact current price and the REAL levels, not old ones.")

    def _check_market_alert(self):
        """Detect a SIGNIFICANT live gold event (round-level cross or sharp move) and
        return a brain prompt to announce it live, or None. Tracks price history; rate-
        limited so it doesn't spam."""
        md = getattr(self, "market", None)
        if md is None:
            return None
        try:
            price = float(md.price)
        except Exception:
            return None
        if not price or price <= 0:
            return None
        now = time.monotonic()
        if not hasattr(self, "_ma_hist"):
            import collections as _c
            self._ma_hist = _c.deque(maxlen=60)
            self._ma_last = 0.0
            self._ma_prev = price
        self._ma_hist.append((now, price))
        if now - self._ma_last < 75:        # at most ~1 alert / 75s
            self._ma_prev = price
            return None
        alert = None
        lvl = 25                            # gold "round" levels every $25
        if int(price // lvl) != int(self._ma_prev // lvl):     # crossed a round level
            crossed = (int(price // lvl) * lvl if price > self._ma_prev
                       else (int(price // lvl) + 1) * lvl)
            alert = (f"BREAKING: gold just crossed ${crossed:,} and is now ${price:,.0f}. "
                     "React live to this level break — support, resistance, or breakout? "
                     "One short, energetic sentence.")
        else:                               # sharp % move over ~90s
            ref = next((p for t, p in self._ma_hist if now - t >= 80), None)
            if ref:
                pct = (price - ref) / ref * 100.0
                if abs(pct) >= 0.22:
                    d = "spiking UP" if pct > 0 else "DROPPING"
                    alert = (f"Gold is {d} fast — now ${price:,.0f}, "
                             f"{'+' if pct > 0 else '-'}{abs(pct):.2f}% in minutes. React live to "
                             "this move and what it means for traders. Short and energetic.")
        self._ma_prev = price
        if alert:
            self._ma_last = now
        return alert

    def _market_monitor(self):
        """Background: watch the live price + the economic calendar, queue an alert
        when something big happens (sharp move / level break / imminent news)."""
        import time as _t
        cal = None
        try:
            from econ_calendar import EconCalendar
            cal = EconCalendar()
        except Exception as exc:
            self._log_msg(f"[econ] calendar unavailable: {exc}")
        while getattr(self, "running", False):
            try:
                if (getattr(self, "autotalk_var", None) and self.autotalk_var.get()):
                    alert = self._check_market_alert()
                    if alert:
                        self._event_q.put_nowait(("market", alert))
                    if cal is not None:                  # imminent high-impact news
                        news = cal.next_alert()
                        if news:
                            self._event_q.put_nowait(("market", news))
                    self._poll_tick()                    # buy/sell poll lifecycle
            except Exception:
                pass
            _t.sleep(8)

    def _watchdog(self):
        """Keep the stream alive unattended: if the render thread DIES, restart it; if
        it STALLS (no new frame for a while), log it. Bounded restarts so a hard fault
        doesn't loop forever."""
        import time as _t
        self._wd_restarts = getattr(self, "_wd_restarts", 0)
        _t.sleep(15)                       # let the first boot settle
        while getattr(self, "running", False):
            _t.sleep(5)
            try:
                if not getattr(self, "running", False):
                    break
                w = getattr(self, "_worker", None)
                if w is not None and not w.is_alive():        # render thread crashed
                    if self._wd_restarts < 5:
                        self._wd_restarts += 1
                        self._log_msg(f"[watchdog] render thread died — restarting "
                                      f"({self._wd_restarts}/5)…")
                        self._last_frame_t = _t.monotonic()
                        self._worker = threading.Thread(target=self._loop, daemon=True)
                        self._worker.start()
                    else:
                        self._log_msg("[watchdog] too many restarts — needs a manual look.")
                        _t.sleep(60)
                else:
                    lt = getattr(self, "_last_frame_t", 0)
                    if lt and (_t.monotonic() - lt) > 20:     # alive but frozen
                        self._log_msg("[watchdog] render stalled >20s (feed/GPU hiccup).")
                        self._last_frame_t = _t.monotonic()   # don't spam the log
            except Exception:
                pass

    def _market_active(self):
        """Is gold MOVING right now? True if the recent price range is a meaningful
        % of price (so the host leans into market talk; otherwise it works the chat).
        Uses the price history the market monitor keeps. Tunable threshold."""
        try:
            hist = getattr(self, "_ma_hist", None)
            if hist and len(hist) >= 4:
                recent = [p for (_, p) in list(hist)[-6:]]
                avg = sum(recent) / len(recent)
                rng = (max(recent) - min(recent)) / avg if avg else 0.0
                thr = float(os.environ.get("AVATAR_MARKET_ACTIVE_PCT", "0.0012"))
                return rng >= thr
        except Exception:
            pass
        return False

    def _autotalk_loop(self):
        """Background host: the brain writes gold commentary and the Arabic-accent TTS
        speaks it (mouth lip-syncs). PIPELINED — the brain generates the NEXT line
        WHILE the current one is synthesizing/playing, so voice generation never
        pauses the LLM. A 1-line look-ahead keeps it bounded so commentary stays fresh."""
        import time as _t
        # Look-ahead = 2 so a NEXT line is always synthesized and ready the instant
        # the current one ends — no dead air between lines (streamers never leave
        # silence). The music bed covers any micro-gap while a line generates.
        LEAD = 2
        i = 0
        while getattr(self, "running", False):
            try:
                tts = self.tts
                if (not getattr(self, "autotalk_var", None) or not self.autotalk_var.get()
                        or self.brain is None or not self.brain.ok or tts is None):
                    _t.sleep(0.4)
                    continue
                # YIELD to the user: if you just ASKed/SPEAKed, stay quiet so the
                # avatar answers YOU and doesn't talk over it.
                if _t.monotonic() < self._user_active_until:
                    _t.sleep(0.3)
                    continue
                # pace to the look-ahead — NOT blocked by the voice; the brain just
                # doesn't run more than LEAD lines ahead of playback.
                if tts.pending > LEAD:
                    _t.sleep(0.2)
                    continue
                # PRIORITY 1: react to gifts / follows / shares / like-milestones and
                # BREAKING market moves (the monitor queues those) — always first.
                if self._react_one_event():
                    continue
                # ADAPTIVE BALANCE: when gold is MOVING, focus on the market (analysis
                # beats, only glance at the chat occasionally); when it's QUIET, work
                # the COMMENT SECTION + engagement so there's never dead air.
                active = self._market_active()
                self._mix = getattr(self, "_mix", 0) + 1
                if active:
                    pool = self._MARKET_BEATS
                    # still answer the chat, but sparingly (~1 in 4) so market leads
                    if self._mix % 4 == 0 and self._answer_one_comment():
                        continue
                else:
                    pool = self._ENGAGE_BEATS
                    # quiet market -> viewers come first, answer comments eagerly
                    if self._answer_one_comment():
                        continue
                beat = pool[i % len(pool)]; i += 1
                ctx = self._live_market_ctx()    # REAL gold price + recent move, fetched NOW
                line = None
                try:
                    line = self._generate(beat + ctx)   # ONE-at-a-time brain access
                except Exception as exc:
                    self._log_msg(f"[autotalk] brain: {exc}")
                # if you interacted while it was generating, drop this line
                if line and self.autotalk_var.get() and _t.monotonic() >= self._user_active_until:
                    self._log_msg("avatar> " + line)
                    self.tts.speak(line)
            except Exception as exc:
                self._log_msg(f"[autotalk] {exc}")
                _t.sleep(2.0)

    def _on_comment(self, user, text):
        """Called from the TikTok reader thread for every live comment — queue it
        (or count it as a poll vote if a buy/sell poll is running)."""
        try:
            self._feed_msg(f"{user}:  {text}", "q")     # show ALL comments live
            poll = self._poll
            if poll is not None:
                t = (text or "").strip().lower()
                if t in ("1", "buy", "long", "buy gold", "bull"):
                    poll["buy"] += 1; return
                if t in ("2", "sell", "short", "sell gold", "bear"):
                    poll["sell"] += 1; return
            self._comment_q.put_nowait((user, text))
        except Exception:
            pass        # queue full = drop (we're behind on a comment flood)

    def _poll_tick(self):
        """Start a buy/sell poll periodically while live; close it + announce the result."""
        try:
            now = time.monotonic()
            if self._poll is None:
                if (self.tiktok is not None and now - self._poll_last > 300
                        and getattr(self, "autotalk_var", None) and self.autotalk_var.get()):
                    self._poll = {"buy": 0, "sell": 0, "end": now + 45}
                    self._poll_last = now
                    self._event_q.put_nowait(("poll_start",))
            elif now >= self._poll["end"]:
                b, s = self._poll["buy"], self._poll["sell"]
                self._poll = None
                self._event_q.put_nowait(("poll_result", b, s))
        except Exception:
            pass

    # --- LIVE EVENT handlers (gifts / follows / likes / shares) -------------
    def _on_gift(self, user, gift, count, coins):
        try:
            self._log_msg(f"🎁 {user} sent {count}x {gift} ({coins} coins)")
            self._feed_msg(f"\U0001f381 {user} sent {count}x {gift} ({coins} coins)", "ev")
            self._sess_coins += max(0, int(coins))
            self._event_q.put_nowait(("gift", user, gift, count, coins))
            if self._sess_coins >= self._coin_goal:           # gift goal reached!
                reached = self._coin_goal
                self._coin_goal += int(os.environ.get("AVATAR_COIN_GOAL", "200"))
                self._event_q.put_nowait(("goal", reached))
        except Exception:
            pass

    def _on_follow(self, user):
        try:
            self._sess_follows += 1
            self._feed_msg(f"➕ {user} followed", "ev")
            self._event_q.put_nowait(("follow", user))
        except Exception:
            pass

    def _on_share(self, user):
        try:
            self._feed_msg(f"↪ {user} shared the stream", "ev")
            self._event_q.put_nowait(("share", user))
        except Exception:
            pass

    def _on_like(self, user, total):
        # celebrate only when we CROSS a milestone (likes fire constantly otherwise)
        try:
            self._sess_likes = max(self._sess_likes, int(total or 0))
            if total and total >= self._next_like_ms:
                self._event_q.put_nowait(("likes", total))
                step = 500 if total < 1000 else (1000 if total < 10000 else 5000)
                self._next_like_ms = ((total // step) + 1) * step
        except Exception:
            pass

    def _react_one_event(self):
        """Speak the next live-event reaction (gift/follow/share/likes). Gifts come
        first. Returns True if the avatar spoke (loop skips its scripted line)."""
        try:
            if self._event_q.empty() or self.tts is None:
                return False
            ev = self._event_q.get_nowait()
            kind = ev[0]
            reply = None
            if kind == "market":                   # autonomous — no responder needed
                reply = self._generate(ev[1])      # brain phrases the live alert
            elif kind == "goal":                   # gift-goal reached celebration
                reply = (self.responder.react_goal(ev[1]) if self.responder is not None
                         else f"We just smashed the {ev[1]}-coin goal — thank you all!")
            elif kind == "poll_start":
                reply = ("Quick poll, fam — are you BUYING or SELLING gold right now? "
                         "Comment 1 for buy, 2 for sell, let's see the chat!")
            elif kind == "poll_result":
                b, s = ev[1], ev[2]; tot = b + s
                if tot == 0:
                    reply = "Nobody voted that round — next time hit 1 for buy, 2 for sell!"
                else:
                    bp = round(b / tot * 100)
                    lead = "BUYING" if b >= s else "SELLING"
                    reply = (f"Poll's in — the chat is {lead} gold! Buy {bp}%, sell {100 - bp}%, "
                             f"{tot} votes. Let's trade it.")
            elif self.responder is not None:
                if kind == "gift":
                    reply = self.responder.react_gift(ev[1], ev[2], ev[3], ev[4])
                elif kind == "follow":
                    reply = self.responder.react_follow(ev[1])
                elif kind == "share":
                    reply = self.responder.react_share(ev[1])
                elif kind == "likes":
                    reply = self.responder.react_likes(ev[1])
            if not reply:
                return False
            if reply:
                self._log_msg(f"avatar→{kind}> {reply}")
                self.tts.speak(reply)
                return True
            return False
        except Exception as exc:
            self._log_msg(f"[events] {exc}")
            return False

    def _answer_one_comment(self):
        """Pop the next live comment; if it's worth answering, speak the answer.
        Returns True if the avatar spoke (so the loop skips its scripted line)."""
        try:
            # lazily build the responder once the brain finished loading (the reader
            # may have started before START was pressed).
            if self.responder is None and self.brain is not None:
                try:
                    from comment_responder import CommentResponder
                    self.responder = CommentResponder(self.brain, get_context=self._live_market_ctx)
                except Exception:
                    pass
            if self.responder is None or self._comment_q.empty() or self.tts is None:
                return False
            # drain a few at once but only answer ONE per cycle (keeps it fresh, not
            # a backlog read-out); newer comments matter more so take the latest.
            user, text = None, None
            while not self._comment_q.empty():
                user, text = self._comment_q.get_nowait()
            reply = self.responder.respond(user, text)     # filter + research + answer
            if reply:
                self._log_msg(f"↳ {user}: {text}")
                self._log_msg(f"avatar→{user}> {reply}")
                self._feed_msg(f"\U0001f916 → {user}:  {reply}", "a")
                self.tts.speak(reply)
                return True
            return False
        except Exception as exc:
            self._log_msg(f"[comments] {exc}")
            return False

    def _feed_msg(self, text, kind="sys"):
        """Append one line to the live-comments feed below the avatar. Thread-safe.
        kind: 'q' viewer comment, 'a' avatar reply, 'ev' gift/follow, 'sys' note."""
        def _apply():
            try:
                self.feed.configure(state="normal")
                self.feed.insert("end", text + "\n", kind)
                if int(self.feed.index("end-1c").split(".")[0]) > 250:
                    self.feed.delete("1.0", "80.0")     # cap scrollback
                self.feed.see("end")
                self.feed.configure(state="disabled")
            except Exception:
                pass
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _set_live_light(self, state):
        """Drive the live-status dot. state: 'live' (green), 'off' (red),
        'none' (grey, no handle). Thread-safe (marshalled onto the Tk thread)."""
        dot, glow = {
            "live": ("#27ff9e", "#0c4a32"),   # bright green + dark green halo
            "off":  ("#ff3b5c", "#4a0c1a"),   # red + dark red halo
            "none": ("#3a3f4a", ""),          # grey, no halo
        }.get(state, ("#3a3f4a", ""))
        word, wcol = {
            "live": ("LIVE", "#27ff9e"),
            "off":  ("offline", "#ff3b5c"),
            "none": ("no handle", MUTED),
        }.get(state, ("no handle", MUTED))

        def _apply():
            try:
                self.live_light.itemconfig(self._live_dot, fill=dot)
                self.live_light.itemconfig(self._live_glow, fill=glow)
            except Exception:
                pass
            try:                                  # mirror onto the docked feed panel
                self.feed_light.itemconfig(self._feed_dot, fill=dot)
                self.feed_status.configure(text=word, fg=wcol)
            except Exception:
                pass
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _live_status_loop(self):
        """ALWAYS-ON: continuously poll TikTok for whether the entered @handle is
        LIVE and drive the green/red light in real time. When the handle goes live
        and 'Answer live comments' is ticked, auto-connects the reader so it never
        misses the start of a stream. Cheap is_live() check, no long sleeps."""
        import asyncio
        import time as _t
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
        except Exception:
            pass
        last = None
        client = None
        client_handle = None
        while not getattr(self, "_live_stop", False):
            handle = (self.handle_var.get() or "").strip()
            if not handle:
                if last is not None:
                    self._set_live_light("none")
                    last = None
                _t.sleep(1.2)
                continue
            if not handle.startswith("@"):
                handle = "@" + handle
            try:
                from TikTokLive import TikTokLiveClient
                # reuse ONE client per handle so we don't leak an HTTP session
                # on every poll over a multi-hour stream.
                if client is None or handle != client_handle:
                    client = TikTokLiveClient(unique_id=handle)
                    client_handle = handle
                live = bool(asyncio.get_event_loop().run_until_complete(
                    client.is_live()))
            except Exception:
                live = False
            self._handle_live = live
            self._set_live_light("live" if live else "off")
            if live != last:
                self._log_msg(f"[live] {handle} is "
                              + ("LIVE \U0001f7e2 — comments incoming" if live
                                 else "offline \U0001f534"))
                last = live
            # auto-connect the comment reader the moment the stream goes live
            if (live and self.comments_var.get() and self.tiktok is None
                    and self.brain is not None):
                try:
                    self.root.after(0, self._on_comments)
                except Exception:
                    pass
            _t.sleep(3.0)

    def _on_comments(self):
        """Toggle the live TikTok comment responder on/off."""
        try:
            if not self.comments_var.get():
                if self.tiktok is not None:
                    self.tiktok.stop(); self.tiktok = None
                self._log_msg("[comments] off")
                return
            handle = (self.handle_var.get() or "").strip()
            if not handle:
                self._log_msg("[comments] enter your TikTok @handle first.")
                self.comments_var.set(False); return
            if not handle.startswith("@"):
                handle = "@" + handle
            from comment_responder import CommentResponder
            from tiktok_comments import TikTokComments
            # The reader can run without the brain (it just READS comments); answers
            # begin as soon as the brain is loaded (responder is created lazily).
            if self.responder is None and self.brain is not None:
                self.responder = CommentResponder(self.brain, get_context=self._live_market_ctx)
            if self.tiktok is None:
                self.tiktok = TikTokComments(handle, self._on_comment,
                                             on_gift=self._on_gift, on_follow=self._on_follow,
                                             on_like=self._on_like, on_share=self._on_share)
                self.tiktok.start()
            note = "" if self.brain is not None else " (answers begin after START)"
            self._log_msg(f"[comments] reading {handle} — comments + gifts/follows{note}")
        except Exception as exc:
            self._log_msg(f"[comments] failed: {exc}")
            self.comments_var.set(False)

    def _stats_overlay(self, fr):
        """Top strip: session LIKES / COINS + a gift-goal progress bar. Only shown
        while connected to a TikTok live (so it doesn't clutter solo use)."""
        if self.tiktok is None:
            return fr
        try:
            S = FRAME_SIZE
            likes, coins, goal = self._sess_likes, self._sess_coins, self._coin_goal
            prog = min(1.0, coins / max(1, goal))
            ov = fr.copy()
            cv2.rectangle(ov, (0, 0), (S, 28), (10, 12, 16), -1)
            fr = cv2.addWeighted(ov, 0.5, fr, 0.5, 0)
            cv2.putText(fr, f"LIKES {likes:,}    COINS {coins:,}    GOAL {coins}/{goal}",
                        (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 235, 255), 1, cv2.LINE_AA)
            cv2.rectangle(fr, (10, 25), (S - 10, 27), (40, 48, 60), -1)
            cv2.rectangle(fr, (10, 25), (10 + int((S - 20) * prog), 27), (0, 215, 255), -1)
            return fr
        except Exception:
            return fr

    def _perf_overlay(self, fr):
        """Small top-right CPU / GPU / VRAM readout (live load), colour-coded green→
        amber→red. Lets you watch the resource governor working on stream."""
        if self.monitor is None or not getattr(self, "perf_var", None) or not self.perf_var.get():
            return fr
        try:
            S = FRAME_SIZE
            rows = [("CPU", self.monitor.cpu), ("GPU", self.monitor.gpu),
                    ("VRAM", self.monitor.vram)]

            def col(v):
                return (110, 240, 130) if v < 70 else ((60, 200, 255) if v < 88 else (80, 80, 255))
            x = S - 132
            ov = fr.copy()
            cv2.rectangle(ov, (x - 8, 6), (S - 4, 58), (10, 12, 16), -1)
            fr = cv2.addWeighted(ov, 0.55, fr, 0.45, 0)
            for i, (lbl, v) in enumerate(rows):
                yy = 20 + i * 14
                c = col(v)
                cv2.putText(fr, f"{lbl:4}", (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (170, 190, 210), 1, cv2.LINE_AA)
                bx = x + 36
                cv2.rectangle(fr, (bx, yy - 8), (bx + 56, yy - 2), (40, 44, 54), -1)
                cv2.rectangle(fr, (bx, yy - 8), (bx + int(56 * min(1.0, v / 100.0)), yy - 2), c, -1)
                cv2.putText(fr, f"{v:3.0f}%", (bx + 60, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            c, 1, cv2.LINE_AA)
            if self.monitor.saturated():
                cv2.putText(fr, "BALANCING", (x, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (60, 200, 255), 1, cv2.LINE_AA)
            return fr
        except Exception:
            return fr

    def _broadcast_frame(self, fr, scale=0.82):
        """BROADCAST framing: render the avatar at a natural size on a soft blur of
        itself. This shows the (96px) lip-sync mouth NEAR 1:1 instead of stretched
        across a full-screen face — so the talking mouth reads SHARP, with no GPU lag.
        Also a cleaner streaming look."""
        try:
            S = FRAME_SIZE
            small = cv2.resize(fr, (96, 96))
            small = cv2.GaussianBlur(small, (0, 0), 5)
            bg = cv2.resize(small, (S, S))
            bg = (bg.astype(np.float32) * 0.5).astype(np.uint8)   # darken the backdrop
            w = max(64, min(S, int(S * scale)))
            av = cv2.resize(fr, (w, w))
            x = (S - w) // 2
            y = min(S - w, int(S * 0.05))
            bg[y:y + w, x:x + w] = av
            return bg
        except Exception:
            return fr

    def _trader_scene(self, avatar, chart, speaking):
        """Merged trading stream: the live CHART fills the frame and the avatar host
        sits in a picture-in-picture corner narrating the market (folds the ai_trader
        concept into one app)."""
        try:
            base = chart.render(speaking=speaking)
            if base.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
                base = cv2.resize(base, (FRAME_SIZE, FRAME_SIZE))
            pw = int(FRAME_SIZE * 0.40)                 # square PiP, bottom-right
            pip = cv2.resize(avatar, (pw, pw))
            m = 14
            x2 = FRAME_SIZE - m; x1 = x2 - pw
            y2 = FRAME_SIZE - m; y1 = y2 - pw
            cv2.rectangle(base, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (18, 18, 18), -1)
            cv2.rectangle(base, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 215, 255), 2)
            base[y1:y2, x1:x2] = pip
            return base
        except Exception:
            return avatar

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
        heavy = key in ("maya1", "chatterbox", "multilingual")
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
        self._restore_every_base = p.get("restore_every", 2)   # loop doubles it while talking
        if self.engines and "restore" in self.engines:
            try:
                self.engines["restore"].every_n = self._restore_every_base
            except Exception:
                pass
        self._log_msg(f"[studio] quality: {self.quality_var.get()} "
                      f"(LP every {p['lp']}, enhance {p['enhance']}, "
                      f"body {'on' if p['body'] else 'off'}, "
                      f"restore every {p.get('restore_every', 2)})")

    def _on_stab(self):
        lvl = self.stab_var.get() / 100.0
        if self.engines:
            try:
                self.engines["lp"].set_stabilization(lvl)
            except Exception:
                pass
        if self.swap_engine is not None:          # also stabilize the face-swap
            try:
                self.swap_engine.set_stabilization(lvl)
            except Exception:
                pass

    def _on_gaze(self):
        if self.engines:
            try:
                self.engines["lp"].set_gaze(self.gaze_var.get(),
                                            self.gaze_var2.get() / 100.0)
            except Exception:
                pass

    def _on_music(self):
        if self.music is not None:
            on = bool(self.music_var.get())
            self.music.set_active(on and self.running)
            self._log_msg("[studio] background music " + ("ON" if on else "off"))
        self._sync_music_btn()

    def _toggle_music(self):
        """Top mute button: flip the background music on/off."""
        if getattr(self, "music_var", None) is None:
            return
        self.music_var.set(not self.music_var.get())
        self._on_music()

    def _toggle_speech(self):
        """Top mute button: silence the bot's VOICE (lips keep moving)."""
        if self.tts is None:
            self._log_msg("[studio] press START first.")
            return
        self.tts.set_muted(not self.tts.muted)
        muted = self.tts.muted
        self.speech_btn.configure(text="🔇 SPEECH" if muted else "🎤 SPEECH",
                                  fg=RED if muted else MAG,
                                  highlightbackground=self._mix(RED if muted else MAG, BG, 0.5))
        self._log_msg("[studio] bot speech " + ("MUTED" if muted else "on"))

    def _run_autoconfig(self):
        """Background: probe + benchmark the machine, then apply the picked config
        and reveal it in the top bar (the loading bar runs meanwhile)."""
        global AUTO_PROFILE
        try:
            from auto_config import detect, choose, apply
            res = detect()                 # imports torch + runs the GPU benchmark
            cfg = choose(res)
            env = apply(cfg)               # sets AVATAR_* env (setdefault)
            AUTO_PROFILE = {"res": res, "cfg": cfg, "env": env}
        except Exception as exc:
            AUTO_PROFILE = None
            print(f"[AUTO-CONFIG] failed ({exc}).")
        try:
            self.root.after(0, self._autoconfig_done)
        except Exception:
            pass

    def _autoconfig_done(self):
        """On the UI thread: stop the loading bar, show the chosen model, set the
        voice dropdown to the auto pick, and ENABLE START."""
        try:
            self.bench_bar.stop()
            self.bench_bar.pack_forget()
        except Exception:
            pass
        self.info_lbl.configure(text=self._autocfg_text(), fg=MUTED)
        if AUTO_PROFILE:
            r, cfg = AUTO_PROFILE["res"], AUTO_PROFILE["cfg"]
            lbl = next((l for l, k in VOICE_MODES if k == cfg["tts"]), None)
            if lbl:
                try:
                    self.voicemode_var.set(lbl)
                except Exception:
                    pass
            self._log_msg(f"[auto-config] {r['gpu']} · {r['vram_free']:.1f}GB free "
                          f"· {r['tflops']:.0f} TFLOP/s")
            self._log_msg("[auto-config] -> " + cfg["why"])
        try:
            self.start_btn.configure(state="normal")
        except Exception:
            pass

    def _autocfg_text(self):
        """Top readout: chosen LLM brain + GPU benchmark."""
        try:
            if AUTO_PROFILE:
                b = AUTO_PROFILE["cfg"]["brain"]
                tf = AUTO_PROFILE["res"]["tflops"]
                return f"🧠 {b}   ⚡ {tf:.0f} TFLOP/s"
        except Exception:
            pass
        return "🧠 " + os.environ.get("AVATAR_BRAIN_MODEL", "?")

    def _update_info(self):
        """Refresh the readout with the ACTUAL loaded brain (it may differ from the
        auto-pick if that model wasn't pulled and the brain fell back)."""
        txt = self._autocfg_text()
        try:
            if self.brain is not None and getattr(self.brain, "model", None):
                tf = AUTO_PROFILE["res"]["tflops"] if AUTO_PROFILE else 0
                txt = f"🧠 {self.brain.model}" + (f"   ⚡ {tf:.0f} TFLOP/s" if tf else "")
            self.info_lbl.configure(text=txt)
        except Exception:
            pass

    def _sync_music_btn(self):
        """Make the top button reflect the current music state."""
        if getattr(self, "music_btn", None) is None:
            return
        on = bool(getattr(self, "music_var", None) and self.music_var.get())
        try:
            if on:
                self.music_btn.configure(text="♪ MUSIC", fg=CYAN,
                                         highlightbackground=self._mix(CYAN, BG, 0.5))
            else:
                self.music_btn.configure(text="🔇 MUTED", fg=RED,
                                         highlightbackground=self._mix(RED, BG, 0.5))
        except Exception:
            pass

    def _on_liplock(self):
        if self.engines:
            try:
                self.engines["lp"].set_lip_lock(self.liplock_var.get())
                self._log_msg("[studio] lips: "
                              + ("BOT VOICE only (your mouth ignored)"
                                 if self.liplock_var.get() else "follow your webcam mouth"))
            except Exception:
                pass

    def _on_swap(self):
        # FACE-SWAP shows YOUR real head/mouth — mutually exclusive with bot-only
        # lips. Turning it on disables the lip-lock.
        if self.swap_var.get() and getattr(self, "liplock_var", None) and self.liplock_var.get():
            self.liplock_var.set(False)
            self._log_msg("[studio] FACE-SWAP on — shows YOUR real mouth "
                          "(lip-lock off). Untick it for bot-only lips.")

    def _on_autotalk(self):
        on = bool(self.autotalk_var.get())
        if on and (self.brain is None or not getattr(self.brain, "ok", False)):
            self._log_msg("[studio] auto-talk needs the AI brain — it'll start once ready.")
        self._log_msg("[studio] auto-talk " + ("ON — bot hosts by itself" if on else "off"))

    def _on_minface(self):
        if self.engines:
            try:
                self.engines["lp"].min_good_face = max(0.04, self.minface_var.get() / 100.0)
            except Exception:
                pass

    def _on_character(self, *args):
        """Switch the face-swap character identity live."""
        folder = {"White Haddan": "haddan_white", "Haddan": "Haddan",
                  "White man": "character_src"}.get(self.char_var.get())
        if not folder or self.swap_engine is None:
            return
        d = os.path.join(PROJECT_DIR, folder)
        if not os.path.isdir(d):
            self._log_msg(f"[studio] character folder missing: {folder}")
            return
        try:
            n = self.swap_engine.set_source_from_folder(d)
            self.swap_engine._lock_emb = None      # re-lock onto the new look
            self._log_msg(f"[studio] character -> {self.char_var.get()} ({n} photos)")
        except Exception as exc:
            self._log_msg(f"[studio] character switch failed: {exc}")

    def _on_hair(self, *args):
        """Set the hair/beard recolour target live."""
        if self.swap_engine is not None:
            self.swap_engine._hair_color = self.hair_var.get()
            self._log_msg(f"[studio] hair colour -> {self.hair_var.get()}")

    def _on_eye(self, *args):
        """Set the iris recolour target live."""
        if self.swap_engine is not None:
            self.swap_engine._eye_color = self.eye_var.get()
            self._log_msg(f"[studio] eye colour -> {self.eye_var.get()}")

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
        self._live_stop = True
        if self.tiktok is not None:
            try:
                self.tiktok.stop()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        for fn in (lambda: self.cap.release() if self.cap else None,
                   lambda: self.obs_cam.close() if self.obs_cam else None,
                   lambda: self._tv_proc.terminate() if self._tv_proc else None,
                   lambda: self.tts.shutdown() if self.tts else None,
                   lambda: self.music.stop() if self.music else None):
            try:
                fn()
            except Exception:
                pass
        self.root.destroy()


_SINGLE_INSTANCE_HANDLE = None


def _acquire_single_instance():
    """Allow only ONE Avatar Studio at a time (so only one bot can speak). Uses a
    named Windows mutex — held for the process lifetime, released automatically when
    the process exits. Returns False if another instance already holds it."""
    global _SINGLE_INSTANCE_HANDLE
    try:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        h = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\AvatarStudioSingleInstance")
        if not h or ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        _SINGLE_INSTANCE_HANDLE = h        # keep the handle alive
        return True
    except Exception:
        return True                         # non-Windows / failure: don't block


def main():
    if not _acquire_single_instance():
        print("[studio] Avatar Studio is ALREADY RUNNING — only one instance is "
              "allowed (so only one bot speaks). Exiting this one.")
        return
    root = tk.Tk()
    AvatarStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
