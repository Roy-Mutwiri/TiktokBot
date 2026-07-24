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
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageTk

from realtime_avatar import _character_path, _open_webcam, FRAME_SIZE, FPS
from tts_stream_engine import MALE_VOICES

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PREVIEW_SIZE = 512
PREVIEW_W = 390
PREVIEW_H = 844
TIKTOK_PORTRAIT_W = 1080
TIKTOK_PORTRAIT_H = 2340
TIKTOK_CHART_H = 1380
TIKTOK_FACE_H = TIKTOK_PORTRAIT_H - TIKTOK_CHART_H
STREAMER_FACE_CROP = (0.745, 0.61, 0.25, 0.36)
STREAMER_CHART_CROP = (0.0, 0.04, 0.72, 0.84)
FACE_STRIP_PRESETS = {
    "Short face only": 660,
    "Medium face only": 780,
    "Tall face only": TIKTOK_FACE_H,
}
FACE_STRIP_LABELS = list(FACE_STRIP_PRESETS.keys())
DEFAULT_FACE_STRIP_LABEL = "Tall face only"
TARGET_FRAME_TIME = 1.0 / FPS
APP_USER_MODEL_ID = "TiktokBot.AvatarStudio"
APP_ICON_PATH = os.path.join(PROJECT_DIR, "assets", "avatar_studio.ico")
APP_ICON_PNG_PATH = os.path.join(PROJECT_DIR, "assets", "avatar_studio.png")
APP_ICON_FALLBACK = os.path.join(PROJECT_DIR, "haddan_white", "wh_00.png")

# Face-loss -> trading-chart scene
NO_FACE_SECONDS = 3.0          # no face for this long -> switch to charts
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
    ("Fluent live - Kokoro", "kokoro"),
    ("Arabic + English · XTTS-v2 (cloned)", "xtts"),
]
VOICE_MODE_LABELS = [m[0] for m in VOICE_MODES]
VOICE_MODE_KEY = dict(VOICE_MODES)
YOUTUBE_VOICE_PERSONAS = [
    ("Marcus - deep male", "deep_male"),
    ("Omar - warm male", "warm_male"),
    ("Ethan - young male", "young_male"),
    ("David - broadcast male", "broadcast_male"),
    ("Layla - natural woman", "natural_woman"),
    ("Sofia - warm woman", "warm_woman"),
    ("Maya - bright woman", "bright_woman"),
    ("Nora - low woman", "low_woman"),
]
YOUTUBE_PERSONA_KEY = dict(YOUTUBE_VOICE_PERSONAS)
YOUTUBE_PERSONA_LABELS = [item[0] for item in YOUTUBE_VOICE_PERSONAS]

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
BG       = "#050608"      # near-black application canvas
SURFACE  = "#0c0f15"      # raised panel face
SURFACE2 = "#080a0f"      # recessed fields / entries
BORDER   = "#202632"      # neutral hairline borders
FG       = "#f4f6fb"      # high-contrast primary text
MUTED    = "#8a93a5"      # secondary text / control labels
FAINT    = "#50596b"      # captions / inactive chrome

CYAN     = "#58dff8"      # primary interface accent
CYAN_HI  = "#a3efff"
CYAN_INK = "#03171c"
MAG      = "#ff4fa3"      # interaction / comment accent
MAG_HI   = "#ff91c7"
MINT     = "#4df0b5"      # live / go
MINT_HI  = "#9affd8"
MINT_INK = "#02160e"
AMBER    = "#f5b84b"      # warnings / recenter
RED      = "#ff526b"      # stopped / error

# Windows Segoe MDL2 icon glyphs. Stored as escapes so the source stays portable
# while the UI gets real app-style icons on Windows.
ICONS = {
    "dashboard": "\ue80f", "live": "\ue768", "comments": "\ue8bd",
    "voice": "\ue720", "face": "\ue8b2", "lips": "\ue9d9",
    "scenes": "\ue7c3", "analytics": "\ue9d2", "settings": "\ue713",
    "play": "\ue768", "pause": "\ue769", "mute": "\ue74f",
    "record": "\ue7c8", "stop": "\ue71a", "gear": "\ue713",
    "preview": "\ue714", "cpu": "\ue950", "gpu": "\ue7f4",
    "ram": "\ue964", "clock": "\ue823", "heart": "\ue00b",
    "viewers": "\ue716", "link": "\ue71b", "mic": "\ue720",
}

def _configure_windows_app_identity():
    """Give the studio its own Windows taskbar group instead of Python's."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:
        pass


def _set_window_icon(root):
    """Set both the native Windows icon and Tk's window icon."""
    try:
        if os.path.exists(APP_ICON_PATH):
            root.iconbitmap(default=APP_ICON_PATH)
    except Exception:
        pass

    image_path = (
        APP_ICON_PNG_PATH
        if os.path.exists(APP_ICON_PNG_PATH)
        else APP_ICON_FALLBACK
    )
    try:
        if os.path.exists(image_path):
            root._avatar_taskbar_icon = tk.PhotoImage(file=image_path)
            root.iconphoto(True, root._avatar_taskbar_icon)
    except Exception:
        pass


def _show_frameless_window_in_taskbar(root):
    """Make a Tk overrideredirect window a normal Windows taskbar app."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        gwl_exstyle = -20
        ws_ex_toolwindow = 0x00000080
        ws_ex_appwindow = 0x00040000
        style = ctypes.windll.user32.GetWindowLongW(hwnd, gwl_exstyle)
        style = (style & ~ws_ex_toolwindow) | ws_ex_appwindow
        ctypes.windll.user32.SetWindowLongW(hwnd, gwl_exstyle, style)

        # Windows refreshes taskbar registration when the top-level is remapped.
        root.withdraw()
        root.after(10, root.deiconify)
    except Exception:
        pass


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
        self.live_mic = None                 # LiveVoiceEngine when Live-Mic mode is on
        self.ai_mouth_var = tk.BooleanVar(value=True)
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
        self.brain_pool = None               # parallel commentary prefetch pool
        self.tiktok = None                   # LIVE TikTok comment reader
        self._handles_file = os.path.join(PROJECT_DIR, "tiktok_handles.json")
        self._handles = self._load_handles()  # remembered @handles for the dropdown
        self._handle_text = os.environ.get("AVATAR_TIKTOK_USER", "")
        self.responder = None                # comment filter + answerer (web research)
        self._comment_q = queue.Queue(maxsize=80)
        self.comment_voice_var = tk.BooleanVar(
            value=os.environ.get("AVATAR_COMMENT_READER_VOICE", "0") == "1")
        self._comment_voice_q = queue.Queue(maxsize=12)
        self._comment_voice_event = threading.Event()
        self._comment_voice_last_t = 0.0
        self._comment_voice_seen = {}
        self._event_q = queue.Queue(maxsize=40)   # market alerts / polls (may use the LLM)
        import collections as _c
        # TOP-PRIORITY instant reactions (follows/gifts/shares/likes/goals) — already
        # built as ready-to-speak text (offline templates, no LLM), spoken the moment
        # the loop ticks, jumping ahead of filler commentary. Bounded so a gift flood
        # doesn't pile up (keep the most recent).
        self._prio_events = _c.deque(maxlen=10)
        # follows arrive in BURSTS — buffer the names and thank them in one batched
        # line (so a burst is one quick shout-out, not 8 slow separate ones).
        self._pending_follows = _c.deque(maxlen=60)
        self._follow_batch_after = None
        self._follow_batch_first_t = 0.0
        self._ready_speech_lock = threading.Lock()
        self._ready_speech = None
        self._ready_speech_deferred = None
        self._ready_speech_slots = {"urgent": None, "comment": None}
        self._ready_speech_deferred_slots = {"urgent": None, "comment": None}
        self._ready_speech_token = 0
        self._live_response_event = threading.Event()
        self._comment_times = _c.deque(maxlen=1000)
        self._next_like_ms = 500                  # next likes milestone to celebrate
        # SESSION STATS + gift goal (on-screen bar + CTAs)
        self._sess_viewers = None
        self._sess_likes = 0
        self._sess_coins = 0
        self._sess_follows = 0
        self._viewer_scores = {}
        self._session_started_at = None
        self._coin_goal = int(os.environ.get("AVATAR_COIN_GOAL", "200"))
        self._poll = None                         # active buy/sell poll {buy,sell,end}
        self._poll_last = 0.0                     # last poll start (monotonic)
        self.market = None                   # currently active live market feed
        self.market_gold = None              # PAXG proxy while XAUUSD is open
        self.market_btc = None               # BTCUSDT, active when gold is closed
        self._market_symbol = None
        self._market_transition_announced = None
        self._thinking = False               # True while the brain is generating
        self.cap = None
        self.camera_enabled = True
        self._camera_lock = threading.RLock()
        self.obs_cam = None
        self._tv_proc = None                # the AI-driven TradingView browser
        self.lp_interval = 2
        self._char_path = None               # chosen character image (any face)

        self._latest = None                 # latest final frame (BGR ndarray)
        self._latest_serial = 0
        self._drawn_serial = -1
        self._last_preview_draw_t = 0.0
        self._last_streamer_face_frame = None
        self._frame_lock = threading.Lock()
        self._log_q = queue.Queue()
        self._fps = 0.0
        self._diag = ""                      # per-stage ms readout
        self._speaking = False
        self._youtube_busy = False
        self._youtube_mode = "market"         # "market" | "youtube"
        self._youtube_chunks = []
        self._youtube_title = ""
        self._youtube_index = 0
        self._youtube_pump_started = False
        self._youtube_audio = None
        self._youtube_audio_mode = False
        self._youtube_audio_status = ""
        self._youtube_muted = False
        self._youtube_queue_urls = []
        self._youtube_queue_slots = []
        self._youtube_queue_index = 0
        self._youtube_queue_active = False
        self._youtube_queue_advancing = False
        self._youtube_queue_prewarmed = set()
        self._youtube_queue_prewarm_started = False
        self._youtube_queue_signature = ()
        self._youtube_prewarm_wait_started = False
        self.youtube_status_vars = []
        self.youtube_smooth_var = tk.BooleanVar(value=True)
        self.face_strip_var = tk.StringVar(value=DEFAULT_FACE_STRIP_LABEL)
        self.low_lag_scene_var = tk.BooleanVar(value=True)
        self._mic_monitor_muted = False
        self._audio_meters = {}
        self._audio_meter_levels = {}
        self._youtube_mohammed_mode = False
        self._youtube_start_seconds = 0.0
        self._youtube_end_seconds = None
        self._youtube_duration = 0.0
        self._youtube_scene_clock_started_at = None
        self._youtube_clock_position = 0.0
        self._youtube_clock_anchor_t = None
        self._youtube_progress_value = 0.0
        self._youtube_progress_text = "Idle"
        self._youtube_link_notice = ""
        self._youtube_link_notice_color = AMBER
        self._youtube_link_notice_until = 0.0
        self._scene_capture_image = None
        self._scene_capture_bgr = None
        self._scene_display_image = None
        self._scene_capture_tk = None
        self._scene_capture_bbox = None
        self._scene_window_hwnd = None
        self._scene_window_crop = None
        self._scene_window_title = ""
        self._scene_window_capture_method = None
        self._scene_preview_job = None
        self._scene_capture_lock = threading.Lock()
        self._scene_capture_stop = threading.Event()
        self._scene_capture_thread = None
        self._scene_capture_serial = 0
        self._scene_rendered_serial = -1
        self._scene_preview_size = (640, 210)
        self._scene_source = None
        self._youtube_scene = None
        self._youtube_scene_url = ""
        self._youtube_scene_crop = (0.0, 0.0, 1.0, 1.0)
        self._youtube_scene_attach_job = None
        self._youtube_scene_raw_image = None
        self._scene_text_enabled = False
        self._scene_text = ""
        self._scene_text_font = "Segoe UI"
        self._scene_text_size = 72
        self._scene_text_color = "#ffffff"
        self._scene_text_bg = "#000000"
        self._scene_text_outline = "#00e5ff"
        self._scene_text_behavior = "Static"
        self._scene_text_position = "Top"
        self._scene_text_opacity = 74
        self._scene_text_items = []
        self._scene_text_active_index = 0
        self.scene_slot = None
        self.scene_preview_lbl = None
        self.scene_add_btn = None
        self.scene_edit_btn = None
        self.scene_reset_btn = None
        self.scene_time_lbl = None
        self.scene_face_strip = None
        self._active_face_variant = 1
        self._face_variant_buttons = {}
        self._scene_face_box = None
        self._scene_face_detect_count = 0
        self._scene_face_detector = None
        self._avatar_face_detector = None
        self._character_face_cache = None
        self._audio_handoff_lock = threading.Lock()
        self._audio_handoff_token = 0
        self._audio_handoff_state = None
        self._ready_playback_active = False
        self._mohammed_voice_ready = False
        self._mohammed_voice_warming = False
        self._worker = None
        self._render_recovery_until = 0.0
        self._render_stall_count = 0
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
        _set_window_icon(root)
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        # Match the large center tile in Windows 11's 25/50/25 snap layout.
        # Keep enough width for the control rails on smaller displays, while
        # using nearly the full work-area height for the stacked live monitors.
        win_w = min(sw - 32, max(1500, int(sw * 0.50)))
        win_h = max(780, sh - 70)
        _wx = max(0, (sw - win_w) // 2); _wy = max(0, (sh - win_h) // 2)
        root.geometry(f"{win_w}x{win_h}+{_wx}+{_wy}")
        root.minsize(900, 600)
        root.resizable(True, True)
        # frameless window -> our own futuristic title bar with custom controls
        self._drag = None
        self._resize_drag = None
        self._resize_handles = []
        self._tb_buttons = {}
        self._tb_hover = None
        try:
            root.overrideredirect(True)
            root.bind("<Map>", self._restore_override)
            root.after(20, lambda: _show_frameless_window_in_taskbar(root))
            root.after(80, lambda: (root.lift(), root.focus_force()))
        except Exception:
            pass

        self._init_style()
        self._build_ui()
        self._install_resize_handles()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()                     # start the UI refresh loop
        self._animate()                     # start the HUD animation loop
        # ALWAYS-ON live-status monitor: from the moment the app opens it keeps
        # checking whether the entered @handle is LIVE on TikTok and drives the
        # green/red light by the SPEECH button. No START needed, never sleeps long.
        self._live_stop = False
        self._safe_thread("_live_status_loop")
        self._safe_thread("_live_response_loop")
        self._safe_thread("_comment_reader_voice_loop")
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

    def _safe_thread(self, method_name, *args):
        """Start a daemon thread for self.<method_name>, but NEVER let a missing/
        broken method (e.g. an autosync mid-write) crash startup. Returns the
        thread or None."""
        try:
            fn = getattr(self, method_name, None)
            if fn is None:
                print(f"[studio] optional thread {method_name} unavailable — skipping.")
                return None
            t = threading.Thread(target=fn, args=args, daemon=True)
            t.start()
            return t
        except Exception as exc:
            print(f"[studio] could not start {method_name}: {exc}")
            return None

    def _launch_tradingview(self, symbol=None):
        """Open TradingView.com in a browser the AI drives (scroll/zoom/draw)."""
        import subprocess
        if os.environ.get("AVATAR_TV", "1") == "0":
            return
        if bool(getattr(self, "low_lag_scene_var", None)
                and self.low_lag_scene_var.get()):
            self._log_msg("[studio] TradingView pilot skipped in low-lag scene mode.")
            return
        proj = os.path.dirname(os.path.abspath(__file__))
        if symbol is None:
            try:
                from market_session import xauusd_session
                symbol = "OANDA:XAUUSD" if xauusd_session().is_open else "BINANCE:BTCUSDT"
            except Exception:
                symbol = "OANDA:XAUUSD"
        sym = os.environ.get("AVATAR_TV_SYMBOL", symbol)
        lang = os.environ.get("AVATAR_TTS_LANG", "en")
        if getattr(sys, "frozen", False):
            args = [sys.executable, "--tradingview-pilot"]
        else:
            args = [sys.executable, os.path.join(proj, "tradingview_pilot.py")]
        args.extend([
            "--symbol", sym, "--lang", lang if lang in ("en", "ar") else "en"
        ])
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
        b._normal_bg = bg
        b._normal_border = border
        b._hover_bg = hover
        b._hover_border = hover_border

        def en(_):
            if str(b["state"]) != "disabled":
                b.configure(bg=getattr(b, "_hover_bg", hover),
                            highlightbackground=getattr(b, "_hover_border", hover_border))

        def lv(_):
            if str(b["state"]) != "disabled":
                b.configure(bg=getattr(b, "_normal_bg", bg),
                            highlightbackground=getattr(b, "_normal_border", border))
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
        try:
            ready = self._ready_speech_snapshot()
            if ready and ready.get("status") in ("preparing", "ready"):
                self._topdraw()
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
        if h == "camera":
            self.toggle_camera(); return
        if h == "ready_speech":
            self._play_ready_speech(); return
        if h == "ready_thanks_speak":
            self._play_ready_speech("urgent"); return
        if h == "ready_thanks_skip":
            self._skip_ready_speech("urgent"); return
        if h == "ready_comment_speak":
            self._play_ready_speech("comment"); return
        if h == "ready_comment_skip":
            self._skip_ready_speech("comment"); return
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
                self.root.after_idle(self._raise_resize_handles)
        except Exception:
            pass

    def _install_resize_handles(self):
        """Add draggable edges and corners to the frameless main window."""
        specs = (
            ("n",  0, 0, "nw", 1, 6, "sb_v_double_arrow"),
            ("s",  0, 1, "sw", 1, 6, "sb_v_double_arrow"),
            ("w",  0, 0, "nw", 6, 1, "sb_h_double_arrow"),
            ("e",  1, 0, "ne", 6, 1, "sb_h_double_arrow"),
            ("nw", 0, 0, "nw", 14, 14, "size_nw_se"),
            ("ne", 1, 0, "ne", 14, 14, "size_ne_sw"),
            ("sw", 0, 1, "sw", 14, 14, "size_ne_sw"),
            ("se", 1, 1, "se", 14, 14, "size_nw_se"),
        )
        for edge, relx, rely, anchor, width, height, cursor in specs:
            handle = tk.Frame(self.root, bg=BG, cursor=cursor, bd=0)
            options = {"relx": relx, "rely": rely, "anchor": anchor,
                       "width": width, "height": height}
            if edge in ("n", "s"):
                options["relwidth"] = 1
            elif edge in ("w", "e"):
                options["relheight"] = 1
            handle.place(**options)
            handle.bind(
                "<ButtonPress-1>",
                lambda event, side=edge: self._begin_window_resize(event, side))
            handle.bind("<B1-Motion>", self._resize_window)
            handle.bind("<ButtonRelease-1>", self._end_window_resize)
            self._resize_handles.append(handle)
        self.root.bind("<Configure>", lambda _event: self._raise_resize_handles(),
                       add="+")
        self._raise_resize_handles()

    def _raise_resize_handles(self):
        for handle in self._resize_handles:
            try:
                handle.lift()
            except Exception:
                pass

    def _begin_window_resize(self, event, edge):
        self._resize_drag = (
            edge, event.x_root, event.y_root,
            self.root.winfo_x(), self.root.winfo_y(),
            self.root.winfo_width(), self.root.winfo_height(),
        )

    def _resize_window(self, event):
        if self._resize_drag is None:
            return
        edge, start_x, start_y, x, y, width, height = self._resize_drag
        dx, dy = event.x_root - start_x, event.y_root - start_y
        min_width, min_height = self.root.minsize()
        new_x, new_y, new_width, new_height = x, y, width, height

        if "e" in edge:
            new_width = max(min_width, width + dx)
        if "s" in edge:
            new_height = max(min_height, height + dy)
        if "w" in edge:
            new_width = max(min_width, width - dx)
            new_x = x + width - new_width
        if "n" in edge:
            new_height = max(min_height, height - dy)
            new_y = y + height - new_height

        self.root.geometry(
            f"{new_width}x{new_height}+{new_x}+{new_y}")

    def _end_window_resize(self, _event=None):
        self._resize_drag = None

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
        self.youtube_smooth_btn = tk.Button(
            ph, text="SMOOTH VOICE", command=self._toggle_youtube_smooth,
            bg=self._mix(SURFACE2, MINT, 0.18), fg=MINT,
            font=("Consolas", 10, "bold"), relief="flat", bd=0,
            padx=10, cursor="hand2",
            activebackground=self._mix(SURFACE2, MINT, 0.28),
            activeforeground=MINT, highlightthickness=1,
            highlightbackground=self._mix(MINT, BG, 0.45))
        self.youtube_smooth_btn.pack(side="right", padx=(0, 10))
        # Switch between voice-driven lip-sync and the avatar's untouched mouth.
        self.mouth_btn = tk.Button(ph, text="AI MOUTH", command=self._toggle_ai_mouth,
                                   bg=SURFACE2, fg=MINT, font=("Consolas", 10, "bold"),
                                   relief="flat", bd=0, padx=10, cursor="hand2",
                                   activebackground=self._mix(SURFACE2, MINT, 0.2),
                                   activeforeground=MINT, highlightthickness=1,
                                   highlightbackground=self._mix(MINT, BG, 0.5))
        self.mouth_btn.pack(side="right", padx=(0, 10))
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
        self.root.after(1000, self._youtube_clock_tick)

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
        # REFRESH (reconnect the reader + re-check live) and RESET (clear the feed)
        tk.Button(ch, text="↻ Refresh", command=self._on_comments_refresh,
                  bg=SURFACE2, fg=CYAN, font=("Consolas", 8, "bold"), relief="flat",
                  bd=0, padx=8, cursor="hand2", highlightthickness=1,
                  activebackground=self._mix(SURFACE2, CYAN, 0.25), activeforeground=CYAN,
                  highlightbackground=self._mix(CYAN, BG, 0.5)).pack(side="left", padx=(12, 0))
        tk.Button(ch, text="⟲ Reset", command=self._on_comments_reset,
                  bg=SURFACE2, fg=AMBER, font=("Consolas", 8, "bold"), relief="flat",
                  bd=0, padx=8, cursor="hand2", highlightthickness=1,
                  activebackground=self._mix(SURFACE2, AMBER, 0.25), activeforeground=AMBER,
                  highlightbackground=self._mix(AMBER, BG, 0.5)).pack(side="left", padx=(6, 0))
        # @handle entry + Answer toggle on the right
        self.comments_var = tk.BooleanVar(value=True)
        self._check(ch, "Answer", self.comments_var,
                    self._on_comments).pack(side="right")
        self._check(ch, "Voice", self.comment_voice_var,
                    self._on_comment_voice).pack(side="right", padx=(0, 10))
        self.handle_var = tk.StringVar(value=os.environ.get("AVATAR_TIKTOK_USER", ""))
        # editable DROPDOWN: type a new @handle, or click the arrow to pick one you've
        # used before (remembered across sessions, each handle listed once).
        self.handle_combo = ttk.Combobox(ch, textvariable=self.handle_var, width=14,
                                         values=self._handles, style="Studio.TCombobox",
                                         font=("Segoe UI", 10))
        self.handle_combo.pack(side="right", padx=(0, 8), ipady=2)
        # picking from the dropdown drops it into the blank handle space + re-checks live
        self.handle_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._on_handle_pick())
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
        # let the wheel scroll the FEED (and not the right rail) while hovering it
        def _feed_wheel(e):
            self.feed.yview_scroll(int(-(e.delta or 0) / 120), "units")
            return "break"
        self.feed.bind("<MouseWheel>", _feed_wheel)
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
        self.minface_var = tk.IntVar(value=6)
        ttk.Spinbox(r, from_=4, to=40, increment=2, width=5,
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
        # fairer-skin tone: shifts ALL visible skin toward a lighter Caucasian tone
        # (skin-gated, won't touch shirt/beard/eyes/bg). 0 = source tone, 100 = fairest.
        r = self._row(c, "Skin tone (fairer)")
        self.skintone_var = tk.IntVar(value=50)
        ttk.Scale(r, from_=0, to=100, variable=self.skintone_var, length=150,
                  style="Studio.Horizontal.TScale",
                  command=lambda e: self._on_skintone()).pack(side="right")

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
        from trading_backgrounds import BACKGROUND_PRESETS
        self.background_on_var = tk.BooleanVar(value=True)
        self._check(c, "Trading background", self.background_on_var,
                    self._on_background_toggle).pack(fill="x", pady=3)
        r = self._row(c, "Background preset")
        self.background_var = tk.StringVar(value="Wall Street LED / Midnight Blue")
        self.background_combo = ttk.Combobox(
            r, textvariable=self.background_var, values=BACKGROUND_PRESETS,
            state="readonly", width=31, style="Studio.TCombobox")
        self.background_combo.pack(side="right")
        self.background_var.trace_add("write", self._on_background)
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

        # ---- LIVE MIC (voice changer) -------------------------------------
        # YOU talk into the mic -> voice changer -> the avatar's mouth syncs to
        # it (instead of the typed/AI voice). Coexists with the AI voice: turning
        # it ON mutes the AI host so the two never fight for the mouth.
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(2, 6))
        self.livemic_var = tk.BooleanVar(value=False)
        self._check(c, "Live Mic — speak as the avatar (voice changer)",
                    self.livemic_var, self._on_live_mic).pack(fill="x", pady=2)
        try:
            from voice_changer_engine import list_input_devices
            _mics = [f"{i}: {n}" for i, n in list_input_devices()]
        except Exception:
            _mics = []
        tk.Label(c, text="Mic input", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.micdev_var = tk.StringVar(value=(_mics[0] if _mics else "default"))
        ttk.Combobox(c, textvariable=self.micdev_var,
                     values=(_mics or ["default"]), state="readonly",
                     style="Studio.TCombobox").pack(fill="x", pady=(3, 6))
        self.micdev_var.trace_add("write", self._on_micdev)
        tk.Label(c, text="Voice changer", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        # label -> converter key (see voice_changer_engine.make_converter)
        self.VC_MODES = [("Persona voice (RVC)", "rvc"),
                         ("Pitch / formant (DSP)", "dsp"),
                         ("Passthrough (no change)", "passthrough")]
        self.vcmode_var = tk.StringVar(value=self.VC_MODES[0][0])
        ttk.Combobox(c, textvariable=self.vcmode_var,
                     values=[m[0] for m in self.VC_MODES], state="readonly",
                     style="Studio.TCombobox").pack(fill="x", pady=(3, 9))
        self.vcmode_var.trace_add("write", self._on_vcmode)
        # Mic boost: quiet mics (e.g. the webcam mic) need a software gain so
        # normal speech clears the gate and the monitor is audible. ~5x suits the
        # Logi C270 webcam mic; a close headset mic wants ~1x.
        tk.Label(c, text="Mic boost", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.micgain_var = tk.DoubleVar(value=float(os.environ.get("AVATAR_MIC_GAIN", "5.0")))
        tk.Scale(c, from_=1.0, to=12.0, resolution=0.5, orient="horizontal",
                 variable=self.micgain_var, command=self._on_micgain,
                 bg=SURFACE, fg=FG, troughcolor=SURFACE2, highlightthickness=0,
                 bd=0, font=("Segoe UI", 8)).pack(fill="x", pady=(0, 9))

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

        # ---- YOUTUBE SPEAK -------------------------------------------------
        c = self._card(right, "YOUTUBE SPEAK")
        tk.Label(c, text="Add YouTube links in order. Video 2 starts when Video 1 ends.",
                 bg=SURFACE, fg=FAINT, font=("Segoe UI", 8),
                 wraplength=300, justify="left").pack(anchor="w", pady=(0, 4))
        self.youtube_entries = []
        self.youtube_status_vars = []
        for i in range(10):
            item = tk.Frame(c, bg=SURFACE)
            item.pack(fill="x", pady=(0, 5))
            row = tk.Frame(item, bg=SURFACE)
            row.pack(fill="x")
            tk.Label(row, text=f"YOUTUBE {i + 1}", bg=SURFACE, fg=FAINT,
                     font=("Consolas", 8, "bold"), width=10,
                     anchor="w").pack(side="left")
            entry = tk.Entry(
                row, bg=SURFACE2, fg=FG, insertbackground=AMBER,
                font=("Segoe UI", 9), relief="flat",
                highlightthickness=1, highlightbackground=BORDER,
                highlightcolor=AMBER)
            entry.pack(side="left", fill="x", expand=True, ipady=4)
            entry.bind("<Return>", self._on_youtube_enter)
            entry.bind("<<Paste>>", self._on_youtube_link_changed)
            entry.bind("<KeyRelease>", self._on_youtube_link_changed)
            self.youtube_entries.append(entry)
            status_var = tk.StringVar(value="EMPTY")
            self.youtube_status_vars.append(status_var)
            tk.Label(
                item, textvariable=status_var, bg=SURFACE, fg=FAINT,
                font=("Consolas", 7), anchor="w").pack(
                    fill="x", padx=(78, 0), pady=(1, 0))
        self.youtube_entry = self.youtube_entries[0]
        self.youtube_persona_var = tk.StringVar(value=YOUTUBE_PERSONA_LABELS[0])
        persona_combo = ttk.Combobox(
            c, textvariable=self.youtube_persona_var,
            values=YOUTUBE_PERSONA_LABELS, state="readonly",
            style="Studio.TCombobox",
        )
        persona_combo.pack(fill="x", pady=(0, 7))
        persona_combo.bind("<<ComboboxSelected>>", self._on_youtube_persona_change)
        range_row = tk.Frame(c, bg=SURFACE); range_row.pack(fill="x", pady=(0, 7))
        tk.Label(range_row, text="FROM", bg=SURFACE, fg=FAINT,
                 font=("Consolas", 8, "bold")).pack(side="left")
        self.youtube_from_var = tk.StringVar(value="")
        tk.Entry(range_row, textvariable=self.youtube_from_var, width=8,
                 bg=SURFACE2, fg=FG, insertbackground=AMBER, relief="flat",
                 font=("Consolas", 9)).pack(side="left", padx=(5, 10), ipady=3)
        tk.Label(range_row, text="TO", bg=SURFACE, fg=FAINT,
                 font=("Consolas", 8, "bold")).pack(side="left")
        self.youtube_to_var = tk.StringVar(value="")
        tk.Entry(range_row, textvariable=self.youtube_to_var, width=8,
                 bg=SURFACE2, fg=FG, insertbackground=AMBER, relief="flat",
                 font=("Consolas", 9)).pack(side="left", padx=(5, 8), ipady=3)
        tk.Label(range_row, text="min or mm:ss", bg=SURFACE, fg=FAINT,
                 font=("Consolas", 8)).pack(side="left")
        self.youtube_btn = self._btn(
            c, "SPEAK YOUTUBE", self.speak_youtube, bg=AMBER, fg=CYAN_INK,
            hover=self._mix(AMBER, "#ffffff", 0.18), border=AMBER,
            hover_border="#ffffff", font=("Consolas", 10, "bold"),
            state="disabled")
        self.youtube_btn.pack(fill="x", ipady=6)
        self.youtube_audio_btn = self._btn(
            c, "ALTER REAL YOUTUBE VOICE", self.speak_youtube_audio, bg=SURFACE2, fg=AMBER,
            hover=self._mix(SURFACE2, AMBER, 0.2), border=AMBER,
            hover_border="#ffffff", font=("Consolas", 10, "bold"),
            state="disabled")
        self.youtube_audio_btn.pack(fill="x", pady=(6, 0), ipady=6)
        jump_row = tk.Frame(c, bg=SURFACE); jump_row.pack(fill="x", pady=(6, 0))
        self.youtube_back_btn = self._btn(
            jump_row, "BACK VIDEO", self.youtube_previous_video,
            bg=SURFACE2, fg=AMBER, hover=self._mix(SURFACE2, AMBER, 0.2),
            border=AMBER, hover_border="#ffffff",
            font=("Consolas", 9, "bold"), state="disabled")
        self.youtube_back_btn.pack(side="left", fill="x", expand=True, ipady=5)
        self.youtube_next_btn = self._btn(
            jump_row, "NEXT VIDEO", self.youtube_next_video,
            bg=SURFACE2, fg=AMBER, hover=self._mix(SURFACE2, AMBER, 0.2),
            border=AMBER, hover_border="#ffffff",
            font=("Consolas", 9, "bold"), state="disabled")
        self.youtube_next_btn.pack(side="left", fill="x", expand=True,
                                   padx=(8, 0), ipady=5)
        ymode = tk.Frame(c, bg=SURFACE); ymode.pack(fill="x", pady=(7, 0))
        self.youtube_light = tk.Canvas(ymode, width=18, height=18, bg=SURFACE,
                                       highlightthickness=0)
        self.youtube_light_dot = self.youtube_light.create_oval(
            4, 4, 14, 14, fill="#3a3f4a", outline="")
        self.youtube_light.pack(side="left", padx=(0, 6))
        self.youtube_status_lbl = tk.Label(
            ymode, text="MARKET MODE", bg=SURFACE, fg=MUTED,
            font=("Consolas", 9, "bold"))
        self.youtube_status_lbl.pack(side="left")
        self.youtube_time_lbl = tk.Label(
            c, text="YOUTUBE TIME 00:00", bg=SURFACE, fg=FAINT,
            font=("Consolas", 9))
        self.youtube_time_lbl.pack(anchor="w", pady=(5, 0))
        ybtns = tk.Frame(c, bg=SURFACE); ybtns.pack(fill="x", pady=(7, 0))
        self.youtube_resume_btn = self._btn(
            ybtns, "YOUTUBE", self.resume_youtube, bg=SURFACE2, fg=AMBER,
            hover=self._mix(SURFACE2, AMBER, 0.2), border=AMBER,
            hover_border="#ffffff", font=("Consolas", 9, "bold"),
            state="disabled")
        self.youtube_resume_btn.pack(side="left", fill="x", expand=True, ipady=5)
        self.market_mode_btn = self._btn(
            ybtns, "MARKET", self.resume_market, bg=SURFACE2, fg=MINT,
            hover=self._mix(SURFACE2, MINT, 0.2), border=MINT,
            hover_border="#ffffff", font=("Consolas", 9, "bold"),
            state="disabled")
        self.market_mode_btn.pack(side="left", fill="x", expand=True,
                                  padx=(7, 0), ipady=5)
        self._sync_youtube_buttons()

        # ---- ACTIVITY LOG --------------------------------------------------
        c = self._card(right, "ACTIVITY LOG")
        self.log = tk.Text(c, height=8, bg=SURFACE2, fg=MUTED, relief="flat",
                           font=("Consolas", 8), wrap="word", state="disabled",
                           padx=8, pady=6, highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=BORDER)
        self.log.pack(fill="both", expand=True)

    # -------------------------------------------------------------------------
    # POLISHED DASHBOARD UI (overrides the older cockpit layout above)
    # -------------------------------------------------------------------------
    def _dash_panel(self, parent, title, accent=CYAN, *, fill="both", expand=False,
                    padx=0, pady=0):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill=fill, expand=expand, padx=padx, pady=pady)
        cv = tk.Canvas(outer, bg=BG, height=80, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)
        body = tk.Frame(cv, bg=SURFACE)
        win = cv.create_window(14, 44, anchor="nw", window=body)
        body._dash_outer = outer
        body._dash_canvas = cv
        icon_map = {
            "Comment Reader": ICONS["comments"], "Live Preview": ICONS["preview"],
            "AI Voice": ICONS["voice"], "Face Swap": ICONS["face"],
            "Lip Sync": ICONS["mic"], "Stream Output": ICONS["link"],
            "Session & Performance": ICONS["cpu"], "Realism": ICONS["face"],
            "Scene & Output": ICONS["scenes"], "Ask The Avatar": ICONS["comments"],
            "Speak": ICONS["voice"], "YouTube Speak": ICONS["play"],
            "Activity Log": ICONS["analytics"],
        }

        def redraw(_=None):
            w, h = cv.winfo_width(), cv.winfo_height()
            if w <= 2 or h <= 2:
                return
            if not expand:
                body.update_idletasks()
                want_h = max(80, body.winfo_reqheight() + 56)
                if abs(want_h - h) > 2:
                    cv.configure(height=want_h)
                    h = want_h
            cv.delete("panel")
            # One-pixel drop shadow and neutral shell. Accent is reserved for a
            # short top rail and the module icon, matching the MotionSites style.
            self._round_rect(
                cv, 2, 3, w - 1, h - 1, 8, fill="#020305",
                outline="", tags="panel")
            self._round_rect(
                cv, 1, 1, w - 2, h - 3, 8, fill=SURFACE,
                outline=BORDER, width=1, tags="panel")
            cv.create_rectangle(
                2, 2, w - 3, 37, fill="#0e1219", outline="", tags="panel")
            cv.create_rectangle(
                14, 1, min(w - 18, 76), 2, fill=accent,
                outline="", tags="panel")
            cv.create_line(
                14, 38, w - 14, 38, fill="#171c26", width=1, tags="panel")
            self._round_rect(
                cv, 13, 10, 31, 28, 5,
                fill=self._mix(SURFACE2, accent, 0.12),
                outline=self._mix(BORDER, accent, 0.28), width=1, tags="panel")
            cv.create_text(
                22, 19, text=icon_map.get(title, ICONS["settings"]),
                fill=accent, font=("Segoe MDL2 Assets", 9), tags="panel")
            cv.create_text(
                39, 19, text=title, anchor="w", fill=FG,
                font=("Segoe UI", 9, "bold"), tags="panel")
            cv.create_oval(
                w - 23, 16, w - 17, 22,
                fill=self._mix(SURFACE, accent, 0.65), outline="", tags="panel")
            cv.itemconfigure(win, width=max(40, w - 28), height=max(20, h - 56))
            cv.tag_lower("panel")

        cv.bind("<Configure>", redraw)
        body.bind("<Configure>", redraw)
        return body

    def _metric_tile(self, parent, label, value, change="", accent=CYAN):
        if not hasattr(self, "_metric_value_labels"):
            self._metric_value_labels = {}
            self._metric_change_labels = {}
            self._metric_tiles = {}
            self._metric_sparks = {}
            self._metric_accents = {}
            self._metric_targets = {}
            self._metric_anim_tokens = {}
        metric_icons = {
            "Viewers": ICONS["viewers"], "Likes": ICONS["heart"],
            "Comments / min": ICONS["comments"], "CPU": ICONS["cpu"],
            "GPU": ICONS["gpu"], "VRAM": ICONS["ram"], "Uptime": ICONS["clock"],
        }
        f = tk.Frame(
            parent, bg="#090c11", highlightthickness=1,
            highlightbackground="#151a23")
        header = tk.Frame(f, bg="#090c11"); header.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(header, text=metric_icons.get(label, ICONS["analytics"]),
                 bg="#090c11", fg=accent, font=("Segoe MDL2 Assets", 9)).pack(side="left")
        tk.Label(header, text=label, bg="#090c11", fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(5, 0))
        row = tk.Frame(f, bg="#090c11"); row.pack(fill="x", padx=10)
        value_lbl = tk.Label(row, text=value, bg="#090c11", fg=FG, font=("Segoe UI", 13, "bold"))
        value_lbl.pack(side="left")
        self._metric_value_labels[label] = value_lbl
        change_lbl = tk.Label(
            row, text=change, bg="#090c11", fg=accent,
            font=("Segoe UI", 8, "bold"))
        change_lbl.pack(side="left", padx=(8, 0))
        self._metric_change_labels[label] = change_lbl
        mini = tk.Canvas(f, height=18, bg="#090c11", highlightthickness=0)
        mini.pack(fill="x", padx=10, pady=(3, 7))
        pts = [(0, 15), (12, 11), (24, 14), (36, 7), (48, 12), (60, 5), (72, 9),
               (84, 3), (96, 8), (108, 4), (120, 10)]
        spark_items = []
        for i in range(len(pts) - 1):
            spark_items.append(mini.create_line(
                *pts[i], *pts[i + 1], fill=accent, width=1))
        self._metric_tiles[label] = f
        self._metric_sparks[label] = (mini, spark_items, pts)
        self._metric_accents[label] = accent
        self._metric_targets[label] = str(value)
        return f

    def _mini_row(self, parent, label, value, color=FG):
        if not hasattr(self, "_row_value_labels"):
            self._row_value_labels = {}
        r = tk.Frame(parent, bg=SURFACE)
        r.pack(fill="x", pady=3)
        tk.Label(r, text=label, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        value_lbl = tk.Label(r, text=value, bg=SURFACE, fg=color, font=("Segoe UI", 8, "bold"))
        value_lbl.pack(side="right")
        self._row_value_labels[label] = value_lbl
        return r

    def _range_row(self, parent, label, var, command=None, frm=0, to=100):
        r = tk.Frame(parent, bg=SURFACE)
        r.pack(fill="x", pady=4)
        tk.Label(r, text=label, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        ttk.Scale(r, from_=frm, to=to, variable=var, length=135,
                  style="Studio.Horizontal.TScale", command=command).pack(side="right")
        return r

    def _thumb(self, parent, label, accent, live=False):
        wrap = tk.Frame(parent, bg=SURFACE)
        cv = tk.Canvas(wrap, width=72, height=72, bg=SURFACE2, highlightthickness=1,
                       highlightbackground=self._mix(BORDER, accent, 0.35))
        cv.pack()
        cv.create_rectangle(0, 0, 72, 72, fill=self._mix(SURFACE2, accent, 0.12), outline="")
        cv.create_oval(23, 12, 49, 38, fill=self._mix("#f1c7b7", accent, 0.08), outline="")
        cv.create_arc(18, 20, 54, 64, start=20, extent=140, fill="#171018", outline="")
        cv.create_oval(28, 25, 31, 28, fill="#151515", outline="")
        cv.create_oval(41, 25, 44, 28, fill="#151515", outline="")
        cv.create_line(32, 43, 42, 43, fill=MAG, width=2)
        if live:
            cv.create_oval(56, 56, 64, 64, fill=MINT, outline="")
        tk.Label(wrap, text=label, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(pady=(4, 0))
        return wrap

    def _bind_mousewheel_tree(self, widget, callback):
        widget.bind("<MouseWheel>", callback, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel_tree(child, callback)

    def _flash_widget(self, widget, color=None):
        color = color or CYAN
        try:
            old = widget.cget("highlightbackground")
            widget.configure(highlightthickness=2, highlightbackground=color)
            self.root.after(650, lambda: widget.configure(highlightthickness=1,
                                                          highlightbackground=old))
        except Exception:
            pass

    def _set_nav_active(self, label):
        for name, parts in getattr(self, "_nav_items", {}).items():
            active = name == label
            bg = "#11151d" if active else "#07090d"
            fg = MAG if active else FG
            icon_fg = MAG if active else MUTED
            for w in parts.get("widgets", ()):
                try:
                    w.configure(bg=bg)
                except Exception:
                    pass
            try:
                parts["icon"].configure(bg=bg, fg=icon_fg)
                parts["label"].configure(bg=bg, fg=fg)
                parts["rail"].configure(bg=MAG if active else bg)
                parts["frame"].configure(highlightthickness=1 if active else 0,
                                         highlightbackground=self._mix(MAG, BG, 0.35))
            except Exception:
                pass

    def _scroll_right_to(self, target):
        canvas = getattr(self, "_right_canvas", None)
        frame = getattr(self, "_right_frame", None)
        if canvas is None or frame is None or target is None:
            return
        self.root.update_idletasks()
        outer = getattr(target, "_dash_outer", target)
        total = canvas.bbox("all")
        if not total:
            return
        y = max(0, outer.winfo_y() - 4)
        span = max(1, total[3] - total[1])
        canvas.yview_moveto(min(1.0, y / span))
        self._flash_widget(outer, MAG)

    def _nav_go(self, label):
        self._set_nav_active(label)
        targets = getattr(self, "_nav_targets", {})
        if label == "Dashboard":
            try:
                self._right_canvas.yview_moveto(0)
            except Exception:
                pass
            self._flash_widget(getattr(self, "preview_stage_wrap", self.root), CYAN)
            return
        if label == "Comments":
            try:
                self.handle_combo.focus_set()
            except Exception:
                pass
            self._flash_widget(getattr(self, "_comments_outer", self.root), MAG)
            return
        target = targets.get(label)
        if target is not None:
            self._scroll_right_to(target)

    def _build_sidebar_scene_slot(self):
        if self.scene_slot is None:
            return
        self.scene_preview_lbl = None
        for child in self.scene_slot.winfo_children():
            child.destroy()
        holder = tk.Frame(
            self.scene_slot, bg="#0c1016", highlightthickness=1,
            highlightbackground=self._mix(BORDER, CYAN, 0.42))
        holder.pack(fill="both", expand=True)
        header = tk.Frame(holder, bg="#0c1016")
        header.pack(fill="x", padx=9, pady=(7, 5))
        tk.Label(
            header, text="LIVE SCENE", bg="#0c1016",
            fg=CYAN if self._scene_capture_image is not None else MUTED,
            font=("Segoe UI", 8, "bold")).pack(side="left")
        self.scene_time_lbl = tk.Label(
            header, text=self._scene_time_text(), bg="#0c1016", fg=FAINT,
            font=("Consolas", 7))
        self.scene_time_lbl.pack(side="left", padx=(8, 0))
        self.scene_add_btn = tk.Button(
            header,
            text="Screen Scene" if self._scene_capture_image is not None else "Add Scene",
            command=self._add_scene,
            bg=self._mix(CYAN, "#1d7cff", 0.35), fg="#ffffff",
            activebackground=self._mix(CYAN, "#ffffff", 0.20),
            activeforeground="#ffffff", relief="flat", bd=0,
            font=("Segoe UI", 8, "bold"), cursor="hand2",
            highlightthickness=1, highlightbackground=CYAN, padx=10, pady=4)
        self.scene_add_btn.pack(side="right")
        has_text_overlay = self._scene_text_overlay_active()
        self.scene_text_btn = tk.Button(
            header, text="T", command=self._open_scene_text_overlay_editor,
            bg=self._mix(SURFACE2, CYAN, 0.18),
            fg=CYAN if has_text_overlay else MUTED,
            activebackground=self._mix(SURFACE2, CYAN, 0.30),
            activeforeground=CYAN, relief="flat", bd=0,
            font=("Segoe UI", 9, "bold"), cursor="hand2",
            highlightthickness=1,
            highlightbackground=CYAN if has_text_overlay else BORDER,
            padx=9, pady=4)
        self.scene_text_btn.pack(side="right", padx=(0, 6))
        if self._scene_source == "youtube":
            self.scene_reset_btn = tk.Button(
                header, text="Reset Video", command=self._reset_youtube_scene_video,
                bg=self._mix(SURFACE2, RED, 0.18), fg=RED,
                activebackground=self._mix(SURFACE2, RED, 0.30),
                activeforeground=RED, relief="flat", bd=0,
                font=("Segoe UI", 8, "bold"), cursor="hand2",
                highlightthickness=1, highlightbackground=RED,
                padx=9, pady=4)
            self.scene_reset_btn.pack(side="right", padx=(0, 6))
            self.scene_edit_btn = tk.Button(
                header, text="Edit Crop", command=self._edit_youtube_scene_crop,
                bg=self._mix(SURFACE2, AMBER, 0.22), fg=AMBER,
                activebackground=self._mix(SURFACE2, AMBER, 0.34),
                activeforeground=AMBER, relief="flat", bd=0,
                font=("Segoe UI", 8, "bold"), cursor="hand2",
                highlightthickness=1, highlightbackground=AMBER,
                padx=9, pady=4)
            self.scene_edit_btn.pack(side="right", padx=(0, 6))
        if self._scene_capture_image is not None:
            preview_row = tk.Frame(holder, bg="#050608")
            preview_row.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.scene_face_strip = tk.Frame(
                preview_row, bg="#080c12", width=154)
            self.scene_face_strip.pack(side="left", fill="y", padx=(0, 6))
            self.scene_face_strip.pack_propagate(False)
            self.scene_face_strip.grid_columnconfigure(0, weight=1, uniform="faces")
            self.scene_face_strip.grid_columnconfigure(1, weight=1, uniform="faces")
            for row in range(4):
                self.scene_face_strip.grid_rowconfigure(row, weight=1, uniform="faces")
            for index in range(1, 9):
                slot = tk.Frame(
                    self.scene_face_strip, bg="#0d131c",
                    highlightthickness=1, highlightbackground=BORDER)
                row = (index - 1) % 4
                column = (index - 1) // 4
                slot.grid(
                    row=row, column=column, sticky="nsew",
                    padx=(0 if column == 0 else 3, 0),
                    pady=(0 if row == 0 else 3, 0))
                button = tk.Button(
                    slot, text=f"FACE {index}",
                    command=lambda value=index: self._set_face_variant(value),
                    bg="#123128" if index == self._active_face_variant else "#0d131c",
                    fg=MINT if index == self._active_face_variant else MUTED,
                    activebackground="#173d32", activeforeground=MINT,
                    relief="flat", bd=0, cursor="hand2",
                    font=("Segoe UI", 8, "bold"))
                button.pack(fill="both", expand=True)
                self._face_variant_buttons[index] = button
            self.scene_preview_lbl = tk.Label(
                preview_row, bg="#050608", bd=0, highlightthickness=1,
                highlightbackground=self._mix(CYAN, BG, 0.35))
            self.scene_preview_lbl.pack(side="left", fill="both", expand=True)
            self.scene_preview_lbl.bind(
                "<Configure>", self._on_scene_preview_resize)
        else:
            empty = tk.Label(
                holder, text="No scene selected",
                bg="#050608", fg=MUTED, font=("Segoe UI", 9))
            empty.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        if self._scene_capture_image is not None:
            self._refresh_scene_preview()

    def _set_face_variant(self, variant):
        self._active_face_variant = max(1, min(8, int(variant)))
        for index, button in self._face_variant_buttons.items():
            active = index == self._active_face_variant
            try:
                button.configure(
                    bg="#123128" if active else "#0d131c",
                    fg=MINT if active else MUTED)
            except Exception:
                pass
        self._log_msg(
            f"[test] visual fingerprint preset Face {self._active_face_variant}")

    def _detect_scene_face(self, frame):
        """Return a stable face box for the locked scene (RGB PIL image)."""
        self._scene_face_detect_count += 1
        if (self._scene_face_box is not None
                and self._scene_face_detect_count % 20 != 1):
            return self._scene_face_box
        try:
            if self._scene_face_detector is None:
                cascade = os.path.join(
                    cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                self._scene_face_detector = cv2.CascadeClassifier(cascade)
            rgb = np.asarray(frame.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            scale = min(1.0, 720.0 / max(gray.shape))
            scan = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
            faces = self._scene_face_detector.detectMultiScale(
                scan, scaleFactor=1.08, minNeighbors=4,
                minSize=(max(30, scan.shape[1] // 18),
                         max(30, scan.shape[0] // 18)))
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                inv = 1.0 / scale
                x, y, w, h = [int(round(v * inv)) for v in (x, y, w, h)]
                pad_x, pad_y = int(w * 0.18), int(h * 0.22)
                self._scene_face_box = (
                    max(0, x - pad_x), max(0, y - pad_y),
                    min(frame.width, x + w + pad_x),
                    min(frame.height, y + h + pad_y))
        except Exception:
            pass
        if self._scene_face_box is None:
            # Selected scenes normally center the presenter. This fallback is
            # deliberately conservative and remains stable for detector tests.
            side = int(min(frame.width, frame.height) * 0.62)
            cx, cy = frame.width // 2, int(frame.height * 0.42)
            self._scene_face_box = (
                max(0, cx - side // 2), max(0, cy - side // 2),
                min(frame.width, cx + side // 2),
                min(frame.height, cy + side // 2))
        return self._scene_face_box

    def _apply_scene_face_variant(self, frame):
        """Apply a strong deterministic benchmark transform to the scene face."""
        variant = int(getattr(self, "_active_face_variant", 1) or 1)
        if variant == 1 or frame is None:
            return frame
        rgb = np.asarray(frame.convert("RGB")).copy()
        h, w = rgb.shape[:2]
        box = self._detect_scene_face(frame)
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        base = rgb[y1:y2, x1:x2].copy()
        roi = base.copy()
        rh, rw = roi.shape[:2]
        if variant == 2:
            # Narrower, warmer face with lifted midtones.
            narrow = cv2.resize(roi, (max(8, int(rw * 0.84)), rh),
                                interpolation=cv2.INTER_CUBIC)
            roi = cv2.copyMakeBorder(
                narrow, 0, 0, (rw - narrow.shape[1]) // 2,
                rw - narrow.shape[1] - (rw - narrow.shape[1]) // 2,
                cv2.BORDER_REFLECT_101)
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(
                lab[:, :, 1].astype(np.int16) + 12, 0, 255).astype(np.uint8)
            lab[:, :, 2] = np.clip(
                lab[:, :, 2].astype(np.int16) + 20, 0, 255).astype(np.uint8)
            roi = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            roi = cv2.convertScaleAbs(roi, alpha=1.08, beta=7)
        elif variant == 3:
            # Wider lower face, cooler palette, stronger edge structure.
            src = np.float32([[0, 0], [rw - 1, 0], [0, rh - 1], [rw - 1, rh - 1]])
            dx = rw * 0.10
            dst = np.float32([[dx, 0], [rw - 1 - dx, 0],
                              [0, rh - 1], [rw - 1, rh - 1]])
            roi = cv2.warpPerspective(
                roi, cv2.getPerspectiveTransform(src, dst), (rw, rh),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            blur = cv2.GaussianBlur(roi, (0, 0), 1.4)
            roi = cv2.addWeighted(roi, 1.65, blur, -0.65, 0)
            roi[:, :, 2] = np.clip(
                roi[:, :, 2].astype(np.int16) - 16, 0, 255).astype(np.uint8)
            roi[:, :, 0] = np.clip(
                roi[:, :, 0].astype(np.int16) + 18, 0, 255).astype(np.uint8)
        elif variant == 4:
            # Vertically compact, desaturated, high local contrast plus fixed
            # low-amplitude texture for reproducible fingerprint testing.
            compact = cv2.resize(roi, (rw, max(8, int(rh * 0.86))),
                                 interpolation=cv2.INTER_CUBIC)
            roi = cv2.copyMakeBorder(
                compact, (rh - compact.shape[0]) // 2,
                rh - compact.shape[0] - (rh - compact.shape[0]) // 2,
                0, 0, cv2.BORDER_REFLECT_101)
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(6, 6))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            roi = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            roi = cv2.addWeighted(roi, 0.38, gray3, 0.62, 0)
            yy, xx = np.indices((rh, rw))
            texture = (((xx * 13 + yy * 7) % 17) - 8).astype(np.int16)
            roi = np.clip(
                roi.astype(np.int16) + texture[:, :, None],
                0, 255).astype(np.uint8)
        elif variant == 5:
            # Uneven mixed lighting: warm key light and cool opposing fill.
            yy, xx = np.indices((rh, rw), dtype=np.float32)
            mix = xx / max(1.0, rw - 1.0)
            warm = np.zeros_like(roi, dtype=np.float32)
            warm[:, :, 0] = 30.0 * (1.0 - mix)
            warm[:, :, 1] = 12.0 * (1.0 - mix)
            warm[:, :, 2] = 28.0 * mix
            shade = (0.72 + 0.46 * (1.0 - np.abs(mix - 0.42)))[:, :, None]
            roi = np.clip(
                roi.astype(np.float32) * shade + warm,
                0, 255).astype(np.uint8)
        elif variant == 6:
            # Low-resolution acquisition followed by blocky recompression.
            low_w, low_h = max(24, rw // 5), max(24, rh // 5)
            low = cv2.resize(roi, (low_w, low_h), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 28])
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                low = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(low, (rw, rh), interpolation=cv2.INTER_NEAREST)
            roi = ((roi.astype(np.uint16) // 24) * 24).clip(0, 255).astype(np.uint8)
        elif variant == 7:
            # Fixed partial occlusion benchmark across eyes and upper cheeks.
            overlay = roi.copy()
            top = int(rh * 0.24)
            bottom = int(rh * 0.60)
            cv2.rectangle(
                overlay, (int(rw * 0.08), top), (int(rw * 0.92), bottom),
                (24, 28, 34), -1)
            cv2.line(
                overlay, (int(rw * 0.08), bottom),
                (int(rw * 0.92), bottom), (95, 110, 126), 2)
            roi = cv2.addWeighted(overlay, 0.88, roi, 0.12, 0)
        elif variant == 8:
            # Deterministic perspective/pose stress test.
            src = np.float32([
                [0, 0], [rw - 1, 0], [0, rh - 1], [rw - 1, rh - 1]])
            dst = np.float32([
                [rw * 0.17, rh * 0.05], [rw * 0.91, 0],
                [rw * 0.04, rh * 0.94], [rw * 0.98, rh - 1]])
            roi = cv2.warpPerspective(
                roi, cv2.getPerspectiveTransform(src, dst), (rw, rh),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            matrix = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), -5.0, 1.03)
            roi = cv2.warpAffine(
                roi, matrix, (rw, rh), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101)
            roi = cv2.convertScaleAbs(roi, alpha=1.12, beta=-8)
        mask = np.zeros((y2 - y1, x2 - x1), np.float32)
        cv2.ellipse(
            mask, ((x2 - x1) // 2, (y2 - y1) // 2),
            (max(1, (x2 - x1) // 2 - 2), max(1, (y2 - y1) // 2 - 2)),
            0, 0, 360, 1.0, -1)
        feather = max(5, int(min(mask.shape) * 0.08) | 1)
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)[:, :, None]
        rgb[y1:y2, x1:x2] = np.clip(
            base * (1.0 - mask) + roi.astype(np.float32) * mask,
            0, 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    def _refresh_scene_preview(self):
        if self.scene_preview_lbl is None:
            return
        try:
            with self._scene_capture_lock:
                display = self._scene_display_image
            if display is None:
                return
            self._scene_capture_tk = ImageTk.PhotoImage(display)
            self.scene_preview_lbl.configure(image=self._scene_capture_tk)
        except Exception as exc:
            self._log_msg(f"[scene] preview failed: {exc}")

    def _on_scene_preview_resize(self, event):
        size = (max(240, int(event.width)), max(120, int(event.height)))
        with self._scene_capture_lock:
            self._scene_preview_size = size

    def _schedule_scene_preview(self):
        if self._scene_preview_job is not None:
            return
        self._scene_preview_job = self.root.after(33, self._scene_preview_tick)
        if self._scene_capture_thread is None or not self._scene_capture_thread.is_alive():
            self._scene_capture_stop.clear()
            self._scene_capture_thread = threading.Thread(
                target=self._scene_capture_loop, daemon=True)
            self._scene_capture_thread.start()

    def _scene_capture_loop(self):
        """Capture outside Tk's thread so screen I/O cannot freeze the UI."""
        youtube_frame_serial = -1
        last_raw_image_t = 0.0
        while not self._scene_capture_stop.is_set():
            with self._scene_capture_lock:
                source = self._scene_source
                bbox = self._scene_capture_bbox
                hwnd = self._scene_window_hwnd
                window_crop = self._scene_window_crop
                capture_method = self._scene_window_capture_method
                youtube_scene = self._youtube_scene
                youtube_crop = self._youtube_scene_crop
            if source == "youtube":
                if youtube_scene is None:
                    frame_serial, frame_array = -1, None
                else:
                    snapshot = getattr(youtube_scene, "frame_snapshot", None)
                    if callable(snapshot):
                        frame_serial, frame_array = snapshot()
                    else:
                        frame_serial, frame_array = 0, youtube_scene.frame()
                if frame_array is None:
                    self._scene_capture_stop.wait(0.08)
                    continue
                if frame_serial == youtube_frame_serial:
                    self._scene_capture_stop.wait(0.01)
                    continue
                youtube_frame_serial = frame_serial
                try:
                    from youtube_video import normalized_crop
                    cropped = normalized_crop(frame_array, youtube_crop)
                    frame_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                    frame = Image.fromarray(cropped, "RGB")
                    frame = self._apply_scene_face_variant(frame)
                    with self._scene_capture_lock:
                        target = self._scene_preview_size
                    now_raw = time.monotonic()
                    raw_image = None
                    if now_raw - last_raw_image_t >= 2.0:
                        raw_image = Image.fromarray(frame_array, "RGB")
                        last_raw_image_t = now_raw
                    preview = frame.copy()
                    preview.thumbnail(target, Image.BILINEAR)
                    display = Image.new("RGB", target, "#050608")
                    display.paste(
                        preview,
                        ((target[0] - preview.width) // 2,
                         (target[1] - preview.height) // 2))
                    with self._scene_capture_lock:
                        if self._scene_source == "youtube":
                            self._scene_capture_image = frame
                            self._scene_capture_bgr = frame_bgr
                            if raw_image is not None:
                                self._youtube_scene_raw_image = raw_image
                            self._scene_display_image = display
                            self._scene_capture_serial += 1
                except Exception as exc:
                    self._log_msg(f"[scene] YouTube frame failed: {exc}")
                    self._scene_capture_stop.wait(0.5)
                self._scene_capture_stop.wait(0.04)
                continue
            if bbox is None:
                self._scene_capture_stop.wait(0.20)
                continue
            try:
                frame = None
                if hwnd and window_crop:
                    candidates = self._capture_window_images(
                        hwnd, preferred=capture_method)
                    cropped = [
                        (method, image.crop(window_crop).convert("RGB"))
                        for method, image in candidates
                        if image is not None
                    ]
                    usable = [
                        (method, image) for method, image in cropped
                        if self._scene_frame_score(image) >= 5.0
                    ]
                    if usable:
                        method, frame = max(
                            usable, key=lambda pair: self._scene_frame_score(pair[1]))
                        with self._scene_capture_lock:
                            self._scene_window_capture_method = method
                    elif capture_method is not None:
                        with self._scene_capture_lock:
                            self._scene_window_capture_method = None
                if frame is None:
                    # Keep the last valid locked frame when a GPU window returns
                    # black. Desktop fallback would leak whatever overlaps it.
                    if hwnd and window_crop:
                        self._scene_capture_stop.wait(0.10)
                        continue
                    frame = ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
                frame = self._apply_scene_face_variant(frame)
                with self._scene_capture_lock:
                    target = self._scene_preview_size
                frame.thumbnail(target, Image.BILINEAR)
                display = Image.new("RGB", target, "#050608")
                display.paste(
                    frame,
                    ((target[0] - frame.width) // 2,
                     (target[1] - frame.height) // 2))
                with self._scene_capture_lock:
                    if bbox == self._scene_capture_bbox:
                        self._scene_capture_image = frame
                        self._scene_display_image = display
                        self._scene_capture_serial += 1
            except Exception as exc:
                self._log_msg(f"[scene] live capture failed: {exc}")
                self._scene_capture_stop.wait(0.5)
                continue
            self._scene_capture_stop.wait(0.06)

    @staticmethod
    def _window_at_point(x, y):
        if sys.platform != "win32":
            return None
        try:
            import ctypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            user32 = ctypes.windll.user32
            hwnd = user32.WindowFromPoint(POINT(int(x), int(y)))
            if not hwnd:
                return None
            root = user32.GetAncestor(hwnd, 2)  # GA_ROOT
            return int(root or hwnd)
        except Exception:
            return None

    @staticmethod
    def _window_rect(hwnd):
        if not hwnd or sys.platform != "win32":
            return None
        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long),
                ]

            rect = RECT()
            if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
            return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception:
            return None

    @staticmethod
    def _window_title(hwnd):
        if not hwnd or sys.platform != "win32":
            return ""
        try:
            import ctypes
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(max(2, length + 1))
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, len(buf))
            return buf.value.strip()
        except Exception:
            return ""

    @staticmethod
    def _scene_frame_score(image):
        """Reject flat black GPU frames while allowing legitimately dark UIs."""
        try:
            gray = np.asarray(image.resize((96, 54)).convert("L"), dtype=np.float32)
            contrast = float(gray.std())
            edges = (
                float(np.abs(np.diff(gray, axis=0)).mean())
                + float(np.abs(np.diff(gray, axis=1)).mean())
            )
            dynamic_range = float(gray.max() - gray.min())
            return contrast + edges * 1.5 + dynamic_range * 0.25
        except Exception:
            return 0.0

    @staticmethod
    def _capture_window_images(hwnd, preferred=None):
        """Return native window captures from multiple Windows rendering paths."""
        rect = AvatarStudio._window_rect(hwnd)
        if rect is None:
            return []
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width < 2 or height < 2:
            return []
        images = []
        if preferred in (None, "pillow"):
            try:
                image = ImageGrab.grab(window=hwnd)
                if image is not None and image.width >= 2 and image.height >= 2:
                    images.append(("pillow", image.convert("RGB")))
            except Exception:
                pass
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            window_dc = user32.GetWindowDC(hwnd)
            memory_dc = gdi32.CreateCompatibleDC(window_dc)
            bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
            old = gdi32.SelectObject(memory_dc, bitmap)
            try:
                def _read_bitmap():
                    bmi = ctypes.create_string_buffer(40)
                    ctypes.memset(bmi, 0, 40)
                    ctypes.cast(bmi, ctypes.POINTER(wintypes.DWORD))[0] = 40
                    ctypes.cast(bmi, ctypes.POINTER(wintypes.LONG))[1] = width
                    ctypes.cast(bmi, ctypes.POINTER(wintypes.LONG))[2] = -height
                    ctypes.cast(bmi, ctypes.POINTER(wintypes.WORD))[6] = 1
                    ctypes.cast(bmi, ctypes.POINTER(wintypes.WORD))[7] = 32
                    raw = ctypes.create_string_buffer(width * height * 4)
                    if not gdi32.GetDIBits(
                            memory_dc, bitmap, 0, height, raw, bmi, 0):
                        return None
                    return Image.frombuffer(
                        "RGB", (width, height), raw,
                        "raw", "BGRX", 0, 1).copy()

                methods = (
                    ((2, "print-full"), (0, "print-basic"), (3, "print-client"))
                    if preferred is None else
                    tuple(
                        pair for pair in (
                            (2, "print-full"), (0, "print-basic"),
                            (3, "print-client"))
                        if pair[1] == preferred)
                )
                for flag, method in methods:
                    if user32.PrintWindow(hwnd, memory_dc, flag):
                        image = _read_bitmap()
                        if image is not None:
                            images.append((method, image))
                # Some accelerated windows expose useful pixels through their
                # window DC even when PrintWindow returns a black surface.
                if preferred in (None, "bitblt") and gdi32.BitBlt(
                        memory_dc, 0, 0, width, height, window_dc,
                        0, 0, 0x00CC0020 | 0x40000000):
                    image = _read_bitmap()
                    if image is not None:
                        images.append(("bitblt", image))
            finally:
                gdi32.SelectObject(memory_dc, old)
                gdi32.DeleteObject(bitmap)
                gdi32.DeleteDC(memory_dc)
                user32.ReleaseDC(hwnd, window_dc)
        except Exception:
            pass
        return images

    @staticmethod
    def _capture_window_image(hwnd):
        """Compatibility helper returning the best available native capture."""
        images = AvatarStudio._capture_window_images(hwnd)
        if not images:
            return None
        return max(images, key=lambda pair: AvatarStudio._scene_frame_score(
            pair[1]))[1]

    def _scene_preview_tick(self):
        self._scene_preview_job = None
        with self._scene_capture_lock:
            bbox = self._scene_capture_bbox
            source = self._scene_source
        if bbox is None and source != "youtube":
            return
        try:
            with self._scene_capture_lock:
                serial = self._scene_capture_serial
            if serial != self._scene_rendered_serial:
                self._refresh_scene_preview()
                self._scene_rendered_serial = serial
        except Exception as exc:
            self._log_msg(f"[scene] live preview failed: {exc}")
        finally:
            if self._scene_capture_bbox is not None or self._scene_source == "youtube":
                self._schedule_scene_preview()

    def _list_monitors(self):
        if sys.platform == "win32":
            try:
                import ctypes
                monitors = []

                class RECT(ctypes.Structure):
                    _fields_ = [
                        ("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long),
                    ]

                def _callback(_hmon, _hdc, rect_ptr, _data):
                    rect = rect_ptr.contents
                    monitors.append({
                        "left": int(rect.left),
                        "top": int(rect.top),
                        "width": int(rect.right - rect.left),
                        "height": int(rect.bottom - rect.top),
                    })
                    return 1

                proc = ctypes.WINFUNCTYPE(
                    ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                    ctypes.POINTER(RECT), ctypes.c_longlong)
                ctypes.windll.user32.EnumDisplayMonitors(0, 0, proc(_callback), 0)
                if monitors:
                    return monitors
            except Exception:
                pass
        return [{
            "left": 0, "top": 0,
            "width": self.root.winfo_screenwidth(),
            "height": self.root.winfo_screenheight(),
        }]

    def _start_scene_snip(self):
        monitors = self._list_monitors()
        if len(monitors) > 1:
            self._choose_scene_monitor(monitors)
        else:
            self._open_scene_snipper(monitors[0])

    def _add_scene(self):
        """Use the pasted YouTube video first, otherwise open screen selection."""
        url = ""
        try:
            url = self._youtube_primary_url()
        except Exception:
            pass
        if "youtu" in url.lower():
            self._attach_youtube_scene(url, force=True)
            return
        self._start_scene_snip()

    def _choose_scene_monitor(self, monitors):
        win = tk.Toplevel(self.root)
        win.title("Select screen")
        win.configure(bg=BG)
        win.transient(self.root)
        win.attributes("-topmost", True)
        tk.Label(
            win, text="Select screen to snip from", bg=BG, fg=FG,
            font=("Segoe UI", 10, "bold")).pack(padx=18, pady=(14, 8))
        for i, mon in enumerate(monitors, start=1):
            tk.Button(
                win, text=f"Screen {i}  {mon['width']}x{mon['height']}",
                command=lambda m=mon: (win.destroy(), self._open_scene_snipper(m)),
                bg=SURFACE2, fg=FG,
                activebackground=self._mix(SURFACE2, CYAN, 0.25),
                activeforeground=FG, relief="flat", bd=0, cursor="hand2",
                font=("Segoe UI", 9), padx=18, pady=8,
                highlightthickness=1, highlightbackground=BORDER).pack(
                    fill="x", padx=18, pady=4)
        tk.Button(
            win, text="Cancel", command=win.destroy,
            bg=self._mix(SURFACE2, RED, 0.14), fg=RED,
            activebackground=self._mix(SURFACE2, RED, 0.24),
            activeforeground=RED, relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 9, "bold"), padx=18, pady=7).pack(
                fill="x", padx=18, pady=(8, 14))
        win.update_idletasks()
        x = self.root.winfo_x() + max(40, (self.root.winfo_width() - win.winfo_width()) // 2)
        y = self.root.winfo_y() + 90
        win.geometry(f"+{x}+{y}")

    def _open_scene_snipper(self, monitor):
        try:
            self.root.withdraw()
            self.root.after(180, lambda: self._show_scene_snipper(monitor))
        except Exception:
            self._show_scene_snipper(monitor)

    def _show_scene_snipper(self, monitor):
        try:
            bbox = (
                monitor["left"], monitor["top"],
                monitor["left"] + monitor["width"],
                monitor["top"] + monitor["height"],
            )
            shot = ImageGrab.grab(bbox=bbox, all_screens=True)
        except Exception as exc:
            self.root.deiconify()
            self._log_msg(f"[scene] screen capture failed: {exc}")
            return

        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.geometry(
            f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}")
        cv = tk.Canvas(overlay, highlightthickness=0, bd=0, cursor="crosshair")
        cv.pack(fill="both", expand=True)
        bg = shot.resize((monitor["width"], monitor["height"]), Image.LANCZOS)
        tk_bg = ImageTk.PhotoImage(bg)
        overlay._scene_bg = tk_bg
        cv.create_image(0, 0, image=tk_bg, anchor="nw")
        cv.create_rectangle(
            0, 0, monitor["width"], monitor["height"],
            fill="#000000", stipple="gray50", outline="")
        cv.create_text(
            monitor["width"] // 2, 28,
            text="Drag to select a scene region. Enter confirms. Esc cancels.",
            fill="#ffffff", font=("Segoe UI", 12, "bold"))
        state = {"start": None, "rect": None, "box": None}

        def _draw_box(x1, y1, x2, y2):
            if state["rect"] is not None:
                cv.delete(state["rect"])
            state["rect"] = cv.create_rectangle(x1, y1, x2, y2, outline=CYAN, width=3)
            state["box"] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        def _down(event):
            state["start"] = (event.x, event.y)
            _draw_box(event.x, event.y, event.x + 1, event.y + 1)

        def _drag(event):
            if state["start"] is None:
                return
            sx, sy = state["start"]
            _draw_box(sx, sy, event.x, event.y)

        def _cancel(_event=None):
            overlay.destroy()
            self.root.deiconify()
            self.root.lift()

        def _confirm(_event=None):
            box = state.get("box")
            if not box or box[2] - box[0] < 8 or box[3] - box[1] < 8:
                return
            crop = shot.crop(box)
            abs_box = (
                monitor["left"] + box[0],
                monitor["top"] + box[1],
                monitor["left"] + box[2],
                monitor["top"] + box[3],
            )
            # Remove the selection overlay before asking Windows which native
            # window owns the selected region.
            overlay.withdraw()
            overlay.update_idletasks()
            center_x = (abs_box[0] + abs_box[2]) // 2
            center_y = (abs_box[1] + abs_box[3]) // 2
            hwnd = self._window_at_point(center_x, center_y)
            window_rect = self._window_rect(hwnd)
            window_crop = None
            if window_rect is not None:
                window_crop = (
                    max(0, abs_box[0] - window_rect[0]),
                    max(0, abs_box[1] - window_rect[1]),
                    min(window_rect[2] - window_rect[0],
                        abs_box[2] - window_rect[0]),
                    min(window_rect[3] - window_rect[1],
                        abs_box[3] - window_rect[1]),
                )
                if (window_crop[2] - window_crop[0] < 8
                        or window_crop[3] - window_crop[1] < 8):
                    window_crop = None
            with self._scene_capture_lock:
                youtube_scene = self._youtube_scene
                self._scene_source = "screen"
                self._scene_capture_bbox = abs_box
                self._scene_window_hwnd = hwnd if window_crop is not None else None
                self._scene_window_crop = window_crop
                self._scene_window_title = (
                    self._window_title(hwnd) if window_crop is not None else "")
                self._scene_window_capture_method = None
                self._scene_face_box = None
                self._scene_face_detect_count = 0
                self._scene_capture_image = crop.convert("RGB")
                initial = self._scene_capture_image.copy()
                initial.thumbnail(self._scene_preview_size, Image.BILINEAR)
                self._scene_display_image = Image.new(
                    "RGB", self._scene_preview_size, "#050608")
                self._scene_display_image.paste(
                    initial,
                    ((self._scene_preview_size[0] - initial.width) // 2,
                     (self._scene_preview_size[1] - initial.height) // 2))
                self._scene_capture_serial += 1
            if youtube_scene is not None:
                try:
                    youtube_scene.stop()
                except Exception:
                    pass
                self._youtube_scene = None
                self._youtube_scene_url = ""
            overlay.destroy()
            self.root.deiconify()
            self.root.lift()
            self._build_sidebar_scene_slot()
            self._schedule_scene_preview()
            if window_crop is not None:
                title = self._scene_window_title or "selected window"
                self._log_msg(
                    f"[scene] locked to window: {title} "
                    f"({crop.width}x{crop.height})")
            else:
                self._log_msg(
                    f"[scene] screen region added {crop.width}x{crop.height}; "
                    "window lock unavailable")

        cv.bind("<Button-1>", _down)
        cv.bind("<B1-Motion>", _drag)
        cv.bind("<ButtonRelease-1>", _drag)
        overlay.bind("<Return>", _confirm)
        overlay.bind("<Escape>", _cancel)
        overlay.bind("<Double-Button-1>", _confirm)
        overlay.focus_force()

    def _open_scene_text_overlay_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Scene Text")
        win.configure(bg=SURFACE)
        win.transient(self.root)
        win.grab_set()
        win.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 160,
            self.root.winfo_rooty() + 120))

        items = [dict(item) for item in self._scene_text_overlay_items()]
        if not items:
            items = [self._default_scene_text_item("Text 1")]
        active_index = max(0, min(len(items) - 1, int(getattr(self, "_scene_text_active_index", 0) or 0)))

        enabled_var = tk.BooleanVar()
        text_var = tk.StringVar()
        font_var = tk.StringVar()
        size_var = tk.IntVar()
        color_var = tk.StringVar()
        bg_var = tk.StringVar()
        outline_var = tk.StringVar()
        behavior_var = tk.StringVar()
        position_var = tk.StringVar()
        opacity_var = tk.IntVar()
        x_var = tk.IntVar()
        y_var = tk.IntVar()

        body = tk.Frame(win, bg=SURFACE)
        body.pack(fill="both", expand=True, padx=14, pady=14)
        tk.Label(body, text="Scene text overlays", bg=SURFACE, fg=CYAN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        main = tk.Frame(body, bg=SURFACE)
        main.pack(fill="both", expand=True, pady=(8, 0))
        list_frame = tk.Frame(main, bg=SURFACE)
        list_frame.pack(side="left", fill="y", padx=(0, 12))
        tk.Label(list_frame, text="TEXTS", bg=SURFACE, fg=FAINT,
                 font=("Consolas", 8, "bold")).pack(anchor="w")
        listbox = tk.Listbox(
            list_frame, height=9, width=18, bg=SURFACE2, fg=FG,
            selectbackground=self._mix(CYAN, BG, 0.35),
            selectforeground=FG, relief="flat", highlightthickness=1,
            highlightbackground=BORDER, exportselection=False,
            font=("Segoe UI", 9))
        listbox.pack(fill="y", pady=(3, 7))

        editor = tk.Frame(main, bg=SURFACE)
        editor.pack(side="left", fill="both", expand=True)
        tk.Checkbutton(
            editor, text="Show selected text", variable=enabled_var, bg=SURFACE,
            fg=FG, selectcolor=SURFACE2, activebackground=SURFACE,
            activeforeground=CYAN, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 4))

        tk.Entry(
            editor, textvariable=text_var, bg=SURFACE2, fg=FG,
            insertbackground=CYAN, relief="flat", font=("Segoe UI", 11),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=CYAN).pack(fill="x", ipady=6, pady=(0, 8))

        grid = tk.Frame(editor, bg=SURFACE)
        grid.pack(fill="x")
        for col in range(2):
            grid.grid_columnconfigure(col, weight=1)

        def _field(row, col, label, widget):
            tk.Label(grid, text=label, bg=SURFACE, fg=FAINT,
                     font=("Consolas", 8, "bold")).grid(
                         row=row * 2, column=col, sticky="w",
                         padx=(0 if col == 0 else 10, 0), pady=(4, 2))
            widget.grid(
                row=row * 2 + 1, column=col, sticky="ew",
                padx=(0 if col == 0 else 10, 0), pady=(0, 4))

        _field(0, 0, "FONT", ttk.Combobox(
            grid, textvariable=font_var,
            values=("Segoe UI", "Arial", "Consolas", "Impact"),
            state="readonly", style="Studio.TCombobox"))
        _field(0, 1, "SIZE", ttk.Spinbox(
            grid, from_=28, to=160, increment=4, textvariable=size_var,
            style="Studio.TSpinbox"))
        _field(1, 0, "BEHAVIOR", ttk.Combobox(
            grid, textvariable=behavior_var,
            values=("Static", "Ticker", "Pulse"),
            state="readonly", style="Studio.TCombobox"))
        _field(1, 1, "POSITION", ttk.Combobox(
            grid, textvariable=position_var,
            values=("Top", "Middle", "Bottom"),
            state="readonly", style="Studio.TCombobox"))
        _field(2, 0, "BACKGROUND %", ttk.Spinbox(
            grid, from_=0, to=100, increment=5, textvariable=opacity_var,
            style="Studio.TSpinbox"))
        _field(3, 0, "X %", ttk.Spinbox(
            grid, from_=0, to=100, increment=2, textvariable=x_var,
            style="Studio.TSpinbox"))
        _field(3, 1, "Y %", ttk.Spinbox(
            grid, from_=0, to=100, increment=2, textvariable=y_var,
            style="Studio.TSpinbox"))

        drag_canvas = tk.Canvas(
            editor, width=180, height=320, bg="#050608",
            highlightthickness=1, highlightbackground=BORDER, bd=0,
            cursor="fleur")
        drag_canvas.pack(anchor="w", pady=(8, 0))
        drag_canvas.create_rectangle(0, 0, 179, 319, outline=self._mix(CYAN, BG, 0.45))
        drag_canvas.create_line(0, 188, 180, 188, fill=self._mix(AMBER, BG, 0.45))

        def _refresh_drag_preview(*_args):
            try:
                drag_canvas.delete("textitem")
                x = int(max(0, min(100, x_var.get() or 50)) * 1.8)
                y = int(max(0, min(100, y_var.get() or 8)) * 3.2)
                label = text_var.get().strip() or "Text"
                label = label[:16] + ("..." if len(label) > 16 else "")
                color = color_var.get() if str(color_var.get()).startswith("#") else CYAN
                drag_canvas.create_rectangle(
                    max(2, x - 54), max(2, y - 16),
                    min(178, x + 54), min(318, y + 16),
                    fill=self._mix(SURFACE2, CYAN, 0.12), outline=color,
                    tags="textitem")
                drag_canvas.create_text(
                    x, y, text=label, fill=color, font=("Segoe UI", 8, "bold"),
                    tags="textitem")
            except Exception:
                pass

        def _drag_text(event):
            x_var.set(max(0, min(100, int(round(event.x / 1.8)))))
            y_var.set(max(0, min(100, int(round(event.y / 3.2)))))
            _refresh_drag_preview()

        drag_canvas.bind("<Button-1>", _drag_text)
        drag_canvas.bind("<B1-Motion>", _drag_text)
        for var in (text_var, color_var, x_var, y_var):
            try:
                var.trace_add("write", _refresh_drag_preview)
            except Exception:
                pass

        def _choose(var):
            try:
                from tkinter import colorchooser
                result = colorchooser.askcolor(color=var.get(), parent=win)
                if result and result[1]:
                    var.set(result[1])
            except Exception:
                pass

        swatches = tk.Frame(grid, bg=SURFACE)
        for label, var in (("Text", color_var), ("Box", bg_var), ("Line", outline_var)):
            tk.Button(
                swatches, text=label, command=lambda v=var: _choose(v),
                bg=SURFACE2, fg=FG,
                activebackground=self._mix(SURFACE2, CYAN, 0.20),
                relief="flat", bd=0, font=("Segoe UI", 8, "bold"),
                highlightthickness=1, highlightbackground=BORDER,
                padx=8, pady=4).pack(side="left", padx=(0, 6))
        _field(2, 1, "COLORS", swatches)

        def _save_current():
            if not items:
                return
            item = items[active_index]
            item["enabled"] = bool(enabled_var.get())
            item["text"] = text_var.get().strip()
            item["font"] = font_var.get()
            item["size"] = max(12, int(size_var.get() or 72))
            item["color"] = color_var.get()
            item["bg"] = bg_var.get()
            item["outline"] = outline_var.get()
            item["behavior"] = behavior_var.get()
            item["position"] = position_var.get()
            item["opacity"] = max(0, min(100, int(opacity_var.get() or 0)))
            item["x"] = max(0, min(100, int(x_var.get() or 50)))
            item["y"] = max(0, min(100, int(y_var.get() or 8)))

        def _refresh_list():
            listbox.delete(0, "end")
            for index, item in enumerate(items, start=1):
                name = item.get("text") or f"Text {index}"
                if len(name) > 18:
                    name = name[:17] + "..."
                prefix = "✓ " if item.get("enabled", True) else "  "
                listbox.insert("end", prefix + name)
            if items:
                listbox.selection_clear(0, "end")
                listbox.selection_set(active_index)
                listbox.activate(active_index)

        def _load(index):
            nonlocal active_index
            active_index = max(0, min(len(items) - 1, int(index)))
            item = items[active_index]
            enabled_var.set(bool(item.get("enabled", True)))
            text_var.set(str(item.get("text", "")))
            font_var.set(str(item.get("font", "Segoe UI")))
            size_var.set(int(item.get("size", 72) or 72))
            color_var.set(str(item.get("color", "#ffffff")))
            bg_var.set(str(item.get("bg", "#000000")))
            outline_var.set(str(item.get("outline", "#00e5ff")))
            behavior_var.set(str(item.get("behavior", "Static")))
            position_var.set(str(item.get("position", "Top")))
            opacity_var.set(int(item.get("opacity", 74) or 74))
            x_var.set(int(item.get("x", 50) or 50))
            y_var.set(int(item.get("y", 8) or 8))
            _refresh_list()
            _refresh_drag_preview()

        def _select(_event=None):
            nonlocal active_index
            if not listbox.curselection():
                return
            _save_current()
            _load(listbox.curselection()[0])

        listbox.bind("<<ListboxSelect>>", _select)

        def _add():
            nonlocal active_index
            _save_current()
            items.append(self._default_scene_text_item(f"Text {len(items) + 1}"))
            _load(len(items) - 1)

        def _duplicate():
            nonlocal active_index
            _save_current()
            clone = dict(items[active_index])
            clone["text"] = (clone.get("text") or "Text") + " copy"
            items.insert(active_index + 1, clone)
            _load(active_index + 1)

        def _delete():
            nonlocal active_index
            if not items:
                return
            del items[active_index]
            if not items:
                items.append(self._default_scene_text_item("Text 1"))
            _load(min(active_index, len(items) - 1))

        list_actions = tk.Frame(list_frame, bg=SURFACE)
        list_actions.pack(fill="x")
        for label, cmd in (("+", _add), ("Copy", _duplicate), ("Del", _delete)):
            tk.Button(
                list_actions, text=label, command=cmd, bg=SURFACE2, fg=CYAN,
                activebackground=self._mix(SURFACE2, CYAN, 0.20),
                relief="flat", bd=0, font=("Segoe UI", 8, "bold"),
                highlightthickness=1, highlightbackground=BORDER,
                padx=6, pady=3).pack(side="left", padx=(0, 4))

        actions = tk.Frame(body, bg=SURFACE)
        actions.pack(fill="x", pady=(12, 0))

        def _apply():
            _save_current()
            self._scene_text_items = [dict(item) for item in items if item.get("text")]
            self._scene_text_active_index = active_index
            self._sync_legacy_scene_text_from_items()
            self._build_sidebar_scene_slot()
            win.destroy()

        def _clear():
            self._scene_text_items = []
            self._sync_legacy_scene_text_from_items()
            self._build_sidebar_scene_slot()
            win.destroy()

        self._btn(actions, "Apply", _apply, bg=CYAN, fg=CYAN_INK,
                  hover=CYAN_HI, border=CYAN,
                  font=("Segoe UI", 9, "bold")).pack(side="right", ipadx=12, ipady=5)
        self._btn(actions, "Clear", _clear, bg=SURFACE2, fg=RED,
                  hover=self._mix(SURFACE2, RED, 0.18), border=RED,
                  font=("Segoe UI", 9, "bold")).pack(side="right", padx=(0, 8), ipadx=12, ipady=5)
        _load(active_index)

    @staticmethod
    def _hex_to_rgb(value, fallback=(255, 255, 255)):
        try:
            value = str(value or "").strip()
            if value.startswith("#"):
                value = value[1:]
            if len(value) == 3:
                value = "".join(ch * 2 for ch in value)
            if len(value) != 6:
                return fallback
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return fallback

    @staticmethod
    def _scene_text_font_path(family):
        family = str(family or "").lower()
        windir = os.environ.get("WINDIR", r"C:\Windows")
        fonts = os.path.join(windir, "Fonts")
        if "consol" in family:
            return os.path.join(fonts, "consolab.ttf")
        if "impact" in family:
            return os.path.join(fonts, "impact.ttf")
        if "arial" in family:
            return os.path.join(fonts, "arialbd.ttf")
        return os.path.join(fonts, "segoeuib.ttf")

    @staticmethod
    def _default_scene_text_item(text="Text"):
        return {
            "enabled": True,
            "text": text,
            "font": "Segoe UI",
            "size": 72,
            "color": "#ffffff",
            "bg": "#000000",
            "outline": "#00e5ff",
            "behavior": "Static",
            "position": "Top",
            "opacity": 74,
            "x": 50,
            "y": 8,
        }

    def _legacy_scene_text_item(self):
        return {
            "enabled": bool(getattr(self, "_scene_text_enabled", False)),
            "text": str(getattr(self, "_scene_text", "") or ""),
            "font": str(getattr(self, "_scene_text_font", "Segoe UI")),
            "size": int(getattr(self, "_scene_text_size", 72) or 72),
            "color": str(getattr(self, "_scene_text_color", "#ffffff")),
            "bg": str(getattr(self, "_scene_text_bg", "#000000")),
            "outline": str(getattr(self, "_scene_text_outline", "#00e5ff")),
            "behavior": str(getattr(self, "_scene_text_behavior", "Static")),
            "position": str(getattr(self, "_scene_text_position", "Top")),
            "opacity": int(getattr(self, "_scene_text_opacity", 74) or 74),
            "x": int(getattr(self, "_scene_text_x", 50) or 50),
            "y": int(getattr(self, "_scene_text_y", 8) or 8),
        }

    def _scene_text_overlay_items(self):
        items = getattr(self, "_scene_text_items", None)
        if items:
            return [dict(item) for item in items if str(item.get("text", "")).strip()]
        legacy = self._legacy_scene_text_item()
        if legacy["enabled"] and legacy["text"].strip():
            return [legacy]
        return []

    def _scene_text_overlay_active(self):
        return any(
            bool(item.get("enabled", True)) and str(item.get("text", "")).strip()
            for item in self._scene_text_overlay_items())

    def _sync_legacy_scene_text_from_items(self):
        items = self._scene_text_overlay_items()
        first = items[0] if items else self._default_scene_text_item("")
        self._scene_text_enabled = bool(first.get("enabled", False)) and bool(first.get("text"))
        self._scene_text = str(first.get("text", ""))
        self._scene_text_font = str(first.get("font", "Segoe UI"))
        self._scene_text_size = int(first.get("size", 72) or 72)
        self._scene_text_color = str(first.get("color", "#ffffff"))
        self._scene_text_bg = str(first.get("bg", "#000000"))
        self._scene_text_outline = str(first.get("outline", "#00e5ff"))
        self._scene_text_behavior = str(first.get("behavior", "Static"))
        self._scene_text_position = str(first.get("position", "Top"))
        self._scene_text_opacity = int(first.get("opacity", 74) or 74)
        self._scene_text_x = int(first.get("x", 50) or 50)
        self._scene_text_y = int(first.get("y", 8) or 8)

    def _draw_scene_text_item(self, image, item, index=0, count=1):
        h, w = image.size[1], image.size[0]
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        text = str(item.get("text", "") or "").strip()
        size = max(12, int(item.get("size", 72) or 72))
        behavior = str(item.get("behavior", "Static") or "Static")
        if behavior == "Pulse":
            size = int(round(size * (1.0 + 0.08 * math.sin(time.monotonic() * 4.0))))
        try:
            font = ImageFont.truetype(
                self._scene_text_font_path(item.get("font", "")),
                size=size)
        except Exception:
            font = ImageFont.load_default()
        text_color = self._hex_to_rgb(item.get("color", "#ffffff"))
        bg_color = self._hex_to_rgb(item.get("bg", "#000000"), (0, 0, 0))
        outline_color = self._hex_to_rgb(item.get("outline", "#00e5ff"), (0, 229, 255))
        opacity = max(0, min(100, int(item.get("opacity", 74) or 0)))
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=3)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x = max(22, int(size * 0.36))
        pad_y = max(14, int(size * 0.22))
        box_w = min(w - 48, tw + pad_x * 2)
        box_h = min(h - 48, th + pad_y * 2)
        position = str(item.get("position", "Top") or "Top")
        stack_gap = 14
        has_free_position = "x" in item or "y" in item
        if has_free_position:
            x = int(w * max(0, min(100, int(item.get("x", 50) or 50))) / 100.0 - box_w / 2)
            y = int(h * max(0, min(100, int(item.get("y", 8) or 8))) / 100.0 - box_h / 2)
            y += index * stack_gap
        elif position == "Middle":
            y = int(h * 0.42 - ((box_h + stack_gap) * count - stack_gap) / 2)
            y += index * (box_h + stack_gap)
            x = (w - box_w) // 2
        elif position == "Bottom":
            y = h - 76 - ((box_h + stack_gap) * (count - index) - stack_gap)
            x = (w - box_w) // 2
        else:
            y = 54 + index * (box_h + stack_gap)
            x = (w - box_w) // 2
        y = max(18, min(h - box_h - 18, y))
        if behavior == "Ticker":
            span = w + box_w
            x = int(w - ((time.monotonic() * 150.0 + index * 120.0) % span))
        else:
            x = max(18, min(w - box_w - 18, x))
        rect = (x, y, x + box_w, y + box_h)
        draw.rounded_rectangle(
            rect, radius=20,
            fill=(*bg_color, int(255 * opacity / 100.0)),
            outline=(*outline_color, 230), width=4)
        tx = x + max(0, (box_w - tw) // 2)
        ty = y + max(0, (box_h - th) // 2) - max(0, bbox[1])
        draw.text(
            (tx, ty), text, font=font, fill=(*text_color, 255),
            stroke_width=3, stroke_fill=(*outline_color, 210))
        return Image.alpha_composite(image, layer)

    def _apply_scene_text_overlay(self, frame):
        items = [
            item for item in self._scene_text_overlay_items()
            if bool(item.get("enabled", True)) and str(item.get("text", "")).strip()
        ]
        if not items:
            return frame
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            for index, item in enumerate(items):
                image = self._draw_scene_text_item(image, item, index, len(items))
            image = image.convert("RGB")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except Exception:
            return frame

    def _resize_preview_stage(self, event=None):
        wrap = getattr(self, "preview_stage_wrap", None)
        stage = getattr(self, "preview_stage", None)
        if wrap is None or stage is None:
            return
        w = max(1, wrap.winfo_width() - 20)
        h = max(1, wrap.winfo_height() - 20)
        target_ratio = PREVIEW_W / PREVIEW_H
        if w / h > target_ratio:
            ph = h
            pw = int(ph * target_ratio)
        else:
            pw = w
            ph = int(pw / target_ratio)
        min_h = min(h, 640)
        min_w = int(min_h * target_ratio)
        if pw < min_w or ph < min_h:
            if w >= min_w and h >= min_h:
                pw, ph = min_w, min_h
        old_size = getattr(self, "_preview_draw_size", None)
        self._preview_draw_size = (pw, ph)
        stage.configure(width=pw, height=ph)
        stage.place(relx=0.5, rely=0.5, anchor="center")
        if getattr(self, "preview_overlay", None) is not None:
            self.preview_overlay.place_configure(width=pw)
        if old_size != (pw, ph) and not getattr(self, "running", False):
            try:
                self._show_placeholder()
            except Exception:
                pass

    def _build_ui(self):
        self.root.configure(bg=BG)

        # Top app bar.
        TBH = 54
        topcv = tk.Canvas(self.root, bg=BG, height=TBH, highlightthickness=0, bd=0)
        topcv.pack(side="top", fill="x")
        self._topcv = topcv
        self._sweep_y = TBH - 5
        self._sweep = topcv.create_oval(0, 0, 0, 0, fill=CYAN, outline="")

        def _topdraw(_=None):
            w = topcv.winfo_width()
            if w <= 1:
                return
            topcv.delete("tb")
            topcv.create_rectangle(0, 0, w, TBH, fill="#07080b", outline="", tags="tb")
            topcv.create_line(0, 0, w, 0, fill="#242936", width=1, tags="tb")
            topcv.create_oval(17, 14, 43, 40, fill="#0d1016",
                              outline=self._mix(BORDER, CYAN, 0.48), width=1, tags="tb")
            topcv.create_text(29, 27, text="♪", fill=CYAN,
                              font=("Segoe UI", 19, "bold"), tags="tb")
            topcv.create_text(33, 27, text="♪", fill=MAG,
                              font=("Segoe UI", 17, "bold"), tags="tb")
            topcv.create_text(31, 26, text="♪", fill="#ffffff",
                              font=("Segoe UI", 16, "bold"), tags="tb")
            topcv.create_text(50, 20, text="TikTok Live Bot", anchor="w", fill=FG,
                              font=("Segoe UI", 13, "bold"), tags="tb")
            topcv.create_text(50, 36, text="local build", anchor="w", fill=MUTED,
                              font=("Segoe UI", 7), tags="tb")
            live_color = RED if self.running else self._mix(SURFACE2, RED, 0.22)
            live_text = "LIVE" if self.running else "IDLE"
            timer = self._format_duration(self._uptime_seconds())
            chat_on = bool(getattr(self, "tiktok", None) is not None)
            chat_live = bool(getattr(self, "_handle_live", False))
            chat_text = "Chat Connected" if chat_on else ("TikTok Live" if chat_live else "Chat Offline")
            chat_dot = MINT if chat_on else (AMBER if chat_live else FAINT)
            self._round_rect(topcv, 190, 16, 236, 38, 6, fill=live_color, outline="", tags="tb")
            topcv.create_text(213, 27, text=live_text, fill="#ffffff",
                              font=("Segoe UI", 9, "bold"), tags="tb")
            topcv.create_text(248, 27, text=timer, anchor="w", fill=FG,
                              font=("Consolas", 10), tags="tb")
            self._round_rect(topcv, 314, 14, 426, 40, 6, fill=SURFACE, outline=BORDER, tags="tb")
            topcv.create_oval(324, 23, 332, 31, fill=chat_dot, outline="", tags="tb")
            topcv.create_text(340, 23, text=("Connected" if chat_on else "Waiting"),
                              anchor="w", fill=FG,
                              font=("Segoe UI", 8), tags="tb")
            topcv.create_text(340, 34, text=chat_text, anchor="w", fill=MUTED,
                              font=("Segoe UI", 7), tags="tb")
            ready = self._ready_speech_snapshot()
            ready_x1 = 442
            ready_x2 = min(w - 286, 650)
            if ready_x2 > ready_x1 + 80:
                status = ready.get("status") if ready else None
                is_ready = status == "ready"
                is_preparing = status == "preparing"
                kind = ready.get("kind") if ready else None
                pulse = 0.5 + 0.5 * math.sin(
                    time.monotonic() * (9.0 if is_ready else 5.5))
                if is_ready:
                    action = "SPEAK THANKS" if kind == "urgent" else "SPEAK COMMENT"
                    ready_text = f"READY  •  {action}"
                    ready_fill = self._mix("#075f36", "#72ff67", pulse)
                    ready_outline = self._mix(MINT, "#ffffff", pulse * 0.9)
                    ready_fg = "#041009" if pulse > 0.42 else "#ffffff"
                elif is_preparing:
                    ready_text = "PREPARING VOICE"
                    ready_fill = self._mix("#472400", "#ff8a00", pulse)
                    ready_outline = self._mix(AMBER, "#fff36b", pulse)
                    ready_fg = "#fffdf2"
                else:
                    ready_text = "NOTHING READY"
                    ready_fill, ready_outline, ready_fg = SURFACE, BORDER, MUTED
                if is_ready or is_preparing:
                    halo = MINT if is_ready else AMBER
                    self._round_rect(
                        topcv, ready_x1 - 3, 11, ready_x2 + 3, 43, 8,
                        fill="", outline=self._mix(BG, halo, 0.35 + pulse * 0.6),
                        width=2, tags="tb")
                self._round_rect(topcv, ready_x1, 14, ready_x2, 40, 6,
                                 fill=ready_fill, outline=ready_outline,
                                 width=2 if is_ready or is_preparing else 1,
                                 tags="tb")
                topcv.create_oval(ready_x1 + 10, 23, ready_x1 + 18, 31,
                                  fill=(
                                      self._mix(MINT, "#ffffff", pulse)
                                      if is_ready else
                                      self._mix(AMBER, "#fff36b", pulse)
                                      if is_preparing else FAINT),
                                  outline="", tags="tb")
                topcv.create_text((ready_x1 + ready_x2) // 2 + 6, 27, text=ready_text,
                                  fill=ready_fg,
                                  font=("Segoe UI", 9 if is_ready else 8, "bold"),
                                  tags="tb")
                self._tb_buttons["ready_speech"] = (ready_x1, 14, ready_x2, 40)
            ready_hits = []
            split_x1 = 442
            split_x2 = min(w - 286, 930)

            def _draw_ready_lane(slot, label, x1, x2):
                item = self._ready_speech_snapshot(slot)
                status = item.get("status") if item else None
                is_ready = status == "ready"
                is_preparing = status == "preparing"
                pulse = 0.5 + 0.5 * math.sin(
                    time.monotonic() * (9.0 if is_ready else 5.5))
                skip_w = 46
                speak_x2 = max(x1 + 58, x2 - skip_w - 4)
                if is_ready:
                    text = f"{label} READY"
                    fill = self._mix("#075f36", "#72ff67", pulse)
                    outline = self._mix(MINT, "#ffffff", pulse * 0.9)
                    fg = "#041009" if pulse > 0.42 else "#ffffff"
                    dot = self._mix(MINT, "#ffffff", pulse)
                    width = 2
                elif is_preparing:
                    text = f"{label} PREP"
                    fill = self._mix("#472400", "#ff8a00", pulse)
                    outline = self._mix(AMBER, "#fff36b", pulse)
                    fg = "#fffdf2"
                    dot = self._mix(AMBER, "#fff36b", pulse)
                    width = 2
                else:
                    text = f"{label} WAIT"
                    fill, outline, fg, dot, width = SURFACE, BORDER, MUTED, FAINT, 1
                self._round_rect(topcv, x1 - 2, 12, x2 + 2, 42, 8,
                                 fill="#07080b", outline="", tags="tb")
                if is_ready or is_preparing:
                    halo = MINT if is_ready else AMBER
                    self._round_rect(
                        topcv, x1 - 3, 11, x2 + 3, 43, 8,
                        fill="", outline=self._mix(BG, halo, 0.35 + pulse * 0.6),
                        width=2, tags="tb")
                self._round_rect(topcv, x1, 14, speak_x2, 40, 6,
                                 fill=fill, outline=outline, width=width, tags="tb")
                topcv.create_oval(x1 + 8, 23, x1 + 16, 31,
                                  fill=dot, outline="", tags="tb")
                topcv.create_text((x1 + speak_x2) // 2 + 6, 27, text=text,
                                  fill=fg, font=("Segoe UI", 8, "bold"),
                                  tags="tb")
                skip_fill = self._mix(SURFACE2, RED, 0.26 if item else 0.08)
                self._round_rect(topcv, speak_x2 + 4, 14, x2, 40, 6,
                                 fill=skip_fill, outline=RED if item else BORDER,
                                 width=1, tags="tb")
                topcv.create_text((speak_x2 + 4 + x2) // 2, 27, text="SKIP",
                                  fill=RED if item else FAINT,
                                  font=("Segoe UI", 7, "bold"), tags="tb")
                key = "comment" if slot == "comment" else "thanks"
                ready_hits.append((f"ready_{key}_speak", (x1, 14, speak_x2, 40)))
                ready_hits.append((f"ready_{key}_skip", (speak_x2 + 4, 14, x2, 40)))

            if split_x2 > split_x1 + 190:
                split_gap = 8
                lane_w = (split_x2 - split_x1 - split_gap) // 2
                _draw_ready_lane("urgent", "THANKS", split_x1, split_x1 + lane_w)
                _draw_ready_lane(
                    "comment", "COMMENT", split_x1 + lane_w + split_gap,
                    split_x1 + lane_w + split_gap + lane_w)
            notice = str(getattr(self, "_youtube_link_notice", "") or "")
            notice_until = float(
                getattr(self, "_youtube_link_notice_until", 0.0) or 0.0)
            if notice and time.monotonic() < notice_until and w > 1040:
                notice_color = getattr(
                    self, "_youtube_link_notice_color", AMBER)
                nx1 = max(664, min(w - 760, w // 2 - 230))
                nx2 = min(w - 286, nx1 + 460)
                pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 7.0)
                self._round_rect(
                    topcv, nx1 - 3, 11, nx2 + 3, 43, 8,
                    fill="", outline=self._mix(BG, notice_color, 0.35 + pulse * 0.5),
                    width=2, tags="tb")
                self._round_rect(
                    topcv, nx1, 14, nx2, 40, 6,
                    fill=self._mix(SURFACE2, notice_color, 0.26),
                    outline=notice_color, width=2, tags="tb")
                topcv.create_text(
                    (nx1 + nx2) // 2, 27, text=notice,
                    fill=self._mix(notice_color, "#ffffff", 0.25),
                    font=("Segoe UI", 8, "bold"), tags="tb")
            camera_x1, camera_x2 = w - 274, w - 234
            camera_on = bool(self.camera_enabled)
            camera_outline = MINT if camera_on else RED
            camera_fill = self._mix(
                SURFACE2, camera_outline,
                0.22 if self._tb_hover == "camera" else 0.10)
            self._round_rect(
                topcv, camera_x1, 14, camera_x2, 40, 6,
                fill=camera_fill, outline=camera_outline, tags="tb")
            eye_cx = (camera_x1 + camera_x2) // 2
            topcv.create_oval(
                eye_cx - 10, 20, eye_cx + 10, 34,
                outline=camera_outline, width=2, tags="tb")
            topcv.create_oval(
                eye_cx - 3, 24, eye_cx + 3, 30,
                fill=camera_outline, outline="", tags="tb")
            if not camera_on:
                topcv.create_line(
                    eye_cx - 11, 18, eye_cx + 11, 36,
                    fill=RED, width=2, tags="tb")

            self._round_rect(topcv, w - 226, 14, w - 116, 40, 6, fill=SURFACE, outline=BORDER, tags="tb")
            topcv.create_text(w - 210, 27, text="Default Profile", anchor="w", fill=FG,
                              font=("Segoe UI", 8), tags="tb")
            topcv.create_text(w - 133, 27, text="v", anchor="w", fill=MUTED,
                              font=("Segoe UI", 8), tags="tb")
            bw, bh, gap = 31, 24, 8
            ex_x = w - 12 - bw
            mn_x = ex_x - gap - bw
            by = 15
            self._tb_buttons = {"min": (mn_x, by, mn_x + bw, by + bh),
                                "exit": (ex_x, by, ex_x + bw, by + bh),
                                "camera": (camera_x1, 14, camera_x2, 40)}
            if ready_x2 > ready_x1 + 80 and not ready_hits:
                self._tb_buttons["ready_speech"] = (ready_x1, 14, ready_x2, 40)
            for name, bounds in ready_hits:
                self._tb_buttons[name] = bounds
            self._draw_winbtn(topcv, "min", mn_x, by, bw, bh)
            self._draw_winbtn(topcv, "exit", ex_x, by, bw, bh)
            topcv.create_line(0, TBH - 1, w, TBH - 1, fill=BORDER, tags="tb")
            topcv.tag_raise(self._sweep)

        self._topdraw = _topdraw
        topcv.bind("<Configure>", _topdraw)
        topcv.bind("<Motion>", self._tb_motion)
        topcv.bind("<Button-1>", self._tb_press)
        topcv.bind("<B1-Motion>", self._tb_drag)
        topcv.bind("<ButtonRelease-1>", self._tb_release)

        # Main shell.
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg="#07090d", width=178)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        nav = [(ICONS["dashboard"], "Dashboard"), (ICONS["live"], "Live Control"),
               (ICONS["comments"], "Comments"), (ICONS["voice"], "Voice"),
               (ICONS["face"], "Face Swap"), (ICONS["lips"], "Lip Sync"),
               (ICONS["scenes"], "Scenes"), (ICONS["analytics"], "Analytics"),
               (ICONS["settings"], "Settings")]
        self._nav_items = {}
        for i, (ico, label) in enumerate(nav):
            active = label == "Live Control"
            r = tk.Frame(sidebar, bg="#11151d" if active else "#07090d",
                         highlightthickness=1 if active else 0,
                         highlightbackground=self._mix(MAG, BG, 0.35))
            r.pack(fill="x", padx=10, pady=(14 if i == 0 else 4, 0), ipady=9)
            r.configure(cursor="hand2")
            rail = tk.Frame(r, bg=MAG if active else r["bg"], width=3)
            rail.pack(side="left", fill="y")
            icon_lbl = tk.Label(r, text=ico, bg=r["bg"], fg=MAG if active else MUTED,
                                font=("Segoe MDL2 Assets", 13), width=4,
                                cursor="hand2")
            icon_lbl.pack(side="left", padx=(7, 2))
            text_lbl = tk.Label(r, text=label, bg=r["bg"], fg=MAG if active else FG,
                                font=("Segoe UI", 9), cursor="hand2")
            text_lbl.pack(side="left")
            self._nav_items[label] = {
                "frame": r, "rail": rail, "icon": icon_lbl, "label": text_lbl,
                "widgets": (r, rail, icon_lbl, text_lbl),
            }
            for w in (r, rail, icon_lbl, text_lbl):
                w.bind("<Button-1>", lambda _e, name=label: self._nav_go(name))
            if label == "Comments":
                self.nav_comments_badge = tk.Label(r, text="0", bg=RED, fg="#ffffff",
                                                   font=("Segoe UI", 7, "bold"),
                                                   padx=5, pady=1, cursor="hand2")
                self.nav_comments_badge.pack(side="right", padx=8)
                self._nav_items[label]["widgets"] = (
                    r, rail, icon_lbl, text_lbl, self.nav_comments_badge)
                self.nav_comments_badge.bind(
                    "<Button-1>", lambda _e, name=label: self._nav_go(name))
        tk.Frame(sidebar, bg="#07090d").pack(side="top", fill="both", expand=True)
        profile = tk.Frame(sidebar, bg="#0c1016",
                           highlightthickness=1, highlightbackground=BORDER)
        profile.pack(side="bottom", fill="x", padx=12, pady=12, ipady=8)
        self._thumb(profile, "Profile", MINT, live=True).pack(side="left", padx=(8, 4))
        tk.Label(profile, text=os.environ.get("USERNAME", "Local Profile"), bg="#0c1016", fg=FG,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(14, 0))
        tk.Label(profile, text="Local", bg="#111721", fg=CYAN,
                 font=("Segoe UI", 7), padx=5).pack(anchor="w", pady=(3, 0))

        main = tk.Frame(shell, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        content = tk.Frame(main, bg=BG)
        content.pack(fill="both", expand=True)

        left_col = tk.Frame(content, bg=BG, width=250)
        left_col.pack(side="left", fill="both", padx=(0, 10))
        left_col.pack_propagate(False)

        # Pack fixed rails before the expanding preview. Tk's packer allocates
        # space in declaration order; packing center first can starve this rail.
        right_col = tk.Frame(content, bg=BG, width=430)
        right_col.pack(side="right", fill="both")
        right_col.pack_propagate(False)

        center_col = tk.Frame(content, bg=BG)
        center_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_canvas = tk.Canvas(right_col, bg=BG, highlightthickness=0, bd=0)
        right_vsb = ttk.Scrollbar(right_col, orient="vertical",
                                  command=right_canvas.yview,
                                  style="Studio.Vertical.TScrollbar")
        right_canvas.configure(yscrollcommand=right_vsb.set)
        right_vsb.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        right = tk.Frame(right_canvas, bg=BG)
        right_win = right_canvas.create_window((0, 0), window=right, anchor="nw")
        self._right_canvas = right_canvas
        self._right_frame = right

        def _sync_right_scroll(_=None):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfigure(right_win, width=right_canvas.winfo_width())
        right.bind("<Configure>", _sync_right_scroll)
        right_canvas.bind("<Configure>", _sync_right_scroll)

        def _right_wheel(e):
            right_canvas.yview_scroll(int(-(e.delta or 0) / 120), "units")
            return "break"
        right_canvas.bind("<Enter>", lambda e: right_canvas.bind_all("<MouseWheel>", _right_wheel))
        right_canvas.bind("<Leave>", lambda e: right_canvas.unbind_all("<MouseWheel>"))
        self._right_wheel = _right_wheel

        # Comment Reader.
        comments = self._dash_panel(left_col, "Comment Reader", MAG, expand=True)
        self._comments_outer = getattr(comments, "_dash_outer", comments)
        top = tk.Frame(comments, bg=SURFACE); top.pack(fill="x")
        tk.Label(top, text="Auto-read", bg=SURFACE, fg=FG, font=("Segoe UI", 8)).pack(side="left")
        self.comments_var = tk.BooleanVar(value=True)
        self._check(top, "", self.comments_var, self._on_comments).pack(side="right")
        self._check(top, "Voice", self.comment_voice_var,
                    self._on_comment_voice).pack(side="right", padx=(0, 8))
        self.handle_var = tk.StringVar(value=os.environ.get("AVATAR_TIKTOK_USER", ""))
        self.handle_combo = ttk.Combobox(comments, textvariable=self.handle_var, width=18,
                                         values=self._handles, style="Studio.TCombobox",
                                         font=("Segoe UI", 8))
        self.handle_combo.pack(fill="x", pady=(8, 6))
        self.handle_combo.bind("<<ComboboxSelected>>", lambda e: self._on_handle_pick())
        self.handle_var.trace_add("write", lambda *_: setattr(
            self, "_handle_text", self.handle_var.get()))
        lang = tk.Frame(comments, bg=SURFACE); lang.pack(fill="x", pady=(0, 7))
        tk.Label(lang, text="Language", bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        ttk.Combobox(lang, values=["English (US)", "Arabic", "Auto"], state="readonly",
                     width=13, style="Studio.TCombobox").pack(side="right")
        self.feed_light = tk.Canvas(comments, width=1, height=1, bg=SURFACE, highlightthickness=0)
        self._feed_dot = self.feed_light.create_oval(0, 0, 1, 1, fill="#3a3f4a", outline="")
        self.feed_status = tk.Label(comments, text="no handle", bg=SURFACE, fg=MUTED,
                                    font=("Segoe UI", 8))
        self.feed_status.pack(anchor="w")
        self._answer_bar = tk.Frame(comments, bg=self._mix(SURFACE, MAG, 0.16))
        self._answer_bar.pack(fill="x", pady=(6, 5))
        self.answering_lbl = tk.Label(self._answer_bar, text="idle - waiting for a question",
                                      bg=self._answer_bar["bg"], fg=MUTED, font=("Segoe UI", 8),
                                      anchor="w", justify="left", wraplength=210)
        self.answering_lbl.pack(fill="x", padx=8, pady=5)
        fb = tk.Frame(comments, bg=SURFACE); fb.pack(fill="both", expand=True)
        fsb = tk.Scrollbar(fb); fsb.pack(side="right", fill="y")
        self.feed = tk.Text(fb, height=11, bg="#090e17", fg=FG, relief="flat", bd=0,
                            font=("Segoe UI", 8), wrap="word", padx=8, pady=6,
                            state="disabled", yscrollcommand=fsb.set)
        self.feed.pack(side="left", fill="both", expand=True)
        fsb.config(command=self.feed.yview)
        self.feed.tag_config("q", foreground=CYAN)
        self.feed.tag_config("a", foreground=MINT)
        self.feed.tag_config("ev", foreground=AMBER)
        self.feed.tag_config("sys", foreground=MUTED)
        self._feed_msg("enter your @handle and go live - real comments appear here.", "sys")
        graph = tk.Canvas(comments, height=36, bg=SURFACE, highlightthickness=0)
        graph.pack(fill="x", pady=(8, 0))
        graph.create_text(2, 8, text="Comments / min", anchor="w", fill=MUTED, font=("Segoe UI", 7))
        self.comments_min_item = graph.create_text(2, 27, text="0", anchor="w", fill=FG,
                                                   font=("Segoe UI", 10, "bold"))
        self.comments_graph = graph

        # Live Preview.
        preview_panel = self._dash_panel(center_col, "Live Preview", CYAN, expand=True)
        controls = tk.Frame(preview_panel, bg=SURFACE); controls.pack(fill="x", pady=(0, 8))
        self.status_canvas = tk.Canvas(controls, width=22, height=22, bg=SURFACE, highlightthickness=0)
        self.status_canvas.pack(side="left")
        self.status_glow = self.status_canvas.create_oval(3, 3, 19, 19, outline=SURFACE, width=2)
        self.status_dot = self.status_canvas.create_oval(7, 7, 15, 15, fill=RED, outline="")
        self.status_lbl = tk.Label(controls, text="OFFLINE", bg=SURFACE, fg=FG,
                                   font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(side="left", padx=(6, 12))
        self.info_lbl = tk.Label(controls, text="benchmarking GPU...", bg=SURFACE, fg=AMBER,
                                 font=("Segoe UI", 8))
        self.info_lbl.pack(side="left")
        self.bench_bar = ttk.Progressbar(controls, mode="indeterminate", length=95)
        self.bench_bar.pack(side="left", padx=(8, 0))
        self.bench_bar.start(14)
        self.fps_lbl = tk.Label(controls, text="", bg=SURFACE, fg=CYAN,
                                font=("Consolas", 9))
        self.fps_lbl.pack(side="right")
        tk.Label(controls, text=f"{TIKTOK_PORTRAIT_W}x{TIKTOK_PORTRAIT_H}", bg=SURFACE2, fg=MUTED,
                 font=("Segoe UI", 8), padx=10, pady=3,
                 highlightthickness=1, highlightbackground=BORDER).pack(side="right", padx=(0, 8))
        tk.Label(controls, text="TikTok portrait", bg=SURFACE2, fg=MUTED,
                 font=("Segoe UI", 8), padx=10, pady=3,
                 highlightthickness=1, highlightbackground=BORDER).pack(side="right", padx=(0, 6))
        self.live_light = tk.Canvas(controls, width=18, height=18, bg=SURFACE, highlightthickness=0)
        self._live_glow = self.live_light.create_oval(1, 1, 17, 17, fill="", outline="")
        self._live_dot = self.live_light.create_oval(5, 5, 13, 13, fill="#3a3f4a", outline="")
        self.live_light.pack(side="right", padx=(0, 10))

        stage_wrap = tk.Frame(
            preview_panel, bg="#030406", highlightthickness=1,
            highlightbackground="#202633")
        stage_wrap.pack(fill="both", expand=True)
        stage = tk.Frame(stage_wrap, bg="#000000", width=PREVIEW_W, height=PREVIEW_H)
        stage.place(relx=0.5, rely=0.5, anchor="center")
        stage.pack_propagate(False)
        self.preview_stage = stage
        self.preview_stage_wrap = stage_wrap
        self._preview_draw_size = (PREVIEW_W, PREVIEW_H)
        self.preview = tk.Label(stage, bg="#000000", bd=0)
        self.preview.pack(fill="both", expand=True)
        overlay = tk.Frame(stage_wrap, bg="#080b10")
        overlay.place(relx=0.5, rely=0.98, anchor="s", width=PREVIEW_W, height=38)
        self.preview_overlay = overlay
        self.diag_lbl = tk.Label(overlay, text="// ready", bg="#080b10",
                                 fg=self._mix(CYAN, BG, 0.25), font=("Consolas", 8),
                                 anchor="w")
        self.diag_lbl.pack(fill="both", padx=10)
        stage_wrap.bind("<Configure>", self._resize_preview_stage)
        self._show_placeholder()
        self.root.after(1000, self._youtube_clock_tick)

        # Large, continuously refreshed screen-region monitor below the avatar.
        self.scene_slot = tk.Frame(
            preview_panel, bg="#07090d", height=260)
        self.scene_slot.pack(fill="x", pady=(8, 0))
        self.scene_slot.pack_propagate(False)
        self._build_sidebar_scene_slot()

        # Quick Actions.
        actions = tk.Frame(center_col, bg="#090c11", highlightthickness=1,
                           highlightbackground="#202632")
        actions.pack(fill="x", pady=(10, 0), ipady=8)
        self.start_btn = self._btn(actions, f"{ICONS['play']}  Go Live", self.start, bg=MINT, fg=MINT_INK,
                                   hover=self._mix(MINT, "#ffffff", 0.18), border=MINT,
                                   font=("Segoe UI", 9, "bold"))
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(10, 5), ipady=6)
        self.stop_btn = self._btn(actions, f"{ICONS['stop']}  Emergency Stop", self.stop,
                                  bg=self._mix(SURFACE2, RED, 0.16), fg=RED,
                                  hover=self._mix(SURFACE2, RED, 0.26), border=RED,
                                  font=("Segoe UI", 9, "bold"), state="disabled")
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=(5, 10), ipady=6)
        self.mouth_btn = tk.Button(actions, text=f"{ICONS['lips']}  Lip Sync", command=self._toggle_ai_mouth,
                                   bg=SURFACE2, fg=MINT, relief="flat", bd=0,
                                   font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.mouth_btn.pack(side="left", fill="x", expand=True, padx=5, ipady=6)
        self.youtube_smooth_btn = tk.Button(
            actions, text="Smooth Voice", command=self._toggle_youtube_smooth,
            bg=self._mix(SURFACE2, MINT, 0.18), fg=MINT, relief="flat", bd=0,
            font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.youtube_smooth_btn.pack(side="left", fill="x", expand=True, padx=5, ipady=6)
        self.viewers_btn = tk.Button(
            actions, text=f"{ICONS['viewers']}  Viewers",
            command=self._speak_top_viewers,
            bg=self._mix(SURFACE2, MAG, 0.18), fg=FG, relief="flat", bd=0,
            font=("Segoe UI", 9, "bold"), cursor="hand2",
            activebackground=self._mix(SURFACE2, MAG, 0.30),
            activeforeground=FG)
        self.viewers_btn.pack(side="left", fill="x", expand=True, padx=5, ipady=6)
        audio_mutes = tk.Frame(center_col, bg="#090c11", highlightthickness=1,
                               highlightbackground="#202632")
        audio_mutes.pack(fill="x", pady=(8, 0))
        tk.Label(audio_mutes, text="AUDIO", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 7, "bold"), width=7).pack(
                     side="left", fill="y", padx=(4, 2), pady=7)
        audio_channels = tk.Frame(audio_mutes, bg=SURFACE)
        audio_channels.pack(side="left", fill="x", expand=True, padx=(0, 5), pady=5)
        for column in range(4):
            audio_channels.grid_columnconfigure(column, weight=1, uniform="audio")
        self.speech_btn = self._audio_source_control(
            audio_channels, "ai", self._toggle_speech, column=0)
        self.youtube_mute_btn = self._audio_source_control(
            audio_channels, "youtube", self._toggle_youtube_mute, column=1)
        self.music_btn = self._audio_source_control(
            audio_channels, "music", self._toggle_music, column=2)
        self.mic_mute_btn = self._audio_source_control(
            audio_channels, "mic", self._toggle_mic_monitor_mute, column=3)
        self._sync_audio_mute_buttons()
        self._sync_youtube_smooth_button()

        # Right control grid.
        voice = self._dash_panel(right, "AI Voice", MAG, fill="x", pady=(0, 8))
        _auto_tts = os.environ.get("AVATAR_TTS", "")
        _def_label = next((lbl for lbl, key in VOICE_MODES if key == _auto_tts), VOICE_MODE_LABELS[0])
        self.voicemode_var = tk.StringVar(value=_def_label)
        ttk.Combobox(voice, textvariable=self.voicemode_var, values=VOICE_MODE_LABELS,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(0, 6))
        self.voicemode_var.trace_add("write", self._on_voice_mode)
        self.voice_var = tk.StringVar(value=MALE_VOICES[0])
        ttk.Combobox(voice, textvariable=self.voice_var, values=MALE_VOICES,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(0, 6))
        self.voice_var.trace_add("write", self._on_voice)
        self.gaze_var = tk.BooleanVar(value=True)
        self.gaze_var2 = tk.IntVar(value=55)
        self._range_row(voice, "Speed", self.gaze_var2, lambda e: self._on_gaze())
        self.skin_var = tk.IntVar(value=70)
        self._range_row(voice, "Pitch", self.skin_var, lambda e: self._on_skin())
        self.speak_btn = self._btn(voice, f"{ICONS['voice']}  Speak Test", self.speak,
                                   bg=self._mix(SURFACE2, MAG, 0.22),
                                   fg=FG, hover=self._mix(SURFACE2, MAG, 0.34),
                                   border=self._mix(MAG, BG, 0.35), font=("Segoe UI", 8, "bold"),
                                   state="disabled")
        self.speak_btn.pack(fill="x", ipady=5, pady=(4, 0))

        fs = self._dash_panel(right, "Face Swap", CYAN, fill="x", pady=(0, 8))
        thumbs = tk.Frame(fs, bg=SURFACE); thumbs.pack(fill="x", pady=(0, 6))
        self._thumb(thumbs, "Source Face", CYAN, live=True).pack(side="left")
        self._thumb(thumbs, "Target Avatar", MAG).pack(side="right")
        self.swap_var = tk.BooleanVar(value=True)
        self.char_var = tk.StringVar(value="White Haddan")
        ttk.Combobox(fs, textvariable=self.char_var,
                     values=["White Haddan", "Haddan", "White man"], state="readonly",
                     style="Studio.TCombobox").pack(fill="x", pady=(0, 5))
        self.char_var.trace_add("write", self._on_character)
        self.skintone_var = tk.IntVar(value=50)
        self._range_row(fs, "Strength", self.skintone_var, lambda e: self._on_skintone())
        brow = tk.Frame(fs, bg=SURFACE); brow.pack(fill="x", pady=(2, 0))
        self.char_btn = self._btn(brow, f"{ICONS['play']}  Running", self._load_character,
                                  bg=self._mix(SURFACE2, MINT, 0.15),
                                  fg=MINT, hover=self._mix(SURFACE2, MINT, 0.25),
                                  border=self._mix(MINT, BG, 0.4), font=("Segoe UI", 8, "bold"))
        self.char_btn.pack(side="left", fill="x", expand=True, ipady=5)
        self.recenter_btn = self._btn(brow, f"{ICONS['stop']}  Stop", self.recenter,
                                      bg=self._mix(SURFACE2, RED, 0.15),
                                      fg=RED, hover=self._mix(SURFACE2, RED, 0.25),
                                      border=self._mix(RED, BG, 0.4), font=("Segoe UI", 8, "bold"),
                                      state="disabled")
        self.recenter_btn.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=5)

        lip = self._dash_panel(right, "Lip Sync", MAG, fill="x", pady=(0, 8))
        self.livemic_var = tk.BooleanVar(value=False)
        try:
            from voice_changer_engine import list_input_devices
            _mics = [f"{i}: {n}" for i, n in list_input_devices()]
        except Exception:
            _mics = []
        self.micdev_var = tk.StringVar(value=(_mics[0] if _mics else "default"))
        ttk.Combobox(lip, textvariable=self.micdev_var, values=(_mics or ["default"]),
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(0, 7))
        self.micdev_var.trace_add("write", self._on_micdev)
        meter = tk.Canvas(lip, height=20, bg=SURFACE, highlightthickness=0)
        meter.pack(fill="x")
        for i in range(26):
            col = MINT if i < 19 else AMBER if i < 23 else RED
            meter.create_rectangle(4 + i * 8, 4, 9 + i * 8, 18, fill=col, outline="")
        self.liplock_var = tk.BooleanVar(value=True)
        self.ai_mouth_var.set(True)
        self.micgain_var = tk.DoubleVar(value=float(os.environ.get("AVATAR_MIC_GAIN", "5.0")))
        self._range_row(lip, "Sync intensity", self.micgain_var, self._on_micgain, frm=1.0, to=12.0)
        self._mini_row(lip, "Latency", "120 ms", MINT)

        out = self._dash_panel(right, "Stream Output", MINT, fill="x", pady=(0, 8))
        self._mini_row(out, "TikTok", "Disconnected", MUTED)
        self._mini_row(out, "OBS Virtual Camera", "Off", MUTED)
        self._mini_row(out, "Resolution", f"{TIKTOK_PORTRAIT_W}x{TIKTOK_PORTRAIT_H} portrait")
        self._mini_row(out, "FPS", "0 FPS")
        self._mini_row(out, "Video Bitrate", "preview only")
        self._mini_row(out, "Audio Bitrate", "TTS off")
        self._mini_row(out, "Output", "idle", MUTED)

        # Full operational controls from the previous studio layout.
        self.quality_var = tk.StringVar(value="Delulu (recommended)")
        self.interval_var = tk.IntVar(value=2)
        self.stab_var = tk.IntVar(value=20)
        self.minface_var = tk.IntVar(value=6)
        self.pose_var = tk.StringVar(value="Safe (no melt)")
        self.turncap_var = tk.IntVar(value=30)
        self.tilt_var = tk.IntVar(value=10)
        self.autotalk_var = tk.BooleanVar(value=True)
        self.restore_var = tk.BooleanVar(value=True)
        self.body_var = tk.BooleanVar(value=True)
        self.music_var = tk.BooleanVar(value=True)
        self.multiref_var = tk.BooleanVar(value=False)
        self.hair_var = tk.StringVar(value="gray")
        self.eye_var = tk.StringVar(value="gray")
        from trading_backgrounds import BACKGROUND_PRESETS
        self.background_on_var = tk.BooleanVar(value=True)
        self.background_var = tk.StringVar(value="Wall Street LED / Midnight Blue")
        self.chart_var = tk.BooleanVar(value=False)
        self.trader_var = tk.BooleanVar(value=False)
        self.broadcast_var = tk.BooleanVar(value=True)
        self.perf_var = tk.BooleanVar(value=True)
        self.obs_var = tk.BooleanVar(value=False)
        self.VC_MODES = [("Persona voice (RVC)", "rvc"),
                         ("Pitch / formant (DSP)", "dsp"),
                         ("Passthrough (no change)", "passthrough")]
        self.vcmode_var = tk.StringVar(value=self.VC_MODES[0][0])
        self._sync_audio_mute_buttons()

        session = self._dash_panel(right, "Session & Performance", CYAN, fill="x", pady=(0, 8))
        self._mini_row(session, "Quality preset", "")
        ttk.Combobox(session, textvariable=self.quality_var, values=QUALITY_LABELS,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(0, 5))
        self.quality_var.trace_add("write", self._on_quality)
        self._mini_row(session, "Head update", "")
        ttk.Spinbox(session, from_=1, to=4, width=5, textvariable=self.interval_var,
                    command=self._on_interval, style="Studio.TSpinbox").pack(fill="x", pady=(0, 5))
        self._range_row(session, "Stabilization", self.stab_var, lambda e: self._on_stab())
        self._mini_row(session, "Min face size", "")
        ttk.Spinbox(session, from_=4, to=40, increment=2, width=5,
                    textvariable=self.minface_var, command=self._on_minface,
                    style="Studio.TSpinbox").pack(fill="x", pady=(0, 5))
        ttk.Combobox(session, textvariable=self.pose_var, values=POSE_LABELS,
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(0, 5))
        self.pose_var.trace_add("write", self._on_pose)
        pose_row = tk.Frame(session, bg=SURFACE); pose_row.pack(fill="x")
        ttk.Spinbox(pose_row, from_=20, to=90, increment=5, width=8,
                    textvariable=self.turncap_var, command=self._on_turncap,
                    style="Studio.TSpinbox").pack(side="left", fill="x", expand=True)
        ttk.Spinbox(pose_row, from_=8, to=30, increment=1, width=8,
                    textvariable=self.tilt_var, command=self._on_tilt,
                    style="Studio.TSpinbox").pack(side="left", fill="x", expand=True, padx=(8, 0))

        real = self._dash_panel(right, "Realism", MAG, fill="x", pady=(0, 8))
        self._check(real, "Auto-talk AI host", self.autotalk_var, self._on_autotalk).pack(fill="x", pady=2)
        self._check(real, "Face restoration GFPGAN", self.restore_var).pack(fill="x", pady=2)
        self._check(real, "Live body motion", self.body_var).pack(fill="x", pady=2)
        self._check(real, "Background music", self.music_var, self._on_music).pack(fill="x", pady=2)
        self._check(real, "Extended turning", self.multiref_var, self._on_multiref).pack(fill="x", pady=2)
        self._check(real, "Face-swap mode", self.swap_var, self._on_swap).pack(fill="x", pady=2)
        self._check(real, "Voice-driven mouth", self.liplock_var, self._on_liplock).pack(fill="x", pady=2)
        self._range_row(real, "Skin detail", self.skin_var, lambda e: self._on_skin())
        self._range_row(real, "Skin tone", self.skintone_var, lambda e: self._on_skintone())
        ttk.Combobox(real, textvariable=self.hair_var,
                     values=["brown", "black", "blonde", "gray", "none"],
                     state="readonly", style="Studio.TCombobox").pack(fill="x", pady=(5, 4))
        self.hair_var.trace_add("write", self._on_hair)
        ttk.Combobox(real, textvariable=self.eye_var,
                     values=["off", "blue", "green", "hazel", "brown", "amber", "gray"],
                     state="readonly", style="Studio.TCombobox").pack(fill="x")
        self.eye_var.trace_add("write", self._on_eye)

        scene = self._dash_panel(right, "Scene & Output", MINT, fill="x", pady=(0, 8))
        self._check(scene, "Trading background", self.background_on_var,
                    self._on_background_toggle).pack(fill="x", pady=2)
        self.background_combo = ttk.Combobox(scene, textvariable=self.background_var,
                                             values=BACKGROUND_PRESETS, state="readonly",
                                             style="Studio.TCombobox")
        self.background_combo.pack(fill="x", pady=(2, 6))
        self.background_var.trace_add("write", self._on_background)
        self._check(scene, "Show live charts when face is lost", self.chart_var).pack(fill="x", pady=2)
        self._check(scene, "Trader scene chart + avatar PiP", self.trader_var).pack(fill="x", pady=2)
        self._check(scene, "Broadcast framing", self.broadcast_var).pack(fill="x", pady=2)
        self._check(scene, "Low-lag scene mode", self.low_lag_scene_var).pack(fill="x", pady=2)
        self._mini_row(scene, "Bottom face length", "")
        ttk.Combobox(scene, textvariable=self.face_strip_var,
                     values=FACE_STRIP_LABELS, state="readonly",
                     style="Studio.TCombobox").pack(fill="x", pady=(0, 6))
        self.face_strip_var.trace_add("write", self._on_face_strip_length)
        self._check(scene, "Show CPU/GPU monitor", self.perf_var).pack(fill="x", pady=2)
        self._check(scene, "Send to OBS virtual camera", self.obs_var).pack(fill="x", pady=2)
        ttk.Combobox(scene, textvariable=self.vcmode_var,
                     values=[m[0] for m in self.VC_MODES], state="readonly",
                     style="Studio.TCombobox").pack(fill="x", pady=(6, 0))
        self.vcmode_var.trace_add("write", self._on_vcmode)

        ask_panel = self._dash_panel(right, "Ask The Avatar", MAG, fill="x", pady=(0, 8))
        self.ask_entry = tk.Text(ask_panel, height=2, bg=SURFACE2, fg=FG,
                                 insertbackground=MAG, font=("Segoe UI", 10),
                                 relief="flat", wrap="word", padx=8, pady=6,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 highlightcolor=MAG)
        self.ask_entry.pack(fill="x", pady=(0, 6))
        self.ask_entry.bind("<Return>", self._on_ask_enter)
        self.ask_btn = self._btn(ask_panel, f"{ICONS['comments']}  Ask Avatar", self.ask,
                                 bg=MAG, fg=CYAN_INK,
                                 hover=MAG_HI, border=MAG, font=("Segoe UI", 9, "bold"),
                                 state="disabled")
        self.ask_btn.pack(fill="x", ipady=5)

        speak_panel = self._dash_panel(right, "Speak", CYAN, fill="x", pady=(0, 8))
        self.entry = tk.Text(speak_panel, height=3, bg=SURFACE2, fg=FG,
                             insertbackground=CYAN, font=("Segoe UI", 10),
                             relief="flat", wrap="word", padx=8, pady=6,
                             highlightthickness=1, highlightbackground=BORDER,
                             highlightcolor=CYAN)
        self.entry.pack(fill="x", pady=(0, 6))
        self.entry.bind("<Return>", self._on_enter)
        speak_row = tk.Frame(speak_panel, bg=SURFACE); speak_row.pack(fill="x")
        self.speak_btn = self._btn(speak_row, f"{ICONS['voice']}  Speak", self.speak,
                                   bg=CYAN, fg=CYAN_INK,
                                   hover=CYAN_HI, border=CYAN, font=("Segoe UI", 9, "bold"),
                                   state="disabled")
        self.speak_btn.pack(side="left", fill="x", expand=True, ipady=5)
        self.mute_btn = tk.Button(speak_row, text=f"{ICONS['mute']}  Mute", command=self.toggle_mute,
                                  bg=SURFACE2, fg=MUTED, font=("Segoe UI", 9, "bold"),
                                  relief="flat", bd=0, cursor="hand2", state="disabled",
                                  activebackground=self._mix(SURFACE2, RED, 0.16),
                                  highlightthickness=1,
                                  highlightbackground=self._mix(MUTED, BG, 0.45))
        self.mute_btn.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=5)
        for t in QUICK_PHRASES:
            self._chip(speak_panel, t[:42] + ("..." if len(t) > 42 else ""),
                       lambda x=t: self._speak_text(x), full=True).pack(fill="x", pady=(6, 0))

        youtube_panel = self._dash_panel(right, "YouTube Speak", AMBER, fill="x", pady=(0, 8))
        tk.Label(
            youtube_panel, text="Add YouTube links in order. Video 2 starts when Video 1 ends.",
            bg=SURFACE, fg=FAINT, font=("Segoe UI", 8),
            wraplength=300, justify="left").pack(anchor="w", pady=(0, 4))
        self.youtube_entries = []
        self.youtube_status_vars = []
        for i in range(10):
            item = tk.Frame(youtube_panel, bg=SURFACE)
            item.pack(fill="x", pady=(0, 5))
            row = tk.Frame(item, bg=SURFACE)
            row.pack(fill="x")
            tk.Label(row, text=f"YOUTUBE {i + 1}", bg=SURFACE, fg=FAINT,
                     font=("Consolas", 8, "bold"), width=10,
                     anchor="w").pack(side="left")
            entry = tk.Entry(
                row, bg=SURFACE2, fg=FG, insertbackground=AMBER,
                font=("Segoe UI", 9), relief="flat",
                highlightthickness=1, highlightbackground=BORDER,
                highlightcolor=AMBER)
            entry.pack(side="left", fill="x", expand=True, ipady=4)
            entry.bind("<Return>", self._on_youtube_enter)
            entry.bind("<<Paste>>", self._on_youtube_link_changed)
            entry.bind("<KeyRelease>", self._on_youtube_link_changed)
            self.youtube_entries.append(entry)
            status_var = tk.StringVar(value="EMPTY")
            self.youtube_status_vars.append(status_var)
            tk.Label(
                item, textvariable=status_var, bg=SURFACE, fg=FAINT,
                font=("Consolas", 7), anchor="w").pack(
                    fill="x", padx=(78, 0), pady=(1, 0))
        self.youtube_entry = self.youtube_entries[0]
        self.youtube_persona_var = tk.StringVar(value=YOUTUBE_PERSONA_LABELS[0])
        persona_combo = ttk.Combobox(
            youtube_panel, textvariable=self.youtube_persona_var,
            values=YOUTUBE_PERSONA_LABELS, state="readonly",
            style="Studio.TCombobox",
        )
        persona_combo.pack(fill="x", pady=(0, 6))
        persona_combo.bind("<<ComboboxSelected>>", self._on_youtube_persona_change)
        yr = tk.Frame(youtube_panel, bg=SURFACE); yr.pack(fill="x", pady=(0, 6))
        self.youtube_from_var = tk.StringVar(value="")
        self.youtube_to_var = tk.StringVar(value="")
        tk.Entry(yr, textvariable=self.youtube_from_var, width=8, bg=SURFACE2,
                 fg=FG, insertbackground=AMBER, relief="flat").pack(side="left", fill="x", expand=True)
        tk.Entry(yr, textvariable=self.youtube_to_var, width=8, bg=SURFACE2,
                 fg=FG, insertbackground=AMBER, relief="flat").pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.youtube_btn = self._btn(youtube_panel, f"{ICONS['play']}  Speak YouTube", self.speak_youtube,
                                     bg=AMBER, fg=CYAN_INK, hover=self._mix(AMBER, "#ffffff", 0.18),
                                     border=AMBER, font=("Segoe UI", 9, "bold"), state="disabled")
        self.youtube_btn.pack(fill="x", ipady=5)
        self.youtube_audio_btn = self._btn(youtube_panel, f"{ICONS['voice']}  Alter Real Voice", self.speak_youtube_audio,
                                           bg=SURFACE2, fg=AMBER,
                                           hover=self._mix(SURFACE2, AMBER, 0.2),
                                           border=AMBER, font=("Segoe UI", 9, "bold"),
                                           state="disabled")
        self.youtube_audio_btn.pack(fill="x", ipady=5, pady=(6, 0))
        jump_row = tk.Frame(youtube_panel, bg=SURFACE)
        jump_row.pack(fill="x", pady=(6, 0))
        self.youtube_back_btn = self._btn(
            jump_row, f"{ICONS['play']}  Back", self.youtube_previous_video,
            bg=SURFACE2, fg=AMBER, hover=self._mix(SURFACE2, AMBER, 0.2),
            border=AMBER, font=("Segoe UI", 8, "bold"),
            state="disabled")
        self.youtube_back_btn.pack(side="left", fill="x", expand=True, ipady=4)
        self.youtube_next_btn = self._btn(
            jump_row, f"{ICONS['play']}  Next", self.youtube_next_video,
            bg=SURFACE2, fg=AMBER, hover=self._mix(SURFACE2, AMBER, 0.2),
            border=AMBER, font=("Segoe UI", 8, "bold"),
            state="disabled")
        self.youtube_next_btn.pack(side="left", fill="x", expand=True,
                                   padx=(8, 0), ipady=4)
        ystatus = tk.Frame(youtube_panel, bg=SURFACE); ystatus.pack(fill="x", pady=(7, 0))
        self.youtube_light = tk.Canvas(ystatus, width=18, height=18, bg=SURFACE, highlightthickness=0)
        self.youtube_light_dot = self.youtube_light.create_oval(4, 4, 14, 14, fill="#3a3f4a", outline="")
        self.youtube_light.pack(side="left")
        self.youtube_status_lbl = tk.Label(ystatus, text="MARKET MODE", bg=SURFACE, fg=MUTED,
                                           font=("Segoe UI", 8, "bold"))
        self.youtube_status_lbl.pack(side="left", padx=(6, 0))
        self.youtube_time_lbl = tk.Label(youtube_panel, text="YOUTUBE TIME 00:00",
                                         bg=SURFACE, fg=FAINT, font=("Consolas", 8))
        self.youtube_time_lbl.pack(anchor="w", pady=(4, 0))
        self.youtube_progress = tk.Canvas(
            youtube_panel, height=18, bg=SURFACE, highlightthickness=0, bd=0)
        self.youtube_progress.pack(fill="x", pady=(5, 0))
        self._youtube_progress_display = 0.0
        self._youtube_progress_after = None
        self.youtube_progress.bind(
            "<Configure>", lambda _e: self._draw_youtube_progress())
        self._draw_youtube_progress()
        self.youtube_progress_lbl = tk.Label(
            youtube_panel, text="Idle", bg=SURFACE, fg=FAINT,
            font=("Consolas", 8))
        self.youtube_progress_lbl.pack(anchor="w", pady=(2, 0))
        ybuttons = tk.Frame(youtube_panel, bg=SURFACE); ybuttons.pack(fill="x", pady=(6, 0))
        self.youtube_resume_btn = self._btn(ybuttons, f"{ICONS['play']}  YouTube", self.resume_youtube,
                                            bg=SURFACE2, fg=AMBER, hover=self._mix(SURFACE2, AMBER, 0.2),
                                            border=AMBER, font=("Segoe UI", 8, "bold"),
                                            state="disabled")
        self.youtube_resume_btn.pack(side="left", fill="x", expand=True, ipady=4)
        self.market_mode_btn = self._btn(ybuttons, f"{ICONS['analytics']}  Market", self.resume_market,
                                         bg=SURFACE2, fg=MINT, hover=self._mix(SURFACE2, MINT, 0.2),
                                         border=MINT, font=("Segoe UI", 8, "bold"),
                                         state="disabled")
        self.market_mode_btn.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=4)
        self._sync_youtube_buttons()

        log_panel = self._dash_panel(right, "Activity Log", CYAN, fill="x", pady=(0, 8))
        self.log = tk.Text(log_panel, height=8, bg=SURFACE2, fg=MUTED, relief="flat",
                           font=("Consolas", 8), wrap="word", state="disabled",
                           padx=8, pady=6, highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=BORDER)
        self.log.pack(fill="both", expand=True)
        self._bind_mousewheel_tree(right, _right_wheel)
        self._nav_targets = {
            "Live Control": voice,
            "Voice": voice,
            "Face Swap": fs,
            "Lip Sync": lip,
            "Scenes": scene,
            "Analytics": log_panel,
            "Settings": session,
        }

        # Hidden/compact controls and variables still used by the engine.
        hidden = tk.Frame(main, bg=BG)

        # Bottom analytics strip.
        strip = tk.Frame(main, bg="#090c11", highlightthickness=1,
                         highlightbackground="#202632")
        strip.pack(fill="x", pady=(10, 0), ipady=8)
        metrics = [("Viewers", "0", "", MINT), ("Likes", "0", "", MINT),
                   ("Comments / min", "0", "", MINT), ("CPU", "0%", "", CYAN),
                   ("GPU", "0%", "", MAG), ("VRAM", "0%", "", CYAN),
                   ("Uptime", "00:00:00", "", MAG)]
        for column in range(len(metrics)):
            strip.grid_columnconfigure(column, weight=1, uniform="metrics")
        for column, (label, val, chg, acc) in enumerate(metrics):
            self._metric_tile(strip, label, val, chg, acc).grid(
                row=0, column=column, sticky="nsew", padx=4)

    # -------------------------------------------------------------------------
    # PREVIEW / UI REFRESH (Tk main thread only)
    # -------------------------------------------------------------------------
    def _format_duration(self, seconds):
        seconds = max(0, int(seconds or 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _uptime_seconds(self):
        if not self.running or self._session_started_at is None:
            return 0
        return time.monotonic() - self._session_started_at

    def _comments_per_minute(self):
        now = time.time()
        while self._comment_times and now - self._comment_times[0] > 60:
            self._comment_times.popleft()
        return len(self._comment_times)

    def _set_metric(self, label, value, change=""):
        value = str(value)
        lbl = getattr(self, "_metric_value_labels", {}).get(label)
        cl = getattr(self, "_metric_change_labels", {}).get(label)
        if lbl is None:
            return

        targets = getattr(self, "_metric_targets", {})
        previous_target = targets.get(label)
        if previous_target == value:
            if change and cl is not None:
                cl.configure(text=str(change))
            return
        targets[label] = value

        previous_number = self._metric_number(previous_target)
        next_number = self._metric_number(value)
        engagement = label in ("Viewers", "Likes", "Comments / min")
        if engagement and previous_number is not None and next_number is not None:
            delta = next_number - previous_number
            self._animate_metric_count(
                label, previous_number, next_number,
                suffix="%" if value.endswith("%") else "")
            if cl is not None:
                cl.configure(text=f"+{delta:,}" if delta > 0 else "")
        else:
            lbl.configure(text=value)
            if cl is not None:
                cl.configure(text=str(change or ""))

        if previous_target is not None and label != "Uptime":
            self._pulse_metric(label, strong=engagement)

    @staticmethod
    def _metric_number(value):
        if value is None:
            return None
        text = str(value).strip().replace(",", "").replace("%", "")
        if not text or text.lower() == "n/a" or ":" in text:
            return None
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None

    def _animate_metric_count(self, label, start, end, suffix=""):
        """Count smoothly to a new engagement total without blocking Tk."""
        token = self._metric_anim_tokens.get(label, 0) + 1
        self._metric_anim_tokens[label] = token
        steps = 10

        def tick(step=1):
            if self._metric_anim_tokens.get(label) != token:
                return
            eased = 1.0 - (1.0 - step / steps) ** 3
            current = int(round(start + (end - start) * eased))
            lbl = self._metric_value_labels.get(label)
            if lbl is not None:
                lbl.configure(text=f"{current:,}{suffix}")
            if step < steps:
                self.root.after(28, lambda: tick(step + 1))

        tick()

    def _pulse_metric(self, label, strong=False):
        """Flash a tile and kick its sparkline when a real value changes."""
        tile = getattr(self, "_metric_tiles", {}).get(label)
        lbl = getattr(self, "_metric_value_labels", {}).get(label)
        spark = getattr(self, "_metric_sparks", {}).get(label)
        accent = getattr(self, "_metric_accents", {}).get(label, CYAN)
        if tile is None or lbl is None or spark is None:
            return
        token = self._metric_anim_tokens.get(f"pulse:{label}", 0) + 1
        self._metric_anim_tokens[f"pulse:{label}"] = token
        canvas, items, points = spark
        frames = 12 if strong else 7

        def frame(index=0):
            if self._metric_anim_tokens.get(f"pulse:{label}") != token:
                return
            phase = index / max(1, frames - 1)
            energy = math.sin(math.pi * phase)
            border = self._mix("#151a23", accent, energy * (0.9 if strong else 0.5))
            tile.configure(highlightbackground=border)
            lbl.configure(
                fg=self._mix(FG, accent, energy * (0.72 if strong else 0.35)),
                font=("Segoe UI", 15 if strong and energy > 0.45 else 13, "bold"))
            for i, item in enumerate(items):
                x1, base_y1 = points[i]
                x2, base_y2 = points[i + 1]
                wave = math.sin((i + index) * 1.35) * energy
                lift = (7 if strong else 3) * wave
                canvas.coords(item, x1, base_y1 - lift, x2, base_y2 - lift)
                canvas.itemconfigure(
                    item, fill=self._mix(accent, "#ffffff", energy * 0.45),
                    width=2 if energy > 0.35 else 1)
            if index < frames - 1:
                self.root.after(42, lambda: frame(index + 1))
            else:
                tile.configure(highlightbackground="#151a23")
                lbl.configure(fg=FG, font=("Segoe UI", 13, "bold"))
                for i, item in enumerate(items):
                    canvas.coords(item, *points[i], *points[i + 1])
                    canvas.itemconfigure(item, fill=accent, width=1)
                change_lbl = self._metric_change_labels.get(label)
                if change_lbl is not None:
                    self.root.after(850, lambda: change_lbl.configure(text=""))

        frame()

    def _set_row_value(self, label, value, color=None):
        lbl = getattr(self, "_row_value_labels", {}).get(label)
        if lbl is not None:
            kw = {"text": str(value)}
            if color:
                kw["fg"] = color
            lbl.configure(**kw)

    def _update_live_dashboard(self):
        cpm = self._comments_per_minute()
        viewers = getattr(self, "_sess_viewers", None)
        self._set_metric(
            "Viewers", f"{int(viewers):,}" if viewers is not None else "n/a")
        self._set_metric("Likes", f"{int(getattr(self, '_sess_likes', 0)):,}")
        self._set_metric("Comments / min", str(cpm))
        self._set_metric("Uptime", self._format_duration(self._uptime_seconds()))
        if getattr(self, "comments_min_item", None) is not None:
            self.comments_graph.itemconfigure(self.comments_min_item, text=str(cpm))
            if getattr(self, "nav_comments_badge", None) is not None:
                self.nav_comments_badge.configure(text=str(cpm))
            self.comments_graph.delete("spark")
            now = time.time()
            buckets = []
            for i in range(8):
                lo = now - (8 - i) * 7.5
                hi = lo + 7.5
                buckets.append(sum(1 for t in self._comment_times if lo <= t < hi))
            maxv = max(1, max(buckets or [0]))
            pts = []
            for i, v in enumerate(buckets):
                x = 96 + i * 16
                y = 30 - int((v / maxv) * 22)
                pts.append((x, y))
            for i in range(len(pts) - 1):
                self.comments_graph.create_line(*pts[i], *pts[i + 1], fill=MAG,
                                                width=2, tags="spark")
        mon = self.monitor
        if mon is not None and getattr(mon, "ready", False):
            self._set_metric("CPU", f"{getattr(mon, 'cpu_live', mon.cpu):.0f}%")
            self._set_metric("GPU", f"{getattr(mon, 'gpu_live', mon.gpu):.0f}%")
            self._set_metric("VRAM", f"{getattr(mon, 'vram_live', mon.vram):.0f}%")
        else:
            self._set_metric("CPU", "n/a")
            self._set_metric("GPU", "n/a")
            self._set_metric("VRAM", "n/a")
        tiktok_connected = bool(getattr(self, "tiktok", None) is not None)
        obs_connected = bool(getattr(self, "obs_cam", None) is not None)
        self._set_row_value("TikTok", "Connected" if tiktok_connected else "Disconnected",
                            MINT if tiktok_connected else MUTED)
        self._set_row_value("OBS Virtual Camera", "Connected" if obs_connected else "Off",
                            MINT if obs_connected else MUTED)
        self._set_row_value(
            "Resolution", f"{TIKTOK_PORTRAIT_W}x{TIKTOK_PORTRAIT_H} portrait")
        self._set_row_value("FPS", f"{self._fps:.1f} FPS" if self.running else "0 FPS")
        self._set_row_value("Video Bitrate", "OBS virtual cam" if obs_connected else "preview only")
        self._set_row_value("Audio Bitrate", "TTS active" if self.tts is not None else "TTS off")
        stable = bool(self.running and self._fps > 0)
        self._set_row_value("Output", "stable" if stable else "idle",
                            MINT if stable else MUTED)
        try:
            self._topdraw()
        except Exception:
            pass

    def _show_placeholder(self):
        W, H = getattr(self, "_preview_draw_size", (PREVIEW_W, PREVIEW_H))
        img = np.full((H, W, 3), 9, np.uint8)
        for y in range(H):
            t = y / max(1, H - 1)
            img[y, :, :] = (18 + int(18 * t), 8 + int(16 * t), 18 + int(34 * t))
        cx, cy = W // 2, H // 2
        cv2.circle(img, (cx, cy - 38), max(54, W // 6), (70, 34, 86), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - 44), max(40, W // 9), (168, 128, 118), -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - W // 10, cy + 12), (cx + W // 10, cy + H // 4),
                      (36, 35, 55), -1)
        cv2.putText(img, "Preview Standby", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (245, 236, 225), 1, cv2.LINE_AA)
        cv2.putText(img, "waiting for live frame", (22, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (190, 165, 188), 1, cv2.LINE_AA)
        cv2.rectangle(img, (W - 58, 19), (W - 17, 41), (70, 70, 78), -1)
        cv2.putText(img, "IDLE", (W - 51, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y0 = H - 112
        for i, txt in enumerate(("No TikTok comments connected",
                                 "Enter a real handle to read chat",
                                 "Bot replies appear after Start")):
            yy = y0 + i * 30
            cv2.rectangle(img, (18, yy), (min(W - 18, 250), yy + 24), (24, 22, 32), -1)
            cv2.putText(img, txt, (26, yy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (230, 225, 235), 1, cv2.LINE_AA)
        self._draw(img)
        return
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
        target = getattr(self, "_preview_draw_size", (PREVIEW_W, PREVIEW_H))
        if im.size != target:
            im = self._letterbox_image(im, target)
        tkimg = ImageTk.PhotoImage(im)
        self.preview.configure(image=tkimg)
        self.preview.image = tkimg          # keep a reference

    @staticmethod
    def _letterbox_image(image, target, fill="#000000"):
        tw, th = max(1, int(target[0])), max(1, int(target[1]))
        iw, ih = image.size
        if iw <= 0 or ih <= 0:
            return Image.new("RGB", (tw, th), fill)
        scale = min(tw / float(iw), th / float(ih))
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (tw, th), fill)
        canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

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
            now_preview = time.monotonic()
            with self._frame_lock:
                serial = self._latest_serial
                should_draw = (
                    self._latest is not None
                    and serial != self._drawn_serial
                    and now_preview - self._last_preview_draw_t >= (1.0 / 12.0)
                )
                frame = self._latest.copy() if should_draw else None
            if frame is not None:
                self._drawn_serial = serial
                self._last_preview_draw_t = now_preview
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
                self.music.set_speaking(self._any_speech_active())
            except Exception:
                pass
        self._update_audio_meters()
        self._update_live_dashboard()
        self.root.after(50, self._poll_ui)   # ~20 Hz UI refresh; preview draw is capped lower

    def _any_speech_active(self):
        """Authoritative speech state across TTS, YouTube audio, and live mic."""
        return any((
            bool(self.tts is not None and getattr(self.tts, "speaking", False)),
            bool(self._youtube_audio is not None
                 and getattr(self._youtube_audio, "speaking", False)),
            bool(self.live_mic is not None
                 and getattr(self.live_mic, "speaking", False)),
        ))

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
            supporter_warmup = None
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
                lp.min_good_face = max(0.03, self.minface_var.get() / 100.0)
                lp._multi = bool(self.multiref_var.get()) and len(getattr(lp, "_refs", [])) > 1
                lp.set_stabilization(self.stab_var.get() / 100.0)
                lp.set_gaze(self.gaze_var.get(), self.gaze_var2.get() / 100.0)
                lp.set_lip_lock(self.ai_mouth_var.get())
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
            try:
                import reactions
                response_lines = reactions.ready_lines()

                def _warm_supporter_responses():
                    try:
                        self._log_msg(
                            f"[studio] caching {len(response_lines)} supporter "
                            "response pieces in background...")
                        tts.prerender(response_lines)
                        self._log_msg(
                            "   -> follow/gift/share response cache ready")
                    except Exception as exc:
                        self._log_msg(
                            f"[studio] supporter response warmup failed ({exc}).")

                supporter_warmup = _warm_supporter_responses
            except Exception as exc:
                self._log_msg(f"[studio] supporter response warmup failed ({exc}).")
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
            cap = _open_webcam() if self.camera_enabled else None
            if cap is not None and not self.camera_enabled:
                cap.release()
                cap = None
            if cap is None:
                self._log_msg("[studio] NO WEBCAM — driving with a static frame.")
            obs = None
            if self.obs_var.get():
                try:
                    import pyvirtualcam
                    obs = pyvirtualcam.Camera(
                        width=TIKTOK_PORTRAIT_W, height=TIKTOK_PORTRAIT_H,
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
            # Keep both feeds warm. Session routing uses gold while spot XAUUSD is
            # open and moves to BTCUSD during the weekend/daily maintenance break.
            try:
                from market_data import MarketData
                self.market_gold = MarketData("PAXGUSDT", "1m")
                self.market_btc = MarketData("BTCUSDT", "1m")
                self.market_gold.start()
                self.market_btc.start()
                self._select_active_market(force=True)
                self._log_msg("   -> " + self.market_gold.startup_check()[1])
                self._log_msg("   -> " + self.market_btc.startup_check()[1])
            except Exception as exc:
                self.market = None
                self.market_gold = None
                self.market_btc = None
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
                    try:                       # apply the fairer-skin slider at boot
                        self.swap_engine.skin_lighten = self.skintone_var.get() / 100.0
                    except Exception:
                        pass
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
            with self._camera_lock:
                self.cap = cap
            self.obs_cam = obs
            self._live_response_event.set()
            self.root.after(0, self._sync_audio_mute_buttons)

            self.running = True
            self._session_started_at = time.monotonic()
            self.booting = False
            # SCENE cue: the avatar scene is going live now.
            try:
                from startup_sound import play_scene_sound
                play_scene_sound()
            except Exception:
                pass
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
            if supporter_warmup is not None:
                threading.Thread(
                    target=supporter_warmup, daemon=True).start()
            # PARALLEL LLM PREFETCH POOL: several worker threads generate commentary
            # lines AHEAD of time and route each to whatever compute is free (GPU idle
            # -> GPU; GPU busy with swap/voice -> a CPU-resident model if configured,
            # else back off) — the same CPU/GPU governor idea applied to the LLM. The
            # auto-talk loop then PULLS a ready line instantly, so the host never goes
            # quiet waiting on the model. De-dupes so the buffer never hands a repeat.
            if self.brain_pool is not None:          # STOP->START: kill the old pool
                try:
                    self.brain_pool.stop()
                except Exception:
                    pass
            self.brain_pool = None
            try:
                from llm_pool import BrainPool
                pool_beats = (list(self._MARKET_BEATS) * 2 + list(self._SHORT_BEATS)
                              + list(self._ENGAGE_BEATS))     # chart-forward weighting
                self.brain_pool = BrainPool(self.brain, monitor=self.monitor,
                                            beats=pool_beats,
                                            get_context=self._live_stream_ctx)
                self.brain_pool.start()
                self._log_msg("[studio] LLM prefetch pool ON — " + self.brain_pool.status())
            except Exception as exc:
                self._log_msg(f"[studio] LLM pool off ({exc}) — inline generation.")
            # AUTO-TALK: the brain writes + speaks gold commentary on its own, PIPELINED
            # (generates the next line while the current one plays — voice gen never
            # pauses the LLM).
            self._queue_initial_market_status()
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
                for b in (self.speak_btn, self.ask_btn, self.mute_btn,
                          self.recenter_btn, self.youtube_btn,
                          self.youtube_audio_btn, self.youtube_resume_btn,
                          self.market_mode_btn):
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

    def _warm_mohammed_voice_background(self):
        if self.tts is None or self._mohammed_voice_ready or self._mohammed_voice_warming:
            return
        self._mohammed_voice_warming = True
        try:
            self.root.after(0, lambda: self.youtube_audio_btn.configure(
                state="disabled", text="MOHAMMED LOADING..."))
            self._set_youtube_progress(5, "Background: loading Mohammed voice...")
        except Exception:
            pass

        def _worker():
            try:
                self._log_msg("[youtube] Mohammed voice prewarm: loading XTTS now...")
                self.tts.set_backend("xtts")
                msg = self.tts.warm_backend()
                self._mohammed_voice_ready = True
                self._log_msg("[youtube] Mohammed voice ready: " + msg)
                self._set_youtube_progress(100, "Mohammed voice ready")
            except Exception as exc:
                self._log_msg(f"[youtube] Mohammed voice prewarm failed: {exc}")
                self._set_youtube_progress(0, f"Mohammed prewarm failed: {exc}")
            finally:
                self._mohammed_voice_warming = False
                try:
                    self.root.after(0, lambda: self.youtube_audio_btn.configure(
                        state="normal",
                        text="ALTER REAL YOUTUBE VOICE"))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

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
            fast_avatar = (getattr(self, "_last_streamer_face_frame", None)
                           or self._default_avatar_face_frame())
            if (fast_avatar is not None
                    and bool(getattr(self, "low_lag_scene_var", None)
                             and self.low_lag_scene_var.get())):
                scene_direct = self._scene_portrait_frame(fast_avatar)
                if scene_direct is not None:
                    self._last_frame_t = time.monotonic()
                    with self._frame_lock:
                        self._latest = scene_direct
                        self._latest_serial += 1
                    if self.obs_cam is not None:
                        try:
                            self.obs_cam.send(np.ascontiguousarray(scene_direct))
                        except Exception:
                            pass
                    frame_count += 1
                    if frame_count % 15 == 0:
                        now = time.perf_counter()
                        self._fps = 15.0 / max(0.001, now - fps_t)
                        fps_t = now
                        self._diag = (
                            f"scene-fast  fps:{self._fps:.1f}  "
                            f"{TIKTOK_PORTRAIT_W}x{TIKTOK_PORTRAIT_H}")
                    now_m = time.monotonic()
                    delay = max(0.0, next_tick - now_m)
                    if delay:
                        time.sleep(min(delay, 0.02))
                    next_tick = max(next_tick + TARGET_FRAME_TIME,
                                    time.monotonic())
                    continue

            if not self.camera_enabled:
                scene = self._scene_portrait_frame(
                    getattr(self, "_last_streamer_face_frame", None))
                if scene is not None:
                    with self._frame_lock:
                        self._latest = scene
                        self._latest_serial += 1
                    if self.obs_cam is not None:
                        try:
                            self.obs_cam.send(np.ascontiguousarray(scene))
                        except Exception:
                            pass
                    time.sleep(0.1)
                    continue
                msg = np.full((TIKTOK_PORTRAIT_H, TIKTOK_PORTRAIT_W, 3), 24, np.uint8)
                cv2.putText(msg, "CAMERA OFF", (118, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (181, 240, 77), 2, cv2.LINE_AA)
                cv2.putText(msg, "Use the eye button in the top bar to enable it.",
                            (54, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (200, 200, 200), 1, cv2.LINE_AA)
                with self._frame_lock:
                    self._latest = msg
                    self._latest_serial += 1
                time.sleep(0.1)
                continue

            # No real camera? Show a clear message instead of running the
            # pipeline on a blank frame (which would just sit on charts).
            with self._camera_lock:
                cap_available = self.cap is not None
            if not cap_available:
                msg = np.full((TIKTOK_PORTRAIT_H, TIKTOK_PORTRAIT_W, 3), 24, np.uint8)
                cv2.putText(msg, "NO WEBCAM", (120, 230), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (60, 60, 230), 2, cv2.LINE_AA)
                cv2.putText(msg, "Your camera is busy in another app.", (70, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(msg, "Close the browser tab / video call, then STOP+START.",
                            (40, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1,
                            cv2.LINE_AA)
                with self._frame_lock:
                    self._latest = msg
                    self._latest_serial += 1
                time.sleep(0.1)
                continue

            _t = time.perf_counter()
            driving = last_frame
            with self._camera_lock:
                cap = self.cap
                ok, fr = cap.read() if cap is not None else (False, None)
            if cap is not None:
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
            speaking_now = self._any_speech_active()
            _speed = 2 if speaking_now else 1
            recovery_mode = time.monotonic() < getattr(
                self, "_render_recovery_until", 0.0)
            if recovery_mode:
                _speed = max(_speed, 3)

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
            if self.swap_var.get() and not recovery_mode:
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
            elif self.swap_var.get() and cached_face is not None:
                ai = cached_face
                did_swap = True

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
            if (self.body_var.get() and not recovery_mode
                    and not did_swap and getattr(lp, "_face_found", False)):
                try:
                    ai = self.engines["body"].process(driving, ai)
                except Exception:
                    pass
            t_body += time.perf_counter() - _t

            # --- GFPGAN restoration: fix the plastic look on the FACE crop ----
            _t = time.perf_counter()
            if (self.restore_var.get() and not recovery_mode
                    and not did_swap and getattr(lp, "_face_found", False)):
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
            face_ok = (
                not getattr(lp, "fallback_mode", False)
                and (
                    getattr(lp, "_face_found", False)
                    or bool(did_swap and self.swap_engine is not None
                            and getattr(self.swap_engine, "last_found", False))
                    or cached_face is not None
                )
            )
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
                mt.allow_deblur = (
                    False if recovery_mode
                    else (self.monitor is None or self.monitor.gpu_free(78)))

            # "the bot is ACTUALLY talking" — from the TTS, NOT the mouth engine
            # (which we may keep alive with idle silence below).
            self._speaking = self._any_speech_active()
            ai_mouth = bool(self.ai_mouth_var.get())
            # AI mode has exclusive mouth ownership. Keeping the native mouth
            # active underneath creates visible duplicate lips at crop edges.
            lips_from_bot = ai_mouth
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
                if (ai_mouth and self._speaking) or lips_from_bot:
                    try:
                        # During speech LP/restore may run at a lower cadence.
                        # If the operator turns, a stale mouth box makes the
                        # generated mouth land between poses and look smeared.
                        if lp_fresh or motion > MOTION_THRESH or cached_bbox is None:
                            _mh = (getattr(self.swap_engine, "last_mouth", None)
                                   if did_swap and self.swap_engine is not None else None)
                            cached_bbox = comp.detect_mouth_bbox(ai, _mh)
                        mouth = mt.process_mouth(ai, cached_bbox)
                        ai = comp.blend_mouth(ai, mouth, cached_bbox, exclusive=True)
                    except Exception as exc:
                        errs += 1
                        if errs <= 3:
                            self._log_msg(f"[studio] mouth error: {exc}")
                # HD-restore the FINAL face (swap + bot mouth + real eyes) together so
                # the MOUTH gets the same CodeFormer sharpness as the eyes/skin.
                if did_swap and not recovery_mode and self.swap_engine is not None:
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
                    if recovery_mode:
                        _dev, _lvl = "cpu", "light"
                    if did_swap:
                        enh.set_level(_lvl)
                        enh.set_protect_head(
                            getattr(self.swap_engine, "last_head", None))
                    else:
                        enh.set_protect_head(None)
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

            self._last_streamer_face_frame = final.copy()
            scene_final = self._scene_portrait_frame(final)
            if scene_final is not None:
                final = scene_final

            last_final = final            # remember for the "generating" hold
            self._last_frame_t = time.monotonic()   # heartbeat for the watchdog
            with self._frame_lock:
                self._latest = final
                self._latest_serial += 1
            if self.obs_cam is not None:
                try:
                    if final.shape[:2] != (TIKTOK_PORTRAIT_H, TIKTOK_PORTRAIT_W):
                        out = self._cover_resize_bgr(
                            final, (TIKTOK_PORTRAIT_W, TIKTOK_PORTRAIT_H))
                    else:
                        out = final
                    self.obs_cam.send(np.ascontiguousarray(out))
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
        self._session_started_at = None
        self._clear_ready_speech()
        self._clear_answering()
        if self.live_mic is not None:        # release the mic + monitor streams
            try:
                self.live_mic.shutdown()
            except Exception:
                pass
            self.live_mic = None
            try:
                self.livemic_var.set(False)
            except Exception:
                pass
        if self._youtube_audio is not None:
            try:
                self._youtube_audio.stop()
            except Exception:
                pass
            self._youtube_audio = None
            self._youtube_audio_mode = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._release_camera()
        for fn in (lambda: self.obs_cam.close() if self.obs_cam else None,):
            try:
                fn()
            except Exception:
                pass
        self.cap = None; self.obs_cam = None
        self._latest = None
        self.stop_btn.configure(state="disabled")
        for b in (self.speak_btn, self.ask_btn, self.mute_btn,
                  self.recenter_btn, self.youtube_btn,
                  self.youtube_audio_btn, self.youtube_resume_btn,
                  self.market_mode_btn):
            b.configure(state="disabled")
        self.start_btn.configure(state="normal", text="START")
        self._set_status("stopped", RED)
        self.fps_lbl.configure(text="")
        self._show_placeholder()
        self._log_msg("[studio] stopped (engines kept warm; START to resume).")

    # -------------------------------------------------------------------------
    # SPEAK / CONTROLS
    # -------------------------------------------------------------------------
    def _release_camera(self):
        with self._camera_lock:
            cap, self.cap = self.cap, None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _enable_camera_capture(self):
        self._log_msg("[studio] enabling camera...")
        cap = _open_webcam()
        if not self.camera_enabled:
            if cap is not None:
                cap.release()
            return
        with self._camera_lock:
            old_cap, self.cap = self.cap, cap
        if old_cap is not None and old_cap is not cap:
            try:
                old_cap.release()
            except Exception:
                pass
        self._log_msg(
            "[studio] camera enabled."
            if cap is not None else
            "[studio] camera unavailable or busy.")

    def toggle_camera(self):
        self.camera_enabled = not self.camera_enabled
        if self.camera_enabled:
            self._log_msg("[studio] camera requested ON.")
            if self.running:
                threading.Thread(
                    target=self._enable_camera_capture, daemon=True).start()
        else:
            self._release_camera()
            self._log_msg("[studio] camera disabled and released.")
        try:
            self._topdraw()
        except Exception:
            pass

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
        if self.live_mic is not None:
            self._log_msg("[studio] Live Mic is on — just talk into the mic (typed SPEAK is disabled).")
            return
        self._user_priority()                 # you take precedence over auto-host
        self._speak_exclusive(txt)
        self._log_msg("> " + txt)

    def _speak_exclusive(self, text, priority=0):
        """Queue AI speech while silencing, but not pausing, YouTube audio."""
        if self.tts is None or not self.running:
            return False
        handoff_token = self._pause_youtube_for_ready_speech()
        try:
            self.tts.set_playback_voice_match(None)
            self.tts.set_muted(False)
            self._sync_audio_mute_buttons()
        except Exception:
            pass
        accepted = self.tts.speak(text, priority=priority)
        if not accepted:
            if handoff_token is not None:
                threading.Thread(
                    target=self._wait_and_restore_youtube,
                    args=(handoff_token,), daemon=True
                ).start()
            return False
        if handoff_token is not None:
            threading.Thread(
                target=self._wait_and_restore_youtube,
                args=(handoff_token,), daemon=True
            ).start()
        return True

    # ---- YOUTUBE SPEAK -----------------------------------------------------
    def _on_youtube_enter(self, event):
        self.speak_youtube_audio()
        return "break"

    def _on_youtube_link_changed(self, _event=None):
        if self._youtube_scene_attach_job is not None:
            try:
                self.root.after_cancel(self._youtube_scene_attach_job)
            except Exception:
                pass
        # <<Paste>> fires before Tk's default binding inserts clipboard text.
        self._youtube_scene_attach_job = self.root.after(
            650, self._attach_youtube_scene_from_entry)

    def _attach_youtube_scene_from_entry(self):
        self._youtube_scene_attach_job = None
        url = self._youtube_primary_url()
        if "youtu" not in url.lower():
            return
        self._prepare_youtube_queue(url, auto_prewarm=False)
        self._attach_youtube_scene(url)
        self._prewarm_youtube_queue_after_current_ready()

    def _youtube_entry_text(self):
        entries = getattr(self, "youtube_entries", None)
        if entries:
            values = []
            for entry in entries:
                try:
                    values.append(entry.get().strip())
                except TypeError:
                    values.append(entry.get("1.0", "end").strip())
                except Exception:
                    pass
            return "\n".join(values)
        try:
            return self.youtube_entry.get("1.0", "end")
        except TypeError:
            try:
                return self.youtube_entry.get()
            except Exception:
                return ""
        except Exception:
            return ""

    def _youtube_entry_value(self, entry):
        try:
            return entry.get().strip()
        except TypeError:
            return entry.get("1.0", "end").strip()
        except Exception:
            return ""

    def _youtube_link_rows(self, update_status=True):
        entries = getattr(self, "youtube_entries", None)
        if not entries:
            rows = []
            seen = set()
            for raw in self._youtube_entry_text().replace(",", "\n").splitlines():
                url = raw.strip()
                if not url or "youtu" not in url.lower() or url in seen:
                    continue
                seen.add(url)
                rows.append((len(rows), url))
                if len(rows) >= 10:
                    break
            return rows
        rows = []
        seen = set()
        for slot, entry in enumerate(entries[:10]):
            url = self._youtube_entry_value(entry)
            if not url:
                if update_status:
                    self._set_youtube_slot_status(slot, "EMPTY")
                continue
            if "youtu" not in url.lower():
                if update_status:
                    self._set_youtube_slot_status(slot, "WAITING - paste a YouTube link")
                continue
            if url in seen:
                if update_status:
                    self._set_youtube_slot_status(slot, "DUPLICATE - skipped")
                continue
            seen.add(url)
            rows.append((slot, url))
            if update_status:
                self._set_youtube_slot_status(slot, "READY TO LOAD")
        return rows

    def _youtube_links_from_entry(self):
        return [url for _slot, url in self._youtube_link_rows()]

    def _youtube_primary_url(self):
        rows = self._youtube_link_rows()
        if rows:
            return rows[0][1]
        return self._youtube_entry_text().strip()

    def _set_youtube_slot_status(self, slot, text):
        try:
            slot = int(slot)
        except Exception:
            return
        vars_ = getattr(self, "youtube_status_vars", None) or []
        if slot < 0 or slot >= len(vars_):
            return
        value = str(text or "").strip() or "WAITING"
        if len(value) > 96:
            value = value[:93] + "..."

        def _apply():
            try:
                vars_[slot].set(value)
            except Exception:
                pass

        try:
            self.root.after(0, _apply)
        except Exception:
            _apply()

    def _youtube_current_slot(self):
        slots = getattr(self, "_youtube_queue_slots", []) or []
        idx = int(getattr(self, "_youtube_queue_index", 0) or 0)
        if 0 <= idx < len(slots):
            return slots[idx]
        return None

    def _set_current_youtube_slot_status(self, text):
        slot = self._youtube_current_slot()
        if slot is not None:
            self._set_youtube_slot_status(slot, text)

    def _youtube_slot_for_url(self, url):
        needle = (url or "").strip()
        if not needle:
            return None
        for slot, candidate in self._youtube_link_rows(update_status=False):
            if candidate == needle:
                return slot
        return None

    def _youtube_queue_label(self):
        total = len(getattr(self, "_youtube_queue_urls", []) or [])
        if not getattr(self, "_youtube_queue_active", False) or total <= 1:
            return ""
        idx = min(total, max(0, int(getattr(self, "_youtube_queue_index", 0))) + 1)
        return f" VIDEO {idx}/{total}"

    def _prepare_youtube_queue(self, start_url=None, auto_prewarm=True):
        rows = self._youtube_link_rows()
        urls = [url for _slot, url in rows]
        slots = [slot for slot, _url in rows]
        if start_url and start_url not in urls:
            urls = [start_url] + urls
            slots = [0] + slots
        urls = urls[:10]
        slots = slots[:10]
        signature = tuple(zip(slots, urls))
        previous_signature = tuple(getattr(self, "_youtube_queue_signature", ()) or ())
        changed = signature != previous_signature
        self._youtube_queue_urls = urls
        self._youtube_queue_slots = slots
        self._youtube_queue_signature = signature
        self._youtube_queue_index = 0
        if start_url:
            try:
                self._youtube_queue_index = self._youtube_queue_urls.index(start_url)
            except ValueError:
                self._youtube_queue_index = 0
        self._youtube_queue_active = len(self._youtube_queue_urls) > 1
        self._youtube_queue_advancing = False
        if changed:
            self._youtube_queue_prewarmed = set()
            self._youtube_queue_prewarm_started = False
            self._youtube_prewarm_wait_started = False
        for order, slot in enumerate(self._youtube_queue_slots):
            self._set_youtube_slot_status(
                slot, f"QUEUED - video {order + 1}/{len(self._youtube_queue_urls)}")
        if self._youtube_queue_active and changed:
            self._log_msg(
                f"[youtube] playlist loaded: {len(self._youtube_queue_urls)} videos.")
        if self._youtube_queue_active and auto_prewarm:
            self._prewarm_youtube_queue()
        return self._youtube_queue_urls

    def _prewarm_youtube_queue_after_current_ready(self):
        if (not getattr(self, "_youtube_queue_active", False)
                or getattr(self, "_youtube_prewarm_wait_started", False)):
            return
        self._youtube_prewarm_wait_started = True

        def _worker():
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                if not getattr(self, "_youtube_queue_active", False):
                    return
                scene = getattr(self, "_youtube_scene", None)
                if scene is None:
                    time.sleep(0.25)
                    continue
                if bool(getattr(scene, "video_ready", False)):
                    self._log_msg(
                        "[youtube] first video ready; downloading next queued video.")
                    self._prewarm_youtube_queue()
                    return
                status = str(getattr(scene, "status", "") or "").lower()
                if "video scene failed" in status:
                    self._log_msg(
                        "[youtube] first video failed; starting queued preload anyway.")
                    self._prewarm_youtube_queue()
                    return
                time.sleep(0.25)
            self._log_msg(
                "[youtube] first video wait timed out; starting queued preload.")
            self._prewarm_youtube_queue()

        threading.Thread(target=_worker, daemon=True).start()

    def _prewarm_youtube_queue(self):
        if (not getattr(self, "_youtube_queue_active", False)
                or getattr(self, "_youtube_queue_prewarm_started", False)):
            return
        self._youtube_queue_prewarm_started = True

        def _worker():
            urls = list(getattr(self, "_youtube_queue_urls", []) or [])
            slots = list(getattr(self, "_youtube_queue_slots", []) or [])
            start = max(0, int(getattr(self, "_youtube_queue_index", 0))) + 1
            for i, url in enumerate(urls[start:], start=start):
                slot = slots[i] if i < len(slots) else i
                if not getattr(self, "_youtube_queue_active", False):
                    break
                if url in self._youtube_queue_prewarmed:
                    continue
                try:
                    self._set_youtube_slot_status(
                        slot, f"PRELOADING - video {i + 1}/{len(urls)}")
                    self._log_msg(
                        f"[youtube] preloading next video {i + 1}/{len(urls)}...")
                    try:
                        from youtube_audio import _resolve_audio_source
                        _resolve_audio_source(
                            url,
                            lambda msg, n=i, s=slot: (
                                self._log_msg(
                                    f"[youtube] preload audio {n + 1}: {msg}"),
                                self._set_youtube_slot_status(
                                    s, f"AUDIO: {msg}")))
                    except Exception as exc:
                        self._set_youtube_slot_status(
                            slot, f"AUDIO PRELOAD SKIPPED: {exc}")
                        self._log_msg(
                            f"[youtube] preload audio {i + 1} skipped: {exc}")
                    try:
                        from youtube_video import resolve_youtube_video
                        resolve_youtube_video(
                            url,
                            lambda msg, n=i, s=slot: (
                                self._log_msg(
                                    f"[youtube] preload video {n + 1}: {msg}"),
                                self._set_youtube_slot_status(
                                    s, f"VIDEO: {msg}")))
                    except Exception as exc:
                        self._set_youtube_slot_status(
                            slot, f"VIDEO PRELOAD SKIPPED: {exc}")
                        self._log_msg(
                            f"[youtube] preload video {i + 1} skipped: {exc}")
                    self._youtube_queue_prewarmed.add(url)
                    self._set_youtube_slot_status(
                        slot, f"READY - video {i + 1}/{len(urls)} cached")
                except Exception as exc:
                    self._set_youtube_slot_status(slot, f"PRELOAD FAILED: {exc}")
                    self._log_msg(f"[youtube] preload failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _attach_youtube_scene(
            self, url, force=False, preserve_crop=False,
            force_refresh_cache=False):
        url = (url or "").strip()
        if (not url or (
                url == self._youtube_scene_url
                and self._scene_source == "youtube"
                and not force)):
            return
        try:
            from youtube_video import YouTubeVideoScene
        except Exception as exc:
            self._log_msg(f"[scene] YouTube video unavailable: {exc}")
            return
        if self._youtube_scene is not None:
            try:
                self._youtube_scene.stop()
            except Exception:
                pass

        def _status(message):
            self._log_msg(f"[scene] {message}")
            self._on_youtube_video_status(message)
            slot = self._youtube_slot_for_url(url)
            if slot is not None:
                self._set_youtube_slot_status(slot, f"VIDEO: {message}")
            try:
                with open(os.path.join(PROJECT_DIR, "youtube_scene_status.log"),
                          "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            except Exception:
                pass
            if "ready" in message:
                try:
                    self.root.after(0, self._build_sidebar_scene_slot)
                except Exception:
                    pass

        previous_crop = self._youtube_scene_crop
        self._youtube_scene_url = url
        self._youtube_scene_crop = (
            previous_crop if preserve_crop else (0.0, 0.0, 1.0, 1.0))
        self._youtube_scene_raw_image = None
        self._youtube_scene_clock_started_at = None
        preview_running = (
            self._youtube_mode == "youtube"
            or (
                self._youtube_audio is None
                and not self._youtube_chunks
                and not self._youtube_busy
            )
        )
        self._youtube_clock_reset(
            self._youtube_position_seconds(),
            running=preview_running)
        self._scene_face_box = None
        self._scene_face_detect_count = 0
        self._youtube_scene = YouTubeVideoScene(
            self._youtube_position_seconds, status_callback=_status,
            force_refresh_cache=bool(force_refresh_cache))
        with self._scene_capture_lock:
            self._scene_source = "youtube"
            self._scene_capture_bbox = None
            self._scene_window_hwnd = None
            self._scene_window_crop = None
            self._scene_capture_bgr = None
            self._scene_capture_image = Image.new(
                "RGB", self._scene_preview_size, "#050608")
            self._scene_display_image = self._scene_capture_image.copy()
            self._scene_capture_serial += 1
        self._youtube_scene.start(url)
        self._build_sidebar_scene_slot()
        self._schedule_scene_preview()
        self._log_msg("[scene] YouTube video attached; loading synchronized preview.")

    def _reset_youtube_scene_video(self):
        """Restart the YouTube video decoder without touching current audio."""
        if self._scene_source != "youtube":
            self._log_msg("[scene] no YouTube video scene is active.")
            return
        url = (self._youtube_scene_url or "").strip()
        if not url:
            try:
                url = self._youtube_primary_url()
            except Exception:
                url = ""
        if not url:
            self._log_msg("[scene] paste a YouTube link before resetting video.")
            return
        pos = self._youtube_position_seconds()
        self._announce_youtube_link_state(url)
        self._log_msg(
            f"[scene] resetting YouTube video at {self._fmt_time(pos)}; voice keeps playing.")
        self._attach_youtube_scene(
            url, force=True, preserve_crop=True)

    def _youtube_position_seconds(self):
        if self._youtube_audio is not None:
            return float(getattr(self._youtube_audio, "position_seconds", 0.0) or 0.0)
        start = float(getattr(self, "_youtube_start_seconds", 0.0) or 0.0)
        if self._youtube_chunks:
            return self._youtube_clock_current()
        if self._scene_source == "youtube" and self._youtube_scene is not None:
            return self._youtube_clock_current()
        return start

    def _youtube_clock_reset(self, position=None, running=None):
        start = float(getattr(self, "_youtube_start_seconds", 0.0) or 0.0)
        self._youtube_clock_position = float(start if position is None else position)
        if running is None:
            running = getattr(self, "_youtube_mode", "market") == "youtube"
        self._youtube_clock_anchor_t = time.monotonic() if running else None

    def _youtube_clock_current(self):
        position = float(getattr(self, "_youtube_clock_position", 0.0) or 0.0)
        anchor = getattr(self, "_youtube_clock_anchor_t", None)
        if anchor is not None:
            position += max(0.0, time.monotonic() - float(anchor))
        end = getattr(self, "_youtube_end_seconds", None)
        duration = float(getattr(self, "_youtube_duration", 0.0) or 0.0)
        if duration <= 0 and self._youtube_scene is not None:
            duration = float(getattr(self._youtube_scene, "duration", 0.0) or 0.0)
        limit = float(end) if end is not None else duration
        if limit > 0:
            position = min(position, limit)
        return max(0.0, position)

    def _youtube_clock_pause(self):
        self._youtube_clock_position = self._youtube_clock_current()
        self._youtube_clock_anchor_t = None

    def _youtube_clock_resume(self):
        self._youtube_clock_position = self._youtube_clock_current()
        self._youtube_clock_anchor_t = time.monotonic()

    def _scene_time_text(self):
        if self._scene_source != "youtube":
            return ""
        position = self._youtube_position_seconds()
        duration = 0.0
        if self._youtube_scene is not None:
            duration = float(getattr(self._youtube_scene, "duration", 0.0) or 0.0)
        return (
            f"{self._fmt_time(position)} / {self._fmt_time(duration)}"
            if duration > 0 else self._fmt_time(position)
        )

    def _edit_youtube_scene_crop(self):
        with self._scene_capture_lock:
            image = self._youtube_scene_raw_image
            current = self._youtube_scene_crop
        if image is None:
            self._log_msg("[scene] wait for the first YouTube frame, then edit crop.")
            return

        win = tk.Toplevel(self.root)
        win.title("Edit YouTube Scene Crop")
        win.configure(bg=BG)
        win.transient(self.root)
        win.attributes("-topmost", True)
        width, height = 800, 450
        canvas = tk.Canvas(
            win, width=width, height=height, bg="#000000",
            highlightthickness=1, highlightbackground=AMBER,
            cursor="crosshair")
        canvas.pack(padx=12, pady=(12, 8))
        shown = image.resize((width, height), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(shown)
        canvas.create_image(0, 0, image=tk_image, anchor="nw")
        win._youtube_crop_image = tk_image
        state = {"start": None, "rect": None, "crop": current}

        def _draw(crop):
            if state["rect"] is not None:
                canvas.delete(state["rect"])
            x1, y1, x2, y2 = crop
            state["rect"] = canvas.create_rectangle(
                x1 * width, y1 * height, x2 * width, y2 * height,
                outline=AMBER, width=3)
            state["crop"] = crop

        def _down(event):
            state["start"] = (event.x, event.y)

        def _drag(event):
            if state["start"] is None:
                return
            sx, sy = state["start"]
            x1, x2 = sorted((max(0, sx), min(width, event.x)))
            y1, y2 = sorted((max(0, sy), min(height, event.y)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                return
            _draw((x1 / width, y1 / height, x2 / width, y2 / height))

        def _apply():
            self._youtube_scene_crop = state["crop"]
            self._scene_face_box = None
            self._scene_face_detect_count = 0
            win.destroy()
            self._log_msg("[scene] YouTube crop updated.")

        def _reset():
            _draw((0.0, 0.0, 1.0, 1.0))

        _draw(current)
        canvas.bind("<Button-1>", _down)
        canvas.bind("<B1-Motion>", _drag)
        canvas.bind("<ButtonRelease-1>", _drag)
        controls = tk.Frame(win, bg=BG)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        tk.Button(
            controls, text="Reset Full Video", command=_reset,
            bg=SURFACE2, fg=FG, relief="flat", cursor="hand2",
            font=("Segoe UI", 9), padx=12, pady=6).pack(side="left")
        tk.Button(
            controls, text="Apply Crop", command=_apply,
            bg=self._mix(AMBER, "#ffffff", 0.08), fg="#111111",
            relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
            padx=16, pady=6).pack(side="right")
        win.bind("<Return>", lambda _event: _apply())
        win.bind("<Escape>", lambda _event: win.destroy())

    def _youtube_link_cache_state(self, url):
        """Return (seen_before, video_id) for the pasted YouTube link."""
        try:
            from youtube_cache import audio_dir, cache_summary
            summary = cache_summary(url)
            seen = bool(summary.get("has_transcript") or summary.get("has_audio"))
            video_id = summary.get("video_id", "")
            try:
                from youtube_video import _preview_video_candidates
                seen = seen or bool(_preview_video_candidates(audio_dir(url)))
            except Exception:
                pass
            return seen, video_id
        except Exception:
            return False, ""

    def _announce_youtube_link_state(self, url):
        seen, video_id = self._youtube_link_cache_state(url)
        if seen:
            notice = "LINK WAS HERE BEFORE - USING CACHE"
            color = AMBER
            progress = "Link was here before - using cached video"
            log = f"[youtube] link was here before; trusting cache ({video_id})"
        else:
            notice = "NEW LINK LOADED"
            color = MINT
            progress = "New link loaded - downloading video"
            log = f"[youtube] new link loaded ({video_id or 'uncached'})"
        self._youtube_link_notice = notice
        self._youtube_link_notice_color = color
        self._youtube_link_notice_until = time.monotonic() + 8.0
        self._set_youtube_progress(3, progress)
        self._log_msg(log)
        try:
            self._topdraw()
            self.root.after(8200, self._topdraw)
        except Exception:
            pass
        return seen

    def speak_youtube(self):
        url = self._youtube_primary_url()
        if not url:
            return
        self._announce_youtube_link_state(url)
        if self.tts is None or not self.running:
            self._attach_youtube_scene(url)
            self._log_msg("[studio] press START first.")
            return
        if self.live_mic is not None:
            self._log_msg("[studio] Live Mic is on - YouTube Speak is disabled.")
            return
        if self._youtube_busy:
            self._log_msg("[studio] YouTube Speak is already loading a video.")
            return
        try:
            start_s, end_s = self._youtube_range()
        except ValueError as exc:
            self._log_msg(f"[youtube] range error: {exc}")
            return
        self._youtube_start_seconds = float(start_s or 0.0)
        self._youtube_end_seconds = end_s
        self._youtube_clock_reset(self._youtube_start_seconds, running=False)
        self._youtube_busy = True
        self.youtube_btn.configure(state="disabled", text="LOADING...")
        self._youtube_audio_mode = False
        self._youtube_mohammed_mode = False
        try:
            if self._youtube_audio is not None:
                self._youtube_audio.stop()
                self._youtube_audio = None
            self.tts.set_muted(False)
        except Exception:
            pass
        self._attach_youtube_scene(url, force=True, preserve_crop=True)
        self._user_priority()
        self._log_msg("[youtube] reading captions...")
        self._start_youtube_pump()

        def _worker():
            try:
                from youtube_cache import cache_summary
                from youtube_speaker import fetch_youtube_transcript, chunk_for_speech
                summary = cache_summary(url)
                if summary["has_transcript"]:
                    self._log_msg(f"[youtube] db hit: captions already saved ({summary['video_id']})")
                else:
                    self._log_msg(f"[youtube] db miss: new captions link ({summary['video_id']})")

                def _caption_status(msg):
                    self._log_msg(f"[youtube] captions: {msg}")

                title, transcript = fetch_youtube_transcript(
                    url, start_seconds=start_s, end_seconds=end_s,
                    status_callback=_caption_status)
                chunks = chunk_for_speech(transcript)
                if not chunks:
                    raise RuntimeError("no speakable transcript text found")
                first_batch = chunks[:min(3, len(chunks))]
                if first_batch:
                    self._log_msg(
                        f"[youtube] preparing first Mohammed chunks 0/{len(first_batch)}...")

                    def _pre_progress(i, total, text):
                        self._log_msg(
                            f"[youtube] preparing Mohammed {i}/{total}: {text[:48]}")

                    self.tts.prerender(first_batch, progress=_pre_progress)
                    self._log_msg("[youtube] first Mohammed chunks ready.")
                self._youtube_title = title
                self._youtube_chunks = chunks
                self._youtube_index = 0
                self._youtube_clock_reset(self._youtube_start_seconds, running=False)
                rg = self._youtube_range_label(start_s, end_s)
                self._log_msg(f"[youtube] {title}{rg} -> {len(chunks)} voice chunks")
                self._wait_for_youtube_video_ready(url)
                self._set_youtube_mode("youtube")
            except Exception as exc:
                self._log_msg(f"[youtube] failed: {exc}")
            finally:
                def _done():
                    self._youtube_busy = False
                    text = "SPEAK YOUTUBE"
                    if self.running:
                        self.youtube_btn.configure(state="normal", text=text)
                    else:
                        self.youtube_btn.configure(state="disabled", text=text)
                try:
                    self.root.after(0, _done)
                except Exception:
                    self._youtube_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def resume_youtube(self):
        if self._youtube_audio is None and not self._youtube_chunks:
            self._log_msg("[youtube] paste a link and press SPEAK YOUTUBE first.")
            return
        self._set_youtube_mode("youtube")

    def resume_market(self):
        self._set_youtube_mode("market")

    def _set_youtube_mode(self, mode):
        mode = "youtube" if mode == "youtube" else "market"
        previous = getattr(self, "_youtube_mode", "market")
        self._youtube_mode = mode
        try:
            if self.tts is not None:
                # Drop queued low-priority YouTube/auto filler. The current spoken
                # chunk finishes naturally; the next source starts cleanly.
                self.tts.clear_pending(below=1)
        except Exception:
            pass
        try:
            if getattr(self, "autotalk_var", None) is not None:
                self.root.after(0, lambda: self.autotalk_var.set(mode == "market"))
        except Exception:
            pass
        try:
            self.root.after(0, self._sync_youtube_status)
        except Exception:
            pass
        if mode == "youtube":
            if previous != "youtube":
                self._youtube_clock_resume()
            try:
                if self._youtube_audio is not None:
                    if self.tts is not None:
                        self.tts.set_muted(True)
                    self._youtube_audio.resume()
            except Exception:
                pass
            self._start_youtube_pump()
            if self._youtube_chunks:
                self._bump_youtube_progress(
                    self._youtube_playback_progress(), "YouTube speech running")
            self._log_msg("[youtube] resumed from saved position.")
        else:
            if previous == "youtube":
                self._youtube_clock_pause()
            try:
                if self._youtube_audio is not None:
                    self._youtube_audio.pause()
                if self.tts is not None:
                    self.tts.set_muted(False)
            except Exception:
                pass
            self._bump_youtube_progress(
                self._youtube_progress_value, "Paused - market mode")
            self._log_msg("[youtube] paused. Market analysis resumed.")

    def _sync_youtube_status(self):
        try:
            active = self._youtube_mode == "youtube"
            audio_active = self._youtube_audio is not None and self._youtube_audio_mode
            done = bool(self._youtube_chunks) and self._youtube_index >= len(self._youtube_chunks)
            if self._youtube_audio_status == "failed":
                text, color = "YOUTUBE FAILED", RED
            elif done:
                text, color = "YOUTUBE DONE", MINT
            elif active:
                if audio_active:
                    status = getattr(self._youtube_audio, "status", None) or self._youtube_audio_status
                    text = ("REAL YOUTUBE VOICE - " + status.upper()
                            if status else "ALTERED REAL YOUTUBE VOICE")
                else:
                    text = "SPEAKING FROM YOUTUBE"
                color = AMBER
            else:
                text, color = "MARKET MODE", MINT
            self.youtube_status_lbl.configure(text=text, fg=color)
            self.youtube_light.itemconfig(
                self.youtube_light_dot,
                fill=color if active or done else "#3a3f4a")
            self.youtube_time_lbl.configure(text=self._youtube_time_text())
            self._sync_youtube_buttons()
            if self._youtube_chunks:
                self.youtube_resume_btn.configure(state="normal")
        except Exception:
            pass

    def _set_youtube_progress(self, value, text=None):
        self._youtube_progress_value = max(0.0, min(100.0, float(value or 0.0)))
        if text is not None:
            self._youtube_progress_text = str(text)

        def _apply():
            try:
                pending = getattr(self, "_youtube_progress_after", None)
                if pending is not None:
                    self.root.after_cancel(pending)
                    self._youtube_progress_after = None
                self._animate_youtube_progress()
                if getattr(self, "youtube_progress_lbl", None) is not None:
                    self.youtube_progress_lbl.configure(text=self._youtube_progress_text)
            except Exception:
                pass

        try:
            self.root.after(0, _apply)
        except Exception:
            _apply()

    def _animate_youtube_progress(self):
        """Ease the custom YouTube meter toward its latest real progress value."""
        self._youtube_progress_after = None
        current = float(getattr(self, "_youtube_progress_display", 0.0))
        target = float(getattr(self, "_youtube_progress_value", 0.0))
        delta = target - current
        if abs(delta) < 0.15:
            self._youtube_progress_display = target
            self._draw_youtube_progress()
            return
        self._youtube_progress_display = current + delta * 0.22
        self._draw_youtube_progress()
        self._youtube_progress_after = self.root.after(
            16, self._animate_youtube_progress)

    def _draw_youtube_progress(self):
        """Draw a compact neon processing rail without native-theme artifacts."""
        canvas = getattr(self, "youtube_progress", None)
        if canvas is None:
            return
        width = max(80, canvas.winfo_width())
        height = max(18, canvas.winfo_height())
        value = max(0.0, min(
            100.0, float(getattr(self, "_youtube_progress_display", 0.0))))
        canvas.delete("all")

        x1, y1, x2, y2 = 1, 3, width - 1, height - 3
        self._round_rect(
            canvas, x1, y1, x2, y2, 6,
            fill="#07121c", outline=self._mix(BORDER, CYAN, 0.25), width=1)

        inner_x1, inner_y1 = x1 + 2, y1 + 2
        inner_x2, inner_y2 = x2 - 2, y2 - 2
        fill_x = inner_x1 + (inner_x2 - inner_x1) * value / 100.0
        if value > 0.0:
            glow_x = min(inner_x2, fill_x + 4)
            self._round_rect(
                canvas, inner_x1, inner_y1, glow_x, inner_y2, 4,
                fill=self._mix("#07121c", CYAN, 0.34), outline="")
            self._round_rect(
                canvas, inner_x1, inner_y1, fill_x, inner_y2, 4,
                fill=CYAN, outline="")
            if fill_x - inner_x1 > 8:
                canvas.create_line(
                    fill_x - 1, inner_y1 + 1, fill_x - 1, inner_y2 - 1,
                    fill="#d7ffff", width=2)

        for pct in (25, 50, 75):
            tick_x = inner_x1 + (inner_x2 - inner_x1) * pct / 100.0
            if tick_x > fill_x + 2:
                canvas.create_line(
                    tick_x, inner_y1 + 2, tick_x, inner_y2 - 2,
                    fill=self._mix(BORDER, CYAN, 0.22))

        canvas.create_text(
            width - 7, height / 2, text=f"{int(round(value)):02d}%",
            anchor="e", fill=(
                "#031016" if value >= 96 else self._mix(FG, CYAN, 0.35)),
            font=("Consolas", 7, "bold"))

    def _bump_youtube_progress(self, value, text=None):
        if value > self._youtube_progress_value:
            self._set_youtube_progress(value, text)
        elif text is not None:
            self._set_youtube_progress(self._youtube_progress_value, text)

    def _on_youtube_video_status(self, message):
        text = str(message or "")
        low = text.lower()
        if low.startswith("video preview download ") and "%" in low:
            try:
                pct_text = low.split("video preview download ", 1)[1].split("%", 1)[0]
                pct = float(pct_text)
                self._set_youtube_progress(
                    min(100.0, max(0.0, pct)),
                    f"Video downloading {int(pct)}%")
            except Exception:
                self._set_youtube_progress(
                    self._youtube_progress_value, "Video downloading...")
        elif "downloading low-res video preview" in low:
            self._set_youtube_progress(0, "Video download starting...")
        elif "video preview found in local cache" in low:
            self._set_youtube_progress(100, "Video preview ready")
        elif "video preview saved in local cache" in low:
            self._set_youtube_progress(100, "Video download complete")
        elif "video first frame ready" in low:
            self._set_youtube_progress(100, "Video ready")

    def _wait_for_youtube_video_ready(self, url, timeout=240.0):
        deadline = time.monotonic() + float(timeout)
        announced = False
        while time.monotonic() < deadline:
            scene = getattr(self, "_youtube_scene", None)
            if scene is None:
                return False
            if getattr(scene, "url", "") and url and scene.url.strip() != url.strip():
                return False
            if bool(getattr(scene, "video_ready", False)):
                return True
            status = str(getattr(scene, "status", "") or "").lower()
            if "video scene failed" in status:
                return False
            if not announced:
                announced = True
                self._set_youtube_progress(
                    max(5.0, self._youtube_progress_value),
                    "Waiting for video before sound...")
                self._log_msg("[youtube] waiting for video before starting sound.")
            time.sleep(0.25)
        self._log_msg("[youtube] video wait timed out; starting sound anyway.")
        return False

    def _style_state_button(self, btn, on, text=None):
        """Make mode buttons obvious: green means active/on, red means off."""
        if btn is None:
            return
        color = MINT if on else RED
        bg = self._mix(SURFACE2, color, 0.26 if on else 0.10)
        hover = self._mix(SURFACE2, color, 0.38 if on else 0.20)
        try:
            if text is not None:
                btn.configure(text=text)
            btn.configure(bg=bg, fg=color, activebackground=hover,
                          activeforeground=color, highlightbackground=color,
                          highlightcolor=color)
            btn._normal_bg = bg
            btn._normal_border = color
            btn._hover_bg = hover
            btn._hover_border = self._mix(color, "#ffffff", 0.35)
        except Exception:
            pass

    def _sync_youtube_buttons(self):
        yt_on = self._youtube_mode == "youtube"
        market_on = not yt_on
        audio_on = yt_on and self._youtube_audio is not None and self._youtube_audio_mode
        caption_on = yt_on and bool(self._youtube_chunks) and not audio_on
        self._style_state_button(self.youtube_btn, caption_on, "SPEAK YOUTUBE")
        self._style_state_button(
            self.youtube_audio_btn, audio_on, "ALTER REAL YOUTUBE VOICE")
        self._style_state_button(self.youtube_resume_btn, yt_on, "YOUTUBE")
        self._style_state_button(self.market_mode_btn, market_on, "MARKET")
        total = len(getattr(self, "_youtube_queue_urls", []) or [])
        idx = int(getattr(self, "_youtube_queue_index", 0) or 0)
        if total <= 0:
            total = len(self._youtube_links_from_entry())
            idx = 0
        prev_enabled = total > 1 and idx > 0 and not self._youtube_busy
        next_enabled = total > 1 and idx < total - 1 and not self._youtube_busy
        try:
            self.youtube_back_btn.configure(
                state="normal" if prev_enabled else "disabled")
            self.youtube_next_btn.configure(
                state="normal" if next_enabled else "disabled")
        except Exception:
            pass

    def _youtube_clock_tick(self):
        try:
            if getattr(self, "youtube_time_lbl", None) is not None:
                self.youtube_time_lbl.configure(text=self._youtube_time_text())
            if getattr(self, "scene_time_lbl", None) is not None:
                self.scene_time_lbl.configure(text=self._scene_time_text())
            if (self._youtube_audio is not None
                    and self._youtube_mode == "youtube"
                    and self._scene_source != "youtube"
                    and self._youtube_scene_attach_job is None):
                self._on_youtube_link_changed()
        except Exception:
            pass
        try:
            self.root.after(1000, self._youtube_clock_tick)
        except Exception:
            pass

    def _youtube_time_text(self):
        queue_label = self._youtube_queue_label()
        if self._youtube_audio is not None:
            pos = self._youtube_position_seconds()
            dur = float(getattr(self._youtube_audio, "duration", 0.0))
            status = (getattr(self._youtube_audio, "status", "") or "").upper()
            if dur > 0:
                return f"YOUTUBE{queue_label} TIME {self._fmt_time(pos)} / {self._fmt_time(dur)}  {status}"
            return f"YOUTUBE{queue_label} TIME {self._fmt_time(pos)}  {status}"
        if self._youtube_chunks:
            total = len(self._youtube_chunks)
            idx = min(total, max(0, self._youtube_index))
            pct = int(round(idx / max(1, total) * 100))
            start = float(getattr(self, "_youtube_start_seconds", 0.0) or 0.0)
            end = getattr(self, "_youtube_end_seconds", None)
            dur = float(getattr(self, "_youtube_duration", 0.0) or 0.0)
            if dur <= 0 and self._youtube_scene is not None:
                dur = float(
                    getattr(self._youtube_scene, "duration", 0.0) or 0.0)
            range_end = float(end) if end is not None else (dur if dur > 0 else start)
            pos = self._youtube_position_seconds()
            if dur > 0:
                return f"YOUTUBE{queue_label} TIME {self._fmt_time(pos)} / {self._fmt_time(dur)}  TEXT {idx}/{total} {pct}%"
            return f"YOUTUBE{queue_label} TIME {self._fmt_time(pos)}  TEXT {idx}/{total} {pct}%"
        return "YOUTUBE TIME 00:00"

    def _youtube_playback_progress(self):
        if not self._youtube_chunks:
            return 0.0
        total = len(self._youtube_chunks)
        idx = min(total, max(0, self._youtube_index))
        return 90.0 + 10.0 * (idx / max(1, total))

    def _youtube_range(self):
        start = self._parse_youtube_time(self.youtube_from_var.get())
        end = self._parse_youtube_time(self.youtube_to_var.get())
        if start is not None and end is not None and end <= start:
            raise ValueError("TO must be after FROM")
        return start, end

    def _youtube_range_label(self, start, end):
        if start is None and end is None:
            return ""
        a = self._fmt_time(start or 0)
        b = self._fmt_time(end) if end is not None else "end"
        return f" [{a} -> {b}]"

    @staticmethod
    def _parse_youtube_time(value):
        s = (value or "").strip()
        if not s:
            return None
        try:
            if ":" not in s:
                return float(s) * 60.0
            parts = [float(p) for p in s.split(":")]
            if len(parts) == 2:
                return parts[0] * 60.0 + parts[1]
            if len(parts) == 3:
                return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        except Exception:
            pass
        raise ValueError("use minutes, mm:ss, or hh:mm:ss")

    @staticmethod
    def _fmt_time(seconds):
        if seconds is None:
            return "00:00"
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def speak_youtube_audio(self, start_override=None, queue_advance=False):
        """Play the real YouTube recording through a voice-only transform.

        This path never transcribes the video and never invokes TTS. It preserves
        the source performance and changes only its audible voice characteristics.
        """
        if queue_advance and getattr(self, "_youtube_queue_active", False):
            urls = getattr(self, "_youtube_queue_urls", []) or []
            idx = int(getattr(self, "_youtube_queue_index", 0) or 0)
            url = urls[idx] if 0 <= idx < len(urls) else ""
        else:
            url = self._youtube_primary_url()
        if not url:
            return
        if not queue_advance:
            self._prepare_youtube_queue(url, auto_prewarm=False)
        current_slot = self._youtube_current_slot()
        if current_slot is None:
            current_slot = self._youtube_slot_for_url(url)
        if current_slot is not None:
            self._set_youtube_slot_status(current_slot, "LOADING - checking cache/video/audio")
        self._announce_youtube_link_state(url)
        if not self.running or self.engines is None:
            self._attach_youtube_scene(url)
            self._log_msg("[studio] press START first.")
            return
        if self.live_mic is not None:
            self._log_msg("[studio] Live Mic is on - real YouTube voice is disabled.")
            return
        if self._youtube_busy:
            self._log_msg("[studio] YouTube audio is already loading.")
            return
        if queue_advance:
            start_s, end_s = 0.0, None
        else:
            try:
                start_s, end_s = self._youtube_range()
            except ValueError as exc:
                self._log_msg(f"[youtube] range error: {exc}")
                return
        if start_override is not None:
            try:
                start_s = max(0.0, float(start_override))
            except (TypeError, ValueError):
                start_s = 0.0
            if end_s is not None and start_s >= float(end_s):
                self._log_msg("[youtube] selected voice change after the selected range ended.")
                return
        self.youtube_audio_btn.configure(state="disabled", text="LOADING REAL AUDIO...")
        self._youtube_busy = True
        self._youtube_audio_mode = True
        self._youtube_audio_status = ""
        self._youtube_mohammed_mode = False
        self._youtube_start_seconds = float(start_s or 0.0)
        self._youtube_end_seconds = end_s
        self._youtube_clock_reset(self._youtube_start_seconds, running=False)
        persona_label = self.youtube_persona_var.get()
        persona_key = YOUTUBE_PERSONA_KEY.get(persona_label, "deep_male")
        self._youtube_duration = 0.0
        self._youtube_chunks = []
        self._youtube_index = 0
        try:
            if self._youtube_audio is not None:
                self._youtube_audio.stop()
                self._youtube_audio = None
            if self.tts is not None:
                self.tts.clear_pending()
                self.tts.set_muted(True)
        except Exception:
            pass
        self._attach_youtube_scene(
            url, force=True, preserve_crop=True)
        self._user_priority()
        self._log_msg(
            "[youtube] loading the original recording; TTS/model generation is off.")
        self._set_youtube_progress(5, "Loading original YouTube audio...")

        def _worker():
            try:
                from youtube_audio import YouTubeAudioPlayer

                def _audio_status(msg):
                    self._youtube_audio_status = msg
                    self._log_msg(f"[youtube] real audio: {msg}")
                    if current_slot is not None:
                        self._set_youtube_slot_status(
                            current_slot, f"AUDIO: {msg}")
                    if msg.startswith("voice isolation ") and "%" in msg:
                        try:
                            phase_pct = int(
                                msg.split("voice isolation ", 1)[1].split("%", 1)[0])
                            self._bump_youtube_progress(
                                35.0 + 47.0 * phase_pct / 100.0, msg)
                        except (TypeError, ValueError):
                            self._bump_youtube_progress(35, msg)
                    elif "vocals ready" in msg or "isolated vocals" in msg:
                        self._bump_youtube_progress(82, msg)
                    elif "download" in msg or "cache" in msg:
                        self._bump_youtube_progress(35, msg)
                    elif "ffmpeg" in msg or "buffer" in msg:
                        self._bump_youtube_progress(86, msg)
                    elif "playing" in msg:
                        self._bump_youtube_progress(94, "Playing altered original voice")
                    elif msg == "ended":
                        label = self._youtube_queue_label()
                        if current_slot is not None:
                            self._set_youtube_slot_status(current_slot, "ENDED")
                        self._set_youtube_progress(
                            100, f"Original YouTube audio finished{label}")
                        self._advance_youtube_queue_after_current()
                    try:
                        self.root.after(0, self._sync_youtube_status)
                    except Exception:
                        pass

                player = YouTubeAudioPlayer(
                    self.engines["mt"],
                    converter_kind=os.environ.get(
                        "AVATAR_YOUTUBE_CONVERTER", "youtube-disguise"),
                    persona=persona_key,
                    smooth_transition=bool(self.youtube_smooth_var.get()),
                    monitor=True,
                    status_callback=_audio_status,
                )
                self._youtube_audio = player
                player.set_muted(self._youtube_muted)
                self._wait_for_youtube_video_ready(url)
                if current_slot is not None:
                    self._set_youtube_slot_status(current_slot, "VIDEO READY - starting audio")
                player.start(url, start_seconds=start_s, end_seconds=end_s)
                self._youtube_title = player.title
                self._youtube_duration = player.duration
                self._set_youtube_mode("youtube")
                self._set_youtube_progress(85, "Starting altered original voice...")
                if current_slot is not None:
                    self._set_youtube_slot_status(
                        current_slot,
                        "PLAYING - next videos downloading in background")
                self._prewarm_youtube_queue()
                self._log_msg(
                    f"[youtube] original performance: {player.title}"
                    f"{self._youtube_range_label(start_s, end_s)}"
                    f" -> {persona_label}; voice transform only")
            except Exception as exc:
                self._youtube_audio_mode = False
                self._youtube_audio_status = "failed"
                self._youtube_audio = None
                self._set_youtube_progress(0, f"Failed: {exc}")
                if current_slot is not None:
                    self._set_youtube_slot_status(current_slot, f"FAILED: {exc}")
                self._log_msg(f"[youtube] real audio failed: {exc}")
                try:
                    if self.tts is not None:
                        self.tts.set_muted(False)
                except Exception:
                    pass
            finally:
                def _done():
                    self._youtube_busy = False
                    self.youtube_audio_btn.configure(
                        state="normal",
                        text="ALTER REAL YOUTUBE VOICE")
                try:
                    self.root.after(0, _done)
                except Exception:
                    self._youtube_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _advance_youtube_queue_after_current(self):
        if not getattr(self, "_youtube_queue_active", False):
            return False
        if getattr(self, "_youtube_queue_advancing", False):
            return False
        urls = getattr(self, "_youtube_queue_urls", []) or []
        next_index = int(getattr(self, "_youtube_queue_index", 0) or 0) + 1
        if next_index >= len(urls):
            self._youtube_queue_active = False
            self._youtube_queue_advancing = False
            self._log_msg("[youtube] playlist finished all videos.")
            try:
                if self.tts is not None:
                    self.tts.set_muted(False)
            except Exception:
                pass
            return False
        self._youtube_queue_index = next_index
        self._youtube_queue_advancing = True
        next_slot = None
        slots = getattr(self, "_youtube_queue_slots", []) or []
        if next_index < len(slots):
            next_slot = slots[next_index]
            self._set_youtube_slot_status(
                next_slot, f"STARTING NOW - video {next_index + 1}/{len(urls)}")
        self._set_youtube_progress(
            2, f"Starting next video {next_index + 1}/{len(urls)}...")
        self._log_msg(
            f"[youtube] video ended; switching to playlist video "
            f"{next_index + 1}/{len(urls)}.")

        def _start_next():
            self._youtube_queue_advancing = False
            try:
                self.speak_youtube_audio(queue_advance=True)
            except Exception as exc:
                self._log_msg(f"[youtube] playlist advance failed: {exc}")

        try:
            self.root.after(1, _start_next)
        except Exception:
            _start_next()
        return True

    def youtube_next_video(self):
        self._jump_youtube_queue(1)

    def youtube_previous_video(self):
        self._jump_youtube_queue(-1)

    def _jump_youtube_queue(self, step):
        rows = self._youtube_link_rows()
        if not rows:
            self._log_msg("[youtube] add YouTube links before using next/back.")
            return False
        if not getattr(self, "_youtube_queue_urls", None):
            self._prepare_youtube_queue(rows[0][1], auto_prewarm=False)
        urls = getattr(self, "_youtube_queue_urls", []) or []
        slots = getattr(self, "_youtube_queue_slots", []) or []
        if not urls:
            return False
        current = int(getattr(self, "_youtube_queue_index", 0) or 0)
        target = max(0, min(len(urls) - 1, current + int(step or 0)))
        if target == current:
            direction = "next" if int(step or 0) > 0 else "back"
            self._log_msg(f"[youtube] no {direction} video available.")
            return False
        old_slot = slots[current] if current < len(slots) else None
        target_slot = slots[target] if target < len(slots) else None
        if old_slot is not None:
            self._set_youtube_slot_status(old_slot, "SKIPPED BY USER")
        if target_slot is not None:
            self._set_youtube_slot_status(
                target_slot, f"STARTING NOW - video {target + 1}/{len(urls)}")
        self._youtube_queue_index = target
        self._youtube_queue_active = len(urls) > 1
        self._youtube_queue_advancing = False
        self._youtube_start_seconds = 0.0
        self._youtube_end_seconds = None
        self._youtube_clock_reset(0.0, running=False)
        try:
            if self._youtube_audio is not None:
                self._youtube_audio.stop()
                self._youtube_audio = None
        except Exception:
            pass
        self._youtube_busy = False
        self._youtube_audio_mode = False
        self._youtube_audio_status = ""
        self._youtube_chunks = []
        self._youtube_index = 0
        self._set_youtube_progress(
            2, f"User selected video {target + 1}/{len(urls)}")
        self._log_msg(
            f"[youtube] user selected playlist video {target + 1}/{len(urls)}.")
        if not self.running or self.engines is None:
            self._attach_youtube_scene(urls[target], force=True, preserve_crop=True)
            self._log_msg("[studio] press START to play audio for the selected video.")
            self._sync_youtube_status()
            return True
        try:
            self.speak_youtube_audio(queue_advance=True)
        except Exception as exc:
            self._log_msg(f"[youtube] jump failed: {exc}")
            if target_slot is not None:
                self._set_youtube_slot_status(target_slot, f"FAILED: {exc}")
            return False
        return True

    def _on_youtube_persona_change(self, event=None):
        """Apply a newly selected persona immediately to active YouTube audio."""
        label = self.youtube_persona_var.get()
        self._log_msg(f"[youtube] alter voice persona -> {label}")
        if not (self._youtube_audio is not None
                and self._youtube_audio_mode
                and self._youtube_mode == "youtube"):
            return
        resume_at = None
        try:
            resume_at = float(getattr(self._youtube_audio, "position_seconds", 0.0) or 0.0)
        except Exception:
            resume_at = None
        try:
            self._youtube_audio.stop()
        except Exception:
            pass
        self._youtube_audio = None
        self._youtube_audio_mode = False
        self._youtube_busy = False
        self._log_msg(
            f"[youtube] continuing from {self._fmt_time(resume_at or 0.0)} with {label}")
        self.root.after(
            120, lambda: self.speak_youtube_audio(start_override=resume_at))

    def _start_youtube_pump(self):
        if self._youtube_pump_started:
            return
        self._youtube_pump_started = True

        def _pump():
            import time as _t
            while True:
                try:
                    if not getattr(self, "running", False):
                        _t.sleep(0.5)
                        continue
                    if self._youtube_mode != "youtube":
                        _t.sleep(0.25)
                        continue
                    if not self._youtube_chunks:
                        _t.sleep(0.25)
                        continue
                    if self._youtube_index >= len(self._youtube_chunks):
                        self._set_youtube_progress(100, "YouTube speech finished")
                        self._set_youtube_mode("market")
                        self._log_msg("[youtube] finished. Market analysis resumed.")
                        _t.sleep(1.0)
                        continue
                    if self.tts is None or getattr(self.tts, "pending", 0) >= 2:
                        _t.sleep(0.25)
                        continue
                    chunk = self._youtube_chunks[self._youtube_index]
                    self._youtube_index += 1
                    self._speak_exclusive(chunk, priority=0)
                    self._set_youtube_progress(
                        self._youtube_playback_progress(),
                        f"Speaking chunk {self._youtube_index}/{len(self._youtube_chunks)}")
                    if (self._youtube_index % 10) == 0:
                        self._log_msg(
                            f"[youtube] progress {self._youtube_index}/{len(self._youtube_chunks)}")
                    try:
                        self.root.after(0, self._sync_youtube_status)
                    except Exception:
                        pass
                except Exception as exc:
                    self._log_msg(f"[youtube] pump error: {exc}")
                    _t.sleep(1.0)

        threading.Thread(target=_pump, daemon=True).start()

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
        if self.live_mic is not None:
            self._log_msg("[studio] Live Mic is on — answer your viewers out loud into the mic (ASK is disabled).")
            return
        if self.brain is None or not self.brain.ok:
            why = self.brain.startup_check()[1] if self.brain else "brain not started"
            self._log_msg(f"[studio] AI brain unavailable ({why}) — speaking as-is.")
            self._speak_exclusive(txt)
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
                self._speak_exclusive(reply)
            else:
                self._log_msg("[studio] no answer — speaking your text as-is.")
                self._speak_exclusive(txt)
        threading.Thread(target=_think, daemon=True).start()

    # ----- AUTO-TALK: continuous self-generated gold commentary -----------------
    # MARKET-focused beats — used when gold is actually moving (lean into analysis,
    # levels, reactions). Mixed length so pacing stays unpredictable/human.
    _MARKET_BEATS = [
        "Give a live update on the ACTIVE MARKET and what you're watching.",
        "Call out active-market support or resistance and what confirms the next move.",
        "React to the current move using the exact fresh data and nearest level.",
        "Give a concise bullish, bearish, or neutral read on the active market.",
        "Walk the chat through the gold chart right now — the trend and the nearest level that matters.",
        "Point out what gold is doing around the current price and the level you'd watch for a move.",
    ]
    # ENGAGEMENT-focused beats — used when the market is quiet (work the chat, CTAs,
    # questions, hype) so there's never dead air.
    _ENGAGE_BEATS = [
        "Fire off a punchy short line telling the chat to smash like. One sentence max.",
        "Push the gift goal — tell viewers to send a rose to unlock the next gold signal.",
        "Ask the chat a fun question about gold or their trades, then IMMEDIATELY answer it yourself and keep rolling — never leave a silent pause waiting.",
        "Welcome the room warmly, call out that you see new people coming in, and tell them to hit follow and smash the like.",
        "Tease that a big gold signal is coming up soon and tell them to send a rose to unlock it — build anticipation so nobody leaves.",
        "Banter with the chat — answer the vibe of the room, shout people out, keep it warm and fun.",
    ]
    # SHORT snappy one-breath lines — FAST to generate. Used to refill the voice
    # buffer the instant it runs dry so the avatar never goes silent. Chart-flavoured
    # so even the filler keeps the show on gold.
    _SHORT_BEATS = [
        "One accurate line reacting to the ACTIVE MARKET right now.",
        "Quick call: is the active market pushing, fading, or ranging?",
        "Say the current price and exact next level you're watching.",
        "One quick line hyping the chat to smash that like while you watch gold.",
        "Your gut read on the gold chart this second — one punchy sentence.",
        "Drop a quick 'eyes on this level' callout for gold. One short line.",
    ]
    # DEEP dives — long, passionate, rambling. Only fired when there's a voice cushion
    # (a queued line already playing) so the slow generation can't create dead air.
    _DEEP_BEATS = [
        "Take your time and go DEEP — read the real technicals out loud (trend, RSI, the exact support and resistance) and walk the chat through your full thinking like a sharp analyst, ramble a bit.",
        "Tell a little story or tangent about trading gold — a lesson, a past move — then bring it back to THIS move today.",
        "Really hype the room for a while — build the energy up, talk to the chat, react, go on a passionate rant about where gold is headed.",
        "Break down your whole game plan on gold from here — the level you want, your invalidation, and what gets you excited, out loud like a pro.",
    ]
    _AUTOTALK_BEATS = _MARKET_BEATS + _ENGAGE_BEATS    # union (compat)

    def _select_active_market(self, force=False):
        """Route open XAUUSD hours to gold and closed hours to live BTCUSD."""
        from datetime import datetime
        from market_session import xauusd_session, spoken_reopen

        session = xauusd_session()
        symbol = "XAUUSD" if session.is_open else "BTCUSD"
        md = self.market_gold if session.is_open else self.market_btc
        changed = force or symbol != self._market_symbol
        self.market = md
        self._market_symbol = symbol
        self._market_session = session
        if changed:
            self._ma_hist = None
            self._ma_prev = None
            if self.brain_pool is not None:
                try:
                    self.brain_pool.clear()
                except Exception:
                    pass
            chart = self.engines.get("chart") if self.engines else None
            price = float(md.price) if md is not None else 0.0
            if chart is not None and price > 0:
                chart.set_market(symbol, price)
            if not force and self._tv_proc is not None:
                try:
                    self._tv_proc.terminate()
                except Exception:
                    pass
                self._tv_proc = None
                tv_symbol = "OANDA:XAUUSD" if symbol == "XAUUSD" else "BINANCE:BTCUSDT"
                self.root.after(0, lambda s=tv_symbol: self._launch_tradingview(s))
            if not force:
                if session.is_open:
                    prompt = (
                        "Gold has reopened. Tell viewers naturally that XAUUSD is open "
                        "again, switch back to gold, and give one fresh observation.")
                else:
                    reopen = spoken_reopen(session, datetime.now().astimezone().tzinfo)
                    prompt = (
                        f"Tell viewers naturally that XAUUSD is closed for the "
                        f"{session.reason} and reopens {reopen}. Explain that you are "
                        "switching to live BTCUSD analysis now.")
                try:
                    self._event_q.put_nowait(("market", prompt + self._live_market_ctx()))
                except Exception:
                    pass
            self._log_msg(f"[market] active instrument -> {symbol}")
        return symbol, md, session

    def _queue_initial_market_status(self):
        """Make the first market line explain a closed-gold BTC handoff."""
        from datetime import datetime
        from market_session import spoken_reopen

        symbol, _md, session = self._select_active_market()
        if symbol != "BTCUSD":
            return
        reopen = spoken_reopen(session, datetime.now().astimezone().tzinfo)
        prompt = (
            f"Open with a natural live update: XAUUSD is closed for the "
            f"{session.reason} and reopens {reopen}. Tell viewers you are switching "
            "to BTCUSD, then give one concise Bitcoin observation from the exact data.")
        try:
            self._event_q.put_nowait(("market", prompt + self._live_market_ctx()))
        except Exception:
            pass

    def _live_market_ctx(self):
        """Build a REAL-TIME market context string for the brain — the actual live
        gold price + recent % move, fetched at THIS moment. Falls back to the
        simulated chart only if the live feed is down. Also keeps the on-screen
        chart price synced to reality so the visuals match the commentary."""
        from datetime import datetime
        symbol, md, session = self._select_active_market()
        price = None
        tk = {}
        if md is not None:
            try:
                tk = md.live_ticker() or {}          # EXACT real-time price + 24h stats
                price = tk.get("price") or md.price
                if price and price > 0:
                    chart = self.engines.get("chart") if self.engines else None
                    if chart is not None:
                        chart.set_market(symbol, price)
            except Exception:
                price = None
        quote_age = max(0.0, time.time() - getattr(md, "_tick_t", 0.0)) if md else 1e9
        if not price or quote_age > 10.0:
            return (f" ACTIVE MARKET: {symbol}. LIVE DATA IS UNAVAILABLE OR STALE. "
                    "Say data is temporarily unavailable and do not quote a price, "
                    "level, trend, RSI, signal, or trade idea.")
        # assemble the PRECISE, real, this-second data the model must reason over
        facts = [f"price ${price:,.2f}", f"quote age {quote_age:.1f} seconds"]
        if tk:
            facts.append(f"24h change {tk['change_pct']:+.2f}%")
            facts.append(f"24h high ${tk['high']:,.0f}, 24h low ${tk['low']:,.0f}")
            rng = tk["high"] - tk["low"]
            if rng > 0:
                facts.append(f"trading {(price - tk['low']) / rng * 100:.0f}% up its 24h range")
        try:
            import market_ta
            a = market_ta.analyze(md.snapshot()) if md is not None else None
            if a:
                if a.get("rsi") is not None:
                    facts.append(f"RSI {a['rsi']:.0f}")
                facts.append(f"short-term trend {a['trend']}")
                res, sup = a.get("resistance"), a.get("support")
                if res:
                    facts.append(f"nearest resistance ${res:,.0f} ({(res - price) / price * 100:+.2f}%)")
                if sup:
                    facts.append(f"nearest support ${sup:,.0f} ({(sup - price) / price * 100:+.2f}%)")
        except Exception:
            pass
        data = "; ".join(facts)
        news = getattr(self, "_gold_news", "") if symbol == "XAUUSD" else ""
        news_clause = (f" Latest real market headline: \"{news}\" — you MAY weave in why "
                       "gold is moving if it fits, but never invent news.") if news else ""
        if symbol == "XAUUSD":
            source = ("XAUUSD is open. The live source is Binance PAXGUSDT, a "
                      "tokenized-gold proxy, not an exact OTC XAUUSD broker quote.")
            session_note = ""
        else:
            from market_session import spoken_reopen
            reopen = spoken_reopen(session, datetime.now().astimezone().tzinfo)
            source = "XAUUSD is closed. The active source is live Binance BTCUSDT."
            session_note = (
                f" State naturally that gold is closed for the {session.reason} and "
                f"reopens {reopen}; then analyze Bitcoin, not gold.")
        return (f" ACTIVE MARKET DATA RIGHT NOW: {symbol}; {data}. {source}{session_note}"
                f"{news_clause} Use ONLY these supplied numbers. Never invent a price, "
                "level, candle, news event, market status, or future outcome. Separate "
                "observed facts from conditional scenarios. Analyze price versus support "
                "and resistance, trend, RSI, range, and invalidation. Say every number "
                "in plain spoken words. Do not use markdown or math notation.")
        return (f" LIVE GOLD DATA RIGHT NOW (XAUUSD, real, this second): {data}.{news_clause} "
                "Reason like a sharp analyst USING ONLY THESE EXACT NUMBERS — never "
                "invent a price or level. Read what's ACTUALLY happening: where price "
                "sits vs support/resistance, the trend, RSI, the 24h move, and what you "
                "are watching next. Be precise, accurate and real, not vague hype. "
                "Say every number in plain spoken words like a person talking out loud. "
                "NEVER use dollar signs, LaTeX, backslashes, math notation or markdown — "
                "just speak the numbers.")

    def _check_market_alert(self):
        """Detect a SIGNIFICANT live gold event (round-level cross or sharp move) and
        return a brain prompt to announce it live, or None. Tracks price history; rate-
        limited so it doesn't spam."""
        symbol, md, _session = self._select_active_market()
        if md is None:
            return None
        try:
            price = float(md.price)
        except Exception:
            return None
        quote_age = time.time() - getattr(md, "_tick_t", 0.0)
        if not price or price <= 0 or quote_age > 10.0:
            return None
        now = time.monotonic()
        if not getattr(self, "_ma_hist", None):
            import collections as _c
            self._ma_hist = _c.deque(maxlen=60)
            self._ma_last = 0.0
            self._ma_prev = price
        self._ma_hist.append((now, price))
        if now - self._ma_last < 75:        # at most ~1 alert / 75s
            self._ma_prev = price
            return None
        alert = None
        lvl = 25 if symbol == "XAUUSD" else 500
        asset = "gold" if symbol == "XAUUSD" else "Bitcoin"
        if int(price // lvl) != int(self._ma_prev // lvl):     # crossed a round level
            crossed = (int(price // lvl) * lvl if price > self._ma_prev
                       else (int(price // lvl) + 1) * lvl)
            alert = (f"BREAKING: {asset} just crossed ${crossed:,} and is now ${price:,.0f}. "
                     "React live to this level break — support, resistance, or breakout? "
                     "One short, energetic sentence.")
        else:                               # sharp % move over ~90s
            ref = next((p for t, p in self._ma_hist if now - t >= 80), None)
            if ref:
                pct = (price - ref) / ref * 100.0
                move_threshold = 0.22 if symbol == "XAUUSD" else 0.35
                if abs(pct) >= move_threshold:
                    d = "spiking UP" if pct > 0 else "DROPPING"
                    alert = (f"Gold is {d} fast — now ${price:,.0f}, "
                             f"{'+' if pct > 0 else '-'}{abs(pct):.2f}% in minutes. React live to "
                             "this move and what it means for traders. Short and energetic.")
        if alert and symbol == "BTCUSD":
            alert = alert.replace("Gold", "Bitcoin").replace("gold", "Bitcoin")
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
                    if self._market_symbol == "XAUUSD" and cal is not None:
                        news = cal.next_alert()
                        if news:
                            self._event_q.put_nowait(("market", news))
                    if self._market_symbol == "XAUUSD":
                        self._refresh_gold_news()
                    self._poll_tick()                    # buy/sell poll lifecycle
            except Exception:
                pass
            _t.sleep(8)

    def _refresh_gold_news(self):
        """Cache the top real gold-market headline (~every 6 min) so the host can
        reference WHY gold is moving, grounded in actual news, not made up."""
        import time as _t
        now = _t.monotonic()
        if now - getattr(self, "_news_t", 0.0) < 360:
            return
        self._news_t = now
        def _fetch():
            try:
                import web_research
                res = web_research.research("gold XAUUSD price today news driver fed dollar",
                                            max_results=3)
                if res:
                    # keep the first concrete headline line
                    for ln in res.splitlines():
                        ln = ln.strip(" -•").strip()
                        if len(ln) > 30:
                            self._gold_news = ln[:200]
                            break
            except Exception:
                pass
        threading.Thread(target=_fetch, daemon=True).start()

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
                        self._render_stall_count = getattr(
                            self, "_render_stall_count", 0) + 1
                        self._render_recovery_until = _t.monotonic() + 45.0
                        self._log_msg(
                            "[watchdog] render stalled >20s; entering light recovery "
                            "and refreshing camera feed.")
                        try:
                            released = False
                            lock = getattr(self, "_camera_lock", None)
                            if lock is not None and lock.acquire(blocking=False):
                                try:
                                    cap, self.cap = self.cap, None
                                finally:
                                    lock.release()
                                if cap is not None:
                                    try:
                                        cap.release()
                                    except Exception:
                                        pass
                                released = True
                            if released and getattr(self, "camera_enabled", False):
                                threading.Thread(
                                    target=self._enable_camera_capture,
                                    daemon=True).start()
                        except Exception:
                            pass
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
        LEAD = 2                  # fresher analysis while retaining one-line cushion
        i = 0
        while getattr(self, "running", False):
            try:
                tts = self.tts
                if (self.live_mic is not None       # YOU own the mouth in Live-Mic mode
                        or getattr(self, "_youtube_mode", "market") == "youtube"
                        or tts is None):
                    _t.sleep(0.4)
                    continue
                # YIELD to the user: if you just ASKed/SPEAKed, stay quiet so the
                # avatar answers YOU and doesn't talk over it.
                if _t.monotonic() < self._user_active_until:
                    _t.sleep(0.3)
                    continue
                if self._ready_playback_active:
                    _t.sleep(0.1)
                    continue
                # PRIORITY 1 (checked BEFORE the buffer pace): react to gifts /
                # follows / shares / like-milestones IMMEDIATELY — thank supporters
                # without waiting for the line buffer to drain.
                if self._ready_speech_snapshot() is not None:
                    _t.sleep(0.15)
                    continue
                # PRIORITY 2: ANSWER THE CHAT in real time — checked BEFORE the buffer
                # pace, so a busy voice queue never delays a viewer's reply. If a
                # comment is waiting, answer it now (the reply jumps the filler queue);
                # only fall through to commentary when the chat is quiet.
                if (not getattr(self, "autotalk_var", None)
                        or not self.autotalk_var.get()
                        or self.brain is None or not self.brain.ok):
                    _t.sleep(0.4)
                    continue
                # pace COMMENTARY (filler only) to the look-ahead so we don't over-queue.
                if tts.pending > LEAD:
                    _t.sleep(0.2)
                    continue
                import random as _r
                active = self._market_active()
                mode = "market" if (active or _r.random() < 0.78) else "engage"
                # keep talking with no dead air: instant pre-generated line first
                line = None
                if self.brain_pool is not None:
                    line = self.brain_pool.get(timeout=0.1)
                if line is None:
                    pool = self._MARKET_BEATS if mode == "market" else self._ENGAGE_BEATS
                    pend = getattr(self.tts, "pending", 0) or 0
                    if pend <= 0:
                        beat = self._SHORT_BEATS[i % len(self._SHORT_BEATS)]
                    elif pend >= 2 and i % 7 == 0:
                        beat = self._DEEP_BEATS[i % len(self._DEEP_BEATS)]
                    else:
                        beat = pool[i % len(pool)]
                    i += 1
                    ctx = self._live_market_ctx()    # REAL gold price, fetched NOW
                    try:
                        line = self._generate(beat + ctx)   # ONE-at-a-time brain access
                    except Exception as exc:
                        self._log_msg(f"[autotalk] brain: {exc}")
                # if you interacted while it was generating, drop this line
                if line and self.autotalk_var.get() and _t.monotonic() >= self._user_active_until:
                    self._log_msg("avatar> " + line)
                    self._speak_exclusive(line)
                    self._last_spoke_t = _t.monotonic()
            except Exception as exc:
                self._log_msg(f"[autotalk] {exc}")
                _t.sleep(2.0)

    def _on_comment(self, user, text):
        """Called from the TikTok reader thread for every live comment — queue it
        (or count it as a poll vote if a buy/sell poll is running)."""
        try:
            self._mark_viewer(user, comments=1)
            self._comment_times.append(time.time())
            self._feed_msg(f"{user}:  {text}", "q")     # show ALL comments live
            self._queue_comment_reader_voice(user, text)
            poll = self._poll
            if poll is not None:
                t = (text or "").strip().lower()
                if t in ("1", "buy", "long", "buy gold", "bull"):
                    poll["buy"] += 1; return
                if t in ("2", "sell", "short", "sell gold", "bear"):
                    poll["sell"] += 1; return
            if bool(getattr(self, "comments_var", None) and self.comments_var.get()):
                self._comment_q.put_nowait((user, text))
                self._live_response_event.set()
        except Exception:
            pass        # queue full = drop (we're behind on a comment flood)

    def _queue_comment_reader_voice(self, user, text):
        if not bool(getattr(self, "comment_voice_var", None)
                    and self.comment_voice_var.get()):
            return False
        line = self._format_comment_reader_voice(user, text)
        if not line:
            return False
        try:
            self._comment_voice_q.put_nowait((time.monotonic(), line))
            self._comment_voice_event.set()
            return True
        except Exception:
            return False

    def _format_comment_reader_voice(self, user, text):
        user = self._clean_comment_voice_user(user)
        text = self._clean_comment_voice_text(text, limit=180)
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith(("http://", "https://")) or " tiktok.com/" in lowered:
            return ""
        key = (user.lower(), text.lower())
        now = time.monotonic()
        seen = getattr(self, "_comment_voice_seen", None)
        if seen is None:
            seen = self._comment_voice_seen = {}
        # Do not repeat the same viewer/comment spam inside a short live burst.
        last = float(seen.get(key, 0.0) or 0.0)
        if now - last < 45.0:
            return ""
        seen[key] = now
        if len(seen) > 200:
            cutoff = now - 120.0
            for old_key, ts in list(seen.items()):
                if ts < cutoff:
                    seen.pop(old_key, None)
        return f"{user} says: {text}" if user else text

    @staticmethod
    def _clean_comment_voice_user(value):
        import re
        user = str(value or "")
        user = re.sub(r"\s+", " ", user).strip(" @:|-")
        if len(user) > 32:
            user = user[:32].strip()
        return user

    @staticmethod
    def _clean_comment_voice_text(value, limit=180):
        import re
        text = str(value or "")
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip(" :|-")
        if not text:
            return ""
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0].strip()
        return text

    def _comment_reader_voice_loop(self):
        while not getattr(self, "_live_stop", False):
            event = getattr(self, "_comment_voice_event", None)
            if event is None:
                time.sleep(0.5)
                continue
            event.wait(timeout=1.0)
            event.clear()
            while not getattr(self, "_live_stop", False):
                try:
                    q = getattr(self, "_comment_voice_q", None)
                    if q is None or q.empty():
                        break
                    if not bool(getattr(self, "comment_voice_var", None)
                                and self.comment_voice_var.get()):
                        self._drain_comment_reader_voice()
                        break
                    if (not getattr(self, "running", False)
                            or self.tts is None
                            or self.live_mic is not None
                            or getattr(self, "_youtube_mode", "market") == "youtube"
                            or self._ready_playback_active
                            or self._ready_speech_any()):
                        time.sleep(0.35)
                        continue
                    if getattr(self.tts, "pending", 0) > 0:
                        time.sleep(0.25)
                        continue
                    gap = float(os.environ.get("AVATAR_COMMENT_READER_GAP", "3.5"))
                    wait = gap - (time.monotonic() - self._comment_voice_last_t)
                    if wait > 0:
                        time.sleep(min(wait, 0.5))
                        continue
                    _ts, line = q.get_nowait()
                    if self._speak_exclusive(line, priority=0):
                        self._comment_voice_last_t = time.monotonic()
                        self._log_msg(f"[comment voice] {line}")
                except queue.Empty:
                    break
                except Exception as exc:
                    self._log_msg(f"[comment voice] {exc}")
                    time.sleep(1.0)

    def _drain_comment_reader_voice(self):
        q = getattr(self, "_comment_voice_q", None)
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass

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
    # Reactions are built INSTANTLY from offline templates (engines/reactions.py,
    # no LLM) the moment the event fires, and pushed to the TOP-priority queue so
    # the avatar thanks the supporter immediately — gifts/follows lead the front.
    @staticmethod
    def _reactions():
        try:
            import reactions
            return reactions
        except Exception:
            return None

    def _on_gift(self, user, gift, count, coins):
        try:
            self._log_msg(f"🎁 {user} sent {count}x {gift} ({coins} coins)")
            self._feed_msg(f"\U0001f381 {user} sent {count}x {gift} ({coins} coins)", "ev")
            self._sess_coins += max(0, int(coins))
            self._mark_viewer(user, gifts=max(1, int(count or 1)), coins=max(0, int(coins or 0)))
            rx = self._reactions()
            if rx is not None:                    # gifts to the FRONT (lead priority)
                self._stage_urgent_event(
                    self._with_live_room_context(rx.ready_gift(user, gift, coins)),
                    f"gift from {user}", front=True)
            if self._sess_coins >= self._coin_goal:           # gift goal reached!
                reached = self._coin_goal
                self._coin_goal += int(os.environ.get("AVATAR_COIN_GOAL", "200"))
                if rx is not None:
                    self._prio_events.append(
                        self._with_live_room_context(rx.goal(reached)))
                    self._live_response_event.set()
        except Exception:
            pass

    def _on_follow(self, user):
        try:
            self._sess_follows += 1
            self._feed_msg(f"➕ {user} followed", "ev")
            self._mark_viewer(user, follows=1)
            self._log_msg(f"[follow] received from {user}")
            self._queue_follow_thanks(user)
        except Exception:
            pass

    def _queue_follow_thanks(self, user):
        """Debounce follows so bursty follow spam becomes one human thank-you."""
        name = (user or "").strip()
        if not name:
            return
        now = time.monotonic()
        if not self._pending_follows:
            self._follow_batch_first_t = now
        if name not in self._pending_follows:
            self._pending_follows.append(name)
        try:
            pending = getattr(self, "_follow_batch_after", None)
            if pending is not None:
                self.root.after_cancel(pending)
        except Exception:
            pass
        elapsed = now - float(getattr(self, "_follow_batch_first_t", now) or now)
        delay_ms = 80 if elapsed >= 2.2 or len(self._pending_follows) >= 10 else 1100
        try:
            self._follow_batch_after = self.root.after(
                delay_ms, self._flush_follow_batch)
        except Exception:
            self._follow_batch_after = None
            self._live_response_event.set()

    def _flush_follow_batch(self):
        self._follow_batch_after = None
        self._follow_batch_first_t = 0.0
        self._live_response_event.set()

    def _on_share(self, user):
        try:
            self._feed_msg(f"↪ {user} shared the stream", "ev")
            self._mark_viewer(user, shares=1)
            rx = self._reactions()
            if rx is not None:
                self._log_msg(f"[share] received from {user}")
                self._stage_urgent_event(
                    self._with_live_room_context(rx.ready_share(user)),
                    f"share from {user}")
        except Exception:
            pass

    def _live_room_context(self):
        """Short spoken room snapshot for live reactions and comment answers."""
        try:
            parts = []
            viewers = getattr(self, "_sess_viewers", None)
            if viewers is not None:
                parts.append(f"{max(0, int(viewers)):,} viewers")
            likes = max(0, int(getattr(self, "_sess_likes", 0) or 0))
            if likes:
                parts.append(f"{likes:,} likes")
            if not parts:
                return ""
            if len(parts) == 1:
                return f"Right now we have {parts[0]} in the room."
            return f"Right now we have {parts[0]} and {parts[1]} in the room."
        except Exception:
            return ""

    def _with_live_room_context(self, text):
        ctx = self._live_room_context()
        text = (text or "").strip()
        if not text or not ctx:
            return text
        if "Right now we have" in text:
            return text
        return f"{text} [[CUT]] {ctx}"

    def _viewer_record(self, user):
        name = (user or "").strip()
        if not name:
            return None
        key = name.lower()
        rec = self._viewer_scores.get(key)
        if rec is None:
            rec = {
                "name": name, "comments": 0, "coins": 0, "gifts": 0,
                "follows": 0, "shares": 0, "likes": 0, "joins": 0,
                "last": time.monotonic(),
            }
            self._viewer_scores[key] = rec
        else:
            rec["name"] = name
            rec["last"] = time.monotonic()
        return rec

    def _mark_viewer(self, user, *, comments=0, coins=0, gifts=0,
                     follows=0, shares=0, likes=0, joins=0):
        rec = self._viewer_record(user)
        if rec is None:
            return
        rec["comments"] += int(comments or 0)
        rec["coins"] += int(coins or 0)
        rec["gifts"] += int(gifts or 0)
        rec["follows"] += int(follows or 0)
        rec["shares"] += int(shares or 0)
        rec["likes"] += int(likes or 0)
        rec["joins"] += int(joins or 0)

    @staticmethod
    def _viewer_score(rec):
        return (
            int(rec.get("coins", 0)) * 5
            + int(rec.get("gifts", 0)) * 90
            + int(rec.get("shares", 0)) * 45
            + int(rec.get("follows", 0)) * 35
            + int(rec.get("comments", 0)) * 18
            + int(rec.get("likes", 0)) * 2
            + int(rec.get("joins", 0))
        )

    def _top_viewers(self, limit=8):
        rows = list(getattr(self, "_viewer_scores", {}).values())
        rows.sort(key=lambda rec: (self._viewer_score(rec), rec.get("last", 0.0)),
                  reverse=True)
        return [rec for rec in rows if self._viewer_score(rec) > 0][:limit]

    @staticmethod
    def _join_names(names):
        names = [str(n).strip() for n in names if str(n).strip()]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    def _top_viewers_line(self):
        top = self._top_viewers(8)
        if not top:
            return ("Big love to everyone watching. Thank you for being here "
                    "and supporting the live.")
        names = [rec["name"] for rec in top[:5]]
        extra = len(top) - len(names)
        joined = self._join_names(names)
        if extra > 0:
            joined = f"{joined}, and {extra} more"
        top_gifter = next((rec for rec in top if int(rec.get("coins", 0)) > 0), None)
        gift_part = ""
        if top_gifter is not None:
            gift_part = f" Extra thanks to {top_gifter['name']} for the gifts."
        return (
            f"Shout out to our top viewers: {joined}.{gift_part} "
            "Thank you for supporting the live."
        ).replace("  ", " ").strip()

    def _speak_top_viewers(self):
        if self.tts is None or not self.running:
            self._log_msg("[viewers] press START first.")
            return
        line = self._top_viewers_line()
        try:
            self.tts.set_muted(False)
            self.tts.clear_pending(below=2)
        except Exception:
            pass
        accepted = self._speak_exclusive(line, priority=2)
        if not accepted:
            self._log_msg("[viewers] TTS rejected viewer shout-out")
            return
        self._log_msg("[viewers] top viewers shout-out queued")
        self._log_msg("avatar> " + line)
        self._last_spoke_t = time.monotonic()
        self._user_active_until = time.monotonic() + 5.0

    def _live_stream_ctx(self):
        market = ""
        try:
            market = self._live_market_ctx() or ""
        except Exception:
            market = ""
        room = self._live_room_context()
        if room:
            return (market + " " + f"Live room context: {room}").strip()
        return market

    def _stage_urgent_event(self, text, detail, front=False):
        """Make a pre-rendered acknowledgement green directly from TikTok."""
        text = self._supporter_text_only(text)
        if self._prepare_speech_button(
                text, kind="urgent", priority=2, detail=detail):
            return True
        if front:
            self._prio_events.appendleft(text)
        else:
            self._prio_events.append(text)
        self._live_response_event.set()
        return False

    @staticmethod
    def _supporter_text_only(text):
        """Keep appreciation focused on the person, not a detached room count."""
        text = (text or "").strip()
        marker = "[[CUT]] Right now we have"
        if marker in text:
            text = text.split(marker, 1)[0].strip()
        return text

    def _on_like(self, user, total):
        # celebrate only when we CROSS a milestone (likes fire constantly otherwise)
        try:
            self._mark_viewer(user, likes=1)
            self._sess_likes = max(self._sess_likes, int(total or 0))
            if total and total >= self._next_like_ms:
                rx = self._reactions()
                if rx is not None:
                    self._prio_events.append(
                        self._with_live_room_context(rx.likes(total)))
                    self._live_response_event.set()
                step = 500 if total < 1000 else (1000 if total < 10000 else 5000)
                self._next_like_ms = ((total // step) + 1) * step
        except Exception:
            pass

    def _on_viewers(self, total):
        """Receive TikTok's concurrent room count immediately when it changes."""
        try:
            self._sess_viewers = max(0, int(total or 0))
        except Exception:
            pass

    def _on_join(self, user):
        try:
            self._mark_viewer(user, joins=1)
        except Exception:
            pass

    def _react_one_event(self):
        """Prepare supporter reactions; keep market alerts and polls automatic."""
        try:
            if self.tts is None:
                return False
            current = self._ready_speech_snapshot("urgent")
            if current is not None and int(current.get("priority", 0)) >= 2:
                return False
            # 1) INSTANT appreciation — speak NOW, jump ahead of any filler commentary
            if self._prio_events:
                try:
                    txt = self._prio_events.popleft()
                except IndexError:
                    txt = None
                if txt:
                    return self._prepare_speech_button(
                        self._supporter_text_only(txt), kind="urgent", priority=2,
                        detail="supporter appreciation"
                    )
            # 1b) BATCHED follows — thank a burst of follows in one quick line
            if self._pending_follows:
                if getattr(self, "_follow_batch_after", None) is not None:
                    return False
                names = []
                seen = set()
                while self._pending_follows:
                    try:
                        name = self._pending_follows.popleft()
                    except IndexError:
                        break
                    key = str(name).strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        names.append(name)
                rx = self._reactions()
                txt = rx.follow_many(names) if rx is not None else None
                if txt:
                    return self._prepare_speech_button(
                        self._supporter_text_only(txt),
                        kind="urgent", priority=2, detail="new followers"
                    )
            # 2) market alerts / polls (secondary — may phrase via the brain)
            if self._event_q.empty():
                return False
            ev = self._event_q.get_nowait()
            kind = ev[0]
            reply = None
            if kind == "market":                   # autonomous — brain phrases the alert
                reply = self._generate(ev[1])
            elif kind == "goal":                   # legacy path (now usually instant)
                rx = self._reactions()
                reply = (rx.goal(ev[1]) if rx is not None
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
            if not reply:
                return False
            self._log_msg(f"avatar→{kind}> {reply}")
            try:
                self.tts.clear_pending(below=1)     # announcements clear filler, keep replies
            except Exception:
                pass
            self._speak_exclusive(reply, priority=1)
            self._last_spoke_t = time.monotonic()
            return True
        except Exception as exc:
            self._log_msg(f"[events] {exc}")
            return False

    def _live_response_loop(self):
        """Wake immediately for TikTok activity; no timed polling while connected."""
        while not getattr(self, "_live_stop", False):
            self._live_response_event.wait()
            self._live_response_event.clear()
            if getattr(self, "_live_stop", False):
                break
            try:
                if self.tts is None or not self.running:
                    continue
                if self._ready_playback_active:
                    continue
                if self._prio_events or self._pending_follows:
                    if self._ready_speech_snapshot("urgent") is None:
                        self._react_one_event()
                    continue
                if (self.brain is not None and self.brain.ok
                        and not self._comment_q.empty()
                        and self._ready_speech_snapshot("comment") is None):
                    if self._answer_one_comment():
                        self._last_comment_t = time.monotonic()
            except Exception as exc:
                self._log_msg(f"[live dispatcher] {exc}")

    def _answer_one_comment(self):
        """Prepare one worthwhile comment answer for release from the top button."""
        try:
            # lazily build the responder once the brain finished loading (the reader
            # may have started before START was pressed).
            if self.responder is None and self.brain is not None:
                try:
                    from comment_responder import CommentResponder
                    self.responder = CommentResponder(self.brain, get_context=self._live_stream_ctx)
                except Exception:
                    pass
            if self.responder is None or self._comment_q.empty() or self.tts is None:
                return False
            # drain a few at once but only answer ONE per cycle (keeps it fresh, not
            # a backlog read-out); newer comments matter more so take the latest.
            user, text = None, None
            while not self._comment_q.empty():
                user, text = self._comment_q.get_nowait()
            # filter + research + answer; on_answering fires the instant it commits
            # to a genuine answer, so the NOW-ANSWERING bar shows real ones only.
            reply = self.responder.respond(user, text, on_answering=self._set_answering)
            if reply:
                reply = self._with_live_room_context(reply)
                self._log_msg(f"↳ {user}: {text}")
                self._log_msg(f"avatar→{user}> {reply}")
                self._feed_msg(f"\U0001f916 → {user}:  {reply}", "a")
                return self._prepare_speech_button(
                    reply, kind="comment", priority=1,
                    detail=f"{user}: {text}", clear_answering=True
                )
            self._clear_answering()
            return False
        except Exception as exc:
            self._log_msg(f"[comments] {exc}")
            return False

    @staticmethod
    def _ready_speech_slot(kind):
        return "comment" if kind == "comment" else "urgent"

    def _ready_speech_snapshot(self, kind=None):
        lock = getattr(self, "_ready_speech_lock", None)
        if lock is None:
            return None
        with lock:
            slots = getattr(self, "_ready_speech_slots", None)
            if slots is None:
                item = self._ready_speech
                return dict(item) if item is not None else None
            if kind is not None:
                item = slots.get(self._ready_speech_slot(kind))
                return dict(item) if item is not None else None
            items = [item for item in slots.values() if item is not None]
            if not items:
                return None
            items.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
            return dict(items[0])

    def _ready_speech_any(self):
        return self._ready_speech_snapshot() is not None

    def _clear_ready_speech(self):
        """Discard staged audio and invalidate any preparation still in flight."""
        with self._ready_speech_lock:
            self._ready_speech_token += 1
            self._ready_speech = None
            self._ready_speech_deferred = None
            slots = getattr(self, "_ready_speech_slots", None)
            if slots is not None:
                for key in slots:
                    slots[key] = None
            deferred = getattr(self, "_ready_speech_deferred_slots", None)
            if deferred is not None:
                for key in deferred:
                    deferred[key] = None

    def _clear_ready_speech_slot(self, kind):
        slot = self._ready_speech_slot(kind)
        with self._ready_speech_lock:
            self._ready_speech_token += 1
            slots = getattr(self, "_ready_speech_slots", None)
            if slots is not None:
                slots[slot] = None
            deferred = getattr(self, "_ready_speech_deferred_slots", None)
            if deferred is not None:
                deferred[slot] = None
            if slot == "urgent":
                self._ready_speech = None
                self._ready_speech_deferred = None

    def _prepare_speech_button(self, text, kind, priority, detail="",
                               clear_answering=False):
        """Reserve the top button, render the voice, then mark it green."""
        text = (text or "").strip()
        if not text or self.tts is None:
            return False
        slot = self._ready_speech_slot(kind)
        with self._ready_speech_lock:
            slots = getattr(self, "_ready_speech_slots", None)
            if slots is None:
                slots = self._ready_speech_slots = {"urgent": None, "comment": None}
            deferred_slots = getattr(self, "_ready_speech_deferred_slots", None)
            if deferred_slots is None:
                deferred_slots = self._ready_speech_deferred_slots = {
                    "urgent": None, "comment": None}
            current = slots.get(slot)
            if current is not None and int(current.get("priority", 0)) >= int(priority):
                return False
            if current is not None:
                deferred_slots[slot] = {
                    key: current.get(key) for key in (
                        "text", "kind", "priority", "detail", "clear_answering"
                    )
                }
            self._ready_speech_token += 1
            token = self._ready_speech_token
            item = {
                "token": token, "text": text, "kind": kind,
                "priority": int(priority), "detail": detail,
                "status": "preparing", "clear_answering": bool(clear_answering),
            }
            slots[slot] = item
            if slot == "urgent":
                self._ready_speech = item
        try:
            self.tts.clear_pending(below=priority)
        except Exception:
            pass

        already_ready = False
        try:
            already_ready = bool(self.tts.is_prepared(text))
        except Exception:
            pass
        if already_ready:
            with self._ready_speech_lock:
                current = self._ready_speech_slots.get(slot)
                if current is not None and current.get("token") == token:
                    current["status"] = "ready"
            self._log_msg(f"[ready] {kind}: press the green top button")
            return True

        def _prepare():
            ok = False
            try:
                ok = bool(self.tts is not None and self.tts.prepare(text))
            except Exception as exc:
                self._log_msg(f"[ready voice] {exc}")
            with self._ready_speech_lock:
                current = self._ready_speech_slots.get(slot)
                if current is None or current.get("token") != token:
                    return
                if ok:
                    current["status"] = "ready"
                else:
                    self._ready_speech_slots[slot] = None
                    if slot == "urgent":
                        self._ready_speech = None
            if ok:
                self._log_msg(f"[ready] {kind}: press the green top button")
            elif clear_answering:
                self._clear_answering()

        threading.Thread(target=_prepare, daemon=True).start()
        return True

    def _play_ready_speech(self, kind=None):
        """Release prepared audio immediately when the green button is pressed."""
        slot = self._ready_speech_slot(kind) if kind is not None else None
        with self._ready_speech_lock:
            if slot is None:
                ready_items = [
                    item for item in self._ready_speech_slots.values()
                    if item is not None and item.get("status") == "ready"]
                ready_items.sort(
                    key=lambda item: int(item.get("priority", 0)), reverse=True)
                item = ready_items[0] if ready_items else None
                slot = self._ready_speech_slot(
                    item.get("kind")) if item is not None else "urgent"
            else:
                item = self._ready_speech_slots.get(slot)
            if item is None or item.get("status") != "ready":
                self._log_msg("[ready] click ignored: voice is not ready")
                return
        if self.tts is None or not self.running:
            self._log_msg("[ready] cannot speak: TTS is not running")
            return
        if self.live_mic is not None:
            self._log_msg("[ready] AI voice activated over Live Mic for acknowledgement")
        self._ready_playback_active = True
        self._user_active_until = time.monotonic() + 60.0
        try:
            self.tts.clear_pending()
            soft_interrupt = getattr(self.tts, "soft_interrupt_current", None)
            if callable(soft_interrupt):
                soft_interrupt()
        except Exception:
            pass
        accepted = self._speak_exclusive(
            item["text"], priority=int(item["priority"]))
        if not accepted:
            self._ready_playback_active = False
            self._user_active_until = time.monotonic() + 1.0
            self._log_msg("[ready] TTS rejected the line; button remains ready")
            return
        with self._ready_speech_lock:
            current = self._ready_speech_slots.get(slot)
            if current is None or current.get("token") != item.get("token"):
                return
            self._ready_speech_slots[slot] = None
            if slot == "urgent":
                self._ready_speech = None
            deferred = self._ready_speech_deferred_slots.get(slot)
            self._ready_speech_deferred_slots[slot] = None
        self._log_msg("[ready] pressed; speech queued")
        self._log_msg("avatar> " + item["text"])
        self._last_spoke_t = time.monotonic()
        threading.Thread(
            target=self._finish_ready_playback,
            args=(item, slot, deferred), daemon=True
        ).start()

    def _finish_ready_playback(self, item, slot, deferred):
        """Hold market speech until the selected acknowledgement fully ends."""
        deadline = time.monotonic() + 60.0
        started_at = time.monotonic()
        busy_seen = False
        idle_since = None
        while time.monotonic() < deadline:
            tts = self.tts
            if tts is None or not self.running:
                break
            busy = bool(
                getattr(tts, "pending", 0) > 0
                or getattr(tts, "synthesizing", False)
                or getattr(tts, "speaking", False)
            )
            if busy:
                busy_seen = True
                idle_since = None
            elif busy_seen:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= 0.35:
                    break
            elif time.monotonic() - started_at >= 2.0:
                break
            time.sleep(0.03)
        self._ready_playback_active = False
        self._user_active_until = time.monotonic() + 3.0
        if item.get("clear_answering"):
            self._clear_answering()
        if deferred is not None and slot == "comment":
            self._prepare_speech_button(**deferred)
        else:
            self._live_response_event.set()

    def _skip_ready_speech(self, kind):
        slot = self._ready_speech_slot(kind)
        item = self._ready_speech_snapshot(slot)
        if item is None:
            self._log_msg("[ready] nothing to skip")
            return
        self._clear_ready_speech_slot(slot)
        if slot == "comment" or item.get("clear_answering"):
            self._clear_answering()
        if slot == "urgent":
            try:
                self._pending_follows.clear()
            except Exception:
                pass
        self._log_msg(f"[ready] skipped {slot}")
        self._live_response_event.set()

    def _pause_youtube_for_ready_speech(self):
        """Silence YouTube output while keeping its playback clock advancing."""
        player = self._youtube_audio
        if player is None:
            return None
        with self._audio_handoff_lock:
            self._audio_handoff_token += 1
            token = self._audio_handoff_token
            if self._audio_handoff_state is None:
                self._audio_handoff_state = {
                    "player": player,
                    "was_running": bool(getattr(player, "_running", False)),
                    "was_paused": bool(getattr(player, "_paused", None)
                                       and player._paused.is_set()),
                    "youtube_duck_gain": float(
                        getattr(player, "duck_gain", 0.22)),
                    "youtube_output_target": float(
                        getattr(player, "_target_output_gain", 1.0)),
                    "tts_was_muted": bool(getattr(self.tts, "muted", False)),
                    "tts_voice_match": getattr(
                        self.tts, "_playback_match_persona", None),
                }
                try:
                    if self.tts is not None:
                        # AI acknowledgements must remain close, clear, and
                        # intelligible instead of inheriting YouTube's persona
                        # pitch/formant filter.
                        self.tts.set_playback_voice_match(None)
                    # Zero-gain ducking keeps FFmpeg consuming the live stream,
                    # position_seconds moving, and the UI clock current.
                    player.set_ducked(True, gain=0.0)
                    self._log_msg(
                        "[audio] YouTube voice muted; playback continues")
                except Exception as exc:
                    self._log_msg(
                        f"[ready] could not mute YouTube audio: {exc}")
            return token

    def _wait_and_restore_youtube(self, token):
        """Restore YouTube volume after the urgent TTS queue becomes idle."""
        deadline = time.monotonic() + 45.0
        started_at = time.monotonic()
        busy_seen = False
        idle_since = None
        while time.monotonic() < deadline:
            tts = self.tts
            if tts is None:
                break
            busy = bool(
                getattr(tts, "pending", 0) > 0
                or getattr(tts, "synthesizing", False)
                or getattr(tts, "speaking", False)
            )
            if busy:
                busy_seen = True
                idle_since = None
            elif busy_seen:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= 0.25:
                    break
            elif time.monotonic() - started_at >= 1.0:
                break
            time.sleep(0.02)
        self._restore_youtube_after_ready_speech(token)

    def _restore_youtube_after_ready_speech(self, token):
        if token is None:
            return
        with self._audio_handoff_lock:
            if token != self._audio_handoff_token:
                return
            state = self._audio_handoff_state
            self._audio_handoff_state = None
            if not state:
                return
            player = state.get("player")
            if player is None or player is not self._youtube_audio:
                return
            try:
                player.duck_gain = float(
                    state.get("youtube_duck_gain", 0.22))
                player._target_output_gain = float(
                    state.get("youtube_output_target", 1.0))
                if self.tts is not None:
                    self.tts.set_playback_voice_match(
                        state.get("tts_voice_match"))
                    self.tts.set_muted(
                        bool(state.get("tts_was_muted", False)))
                self._log_msg(
                    "[audio] AI speech finished; YouTube volume restored")
            except Exception as exc:
                self._log_msg(
                    f"[ready] could not restore YouTube volume: {exc}")

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

    def _set_answering(self, user, text, mode="answering"):
        """Show the comment the AI has just committed to answering. Thread-safe."""
        icon, col = (("\U0001f44b", "#27ff9e") if mode == "greeting"
                     else ("\U0001f4ad", "#ffd24d"))
        verb = "welcoming" if mode == "greeting" else "answering"
        t = (text or "").strip()
        if len(t) > 90:
            t = t[:90].rstrip() + "…"
        msg = f"{icon}  {verb} {user}:  {t}"
        bg = self._mix(SURFACE, MAG, 0.22)

        def _apply():
            try:
                self._answer_bar.configure(bg=bg)
                self.answering_lbl.configure(text=msg, fg=col, bg=bg)
            except Exception:
                pass
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _clear_answering(self):
        """Return the NOW-ANSWERING bar to idle. Thread-safe."""
        bg = self._mix(SURFACE, MAG, 0.22)

        def _apply():
            try:
                self.answering_lbl.configure(text="○  idle — waiting for a question",
                                             fg=MUTED, bg=bg)
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
            handle = (getattr(self, "_handle_text", "") or "").strip()
            if not handle:
                if last is not None:
                    self._set_live_light("none")
                    last = None
                _t.sleep(1.2)
                continue
            unique_id = handle.lstrip("@")
            display_handle = "@" + unique_id
            try:
                from TikTokLive import TikTokLiveClient
                # reuse ONE client per handle so we don't leak an HTTP session
                # on every poll over a multi-hour stream.
                if client is None or unique_id != client_handle:
                    client = TikTokLiveClient(unique_id=unique_id)
                    client_handle = unique_id
                live = bool(asyncio.get_event_loop().run_until_complete(
                    client.is_live()))
            except Exception:
                # A network/API failure is unknown, not an offline transition.
                # Keep the last confirmed state and retry on the next poll.
                _t.sleep(3.0)
                continue
            self._handle_live = live
            self._set_live_light("live" if live else "off")
            if live != last:
                self._log_msg(f"[live] {display_handle} is "
                              + ("LIVE \U0001f7e2 — comments incoming" if live
                                 else "offline \U0001f534"))
                if live:
                    self._save_handle(display_handle)
                if last is not None:
                    try:
                        from startup_sound import (
                            play_live_offline_sound,
                            play_live_online_sound,
                        )
                        if live:
                            play_live_online_sound()
                        else:
                            play_live_offline_sound()
                    except Exception:
                        pass
                last = live
            # auto-connect the comment reader the moment the stream goes live
            if live and self._wants_comment_reader() and self.tiktok is None:
                try:
                    self.root.after(0, self._connect_comment_reader)
                except Exception:
                    pass
            _t.sleep(3.0)

    def _load_handles(self):
        """Load the remembered @handle list (most-recent first, each once)."""
        try:
            import json
            with open(self._handles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                seen, out = set(), []
                for h in data:
                    h = str(h).strip()
                    if h and h.lower() not in seen:
                        seen.add(h.lower())
                        out.append(h)
                return out
        except Exception:
            pass
        return []

    def _save_handle(self, handle):
        """Remember a used @handle (deduped, most-recent first) + refresh the dropdown."""
        h = (handle or "").strip()
        if not h:
            return
        if not h.startswith("@"):
            h = "@" + h
        low = h.lower()
        cur = [x for x in getattr(self, "_handles", []) if x.lower() != low]
        self._handles = ([h] + cur)[:40]
        try:
            import json
            with open(self._handles_file, "w", encoding="utf-8") as f:
                json.dump(self._handles, f, ensure_ascii=False)
        except Exception:
            pass

        def _apply():
            try:
                self.handle_combo["values"] = self._handles
            except Exception:
                pass
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _on_handle_pick(self):
        """A handle was picked from the dropdown — it's now in the field; remember it
        (moves it to the top) so the live monitor switches to it on its next check."""
        h = (self.handle_var.get() or "").strip()
        if h:
            self._save_handle(h)
            self._log_msg(f"[comments] handle set to {h}")

    def _drain_queue(self, q):
        """Empty a Queue without raising."""
        try:
            while not q.empty():
                q.get_nowait()
        except Exception:
            pass

    def _on_comments_refresh(self):
        """RECONNECT the live comment reader for the current @handle and force a fresh
        live check — use it if the feed stalls or you switched handles. Drops stale
        queued comments so it doesn't replay an old backlog."""
        try:
            handle = (self.handle_var.get() or "").strip()
            self._drain_queue(self._comment_q)        # forget stale comments
            self._drain_comment_reader_voice()
            self._clear_ready_speech()
            self._clear_answering()
            if self.tiktok is not None:               # drop the old connection
                try:
                    self.tiktok.stop()
                except Exception:
                    pass
                self.tiktok = None
            self._set_live_light("none")              # monitor re-evaluates on next poll
            if handle and self._wants_comment_reader():
                self._connect_comment_reader()        # reconnect right now
                self._log_msg(f"[comments] refreshed — reconnecting {handle}…")
            else:
                self._log_msg("[comments] refreshed — re-checking live status…")
        except Exception as exc:
            self._log_msg(f"[comments] refresh failed: {exc}")

    def _on_comments_reset(self):
        """Clear the comment FEED and every queued comment / reaction (fresh slate).
        Leaves the connection + your @handle alone — just wipes what's on screen."""
        try:
            self._drain_queue(self._comment_q)
            self._drain_comment_reader_voice()
            self._drain_queue(self._event_q)
            try:
                self._prio_events.clear()
                self._pending_follows.clear()
                self._viewer_scores.clear()
            except Exception:
                pass
            self._clear_ready_speech()
            self._clear_answering()
            try:                                       # we're on the Tk thread here
                self.feed.configure(state="normal")
                self.feed.delete("1.0", "end")
                self.feed.insert("end", "feed cleared — waiting for comments…\n", "sys")
                self.feed.configure(state="disabled")
            except Exception:
                pass
            self._log_msg("[comments] feed reset.")
        except Exception as exc:
            self._log_msg(f"[comments] reset failed: {exc}")

    def _on_comment_voice(self):
        enabled = bool(getattr(self, "comment_voice_var", None)
                       and self.comment_voice_var.get())
        if enabled:
            if getattr(self, "tiktok", None) is None:
                self._connect_comment_reader()
            event = getattr(self, "_comment_voice_event", None)
            if event is not None:
                event.set()
        else:
            self._drain_comment_reader_voice()
        self._log_msg("[comment voice] " + ("ON" if enabled else "off"))

    def _wants_comment_reader(self):
        return bool((getattr(self, "comments_var", None) and self.comments_var.get())
                    or (getattr(self, "comment_voice_var", None)
                        and self.comment_voice_var.get()))

    def _connect_comment_reader(self):
        handle = (self.handle_var.get() or "").strip()
        if not handle:
            self._log_msg("[comments] enter your TikTok @handle first.")
            return False
        if not handle.startswith("@"):
            handle = "@" + handle
        self._save_handle(handle)
        from tiktok_comments import TikTokComments
        if self.responder is None and self.brain is not None:
            from comment_responder import CommentResponder
            self.responder = CommentResponder(self.brain, get_context=self._live_stream_ctx)
        if self.tiktok is None:
            self.tiktok = TikTokComments(handle, self._on_comment,
                                         on_gift=self._on_gift, on_follow=self._on_follow,
                                         on_like=self._on_like, on_share=self._on_share,
                                         on_viewers=self._on_viewers,
                                         on_join=self._on_join)
            self.tiktok.start()
        answer_note = "" if bool(getattr(self, "comments_var", None)
                                 and self.comments_var.get()) else " (read only)"
        brain_note = "" if self.brain is not None else " (answers begin after START)"
        self._log_msg(f"[comments] reading {handle}{answer_note} - comments + gifts/follows{brain_note}")
        return True

    def _on_comments(self):
        """Toggle the live TikTok comment responder on/off."""
        try:
            if not self.comments_var.get():
                if self.tiktok is not None and not self._wants_comment_reader():
                    self.tiktok.stop(); self.tiktok = None
                self._sess_viewers = None
                self._clear_ready_speech_slot("comment")
                self._drain_queue(self._comment_q)
                self._clear_answering()
                self._log_msg("[comments] answers off"
                              + ("; voice reader still on" if self.tiktok is not None else ""))
                return
            handle = (self.handle_var.get() or "").strip()
            if not handle:
                self._log_msg("[comments] enter your TikTok @handle first.")
                self.comments_var.set(False); return
            if not handle.startswith("@"):
                handle = "@" + handle
            self._save_handle(handle)             # remember it for the dropdown
            from comment_responder import CommentResponder
            from tiktok_comments import TikTokComments
            # The reader can run without the brain (it just READS comments); answers
            # begin as soon as the brain is loaded (responder is created lazily).
            if self.responder is None and self.brain is not None:
                self.responder = CommentResponder(self.brain, get_context=self._live_stream_ctx)
            if self.tiktok is None:
                self.tiktok = TikTokComments(handle, self._on_comment,
                                             on_gift=self._on_gift, on_follow=self._on_follow,
                                             on_like=self._on_like, on_share=self._on_share,
                                             on_viewers=self._on_viewers,
                                             on_join=self._on_join)
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

    @staticmethod
    def _cover_resize_bgr(frame, size, crop_x=0.5, crop_y=0.5):
        """Resize/crop a BGR frame to exactly fill size=(w, h)."""
        out_w, out_h = size
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((out_h, out_w, 3), np.uint8)
        scale = max(out_w / float(w), out_h / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
        crop_x = max(0.0, min(1.0, float(crop_x)))
        crop_y = max(0.0, min(1.0, float(crop_y)))
        x = max(0, min(nw - out_w, int(round((nw - out_w) * crop_x))))
        y = max(0, min(nh - out_h, int(round((nh - out_h) * crop_y))))
        return resized[y:y + out_h, x:x + out_w]

    @staticmethod
    def _contain_resize_bgr(frame, size, scale_pad=0.96):
        """Resize a BGR frame to fit inside size=(w, h), preserving full view."""
        out_w, out_h = size
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((out_h, out_w, 3), np.uint8)
        scale = min(out_w / float(w), out_h / float(h)) * float(scale_pad)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        x = (out_w - nw) // 2
        y = (out_h - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas

    @staticmethod
    def _sharpen_bgr(frame, amount=0.42):
        """Light unsharp mask after resize; keeps UI text crisp without halos."""
        try:
            blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
            sharp = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
            return sharp
        except Exception:
            return frame

    def _scene_image_bgr(self):
        with self._scene_capture_lock:
            image = self._scene_capture_image
            bgr = getattr(self, "_scene_capture_bgr", None)
            source = self._scene_source
            raw_youtube = self._youtube_scene_raw_image
            youtube_scene = getattr(self, "_youtube_scene", None)
        if image is None or source not in ("youtube", "screen"):
            return None, None
        if (source == "youtube"
                and (self._is_nearly_black_bgr(bgr)
                     or (bgr is None and self._is_nearly_black_image(image)))):
            direct_bgr, direct_image = self._youtube_scene_direct_bgr(youtube_scene)
            if direct_bgr is not None:
                return direct_bgr, direct_image
        if bgr is not None:
            return bgr, image
        if (source == "youtube"
                and bool(getattr(self, "low_lag_scene_var", None)
                         and self.low_lag_scene_var.get())
                and raw_youtube is not None):
            image = raw_youtube
        try:
            rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), image
        except Exception:
            return None, None

    @staticmethod
    def _is_nearly_black_bgr(frame, threshold=10):
        try:
            if frame is None or frame.size == 0:
                return False
            return float(np.mean(frame)) < float(threshold)
        except Exception:
            return False

    @staticmethod
    def _is_nearly_black_image(image, threshold=10):
        try:
            if image is None:
                return False
            rgb = np.asarray(image.convert("RGB"))
            return float(np.mean(rgb)) < float(threshold)
        except Exception:
            return False

    @staticmethod
    def _youtube_scene_direct_bgr(youtube_scene):
        try:
            if youtube_scene is None:
                return None, None
            snapshot = getattr(youtube_scene, "frame_snapshot", None)
            if callable(snapshot):
                _serial, frame = snapshot()
            else:
                frame_getter = getattr(youtube_scene, "frame", None)
                frame = frame_getter() if callable(frame_getter) else None
            if frame is None:
                return None, None
            image = Image.fromarray(frame, "RGB")
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            return bgr, image
        except Exception:
            return None, None

    @staticmethod
    def _active_content_crop_bgr(frame, threshold=12):
        """Remove decoder letterbox/pillarbox padding before layout decisions."""
        try:
            if frame is None or frame.size == 0:
                return frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = gray > int(threshold)
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                return frame
            h, w = frame.shape[:2]
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            if (x2 - x1) < w * 0.20 or (y2 - y1) < h * 0.20:
                return frame
            pad_x = max(0, int(round((x2 - x1) * 0.01)))
            pad_y = max(0, int(round((y2 - y1) * 0.01)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            if x1 <= 1 and y1 <= 1 and x2 >= w - 1 and y2 >= h - 1:
                return frame
            return frame[y1:y2, x1:x2]
        except Exception:
            return frame

    def _face_strip_height(self):
        try:
            label = self.face_strip_var.get()
        except Exception:
            label = DEFAULT_FACE_STRIP_LABEL
        height = FACE_STRIP_PRESETS.get(label, FACE_STRIP_PRESETS[DEFAULT_FACE_STRIP_LABEL])
        return max(480, min(TIKTOK_PORTRAIT_H - 720, int(height)))

    def _chart_height(self):
        return TIKTOK_PORTRAIT_H - self._face_strip_height()

    def _avatar_face_slot(self, frame, slot_h):
        """Crop fallback avatar frames around the head so the bottom slot stays face-only."""
        try:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return np.zeros((slot_h, TIKTOK_PORTRAIT_W, 3), np.uint8)
            box = self._detect_avatar_face_box(frame)
            if box is not None:
                x1, y1, x2, y2 = box
                face_w = max(1, x2 - x1)
                face_h = max(1, y2 - y1)
                aspect = TIKTOK_PORTRAIT_W / float(slot_h)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                crop_h = min(float(h), max(face_h * 2.55, face_w / aspect * 2.05))
                crop_w = min(float(w), max(crop_h * aspect, face_w * 2.25))
                crop_h = min(float(h), max(crop_h, crop_w / aspect))
                crop_w = min(float(w), crop_h * aspect)
                sx = int(round(cx - crop_w * 0.50))
                sy = int(round(cy - crop_h * 0.58))
                sx = max(0, min(sx, int(w - crop_w)))
                sy = max(0, min(sy, int(h - crop_h)))
                crop = frame[sy:sy + int(crop_h), sx:sx + int(crop_w)]
                if crop.size:
                    face = self._cover_resize_bgr(crop, (TIKTOK_PORTRAIT_W, slot_h))
                    face = self._sharpen_bgr(face, amount=0.48)
                    return cv2.convertScaleAbs(face, alpha=1.04, beta=3)
            target_aspect = TIKTOK_PORTRAIT_W / float(slot_h)
            crop_w = w
            crop_h = int(round(crop_w / target_aspect))
            if crop_h > h:
                crop_h = h
                crop_w = int(round(crop_h * target_aspect))
            x1 = max(0, min(w - crop_w, (w - crop_w) // 2))
            y1 = max(0, min(h - crop_h, int(h * 0.08)))
            crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            if crop.size == 0:
                crop = frame
            face = self._cover_resize_bgr(crop, (TIKTOK_PORTRAIT_W, slot_h))
            face = self._sharpen_bgr(face, amount=0.44)
            return cv2.convertScaleAbs(face, alpha=1.04, beta=3)
        except Exception:
            return self._cover_resize_bgr(frame, (TIKTOK_PORTRAIT_W, slot_h))

    def _default_avatar_face_frame(self):
        """Static character frame used until the live avatar has produced one."""
        cached = getattr(self, "_character_face_cache", None)
        if cached is not None:
            return cached
        try:
            path = getattr(self, "_char_path", None) or _character_path()
            frame = cv2.imread(path) if path else None
            if frame is not None and frame.size:
                self._character_face_cache = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                return self._character_face_cache
        except Exception:
            pass
        return None

    def _detect_avatar_face_box(self, frame):
        """Return the strongest avatar face box in BGR frame coordinates."""
        try:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return None
            try:
                mp_detector = getattr(self, "_avatar_mp_face_detector", None)
                if mp_detector is None and not getattr(self, "_avatar_mp_tried", False):
                    self._avatar_mp_tried = True
                    import mediapipe as mp
                    mp_detector = mp.solutions.face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.25)
                    self._avatar_mp_face_detector = mp_detector
                if mp_detector is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_detector.process(rgb)
                    if res.detections:
                        r = max((d.location_data.relative_bounding_box
                                 for d in res.detections),
                                key=lambda b: b.width * b.height)
                        x1 = int(max(0, min(w - 1, r.xmin * w)))
                        y1 = int(max(0, min(h - 1, r.ymin * h)))
                        x2 = int(max(x1 + 1, min(w, (r.xmin + r.width) * w)))
                        y2 = int(max(y1 + 1, min(h, (r.ymin + r.height) * h)))
                        return (x1, y1, x2, y2)
            except Exception:
                pass
            detector = getattr(self, "_avatar_face_detector", None)
            if detector is None:
                cascade = os.path.join(
                    cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                detector = cv2.CascadeClassifier(cascade)
                self._avatar_face_detector = detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = min(1.0, 720.0 / max(gray.shape))
            scan = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
            faces = detector.detectMultiScale(
                scan, scaleFactor=1.08, minNeighbors=3,
                minSize=(max(24, scan.shape[1] // 16),
                         max(24, scan.shape[0] // 16)))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
                inv = 1.0 / scale
                x, y, fw, fh = [int(round(v * inv)) for v in (x, y, fw, fh)]
                return (x, y, x + fw, y + fh)
        except Exception:
            pass
        return None

    def _portrait_presenter_region_crop(self, scene_bgr):
        """Force scene sources into chart-on-top, presenter/video-on-bottom."""
        h, w = scene_bgr.shape[:2]
        slot_h = self._face_strip_height()
        if h <= 0 or w <= 0:
            return self._empty_face_slot(slot_h)
        portrait_source = h > w * 1.25
        if portrait_source:
            split_y = int(round(h * (self._chart_height() / TIKTOK_PORTRAIT_H)))
            y1 = max(0, min(h - 1, split_y))
            presenter = scene_bgr[y1:h, 0:w]
        else:
            fx, fy, fw, fh = STREAMER_FACE_CROP
            x1 = max(0, min(w - 1, int(w * fx)))
            y1 = max(0, min(h - 1, int(h * fy)))
            x2 = max(x1 + 1, min(w, int(w * (fx + fw))))
            y2 = max(y1 + 1, min(h, int(h * (fy + fh))))
            presenter = scene_bgr[y1:y2, x1:x2]
        if presenter.size == 0:
            return self._empty_face_slot(slot_h)
        face = self._cover_resize_bgr(
            presenter, (TIKTOK_PORTRAIT_W, slot_h))
        return cv2.convertScaleAbs(face, alpha=1.04, beta=3)

    def _detect_presenter_face_box(self, scene_image):
        """Detect the presenter's face anywhere in the scene."""
        try:
            if self._scene_face_detector is None:
                cascade = os.path.join(
                    cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                self._scene_face_detector = cv2.CascadeClassifier(cascade)
            rgb = np.asarray(scene_image.convert("RGB"))
            h, w = rgb.shape[:2]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            scale = min(1.0, 720.0 / max(gray.shape))
            scan = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
            faces = self._scene_face_detector.detectMultiScale(
                scan, scaleFactor=1.08, minNeighbors=4,
                minSize=(max(24, scan.shape[1] // 24),
                         max(24, scan.shape[0] // 24)))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
                inv = 1.0 / scale
                x, y, fw, fh = [int(round(v * inv)) for v in (x, y, fw, fh)]
                return (x, y, x + fw, y + fh)
        except Exception:
            pass
        return None

    def _detect_presenter_face_box_bgr(self, scene_bgr):
        """Detect a presenter face directly on the active scene crop."""
        try:
            rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            return self._detect_presenter_face_box(image)
        except Exception:
            return None

    def _portrait_chart_crop(self, scene_bgr, face_box):
        h, w = scene_bgr.shape[:2]
        chart_h = self._chart_height()
        portrait_source = h > w * 1.25
        if portrait_source:
            crop_h = int(round(h * (chart_h / TIKTOK_PORTRAIT_H)))
            crop_h = min(h, max(1, crop_h))
            chart = scene_bgr[0:crop_h, 0:w]
        else:
            cx, cy, cw, ch = STREAMER_CHART_CROP
            x = int(w * cx)
            y = int(h * cy)
            crop_w = int(w * cw)
            crop_h = int(h * ch)
            if y + crop_h > h:
                crop_h = h - y
            chart = scene_bgr[y:y + crop_h, x:x + crop_w]
        if chart.size == 0:
            chart = scene_bgr
        if portrait_source:
            chart = self._cover_resize_bgr(
                chart, (TIKTOK_PORTRAIT_W, chart_h))
        else:
            chart = self._cover_resize_bgr(
                chart, (TIKTOK_PORTRAIT_W, chart_h), crop_x=0.68)
        return cv2.convertScaleAbs(chart, alpha=1.08, beta=4)

    def _portrait_face_crop(self, scene_bgr, scene_image, face_box):
        h, w = scene_bgr.shape[:2]
        slot_h = self._face_strip_height()
        portrait_source = h > w * 1.25
        if portrait_source and face_box is None:
            return self._portrait_presenter_region_crop(scene_bgr)
        if face_box is None:
            return self._portrait_presenter_region_crop(scene_bgr)
        else:
            bx1, by1, bx2, by2 = [int(v) for v in face_box]
            face_w = max(1, bx2 - bx1)
            face_h = max(1, by2 - by1)
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            target_aspect = TIKTOK_PORTRAIT_W / float(slot_h)
            if portrait_source:
                crop_w = max(w * 0.72, face_w * 3.20)
                crop_h = max(crop_w / target_aspect, face_h * 2.70)
                crop_w = min(crop_w, w)
                crop_h = min(crop_h, h * 0.42)
            else:
                crop_w = max(w * 0.30, face_w * 2.80)
                crop_h = max(crop_w / target_aspect, face_h * 2.70)
                crop_w = min(crop_w, w * 0.46)
                crop_h = min(crop_h, h * 0.52)
            x1 = int(cx - crop_w * 0.50)
            y1 = int(cy - crop_h * (0.58 if portrait_source else 0.58))
            x1 = max(0, min(x1, int(w - crop_w)))
            y1 = max(0, min(y1, int(h - crop_h)))
            x2 = min(w, int(x1 + crop_w))
            y2 = min(h, int(y1 + crop_h))
        face = scene_bgr[y1:y2, x1:x2]
        if face.size == 0:
            face = scene_bgr
        face = self._cover_resize_bgr(
            face, (TIKTOK_PORTRAIT_W, slot_h))
        face = self._sharpen_bgr(face, amount=0.48)
        return cv2.convertScaleAbs(face, alpha=1.04, beta=3)

    def _portrait_source_reference_layout(self, scene_bgr):
        """Reframe portrait YouTube videos into chart top + full presenter bottom."""
        h, w = scene_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None
        chart_h = int(round(TIKTOK_PORTRAIT_H * 0.58))
        face_h = TIKTOK_PORTRAIT_H - chart_h

        chart_y2 = max(1, min(h, int(round(h * 0.64))))
        chart = scene_bgr[0:chart_y2, 0:w]

        face_x1 = max(0, min(w - 1, int(round(w * 0.47))))
        face_y1 = max(0, min(h - 1, int(round(h * 0.64))))
        face = scene_bgr[face_y1:h, face_x1:w]
        if face.size == 0:
            face = scene_bgr[chart_y2:h, 0:w]
        if chart.size == 0:
            chart = scene_bgr

        top = self._cover_resize_bgr(chart, (TIKTOK_PORTRAIT_W, chart_h))
        bottom = self._cover_resize_bgr(face, (TIKTOK_PORTRAIT_W, face_h))
        top = cv2.convertScaleAbs(top, alpha=1.06, beta=3)
        bottom = self._sharpen_bgr(bottom, amount=0.30)
        bottom = cv2.convertScaleAbs(bottom, alpha=1.04, beta=3)

        canvas = np.zeros(
            (TIKTOK_PORTRAIT_H, TIKTOK_PORTRAIT_W, 3), np.uint8)
        canvas[0:chart_h, 0:TIKTOK_PORTRAIT_W] = top
        canvas[chart_h:TIKTOK_PORTRAIT_H, 0:TIKTOK_PORTRAIT_W] = bottom
        cv2.line(canvas, (0, chart_h), (TIKTOK_PORTRAIT_W, chart_h),
                 (18, 19, 22), 3, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _empty_face_slot(height=None):
        height = int(height or TIKTOK_FACE_H)
        return np.full((height, TIKTOK_PORTRAIT_W, 3), 8, np.uint8)

    def _scene_portrait_frame(self, avatar_fallback=None):
        """TikTok Live portrait scene: chart/source focus above, detected face below."""
        chart_h = self._chart_height()
        face_h = self._face_strip_height()
        if avatar_fallback is None:
            avatar_fallback = self._default_avatar_face_frame()
        scene_bgr, scene_image = self._scene_image_bgr()
        if scene_bgr is None or scene_image is None:
            if avatar_fallback is None:
                return None
            top = self._cover_resize_bgr(
                avatar_fallback, (TIKTOK_PORTRAIT_W, chart_h))
            bottom = self._avatar_face_slot(avatar_fallback, face_h)
            return self._apply_scene_text_overlay(np.vstack((top, bottom)))
        if bool(getattr(self, "low_lag_scene_var", None)
                and self.low_lag_scene_var.get()):
            scene_bgr = self._active_content_crop_bgr(scene_bgr)
        portrait_source = scene_bgr.shape[0] > scene_bgr.shape[1] * 1.25
        if portrait_source:
            return self._apply_scene_text_overlay(
                self._portrait_source_reference_layout(scene_bgr))
        low_lag = bool(getattr(self, "low_lag_scene_var", None)
                       and self.low_lag_scene_var.get())
        if portrait_source or low_lag:
            face_box = None
        else:
            try:
                face_box = self._detect_presenter_face_box(scene_image)
            except Exception:
                face_box = None
        chart = self._portrait_chart_crop(scene_bgr, face_box)
        if face_box is not None:
            face = self._portrait_face_crop(scene_bgr, scene_image, face_box)
        elif portrait_source or low_lag:
            face = self._portrait_presenter_region_crop(scene_bgr)
        elif avatar_fallback is not None:
            face = self._avatar_face_slot(avatar_fallback, face_h)
        else:
            face = self._portrait_presenter_region_crop(scene_bgr)
        canvas = np.zeros(
            (TIKTOK_PORTRAIT_H, TIKTOK_PORTRAIT_W, 3), np.uint8)
        canvas[0:chart_h, 0:TIKTOK_PORTRAIT_W] = chart
        canvas[chart_h:TIKTOK_PORTRAIT_H, 0:TIKTOK_PORTRAIT_W] = face
        cv2.line(canvas, (0, chart_h), (TIKTOK_PORTRAIT_W, chart_h),
                 (20, 22, 26), 4, cv2.LINE_AA)
        return self._apply_scene_text_overlay(canvas)

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
        # In Live-Mic mode the AI voice is already muted; the button mutes the
        # voice-changer MONITOR instead (the avatar mouth keeps tracking your mic).
        if self.live_mic is not None:
            self.live_mic.set_muted(not self.live_mic.muted)
            muted = self.live_mic.muted
        elif self.tts is not None:
            self.tts.set_muted(not self.tts.muted)
            muted = self.tts.muted
        else:
            return
        self.mute_btn.configure(text="UNMUTE" if muted else "MUTE",
                                bg=RED if muted else BG2,
                                fg="#ffffff" if muted else FG)
        self._log_msg("[studio] muted" if muted else "[studio] unmuted")

    def _on_voice(self, *args):
        if self.tts is not None:
            self.tts.set_voice(self.voice_var.get())

    # ----- LIVE MIC (voice changer) -----------------------------------------
    def _vc_key(self):
        """Selected voice-changer converter key (rvc / dsp / passthrough)."""
        return dict(getattr(self, "VC_MODES", [])).get(self.vcmode_var.get(), "rvc")

    def _mic_device_index(self):
        """Parse 'N: name' from the mic dropdown -> device index, or None (default)."""
        s = self.micdev_var.get()
        if ":" in s:
            try:
                return int(s.split(":", 1)[0])
            except Exception:
                return None
        return None

    def _on_live_mic(self, *args):
        if bool(self.livemic_var.get()):
            self._start_live_mic()
        else:
            self._stop_live_mic()

    def _start_live_mic(self):
        """Build + start the LiveVoiceEngine on the SAME mouth engine the TTS feeds,
        and mute the AI host so only one source drives the mouth at a time."""
        if not self.running or self.engines is None:
            self._log_msg("[studio] press START first, then enable Live Mic.")
            self.livemic_var.set(False)
            return
        if self.live_mic is not None:
            return
        try:
            from voice_changer_engine import LiveVoiceEngine, make_converter
            mt = self.engines.get("mt")
            eng = LiveVoiceEngine(mt, converter=make_converter(self._vc_key()),
                                  in_device=self._mic_device_index())
            try:
                eng.set_gain(self.micgain_var.get())
            except Exception:
                pass
            ok, msg = eng.startup_check()
            if not ok:
                self._log_msg(f"[studio] live mic unavailable: {msg}")
                self.livemic_var.set(False)
                return
            if self.tts is not None:
                try:
                    self.tts.clear_pending()
                except Exception:
                    pass
                self.tts.set_muted(True)        # avatar speaks from YOUR mic now
            eng.set_muted(self._mic_monitor_muted)
            eng.start()
            self.live_mic = eng
            self._sync_audio_mute_buttons()
            self._log_msg(f"[studio] LIVE MIC on — {msg}. AI host muted; talk into the mic.")
        except Exception as exc:
            self._log_msg(f"[studio] live mic failed: {exc}")
            self.live_mic = None
            self.livemic_var.set(False)

    def _stop_live_mic(self):
        eng = self.live_mic
        self.live_mic = None
        if eng is not None:
            try:
                eng.shutdown()
            except Exception:
                pass
            if self.tts is not None:
                self.tts.set_muted(False)       # restore the AI voice
            self._log_msg("[studio] live mic off — AI voice restored.")
        self._sync_audio_mute_buttons()

    def _on_vcmode(self, *args):
        """Hot-swap the voice changer (RVC / DSP / passthrough) while live."""
        if self.live_mic is None:
            return
        try:
            from voice_changer_engine import make_converter
            self.live_mic.set_converter(make_converter(self._vc_key()))
            self._log_msg(f"[studio] voice changer -> {self.vcmode_var.get()}")
        except Exception as exc:
            self._log_msg(f"[studio] voice changer switch failed: {exc}")

    def _on_micdev(self, *args):
        """Restart the live mic on the newly selected input device."""
        if self.live_mic is None:
            return
        self._stop_live_mic()
        self._start_live_mic()

    def _on_micgain(self, *args):
        """Live mic-boost slider — apply immediately while talking."""
        if self.live_mic is not None:
            try:
                self.live_mic.set_gain(self.micgain_var.get())
            except Exception:
                pass

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

    def _on_background(self, *args):
        """Apply a background preset immediately."""
        name = self.background_var.get()
        enabled = name != "No Background"
        self.background_on_var.set(enabled)
        if enabled:
            self._last_background = name
        try:
            import enhance_engine as ee
            ee.set_background_preset(name)
        except Exception as exc:
            self._log_msg(f"[studio] background switch failed: {exc}")
            return
        self._log_msg("[studio] background -> " + name)

    def _on_background_toggle(self, *args):
        """Explicitly disable replacement or restore the last selected preset."""
        enabled = bool(self.background_on_var.get())
        if enabled:
            name = getattr(self, "_last_background",
                           "Wall Street LED / Midnight Blue")
            if name == "No Background":
                name = "Wall Street LED / Midnight Blue"
        else:
            current = self.background_var.get()
            if current != "No Background":
                self._last_background = current
            name = "No Background"
        if self.background_var.get() != name:
            self.background_var.set(name)
        else:
            self._on_background()

    def _on_face_strip_length(self, *args):
        try:
            label = self.face_strip_var.get()
            height = self._face_strip_height()
            self._log_msg(
                f"[scene] bottom face length: {label} ({height}px, face-only crop).")
        except Exception:
            pass

    def _toggle_music(self):
        """Top mute button: flip the background music on/off."""
        if getattr(self, "music_var", None) is None:
            return
        self.music_var.set(not self.music_var.get())
        self._on_music()
        self._sync_audio_mute_buttons()

    def _toggle_speech(self):
        """Top mute button: silence the bot's VOICE (lips keep moving)."""
        if self.tts is None:
            self._log_msg("[studio] press START first.")
            return
        self.tts.set_muted(not self.tts.muted)
        muted = self.tts.muted
        self._log_msg("[studio] bot speech " + ("MUTED" if muted else "on"))
        self._sync_audio_mute_buttons()

    def _toggle_youtube_mute(self):
        """Mute only the original/altered YouTube voice source."""
        self._youtube_muted = not self._youtube_muted
        if self._youtube_audio is not None:
            self._youtube_audio.set_muted(self._youtube_muted)
        self._log_msg("[studio] YouTube voice "
                      + ("MUTED" if self._youtube_muted else "on"))
        self._sync_audio_mute_buttons()

    def _toggle_mic_monitor_mute(self):
        """Mute only the local Live Mic monitor output."""
        self._mic_monitor_muted = not self._mic_monitor_muted
        if self.live_mic is not None:
            self.live_mic.set_muted(self._mic_monitor_muted)
        self._log_msg("[studio] Live Mic monitor "
                      + ("MUTED" if self._mic_monitor_muted else "on"))
        self._sync_audio_mute_buttons()

    def _toggle_youtube_smooth(self):
        """Blend real YouTube voice starts/stops so source changes are less obvious."""
        if getattr(self, "youtube_smooth_var", None) is None:
            return
        enabled = not bool(self.youtube_smooth_var.get())
        self.youtube_smooth_var.set(enabled)
        if self._youtube_audio is not None:
            self._youtube_audio.smooth_transition = enabled
        self._sync_youtube_smooth_button()
        self._log_msg("[studio] YouTube voice smoothing "
                      + ("ON" if enabled else "OFF"))

    def _sync_youtube_smooth_button(self):
        btn = getattr(self, "youtube_smooth_btn", None)
        if btn is None or getattr(self, "youtube_smooth_var", None) is None:
            return
        enabled = bool(self.youtube_smooth_var.get())
        color = MINT if enabled else RED
        bg = self._mix(SURFACE2, color, 0.22 if enabled else 0.10)
        try:
            btn.configure(
                text="SMOOTH VOICE" if enabled else "VOICE HARD CUT",
                bg=bg,
                fg=color,
                activebackground=self._mix(SURFACE2, color, 0.32),
                activeforeground=color,
                highlightbackground=self._mix(color, BG, 0.45),
            )
        except Exception:
            pass

    def _sync_audio_mute_buttons(self):
        """Label every audible source and show its independent mute state."""
        states = (
            (getattr(self, "speech_btn", None), "AI",
             bool(self.tts is not None and self.tts.muted)),
            (getattr(self, "youtube_mute_btn", None), "YOUTUBE",
             bool(self._youtube_muted)),
            (getattr(self, "music_btn", None), "MUSIC",
             not bool(getattr(self, "music_var", None) and self.music_var.get())),
            (getattr(self, "mic_mute_btn", None), "MIC",
             bool(self._mic_monitor_muted)),
        )
        for btn, label, muted in states:
            if btn is None:
                continue
            try:
                btn.configure(
                    text=f"{label}  {'OFF' if muted else 'ON'}",
                    fg=RED if muted else MINT,
                    activeforeground=RED if muted else MINT,
                    activebackground=self._mix(
                        SURFACE2, RED if muted else MINT, 0.18),
                )
            except Exception:
                pass

    def _audio_source_control(self, parent, key, command, column):
        """Create one mute button with a compact live output meter below it."""
        wrap = tk.Frame(
            parent, bg=SURFACE2, highlightthickness=1,
            highlightbackground=BORDER)
        wrap.grid(row=0, column=column, sticky="nsew", padx=2)
        btn = tk.Button(
            wrap, command=command, bg=SURFACE2, fg=MINT,
            relief="flat", bd=0, font=("Segoe UI", 8, "bold"),
            cursor="hand2")
        btn.pack(fill="x", ipady=1)
        meter = tk.Canvas(
            wrap, height=7, bg=SURFACE2, highlightthickness=0, bd=0)
        meter.pack(fill="x", padx=5, pady=(1, 4))
        bars = []
        for i in range(10):
            bars.append(meter.create_rectangle(
                2 + i * 7, 2, 6 + i * 7, 5, fill=BORDER, outline=""))
        self._audio_meters[key] = (wrap, meter, bars)
        self._audio_meter_levels[key] = 0.0
        return btn

    @staticmethod
    def _meter_value(rms):
        """Map normal audio RMS values to a readable 0..1 meter."""
        if rms <= 0.001:
            return 0.0
        return max(0.0, min(1.0, (rms - 0.001) / 0.24))

    def _update_audio_meters(self):
        """Refresh meters from the actual output level of every audio source."""
        tts = self.tts
        youtube = self._youtube_audio
        music = self.music
        mic = self.live_mic
        raw = {
            "ai": (
                getattr(tts, "audio_level", 0.0)
                if tts is not None and getattr(tts, "speaking", False)
                and not getattr(tts, "muted", False) else 0.0),
            "youtube": (
                getattr(youtube, "audio_level", 0.0)
                if youtube is not None and getattr(youtube, "speaking", False)
                and not self._youtube_muted else 0.0),
            "music": (
                getattr(music, "audio_level", 0.0)
                if music is not None and getattr(music, "active", False)
                and getattr(self, "music_var", None)
                and self.music_var.get() else 0.0),
            "mic": (
                getattr(mic, "last_rms", 0.0)
                if mic is not None and getattr(mic, "speaking", False)
                and not self._mic_monitor_muted else 0.0),
        }
        for key, rms in raw.items():
            target = self._meter_value(float(rms or 0.0))
            previous = self._audio_meter_levels.get(key, 0.0)
            level = target if target > previous else previous * 0.72
            self._audio_meter_levels[key] = level
            meter_info = self._audio_meters.get(key)
            if meter_info is None:
                continue
            wrap, canvas, bars = meter_info
            width = max(54, canvas.winfo_width())
            gap = 2
            bar_w = max(2, (width - 4 - gap * (len(bars) - 1)) / len(bars))
            lit = int(round(level * len(bars)))
            active = lit > 0
            wrap.configure(
                highlightbackground=(
                    self._mix(MINT, BORDER, 0.45) if active else BORDER))
            for i, item in enumerate(bars):
                x1 = 2 + i * (bar_w + gap)
                color = (MINT if i < 7 else AMBER if i < 9 else RED)
                canvas.coords(item, x1, 2, x1 + bar_w, 5)
                canvas.itemconfigure(
                    item, fill=color if i < lit else BORDER)

    def _toggle_ai_mouth(self):
        """Switch AI voice lip-sync on/off without stopping speech."""
        enabled = not bool(self.ai_mouth_var.get())
        self.ai_mouth_var.set(enabled)
        self.liplock_var.set(enabled)
        self._sync_mouth_btn()
        if self.engines:
            try:
                self.engines["lp"].set_lip_lock(enabled)
            except Exception:
                pass
        self._log_msg("[studio] mouth mode: "
                      + ("AI VOICE lip-sync" if enabled else "AVATAR native mouth"))

    def _sync_mouth_btn(self):
        """Reflect the active mouth source in the top-bar button."""
        if getattr(self, "mouth_btn", None) is None:
            return
        enabled = bool(self.ai_mouth_var.get())
        color = MINT if enabled else AMBER
        self.mouth_btn.configure(
            text="AI MOUTH" if enabled else "AVATAR MOUTH",
            fg=color,
            highlightbackground=self._mix(color, BG, 0.5))

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
        self._sync_audio_mute_buttons()

    def _on_liplock(self):
        enabled = bool(self.liplock_var.get())
        self.ai_mouth_var.set(enabled)
        self._sync_mouth_btn()
        if self.engines:
            try:
                self.engines["lp"].set_lip_lock(enabled)
                self._log_msg("[studio] lips: "
                              + ("AI VOICE only (native mouth suppressed)"
                                 if enabled else "AVATAR native mouth"))
            except Exception:
                pass

    def _on_swap(self):
        # FACE-SWAP shows YOUR real head/mouth — mutually exclusive with bot-only
        # lips. Turning it on disables the lip-lock.
        if self.swap_var.get() and not self.ai_mouth_var.get():
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
                self.engines["lp"].min_good_face = max(0.03, self.minface_var.get() / 100.0)
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

    def _on_skintone(self):
        """Live-set the fairer-skin tone strength on the face-swap engine (0..1)."""
        try:
            if getattr(self, "swap_engine", None) is not None:
                self.swap_engine.skin_lighten = self.skintone_var.get() / 100.0
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
        self._scene_capture_stop.set()
        if self._scene_preview_job is not None:
            try:
                self.root.after_cancel(self._scene_preview_job)
            except Exception:
                pass
            self._scene_preview_job = None
        if self._youtube_scene is not None:
            try:
                self._youtube_scene.stop()
            except Exception:
                pass
        self._live_response_event.set()
        if self.brain_pool is not None:
            try:
                self.brain_pool.stop()
            except Exception:
                pass
        if self.tiktok is not None:
            try:
                self.tiktok.stop()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._release_camera()
        for fn in (lambda: self.obs_cam.close() if self.obs_cam else None,
                   lambda: self._tv_proc.terminate() if self._tv_proc else None,
                   lambda: self.market_gold.stop() if self.market_gold else None,
                   lambda: self.market_btc.stop() if self.market_btc else None,
                   lambda: self._youtube_audio.stop() if self._youtube_audio else None,
                   lambda: self.live_mic.shutdown() if self.live_mic else None,
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
        mutex_name = os.environ.get(
            "AVATAR_INSTANCE_MUTEX", "Global\\AvatarStudioSingleInstance"
        )
        h = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
        if not h or ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        _SINGLE_INSTANCE_HANDLE = h        # keep the handle alive
        return True
    except Exception:
        return True                         # non-Windows / failure: don't block


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--tradingview-pilot":
        from tradingview_pilot import main as tradingview_main
        return tradingview_main([sys.argv[0], *sys.argv[2:]])
    if not _acquire_single_instance():
        print("[studio] Avatar Studio is ALREADY RUNNING — only one instance is "
              "allowed (so only one bot speaks). Exiting this one.")
        return 1
    _configure_windows_app_identity()
    root = tk.Tk()
    AvatarStudio(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
