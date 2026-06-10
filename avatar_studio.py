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

# Quality presets -> (lp_interval, enhance level, body motion). Smooth maximises
# fps (LP amortised + light enhance + no body warp); Sharp maximises detail.
QUALITY_PRESETS = {
    "Smooth (max fps)":      dict(lp=3, enhance="light", body=False),
    "Balanced":              dict(lp=2, enhance="light", body=True),
    "Sharp (max detail)":    dict(lp=1, enhance="full",  body=True),
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

# Dark palette
BG = "#15171c"; BG2 = "#1e2128"; FG = "#e6e6e6"
ACCENT = "#00d7ff"; GREEN = "#22cc55"; RED = "#dd3344"; ENTRY_BG = "#0f1115"


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

        root.title("AVATAR STUDIO — live test")
        root.configure(bg=BG)
        root.geometry("980x640")
        root.minsize(900, 600)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()                     # start the UI refresh loop

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # left: live preview
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self.preview = tk.Label(left, bg="#000000", width=PREVIEW_SIZE,
                                height=PREVIEW_SIZE)
        self.preview.pack(fill="both", expand=True)
        self._show_placeholder()

        # right: controls
        right = tk.Frame(self.root, bg=BG, width=360)
        right.pack(side="right", fill="y", padx=(0, 12), pady=12)
        right.pack_propagate(False)

        tk.Label(right, text="AVATAR STUDIO", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w")

        # status row
        srow = tk.Frame(right, bg=BG); srow.pack(fill="x", pady=(8, 4))
        self.status_canvas = tk.Canvas(srow, width=16, height=16, bg=BG,
                                       highlightthickness=0)
        self.status_canvas.pack(side="left")
        self.status_dot = self.status_canvas.create_oval(2, 2, 14, 14,
                                                         fill=RED, outline="")
        self.status_lbl = tk.Label(srow, text="stopped", bg=BG, fg=FG,
                                   font=("Segoe UI", 10))
        self.status_lbl.pack(side="left", padx=6)
        self.fps_lbl = tk.Label(srow, text="", bg=BG, fg="#8a93a3",
                                font=("Consolas", 9))
        self.fps_lbl.pack(side="right")

        # start/stop
        self.start_btn = tk.Button(right, text="START", command=self.start,
                                   bg=GREEN, fg="#04240f", font=("Segoe UI", 12, "bold"),
                                   relief="flat", cursor="hand2", activebackground="#33dd6a")
        self.start_btn.pack(fill="x", ipady=6, pady=(8, 4))
        self.stop_btn = tk.Button(right, text="STOP", command=self.stop,
                                  bg=BG2, fg=FG, font=("Segoe UI", 12, "bold"),
                                  relief="flat", cursor="hand2", state="disabled")
        self.stop_btn.pack(fill="x", ipady=6, pady=(0, 4))
        self.char_btn = tk.Button(
            right, text="LOAD CHARACTER…  (any face image)",
            command=self._load_character, bg="#16263a", fg="#7fc8ff",
            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
            activebackground="#1d3550")
        self.char_btn.pack(fill="x", ipady=4, pady=(0, 4))
        self.recenter_btn = tk.Button(
            right, text="RECENTER POSE  (face camera upright)",
            command=self.recenter, bg="#3a2f12", fg="#ffcf66",
            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
            state="disabled", activebackground="#4a3c18")
        self.recenter_btn.pack(fill="x", ipady=4, pady=(0, 8))

        # head-update interval
        irow = tk.Frame(right, bg=BG); irow.pack(fill="x", pady=2)
        tk.Label(irow, text="Head update (LP every N frames)", bg=BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side="left")
        self.interval_var = tk.IntVar(value=2)
        ttk.Spinbox(irow, from_=1, to=4, width=4, textvariable=self.interval_var,
                    command=self._on_interval).pack(side="right")

        # quality preset (sets fps/detail tradeoff: LP interval + enhance + body)
        qrow = tk.Frame(right, bg=BG); qrow.pack(fill="x", pady=2)
        tk.Label(qrow, text="Quality", bg=BG, fg=FG, font=("Segoe UI", 8)).pack(side="left")
        self.quality_var = tk.StringVar(value="Balanced")
        ttk.Combobox(qrow, textvariable=self.quality_var, values=QUALITY_LABELS,
                     state="readonly", width=18).pack(side="right")
        self.quality_var.trace_add("write", self._on_quality)

        # stabilization (smooths head pose/expression — reduces melt/jitter)
        srow2 = tk.Frame(right, bg=BG); srow2.pack(fill="x", pady=2)
        tk.Label(srow2, text="Stabilization", bg=BG, fg=FG,
                 font=("Segoe UI", 8)).pack(side="left")
        self.stab_var = tk.IntVar(value=40)
        ttk.Scale(srow2, from_=0, to=100, variable=self.stab_var, length=140,
                  command=lambda e: self._on_stab()).pack(side="right")

        # gaze lock (keep eyes toward camera even when you glance away)
        grow = tk.Frame(right, bg=BG); grow.pack(fill="x", pady=2)
        self.gaze_var = tk.BooleanVar(value=True)
        tk.Checkbutton(grow, text="Lock gaze", variable=self.gaze_var, bg=BG, fg=FG,
                       selectcolor=BG2, activebackground=BG, activeforeground=FG,
                       command=self._on_gaze, font=("Segoe UI", 8)).pack(side="left")
        self.gaze_var2 = tk.IntVar(value=80)
        ttk.Scale(grow, from_=0, to=100, variable=self.gaze_var2, length=120,
                  command=lambda e: self._on_gaze()).pack(side="right")

        # live per-stage timing readout
        self.diag_lbl = tk.Label(right, text="", bg=BG, fg="#7fa6c0",
                                 font=("Consolas", 8))
        self.diag_lbl.pack(anchor="w", pady=(2, 0))

        # min face size gate (below this -> too far -> hold/charts, never a bad face)
        mrow = tk.Frame(right, bg=BG); mrow.pack(fill="x", pady=2)
        tk.Label(mrow, text="Min face size % (smaller = allow farther)", bg=BG,
                 fg=FG, font=("Segoe UI", 8)).pack(side="left")
        self.minface_var = tk.IntVar(value=9)
        ttk.Spinbox(mrow, from_=6, to=40, increment=2, width=4,
                    textvariable=self.minface_var,
                    command=self._on_minface).pack(side="right")

        # max head turn before it caps (smaller = cleaner, less range) — LIVE
        trow = tk.Frame(right, bg=BG); trow.pack(fill="x", pady=2)
        tk.Label(trow, text="Max turn deg (smaller = cleaner)", bg=BG,
                 fg=FG, font=("Segoe UI", 8)).pack(side="left")
        self.turncap_var = tk.IntVar(value=22)
        ttk.Spinbox(trow, from_=10, to=40, increment=2, width=4,
                    textvariable=self.turncap_var,
                    command=self._on_turncap).pack(side="right")

        # OBS toggle
        self.obs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Also send to OBS virtual camera",
                       variable=self.obs_var, bg=BG, fg=FG, selectcolor=BG2,
                       activebackground=BG, activeforeground=FG,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

        # face-loss -> trading chart scene toggle
        self.chart_var = tk.BooleanVar(value=True)
        tk.Checkbutton(right, text="Show live charts when face is lost",
                       variable=self.chart_var, bg=BG, fg=FG, selectcolor=BG2,
                       activebackground=BG, activeforeground=FG,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 0))

        # webcam-driven upper-body motion (shoulders sway/lean with you)
        self.body_var = tk.BooleanVar(value=True)
        tk.Checkbutton(right, text="Live body motion (torso follows you)",
                       variable=self.body_var, bg=BG, fg=FG, selectcolor=BG2,
                       activebackground=BG, activeforeground=FG,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 0))

        # extended turning via multi-view references (OFF = bulletproof default)
        self.multiref_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Extended turning (multi-view)  [experimental]",
                       variable=self.multiref_var, bg=BG, fg=FG, selectcolor=BG2,
                       activebackground=BG, activeforeground=FG,
                       command=self._on_multiref,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 8))

        # voice mode (which TTS backend) — switch live
        vmrow = tk.Frame(right, bg=BG); vmrow.pack(fill="x", pady=(6, 2))
        tk.Label(vmrow, text="Voice mode", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).pack(side="left")
        self.voicemode_var = tk.StringVar(value=VOICE_MODE_LABELS[0])
        ttk.Combobox(vmrow, textvariable=self.voicemode_var,
                     values=VOICE_MODE_LABELS, state="readonly", width=24
                     ).pack(side="left", padx=6)
        self.voicemode_var.trace_add("write", self._on_voice_mode)

        # emotion-tag quick inserts (Maya1 performs these; other voices ignore)
        erow = tk.Frame(right, bg=BG); erow.pack(fill="x", pady=(0, 2))
        tk.Label(erow, text="Insert:", bg=BG, fg="#8a93a3",
                 font=("Segoe UI", 8)).pack(side="left")
        for tag in ("<laugh>", "<sigh>", "<chuckle>", "<gasp>", "<whisper>"):
            tk.Button(erow, text=tag, bg=BG2, fg=FG, relief="flat",
                      font=("Consolas", 8), cursor="hand2", padx=3,
                      command=lambda t=tag: self._insert_tag(t)).pack(side="left", padx=1)

        # speak entry
        tk.Label(right, text="Make the avatar speak:", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.entry = tk.Text(right, height=3, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                             font=("Segoe UI", 11), relief="flat", wrap="word",
                             padx=6, pady=4)
        self.entry.pack(fill="x", pady=(2, 4))
        self.entry.bind("<Return>", self._on_enter)

        brow = tk.Frame(right, bg=BG); brow.pack(fill="x")
        self.speak_btn = tk.Button(brow, text="SPEAK", command=self.speak,
                                   bg=ACCENT, fg="#04222a", font=("Segoe UI", 11, "bold"),
                                   relief="flat", cursor="hand2", state="disabled")
        self.speak_btn.pack(side="left", fill="x", expand=True, ipady=3)
        self.mute_btn = tk.Button(brow, text="MUTE", command=self.toggle_mute,
                                  bg=BG2, fg=FG, font=("Segoe UI", 11, "bold"),
                                  relief="flat", width=8, cursor="hand2", state="disabled")
        self.mute_btn.pack(side="left", padx=(6, 0), ipady=3)

        # quick phrases
        qp = tk.Frame(right, bg=BG); qp.pack(fill="x", pady=(8, 4))
        for i, t in enumerate(QUICK_PHRASES):
            tk.Button(qp, text=t[:30] + ("..." if len(t) > 30 else ""),
                      bg=BG2, fg=FG, relief="flat", font=("Segoe UI", 8),
                      anchor="w", cursor="hand2",
                      command=lambda x=t: self._speak_text(x)).pack(fill="x", pady=1)

        # voice
        vrow = tk.Frame(right, bg=BG); vrow.pack(fill="x", pady=(6, 4))
        tk.Label(vrow, text="Voice", bg=BG, fg=FG, font=("Segoe UI", 9)).pack(side="left")
        self.voice_var = tk.StringVar(value=MALE_VOICES[0])
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        ttk.Combobox(vrow, textvariable=self.voice_var, values=MALE_VOICES,
                     state="readonly", width=26).pack(side="left", padx=6)
        self.voice_var.trace_add("write", self._on_voice)

        # log
        tk.Label(right, text="Log", bg=BG, fg=FG, font=("Segoe UI", 9)).pack(anchor="w",
                                                                             pady=(6, 0))
        self.log = tk.Text(right, height=7, bg=ENTRY_BG, fg="#9aa3b2", relief="flat",
                           font=("Consolas", 8), wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(2, 0))

    # -------------------------------------------------------------------------
    # PREVIEW / UI REFRESH (Tk main thread only)
    # -------------------------------------------------------------------------
    def _show_placeholder(self):
        img = np.full((PREVIEW_SIZE, PREVIEW_SIZE, 3), 22, np.uint8)
        cv2.putText(img, "press START", (150, 256), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (90, 90, 90), 2, cv2.LINE_AA)
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
            self.engines = {"lp": lp, "mt": mt, "comp": comp, "enh": enhance_engine,
                            "chart": TradingView("XAUUSD"), "body": BodyMotionEngine()}
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
        t_read = t_lp = t_body = t_enh = 0.0
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
                rd, lpm, bd, en = (x / 15 * 1000 for x in (t_read, t_lp, t_body, t_enh))
                self._diag = (f"{self._fps:.1f}fps | read {rd:.0f} | LP {lpm:.0f} | "
                              f"body {bd:.0f} | enh {en:.0f} ms")
                print("[DIAG] " + self._diag)
                t_read = t_lp = t_body = t_enh = 0.0

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
        """Live-set the head-turn caps (module globals read per frame). Pitch and
        roll scale with yaw. No restart needed."""
        try:
            import liveportrait_engine as lpe
            v = float(self.turncap_var.get())
            lpe.YAW_CAP = v
            lpe.PITCH_CAP = max(8.0, v * 0.72)
            lpe.ROLL_CAP = max(10.0, v * 0.9)
            self._log_msg(f"[studio] max turn -> {v:.0f}deg (cleaner if smaller)")
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
