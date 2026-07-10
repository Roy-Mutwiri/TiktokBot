# =============================================================================
# realtime_avatar.py  —  MAIN (run this to go live)
# -----------------------------------------------------------------------------
# Real-time photoreal AI avatar for TikTok Live. YOUR webcam drives the head
# motion, expression, blinks and leans of an AI character; you type text, an AI
# voice speaks it, and the mouth syncs to THAT voice (not your real mouth).
#
# Per-frame pipeline (this exact order):
#   1. capture webcam frame
#   2. LivePortrait : webcam -> animated AI face (full head motion + expression)
#   3. if speaking  : MuseTalk syncs ONLY the mouth bbox to the current TTS audio
#   4. compositor   : feathered, colour-matched blend of the synced mouth back on
#   5. enhance      : studio bg, ticker, LIVE badge, speaking indicator, grade
#   6. pyvirtualcam -> OBS Video Capture Device -> TikTok Live
#
# Text + control come from control_gui.py over a local socket (127.0.0.1:9998).
#
#   python realtime_avatar.py
# =============================================================================

import os
import sys
import time
import socket
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = os.path.join(PROJECT_DIR, "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CHARACTER_CANDIDATES = [
    os.path.join(PROJECT_DIR, "character_views", "character.jpg"),  # extracted from video
    os.path.join(PROJECT_DIR, "character.jpg"),
    os.path.join(PROJECT_DIR, "ai-face", "character.jpg"),
]
FRAME_SIZE = 512
FPS = 25
TARGET_FRAME_TIME = 1.0 / FPS
WEBCAM_INDEX = int(os.environ.get("AVATAR_WEBCAM", "0"))
WEBCAM_W, WEBCAM_H = 640, 480

# LivePortrait (~80 ms/frame) is the cost center. Run it every Nth webcam frame
# and reuse its output between, while the mouth still syncs EVERY frame (lip-sync
# stays full-rate; only head pose updates at FPS/N). Measured on the 5060 Ti:
#   1 = ~9-10 fps (smoothest head)   2 = ~16-18 fps (default)   3 = ~21 fps (choppier)
# Override with the AVATAR_LP_INTERVAL env var.
LP_INTERVAL = max(1, int(os.environ.get("AVATAR_LP_INTERVAL", "2")))

# Face-loss -> trading-chart fallback scene. When the webcam can't see the face
# (operator looks away/down) for NO_FACE_SECONDS, the output crossfades to a
# live-moving trading chart, then back when the face returns.
SHOW_CHART_ON_FACE_LOSS = os.environ.get("AVATAR_CHART_ON_LOSS", "1") == "1"
NO_FACE_SECONDS = 3.0
CHART_FADE_STEP = 0.12

# LIVE MIC mode: instead of (or alongside) the AI voice, YOUR mic drives the mouth
# through a voice changer (engines/voice_changer_engine.py). The render loop is
# unchanged — it only polls musetalk.is_speaking, which the live engine sets via
# the same feed_audio() the TTS used. AVATAR_LIVE_MIC=1 turns it on; AVATAR_VC
# picks the converter (passthrough / dsp / rvc). When live mic is on we suppress
# the TTS greeting + chart narration so the two don't fight for the mouth.
LIVE_MIC = os.environ.get("AVATAR_LIVE_MIC", "0") == "1"
LIVE_MIC_CONVERTER = os.environ.get("AVATAR_VC", "rvc")

# Auto-drop background replacement first if sustained fps falls below this.
MIN_FPS = 18.0
STATS_EVERY = 250              # frames between [MAIN] perf lines
SLOW_FRAME_MS = 250.0

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 9998

BANNER = r"""
+=====================================================+
|        REALTIME AI AVATAR  -  TikTok Live           |
|  Webcam drives the head  -  you type, AI speaks      |
|  MuseTalk syncs the mouth to the AI voice            |
+=====================================================+
"""


def _character_path():
    """Return the first character.jpg that exists."""
    for c in CHARACTER_CANDIDATES:
        if os.path.exists(c):
            return c
    return CHARACTER_CANDIDATES[0]


# -----------------------------------------------------------------------------
# KEYBOARD (non-blocking, Windows msvcrt)
# -----------------------------------------------------------------------------
try:
    import msvcrt
    HAVE_MSVCRT = True
except ImportError:
    HAVE_MSVCRT = False


def _poll_key():
    """Return a pressed key char (lowercase) or None. Non-blocking."""
    if HAVE_MSVCRT and msvcrt.kbhit():
        try:
            return msvcrt.getch().decode(errors="ignore").lower()
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# CONTROL SOCKET SERVER
# -----------------------------------------------------------------------------
class ControlServer:
    """Local socket server; control_gui.py connects and sends newline commands.

    Protocol (one command per line, UTF-8):
        SPEAK <text>     speak a line
        MUTE 1 | MUTE 0  mute/unmute audio (mouth still animates)
        VOICE <name>     switch edge-tts voice
        QUIT             graceful shutdown
    """

    def __init__(self, tts, on_quit):
        self.tts = tts
        self.on_quit = on_quit
        self.connected = False
        self._running = True
        self._sock = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((CONTROL_HOST, CONTROL_PORT))
            self._sock.listen(1)
            self._sock.settimeout(1.0)
        except Exception as exc:
            print(f"[MAIN] control server bind failed ({exc}); GUI disabled.")
            return
        print(f"[MAIN] control server on {CONTROL_HOST}:{CONTROL_PORT}")
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            self.connected = True
            print("[MAIN] control GUI connected.")
            self._handle(conn)
            self.connected = False
            print("[MAIN] control GUI disconnected.")

    def _handle(self, conn):
        conn.settimeout(1.0)
        buf = ""
        while self._running:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            if not data:
                break
            buf += data.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._dispatch(line.strip())
        try:
            conn.close()
        except Exception:
            pass

    def _dispatch(self, line):
        if not line:
            return
        cmd, _, arg = line.partition(" ")
        cmd = cmd.upper()
        if cmd == "SPEAK":
            self.tts.speak(arg)
        elif cmd == "MUTE":
            self.tts.set_muted(arg.strip() in ("1", "true", "on", "True"))
            print(f"[MAIN] {'MUTED' if self.tts.muted else 'UNMUTED'}")
        elif cmd == "VOICE":
            self.tts.set_voice(arg.strip())
        elif cmd == "QUIT":
            self.on_quit()
        else:
            # bare text with no command keyword -> speak it
            self.tts.speak(line)

    def shutdown(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# STARTUP
# -----------------------------------------------------------------------------
def startup(cam=None, backend=None, hints=True):
    """Build every engine in order, printing progress. Returns a dict.

    cam     : a pre-built pyvirtualcam.Camera to publish into. avatar_camera.py
              injects a branded one; left None, one is created here.
    backend : pyvirtualcam backend ('obs' / 'unitycapture') used only when this
              function creates the camera itself.
    hints   : print the trailing OBS / control_gui usage hints (the camera
              launcher prints its own, so it passes hints=False).
    """
    print(BANNER)

    # JARVIS-style boot cue the moment we come online (non-blocking, crash-proof).
    try:
        from startup_sound import play_startup_sound
        play_startup_sound()
    except Exception:
        pass

    import pyvirtualcam
    from liveportrait_engine import LivePortraitEngine
    from musetalk_engine import MuseTalkEngine
    from compositor import Compositor
    from tts_stream_engine import TTSStreamEngine
    from trading_view import TradingView
    import enhance_engine

    char = _character_path()

    print("[1/5] LivePortrait loading...")
    liveportrait = LivePortraitEngine(char)
    print(f"      -> {liveportrait.startup_check()[1]}")

    print("[2/5] MuseTalk loading...")
    musetalk = MuseTalkEngine(char)
    print(f"      -> {musetalk.startup_check()[1]}")
    compositor = Compositor()
    print(f"      -> {compositor.startup_check()[1]}")

    print("[3/5] TTS engine...")
    tts = TTSStreamEngine(musetalk)
    print(f"      -> {tts.startup_check()[1]}")

    # LIVE MIC: build the voice-changer engine on the SAME mouth engine the TTS
    # feeds. Only one source may feed the mouth at a time, so when it's on we mute
    # the TTS (keeps the object alive for comment/brain features, but no audio).
    live_mic = None
    if LIVE_MIC:
        try:
            from voice_changer_engine import LiveVoiceEngine, make_converter
            live_mic = LiveVoiceEngine(musetalk, converter=make_converter(LIVE_MIC_CONVERTER))
            ok, msg = live_mic.startup_check()
            print(f"      -> live mic: {msg}")
            if ok:
                tts.set_muted(True)    # avatar speaks from YOUR mic, not the AI voice
                live_mic.start()
            else:
                live_mic = None
        except Exception as exc:
            print(f"      -> live mic unavailable ({exc}); using AI voice.")
            live_mic = None

    print("[4/5] Webcam opening...")
    cap = _open_webcam()
    if cap is None:
        print("      -> NO WEBCAM — driving with a static frame (head won't move).")
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"      -> webcam {WEBCAM_INDEX} @ {w}x{h}")

    print("[5/5] Virtual camera...")
    print(f"      -> enhance: {enhance_engine.startup_check()[1]}")
    if cam is None:
        _kw = dict(width=FRAME_SIZE, height=FRAME_SIZE, fps=FPS,
                   fmt=pyvirtualcam.PixelFormat.BGR)
        if backend:
            _kw["backend"] = backend
        cam = pyvirtualcam.Camera(**_kw)
    print(f"      -> virtual camera: {cam.device}")

    # CAMERA cue: the virtual camera is now detected / ready.
    try:
        from startup_sound import play_camera_sound
        play_camera_sound()
    except Exception:
        pass

    if hints:
        print("\n  AVATAR LIVE — open OBS, add Video Capture Device -> select the "
              "virtual cam.")
        print("  Run  python control_gui.py  in another terminal to type what the AI says.")
        print("  Keys here:  [M] mute   [Q] quit\n")

    # Optional spoken "JARVIS" greeting once everything is online, in the avatar's
    # own voice (mouth-synced by the run loop). Set AVATAR_GREETING="0" to mute it,
    # or to any custom line to change it.
    greeting = os.environ.get(
        "AVATAR_GREETING", "Good evening. All systems are now online and operational.")
    if greeting and greeting != "0" and live_mic is None:
        try:
            tts.speak(greeting)
        except Exception:
            pass

    # Chart scene: the AI-operated analytical chart (real data + drawings +
    # spoken analysis) when AVATAR_AICHART is set, else the classic synthetic one.
    ai_sym = os.environ.get("AVATAR_AICHART", "").strip()
    if ai_sym:
        try:
            from chart_pilot import ChartPilot
            sym = "PAXGUSDT" if ai_sym.lower() in ("1", "on", "true", "yes") else ai_sym.upper()
            chart = ChartPilot(sym, os.environ.get("AVATAR_AICHART_TF", "15m"),
                               size=(FRAME_SIZE, FRAME_SIZE),
                               display_name=os.environ.get("AVATAR_AICHART_NAME", "XAU/USD (gold)"),
                               narrate_lang=os.environ.get("AVATAR_TTS_LANG", "en")
                               if os.environ.get("AVATAR_TTS_LANG", "en") in ("en", "ar") else "en")
            print(f"      -> AI chart pilot: {chart.startup_check()[1]}")
        except Exception as exc:
            print(f"      -> AI chart unavailable ({exc}); using synthetic chart.")
            chart = TradingView("XAUUSD")
    else:
        chart = TradingView("XAUUSD")

    return {"liveportrait": liveportrait, "musetalk": musetalk,
            "compositor": compositor, "tts": tts, "enhance": enhance_engine,
            "cap": cap, "cam": cam, "chart": chart, "live_mic": live_mic}


# Virtual / non-physical cameras we must NEVER grab as input — feeding the OBS
# virtual cam back in is a loop and shows its placeholder logo (no face).
VIRTUAL_CAM_PATTERNS = ("obs", "virtual", "lsvcam", "splitcam", "manycam",
                        "droidcam", "ndi", "xsplit", "snap camera")


def _list_cameras():
    """Return [(index, name), ...] via pygrabber, or None if unavailable."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        return None


def _try_open(idx):
    """Open a camera index and require a real frame read. Returns cap or None."""
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            for _ in range(5):
                ok, fr = cap.read()
                if ok and fr is not None:
                    return cap
        cap.release()
    return None


def _open_webcam():
    """Open the REAL webcam and return it (or None).

    Uses device names (pygrabber) to prefer a physical camera and to NEVER grab a
    virtual camera (OBS/LSVCam/etc. — those show a placeholder with no face, which
    is exactly the "stuck on charts" symptom). If the real camera is busy (held by
    another app, e.g. a browser tab) it reports that clearly instead of silently
    grabbing the wrong device.
    """
    cams = _list_cameras()
    if cams:
        real = [(i, n) for i, n in cams
                if not any(p in n.lower() for p in VIRTUAL_CAM_PATTERNS)]
        env = os.environ.get("AVATAR_WEBCAM")
        order = ([int(env)] if env and env.isdigit() else []) + [i for i, _ in real]
        seen = set()
        for idx in order:
            if idx in seen:
                continue
            seen.add(idx)
            cap = _try_open(idx)
            if cap is not None:
                name = dict(cams).get(idx, "?")
                print(f"[MAIN] webcam: '{name}' (index {idx}) delivering frames.")
                from webcam_thread import ThreadedCamera
                return ThreadedCamera(cap)        # latest-frame buffer (low latency)
        names = ", ".join(f"{i}:'{n}'" for i, n in cams)
        busy = real[0][1] if real else "your webcam"
        print(f"[MAIN] REAL webcam unavailable — '{busy}' is busy or off. "
              f"Devices seen: [{names}]. Close the app using the camera "
              f"(a browser tab / video call / OBS), then press START again.")
        return None
    # no pygrabber: scan indices, require a real frame
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for idx in [WEBCAM_INDEX] + [i for i in range(0, 5) if i != WEBCAM_INDEX]:
            cap = _try_open(idx)
            if cap is not None:
                print(f"[MAIN] webcam: index {idx} (backend {be}) delivering frames.")
                from webcam_thread import ThreadedCamera
                return ThreadedCamera(cap)        # latest-frame buffer (low latency)
    print("[MAIN] no webcam delivered frames (in use by another app?).")
    return None


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def run(eng=None):
    """Start everything and run the avatar until quit.

    eng : a pre-built engine dict from startup(); when None we build one. The
          camera launcher (avatar_camera.py) builds it with a branded camera and
          passes it in so the device name / instructions are printed first.
    """
    if eng is None:
        try:
            eng = startup()
        except Exception as exc:
            print(f"[MAIN] startup failed: {exc}")
            return

    liveportrait = eng["liveportrait"]; musetalk = eng["musetalk"]
    compositor = eng["compositor"]; tts = eng["tts"]; enhance = eng["enhance"]
    cap = eng["cap"]; cam = eng["cam"]; chart = eng["chart"]
    live_mic = eng.get("live_mic")

    state = {"running": True}

    def _quit():
        state["running"] = False

    control = ControlServer(tts, _quit)

    blank = np.full((FRAME_SIZE, FRAME_SIZE, 3), 60, dtype=np.uint8)
    last_frame = blank.copy()
    cached_face = None
    cached_bbox = None
    frame_count = 0
    prev_small = None
    lp_ms = mt_ms = 0.0
    drop_bg = False
    noface = 0                # consecutive no-face frames
    chart_fade = 0.0          # 0 = avatar, 1 = trading chart
    in_chart = False
    stats_timer = time.perf_counter()
    session_start = time.perf_counter()
    next_tick = time.monotonic()

    # SCENE cue: the avatar scene is now live and about to stream.
    try:
        from startup_sound import play_scene_sound
        play_scene_sound()
    except Exception:
        pass

    try:
        while state["running"]:
            t0 = time.perf_counter()

            # 1) capture webcam frame (driving motion)
            driving = last_frame
            if cap is not None:
                ok, frame = cap.read()
                if ok and frame is not None:
                    driving = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                    last_frame = driving

            small = cv2.cvtColor(cv2.resize(driving, (64, 64)),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            motion = float(np.mean(np.abs(small - prev_small))) \
                if prev_small is not None else 99.0
            prev_small = small

            # 2) LivePortrait: webcam -> animated AI face (every LP_INTERVAL frames)
            t1 = time.perf_counter()
            lp_fresh = False
            try:
                if getattr(liveportrait, "fallback_mode", False):
                    ai_face = liveportrait.process_frame(driving)
                    lp_fresh = True
                elif (cached_face is None or (frame_count % LP_INTERVAL) == 0
                      or motion > 4.0):
                    ai_face = liveportrait.process_frame(driving)
                    cached_face = ai_face
                    lp_fresh = True
                else:
                    ai_face = cached_face
            except Exception as exc:
                print(f"[MAIN] LP error: {exc}")
                ai_face = driving
            lp_ms = (time.perf_counter() - t1) * 1000

            # face-loss -> trading chart scene (crossfade). Disabled in LP fallback.
            # Charts only when there is NO face at all (operator left / looked
            # away). A small/far face still shows the avatar (held) — not charts.
            face_ok = (
                not getattr(liveportrait, "fallback_mode", False)
                and (getattr(liveportrait, "_face_found", False)
                     or cached_face is not None)
            )
            noface = 0 if face_ok else noface + 1
            want_chart = (SHOW_CHART_ON_FACE_LOSS
                          and not getattr(liveportrait, "fallback_mode", False)
                          and noface >= int(FPS * NO_FACE_SECONDS))
            if want_chart and chart_fade < 1.0:
                chart_fade = min(1.0, chart_fade + CHART_FADE_STEP)
            elif not want_chart and chart_fade > 0.0:
                chart_fade = max(0.0, chart_fade - CHART_FADE_STEP)
            if want_chart and not in_chart:
                in_chart = True; chart.reset_price_drift()
                print("[MAIN] no face — showing live charts.")
            elif not want_chart and in_chart and chart_fade <= 0.0:
                in_chart = False; print("[MAIN] face back — avatar resumed.")

            # 3+4) mouth sync + composite (only while speaking).
            # The AI face is unchanged on cached-LP frames, so the mouth bbox is
            # only re-detected when LivePortrait produced a fresh face — the rest
            # of the time we reuse the cached bbox (saves a FaceMesh pass/frame).
            t2 = time.perf_counter()
            speaking = bool(getattr(musetalk, "is_speaking", False))
            # While the AI chart is on screen, let the avatar SPEAK its live
            # market read (the ChartPilot paces one line per analysis move).
            if in_chart and not speaking and live_mic is None and hasattr(chart, "next_narration"):
                _line = chart.next_narration()
                if _line:
                    try:
                        tts.speak(_line)
                    except Exception:
                        pass
            if chart_fade >= 1.0:
                # fully on charts — skip the now-hidden avatar mouth/enhance work
                final = chart.render(speaking=speaking)
                mt_ms = 0.0
            else:
                if speaking:
                    try:
                        if lp_fresh or motion > 4.0 or cached_bbox is None:
                            cached_bbox = compositor.detect_mouth_bbox(ai_face)
                        bbox = cached_bbox
                        mouth = musetalk.process_mouth(ai_face, bbox)
                        ai_face = compositor.blend_mouth(ai_face, mouth, bbox)
                    except Exception as exc:
                        print(f"[MAIN] mouth/composite error: {exc}")
                mt_ms = (time.perf_counter() - t2) * 1000

                # 5) enhance (studio bg, ticker, badges, grade)
                try:
                    final = enhance.enhance_frame(ai_face, is_speaking=speaking)
                except Exception as exc:
                    print(f"[MAIN] enhance error: {exc}")
                    final = ai_face
                if chart_fade > 0.0:      # crossfade avatar <-> chart
                    final = cv2.addWeighted(final, 1.0 - chart_fade,
                                            chart.render(speaking=speaking), chart_fade, 0)

            # 6) send to OBS
            if final.shape[0] != FRAME_SIZE or final.shape[1] != FRAME_SIZE:
                final = cv2.resize(final, (FRAME_SIZE, FRAME_SIZE))
            cam.send(np.ascontiguousarray(final))

            # --- local hotkeys ---
            key = _poll_key()
            if key == "m":
                tts.set_muted(not tts.muted)
                print(f"[MAIN] {'MUTED' if tts.muted else 'UNMUTED'}")
            elif key == "q":
                state["running"] = False

            # --- perf stats ---
            frame_count += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > SLOW_FRAME_MS:
                print(f"[MAIN] slow frame {elapsed_ms:.0f}ms "
                      f"(LP {lp_ms:.0f} MT {mt_ms:.0f})")
            if frame_count % STATS_EVERY == 0:
                fps = STATS_EVERY / (time.perf_counter() - stats_timer)
                gpu = _vram_mb()
                print(f"[MAIN] {fps:.1f}fps | LP:{lp_ms:.0f}ms | MT:{mt_ms:.0f}ms | "
                      f"VRAM:{gpu:.0f}MB | bg:{'off' if drop_bg else 'on'}")
                if fps < MIN_FPS and not drop_bg:
                    drop_bg = True
                    enhance.background_composite = lambda f: f   # skip bg cut once
                    print("[MAIN] fps < 18 — dropping background replacement to recover.")
                stats_timer = time.perf_counter()

            # --- pacing (always yield) ---
            next_tick += TARGET_FRAME_TIME
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C — shutting down.")
    finally:
        _shutdown(eng, control, frame_count, session_start)


def _vram_mb():
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _shutdown(eng, control, frame_count, session_start):
    """Graceful shutdown: release webcam, camera, stop TTS, print session stats."""
    runtime = time.perf_counter() - session_start
    for fn in (control.shutdown,
               (eng.get("live_mic").shutdown if eng.get("live_mic") else (lambda: None)),
               eng["tts"].shutdown,
               lambda: eng["cap"].release() if eng["cap"] else None,
               eng["cam"].close):
        try:
            fn()
        except Exception:
            pass
    words = getattr(eng["tts"], "words_spoken", 0)
    lines = getattr(eng["tts"], "lines_spoken", 0)
    fps = frame_count / runtime if runtime > 0 else 0
    print("\n[MAIN] ===== SESSION STATS =====")
    print(f"[MAIN] runtime:   {runtime/60:.1f} min ({frame_count} frames, {fps:.1f} fps avg)")
    print(f"[MAIN] lines spoken: {lines}  ({words} words)")
    print("[MAIN] Avatar session ended.")


if __name__ == "__main__":
    run()
