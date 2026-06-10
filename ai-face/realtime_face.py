# =============================================================================
# realtime_face.py
# -----------------------------------------------------------------------------
# Real-time AI talking-face engine for TikTok Live streaming.
#
# Architecture:
#   - Socket server thread:   listens on 127.0.0.1:9999 for text lines from the
#                             control panel and pushes them onto text_queue.
#   - TTS + lip-sync worker:  pops text -> edge-tts WAV -> Wav2Lip inference ->
#                             decodes the result video into frames -> (optional)
#                             GFPGAN face restoration for sharp, human lips ->
#                             packages a finished "clip" (frames + audio) onto
#                             clip_queue. The ENTIRE clip is rendered before it
#                             is queued, so playback never stutters.
#   - Main thread:            pyvirtualcam loop at 25 fps. When a clip is ready
#                             it starts the audio (so the stream actually has a
#                             voice) and plays the frames in sync; otherwise it
#                             shows the idle frame.
#
# OBS picks up the virtual camera (video) and Desktop Audio (the spoken voice)
# and streams both to TikTok Live Studio.
# =============================================================================

import os
import sys
import time
import socket
import shutil
import threading
import subprocess
import queue

# Force UTF-8 stdout so status glyphs ([*] [✓] [!]) don't crash the Windows
# console, whose default code page (cp1252) can't encode them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

try:
    import cv2
except ImportError:
    print("[!] opencv-python not installed. Run:  pip install opencv-python")
    raise

try:
    import pyvirtualcam
except ImportError:
    print("[!] pyvirtualcam not installed. Run:  pip install pyvirtualcam")
    raise

# winsound is part of the Windows standard library; used to play the spoken
# audio out of the default playback device (which OBS captures as Desktop Audio).
try:
    import winsound
    HAVE_WINSOUND = True
except ImportError:
    HAVE_WINSOUND = False


# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS  (override paths / sizes / quality here)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHARACTER_IMAGE = os.path.join(PROJECT_ROOT, "character.jpg")
WAV2LIP_PATH = os.path.join(PROJECT_ROOT, "Wav2Lip")
WAV2LIP_INFERENCE = os.path.join(WAV2LIP_PATH, "inference.py")
CHECKPOINT = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip_gan.pth")

WAV_PATH = os.path.join(PROJECT_ROOT, "speech_output.wav")
RESULT_VIDEO = os.path.join(PROJECT_ROOT, "result_voice.mp4")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "clips")        # per-clip audio lives here

OUTPUT_WIDTH = 720      # virtual-camera resolution (higher = clearer on stream)
OUTPUT_HEIGHT = 720
FPS = 25

# Quality: run GFPGAN face restoration on every rendered frame. This is what
# makes the lips look sharp and human instead of soft/blurry. Set to False for
# faster (but softer) rendering on weaker GPUs.
ENHANCE = True

# A/V sync fine-tune, in seconds. The video is slaved to the audio clock; this
# shifts the video relative to the audio to absorb any constant device latency.
#   - lips move BEFORE you hear the voice  -> make this MORE negative (e.g. -0.10)
#   - you hear the voice BEFORE the lips    -> make this MORE positive (e.g. +0.10)
AUDIO_OFFSET = 0.0

HOST = "127.0.0.1"
PORT = 9999

CLIP_QUEUE_MAXSIZE = 50         # how many finished clips may wait to play
TEXT_QUEUE_MAXSIZE = 100        # how many queued lines we allow

# -----------------------------------------------------------------------------
# SHARED STATE
# -----------------------------------------------------------------------------
text_queue = queue.Queue(maxsize=TEXT_QUEUE_MAXSIZE)       # incoming text lines
clip_queue = queue.Queue(maxsize=CLIP_QUEUE_MAXSIZE)       # finished clips ready to play
SPEAKING = False                                           # True while a clip is on screen
stop_event = threading.Event()                            # signals shutdown
idle_frame = None                                         # static character frame (BGR)
idle_display = None                                       # idle frame WITH gray overlay (precomputed)
_clip_counter = 0                                         # unique id for per-clip audio


# -----------------------------------------------------------------------------
# OPTIONAL ENHANCER (loaded lazily; engine still runs if it is unavailable)
# -----------------------------------------------------------------------------
def _maybe_enhance(frame, out_size):
    """Enhance a frame with GFPGAN if ENHANCE is on, else just resize it.

    Always returns a frame at out_size. Never raises.
    """
    if ENHANCE:
        try:
            import enhance_engine
            return enhance_engine.enhance_bgr(frame, out_size=out_size)
        except Exception as exc:
            print("[!] Enhance unavailable, using plain frame:", exc)
    return cv2.resize(frame, out_size)


# -----------------------------------------------------------------------------
# IMAGE / FRAME HELPERS
# -----------------------------------------------------------------------------
def load_idle_frame():
    """Load the character image into a static idle frame (enhanced if enabled).

    Falls back to a generated placeholder frame if the image is missing.
    """
    global idle_frame, idle_display
    if os.path.exists(CHARACTER_IMAGE):
        img = cv2.imread(CHARACTER_IMAGE)
        if img is None:
            print("[!] Could not decode", CHARACTER_IMAGE, "- using placeholder.")
            img = _placeholder_image()
    else:
        print("[!] character.jpg not found - using placeholder. Replace it with your face image.")
        img = _placeholder_image()

    # Enhance the idle face once so it matches the crisp speaking frames.
    if ENHANCE and os.path.exists(CHARACTER_IMAGE):
        print("[*] Enhancing idle character frame (one-time)...")
        idle_frame = _maybe_enhance(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    else:
        idle_frame = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # Precompute the idle frame WITH its gray status dot so the camera loop can
    # send it with zero per-frame work (no copy, no draw).
    idle_display = draw_status_overlay(idle_frame, False)
    return idle_frame


def draw_overlay_inplace(frame, speaking):
    """Draw the status circle directly into a frame (no copy). Returns it."""
    center = (OUTPUT_WIDTH - 34, 34)
    color = (0, 220, 0) if speaking else (120, 120, 120)   # BGR
    cv2.circle(frame, center, 16, color, thickness=-1)
    cv2.circle(frame, center, 16, (255, 255, 255), thickness=2)
    return frame


def _placeholder_image():
    """Generate a simple gray placeholder image with a label."""
    img = np.full((512, 512, 3), 40, dtype=np.uint8)
    cv2.putText(img, "character.jpg", (60, 246),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "(replace me)", (110, 296),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
    return img


def draw_status_overlay(frame, speaking):
    """Draw a status circle in the top-right corner.

    Green circle while speaking, gray while idle. Returns a new frame copy.
    """
    out = frame.copy()
    center = (OUTPUT_WIDTH - 34, 34)
    color = (0, 220, 0) if speaking else (120, 120, 120)   # BGR
    cv2.circle(out, center, 16, color, thickness=-1)
    cv2.circle(out, center, 16, (255, 255, 255), thickness=2)
    return out


# -----------------------------------------------------------------------------
# WAV2LIP INFERENCE
# -----------------------------------------------------------------------------
def run_wav2lip():
    """Run Wav2Lip inference as a subprocess.

    Uses sys.executable so the subprocess matches the current Python env, and
    sets CUDA_VISIBLE_DEVICES=0 to force GPU usage.

    Returns the path to the generated result video, or None on failure.
    """
    if not os.path.exists(WAV2LIP_INFERENCE):
        print("[!] Wav2Lip inference.py not found at", WAV2LIP_INFERENCE)
        print("    Clone it:  git clone https://github.com/Rudrabha/Wav2Lip")
        return None

    if not os.path.exists(CHECKPOINT):
        print("[!] Checkpoint not found at", CHECKPOINT)
        print("    Download wav2lip_gan.pth and place it in Wav2Lip/checkpoints/")
        return None

    if not os.path.exists(WAV_PATH):
        print("[!] Audio file not found at", WAV_PATH)
        return None

    if os.path.exists(RESULT_VIDEO):
        try:
            os.remove(RESULT_VIDEO)
        except OSError:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    command = [
        sys.executable,
        WAV2LIP_INFERENCE,
        "--checkpoint_path", CHECKPOINT,
        "--face", CHARACTER_IMAGE,
        "--audio", WAV_PATH,
        "--outfile", RESULT_VIDEO,
        "--resize_factor", "1",
        "--nosmooth",
    ]

    print("[*] Wav2Lip: running inference (GPU)...")
    try:
        result = subprocess.run(
            command,
            cwd=WAV2LIP_PATH,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        print("[!] Wav2Lip subprocess failed to start:", exc)
        return None

    if result.returncode != 0:
        print("[!] Wav2Lip inference failed (exit code", result.returncode, ")")
        tail = result.stderr.decode(errors="ignore").strip().splitlines()[-15:]
        for line in tail:
            print("    ", line)
        return None

    if not os.path.exists(RESULT_VIDEO):
        print("[!] Wav2Lip finished but no output video was produced.")
        return None

    print("[✓] Wav2Lip: inference complete ->", RESULT_VIDEO)
    return RESULT_VIDEO


def render_clip(video_path):
    """Decode a video into a fully-rendered list of output frames.

    Each frame is resized (and GFPGAN-enhanced when ENHANCE is on) to the
    output resolution. The whole clip is built before playback so frames never
    arrive late and stutter.

    Returns a list of BGR frames (possibly empty).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Could not open result video:", video_path)
        return []

    frames = []
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_maybe_enhance(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT)))
    cap.release()
    return frames


# -----------------------------------------------------------------------------
# WORKER HELPERS
# -----------------------------------------------------------------------------
def _render_frames(runtime, wav_path):
    """Produce finished output frames for a wav.

    Prefers the resident in-process runtime (fast). If it is unavailable, falls
    back to the Wav2Lip subprocess + per-frame decode/enhance path.
    """
    if runtime is not None:
        frames = runtime.infer(wav_path)              # in-memory Wav2Lip
        if not frames:
            return []
        if ENHANCE:
            try:
                import enhance_engine
                return enhance_engine.enhance_frames(
                    frames, out_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
            except Exception as exc:
                print("[!] Enhance failed, using plain frames:", exc)
        return [cv2.resize(f, (OUTPUT_WIDTH, OUTPUT_HEIGHT)) for f in frames]

    # --- Fallback: subprocess Wav2Lip + decode ---
    video = run_wav2lip()
    if not video:
        return []
    return render_clip(video)


def _warmup(runtime):
    """Trigger CUDA kernel autotuning once so the first real line is fast.

    Renders a short silent clip through the full pipeline. Best-effort.
    """
    if runtime is None:
        return
    import wave
    warm_wav = os.path.join(CLIPS_DIR, "_warmup.wav")
    try:
        with wave.open(warm_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)        # 0.5s of silence
        print("[*] Worker: warming up GPU kernels (one-time)...")
        _render_frames(runtime, warm_wav)
        print("[✓] Worker: warm-up complete.")
    except Exception as exc:
        print("[!] Worker: warm-up skipped:", exc)
    finally:
        if os.path.exists(warm_wav):
            try:
                os.remove(warm_wav)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# WORKER THREAD: TTS + LIP-SYNC + ENHANCE PIPELINE
# -----------------------------------------------------------------------------
def tts_lipsync_worker():
    """Continuously turn queued text into finished, ready-to-play clips."""
    global _clip_counter

    try:
        from tts_engine import speak
    except Exception as exc:
        print("[!] Could not import tts_engine.speak:", exc)
        print("    The worker will exit. Fix TTS setup and restart.")
        return

    # Load the resident in-process runtime (Wav2Lip + s3fd kept in VRAM). This
    # eliminates the ~10s subprocess cold-start per line. If it can't load we
    # fall back to the subprocess path.
    runtime = None
    try:
        import face_runtime
        runtime = face_runtime.FaceRuntime(
            checkpoint=CHECKPOINT, character_image=CHARACTER_IMAGE)
    except Exception as exc:
        print("[!] In-process runtime unavailable, using subprocess fallback:", exc)
        runtime = None

    if ENHANCE:
        try:
            import enhance_engine
            enhance_engine.is_available()
        except Exception as exc:
            print("[!] Enhancer warm-up skipped:", exc)

    _warmup(runtime)
    print("[✓] Worker: TTS + lip-sync pipeline ready.")

    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if text is None:                # shutdown sentinel
            break

        print("[*] Worker: processing ->", repr(text))
        t0 = time.time()
        try:
            # 1) Text -> WAV
            speak(text, WAV_PATH)

            # 2) Keep a private copy of this line's audio (the shared WAV_PATH
            #    gets overwritten by the next line).
            _clip_counter += 1
            clip_wav = os.path.join(CLIPS_DIR, f"clip_{_clip_counter}.wav")
            try:
                shutil.copyfile(WAV_PATH, clip_wav)
            except OSError:
                clip_wav = None

            # 3) WAV (+ image) -> finished, enhanced frames.
            frames = _render_frames(runtime, WAV_PATH)
            if not frames:
                print("[!] No frames rendered for this line.")
                continue

            # Bake the green "speaking" dot in here (worker thread) so the main
            # camera loop can send frames with zero per-frame CPU work.
            for f in frames:
                draw_overlay_inplace(f, True)

            clip = {"frames": frames, "wav": clip_wav}
            clip_queue.put(clip)
            print(f"[✓] Worker: clip ready ({len(frames)} frames, "
                  f"{time.time() - t0:.1f}s). Queued for playback.")
        except Exception as exc:
            print("[!] Worker error while processing line:", exc)
        finally:
            text_queue.task_done()

    print("[*] Worker: stopped.")


# -----------------------------------------------------------------------------
# SOCKET SERVER THREAD
# -----------------------------------------------------------------------------
def socket_server():
    """Listen for newline-delimited text lines and enqueue them."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except OSError as exc:
        print("[!] Could not bind socket on", f"{HOST}:{PORT}", "->", exc)
        print("    Is another realtime_face.py already running?")
        stop_event.set()
        return

    server.listen(5)
    server.settimeout(1.0)
    print("[✓] Socket server listening on", f"{HOST}:{PORT}")

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        print("[✓] Control panel connected from", addr)
        threading.Thread(target=_handle_client, args=(conn, addr), daemon=True).start()

    server.close()
    print("[*] Socket server stopped.")


def _handle_client(conn, addr):
    """Read newline-delimited text from one client connection."""
    conn.settimeout(1.0)
    buffer = ""
    with conn:
        while not stop_event.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    text_queue.put(line, timeout=1.0)
                    print("[✓] Queued line:", repr(line))
                except queue.Full:
                    print("[!] Text queue full - dropping line:", repr(line))
    print("[*] Control panel disconnected:", addr)


# -----------------------------------------------------------------------------
# AUDIO PLAYBACK
# -----------------------------------------------------------------------------
def start_audio(wav_path):
    """Begin playing a WAV asynchronously through the default output device.

    OBS captures this via 'Desktop Audio'. No-op if winsound or the file is
    unavailable.
    """
    if not HAVE_WINSOUND or not wav_path or not os.path.exists(wav_path):
        return
    try:
        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as exc:
        print("[!] Could not play audio:", exc)


def stop_audio():
    """Stop any audio currently playing."""
    if HAVE_WINSOUND:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# MAIN: VIRTUAL CAMERA LOOP (plays clips with synced audio)
# -----------------------------------------------------------------------------
def main():
    """Start all threads and run the pyvirtualcam loop at FPS."""
    global SPEAKING

    print("=" * 64)
    print(" AI Talking Face Engine - realtime_face.py")
    print("=" * 64)

    # Fresh clips directory for per-line audio.
    try:
        if os.path.isdir(CLIPS_DIR):
            shutil.rmtree(CLIPS_DIR, ignore_errors=True)
        os.makedirs(CLIPS_DIR, exist_ok=True)
    except OSError:
        pass

    load_idle_frame()

    threading.Thread(target=socket_server, daemon=True).start()
    threading.Thread(target=tts_lipsync_worker, daemon=True).start()

    # Playback state for the current clip.
    current = None          # dict: frames + wav
    clip_start = 0.0        # monotonic time when this clip's audio began (master clock)

    try:
        with pyvirtualcam.Camera(width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
                                 fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            print("[✓] Virtual camera started:", cam.device)
            print(f"[*] Resolution {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @ {FPS}fps"
                  + ("  | GFPGAN enhance: ON" if ENHANCE else ""))
            print("[*] OBS: Add Source -> Video Capture Device -> select this camera.")
            print("[*] OBS: enable 'Desktop Audio' so the spoken voice is captured.")
            print("[*] Type in the control panel to make the face speak.")

            # Explicit frame pacing. We do NOT use cam.sleep_until_next_frame()
            # because it stops sleeping once it thinks it's "behind" — which
            # happens the instant the render worker steals CPU — turning this
            # into a busy-spin that starves the worker. Instead we always
            # time.sleep() to the next tick (yielding the GIL), and if we fall
            # behind we just reset the clock rather than spin. The audio clock
            # selects the right frame, so a slightly low frame rate never
            # desyncs.
            frame_interval = 1.0 / FPS
            next_tick = time.monotonic()

            while not stop_event.is_set():
                # Start a new clip if we aren't playing one. The audio is the
                # master clock: start it, timestamp the start, then pick the
                # video frame by elapsed time below.
                if current is None:
                    try:
                        current = clip_queue.get_nowait()
                    except queue.Empty:
                        current = None
                    if current is not None:
                        SPEAKING = True
                        start_audio(current.get("wav"))
                        clip_start = time.monotonic()

                if current is not None:
                    frames = current["frames"]
                    nframes = len(frames)
                    elapsed = time.monotonic() - clip_start + AUDIO_OFFSET
                    target = int(elapsed * FPS)

                    if target >= nframes:
                        # The clip's video has played out; clean up and go idle.
                        wav = current.get("wav")
                        if wav and os.path.exists(wav):
                            try:
                                os.remove(wav)
                            except OSError:
                                pass
                        current = None
                        SPEAKING = False
                        cam.send(idle_display)
                    else:
                        # frames already carry the baked-in green dot.
                        cam.send(frames[target if target > 0 else 0])
                else:
                    SPEAKING = False
                    cam.send(idle_display)

                # Sleep to the next tick, always yielding so the worker can run.
                next_tick += frame_interval
                now = time.monotonic()
                sleep_for = next_tick - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = now            # fell behind: reset, don't spin

    except RuntimeError as exc:
        print("[!] Virtual camera error:", exc)
        print("    Make sure OBS is installed and 'Start Virtual Camera' has been")
        print("    run at least once so the OBS Virtual Camera device exists.")
    except KeyboardInterrupt:
        print("\n[*] Shutting down (Ctrl+C).")
    finally:
        stop_event.set()
        stop_audio()
        try:
            text_queue.put_nowait(None)     # wake the worker for clean exit
        except queue.Full:
            pass
        time.sleep(0.3)
        print("[✓] Engine stopped.")


if __name__ == "__main__":
    main()
