# =============================================================================
# avatar_camera.py  —  AVATAR CAMERA (your avatar AS a webcam)
# -----------------------------------------------------------------------------
# Turns the Avatar Studio pipeline into a real, system-wide CAMERA that any app
# detects — Zoom, Teams, Google Meet, Chrome/Edge, OBS, Discord, TikTok Live
# Studio, the Windows Camera app, anything that lists webcams.
#
# It boots the exact same pipeline as realtime_avatar.py (webcam -> AI face ->
# mouth-sync -> enhance) but is CAMERA-FIRST: it ALWAYS publishes every composed
# frame into a virtual-camera device, and prints the precise device name to pick
# in your other app.  No OBS scene wiring required — just select the device.
#
#   python avatar_camera.py                 # go live on the best backend
#   python avatar_camera.py --list          # show camera devices on this PC
#   python avatar_camera.py --backend obs   # force a specific backend
#   python avatar_camera.py --native        # feed the native "Avatar Studio
#                                           # Camera" that appears only while
#                                           # running (see native_camera\README)
#   python avatar_camera.py --dshow         # feed the DirectShow "Avatar Studio
#                                           # Camera" - a normal webcam with NO
#                                           # virtual-camera tag (OBS/Zoom/TikTok)
#
# Type what the avatar says with:  python control_gui.py   (another terminal)
# Stop with [Q] here (or Ctrl+C).
# =============================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = os.path.join(PROJECT_DIR, "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

# pyvirtualcam backends to try, in order of preference. 'obs' = the OBS Virtual
# Camera DirectShow device (installed with OBS Studio, present on this machine);
# 'unitycapture' = the Unity Capture filter if you have it instead.
BACKENDS = ("obs", "unitycapture")

# How a backend name maps to the friendly device name other apps will SHOW in
# their camera dropdown — so we can tell the user exactly what to click.
BACKEND_DEVICE_HINT = {
    "obs": "OBS Virtual Camera",
    "unitycapture": "Unity Video Capture",
}


def _list_devices():
    """Print every camera Windows currently exposes (what other apps see)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        devs = FilterGraph().get_input_devices()
    except Exception as exc:
        print(f"  (could not enumerate devices: {exc})")
        return
    if not devs:
        print("  (no camera devices found)")
        return
    print("  Cameras this PC exposes to other apps:")
    for i, name in enumerate(devs):
        tag = "  <- virtual" if any(
            k in name.lower() for k in ("obs", "virtual", "unity")) else ""
        print(f"    [{i}] {name}{tag}")


def _open_camera(width, height, fps, prefer=None):
    """Open the first working virtual-camera backend.

    Returns (cam, backend_name). Raises RuntimeError with install guidance if no
    backend is available.
    """
    import pyvirtualcam
    order = ([prefer] if prefer else []) + [b for b in BACKENDS if b != prefer]
    errors = []
    for be in order:
        try:
            cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR, backend=be)
            return cam, be
        except Exception as exc:
            errors.append(f"    - {be}: {exc}")
    detail = "\n".join(errors) if errors else "    (no backends attempted)"
    raise RuntimeError(
        "No virtual-camera backend is available.\n" + detail + "\n\n"
        "  Fix: install OBS Studio (https://obsproject.com) — it registers the\n"
        "  'OBS Virtual Camera' device this bridge publishes into. After install,\n"
        "  you do NOT need to open OBS; just rerun:  python avatar_camera.py")


_BANNER = r"""
  ___ _   _____ _____ ___    ___    ___ ___ __  __ ___ ___ ___
 / _ \ \ / / _ |_   _/ _ \  | _ \  / __| _ \  \/  | __| _ \   |
| (_| |\ V / __ | | || (_) | |   / | (__|   / |\/| | _||   / - |
 \___/  \_/_/ |_| |_| \___/  |_|_\  \___|_|_\_|  |_|___|_|_\___|
            A V A T A R   C A M E R A   //   live virtual webcam
"""


def _print_online_box(device_name, backend):
    """Loud, copy-paste-able 'how to select me' banner."""
    line = "=" * 68
    print("\n" + line)
    print("  AVATAR CAMERA IS ONLINE  ●  publishing every frame")
    print(line)
    print(f"  In ANY app's camera menu, pick:   >>>  {device_name}  <<<")
    print(f"  (backend: {backend})")
    print()
    print("  Works in: Zoom · Teams · Google Meet · Chrome/Edge · Discord ·")
    print("            OBS · TikTok Live Studio · Windows Camera app")
    print()
    print("  Speak:  run  python control_gui.py  in another terminal and type.")
    print("  Stop:   press [Q] in this window  (or Ctrl+C)")
    print(line + "\n")


class _SharedMemCam:
    """A drop-in stand-in for pyvirtualcam.Camera that publishes frames to the
    native virtual camera via the shared-memory file (see avatar_sharedframe).
    realtime_avatar.run() only calls .send()/.device/.close(), so this is enough.
    """

    def __init__(self, width, height):
        import avatar_sharedframe as sf
        self._w = sf.SharedFrameWriter(width, height)
        self.device = "Avatar Studio Camera (native)"
        self.path = self._w.path

    def send(self, frame):
        self._w.write(frame)

    def sleep_until_next_frame(self):
        pass

    def close(self):
        self._w.close()


def _run_native(prefer_name):
    """--native: feed the native MFCreateVirtualCamera device via shared memory.

    The camera DEVICE appears only while the native host (vcam_host.exe) holds
    it, so we launch the host (if built) and feed frames into the shared file it
    reads. If the host isn't built yet, we still write frames (and tell the user
    how to build it) so the pipeline can be verified.
    """
    import os
    import subprocess
    import realtime_avatar as ra

    cam = _SharedMemCam(ra.FRAME_SIZE, ra.FRAME_SIZE)
    print(f"[NATIVE] writing avatar frames to shared file: {cam.path}")

    host = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "native_camera", "vcam_host.exe")
    proc = None
    if os.path.exists(host):
        try:
            proc = subprocess.Popen([host, prefer_name or "Avatar Studio Camera"])
            print(f"[NATIVE] launched host -> '{prefer_name or 'Avatar Studio Camera'}' "
                  "should now appear in apps' camera lists.")
        except Exception as exc:
            print(f"[NATIVE] could not launch host ({exc}); frames still written.")
    else:
        print("[NATIVE] native host not built yet — frames are being written, but the")
        print("[NATIVE] camera device won't appear until you build it:")
        print("[NATIVE]   1) install VS Build Tools (see native_camera\\README.md)")
        print("[NATIVE]   2) native_camera\\setup.ps1  ->  build.ps1  ->  install.ps1")

    try:
        eng = ra.startup(cam=cam, hints=False)
    except Exception as exc:
        print(f"[NATIVE] engine startup failed: {exc}")
        cam.close()
        if proc:
            proc.terminate()
        return 1
    try:
        ra.run(eng)
    finally:
        if proc:
            proc.terminate()
    return 0


def _run_dshow():
    """--dshow: feed the DirectShow "Avatar Studio Camera" (no virtual tag).

    That camera is a registered DirectShow filter (native_camera_dshow), so it is
    always present in apps that block/flag virtual cameras (OBS, Zoom, Discord,
    Chrome, TikTok Live Studio) as an ordinary webcam. We only need to write the
    avatar frames into the shared file it reads - no host process required.
    """
    import realtime_avatar as ra
    cam = _SharedMemCam(ra.FRAME_SIZE, ra.FRAME_SIZE)
    print(f"[DSHOW] feeding 'Avatar Studio Camera' (DirectShow, no virtual tag).")
    print(f"[DSHOW] writing avatar frames to: {cam.path}")
    print("[DSHOW] pick 'Avatar Studio Camera' in OBS / Zoom / Discord / Chrome /")
    print("[DSHOW] TikTok Live Studio. (If it's missing, run native_camera_dshow\\install.ps1)")
    try:
        eng = ra.startup(cam=cam, hints=False)
    except Exception as exc:
        print(f"[DSHOW] engine startup failed: {exc}")
        cam.close()
        return 1
    ra.run(eng)
    return 0


def main(argv):
    prefer = None
    args = [a for a in argv[1:]]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    print(_BANNER)

    if "--list" in args:
        _list_devices()
        return 0

    if "--dshow" in args:
        return _run_dshow()

    if "--native" in args:
        name = None
        if "--name" in args:
            j = args.index("--name")
            if j + 1 < len(args):
                name = args[j + 1]
        return _run_native(name)

    if "--backend" in args:
        i = args.index("--backend")
        if i + 1 < len(args):
            prefer = args[i + 1].lower()
        else:
            print("  --backend needs a value (obs | unitycapture)")
            return 2

    # Import the pipeline lazily so --list / --help stay instant.
    import realtime_avatar as ra

    # 1) Claim the virtual camera FIRST so we can show the exact device name and
    #    fail fast (with install help) before spending ~30s loading the engines.
    try:
        cam, backend = _open_camera(ra.FRAME_SIZE, ra.FRAME_SIZE, ra.FPS, prefer)
    except RuntimeError as exc:
        print(f"[CAMERA] {exc}")
        return 1
    device_name = getattr(cam, "device", None) or BACKEND_DEVICE_HINT.get(
        backend, "virtual camera")
    print(f"[CAMERA] claimed '{device_name}' via the {backend} backend.")
    print("[CAMERA] loading the avatar engines (first run can take ~30s)...\n")

    # 2) Build the rest of the pipeline around OUR camera and run the same loop
    #    realtime_avatar uses (it already sends every composed frame to cam).
    try:
        eng = ra.startup(cam=cam, backend=backend, hints=False)
    except Exception as exc:
        print(f"[CAMERA] engine startup failed: {exc}")
        try:
            cam.close()
        except Exception:
            pass
        return 1

    _print_online_box(device_name, backend)
    ra.run(eng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
