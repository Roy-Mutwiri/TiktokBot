# =============================================================================
# setup_models.py  —  model download / readiness checker
# -----------------------------------------------------------------------------
# Run ONCE before going live:
#
#   python setup_models.py
#
# It checks every model the realtime avatar needs, downloads the ones it can via
# huggingface_hub.snapshot_download, and prints a clear checklist of what is
# ready and exact instructions / URLs for anything that needs a manual step
# (clone a repo, pip install a dep, etc.). It never deletes anything and is safe
# to re-run.
#
# Models:
#   1. LivePortrait  (KwaiVGI/LivePortrait)         webcam -> AI face   [installed]
#   2. MuseTalk      (TMElyralab/MuseTalk)          mouth sync to TTS   [optional]
#   3. Wav2Lip       (fallback mouth sync)                              [installed]
#   4. edge-tts                                     AI voice
#   5. MediaPipe                                    mouth bbox + bg seg
# =============================================================================

import os
import sys
import importlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_DIR)

LIVEPORTRAIT_DIR = os.path.join(PARENT_DIR, "LivePortrait")
MUSETALK_DIR = os.path.join(PARENT_DIR, "MuseTalk")
MUSETALK_IN_PROJECT = os.path.join(PROJECT_DIR, "MuseTalk")
WAV2LIP_CKPT = os.path.join(PROJECT_DIR, "ai-face", "Wav2Lip",
                            "checkpoints", "wav2lip_gan.pth")

OK = "[ READY ]"
MISS = "[ TODO  ]"
WARN = "[ NOTE  ]"

_results = []     # (status, line) tuples for the final summary
_todos = []       # manual instructions to print at the end


def _have(modname):
    try:
        importlib.import_module(modname)
        return True
    except Exception:
        return False


def _report(status, line):
    _results.append((status, line))
    print(f"{status} {line}")


# -----------------------------------------------------------------------------
# 1. LivePortrait
# -----------------------------------------------------------------------------
def check_liveportrait():
    print("\n--- 1. LivePortrait (webcam -> AI character face) ---")
    repo_ok = os.path.isdir(os.path.join(LIVEPORTRAIT_DIR, "src"))
    weights = os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights", "liveportrait",
                           "base_models")
    weights_ok = os.path.isdir(weights) and len(os.listdir(weights)) > 0

    if repo_ok and weights_ok:
        _report(OK, f"LivePortrait repo + weights present at {LIVEPORTRAIT_DIR}")
        return
    if not repo_ok:
        _report(MISS, "LivePortrait repo not found.")
        _todos.append(
            "LivePortrait repo:\n"
            f"    git clone https://github.com/KwaiVGI/LivePortrait \"{LIVEPORTRAIT_DIR}\"")
    if not weights_ok:
        _report(MISS, "LivePortrait weights missing — downloading via huggingface_hub...")
        _try_snapshot("KwaiVGI/LivePortrait",
                      os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights"),
                      "LivePortrait weights (KwaiVGI/LivePortrait)")


# -----------------------------------------------------------------------------
# 2. MuseTalk
# -----------------------------------------------------------------------------
def check_musetalk():
    print("\n--- 2. MuseTalk (mouth sync to AI voice) ---")
    mt_dir = MUSETALK_DIR if os.path.isdir(MUSETALK_DIR) else MUSETALK_IN_PROJECT
    repo_ok = os.path.isdir(os.path.join(mt_dir, "musetalk"))

    # python deps MuseTalk needs beyond the base set
    deps = {"diffusers": _have("diffusers"), "transformers": _have("transformers"),
            "einops": _have("einops"), "omegaconf": _have("omegaconf")}
    missing_deps = [d for d, ok in deps.items() if not ok]

    weights_dir = os.path.join(mt_dir, "models")
    musetalk_bin = os.path.join(weights_dir, "musetalk", "pytorch_model.bin")
    vae_ok = os.path.isdir(os.path.join(weights_dir, "sd-vae-ft-mse"))
    unet_ok = os.path.exists(musetalk_bin)
    whisper_ok = os.path.isdir(os.path.join(weights_dir, "whisper"))

    if repo_ok and unet_ok and vae_ok and whisper_ok and not missing_deps:
        _report(OK, f"MuseTalk repo + weights + deps present at {mt_dir}")
        return

    _report(WARN, "MuseTalk not fully installed — the avatar runs NOW using the "
                  "Wav2Lip fallback for mouth sync. Install MuseTalk to upgrade.")

    if not repo_ok:
        _todos.append(
            "MuseTalk repo:\n"
            f"    git clone https://github.com/TMElyralab/MuseTalk \"{MUSETALK_DIR}\"")
    if missing_deps:
        _todos.append("MuseTalk python deps:\n"
                      "    pip install " + " ".join(missing_deps))
    if not (unet_ok and vae_ok and whisper_ok):
        # Try to grab the MuseTalk model repo automatically into <repo>/models.
        if repo_ok:
            _report(MISS, "MuseTalk weights missing — downloading via huggingface_hub...")
            _try_snapshot("TMElyralab/MuseTalk", weights_dir,
                          "MuseTalk weights (TMElyralab/MuseTalk)")
        else:
            _todos.append(
                "MuseTalk weights (after cloning the repo, into MuseTalk/models/):\n"
                "    - UNet + config : https://huggingface.co/TMElyralab/MuseTalk "
                "(musetalk/musetalk.json + musetalk/pytorch_model.bin)\n"
                "    - VAE           : https://huggingface.co/stabilityai/sd-vae-ft-mse "
                "-> models/sd-vae-ft-mse/\n"
                "    - Whisper tiny  : https://huggingface.co/openai/whisper-tiny "
                "-> models/whisper/\n"
                "    - DWPose        : https://huggingface.co/yzd-v/DWPose "
                "(dw-ll_ucoco_384.pth) -> models/dwpose/\n"
                "    - Face parse    : 79999_iter.pth + resnet18-5c106cde.pth "
                "-> models/face-parse-bisent/\n"
                "    (MuseTalk's own download_weights.sh / .bat fetches all of these.)")


# -----------------------------------------------------------------------------
# 3. Wav2Lip (fallback)
# -----------------------------------------------------------------------------
def check_wav2lip():
    print("\n--- 3. Wav2Lip (fallback mouth sync) ---")
    if os.path.exists(WAV2LIP_CKPT):
        _report(OK, "Wav2Lip checkpoint present (fallback mouth sync ready).")
    else:
        _report(MISS, "Wav2Lip checkpoint missing — fallback mouth sync disabled.")
        _todos.append(
            "Wav2Lip checkpoint (fallback lip-sync):\n"
            "    Download wav2lip_gan.pth and place at:\n"
            f"    {WAV2LIP_CKPT}\n"
            "    Mirror: https://huggingface.co/numz/wav2lip_studio "
            "(or the original Wav2Lip release).")


# -----------------------------------------------------------------------------
# 4 + 5. Python packages (edge-tts, mediapipe, etc.)
# -----------------------------------------------------------------------------
def check_packages():
    print("\n--- 4/5. Python packages ---")
    core = {
        "edge_tts": "edge-tts",
        "cv2": "opencv-python",
        "pyvirtualcam": "pyvirtualcam",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "torch": "torch (cu128 build for RTX 50-series)",
        "huggingface_hub": "huggingface_hub",
        "sounddevice": "sounddevice",
        "librosa": "librosa",
        "soundfile": "soundfile",
    }
    missing = []
    for mod, pkg in core.items():
        if _have(mod):
            _report(OK, f"{pkg}")
        else:
            _report(MISS, f"{pkg}  (import '{mod}' failed)")
            missing.append(pkg.split(" ")[0])

    # ffmpeg is needed to decode edge-tts MP3 -> PCM
    if _ffmpeg_on_path():
        _report(OK, "ffmpeg on PATH")
    else:
        _report(MISS, "ffmpeg not found on PATH")
        _todos.append("ffmpeg (decodes the AI voice audio):\n"
                      "    winget install Gyan.FFmpeg   (then restart the terminal)")

    if _have("torch"):
        try:
            import torch
            if torch.cuda.is_available():
                _report(OK, f"CUDA available — {torch.cuda.get_device_name(0)}")
            else:
                _report(WARN, "torch installed but CUDA NOT available — will be slow. "
                              "Install the cu128 build for RTX 50-series.")
        except Exception:
            pass

    if missing:
        _todos.append("Missing python packages:\n    pip install " + " ".join(missing))


def _ffmpeg_on_path():
    import shutil
    import glob
    if shutil.which("ffmpeg"):
        return True
    local = os.environ.get("LOCALAPPDATA", "")
    winget = os.path.join(local, "Microsoft", "WinGet", "Packages")
    for pat in (os.path.join(winget, "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe"),
                os.path.join(winget, "*FFmpeg*", "*", "bin", "ffmpeg.exe")):
        if glob.glob(pat):
            return True
    return False


# -----------------------------------------------------------------------------
# huggingface_hub download helper
# -----------------------------------------------------------------------------
def _try_snapshot(repo_id, local_dir, label):
    """Download a HF repo snapshot into local_dir. Prints a manual hint on failure."""
    if not _have("huggingface_hub"):
        _todos.append(f"{label}:\n    pip install huggingface_hub, then re-run this script.")
        return
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        print(f"        downloading {repo_id} -> {local_dir} (this can take a while)...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir,
                          local_dir_use_symlinks=False)
        _report(OK, f"{label} downloaded.")
    except Exception as exc:
        _report(MISS, f"{label} auto-download failed ({exc}).")
        _todos.append(
            f"{label}:\n"
            f"    python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{repo_id}', local_dir=r'{local_dir}', "
            f"local_dir_use_symlinks=False)\"\n"
            f"    or browse https://huggingface.co/{repo_id}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" REALTIME AVATAR — MODEL SETUP / READINESS CHECK")
    print("=" * 60)

    check_packages()
    check_liveportrait()
    check_musetalk()
    check_wav2lip()

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    ready = sum(1 for s, _ in _results if s == OK)
    todo = sum(1 for s, _ in _results if s == MISS)
    note = sum(1 for s, _ in _results if s == WARN)
    print(f"  {ready} ready | {todo} to do | {note} notes")

    can_run = (os.path.isdir(os.path.join(LIVEPORTRAIT_DIR, "src"))
               and (_have("pyvirtualcam") and _have("cv2") and _have("edge_tts")))
    if can_run:
        print("\n  You can run the avatar now:")
        print("     python realtime_avatar.py     (terminal 1)")
        print("     python control_gui.py         (terminal 2)")
        if not (os.path.isdir(MUSETALK_DIR) or os.path.isdir(MUSETALK_IN_PROJECT)):
            print("  Mouth sync uses the Wav2Lip fallback until MuseTalk is installed.")
    else:
        print("\n  Finish the TODO items below before running realtime_avatar.py.")

    if _todos:
        print("\n" + "-" * 60)
        print(" MANUAL STEPS")
        print("-" * 60)
        for i, t in enumerate(_todos, 1):
            print(f"\n  {i}. {t}")
    print()


if __name__ == "__main__":
    main()
