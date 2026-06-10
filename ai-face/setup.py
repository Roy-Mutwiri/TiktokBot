# =============================================================================
# setup.py
# -----------------------------------------------------------------------------
# Auto-installer / environment checker for the AI talking-face system.
#
# Run:  python setup.py
#
# It will:
#   - verify Python >= 3.8
#   - pip install the required packages
#   - check that ffmpeg is on PATH
#   - check the Wav2Lip clone, checkpoint, and face-detection model
#   - check character.jpg
#   - print a final readiness checklist
# =============================================================================

import os
import sys
import shutil
import subprocess

# Force UTF-8 stdout so status glyphs ([*] [✓] [!]) don't crash the Windows
# console, whose default code page (cp1252) can't encode them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PIP_PACKAGES = ["edge-tts", "opencv-python", "pyvirtualcam", "numpy",
                "librosa", "numba", "scipy", "tqdm", "gfpgan"]

GFPGAN_MODEL = os.path.join(PROJECT_ROOT, "models", "GFPGANv1.4.pth")

WAV2LIP_PATH = os.path.join(PROJECT_ROOT, "Wav2Lip")
CHECKPOINT = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip_gan.pth")
S3FD_MODEL = os.path.join(WAV2LIP_PATH, "face_detection", "detection", "sfd",
                          "s3fd.pth")
S3FD_MODEL_ALT = os.path.join(WAV2LIP_PATH, "face_detection", "detection", "sfd",
                              "s3fd-619a316812.pth")
CHARACTER_IMAGE = os.path.join(PROJECT_ROOT, "character.jpg")

MIN_PYTHON = (3, 8)


# -----------------------------------------------------------------------------
# INDIVIDUAL CHECKS
# -----------------------------------------------------------------------------
def check_python_version():
    """Verify the running Python is >= MIN_PYTHON. Returns bool."""
    print("[*] Checking Python version...")
    if sys.version_info < MIN_PYTHON:
        print(f"[!] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
              f"found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"[✓] Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True


def install_pip_packages():
    """pip install all required packages into the current environment.

    Returns True if the install command succeeded.
    """
    print("[*] Installing pip packages:", ", ".join(PIP_PACKAGES))
    command = [sys.executable, "-m", "pip", "install"] + PIP_PACKAGES
    try:
        result = subprocess.run(command, check=False)
    except Exception as exc:
        print("[!] pip install failed to run:", exc)
        return False

    if result.returncode != 0:
        print("[!] pip install returned a non-zero exit code.")
        return False
    print("[✓] pip packages installed.")
    return True


def check_ffmpeg():
    """Check whether ffmpeg is available on PATH. Returns bool."""
    print("[*] Checking ffmpeg...")
    if shutil.which("ffmpeg"):
        print("[✓] ffmpeg found on PATH.")
        return True
    print("[!] ffmpeg NOT found on PATH.")
    print("    Install it with one of:")
    print("      winget install ffmpeg")
    print("      choco install ffmpeg")
    print("    Then restart your terminal so PATH refreshes.")
    return False


def check_wav2lip():
    """Check the Wav2Lip clone exists. Returns bool."""
    print("[*] Checking Wav2Lip repository...")
    if os.path.isdir(WAV2LIP_PATH) and os.path.exists(
            os.path.join(WAV2LIP_PATH, "inference.py")):
        print("[✓] Wav2Lip repository found.")
        return True
    print("[!] Wav2Lip not found. Clone it into the project folder:")
    print("      git clone https://github.com/Rudrabha/Wav2Lip")
    return False


def check_checkpoint():
    """Check the wav2lip_gan.pth checkpoint exists. Returns bool."""
    print("[*] Checking Wav2Lip checkpoint...")
    if os.path.exists(CHECKPOINT):
        print("[✓] wav2lip_gan.pth checkpoint found.")
        return True
    print("[!] Checkpoint missing:", CHECKPOINT)
    print("    Download wav2lip_gan.pth from the Wav2Lip model release and place it in:")
    print("      Wav2Lip/checkpoints/wav2lip_gan.pth")
    print("    Model list: https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
    return False


def check_s3fd():
    """Check the s3fd face-detection model exists. Returns bool."""
    print("[*] Checking face-detection model (s3fd)...")
    if os.path.exists(S3FD_MODEL) or os.path.exists(S3FD_MODEL_ALT):
        print("[✓] s3fd face-detection model found.")
        return True
    print("[!] s3fd model missing.")
    print("    Download s3fd-619a316812.pth and place it (renamed to s3fd.pth) in:")
    print("      Wav2Lip/face_detection/detection/sfd/s3fd.pth")
    return False


def check_character_image():
    """Check character.jpg exists. Returns bool."""
    print("[*] Checking character.jpg...")
    if os.path.exists(CHARACTER_IMAGE):
        print("[✓] character.jpg found.")
        return True
    print("[!] character.jpg missing.")
    print("    Add a 512x512 front-facing, neutral-expression face image named")
    print("    character.jpg in the project root.")
    return False


def check_gfpgan_model():
    """Check the GFPGAN face-restoration model exists. Returns bool."""
    print("[*] Checking GFPGAN model (clarity / human lips)...")
    if os.path.exists(GFPGAN_MODEL):
        print("[✓] GFPGANv1.4.pth found.")
        return True
    print("[!] GFPGANv1.4.pth missing (needed when ENHANCE=True).")
    print("    Download into models/ :")
    print("    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
    return False


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    """Run all checks and print a final readiness checklist."""
    print("=" * 64)
    print(" AI Talking Face - Setup & Environment Check")
    print("=" * 64)

    results = {}

    results["Python >= 3.8"] = check_python_version()
    if not results["Python >= 3.8"]:
        print("[!] Aborting: incompatible Python version.")
        sys.exit(1)

    results["pip packages"] = install_pip_packages()
    results["ffmpeg"] = check_ffmpeg()
    results["Wav2Lip repo"] = check_wav2lip()
    results["wav2lip_gan.pth"] = check_checkpoint()
    results["s3fd model"] = check_s3fd()
    results["GFPGAN model"] = check_gfpgan_model()
    results["character.jpg"] = check_character_image()

    # ---- Final checklist ----------------------------------------------------
    print()
    print("=" * 64)
    print(" READINESS CHECKLIST")
    print("=" * 64)
    for name, ok in results.items():
        mark = "[✓]" if ok else "[!]"
        status = "ready" if ok else "NEEDS SETUP"
        print(f" {mark} {name:<22} {status}")
    print("=" * 64)

    if all(results.values()):
        print("[✓] Everything is ready! Start with:")
        print("      Terminal 1:  python realtime_face.py")
        print("      Terminal 2:  python gui_panel.py")
    else:
        print("[*] Resolve the items marked NEEDS SETUP above, then re-run setup.py.")
        print("    See README.md for detailed download links and instructions.")


if __name__ == "__main__":
    main()
