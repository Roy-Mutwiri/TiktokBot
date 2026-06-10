# =============================================================================
# tts_engine.py
# -----------------------------------------------------------------------------
# Text-to-Speech engine using Microsoft edge-tts (free, no API key required).
#
# Pipeline:
#   text  ->  edge-tts (async)  ->  speech_output.mp3
#         ->  ffmpeg  ->  speech_output.wav (16000 Hz, mono, 16-bit PCM)
#
# Wav2Lip expects a 16 kHz mono WAV, which is exactly what we produce here.
# =============================================================================

import os
import sys
import asyncio
import subprocess

# Force UTF-8 stdout so status glyphs ([*] [✓] [!]) don't crash the Windows
# console, whose default code page (cp1252) can't encode them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Try to import edge_tts. If it's missing we fail loudly with instructions.
try:
    import edge_tts
except ImportError:
    print("[!] edge-tts is not installed. Run:  pip install edge-tts")
    raise

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS  (change these to customize behaviour)
# -----------------------------------------------------------------------------
VOICE = "en-US-ChristopherNeural"        # edge-tts voice (run `edge-tts --list-voices` to see all)
RATE = "+0%"                             # speaking rate, e.g. "+10%" for faster
VOLUME = "+0%"                           # volume adjustment, e.g. "+10%"
PITCH = "+0Hz"                           # pitch adjustment, e.g. "+5Hz"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MP3_PATH = os.path.join(PROJECT_ROOT, "speech_output.mp3")
WAV_PATH = os.path.join(PROJECT_ROOT, "speech_output.wav")

SAMPLE_RATE = 16000                      # Wav2Lip requires 16 kHz audio
CHANNELS = 1                             # mono


# -----------------------------------------------------------------------------
# CORE ASYNC SYNTHESIS
# -----------------------------------------------------------------------------
async def _synthesize_async(text, mp3_path):
    """Generate speech MP3 from text using edge-tts (asynchronous)."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        volume=VOLUME,
        pitch=PITCH,
    )
    await communicate.save(mp3_path)


# -----------------------------------------------------------------------------
# MP3 -> WAV CONVERSION
# -----------------------------------------------------------------------------
def _convert_mp3_to_wav(mp3_path, wav_path):
    """Convert an MP3 file to a 16 kHz mono WAV using ffmpeg.

    Raises RuntimeError if ffmpeg fails or is not installed.
    """
    command = [
        "ffmpeg",
        "-y",                       # overwrite output without asking
        "-i", mp3_path,             # input mp3
        "-ar", str(SAMPLE_RATE),    # resample to 16 kHz
        "-ac", str(CHANNELS),       # downmix to mono
        "-acodec", "pcm_s16le",     # 16-bit PCM
        wav_path,                   # output wav
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install it with `winget install ffmpeg` "
            "or `choco install ffmpeg`, then restart your terminal."
        )

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore")
        raise RuntimeError("ffmpeg conversion failed:\n" + stderr)


# -----------------------------------------------------------------------------
# PUBLIC SYNCHRONOUS API
# -----------------------------------------------------------------------------
def speak(text, wav_path=WAV_PATH):
    """Convert text to a 16 kHz mono WAV file (synchronous wrapper).

    Args:
        text:     The text to synthesize.
        wav_path: Destination WAV path (defaults to speech_output.wav).

    Returns:
        The path to the generated WAV file.

    Raises:
        ValueError:   If text is empty.
        RuntimeError: If synthesis or conversion fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    print("[*] TTS: synthesizing speech...")

    # Run the async edge-tts call inside a fresh event loop. This works
    # whether or not we're called from a thread that already has a loop.
    try:
        asyncio.run(_synthesize_async(text, MP3_PATH))
    except RuntimeError:
        # asyncio.run() fails if a loop is already running in this thread.
        # Fall back to a manually managed loop.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_synthesize_async(text, MP3_PATH))
        finally:
            loop.close()

    if not os.path.exists(MP3_PATH) or os.path.getsize(MP3_PATH) == 0:
        raise RuntimeError("edge-tts produced no audio (check internet connection).")

    print("[*] TTS: converting MP3 -> WAV (16kHz mono)...")
    _convert_mp3_to_wav(MP3_PATH, wav_path)

    print("[✓] TTS: audio ready ->", wav_path)
    return wav_path


# -----------------------------------------------------------------------------
# STANDALONE TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "Hello, this is a test of the text to speech engine."
    try:
        speak(sample)
        print("[✓] Done. Play speech_output.wav to verify.")
    except Exception as exc:
        print("[!] TTS test failed:", exc)
        sys.exit(1)
