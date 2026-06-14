# =============================================================================
# engines/wav2lip_engine.py
# -----------------------------------------------------------------------------
# Real-time, frame-by-frame Wav2Lip lip-sync that runs ON TOP of the animated
# face. It keeps a rolling audio buffer fed by the TTS engine and, while the
# avatar is speaking, syncs ONLY the mouth region of each frame — preserving the
# head movement and eyes produced upstream by LivePortrait.
# =============================================================================

import os
import sys
import collections
import threading
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Wav2Lip lives in ai-face/ in this project; allow a few locations.
WAV2LIP_CANDIDATES = [
    os.path.join(PROJECT_DIR, "Wav2Lip"),
    os.path.join(PROJECT_DIR, "ai-face", "Wav2Lip"),
]
CHECKPOINT_NAME = os.path.join("checkpoints", "wav2lip_gan.pth")

SAMPLE_RATE = 16000
FPS = 25
IMG_SIZE = 96
MEL_STEP_SIZE = 16
BLEND_FACTOR = 0.88            # Wav2Lip vs original in the mouth region
SILENCE_THRESHOLD = 8          # frames of no new audio before is_speaking=False
AUDIO_WINDOW = 6400            # samples (~400ms) used to build the mel
MIN_AUDIO = 3200               # need at least this much audio to sync
DETECT_INTERVAL = 6            # re-run face detection every N frames (cache between)
MOUTH_REGION_TOP = 0.45        # blend only below this fraction of the face box
STREAM_TIMEOUT = SILENCE_THRESHOLD / FPS


class Wav2LipEngine:
    """Keeps a Wav2Lip model resident and lip-syncs the mouth each frame."""

    def __init__(self):
        """Load the model + detector, prime buffers, warm up."""
        self.ready = False
        self.fallback = False
        self.is_speaking = False
        self.audio_buffer = collections.deque(maxlen=SAMPLE_RATE * 2)   # 2s
        self._audio_lock = threading.Lock()
        self._timeline_audio = None
        self._timeline_start = 0.0
        self._last_audio_feed = 0.0
        self._silence_frames = 0
        self._face_box = None
        self._frame_counter = 0
        self._err_printed = False

        root = self._find_wav2lip()
        if root is None:
            self.fallback = True
            print("[WAV2LIP] Wav2Lip not found — lip-sync disabled (face still shows).")
            return
        try:
            self._load(root)
            self._warmup()
            self.ready = True
            self._print_gpu()
            print("[WAV2LIP] Ready — real-time mouth sync enabled.")
        except Exception as exc:
            self.fallback = True
            print(f"[WAV2LIP] init failed ({exc}) — lip-sync disabled.")

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Report lip-sync readiness. Returns (ok, message)."""
        if self.fallback:
            return True, "FALLBACK — no lip-sync (Wav2Lip/checkpoint missing)."
        return True, "Wav2Lip lip-sync active."

    # -------------------------------------------------------------------------
    def _find_wav2lip(self):
        """Return the Wav2Lip root that has a checkpoint, else None."""
        for root in WAV2LIP_CANDIDATES:
            if os.path.exists(os.path.join(root, CHECKPOINT_NAME)) and \
               os.path.exists(os.path.join(root, "models")):
                return root
        return None

    def _load(self, root):
        """Load the Wav2Lip generator, audio module, and s3fd detector."""
        import torch
        if root not in sys.path:
            sys.path.insert(0, root)
        import audio as w2l_audio
        from models import Wav2Lip
        import face_detection

        self._torch = torch
        self._audio = w2l_audio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True

        ckpt_path = os.path.join(root, CHECKPOINT_NAME)
        model = Wav2Lip()
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

        self.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=self.device)

    # -------------------------------------------------------------------------
    # AUDIO FEED
    # -------------------------------------------------------------------------
    def feed_audio(self, audio_chunk):
        """Append live/streaming 16 kHz audio to the rolling buffer."""
        try:
            chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
            with self._audio_lock:
                self._timeline_audio = None
                self.audio_buffer.extend(chunk)
                self._last_audio_feed = time.monotonic()
            self.is_speaking = True
            self._silence_frames = 0
        except Exception as exc:
            print(f"[WAV2LIP] feed_audio error: {exc}")

    def start_audio(self, audio, start_time=None):
        """Start a clocked utterance.

        Generated speech is supplied in full so each rendered frame can select
        the mel window for its actual wall-clock position instead of repeatedly
        using whichever chunk most recently reached an append-only buffer.
        """
        pcm = np.asarray(audio, dtype=np.float32).flatten().copy()
        with self._audio_lock:
            self.audio_buffer.clear()
            self._timeline_audio = pcm
            self._timeline_start = float(start_time or time.monotonic())
            self._last_audio_feed = self._timeline_start
        self.is_speaking = bool(len(pcm))
        self._silence_frames = 0

    def end_audio(self):
        """Finish the current utterance and discard stale phonetic context."""
        with self._audio_lock:
            self._timeline_audio = None
            self.audio_buffer.clear()
        self.is_speaking = False
        self._silence_frames = 0

    # -------------------------------------------------------------------------
    # FRAME PROCESSING
    # -------------------------------------------------------------------------
    def process_frame(self, face_frame, raw_output=False):
        """Return face_frame with the mouth lip-synced to recent audio."""
        if self.fallback or not self.ready:
            return face_frame

        if not self._audio_active():
            return face_frame

        try:
            box = self._get_face_box(face_frame)
            if box is None:
                return face_frame
            x1, y1, x2, y2 = box
            crop = face_frame[y1:y2, x1:x2]
            if crop.size == 0:
                return face_frame

            mel = self._build_mel()
            if mel is None:
                return face_frame

            synced = self._run_wav2lip(crop, mel)              # uint8, crop size
            if raw_output:
                result = face_frame.copy()
                result[y1:y2, x1:x2] = synced
                return result
            return self._blend_mouth(face_frame, synced, box)
        except Exception as exc:
            if not self._err_printed:
                print(f"[WAV2LIP] frame error ({exc}) — passing face through.")
                self._err_printed = True
            return face_frame

    def _get_face_box(self, frame):
        """Detect the face box, cached for DETECT_INTERVAL frames."""
        self._frame_counter += 1
        if self._face_box is not None and (self._frame_counter % DETECT_INTERVAL) != 0:
            return self._face_box
        preds = self.detector.get_detections_for_batch(np.array([frame]))
        rect = preds[0]
        if rect is None:
            return self._face_box                # keep last good box if any
        h, w = frame.shape[:2]
        x1 = max(0, int(rect[0])); y1 = max(0, int(rect[1]))
        x2 = min(w, int(rect[2])); y2 = min(h, int(rect[3]) + 10)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return self._face_box
        self._face_box = (x1, y1, x2, y2)
        return self._face_box

    def _audio_active(self):
        """Update and return whether timeline/stream audio is still current."""
        now = time.monotonic()
        with self._audio_lock:
            timeline = self._timeline_audio
            start = self._timeline_start
            last_feed = self._last_audio_feed
            buffered = len(self.audio_buffer)
        if timeline is not None:
            active = 0.0 <= now - start < (len(timeline) / SAMPLE_RATE)
        else:
            active = buffered >= MIN_AUDIO and (now - last_feed) <= STREAM_TIMEOUT
        self.is_speaking = bool(active)
        return self.is_speaking

    def _current_audio_window(self, now=None):
        """Return the audio ending at the current utterance playback cursor."""
        now = time.monotonic() if now is None else float(now)
        with self._audio_lock:
            timeline = self._timeline_audio
            start = self._timeline_start
            if timeline is None:
                return np.array(list(self.audio_buffer)[-AUDIO_WINDOW:], dtype=np.float32)
            end = int(max(0.0, now - start) * SAMPLE_RATE)
            end = min(len(timeline), end)
            begin = max(0, end - AUDIO_WINDOW)
            window = timeline[begin:end].copy()
        if len(window) < AUDIO_WINDOW:
            window = np.pad(window, (AUDIO_WINDOW - len(window), 0))
        return window

    def _build_mel(self, now=None):
        """Build the mel chunk at the current clocked audio position."""
        window = self._current_audio_window(now)
        if len(window) < MIN_AUDIO:
            return None
        mel = self._audio.melspectrogram(window)              # (80, T)
        if np.isnan(mel).any() or mel.shape[1] < MEL_STEP_SIZE:
            return None
        chunk = mel[:, -MEL_STEP_SIZE:]                       # (80,16)
        chunk = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,80,16)
        return self._torch.from_numpy(chunk).to(self.device)

    def _run_wav2lip(self, crop, mel_t):
        """Run the generator on a face crop; return a synced crop (uint8)."""
        torch = self._torch
        face = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        masked = face.copy()
        masked[IMG_SIZE // 2:, :] = 0
        img = np.concatenate((masked, face), axis=2).astype(np.float32) / 255.0  # (96,96,6)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))[np.newaxis]).to(self.device)
        with torch.no_grad():
            pred = self.model(mel_t, img_t)                  # (1,3,96,96)
        pred = (pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        # LANCZOS4 (not the default soft LINEAR) to upscale the 96px mouth back —
        # crisper lips/teeth at the source = less of the "blur taking over".
        return cv2.resize(pred, (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_LANCZOS4)

    def _blend_mouth(self, frame, synced_crop, box):
        """Blend only the lower (mouth) portion of the synced crop back in."""
        x1, y1, x2, y2 = box
        h = y2 - y1
        result = frame.copy()
        region = result[y1:y2, x1:x2].astype(np.float32)
        synced = synced_crop.astype(np.float32)

        # Vertical mask: 0 in the upper face, ramping to 1 over the mouth area,
        # so LivePortrait's eyes/forehead/head motion are preserved.
        mask = np.zeros((h, 1), dtype=np.float32)
        top = int(h * MOUTH_REGION_TOP)
        ramp = max(1, int(h * 0.12))
        mask[top + ramp:, 0] = 1.0
        for i in range(ramp):
            mask[top + i, 0] = i / ramp
        mask = mask[:, :, None] * BLEND_FACTOR

        blended = synced * mask + region * (1.0 - mask)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    # -------------------------------------------------------------------------
    def _warmup(self):
        """Warm s3fd + the generator on a REAL face so cuDNN autotunes.

        A blank frame has no detectable face, so process_frame would return
        early and the model would never run — leaving the autotuning cost to hit
        the first real speaking frames (hundreds of ms). We use the character
        image (or a face-like blob) so detection + inference actually execute.
        """
        face = None
        for p in (os.path.join(PROJECT_DIR, "ai-face", "character.jpg"),
                  os.path.join(PROJECT_DIR, "character.jpg")):
            if os.path.exists(p):
                face = cv2.imread(p)
                break
        if face is None:
            face = np.full((512, 512, 3), 110, np.uint8)
            cv2.circle(face, (256, 250), 130, (175, 165, 155), -1)
            cv2.circle(face, (215, 220), 14, (90, 90, 90), -1)
            cv2.circle(face, (300, 220), 14, (90, 90, 90), -1)
        face = cv2.resize(face, (512, 512))

        # Warm the s3fd face detector DIRECTLY — its first GPU forward costs
        # ~12s (cuDNN autotune) and is the single biggest cold-start stall.
        try:
            for _ in range(2):
                self.detector.get_detections_for_batch(np.array([face]))
        except Exception:
            pass

        # Exercise the two slow-on-first-call paths DIRECTLY so their one-time
        # cost is paid here at startup, not on the first real speaking frame:
        #   - librosa.melspectrogram triggers numba JIT (~tens of seconds once)
        #   - the generator's first CUDA forward triggers cuDNN autotuning
        # We call them explicitly (not via process_frame) so they run even if
        # s3fd doesn't detect a face in the warm-up image.
        self.feed_audio((np.random.randn(AUDIO_WINDOW) * 0.1).astype(np.float32))
        mel = None
        for _ in range(3):
            mel = self._build_mel()
            self.feed_audio((np.random.randn(640) * 0.1).astype(np.float32))
        crop = (np.random.rand(120, 120, 3) * 255).astype(np.uint8)
        if mel is not None:
            for _ in range(3):
                self._run_wav2lip(crop, mel)
        # Also run the full path once so s3fd + blend autotune too.
        for _ in range(2):
            self.process_frame(face)

        self.audio_buffer.clear()
        self._timeline_audio = None
        self.is_speaking = False
        self._face_box = None
        self._frame_counter = 0

    def _print_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[WAV2LIP] GPU memory: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
        except Exception:
            pass


if __name__ == "__main__":
    eng = Wav2LipEngine()
    print("[WAV2LIP] startup_check:", eng.startup_check())
