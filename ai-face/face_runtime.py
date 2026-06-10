# =============================================================================
# face_runtime.py
# -----------------------------------------------------------------------------
# Persistent, in-process inference runtime for the talking face.
#
# Instead of spawning a fresh Wav2Lip subprocess per line (which reloads the
# model from disk every time, ~10s, and round-trips video through files), this
# loads everything ONCE and keeps it resident in VRAM:
#
#   - s3fd face detector  (Wav2Lip/face_detection)
#   - Wav2Lip generator   (wav2lip_gan.pth)
#
# Because the character image is static, the face is detected a SINGLE time and
# the crop is reused for every frame. A line then becomes:
#     wav -> mel -> batched GPU inference -> paste mouth back -> frames
# entirely in memory. This drops per-line latency from ~30s to a few seconds.
# =============================================================================

import os
import sys

# Force UTF-8 stdout so status glyphs don't crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_PATH = os.path.join(PROJECT_ROOT, "Wav2Lip")
CHECKPOINT = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip_gan.pth")
CHARACTER_IMAGE = os.path.join(PROJECT_ROOT, "character.jpg")

IMG_SIZE = 96                  # Wav2Lip face crop size
MEL_STEP_SIZE = 16             # mel window per frame
FPS = 25                       # frames per second of generated video
PADS = (0, 10, 0, 0)           # (top, bottom, left, right) padding around face box
WAV2LIP_BATCH = 64             # FIXED frames per GPU batch (padded) — must stay
                               # constant so cuDNN autotunes the conv shape once.
SAMPLE_RATE = 16000

# Make the Wav2Lip package importable (audio.py, models/, face_detection/).
if WAV2LIP_PATH not in sys.path:
    sys.path.insert(0, WAV2LIP_PATH)


# =============================================================================
# RUNTIME
# =============================================================================
class FaceRuntime:
    """Holds the resident models and renders lip-synced frames in memory."""

    def __init__(self, checkpoint=CHECKPOINT, character_image=CHARACTER_IMAGE,
                 device=None):
        """Load the detector + Wav2Lip model and lock onto the character face.

        Raises on missing models, missing image, or no detectable face.
        """
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint}")

        # Import Wav2Lip pieces (after sys.path insert).
        import face_detection
        from models import Wav2Lip

        print(f"[*] Runtime: loading Wav2Lip + s3fd on {self.device} (one-time)...")
        self._Wav2Lip = Wav2Lip
        self.model = self._load_model(checkpoint)
        self.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=self.device)

        self.full_img = None
        self.coords = None
        self.face_input6 = None     # precomputed (96,96,6) masked+full / 255
        self.set_character(character_image)
        print("[✓] Runtime: ready (models resident, face locked).")

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    def _load_model(self, path):
        """Load Wav2Lip weights into an eval-mode model on the target device."""
        torch = self._torch
        model = self._Wav2Lip()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean)
        return model.to(self.device).eval()

    # -------------------------------------------------------------------------
    # CHARACTER FACE (detected once, reused for every frame)
    # -------------------------------------------------------------------------
    def set_character(self, image_path):
        """Detect the face in the character image once and cache the crop."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read character image: {image_path}")
        self.full_img = img
        h, w = img.shape[:2]

        preds = self.detector.get_detections_for_batch(np.array([img]))
        rect = preds[0]
        if rect is None:
            raise ValueError("No face detected in character image. Use a clear, "
                             "front-facing 512x512 photo.")

        pady1, pady2, padx1, padx2 = PADS
        y1 = max(0, rect[1] - pady1)
        y2 = min(h, rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(w, rect[2] + padx2)
        self.coords = (y1, y2, x1, x2)

        face = cv2.resize(img[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        masked = face.copy()
        masked[IMG_SIZE // 2:, :] = 0       # mask lower half (mouth region)
        # (96,96,6): [masked | full] / 255, channel-last, ready to transpose.
        self.face_input6 = (np.concatenate((masked, face), axis=2)
                            .astype(np.float32) / 255.0)

    # -------------------------------------------------------------------------
    # MEL CHUNKING
    # -------------------------------------------------------------------------
    def _mel_chunks(self, wav_path):
        """Load a wav and split its mel-spectrogram into per-frame windows."""
        import audio as w2l_audio
        wav = w2l_audio.load_wav(wav_path, SAMPLE_RATE)
        mel = w2l_audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN (bad audio).")

        chunks = []
        idx_multiplier = 80.0 / FPS
        i = 0
        while True:
            start = int(i * idx_multiplier)
            if start + MEL_STEP_SIZE > len(mel[0]):
                chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
                break
            chunks.append(mel[:, start:start + MEL_STEP_SIZE])
            i += 1
        return chunks

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    def infer(self, wav_path):
        """Render the lip-synced frames for a wav, fully in memory.

        Returns a list of full-resolution BGR frames (the character image with
        the generated mouth pasted in). The mouth region is still Wav2Lip's
        96x96 output; run a face enhancer afterwards for HD clarity.
        """
        torch = self._torch
        mel_chunks = self._mel_chunks(wav_path)
        n = len(mel_chunks)
        if n == 0:
            return []

        y1, y2, x1, x2 = self.coords
        bw, bh = x2 - x1, y2 - y1

        # The face input (6-channel) is identical for every frame (static head),
        # so build the transposed tensor template once.
        img_chw = np.transpose(self.face_input6, (2, 0, 1))   # (6,96,96)

        # The img tensor is identical for a full fixed-size batch (static head).
        img_arr = np.repeat(img_chw[np.newaxis, ...], WAV2LIP_BATCH, axis=0)  # (B,6,96,96)
        img_t = torch.from_numpy(img_arr).to(self.device)

        frames = []
        with torch.no_grad():
            for b in range(0, n, WAV2LIP_BATCH):
                batch = mel_chunks[b:b + WAV2LIP_BATCH]
                m = len(batch)
                # Pad the last batch to the FIXED batch size so the conv input
                # shape never changes — otherwise cuDNN re-autotunes (seconds)
                # on every clip of a different length.
                if m < WAV2LIP_BATCH:
                    batch = batch + [batch[-1]] * (WAV2LIP_BATCH - m)

                mel_arr = np.asarray(batch, dtype=np.float32)[:, np.newaxis, :, :]  # (B,1,80,16)
                mel_t = torch.from_numpy(mel_arr).to(self.device)

                pred = self.model(mel_t, img_t)                       # (B,3,96,96)
                pred = (pred[:m].cpu().numpy().transpose(0, 2, 3, 1) * 255.0)  # keep first m

                for p in pred:
                    mouth = cv2.resize(p.astype(np.uint8), (bw, bh))
                    frame = self.full_img.copy()
                    frame[y1:y2, x1:x2] = mouth
                    frames.append(frame)
        return frames


# -----------------------------------------------------------------------------
# STANDALONE TEST  (python face_runtime.py some_speech.wav)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    if len(sys.argv) < 2:
        print("Usage: python face_runtime.py <wav_file>")
        sys.exit(1)

    t0 = time.time()
    rt = FaceRuntime()
    t1 = time.time()
    print(f"[*] Load time: {t1 - t0:.1f}s")

    frames = rt.infer(sys.argv[1])
    t2 = time.time()
    print(f"[✓] Rendered {len(frames)} frames in {t2 - t1:.2f}s "
          f"({(t2 - t1) / max(1, len(frames)) * 1000:.0f} ms/frame)")
    if frames:
        cv2.imwrite("_runtime_test_frame.jpg", frames[len(frames) // 2])
        print("[✓] Wrote middle frame -> _runtime_test_frame.jpg")
