# =============================================================================
# engines/musetalk_engine.py
# -----------------------------------------------------------------------------
# LAYER 2 of the avatar pipeline: overrides ONLY the mouth region of the
# LivePortrait output so the lips track the TTS audio (the typed text), NOT the
# operator's real mouth. Active only while speaking; silent frames are left to
# LivePortrait untouched.
#
# Two execution modes:
#   * REAL MuseTalk  — UNet latent inpainting on the mouth crop conditioned on
#                      Whisper audio features, decoded by sd-vae-ft-mse. Loaded
#                      only if ./MuseTalk + all weights + (diffusers, transformers,
#                      einops, omegaconf) are present.
#   * FALLBACK       — the proven resident Wav2Lip engine syncs the mouth instead
#                      (already installed in this repo, ~9 ms/frame). The system
#                      is fully functional in this mode; install MuseTalk to
#                      upgrade mouth quality. If even Wav2Lip is missing, the
#                      mouth is passed through unchanged (face still animates).
#
# Public API (identical in both modes):
#   feed_audio(chunk)                      append 16 kHz mono float audio
#   is_speaking                            True while recent audio is buffered
#   process_mouth(lp_face_frame, bbox)     -> synced mouth crop (bbox-sized BGR)
# =============================================================================

import os
import sys
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MUSETALK_CANDIDATES = [
    os.path.join(os.path.dirname(PROJECT_DIR), "MuseTalk"),   # sibling of TiktokBot
    os.path.join(PROJECT_DIR, "MuseTalk"),                    # inside TiktokBot
    os.environ.get("MUSETALK_PATH", ""),
]
FACE_REGION_SIZE = 256          # MuseTalk native mouth crop resolution
BLEND_FACTOR = 0.9              # MuseTalk vs original inside the crop (real path)

SAMPLE_RATE = 16000
AUDIO_WINDOW_MS = 200
AUDIO_WINDOW = int(SAMPLE_RATE * AUDIO_WINDOW_MS / 1000)   # 3200 samples
MIN_AUDIO = 1600                # need >=100ms of audio to drive the mouth
SILENCE_FRAMES = 8              # frames with no fresh audio before is_speaking=False


class MuseTalkEngine:
    """Mouth-region lip-sync to TTS audio (MuseTalk, Wav2Lip fallback)."""

    def __init__(self, character_path=None, use_wav2lip_fallback=True):
        """Try real MuseTalk; otherwise wire the Wav2Lip fallback. Warm up."""
        self.character_path = character_path
        self.real_ready = False
        self.fallback_mode = False
        self.is_speaking = False

        self.audio_buffer = collections.deque(maxlen=SAMPLE_RATE * 2)   # 2 s
        self._silence = 0
        self._err_printed = False
        self._w2l = None          # Wav2Lip fallback engine (if used)

        # --- try the real MuseTalk pipeline first ---
        mt_path = self._find_musetalk()
        if mt_path is not None:
            try:
                self._load_musetalk(mt_path)
                self._warmup_real()
                self.real_ready = True
                self._print_gpu()
                print("[MUSETALK] Ready — UNet mouth inpainting active.")
                return
            except Exception as exc:
                print(f"[MUSETALK] real init failed ({exc}) — using fallback.")

        # --- fallback: resident Wav2Lip mouth sync ---
        self.fallback_mode = True
        if use_wav2lip_fallback:
            try:
                from wav2lip_engine import Wav2LipEngine
                self._w2l = Wav2LipEngine()
                if getattr(self._w2l, "fallback", False):
                    print("[MUSETALK] FALLBACK — Wav2Lip unavailable too; mouth "
                          "passthrough (face still animates, no lip-sync).")
                    self._w2l = None
                else:
                    print("[MUSETALK] FALLBACK — Wav2Lip mouth sync active. "
                          "Install MuseTalk (python setup_models.py) to upgrade.")
            except Exception as exc:
                print(f"[MUSETALK] FALLBACK — Wav2Lip load failed ({exc}); "
                      "mouth passthrough.")
                self._w2l = None
        else:
            print("[MUSETALK] FALLBACK — mouth passthrough (no lip-sync).")

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Report mouth-sync readiness. Returns (ok, message)."""
        if self.real_ready:
            return True, "MuseTalk active — audio-driven mouth inpainting."
        if self._w2l is not None:
            return True, "FALLBACK Wav2Lip — mouth synced (MuseTalk not installed)."
        return True, "FALLBACK passthrough — no lip-sync (no MuseTalk/Wav2Lip)."

    # -------------------------------------------------------------------------
    # AUDIO FEED  (shared by both modes)
    # -------------------------------------------------------------------------
    def feed_audio(self, audio_chunk):
        """Append TTS audio (float32 mono 16 kHz) and mark speaking."""
        try:
            self.audio_buffer.extend(np.asarray(audio_chunk, dtype=np.float32).flatten())
            self.is_speaking = True
            self._silence = 0
            if self._w2l is not None:
                self._w2l.feed_audio(audio_chunk)     # keep fallback buffer in sync
        except Exception as exc:
            print(f"[MUSETALK] feed_audio error: {exc}")

    def _update_speaking(self):
        """Decay is_speaking after SILENCE_FRAMES with no fresh audio. Returns
        True if there is enough buffered audio to drive the mouth this frame."""
        if len(self.audio_buffer) < MIN_AUDIO:
            self._silence += 1
            if self._silence > SILENCE_FRAMES:
                self.is_speaking = False
            return False
        return self.is_speaking

    # -------------------------------------------------------------------------
    # FRAME PROCESSING
    # -------------------------------------------------------------------------
    def process_mouth(self, lp_face_frame, mouth_bbox):
        """Return a mouth crop (bbox-sized BGR) synced to the current audio.

        If not speaking / no audio, returns the unchanged crop so the compositor
        blends LivePortrait's own mouth straight back (a no-op). Never raises.
        """
        x1, y1, x2, y2 = mouth_bbox
        try:
            base_crop = lp_face_frame[y1:y2, x1:x2].copy()
        except Exception:
            return None

        if not self._update_speaking():
            return base_crop

        try:
            if self.real_ready:
                return self._process_real(lp_face_frame, mouth_bbox, base_crop)
            if self._w2l is not None:
                synced_full = self._w2l.process_frame(lp_face_frame)
                return synced_full[y1:y2, x1:x2].copy()
            return base_crop
        except Exception as exc:
            if not self._err_printed:
                print(f"[MUSETALK] frame error ({exc}) — mouth passthrough.")
                self._err_printed = True
            return base_crop

    # -------------------------------------------------------------------------
    # REAL MuseTalk PATH
    # -------------------------------------------------------------------------
    def _process_real(self, lp_face_frame, bbox, base_crop):
        """MuseTalk UNet mouth inpainting conditioned on Whisper audio features."""
        torch = self._torch
        x1, y1, x2, y2 = bbox

        # 1) audio features from the most recent window
        window = np.array(list(self.audio_buffer)[-AUDIO_WINDOW:], dtype=np.float32)
        audio_feat = self._audio_features(window)        # (1, T, C) tensor on device
        if audio_feat is None:
            return base_crop

        # 2) mouth crop -> 256x256, masked lower half (MuseTalk inpaints it)
        crop = cv2.resize(base_crop, (FACE_REGION_SIZE, FACE_REGION_SIZE))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(crop_rgb.transpose(2, 0, 1))[None].to(self.device)
        img = (img * 2.0 - 1.0).to(self._dtype)          # [-1,1] for the VAE

        masked = img.clone()
        masked[:, :, FACE_REGION_SIZE // 2:, :] = 0       # zero lower (mouth) half

        with torch.no_grad():
            # encode both halves to latents, concat on channel dim (MuseTalk input)
            lat_img = self._vae.encode(img).latent_dist.sample() * self._vae_scale
            lat_msk = self._vae.encode(masked).latent_dist.sample() * self._vae_scale
            lat_in = torch.cat([lat_msk, lat_img], dim=1).to(self._dtype)

            ts = torch.tensor([0], device=self.device)
            pred = self._unet(lat_in, ts, encoder_hidden_states=audio_feat).sample
            out = self._vae.decode(pred / self._vae_scale).sample      # (1,3,256,256)

        out = ((out.float()[0].clamp(-1, 1) + 1.0) / 2.0 * 255.0)
        out = out.detach().cpu().numpy().transpose(1, 2, 0)
        out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (x2 - x1, y2 - y1))

        # keep some of the original crop so identity/skin is preserved
        blended = (out.astype(np.float32) * BLEND_FACTOR +
                   base_crop.astype(np.float32) * (1.0 - BLEND_FACTOR))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _audio_features(self, window):
        """Whisper encoder features for the audio window -> conditioning tensor."""
        try:
            torch = self._torch
            feat = self._whisper_feature(window)         # provided by MuseTalk utils
            if feat is None:
                return None
            t = torch.as_tensor(feat, device=self.device).to(self._dtype)
            if t.dim() == 2:
                t = t[None]
            return t
        except Exception:
            return None

    def _find_musetalk(self):
        """Return the first MuseTalk dir that looks complete, else None."""
        for cand in MUSETALK_CANDIDATES:
            if cand and os.path.isdir(cand) and \
               os.path.isdir(os.path.join(cand, "musetalk")):
                return cand
        return None

    def _load_musetalk(self, mt_path):
        """Load UNet + VAE + Whisper feature extractor from a MuseTalk checkout.

        MuseTalk's helper APIs differ across revisions, so this resolves the
        pieces defensively and raises (-> fallback) if anything is missing.
        Importing also requires diffusers/transformers/einops/omegaconf.
        """
        import torch
        # hard dependency check — fail fast into the fallback if absent
        import diffusers          # noqa: F401
        import transformers       # noqa: F401
        import einops             # noqa: F401

        if mt_path not in sys.path:
            sys.path.insert(0, mt_path)

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32

        # MuseTalk ships a load_all_model() helper that returns (audio_proc,
        # vae, unet, pe) in most revisions. Use it when present.
        from musetalk.utils.utils import load_all_model
        audio_processor, vae, unet, pe = load_all_model()

        self._vae = (vae.vae if hasattr(vae, "vae") else vae).to(self.device).eval()
        self._vae_scale = getattr(getattr(self._vae, "config", None),
                                  "scaling_factor", 0.18215)
        self._unet = (unet.model if hasattr(unet, "model") else unet).to(self.device).eval()
        self._pe = pe.to(self.device) if hasattr(pe, "to") else pe

        try:
            self._vae = self._vae.to(self._dtype)
            self._unet = self._unet.to(self._dtype)
        except Exception:
            self._dtype = torch.float32   # some builds dislike fp16 weights

        # Whisper feature extractor: positional-encode the audio processor output.
        def _whisper_feature(window):
            feats = audio_processor.get_audio_feature(window) \
                if hasattr(audio_processor, "get_audio_feature") else None
            if feats is None:
                return None
            return self._pe(feats) if callable(self._pe) else feats
        self._whisper_feature = _whisper_feature

    def _warmup_real(self):
        """Exercise the real path once so CUDA kernels autotune at startup."""
        face = None
        if self.character_path and os.path.exists(self.character_path):
            face = cv2.imread(self.character_path)
        if face is None:
            face = np.full((512, 512, 3), 120, np.uint8)
        face = cv2.resize(face, (512, 512))
        self.feed_audio((np.random.randn(AUDIO_WINDOW) * 0.05).astype(np.float32))
        bbox = (180, 320, 332, 440)
        for _ in range(3):
            self.process_mouth(face, bbox)
        self.audio_buffer.clear()
        self.is_speaking = False
        self._silence = 0

    def _print_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[MUSETALK] GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB")
        except Exception:
            pass


if __name__ == "__main__":
    char = os.path.join(PROJECT_DIR, "ai-face", "character.jpg")
    if not os.path.exists(char):
        char = os.path.join(PROJECT_DIR, "character.jpg")
    eng = MuseTalkEngine(char)
    print("[MUSETALK] startup_check:", eng.startup_check())
    eng.feed_audio((np.random.randn(6400) * 0.1).astype(np.float32))
    frame = np.full((512, 512, 3), 110, np.uint8)
    crop = eng.process_mouth(frame, (180, 320, 332, 440))
    print("[MUSETALK] mouth crop:", None if crop is None else crop.shape)
