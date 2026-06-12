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
BLEND_FACTOR = float(os.environ.get("AVATAR_MOUTH_BLEND", "1.0"))  # 1.0 = full bot mouth (no operator leak)
MOUTH_SHARP = float(os.environ.get("AVATAR_MOUTH_SHARP", "0.55"))  # unsharp
MOUTH_POP = float(os.environ.get("AVATAR_MOUTH_POP", "0.0"))       # off: the contrast darkened the mouth/nose into shadows
MOUTH_SAT = float(os.environ.get("AVATAR_MOUTH_SAT", "0.0"))       # off (was garish)
MOUTH_CLAHE = float(os.environ.get("AVATAR_MOUTH_CLAHE", "0.3"))    # eased: high CLAHE deepened mouth/nose shadows
# HI-RES crisp: sharpen the talking mouth at 2x then resample back. OFF — on top of
# the existing sharpen + the final enhance it over-crunches the 96px mouth.
MOUTH_HIRES = float(os.environ.get("AVATAR_MOUTH_HIRES", "0.5"))
# CLEAN-then-crisp: bilateral denoise the mouth BEFORE sharpening so the crisp defines
# real lip/teeth edges instead of amplifying 96px sensor noise (the "crunch"). 0 = off.
# EASED to 0.4 — at 1.0 it over-smoothed the moving/open mouth = "more blur".
MOUTH_DENOISE = float(os.environ.get("AVATAR_MOUTH_DENOISE", "0.0"))
# INTERIOR: lift the very-dark open-mouth interior toward a natural dark-red so the
# open mouth doesn't render as black holes/"spots" (Wav2Lip's dark interior). 0 = off.
MOUTH_INTERIOR = float(os.environ.get("AVATAR_MOUTH_INTERIOR", "1.0"))
# DE-BLUR the speaking mouth: Real-ESRGAN detail on the (small) mouth crop — ~0ms,
# adds genuine sharpness so the open/talking mouth isn't soft. 0 = off.
DEBLUR_MOUTH = float(os.environ.get("AVATAR_MOUTH_DEBLUR", "0"))  # OFF: redundant (CPU crisp already = same sharpness) + GPU op = lag risk

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

        # --- mouth engine selection ---------------------------------------------
        # MuseTalk's 256 VAE decode is SOFT; Wav2Lip generates a crisp full mouth
        # (and replaces it entirely, so no "operator upper-lip" leak). Default to
        # Wav2Lip for a sharp, leak-free, bot-driven mouth. AVATAR_MOUTH_ENGINE=
        # musetalk forces the (softer) MuseTalk path.
        _engine = os.environ.get("AVATAR_MOUTH_ENGINE", "wav2lip").lower()
        # --- try the real MuseTalk pipeline first (only if requested) ---
        mt_path = self._find_musetalk() if _engine == "musetalk" else None
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
        # PRELOAD the mouth de-blur upscaler now (at boot) so it never lazy-loads
        # mid-speech and stalls the avatar the moment the bot starts talking.
        if DEBLUR_MOUTH > 0:
            try:
                self._get_upscaler()
            except Exception:
                pass

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

    def _get_upscaler(self):
        """Lazy Real-ESRGAN (used to de-blur the speaking mouth crop). None if absent."""
        if getattr(self, "_up_tried", False):
            return getattr(self, "_upscaler", None)
        self._up_tried = True
        self._upscaler = None
        try:
            from upscale_engine import UpscaleEngine
            u = UpscaleEngine()
            self._upscaler = u if getattr(u, "ready", False) else None
        except Exception as exc:
            print(f"[MUSETALK] mouth de-blur unavailable ({exc})")
        return self._upscaler

    def _pop_mouth(self, mouth):
        """Make the mouth movement READ CLEARLY: unsharp + local contrast (the dark
        opening pops, lips define) + a touch of saturation (redder lips). Tunable via
        AVATAR_MOUTH_SHARP / AVATAR_MOUTH_POP / AVATAR_MOUTH_SAT."""
        try:
            if mouth.shape[0] < 6 or mouth.shape[1] < 6:
                return mouth
            if MOUTH_DENOISE > 0:                            # CLEAN first (edge-preserving)
                mouth = cv2.bilateralFilter(mouth, 5, int(30 * MOUTH_DENOISE), 5)
            m = mouth.astype(np.float32)
            if MOUTH_SHARP > 0:                              # TWO-SCALE unsharp = crisp
                blur1 = cv2.GaussianBlur(m, (0, 0), 1.0)     # fine edges (lips/teeth)
                m = m * (1 + MOUTH_SHARP) - blur1 * MOUTH_SHARP
                blur2 = cv2.GaussianBlur(m, (0, 0), 2.6)     # medium detail / structure
                m = m * (1 + MOUTH_SHARP * 0.5) - blur2 * (MOUTH_SHARP * 0.5)
            if MOUTH_POP > 0:                                # contrast: opening darkens, lips define
                m = (m - 128.0) * (1.0 + MOUTH_POP) + 128.0
            m = np.clip(m, 0, 255).astype(np.uint8)
            if MOUTH_CLAHE > 0:                              # CPU local contrast defines a soft
                if getattr(self, "_mclahe", None) is None:   # (e.g. White-Haddan) mouth, NO GPU lag
                    self._mclahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(4, 4))
                lab = cv2.cvtColor(m, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.addWeighted(lab[:, :, 0], 1.0 - MOUTH_CLAHE,
                                               self._mclahe.apply(lab[:, :, 0]), MOUTH_CLAHE, 0)
                m = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if MOUTH_SAT > 0:                                # redder, livelier lips
                hsv = cv2.cvtColor(m, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + MOUTH_SAT), 0, 255)
                m = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if MOUTH_HIRES > 0 and m.shape[0] >= 8:          # HI-RES CPU crisp (lips/teeth)
                h, w = m.shape[:2]
                big = cv2.resize(m, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(big, (0, 0), 1.6)
                big = cv2.addWeighted(big, 1.0 + MOUTH_HIRES, blur, -MOUTH_HIRES, 0)
                m = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
            if MOUTH_INTERIOR > 0:                           # lift black interior -> dark red
                b, g, rr = cv2.split(m.astype(np.float32))
                gray = (0.114 * b + 0.587 * g + 0.299 * rr)
                dark = np.clip((52.0 - gray) / 52.0, 0, 1) * MOUTH_INTERIOR
                rr = np.clip(rr + dark * 42, 0, 255)         # warm (red) + lift the opening
                g = np.clip(g + dark * 14, 0, 255)
                b = np.clip(b + dark * 14, 0, 255)
                m = cv2.merge([b, g, rr]).astype(np.uint8)
            return m
        except Exception:
            return mouth

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

        speaking = self._update_speaking()
        bot_only = getattr(self, "bot_only", False)
        # bot_only (lip-lock): the mouth is ALWAYS the bot's — synced while it talks,
        # a CLOSED bot mouth when silent — so the operator's webcam mouth NEVER shows.
        # Without bot_only, fall back to the webcam mouth when not speaking.
        if not speaking and not bot_only:
            return base_crop

        try:
            if self.real_ready:
                if not speaking:
                    # SILENT + bot_only -> render a closed bot mouth (feed silence),
                    # throttled to every ~8 frames + cached (cheap when not talking).
                    self._cm_fc = getattr(self, "_cm_fc", 0) + 1
                    if getattr(self, "_closed_mouth", None) is None or (self._cm_fc % 8) == 0:
                        self.audio_buffer.extend(np.zeros(AUDIO_WINDOW, np.float32))
                        self._closed_mouth = self._process_real(lp_face_frame, mouth_bbox, base_crop)
                    return cv2.resize(self._closed_mouth, (x2 - x1, y2 - y1))
                # SPEAKING — live synced; ~76ms forward, so every 2nd frame reuse last.
                self._mt_fc = getattr(self, "_mt_fc", 0) + 1
                if (self._mt_fc % 2) == 0 and getattr(self, "_mt_last", None) is not None:
                    return cv2.resize(self._mt_last, (x2 - x1, y2 - y1))
                mouth = self._process_real(lp_face_frame, mouth_bbox, base_crop)
                self._mt_last = mouth
                return mouth
            if self._w2l is not None:
                synced_full = self._w2l.process_frame(lp_face_frame)
                mouth = synced_full[y1:y2, x1:x2].copy()
                mouth = self._pop_mouth(mouth)
                # DE-BLUR the SPEAKING mouth with Real-ESRGAN — but ONLY when the
                # governor says the GPU is free (allow_deblur), so it sharpens during
                # playback yet never fights the voice-synthesis GPU spike = no lag.
                if (speaking and DEBLUR_MOUTH > 0 and getattr(self, "allow_deblur", True)):
                    up = self._get_upscaler()
                    if up is not None:
                        mouth = up.enhance_crop(mouth, DEBLUR_MOUTH)
                # LIGHT temporal smoothing — kills frame-to-frame shimmer/jitter without
                # smearing the lip movement (the bbox is now position-stabilised too).
                if MOUTH_TEMPORAL > 0:
                    pm = getattr(self, "_prev_mouth", None)
                    if pm is not None and pm.shape == mouth.shape:
                        mouth = cv2.addWeighted(mouth, 1.0 - MOUTH_TEMPORAL, pm,
                                                MOUTH_TEMPORAL, 0)
                    self._prev_mouth = mouth.copy()
                return mouth
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
        """MuseTalk V1.5 UNet mouth inpainting conditioned on Whisper features."""
        torch = self._torch
        x1, y1, x2, y2 = bbox

        # 1) audio conditioning for the CURRENT moment from the recent window
        window = np.array(list(self.audio_buffer)[-AUDIO_WINDOW:], dtype=np.float32)
        audio_feat = self._audio_features(window)        # (1, 50, 384) on device
        if audio_feat is None:
            return base_crop

        # 2) MuseTalk needs a SQUARE face crop with the mouth in the LOWER (inpainted)
        # half — a wide mouth box stretched to 256² distorts -> garbage, and a centred
        # mouth leaks the operator's upper lip (the kept upper half). So build a square
        # region whose lower half holds the whole mouth, run MuseTalk, then lift the
        # original mouth bbox back out of the result.
        H, W = lp_face_frame.shape[:2]
        mcx = (x1 + x2) // 2
        lip_y = (y1 + y2) // 2                            # lip line (bbox centre)
        mw = max(8, x2 - x1)
        # TIGHTER square (mouth fills more of the 256 crop = sharper, less blur) while
        # still centring ABOVE the lips so the lip line sits ~62% down — in MuseTalk's
        # inpainted (bot) lower half, not the kept upper half.
        side = int(max(mw, y2 - y1) * 1.45)
        scy = lip_y - int(side * 0.12)
        sy1 = max(0, scy - side // 2); sy2 = min(H, sy1 + side); sy1 = max(0, sy2 - side)
        sx1 = max(0, mcx - side // 2); sx2 = min(W, sx1 + side); sx1 = max(0, sx2 - side)
        sq = lp_face_frame[sy1:sy2, sx1:sx2]
        if sq.shape[0] < 8 or sq.shape[1] < 8:
            return base_crop
        sq256 = cv2.resize(sq, (FACE_REGION_SIZE, FACE_REGION_SIZE))
        with torch.no_grad():
            latents = self._vae.get_latents_for_unet(sq256).to(self.device, self._dtype)
            pred = self._unet.model(latents, self._ts,
                                    encoder_hidden_states=audio_feat).sample
            pred = pred.to(self._vae.vae.dtype)
            recon = self._vae.decode_latents(pred)        # (1,256,256,3) BGR uint8
        recon = recon[0] if recon.ndim == 4 else recon
        recon = cv2.resize(recon, (sx2 - sx1, sy2 - sy1)) # back to the square's size
        # lift the original mouth bbox region out of the square result
        oy1, oy2 = y1 - sy1, y2 - sy1
        ox1, ox2 = x1 - sx1, x2 - sx1
        out = recon[oy1:oy2, ox1:ox2]
        if out.shape[:2] != base_crop.shape[:2]:
            out = cv2.resize(out, (x2 - x1, y2 - y1))
        # the 256 VAE decode is soft -> sharpen + pop so the mouth reads clearly
        out = self._pop_mouth(out)
        blended = (out.astype(np.float32) * BLEND_FACTOR +
                   base_crop.astype(np.float32) * (1.0 - BLEND_FACTOR))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _audio_features(self, window):
        """V1.5: Whisper-encode the recent audio window and return the per-frame
        conditioning (1, 50, 384) for the CURRENT moment (the end of the window)."""
        try:
            torch = self._torch
            feats = self._audio_proc.feature_extractor(
                window, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_features
            feats = feats.to(self.device).to(self._dtype)
            with torch.no_grad():
                hs = self._whisper.encoder(feats, output_hidden_states=True).hidden_states
            hs = torch.stack(hs, dim=2)                  # (1, T_audio, layers, 384)
            # V1.5 conditioning = 10 audio steps (2*(2+2+1)) at the END of the real
            # (non-padding) audio; x5 whisper layers -> the 50-length (1,50,384) cond.
            real = int(round(len(window) / SAMPLE_RATE * 50))      # 50 audio-fps
            real = max(10, min(real, hs.shape[1]))
            clip = hs[:, real - 10:real]                 # (1, 10, layers, 384)
            clip = self._rearrange(clip, "b c h w -> b (c h) w")   # (1, 10*layers, 384)
            return self._pe(clip.to(self._dtype)).to(self._dtype)  # pe can upcast -> recast
        except Exception as exc:
            if getattr(self, "_af_errs", 0) < 3:
                self._af_errs = getattr(self, "_af_errs", 0) + 1
                print(f"[MUSETALK] audio feature error: {exc}")
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

        # MuseTalk V1.5: load_all_model() -> (vae_wrapper, unet_wrapper, pe). The
        # whisper model + AudioProcessor are separate. load_all_model + the VAE use
        # RELATIVE "models/..." paths, so we chdir into the checkout during load.
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel
        from einops import rearrange
        self._rearrange = rearrange
        _cwd = os.getcwd()
        try:
            os.chdir(mt_path)
            vae, unet, pe = load_all_model(
                unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
                vae_type="sd-vae",
                unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
                device=self.device)
            whisper_dir = os.path.join("models", "whisper")
            self._audio_proc = AudioProcessor(feature_extractor_path=whisper_dir)
            self._whisper = (WhisperModel.from_pretrained(whisper_dir)
                             .to(device=self.device, dtype=self._dtype).eval())
        finally:
            os.chdir(_cwd)

        # vae is a VAE wrapper (get_latents_for_unet / decode_latents); unet wraps
        # a UNet2DConditionModel at .model. Keep the WRAPPERS (their helpers do the
        # masking + scaling correctly).
        vae.vae = vae.vae.to(device=self.device, dtype=self._dtype).eval()
        unet.model = unet.model.to(device=self.device, dtype=self._dtype).eval()
        self._vae = vae
        self._unet = unet
        self._pe = pe.to(self.device) if hasattr(pe, "to") else pe
        self._ts = torch.tensor([0], device=self.device)

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
