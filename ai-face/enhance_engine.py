# =============================================================================
# enhance_engine.py
# -----------------------------------------------------------------------------
# Face-restoration enhancer that makes Wav2Lip output look sharp and human.
#
# Wav2Lip generates the mouth region at only 96x96 and pastes it back, so the
# lips look soft/blurry. GFPGAN restores facial detail (especially the mouth),
# upscales, and produces a clean, "HD" looking face.
#
# The GFPGAN model is loaded ONCE and kept resident in VRAM, so enhancing a
# rendered clip just streams frames through it.
# =============================================================================

import os
import sys

# Force UTF-8 stdout so status glyphs ([*] [✓] [!]) don't crash the Windows
# console, whose default code page (cp1252) can't encode them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GFPGAN_MODEL = os.path.join(PROJECT_ROOT, "models", "GFPGANv1.4.pth")

UPSCALE = 2          # 1 = restore at same size, 2 = restore + 2x upscale (clearer)
FIDELITY = 0.5       # GFPGAN blend weight: lower = stronger restoration, higher = closer to input
ONLY_CENTER_FACE = True   # only restore the main (center) face

# Speed: the GFPGAN network forward is the bottleneck (~200ms/frame one-at-a-
# time). Batching many faces through it at once and using fp16 cuts that by an
# order of magnitude. These tune the batched fast path (enhance_frames()).
BATCH_SIZE = 8       # faces per network forward (lower if you hit VRAM limits)
# bf16 autocast: ~2x faster than fp32 and, unlike pure fp16, keeps fp32's
# exponent range so GFPGAN's StyleGAN decoder doesn't overflow into garbage.
USE_BF16 = True

# Lazily-created singletons so importing this module is cheap and model load
# happens exactly once, on first use.
_restorer = None
_load_failed = False


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def _get_restorer():
    """Load (once) and return the GFPGANer instance, or None if unavailable.

    Caches the result so the heavy model load happens a single time.
    """
    global _restorer, _load_failed
    if _restorer is not None:
        return _restorer
    if _load_failed:
        return None

    if not os.path.exists(GFPGAN_MODEL):
        print("[!] Enhancer: GFPGAN model not found at", GFPGAN_MODEL)
        print("    Download GFPGANv1.4.pth into the models/ folder to enable enhancement.")
        _load_failed = True
        return None

    try:
        import torch
        from gfpgan import GFPGANer
    except Exception as exc:
        print("[!] Enhancer: could not import gfpgan:", exc)
        print("    Install it with:  pip install gfpgan")
        _load_failed = True
        return None

    # cuDNN autotuning picks the fastest conv kernels for our fixed 512x512
    # face size — roughly a 4x speedup on the GFPGAN net vs the default kernels.
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Enhancer: loading GFPGAN on {device} (one-time)...")
    try:
        _restorer = GFPGANer(
            model_path=GFPGAN_MODEL,
            upscale=UPSCALE,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,        # no background upsampler (keep it fast)
            device=device,
        )
    except Exception as exc:
        print("[!] Enhancer: failed to initialize GFPGAN:", exc)
        _load_failed = True
        return None

    print("[✓] Enhancer: GFPGAN ready.")
    return _restorer


def is_available():
    """Return True if the enhancer can run (model + library present)."""
    return _get_restorer() is not None


# -----------------------------------------------------------------------------
# FRAME ENHANCEMENT
# -----------------------------------------------------------------------------
def enhance_bgr(frame, out_size=None):
    """Restore/sharpen the face in a single BGR frame.

    Args:
        frame:    BGR image (numpy array) as produced by Wav2Lip.
        out_size: Optional (w, h) to resize the result to. If None, the frame
                  is returned at the enhancer's native (possibly upscaled) size.

    Returns:
        The enhanced BGR frame. If the enhancer is unavailable or fails on this
        frame, the original frame is returned unchanged (so playback never
        breaks).
    """
    restorer = _get_restorer()
    if restorer is None:
        if out_size is not None:
            return cv2.resize(frame, out_size)
        return frame

    try:
        # has_aligned=False -> GFPGAN detects + aligns the face itself.
        # paste_back=True   -> returns the full image with the restored face.
        _, _, restored = restorer.enhance(
            frame,
            has_aligned=False,
            only_center_face=ONLY_CENTER_FACE,
            paste_back=True,
            weight=FIDELITY,
        )
        if restored is None:
            restored = frame
    except Exception as exc:
        # A single bad frame should not kill the stream.
        print("[!] Enhancer: frame restore failed, using original:", exc)
        restored = frame

    if out_size is not None:
        restored = cv2.resize(restored, out_size)
    return restored


# -----------------------------------------------------------------------------
# BATCHED FRAME ENHANCEMENT (fast path for a whole clip)
# -----------------------------------------------------------------------------
def _autocast_ctx():
    """Return a bf16 autocast context on CUDA (or a no-op elsewhere)."""
    import torch
    import contextlib
    restorer = _get_restorer()
    if USE_BF16 and restorer is not None and restorer.device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def enhance_frames(frames, out_size=None):
    """Enhance a whole list of frames fast.

    All frames share the same static head, so the face is detected/aligned
    ONCE (on the first frame) and that alignment is reused for every frame.
    The restoration network is then run in batches (and in fp16), which is far
    faster than enhancing frames one at a time.

    Falls back to per-frame enhance_bgr (or plain resize) if anything about the
    fast path is unavailable, so output is always produced.
    """
    if not frames:
        return []

    restorer = _get_restorer()
    if restorer is None:
        return [cv2.resize(f, out_size) if out_size is not None else f for f in frames]

    try:
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        fh = restorer.face_helper
        device = restorer.device
        net = restorer.gfpgan
        upscale = float(fh.upscale_factor)

        # --- 1) Detect + align ONCE on the first frame; cache the affine. ---
        fh.clean_all()
        fh.read_image(frames[0])
        fh.get_face_landmarks_5(only_center_face=ONLY_CENTER_FACE, eye_dist_threshold=5)
        fh.align_warp_face()
        if not fh.affine_matrices:
            raise RuntimeError("no face on first frame")
        affine = fh.affine_matrices[0]
        face_size = fh.face_size

        # --- 2) Warp every frame's face crop with the cached affine. ---
        crops = [cv2.warpAffine(f, affine, face_size, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(135, 133, 132)) for f in frames]

        # --- 3) Run the restoration net in FIXED-size batches (bf16). ---
        # The batch size must stay constant (we pad the last batch) so cuDNN
        # autotunes the conv shape exactly once instead of re-tuning (which costs
        # seconds) on every clip whose frame count isn't a multiple of the batch.
        restored_faces = []
        for i in range(0, len(crops), BATCH_SIZE):
            sub = crops[i:i + BATCH_SIZE]
            m = len(sub)
            if m < BATCH_SIZE:
                sub = sub + [sub[-1]] * (BATCH_SIZE - m)     # pad to fixed size
            tensors = []
            for c in sub:
                t = img2tensor(c / 255.0, bgr2rgb=True, float32=True)
                normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                tensors.append(t)
            batch = torch.stack(tensors).to(device=device)
            with torch.no_grad(), _autocast_ctx():
                out = net(batch, return_rgb=False, weight=FIDELITY)[0]
            out = out[:m].float()                             # keep only real frames
            for o in out:
                restored_faces.append(
                    tensor2img(o, rgb2bgr=True, min_max=(-1, 1)).astype("uint8"))

        # --- 4) Precompute the paste mask + static background ONCE. ---
        # The head is static, so the soft blend mask and everything outside the
        # face (hair, neck, shirt) are identical for every frame. Only the
        # restored face pixels change. This collapses GFPGAN's expensive
        # per-frame paste (face-parse net + big Gaussian blurs) into a single
        # warp + blend per frame. We warp straight to the OUTPUT size (baking the
        # final scale into the inverse affine) so there is no 1024->720 resize.
        h, w = frames[0].shape[:2]
        if out_size is not None:
            out_w, out_h = out_size
        else:
            out_w, out_h = int(w * upscale), int(h * upscale)
        sx, sy = out_w / w, out_h / h
        s = (sx + sy) / 2.0

        inv_affine = cv2.invertAffineTransform(affine)
        inv_affine[0, :] *= sx
        inv_affine[1, :] *= sy
        inv_affine[0, 2] += 0.5 * sx
        inv_affine[1, 2] += 0.5 * sy

        mask = np.ones(face_size, dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inv_affine, (out_w, out_h))
        k = max(1, int(2 * s))
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((k, k), np.uint8))
        total_area = float(np.sum(inv_mask_erosion))
        w_edge = int(total_area ** 0.5) // 20
        erosion_radius = max(1, w_edge * 2)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = max(1, w_edge * 2)
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)[:, :, None]

        # mask_a multiplies the restored face; bg_term is the constant backdrop.
        mask_a = inv_soft_mask * inv_mask_erosion[:, :, None]
        bg = cv2.resize(frames[0], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        bg_term = (1.0 - inv_soft_mask) * bg

        # --- 5) Fast paste per frame: warp restored face + blend. ---
        results = []
        for rf in restored_faces:
            inv_restored = cv2.warpAffine(rf, inv_affine, (out_w, out_h)).astype(np.float32)
            out_img = mask_a * inv_restored + bg_term
            results.append(np.clip(out_img, 0, 255).astype(np.uint8))
        return results

    except Exception as exc:
        # Fast path failed -> safe per-frame fallback.
        print("[!] Enhancer: batched path failed, falling back per-frame:", exc)
        return [enhance_bgr(f, out_size=out_size) for f in frames]


# -----------------------------------------------------------------------------
# STANDALONE TEST  (python enhance_engine.py input.jpg output.jpg)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python enhance_engine.py <input_image> <output_image>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    img = cv2.imread(src)
    if img is None:
        print("[!] Could not read", src)
        sys.exit(1)

    if not is_available():
        print("[!] Enhancer unavailable; cannot run test.")
        sys.exit(1)

    result = enhance_bgr(img)
    cv2.imwrite(dst, result)
    print(f"[✓] Enhanced {src} ({img.shape[1]}x{img.shape[0]}) -> "
          f"{dst} ({result.shape[1]}x{result.shape[0]})")
