# Fine-tune XTTS-v2 on the Mohamed voice (84 short segments). EPOCHS via env.
import os, sys, shutil
import torch
import transformers.pytorch_utils as _p
if not hasattr(_p, "isin_mps_friendly"):
    _p.isin_mps_friendly = lambda e, t: torch.isin(e, t)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "voice_train", "run")
CKPT = os.path.join(OUT, "XTTS_v2.0_original_model_files")
os.makedirs(CKPT, exist_ok=True)

# pre-copy the already-downloaded base files so train_gpt skips the ~2GB download
base = os.path.expandvars(r"%LOCALAPPDATA%/tts/tts_models--multilingual--multi-dataset--xtts_v2")
for f in ("model.pth", "config.json", "vocab.json", "dvae.pth", "mel_stats.pth"):
    dst = os.path.join(CKPT, f)
    src = os.path.join(base, f)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[prep] copied {f}")

from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = int(os.environ.get("BATCH", "3"))
GRAD = int(os.environ.get("GRAD", "8"))
print(f"[train] EPOCHS={EPOCHS} BATCH={BATCH} GRAD={GRAD}")
cfg, ckpt, vocab, out_path, spk = train_gpt(
    "en", EPOCHS, BATCH, GRAD,
    os.path.join(PROJ, "voice_train", "metadata_train.csv"),
    os.path.join(PROJ, "voice_train", "metadata_eval.csv"),
    OUT, max_audio_length=264600)
print("TRAIN_DONE")
print("OUT_PATH=" + str(out_path))
print("CONFIG=" + str(cfg))
print("VOCAB=" + str(vocab))
print("SPEAKER_REF=" + str(spk))
