# Fine-tune XTTS-v2 on the Mohamed voice — Windows-safe (num_loader_workers=0 to
# avoid the multiprocessing data-loader deadlock that hung the first attempt).
import os, sys, gc
import torch
import transformers.pytorch_utils as _p
if not hasattr(_p, "isin_mps_friendly"):
    _p.isin_mps_friendly = lambda e, t: torch.isin(e, t)

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJ, "voice_train", "run", "run", "training")  # reuse prev dir (base files there)
CK = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")
DVAE = os.path.join(CK, "dvae.pth"); MEL = os.path.join(CK, "mel_stats.pth")
VOCAB = os.path.join(CK, "vocab.json"); CKPT = os.path.join(CK, "model.pth")
CFG = os.path.join(CK, "config.json")
for f in (DVAE, MEL, VOCAB, CKPT):
    assert os.path.isfile(f), f"missing base file {f}"

EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = int(os.environ.get("BATCH", "3"))
GRAD = int(os.environ.get("GRAD", "4"))
print(f"[train] EPOCHS={EPOCHS} BATCH={BATCH} GRAD={GRAD} workers=0 (windows-safe)", flush=True)

ds = BaseDatasetConfig(formatter="coqui", dataset_name="moha",
                       path=os.path.join(PROJ, "voice_train"),
                       meta_file_train=os.path.join(PROJ, "voice_train", "metadata_train.csv"),
                       meta_file_val=os.path.join(PROJ, "voice_train", "metadata_eval.csv"),
                       language="en")

model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150,
    debug_loading_failures=False, max_wav_length=264600, max_text_length=200,
    mel_norm_file=MEL, dvae_checkpoint=DVAE, xtts_checkpoint=CKPT, tokenizer_file=VOCAB,
    gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(
    epochs=EPOCHS, output_path=OUT_PATH, model_args=model_args,
    run_name="GPT_XTTS_MOHA", project_name="XTTS_trainer",
    dashboard_logger="tensorboard", logger_uri=None, audio=audio_config,
    batch_size=BATCH, batch_group_size=48, eval_batch_size=BATCH,
    num_loader_workers=0, num_eval_loader_workers=0,        # <-- THE FIX (no MP deadlock)
    eval_split_max_size=256, print_step=5, plot_step=100, log_model_step=100,
    save_step=200, save_n_checkpoints=1, save_checkpoints=True, print_eval=False,
    optimizer="AdamW", optimizer_wd_only_on_weights=True,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06, lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[])

print("[train] init model from config...", flush=True)
model = GPTTrainer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    [ds], eval_split=True, eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size)
print(f"[train] {len(train_samples)} train / {len(eval_samples)} eval samples — starting fit()", flush=True)
trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                grad_accum_steps=GRAD),
    config, output_path=OUT_PATH, model=model,
    train_samples=train_samples, eval_samples=eval_samples)
trainer.fit()
samples_len = [len(it["text"].split(" ")) for it in train_samples]
speaker_ref = train_samples[samples_len.index(max(samples_len))]["audio_file"]
print("TRAIN_DONE", flush=True)
print("OUT_PATH=" + str(trainer.output_path), flush=True)
print("SPEAKER_REF=" + str(speaker_ref), flush=True)
del model, trainer; gc.collect()
