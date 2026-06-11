# TiktokBot — Personal AI Avatar & Face-Dataset Pipeline

An end-to-end toolkit for building a **photorealistic talking-head avatar** of a
single person, driven entirely from that person's own existing YouTube footage.
It harvests the best face footage from a large back-catalogue, builds a clean
training dataset, and runs a real-time / autonomous avatar with neural
text-to-speech, lip-sync, and face restoration.

The reference subject is the YouTube channel **[@ghaithabohlal](https://www.youtube.com/@ghaithabohlal)**
(2,700+ uploads, mostly Arabic gold/forex market analysis). Most uploads are
chart / screen-share recordings; only a fraction contain clear face-cam
footage. The hard part — and the thing this repo solves efficiently — is
**finding the ~20–30 minutes of the best, most consistent face footage**
without scanning every video.

> **Status:** active personal project. The `data_prep/` staged harvester is the
> newest and most polished subsystem. The avatar runtime (`ai-face/`, the
> `*_avatar.py` scripts, `engines/`) is functional but evolves frequently.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [High-level architecture](#high-level-architecture)
3. [The staged face harvester (`data_prep/`)](#the-staged-face-harvester-data_prep)
   - [Stage 1 — rank by thumbnail](#stage-1--rank-by-thumbnail)
   - [Stage 2 — scan candidates](#stage-2--scan-candidates)
   - [Stage 3 — build dataset (the consistency filter)](#stage-3--build-dataset-the-consistency-filter)
   - [Stage 4 — quality report (go / no-go)](#stage-4--quality-report-go--no-go)
   - [Run order & outputs](#run-order--outputs)
   - [Tuning knobs](#tuning-knobs)
4. [The avatar runtime](#the-avatar-runtime)
5. [Hardware & environment](#hardware--environment)
6. [Installation](#installation)
7. [Dependency landmines (read before installing)](#dependency-landmines-read-before-installing)
8. [Repository layout](#repository-layout)
9. [Troubleshooting](#troubleshooting)
10. [Design principles](#design-principles)
11. [Disclaimer & ethics](#disclaimer--ethics)

---

## Why this exists

Training a sharp avatar needs **consistent** footage: the same person, similar
lighting, similar camera, mostly frontal. Feeding a model a random grab-bag of
clips — some bright, some dark, some side-profile, some tiny-in-frame — produces
a blurry, unstable avatar that "averages" all those looks together.

A channel with 2,700 videos is a goldmine *and* a trap:

- **Goldmine:** somewhere in there are tens of minutes of clean face-cam.
- **Trap:** brute-force downloading and scanning all 2,700 videos would take
  **days**, burn hundreds of GB of disk, and — worse — drag low-quality and
  inconsistent footage into the dataset, *hurting* the avatar.

The harvester is designed around one idea: **spend compute in proportion to how
promising a video is, and stop the moment you have enough good, consistent
footage.** Cheap signals first (thumbnails), expensive signals last (full-res
cuts), with hard early-stops at every stage.

---

## High-level architecture

```
                       ┌─────────────────────────────────────────────┐
                       │  data_prep/  — find the best face footage    │
                       │                                              │
  YouTube channel ───► │  stage1  rank 2700 by THUMBNAIL (no DL)      │
                       │     │      ~free, minutes                    │
                       │     ▼                                        │
                       │  stage2  scan top 150 at 720p, 1fps          │
                       │     │      downloads only candidates         │
                       │     ▼      early-stop @ ~50 min footage      │
                       │  stage3  CONSISTENCY FILTER + cut clips      │
                       │     │      insightface cluster → 1 look      │
                       │     ▼      full-res frame-accurate ffmpeg    │
                       │  stage4  go / no-go quality report           │
                       └─────────────────────────┬───────────────────┘
                                                 │  face_segments/  (~25 min)
                                                 ▼
                       ┌─────────────────────────────────────────────┐
                       │  avatar runtime                              │
                       │  • process_dataset.py  (SyncTalk prep)       │
                       │  • realtime_avatar.py / autonomous_avatar.py │
                       │  • engines/  (TTS, lip-sync, restoration)    │
                       │  • ai-face/  (Wav2Lip, CodeFormer, ESRGAN)   │
                       └─────────────────────────────────────────────┘
```

Two loosely-coupled halves:

1. **`data_prep/`** turns a noisy 2,700-video catalogue into a tight, consistent
   ~25-minute face dataset. This is the part you run **once** to bootstrap.
2. **The avatar runtime** consumes that dataset (plus reference images and voice
   clips) to drive a live or autonomous avatar with TTS + lip-sync + HD face
   restoration.

---

## The staged face harvester (`data_prep/`)

Four scripts, run in order, each cheaper-to-more-expensive than the last. Every
stage **caches** its results so re-runs are nearly free, and every stage honours
an **early-stop** so you never do more work than needed.

Console output is prefixed `[STAGE1]` / `[STAGE2]` / `[STAGE3]` / `[REPORT]` so
you always know which stage is talking.

### Stage 1 — rank by thumbnail

**`data_prep/stage1_rank_by_thumbnail.py`**

- Pulls the **flat** video list + metadata for the whole channel via `yt-dlp`
  in a single call — **no videos are downloaded**, not even one.
- For each video, downloads only the **thumbnail** (a ~25 KB JPEG, fetched
  directly from `i.ytimg.com/vi/<id>/hqdefault.jpg`).
- Runs **MediaPipe** face detection on each thumbnail and scores a
  `face_likelihood` from:
  - detector confidence,
  - bounding-box area (bigger face = better),
  - a **frontal-ness** proxy built from the 6 detection keypoints
    (eye/nose symmetry → yaw & roll estimate),
  - a weak **title-keyword** nudge (Arabic + English): titles mentioning
    *live / analysis / interview* get a small boost; pure
    *price-update / signal / chart* titles get a small penalty.
- Writes `data_prep/candidates_ranked.json` — **all** videos ranked, highest
  face-likelihood first — and shortlists the top `TAKE_TOP` (default **150**)
  for stage 2.

Thumbnail scores are cached in `data_prep/cache/thumb_scores.json`; re-runs skip
everything already scored.

> Example run on the reference channel: *"Of 1809 videos, ~1086 show a clear
> face in the thumbnail."* The other ~700 are charts/screen-share and are never
> downloaded.

### Stage 2 — scan candidates

**`data_prep/stage2_scan_candidates.py`**

- Takes the stage-1 shortlist and, in ranked order, downloads each candidate at
  **≤720p** (enough to scan; cached in `data_prep/scan_videos/`).
- Samples frames at **1 fps** and scores each frame for face presence/quality,
  applying hard gates:
  - face bbox **> 12%** of frame,
  - **yaw within ±30°**,
  - sharpness (Laplacian variance) above a floor,
  - brightness within a sane band.
- Groups passing frames into **segments** (≥ 2 s, bridging dropouts ≤ 1 s) and
  scores each segment `avg_quality × duration`.
- Saves a **representative face crop** per segment (best frame) into
  `data_prep/seg_reps/` for stage-3's appearance embedding.
- Writes `data_prep/segments_scored.json` (segments ranked — **not yet cut**).

Per-video scan results are cached (`data_prep/cache/scan_<id>.json`), and the
stage **stops early** once it has gathered roughly `TARGET_MINUTES × 2` (≈ 50)
minutes of good footage — deliberately over-collecting so stage 3 has room to
*reject* inconsistent looks rather than being forced to keep them.

### Stage 3 — build dataset (the consistency filter)

**`data_prep/stage3_build_dataset.py`** — *the script that makes the avatar
sharp instead of blurry.*

1. Computes a 512-d **InsightFace** (`buffalo_l`) embedding for every segment's
   representative face.
2. **Clusters** segments by appearance using agglomerative clustering on cosine
   distance — grouping by *look / lighting / camera*, without having to know the
   number of distinct looks in advance.
3. Picks the **largest cluster by total duration** as the base identity. This is
   the dominant, most-consistent look across the catalogue.
4. From that cluster only, takes the **highest-quality segments until
   `TARGET_MINUTES` (default 25)** is reached, then **stops**. Outlier-looking
   segments are rejected *even if individually high quality* — mixing looks is
   exactly what blurs an avatar.
5. **Cuts** the chosen segments from the **full-resolution** source (re-downloaded
   per selected video, cached in `data_prep/full_videos/`) using **frame-accurate
   `ffmpeg`** (output-seek + near-lossless CRF re-encode + audio) into
   `data_prep/face_segments/seg_001.mp4`, `seg_002.mp4`, …

Writes `data_prep/selected_segments.json` (the full selection record, including
cluster sizes and which videos were used) for stage 4.

### Stage 4 — quality report (go / no-go)

**`data_prep/segment_quality_report.py`** — read-only; cuts/downloads nothing.

Prints a blunt **GO / NO-GO** verdict plus the evidence:

- **Pose distribution** — yaw binned into left / frontal / right.
- **Lighting & sharpness** — brightness / sharpness / face-area / quality stats.
- **Consistency** — how dominant the chosen appearance cluster is (want ≥ 70 %
  of scanned segments in one cluster).
- **Efficiency** — how many of the channel's videos were actually scanned vs.
  skipped entirely.

If it says NO-GO, it tells you *why* and which knob to turn (scan more
candidates, loosen/tighten clustering, relax frame gates, etc.).

### Run order & outputs

```bash
python data_prep/stage1_rank_by_thumbnail.py   # → candidates_ranked.json
python data_prep/stage2_scan_candidates.py     # → segments_scored.json, seg_reps/
python data_prep/stage3_build_dataset.py       # → face_segments/*.mp4, selected_segments.json
python data_prep/segment_quality_report.py     # → go / no-go report (console)
# then hand data_prep/face_segments/ to process_dataset.py (SyncTalk prep)
```

| Artifact | Produced by | Committed? |
|---|---|---|
| `candidates_ranked.json` | stage 1 | no (gitignored) |
| `cache/thumb_scores.json`, `cache/scan_*.json` | stages 1–2 | no |
| `thumbnails/`, `scan_videos/`, `full_videos/`, `seg_reps/` | stages 1–3 | no |
| `segments_scored.json`, `selected_segments.json` | stages 2–3 | no |
| `face_segments/*.mp4` | stage 3 | no (media) |
| the four `stage*.py` / `*_report.py` scripts | you | **yes** |

All generated data is `.gitignore`d — only the code is version-controlled.

### Tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `TAKE_TOP` | stage 1 | 150 | size of the candidate shortlist |
| `DETECT_CONFIDENCE` | stage 1 | 0.3 | thumbnail face-detect leniency |
| `SAMPLE_FPS` | stage 2 | 1.0 | frame sampling rate while scanning |
| `MIN_FACE_AREA` | stage 2 | 0.12 | min face size (fraction of frame) |
| `MAX_YAW_DEG` | stage 2 | 30 | max head turn accepted |
| `SCAN_MULTIPLIER` | stage 2 | 2.0 | how much to over-collect before stop |
| `TARGET_MINUTES` | stage 3 | 25 | final dataset length, then **stop** |
| `CLUSTER_DISTANCE` | stage 3 | 0.55 | cosine threshold for "same look" |
| `CUT_CRF` | stage 3 | 18 | cut re-encode quality (lower = better) |

---

## The avatar runtime

Once `data_prep/face_segments/` exists, the runtime turns it (plus a reference
image and voice samples) into a driveable avatar. Key entry points:

- **`realtime_avatar.py`** — webcam → AI face (LivePortrait) + neural mouth-sync
  (MuseTalk, with a Wav2Lip fallback), controllable over a socket from
  `control_gui.py`. Real-time, interactive.
- **`autonomous_avatar.py`** — a hands-off avatar loop (idle motion + speech)
  that runs without a webcam driver.
- **`avatar_studio.py`** — a higher-level "studio" front-end that wires the
  engines together (background, character, TTS, restoration).
- **`build_character.py`** — generates the reference character views.
- **`engines/`** — modular building blocks: TTS engines (Kokoro, Maya1,
  Chatterbox, edge-tts), lip-sync, and the realism stack.
- **`ai-face/`** — the photoreal pipeline: **Wav2Lip** (lip-sync) →
  **CodeFormer** (face restoration) → **Real-ESRGAN ×2** (detail upscale),
  with GFPGAN as a restorer fallback. Weights live under `ai-face/models/`
  (gitignored; fetched by `setup_models.py`).

### Text-to-speech options

| Engine | What it's for |
|---|---|
| **Kokoro** (`hexgrad/Kokoro-82M`) | fast local neural TTS, default |
| **Maya1** (`maya-research/maya1`) | expressive 3B speech-LLM, emotion/laughs via inline tags |
| **Chatterbox** (Resemble AI) | clone a *real* voice from a reference clip |
| **edge-tts** | cloud fallback, no GPU needed |

See `AVATAR_README.md` and `ai-face/README.md` for the runtime in depth.

---

## Hardware & environment

- **GPU:** NVIDIA **RTX 5060 Ti** (Blackwell architecture). Blackwell requires
  the **CUDA 12.8** PyTorch wheels — `cu128`, *not* the default `pip install
  torch` build and *not* `cu121`.
- **OS:** Windows 11 (PowerShell-first; a Bash shell is also available).
- **ffmpeg:** required for yt-dlp stream merging and for stage-3 cuts. It does
  **not** need to be on `PATH` — the harvester falls back to the binary shipped
  by `imageio-ffmpeg` automatically.

```bash
# PyTorch for Blackwell (RTX 50-series):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# RTX 30/40-series instead:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Installation

```bash
# 1. PyTorch FIRST, matched to your GPU (see above) — do NOT let another
#    package pull in a generic torch build.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt

# 3. Pin numpy back down (insightface/opencv may try to pull numpy 2.x, which
#    breaks mediapipe — see landmines below)
pip install "numpy==1.26.4"

# 4. Verify models & weights
python setup_models.py
```

The harvester specifically needs: `yt-dlp`, `mediapipe`, `opencv-python`,
`insightface`, `scikit-learn`, `numpy`, and `imageio-ffmpeg` — all in
`requirements.txt`. InsightFace downloads its `buffalo_l` model (~280 MB) to
`~/.insightface/` on first use.

---

## Dependency landmines (read before installing)

This stack has a few sharp edges, learned the hard way:

- **numpy 2.x breaks MediaPipe.** `insightface` and recent `opencv-python`
  wheels declare `numpy>=2`, but `mediapipe` requires `numpy<2`. **Pin
  `numpy==1.26.4`** and ignore the pip resolver warnings — at runtime all of
  numpy 1.26 / cv2 4.11 / mediapipe / insightface / sklearn import together
  fine.
- **Don't let anything reinstall torch.** Several optional packages
  (notably `chatterbox-tts`) pin `torch==2.6.0`, which **breaks the
  cu128/Blackwell GPU**. Install those with `--no-deps` and add their runtime
  deps manually. See the detailed comments in `requirements.txt`.
- **ffmpeg need not be on PATH.** The harvester resolves it via
  `imageio-ffmpeg`; if you *do* put a real ffmpeg on PATH it'll prefer that.
- **InsightFace runs on CPU** unless `onnxruntime-gpu` is installed. For the
  handful of embeddings stage 3 needs, CPU is fine (a minute or so total) and
  avoids disturbing the cu128 torch setup.

---

## Repository layout

```
TiktokBot/
├── data_prep/                     # ← the staged face harvester (this README's focus)
│   ├── stage1_rank_by_thumbnail.py
│   ├── stage2_scan_candidates.py
│   ├── stage3_build_dataset.py
│   ├── segment_quality_report.py
│   ├── thumbnails/  scan_videos/  full_videos/  seg_reps/  cache/   (all gitignored)
│   └── face_segments/             # final ~25 min dataset (gitignored media)
├── ai-face/                       # Wav2Lip + CodeFormer + Real-ESRGAN pipeline
│   ├── face_runtime.py  enhance_engine.py  tts_engine.py
│   ├── models/                    # weights (gitignored, fetched by setup)
│   └── Wav2Lip/
├── engines/                       # modular TTS / lip-sync / realism engines
├── harvester/                     # earlier streamer-footage motion harvester
├── realtime_avatar.py             # webcam-driven live avatar
├── autonomous_avatar.py           # hands-off avatar loop
├── avatar_studio.py               # studio front-end
├── build_character.py             # reference-view generator
├── control_gui.py                 # socket control panel
├── setup_models.py                # model/weight checker + downloader
├── requirements.txt               # dependency list (with landmine notes)
├── AVATAR_README.md               # avatar runtime docs
└── README.md                      # this file
```

> Large/generated content — model weights (`*.pth`, `*.onnx`, …), media
> (`*.mp4`, `*.wav`, …), and all the harvester's downloaded/derived folders —
> is `.gitignore`d. GitHub rejects files > 100 MB; weights are re-downloaded by
> setup, not committed.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every thumbnail scores `face=False` | Almost always a load failure, not the detector. Confirm `data_prep/thumbnails/<id>.jpg` is a real JPEG; the detector itself scores a clean frontal face ~0.9. |
| `yt-dlp` "ffmpeg not found" / can't merge | Install `imageio-ffmpeg` (the harvester auto-uses it) or put ffmpeg on PATH. |
| `ModuleNotFoundError: mediapipe` after install | numpy got upgraded to 2.x. Run `pip install "numpy==1.26.4"`. |
| Avatar looks blurry / "averaged" | Stage 3's consistency cluster is too loose or the dataset mixes looks. Lower `CLUSTER_DISTANCE`, re-run stage 4, and check the dominance %. |
| Stage 2 downloads forever | It early-stops at ~50 min of footage; if the channel is sparse it'll work through more of the 150 shortlist. Lower `SCAN_MULTIPLIER` to stop sooner. |
| Torch can't see the GPU | You installed a non-cu128 build. Reinstall from the `cu128` index (Blackwell). |

---

## Design principles

These are enforced throughout `data_prep/`:

- **Cheap signals before expensive ones.** Thumbnails (≈ free) gate which videos
  ever get downloaded; full-res cuts happen only for the final winners.
- **Stop early, everywhere.** `TAKE_TOP`, `SCAN_MULTIPLIER`, and `TARGET_MINUTES`
  each cap work at a stage boundary.
- **Cache everything.** Re-runs skip already-scored thumbnails and
  already-scanned videos.
- **Consistency over quantity.** A smaller, uniform dataset beats a larger,
  mixed one for avatar sharpness — stage 3 will reject good-but-different
  footage on purpose.
- **Fail soft.** Unavailable / private / deleted videos are skipped, never
  fatal. Every function is implemented — no placeholders.
- **Say what happened.** Stage 4 gives an honest go/no-go with the numbers
  behind it.

---

## Disclaimer & ethics

This project builds an avatar of **the channel owner's own face and voice**,
from **their own** publicly-uploaded footage, for their own content workflow.
Synthesizing a likeness of anyone without their explicit consent is harmful and,
in many places, illegal. Use it only on footage and voices you have the right to
use.
