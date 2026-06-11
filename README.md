# TiktokBot — Personal AI Avatar & Face-Dataset Pipeline

Build a **photorealistic talking-head avatar** of one person, driven entirely
from that person's own existing YouTube catalogue. The repo has two halves:

1. **`data_prep/`** — a staged harvester that mines a large, messy back-catalogue
   for the small amount of clean, *consistent* face footage an avatar actually
   needs, and stops as soon as it has enough.
2. **The avatar runtime** — neural TTS, lip-sync, and face restoration that turn
   that dataset (plus reference images and a voice clip) into a live or
   autonomous avatar.

The reference subject is the channel **[@ghaithabohlal](https://www.youtube.com/@ghaithabohlal)** —
~1,800 listed uploads of Arabic gold/forex market analysis. Crucially, these are
**not** talking-head videos: they're full-screen chart screen-shares with a
**small webcam inset of the presenter in the corner** (the face occupies only
~3–4 % of the frame). That single fact drives almost every design decision
below — most of the catalogue is unusable for a face dataset, and the usable
part has *small* faces, so the harvester is built to find the needle and reject
the haystack cheaply.

> **Status:** active personal project. `data_prep/` is the newest and most
> battle-tested subsystem (it has been run end-to-end against the real channel —
> see [the calibrated run](#a-real-run-on-the-reference-channel)). The avatar
> runtime evolves frequently.

---

## Table of contents

1. [The core problem](#the-core-problem)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [The staged face harvester](#the-staged-face-harvester)
   - [Stage 1 — rank by thumbnail (no downloads)](#stage-1--rank-by-thumbnail-no-downloads)
   - [Stage 2 — scan candidates at 720p](#stage-2--scan-candidates-at-720p)
   - [Stage 3 — the consistency filter + cut](#stage-3--the-consistency-filter--cut)
   - [Stage 4 — go / no-go report](#stage-4--go--no-go-report)
   - [Run it](#run-it)
   - [A real run on the reference channel](#a-real-run-on-the-reference-channel)
   - [Tuning — and the one knob that matters most](#tuning--and-the-one-knob-that-matters-most)
4. [The avatar runtime](#the-avatar-runtime)
5. [Hardware & environment](#hardware--environment)
6. [Installation](#installation)
7. [Dependency landmines](#dependency-landmines)
8. [Repository layout](#repository-layout)
9. [Troubleshooting](#troubleshooting)
10. [Design principles](#design-principles)
11. [Ethics](#ethics)

---

## The core problem

A sharp avatar needs **consistent** footage: same person, similar lighting,
similar camera, mostly frontal. Feed a model a grab-bag of mismatched clips and
it learns the *average* of all of them — a blurry, unstable face. So the dataset
problem is not "get a lot of footage," it's "get ~20–30 minutes of the *same*
look and stop."

The catalogue makes that both possible and painful:

- **Possible** — somewhere in ~1,800 videos there are tens of minutes of the
  presenter's webcam inset, all shot the same way.
- **Painful** — brute-forcing it (download + scan all ~1,800) would take days,
  cost hundreds of GB, and actively *hurt* the avatar by dragging in
  off-looking footage. And because the face is a tiny corner inset, naïve
  full-screen face heuristics reject nearly everything.

The harvester's whole philosophy: **spend compute in proportion to how promising
a video is, and stop the moment you have enough consistent footage.** Cheap
signals first (thumbnails), expensive signals last (full-res cuts), with a hard
early-stop at every stage.

---

## Architecture at a glance

```
                       ┌──────────────────────────────────────────────┐
                       │  data_prep/  — find the best face footage     │
                       │                                               │
  YouTube channel ───► │  stage1  rank ~1800 by THUMBNAIL  (no DL)     │
                       │     │      one yt-dlp call + tiny JPEGs        │
                       │     ▼      ~free, minutes                      │
                       │  stage2  scan shortlist @720p, 1 fps          │
                       │     │      downloads only candidates           │
                       │     ▼      early-stop @ ~50 min footage        │
                       │  stage3  CONSISTENCY FILTER → cut clips        │
                       │     │      insightface embed → cluster →        │
                       │     ▼      ONE dominant look → full-res ffmpeg │
                       │  stage4  go / no-go quality report             │
                       └───────────────────────────┬──────────────────┘
                                                   │  face_segments/  (~25–30 min)
                                                   ▼
                       ┌──────────────────────────────────────────────┐
                       │  avatar runtime                               │
                       │  • process_dataset.py    (SyncTalk prep)      │
                       │  • realtime_avatar.py / autonomous_avatar.py  │
                       │  • engines/   (TTS · lip-sync · restoration)  │
                       │  • ai-face/   (Wav2Lip · CodeFormer · ESRGAN) │
                       └──────────────────────────────────────────────┘
```

You run `data_prep/` **once** to bootstrap a dataset; the runtime consumes it
repeatedly.

---

## The staged face harvester

Four scripts, run in order, each strictly more expensive than the last. Every
stage **caches** its work (re-runs are nearly free) and honours an **early-stop**
so you never over-collect. Console output is prefixed `[STAGE1]` / `[STAGE2]` /
`[STAGE3]` / `[REPORT]`.

### Stage 1 — rank by thumbnail (no downloads)

**`data_prep/stage1_rank_by_thumbnail.py`**

- One `yt-dlp` flat-extraction call lists the whole channel — metadata only,
  **zero video downloads**.
- For each video it fetches just the **thumbnail** (~25 KB, straight from
  `i.ytimg.com/vi/<id>/hqdefault.jpg` — derivable from the id, no per-video page
  load).
- **MediaPipe** face detection scores each thumbnail's `face_likelihood` from
  detector confidence, bounding-box area, and a **frontal-ness** proxy (eye/nose
  symmetry → yaw & roll), nudged by a weak **title-keyword** signal (Arabic +
  English: *live/analysis/interview* up, *price-update/signal/chart* down).
- Writes `candidates_ranked.json` (every video, best first) and shortlists the
  top `TAKE_TOP` (default **150**) for stage 2.

> Thumbnails are the creator's *custom* uploads, which often feature a large
> face even when the video itself is a chart. That's fine — stage 1 is only a
> cheap *prioritiser*; stage 2 does the real verification. Scores cache to
> `cache/thumb_scores.json`.

### Stage 2 — scan candidates at 720p

**`data_prep/stage2_scan_candidates.py`**

- Downloads each shortlisted candidate at **≤720p** (cached in `scan_videos/`),
  in ranked order.
- Samples at **1 fps** and gates each frame with the **full-range** face
  detector (`model_selection=1` — short-range misses the small inset):
  - face bbox **> 3 %** of frame (`MIN_FACE_AREA` — see [the knob that
    matters](#tuning--and-the-one-knob-that-matters-most)),
  - **yaw within ±30°**,
  - sharpness above a (deliberately low) floor — small insets upscale soft,
  - brightness in a sane band.
- Groups passing frames into **segments** (≥ 2 s, bridging dropouts ≤ 1 s),
  scores each `avg_quality × duration`, and saves a **representative face crop**
  per segment to `seg_reps/` for stage 3.
- Writes `segments_scored.json` (ranked — **not yet cut**). Per-video results
  cache to `cache/scan_<id>.json`.
- **Early-stops** once it has ~`TARGET_MINUTES × SCAN_MULTIPLIER` (≈ 50 min) of
  good footage — deliberately over-collecting so stage 3 can *reject*
  inconsistent looks rather than be forced to keep them.

### Stage 3 — the consistency filter + cut

**`data_prep/stage3_build_dataset.py`** — *the script that makes the avatar
sharp instead of blurry.* This is the heart of the whole repo.

1. Compute a 512-d **InsightFace** (`buffalo_l`) embedding for every segment's
   representative face.
2. **Cluster** the embeddings by cosine distance (agglomerative, threshold-based
   — no need to know the number of looks up front). This groups segments by
   *appearance*: look, lighting, camera, framing.
3. Pick the **largest cluster by total duration** — the dominant, most-consistent
   version of the subject across the catalogue.
4. From that cluster *only*, take the **highest-quality segments until
   `TARGET_MINUTES` (default 25) is reached, then stop.** Outlier-looking
   segments are rejected **even if individually excellent** — mixing looks is
   exactly what blurs an avatar.
5. **Cut** the winners from the **full-resolution** source (re-downloaded per
   selected video into `full_videos/`) with **frame-accurate `ffmpeg`**
   (output-seek + near-lossless CRF re-encode + audio) → `face_segments/seg_001.mp4`, …

Writes `selected_segments.json` (the full record: cluster sizes, videos used,
per-segment metrics) for stage 4.

### Stage 4 — go / no-go report

**`data_prep/segment_quality_report.py`** — read-only; cuts/downloads nothing.

Prints a blunt **GO / NO-GO** verdict with the evidence: yaw **pose
distribution**, **lighting/sharpness/area** stats, **cluster dominance** (want
≥ 70 % of scanned segments in one cluster), and **efficiency** (videos scanned
vs. skipped). On NO-GO it names the reason and the knob to turn.

### Run it

```bash
python data_prep/stage1_rank_by_thumbnail.py   # → candidates_ranked.json
python data_prep/stage2_scan_candidates.py     # → segments_scored.json, seg_reps/
python data_prep/stage3_build_dataset.py       # → face_segments/*.mp4, selected_segments.json
python data_prep/segment_quality_report.py     # → go / no-go (console)
# then hand data_prep/face_segments/ to process_dataset.py (SyncTalk prep)
```

| Artifact | Stage | In git? |
|---|---|---|
| the four `stage*.py` / `*_report.py` scripts | you | **yes** |
| `candidates_ranked.json`, `segments_scored.json`, `selected_segments.json` | 1–3 | no |
| `thumbnails/`, `scan_videos/`, `full_videos/`, `seg_reps/`, `cache/` | 1–3 | no |
| `face_segments/*.mp4` | 3 | no (media) |

Everything generated is `.gitignore`d; only code is versioned.

### A real run on the reference channel

The numbers below are from an actual end-to-end run — they show the funnel
doing its job (and why the area gate matters):

| Stage | Result |
|---|---|
| **1** | **1,809** videos listed; **~1,086** showed a face in the thumbnail; top 150 shortlisted. **No videos downloaded.** |
| **2** (correct gates) | Early-stopped after scanning **6** videos → **74 min** of good footage in **15 segments**. |
| **3** | 12/15 segments embedded → **2 appearance clusters**; dominant cluster = **11 segments / 57 min**; selected **~32 min** from **2** source videos; cut to full-res clips. |

So out of ~1,800 videos, the pipeline downloaded **6** at 720p and **2** at full
resolution — and *skipped the other ~1,800 entirely.* That's the design goal:
hours, not days; a few GB, not hundreds.

> **Calibration lesson (baked into the defaults).** The first run used a 12 %
> `MIN_FACE_AREA` (sensible for full-screen talking heads) and found almost
> nothing — 142 of 146 scanned videos yielded **zero** segments, because the
> webcam inset is only ~3–4 % of the frame. Dropping the gate to **3 %** and
> switching to the **full-range** detector turned 1.4 min of footage into 74.
> Pure-chart frames still register < 1 %, so the 3 % gate cleanly rejects them
> while keeping the real inset. (The companion `harvester/clip_quality_filter.py`
> independently learned the same 3 % lesson.)

### Tuning — and the one knob that matters most

| Constant | File | Default | Effect |
|---|---|---|---|
| **`MIN_FACE_AREA`** | **stage 2** | **0.03** | **The critical, footage-dependent knob.** 0.03 fits a small corner webcam inset; raise toward 0.12 only for genuine full-screen talking heads. Too high → empty dataset; too low → chart false-positives leak in. |
| `TAKE_TOP` | stage 1 | 150 | candidate shortlist size |
| `DETECT_CONFIDENCE` | stage 1/2 | 0.3 / 0.5 | face-detect leniency |
| `SAMPLE_FPS` | stage 2 | 1.0 | scan sampling rate |
| `MAX_YAW_DEG` | stage 2 | 30 | max head turn accepted |
| `MIN_SHARPNESS` | stage 2 | 20 | sharpness floor (low — small crops are soft) |
| `SCAN_MULTIPLIER` | stage 2 | 2.0 | how much to over-collect before stopping |
| `TARGET_MINUTES` | stage 3 | 25 | final dataset length, then **stop** |
| `CLUSTER_DISTANCE` | stage 3 | 0.55 | cosine threshold for "same look" (lower = stricter consistency) |
| `CUT_CRF` | stage 3 | 18 | cut re-encode quality (lower = better) |

---

## The avatar runtime

Once `data_prep/face_segments/` exists, the runtime turns it (plus a reference
image and voice samples) into a driveable avatar:

- **`realtime_avatar.py`** — webcam → AI face (LivePortrait) + neural mouth-sync
  (MuseTalk, Wav2Lip fallback), controllable over a socket from `control_gui.py`.
- **`autonomous_avatar.py`** — hands-off idle-motion + speech loop, no webcam.
- **`avatar_studio.py`** — higher-level front-end wiring background, character,
  TTS, and restoration together.
- **`build_character.py`** — generates reference character views.
- **`engines/`** — modular TTS / lip-sync / realism blocks.
- **`ai-face/`** — the photoreal chain: **Wav2Lip** (lip-sync) → **CodeFormer**
  (restoration) → **Real-ESRGAN ×2** (detail upscale), GFPGAN as fallback. This
  chain matters here because the source faces are *small* insets — restoration
  and upscaling do real work. Weights live under `ai-face/models/` (gitignored;
  fetched by `setup_models.py`).

**TTS options:** **Kokoro** (`hexgrad/Kokoro-82M`, fast local default) ·
**Maya1** (`maya-research/maya1`, expressive, emotion via inline tags) ·
**Chatterbox** (Resemble AI, clone a real voice from a reference clip) ·
**edge-tts** (cloud fallback). See `AVATAR_README.md` and `ai-face/README.md`.

---

## Hardware & environment

- **GPU:** NVIDIA **RTX 5060 Ti** (Blackwell). Blackwell needs the **CUDA 12.8**
  PyTorch wheels — `cu128`, **not** the default `pip install torch` and **not**
  `cu121`.
- **OS:** Windows 11 (PowerShell-first; Bash also available).
- **ffmpeg:** required for yt-dlp stream-merging and stage-3 cuts. It does **not**
  need to be on `PATH` — the harvester auto-falls back to the binary shipped by
  `imageio-ffmpeg`.

```bash
# Blackwell / RTX 50-series:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# RTX 30/40-series:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Installation

```bash
# 1. PyTorch FIRST, matched to your GPU (above). Do NOT let another package
#    pull in a generic torch build.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt

# 3. Re-pin numpy (insightface / new opencv try to pull numpy 2.x, which breaks
#    mediapipe — see landmines)
pip install "numpy==1.26.4"

# 4. Verify models & weights
python setup_models.py
```

Harvester-specific deps: `yt-dlp`, `mediapipe`, `opencv-python`, `insightface`,
`scikit-learn`, `numpy`, `imageio-ffmpeg` — all in `requirements.txt`. InsightFace
downloads `buffalo_l` (~280 MB) to `~/.insightface/` on first use.

---

## Dependency landmines

Hard-won, all real:

- **numpy 2.x breaks MediaPipe.** `insightface` and recent `opencv-python` declare
  `numpy>=2`, but `mediapipe` needs `numpy<2`. **Pin `numpy==1.26.4`** and ignore
  the pip resolver warnings — at runtime numpy 1.26 / cv2 4.11 / mediapipe /
  insightface / sklearn import together fine.
- **Don't let anything reinstall torch.** Some optional packages (notably
  `chatterbox-tts`) pin `torch==2.6.0`, which **breaks cu128/Blackwell**. Install
  those `--no-deps` and add runtime deps by hand (see `requirements.txt`).
- **ffmpeg need not be on PATH** — resolved via `imageio-ffmpeg`; a real ffmpeg on
  PATH is preferred if present.
- **InsightFace runs on CPU** unless `onnxruntime-gpu` is installed. For the
  handful of embeddings stage 3 needs, CPU is fine (~a minute) and avoids
  disturbing the cu128 torch setup.

---

## Repository layout

```
TiktokBot/
├── data_prep/                     # the staged face harvester
│   ├── stage1_rank_by_thumbnail.py
│   ├── stage2_scan_candidates.py
│   ├── stage3_build_dataset.py
│   ├── segment_quality_report.py
│   ├── thumbnails/ scan_videos/ full_videos/ seg_reps/ cache/   (gitignored)
│   └── face_segments/             # final ~25–30 min dataset (gitignored media)
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
├── requirements.txt               # deps (with landmine notes)
├── AVATAR_README.md               # avatar runtime docs
└── README.md                      # this file
```

> Model weights (`*.pth`, `*.onnx`, …), media (`*.mp4`, `*.wav`, …), and all the
> harvester's downloaded/derived folders are `.gitignore`d. GitHub rejects files
> > 100 MB; weights are re-downloaded by setup, not committed.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **Stage 2 finds almost no footage** | Your faces are smaller than `MIN_FACE_AREA`. For a corner webcam inset, set it to `0.03` and use the full-range detector (`model_selection=1`). This is the #1 gotcha. |
| Every thumbnail scores `face=False` | A load failure, not the detector — confirm `thumbnails/<id>.jpg` is a real JPEG (the detector scores a clean frontal face ~0.9). |
| `yt-dlp` "ffmpeg not found" / can't merge | `pip install imageio-ffmpeg` (auto-used) or put ffmpeg on PATH. |
| `ModuleNotFoundError: mediapipe` after install | numpy got upgraded to 2.x → `pip install "numpy==1.26.4"`. |
| Avatar looks blurry / "averaged" | Consistency cluster too loose, or dataset mixes looks. Lower `CLUSTER_DISTANCE`, re-run stage 4, check the dominance %. |
| Stage 2 scans far more videos than expected | The channel is sparse in face footage; it works the shortlist until it hits the footage target. Lower `SCAN_MULTIPLIER` to stop sooner. |
| Torch can't see the GPU | Non-cu128 build installed → reinstall from the `cu128` index. |

---

## Design principles

Enforced throughout `data_prep/`:

- **Cheap signals before expensive ones.** Thumbnails (≈ free) decide what ever
  gets downloaded; full-res cuts happen only for the final winners.
- **Stop early, everywhere.** `TAKE_TOP`, `SCAN_MULTIPLIER`, `TARGET_MINUTES`
  each cap work at a stage boundary.
- **Cache everything.** Re-runs skip scored thumbnails and scanned videos.
- **Consistency over quantity.** Stage 3 will reject good-but-different footage on
  purpose — a smaller uniform dataset beats a larger mixed one.
- **Calibrate to the footage, not the assumption.** The gates are tuned to what
  the channel *actually* is (small inset), not to what a talking-head channel
  would be. When you point this at a new source, re-check `MIN_FACE_AREA` first.
- **Fail soft.** Unavailable / private / deleted videos are skipped, never fatal.
  Every function is implemented — no placeholders.
- **Say what happened.** Stage 4 gives an honest go/no-go with the numbers behind
  it.

---

## Ethics

This project builds an avatar of **the channel owner's own face and voice**, from
**their own** publicly-uploaded footage, for their own content workflow.
Synthesizing anyone's likeness without their explicit consent is harmful and, in
many places, illegal. Use it only on footage and voices you have the right to use.
