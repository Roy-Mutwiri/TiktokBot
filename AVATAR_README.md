# Autonomous AI Avatar

A fully local, autonomous AI streamer avatar that learns **real human movements**
from real streamer videos automatically (zero manual recording) and lip-syncs
speech perfectly on top of those movements.

```
real streamer videos
      │  (harvester, run ONCE)
      ▼
motion_clips/  ── labelled 2-4s clips per behaviour (nod, look_left, blink, ...)
      │
      ▼  (autonomous_avatar.py, every stream)
behaviour engine → driving frame (real human motion)
      → LivePortrait  (animate the AI face with that motion)
      → Wav2Lip       (sync the mouth to TTS speech)
      → CodeFormer    (photoreal restore — sharpens the soft 96px mouth/skin)
      → Real-ESRGAN   (optional x2 detail pass, off by default)
      → enhance       (studio bg, ticker, LIVE badge, polish)
      → virtual camera → OBS → TikTok Live
```

Voice options (all local HuggingFace, with automatic fallback):
- **Maya1** — expressive 3B speech-LLM that **laughs, sighs and emotes** via inline
  tags (`<laugh>`, `<sigh>`, `<gasp>`…). Default for the autonomous/OBS avatar.
- **Chatterbox** — clone a **real person's voice** from a ~10s clip (`AVATAR_CLONE_REF`).
- **Kokoro-82M** — fast, natural, low-latency. Default for the realtime webcam studio.
- **edge-tts** — cloud voice, final fallback.

Write emotion tags right in the speech text, e.g. `Haha <laugh> did you see that?`
Non-Maya backends strip the tags automatically so they're never read aloud.

---

## Two commands, ever

### 1. Build the motion library (run once, ~20-40 min)
```powershell
# Edit harvester/download_streamers.py -> STREAMER_SOURCES (add real channel URLs)
python harvester/run_harvester.py
```
Downloads streamer footage → extracts labelled motion clips with MediaPipe →
filters quality + de-duplicates → writes `motion_clips/`.

### 2. Go live (every stream)
```powershell
python autonomous_avatar.py
```
- `T` — type text for the avatar to speak
- `M` — mute / unmute audio
- `R` — force a random reaction
- `Q` — quit

Then in OBS: **Add Source → Video Capture Device → OBS Virtual Camera**, and
enable **Desktop Audio** so the voice is captured.

---

## ⚠️ Important: LivePortrait must be installed for head motion

LivePortrait is **not bundled** and was **not found** on this machine. Without
it the avatar still runs and lip-syncs, but the head stays static (the engine
runs in a clearly-logged FALLBACK mode). To enable real motion driving:

```powershell
# place it as a sibling of TiktokBot/, or inside it, or set LIVEPORTRAIT_PATH
git clone https://github.com/KwaiVGI/LivePortrait
# then download its pretrained weights per its README
```

`engines/liveportrait_engine.py` searches `../LivePortrait`, `./LivePortrait`,
and `$LIVEPORTRAIT_PATH`. The reenactment math is implemented against the real
KwaiVGI wrapper API; validate it once LivePortrait is present.

## Paths note

Wav2Lip and `character.jpg` live under `ai-face/` in this project. The engines
auto-detect them (`engines/wav2lip_engine.py`, `autonomous_avatar.py` check both
the project root and `ai-face/`). The Wav2Lip checkpoint used is
`ai-face/Wav2Lip/checkpoints/wav2lip_gan.pth`.

---

## Components

| File | Role |
|------|------|
| `harvester/download_streamers.py` | yt-dlp channel downloader (edit `STREAMER_SOURCES`) |
| `harvester/motion_extractor.py` | MediaPipe FaceMesh metrics → behaviour-labelled clips |
| `harvester/clip_quality_filter.py` | Quality checks + dedup, approves `clip_NNN.mp4` |
| `harvester/run_harvester.py` | One-command pipeline (download→extract→filter→report) |
| `engines/motion_library.py` | Preloads all clips into RAM |
| `engines/behavior_engine.py` | Weighted state machine (IDLE/SPEAKING/REACTING) |
| `engines/liveportrait_engine.py` | Animate AI face with driving motion (fallback-safe) |
| `engines/wav2lip_engine.py` | Real-time mouth-only lip-sync |
| `engines/face_restore_engine.py` | **CodeFormer** photoreal restore (HF), GFPGAN fallback |
| `engines/upscale_engine.py` | **Real-ESRGAN** x2 detail pass (HF), optional |
| `engines/tts_stream_engine.py` | Voice backends: **Maya1** / **Chatterbox** / **Kokoro** (HF) / edge → audio + synced mouth feed |
| `engines/maya1_tts.py` | **Maya1** expressive speech-LLM (laughs/emotion tags) + SNAC |
| `engines/chatterbox_tts.py` | **Chatterbox** real-voice cloning from a reference clip |
| `engines/enhance_engine.py` | Studio bg, ticker, badges, vignette, shake |
| `autonomous_avatar.py` | Main runtime loop |

## Realism knobs (HuggingFace models)

Three HF-backed stages make the avatar read as real. All are env-toggleable:

| Env var | Default | Effect |
|---------|---------|--------|
| `AVATAR_RESTORE` | `1` | CodeFormer face restoration after Wav2Lip (the big realism win — sharp mouth/skin). |
| `AVATAR_RESTORE_INTERVAL` | `2` | Restore every Nth frame, reuse between. `1` = sharpest lip-sync/slowest; higher = more fps. |
| `AVATAR_RESTORE_FIDELITY` | `0.7` | CodeFormer `w`: lower = more invented detail, higher = closer to the (blurry) input. |
| `AVATAR_TTS` | `maya1` (autonomous) / `auto` (studio) | Voice backend: `maya1` (expressive, laughs), `chatterbox` (clone a real voice), `kokoro` (fast natural), `edge` (cloud), `auto`. |
| `AVATAR_KOKORO_VOICE` | `am_michael` | Kokoro voice (e.g. `am_adam`, `bm_george`, `af_heart`). |
| `AVATAR_MAYA_DESC` | warm male host | Maya1 voice *design* in plain English (age, tone, accent). |
| `AVATAR_CLONE_REF` | — | Path to a ~10s WAV of a real voice for Chatterbox to clone. |
| `AVATAR_CLONE_EXAGGERATION` | `0.6` | Chatterbox emotional intensity (0.3 calm … 1.0 very emotive). |
| `AVATAR_UPSCALE` | `0` | Real-ESRGAN x2 detail pass. **Off** — ~160ms/frame; enable for max crispness at lower fps. |

Model weights (auto-downloaded / vendored): `ai-face/models/codeformer.pth`,
`ai-face/models/RealESRGAN_x2.pth`, Kokoro from `hexgrad/Kokoro-82M`, and
facexlib's detector/parser into the HF cache. CodeFormer arch is vendored in
`engines/codeformer_arch/`.

## Performance reality check

Per-frame on this hardware, LivePortrait (~per-frame encode + warp) **plus**
per-frame Wav2Lip is heavy; hitting a true 25fps (40ms/frame) with both active
is unlikely. The loop self-reports FPS and per-stage latency every ~12s and
warns on slow frames. If it can't keep 25fps, options: run LivePortrait at a
lower internal rate, or disable one stage. Behaviour, TTS sync, enhance, and the
harvester all run comfortably; LivePortrait is the cost center (and currently in
fallback until you install it).

## No paid APIs

Maya1, Chatterbox, Kokoro-82M + edge-tts (all free, no key), CodeFormer,
Real-ESRGAN, Wav2Lip, LivePortrait, MediaPipe, yt-dlp — all local HuggingFace /
open models.
