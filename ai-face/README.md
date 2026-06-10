# AI Talking Face — Real-Time TikTok Live System

A fully local, no-paid-API talking-face engine for TikTok Live streaming.
You type text → it's spoken with Microsoft edge-tts → Wav2Lip lip-syncs your
character image to the audio → frames stream into a virtual camera → OBS picks
it up → TikTok Live Studio goes live.

- **TTS:** edge-tts (free, no API key) — voice `en-US-ChristopherNeural`
- **Lip sync:** Wav2Lip (`wav2lip_gan.pth`)
- **Face restoration (clarity):** GFPGAN (`GFPGANv1.4.pth`) — makes the lips
  sharp and the face look HD/human instead of soft and blurry
- **Virtual camera:** pyvirtualcam (OBS Virtual Camera backend)
- **Audio:** the spoken voice plays out the default device for OBS Desktop Audio
- **Control:** terminal panel or dark-themed tkinter GUI
- **Tested target:** Windows 11, NVIDIA GPU (verified on RTX 5060 Ti / Blackwell)

---

## How it works

```
control panel  --socket(127.0.0.1:9999)-->  realtime_face.py
                                              |
                            text -> edge-tts -> speech_output.wav
                                              |
                            wav + character.jpg -> Wav2Lip -> result_voice.mp4
                                              |
                            decode frames -> GFPGAN restore (sharp lips)
                                              |
                            full clip queued, then played at 25fps:
                              video -> pyvirtualcam     (OBS Video Capture)
                              audio -> default speaker   (OBS Desktop Audio)
                                              |
                                           OBS -> TikTok Live Studio
```

Each clip is **fully rendered and enhanced before it plays**, so the lips never
stutter, and the audio is started in sync with the first frame. A queue lets you
type ahead: lines stack up while the face is still speaking.

### Speed: persistent in-process engine

`face_runtime.py` loads Wav2Lip + the s3fd detector **once** and keeps them
resident in VRAM, detects the character's face a single time, and runs inference
fully in memory — no per-line subprocess, no model reload, no video files. cuDNN
autotuning + fixed (padded) batch sizes mean the conv kernels are tuned once at
startup and reused. The result: after a one-time warm-up at launch (~1 minute of
model loading + GPU autotuning), each line renders in roughly **real time**.

Typical per-line cost on an RTX 5060 Ti (≈4s of speech, 720×720, enhance on):
TTS ~1.5s + Wav2Lip ~0.2s + GFPGAN ~6s ≈ **8s to render**, then it plays. Type
ahead and lines queue seamlessly.

### Clarity / "more human" quality (GFPGAN)

Wav2Lip generates the mouth at only 96×96 and pastes it back, so the lips look
soft. GFPGAN restores the face — sharp lips, skin texture, crisp eyes — and
upscales it. The whole clip is enhanced in fixed batches; because the head is
static the face is detected and the blend mask is built **once**, collapsing
GFPGAN's expensive per-frame paste into a single warp+blend per frame.

Toggle/tune with constants at the top of the files:
- `ENHANCE` (realtime_face.py) — turn restoration on/off
- `UPSCALE`, `FIDELITY`, `BATCH_SIZE`, `USE_BF16` (enhance_engine.py)

> Set `ENHANCE = False` for the fastest (but softer) output. Pure fp16 is
> intentionally disabled (`USE_BF16` is used instead) — GFPGAN's StyleGAN
> decoder overflows in fp16 and produces a flat blob; bf16 keeps fp32's range.

---

## Setup (step by step)

### 1. Clone Wav2Lip into this folder
```powershell
cd ai-face
git clone https://github.com/Rudrabha/Wav2Lip
```

### 2. Download the Wav2Lip checkpoint
Download **`wav2lip_gan.pth`** (~435 MB) and place it in
`ai-face/Wav2Lip/checkpoints/wav2lip_gan.pth`. The original host is often down;
a working HuggingFace mirror:
```powershell
curl -L -o Wav2Lip/checkpoints/wav2lip_gan.pth `
  https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth
```

### 3. Download the face-detection model
Download **`s3fd.pth`** (~86 MB) into
`ai-face/Wav2Lip/face_detection/detection/sfd/s3fd.pth`:
```powershell
curl -L -o Wav2Lip/face_detection/detection/sfd/s3fd.pth `
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
```

### 3b. Download the GFPGAN face-restoration model (for clear, human lips)
Download **`GFPGANv1.4.pth`** (~349 MB) into `ai-face/models/`:
```powershell
mkdir models
curl -L -o models/GFPGANv1.4.pth `
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```
(Skip this only if you set `ENHANCE = False` in `realtime_face.py`.)

### 4. Install ffmpeg
```powershell
winget install ffmpeg
# or:  choco install ffmpeg
```
Restart your terminal afterwards so `PATH` refreshes.

### 5. Install Python dependencies
```powershell
pip install -r requirements.txt
```
Or run the guided installer/checker:
```powershell
python setup.py
```

> **Wav2Lip's own requirements (IMPORTANT — do NOT use its `requirements.txt`):**
> Wav2Lip's pinned versions (torch 1.1.0, numpy 1.17, librosa 0.7) are ancient
> and will not install or run on modern Python / modern GPUs. Install current
> versions instead:
>
> ```powershell
> pip install librosa numba scipy tqdm
> ```
>
> **PyTorch — pick the build for YOUR GPU architecture:**
>
> | GPU | Architecture | Install command |
> |-----|--------------|-----------------|
> | RTX 50-series (5060/5070/5080/5090) | Blackwell (sm_120) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
> | RTX 30/40-series (3060, 4070, ...) | Ampere/Ada | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
>
> A mismatched build installs fine but crashes at inference with
> `no kernel image is available for execution on the device`. Verify with:
> ```powershell
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```
>
> **GFPGAN enhancement deps:**
> ```powershell
> pip install gfpgan
> ```
> On modern torchvision (≥0.17) `basicsr` fails to import with
> `No module named 'torchvision.transforms.functional_tensor'`. This repo's
> install patches it automatically; if you hit it on a fresh machine, edit
> `site-packages/basicsr/data/degradations.py` line 8 to:
> ```python
> from torchvision.transforms.functional import rgb_to_grayscale
> ```
>
> **Wav2Lip code patches (already applied in this repo's clone):** the bundled
> `Wav2Lip` here has been patched for modern Python/torch:
> - `audio.py` — `librosa.filters.mel(...)` switched to keyword args; restored
>   removed `np.float`/`np.int` aliases.
> - `inference.py` and `face_detection/.../sfd_detector.py` — `torch.load(...)`
>   now passes `weights_only=False` (required on torch ≥ 2.6).
>
> If you re-clone Wav2Lip fresh, you must re-apply these patches.

### 6. Add your character image
Replace **`character.jpg`** with your AI face:
- 512×512 pixels
- front-facing
- neutral expression (closed/relaxed mouth works best)

### 7. Install + enable the OBS Virtual Camera
`pyvirtualcam` on Windows uses the **OBS Virtual Camera** backend.
1. Install **OBS Studio**.
2. Open OBS once and click **Start Virtual Camera** at least once so the
   virtual camera device gets registered on Windows. You can stop it again;
   it just needs to exist.

---

## Running

Open two terminals in the `ai-face` folder.

**Terminal 1 — the face engine:**
```powershell
python realtime_face.py
```

**Terminal 2 — the control panel (pick one):**
```powershell
python gui_panel.py        # dark-themed GUI with quick-phrase buttons
# or
python control_panel.py    # simple terminal input loop
```

Then in OBS:
1. **Add Source → Video Capture Device** → select the **virtual camera**
   created by the engine (the face video).
2. **Add the audio:** in OBS Audio Mixer make sure **Desktop Audio** is active
   (Settings → Audio → Desktop Audio = your default speakers). The engine plays
   the spoken voice out the default device, and OBS captures it there. Use
   headphones, or you can mute desktop audio *monitoring* while still streaming
   it, to avoid hearing it yourself.
3. Compose your scene and **go live on TikTok Live Studio**.

Type text in the control panel and the face speaks it. A **green dot** in the
top-right of the video means it's currently speaking; **gray** means idle.

> **Startup warm-up:** on launch the engine loads the models and autotunes GPU
> kernels (~1 minute). Wait for `[✓] Worker: TTS + lip-sync pipeline ready` and
> `[✓] Worker: warm-up complete` before typing. After that each ~4s line renders
> in ~8s and plays. Set `ENHANCE = False` in `realtime_face.py` for near-instant
> (but softer) output.

---

## File overview

| File | Purpose |
|------|---------|
| `realtime_face.py` | Main engine: socket server + worker + virtual-cam loop (audio-clock synced) + synced audio |
| `face_runtime.py` | Persistent in-process Wav2Lip + s3fd (models resident, face locked) |
| `enhance_engine.py` | GFPGAN face restoration — batched fast path (`enhance_frames()`) for sharp, human lips |
| `tts_engine.py` | edge-tts → mp3 → 16 kHz mono wav (`speak()`) |
| `control_panel.py` | Terminal control panel |
| `gui_panel.py` | tkinter GUI control panel |
| `setup.py` | Dependency installer + environment checklist |
| `requirements.txt` | pip packages |
| `character.jpg` | Your face image (placeholder — replace it) |

All tunable values (paths, voice, resolution, port) live as **CONSTANTS at the
top of each file** so they're easy to customize.

---

## Troubleshooting

- **`ffmpeg not found`** — install it (step 4) and restart the terminal.
- **Virtual camera error** — install OBS and click *Start Virtual Camera* once
  so the device exists, then re-run `realtime_face.py`.
- **Wav2Lip inference failed** — confirm the checkpoint and `s3fd.pth` are in
  place and that a CUDA build of PyTorch is installed inside `Wav2Lip/`.
- **`edge-tts produced no audio`** — edge-tts needs an internet connection to
  reach Microsoft's voices (it's still free and key-less).
- **Face not detected** — use a clearer, front-facing 512×512 image.
- **Voice not in sync with the lips** — the engine slaves the video to the
  audio clock, so it won't *drift* over a sentence, but a small **constant**
  offset can come from audio-device latency or OBS. Fix it in two places:
  1. **In the engine:** tune `AUDIO_OFFSET` at the top of `realtime_face.py`.
     Lips move *before* the voice → make it more negative (e.g. `-0.10`); you
     hear the voice *before* the lips move → make it more positive (`+0.10`).
  2. **In OBS:** right-click the **Desktop Audio** source → *Advanced Audio
     Properties* → set a **Sync Offset** (ms) to line audio up with the camera.
  Note Wav2Lip renders the video ~0.1s shorter than the audio, so the very last
  syllable may finish on a closed mouth — this is normal and barely noticeable.

---

## Notes

- No paid APIs anywhere. No OpenAI, no ElevenLabs, no cloud TTS keys.
- edge-tts is free and needs no API key — just `pip install edge-tts`.
- The engine forces `CUDA_VISIBLE_DEVICES=0` for Wav2Lip to use the GPU.
- The Wav2Lip subprocess uses `sys.executable`, so it runs in the same Python
  environment as the engine.
