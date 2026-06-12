# Make a WHITE version of Haddan: SDXL img2img re-renders Haddan's own photos as a
# white/Caucasian man, keeping his face structure but changing skin/ethnicity.
# Output -> haddan_white/ (used as the face-swap source = "white Haddan").
import os, sys, glob, torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
import insightface

PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJECT, "haddan_white"); os.makedirs(OUT, exist_ok=True)
STRENGTH = float(os.environ.get("WH_STRENGTH", "0.5"))

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
app = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

PROMPT = ("RAW photo, photorealistic headshot of a WHITE caucasian man, fair light "
          "skin, gray hair, a FULL THICK DENSE well-groomed salt-and-pepper gray "
          "beard covering the cheeks, jaw AND chin with rich detailed individual "
          "beard hairs (a proper full beard, not patchy), the mustache neatly "
          "trimmed just above the lip so the lips and mouth stay clearly VISIBLE, "
          "same face shape, looking at camera, soft studio light, plain background, "
          "ultra detailed realistic beard hair texture and skin pores, 85mm")
NEG_EXTRA = ", mustache covering lips, hidden mouth, bushy moustache over the lips"
NEG = ("mustache covering lips, hidden mouth, dark skin, tan, cartoon, anime, cgi, plastic, blurry, deformed, multiple "
       "people, hat, watermark, text")

# pick the clearest frontal Haddan photos (detectable, largest face)
cands = []
for p in sorted(glob.glob(os.path.join(PROJECT, "Haddan", "*.png")) +
                glob.glob(os.path.join(PROJECT, "Haddan", "*.jpg")))[:14]:
    im = cv2.imread(p)
    if im is None:
        continue
    faces = app.get(cv2.resize(im, (640, 640)))
    if faces:
        cands.append((max(f.bbox[2]-f.bbox[0] for f in faces), p))
cands.sort(reverse=True)
n = 0
for _, p in cands[:4]:
    img = Image.open(p).convert("RGB").resize((1024, 1024))
    g = torch.Generator("cuda").manual_seed(123 + n)
    out = pipe(PROMPT, negative_prompt=NEG, image=img, strength=STRENGTH,
               num_inference_steps=40, guidance_scale=6.5, generator=g).images[0]
    out.save(os.path.join(OUT, f"wh_{n:02d}.png")); n += 1
    print(f"[WH] saved wh_{n-1:02d} (strength {STRENGTH})", flush=True)
print(f"[WH] DONE {n} white-Haddan images", flush=True)
