# Generate a photorealistic WHITE-MAN source face with SDXL, to use as the
# face-swap source identity. One consistent identity (fixed seed) + a couple of
# slight variations for robustness. The swap then makes the avatar a consistent
# white man with NO color hacks.
import os, sys
import torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_src")
os.makedirs(OUT, exist_ok=True)

print("[GEN] loading SDXL (first run downloads ~7GB)...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

PROMPT = ("RAW photo, photorealistic headshot portrait of a white caucasian man, "
          "early 30s, short light-brown hair, neatly trimmed short beard, neutral "
          "friendly expression, looking straight at the camera, even soft studio "
          "lighting, plain light-gray background, sharp focus, natural detailed "
          "skin texture, 85mm portrait photograph, high detail")
NEG = ("cartoon, anime, illustration, painting, cgi, 3d render, plastic skin, "
       "waxy, blurry, out of focus, deformed, distorted face, asymmetric, "
       "multiple people, hat, sunglasses, watermark, text, dark, low light")

# one base identity + 2 small variations (same seed family) for robust embedding
for i, seed in enumerate([12345, 12346, 12347]):
    g = torch.Generator("cuda").manual_seed(seed)
    img = pipe(PROMPT, negative_prompt=NEG, num_inference_steps=32,
               guidance_scale=6.5, generator=g, height=1024, width=1024).images[0]
    p = os.path.join(OUT, f"whiteman_{i:02d}.png")
    img.save(p)
    print(f"[GEN] saved {p}", flush=True)
print("[GEN] DONE", flush=True)
