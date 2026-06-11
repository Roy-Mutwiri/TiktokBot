# Generate several high-quality WHITE-MAN candidates with SDXL (base + refiner) so
# we can pick the single strongest, most photorealistic identity for the swap.
import os, torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_candidates")
os.makedirs(OUT, exist_ok=True)

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
base.set_progress_bar_config(disable=True)
try:
    ref = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    ref.set_progress_bar_config(disable=True)
except Exception as e:
    print("[GEN] refiner unavailable:", e); ref = None

PROMPT = ("candid RAW photo, close-up headshot of a handsome white caucasian man, "
          "mid 30s, short neat brown hair, short well-groomed beard, blue eyes, "
          "confident neutral expression, looking directly at the camera, soft "
          "natural window light, plain studio background, ultra realistic detailed "
          "skin with pores, sharp focus, shot on 85mm f1.8, professional portrait")
NEG = ("cartoon, anime, illustration, painting, cgi, 3d, render, plastic, waxy, "
       "airbrushed, blurry, deformed, asymmetric, ugly, multiple people, hat, "
       "sunglasses, watermark, text, low quality, dark")

for i, seed in enumerate([101, 202, 303, 404, 505, 606]):
    g = torch.Generator("cuda").manual_seed(seed)
    img = base(PROMPT, negative_prompt=NEG, num_inference_steps=40,
               guidance_scale=6.5, generator=g, height=1024, width=1024,
               denoising_end=0.8 if ref else 1.0,
               output_type="latent" if ref else "pil").images
    if ref:
        img = ref(PROMPT, negative_prompt=NEG, image=img, num_inference_steps=40,
                  denoising_start=0.8, generator=g).images
    img[0].save(os.path.join(OUT, f"cand_{i:02d}.png"))
    print(f"[GEN] saved cand_{i:02d} (seed {seed})", flush=True)
print("[GEN] DONE", flush=True)
