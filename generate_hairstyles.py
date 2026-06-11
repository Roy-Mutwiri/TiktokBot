# Build HAIRSTYLE overlay assets: generate a white-man head per style with SDXL,
# segment the HAIR (BiSeNet parsing, label 17) into an RGBA cut-out, and record
# the face 5-kps so the engine can align the hair to your head. Forward-facing
# overlay (best when you face the camera; 2D, won't rotate in 3D on hard turns).
import os, torch, cv2, numpy as np
from diffusers import StableDiffusionXLPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils import img2tensor
from torchvision.transforms.functional import normalize
import insightface

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hairstyles")
os.makedirs(OUT, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
parse = init_parsing_model(model_name="bisenet", device="cuda")
app = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

STYLES = {
    "buzz":      "very short buzz cut brown hair, clean masculine",
    "fade":      "short brown hair with skin fade sides and textured top, modern barber cut",
    "quiff":     "brown quiff hairstyle, voluminous swept-up front, short sides, men's",
    "undercut":  "brown undercut hairstyle, slicked-back top, shaved sides, sharp masculine",
    "crewcut":   "classic short brown crew cut, neat masculine",
    "curly":     "short curly brown hair, men's tapered",
    "long":      "long brown hair to the shoulders, masculine, slight wave",
}
NEG = "cartoon, anime, cgi, blurry, deformed, hat, multiple people, watermark, text"


def hair_mask(bgr):
    img = cv2.resize(bgr, (512, 512))
    t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    t = t.unsqueeze(0).cuda()
    with torch.no_grad():
        out = parse(t)[0]
    lab = out.argmax(1)[0].cpu().numpy().astype(np.uint8)        # 512x512 labels
    m = (lab == 17).astype(np.uint8) * 255                       # hair = 17
    return cv2.resize(m, (bgr.shape[1], bgr.shape[0]))


for name, desc in STYLES.items():
    g = torch.Generator("cuda").manual_seed(777)
    img = pipe(f"RAW studio photo, headshot of a white man with {desc}, facing the "
               f"camera straight on, plain flat gray background, even soft light, "
               f"sharp focus", negative_prompt=NEG, num_inference_steps=34,
               guidance_scale=6.5, generator=g, height=1024, width=1024).images[0]
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        print(f"[HAIR] {name}: no face, skip"); continue
    kps = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).kps
    alpha = cv2.GaussianBlur(hair_mask(bgr), (0, 0), 2.0)
    rgba = np.dstack([bgr, alpha])
    cv2.imwrite(os.path.join(OUT, f"{name}.png"), rgba)
    np.save(os.path.join(OUT, f"{name}.npy"), kps.astype(np.float32))
    print(f"[HAIR] saved {name} (hair px={int((alpha>20).sum())})", flush=True)
print("[HAIR] DONE", flush=True)
