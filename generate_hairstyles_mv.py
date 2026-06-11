# POSE-AWARE MULTI-VIEW hairstyles. For each style: SDXL generates a frontal head,
# then LivePortrait renders that SAME head at several yaw angles (consistent multi-
# view), BiSeNet segments the hair from each view, and we save an RGBA hair cut-out
# + its face 5-kps per view. At runtime the engine picks the hair view matching
# your head's yaw, so the hair TURNS WITH YOUR HEAD instead of staying flat.
import os, sys, torch, cv2, numpy as np
sys.path.insert(0, "engines")
from diffusers import StableDiffusionXLPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils import img2tensor
from torchvision.transforms.functional import normalize
import insightface

PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJECT, "hairstyles_mv")
os.makedirs(OUT, exist_ok=True)
YAWS = [-50, -28, -12, 0, 12, 28, 50]

STYLES = {
    "buzz":     "very short buzz cut brown hair, masculine",
    "fade":     "short brown hair skin fade sides textured top, barber cut",
    "quiff":    "brown quiff, voluminous swept-up front, short sides",
    "undercut": "brown undercut, slicked-back top, shaved sides",
    "crewcut":  "classic short brown crew cut",
    "curly":    "short curly brown hair tapered",
    "long":     "long brown hair to shoulders, slight wave",
}
NEG = "cartoon, anime, cgi, blurry, deformed, hat, multiple people, watermark, text"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
parse = init_parsing_model(model_name="bisenet", device="cuda")
app = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def hair_rgba(bgr):
    img = cv2.resize(bgr, (512, 512))
    t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    with torch.no_grad():
        lab = parse(t.unsqueeze(0).cuda())[0].argmax(1)[0].cpu().numpy().astype(np.uint8)
    m = cv2.resize((lab == 17).astype(np.uint8) * 255, (bgr.shape[1], bgr.shape[0]))
    return np.dstack([bgr, cv2.GaussianBlur(m, (0, 0), 2.0)])


for name, desc in STYLES.items():
    g = torch.Generator("cuda").manual_seed(777)
    img = pipe(f"RAW studio photo, headshot of a white man with {desc}, facing camera "
               f"straight on, plain flat gray background, even light, sharp focus",
               negative_prompt=NEG, num_inference_steps=32, guidance_scale=6.5,
               generator=g, height=1024, width=1024).images[0]
    src = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    tmp = os.path.join(OUT, f"_{name}_src.png"); cv2.imwrite(tmp, src)
    # LivePortrait renders the SAME head at each yaw (consistent multi-view)
    from liveportrait_engine import LivePortraitEngine
    lp = LivePortraitEngine(tmp); W = lp.wrapper
    if lp.fallback_mode:
        print(f"[HAIR] {name}: LP unavailable, frontal only"); continue
    info = lp._x_s_info
    sdir = os.path.join(OUT, name); os.makedirs(sdir, exist_ok=True)
    saved = []
    for vi, yw in enumerate(YAWS):
        z = torch.zeros_like(info["yaw"])
        R = lp._get_rotation_matrix(z, z + float(yw), z) @ lp._R_s
        x_d = info["scale"] * (info["kp"] @ R + info["exp"]) + info["t"]
        try:
            x_d = W.stitching(lp._x_s, x_d)
        except Exception:
            pass
        view = cv2.cvtColor(W.parse_output(W.warp_decode(lp._f_s, lp._x_s, x_d)["out"])[0],
                            cv2.COLOR_RGB2BGR)
        view = cv2.resize(view, (1024, 1024))
        faces = app.get(view)
        if not faces:
            continue
        kps = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).kps
        cv2.imwrite(os.path.join(sdir, f"v{vi}.png"), hair_rgba(view))
        np.save(os.path.join(sdir, f"v{vi}.npy"), kps.astype(np.float32))
        saved.append(yw)
    np.save(os.path.join(sdir, "yaws.npy"), np.array(saved, np.float32))
    os.remove(tmp)
    del lp
    print(f"[HAIR] {name}: {len(saved)} views {saved}", flush=True)
print("[HAIR] DONE", flush=True)
