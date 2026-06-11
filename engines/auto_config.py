# =============================================================================
# engines/auto_config.py
# -----------------------------------------------------------------------------
# AUTO HARDWARE CONFIG — probes the machine (GPU VRAM, RAM, CPU), runs a quick
# GPU compute benchmark, checks which Ollama models are installed, and AUTO-PICKS
# the heaviest set of models that will fit + run smoothly:
#
#   * voice backend  (chatterbox accent | kokoro | edge)
#   * brain model    (qwen2.5:14b | qwen2.5:7b | llama3.1:8b | llama3.2:3b)
#   * face restore   (CodeFormer | GFPGAN | off)
#   * LP / restore cadence  (tuned to the GPU's measured speed)
#
# It sets the corresponding AVATAR_* env vars (with setdefault, so anything you
# set by hand WINS). Call autoconfigure() BEFORE the engines are imported.
#
#   from auto_config import autoconfigure
#   PROFILE = autoconfigure()      # detects + applies + returns the profile
#
# Run standalone to just SEE the recommendation:  python engines/auto_config.py
# =============================================================================

import os
import sys
import time
import json
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# VRAM cost estimates (GB) used to fit models into the free budget.
VOICE_VRAM = [("chatterbox", 3.5), ("kokoro", 0.4), ("edge", 0.0)]   # preference order
BRAIN_VRAM = [("qwen2.5:14b", 9.0), ("qwen2.5:7b", 4.7),
              ("llama3.1:8b", 5.0), ("llama3.2:3b", 2.0)]            # best -> smallest
VIDEO_RESERVE = 6.0     # GB held back for LivePortrait + mouth-sync + restore + overhead


def _ollama_models():
    """Names of models pulled in Ollama (empty if the server is down)."""
    try:
        host = os.environ.get("AVATAR_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        with urllib.request.urlopen(host + "/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "") for m in (data.get("models") or [])]
    except Exception:
        return []


def _bench_gpu(torch):
    """Rough fp16 matmul TFLOP/s — a quick proxy for GPU compute speed."""
    try:
        dev = "cuda"; n = 4096; iters = 50
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        for _ in range(10):                    # WARM UP cuBLAS (first call inits it)
            c = a @ b
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = time.time() - t
        del a, b, c
        torch.cuda.empty_cache()
        return (2.0 * n**3 * iters) / dt / 1e12
    except Exception:
        return 0.0


def detect():
    """Probe the machine. Returns a dict of resources."""
    res = {"cuda": False, "gpu": "CPU", "vram_total": 0.0, "vram_free": 0.0,
           "ram_gb": 0.0, "cpu": os.cpu_count() or 1, "tflops": 0.0,
           "ollama": _ollama_models()}
    try:
        import torch
        if torch.cuda.is_available():
            res["cuda"] = True
            res["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            res["vram_total"] = total / 1e9
            res["vram_free"] = free / 1e9
            res["tflops"] = _bench_gpu(torch)
    except Exception as exc:
        res["err"] = str(exc)[:80]
    try:
        import psutil
        res["ram_gb"] = psutil.virtual_memory().total / 1e9
    except Exception:
        try:                                   # Windows fallback w/o psutil
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_uint32), ("m", ctypes.c_uint32),
                            ("a", ctypes.c_uint64), ("b", ctypes.c_uint64),
                            ("c", ctypes.c_uint64), ("d", ctypes.c_uint64),
                            ("e", ctypes.c_uint64), ("f", ctypes.c_uint64),
                            ("g", ctypes.c_uint64)]
            ms = MS(); ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            res["ram_gb"] = ms.a / 1e9
        except Exception:
            pass
    return res


def _has(name, pulled):
    """Is a model (or its family, e.g. qwen2.5) pulled in Ollama?"""
    if not pulled:
        return name == "llama3.2:3b"          # assume the safe default is around
    base = name.split(":")[0]
    return any(n == name or n.split(":")[0] == base for n in pulled)


def choose(res):
    """Pick the heaviest models that fit the free VRAM + are installed."""
    if not res["cuda"]:
        return dict(tts="edge", brain="llama3.2:3b", restore="off",
                    lp_interval=4, restore_interval=6, upscale=False,
                    why="no CUDA GPU — cloud voice + lightest brain, no GPU restore.")

    free = res["vram_free"]
    # prefer the Arabic-accent Chatterbox voice IF it still leaves room for a
    # usable brain; else fall back to the light Kokoro voice, else cloud edge.
    pick = None
    for vname, vmem in VOICE_VRAM:
        budget = free - VIDEO_RESERVE - vmem
        for bname, bmem in BRAIN_VRAM:
            if bmem <= budget and _has(bname, res["ollama"]):
                pick = (vname, bname); break
        if pick:
            break
    if not pick:
        pick = ("edge", "llama3.2:3b")
    tts, brain = pick

    restore = "codeformer" if free >= 9 else ("gfpgan" if free >= 6 else "off")
    # cadence: more VRAM + faster GPU -> run LP/restore more often (more quality)
    lp_interval = 1 if free >= 18 else (2 if free >= 11 else 3)
    restore_interval = 2 if free >= 12 else (3 if free >= 8 else 4)
    if res["tflops"] and res["tflops"] < 12:   # slower GPU -> ease the cadence
        lp_interval = max(lp_interval, 3)
        restore_interval = max(restore_interval, 4)
    upscale = free >= 20 and res["tflops"] >= 30   # only on big, fast cards

    why = (f"{free:.1f}GB free VRAM, {res['tflops']:.0f} TFLOP/s -> "
           f"{tts} voice + {brain} brain + {restore} restore.")
    return dict(tts=tts, brain=brain, restore=restore, lp_interval=lp_interval,
                restore_interval=restore_interval, upscale=upscale, why=why)


def apply(cfg):
    """Set the AVATAR_* env vars (setdefault — your own overrides win)."""
    env = {
        "AVATAR_TTS": cfg["tts"],
        "AVATAR_BRAIN_MODEL": cfg["brain"],
        "AVATAR_RESTORE": "0" if cfg["restore"] == "off" else "1",
        "AVATAR_RESTORE_NET": "gfpgan" if cfg["restore"] in ("gfpgan", "off") else "codeformer",
        "AVATAR_RESTORE_INTERVAL": str(cfg["restore_interval"]),
        "AVATAR_LP_INTERVAL": str(cfg["lp_interval"]),
        "AVATAR_UPSCALE": "1" if cfg["upscale"] else "0",
    }
    for k, v in env.items():
        os.environ.setdefault(k, str(v))
    return env


def autoconfigure(verbose=True):
    """Detect -> choose -> apply. Returns a profile dict {res, cfg}."""
    res = detect()
    cfg = choose(res)
    env = apply(cfg)
    if verbose:
        print("=" * 60)
        print("[AUTO-CONFIG] machine:")
        print(f"   GPU : {res['gpu']}  |  VRAM {res['vram_free']:.1f}/{res['vram_total']:.1f} GB free"
              + (f"  |  {res['tflops']:.0f} TFLOP/s" if res['tflops'] else ""))
        print(f"   CPU : {res['cpu']} cores  |  RAM {res['ram_gb']:.0f} GB"
              + (f"  |  Ollama: {len(res['ollama'])} models" if res['ollama'] else "  |  Ollama: down"))
        print(f"[AUTO-CONFIG] -> {cfg['why']}")
        print("=" * 60)
    return {"res": res, "cfg": cfg, "env": env}


def vram_status():
    """Live (free_gb, total_gb) for a runtime watchdog. (0,0) if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return 0.0, 0.0


if __name__ == "__main__":
    autoconfigure(verbose=True)
