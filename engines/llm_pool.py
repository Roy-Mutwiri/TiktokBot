# =============================================================================
# engines/llm_pool.py — parallel, resource-aware commentary prefetch.
#
# The auto-host used to generate ONE line at a time, synchronously, on the render
# thread — so a slow LLM call left the avatar silent. This pool runs SEVERAL
# worker threads that generate lines AHEAD of time, in parallel, and keep a small
# buffer of ready-to-speak lines. The auto-talk loop just PULLS a ready line
# (instant) instead of waiting on the model — so the host never goes quiet.
#
# Resource-aware (the same idea as the CPU/GPU filter governor): each worker asks
# the ResourceMonitor whether the GPU is free. When the avatar is busy with the
# face-swap / voice on the GPU, workers EITHER offload to a CPU-resident model
# (if AVATAR_LLM_CPU_MODEL is set) OR back off so they don't starve the renderer.
# When the GPU is idle they generate fast on it. Either way the buffer stays full.
#
# De-dupe: a line whose first words match one we recently produced is tossed and
# regenerated, so the buffer never hands the host a repeat.
#
#   pool = BrainPool(brain, monitor, beats=[...], get_context=ctx_fn, persona=p)
#   pool.start()
#   line = pool.get(timeout=0.1)   # ready-made line, or None if the buffer is cold
#   pool.stop()
# =============================================================================
import os
import sys
import time
import queue
import threading
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class BrainPool:
    """Keeps a buffer of ready-to-speak host lines, generated in parallel and
    routed to whatever compute is free, so the live never goes quiet."""

    def __init__(self, brain, monitor=None, beats=None, get_context=None,
                 persona=None, workers=None, buffer=None):
        self.brain = brain
        self.monitor = monitor
        self.beats = list(beats) if beats else [
            "Give a quick, lively line about gold right now."]
        self.get_context = get_context
        self.persona = persona
        self._stop = True
        self._threads = []
        self._i = 0
        self._i_lock = threading.Lock()
        self._recent = collections.deque(maxlen=40)      # full-line dedupe signatures
        self._recent_openers = collections.deque(maxlen=14)  # first-3-word openers
        self._recent_lock = threading.Lock()
        self.ready = queue.Queue(
            maxsize=int(buffer or os.environ.get("AVATAR_LLM_BUFFER", "6")))
        self._nworkers = int(workers or os.environ.get("AVATAR_LLM_WORKERS", "2"))
        # cap each prefetched line short (faster to generate = buffer refills quicker
        # = less chance of going quiet; also fewer/shorter TTS chunks).
        self.line_tokens = int(os.environ.get("AVATAR_LLM_LINE_TOKENS", "100"))
        # optional SECOND model that lives on the CPU (a different model name so
        # Ollama keeps it loaded alongside the GPU one with no reload thrash). When
        # set, GPU-busy generations route here instead of fighting for the GPU.
        self.cpu_model = os.environ.get("AVATAR_LLM_CPU_MODEL", "").strip() or None
        self.stats = {"gpu": 0, "cpu": 0, "dup": 0, "made": 0}

    # -------------------------------------------------------------------------
    def start(self):
        if not self._stop:
            return
        self._stop = False
        self._threads = []
        for k in range(max(1, self._nworkers)):
            t = threading.Thread(target=self._worker, args=(k,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop = True

    def status(self):
        return (f"LLM pool: {self.ready.qsize()}/{self.ready.maxsize} ready "
                f"| gpu {self.stats['gpu']} cpu {self.stats['cpu']} "
                f"dup {self.stats['dup']}"
                + (f" | cpu-lane {self.cpu_model}" if self.cpu_model else ""))

    # -------------------------------------------------------------------------
    def _next_beat(self):
        with self._i_lock:
            b = self.beats[self._i % len(self.beats)]
            self._i += 1
            return b, self._i

    @staticmethod
    def _sigs(line):
        w = (line or "").lower().split()
        return " ".join(w[:7]), " ".join(w[:3])   # full signature, opener signature

    def _is_dup(self, line):
        """Reject near-repeats: same opening 7 words OR same opening 3 words as a
        recent line — so the host never hands back a line that STARTS the same."""
        full, opener = self._sigs(line)
        if not full:
            return True
        with self._recent_lock:
            return full in self._recent or (bool(opener) and opener in self._recent_openers)

    def _remember(self, line):
        full, opener = self._sigs(line)
        with self._recent_lock:
            self._recent.append(full)
            if opener:
                self._recent_openers.append(opener)

    def _route(self):
        """Pick a lane from live load (the CPU/GPU governor idea, for the LLM).
        Returns (num_gpu, model, lane, busy). busy=True means the GPU is needed by
        the renderer right now."""
        busy = False
        if self.monitor is not None:
            try:
                busy = not self.monitor.gpu_free(
                    int(os.environ.get("AVATAR_LLM_GPU_FREE", "70")))
            except Exception:
                busy = False
        if busy and self.cpu_model:           # GPU needed -> offload to the CPU model
            return 0, self.cpu_model, "cpu", busy
        if busy:                              # GPU needed, no CPU model -> stay on GPU
            return None, self.brain.model, "gpu", busy   # caller will back off a bit
        return 999, self.brain.model, "gpu", busy        # GPU idle -> full speed

    def _worker(self, k):
        while not self._stop:
            try:
                if self.ready.full():
                    time.sleep(0.12)
                    continue
                if self.brain is None or not getattr(self.brain, "ok", False):
                    time.sleep(0.5)
                    continue
                beat, n = self._next_beat()
                ctx = ""
                if self.get_context:
                    try:
                        ctx = self.get_context() or ""
                    except Exception:
                        ctx = ""
                num_gpu, model, lane, busy = self._route()
                # GPU needed by the renderer: only ease off if we ALREADY have a
                # cushion of ready lines. If the buffer is low, generate anyway (the
                # 3B is light) so the host never runs dry waiting on a free GPU.
                if busy and lane == "gpu" and self.ready.qsize() >= 3:
                    time.sleep(float(os.environ.get("AVATAR_LLM_BUSY_BACKOFF", "0.5")))
                    continue
                seed = (n * 1315423911 + k * 2654435761
                        + int(time.monotonic() * 1000)) % 2147483647
                line = self.brain.generate_line(
                    beat + ctx, persona=self.persona, num_gpu=num_gpu,
                    model=model, seed=seed, max_tokens=self.line_tokens)
                if not line:
                    time.sleep(0.3)
                    continue
                if self._is_dup(line):
                    self.stats["dup"] += 1
                    continue
                self._remember(line)
                self.stats[lane] = self.stats.get(lane, 0) + 1
                self.stats["made"] += 1
                try:
                    self.ready.put((line, lane), timeout=1.0)
                except queue.Full:
                    pass
            except Exception:
                time.sleep(0.4)

    def get(self, timeout=0.1):
        """Pop a ready line (instant if the buffer is warm), else None."""
        try:
            line, _lane = self.ready.get(timeout=timeout)
            return line
        except queue.Empty:
            return None
