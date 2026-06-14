# =============================================================================
# engines/resource_monitor.py — ALWAYS-ON live CPU/GPU/VRAM monitor + adaptive
# load router, so the avatar never lags.
#
# Samples CPU% (psutil), GPU% + VRAM% (NVML) on a background thread every 100ms and
# exposes:
#   .cpu / .gpu / .vram        smoothed live load (%)
#   .route_filters()           "gpu" or "cpu" — where to run the MOVABLE filter work
#                              (enhance/mouth crisp have both a cv2-CPU and a kornia-GPU
#                              path) so it lands on whichever device is freer
#   .quality()                 "full" or "light" — drops optional heavy passes (with
#                              hysteresis) when BOTH are saturated, so fps holds = no lag
#   .summary()                 "CPU 46  GPU 44  VRAM 54" for the on-screen/diag readout
#
# Note: the face-swap + voice are GPU-locked (TensorRT/Chatterbox) and stay on GPU;
# only the dual-path filter work is routed. Never raises.
# =============================================================================
import os
import time
import threading

INTERVAL = max(0.05, float(os.environ.get("AVATAR_MONITOR_INTERVAL", "0.1")))


class ResourceMonitor:
    def __init__(self):
        self.cpu = 0.0
        self.gpu = 0.0
        self.vram = 0.0
        self.cpu_live = 0.0
        self.gpu_live = 0.0
        self.vram_live = 0.0
        self.ready = False
        self._light = False          # current degraded state (hysteresis)
        self._stop = False
        self._psutil = None
        self._nvml = None
        self._h = None
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)        # prime the counter
        except Exception:
            self._psutil = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
        # ALWAYS ON: start sampling immediately, run for the whole process life.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                if self._psutil is not None:
                    c = self._psutil.cpu_percent(interval=None)
                    self.cpu_live = float(c)
                    self.cpu = 0.5 * self.cpu + 0.5 * c
                if self._nvml is not None:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._h)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(self._h)
                    self.gpu_live = float(u.gpu)
                    self.vram_live = m.used / m.total * 100.0
                    self.gpu = 0.5 * self.gpu + 0.5 * float(u.gpu)
                    self.vram = self.vram_live
                self.ready = True
            except Exception:
                pass
            time.sleep(INTERVAL)

    # -- adaptive routing -----------------------------------------------------
    def route_filters(self):
        """Where to run the next MOVABLE filter task. Send it to the freer device;
        never pile onto the GPU if VRAM is tight (would risk an OOM)."""
        if not self.ready:
            return "cpu"
        if self.vram > 88:                  # no VRAM headroom -> keep work on CPU
            return "cpu"
        if self.cpu > 78 and self.gpu < 72:  # CPU pinned, GPU has room -> offload to GPU
            return "gpu"
        if self.gpu > 85 and self.cpu < 78:  # GPU pinned, CPU has room -> keep on CPU
            return "cpu"
        return "cpu"                         # default: filters are cheap on CPU; leave
        #                                      the GPU for the swap/voice

    def quality(self):
        """'light' (skip optional heavy passes) when BOTH devices are saturated, so
        the framerate holds instead of stuttering. Hysteresis stops it flapping."""
        if not self.ready:
            return "full"
        both = self.cpu + self.gpu
        if self._light:
            if self.cpu < 80 and self.gpu < 80:      # recovered -> back to full
                self._light = False
        else:
            if self.cpu > 92 and self.gpu > 88:      # both maxed -> degrade
                self._light = True
        return "light" if self._light else "full"

    def saturated(self):
        return self.quality() == "light"

    def gpu_free(self, threshold=72):
        """True if the GPU has spare capacity for an OPTIONAL quality pass (de-blur,
        HD restore) right now — and VRAM has headroom (so we never OOM). Used to run
        the heavy 'pro' passes only when they won't cause a stutter."""
        if not self.ready:
            return True                 # before the first sample, allow (boot)
        return self.gpu < threshold and self.vram < 86

    def summary(self):
        return f"CPU {self.cpu:.0f}  GPU {self.gpu:.0f}  VRAM {self.vram:.0f}"

    def stop(self):
        self._stop = True
