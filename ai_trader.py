# =============================================================================
# ai_trader.py  —  the AI market analyst: drives a live chart + talks about it
# -----------------------------------------------------------------------------
# An AI analyst that operates a real, live chart on its own: it scrolls, zooms,
# draws trendlines / support-resistance / fibonacci with an animated tool cursor,
# and speaks its read of the market in real time (English or Arabic, via the
# multilingual voice).
#
#   python ai_trader.py                         # live gold (PAXG), window + voice
#   python ai_trader.py --symbol BTCUSDT        # bitcoin
#   python ai_trader.py --lang ar               # narrate in Arabic
#   python ai_trader.py --cam                   # publish to the virtual camera (OBS/stream)
#   python ai_trader.py --no-speak              # silent (chart only)
#   python ai_trader.py --size 1280x720
#
# It feeds the SAME analysis the avatar can speak, so the chart and the avatar
# stay in sync on stream.
# =============================================================================

import os
import sys
import time
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "engines"))

import numpy as np
import cv2

FRIENDLY = {"PAXGUSDT": "XAU/USD (gold)", "BTCUSDT": "BTC/USD",
            "ETHUSDT": "ETH/USD"}


def _parse_size(s):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 960, 540


class _Narrator:
    """Speaks narration lines on a background thread via the multilingual voice.
    Degrades gracefully to printing if the TTS backend can't load."""

    def __init__(self, lang="en"):
        self.lang = lang
        self.ok = False
        self._eng = None
        self._q = []
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._load_and_run, daemon=True).start()

    def _load_and_run(self):
        try:
            from multilingual_tts import MultilingualTTSBackend
            self._eng = MultilingualTTSBackend()
            self.ok = self._eng.ok
        except Exception as exc:
            print(f"[narrator] voice unavailable ({exc}); printing only.")
        while not self._stop:
            line = None
            with self._lock:
                if self._q:
                    line = self._q.pop(0)
            if line is None:
                time.sleep(0.1); continue
            print(f"[AI says] {line}")
            if self._eng and self._eng.ok:
                try:
                    wav, sr = self._eng.synthesize(line, lang=self.lang)
                    if wav is not None and len(wav):
                        import sounddevice as sd
                        sd.play(wav, sr); sd.wait()
                except Exception:
                    pass

    def say(self, line):
        with self._lock:
            # keep the queue short so narration tracks the live chart
            self._q = self._q[-1:] + [line]

    def stop(self):
        self._stop = True


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="PAXGUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--lang", default="en", choices=["en", "ar"])
    ap.add_argument("--cam", action="store_true", help="publish to the virtual camera")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--window", action="store_true", help="force a preview window")
    args = ap.parse_args(argv[1:])

    from chart_pilot import ChartPilot
    W, H = _parse_size(args.size)
    name = FRIENDLY.get(args.symbol.upper(), args.symbol.upper())
    pilot = ChartPilot(args.symbol, args.interval, size=(W, H),
                       display_name=name, narrate_lang=args.lang)
    print(pilot.startup_check()[1])

    narrator = None if args.no_speak else _Narrator(args.lang)

    cam = None
    if args.cam:
        try:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(width=W, height=H, fps=20,
                                      fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[cam] publishing AI trader to: {cam.device}")
        except Exception as exc:
            print(f"[cam] virtual camera unavailable ({exc}).")

    show = args.window or (not args.cam)
    print("AI TRADER live. Ctrl+C to stop." + ("  [Q] closes the window." if show else ""))
    try:
        while True:
            frame = pilot.render(speaking=True)
            line = pilot.next_narration()
            if line and narrator:
                narrator.say(line)
            elif line:
                print(f"[AI says] {line}")
            if cam is not None:
                cam.send(np.ascontiguousarray(frame))
                cam.sleep_until_next_frame()
            if show:
                cv2.imshow("AI Trader", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if cam is None and not show:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        pilot.stop()
        if narrator:
            narrator.stop()
        if cam is not None:
            cam.close()
        if show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
