# =============================================================================
# engines/chart_pilot.py  —  the AI that OPERATES the chart
# -----------------------------------------------------------------------------
# Ties the live feed + analysis + chart into an autonomous "analyst performance":
# it scrolls, zooms, and draws tools on a loop, re-analysing the live market each
# cycle, and emits a spoken narration line for each move (which the avatar can
# speak via the multilingual TTS).
#
#   pilot = ChartPilot("PAXGUSDT", "15m", size=(512, 512), display_name="XAU/USD")
#   frame = pilot.render(speaking=...)        # drop-in for the chart scene
#   line  = pilot.next_narration()            # -> text to speak, or None
#
# It is a drop-in replacement for the old TradingView scene's render().
# =============================================================================

import time
import threading
from collections import deque

from market_data import MarketData
from market_analysis import analyze
from ai_chart import AIChart


class ChartPilot:
    def __init__(self, symbol="PAXGUSDT", interval="15m", size=(512, 512),
                 display_name=None, narrate_lang="en"):
        self.symbol = symbol
        self.md = MarketData(symbol, interval)
        self.chart = AIChart(display_name or symbol, size)
        self.narrate_lang = narrate_lang          # 'en' or 'ar'
        self.analysis = None
        self._narration = deque(maxlen=8)
        self._seq = []
        self._i = 0
        self._t0 = 0.0
        self._said = set()
        self._lock = threading.Lock()
        self._refresh()
        self.md.start()

    # -- analysis refresh + action plan --------------------------------------
    def _refresh(self):
        ohlc = self.md.snapshot()
        a = analyze(ohlc, self.symbol)
        with self._lock:
            self.analysis = a
            self.chart.set_data(ohlc, a)
            self._seq = self._plan(a)
            self._i = 0
            self._t0 = time.time()
            self._said = set()

    def _plan(self, a):
        """Build the analyst 'performance' for this analysis pass."""
        tl_sup = next((d for d in a.drawings if d["type"] == "trendline" and d["kind"] == "support"), None)
        tl_res = next((d for d in a.drawings if d["type"] == "trendline" and d["kind"] == "resistance"), None)
        seq = []
        seq.append({"kind": "overview", "dur": 6.0, "say": a.narrative})
        seq.append({"kind": "zoom_focus", "dur": 4.5,
                    "say": "Let's zoom into the recent price action."})
        if tl_sup:
            seq.append({"kind": "draw_tl", "tl": tl_sup, "dur": 4.0,
                        "say": "I'll draw the support trendline connecting the higher lows."})
        if tl_res:
            seq.append({"kind": "draw_tl", "tl": tl_res, "dur": 4.0,
                        "say": "And the resistance line along the swing highs."})
        if a.key_level is not None:
            seq.append({"kind": "level", "dur": 3.5,
                        "say": f"The level to watch is {a.key_level:,.1f}."})
        seq.append({"kind": "zoom_out", "dur": 4.0,
                    "say": a.signals and ("Watch for: " + ", ".join(a.signals) + ".") or ""})
        return seq

    # -- per-frame advance ----------------------------------------------------
    def _apply(self, act, first):
        k = act["kind"]
        if k == "overview":
            snap = self.md.snapshot()
            self.chart.follow_live(bars=min(110, len(snap) if snap is not None else 90))
        elif k == "zoom_focus":
            if self.analysis and self.analysis.focus:
                self.chart.focus(*self.analysis.focus)
        elif k == "draw_tl":
            if self.analysis and self.analysis.focus:
                self.chart.focus(*self.analysis.focus)
            if first:
                tl = act["tl"]
                self.chart.begin_tool("trendline", tl["p1"], tl["p2"], tl["color"],
                                      label=tl["kind"][:3].upper(), dur=min(act["dur"] * 0.7, 2.2))
        elif k == "level":
            pass
        elif k == "zoom_out":
            self.chart.follow_live(bars=130)

    def _advance(self):
        with self._lock:
            if not self._seq:
                return
            now = time.time()
            act = self._seq[self._i]
            first = self._i not in self._said
            if first:
                self._said.add(self._i)
                self._apply(act, True)
                say = act.get("say")
                if say:
                    self._narration.append(say)
            else:
                self._apply(act, False)
            if now - self._t0 >= act["dur"]:
                self._i += 1
                self._t0 = now
                if self._i >= len(self._seq):
                    # cycle finished -> re-analyse the live market
                    threading.Thread(target=self._refresh, daemon=True).start()

    # -- public API -----------------------------------------------------------
    def render(self, speaking=False):
        self._advance()
        return self.chart.render(speaking)

    def next_narration(self):
        with self._lock:
            return self._narration.popleft() if self._narration else None

    def reset_price_drift(self):
        # API-compat with the old TradingView scene (no-op here)
        pass

    def startup_check(self):
        ok, msg = self.md.startup_check()
        a = self.analysis
        extra = f"; {a.trend} bias {a.bias}" if a else ""
        return ok, f"AI chart pilot {self.symbol}: {msg}{extra}"

    def stop(self):
        self.md.stop()


if __name__ == "__main__":
    import os, sys, cv2
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sym = sys.argv[1] if len(sys.argv) > 1 else "PAXGUSDT"
    name = {"PAXGUSDT": "XAU/USD (gold)"}.get(sym, sym)
    pilot = ChartPilot(sym, "15m", size=(640, 480), display_name=name)
    print(pilot.startup_check()[1])
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tts_samples")
    os.makedirs(out_dir, exist_ok=True)
    # run ~14s at 20fps, save a frame at each new narration to show the performance
    shots = 0
    for f in range(280):
        frame = pilot.render(speaking=True)
        line = pilot.next_narration()
        if line:
            print(f"[AI says] {line}")
            cv2.imwrite(os.path.join(out_dir, f"chart_step_{shots}.png"), frame)
            shots += 1
        time.sleep(0.05)
    print(f"saved {shots} step frames to {out_dir}")
    pilot.stop()
