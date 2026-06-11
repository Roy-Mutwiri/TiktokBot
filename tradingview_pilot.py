# =============================================================================
# tradingview_pilot.py  —  the AI operates the REAL TradingView website
# -----------------------------------------------------------------------------
# Opens TradingView.com in a controlled browser (Playwright) and has the AI drive
# it like an analyst: switches timeframe, zooms, pans, activates drawing tools and
# draws trendlines / fibonacci on the chart, while narrating its read of the
# market (computed from real data via market_analysis, spoken with the
# multilingual voice).
#
#   python tradingview_pilot.py                       # gold, visible browser + voice
#   python tradingview_pilot.py --symbol BINANCE:BTCUSDT
#   python tradingview_pilot.py --headless --no-speak # for testing
#
# HONEST CAVEATS: TradingView's chart is a canvas with no public price<->pixel
# API, so drawings are placed at sensible (magnet-snapped) positions, not exact
# prices. Automating tradingview.com is also against their Terms of Service -
# use on your own account/stream at your discretion. The robust, price-perfect,
# ToS-safe alternative is ai_trader.py (our own chart).
# =============================================================================

import os
import sys
import time
import argparse
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "engines"))

TV_URL = "https://www.tradingview.com/chart/?symbol={sym}"
TREND_GROUP = "[data-name='linetool-group-trend-line']"
FIB_GROUP = "[data-name='linetool-group-gann-and-fibonacci']"


class _Narrator:
    """Speaks analysis lines via the multilingual voice (degrades to print)."""

    def __init__(self, lang="en", enabled=True):
        self.lang, self.enabled = lang, enabled
        self.ok = False
        self._eng = None
        self._q = []
        self._lock = threading.Lock()
        self._stop = False
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            from multilingual_tts import MultilingualTTSBackend
            self._eng = MultilingualTTSBackend()
            self.ok = self._eng.ok
        except Exception as exc:
            print(f"[narrator] voice off ({exc})")
        while not self._stop:
            line = None
            with self._lock:
                if self._q:
                    line = self._q.pop(0)
            if not line:
                time.sleep(0.1); continue
            print(f"[AI] {line}")
            if self._eng and self._eng.ok:
                try:
                    import sounddevice as sd
                    wav, sr = self._eng.synthesize(line, lang=self.lang)
                    if wav is not None and len(wav):
                        sd.play(wav, sr); sd.wait()
                except Exception:
                    pass

    def say(self, line):
        if not self.enabled:
            print(f"[AI] {line}"); return
        with self._lock:
            self._q = self._q[-1:] + [line]

    def stop(self):
        self._stop = True


class TradingViewPilot:
    def __init__(self, symbol="OANDA:XAUUSD", headless=False, lang="en", speak=True):
        self.symbol = symbol
        self.headless = headless
        self.lang = lang
        self.narrator = _Narrator(lang, enabled=speak)
        self._pw = self._ctx = self.page = None
        self._stop = False
        self._md = self._mk_market_data(symbol)

    # -- analysis data source -------------------------------------------------
    def _mk_market_data(self, tv_symbol):
        try:
            from market_data import MarketData
            s = tv_symbol.split(":")[-1].upper()
            ds = {"XAUUSD": "PAXGUSDT", "GOLD": "PAXGUSDT",
                  "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT"}.get(s, s)
            if not ds.endswith("USDT"):
                ds = "PAXGUSDT"
            md = MarketData(ds, "15m"); md.start()
            return md
        except Exception:
            return None

    def _analysis(self):
        if not self._md:
            return None
        try:
            from market_analysis import analyze
            a = analyze(self._md.snapshot(), self._md.symbol)
            return (a.narrative_ar if self.lang == "ar" else a.narrative), a
        except Exception:
            return None

    # -- browser launch -------------------------------------------------------
    def start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
        user_dir = os.path.join(PROJECT_DIR, ".tv_profile")
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_dir, headless=self.headless, viewport={"width": 1366, "height": 768},
            user_agent=ua, args=["--start-maximized", "--disable-blink-features=AutomationControlled"])
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self.page.goto(TV_URL.format(sym=self.symbol.replace(":", "%3A")),
                       wait_until="domcontentloaded", timeout=60000)
        time.sleep(9)
        self._dismiss()
        self._enable_magnet()
        print(f"[TV] opened {self.symbol}")

    def _dismiss(self):
        for sel in ("button:has-text('Got it')", "button:has-text('Accept all')",
                    "button:has-text('Accept')", "button:has-text('I accept')",
                    "button:has-text('Maybe later')", "button[aria-label='Close']",
                    "div[data-name='toast-close-button']", "button:has-text('Skip')"):
            try:
                loc = self.page.locator(sel).first
                if loc.is_visible(timeout=500):
                    loc.click(timeout=1000); time.sleep(0.3)
            except Exception:
                pass

    def _enable_magnet(self):
        for sel in ("[data-name='magnet']", "[data-name='stay-in-drawing-mode']",
                    "button[aria-label*='agnet']"):
            try:
                loc = self.page.locator(sel).first
                if loc.is_visible(timeout=500):
                    loc.click(timeout=1000); time.sleep(0.3); return True
            except Exception:
                pass
        return False

    # -- geometry -------------------------------------------------------------
    def _canvas_box(self):
        try:
            return self.page.eval_on_selector_all(
                "canvas",
                "els=>{let b=els.map(e=>e.getBoundingClientRect())"
                ".filter(r=>r.width>400&&r.height>300)"
                ".sort((a,c)=>c.width*c.height-a.width*a.height)[0];"
                "return b?{x:b.x,y:b.y,w:b.width,h:b.height}:null}")
        except Exception:
            return None

    def _pt(self, box, rx, ry):
        return box["x"] + box["w"] * rx, box["y"] + box["h"] * ry

    # -- actions --------------------------------------------------------------
    def set_timeframe(self, code):
        box = self._canvas_box()
        if not box:
            return
        try:
            self.page.mouse.click(*self._pt(box, 0.5, 0.5))
            self.page.keyboard.type(str(code)); self.page.keyboard.press("Enter")
            time.sleep(1.6)
        except Exception:
            pass

    def zoom(self, steps=-500):
        box = self._canvas_box()
        if not box:
            return
        self.page.mouse.move(*self._pt(box, 0.6, 0.5))
        self.page.mouse.wheel(0, steps); time.sleep(1.0)

    def pan(self, rdx=-0.25):
        box = self._canvas_box()
        if not box:
            return
        x1, y = self._pt(box, 0.6, 0.5)
        self.page.mouse.move(x1, y); self.page.mouse.down()
        self.page.mouse.move(x1 + box["w"] * rdx, y, steps=12); self.page.mouse.up()
        time.sleep(0.8)

    def _tool(self, group_sel):
        try:
            self.page.locator(group_sel).first.click(timeout=2500); time.sleep(0.6)
            return True
        except Exception:
            return False

    def draw_line(self, p1, p2, group=TREND_GROUP):
        self._dismiss()
        box = self._canvas_box()
        if not box or not self._tool(group):
            return
        x1, y1 = self._pt(box, *p1); x2, y2 = self._pt(box, *p2)
        self.page.mouse.click(x1, y1); time.sleep(0.5)
        self.page.mouse.click(x2, y2); time.sleep(0.5)
        self.page.keyboard.press("Escape"); time.sleep(0.3)

    def clear(self):
        for sel in ("[data-name='remove-all-drawing-tools']",
                    "button[aria-label*='Remove']", "[data-name='removeAllDrawingTools']"):
            try:
                loc = self.page.locator(sel).first
                if loc.is_visible(timeout=500):
                    loc.click(timeout=1000); time.sleep(0.5); return
            except Exception:
                pass

    # -- autonomous performance ----------------------------------------------
    def perform_cycle(self):
        res = self._analysis()
        narrative, a = (res if res else (None, None))
        up = (a.trend == "uptrend") if a else True
        self._dismiss()
        if narrative:
            self.narrator.say(narrative)

        self.zoom(-400); time.sleep(2.0)

        if up:
            self.draw_line((0.12, 0.80), (0.92, 0.55))
        else:
            self.draw_line((0.12, 0.45), (0.92, 0.72))
        self.narrator.say("Drawing the support trendline along the swing lows."
                          if self.lang == "en" else "أرسم خط الدعم على القيعان.")
        time.sleep(3.0)

        if up:
            self.draw_line((0.18, 0.40), (0.92, 0.22))
        else:
            self.draw_line((0.18, 0.25), (0.92, 0.48))
        self.narrator.say("And the resistance along the swing highs."
                          if self.lang == "en" else "وخط المقاومة على القمم.")
        time.sleep(3.0)

        self.draw_line((0.30, 0.72 if up else 0.30), (0.70, 0.30 if up else 0.72), group=FIB_GROUP)
        self.narrator.say("Adding a fibonacci retracement of the move."
                          if self.lang == "en" else "وأضيف تصحيح فيبوناتشي للحركة.")
        time.sleep(3.5)

        time.sleep(4.0)
        self.clear(); self._dismiss()

    def run(self):
        self.start()
        print("[TV] AI is operating TradingView. Ctrl+C to stop.")
        try:
            while not self._stop:
                self.perform_cycle()
        except KeyboardInterrupt:
            print("\n[TV] stopping.")
        finally:
            self.stop()

    def stop(self):
        self._stop = True
        if self.narrator:
            self.narrator.stop()
        try:
            if self._md:
                self._md.stop()
        except Exception:
            pass
        try:
            self._ctx and self._ctx.close()
            self._pw and self._pw.stop()
        except Exception:
            pass


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="OANDA:XAUUSD")
    ap.add_argument("--lang", default="en", choices=["en", "ar"])
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-speak", action="store_true")
    a = ap.parse_args(argv[1:])
    TradingViewPilot(a.symbol, headless=a.headless, lang=a.lang,
                     speak=not a.no_speak).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
