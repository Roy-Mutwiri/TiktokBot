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
RECT_GROUP = "[data-name='linetool-group-geometric-shapes']"
POS_GROUP = "[data-name='linetool-group-prediction-and-measurement']"


class _Narrator:
    """Speaks analysis lines via the multilingual voice (degrades to print)."""

    def __init__(self, lang="en", enabled=True):
        self.lang, self.enabled = lang, enabled
        self.ok = False
        self._eng = None
        self._q = []
        self._lock = threading.Lock()
        self._stop = False
        self._loaded = threading.Event()
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()
        else:
            self._loaded.set()

    def _run(self):
        try:
            from multilingual_tts import MultilingualTTSBackend
            self._eng = MultilingualTTSBackend()
            self.ok = self._eng.ok
        except Exception as exc:
            print(f"[narrator] voice off ({exc})")
        self._loaded.set()
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

    def speak_sync(self, line):
        """Speak `line` and BLOCK until it's finished (so we delete the drawing
        only after the explanation is done). Falls back to a reading-time pause."""
        print(f"[AI] {line}")
        self._loaded.wait(timeout=60)
        if self.enabled and self._eng and self._eng.ok:
            try:
                import sounddevice as sd
                wav, sr = self._eng.synthesize(line, lang=self.lang)
                if wav is not None and len(wav):
                    sd.play(wav, sr); sd.wait(); return
            except Exception:
                pass
        time.sleep(min(9.0, max(3.0, len(line) / 16.0)))

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

    # -- one-tool-at-a-time helpers ------------------------------------------
    def _draw_points(self, points, pause=0.7):
        """Click a sequence of canvas-relative points (slowly), drawing a tool."""
        box = self._canvas_box()
        if not box:
            return False
        for rx, ry in points:
            x, y = self._pt(box, rx, ry)
            self.page.mouse.move(x, y, steps=8); time.sleep(0.25)
            self.page.mouse.click(x, y); time.sleep(pause)
        return True

    def _draw_with_tool(self, group_sel, points):
        self._dismiss()
        if not self._tool(group_sel):
            return False
        ok = self._draw_points(points)
        try:
            self.page.keyboard.press("Escape"); time.sleep(0.3)
        except Exception:
            pass
        return ok

    def _delete_last(self, n=1):
        """Undo the last drawing(s) so the chart is clean for the next tool."""
        try:
            self.page.keyboard.press("Escape"); time.sleep(0.2)
            for _ in range(max(1, n)):
                self.page.keyboard.press("Control+z"); time.sleep(0.7)
        except Exception:
            pass

    def _tool_plan(self, up):
        """The ordered tools the AI explores one by one, with what to say."""
        return [
            dict(group=TREND_GROUP,
                 points=[(0.12, 0.80 if up else 0.45), (0.92, 0.56 if up else 0.72)],
                 en="First, the trend line. I connect the swing lows; as long as price holds above this line, the uptrend stays intact."
                    if up else "First, the trend line. I connect the swing highs; while price stays below it, the downtrend holds.",
                 ar="أولاً، خط الاتجاه. أصل بين القيعان؛ وطالما السعر فوق هذا الخط يبقى الاتجاه الصاعد سليماً."
                    if up else "أولاً، خط الاتجاه. أصل بين القمم؛ وطالما السعر تحته يبقى الاتجاه هابطاً."),
            dict(group=TREND_GROUP,
                 points=[(0.16, 0.40 if up else 0.26), (0.92, 0.24 if up else 0.50)],
                 en="Next, the opposite trend line. A clean break of it would signal a shift in momentum.",
                 ar="ثم خط الاتجاه المقابل. واختراقه بوضوح يشير إلى تغيّر في الزخم."),
            dict(group=FIB_GROUP,
                 points=[(0.30, 0.72 if up else 0.30), (0.66, 0.32 if up else 0.72)],
                 en="Now a fibonacci retracement of the move. The point six one eight golden pocket is where pullbacks often find support.",
                 ar="الآن تصحيح فيبوناتشي للحركة. ومنطقة 0.618 الذهبية غالباً ما يجد عندها السعر دعماً."),
            dict(group=POS_GROUP,
                 points=[(0.60, 0.58), (0.86, 0.40 if up else 0.74)],
                 en="And the position tool lays out the entry, the target, and the risk to reward of the trade idea.",
                 ar="وأداة المركز تُظهر الدخول والهدف ونسبة المخاطرة إلى العائد لفكرة الصفقة."),
        ]

    # -- deliberate, one-tool-at-a-time performance --------------------------
    def perform_cycle(self):
        res = self._analysis()
        narrative, a = (res if res else (None, None))
        up = (a.trend == "uptrend") if a else True
        self._dismiss()
        self.zoom(-350); time.sleep(1.5)
        if narrative:
            self.narrator.speak_sync(narrative)          # the overall read first
        for tool in self._tool_plan(up):
            if self._stop:
                break
            ok = self._draw_with_tool(tool["group"], tool["points"])   # draw it
            time.sleep(0.6)
            self.narrator.speak_sync(tool["ar"] if self.lang == "ar" else tool["en"])  # explain it
            time.sleep(0.8)
            if ok:
                self._delete_last()                       # delete once done explaining
            time.sleep(1.1)
        # close any stray menu/tool, then loop with a fresh read
        try:
            self.page.keyboard.press("Escape")
        except Exception:
            pass
        self._dismiss()

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
