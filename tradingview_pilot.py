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

    def speak_async(self, line):
        """Synthesize `line`, START playing it on a background thread, and return
        the audio duration (seconds) so the caller can DRAW while it speaks.
        Sets self._done when playback ends; await it with wait_done()."""
        print(f"[AI] {line}")
        self._loaded.wait(timeout=60)
        self._done = threading.Event()
        if self.enabled and self._eng and self._eng.ok:
            try:
                wav, sr = self._eng.synthesize(line, lang=self.lang)
                if wav is not None and len(wav):
                    dur = len(wav) / float(sr)

                    def _play():
                        try:
                            import sounddevice as sd
                            sd.play(wav, sr); sd.wait()
                        except Exception:
                            pass
                        self._done.set()
                    threading.Thread(target=_play, daemon=True).start()
                    return dur
            except Exception:
                pass
        dur = min(12.0, max(4.0, len(line) / 15.0))         # no-voice reading time

        def _wait():
            time.sleep(dur); self._done.set()
        threading.Thread(target=_wait, daemon=True).start()
        return dur

    def wait_done(self, timeout=25):
        d = getattr(self, "_done", None)
        if d:
            d.wait(timeout)

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
        # An unclean previous exit leaves Chromium singleton lock files behind, which
        # make launch_persistent_context fail with "profile already in use". Clear the
        # stale locks first so the browser always opens.
        for _lock in ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"):
            try:
                _p = os.path.join(user_dir, _lock)
                if os.path.lexists(_p):
                    os.remove(_p)
            except Exception:
                pass
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

    def _draw_points_timed(self, points, total_dur):
        """Click the points spread over ~total_dur, with slow visible mouse
        glides, so the DRAWING tracks the spoken explanation in real time."""
        box = self._canvas_box()
        if not box:
            return False
        n = max(1, len(points))
        budget = max(2.0, total_dur - 0.8)
        per = budget / n
        for rx, ry in points:
            x, y = self._pt(box, rx, ry)
            self.page.mouse.move(x, y, steps=26)          # slow, visible glide
            time.sleep(min(0.8, per * 0.35))
            self.page.mouse.click(x, y)
            time.sleep(min(per * 0.65, 3.0))
        return True

    def _tool_plan(self, a, up):
        """Tools the AI explores one by one, with explanations tied to the LIVE
        numbers (price / levels / fib), so every drawing is explained for real."""
        price = a.price if a else 0.0
        res_list = sorted([p for p, s, k in (a.levels if a else []) if p > price])
        sup_list = sorted([p for p, s, k in (a.levels if a else []) if p < price], reverse=True)
        res = res_list[0] if res_list else price * 1.01
        sup = sup_list[0] if sup_list else price * 0.99
        rng = max(res - sup, price * 0.008)
        golden = sup + 0.618 * rng
        target = price + rng
        nm = {"XAUUSD": "gold", "GOLD": "gold", "BTCUSD": "bitcoin", "BTCUSDT": "bitcoin",
              "ETHUSD": "ethereum", "ETHUSDT": "ethereum"}.get(
            self.symbol.split(":")[-1].upper(), self.symbol.split(":")[-1])
        nm_ar = {"gold": "الذهب", "bitcoin": "البيتكوين", "ethereum": "الإيثيريوم"}.get(nm, nm)

        def f(x):
            return f"{x:,.0f}"

        return [
            dict(group=TREND_GROUP,
                 points=[(0.12, 0.80 if up else 0.45), (0.92, 0.56 if up else 0.72)],
                 en=(f"Watch this — I'm drawing the support trend line, connecting the swing lows. "
                     f"As long as {nm} holds above it, the buyers stay in control; it's rising right into price near {f(price)}."
                     if up else
                     f"I'm drawing the resistance trend line down the swing highs. While {nm} stays below it, sellers keep the upper hand near {f(price)}."),
                 ar=(f"انظر، أرسم خط الدعم واصلاً بين القيعان. وطالما بقي {nm_ar} فوقه فالمشترون مسيطرون، وهو يصعد قرب {f(price)}."
                     if up else
                     f"أرسم خط المقاومة على القمم. وطالما بقي {nm_ar} تحته يبقى البائعون مسيطرين قرب {f(price)}.")),
            dict(group=TREND_GROUP,
                 points=[(0.16, 0.40 if up else 0.26), (0.92, 0.24 if up else 0.50)],
                 en=(f"Up here is the resistance line, capping the swing highs around {f(res)}. "
                     f"A clean break above {f(res)} is the trigger for the next leg higher."),
                 ar=(f"وهنا خط المقاومة، يحدّ القمم قرب {f(res)}. واختراق {f(res)} بوضوح هو إشارة الموجة الصاعدة التالية.")),
            dict(group=FIB_GROUP,
                 points=[(0.30, 0.72 if up else 0.30), (0.66, 0.32 if up else 0.72)],
                 en=(f"Now I'll map the fibonacci of this leg. The level that matters is the point-six-one-eight "
                     f"retracement around {f(golden)} — the golden pocket, where dips usually get bought."),
                 ar=(f"الآن أرسم فيبوناتشي لهذه الموجة. وأهم مستوى هو 0.618 قرب {f(golden)}، المنطقة الذهبية حيث يُشترى التصحيح عادة.")),
            dict(group=POS_GROUP,
                 points=[(0.60, 0.58), (0.86, 0.40 if up else 0.74)],
                 en=(f"And here's the trade idea — a long from {f(price)}, targeting {f(target)} at resistance, "
                     f"with the stop tucked under support at {f(sup)}. That's a clean risk-to-reward."),
                 ar=(f"وهذه فكرة الصفقة: شراء من {f(price)} بهدف {f(target)} عند المقاومة، ووقف الخسارة تحت الدعم عند {f(sup)}. نسبة مخاطرة إلى عائد جيدة.")),
        ]

    # -- synchronized performance: DRAW while it EXPLAINS --------------------
    def perform_cycle(self):
        res = self._analysis()
        narrative, a = (res if res else (None, None))
        up = (a.trend == "uptrend") if a else True
        self._dismiss()
        # overall live read while slowly framing the chart
        if narrative:
            dur = self.narrator.speak_async(narrative)
            self.zoom(-300)
            for _ in range(max(1, int(dur / 1.4))):
                if self._stop:
                    break
                time.sleep(1.2)
            self.narrator.wait_done()
        else:
            self.zoom(-300); time.sleep(1.0)
        # tools, one by one — arm tool, then DRAW it AS the voice explains it
        for tool in self._tool_plan(a, up):
            if self._stop:
                break
            self._dismiss()
            self._tool(tool["group"])                     # arm the drawing tool
            dur = self.narrator.speak_async(tool["ar"] if self.lang == "ar" else tool["en"])
            self._draw_points_timed(tool["points"], dur)  # draw in sync with speech
            self.narrator.wait_done()
            try:
                self.page.keyboard.press("Escape")
            except Exception:
                pass
            time.sleep(0.5)
            self._delete_last()                            # clear it for the next tool
            time.sleep(0.8)
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
