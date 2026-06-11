# =============================================================================
# engines/ai_chart.py  —  AI-operated analytical chart (camera + drawing tools)
# -----------------------------------------------------------------------------
# Renders REAL candles and lets the AI "operate" the chart like a human analyst:
#   * camera  : scroll (pan) and zoom, eased smoothly toward targets each frame
#   * tools   : draws trendlines, horizontal S/R, fibonacci, zones, swing markers
#               and labels straight from market_analysis, with an animated pen
#               cursor so it looks like the AI is actively drawing
#
# Drop-in for the old TradingView scene: same `render()` returning a BGR frame.
#   chart = AIChart("PAXGUSDT", size=(512, 512))
#   chart.set_data(ohlc, analysis)            # from MarketData + analyze()
#   chart.focus(start_idx, end_idx)           # AI camera move (optional)
#   frame = chart.render()                    # call once per video frame
# =============================================================================

import time

import numpy as np
import cv2

# palette (BGR) — matches the futuristic studio theme
BG = (14, 10, 6)
GRID = (40, 32, 24)
FG = (220, 225, 235)
MUTED = (150, 150, 160)
UP = (120, 220, 130)
DN = (95, 95, 240)
CYAN = (255, 230, 60)
WICK_UP = (90, 180, 100)
WICK_DN = (80, 80, 200)
AXIS_W = 70
HEADER_H = 46
TIME_H = 22


def _ease(cur, tgt, rate):
    return cur + (tgt - cur) * rate


class AIChart:
    def __init__(self, symbol="PAXGUSDT", size=(512, 512)):
        self.symbol = symbol
        self.W, self.H = size
        self.ohlc = None
        self.analysis = None
        # camera (float for smooth easing)
        self.view_end = 0.0          # rightmost visible bar index
        self.bars = 90.0             # visible bar count (zoom)
        self.view_end_t = 0.0
        self.bars_t = 90.0
        self.pmin = self.pmax = None
        self.pmin_t = self.pmax_t = None
        # animated tool the AI is currently "drawing"
        self.active = None           # {"type","p1","p2","progress","color","label"}
        self._t0 = time.time()

    # -- data / camera --------------------------------------------------------
    def set_data(self, ohlc, analysis=None):
        self.ohlc = ohlc
        if analysis is not None:
            self.analysis = analysis
        if ohlc is not None and len(ohlc):
            n = len(ohlc)
            if self.view_end_t == 0.0:        # first load: snap to latest
                self.view_end = self.view_end_t = n - 1
                self.bars = self.bars_t = min(90.0, n)
            else:
                # follow the latest candle if we were already at the right edge
                if self.view_end_t >= n - 3:
                    self.view_end_t = n - 1

    def focus(self, start_idx, end_idx, pad=4):
        """AI camera move: frame [start_idx, end_idx] (scroll + zoom together)."""
        if self.ohlc is None:
            return
        n = len(self.ohlc)
        end_idx = min(n - 1, end_idx + pad)
        start_idx = max(0, start_idx - pad)
        self.view_end_t = float(end_idx)
        self.bars_t = float(max(20, min(n, end_idx - start_idx + 1)))

    def follow_live(self, bars=90):
        if self.ohlc is None:
            return
        self.view_end_t = float(len(self.ohlc) - 1)
        self.bars_t = float(bars)

    def begin_tool(self, kind, p1, p2, color, label="", dur=1.1):
        """Start animating a tool stroke from p1=(idx,price) to p2=(idx,price)."""
        self.active = {"type": kind, "p1": p1, "p2": p2, "color": color,
                       "label": label, "progress": 0.0, "dur": dur,
                       "t0": time.time()}

    # -- per-frame step (easing) ---------------------------------------------
    def _visible_range(self):
        n = len(self.ohlc)
        i1 = int(round(self.view_end))
        i1 = max(0, min(n - 1, i1))
        b = int(round(self.bars))
        i0 = max(0, i1 - b + 1)
        return i0, i1

    def _step(self):
        # ease camera
        self.view_end = _ease(self.view_end, self.view_end_t, 0.18)
        self.bars = _ease(self.bars, self.bars_t, 0.12)
        i0, i1 = self._visible_range()
        seg = self.ohlc[i0:i1 + 1]
        lo = float(seg[:, 3].min()); hi = float(seg[:, 2].max())
        pad = (hi - lo) * 0.08 + 1e-6
        self.pmin_t, self.pmax_t = lo - pad, hi + pad
        if self.pmin is None:
            self.pmin, self.pmax = self.pmin_t, self.pmax_t
        else:
            self.pmin = _ease(self.pmin, self.pmin_t, 0.16)
            self.pmax = _ease(self.pmax, self.pmax_t, 0.16)
        # advance active tool stroke
        if self.active is not None:
            p = (time.time() - self.active["t0"]) / max(0.2, self.active["dur"])
            self.active["progress"] = min(1.0, p)

    # -- coordinate mapping ---------------------------------------------------
    def _geom(self):
        pl, pr = 6, self.W - AXIS_W
        pt, pb = HEADER_H, self.H - TIME_H
        return pl, pr, pt, pb

    def _x_of(self, idx, i0, i1, pl, pr):
        span = max(1, i1 - i0)
        return int(pl + (idx - i0) / span * (pr - pl - 6))

    def _y_of(self, price, pt, pb):
        rng = max(1e-9, self.pmax - self.pmin)
        return int(pt + (self.pmax - price) / rng * (pb - pt))

    # -- render ---------------------------------------------------------------
    def render(self, speaking=False):
        img = np.full((self.H, self.W, 3), BG, np.uint8)
        if self.ohlc is None or len(self.ohlc) < 5:
            cv2.putText(img, "loading market data...", (16, self.H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MUTED, 1, cv2.LINE_AA)
            return img
        self._step()
        i0, i1 = self._visible_range()
        pl, pr, pt, pb = self._geom()
        xo = lambda i: self._x_of(i, i0, i1, pl, pr)
        yo = lambda p: self._y_of(p, pt, pb)

        # grid + price axis
        for gy in range(5):
            y = int(pt + gy / 4 * (pb - pt))
            cv2.line(img, (pl, y), (pr, y), GRID, 1)
            price = self.pmax - gy / 4 * (self.pmax - self.pmin)
            cv2.putText(img, f"{price:,.1f}", (pr + 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, MUTED, 1, cv2.LINE_AA)
        cv2.line(img, (pr, pt), (pr, pb), GRID, 1)

        # candles
        span = max(1, i1 - i0)
        cw = max(1, int((pr - pl - 6) / span * 0.62))
        for i in range(i0, i1 + 1):
            o, h, l, c = self.ohlc[i, 1], self.ohlc[i, 2], self.ohlc[i, 3], self.ohlc[i, 4]
            x = xo(i)
            up = c >= o
            col = UP if up else DN
            cv2.line(img, (x, yo(h)), (x, yo(l)), WICK_UP if up else WICK_DN, 1)
            y1, y2 = yo(o), yo(c)
            if abs(y1 - y2) < 1:
                y2 = y1 + 1
            cv2.rectangle(img, (x - cw // 2, min(y1, y2)), (x + cw // 2, max(y1, y2)), col, -1)

        self._draw_overlays(img, i0, i1, xo, yo, pl, pr, pt, pb)
        self._draw_active_tool(img, xo, yo)
        self._draw_header(img, speaking)
        self._draw_price_tag(img, xo, yo, i1, pr)
        return img

    def _draw_overlays(self, img, i0, i1, xo, yo, pl, pr, pt, pb):
        if not self.analysis:
            return
        for d in self.analysis.drawings:
            t = d["type"]
            if t == "ema":
                vals = d["values"]
                pts = [(xo(i), yo(float(vals[i]))) for i in range(i0, i1 + 1)]
                for j in range(1, len(pts)):
                    cv2.line(img, pts[j - 1], pts[j], d["color"], 1, cv2.LINE_AA)
            elif t == "hline":
                p = d["price"]
                if not (self.pmin <= p <= self.pmax):
                    continue
                y = yo(p)
                col = (90, 220, 90) if d["kind"] == "support" else (95, 95, 240)
                for x in range(pl, pr, 10):
                    cv2.line(img, (x, y), (x + 5, y), col, 1)
                cv2.putText(img, d["label"], (pl + 4, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
            elif t == "trendline":
                (a_i, a_p), (b_i, b_p) = d["p1"], d["p2"]
                if a_i < i0 - 5 and b_i < i0:
                    continue
                cv2.line(img, (xo(a_i), yo(a_p)), (xo(b_i), yo(b_p)), d["color"], 2, cv2.LINE_AA)
            elif t == "fib":
                hi, lo = d["hi"], d["lo"]
                x_l = xo(max(i0, min(d["hi_i"], d["lo_i"])))
                for f, lvl in d["levels"]:
                    if not (self.pmin <= lvl <= self.pmax):
                        continue
                    y = yo(lvl)
                    cv2.line(img, (x_l, y), (pr, y), (120, 110, 170), 1, cv2.LINE_AA)
                    cv2.putText(img, f"{f:.3f}", (x_l + 2, y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 140, 200), 1, cv2.LINE_AA)
            elif t == "marker":
                i = d["idx"]
                if not (i0 <= i <= i1):
                    continue
                x, y = xo(i), yo(d["price"])
                if d["kind"] == "high":
                    cv2.drawMarker(img, (x, y - 6), (90, 200, 250), cv2.MARKER_TRIANGLE_DOWN, 7, 1)
                else:
                    cv2.drawMarker(img, (x, y + 6), (130, 230, 140), cv2.MARKER_TRIANGLE_UP, 7, 1)

    def _draw_active_tool(self, img, xo, yo):
        a = self.active
        if not a:
            return
        (a_i, a_p), (b_i, b_p) = a["p1"], a["p2"]
        prog = a["progress"]
        x1, y1 = xo(a_i), yo(a_p)
        x2, y2 = xo(b_i), yo(b_p)
        cx = int(x1 + (x2 - x1) * prog); cy = int(y1 + (y2 - y1) * prog)
        cv2.line(img, (x1, y1), (cx, cy), a["color"], 2, cv2.LINE_AA)
        # pen cursor at the leading edge
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 7, a["color"], 1, cv2.LINE_AA)
        cv2.line(img, (cx + 5, cy - 5), (cx + 13, cy - 13), (255, 255, 255), 1, cv2.LINE_AA)
        if prog >= 1.0 and a["label"]:
            cv2.putText(img, a["label"], (x2 + 4, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, a["color"], 1, cv2.LINE_AA)

    def _draw_header(self, img, speaking):
        a = self.analysis
        name = self.symbol
        price = a.price if a else float(self.ohlc[-1, 4])
        cv2.rectangle(img, (0, 0), (self.W, HEADER_H), (22, 16, 10), -1)
        cv2.putText(img, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, f"{price:,.1f}", (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
        if a:
            tx = min(self.W - 230, 24 + tw)        # never collide with the title
            tcol = UP if a.bias == "bullish" else DN if a.bias == "bearish" else MUTED
            cv2.putText(img, a.trend.upper(), (tx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tcol, 1, cv2.LINE_AA)
            cv2.putText(img, f"RSI {a.rsi:.0f}", (tx, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (90, 200, 250) if a.rsi >= 70 else (130, 230, 140) if a.rsi <= 30 else MUTED, 1, cv2.LINE_AA)
        # AI badge + LIVE dot
        cv2.putText(img, "AI ANALYST", (self.W - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 60), 1, cv2.LINE_AA)
        dot = (60, 70, 240)
        if (int((time.time() - self._t0) * 2) % 2) == 0:
            cv2.circle(img, (self.W - 162, 34), 4, dot, -1, cv2.LINE_AA)
        cv2.putText(img, "LIVE", (self.W - 150, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot, 1, cv2.LINE_AA)

    def _draw_price_tag(self, img, xo, yo, i1, pr):
        price = float(self.ohlc[i1, 4])
        y = yo(price)
        up = self.analysis.bias == "bullish" if self.analysis else True
        col = UP if up else DN
        cv2.line(img, (6, y), (pr, y), (col[0] // 2, col[1] // 2, col[2] // 2), 1)
        cv2.rectangle(img, (pr, y - 8), (self.W, y + 8), col, -1)
        cv2.putText(img, f"{price:,.1f}", (pr + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (10, 10, 10), 1, cv2.LINE_AA)

    def startup_check(self):
        return True, f"AI chart ready ({self.symbol}, {self.W}x{self.H})"
