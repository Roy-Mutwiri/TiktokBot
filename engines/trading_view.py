# =============================================================================
# engines/trading_view.py
# -----------------------------------------------------------------------------
# A self-contained, real-time animated trading chart used as the FALLBACK scene
# when the operator's face isn't visible to the webcam. Each render() advances a
# random-walk price one tick, grows the forming candle, scrolls the series, and
# draws a convincing live trading screen: candlesticks, an EMA line, a price
# axis, volume bars, a header with the live price + day change, a pulsing LIVE
# badge and a scrolling ticker.
#
#   chart = TradingView("XAUUSD")
#   frame = chart.render()          # 512x512 BGR, call once per video frame
# =============================================================================

import os
import sys
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2

# random is fine here (normal Python process, not the workflow sandbox)
import random as _random

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SIZE = 512
MAX_CANDLES = 60               # visible candles
FRAMES_PER_CANDLE = 8          # ~0.5s per candle at ~15fps
MARGIN_TOP = 46
MARGIN_BOTTOM = 40             # ticker bar
AXIS_W = 64                    # right-side price axis width
VOL_H = 46                     # volume panel height (above ticker)

BG = (12, 12, 16)
GRID = (34, 34, 40)
UP = (96, 200, 110)            # bullish candle (BGR — green)
DOWN = (70, 70, 232)           # bearish candle (BGR — red)
EMA_COLOR = (230, 190, 70)     # moving-average line (cyan-ish)
TEXT = (220, 220, 220)
SUBTLE = (140, 140, 150)
GOLD = (0, 215, 255)
LIVE_RED = (0, 0, 220)


class TradingView:
    """Animated candlestick chart; advance one tick per render()."""

    def __init__(self, symbol="XAUUSD", start_price=2375.0):
        self.symbol = symbol
        self.price = float(start_price)
        self.day_open = float(start_price)
        self.candles = collections.deque(maxlen=MAX_CANDLES)   # [o,h,l,c,vol]
        self._ema = float(start_price)
        self._frame = 0
        self._ticker_off = 0
        self._blink = 0

        # seed a plausible history so the first frame already looks live
        p = start_price
        for _ in range(MAX_CANDLES):
            o = p
            c = o + _random.uniform(-2.5, 2.5)
            hi = max(o, c) + _random.uniform(0, 1.8)
            lo = min(o, c) - _random.uniform(0, 1.8)
            vol = _random.uniform(0.3, 1.0)
            self.candles.append([o, hi, lo, c, vol])
            p = c
        self.price = p
        self.day_open = p - _random.uniform(-12, 12)
        self._cur = [p, p, p, p, 0.0]     # forming candle O,H,L,C,vol

    # -------------------------------------------------------------------------
    def startup_check(self):
        return True, f"trading view ready ({self.symbol})"

    def reset_price_drift(self):
        """Re-anchor the day-open to 'now' (called when the scene re-activates)."""
        self.day_open = self.price - _random.uniform(-8, 8)

    def set_market(self, symbol, price):
        """Switch instruments and rebuild scale-appropriate visual history."""
        symbol = str(symbol)
        price = float(price)
        if symbol == self.symbol and abs(price - self.price) < max(1.0, price * 0.02):
            self.price = price
            return
        self.symbol = symbol
        self.price = price
        self.day_open = price
        self._ema = price
        self.candles.clear()
        step = max(price * 0.00035, 0.15)
        p = price
        for _ in range(MAX_CANDLES):
            o = p
            c = max(0.01, o + _random.uniform(-step, step))
            hi = max(o, c) + _random.uniform(0, step * 0.7)
            lo = min(o, c) - _random.uniform(0, step * 0.7)
            self.candles.append([o, hi, lo, c, _random.uniform(0.3, 1.0)])
            p = c
        self.price = price
        self._cur = [price, price, price, price, 0.0]

    # -------------------------------------------------------------------------
    def _tick(self):
        """Advance the price one step and update the forming candle."""
        drift = _random.uniform(-1.3, 1.3)
        if _random.random() < 0.05:           # occasional sharper move
            drift *= 4.0
        self.price = max(1.0, self.price + drift)

        o, h, l, c, vol = self._cur
        c = self.price
        h = max(h, c); l = min(l, c)
        vol += abs(drift)
        self._cur = [o, h, l, c, vol]
        self._ema = self._ema * 0.92 + self.price * 0.08

        self._frame += 1
        if self._frame % FRAMES_PER_CANDLE == 0:
            self.candles.append(self._cur)
            self._cur = [c, c, c, c, 0.0]     # next candle opens at last close

    # -------------------------------------------------------------------------
    def render(self, speaking=False):
        """Advance one tick and draw the chart. Returns a 512x512 BGR frame."""
        try:
            self._tick()
            img = np.full((SIZE, SIZE, 3), BG, np.uint8)

            series = list(self.candles) + [self._cur]
            highs = [c[1] for c in series]
            lows = [c[2] for c in series]
            pmax = max(highs); pmin = min(lows)
            if pmax - pmin < 1e-3:
                pmax += 1; pmin -= 1
            pad = (pmax - pmin) * 0.08
            pmax += pad; pmin -= pad

            plot_l = 6
            plot_r = SIZE - AXIS_W
            plot_t = MARGIN_TOP
            plot_b = SIZE - MARGIN_BOTTOM - VOL_H
            plot_h = plot_b - plot_t
            plot_w = plot_r - plot_l

            def y_of(p):
                return int(plot_t + (pmax - p) / (pmax - pmin) * plot_h)

            # --- grid + price axis labels ---
            for g in range(5):
                yy = plot_t + int(g * plot_h / 4)
                cv2.line(img, (plot_l, yy), (plot_r, yy), GRID, 1)
                pv = pmax - (pmax - pmin) * g / 4
                cv2.putText(img, f"{pv:,.1f}", (plot_r + 6, yy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, SUBTLE, 1, cv2.LINE_AA)
            for g in range(0, MAX_CANDLES, 10):
                xx = plot_l + int(g * plot_w / MAX_CANDLES)
                cv2.line(img, (xx, plot_t), (xx, plot_b), GRID, 1)

            # --- candles + volume ---
            n = len(series)
            cw = max(2, int(plot_w / MAX_CANDLES))
            body = max(1, cw - 2)
            volmax = max(c[4] for c in series) or 1.0
            for i, (o, h, l, c, vol) in enumerate(series):
                cx = plot_l + int((i + 0.5) * plot_w / MAX_CANDLES)
                col = UP if c >= o else DOWN
                cv2.line(img, (cx, y_of(h)), (cx, y_of(l)), col, 1)
                y1 = y_of(max(o, c)); y2 = y_of(min(o, c))
                if y2 - y1 < 1:
                    y2 = y1 + 1
                cv2.rectangle(img, (cx - body // 2, y1), (cx + body // 2, y2), col, -1)
                vh = int((vol / volmax) * VOL_H)
                cv2.rectangle(img, (cx - body // 2, SIZE - MARGIN_BOTTOM - vh),
                              (cx + body // 2, SIZE - MARGIN_BOTTOM),
                              (col[0] // 2, col[1] // 2, col[2] // 2), -1)

            # --- EMA line ---
            ema = self.day_open
            pts = []
            for i, (o, h, l, c, vol) in enumerate(series):
                ema = ema * 0.9 + c * 0.1
                cx = plot_l + int((i + 0.5) * plot_w / MAX_CANDLES)
                pts.append((cx, y_of(ema)))
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, EMA_COLOR, 1,
                              cv2.LINE_AA)

            # --- live price line + tag ---
            yp = y_of(self.price)
            for x in range(plot_l, plot_r, 10):
                cv2.line(img, (x, yp), (x + 5, yp), (90, 90, 100), 1)
            up = self.price >= self.day_open
            tagcol = UP if up else DOWN
            cv2.rectangle(img, (plot_r, yp - 9), (SIZE, yp + 9), tagcol, -1)
            cv2.putText(img, f"{self.price:,.1f}", (plot_r + 4, yp + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (15, 15, 15), 1, cv2.LINE_AA)

            # --- header ---
            chg = self.price - self.day_open
            pct = chg / self.day_open * 100 if self.day_open else 0
            cv2.putText(img, f"{self.symbol}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, "M1", (118, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, SUBTLE,
                        1, cv2.LINE_AA)
            arrow = "+" if chg >= 0 else ""
            cv2.putText(img, f"{self.price:,.2f}  {arrow}{chg:,.2f} ({arrow}{pct:.2f}%)",
                        (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tagcol, 1, cv2.LINE_AA)

            # --- pulsing LIVE badge (top-right) ---
            self._blink = (self._blink + 1) % 40
            pulse = 0.4 + 0.6 * abs(20 - self._blink) / 20.0
            cv2.circle(img, (SIZE - 78, 22), 5, tuple(int(c * pulse) for c in LIVE_RED), -1)
            cv2.putText(img, "LIVE", (SIZE - 66, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (60, 60, 230), 1, cv2.LINE_AA)

            # --- scrolling ticker ---
            cv2.rectangle(img, (0, SIZE - MARGIN_BOTTOM + 6), (SIZE, SIZE), (10, 10, 12), -1)
            tk = (f"  {self.symbol} {self.price:,.2f}   GOLD SIGNALS   @xauusa2   "
                  f"DAY {arrow}{pct:.2f}%   ") * 3
            (tw, _), _ = cv2.getTextSize(tk, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            self._ticker_off = (self._ticker_off - 2) % tw
            x = -self._ticker_off
            while x < SIZE:
                cv2.putText(img, tk, (x, SIZE - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            GOLD, 1, cv2.LINE_AA)
                x += tw
            return img
        except Exception as exc:
            img = np.full((SIZE, SIZE, 3), BG, np.uint8)
            cv2.putText(img, "CHARTS", (180, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        GOLD, 2, cv2.LINE_AA)
            return img


if __name__ == "__main__":
    tv = TradingView("XAUUSD")
    print("[TRADING]", tv.startup_check()[1])
    out = None
    for _ in range(30):
        out = tv.render()
    cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "_chart_test.jpg"), out)
    print("[TRADING] wrote _chart_test.jpg", out.shape)
