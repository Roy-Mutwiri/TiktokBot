# =============================================================================
# engines/market_analysis.py  —  the AI's market-analysis brain
# -----------------------------------------------------------------------------
# Pure-numpy technical analysis over an OHLCV buffer. Produces:
#   * indicators      : EMA(20/50), RSI(14), ATR(14)
#   * structure       : swing highs/lows (fractal pivots)
#   * support/resist  : clustered horizontal levels with strength
#   * trendlines      : fitted through recent swing highs and lows
#   * fibonacci       : retracement of the dominant recent swing
#   * trend / signals : HH-HL vs LH-LL, RSI extremes, level tests, breakouts
#   * drawings        : a render-ready list the AI chart draws as "tools"
#   * narrative       : an English (+ Arabic) spoken summary for the avatar
#
# Coordinates in `drawings` are (bar_index, price) so the chart camera can map
# them to pixels at any zoom/scroll.
# =============================================================================

import numpy as np


def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close, n=14):
    d = np.diff(close)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = np.empty(len(close)); rd = np.empty(len(close))
    ru[:] = np.nan; rd[:] = np.nan
    if len(close) <= n:
        return np.full(len(close), 50.0)
    au = up[:n].mean(); ad = dn[:n].mean()
    ru[n] = au; rd[n] = ad
    for i in range(n + 1, len(close)):
        au = (au * (n - 1) + up[i - 1]) / n
        ad = (ad * (n - 1) + dn[i - 1]) / n
        ru[i] = au; rd[i] = ad
    rs = ru / np.where(rd == 0, 1e-9, rd)
    out = 100 - 100 / (1 + rs)
    out[:n + 1] = out[n + 1] if len(out) > n + 1 else 50.0
    return np.nan_to_num(out, nan=50.0)


def atr(high, low, close, n=14):
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    out = np.empty(len(close)); out[0] = tr[0] if len(tr) else 0.0
    a = 1.0 / n
    cur = tr[:n].mean() if len(tr) >= n else (tr.mean() if len(tr) else 0.0)
    for i in range(1, len(close)):
        t = tr[i - 1] if i - 1 < len(tr) else cur
        cur = a * t + (1 - a) * cur
        out[i] = cur
    return out


def pivots(high, low, k=3):
    """Fractal swing points. Returns (idx_highs, idx_lows)."""
    ph, pl = [], []
    n = len(high)
    for i in range(k, n - k):
        win_h = high[i - k:i + k + 1]
        win_l = low[i - k:i + k + 1]
        if high[i] == win_h.max() and high[i] > high[i - 1] and high[i] >= high[i + 1]:
            ph.append(i)
        if low[i] == win_l.min() and low[i] < low[i - 1] and low[i] <= low[i + 1]:
            pl.append(i)
    return ph, pl


def cluster_levels(prices, tol):
    """Group nearby pivot prices into S/R levels. Returns [(price, strength)]."""
    if not len(prices):
        return []
    s = np.sort(np.asarray(prices, dtype=np.float64))
    groups = [[s[0]]]
    for p in s[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [(float(np.mean(g)), len(g)) for g in groups]


def _fit_line(idxs, vals):
    """Least-squares slope/intercept for points (idx, val)."""
    if len(idxs) < 2:
        return None
    A = np.vstack([np.asarray(idxs, float), np.ones(len(idxs))]).T
    m, b = np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)[0]
    return float(m), float(b)


class Analysis:
    """Container for one analysis pass + the render-ready drawing list."""

    def __init__(self):
        self.price = 0.0
        self.trend = "range"
        self.bias = "neutral"
        self.rsi = 50.0
        self.atr = 0.0
        self.ema20 = None
        self.ema50 = None
        self.levels = []          # [(price, strength, kind)]  kind: 'support'/'resistance'
        self.drawings = []        # render-ready
        self.focus = None         # (start_idx, end_idx) the camera should frame
        self.signals = []         # human-readable bullet signals
        self.narrative = ""
        self.narrative_ar = ""
        self.key_level = None


def analyze(ohlc, symbol="GOLD"):
    """Run the full analysis over ohlc [N,6] (ts,o,h,l,c,v). Returns Analysis."""
    a = Analysis()
    if ohlc is None or len(ohlc) < 30:
        return a
    o, h, l, c, v = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 4], ohlc[:, 5]
    n = len(c)
    price = float(c[-1])
    a.price = price
    e20, e50 = ema(c, 20), ema(c, 50)
    r = rsi(c, 14)
    at = atr(h, l, c, 14)
    a.ema20, a.ema50 = e20, e50
    a.rsi = float(r[-1])
    a.atr = float(at[-1])
    tol = max(a.atr * 0.6, price * 0.0015)

    ph, pl = pivots(h, l, k=3)
    # ---- support / resistance from pivots --------------------------------
    res = cluster_levels([h[i] for i in ph], tol)
    sup = cluster_levels([l[i] for i in pl], tol)
    levels = []
    for p, s in res:
        levels.append((p, s, "resistance" if p >= price else "support"))
    for p, s in sup:
        levels.append((p, s, "support" if p <= price else "resistance"))
    # keep the strongest few near current price
    levels.sort(key=lambda t: (abs(t[0] - price)))
    a.levels = levels[:6]

    # ---- trend from swing structure --------------------------------------
    last_h = [h[i] for i in ph[-3:]]
    last_l = [l[i] for i in pl[-3:]]
    up = len(last_h) >= 2 and last_h[-1] > last_h[0] and len(last_l) >= 2 and last_l[-1] > last_l[0]
    dn = len(last_h) >= 2 and last_h[-1] < last_h[0] and len(last_l) >= 2 and last_l[-1] < last_l[0]
    ema_up = e20[-1] > e50[-1] and e20[-1] > e20[-5]
    ema_dn = e20[-1] < e50[-1] and e20[-1] < e20[-5]
    if up or (ema_up and not dn):
        a.trend, a.bias = "uptrend", "bullish"
    elif dn or (ema_dn and not up):
        a.trend, a.bias = "downtrend", "bearish"
    else:
        a.trend, a.bias = "range", "neutral"

    # ---- fibonacci on the dominant recent swing --------------------------
    win = min(n, 120)
    seg = slice(n - win, n)
    hi_i = int(np.argmax(h[seg])) + (n - win)
    lo_i = int(np.argmin(l[seg])) + (n - win)
    swing_hi, swing_lo = float(h[hi_i]), float(l[lo_i])
    fib_levels = []
    if swing_hi - swing_lo > tol:
        up_move = hi_i > lo_i
        for f in (0.236, 0.382, 0.5, 0.618, 0.786):
            lvl = swing_hi - (swing_hi - swing_lo) * f if up_move else swing_lo + (swing_hi - swing_lo) * f
            fib_levels.append((f, lvl))

    # ---- build render-ready drawings -------------------------------------
    dr = a.drawings
    # EMAs
    dr.append({"type": "ema", "n": 20, "values": e20, "color": (80, 200, 255), "label": "EMA20"})
    dr.append({"type": "ema", "n": 50, "values": e50, "color": (200, 140, 80), "label": "EMA50"})
    # S/R horizontal levels (strongest, nearest)
    for p, s, kind in a.levels:
        dr.append({"type": "hline", "price": p, "strength": s, "kind": kind,
                   "label": f"{kind[:3].upper()} {p:,.1f}"})
    # trendlines through last swing highs / lows
    if len(ph) >= 2:
        fit = _fit_line(ph[-3:], [h[i] for i in ph[-3:]])
        if fit:
            m, b = fit
            dr.append({"type": "trendline", "kind": "resistance",
                       "p1": (ph[-3 if len(ph) >= 3 else -2], m * ph[-3 if len(ph) >= 3 else -2] + b),
                       "p2": (n - 1, m * (n - 1) + b), "color": (90, 90, 240)})
    if len(pl) >= 2:
        fit = _fit_line(pl[-3:], [l[i] for i in pl[-3:]])
        if fit:
            m, b = fit
            dr.append({"type": "trendline", "kind": "support",
                       "p1": (pl[-3 if len(pl) >= 3 else -2], m * pl[-3 if len(pl) >= 3 else -2] + b),
                       "p2": (n - 1, m * (n - 1) + b), "color": (90, 220, 90)})
    # fibonacci
    if fib_levels:
        dr.append({"type": "fib", "hi": swing_hi, "lo": swing_lo,
                   "hi_i": hi_i, "lo_i": lo_i, "levels": fib_levels})
    # swing markers
    for i in ph[-4:]:
        dr.append({"type": "marker", "idx": i, "price": float(h[i]), "kind": "high"})
    for i in pl[-4:]:
        dr.append({"type": "marker", "idx": i, "price": float(l[i]), "kind": "low"})

    # ---- signals + key level --------------------------------------------
    nearest = min(a.levels, key=lambda t: abs(t[0] - price)) if a.levels else None
    if nearest:
        a.key_level = nearest[0]
        if abs(price - nearest[0]) <= tol:
            a.signals.append(f"testing {nearest[2]} at {nearest[0]:,.1f}")
    if a.rsi >= 70:
        a.signals.append(f"RSI {a.rsi:.0f} overbought")
    elif a.rsi <= 30:
        a.signals.append(f"RSI {a.rsi:.0f} oversold")
    # breakout: close beyond the most recent resistance/support
    res_above = [p for p, s, k in a.levels if k == "resistance" and p > price]
    sup_below = [p for p, s, k in a.levels if k == "support" and p < price]
    if res_above and price > min(res_above) - tol * 0.2 and a.bias == "bullish":
        a.signals.append("pressing resistance - breakout watch")

    # ---- focus region for the camera (recent structure) ------------------
    f0 = max(0, min(lo_i, hi_i) - 6)
    a.focus = (f0, n - 1)

    a.narrative = _narrate_en(symbol, a, res_above, sup_below)
    a.narrative_ar = _narrate_ar(symbol, a, res_above, sup_below)
    return a


def _sym_name(symbol):
    return {"PAXGUSDT": "Gold", "XAUUSD": "Gold", "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum"}.get(symbol.upper(), symbol.upper())


def _narrate_en(symbol, a, res_above, sup_below):
    name = _sym_name(symbol)
    t = {"uptrend": "in an uptrend, printing higher highs and higher lows",
         "downtrend": "in a downtrend, making lower highs and lower lows",
         "range": "ranging, with no clear direction yet"}[a.trend]
    s = f"{name} is {t}. Price is {a.price:,.1f}, RSI at {a.rsi:.0f}. "
    if a.key_level is not None:
        s += f"The key level to watch is {a.key_level:,.1f}. "
    if res_above:
        s += f"Resistance sits above at {min(res_above):,.1f}; "
    if sup_below:
        s += f"support is below at {max(sup_below):,.1f}. "
    if a.signals:
        s += "Right now: " + ", ".join(a.signals) + "."
    return s.strip()


def _narrate_ar(symbol, a, res_above, sup_below):
    name = {"Gold": "الذهب", "Bitcoin": "البيتكوين",
            "Ethereum": "الإيثيريوم"}.get(_sym_name(symbol), _sym_name(symbol))
    t = {"uptrend": "في اتجاه صاعد ويسجل قمماً وقيعاناً أعلى",
         "downtrend": "في اتجاه هابط ويسجل قمماً وقيعاناً أدنى",
         "range": "في نطاق عرضي بدون اتجاه واضح"}[a.trend]
    s = f"{name} {t}. السعر الآن {a.price:,.1f} ومؤشر القوة النسبية عند {a.rsi:.0f}. "
    if res_above:
        s += f"المقاومة فوق عند {min(res_above):,.1f}، "
    if sup_below:
        s += f"والدعم أسفل عند {max(sup_below):,.1f}. "
    return s.strip()
