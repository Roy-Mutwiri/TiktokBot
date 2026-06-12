# =============================================================================
# engines/market_ta.py — real technical analysis from live OHLCV candles.
#
# Turns the MarketData snapshot ([N,6] = ts,o,h,l,c,v) into ACTUAL levels the host
# can talk about — support/resistance, trend (EMA20/50), RSI, session range — so the
# commentary is real analysis, not generic hype. Never raises; returns "" if no data.
# =============================================================================
import numpy as np


def _ema(c, period):
    if len(c) == 0:
        return None
    if len(c) < period:
        return float(c[-1])
    k = 2.0 / (period + 1.0)
    e = float(c[0])
    for x in c:
        e = float(x) * k + e * (1.0 - k)
    return e


def _rsi(c, period=14):
    if len(c) < period + 1:
        return None
    d = np.diff(c[-(period + 1):])
    gain = d[d > 0].sum()
    loss = -d[d < 0].sum()
    if loss == 0:
        return 100.0
    rs = (gain / period) / (loss / period)
    return 100.0 - 100.0 / (1.0 + rs)


def analyze(ohlc):
    """Return a dict of real TA, or None if not enough data."""
    if ohlc is None or len(ohlc) < 20:
        return None
    try:
        h = ohlc[:, 2].astype(float)
        l = ohlc[:, 3].astype(float)
        c = ohlc[:, 4].astype(float)
        price = float(c[-1])
        n = len(c)
        win = min(n, 60)
        sess_hi = float(h[-win:].max())
        sess_lo = float(l[-win:].min())
        res = float(h[-min(n, 30):].max())       # nearest swing resistance
        sup = float(l[-min(n, 30):].min())       # nearest swing support
        ema20 = _ema(c[-min(n, 60):], 20)
        ema50 = _ema(c[-min(n, 120):], 50)
        rsi = _rsi(c)
        if price > ema20 and (ema50 is None or ema20 >= ema50):
            trend = "uptrend"
        elif price < ema20 and (ema50 is None or ema20 <= ema50):
            trend = "downtrend"
        else:
            trend = "ranging"
        ref = float(c[0])
        pct = (price - ref) / ref * 100.0 if ref else 0.0
        return dict(price=price, high=sess_hi, low=sess_lo, support=sup,
                    resistance=res, ema20=ema20, ema50=ema50, rsi=rsi,
                    trend=trend, pct=pct)
    except Exception:
        return None


def ta_summary(ohlc):
    """One-line REAL TA string for the brain to ground its commentary in."""
    a = analyze(ohlc)
    if not a:
        return ""
    parts = [f"trend is {a['trend']}"]
    if a["rsi"] is not None:
        cond = ("overbought" if a["rsi"] > 70 else
                "oversold" if a["rsi"] < 30 else "neutral")
        parts.append(f"RSI {a['rsi']:.0f} ({cond})")
    parts.append(f"nearest support ${a['support']:,.0f}")
    parts.append(f"nearest resistance ${a['resistance']:,.0f}")
    parts.append(f"session range ${a['low']:,.0f}-${a['high']:,.0f}")
    return " REAL TECHNICALS: " + ", ".join(parts) + ". Use these EXACT levels."
