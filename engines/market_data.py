# =============================================================================
# engines/market_data.py  —  live market data feed for the AI trader chart
# -----------------------------------------------------------------------------
# Pulls real OHLCV candles from Binance's public REST API (no key needed), keeps
# a rolling buffer, and refreshes the latest candle on a background poller. Auto-
# fails over across binance hosts; if the network is down it synthesizes a
# plausible random-walk so the chart never goes blank.
#
#   md = MarketData("PAXGUSDT", "5m")   # PAXG ~ live gold price
#   md.start()                          # background updates
#   ohlc = md.snapshot()                # np.array [N,6] = ts,o,h,l,c,v  (ms, price)
#
# Symbols (no key): PAXGUSDT (gold), BTCUSDT, ETHUSDT, ... any Binance spot pair.
# =============================================================================

import time
import threading

import numpy as np

try:
    import requests
except Exception:
    requests = None

# Public Binance REST hosts, tried in order (geo / outage failover).
_HOSTS = ["api.binance.com", "data-api.binance.vision", "api.binance.us"]
# interval -> seconds, for the synthetic fallback + poll cadence.
_INTERVAL_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                 "1h": 3600, "4h": 14400, "1d": 86400}


class MarketData:
    """Rolling live OHLCV buffer for one symbol/interval."""

    def __init__(self, symbol="PAXGUSDT", interval="5m", limit=320):
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = int(limit)
        self.host = None
        self.live = False                 # True once real data has loaded
        self._ohlc = None                 # np.array [N,6]
        self._tick = {}                   # latest real-time 24h ticker (last price + stats)
        self._tick_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._load_initial()

    # ---- real-time ticker (instant last-traded price + true 24h stats) ------
    def live_ticker(self, max_age=2.0):
        """Binance 24h ticker: the EXACT last-traded price + real 24h change/high/
        low/volume, refreshed at most every `max_age` seconds. This is the most
        real-time price (vs the forming candle's close)."""
        now = time.time()
        if self._tick and now - self._tick_t < max_age:
            return self._tick
        if requests is None:
            return self._tick
        hosts = ([self.host] if self.host else []) + [h for h in _HOSTS if h != self.host]
        for h in hosts:
            try:
                r = requests.get(f"https://{h}/api/v3/ticker/24hr",
                                 params={"symbol": self.symbol}, timeout=6)
                if r.status_code == 200:
                    d = r.json()
                    self._tick = {
                        "price": float(d["lastPrice"]),
                        "change_pct": float(d["priceChangePercent"]),
                        "high": float(d["highPrice"]),
                        "low": float(d["lowPrice"]),
                        "open": float(d["openPrice"]),
                        "volume": float(d["volume"]),
                    }
                    self._tick_t = now
                    self.host = h
                    self.live = True
                    return self._tick
            except Exception:
                continue
        return self._tick

    # -------------------------------------------------------------------------
    def _get(self, params):
        if requests is None:
            return None
        hosts = ([self.host] if self.host else []) + [h for h in _HOSTS if h != self.host]
        for h in hosts:
            try:
                r = requests.get(f"https://{h}/api/v3/klines", params=params, timeout=8)
                if r.ok:
                    self.host = h
                    return r.json()
            except Exception:
                continue
        return None

    @staticmethod
    def _to_arr(raw):
        return np.array([[float(x[0]), float(x[1]), float(x[2]),
                          float(x[3]), float(x[4]), float(x[5])] for x in raw],
                        dtype=np.float64)

    def _load_initial(self):
        raw = self._get({"symbol": self.symbol, "interval": self.interval,
                         "limit": self.limit})
        if raw:
            with self._lock:
                self._ohlc = self._to_arr(raw)
            self.live = True
        else:
            with self._lock:
                self._ohlc = self._synth(self.limit)
            self.live = False

    # ---- synthetic fallback (offline) ---------------------------------------
    def _synth(self, n):
        sec = _INTERVAL_SEC.get(self.interval, 300)
        now = time.time()
        base = {"PAXGUSDT": 4150.0, "BTCUSDT": 63000.0, "ETHUSDT": 1670.0}.get(
            self.symbol, 100.0)
        rng = np.random.default_rng(abs(hash(self.symbol)) % (2**32))
        price = base
        rows = []
        for i in range(n):
            o = price
            drift = rng.normal(0, base * 0.0018)
            c = max(0.01, o + drift)
            hi = max(o, c) + abs(rng.normal(0, base * 0.0009))
            lo = min(o, c) - abs(rng.normal(0, base * 0.0009))
            v = abs(rng.normal(100, 40))
            ts = (now - (n - i) * sec) * 1000.0
            rows.append([ts, o, hi, lo, c, v])
            price = c
        return np.array(rows, dtype=np.float64)

    # ---- live update --------------------------------------------------------
    def update(self):
        """Fetch the last few candles and merge (refreshes the forming bar)."""
        raw = self._get({"symbol": self.symbol, "interval": self.interval,
                         "limit": 3})
        if not raw:
            if not self.live:
                self._synth_step()       # keep the offline chart moving
            return False
        new = self._to_arr(raw)
        with self._lock:
            if self._ohlc is None:
                self._ohlc = new
            else:
                base = self._ohlc
                base_ts = set(base[:, 0].tolist())
                # replace the forming/last candle, append genuinely-new ones
                keep = base[base[:, 0] < new[0, 0]]
                merged = np.vstack([keep, new])
                if len(merged) > self.limit:
                    merged = merged[-self.limit:]
                self._ohlc = merged
            self.live = True
        return True

    def _synth_step(self):
        with self._lock:
            if self._ohlc is None:
                self._ohlc = self._synth(self.limit)
                return
            last = self._ohlc[-1]
            base = float(last[4])
            drift = np.random.normal(0, base * 0.0016)
            c = max(0.01, base + drift)
            o = float(last[4])
            row = [last[0] + _INTERVAL_SEC.get(self.interval, 300) * 1000.0,
                   o, max(o, c) + abs(np.random.normal(0, base * 0.0008)),
                   min(o, c) - abs(np.random.normal(0, base * 0.0008)), c,
                   abs(np.random.normal(100, 40))]
            self._ohlc = np.vstack([self._ohlc[-self.limit + 1:], row])

    # ---- background poller ---------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="market-data",
                                        daemon=True)
        self._thread.start()

    def _loop(self):
        # poll a few times per candle so the forming bar looks live
        period = max(2.0, _INTERVAL_SEC.get(self.interval, 300) / 12.0)
        while not self._stop.is_set():
            try:
                self.update()
            except Exception:
                pass
            self._stop.wait(period)

    def stop(self):
        self._stop.set()

    # ---- access -------------------------------------------------------------
    def snapshot(self):
        with self._lock:
            return None if self._ohlc is None else self._ohlc.copy()

    @property
    def price(self):
        # prefer the REAL-TIME last-traded price; fall back to the forming candle
        if self._tick and time.time() - self._tick_t < 6.0:
            return self._tick["price"]
        with self._lock:
            return float(self._ohlc[-1, 4]) if self._ohlc is not None and len(self._ohlc) else 0.0

    def startup_check(self):
        s = self.snapshot()
        n = 0 if s is None else len(s)
        src = "LIVE" if self.live else "offline/synthetic"
        return (s is not None and n > 0,
                f"market data {self.symbol} {self.interval}: {n} candles ({src})")


if __name__ == "__main__":
    md = MarketData("PAXGUSDT", "5m", limit=50)
    print(md.startup_check())
    s = md.snapshot()
    print("last 3 closes:", s[-3:, 4].tolist(), "price:", md.price)
