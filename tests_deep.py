# =============================================================================
# tests_deep.py — deep tests for the two reported bugs:
#   (A) background music repeats / never rotates
#   (B) the AI host speaks NON-real-time market info (fake simulated price)
# Run:  python tests_deep.py
# =============================================================================
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "engines")          # engines/ wins over any root duplicate

import numpy as np

_p = _f = 0
def chk(name, cond, extra=""):
    global _p, _f
    ok = bool(cond)
    print(f"  [{'✓' if ok else '✗'}] {name}" + (f"  — {extra}" if extra else ""))
    if ok: _p += 1
    else:  _f += 1
    return ok


# =========================================================================
print("\n== A) MUSIC — variety, rotation, randomized start, seamless ==")
import bg_music
from bg_music import _make_track, BackgroundMusic, SR, SONG_SECONDS

# A1: 50 tracks all byte-unique
tracks = [_make_track(s) for s in range(1, 51)]
hashes = {hash(t[0][:6000].tobytes()) for t in tracks}
chk("50 tracks all unique", len(hashes) == 50, f"{len(hashes)}/50 unique")

# A2: genuine VARIETY — multiple styles, ≥5 distinct BPMs, RMS spread
styles = {t[1]["style"] for t in tracks}
bpms = {t[1]["bpm"] for t in tracks}
rms = [float(np.sqrt((t[0] ** 2).mean())) for t in tracks]
chk("≥3 distinct phonk styles", len(styles) >= 3, f"styles={sorted(styles)}")
chk("≥5 distinct tempos", len(bpms) >= 5, f"{len(bpms)} BPMs")
chk("tracks differ in energy (RMS spread)", (max(rms) - min(rms)) > 0.02,
    f"rms {min(rms):.3f}..{max(rms):.3f}")

# A3: tracks are LONGER than the old 4s loop (less repetitive)
dur = len(tracks[0][0]) / SR
chk("loop length > 5s (was 4s)", dur > 5.0, f"{dur:.1f}s")

# A4: pairwise spectral difference — adjacent tracks actually sound different
def spec(x):
    f = np.abs(np.fft.rfft(x[:SR]))
    return f / (np.linalg.norm(f) + 1e-9)
sims = []
for i in range(0, 10):
    a, b = spec(tracks[i][0]), spec(tracks[i + 1][0])
    sims.append(float(np.dot(a, b)))
chk("adjacent tracks spectrally distinct (cos<0.97)", max(sims) < 0.97,
    f"max cos-sim {max(sims):.3f}")

# A5: randomized START — two instances shuffle to different orders
m1 = BackgroundMusic(); m2 = BackgroundMusic()
chk("playlist order randomized per session", m1._order != m2._order,
    f"first track {m1._order[0]} vs {m2._order[0]}")

# A6: ROTATION — driving the callback past SONG_SECONDS advances the track
m = BackgroundMusic(); m.active = True
time.sleep(1.0)                              # let _gen_loop prerender _next
m._song_until = time.monotonic() - 0.01      # due now
first_oi = m._oi
out = np.zeros((1024, 1), dtype=np.float32)
for _ in range(700):                         # ~16s of frames -> must cross a seam
    m._callback(out, 1024, None, None)
    if m._oi != first_oi:
        break
chk("playlist ADVANCES to a new track", m._oi != first_oi, f"_oi {first_oi}->{m._oi}")

# A7: over a longer run it keeps cycling (≥3 changes). Sleep between advances so
# the background generator can pre-render the next track (as it does live).
m3 = BackgroundMusic(); m3.active = True
changes = 0; last = m3._oi
for _ in range(8):
    time.sleep(0.35)                          # let _gen_loop prerender _next
    m3._song_until = time.monotonic() - 0.01  # song due now
    for _ in range(800):                      # drive callbacks across a seam
        m3._callback(out, 1024, None, None)
        if m3._oi != last:
            changes += 1; last = m3._oi; break
chk("keeps cycling (≥3 track changes)", changes >= 3, f"{changes} changes")
m3.stop()

# A8: seamless loop (start≈end so no click at the wrap)
seam = abs(float(tracks[0][0][0]) - float(tracks[0][0][-1]))
chk("loop seam click-free", seam < 0.05, f"|start-end|={seam:.3f}")

# A9: DUCK — music is much quieter while the bot speaks
chk("duck multiplier < 0.3 (voice dominates)", bg_music.DUCK < 0.3, f"DUCK={bg_music.DUCK}")
m.stop(); m1.stop(); m2.stop()


# =========================================================================
print("\n== B) REAL-TIME MARKET — live price, not the simulated fake ==")
from market_data import MarketData
md = MarketData("PAXGUSDT", "1m")
snap = md.snapshot()
chk("market data has candles", snap is not None and len(snap) > 0,
    f"{0 if snap is None else len(snap)} candles, live={md.live}")
price = md.price
chk("live gold price is realistic ($1k-$10k)", 1000 < price < 10000, f"${price:,.0f}")
# the simulated TradingView starts at a FAKE 2375 — confirm the live feed differs
from trading_view import TradingView
fake = TradingView("XAUUSD").price
chk("live price != simulated fake", abs(price - fake) > 50,
    f"live ${price:,.0f} vs sim ${fake:,.0f}")
if md.live:
    md.update()
    chk("update() refreshes without error", md.price > 0, f"${md.price:,.0f}")
else:
    chk("offline fallback still yields a price", price > 0, "(network down -> synthetic)")


# =========================================================================
print("\n== B2) LIVE CONTEXT BUILDER — host quotes the CURRENT price ==")
class _FakeChart:
    price = 2375.0; day_open = 2375.0
class _Studio:
    pass
# emulate _live_market_ctx with a live feed present
class _MD:
    price = 4217.0
    def snapshot(self):
        base = np.linspace(4180, 4217, 30)
        return np.column_stack([np.arange(30), base, base, base, base, base])
import avatar_studio
s = _Studio()
s.market = _MD()
s.engines = {"chart": _FakeChart()}
ctx = avatar_studio.AvatarStudio._live_market_ctx(s)
chk("context contains the LIVE price (4,217)", "4,217" in ctx, ctx[:70])
chk("context says talk about the current price", "current price" in ctx.lower(), "")
chk("live feed overrides the fake chart price", "2,375" not in ctx, "")
chk("chart price synced to live", abs(s.engines["chart"].price - 4217.0) < 1, "")
# feed-down fallback uses the chart (whose price was synced to live earlier)
s.market = None
ctx2 = avatar_studio.AvatarStudio._live_market_ctx(s)
chk("fallback to chart when feed down still yields a price",
    ctx2 != "" and "gold" in ctx2.lower(), ctx2[:50])


# =========================================================================
print("\n" + "#" * 56)
print(f"RESULT: {_p} passed, {_f} failed  (of {_p + _f})")
print("#" * 56)
sys.exit(1 if _f else 0)
