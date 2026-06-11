# AI Trader — an AI that operates a live chart and analyses the market

The AI drives a real, live chart on its own: it **scrolls, zooms, and draws
tools** (trendlines, support/resistance, fibonacci, swing markers) with an
animated pen cursor, **analyses the market in real time**, and **speaks** its
read (English or Arabic via the multilingual voice).

> Why our own chart instead of automating the TradingView website? The public
> TradingView site is a canvas with no price↔pixel API — driving it means a
> headless browser + OCR calibration + fragile pixel clicks, and it's against
> their ToS. Owning the renderer gives pixel-perfect drawing, full camera
> control, real data, and zero brittleness — and it drops straight into the
> avatar's chart scene.

## Run it standalone

```
python ai_trader.py                  # live gold (PAXG), preview window + voice
python ai_trader.py --symbol BTCUSDT # bitcoin
python ai_trader.py --lang ar        # narrate in Arabic
python ai_trader.py --cam            # publish to the virtual camera (OBS / stream)
python ai_trader.py --no-speak       # chart only
python ai_trader.py --size 1280x720
```

`--cam` sends the AI analyst chart to the virtual camera, so you can drop it
straight onto a stream (and the avatar can be a PiP over it).

## Use it as the avatar's chart scene

Set `AVATAR_AICHART` and the avatar's "chart" scene becomes the AI analyst, and
the **avatar speaks the analysis** while it's on screen:

```
set AVATAR_AICHART=1            # gold (PAXGUSDT); or set a symbol e.g. BTCUSDT
set AVATAR_AICHART_TF=15m       # timeframe (1m/5m/15m/1h/4h/1d)
set AVATAR_AICHART_NAME=XAU/USD (gold)
set AVATAR_TTS_LANG=ar          # narrate the analysis in Arabic (default en)
python realtime_avatar.py
```

(Default — env unset — keeps the old synthetic chart, so nothing changes unless
you opt in.)

## How it works

| module | role |
|--------|------|
| `engines/market_data.py` | live OHLCV from Binance public REST (no key); rolling buffer, host failover, offline synthetic fallback. PAXGUSDT ≈ live gold. |
| `engines/market_analysis.py` | the brain: EMA/RSI/ATR, swing pivots → support/resistance, fitted trendlines, fibonacci, trend (HH-HL vs LH-LL), signals + EN/AR narration |
| `engines/ai_chart.py` | renders real candles with a camera (eased scroll/zoom) and draws the analysis as tools, with an animated pen cursor |
| `engines/chart_pilot.py` | the operator: re-analyses the live market on a loop and sequences camera moves + tool draws + narration |
| `ai_trader.py` | standalone runner (window / virtual camera / voice) |

Coordinates in the analysis are `(bar_index, price)`, so every drawing maps to
pixels correctly at any zoom or scroll position.

## Data / symbols

Any Binance spot pair, no API key: `PAXGUSDT` (gold), `BTCUSDT`, `ETHUSDT`, …
Hosts auto-fail-over across `api.binance.com` / `.vision` / `.us`; if the
network is down it falls back to a synthetic walk so the chart never blanks.

## Notes

- Analysis is technical/structural (trend, levels, momentum) — it is **not
  financial advice** and places no trades; it visualises and narrates the read.
- RTF/latency: rendering is real-time; analysis runs once per ~20s cycle.
