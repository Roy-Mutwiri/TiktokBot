# =============================================================================
# engines/web_research.py — quick, resilient web research for live answers.
#
# Given a viewer's question, fetch a few clean web snippets the brain can ground
# its answer in (so it isn't guessing at current prices / news). Hardened for a
# live stream: retries across DuckDuckGo backends, a news fallback for time-
# sensitive questions, and a short TTL cache so a flood of similar questions (and
# transient rate-limits) don't stall or spam the network. Never raises — returns
# "" if the web is unreachable.
# =============================================================================
import os
import re
import time
import threading

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = float(os.environ.get("AVATAR_WEB_CACHE_TTL", "300"))   # 5 min
_RETRIES = int(os.environ.get("AVATAR_WEB_RETRIES", "3"))
_MAXLEN = int(os.environ.get("AVATAR_WEB_MAXLEN", "1500"))
# DuckDuckGo backends to rotate through on failure (the library accepts a
# `backend` kwarg; unknown values are ignored gracefully via the TypeError guard).
_BACKENDS = [b.strip() for b in os.environ.get(
    "AVATAR_WEB_BACKENDS", "auto,html,lite").split(",") if b.strip()]
_NEWS_RE = re.compile(r"\b(news|today|now|latest|breaking|price|forecast|"
                      r"this week|right now|current)\b", re.I)


def _clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _cache_get(q):
    with _CACHE_LOCK:
        v = _CACHE.get(q)
        if v and (time.time() - v[0]) < _CACHE_TTL:
            return v[1]
    return None


def _cache_put(q, val):
    with _CACHE_LOCK:
        _CACHE[q] = (time.time(), val)
        if len(_CACHE) > 256:                       # bound the cache
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _CACHE.pop(oldest, None)


def _rows(ddgs, q, max_results, backend, news):
    """Run one query (text or news) with an optional backend, tolerating older/
    newer ddgs signatures."""
    if news:
        try:
            return list(ddgs.news(q, max_results=max_results))
        except TypeError:
            return list(ddgs.news(q))
    for kwargs in ({"max_results": max_results, "backend": backend},
                   {"max_results": max_results}, {}):
        try:
            return list(ddgs.text(q, **kwargs))
        except TypeError:
            continue
    return []


def _format(rows):
    out = []
    for r in rows or []:
        title = _clean(r.get("title"))
        body = _clean(r.get("body") or r.get("excerpt") or r.get("snippet") or "")
        if body:
            out.append(f"- {title}: {body}" if title else f"- {body}")
    return out


def research(query, max_results=4, timeout=6):
    """Return a short block of clean web snippets for `query` (or "" on failure).
    Retries across backends, falls back to news for time-sensitive questions, and
    caches results for a few minutes. Best-effort and silent on failure."""
    q = (query or "").strip()
    if not q:
        return ""
    if len(q) > 300:
        q = q[:300]
    cached = _cache_get(q)
    if cached is not None:
        return cached
    try:
        from ddgs import DDGS
    except Exception:
        return ""

    rows = []
    for attempt in range(max(1, _RETRIES)):
        backend = _BACKENDS[attempt % len(_BACKENDS)] if _BACKENDS else "auto"
        try:
            with DDGS(timeout=timeout) as ddgs:
                rows = _rows(ddgs, q, max_results, backend, news=False)
            if rows:
                break
        except Exception as exc:
            print(f"[RESEARCH] try {attempt + 1}/{_RETRIES} ({backend}) "
                  f"failed: {str(exc)[:60]}")
            time.sleep(0.5 * (attempt + 1))         # gentle backoff

    # time-sensitive question and still thin -> try the news index
    if not rows and _NEWS_RE.search(q):
        try:
            with DDGS(timeout=timeout) as ddgs:
                rows = _rows(ddgs, q, max_results, _BACKENDS[0] if _BACKENDS else "auto",
                             news=True)
        except Exception as exc:
            print(f"[RESEARCH] news fallback failed: {str(exc)[:60]}")

    text = "\n".join(_format(rows))[:_MAXLEN]
    _cache_put(q, text)                              # cache even "" (rate-limit guard)
    return text


if __name__ == "__main__":
    for query in ["gold price today XAUUSD", "why is gold rising", "asdfqwerzxcv nonsense"]:
        t = time.time()
        out = research(query)
        print(f"\nQ: {query}  ({time.time() - t:.1f}s, {len(out)} chars)")
        print(out[:300] or "(empty)")
