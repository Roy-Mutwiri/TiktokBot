# =============================================================================
# engines/web_research.py — quick web research for answering live questions.
#
# Given a viewer's question, fetch a few clean web snippets (DuckDuckGo) the brain
# can ground its answer in (so it isn't guessing at current prices/news). Fast,
# best-effort, never raises — returns "" if the web is unreachable.
# =============================================================================
import os
import re


def research(query, max_results=4, timeout=6):
    """Return a short block of clean web snippets for `query` (or "" on failure)."""
    try:
        from ddgs import DDGS
    except Exception:
        return ""
    try:
        q = (query or "").strip()
        if not q:
            return ""
        out = []
        with DDGS(timeout=timeout) as ddgs:
            for r in ddgs.text(q, max_results=max_results):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                body = re.sub(r"\s+", " ", body)
                if body:
                    out.append(f"- {title}: {body}")
        return "\n".join(out)[:1500]
    except Exception as exc:
        print(f"[RESEARCH] web search failed ({exc}).")
        return ""
