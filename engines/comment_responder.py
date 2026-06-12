# =============================================================================
# engines/comment_responder.py — decide which live comments to answer, and answer.
#
# Pipeline per comment:
#   1) RULE pre-filter   — drop obvious spam (emoji-only, too short, links, "first",
#                          follow-bait, duplicates) for free (no LLM cost).
#   2) LLM triage        — QUESTION / GREETING / SKIP (fast, history-free).
#   3) ANSWER            — QUESTION: web-research + brain answers as the host (human,
#                          1-2 sentences, addresses the viewer by name).
#                          GREETING: a brief warm welcome by name (engaging mode).
#
# Returns a spoken-ready string (or None to ignore). Never raises.
# =============================================================================
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import web_research
except Exception:
    web_research = None

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⌀-⏿]+")
_URL = re.compile(r"https?://|www\.|\.com|\.net|t\.me/", re.I)
_SPAM = re.compile(r"\b(first|follow me|follow back|f4f|sub4sub|check my|free follow|"
                   r"promo|gift me|send gift|join my)\b", re.I)
# trading words that make a short comment worth answering even without a '?'
_TOPIC = re.compile(r"\b(gold|xau|xauusd|buy|sell|long|short|price|target|support|"
                    r"resist|trend|entry|stop|tp|sl|forecast|bull|bear|trade|signal|"
                    r"chart|market|fed|dollar|dxy|news)\b", re.I)

ENGAGING = os.environ.get("AVATAR_COMMENTS_ENGAGING", "1") == "1"   # welcome viewers too
USE_WEB = os.environ.get("AVATAR_COMMENTS_WEB", "1") == "1"         # web-research answers


class CommentResponder:
    def __init__(self, brain, get_context=None):
        """brain = LLMBrain; get_context() -> short live-market context string."""
        self.brain = brain
        self.get_context = get_context
        self._recent = {}          # text -> last time seen (dedupe)
        self._last_greet = 0.0

    # -- 1) cheap rule filter -------------------------------------------------
    def _rule_skip(self, text):
        t = (text or "").strip()
        if len(t) < 2:
            return True
        stripped = _EMOJI.sub("", t).strip()
        if len(stripped) < 2:               # emoji-only
            return True
        if _URL.search(t) or _SPAM.search(t):
            return True
        # de-dupe identical comments within 60s (spam floods)
        now = time.time()
        self._recent = {k: v for k, v in self._recent.items() if now - v < 60}
        key = t.lower()
        if key in self._recent:
            return True
        self._recent[key] = now
        return False

    # -- 2) triage QUESTION / GREETING / SKIP --------------------------------
    def _triage(self, text):
        t = text.strip()
        # obvious question shortcuts (avoid an LLM call)
        if "?" in t or _TOPIC.search(t):
            return "QUESTION"
        if self.brain is None:
            return "GREETING" if ENGAGING else "SKIP"
        label = self.brain.quick(
            f"Viewer comment on a live gold-trading stream: \"{t}\"\n"
            "Reply with ONE word: QUESTION (a real question or request worth answering), "
            "GREETING (a hello / nice remark / their name), or SKIP (spam / noise / "
            "meaningless). One word only.",
            system="You classify live-stream comments. Reply with exactly one word.",
            max_tokens=4)
        label = (label or "").strip().upper()
        for k in ("QUESTION", "GREETING", "SKIP"):
            if k in label:
                return k
        return "SKIP"

    @staticmethod
    def _say(reply):
        """Strip wrapping quotes / stray markup so the TTS speaks clean text."""
        r = (reply or "").strip().strip('"').strip("'").strip("`").strip()
        return r or None

    # -- 3) compose the spoken answer ----------------------------------------
    def respond(self, user, text):
        """Return a spoken-ready reply for this comment, or None to ignore."""
        try:
            if self._rule_skip(text):
                return None
            kind = self._triage(text)
            if kind == "SKIP":
                return None
            if kind == "GREETING":
                if not ENGAGING:
                    return None
                now = time.time()
                if now - self._last_greet < 25:    # don't greet too often
                    return None
                self._last_greet = now
                reply = self.brain.quick(
                    f"A viewer named {user} said \"{text}\" on your live gold-trading "
                    "stream. Give a SHORT, warm one-line welcome/acknowledgement by name, "
                    "like a friendly human host. Max 14 words. No emojis.",
                    system="You are a warm, natural live-stream host.", max_tokens=40)
                return reply

            # QUESTION -> research + answer as the host
            ctx = ""
            if self.get_context:
                try:
                    ctx = self.get_context() or ""
                except Exception:
                    ctx = ""
            web = web_research.research(text) if (USE_WEB and web_research) else ""
            grounding = ""
            if ctx:
                grounding += f"\nLive market context: {ctx}"
            if web:
                grounding += f"\nWeb research:\n{web}"
            reply = self.brain.quick(
                f"You are the human host of a live gold/XAUUSD trading stream. A viewer "
                f"named {user} asked: \"{text}\".{grounding}\n\n"
                "Answer them directly as a real person on the mic — natural, confident, "
                "helpful, 1-2 short sentences, address them by name once. Use the context/"
                "research if relevant; if unsure, be honest and brief. No emojis, no lists.",
                system="You are a friendly, knowledgeable live trading-stream host. Speak "
                       "naturally as if talking out loud.", max_tokens=110, timeout=40)
            return reply
        except Exception as exc:
            print(f"[COMMENTS] respond error: {exc}")
            return None
