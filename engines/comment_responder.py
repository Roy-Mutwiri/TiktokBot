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

try:
    import reactions as _reactions          # instant offline appreciation templates
except Exception:
    _reactions = None

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⌀-⏿]+")
_URL = re.compile(r"https?://|www\.|\.com|\.net|t\.me/", re.I)
_SPAM = re.compile(r"\b(first|follow me|follow back|f4f|sub4sub|check my|free follow|"
                   r"promo|gift me|send gift|join my)\b", re.I)
# abusive / scam — NEVER read these out on stream (silently ignore)
_TOXIC = re.compile(r"\b(scam|scammer|fake|kys|kill yourself|idiot|moron|stupid|loser|"
                    r"bitch|whore|fuck ?you|fuck off|shut up|trash|garbage|telegram|"
                    r"whatsapp|dm me|inbox me|double your|guaranteed profit|giveaway|"
                    r"investment plan|signal group|join my channel)\b", re.I)
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
        self._last_deep = 0.0      # last time we dug deep into a topic (spacing)
        self._supporters = {}      # user -> {coins, gifts, follow} cumulative support
        import collections as _c
        self._recent_openers = _c.deque(maxlen=10)   # avoid repeating thank-you openers

    # warm Arab-host voice for the appreciation lines (matches the avatar persona)
    _ARAB_HOST_SYS = (
        "You are a warm, charismatic ARAB live-stream host who is genuinely grateful "
        "to your supporters. Drop a natural Arabic word like habibi, wallahi, mashallah, "
        "akhi or ya salam. Speak out loud like a real person. No emojis, no markdown.")

    # rotate the FLAVOUR of every thank-you so it's NEVER the same twice
    _THANK_STYLES = [
        "shout them out loud and proud like they just became a VIP of the stream",
        "be heartfelt and genuinely touched, straight from the heart",
        "be hyped and explosive with energy about it",
        "be warm and playful, share a little joke with them",
        "promise them the next gold signal as your personal thank-you",
        "call them a legend and hype the whole room about them",
        "thank them humbly and sincerely, almost surprised by their kindness",
        "welcome them into the family like they're an old friend",
        "react with delight and tell the chat to show this person some love",
        "keep it short, punchy and full of gratitude",
    ]

    def _vary(self):
        """Random thank-you style + an instruction to not reuse recent openers."""
        import random
        style = random.choice(self._THANK_STYLES)
        clause = f"Style: {style}. "
        if self._recent_openers:
            clause += ("Start in a COMPLETELY different way than these recent openers, "
                       "do NOT begin the same: " + " / ".join(self._recent_openers) + ". ")
        return clause

    def _remember_opener(self, reply):
        if reply:
            self._recent_openers.append(" ".join(reply.split()[:3]).lower())

    def _support(self, user):
        return self._supporters.setdefault(user, {"coins": 0, "gifts": 0, "follow": False})

    # viewers often ANNOUNCE a follow/share/gift in chat (TikTok's structured
    # follow/share events are flaky) — catch those so we ALWAYS appreciate them.
    _FOLLOWED = re.compile(
        r"\b(just followed|i followed|followed you|followed the|now following|new follower|"
        r"i'?m following|gave you a follow|dropped a follow|hit follow|smashed follow|"
        r"followed bro|done follow|followed ya)\b", re.I)
    _SHARED = re.compile(
        r"\b(i shared|just shared|shared your|shared the (live|stream|video)|sharing your|"
        r"shared bro|shared it|i'?ll share|sharing the)\b", re.I)
    _GIFTED = re.compile(
        r"\b(sent you a|sent a gift|gifted you|here'?s a gift|gift for you|sent roses?|"
        r"take my (rose|gift|coins))\b", re.I)

    def _appreciation_from_comment(self, user, text):
        """If a chat comment announces a follow/share/gift, return a warm by-name
        thank-you (so appreciation fires even when TikTok's events don't)."""
        t = text or ""
        if self._FOLLOWED.search(t):
            s = self._support(user)
            if s["follow"]:                  # already thanked this session
                return None
            return self.react_follow(user)
        if self._SHARED.search(t):
            return self.react_share(user)
        if self._GIFTED.search(t):
            r = self.brain.quick(
                f"A viewer named {user} mentioned in chat that they sent you a gift on your "
                "live gold stream. Thank them BY NAME warmly and genuinely for the support. "
                "One short spoken sentence. No emojis.",
                system=self._ARAB_HOST_SYS, max_tokens=40) if self.brain else None
            return self._say(r) or f"Wallahi thank you for the gift {user}, I appreciate you habibi!"
        return None

    # topics worth digging into (keeps the live lively when one comes up)
    _INTERESTING = re.compile(
        r"\b(why|how|what do you think|think about|your take|explain|opinion|strateg|"
        r"long.?term|fed|inflation|recession|interest rate|rate cut|dollar|dxy|war|"
        r"crash|bubble|safe.?haven|invest|portfolio|crypto|bitcoin|btc|silver|stocks?|"
        r"economy|gold standard|central bank|hedge|halal|riba)\b", re.I)
    DEEP_COOLDOWN = float(os.environ.get("AVATAR_DEEP_COOLDOWN", "70"))

    def _is_interesting(self, text):
        """A substantive/topical comment worth a deeper, lively take."""
        t = (text or "").strip()
        if len(t) < 18:
            return False
        return bool(self._INTERESTING.search(t)) or len(t) > 60

    # -- 1) cheap rule filter -------------------------------------------------
    def _rule_skip(self, text):
        t = (text or "").strip()
        if len(t) < 2:
            return True
        stripped = _EMOJI.sub("", t).strip()
        if len(stripped) < 2:               # emoji-only
            return True
        if _URL.search(t) or _SPAM.search(t) or _TOXIC.search(t):
            return True        # spam / scam / abuse — never spoken on stream
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

    @staticmethod
    def _lang(text):
        """Detect the viewer's language so we can answer in it (Arabic vs English)."""
        for ch in (text or ""):
            if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ":
                return "Arabic"
        return "English"

    # -- live-event reactions (gifts / follows / likes / shares) -------------
    def react_gift(self, user, gift, count, coins):
        c = int(coins or 0)
        s = self._support(user)
        repeat = s["gifts"] > 0
        s["coins"] += c
        s["gifts"] += int(count or 1)
        if _reactions is not None:            # INSTANT offline template (no LLM)
            return _reactions.gift(user, gift, count, c)
        amt = f"{count} {gift}s" if count and count > 1 else f"a {gift}"
        # scale the appreciation to how big the gift is
        if c >= 5000:
            tier = ("a HUGE top-tier gift — react like they just made your whole night, "
                    "give them a massive VIP shoutout and call them a legend of the stream")
        elif c >= 500:
            tier = "a big, generous gift — be genuinely moved and hype them up hard"
        elif c >= 50:
            tier = "a generous gift — be warm and clearly grateful"
        else:
            tier = "a kind gift — give a warm, genuine thank-you"
        extra = ""
        if repeat:
            extra = (f" They've supported you before ({s['coins']} coins total now) — "
                     "recognise them as a loyal regular / one of your top supporters.")
        r = self.brain.quick(
            f"A viewer named {user} just sent you {amt} ({c} coins) on your live gold stream. "
            f"That is {tier}. Thank them BY NAME, genuinely and with real energy.{extra} "
            f"{self._vary()}1-2 short spoken sentences. No emojis.",
            system=self._ARAB_HOST_SYS, max_tokens=70) if self.brain else None
        r = self._say(r)
        self._remember_opener(r)
        return r or self._fallback_thanks(user, "gift", gift)

    def react_follow(self, user):
        s = self._support(user)
        s["follow"] = True
        if _reactions is not None:            # INSTANT offline template (no LLM)
            return _reactions.follow(user)
        r = self.brain.quick(
            f"{user} just FOLLOWED your live gold stream. Thank them BY NAME for the follow "
            "with a warm, genuine welcome-to-the-family — like you really appreciate them "
            f"joining. {self._vary()}One short spoken sentence. No emojis.",
            system=self._ARAB_HOST_SYS, max_tokens=44) if self.brain else None
        r = self._say(r)
        self._remember_opener(r)
        return r or self._fallback_thanks(user, "follow")

    def react_share(self, user):
        if _reactions is not None:            # INSTANT offline template (no LLM)
            return _reactions.share(user)
        r = self.brain.quick(
            f"{user} just SHARED your live gold stream with others. Thank them BY NAME warmly "
            f"for spreading the stream — it genuinely helps. {self._vary()}One short spoken "
            "sentence. No emojis.",
            system=self._ARAB_HOST_SYS, max_tokens=42) if self.brain else None
        r = self._say(r)
        self._remember_opener(r)
        return r or self._fallback_thanks(user, "share")

    # varied canned fallbacks (so even with no brain it's never the same line)
    _FB = {
        "follow": ["Welcome to the family {u}, wallahi thank you for the follow habibi!",
                   "Ya salam {u} just followed — mashallah welcome aboard, akhi!",
                   "{u} I see you followed, habibi thank you, you're one of us now!",
                   "Big love {u}, thanks for the follow — welcome to the squad!"],
        "share": ["Ya salam, thank you for sharing the stream {u}, that means a lot habibi!",
                  "Mashallah {u} shared the live — wallahi I appreciate you akhi!",
                  "{u} thank you for spreading the word habibi, you're a real one!"],
        "gift": ["Wallahi thank you so much {u} for the {g}, I appreciate you habibi!",
                 "Ya salam {u}, mashallah thank you for the {g}, you're too kind akhi!",
                 "{u} habibi the {g} — wallahi I'm grateful, thank you so much!"],
    }

    def _fallback_thanks(self, user, kind, gift="gift"):
        import random
        opts = self._FB.get(kind, self._FB["follow"])
        return random.choice(opts).format(u=user, g=gift)

    def react_goal(self, coins):
        if _reactions is not None:            # INSTANT offline template (no LLM)
            return _reactions.goal(coins)
        r = self.brain.quick(
            f"Your live gold-trading stream just HIT its {coins}-coin gift goal. Give an "
            "EXCITED one-line thank-you to everyone and tease the next gold signal/analysis "
            "they unlocked. Max 18 words. No emojis.",
            system="You are a hyped, grateful live-stream host.", max_tokens=46) if self.brain else None
        return self._say(r) or (f"We just smashed the {coins}-coin goal — thank you legends, "
                                "next gold signal coming right up!")

    def react_likes(self, total):
        if _reactions is not None:            # INSTANT offline template (no LLM)
            return _reactions.likes(total)
        r = self.brain.quick(
            f"Your live gold-trading stream just passed {total} likes. Give a short, hyped "
            "one-line thank-you to everyone for the likes. Max 15 words. No emojis.",
            system="You are an enthusiastic live-stream host.", max_tokens=38) if self.brain else None
        return self._say(r) or f"We just passed {total} likes — thank you all, keep them coming!"

    # -- 3) compose the spoken answer ----------------------------------------
    def respond(self, user, text, on_answering=None):
        """Return a spoken-ready reply for this comment, or None to ignore.

        on_answering(user, text, mode) — optional callback fired the moment we
        COMMIT to answering this comment (after spam-filter + triage pass, before
        the slow web-research/LLM step). mode is 'greeting' or 'answering'. Lets the
        UI show 'now answering this comment' for genuine answers only (not spam)."""
        def _commit(mode):
            if on_answering:
                try:
                    on_answering(user, text, mode)
                except Exception:
                    pass
        try:
            if self._rule_skip(text):
                return None
            # ALWAYS appreciate a follow/share/gift announced in chat, first thing.
            appr = self._appreciation_from_comment(user, text)
            if appr:
                _commit("greeting")
                return appr
            lang = self._lang(text)               # answer in the viewer's language
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
                _commit("greeting")
                reply = self.brain.quick(
                    f"A viewer named {user} said \"{text}\" on your live gold-trading "
                    "stream. Give a SHORT, warm one-line welcome/acknowledgement by name, "
                    f"like a friendly human host. Max 14 words. No emojis. Reply in {lang}.",
                    system="You are a warm, natural live-stream host.", max_tokens=60)
                return self._say(reply)

            # QUESTION -> research + answer as the host
            _commit("answering")          # commit BEFORE the slow research/LLM step
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
            # SHORT, punchy answer — a live host fires back fast, one breath. Long
            # rambling replies pile up in the voice queue and fall behind the chat
            # (and small models wander off-topic when given room), so keep it tight.
            reply = self.brain.quick(
                f"You are the human host of a live gold/XAUUSD trading stream. A viewer "
                f"named {user} said: \"{text}\".{grounding}\n\n"
                "Fire back in ONE short, punchy spoken sentence (about 12-20 words, never "
                "more than ~25), like a real streamer answering fast on the mic. Address "
                f"{user} by name once. Stay ON the question and the gold market — do not "
                "wander to unrelated topics. Use the context/levels if relevant; if unsure, "
                f"be honest in one line. No emojis, no lists. Reply in {lang}.",
                system="You are a quick, sharp live trading-stream host. Keep every reply to "
                       "ONE short sentence, on-topic, no rambling.",
                max_tokens=int(os.environ.get("AVATAR_COMMENT_TOKENS", "60")), timeout=30)
            return self._say(reply)
        except Exception as exc:
            print(f"[COMMENTS] respond error: {exc}")
            return None
