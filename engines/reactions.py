# =============================================================================
# engines/reactions.py — INSTANT offline reactions for follows / gifts / shares /
# likes / goals. Pure templates loaded from reactions.json (NO LLM), so a "thank
# you" fires the MOMENT the event happens with zero generation delay.
#
# Picks a RANDOM line per event and avoids the recently-used ones, so it stays
# varied across a long stream. Falls back to a hard-coded line if the file is
# missing. Reload the JSON live with reload().
#
#   import reactions
#   reactions.follow("ali")                 -> "Welcome to the family, ali! ..."
#   reactions.gift("sara", "Rose", 1, 1)    -> tiered by coin value
# =============================================================================
import os
import json
import random
import re
import threading

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reactions.json")
_LOCK = threading.Lock()
_DATA = None
_RECENT = {}          # category -> list of recently used indices (anti-repeat)

_FALLBACK = {
    "follow": "Welcome to the family, {user}, thank you for the follow!",
    "gift_small": "Thank you for the {gift}, {user}, I appreciate you!",
    "gift_mid": "Mashallah {user}, thank you for the {gift}!",
    "gift_big": "Wallahi {user}, thank you so much for the {gift}, legend!",
    "gift_huge": "Ya salam {user}, the {gift}?! Thank you so much, absolute VIP!",
    "share": "Thank you for sharing the stream, {user}!",
    "likes": "We just passed {total} likes — thank you everyone!",
    "goal": "We smashed the {coins}-coin goal — thank you all!",
}


# Reusable pieces are synthesized before the live starts. Combining a shuffled
# opener with a shuffled short/medium/long ending creates hundreds of responses
# without making the live wait for full-sentence generation.
_READY = {
    "intro": [
        "Thank you",
        "Big love to",
        "A special shout-out to",
        "I see you",
        "Much respect to",
        "Ya salam, thank you",
        "Everybody show some love to",
        "Hold on, I have to recognize",
    ],
    "follow": [
        "for the follow. Welcome in!",
        "for joining the family!",
        "for hitting follow. You are one of us now!",
        "for following. Stay close, the next market setup is coming!",
        "for that follow. I appreciate you being part of this live!",
        "for joining us today. Welcome to the room, and enjoy the market action!",
        "for following the stream. Your support helps this community keep growing!",
        "for becoming part of the family. We are watching the charts and moving together!",
        "for the follow. Make yourself at home!",
        "for joining the squad. Much love!",
        "for following. We have plenty more market action ahead, so stay with us!",
        "for that support. Welcome aboard, and let us catch the next move together!",
    ],
    "share": [
        "for sharing the live!",
        "for spreading the word. Much love!",
        "for that share. You are helping the family grow!",
        "for sharing this stream with your people. I really appreciate you!",
        "for helping more traders find the room. That support means a lot!",
        "for sharing the live. More eyes on the charts means more energy in the room!",
        "for sending the stream out. That is real support, and I do not take it for granted!",
        "for helping us reach a bigger audience. Welcome everyone coming in from that share!",
        "for the share. You are a real one!",
        "for putting the live out there. Thank you!",
        "for sharing us with your friends. Let us keep this room moving!",
        "for backing the stream with a share. The whole community appreciates you!",
    ],
    "gift_small": [
        "for the gift. Thank you!",
        "for showing love with that gift!",
        "for the gift. Small gesture, big support!",
        "for sending that gift. I see you and I appreciate you!",
        "for supporting the stream. Every gift brings great energy to the room!",
        "for that thoughtful gift. Thank you for being here and showing love!",
        "for the gift. You just added some extra energy to this live!",
        "for supporting what we are building here. Much respect and thank you!",
        "for the gift. You are appreciated!",
        "for coming through with the support. Big love!",
    ],
    "gift_mid": [
        "for that generous gift. Mashallah, thank you!",
        "for the strong support. Much respect!",
        "for that gift. You really came through for the stream!",
        "for the generous support. That means more than you know!",
        "for showing the room so much love. I appreciate your generosity!",
        "for backing the live in a big way. You just lifted the energy in here!",
        "for that generous gift. The whole room sees the love you are showing!",
        "for supporting the stream like that. We are grateful to have you here!",
        "for the gift. Absolute legend!",
        "for that support. You are family now!",
    ],
    "gift_big": [
        "for the big gift. You are a legend!",
        "for that massive support. Ya salam!",
        "for the big gift. You just changed the energy in the whole room!",
        "for coming through in such a huge way. I truly appreciate you!",
        "for that incredible support. Everybody in the chat, show some love!",
        "for backing this live so strongly. That is a serious act of generosity!",
        "for the huge gift. I will remember that support, and I am grateful!",
        "for lifting up the entire stream. You just earned a major shout-out!",
        "for that big support. Absolute VIP!",
        "for the incredible gift. Much love and respect!",
    ],
    "gift_huge": [
        "for that incredible gift. Absolute VIP!",
        "for the unbelievable support. I am speechless!",
        "for that enormous gift. Stop everything and show this legend some love!",
        "for making a huge moment on this stream. Wallahi, I appreciate you!",
        "for that extraordinary support. You just made the whole room come alive!",
        "for showing generosity on another level. This live will remember your name!",
        "for the incredible gift. Everybody in the chat, give our top supporter some love!",
        "for blessing the stream in such a massive way. You made this a night to remember!",
        "for that legendary support. You are the VIP of the room!",
        "for the huge gift. Ya salam, thank you from the heart!",
    ],
}


def ready_lines():
    """All reusable response pieces that should be synthesized before going live."""
    return list(dict.fromkeys(
        line for pool in _READY.values() for line in pool
    ))


def _spoken_name(user):
    name = re.sub(r"\s+", " ", str(user or "").strip().lstrip("@"))
    return name[:48] or "my friend"


def _ready_pick(category):
    pool = _READY[category]
    with _LOCK:
        key = f"ready:{category}"
        recent = _RECENT.setdefault(key, [])
        fresh = [i for i in range(len(pool)) if i not in recent]
        if not fresh:
            fresh = list(range(len(pool)))
            recent.clear()
        idx = random.choice(fresh)
        recent.append(idx)
        while len(recent) > max(2, len(pool) // 2):
            recent.pop(0)
    return pool[idx]


def _personal(user, category):
    intro = _ready_pick("intro")
    ending = _ready_pick(category)
    return f"{intro} [[CUT]] {_spoken_name(user)}. [[CUT]] {ending}"


def ready_follow(user):
    return _personal(user, "follow")


def ready_share(user):
    return _personal(user, "share")


def ready_gift(user, gift_name="", coins=0):
    c = int(coins or 0)
    tier = ("gift_huge" if c >= 5000 else "gift_big" if c >= 500
            else "gift_mid" if c >= 50 else "gift_small")
    return _personal(user, tier)


def _load():
    global _DATA
    if _DATA is None:
        try:
            with open(_FILE, "r", encoding="utf-8") as f:
                _DATA = json.load(f)
        except Exception as exc:
            print(f"[REACTIONS] could not load {_FILE} ({exc}) — using fallbacks.")
            _DATA = {}
    return _DATA


def reload():
    """Re-read reactions.json (so edits apply without a restart)."""
    global _DATA
    with _LOCK:
        _DATA = None
        _RECENT.clear()
    return _load()


def _pick(category, **kw):
    """Pick a random template from `category`, avoiding the recently-used ones,
    and fill its placeholders. Always returns a spoken-ready string."""
    pool = _load().get(category) or []
    tmpl = None
    if pool:
        with _LOCK:
            recent = _RECENT.setdefault(category, [])
            fresh = [i for i in range(len(pool)) if i not in recent]
            if not fresh:                       # cycled through them all -> reset
                fresh = list(range(len(pool)))
                recent.clear()
            idx = random.choice(fresh)
            recent.append(idx)
            # remember up to half the pool so it doesn't repeat soon
            while len(recent) > max(1, len(pool) // 2):
                recent.pop(0)
        tmpl = pool[idx]
    if tmpl is None:
        tmpl = _FALLBACK.get(category, "Thank you {user}!")
    try:
        return tmpl.format(**kw)
    except Exception:
        return tmpl


def follow(user):
    return _pick("follow", user=str(user))


def follow_many(names):
    """Batch several follows that arrived together into ONE natural line (so a burst
    of follows is thanked instantly, not as 8 separate slow shout-outs)."""
    names = [str(n).strip() for n in (names or []) if str(n).strip()]
    if not names:
        return None
    if len(names) == 1:
        return follow(names[0])
    names = names[:4]                       # name at most 4 so the line stays short
    joined = (f"{names[0]} and {names[1]}" if len(names) == 2
              else ", ".join(names[:-1]) + f" and {names[-1]}")
    return _pick("follow_batch", names=joined)


def share(user):
    return _pick("share", user=str(user))


def likes(total):
    return _pick("likes", total=int(total or 0))


def goal(coins):
    return _pick("goal", coins=int(coins or 0))


def gift(user, gift_name, count=1, coins=0):
    """Tier the appreciation by coin value so a Galaxy lands bigger than a Rose."""
    c = int(coins or 0)
    n = int(count or 1)
    tier = ("gift_huge" if c >= 5000 else "gift_big" if c >= 500
            else "gift_mid" if c >= 50 else "gift_small")
    amt = f"{n} {gift_name}s" if n > 1 else f"a {gift_name}"
    return _pick(tier, user=str(user), gift=str(gift_name), count=n, coins=c, amt=amt)


if __name__ == "__main__":
    print("follow:", follow("ali"))
    print("share :", share("mo"))
    print("likes :", likes(5000))
    print("goal  :", goal(200))
    for cn in (1, 100, 1000, 30000):
        print(f"gift {cn:>5}:", gift("sara", "Rose", 1, cn))
