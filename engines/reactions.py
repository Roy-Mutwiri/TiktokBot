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
