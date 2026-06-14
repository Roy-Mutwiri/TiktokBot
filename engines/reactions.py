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
_READY_USED = {}      # category -> set of used intro/ending combinations

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


def _extend_unique(pool, lines):
    seen = set(pool)
    for line in lines:
        line = str(line or "").strip()
        if line and line not in seen:
            pool.append(line)
            seen.add(line)


_extend_unique(_READY["intro"], [
    "Respect to",
    "Real one alert for",
    "Let me shout out",
    "Chat, welcome",
    "Big respect for",
    "I appreciate",
    "Massive love to",
    "Quick shout-out to",
    "The room sees",
    "Family, say welcome to",
    "Much love and respect to",
    "Wallahi, big love to",
    "Mashallah, welcome",
    "Strong support from",
    "I have to thank",
    "Do not miss",
    "Energy just came from",
    "We have love from",
    "Special respect to",
    "All eyes on",
    "Let us recognize",
    "The family welcomes",
    "This room appreciates",
    "A real supporter just arrived",
    "Big moment for",
    "Nothing but respect for",
    "I see the support from",
    "Give it up for",
    "Welcome to the room",
    "Appreciation going to",
    "Love in the chat for",
    "A proper shout-out for",
    "Strong entrance from",
    "The stream appreciates",
    "Right now I see",
    "That support came from",
])

_extend_unique(_READY["follow"], [
    "for the follow. You are locked in with us now!",
    "for following. Welcome to the room, stay close!",
    "for tapping follow. That support keeps the family growing!",
    "for joining us. The next chart move is coming soon!",
    "for the follow. Good to have sharp eyes in here!",
    "for becoming part of the stream. I appreciate the trust!",
    "for following. You came at a good time, stay with us!",
    "for joining the gold room. Much respect and welcome!",
    "for that follow. You just made the room stronger!",
    "for coming in and following. Make yourself comfortable!",
    "for the follow. We are building something serious here!",
    "for supporting the live. Welcome to the trading family!",
    "for joining. Watch the levels with us, it is getting interesting!",
    "for hitting that follow. I appreciate you more than you know!",
    "for stepping into the family. Welcome, stay active!",
    "for the follow. The room is better with you in it!",
    "for following. You are early for the next setup!",
    "for the support. Welcome in, let us keep the energy high!",
    "for joining us live. Your follow means real support!",
    "for the follow. I see you and I appreciate you!",
    "for following. Stay close, the next signal needs focus!",
    "for joining the squad. Big respect and welcome!",
    "for that follow. You are officially part of the room!",
    "for supporting the stream. We move together here!",
    "for following. That is the kind of support I notice!",
    "for joining. Let us read this market together!",
    "for the follow. Fresh energy just entered the live!",
    "for stepping in. Welcome to the family, enjoy the action!",
    "for following. You are not just watching, you are with us now!",
    "for the follow. I appreciate the love and the timing!",
    "for joining the live. Stay with us, the room is warming up!",
    "for that follow. You helped push this live forward!",
])

_extend_unique(_READY["share"], [
    "for sharing the live. That helps more than you think!",
    "for sending the stream out. You just brought more energy in!",
    "for the share. That is how the family grows!",
    "for sharing us. Real support, and I notice it!",
    "for putting the live in front of more people!",
    "for backing the stream with a share. Much respect!",
    "for the share. You helped the room get louder!",
    "for spreading the live. That support is not small!",
    "for sharing this with your people. I appreciate that!",
    "for the share. More eyes, more energy, better room!",
    "for pushing the live forward with that share!",
    "for sharing. You are helping the whole community grow!",
    "for sending people this way. That is a real supporter move!",
    "for the share. I see who is helping the stream!",
    "for spreading the word. You just opened the door for new people!",
    "for sharing the stream. That is family behavior!",
    "for the share. You helped bring fresh eyes to the charts!",
    "for sending the live out. I appreciate that kind of support!",
    "for sharing. That is how we keep the room alive!",
    "for backing us with a share. Big respect to you!",
    "for the share. You are helping this live travel!",
    "for spreading this stream. The room appreciates you!",
    "for the share. That support carries the live further!",
    "for helping more traders find us. Thank you!",
])

_extend_unique(_READY["gift_small"], [
    "for the gift. Small gift, real support!",
    "for sending that gift. I see the love!",
    "for the support. Every gift adds energy here!",
    "for the gift. You just lifted the room a little higher!",
    "for showing love. That means something to me!",
    "for that gift. I appreciate every bit of support!",
    "for the gift. You are helping keep this live moving!",
    "for coming through. The room sees your support!",
    "for that support. Big heart, much respect!",
    "for the gift. You did not have to, and I appreciate it!",
    "for sending love to the stream. Thank you!",
    "for the gift. That support never goes unnoticed!",
    "for backing the live. Much respect to you!",
    "for the support. You added good energy to the room!",
    "for that gift. Real support from a real one!",
    "for showing up with love. I appreciate you!",
    "for the gift. You helped make this live better!",
    "for that support. I see you clearly!",
    "for the gift. The energy just went up!",
    "for supporting the room. Thank you from the heart!",
])

_extend_unique(_READY["gift_mid"], [
    "for that generous gift. You really came through!",
    "for the strong support. The room feels that!",
    "for that gift. You just raised the energy in here!",
    "for supporting like that. I truly appreciate it!",
    "for the generous love. That is a serious supporter move!",
    "for that gift. You are helping carry this live!",
    "for the support. Everybody can see the love!",
    "for backing the stream. That means a lot to me!",
    "for that generous gift. You are officially family here!",
    "for showing up strong. Big respect and thank you!",
    "for the gift. That is the kind of support I remember!",
    "for supporting the room like that. Much love!",
    "for that strong gift. You just changed the mood!",
    "for the generosity. I appreciate you deeply!",
    "for the gift. Chat sees the support you are showing!",
    "for backing us. You brought real momentum!",
    "for that gift. You are moving like a regular already!",
    "for the support. That was clean, generous, and appreciated!",
])

_extend_unique(_READY["gift_big"], [
    "for the big gift. You just woke the whole room up!",
    "for the massive support. That is VIP energy!",
    "for that big gift. Everybody needs to show you love!",
    "for coming through so strong. I will remember that!",
    "for the huge support. You just made a moment!",
    "for that big gift. The room felt that immediately!",
    "for backing this live like a legend!",
    "for the serious support. That is not normal, thank you!",
    "for the big gift. You just became a name in this room!",
    "for that massive gift. I appreciate you from the heart!",
    "for the huge love. Chat, show respect right now!",
    "for supporting on that level. Absolute legend behavior!",
    "for the big gift. You just gave the stream a new pulse!",
    "for that support. VIP respect to you, seriously!",
    "for the gift. You changed the energy in one second!",
    "for backing us that hard. That is unforgettable!",
])

_extend_unique(_READY["gift_huge"], [
    "for that incredible gift. The whole live has to stop for you!",
    "for the unbelievable support. That is top supporter energy!",
    "for that enormous gift. You just made stream history!",
    "for blessing the room like that. I am honestly grateful!",
    "for that legendary gift. Everybody show respect right now!",
    "for the massive support. You are VIP in this room!",
    "for that huge gift. I will remember your name here!",
    "for lifting the entire stream. That was a serious moment!",
    "for the incredible support. You just made the room explode!",
    "for that gift. This is the kind of support people remember!",
    "for showing love on another level. Absolute top supporter!",
    "for the huge blessing. You just owned this moment!",
    "for that legendary support. The whole family sees you!",
    "for the massive gift. That is not regular support, that is special!",
])


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


def _ready_combo(category):
    """Pick a complete intro+ending pair without repeating until the pair space
    is exhausted. This makes long lives feel much less looped than independently
    shuffling the two chunks."""
    intros = _READY["intro"]
    endings = _READY[category]
    total = len(intros) * len(endings)
    if total <= 0:
        return "", ""
    with _LOCK:
        used = _READY_USED.setdefault(category, set())
        if len(used) >= total:
            used.clear()
        for _ in range(32):
            combo = random.randrange(total)
            if combo not in used:
                break
        else:
            fresh = [i for i in range(total) if i not in used]
            combo = random.choice(fresh) if fresh else random.randrange(total)
        used.add(combo)
    intro = intros[combo // len(endings)]
    ending = endings[combo % len(endings)]
    return intro, ending


def _personal(user, category):
    intro, ending = _ready_combo(category)
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


def _make_lines(openers, endings):
    return [f"{opener} {ending}" for opener in openers for ending in endings]


def _expand_template_data(data):
    """Add a large generated template corpus without storing thousands of JSON lines."""
    banks = {
        "follow": _make_lines([
            "{user}, welcome in!",
            "Big love {user}.",
            "Respect to {user}.",
            "I see you {user}.",
            "Welcome to the family, {user}.",
            "{user}, good to have you here.",
            "Chat, welcome {user}.",
            "Mashallah {user}, welcome.",
            "Strong entrance from {user}.",
            "The room sees you, {user}.",
            "Appreciation to {user}.",
            "Fresh energy from {user}.",
        ], [
            "Thank you for the follow and stay close.",
            "That follow means real support.",
            "You joined at the right time.",
            "We move together in this room.",
            "The family just got stronger.",
            "Keep your eyes on the next setup.",
            "I appreciate you being part of the live.",
            "Make yourself at home here.",
            "That support helps the stream grow.",
            "You are locked in with us now.",
            "Thanks for backing the live.",
            "The chart room welcomes you.",
        ]),
        "follow_batch": _make_lines([
            "Welcome {names}.",
            "Big love to {names}.",
            "Chat, welcome {names}.",
            "Respect to {names}.",
            "Fresh follows from {names}.",
            "The family just grew with {names}.",
        ], [
            "Thank you all for following.",
            "Appreciate every one of those follows.",
            "You are all part of the room now.",
            "Stay close, the next move is coming.",
            "That is real support from the chat.",
            "Welcome in, the energy is climbing.",
        ]),
        "share": _make_lines([
            "{user}, thank you for sharing.",
            "Big respect {user}.",
            "I see that share, {user}.",
            "{user} just helped the stream travel.",
            "Real support from {user}.",
            "Appreciation to {user}.",
            "The room sees your share, {user}.",
            "Much love {user}.",
        ], [
            "You helped bring more people into the room.",
            "That is how this family grows.",
            "The stream moves further because of that.",
            "More eyes on the charts means more energy.",
            "That support is never small to me.",
            "You are helping the live breathe.",
            "That is a real supporter move.",
            "I appreciate you spreading the word.",
        ]),
        "gift_small": _make_lines([
            "Thank you for the {gift}, {user}.",
            "I see the {gift}, {user}.",
            "Big love for the {gift}, {user}.",
            "{user}, that {gift} is appreciated.",
            "Respect for the {gift}, {user}.",
            "{user}, thank you for backing the live.",
        ], [
            "Small gift, real support.",
            "That adds energy to the room.",
            "I appreciate every bit of love.",
            "You did not have to, and I notice it.",
            "That helps keep the live moving.",
            "The room sees your support.",
            "That is a kind supporter move.",
            "Much respect from me.",
        ]),
        "gift_mid": _make_lines([
            "Mashallah {user}, thank you for the {gift}.",
            "Strong support from {user} with the {gift}.",
            "{user}, that {gift} is generous.",
            "Big respect for the {gift}, {user}.",
            "The room sees that {gift}, {user}.",
            "{user}, you came through with the {gift}.",
        ], [
            "You just lifted the energy in here.",
            "That is real generosity.",
            "I appreciate that from the heart.",
            "You are moving like family now.",
            "That support keeps the room alive.",
            "Everybody can see the love.",
            "That is the kind of support I remember.",
            "You brought momentum to the live.",
        ]),
        "gift_big": _make_lines([
            "Wallahi {user}, the {gift}.",
            "Huge support from {user} with the {gift}.",
            "{user}, that {gift} is a big moment.",
            "Chat, show love to {user} for the {gift}.",
            "VIP energy from {user} with the {gift}.",
            "{user}, you just dropped the {gift}.",
        ], [
            "You changed the energy in the whole room.",
            "That is serious support, thank you.",
            "I will remember that one.",
            "Everybody needs to respect that.",
            "You just made the stream louder.",
            "That is not regular support.",
            "Absolute legend behavior.",
            "You just became a name in this room.",
        ]),
        "gift_huge": _make_lines([
            "Stop everything, {user} with the {gift}.",
            "Wallahi {user}, that {gift} is unbelievable.",
            "Chat, {user} just made a huge moment with the {gift}.",
            "{user}, that {gift} is legendary.",
            "Top supporter energy from {user} with the {gift}.",
            "Ya salam {user}, the {gift}.",
        ], [
            "The whole live has to recognize you.",
            "You just made stream history.",
            "I am genuinely grateful for that.",
            "That is VIP support on another level.",
            "Everybody show respect right now.",
            "That support is unforgettable.",
            "You just owned this moment.",
            "The room will remember that.",
        ]),
        "likes": _make_lines([
            "We just passed {total} likes.",
            "{total} likes in the room.",
            "Mashallah, {total} likes.",
            "Look at that, {total} likes.",
            "The room pushed us to {total} likes.",
            "Big milestone, {total} likes.",
        ], [
            "Thank you everyone, keep the pressure on.",
            "Smash like if you are watching this move.",
            "The energy is climbing, thank you family.",
            "Every tap helps the live reach more people.",
            "Keep it going, the room is waking up.",
            "I appreciate every single one of you.",
            "That support keeps the stream moving.",
            "Let us push the next milestone together.",
        ]),
        "goal": _make_lines([
            "We smashed the {coins}-coin goal.",
            "Goal complete at {coins} coins.",
            "{coins} coins hit, family.",
            "The {coins}-coin goal is done.",
            "Chat, the {coins}-coin goal just fell.",
            "Mashallah, {coins} coins reached.",
        ], [
            "Thank you legends, next signal coming.",
            "You unlocked the next gold breakdown.",
            "That support means the next setup is live.",
            "I appreciate the whole room for that.",
            "You earned the next market read.",
            "Now let us get into the next level.",
            "The room delivered, and I appreciate it.",
            "Next analysis is loading because of you.",
        ]),
    }
    for category, lines in banks.items():
        pool = data.setdefault(category, [])
        _extend_unique(pool, lines)


def _load():
    global _DATA
    if _DATA is None:
        try:
            with open(_FILE, "r", encoding="utf-8") as f:
                _DATA = json.load(f)
        except Exception as exc:
            print(f"[REACTIONS] could not load {_FILE} ({exc}) — using fallbacks.")
            _DATA = {}
        _expand_template_data(_DATA)
    return _DATA


def reload():
    """Re-read reactions.json (so edits apply without a restart)."""
    global _DATA
    with _LOCK:
        _DATA = None
        _RECENT.clear()
        _READY_USED.clear()
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
    total = len(names)
    names = names[:4]                       # name at most 4 so the line stays short
    joined = (f"{names[0]} and {names[1]}" if len(names) == 2
              else ", ".join(names[:-1]) + f" and {names[-1]}")
    if total > len(names):
        extra = total - len(names)
        joined = f"{joined}, plus {extra} more"
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
