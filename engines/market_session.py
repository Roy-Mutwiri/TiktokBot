"""Trading-session rules for spot XAUUSD.

Retail spot gold follows the New York forex week: Sunday 6:00 PM through
Friday 5:00 PM America/New_York, with a daily 5:00-6:00 PM maintenance break.
The timezone database handles EST/EDT transitions.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from dateutil import tz


NEW_YORK = tz.gettz("America/New_York")


@dataclass(frozen=True)
class MarketSession:
    is_open: bool
    reason: str
    next_open: datetime | None


def xauusd_session(now=None):
    """Return current XAUUSD session state and the next opening time."""
    if now is None:
        now = datetime.now(NEW_YORK)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=NEW_YORK)
    else:
        now = now.astimezone(NEW_YORK)

    weekday = now.weekday()  # Monday=0, Sunday=6
    clock = now.timetz().replace(tzinfo=None)
    at_5 = datetime.combine(now.date(), time(17, 0), NEW_YORK)
    at_6 = datetime.combine(now.date(), time(18, 0), NEW_YORK)

    if weekday == 5:  # Saturday
        next_open = datetime.combine(now.date() + timedelta(days=1), time(18, 0), NEW_YORK)
        return MarketSession(False, "weekend", next_open)
    if weekday == 6:  # Sunday
        if clock < time(18, 0):
            return MarketSession(False, "weekend", at_6)
        return MarketSession(True, "open", None)
    if weekday == 4 and clock >= time(17, 0):  # Friday close
        next_open = datetime.combine(now.date() + timedelta(days=2), time(18, 0), NEW_YORK)
        return MarketSession(False, "weekend", next_open)
    if time(17, 0) <= clock < time(18, 0):
        return MarketSession(False, "daily maintenance break", at_6)
    return MarketSession(True, "open", None)


def spoken_reopen(session, viewer_tz=None):
    """Human-readable next-open time in New York and optionally viewer timezone."""
    if session.is_open or session.next_open is None:
        return ""
    def _fmt(dt, suffix):
        hour = dt.strftime("%I").lstrip("0") or "12"
        return f"{dt.strftime('%A')} at {hour}:{dt.strftime('%M %p')} {suffix}"

    ny = session.next_open.astimezone(NEW_YORK)
    text = _fmt(ny, "New York time")
    if viewer_tz:
        local = session.next_open.astimezone(viewer_tz)
        text += ", " + _fmt(local, "your local time")
    return text
