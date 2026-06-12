# =============================================================================
# engines/econ_calendar.py — upcoming high-impact economic events (for gold).
#
# Pulls this week's calendar (ForexFactory free JSON), keeps the HIGH-impact USD
# events (CPI, NFP, Fed, etc. — the ones that move gold), and surfaces a heads-up
# when one is imminent so the avatar can warn viewers ("CPI in 30 minutes, expect
# volatility"). Caches the feed; announces each event at most twice (heads-up +
# imminent). Never raises.
# =============================================================================
import os
import time
import datetime

_FEED = os.environ.get("AVATAR_ECON_FEED",
                       "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
_HEADERS = {"User-Agent": "Mozilla/5.0"}
# announce a high-impact event when it's ~this many minutes away (each fires once)
_LEADS = (30, 4)


class EconCalendar:
    def __init__(self, countries=("USD",)):
        self.countries = set(countries)
        self._events = []        # list of dicts: {title, country, time(aware), forecast, previous}
        self._fetched = 0.0
        self._announced = set()  # (title, iso, lead) already announced

    def _refresh(self):
        if self._events and (time.time() - self._fetched) < 7200:   # cache 2h
            return
        try:
            import requests
            data = requests.get(_FEED, timeout=10, headers=_HEADERS).json()
            evs = []
            for e in data:
                if e.get("impact") != "High":
                    continue
                if self.countries and e.get("country") not in self.countries:
                    continue
                try:
                    dt = datetime.datetime.fromisoformat(e["date"])
                except Exception:
                    continue
                evs.append({"title": e.get("title", "an event"),
                            "country": e.get("country", ""),
                            "time": dt,
                            "forecast": e.get("forecast") or "",
                            "previous": e.get("previous") or ""})
            self._events = sorted(evs, key=lambda x: x["time"])
            self._fetched = time.time()
        except Exception as exc:
            print(f"[ECON] calendar fetch failed ({exc}).")

    def next_alert(self):
        """If a high-impact event is imminent (and not yet announced at this lead),
        return a brain prompt to announce it live, else None."""
        try:
            self._refresh()
            if not self._events:
                return None
            now = datetime.datetime.now(datetime.timezone.utc)
            for e in self._events:
                mins = (e["time"] - now).total_seconds() / 60.0
                if mins < -2:
                    continue
                for lead in _LEADS:
                    key = (e["title"], e["time"].isoformat(), lead)
                    if key in self._announced:
                        continue
                    # fire when within `lead` minutes (but not for the larger lead if
                    # we're already inside the smaller one's window only)
                    if 0 < mins <= lead:
                        self._announced.add(key)
                        when = ("in about a minute" if mins < 2
                                else f"in about {int(round(mins))} minutes")
                        fc = f" Forecast {e['forecast']} vs previous {e['previous']}." \
                            if e["forecast"] else ""
                        return (f"HEADS-UP for traders: high-impact {e['country']} news — "
                                f"{e['title']} — is coming {when}.{fc} Warn viewers gold could "
                                "get volatile and to manage risk. One short, natural sentence.")
            return None
        except Exception as exc:
            print(f"[ECON] next_alert error ({exc}).")
            return None
