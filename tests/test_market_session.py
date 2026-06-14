import unittest
from datetime import datetime

from dateutil import tz
from engines.market_session import NEW_YORK, spoken_reopen, xauusd_session


PACIFIC = tz.gettz("America/Los_Angeles")


class MarketSessionTests(unittest.TestCase):
    def test_friday_close_rolls_to_sunday_evening(self):
        now = datetime(2026, 6, 12, 17, 1, tzinfo=NEW_YORK)
        state = xauusd_session(now)
        self.assertFalse(state.is_open)
        self.assertEqual(state.reason, "weekend")
        self.assertEqual(state.next_open,
                         datetime(2026, 6, 14, 18, 0, tzinfo=NEW_YORK))

    def test_saturday_current_weekend_reopens_sunday(self):
        now = datetime(2026, 6, 13, 8, 0, tzinfo=NEW_YORK)
        state = xauusd_session(now)
        self.assertFalse(state.is_open)
        self.assertEqual(state.next_open,
                         datetime(2026, 6, 14, 18, 0, tzinfo=NEW_YORK))
        spoken = spoken_reopen(state, PACIFIC)
        self.assertIn("Sunday at 6:00 PM New York time", spoken)
        self.assertIn("Sunday at 3:00 PM your local time", spoken)

    def test_sunday_reopens_at_six_new_york(self):
        before = xauusd_session(datetime(2026, 6, 14, 17, 59, tzinfo=NEW_YORK))
        after = xauusd_session(datetime(2026, 6, 14, 18, 0, tzinfo=NEW_YORK))
        self.assertFalse(before.is_open)
        self.assertTrue(after.is_open)

    def test_daily_maintenance_break(self):
        state = xauusd_session(datetime(2026, 6, 10, 17, 30, tzinfo=NEW_YORK))
        self.assertFalse(state.is_open)
        self.assertEqual(state.reason, "daily maintenance break")
        self.assertEqual(state.next_open,
                         datetime(2026, 6, 10, 18, 0, tzinfo=NEW_YORK))

    def test_regular_weekday_is_open(self):
        state = xauusd_session(datetime(2026, 6, 10, 12, 0, tzinfo=NEW_YORK))
        self.assertTrue(state.is_open)
        self.assertIsNone(state.next_open)


if __name__ == "__main__":
    unittest.main()
