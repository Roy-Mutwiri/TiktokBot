"""Cookie-retry and error-reporting behaviour for yt-dlp calls.

These cover the failure that made the studio log undiagnosable: every YouTube
problem was reported as "could not find firefox cookies database ... YouTube
asked for sign-in/bot verification", because only the last attempt's exception
survived and the bare "cookies" bot-auth marker matched that message.
"""

import os
import sys
import types
import unittest
from unittest import mock


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import youtube_dlp_options
from youtube_dlp_options import (
    YouTubeDlpAuthError,
    _looks_like_auth_block,
    _option_attempts,
    extract_info_with_retries,
    one_line,
    reset_unreadable_cookie_sources,
)


# Verbatim yt-dlp output from the machine this was diagnosed on.
EDGE_DPAPI_ERROR = (
    "ERROR: ERROR: Failed to decrypt with DPAPI. See  "
    "https://github.com/yt-dlp/yt-dlp/issues/10927  for more info")
CHROME_LOCKED_ERROR = (
    "ERROR: ERROR: Could not copy Chrome cookie database. See  "
    "https://github.com/yt-dlp/yt-dlp/issues/7271  for more info")
FIREFOX_MISSING_ERROR = (
    "ERROR: could not find firefox cookies database in "
    "'C:\\\\Users\\\\user\\\\AppData\\\\Roaming\\\\Mozilla\\\\Firefox\\\\Profiles'")

ALL_BROWSERS = {"AVATAR_YOUTUBE_COOKIE_BROWSERS": "edge,chrome,firefox"}


def make_module(handler):
    """A fake yt_dlp whose extract_info defers to `handler(opts)`."""

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=False):
            return handler(self.opts)

    return types.SimpleNamespace(YoutubeDL=FakeYDL)


def cookie_source(opts):
    """Which cookie jar an attempt was configured with, if any."""
    browser = opts.get("cookiesfrombrowser")
    if browser:
        return browser[0]
    return "file" if opts.get("cookiefile") else None


class CookieErrorReportingTests(unittest.TestCase):
    def setUp(self):
        reset_unreadable_cookie_sources()
        self.addCleanup(reset_unreadable_cookie_sources)

    # -- the production failure -------------------------------------------

    def _broken_cookie_jars(self, first_error):
        """Every browser cookie store unreadable; attempt 0 fails as given."""
        faults = {
            "edge": EDGE_DPAPI_ERROR,
            "chrome": CHROME_LOCKED_ERROR,
            "firefox": FIREFOX_MISSING_ERROR,
        }

        def handler(opts):
            source = cookie_source(opts)
            if source is None:
                raise RuntimeError(first_error)
            raise RuntimeError(faults[source])

        return handler

    def test_root_error_survives_unreadable_cookie_stores(self):
        """The reported cause is attempt 0's error, not the last browser's."""
        handler = self._broken_cookie_jars("Sign in to confirm you're not a bot")
        statuses = []
        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(YouTubeDlpAuthError) as raised:
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True}, status_callback=statuses.append)

        message = str(raised.exception)
        self.assertIn("Sign in to confirm you're not a bot", message)
        # The old code reported the firefox message for every single failure.
        self.assertNotIn("could not find firefox", message)

    def test_auth_error_names_the_unreadable_cookie_sources(self):
        handler = self._broken_cookie_jars("Sign in to confirm you're not a bot")
        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(YouTubeDlpAuthError) as raised:
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True})

        message = str(raised.exception)
        for browser in ("edge", "chrome", "firefox"):
            self.assertIn(f"{browser} cookies", message)

    def test_every_attempt_failure_reaches_the_log(self):
        """Each attempt's real error is logged, not just the survivor."""
        handler = self._broken_cookie_jars("Sign in to confirm you're not a bot")
        statuses = []
        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(YouTubeDlpAuthError):
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True}, status_callback=statuses.append,
                    status_prefix="video preview")

        joined = "\n".join(statuses)
        self.assertIn("video preview: no cookies failed", joined)
        self.assertIn("Sign in to confirm", joined)
        self.assertIn("edge cookies unreadable", joined)
        self.assertIn("Failed to decrypt with DPAPI", joined)
        self.assertIn("chrome cookies unreadable", joined)
        self.assertIn("Could not copy Chrome cookie database", joined)
        self.assertIn("firefox cookies unreadable", joined)

    # -- classification ----------------------------------------------------

    def test_missing_cookie_database_is_not_a_bot_block(self):
        """A local cookie fault must not masquerade as a YouTube auth demand."""
        self.assertFalse(_looks_like_auth_block(FIREFOX_MISSING_ERROR))
        self.assertFalse(_looks_like_auth_block(EDGE_DPAPI_ERROR))
        self.assertFalse(_looks_like_auth_block(CHROME_LOCKED_ERROR))

    def test_genuine_auth_messages_are_still_detected(self):
        self.assertTrue(_looks_like_auth_block("Sign in to confirm you're not a bot"))
        self.assertTrue(_looks_like_auth_block("HTTP Error 403: Forbidden"))
        self.assertTrue(_looks_like_auth_block("Video unavailable"))
        self.assertTrue(_looks_like_auth_block("This video requires login"))

    def test_local_cookie_fault_alone_is_not_reported_as_auth_error(self):
        """No auth block anywhere -> the original error type propagates."""

        def handler(opts):
            if cookie_source(opts) is None:
                raise ValueError("Requested format is not available")
            raise RuntimeError(FIREFOX_MISSING_ERROR)

        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(ValueError) as raised:
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True})

        self.assertIn("Requested format is not available", str(raised.exception))

    def test_non_auth_failure_does_not_trigger_cookie_retries(self):
        """A plain download error should not cost three doomed cookie attempts."""
        seen = []

        def handler(opts):
            seen.append(cookie_source(opts))
            raise RuntimeError("Unable to rename file: disk full")

        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(RuntimeError):
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True})

        self.assertEqual(seen, [None])

    # -- skip list ---------------------------------------------------------

    def test_unreadable_browser_is_skipped_on_the_next_call(self):
        """A browser that already refused is not retried for every download."""
        handler = self._broken_cookie_jars("Sign in to confirm you're not a bot")
        module = make_module(handler)
        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(YouTubeDlpAuthError):
                extract_info_with_retries(
                    module, "https://youtube.test/watch?v=abc", {"quiet": True})

            second = []

            def counting(opts):
                second.append(cookie_source(opts))
                raise RuntimeError("Sign in to confirm you're not a bot")

            with self.assertRaises(YouTubeDlpAuthError):
                extract_info_with_retries(
                    make_module(counting), "https://youtube.test/watch?v=abc",
                    {"quiet": True})

        self.assertEqual(second, [None])

    def test_reset_restores_the_browser_attempts(self):
        handler = self._broken_cookie_jars("Sign in to confirm you're not a bot")
        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            with self.assertRaises(YouTubeDlpAuthError):
                extract_info_with_retries(
                    make_module(handler), "https://youtube.test/watch?v=abc",
                    {"quiet": True})
            self.assertEqual(
                [label for label, _ in _option_attempts({})], ["no cookies"])

            reset_unreadable_cookie_sources()

            self.assertEqual(
                [label for label, _ in _option_attempts({})],
                ["no cookies", "edge cookies", "chrome cookies",
                 "firefox cookies"])

    def test_explicitly_configured_cookie_file_is_always_attempted(self):
        """The skip list must never suppress what the user configured."""
        youtube_dlp_options._remember_unreadable("edge cookies")
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_COOKIES_FROM_BROWSER": "edge"}, clear=False):
            labels = [label for label, _ in _option_attempts({})]

        self.assertEqual(labels, ["configured cookies"])

    # -- log hygiene -------------------------------------------------------

    def test_one_line_strips_colour_codes_and_doubled_prefix(self):
        raw = "\x1b[0;31mERROR:\x1b[0m ERROR: Failed to decrypt\nwith  DPAPI"
        self.assertEqual(one_line(raw), "Failed to decrypt with DPAPI")

    def test_one_line_survives_empty_input(self):
        self.assertEqual(one_line(""), "unknown yt-dlp error")
        self.assertEqual(one_line(None), "unknown yt-dlp error")

    # -- unchanged happy paths --------------------------------------------

    def test_bot_block_still_recovers_via_browser_cookies(self):
        def handler(opts):
            if cookie_source(opts) is None:
                raise RuntimeError("Sign in to confirm you're not a bot")
            return {"title": "OK"}

        statuses = []
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_COOKIE_BROWSERS": "edge"}, clear=False):
            info = extract_info_with_retries(
                make_module(handler), "https://youtube.test/watch?v=abc",
                {"quiet": True}, status_callback=statuses.append)

        self.assertEqual(info["title"], "OK")
        self.assertIn("youtube: retrying with edge cookies", statuses)

    def test_first_attempt_success_makes_no_further_calls(self):
        seen = []

        def handler(opts):
            seen.append(cookie_source(opts))
            return {"title": "OK"}

        with mock.patch.dict(os.environ, ALL_BROWSERS, clear=False):
            info = extract_info_with_retries(
                make_module(handler), "https://youtube.test/watch?v=abc",
                {"quiet": True})

        self.assertEqual(info["title"], "OK")
        self.assertEqual(seen, [None])


if __name__ == "__main__":
    unittest.main()
