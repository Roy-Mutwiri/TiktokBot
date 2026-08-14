import os
import sys

import pytest


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)


@pytest.fixture(autouse=True)
def _forget_unreadable_cookie_sources():
    """Keep the cookie skip list from leaking between tests.

    extract_info_with_retries remembers browsers whose cookie store this
    process could not read, so a test that simulates a locked cookie database
    would otherwise delete that browser's attempt from every later test.
    """
    from youtube_dlp_options import reset_unreadable_cookie_sources

    reset_unreadable_cookie_sources()
    yield
    reset_unreadable_cookie_sources()


@pytest.fixture(autouse=True)
def _never_evict_the_real_cache():
    """Keep the test run away from the user's data/youtube_cache.

    Download paths call enforce_budget() with the real cache directory, so a
    test that exercises one would delete genuinely cached videos. A zero budget
    turns eviction into a no-op for every test that does not set its own.
    """
    previous = os.environ.get("AVATAR_YOUTUBE_CACHE_GB")
    os.environ["AVATAR_YOUTUBE_CACHE_GB"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AVATAR_YOUTUBE_CACHE_GB", None)
        else:
            os.environ["AVATAR_YOUTUBE_CACHE_GB"] = previous
