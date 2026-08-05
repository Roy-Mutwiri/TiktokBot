import os

import pytest


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
