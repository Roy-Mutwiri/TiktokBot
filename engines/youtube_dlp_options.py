import os


BOT_AUTH_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "cookies",
    "login",
    "http error 403",
    "forbidden",
    "video unavailable",
)


class YouTubeDlpAuthError(RuntimeError):
    pass


def extract_info_with_retries(yt_dlp, url, base_opts, download=False,
                              status_callback=None, status_prefix="youtube"):
    """Run yt-dlp, retrying auth/bot blocks with configured browser cookies."""
    attempts = _option_attempts(base_opts)
    last_exc = None
    auth_block_seen = False
    for index, (label, opts) in enumerate(attempts):
        if index > 0:
            _status(status_callback, f"{status_prefix}: retrying with {label}")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=download)
        except Exception as exc:
            last_exc = exc
            if _looks_like_auth_block(exc):
                auth_block_seen = True
                continue
            if auth_block_seen and index > 0:
                continue
            else:
                break
    message = str(last_exc) if last_exc is not None else "unknown yt-dlp error"
    if auth_block_seen or _looks_like_auth_block(message):
        hint = (
            "YouTube asked for sign-in/bot verification. Open YouTube in Edge "
            "or Chrome, make sure you are signed in, then retry. You can also "
            "set AVATAR_YOUTUBE_COOKIES to an exported cookies.txt file or "
            "AVATAR_YOUTUBE_COOKIES_FROM_BROWSER=edge."
        )
        raise YouTubeDlpAuthError(f"{message} ({hint})")
    raise last_exc


def _option_attempts(base_opts):
    base = dict(base_opts or {})
    attempts = []
    explicit = _explicit_cookie_options()
    if explicit:
        first = dict(base)
        first.update(explicit)
        attempts.append(("configured cookies", first))
    else:
        attempts.append(("no cookies", base))

    for browser in _browser_cookie_candidates():
        opts = dict(base)
        opts["cookiesfrombrowser"] = (browser,)
        attempts.append((f"{browser} cookies", opts))
    return attempts


def _explicit_cookie_options():
    cookies_file = os.environ.get("AVATAR_YOUTUBE_COOKIES", "").strip().strip('"')
    if cookies_file:
        return {"cookiefile": cookies_file}
    browser = os.environ.get("AVATAR_YOUTUBE_COOKIES_FROM_BROWSER", "").strip()
    if browser:
        return {"cookiesfrombrowser": _browser_tuple(browser)}
    return {}


def _browser_cookie_candidates():
    configured = os.environ.get(
        "AVATAR_YOUTUBE_COOKIE_BROWSERS", "edge,chrome,firefox")
    explicit = os.environ.get("AVATAR_YOUTUBE_COOKIES_FROM_BROWSER", "").strip()
    if explicit:
        return []
    out = []
    for item in configured.split(","):
        browser = item.strip().lower()
        if browser and browser not in out:
            out.append(browser)
    return tuple(out)


def _browser_tuple(value):
    parts = [part.strip() for part in str(value).split(":") if part.strip()]
    if not parts:
        return ("edge",)
    return tuple(parts)


def _looks_like_auth_block(exc):
    text = str(exc).lower()
    return any(marker in text for marker in BOT_AUTH_MARKERS)


def _status(callback, msg):
    if callback is not None:
        try:
            callback(msg)
        except Exception:
            pass
