import os
import re


# Errors worth retrying with cookies. YouTube rarely says "you are a bot" any
# more: an unauthenticated request that it does not like is answered with a
# player response carrying no formats at all, which yt-dlp reports as "No video
# formats found" or "Unable to extract ...". Those read like a broken video and
# used to end the run on the very first cookieless attempt, so the browser
# cookies that would have satisfied the request were never tried and a link
# that works in the browser came back as "youtube failed".
BOT_AUTH_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "login",
    "http error 403",
    "forbidden",
    "video unavailable",
    "no video formats found",
    "unable to extract",
    "failed to extract any player response",
)

# Reasons a browser's cookie jar cannot be read *on this machine*. These say
# nothing about YouTube, so they must never be reported as the cause of a
# failure. "could not find firefox cookies database" used to match a bare
# "cookies" entry in BOT_AUTH_MARKERS, so an uninstalled browser was dressed up
# as a sign-in demand while the real first-attempt error was discarded.
# Only unambiguous cookie-store problems belong here: a marker that could also
# describe a download failure would send a genuine error down this path.
COOKIE_SOURCE_ERROR_MARKERS = (
    "could not find",       # browser not installed / no profile
    "could not copy",       # cookie DB locked by a running browser
    "failed to decrypt",    # Chromium app-bound encryption (DPAPI)
    "unsupported platform",
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Browsers whose cookie store this process has already found unreadable.
_UNREADABLE_COOKIE_SOURCES = set()


class YouTubeDlpAuthError(RuntimeError):
    pass


def extract_info_with_retries(yt_dlp, url, base_opts, download=False,
                              status_callback=None, status_prefix="youtube"):
    """Run yt-dlp, retrying auth/bot blocks with configured browser cookies."""
    attempts = _option_attempts(base_opts)
    failures = []
    auth_block_seen = False
    for index, (label, opts) in enumerate(attempts):
        if index > 0:
            _status(status_callback, f"{status_prefix}: retrying with {label}")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=download)
        except Exception as exc:
            local_cookie_fault = (
                _uses_cookies(opts) and _matches_cookie_source_error(exc))
            failures.append((label, exc, local_cookie_fault))
            # Every attempt's error is reported as it happens. Only the last one
            # used to survive, which meant the failure that started the retry
            # cascade never reached the log and could not be diagnosed.
            if local_cookie_fault:
                _remember_unreadable(label)
                _status(
                    status_callback,
                    f"{status_prefix}: {label} unreadable ({one_line(exc)})")
                continue
            _status(
                status_callback,
                f"{status_prefix}: {label} failed ({one_line(exc)})")
            if _looks_like_auth_block(exc):
                auth_block_seen = True
                continue
            if auth_block_seen and index > 0:
                continue
            else:
                break
    if not failures:
        raise RuntimeError("yt-dlp produced no attempts")
    root_exc = _root_failure(failures)
    message = one_line(root_exc)
    if auth_block_seen or _looks_like_auth_block(message):
        hint = (
            "YouTube would not serve this link unauthenticated. The reliable "
            "fix on Windows is an exported cookies.txt: sign in to YouTube in "
            "your browser, export cookies for youtube.com, and point "
            "AVATAR_YOUTUBE_COOKIES at the file. Reading cookies straight out "
            "of a browser (AVATAR_YOUTUBE_COOKIES_FROM_BROWSER=edge) is tried "
            "first but often cannot work: current Edge/Chrome encrypt their "
            "cookie store app-bound, and a running Chrome holds it locked."
        )
        unreadable = [
            label for label, exc, local_fault in failures
            if local_fault and exc is not root_exc
        ]
        if unreadable:
            hint += (
                " Browser cookies were unreadable on this machine: "
                + ", ".join(unreadable) + "."
            )
        raise YouTubeDlpAuthError(f"{message} ({hint})")
    raise root_exc


def _root_failure(failures):
    """The error that actually caused the run to fail.

    A browser whose cookie jar cannot be opened is a local dead end, not the
    reason YouTube said no, so it is skipped in favour of the first real error.
    """
    for _label, exc, local_fault in failures:
        if not local_fault:
            return exc
    return failures[0][1]


def _uses_cookies(opts):
    opts = opts or {}
    return bool(opts.get("cookiesfrombrowser") or opts.get("cookiefile"))


def _matches_cookie_source_error(exc):
    text = str(exc).lower()
    return any(marker in text for marker in COOKIE_SOURCE_ERROR_MARKERS)


def one_line(value):
    """Flatten a yt-dlp error into a single readable log line.

    yt-dlp writes colour escapes and doubles its own "ERROR:" prefix, which is
    what made the status log unreadable.
    """
    text = _ANSI_RE.sub("", str(value if value is not None else ""))
    text = " ".join(text.split())
    while text.upper().startswith("ERROR: "):
        text = text[len("ERROR: "):]
    return text or "unknown yt-dlp error"


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
        if browser in _UNREADABLE_COOKIE_SOURCES:
            continue
        opts = dict(base)
        opts["cookiesfrombrowser"] = (browser,)
        attempts.append((f"{browser} cookies", opts))
    return attempts


def _remember_unreadable(label):
    """Stop paying for a cookie store this machine has already refused.

    An uninstalled browser or a locked cookie database does not fix itself
    mid-run, so retrying it on every download only added latency to a failure
    that was already decided.
    """
    browser = str(label or "").split()[0].strip().lower()
    if browser:
        _UNREADABLE_COOKIE_SOURCES.add(browser)


def reset_unreadable_cookie_sources():
    """Forget the per-process skip list (used by tests and after a re-login)."""
    _UNREADABLE_COOKIE_SOURCES.clear()


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
