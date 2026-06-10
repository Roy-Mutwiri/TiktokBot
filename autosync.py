# =============================================================================
# autosync.py — auto-commit + auto-push the whole project on any change
# -----------------------------------------------------------------------------
# Watches the working tree and, whenever files change (your edits, Claude's
# edits, anything), waits for a short quiet period (so it doesn't commit
# mid-save), then commits everything and pushes to the GitHub remote.
#
#   python autosync.py            # run in the foreground (Ctrl+C to stop)
#   start_autosync.bat            # run hidden in the background (Windows)
#
# Honors .gitignore (uses `git add -A`), so weights/videos/caches are never
# pushed. Push failures (offline / auth) are logged and retried next cycle.
# =============================================================================

import os
import sys
import time
import subprocess
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
POLL_SECONDS = 3.0          # how often to check for changes
QUIET_SECONDS = 6.0         # commit once the tree has been unchanged this long
PUSH_TIMEOUT = 120          # seconds before a hung push is abandoned
BRANCH = "main"
LOG_PATH = os.path.join(REPO_DIR, "autosync.log")


def _log(msg):
    """Print + append a timestamped line to autosync.log."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _git(args, timeout=30):
    """Run a git command in the repo. Returns (returncode, output)."""
    try:
        r = subprocess.run(["git"] + args, cwd=REPO_DIR, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="ignore")
        return r.returncode, (r.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:
        return 1, str(exc)


def _status():
    """Return the porcelain status string (empty == clean)."""
    _, out = _git(["status", "--porcelain"])
    return out


def _changed_summary():
    """Short human summary of changed paths for the commit message."""
    out = _status()
    files = [ln[3:] for ln in out.splitlines() if ln.strip()]
    n = len(files)
    sample = ", ".join(os.path.basename(f) for f in files[:4])
    if n > 4:
        sample += f", +{n - 4} more"
    return n, sample


def commit_and_push():
    """Stage everything, commit, and push. Returns True if it pushed."""
    n, sample = _changed_summary()
    if n == 0:
        return False

    code, _ = _git(["add", "-A"])
    if code != 0:
        _log("git add failed; skipping this cycle")
        return False

    # Nothing to commit after add (e.g. only ignored files changed)?
    rc, _ = _git(["diff", "--cached", "--quiet"])
    if rc == 0:
        return False

    msg = f"auto-sync: {n} file(s) — {sample}"
    code, out = _git(["commit", "-m", msg,
                      "-m", "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"])
    if code != 0:
        _log(f"commit failed: {out.splitlines()[-1] if out else '?'}")
        return False
    _log(f"committed: {msg}")

    # Push using the configured credential helper (Git Credential Manager reuses
    # the token you authenticated with earlier — no prompt once stored).
    code, out = _git(["push", "origin", BRANCH], timeout=PUSH_TIMEOUT)
    if code == 0:
        _log("pushed -> origin/" + BRANCH)
        return True

    # Remote moved ahead (edited on GitHub / another machine): pull --rebase, retry.
    if "rejected" in out or "fetch first" in out or "non-fast-forward" in out:
        _log("remote ahead — rebasing and retrying push")
        _git(["pull", "--rebase", "origin", BRANCH], timeout=PUSH_TIMEOUT)
        code, out = _git(["push", "origin", BRANCH], timeout=PUSH_TIMEOUT)
        if code == 0:
            _log("pushed after rebase -> origin/" + BRANCH)
            return True

    tail = out.splitlines()[-1] if out else "?"
    _log(f"push failed ({tail}) — will retry on next change")
    return False


def main():
    """Watch the tree and auto-commit+push after each quiet period."""
    if _git(["rev-parse", "--is-inside-work-tree"])[0] != 0:
        _log("not a git repo — aborting")
        return
    _log(f"autosync started — watching {REPO_DIR} (branch {BRANCH})")
    _log(f"poll={POLL_SECONDS}s quiet={QUIET_SECONDS}s. Ctrl+C to stop.")

    last_state = _status()
    last_change_t = 0.0          # 0 == tree clean / already synced
    try:
        while True:
            time.sleep(POLL_SECONDS)
            state = _status()

            if state != last_state:
                last_state = state
                last_change_t = time.monotonic() if state else 0.0
                continue

            # Stable + dirty for QUIET_SECONDS -> commit & push.
            if state and last_change_t and (time.monotonic() - last_change_t) >= QUIET_SECONDS:
                commit_and_push()
                last_state = _status()
                last_change_t = 0.0
    except KeyboardInterrupt:
        _log("autosync stopped (Ctrl+C)")


if __name__ == "__main__":
    main()
