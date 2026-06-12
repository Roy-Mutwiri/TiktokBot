# =============================================================================
# engines/tiktok_comments.py — read LIVE comments from a TikTok live stream.
#
# Wraps TikTokLive (connects to a public live by @handle, no login) and pushes
# each viewer comment to a callback on a background thread. Reconnects on drop.
# Never crashes the studio — connection errors are logged, not raised.
#
#   tc = TikTokComments("@yourhandle", on_comment=lambda user, text: ...)
#   tc.start()      # connect + stream comments to the callback
#   tc.stop()
# =============================================================================
import os
import sys
import threading
import asyncio
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class TikTokComments:
    """Background TikTok-Live comment reader -> on_comment(user, text)."""

    def __init__(self, username, on_comment):
        u = (username or "").strip()
        self.username = u if u.startswith("@") else "@" + u
        self.on_comment = on_comment
        self.connected = False
        self.status = "idle"
        self._stop = False
        self._thread = None
        self._client = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        # own asyncio loop on this thread; reconnect with backoff while enabled
        asyncio.set_event_loop(asyncio.new_event_loop())
        backoff = 5
        while not self._stop:
            try:
                self._connect_once()
                backoff = 5
            except Exception as exc:
                self.connected = False
                self.status = f"offline ({str(exc)[:40]})"
                print(f"[TIKTOK] {self.username} not live / error: {str(exc)[:80]}")
            if self._stop:
                break
            for _ in range(backoff):                 # wait before retry (host may be offline)
                if self._stop:
                    break
                time.sleep(1)
            backoff = min(30, backoff + 5)

    def _connect_once(self):
        from TikTokLive import TikTokLiveClient
        from TikTokLive.events import ConnectEvent, DisconnectEvent, CommentEvent
        client = TikTokLiveClient(unique_id=self.username)
        self._client = client
        self.status = "connecting"

        @client.on(ConnectEvent)
        async def _on_connect(event):
            self.connected = True
            self.status = "live"
            print(f"[TIKTOK] connected to {self.username} — reading comments.")

        @client.on(DisconnectEvent)
        async def _on_disconnect(event):
            self.connected = False
            self.status = "disconnected"

        @client.on(CommentEvent)
        async def _on_comment(event):
            try:
                user = getattr(event.user, "nickname", None) or getattr(event.user, "unique_id", "viewer")
                text = event.comment or ""
                if text.strip():
                    self.on_comment(str(user), str(text))
            except Exception:
                pass

        client.run()      # blocks this thread's loop until disconnected

    def stop(self):
        self._stop = True
        self.connected = False
        try:
            if self._client is not None:
                self._client.disconnect()
        except Exception:
            pass
