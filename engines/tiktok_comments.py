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

    def __init__(self, username, on_comment, on_gift=None, on_follow=None,
                 on_like=None, on_share=None):
        u = (username or "").strip()
        self.username = u if u.startswith("@") else "@" + u
        self.on_comment = on_comment
        self.on_gift = on_gift          # (user, gift_name, count, coins)
        self.on_follow = on_follow      # (user)
        self.on_like = on_like          # (user, total_likes)
        self.on_share = on_share        # (user)
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

    @staticmethod
    def _name(event):
        u = getattr(event, "user", None)
        return str(getattr(u, "nickname", None) or getattr(u, "unique_id", "viewer"))

    def _connect_once(self):
        from TikTokLive import TikTokLiveClient
        from TikTokLive.events import (ConnectEvent, DisconnectEvent, CommentEvent,
                                       GiftEvent, FollowEvent, LikeEvent, ShareEvent)
        client = TikTokLiveClient(unique_id=self.username)
        self._client = client
        self.status = "connecting"

        @client.on(ConnectEvent)
        async def _on_connect(event):
            self.connected = True
            self.status = "live"
            print(f"[TIKTOK] connected to {self.username} — reading comments + gifts.")

        @client.on(DisconnectEvent)
        async def _on_disconnect(event):
            self.connected = False
            self.status = "disconnected"

        @client.on(CommentEvent)
        async def _on_comment(event):
            try:
                text = event.comment or ""
                if text.strip():
                    self.on_comment(self._name(event), str(text))
            except Exception:
                pass

        if self.on_gift is not None:
            @client.on(GiftEvent)
            async def _on_gift(event):
                try:
                    gift = getattr(event, "gift", None)
                    streakable = bool(getattr(gift, "streakable", False))
                    # for streak gifts, only fire once the streak ENDS (final count)
                    if streakable and getattr(event, "streaking", False):
                        return
                    name = getattr(gift, "name", "a gift")
                    count = int(getattr(event, "repeat_count", 1) or 1)
                    coins = int(getattr(gift, "diamond_count", 0) or 0) * count
                    self.on_gift(self._name(event), str(name), count, coins)
                except Exception:
                    pass

        if self.on_follow is not None:
            @client.on(FollowEvent)
            async def _on_follow(event):
                try:
                    self.on_follow(self._name(event))
                except Exception:
                    pass

        if self.on_like is not None:
            @client.on(LikeEvent)
            async def _on_like(event):
                try:
                    total = int(getattr(event, "total_likes_count", None)
                                or getattr(event, "total", 0) or 0)
                    self.on_like(self._name(event), total)
                except Exception:
                    pass

        if self.on_share is not None:
            @client.on(ShareEvent)
            async def _on_share(event):
                try:
                    self.on_share(self._name(event))
                except Exception:
                    pass

        client.run()      # blocks this thread's loop until disconnected

    def stop(self):
        self._stop = True
        self.connected = False
        self.status = "stopped"
        client = self._client
        if client is None:
            return
        # disconnect() is a COROUTINE running on the client's OWN event loop (the
        # one client.run() spun up on the reader thread). Calling it bare just
        # creates an un-awaited coroutine that never runs (the connection stays
        # open). Schedule it ONTO that loop, thread-safely, so it actually closes.
        loop = (getattr(client, "_asyncio_loop", None)
                or getattr(client, "_loop", None))
        try:
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(client.disconnect(), loop)
            else:
                asyncio.run(client.disconnect())
        except Exception:
            pass
