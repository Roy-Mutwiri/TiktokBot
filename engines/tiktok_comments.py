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

ROOM_METRICS_INTERVAL = max(
    0.05, float(os.environ.get("AVATAR_TIKTOK_METRICS_INTERVAL", "0.1")))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class TikTokComments:
    """Background TikTok-Live comment reader -> on_comment(user, text)."""

    def __init__(self, username, on_comment, on_gift=None, on_follow=None,
                 on_like=None, on_share=None, on_viewers=None):
        u = (username or "").strip()
        self.unique_id = u.lstrip("@")
        self.username = "@" + self.unique_id
        self.on_comment = on_comment
        self.on_gift = on_gift          # (user, gift_name, count, coins)
        self.on_follow = on_follow      # (user)
        self.on_like = on_like          # (user, total_likes)
        self.on_share = on_share        # (user)
        self.on_viewers = on_viewers    # (concurrent_viewers)
        self.connected = False
        self.status = "idle"
        self._stop = False
        self._thread = None
        self._client = None
        self._metrics_task = None
        self._recent_social = {}

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

    def _social_once(self, kind, event):
        """Dispatch once when both custom and raw social listeners receive an event."""
        user = self._name(event)
        now = time.monotonic()
        key = (kind, user)
        if now - self._recent_social.get(key, 0.0) < 2.0:
            return
        self._recent_social[key] = now
        if kind == "follow" and self.on_follow is not None:
            print(f"[TIKTOK] follow event: {user}")
            self.on_follow(user)
        elif kind == "share" and self.on_share is not None:
            print(f"[TIKTOK] share event: {user}")
            self.on_share(user)

    @staticmethod
    def _room_metrics(info):
        """Extract concurrent viewers and total likes from TikTok room info."""
        info = info or {}
        stats = info.get("stats") or {}
        viewers = info.get("user_count")
        if viewers is None:
            viewers = stats.get("user_count")
        likes = info.get("like_count")
        if not likes:
            likes = stats.get("like_count") or stats.get("digg_count")
        return max(0, int(viewers or 0)), max(0, int(likes or 0))

    def _publish_room_metrics(self, info):
        viewers, likes = self._room_metrics(info)
        if self.on_viewers is not None:
            self.on_viewers(viewers)
        if self.on_like is not None:
            self.on_like("", likes)

    async def _poll_room_metrics(self, client):
        """Refresh counters when TikTok omits room/like websocket events."""
        while not self._stop and self.connected:
            try:
                info = await client.web.fetch_room_info(room_id=client.room_id)
                self._publish_room_metrics(info)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[TIKTOK] room metrics refresh failed: {str(exc)[:100]}")
            # The request itself usually takes about one second. Keep only a
            # small yield here so there is no additional artificial delay.
            await asyncio.sleep(ROOM_METRICS_INTERVAL)

    def _connect_once(self):
        from TikTokLive import TikTokLiveClient
        from TikTokLive.events import (ConnectEvent, DisconnectEvent, CommentEvent,
                                       GiftEvent, FollowEvent, LikeEvent, ShareEvent,
                                       SocialEvent, RoomUserSeqEvent)
        client = TikTokLiveClient(unique_id=self.unique_id)
        self._client = client
        self.status = "connecting"

        @client.on(ConnectEvent)
        async def _on_connect(event):
            self.connected = True
            self.status = "live"
            try:
                info = await client.web.fetch_room_info(room_id=client.room_id)
                self._publish_room_metrics(info)
            except Exception as exc:
                print(f"[TIKTOK] initial room metrics failed: {str(exc)[:100]}")
            self._metrics_task = asyncio.create_task(
                self._poll_room_metrics(client))
            print(f"[TIKTOK] connected to {self.username} — reading comments + gifts.")

        @client.on(DisconnectEvent)
        async def _on_disconnect(event):
            self.connected = False
            self.status = "disconnected"
            if self._metrics_task is not None:
                self._metrics_task.cancel()
                self._metrics_task = None
            if self.on_viewers is not None:
                self.on_viewers(0)

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
                    self._social_once("follow", event)
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

        if self.on_viewers is not None:
            @client.on(RoomUserSeqEvent)
            async def _on_room_users(event):
                try:
                    # m_total is TikTok's current room population. total_user is
                    # cumulative visitors and must not be shown as concurrent viewers.
                    viewers = int(getattr(event, "m_total", 0) or 0)
                    self.on_viewers(max(0, viewers))
                except Exception:
                    pass

        if self.on_share is not None:
            @client.on(ShareEvent)
            async def _on_share(event):
                try:
                    self._social_once("share", event)
                except Exception:
                    pass

        @client.on(SocialEvent)
        async def _on_social(event):
            """Fallback when a raw social event is not promoted to a custom type."""
            try:
                display = getattr(getattr(event, "base_message", None),
                                  "display_text", None)
                key = str(getattr(display, "key", "") or "").lower()
                pattern = str(getattr(display, "default_pattern", "") or "").lower()
                marker = key + " " + pattern
                print(f"[TIKTOK] raw social event: {marker[:120]}")
                if "follow" in marker:
                    self._social_once("follow", event)
                elif "share" in marker:
                    self._social_once("share", event)
            except Exception:
                pass

        client.run()      # blocks this thread's loop until disconnected

    def stop(self):
        self._stop = True
        self.connected = False
        self.status = "stopped"
        if self._metrics_task is not None:
            try:
                self._metrics_task.cancel()
            except Exception:
                pass
            self._metrics_task = None
        if self.on_viewers is not None:
            try:
                self.on_viewers(0)
            except Exception:
                pass
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
