# =============================================================================
# engines/tts_stream_engine.py
# -----------------------------------------------------------------------------
# Streaming TTS (edge-tts) that drives the avatar's speech.
#
# CRITICAL TIMING — the AI voice and the synced mouth must start together:
#   1. on dequeue, synthesize the FULL line to PCM first (~300-500 ms), buffer it
#   2. then SIMULTANEOUSLY
#        a) start audio playback (to the default device -> OBS Desktop Audio /
#           a virtual audio cable)
#        b) feed the SAME audio to the mouth engine in real-time-paced chunks
#   3. when audio ends, stop feeding; the mouth engine goes silent on its own
#      after its silence timeout.
#
# The "mouth engine" is anything exposing feed_audio(chunk) + is_speaking — the
# MuseTalkEngine in the realtime avatar, or the Wav2LipEngine in the autonomous
# avatar. The optional behaviour engine (autonomous mode) is notified of speech
# boundaries and keyword reactions; it may be None.
#
# edge-tts streams MP3, decoded to 16 kHz mono float PCM via ffmpeg.
# =============================================================================

import os
import re
import sys
import glob
import hashlib
import shutil
import asyncio
import threading
import tempfile
import subprocess
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np


def _resolve_ffmpeg():
    """Find ffmpeg on PATH, else in the winget install location. Returns a path
    (or just 'ffmpeg' as a last resort). edge-tts audio can't decode without it.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA", "")
    winget = os.path.join(local, "Microsoft", "WinGet", "Packages")
    patterns = [
        os.path.join(winget, "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe"),
        os.path.join(winget, "*FFmpeg*", "*", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    try:
        import imageio_ffmpeg
        found = imageio_ffmpeg.get_ffmpeg_exe()
        if found and os.path.exists(found):
            print(f"[TTS] using bundled ffmpeg at {found}")
            return found
    except Exception:
        pass
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            found = hits[0]
            # add its dir to PATH so child processes resolve it too
            os.environ["PATH"] = os.path.dirname(found) + os.pathsep + os.environ.get("PATH", "")
            print(f"[TTS] using ffmpeg at {found}")
            return found
    print("[TTS] ffmpeg NOT found — AI voice audio cannot be decoded. "
          "Install: winget install Gyan.FFmpeg")
    return "ffmpeg"


FFMPEG = _resolve_ffmpeg()

try:
    import edge_tts
except ImportError:
    print("[TTS] edge-tts not installed. Run:  pip install edge-tts")
    raise

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import winsound
    HAVE_WINSOUND = True
except ImportError:
    HAVE_WINSOUND = False

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
VOICE = "en-US-ChristopherNeural"      # deep confident male (default)
SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000
FEED_CHUNK_SAMPLES = 640               # 40 ms at 16 kHz — one video frame's worth
TAIL_SILENCE = float(os.environ.get("AVATAR_TTS_TAIL_SILENCE", "0.10"))
# The on-screen mouth lags the audio by the render+display pipeline latency, so we
# DELAY audio playback by this to line the voice up with the visible mouth. The
# pipeline is heavier with after-mouth CodeFormer, so default ~150ms. Tune via
# AVATAR_MOUTH_SYNC_DELAY: raise if the mouth still LAGS the voice, lower if it LEADS.
MOUTH_SYNC_DELAY = float(os.environ.get("AVATAR_MOUTH_SYNC_DELAY", "0.15"))

# Male edge-tts voices offered in the control GUI dropdown.
MALE_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-EricNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
]

# --- Voice backend -----------------------------------------------------------
# "auto"   -> use Kokoro (local HuggingFace neural TTS) when it loads, else edge
# "kokoro" -> force Kokoro (no fallback to edge)
# "edge"   -> force edge-tts (the original cloud voice)
# Kokoro (hexgrad/Kokoro-82M) sounds more conversational/human than edge's
# newsreader cadence, and runs fully local so there's no cloud round-trip.
# Default = chatterbox: the male English-in-Arabic-accent cloned voice (it clones
# voice_refs/arabic_accent.wav). Set AVATAR_TTS=kokoro/edge/maya1/multilingual to
# change. The studio's voice dropdown also defaults to this.
TTS_BACKEND = os.environ.get("AVATAR_TTS", "kokoro").lower()
KOKORO_VOICE = os.environ.get("AVATAR_KOKORO_VOICE", "am_michael")  # natural US male
KOKORO_LANG = os.environ.get("AVATAR_KOKORO_LANG", "a")             # 'a' = American Eng
KOKORO_SPEED = float(os.environ.get("AVATAR_KOKORO_SPEED", "1.0"))
KOKORO_SR = 24000                                                  # Kokoro native rate

# Disk cache for synthesized lines (16 kHz mono). Lets the heavy expressive
# backends be pre-rendered once so live playback never stalls the GPU. Disable
# with AVATAR_TTS_CACHE=0.
TTS_CACHE = os.environ.get("AVATAR_TTS_CACHE", "1") == "1"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tts_cache")
CACHE_VERSION = "fluent-v2"


class TTSStreamEngine:
    """Speak queued text, play it, and feed the mouth engine in lockstep."""

    def __init__(self, mouth_engine, behavior=None):
        """Start the async worker thread.

        mouth_engine : object with feed_audio(chunk) + is_speaking (MuseTalk/Wav2Lip)
        behavior     : optional behaviour engine (autonomous mode) or None
        """
        self.mouth = mouth_engine
        self.behavior = behavior
        self.voice = VOICE
        self.muted = False
        self.words_spoken = 0
        self.lines_spoken = 0
        self._running = True
        self._loop = None
        self.speech_queue = None
        self._ready = threading.Event()
        # Local HF backends — all lazily loaded, with edge-tts as the final
        # fallback. Kokoro = fast natural; Maya1 = expressive (laughs/emotion);
        # Chatterbox = clone a real voice.
        self._kokoro = None
        self._kokoro_tried = False
        self._maya1 = None
        self._maya1_tried = False
        self._chatter = None
        self._chatter_tried = False
        self._mltts = None               # Chatterbox Multilingual (Arabic+English+)
        self._mltts_tried = False
        self._xtts = None                # Coqui XTTS-v2 (Arabic + English clone)
        self._xtts_tried = False
        self._eleven = None              # ElevenLabs cloud voice (high quality)
        self._eleven_tried = False
        self._hifi = {}                  # feed-pcm hash -> (hi-fi 24k pcm, sr) for playback
        self._prepared_audio = {}        # chunk text -> PCM, ready for instant playback
        self._prepared_lock = threading.Lock()
        self._interrupt_serial = 0
        self._audio_stream = None
        self._audio_lock = threading.Lock()
        self._playback_match_persona = None
        try:
            self._resample(np.zeros(32, np.float32), SAMPLE_RATE, PLAYBACK_RATE)
        except Exception:
            pass
        self.backend = "edge-tts"
        # Active backend (switchable at runtime via set_backend) — starts from
        # the AVATAR_TTS env default.
        self.tts_backend = TTS_BACKEND
        self.synthesizing = False        # True while a line is being generated
        self.speaking = False            # True while the bot is ACTUALLY talking
        self.audio_level = 0.0           # current playback RMS for UI meters
        # Serializes model loading so a SPEAK fired during a (slow) warm waits
        # for the model instead of racing in, seeing it half-loaded, and silently
        # falling back to a different voice.
        self._load_lock = threading.Lock()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Validate TTS can run; resolve the voice backend. Returns (ok, message)."""
        # Decide/load the backend now so the banner is honest and the (heavy)
        # model load happens here, not on the first SPEAK.
        if self.tts_backend == "elevenlabs" and self._get_eleven() is not None:
            self.backend = self._eleven.startup_check()[1]
        elif self.tts_backend == "elevenlabs" and self._get_xtts() is not None:
            self.backend = "ElevenLabs unavailable — using local XTTS. " \
                + self._xtts.startup_check()[1]
        elif self.tts_backend == "xtts" and self._get_xtts() is not None:
            self.backend = self._xtts.startup_check()[1]
        elif self.tts_backend == "multilingual" and self._get_mltts() is not None:
            self.backend = self._mltts.startup_check()[1]
        elif self.tts_backend == "maya1" and self._get_maya1() is not None:
            self.backend = "Maya1 (expressive — laughs/emotion tags)"
        elif self.tts_backend == "chatterbox" and self._get_chatterbox() is not None:
            self.backend = self._chatter.startup_check()[1]
        elif self.tts_backend in ("auto", "kokoro", "maya1", "chatterbox", "multilingual", "xtts") and \
                self._get_kokoro() is not None:
            extra = " — heavy backend unavailable, using Kokoro" \
                if self.tts_backend in ("maya1", "chatterbox", "multilingual", "xtts") else ""
            self.backend = f"Kokoro/{KOKORO_VOICE}{extra}"
        else:
            self.backend = f"edge-tts/{self.voice}"
        playback = "" if HAVE_WINSOUND else " (no winsound — playback disabled)"
        return True, f"{self.backend}, streaming + synced mouth feed{playback}."

    # -------------------------------------------------------------------------
    def _get_kokoro(self):
        """Lazily build the Kokoro pipeline once. Returns it, or None if Kokoro
        isn't installed/loadable (caller falls back to edge-tts)."""
        with self._load_lock:
            if self._kokoro_tried:
                return self._kokoro
            if self.tts_backend == "edge":
                return None
            try:
                from kokoro import KPipeline
                self._kokoro = KPipeline(lang_code=KOKORO_LANG)
                # Warm one synth so the first real line doesn't pay cuDNN autotune.
                try:
                    for _ in self._kokoro("warming up the voice now",
                                          voice=KOKORO_VOICE, speed=KOKORO_SPEED):
                        pass
                except Exception:
                    pass
                print(f"[TTS] Kokoro ready (voice {KOKORO_VOICE}, lang '{KOKORO_LANG}').")
            except Exception as exc:
                self._kokoro = None
                print(f"[TTS] Kokoro unavailable ({exc}) — using edge-tts.")
            self._kokoro_tried = True
            return self._kokoro

    def _get_maya1(self):
        """Lazily load the Maya1 expressive backend (heavy; ~7GB, ~15-30s).
        Returns it or None. The lock + tried-flag-after-load mean a SPEAK fired
        mid-load WAITS here for the model rather than falling back to Kokoro."""
        with self._load_lock:
            if self._maya1_tried:
                return self._maya1
            try:
                from maya1_tts import Maya1TTS
                eng = Maya1TTS()
                self._maya1 = eng if eng.ok else None
            except Exception as exc:
                self._maya1 = None
                print(f"[TTS] Maya1 unavailable ({exc}).")
            self._maya1_tried = True
            return self._maya1

    def _get_chatterbox(self):
        """Lazily load the Chatterbox voice-clone backend. Returns it or None.
        Lock + tried-after-load: a SPEAK during the warm waits for the model."""
        with self._load_lock:
            if self._chatter_tried:
                return self._chatter
            try:
                from chatterbox_tts import ChatterboxTTSBackend
                eng = ChatterboxTTSBackend()
                self._chatter = eng if eng.ok else None
            except Exception as exc:
                self._chatter = None
                print(f"[TTS] Chatterbox unavailable ({exc}).")
            self._chatter_tried = True
            return self._chatter

    def _get_mltts(self):
        """Lazily load the Chatterbox Multilingual backend (Arabic+English+21).
        Returns it or None. Lock + tried-after-load so a SPEAK mid-warm waits."""
        with self._load_lock:
            if self._mltts_tried:
                return self._mltts
            try:
                from multilingual_tts import MultilingualTTSBackend
                eng = MultilingualTTSBackend()
                self._mltts = eng if eng.ok else None
            except Exception as exc:
                self._mltts = None
                print(f"[TTS] Multilingual TTS unavailable ({exc}).")
            self._mltts_tried = True
            return self._mltts

    def _get_xtts(self):
        """Lazily load Coqui XTTS-v2 (Arabic + English voice clone). Lock +
        tried-after-load so a SPEAK fired mid-warm waits for the model."""
        with self._load_lock:
            if self._xtts_tried:
                return self._xtts
            try:
                from xtts_tts import XTTSBackend
                eng = XTTSBackend()
                self._xtts = eng if eng.ok else None
            except Exception as exc:
                self._xtts = None
                print(f"[TTS] XTTS unavailable ({exc}).")
            self._xtts_tried = True
            return self._xtts

    def _get_eleven(self):
        """Lazily init the ElevenLabs cloud backend (cheap auth check, no model load)."""
        with self._load_lock:
            if self._eleven_tried:
                return self._eleven
            try:
                from elevenlabs_tts import ElevenLabsBackend
                eng = ElevenLabsBackend()
                self._eleven = eng if eng.ok else None
                if eng.ok:
                    print("[TTS] " + eng.startup_check()[1])
                else:
                    print(f"[TTS] ElevenLabs off: {eng.last_error} — using local voice.")
            except Exception as exc:
                self._eleven = None
                print(f"[TTS] ElevenLabs unavailable ({exc}).")
            self._eleven_tried = True
            return self._eleven

    def _synth_eleven(self, text):
        """ElevenLabs synth -> 16 kHz mono float32 PCM (None on failure -> fallback)."""
        wav, sr = self._eleven.synthesize(text)
        if wav is None or len(wav) == 0:
            return None
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    # -------------------------------------------------------------------------
    def _run_loop(self):
        """Background thread that owns the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.speech_queue = asyncio.Queue()
        self._loop.create_task(self._stream_worker())
        self._ready.set()
        self._loop.run_forever()

    # -------------------------------------------------------------------------
    @property
    def pending(self):
        """Lines IN FLIGHT (queued + currently synthesizing/speaking). Lets the
        auto-talk brain pipeline: generate the NEXT line while this one plays,
        without racing ahead unboundedly."""
        try:
            q = self.speech_queue.qsize() if self.speech_queue is not None else 0
        except Exception:
            q = 0
        return q + (1 if (self.synthesizing or self.speaking) else 0)

    def clear_pending(self, below=999):
        """Drop QUEUED (not-yet-started) lines whose priority is < `below`, so a more
        important line plays next instead of behind a backlog. Items with priority
        >= `below` are KEPT (re-queued in order). The currently-playing line always
        finishes. Priorities: 0 = filler commentary, 1 = comment answer, 2 = live
        appreciation (follow/gift). So a comment clears filler (below=1) but never a
        queued thank-you; a thank-you clears both (below=2) so it lands immediately."""
        if self._loop is None or self.speech_queue is None:
            return

        async def _drain():
            kept = []
            try:
                while not self.speech_queue.empty():
                    item = self.speech_queue.get_nowait()
                    prio = item[0] if isinstance(item, tuple) else 0
                    if item is not None and prio >= below:
                        kept.append(item)
            except Exception:
                pass
            for it in kept:                      # put the survivors back, in order
                try:
                    self.speech_queue.put_nowait(it)
                except Exception:
                    pass
        try:
            asyncio.run_coroutine_threadsafe(_drain(), self._loop)
        except Exception:
            pass

    def interrupt_current(self):
        """Stop the active line immediately so urgent speech can take over."""
        self._interrupt_serial += 1
        stream = self._audio_stream
        self._audio_stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if HAVE_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        try:
            self.mouth.end_audio()
        except Exception:
            pass
        self.synthesizing = False
        self._set_speaking(False)

    def speak(self, text, priority=0):
        """Queue text to be spoken (priority: 0 filler, 1 comment, 2 appreciation)."""
        text = (text or "").strip()
        if not text:
            return False
        if self.behavior is not None:
            try:
                self.behavior.scan_text_for_reactions(text)
            except Exception:
                pass
        self.words_spoken += len(text.split())
        self.lines_spoken += 1
        if self._loop is None or self.speech_queue is None or not self._running:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.speech_queue.put((int(priority), text)), self._loop)
            future.result(timeout=2)
            print(f"[TTS] queued priority={int(priority)}: {text[:80]}")
            return True
        except Exception as exc:
            print(f"[TTS] queue failed: {exc}")
            return False

    def set_muted(self, muted):
        """Mute/unmute audio output (mouth still moves when muted)."""
        self.muted = bool(muted)
        if not self.muted:
            return
        # Discard audio already queued in the device. The playback worker checks
        # `muted` per block and exits without cancelling the mouth timeline.
        stream = self._audio_stream
        self._audio_stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if HAVE_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        self.audio_level = 0.0

    def set_voice(self, voice_name):
        """Switch the edge-tts voice for subsequent lines."""
        if voice_name:
            self.voice = voice_name
            print(f"[TTS] voice -> {self.voice}")

    def set_backend(self, name):
        """Switch the voice backend at runtime (maya1 / chatterbox / kokoro /
        edge / auto). Returns (ok, message). The heavy model loads lazily on the
        next synth; call warm_backend() first to pay that cost up front."""
        name = (name or "").strip().lower()
        valid = ("xtts", "elevenlabs", "eleven", "maya1", "chatterbox", "multilingual",
                 "kokoro", "edge", "auto")
        if name == "eleven":
            name = "elevenlabs"
        if name not in valid:
            return False, f"unknown backend {name!r}"
        self.tts_backend = name
        print(f"[TTS] backend -> {name}")
        return True, name

    def set_playback_voice_match(self, persona=None):
        """Temporarily color TTS playback toward the active YouTube voice persona."""
        self._playback_match_persona = (persona or "").strip().lower() or None

    def warm_backend(self):
        """Load the current backend's model now (blocking) so the first SPEAK
        doesn't stall. Safe to call from any thread. Returns the banner string."""
        return self.startup_check()[1]

    # -------------------------------------------------------------------------
    # SYNTHESIS CACHE  (pre-render heavy voices so live playback never stalls)
    # -------------------------------------------------------------------------
    def _cache_key(self, text):
        """Stable filename for (backend, voice/identity, text)."""
        ident = f"{CACHE_VERSION}|{self.tts_backend}|chunk={self._TTS_CHUNK_CHARS}"
        if self.tts_backend == "maya1":
            ident += "|" + os.environ.get("AVATAR_MAYA_DESC", "default")
        elif self.tts_backend == "chatterbox":
            ident += "|" + os.environ.get("AVATAR_CLONE_REF", "default")
        elif self.tts_backend == "multilingual":
            ident += "|" + os.environ.get("AVATAR_CLONE_REF", "default") \
                + "|" + os.environ.get("AVATAR_TTS_LANG", "auto")
        elif self.tts_backend in ("auto", "kokoro"):
            ident += "|" + KOKORO_VOICE
        elif self.tts_backend == "xtts":
            # key on the CLONE REFERENCE so swapping the voice invalidates old cache
            try:
                import xtts_tts
                ident += "|" + ",".join(os.path.basename(p) for p in (xtts_tts.XTTS_REFS or []))
            except Exception:
                ident += "|xtts"
        elif self.tts_backend == "elevenlabs":
            ident += "|eleven|" + os.environ.get("AVATAR_ELEVEN_VOICE", "default")
        else:
            ident += "|" + self.voice
        h = hashlib.sha1(f"{ident}|{text}".encode("utf-8")).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{h}.wav")

    def _cache_load(self, text):
        """Return cached 16 kHz PCM for text, or None."""
        try:
            with self._prepared_lock:
                prepared = self._prepared_audio.get(text)
            if prepared is not None:
                return prepared
        except Exception:
            pass

        if not TTS_CACHE or sf is None:
            return None
        path = self._cache_key(text)
        if not os.path.exists(path):
            return None
        try:
            data, _ = sf.read(path, dtype="float32")
            data = np.asarray(data, dtype=np.float32).flatten()
            if not len(data) or len(data) > SAMPLE_RATE * 45:
                return None
            if not np.all(np.isfinite(data)):
                return None
            with self._prepared_lock:
                self._prepared_audio[text] = data
            return data
        except Exception:
            return None

    def _cache_save(self, text, pcm):
        try:
            with self._prepared_lock:
                self._prepared_audio[text] = np.asarray(pcm, dtype=np.float32)
                while len(self._prepared_audio) > 48:
                    self._prepared_audio.pop(next(iter(self._prepared_audio)))
        except Exception:
            pass
        if not TTS_CACHE or sf is None:
            return
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            sf.write(self._cache_key(text), np.asarray(pcm, dtype=np.float32),
                     SAMPLE_RATE, subtype="PCM_16")
        except Exception:
            pass

    def prepare(self, text, timeout=120):
        """Synthesize every playback chunk without speaking it.

        This returns only after the exact chunks used by ``speak`` are cached in
        memory, allowing the UI to signal that playback is genuinely ready.
        """
        chunks = self._split_for_tts(text)
        if not chunks or self._loop is None:
            return False

        async def _prepare_chunk(chunk):
            cached = self._cache_load(chunk)
            if cached is not None:
                return True
            loop = asyncio.get_event_loop()
            pcm = await loop.run_in_executor(None, self._synthesize_blocking, chunk)
            if pcm is None:
                pcm = await self._synthesize(chunk)
            return pcm is not None and len(pcm) > 0

        for chunk in chunks:
            fut = asyncio.run_coroutine_threadsafe(_prepare_chunk(chunk), self._loop)
            try:
                if not fut.result(timeout=timeout):
                    return False
            except Exception as exc:
                print(f"[TTS] prepare failed for {chunk[:30]!r}: {exc}")
                return False
        return True

    def is_prepared(self, text):
        """Return True when every playback chunk for this line is in memory."""
        chunks = self._split_for_tts(text)
        return bool(chunks) and all(self._cache_load(chunk) is not None for chunk in chunks)

    def prerender(self, texts, progress=None):
        """Synthesize + cache a list of lines NOW (blocking), so they later play
        with zero GPU work. Call at startup for known lines (greeting, demo set)
        when using a heavy backend, so its per-line synth cost is paid up front
        instead of freezing the live video mid-stream."""
        if self._loop is None:
            return
        chunks = {
            chunk
            for text in texts
            for chunk in self._split_for_tts((text or "").strip())
        }
        todo = [chunk for chunk in chunks if self._cache_load(chunk) is None]
        for i, text in enumerate(todo):
            if progress:
                progress(i + 1, len(todo), text)
            fut = asyncio.run_coroutine_threadsafe(self._synthesize(text), self._loop)
            try:
                fut.result(timeout=120)
            except Exception as exc:
                print(f"[TTS] prerender failed for {text[:30]!r}: {exc}")

    # -------------------------------------------------------------------------
    async def _stream_worker(self):
        """Pull a line, split it into short sentence chunks, and play them back to
        back — synthesizing the NEXT chunk while the CURRENT one plays. This (a)
        keeps XTTS off long paragraphs (which distort) and (b) starts speech after
        the FIRST short chunk (~1s) instead of after the whole line, so the avatar
        isn't silent while a long line synthesizes."""
        loop = asyncio.get_event_loop()
        while self._running:
            item = await self.speech_queue.get()
            if item is None or not self._running:        # shutdown sentinel
                break
            text = item[1] if isinstance(item, tuple) else item   # (priority, text)
            try:
                chunks = self._split_for_tts(text)
                if not chunks:
                    continue
                self.synthesizing = True
                # synth chunk 0, then pipeline: synth i+1 in a thread while i plays
                pcm = await loop.run_in_executor(None, self._synthesize_blocking, chunks[0])
                if pcm is None:                          # backend needs the async path
                    pcm = await self._synthesize(chunks[0])
                self._set_speaking(True)
                for idx in range(len(chunks)):
                    serial = self._interrupt_serial
                    nxt = None
                    if idx + 1 < len(chunks) and self._running:
                        nxt = loop.run_in_executor(
                            None, self._synthesize_blocking, chunks[idx + 1])
                    if pcm is not None and len(pcm) > 0:
                        self.synthesizing = False
                        await self._play_and_feed(
                            pcm, final_chunk=(idx == len(chunks) - 1)
                        )
                    if serial != self._interrupt_serial:
                        break
                    if nxt is not None:
                        self.synthesizing = True
                        pcm = await nxt
                        if pcm is None:
                            pcm = await self._synthesize(chunks[idx + 1])
                    else:
                        pcm = None
                self.synthesizing = False
                self._set_speaking(False)
            except Exception as exc:
                print(f"[TTS] speak error: {exc}")
                self.synthesizing = False
                self._set_speaking(False)

    # Max characters per TTS chunk. XTTS-v2 distorts / rushes on long inputs, so we
    # keep each synth call to roughly a sentence or two. Tunable.
    _TTS_CHUNK_CHARS = int(os.environ.get("AVATAR_TTS_CHUNK_CHARS", "150"))

    def _split_for_tts(self, text):
        """Split a line into sentence-ish chunks no longer than _TTS_CHUNK_CHARS, so
        the voice model never gets a long paragraph (the cause of long-speech
        distortion). Short sentences are grouped; an over-long sentence is hard-split
        on commas/spaces."""
        text = (text or "").strip()
        if not text:
            return []
        target = max(60, self._TTS_CHUNK_CHARS)
        parts = re.split(
            r"(\[\[CUT\]\])|(?<=[.!?…])\s+", text,
            flags=re.IGNORECASE
        )
        chunks, cur = [], ""
        for p in parts:
            if p is None:
                continue
            p = p.strip()
            if not p:
                continue
            if p.upper() == "[[CUT]]":
                if cur:
                    chunks.append(cur)
                    cur = ""
                continue
            while len(p) > target:                       # giant sentence -> hard split
                cut = p.rfind(",", 0, target)
                if cut < int(target * 0.5):
                    cut = p.rfind(" ", 0, target)
                if cut <= 0:
                    cut = target
                chunks.append(p[:cut].strip())
                p = p[cut:].strip()
            if not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= target:
                cur += " " + p
            else:
                chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return [c for c in chunks if c]

    def _synthesize_blocking(self, text):
        """SYNCHRONOUS synth for the pipeline executor (so the next chunk can render
        while the current one plays). Cache-aware; covers the sync backends. Returns
        None for the async-only path (edge) so the caller falls back to _synthesize."""
        try:
            cached = self._cache_load(text)
            if cached is not None:
                return cached
            pcm = None
            b = self.tts_backend
            if b == "elevenlabs" and self._get_eleven() is not None:
                pcm = self._synth_eleven(text)
                if pcm is None and self._get_xtts() is not None:
                    pcm = self._synth_xtts(text)         # cloud failed -> local XTTS
            elif b == "xtts" and self._get_xtts() is not None:
                pcm = self._synth_xtts(text)
            elif b == "multilingual" and self._get_mltts() is not None:
                pcm = self._synth_mltts(text)
            elif b == "maya1" and self._get_maya1() is not None:
                pcm = self._synth_maya1(text)
            elif b == "chatterbox" and self._get_chatterbox() is not None:
                pcm = self._synth_chatterbox(text)
            if (pcm is None or len(pcm) == 0) and self._get_kokoro() is not None:
                pcm = self._synth_kokoro(text)            # cheap sync fallback
            if pcm is not None and len(pcm) > 0:
                self._cache_save(text, pcm)
                return pcm
        except Exception as exc:
            print(f"[TTS] chunk synth failed ({exc}) — async fallback.")
        return None

    async def _synthesize(self, text):
        """Cache-aware synth. Heavy backends (Maya1 ~8s/line, Chatterbox ~3s)
        saturate the GPU and would freeze the live video while generating, so
        every result is cached to disk keyed by (backend, voice, text). Repeat
        lines — and anything pre-rendered at startup — then play instantly with
        no GPU work. Kokoro/edge are cheap but caching them is harmless."""
        cached = self._cache_load(text)
        if cached is not None:
            return cached
        pcm = await self._synthesize_uncached(text)
        if pcm is not None and len(pcm) > 0:
            self._cache_save(text, pcm)
        return pcm

    async def _synthesize_uncached(self, text):
        """Synthesize a line to 16 kHz mono float PCM, dispatching on the chosen
        backend with a graceful fallback chain. Returns an ndarray."""
        # ElevenLabs cloud voice — highest quality, natural Arabic + English.
        if self.tts_backend == "elevenlabs" and self._get_eleven() is not None:
            try:
                pcm = self._synth_eleven(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] ElevenLabs synth failed ({exc}) — falling back.")
        # XTTS-v2 (Coqui) — Arabic + English voice clone, auto language per line.
        if self.tts_backend in ("xtts", "elevenlabs") and self._get_xtts() is not None:
            try:
                pcm = self._synth_xtts(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] XTTS synth failed ({exc}) — falling back.")

        # Multilingual backend (Chatterbox ML) — Arabic + English + 21 more, with
        # automatic per-segment language routing for code-switched text.
        if self.tts_backend == "multilingual" and self._get_mltts() is not None:
            try:
                pcm = self._synth_mltts(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] Multilingual synth failed ({exc}) — falling back.")

        # Expressive backend (Maya1) — emotion tags like <laugh>, <sigh>.
        if self.tts_backend == "maya1" and self._get_maya1() is not None:
            try:
                pcm = self._synth_maya1(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] Maya1 synth failed ({exc}) — falling back.")

        # Voice-clone backend (Chatterbox).
        if self.tts_backend == "chatterbox" and self._get_chatterbox() is not None:
            try:
                pcm = self._synth_chatterbox(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] Chatterbox synth failed ({exc}) — falling back.")

        # Kokoro (fast natural) — default for auto/kokoro, and the fallback for
        # the heavy backends above so a line still gets spoken.
        if self.tts_backend in ("auto", "kokoro", "maya1", "chatterbox", "multilingual", "xtts") and \
                self._get_kokoro() is not None:
            try:
                pcm = self._synth_kokoro(text)
                if pcm is not None and len(pcm) > 0:
                    return pcm
            except Exception as exc:
                print(f"[TTS] Kokoro synth failed ({exc}) — falling back to edge.")
        return await self._synthesize_edge(text)

    def _synth_maya1(self, text):
        """Maya1 expressive synth -> 16 kHz mono float32 PCM."""
        wav = self._maya1.synthesize(text)
        if wav is None or len(wav) == 0:
            return None
        from maya1_tts import SAMPLE_RATE as MAYA_SR
        if MAYA_SR != SAMPLE_RATE:
            wav = self._resample(wav, MAYA_SR, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    def _synth_chatterbox(self, text):
        """Chatterbox cloned-voice synth -> 16 kHz mono float32 PCM."""
        wav, sr = self._chatter.synthesize(text)
        if wav is None or len(wav) == 0:
            return None
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    def _synth_mltts(self, text):
        """Chatterbox Multilingual synth (auto Arabic/English/...) -> 16 kHz PCM."""
        wav, sr = self._mltts.synthesize(text)
        if wav is None or len(wav) == 0:
            return None
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    def _synth_xtts(self, text):
        """Coqui XTTS-v2 synth (auto Arabic/English). Returns the 16 kHz feed PCM (for
        the mouth) but STASHES the full-quality 24 kHz audio so playback uses it —
        the listener hears XTTS's native fidelity, not a 16 kHz downsample."""
        wav, sr = self._xtts.synthesize(self._strip_emotion_tags(text))
        if wav is None or len(wav) == 0:
            return None
        wav = np.asarray(wav, dtype=np.float32)
        feed = self._resample(wav, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wav
        feed = np.asarray(feed, dtype=np.float32)
        if sr > SAMPLE_RATE:                      # keep the hi-fi version for playback
            try:
                key = hash(feed[:512].tobytes())
                self._hifi[key] = (wav, int(sr))
                if len(self._hifi) > 8:           # bound the stash
                    self._hifi.pop(next(iter(self._hifi)))
            except Exception:
                pass
        return feed

    # Maya1-style inline emotion tags like <laugh>, <sigh>. Only Maya1 performs
    # them; every other backend must strip them so they aren't read aloud.
    _EMOTION_TAG_RE = re.compile(r"<\s*[a-zA-Z][a-zA-Z _-]{0,18}\s*>")

    @classmethod
    def _strip_emotion_tags(cls, text):
        return cls._EMOTION_TAG_RE.sub(" ", text).replace("  ", " ").strip()

    def _synth_kokoro(self, text):
        """Run Kokoro on a full line, return 16 kHz mono float32 PCM."""
        text = self._strip_emotion_tags(text)
        pipe = self._kokoro
        chunks = []
        for _, _, audio in pipe(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(a.astype(np.float32).flatten())
        if not chunks:
            return None
        wav = np.concatenate(chunks).astype(np.float32)
        if KOKORO_SR != SAMPLE_RATE:               # 24k -> 16k for the mel/feed path
            feed = self._resample(wav, KOKORO_SR, SAMPLE_RATE)
            try:
                key = hash(feed[:512].tobytes())
                self._hifi[key] = (wav, KOKORO_SR)
                if len(self._hifi) > 8:
                    self._hifi.pop(next(iter(self._hifi)))
            except Exception:
                pass
            return np.asarray(feed, dtype=np.float32)
        return wav

    @staticmethod
    def _resample(wav, src_sr, dst_sr):
        """Resample mono float audio without first-call JIT initialization."""
        try:
            from scipy.signal import resample_poly
            divisor = np.gcd(int(src_sr), int(dst_sr))
            return resample_poly(
                wav, int(dst_sr) // divisor, int(src_sr) // divisor
            ).astype(np.float32)
        except Exception:
            n_dst = int(round(len(wav) * dst_sr / src_sr))
            if n_dst <= 1:
                return wav
            x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
            return np.interp(x_new, x_old, wav).astype(np.float32)

    async def _synthesize_edge(self, text):
        """Stream edge-tts MP3, decode to 16 kHz mono float PCM. Returns ndarray."""
        text = self._strip_emotion_tags(text)
        mp3 = bytearray()
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            return None
        return self._decode_mp3(bytes(mp3))

    def _decode_mp3(self, mp3_bytes):
        """Decode MP3 bytes to 16 kHz mono float32 via ffmpeg + soundfile."""
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_mp3.write(mp3_bytes); tmp_mp3.close(); tmp_wav.close()
        try:
            subprocess.run(
                [FFMPEG, "-y", "-loglevel", "error", "-i", tmp_mp3.name,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp_wav.name],
                check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if sf is not None:
                data, _ = sf.read(tmp_wav.name, dtype="float32")
            else:
                import wave
                with wave.open(tmp_wav.name, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return np.asarray(data, dtype=np.float32).flatten()
        except Exception as exc:
            print(f"[TTS] decode error ({exc}). Is ffmpeg on PATH?")
            return None
        finally:
            for p in (tmp_mp3.name, tmp_wav.name):
                try:
                    os.remove(p)
                except OSError:
                    pass

    @staticmethod
    def _condition_playback(pcm):
        """Sanitize and peak-limit speech without adding saturation."""
        audio = np.asarray(pcm, dtype=np.float32).flatten()
        if not len(audio):
            return audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio))) or 1.0
        if peak > 0.94:
            audio = audio * (0.94 / peak)
        return audio.astype(np.float32)

    def _match_youtube_playback_voice(self, audio, rate):
        """Light post-filter so acknowledgements sit closer to YouTube voice mode."""
        persona = self._playback_match_persona
        if not persona or os.environ.get("AVATAR_READY_MATCH_YOUTUBE", "1") != "1":
            return audio
        y = np.asarray(audio, dtype=np.float32).flatten()
        if not len(y):
            return y
        profiles = {
            "deep_male": (-4.0, 0.82, 0.90),
            "warm_male": (-2.0, 0.92, 0.96),
            "young_male": (1.0, 1.03, 1.02),
            "broadcast_male": (-1.0, 0.95, 0.92),
            "natural_woman": (3.0, 1.08, 1.03),
            "warm_woman": (2.0, 1.04, 1.00),
            "bright_woman": (4.0, 1.12, 1.06),
            "low_woman": (1.0, 1.00, 0.96),
        }
        semitones, formant, warmth = profiles.get(persona, profiles["deep_male"])
        try:
            import librosa
            if abs(semitones) >= 0.05:
                y = librosa.effects.pitch_shift(
                    y, sr=int(rate), n_steps=float(semitones), n_fft=2048)
                y = np.asarray(y, dtype=np.float32)
        except Exception:
            pass
        try:
            from scipy.signal import butter, sosfilt
            low = max(40.0, min(180.0, 70.0 * formant))
            high = max(9000.0, min(rate / 2 - 200.0, 14500.0 * formant))
            sos = butter(2, [low, high], btype="bandpass", fs=rate, output="sos")
            y = sosfilt(sos, y).astype(np.float32)
        except Exception:
            pass
        # Add the same warm, controlled center-speaker character as altered YouTube.
        warm = np.empty_like(y)
        lp = 0.0
        for i, sample in enumerate(y):
            lp = 0.985 * lp + 0.015 * float(sample)
            warm[i] = sample + 0.18 * warmth * lp
        y = np.tanh(warm * 1.08) / np.tanh(1.08)
        fade = min(len(y) // 4, max(1, int(rate * 0.08)))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y[:fade] *= ramp
            y[-fade:] *= ramp[::-1]
        peak = float(np.max(np.abs(y))) or 1.0
        if peak > 0.92:
            y = y * (0.92 / peak)
        return y.astype(np.float32)

    def _play_direct(self, pcm, rate):
        """Write speech through one persistent float stream."""
        import sounddevice as sd

        audio = self._condition_playback(pcm)
        if rate != PLAYBACK_RATE:
            audio = self._resample(audio, rate, PLAYBACK_RATE)
        audio = self._match_youtube_playback_voice(audio, PLAYBACK_RATE)
        try:
            with self._audio_lock:
                if self.muted:
                    return
                if self._audio_stream is None:
                    self._audio_stream = sd.OutputStream(
                        samplerate=PLAYBACK_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=960,
                        latency="low",
                    )
                    self._audio_stream.start()
                stream = self._audio_stream
                for start in range(0, len(audio), 960):
                    if self.muted or stream is not self._audio_stream:
                        break
                    block = audio[start:start + 960]
                    self.audio_level = float(
                        np.sqrt(np.mean(block * block)) + 1e-9)
                    stream.write(block.reshape(-1, 1))
        finally:
            self.audio_level = 0.0

    async def _play_and_feed(self, pcm, final_chunk=True):
        """Play audio and drive the mouth from the same monotonic clock.

        Modern mouth engines receive the full utterance once and choose their
        current mel window by elapsed time. Older/live streaming implementations
        retain the 40 ms paced feed fallback.

        Playback uses the stashed HI-FI (24 kHz) version when available so the
        listener hears full fidelity; the mouth is always fed the 16 kHz pcm."""
        serial = self._interrupt_serial
        # find the matching hi-fi (24k) buffer for playback; fall back to the 16k pcm
        play_pcm, play_sr = pcm, SAMPLE_RATE
        try:
            hi = self._hifi.pop(hash(pcm[:512].tobytes()), None)
            if hi is not None:
                play_pcm, play_sr = hi
        except Exception:
            pass

        # a) start audio playback AFTER MOUTH_SYNC_DELAY so the audible voice lines up
        #    with the on-screen mouth (delayed by the render pipeline).
        async def _delayed_play():
            if MOUTH_SYNC_DELAY > 0:
                await asyncio.sleep(MOUTH_SYNC_DELAY)
            if (serial == self._interrupt_serial and not self.muted
                    and self._running):
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        None, self._play_direct, play_pcm, play_sr
                    )
                except Exception as exc:
                    if not self.muted:
                        print(f"[TTS] direct playback failed ({exc}); using winsound.")
                    if (serial == self._interrupt_serial and not self.muted
                            and HAVE_WINSOUND):
                        wav = self._write_temp_wav(play_pcm, play_sr)
                        if wav:
                            try:
                                winsound.PlaySound(
                                    wav, winsound.SND_FILENAME | winsound.SND_SYNC
                                )
                            finally:
                                try:
                                    os.remove(wav)
                                except OSError:
                                    pass
        start = time.monotonic()
        clocked = callable(getattr(self.mouth, "start_audio", None))
        if clocked:
            try:
                self.mouth.start_audio(pcm, start_time=start)
            except Exception:
                clocked = False

        play_task = asyncio.ensure_future(_delayed_play())

        # b) Keep the utterance alive for its exact duration. Clocked engines read
        #    directly from the timeline; legacy engines receive paced 40 ms chunks.
        n = len(pcm)
        pos = 0
        idx = 0
        while pos < n and self._running and serial == self._interrupt_serial:
            if not clocked:
                try:
                    self.mouth.feed_audio(pcm[pos:pos + FEED_CHUNK_SAMPLES])
                except Exception:
                    pass
            pos += FEED_CHUNK_SAMPLES
            idx += 1
            target = start + idx * (FEED_CHUNK_SAMPLES / SAMPLE_RATE)
            await asyncio.sleep(max(0.0, target - time.monotonic()))

        await play_task
        if final_chunk and serial == self._interrupt_serial:
            await asyncio.sleep(TAIL_SILENCE)
        if clocked and final_chunk and serial == self._interrupt_serial:
            try:
                self.mouth.end_audio()
            except Exception:
                pass

    def _set_speaking(self, speaking):
        """Notify the optional behaviour engine of speech boundaries. Also exposes
        `self.speaking` — the authoritative 'the bot is ACTUALLY talking' flag
        (unpolluted by any idle silence keep-alive the studio may feed the mouth)."""
        self.speaking = bool(speaking)
        if not self.speaking:
            self.audio_level = 0.0
        if self.behavior is not None:
            try:
                self.behavior.set_speaking(speaking)
            except Exception:
                pass

    def _write_temp_wav(self, pcm, rate=SAMPLE_RATE):
        """Write float PCM to a temp mono wav for winsound (at `rate`). Returns path."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            data16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
            if sf is not None:
                sf.write(tmp.name, data16, int(rate), subtype="PCM_16")
            else:
                import wave
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(rate))
                    wf.writeframes(data16.tobytes())
            return tmp.name
        except Exception:
            return None

    # -------------------------------------------------------------------------
    def shutdown(self):
        """Stop the worker loop cleanly (wake the queue, then stop)."""
        self._running = False
        with self._audio_lock:
            try:
                if self._audio_stream is not None:
                    self._audio_stream.stop()
                    self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        if self._loop is not None and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.speech_queue.put(None), self._loop)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)


if __name__ == "__main__":
    class _Dummy:
        is_speaking = False
        def feed_audio(self, x): self.is_speaking = True
    eng = TTSStreamEngine(_Dummy())
    print("[TTS] startup_check:", eng.startup_check())
    eng.speak("Hello everyone, welcome to the stream.")
    time.sleep(6)
