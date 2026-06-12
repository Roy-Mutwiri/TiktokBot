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
FEED_CHUNK_SAMPLES = 640               # 40 ms at 16 kHz — one video frame's worth
TAIL_SILENCE = 0.30                    # keep "speaking" this long after audio ends
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
TTS_BACKEND = os.environ.get("AVATAR_TTS", "chatterbox").lower()
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
        self.backend = "edge-tts"
        # Active backend (switchable at runtime via set_backend) — starts from
        # the AVATAR_TTS env default.
        self.tts_backend = TTS_BACKEND
        self.synthesizing = False        # True while a line is being generated
        self.speaking = False            # True while the bot is ACTUALLY talking
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
        if self.tts_backend == "xtts" and self._get_xtts() is not None:
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

    def clear_pending(self):
        """Drop queued (not-yet-started) lines so a higher-priority line (e.g. a
        user's ASK answer) plays next instead of behind an auto-host backlog. The
        currently-playing line is left to finish."""
        if self._loop is None or self.speech_queue is None:
            return
        async def _drain():
            try:
                while not self.speech_queue.empty():
                    self.speech_queue.get_nowait()
            except Exception:
                pass
        try:
            asyncio.run_coroutine_threadsafe(_drain(), self._loop)
        except Exception:
            pass

    def speak(self, text):
        """Queue text to be spoken; trigger any keyword reaction immediately."""
        text = (text or "").strip()
        if not text:
            return
        if self.behavior is not None:
            try:
                self.behavior.scan_text_for_reactions(text)
            except Exception:
                pass
        self.words_spoken += len(text.split())
        self.lines_spoken += 1
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.speech_queue.put(text), self._loop)

    def set_muted(self, muted):
        """Mute/unmute audio output (mouth still moves when muted)."""
        self.muted = bool(muted)

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
        valid = ("xtts", "maya1", "chatterbox", "multilingual", "kokoro", "edge", "auto")
        if name not in valid:
            return False, f"unknown backend {name!r}"
        self.tts_backend = name
        print(f"[TTS] backend -> {name}")
        return True, name

    def warm_backend(self):
        """Load the current backend's model now (blocking) so the first SPEAK
        doesn't stall. Safe to call from any thread. Returns the banner string."""
        return self.startup_check()[1]

    # -------------------------------------------------------------------------
    # SYNTHESIS CACHE  (pre-render heavy voices so live playback never stalls)
    # -------------------------------------------------------------------------
    def _cache_key(self, text):
        """Stable filename for (backend, voice/identity, text)."""
        ident = self.tts_backend
        if self.tts_backend == "maya1":
            ident += "|" + os.environ.get("AVATAR_MAYA_DESC", "default")
        elif self.tts_backend == "chatterbox":
            ident += "|" + os.environ.get("AVATAR_CLONE_REF", "default")
        elif self.tts_backend == "multilingual":
            ident += "|" + os.environ.get("AVATAR_CLONE_REF", "default") \
                + "|" + os.environ.get("AVATAR_TTS_LANG", "auto")
        elif self.tts_backend in ("auto", "kokoro"):
            ident += "|" + KOKORO_VOICE
        else:
            ident += "|" + self.voice
        h = hashlib.sha1(f"{ident}|{text}".encode("utf-8")).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{h}.wav")

    def _cache_load(self, text):
        """Return cached 16 kHz PCM for text, or None."""
        if not TTS_CACHE or sf is None:
            return None
        path = self._cache_key(text)
        if not os.path.exists(path):
            return None
        try:
            data, _ = sf.read(path, dtype="float32")
            return np.asarray(data, dtype=np.float32).flatten()
        except Exception:
            return None

    def _cache_save(self, text, pcm):
        if not TTS_CACHE or sf is None:
            return
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            sf.write(self._cache_key(text), np.asarray(pcm, dtype=np.float32),
                     SAMPLE_RATE, subtype="PCM_16")
        except Exception:
            pass

    def prerender(self, texts, progress=None):
        """Synthesize + cache a list of lines NOW (blocking), so they later play
        with zero GPU work. Call at startup for known lines (greeting, demo set)
        when using a heavy backend, so its per-line synth cost is paid up front
        instead of freezing the live video mid-stream."""
        if self._loop is None:
            return
        todo = [t for t in {(t or "").strip() for t in texts}
                if t and self._cache_load(t) is None]
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
        """Pull text, synthesize the full line, then play + feed in lockstep."""
        while self._running:
            text = await self.speech_queue.get()
            if text is None or not self._running:        # shutdown sentinel
                break
            try:
                self.synthesizing = True
                pcm = await self._synthesize(text)        # FULL line first
                self.synthesizing = False
                if pcm is None or len(pcm) == 0:
                    continue
                await self._play_and_feed(pcm)            # then play + feed together
            except Exception as exc:
                print(f"[TTS] speak error: {exc}")
                self.synthesizing = False
                self._set_speaking(False)

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
        # XTTS-v2 (Coqui) — Arabic + English voice clone, auto language per line.
        if self.tts_backend == "xtts" and self._get_xtts() is not None:
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
        """Coqui XTTS-v2 synth (auto Arabic/English) -> 16 kHz mono float32 PCM."""
        wav, sr = self._xtts.synthesize(self._strip_emotion_tags(text))
        if wav is None or len(wav) == 0:
            return None
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

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
        wav = np.concatenate(chunks)
        if KOKORO_SR != SAMPLE_RATE:               # 24k -> 16k for the mel/feed path
            wav = self._resample(wav, KOKORO_SR, SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    @staticmethod
    def _resample(wav, src_sr, dst_sr):
        """Resample mono float audio. Uses librosa if present, else linear interp."""
        try:
            import librosa
            return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr)
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

    async def _play_and_feed(self, pcm):
        """Play audio and feed the mouth engine 40 ms at a time, paced to playback."""
        self._set_speaking(True)
        wav_holder = {}

        # a) start audio playback AFTER MOUTH_SYNC_DELAY so the audible voice lines up
        #    with the on-screen mouth (delayed by the render pipeline).
        async def _delayed_play():
            if MOUTH_SYNC_DELAY > 0:
                await asyncio.sleep(MOUTH_SYNC_DELAY)
            if HAVE_WINSOUND and not self.muted and self._running:
                wav = self._write_temp_wav(pcm)
                if wav:
                    try:
                        winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    except Exception:
                        pass
                    wav_holder["p"] = wav
        play_task = asyncio.ensure_future(_delayed_play())

        # b) feed the mouth engine immediately, paced to wall-clock (the playback delay
        #    above closes the gap so the lips track what WILL be audible).
        start = time.monotonic()
        n = len(pcm)
        pos = 0
        idx = 0
        while pos < n and self._running:
            try:
                self.mouth.feed_audio(pcm[pos:pos + FEED_CHUNK_SAMPLES])
            except Exception:
                pass
            pos += FEED_CHUNK_SAMPLES
            idx += 1
            target = start + idx * (FEED_CHUNK_SAMPLES / SAMPLE_RATE)
            await asyncio.sleep(max(0.0, target - time.monotonic()))

        await play_task
        await asyncio.sleep(TAIL_SILENCE)
        self._set_speaking(False)
        if wav_holder.get("p"):
            try:
                os.remove(wav_holder["p"])
            except OSError:
                pass

    def _set_speaking(self, speaking):
        """Notify the optional behaviour engine of speech boundaries. Also exposes
        `self.speaking` — the authoritative 'the bot is ACTUALLY talking' flag
        (unpolluted by any idle silence keep-alive the studio may feed the mouth)."""
        self.speaking = bool(speaking)
        if self.behavior is not None:
            try:
                self.behavior.set_speaking(speaking)
            except Exception:
                pass

    def _write_temp_wav(self, pcm):
        """Write float PCM to a temp 16 kHz mono wav for winsound. Returns path."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            data16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
            if sf is not None:
                sf.write(tmp.name, data16, SAMPLE_RATE, subtype="PCM_16")
            else:
                import wave
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(data16.tobytes())
            return tmp.name
        except Exception:
            return None

    # -------------------------------------------------------------------------
    def shutdown(self):
        """Stop the worker loop cleanly (wake the queue, then stop)."""
        self._running = False
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
