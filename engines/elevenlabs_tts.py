# =============================================================================
# engines/elevenlabs_tts.py — ElevenLabs cloud voice backend.
#
# High-quality, very natural cloud TTS (great Arabic + English). Uses the REST API
# directly (stdlib urllib — no extra pip dep) and asks for raw PCM at 16 kHz so the
# bytes drop straight into the avatar's mouth-feed with no decode step.
#
#   key   : ELEVENLABS_API_KEY   (required — paid service)
#   voice : AVATAR_ELEVEN_VOICE  (the voiceId, e.g. evtDwF7UqN7X6HhwtdoK)
#   model : AVATAR_ELEVEN_MODEL  (eleven_flash_v2_5 = fast/cheap multilingual,
#                                 eleven_turbo_v2_5 = balanced,
#                                 eleven_multilingual_v2 = best quality)
#
# Cost: the avatar talks continuously (~42k chars/hour), so this is expensive for
# long streams — that's a deliberate, user-made choice.
# =============================================================================
import os
import sys
import json
import urllib.request
import urllib.error

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip() or \
    os.environ.get("ELEVEN_API_KEY", "").strip()
VOICE_ID = os.environ.get("AVATAR_ELEVEN_VOICE", "evtDwF7UqN7X6HhwtdoK").strip()
MODEL_ID = os.environ.get("AVATAR_ELEVEN_MODEL", "eleven_flash_v2_5").strip()
SAMPLE_RATE = 16000                        # pcm_16000 -> matches the avatar pipeline
# voice settings (0..1): stability = consistency vs expressiveness, similarity =
# how close to the cloned voice, style = exaggeration, speed = 0.7..1.2 pacing.
STABILITY = float(os.environ.get("AVATAR_ELEVEN_STABILITY", "0.45"))
SIMILARITY = float(os.environ.get("AVATAR_ELEVEN_SIMILARITY", "0.8"))
STYLE = float(os.environ.get("AVATAR_ELEVEN_STYLE", "0.0"))
SPEED = float(os.environ.get("AVATAR_ELEVEN_SPEED", "0.96"))     # slightly calm
TIMEOUT = float(os.environ.get("AVATAR_ELEVEN_TIMEOUT", "20"))


class ElevenLabsBackend:
    """ElevenLabs REST TTS. synthesize(text) -> (float32 mono 16 kHz, sr)."""

    def __init__(self, voice_id=None, api_key=None):
        self.voice_id = (voice_id or VOICE_ID).strip()
        self.api_key = (api_key or API_KEY).strip()
        self.model = MODEL_ID
        self.sr = SAMPLE_RATE
        self.ready = False
        self.last_error = None
        self._verify()

    @property
    def ok(self):
        return self.ready

    def _verify(self):
        """Cheap auth check: hit /v1/user (no synthesis cost)."""
        if not self.api_key:
            self.last_error = "ELEVENLABS_API_KEY not set"
            return
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": self.api_key})
            with urllib.request.urlopen(req, timeout=10) as r:
                json.loads(r.read().decode("utf-8"))
            self.ready = True
            self.last_error = None
        except urllib.error.HTTPError as e:
            self.last_error = f"auth failed (HTTP {e.code})"
        except Exception as e:
            self.last_error = f"unreachable ({type(e).__name__})"

    def startup_check(self):
        if self.ready:
            return True, f"ElevenLabs ready (voice {self.voice_id[:8]}…, {self.model})."
        return False, f"ElevenLabs unavailable: {self.last_error}."

    def synthesize(self, text):
        """Synthesise `text` -> (float32 mono 16 kHz, sr). Returns (None, sr) on error
        so the engine can fall back to a local backend (the avatar never goes silent)."""
        text = (text or "").strip()
        if not text or not self.api_key:
            return None, self.sr
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
               f"?output_format=pcm_{self.sr}")
        body = json.dumps({
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": STABILITY,
                "similarity_boost": SIMILARITY,
                "style": STYLE,
                "speed": SPEED,
                "use_speaker_boost": True,
            },
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        })
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read()
            if not raw:
                return None, self.sr
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return pcm, self.sr
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:160]
            except Exception:
                pass
            print(f"[ELEVEN] HTTP {e.code} {detail}")
            self.last_error = f"HTTP {e.code}"
            return None, self.sr
        except Exception as e:
            print(f"[ELEVEN] synth error: {type(e).__name__}: {str(e)[:80]}")
            return None, self.sr


if __name__ == "__main__":
    eng = ElevenLabsBackend()
    print("[ELEVEN]", eng.startup_check()[1])
    if eng.ok:
        import soundfile as sf
        for txt, name in [("Welcome back everyone, gold is moving strong today.", "_el_en.wav"),
                          ("مرحبا بكم، الذهب يتحرك بقوة اليوم", "_el_ar.wav")]:
            wav, sr = eng.synthesize(txt)
            if wav is not None:
                sf.write(name, wav, sr)
                print(f"  wrote {name} ({len(wav)/sr:.2f}s)")
