# =============================================================================
# engines/llm_brain.py
# -----------------------------------------------------------------------------
# The avatar's "brain": a local Ollama model that turns what you (or chat) type
# into the avatar's own in-character spoken answer. Instead of the avatar
# reading your text verbatim, it RESPONDS — then the TTS voice speaks the reply
# and the mouth syncs to it.
#
#   you type:   "is gold a buy here?"
#   brain says: "Honestly, I love this level. We're sitting right on support,
#                so I'm watching for a bounce — but keep your stop tight."
#
# Talks to Ollama's local HTTP API (no extra pip dep — stdlib urllib). Keeps a
# short rolling history so replies stay coherent across turns. Replies are kept
# SHORT and free of markdown/emojis because they are spoken aloud.
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

OLLAMA_HOST = os.environ.get("AVATAR_OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("AVATAR_BRAIN_MODEL", "llama3.2:3b")
HISTORY_TURNS = int(os.environ.get("AVATAR_BRAIN_HISTORY", "6"))   # user+assistant pairs kept
MAX_TOKENS = int(os.environ.get("AVATAR_BRAIN_MAXTOKENS", "120"))  # short = fast + speakable
TEMPERATURE = float(os.environ.get("AVATAR_BRAIN_TEMP", "0.8"))

# The avatar's character. Override with AVATAR_BRAIN_PERSONA.
DEFAULT_PERSONA = os.environ.get("AVATAR_BRAIN_PERSONA", (
    "You are a charismatic live-stream host who trades gold (XAUUSD) and markets. "
    "You're confident, friendly, a little hyped, and you talk like a real person on "
    "a live stream — not a chatbot. Answer the viewer in 1-3 short spoken sentences. "
    "Never use markdown, bullet points, emojis, stage directions, or asterisks. "
    "Don't give financial advice as if it's guaranteed; talk about levels, setups, "
    "and risk like a streamer would. Stay in character at all times."))


class LLMBrain:
    """Ollama-backed conversational brain. respond(text) -> spoken reply string."""

    def __init__(self, model=None, persona=None):
        self.model = model or MODEL
        self.persona = persona or DEFAULT_PERSONA
        self.host = OLLAMA_HOST.rstrip("/")
        self.history = []            # [{"role":"user"/"assistant","content":...}]
        self.available = False
        self.last_error = None
        self._check()

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Returns (ok, message). ok=False if Ollama/model isn't ready."""
        if self.available:
            return True, f"Ollama brain ready (model: {self.model})."
        return False, f"Ollama brain OFFLINE ({self.last_error})."

    @property
    def ok(self):
        return self.available

    def _check(self):
        """Ping the server and verify the model is pulled."""
        try:
            tags = self._get("/api/tags", timeout=4)
            names = [m.get("name", "") for m in (tags.get("models") or [])]
            self._models = names
            if not names:
                self.available = False
                self.last_error = "no models pulled (run: ollama pull " + self.model + ")"
                return
            # accept exact or prefix match (e.g. 'llama3.2:3b' vs 'llama3.2:latest')
            base = self.model.split(":")[0]
            if self.model in names or any(n.split(":")[0] == base for n in names):
                self.available = True
                self.last_error = None
            else:
                # fall back to the first available model so it still works
                self.model = names[0]
                self.available = True
                self.last_error = None
        except Exception as exc:
            self.available = False
            self.last_error = f"server not reachable at {self.host} ({type(exc).__name__})"

    # -------------------------------------------------------------------------
    def respond(self, user_text, persona=None):
        """Generate the avatar's in-character reply. Returns the reply string,
        or None if the brain is unavailable / errored (caller can fall back to
        speaking the raw text)."""
        user_text = (user_text or "").strip()
        if not user_text:
            return None
        if not self.available:
            self._check()
            if not self.available:
                return None

        messages = [{"role": "system", "content": persona or self.persona}]
        messages.extend(self.history[-(HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        }
        try:
            data = self._post("/api/chat", payload, timeout=120)
            reply = (data.get("message", {}) or {}).get("content", "").strip()
            reply = self._clean(reply)
            if not reply:
                return None
            # remember the turn for coherence
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-(HISTORY_TURNS * 2):]
            return reply
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[BRAIN] respond error: {self.last_error}")
            return None

    def reset(self):
        """Forget the conversation (fresh context)."""
        self.history = []

    # -------------------------------------------------------------------------
    @staticmethod
    def _clean(text):
        """Strip markdown/formatting that would sound wrong when spoken."""
        import re
        text = re.sub(r"[*_`#>]+", "", text)              # md emphasis/headers
        text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)  # bullets
        text = re.sub(r"\((?:laughs?|smiles?|chuckles?)\)", "", text, flags=re.I)  # stage dirs
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get(self, path, timeout=10):
        req = urllib.request.Request(self.host + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload, timeout=120):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    brain = LLMBrain()
    print("[BRAIN] startup_check:", brain.startup_check())
    if brain.ok:
        for q in ["hey, is gold a buy right now?",
                  "what about the dollar?"]:
            print("\nYOU:", q)
            print("AVATAR:", brain.respond(q))
    else:
        print("[BRAIN] Start Ollama and pull a model:")
        print("        ollama serve   (usually auto-runs)")
        print(f"        ollama pull {MODEL}")
