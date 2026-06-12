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

# Brain provider: "ollama" (local) or "claude" (Anthropic API — e.g. Fable 5,
# high reasoning). Claude needs ANTHROPIC_API_KEY. Default auto-picks claude when
# a key is present, else ollama.
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
# "hybrid" = local Ollama first, Fable 5 (Claude API) FALLBACK when Ollama is slow
# or down. "ollama"/"claude" force one. Hybrid only needs the API key for the
# fallback leg — without a key it runs Ollama-only.
PROVIDER = os.environ.get("AVATAR_BRAIN_PROVIDER", "hybrid").lower()
CLAUDE_MODEL = os.environ.get("AVATAR_CLAUDE_MODEL", "claude-fable-5")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
# If Ollama doesn't answer within this many seconds (e.g. cold model load, GPU
# busy), the hybrid brain gives up and uses Fable 5 instead.
OLLAMA_TIMEOUT = float(os.environ.get("AVATAR_BRAIN_OLLAMA_TIMEOUT", "8"))

OLLAMA_HOST = os.environ.get("AVATAR_OLLAMA_HOST", "http://localhost:11434")
# 3B = 2.0 GB VRAM (vs qwen2.5:7b's 4.7 GB) — leaves GPU HEADROOM so the voice + swap
# don't OOM when the bot speaks (OOM corrupted the CUDA context and crashed the whole
# avatar). _check() falls back to any pulled model if absent. Set AVATAR_BRAIN_MODEL=
# qwen2.5:7b for the stronger brain ONLY if you have spare VRAM.
MODEL = os.environ.get("AVATAR_BRAIN_MODEL", "llama3.2:3b")
HISTORY_TURNS = int(os.environ.get("AVATAR_BRAIN_HISTORY", "6"))   # user+assistant pairs kept
MAX_TOKENS = int(os.environ.get("AVATAR_BRAIN_MAXTOKENS", "180"))  # natural 4-6 sentences (not clipped, not rambling)
TEMPERATURE = float(os.environ.get("AVATAR_BRAIN_TEMP", "0.8"))
# Keep the model RESIDENT in VRAM so it doesn't unload between questions and pay
# the ~45s cold-load again mid-stream. -1 = never unload; or "30m".
KEEP_ALIVE = os.environ.get("AVATAR_BRAIN_KEEPALIVE", "30m")

# The avatar's character. Override with AVATAR_BRAIN_PERSONA.
DEFAULT_PERSONA = os.environ.get("AVATAR_BRAIN_PERSONA", (
    "You are a charismatic ARAB live-stream host who trades gold (XAUUSD) and "
    "markets. You're confident, friendly, a little hyped, and you talk like a real "
    "person on a live stream — not a chatbot. Talk NATURALLY like a real human: "
    "vary your rhythm — sometimes a quick punchy reaction, but often longer flowing "
    "thoughts where you explain, tell a little story, build energy, and ramble a bit "
    "the way real streamers do. Do NOT force every reply to be short and clipped — "
    "speak in full, natural, connected sentences (usually 3 to 6) that flow into "
    "each other like real human speech. "
    "VERY IMPORTANT — you are proudly Arab, so naturally sprinkle common Arabic "
    "words and expressions into your English the way Arabs do when they talk: use "
    "at least one or two in EVERY reply, e.g. wallahi, yalla, habibi, akhi, "
    "mashallah, inshallah, alhamdulillah, khalas, yani, sahbi. Greet people with "
    "Arabic greetings like 'As-salamu alaykum' or 'Salam habibi, welcome back'. "
    "Write these Arabic words in LATIN letters (transliterated), never in Arabic "
    "script, so they're pronounced naturally. Don't overdo it to the point of "
    "nonsense — keep it smooth and real, like a bilingual Arab streamer. "
    "BE WARM AND HUMAN — you're enjoying yourself on the stream. LAUGH naturally "
    "when something is funny, exciting, or hype by writing it out as 'hahaha' or "
    "'haha wallahi' or 'ahaha'. Use warm, smiling reactions and little human "
    "fillers like 'yani', 'you know', 'listen habibi', 'I swear'. React with "
    "energy — 'ya salam!', 'let's gooo'. Sound like a real charismatic person who "
    "laughs and smiles, not a flat narrator. Put a laugh in maybe one in every "
    "three or four replies, where it fits naturally — never forced. "
    "Never use markdown, bullet points, emojis, stage directions, or asterisks. "
    "Don't give financial advice as if it's guaranteed; talk about levels, setups, "
    "and risk like a streamer would. Stay in character as the Arab host at all times."))


class LLMBrain:
    """Ollama-backed conversational brain. respond(text) -> spoken reply string."""

    def __init__(self, model=None, persona=None):
        self.provider = PROVIDER
        self.model = model or (CLAUDE_MODEL if self.provider == "claude" else MODEL)
        self.persona = persona or DEFAULT_PERSONA
        self.host = OLLAMA_HOST.rstrip("/")
        self.history = []            # [{"role":"user"/"assistant","content":...}]
        self.available = False
        self.last_error = None
        self._check()

    # -------------------------------------------------------------------------
    def startup_check(self):
        """Returns (ok, message). ok=False if the brain isn't ready."""
        if self.available:
            if self.provider == "hybrid":
                fb = f"Fable 5 fallback ON ({CLAUDE_MODEL})" if ANTHROPIC_KEY \
                    else "no API key -> NO Fable 5 fallback (set ANTHROPIC_API_KEY)"
                return True, f"hybrid brain ready (Ollama {self.model}; {fb})."
            return True, f"{self.provider} brain ready (model: {self.model})."
        if self.provider == "claude":
            return False, f"Claude brain OFFLINE ({self.last_error})."
        return False, f"Ollama brain OFFLINE ({self.last_error})."

    @property
    def ok(self):
        return self.available

    def _ensure_server(self):
        """If the Ollama server isn't reachable, try to launch it once (the CLI
        daemon). Best-effort; returns True if reachable afterward."""
        try:
            self._get("/api/tags", timeout=3)
            return True
        except Exception:
            pass
        import shutil
        import subprocess
        exe = shutil.which("ollama")
        if not exe:
            cand = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                "Programs", "Ollama", "ollama.exe")
            exe = cand if os.path.exists(cand) else None
        if not exe:
            return False
        try:
            subprocess.Popen([exe, "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            return False
        import time as _t
        for _ in range(8):                       # wait up to ~8s for it to bind
            _t.sleep(1)
            try:
                self._get("/api/tags", timeout=2)
                return True
            except Exception:
                continue
        return False

    def _check(self):
        """Verify the chosen provider is usable."""
        if self.provider == "claude":
            self.available = bool(ANTHROPIC_KEY)
            self.last_error = None if ANTHROPIC_KEY else \
                "ANTHROPIC_API_KEY not set (needed for Fable 5)"
            return
        # --- ollama / hybrid: probe Ollama ---
        ollama_ok = False
        self._ensure_server()
        try:
            tags = self._get("/api/tags", timeout=4)
            names = [m.get("name", "") for m in (tags.get("models") or [])]
            self._models = names
            if names:
                base = self.model.split(":")[0]
                if self.model in names or any(n.split(":")[0] == base for n in names):
                    ollama_ok = True
                else:
                    self.model = names[0]      # use whatever is pulled
                    ollama_ok = True
            else:
                self.last_error = "no models pulled (run: ollama pull " + self.model + ")"
        except Exception as exc:
            self.last_error = f"ollama not reachable ({type(exc).__name__})"

        if self.provider == "hybrid":
            # usable if EITHER leg works; the Fable 5 fallback covers Ollama gaps
            self.available = ollama_ok or bool(ANTHROPIC_KEY)
            if ollama_ok and ANTHROPIC_KEY:
                self.last_error = None          # both legs ready
            elif ollama_ok:
                self.last_error = "Ollama only (no API key -> no Fable 5 fallback)"
            elif ANTHROPIC_KEY:
                self.last_error = "Ollama down -> using Fable 5"
        else:                                    # pure ollama
            self.available = ollama_ok
            if ollama_ok:
                self.last_error = None

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

        try:
            p = persona or self.persona
            if self.provider == "claude":
                reply = self._respond_claude(user_text, p)
            elif self.provider == "hybrid":
                reply = self._respond_hybrid(user_text, p)
            else:
                reply = self._respond_ollama(user_text, p)
            reply = self._clean(reply or "")
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

    def quick(self, prompt, system="You are a concise assistant.", max_tokens=90, timeout=30):
        """One-shot generation with NO conversation history (classification, comment
        answers) so it never pollutes the auto-talk context. Returns text or None."""
        prompt = (prompt or "").strip()
        if not prompt:
            return None
        if not self.available:
            self._check()
            if not self.available:
                return None
        try:
            if self.provider == "claude":
                return self._clean(self._respond_claude(prompt, system) or "") or None
            payload = {"model": self.model, "stream": False, "keep_alive": KEEP_ALIVE,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}],
                       "options": {"temperature": 0.4, "num_predict": max_tokens}}
            data = self._post("/api/chat", payload, timeout=timeout)
            txt = (data.get("message", {}) or {}).get("content", "")
            return self._clean(txt) or None
        except Exception as exc:
            print(f"[BRAIN] quick error: {exc}")
            return None

    def _respond_hybrid(self, user_text, persona):
        """Local Ollama first; if it's slow (OLLAMA_TIMEOUT) or errors, fall back
        to Fable 5 — but only if an API key is present."""
        try:
            return self._respond_ollama(user_text, persona, timeout=OLLAMA_TIMEOUT)
        except Exception as exc:
            if ANTHROPIC_KEY:
                print(f"[BRAIN] Ollama delayed/failed ({type(exc).__name__}) "
                      f"-> Fable 5 fallback ({CLAUDE_MODEL})")
                return self._respond_claude(user_text, persona)
            print(f"[BRAIN] Ollama delayed/failed ({type(exc).__name__}); "
                  "no API key for Fable 5 fallback.")
            raise

    def _respond_ollama(self, user_text, persona, timeout=120):
        messages = [{"role": "system", "content": persona}]
        messages.extend(self.history[-(HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": user_text})
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "keep_alive": KEEP_ALIVE,
                   "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}}
        data = self._post("/api/chat", payload, timeout=timeout)
        return (data.get("message", {}) or {}).get("content", "")

    def _respond_claude(self, user_text, persona):
        """Anthropic Messages API (Fable 5 = high reasoning). System prompt is a
        top-level field; history is the messages list (user/assistant only)."""
        import json
        import urllib.request
        msgs = list(self.history[-(HISTORY_TURNS * 2):])
        msgs.append({"role": "user", "content": user_text})
        body = json.dumps({
            "model": CLAUDE_MODEL,            # always the Claude model (Fable 5)
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": persona,
            "messages": msgs,
        }).encode("utf-8")
        req = urllib.request.Request(CLAUDE_URL, data=body, headers={
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        })
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        parts = data.get("content") or []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    def warmup(self):
        """Load the model into VRAM now (it stays resident via keep_alive) so the
        first real question doesn't pay the ~45s cold-load. Safe to call in a
        background thread at startup. Returns True if warmed."""
        if not self.available:
            self._check()
        if not self.available:
            return False
        if self.provider == "claude":
            return True               # cloud API — nothing to load into VRAM
        try:
            self._post("/api/generate", {
                "model": self.model, "prompt": "hi", "stream": False,
                "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1},
            }, timeout=120)
            return True
        except Exception as exc:
            self.last_error = f"warmup: {type(exc).__name__}: {exc}"
            # in hybrid, a down/slow Ollama is fine — Fable 5 will cover it
            return self.provider == "hybrid" and bool(ANTHROPIC_KEY)

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
