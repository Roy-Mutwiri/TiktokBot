# =============================================================================
# control_gui.py  —  operator console for realtime_avatar.py
# -----------------------------------------------------------------------------
# Dark-theme tkinter panel that connects to the running avatar over a local
# socket (127.0.0.1:9998) and tells it what to say. Type a line, press Enter (or
# SPEAK), and the AI voice speaks it while the mouth syncs.
#
#   Terminal 1:  python realtime_avatar.py
#   Terminal 2:  python control_gui.py
#
# Controls: text entry + SPEAK, quick-phrase buttons, voice selector, MUTE
# toggle, and a live connection status dot (auto-reconnects).
# =============================================================================

import socket
import threading
import time
import tkinter as tk
from tkinter import ttk

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 9998
RECONNECT_SECONDS = 2.0

VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-EricNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
]

QUICK_PHRASES = [
    ("Greeting", "Hey everyone, welcome back to the stream. Great to have you here."),
    ("Market call", "Gold is pushing into a key resistance level right now. Watch this closely."),
    ("Thanks", "Thank you so much for the support, I really appreciate every one of you."),
    ("Question", "That's a great question. Let me break it down for you step by step."),
    ("Hype", "This is a serious move. Look at that volume coming in right now."),
    ("Risk", "Real talk, risk management is everything in this game. Protect your capital."),
]

# Dark theme palette
BG = "#15171c"
BG2 = "#1e2128"
FG = "#e6e6e6"
ACCENT = "#00d7ff"
GREEN = "#22cc55"
RED = "#dd3344"
ENTRY_BG = "#0f1115"


class ControlGUI:
    """Tkinter console that streams commands to the avatar over a socket."""

    def __init__(self, root):
        self.root = root
        self.sock = None
        self.connected = False
        self.muted = False
        self._running = True
        self._lock = threading.Lock()

        root.title("AVATAR CONTROL — XAUUSD")
        root.configure(bg=BG)
        root.geometry("560x620")
        root.minsize(480, 520)

        self._build_ui()
        threading.Thread(target=self._connect_loop, daemon=True).start()

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # --- header / status ---
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(header, text="AVATAR CONTROL", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        self.status_canvas = tk.Canvas(header, width=16, height=16, bg=BG,
                                       highlightthickness=0)
        self.status_canvas.pack(side="right")
        self.status_dot = self.status_canvas.create_oval(2, 2, 14, 14,
                                                         fill=RED, outline="")
        self.status_label = tk.Label(header, text="connecting...", bg=BG, fg=FG,
                                     font=("Segoe UI", 9))
        self.status_label.pack(side="right", padx=8)

        # --- text entry ---
        entry_frame = tk.Frame(self.root, bg=BG)
        entry_frame.pack(fill="x", padx=14, pady=6)
        self.entry = tk.Text(entry_frame, height=3, bg=ENTRY_BG, fg=FG,
                             insertbackground=FG, font=("Segoe UI", 12),
                             relief="flat", wrap="word", padx=8, pady=6)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: None)   # Shift+Enter = newline
        self.entry.focus_set()

        # --- buttons row ---
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(2, 8))
        self.speak_btn = tk.Button(btn_row, text="SPEAK  (Enter)",
                                   command=self.send_speak, bg=ACCENT, fg="#04222a",
                                   font=("Segoe UI", 11, "bold"), relief="flat",
                                   activebackground="#33e2ff", cursor="hand2")
        self.speak_btn.pack(side="left", fill="x", expand=True, ipady=4)
        self.mute_btn = tk.Button(btn_row, text="MUTE", command=self.toggle_mute,
                                  bg=BG2, fg=FG, font=("Segoe UI", 11, "bold"),
                                  relief="flat", width=10, cursor="hand2")
        self.mute_btn.pack(side="left", padx=(8, 0), ipady=4)

        # --- voice selector ---
        voice_frame = tk.Frame(self.root, bg=BG)
        voice_frame.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(voice_frame, text="Voice", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).pack(side="left")
        self.voice_var = tk.StringVar(value=VOICES[0])
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox", fieldbackground=ENTRY_BG, background=BG2,
                        foreground=FG, arrowcolor=ACCENT)
        self.voice_box = ttk.Combobox(voice_frame, textvariable=self.voice_var,
                                      values=VOICES, state="readonly", width=28)
        self.voice_box.pack(side="left", padx=8)
        self.voice_box.bind("<<ComboboxSelected>>", self._on_voice)

        # --- quick phrases ---
        qp = tk.LabelFrame(self.root, text=" Quick phrases ", bg=BG, fg=ACCENT,
                           font=("Segoe UI", 9, "bold"), relief="flat")
        qp.pack(fill="x", padx=14, pady=4)
        grid = tk.Frame(qp, bg=BG)
        grid.pack(fill="x", pady=4)
        for i, (label, text) in enumerate(QUICK_PHRASES):
            b = tk.Button(grid, text=label, bg=BG2, fg=FG, relief="flat",
                          font=("Segoe UI", 9), cursor="hand2",
                          command=lambda t=text: self.send_text(t))
            b.grid(row=i // 3, column=i % 3, sticky="ew", padx=3, pady=3, ipady=3)
        for c in range(3):
            grid.columnconfigure(c, weight=1)

        # --- log ---
        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=14, pady=(6, 12))
        tk.Label(log_frame, text="Spoken log", bg=BG, fg=FG,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.log = tk.Text(log_frame, bg=ENTRY_BG, fg="#aab2c0", relief="flat",
                           font=("Consolas", 9), wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(2, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------------
    # SOCKET
    # -------------------------------------------------------------------------
    def _connect_loop(self):
        """Keep a connection to the avatar alive; reconnect if it drops."""
        while self._running:
            if not self.connected:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(3.0)
                    s.connect((HOST, PORT))
                    s.settimeout(None)
                    with self._lock:
                        self.sock = s
                        self.connected = True
                    self._set_status(True)
                    self._log("[connected to avatar]")
                except Exception:
                    self._set_status(False)
                    time.sleep(RECONNECT_SECONDS)
            else:
                time.sleep(0.5)

    def _send(self, line):
        """Send one command line; mark disconnected on failure."""
        with self._lock:
            if not self.connected or self.sock is None:
                self._log("[not connected — is realtime_avatar.py running?]")
                return False
            try:
                self.sock.sendall((line + "\n").encode("utf-8"))
                return True
            except Exception:
                self.connected = False
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self._set_status(False)
                self._log("[connection lost — reconnecting...]")
                return False

    # -------------------------------------------------------------------------
    # ACTIONS
    # -------------------------------------------------------------------------
    def _on_enter(self, event):
        self.send_speak()
        return "break"          # prevent the newline being inserted

    def send_speak(self):
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return
        self.entry.delete("1.0", "end")
        self.send_text(text)

    def send_text(self, text):
        if self._send("SPEAK " + text):
            self._log("> " + text)

    def toggle_mute(self):
        self.muted = not self.muted
        self._send("MUTE " + ("1" if self.muted else "0"))
        self.mute_btn.configure(text="UNMUTE" if self.muted else "MUTE",
                                bg=RED if self.muted else BG2,
                                fg="#ffffff" if self.muted else FG)
        self._log("[muted]" if self.muted else "[unmuted]")

    def _on_voice(self, event):
        voice = self.voice_var.get()
        self._send("VOICE " + voice)
        self._log("[voice -> " + voice + "]")

    # -------------------------------------------------------------------------
    # UI HELPERS  (always marshalled onto the tk thread)
    # -------------------------------------------------------------------------
    def _set_status(self, connected):
        def _apply():
            self.status_canvas.itemconfig(self.status_dot,
                                          fill=GREEN if connected else RED)
            self.status_label.configure(text="connected" if connected
                                        else "disconnected")
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _log(self, msg):
        def _apply():
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _on_close(self):
        # Closing the console does NOT stop the avatar (press Q in its terminal).
        self._running = False
        try:
            with self._lock:
                if self.sock:
                    self.sock.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    ControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
