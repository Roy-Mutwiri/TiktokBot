# =============================================================================
# gui_panel.py
# -----------------------------------------------------------------------------
# Tkinter GUI control panel for the AI talking-face engine.
#
# Features:
#   - Dark theme
#   - Connection status indicator (green dot = connected)
#   - Scrolling read-only log of everything sent
#   - Text entry + SPEAK button (Enter also sends)
#   - Row of customizable quick-phrase buttons
#   - Background reconnect so you can launch the GUI before the engine
# =============================================================================

import socket
import threading
import time
import tkinter as tk

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 9999
RECONNECT_DELAY = 2.0           # seconds between reconnect attempts

# Theme colors
BG_COLOR = "#1a1a1a"
INPUT_BG = "#111111"
TEXT_COLOR = "#ffffff"
LOG_BG = "#0d0d0d"
GREEN = "#00cc44"
GRAY = "#555555"
BUTTON_BG = "#222222"

# Quick-phrase buttons (edit these freely)
QUICK_PHRASES = [
    "Gold is looking bullish right now",
    "Welcome to the stream everyone!",
    "Let's talk about today's setup",
    "XAUUSD key level to watch",
    "Great question, let me break that down",
]


# =============================================================================
# CONTROL PANEL APPLICATION
# =============================================================================
class ControlPanel:
    """Tkinter GUI that streams text lines to the face engine over a socket."""

    def __init__(self, root):
        """Build the UI and start the background connection thread."""
        self.root = root
        self.sock = None
        self.connected = False
        self.running = True

        self._build_ui()

        # Start a daemon thread that keeps the socket connected.
        self.conn_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.conn_thread.start()

        # Make sure we tear down cleanly on window close.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION
    # -------------------------------------------------------------------------
    def _build_ui(self):
        """Create all widgets and lay them out."""
        self.root.title("AI Face - Control Panel")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("560x520")
        self.root.minsize(460, 420)

        # --- Top: connection status -----------------------------------------
        top = tk.Frame(self.root, bg=BG_COLOR)
        top.pack(fill="x", padx=12, pady=(12, 6))

        self.status_canvas = tk.Canvas(top, width=18, height=18, bg=BG_COLOR,
                                       highlightthickness=0)
        self.status_canvas.pack(side="left")
        self.status_dot = self.status_canvas.create_oval(3, 3, 15, 15,
                                                         fill=GRAY, outline="")

        self.status_label = tk.Label(top, text="Disconnected", bg=BG_COLOR,
                                     fg=TEXT_COLOR, font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side="left", padx=8)

        # --- Middle: scrolling log ------------------------------------------
        log_frame = tk.Frame(self.root, bg=BG_COLOR)
        log_frame.pack(fill="both", expand=True, padx=12, pady=6)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log = tk.Text(log_frame, bg=LOG_BG, fg=TEXT_COLOR,
                           insertbackground=TEXT_COLOR, font=("Consolas", 10),
                           wrap="word", state="disabled",
                           yscrollcommand=scrollbar.set, height=12,
                           relief="flat", padx=8, pady=8)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log.yview)

        # --- Quick phrases row ----------------------------------------------
        qp_frame = tk.Frame(self.root, bg=BG_COLOR)
        qp_frame.pack(fill="x", padx=12, pady=(0, 6))

        for i, phrase in enumerate(QUICK_PHRASES):
            label = phrase if len(phrase) <= 18 else phrase[:16] + "…"
            btn = tk.Button(qp_frame, text=label, bg=BUTTON_BG, fg=TEXT_COLOR,
                            activebackground="#333333", activeforeground=TEXT_COLOR,
                            relief="flat", font=("Segoe UI", 8),
                            command=lambda p=phrase: self.send_text(p))
            btn.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
            qp_frame.columnconfigure(i, weight=1)

        # --- Bottom: entry + SPEAK button -----------------------------------
        bottom = tk.Frame(self.root, bg=BG_COLOR)
        bottom.pack(fill="x", padx=12, pady=(6, 12))

        self.entry = tk.Entry(bottom, bg=INPUT_BG, fg=TEXT_COLOR,
                              insertbackground=TEXT_COLOR, font=("Segoe UI", 12),
                              relief="flat")
        self.entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.entry.bind("<Return>", self._on_enter)
        self.entry.focus_set()

        self.speak_btn = tk.Button(bottom, text="SPEAK", bg=GREEN, fg="#ffffff",
                                   activebackground="#00992f", activeforeground="#ffffff",
                                   relief="flat", font=("Segoe UI", 11, "bold"),
                                   width=10, command=self._on_speak_click)
        self.speak_btn.pack(side="right", ipady=4)

        self._append_log("[*] Control panel started. Connecting to engine...")

    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------
    def _append_log(self, message):
        """Append a line to the read-only log and scroll to the bottom."""
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT (background thread)
    # -------------------------------------------------------------------------
    def _connection_loop(self):
        """Keep a socket connection alive, reconnecting as needed."""
        while self.running:
            if not self.connected:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.connect((HOST, PORT))
                    self.sock = sock
                    self.connected = True
                    self._set_status(True)
                    self.root.after(0, self._append_log,
                                    f"[✓] Connected to engine at {HOST}:{PORT}")
                except OSError:
                    sock.close()
                    self.connected = False
                    self._set_status(False)
                    time.sleep(RECONNECT_DELAY)
            else:
                time.sleep(0.5)

    def _set_status(self, connected):
        """Update the status dot/label (thread-safe via .after)."""
        def update():
            if connected:
                self.status_canvas.itemconfig(self.status_dot, fill=GREEN)
                self.status_label.config(text="Connected")
            else:
                self.status_canvas.itemconfig(self.status_dot, fill=GRAY)
                self.status_label.config(text="Disconnected")
        self.root.after(0, update)

    # -------------------------------------------------------------------------
    # SENDING TEXT
    # -------------------------------------------------------------------------
    def send_text(self, text):
        """Send a text line to the engine; log success or failure."""
        text = text.strip()
        if not text:
            return

        if not self.connected or self.sock is None:
            self._append_log("[!] Not connected - cannot send: " + repr(text))
            return

        try:
            self.sock.sendall((text + "\n").encode("utf-8"))
            self._append_log("[✓] Sent: " + text)
        except OSError:
            self._append_log("[!] Connection lost while sending. Reconnecting...")
            self.connected = False
            self._set_status(False)
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _on_enter(self, event):
        """Handle the Enter key inside the entry field."""
        self._on_speak_click()
        return "break"

    def _on_speak_click(self):
        """Send whatever is currently in the entry box and clear it."""
        text = self.entry.get()
        self.entry.delete(0, "end")
        self.send_text(text)

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    def on_close(self):
        """Clean up the socket and close the window."""
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.root.destroy()


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """Launch the Tkinter control panel."""
    root = tk.Tk()
    ControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
