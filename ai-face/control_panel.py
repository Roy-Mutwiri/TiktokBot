# =============================================================================
# control_panel.py
# -----------------------------------------------------------------------------
# Terminal control panel for the AI talking-face engine.
#
# Connects to realtime_face.py over a TCP socket and forwards whatever you type
# (newline-delimited) so the face speaks it. Reconnects automatically if the
# engine isn't running yet or the connection drops.
# =============================================================================

import socket
import sys
import time

# Force UTF-8 stdout so status glyphs ([*] [✓] [!]) don't crash the Windows
# console, whose default code page (cp1252) can't encode them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 9999
RECONNECT_DELAY = 2.0       # seconds between reconnect attempts


# -----------------------------------------------------------------------------
# CONNECTION HANDLING
# -----------------------------------------------------------------------------
def connect():
    """Block until a connection to the face engine is established.

    Returns a connected socket. Retries forever until successful or interrupted.
    """
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((HOST, PORT))
            print(f"[✓] Connected to face engine at {HOST}:{PORT}")
            return sock
        except (ConnectionRefusedError, OSError):
            sock.close()
            print(f"[*] Face engine not available. Retrying in {RECONNECT_DELAY:.0f}s... "
                  "(start realtime_face.py)")
            time.sleep(RECONNECT_DELAY)


def send_line(sock, text):
    """Send a single newline-terminated text line to the engine.

    Returns True on success, False if the connection was lost.
    """
    try:
        sock.sendall((text + "\n").encode("utf-8"))
        return True
    except OSError:
        return False


# -----------------------------------------------------------------------------
# MAIN INPUT LOOP
# -----------------------------------------------------------------------------
def main():
    """Run the interactive terminal control panel."""
    print("=" * 60)
    print(" AI Face - Terminal Control Panel")
    print("=" * 60)
    print(" Type text and press Enter to make the face speak it.")
    print(" Type 'quit' or 'exit' (or Ctrl+C) to leave.")
    print("=" * 60)

    sock = connect()

    while True:
        try:
            text = input("speak> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Exiting control panel.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("[*] Exiting control panel.")
            break

        if not send_line(sock, text):
            print("[!] Connection lost. Reconnecting...")
            sock.close()
            sock = connect()
            # Retry sending once after reconnecting.
            if not send_line(sock, text):
                print("[!] Failed to send after reconnect:", repr(text))
                continue

        print("[✓] Sent:", repr(text))

    try:
        sock.close()
    except OSError:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[!] Control panel error:", exc)
        sys.exit(1)
