import os
import sys
import time


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES_DIR = os.path.join(PROJECT_DIR, "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)


def main():
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    t0 = time.time()
    from xtts_tts import XTTSBackend, XTTS_REFS

    print("[prewarm] Mohammed refs:")
    for path in XTTS_REFS:
        print("[prewarm]  - " + os.path.basename(path))

    eng = XTTSBackend()
    ok, msg = eng.startup_check()
    print("[prewarm] " + msg)
    if not ok:
        return 1

    for text in (
        "Welcome back habibi, gold is moving fast today.",
        "مرحبا بكم في البث المباشر.",
    ):
        wav, sr = eng.synthesize(text)
        secs = (len(wav) / float(sr)) if wav is not None and sr else 0.0
        print(f"[prewarm] synthesized {secs:.2f}s @ {sr} Hz")

    print(f"[prewarm] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
