"""
CLI for the Phase-1 camera output pipeline.

  python -m camera_output --source test --fmt nv12 --width 1280 --height 720
  python -m camera_output --source image --path pic.jpg
  python -m camera_output --source video --path clip.mp4

Writes frames into the shared ring until Ctrl+C.
"""
from __future__ import annotations

import argparse
import time

from . import roycam_format as F
from .controller import CameraOutputController
from .sources import StaticImageSource, TestPatternSource, VideoFileSource

_FMT = {"nv12": F.FMT_NV12, "yuy2": F.FMT_YUY2, "rgb32": F.FMT_RGB32}


def main():
    ap = argparse.ArgumentParser(prog="camera_output")
    ap.add_argument("--source", choices=["test", "image", "video"],
                    default="test")
    ap.add_argument("--path", help="file path for image/video source")
    ap.add_argument("--fmt", choices=list(_FMT), default="nv12")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    if args.source == "image":
        src = StaticImageSource(args.path, args.width, args.height)
    elif args.source == "video":
        src = VideoFileSource(args.path, args.width, args.height)
    else:
        src = TestPatternSource(args.width, args.height, args.fps)

    ctl = CameraOutputController(source=src, width=args.width,
                                 height=args.height, fmt_id=_FMT[args.fmt],
                                 fps=args.fps)
    ctl.start()
    print(f"[camera_output] {src.name} -> {args.fmt} "
          f"{args.width}x{args.height}@{args.fps} ring={ctl.writer.path}")
    print("[camera_output] Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
            print(f"  frames_written={ctl.frames_written} "
                  f"err={ctl.last_error}")
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        print("[camera_output] stopped")


if __name__ == "__main__":
    main()
