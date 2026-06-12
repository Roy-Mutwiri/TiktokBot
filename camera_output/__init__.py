"""
camera_output — Phase-1 (WDK-independent) camera frame pipeline for RoyCam.

App side:  FrameSource -> frame_converter -> RingWriter -> shared memory ring
Service:   RingReader -> validate -> (driver bridge / preview / fallback)

This package is the legitimate, own-branded software-camera frame path. It does
not impersonate any vendor and does not claim to be physical USB hardware.
"""
from . import roycam_format as roycam_format  # noqa: F401
from . import frame_converter as frame_converter  # noqa: F401
from .ring import RingReader, RingWriter  # noqa: F401
from .sources import (  # noqa: F401
    CallableSource,
    FrameSource,
    StaticImageSource,
    TestPatternSource,
    VideoFileSource,
)
from .controller import CameraOutputController  # noqa: F401

__all__ = [
    "roycam_format",
    "frame_converter",
    "RingReader",
    "RingWriter",
    "FrameSource",
    "TestPatternSource",
    "StaticImageSource",
    "VideoFileSource",
    "CallableSource",
    "CameraOutputController",
]
