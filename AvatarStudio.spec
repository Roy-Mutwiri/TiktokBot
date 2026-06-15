# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(SPECPATH)


def tree(source, target=None):
    source_path = ROOT / source
    if not source_path.exists():
        return []
    destination = target or source
    return [
        (str(path), str(Path(destination) / path.relative_to(source_path).parent))
        for path in source_path.rglob("*")
        if path.is_file() and "trt_cache" not in path.parts
    ]


datas = []
for source in (
    "assets",
    "character_src",
    "character_views",
    "Haddan",
    "haddan_white",
    "haddan_white_bearded",
    "voice_refs",
    "music/streambeats",
    "models",
    "ai-face/models",
    "ai-face/Wav2Lip/checkpoints",
    "ai-face/Wav2Lip/face_detection/detection/sfd",
):
    datas += tree(source)

for source in (
    "studio_background.jpg",
    "tiktok_handles.json",
    "engines/reactions.json",
):
    path = ROOT / source
    if path.exists():
        datas.append((str(path), str(Path(source).parent)))

playwright_cache = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
for pattern in ("chromium-*", "ffmpeg-*", "winldd-*"):
    for source_path in playwright_cache.glob(pattern):
        if source_path.is_dir():
            for path in source_path.rglob("*"):
                if path.is_file():
                    destination = (
                        Path("ms-playwright")
                        / source_path.name
                        / path.relative_to(source_path).parent
                    )
                    datas.append((str(path), str(destination)))

hiddenimports = sorted(collect_submodules("engines") + [
    "tradingview_pilot",
    "startup_sound",
])

a = Analysis(
    ["avatar_studio.py"],
    pathex=[str(ROOT), str(ROOT / "engines")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "installer" / "runtime_env.py")],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "notebook",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AvatarStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "avatar_studio.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AvatarStudio",
)
