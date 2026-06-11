# assets

## JARVIS boot sound

When the bot starts it plays a boot cue (see `startup_sound.py`).

- By default it **synthesizes** a sci-fi power-up sweep + "systems online" chime
  — nothing to install.
- To use **your own** clip instead, drop a file here named:
  - `jarvis_startup.wav`  (preferred), or
  - `jarvis_startup.mp3`  (needs ffmpeg, which this project already has).

  Whatever you put there plays instead of the synthesized cue.

## Controls (env vars)

- `AVATAR_BOOT_SOUND=0` — disable the boot sound entirely.
- `AVATAR_GREETING="0"` — disable the spoken "all systems online" greeting.
- `AVATAR_GREETING="your line here"` — change what the avatar says on boot.
