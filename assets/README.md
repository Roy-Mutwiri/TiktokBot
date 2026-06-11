# assets

## Startup sound cues

The bot plays three distinct cues as it comes online (see `startup_sound.py`):

| stage | when it fires | default sound |
|-------|---------------|---------------|
| **boot** | the instant the bot starts | deep power-up sweep + "systems online" chime |
| **camera** | the virtual camera is detected / ready | crisp scanner "lock" blips |
| **scene** | the avatar scene goes live / starts streaming | bright rising "go live" triad |

All three are **synthesized** by default — nothing to install.

### Use your own clips

Drop a file here and it plays instead of the synthesized cue:

- `jarvis_startup.wav`  (or `.mp3`) — the boot cue
- `jarvis_camera.wav`   (or `.mp3`) — the camera-detected cue
- `jarvis_scene.wav`    (or `.mp3`) — the scene-live cue

(mp3 needs ffmpeg, which this project already has.)

## Controls (env vars)

- `AVATAR_SOUNDS=0` — disable ALL cues.
- `AVATAR_BOOT_SOUND=0` — disable just the boot cue.
- `AVATAR_GREETING="0"` — disable the spoken "all systems online" greeting.
- `AVATAR_GREETING="your line here"` — change what the avatar says on boot.

## Tip

Preview the cues without starting the bot:

```
python startup_sound.py            # plays boot, camera, scene in sequence
python startup_sound.py camera     # just one
```
