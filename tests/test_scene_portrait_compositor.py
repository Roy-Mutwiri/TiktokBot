import threading

import numpy as np
import pytest
from PIL import Image

import avatar_studio
from engines import youtube_video


def _studio_with_scene(scene_rgb):
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._scene_capture_lock = threading.Lock()
    studio._scene_source = "youtube"
    studio._scene_capture_image = Image.fromarray(scene_rgb, "RGB")
    studio._youtube_scene_raw_image = None
    studio._scene_face_detector = None
    studio._default_avatar_face_frame = lambda: None
    return studio


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_low_lag_fast_path_requires_existing_avatar_face():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio.low_lag_scene_var = _FakeVar(True)
    studio._last_streamer_face_frame = None

    should_fast_path = (
        studio._last_streamer_face_frame is not None
        and studio.low_lag_scene_var.get()
    )

    assert not should_fast_path


def test_scene_uses_presenter_region_before_static_character_fallback():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1180:, :, :] = (220, 20, 20)
    avatar = np.zeros((512, 512, 3), dtype=np.uint8)
    avatar[:, :, :] = (70, 130, 190)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: None
    studio._default_avatar_face_frame = lambda: avatar

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    mean = bottom[8:].mean(axis=(0, 1))
    assert mean[2] > 200
    assert mean[2] > mean[1] * 4
    assert not np.allclose(bottom.mean(axis=(0, 1)), (76, 138, 200), atol=20)


def test_avatar_face_slot_uses_detected_face_box():
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    frame[:, :, :] = (5, 5, 5)
    frame[180:300, 300:420, :] = (80, 130, 190)
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._detect_avatar_face_box = lambda _frame: (300, 180, 420, 300)

    out = studio._avatar_face_slot(frame, 900)

    assert out.shape == (900, avatar_studio.TIKTOK_PORTRAIT_W, 3)
    assert out[:, :, 0].max() > 70


def test_scene_image_bgr_uses_cached_bgr_frame_without_pil_roundtrip():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    image = Image.new("RGB", (4, 4), (1, 2, 3))
    cached = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
    studio._scene_capture_lock = threading.Lock()
    studio._scene_capture_image = image
    studio._scene_capture_bgr = cached
    studio._scene_source = "youtube"
    studio._youtube_scene_raw_image = None

    bgr, returned_image = studio._scene_image_bgr()

    assert bgr is cached
    assert returned_image is image


def test_scene_image_bgr_uses_direct_youtube_frame_when_cache_is_black():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    visible_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    visible_rgb[:, :, :] = (20, 180, 240)

    class FakeYouTubeScene:
        def frame_snapshot(self):
            return 2, visible_rgb

    studio._scene_capture_lock = threading.Lock()
    studio._scene_capture_image = Image.new("RGB", (4, 4), (0, 0, 0))
    studio._scene_capture_bgr = black
    studio._scene_source = "youtube"
    studio._youtube_scene_raw_image = None
    studio._youtube_scene = FakeYouTubeScene()

    bgr, returned_image = studio._scene_image_bgr()

    assert returned_image.size == (4, 4)
    assert bgr[:, :, 0].max() == 240
    assert bgr[:, :, 1].max() == 180
    assert bgr[:, :, 2].max() == 20


def test_scene_image_bgr_uses_direct_youtube_frame_when_cached_bgr_missing():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    visible_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    visible_rgb[:, :, :] = (90, 120, 210)

    class FakeYouTubeScene:
        def frame_snapshot(self):
            return 3, visible_rgb

    studio._scene_capture_lock = threading.Lock()
    studio._scene_capture_image = Image.new("RGB", (4, 4), (0, 0, 0))
    studio._scene_capture_bgr = None
    studio._scene_source = "youtube"
    studio._youtube_scene_raw_image = None
    studio._youtube_scene = FakeYouTubeScene()

    bgr, returned_image = studio._scene_image_bgr()

    assert returned_image.size == (4, 4)
    assert bgr[:, :, 0].max() == 210
    assert bgr[:, :, 1].max() == 120
    assert bgr[:, :, 2].max() == 90


def test_portrait_scene_uses_source_presenter_before_avatar_fallback():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)      # chart
    scene[1180:, :, :] = (220, 20, 20)     # bad table/ad content
    avatar = np.zeros((512, 512, 3), dtype=np.uint8)
    avatar[:, :, :] = (70, 130, 190)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: None

    out = studio._scene_portrait_frame(avatar)

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    top = out[: avatar_studio.TIKTOK_CHART_H]
    bottom = out[avatar_studio.TIKTOK_CHART_H :]
    assert top.mean() > 25
    assert not np.allclose(top.mean(axis=(0, 1)), (76, 138, 200), atol=20)
    mean = bottom[8:].mean(axis=(0, 1))
    assert mean[2] > 200
    assert mean[2] > mean[1] * 4
    assert not np.allclose(bottom.mean(axis=(0, 1)), (76, 138, 200), atol=20)


def test_portrait_scene_without_avatar_forces_lower_presenter_region():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1180:, :, :] = (80, 130, 190)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: None

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    assert np.allclose(bottom[8:].mean(axis=(0, 1)), (201, 138, 86), atol=12)
    assert not np.allclose(bottom.mean(axis=(0, 1)), (20, 40, 80), atol=20)


def test_portrait_scene_split_does_not_run_face_detector():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1180:, :, :] = (80, 130, 190)

    studio = _studio_with_scene(scene)

    def fail_detector(_image):
        raise AssertionError("portrait split should not call face detector")

    studio._detect_presenter_face_box = fail_detector

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[8:].mean() > 100


def test_low_lag_youtube_uses_raw_frame_not_bad_crop():
    cropped = np.zeros((900, 1080, 3), dtype=np.uint8)
    cropped[:, :, :] = (20, 40, 80)
    raw = np.zeros((1920, 1080, 3), dtype=np.uint8)
    raw[:1180, :, :] = (20, 40, 80)
    raw[1180:, :, :] = (80, 130, 190)

    studio = _studio_with_scene(cropped)
    studio._youtube_scene_raw_image = Image.fromarray(raw, "RGB")
    studio.low_lag_scene_var = _FakeVar(True)

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[8:].mean() > 100
    assert np.allclose(bottom[8:].mean(axis=(0, 1)), (201, 138, 86), atol=12)


def test_low_lag_youtube_crops_decoder_pillarbox_before_split():
    frame = np.zeros((608, 1080, 3), dtype=np.uint8)
    x1, x2 = 369, 711
    frame[:374, x1:x2, :] = (20, 40, 80)
    frame[374:, x1:x2, :] = (80, 130, 190)

    studio = _studio_with_scene(frame)
    studio.low_lag_scene_var = _FakeVar(True)

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[8:].mean() > 100
    assert np.allclose(bottom[8:].mean(axis=(0, 1)), (201, 138, 86), atol=12)


def test_low_lag_landscape_youtube_finds_bottom_center_presenter():
    scene = np.zeros((608, 1080, 3), dtype=np.uint8)
    scene[:, :, :] = (6, 8, 10)
    scene[40:390, 0:620, :] = (20, 40, 80)
    scene[350:600, 360:720, :] = (210, 170, 130)

    studio = _studio_with_scene(scene)
    studio._youtube_scene_raw_image = Image.fromarray(scene, "RGB")
    studio.low_lag_scene_var = _FakeVar(True)
    studio._detect_presenter_face_box = lambda _image: (_ for _ in ()).throw(
        AssertionError("low-lag scene should not run detector"))

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[8:, :, 2].mean() > 150
    assert bottom[8:, :, 2].mean() > bottom[8:, :, 0].mean() * 1.4


def test_detected_presenter_face_overrides_avatar_fallback():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1180:, :, :] = (60, 70, 90)
    scene[1320:1520, 420:620, :] = (210, 170, 130)
    avatar = np.zeros((512, 512, 3), dtype=np.uint8)
    avatar[:, :, :] = (20, 220, 20)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: (430, 1330, 610, 1510)

    out = studio._scene_portrait_frame(avatar)
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    assert bottom[8:].mean() > 70
    assert not np.allclose(bottom.mean(axis=(0, 1)), (20, 220, 20), atol=35)


def test_bottom_face_length_dropdown_changes_face_slot_height():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1180:, :, :] = (80, 130, 190)

    studio = _studio_with_scene(scene)
    studio.face_strip_var = _FakeVar("Short face only")
    studio._detect_presenter_face_box = lambda _image: (430, 1330, 610, 1510)

    out = studio._scene_portrait_frame()
    chart_h = avatar_studio.TIKTOK_PORTRAIT_H - avatar_studio.FACE_STRIP_PRESETS["Short face only"]
    bottom = out[chart_h:]

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    assert bottom.shape[0] == avatar_studio.FACE_STRIP_PRESETS["Short face only"]
    assert bottom[8:].mean() > 45


def test_portrait_fast_split_uses_presenter_region_not_detector_crop():
    scene = np.zeros((1920, 1080, 3), dtype=np.uint8)
    scene[:1180, :, :] = (20, 40, 80)
    scene[1525:1840, 360:720, :] = (20, 220, 20)
    scene[1320:1520, 430:610, :] = (210, 170, 130)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: (_ for _ in ()).throw(
        AssertionError("portrait fast split should not run detector"))

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[:, :, 1].max() > 180
    assert bottom[:, :, 1].mean() > bottom[:, :, 2].mean()


def test_landscape_scene_matches_poster_chart_crop_style():
    scene = np.zeros((540, 960, 3), dtype=np.uint8)
    scene[:, :, :] = (5, 5, 5)
    cx, cy, cw, ch = avatar_studio.STREAMER_CHART_CROP
    scene[int(540 * cy) : int(540 * (cy + ch)), int(960 * cx) : int(960 * (cx + cw)), :] = (30, 60, 120)
    avatar = np.zeros((512, 512, 3), dtype=np.uint8)
    avatar[:, :, :] = (80, 120, 160)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: None

    out = studio._scene_portrait_frame(avatar)
    top = out[: avatar_studio.TIKTOK_CHART_H]
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    assert top[:, :, 0].mean() > top[:, :, 1].mean()
    assert bottom[:, :, 0].mean() > 70


def test_landscape_source_without_avatar_forces_presenter_region():
    scene = np.zeros((1080, 1920, 3), dtype=np.uint8)
    scene[:, :, :] = (4, 4, 4)
    scene[int(1080 * 0.06) : int(1080 * 0.88), : int(1920 * 0.74), :] = (25, 70, 130)
    x1 = int(1920 * avatar_studio.STREAMER_FACE_CROP[0])
    y1 = int(1080 * avatar_studio.STREAMER_FACE_CROP[1])
    x2 = int(1920 * (avatar_studio.STREAMER_FACE_CROP[0] + avatar_studio.STREAMER_FACE_CROP[2]))
    y2 = int(1080 * (avatar_studio.STREAMER_FACE_CROP[1] + avatar_studio.STREAMER_FACE_CROP[3]))
    scene[y1:y2, x1:x2, :] = (170, 125, 85)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: None

    out = studio._scene_portrait_frame()
    top = out[: avatar_studio.TIKTOK_CHART_H]
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert out.shape == (
        avatar_studio.TIKTOK_PORTRAIT_H,
        avatar_studio.TIKTOK_PORTRAIT_W,
        3,
    )
    assert top[:, :, 0].mean() > 100
    assert bottom[8:].mean() > 115
    assert bottom[8:, :, 2].mean() > bottom[8:, :, 1].mean()


def test_empty_face_slot_has_no_waiting_text():
    slot = avatar_studio.AvatarStudio._empty_face_slot(480)

    assert slot.shape == (480, avatar_studio.TIKTOK_PORTRAIT_W, 3)
    assert slot.max() == 8


def test_landscape_youtube_pip_crop_excludes_lower_side_panel():
    scene = np.zeros((360, 640, 3), dtype=np.uint8)
    scene[:, :, :] = (5, 5, 5)
    scene[20:225, 0:390, :] = (25, 70, 130)      # chart
    scene[225:360, 0:476, :] = (230, 20, 25)     # unwanted side panel
    scene[220:350, 476:640, :] = (185, 145, 95)  # presenter camera

    studio = _studio_with_scene(scene)
    studio.low_lag_scene_var = _FakeVar(True)
    studio._detect_presenter_face_box = lambda _image: (_ for _ in ()).throw(
        AssertionError("low-lag scene should not run detector"))

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]
    mean = bottom[8:].mean(axis=(0, 1))

    assert mean[2] > 155
    assert mean[1] > 115
    assert mean[0] < 125
    assert not np.allclose(mean, (25, 20, 230), atol=35)


def test_landscape_source_uses_locked_face_crop_not_detector_jitter():
    scene = np.zeros((1080, 1920, 3), dtype=np.uint8)
    scene[:, :, :] = (4, 4, 4)
    fx, fy, fw, fh = avatar_studio.STREAMER_FACE_CROP
    scene[int(1080 * fy) : int(1080 * (fy + fh)),
          int(1920 * fx) : int(1920 * (fx + fw)), :] = (180, 120, 70)
    scene[650:760, 200:320, :] = (20, 220, 20)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: (200, 650, 320, 760)

    out = studio._scene_portrait_frame()
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[:, :, 1].max() > 180
    assert bottom[:, :, 1].mean() > bottom[:, :, 2].mean()
    assert not np.allclose(bottom[8:].mean(axis=(0, 1)), (180, 120, 70), atol=40)


def test_landscape_detected_face_overrides_avatar_fallback():
    scene = np.zeros((1080, 1920, 3), dtype=np.uint8)
    scene[:, :, :] = (4, 4, 4)
    scene[int(1080 * 0.06) : int(1080 * 0.88), : int(1920 * 0.74), :] = (25, 70, 130)
    scene[330:510, 1440:1620, :] = (190, 130, 80)
    avatar = np.zeros((512, 512, 3), dtype=np.uint8)
    avatar[:, :, :] = (20, 220, 20)

    studio = _studio_with_scene(scene)
    studio._detect_presenter_face_box = lambda _image: (1440, 330, 1620, 510)

    out = studio._scene_portrait_frame(avatar)
    bottom = out[avatar_studio.TIKTOK_CHART_H :]

    assert bottom[8:].mean() > 40
    assert bottom[8:, :, 1].mean() > 35
    assert bottom[8:, :, 2].mean() > 30
    assert not np.allclose(bottom.mean(axis=(0, 1)), (20, 220, 20), atol=35)


def test_youtube_scene_prefers_hd_bounded_video_sources():
    info = {
        "formats": [
            {"url": "low", "height": 480, "width": 854, "vcodec": "avc1", "fps": 30, "tbr": 1200},
            {"url": "hd", "height": 720, "width": 1280, "vcodec": "avc1", "fps": 30, "tbr": 3000},
            {"url": "full-hd-too-heavy", "height": 1080, "width": 1920, "vcodec": "avc1", "fps": 30, "tbr": 5000},
            {"url": "qhd-too-heavy", "height": 1440, "width": 2560, "vcodec": "avc1", "fps": 30, "tbr": 9000},
            {"url": "uhd", "height": 2160, "width": 3840, "vcodec": "avc1", "fps": 30, "tbr": 14000},
            {"url": "too-high", "height": 4320, "width": 7680, "vcodec": "avc1", "fps": 30, "tbr": 28000},
        ]
    }

    assert youtube_video._select_video_source(info) == "hd"


def test_youtube_decoder_preserves_landscape_source_before_portrait_composition():
    assert youtube_video.VIDEO_WIDTH == 1280
    assert youtube_video.VIDEO_HEIGHT == 720


def test_default_scene_layout_uses_taller_face_and_wider_chart_context():
    assert avatar_studio.TIKTOK_FACE_H == 960
    assert avatar_studio.TIKTOK_CHART_H == 1380
    assert avatar_studio.STREAMER_CHART_CROP[0] == 0.0
    assert avatar_studio.STREAMER_CHART_CROP[2] >= 0.72
    assert avatar_studio.STREAMER_CHART_CROP[3] >= 0.84


def test_cover_resize_can_bias_crop_to_show_right_side_without_squeeze():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    frame[:, :100, :] = (255, 0, 0)
    frame[:, 100:200, :] = (0, 255, 0)
    frame[:, 200:, :] = (0, 0, 255)

    centered = avatar_studio.AvatarStudio._cover_resize_bgr(
        frame, (100, 100), crop_x=0.5)
    right_biased = avatar_studio.AvatarStudio._cover_resize_bgr(
        frame, (100, 100), crop_x=1.0)

    assert centered[:, :, 1].mean() > centered[:, :, 2].mean()
    assert right_biased[:, :, 2].mean() > right_biased[:, :, 1].mean()


def test_scene_text_overlay_draws_into_frame_when_enabled():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._scene_text_enabled = True
    studio._scene_text = "BUY NOW"
    studio._scene_text_font = "Arial"
    studio._scene_text_size = 72
    studio._scene_text_color = "#ffffff"
    studio._scene_text_bg = "#000000"
    studio._scene_text_outline = "#00e5ff"
    studio._scene_text_behavior = "Static"
    studio._scene_text_position = "Top"
    studio._scene_text_opacity = 80
    frame = np.zeros((avatar_studio.TIKTOK_PORTRAIT_H,
                      avatar_studio.TIKTOK_PORTRAIT_W, 3), dtype=np.uint8)

    out = studio._apply_scene_text_overlay(frame)

    assert out.shape == frame.shape
    assert out[:220].max() > 120
    assert out[300:].max() == 0


def test_scene_text_overlay_draws_multiple_text_items():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._scene_text_enabled = False
    studio._scene_text = ""
    studio._scene_text_items = [
        {
            "enabled": True,
            "text": "TOP NOTE",
            "font": "Arial",
            "size": 56,
            "color": "#ffffff",
            "bg": "#000000",
            "outline": "#00e5ff",
            "behavior": "Static",
            "position": "Top",
            "opacity": 80,
        },
        {
            "enabled": True,
            "text": "BOTTOM NOTE",
            "font": "Arial",
            "size": 56,
            "color": "#ffffff",
            "bg": "#000000",
            "outline": "#ff3366",
            "behavior": "Static",
            "position": "Bottom",
            "opacity": 80,
        },
    ]
    frame = np.zeros((avatar_studio.TIKTOK_PORTRAIT_H,
                      avatar_studio.TIKTOK_PORTRAIT_W, 3), dtype=np.uint8)

    out = studio._apply_scene_text_overlay(frame)

    assert out[:220].max() > 120
    assert out[-220:].max() > 120
    assert out[500:1700].max() == 0


def test_scene_text_overlay_uses_free_xy_position():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._scene_text_enabled = False
    studio._scene_text = ""
    studio._scene_text_items = [{
        "enabled": True,
        "text": "MOVE",
        "font": "Arial",
        "size": 56,
        "color": "#ffffff",
        "bg": "#000000",
        "outline": "#00e5ff",
        "behavior": "Static",
        "position": "Top",
        "opacity": 80,
        "x": 80,
        "y": 70,
    }]
    frame = np.zeros((avatar_studio.TIKTOK_PORTRAIT_H,
                      avatar_studio.TIKTOK_PORTRAIT_W, 3), dtype=np.uint8)

    out = studio._apply_scene_text_overlay(frame)

    assert out[:500].max() == 0
    assert out[1400:1800, 700:1050].max() > 120


def test_scene_text_overlay_size_changes_rendered_area():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    base = {
        "enabled": True,
        "text": "SIZE",
        "font": "Arial",
        "color": "#ffffff",
        "bg": "#000000",
        "outline": "#00e5ff",
        "behavior": "Static",
        "position": "Top",
        "opacity": 80,
        "x": 50,
        "y": 20,
    }
    frame = np.zeros((avatar_studio.TIKTOK_PORTRAIT_H,
                      avatar_studio.TIKTOK_PORTRAIT_W, 3), dtype=np.uint8)
    studio._scene_text_enabled = False
    studio._scene_text = ""
    studio._scene_text_items = [dict(base, size=36)]
    small = studio._apply_scene_text_overlay(frame)
    studio._scene_text_items = [dict(base, size=108)]
    large = studio._apply_scene_text_overlay(frame)

    assert np.count_nonzero(large) > np.count_nonzero(small)


def test_scene_text_overlay_disabled_leaves_frame_unchanged():
    studio = avatar_studio.AvatarStudio.__new__(avatar_studio.AvatarStudio)
    studio._scene_text_enabled = False
    frame = np.full((240, 320, 3), 17, dtype=np.uint8)

    out = studio._apply_scene_text_overlay(frame)

    assert out is frame


def test_live_preview_letterboxes_portrait_without_stretching():
    image = Image.new("RGB", (1080, 1920), (200, 100, 50))

    out = avatar_studio.AvatarStudio._letterbox_image(image, (800, 800))

    assert out.size == (800, 800)
    # 9:16 portrait inside an 800x800 square should become 450x800,
    # leaving black side bars instead of stretching to square.
    assert out.getpixel((0, 400)) == (0, 0, 0)
    assert out.getpixel((799, 400)) == (0, 0, 0)
    assert out.getpixel((400, 400)) == (200, 100, 50)


def test_live_preview_default_is_phone_shaped_not_square():
    ratio = avatar_studio.PREVIEW_W / avatar_studio.PREVIEW_H

    assert ratio < 0.5
    assert avatar_studio.PREVIEW_H > avatar_studio.PREVIEW_W
