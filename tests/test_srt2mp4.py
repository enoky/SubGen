from fractions import Fraction
from pathlib import Path

import pytest

import srt2mp4 as s


def clip(name="clip1.mp4", codec="h264", w=1280, h=704, fps=24, pix="yuv420p", dur=10.0):
    return s.ClipInfo(Path(name), codec, w, h, Fraction(fps), pix, dur)


GPU_CAPS = s.Caps(
    encoders=frozenset({"hevc_nvenc", "h264_nvenc", "libx264"}),
    decoders=frozenset({"h264_cuvid"}),
    filters=frozenset({"scale_cuda", "overlay_cuda", "hwupload"}),
    scale_cuda_opts=frozenset({"format", "interp_algo"}),
    gpu=True,
)
CPU_CAPS = s.Caps(frozenset({"libx264"}), frozenset(), frozenset(), frozenset(), gpu=False)


def test_natural_sort(tmp_path):
    for name in ["clip10.mp4", "clip2.mp4", "Clip1.mp4", "notes.txt", "clip3.MP4"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in s.find_clips(tmp_path)] == ["Clip1.mp4", "clip2.mp4", "clip3.MP4", "clip10.mp4"]


def test_find_clips_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        s.find_clips(tmp_path)


@pytest.mark.parametrize(
    ("size", "box"),
    [
        ((1280, 704), (1252, 704, 14, 0)),  # slightly wider than 16:9: trim the sides
        ((1920, 1080), (1920, 1080, 0, 0)),  # already 16:9
        ((1080, 1080), (1080, 608, 0, 236)),  # taller: trim top and bottom
    ],
)
def test_crop_box(size, box):
    assert s.crop_box(*size) == box


def test_plan_pads_only_beyond_one_frame():
    clips = [clip(dur=10.0), clip("clip2.mp4", dur=10.0)]
    assert s.plan_clips(clips, 20.03).pad_seconds == 0.0
    assert s.plan_clips(clips, 21.0).pad_seconds == pytest.approx(1.0)
    assert s.plan_clips(clips, 19.0).pad_seconds == 0.0  # longer clips are just cut
    assert s.plan_clips(clips, 19.0).end_frame == 19 * 24


def test_attempt_order_on_gpu():
    plan = s.plan_clips([clip(), clip("clip2.mp4")], 20.0)
    assert s.clip_attempts(plan, "auto", GPU_CAPS) == [
        ("gpu", "hevc_nvenc"), ("gpu", "h264_nvenc"),
        ("hybrid", "hevc_nvenc"), ("hybrid", "h264_nvenc"),
        ("cpu", "libx264"),
    ]


def test_attempts_skip_full_gpu_when_padding_or_mixed():
    padded = s.plan_clips([clip()], 15.0)
    assert ("gpu", "hevc_nvenc") not in s.clip_attempts(padded, "auto", GPU_CAPS)
    mixed = s.plan_clips([clip(), clip("b.mp4", w=1920, h=1080)], 20.0)
    assert not mixed.uniform
    assert s.clip_attempts(mixed, "auto", GPU_CAPS)[0] == ("hybrid", "hevc_nvenc")


def test_attempts_without_gpu():
    plan = s.plan_clips([clip()], 10.0)
    assert s.clip_attempts(plan, "auto", CPU_CAPS) == [("cpu", "libx264")]


def build(plan, mode, enc="hevc_nvenc"):
    return s.build_clip_command(plan, mode, enc, Path("clips.txt"), Path("cards.txt"), Path("n.wav"), Path("o.mp4"), GPU_CAPS)


def test_gpu_command():
    cmd = build(s.plan_clips([clip(), clip("clip2.mp4")], 20.0), "gpu")
    joined = " ".join(cmd)
    assert "-c:v h264_cuvid -crop 0x0x14x14" in joined
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale_cuda=w=1920:h=1080:interp_algo=lanczos:format=yuv420p" in graph
    assert "overlay_cuda" in graph
    assert cmd[cmd.index("-map") + 3] == "2:a:0"  # narration audio, never the clips'
    assert "-tag:v hvc1" in joined and "-movflags +faststart" in joined
    assert "-g 48" in joined and "-b:a 384k" in joined and "-ar 48000" in joined


def test_gpu_command_rejects_padding():
    with pytest.raises(ValueError):
        build(s.plan_clips([clip()], 15.0), "gpu")


def test_hybrid_command_pads_and_crops():
    cmd = build(s.plan_clips([clip()], 12.0), "hybrid")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert cmd[cmd.index("-i") - 2:cmd.index("-i")] == ["-safe", "0"]
    assert "-hwaccel" in cmd
    assert "crop=1252:704:14:0,scale=1920:1080:flags=lanczos" in graph
    assert "tpad=stop_mode=clone:stop_duration=2.000000" in graph


def test_mixed_clips_use_concat_filter():
    plan = s.plan_clips([clip(), clip("b.mp4", w=1920, h=1080, fps=30, dur=5)], 15.0)
    cmd = build(plan, "cpu", "libx264")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "-hwaccel" not in cmd
    assert "concat=n=2:v=1:a=0" in graph and "[2:v]format=rgba" in graph
    assert cmd[cmd.index("-map") + 3] == "3:a:0"
    assert "fps=24" in graph  # first clip's frame rate wins


def test_transparent_card_has_outline():
    font = s.ImageFont.truetype(str(s.FONT_PATH), s.VIDEO_STYLE.font_size)
    img = s.render_card("Hello there", font, s.VIDEO_STYLE, transparent=True)
    assert img.mode == "RGBA"
    alpha = img.getchannel("A")
    assert alpha.getpixel((0, 0)) == 0  # background is clear
    opaque = [px for _, px in img.getcolors(maxcolors=1 << 24) if px[3] == 255]
    assert any(px[:3] == (255, 255, 255) for px in opaque)  # white fill
    assert any(px[:3] == (0, 0, 0) for px in opaque)  # black outline


def test_blue_card_unchanged_mode():
    font = s.ImageFont.truetype(str(s.FONT_PATH), s.STYLE.font_size)
    img = s.render_card("Hi", font)
    assert img.mode == "RGB" and img.getpixel((0, 0)) == s.STYLE.bg
