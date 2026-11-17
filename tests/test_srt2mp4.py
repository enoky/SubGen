from fractions import Fraction
from pathlib import Path

import pytest

import srt2mp4 as s


def clip(name="clip1.mp4", codec="h264", w=1280, h=704, fps=24, pix="yuv420p", dur=10.0):
    return s.ClipInfo(Path(name), codec, w, h, Fraction(fps), pix, dur)


GPU_CAPS = s.Caps(
    encoders=frozenset({"hevc_nvenc", "h264_nvenc", "libx264"}),
    decoders=frozenset({"h264_cuvid"}),
    filters=frozenset({"scale_cuda", "overlay_cuda", "hwupload", "hwdownload"}),
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
        ((960, 512), (910, 512, 24, 0)),  # odd margin (25 px) rounded to an even offset
        ((1920, 1080), (1920, 1080, 0, 0)),  # already 16:9
        ((1080, 1080), (1080, 608, 0, 236)),  # taller: trim top and bottom
    ],
)
def test_crop_box(size, box):
    assert s.crop_box(*size) == box


def test_hold_only_beyond_one_frame():
    clips = [clip(dur=10.0), clip("clip2.mp4", dur=10.0)]
    assert s.hold_seconds(clips, 20.03) == 0.0
    assert s.hold_seconds(clips, 21.0) == pytest.approx(1.0)
    assert s.hold_seconds(clips, 19.0) == 0.0  # longer clips are just cut
    assert s.plan_clips(clips, 19.0).end_frame == 19 * 24


def test_majority_signature_ignores_odd_clip():
    clips = [clip("b.mp4", w=1920, h=1080), clip(), clip("c.mp4")]
    assert s.majority_signature(clips) == ("h264", 1280, 704, Fraction(24), "yuv420p")


def test_attempt_order_on_gpu():
    plan = s.plan_clips([clip(), clip("clip2.mp4")], 20.0)
    assert s.clip_attempts(plan, "auto", GPU_CAPS) == [
        ("gpu_overscan", "hevc_nvenc"), ("gpu_overscan", "h264_nvenc"),
        ("gpu_overscan_canvas", "hevc_nvenc"), ("gpu_overscan_canvas", "h264_nvenc"),
        ("gpu_overscan_download", "hevc_nvenc"), ("gpu_overscan_download", "h264_nvenc"),
        ("gpu_upload", "hevc_nvenc"), ("gpu_upload", "h264_nvenc"),
        ("hybrid", "hevc_nvenc"), ("hybrid", "h264_nvenc"),
        ("cpu", "libx264"),
    ]


def test_cpu_filters_skips_gpu_routes():
    plan = s.plan_clips([clip()], 20.0)
    assert s.clip_attempts(plan, "auto", GPU_CAPS, cpu_filters=True) == [
        ("hybrid", "hevc_nvenc"), ("hybrid", "h264_nvenc"), ("cpu", "libx264"),
    ]


def test_attempts_without_gpu():
    plan = s.plan_clips([clip()], 10.0)
    assert s.clip_attempts(plan, "auto", CPU_CAPS) == [("cpu", "libx264")]


def build(plan, mode, enc="hevc_nvenc"):
    return s.build_clip_command(plan, mode, enc, Path("clips.txt"), Path("cards.txt"), Path("n.wav"), Path("o.mp4"), GPU_CAPS)


def test_gpu_overscan_command():
    cmd = build(s.plan_clips([clip(), clip("clip2.mp4")], 20.0), "gpu_overscan")
    joined = " ".join(cmd)
    assert "-c:v h264_cuvid -crop 0x0x14x14" in joined
    graph = cmd[cmd.index("-filter_complex") + 1]
    # scaled 1.5% larger and placed at a negative offset on a 1080p canvas: the border
    # is cut off on the GPU, and frames never come back to the CPU
    assert "[b]scale_cuda=w=1948:h=1096:interp_algo=lanczos:format=yuv420p[big]" in graph
    assert "[a]scale_cuda=w=1920:h=1080:interp_algo=nearest:format=yuv420p[canvas]" in graph
    assert "[canvas][big]overlay_cuda=x=-14:y=-8[bg]" in graph
    assert "[bg][subs]overlay_cuda=x=0:y=0[v]" in graph
    assert "hwdownload" not in graph
    assert cmd[cmd.index("-map") + 3] == "2:a:0"  # narration audio, never the clips'
    assert "-tag:v hvc1" in joined and "-movflags +faststart" in joined
    assert "-g 48" in joined and "-b:a 384k" in joined and "-ar 48000" in joined
    assert joined.count("-i ") == 3  # clips list, cards list, narration: never one input per clip


def test_gpu_overscan_canvas_uploads_one_frame():
    cmd = build(s.plan_clips([clip()], 20.0), "gpu_overscan_canvas")
    graph = cmd[cmd.index("-filter_complex") + 1]
    lavfi = cmd[cmd.index("lavfi") + 2]
    assert lavfi.startswith("color=c=black:s=1920x1080:r=24:d=0.041667")
    assert cmd[cmd.index("-map") + 3] == "2:a:0"  # the canvas input comes after the narration
    assert "[3:v]format=yuv420p,hwupload,loop=loop=-1:size=1[canvas]" in graph
    assert "[canvas][big]overlay_cuda=x=-14:y=-8:shortest=1[bg]" in graph
    assert "hwdownload" not in graph and "split" not in graph


def test_gpu_overscan_download_fallback_crops_on_cpu():
    graph = build(s.plan_clips([clip()], 20.0), "gpu_overscan_download")
    graph = graph[graph.index("-filter_complex") + 1]
    assert "overlay_cuda=x=14:y=8,hwdownload,format=yuv420p,crop=1920:1080:14:8[v]" in graph


def test_overscan_margins_keep_aspect():
    assert s.overscan_margins(1920, 1080) == (14, 8)


def test_command_rejects_mixed_clips():
    with pytest.raises(ValueError):
        build(s.plan_clips([clip(), clip("b.mp4", w=1920, h=1080)], 15.0), "gpu_overscan")


def test_gpu_crop_for_960x512():
    plan = s.plan_clips([clip(w=960, h=512), clip("clip2.mp4", w=960, h=512)], 20.0)
    assert "-crop 0x0x24x26" in " ".join(build(plan, "gpu_overscan"))


def test_gpu_upload_command_crops_on_cpu_then_scales_on_gpu():
    cmd = build(s.plan_clips([clip(w=960, h=512)], 12.0), "gpu_upload")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "-crop" not in cmd and "h264_cuvid" not in cmd
    assert "-hwaccel_output_format" not in cmd  # decoded frames come back for the cheap crop
    assert graph.startswith("[0:v]crop=910:512:24:0,format=nv12,hwupload,scale_cuda=w=1920:h=1080")
    assert "overlay_cuda" in graph


def yuv_frame(w, h, y=100, u=120, v=130, bottom_uv=None):
    luma = bytes([y]) * (w * h)
    cw, ch = w // 2, h // 2

    def plane(val):
        rows = [bytes([val]) * cw for _ in range(ch)]
        if bottom_uv is not None:
            rows[-1] = bytes([bottom_uv]) * cw
        return b"".join(rows)

    return luma + plane(u) + plane(v)


def test_edge_defect_spots_a_green_bottom_line():
    w, h = 64, 36
    ref = yuv_frame(w, h)
    assert s.edge_defect(ref, ref, w, h) == 0
    assert s.edge_defect(yuv_frame(w, h, u=123), ref, w, h) == 0  # uniform small difference is fine
    green = yuv_frame(w, h, bottom_uv=0)  # zeroed chroma shows as green
    assert s.edge_defect(green, ref, w, h) > s.EDGE_DEFECT_LIMIT


def test_verify_routes_keeps_first_clean_gpu_route(tmp_path, monkeypatch):
    plan = s.ClipPlan([clip(w=960, h=512, dur=2.0)], Fraction(24), 48, 64, 36)
    attempts = s.clip_attempts(plan, "auto", GPU_CAPS)
    frames = {"reference_cpu_overscan.mp4": yuv_frame(64, 36), "reference_cpu.mp4": yuv_frame(64, 36),
              "gpu_overscan.mp4": yuv_frame(64, 36, bottom_uv=0),
              "gpu_overscan_canvas.mp4": yuv_frame(64, 36, bottom_uv=0),
              "gpu_overscan_download.mp4": yuv_frame(64, 36, bottom_uv=0), "gpu_upload.mp4": yuv_frame(64, 36)}
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(s, "decode_frame", lambda path, i, w, h: frames[Path(path).name])
    kept = s.verify_routes(attempts, plan, tmp_path, Path("n.wav"), GPU_CAPS)
    assert kept == [
        ("gpu_upload", "hevc_nvenc"), ("gpu_upload", "h264_nvenc"),
        ("hybrid", "hevc_nvenc"), ("hybrid", "h264_nvenc"), ("cpu", "libx264"),
    ]


def test_verify_routes_falls_back_to_cpu_filters(tmp_path, monkeypatch):
    plan = s.ClipPlan([clip(dur=2.0)], Fraction(24), 48, 64, 36)
    attempts = s.clip_attempts(plan, "auto", GPU_CAPS)
    bad = yuv_frame(64, 36, bottom_uv=0)
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(s, "decode_frame", lambda path, i, w, h: yuv_frame(64, 36) if "reference" in str(path) else bad)
    kept = s.verify_routes(attempts, plan, tmp_path, Path("n.wav"), GPU_CAPS)
    assert [m for m, _ in kept] == ["hybrid", "hybrid", "cpu"]


def test_hybrid_command_crops():
    cmd = build(s.plan_clips([clip()], 12.0), "hybrid")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "-hwaccel" in cmd and "-hwaccel_output_format" not in cmd  # frames come back to the CPU
    assert "-filter_complex_threads" in cmd
    assert "crop=1252:704:14:0,scale=1920:1080:flags=lanczos" in graph


def test_cpu_command_has_no_hwaccel():
    cmd = build(s.plan_clips([clip()], 12.0), "cpu", "libx264")
    assert "-hwaccel" not in cmd and "libx264" in cmd


def test_probe_cache_roundtrip(tmp_path, monkeypatch):
    files = []
    for i in range(3):
        f = tmp_path / f"clip{i}.mp4"
        f.write_bytes(b"x" * (i + 1))
        files.append(f)
    calls = []
    monkeypatch.setattr(s, "probe_clip", lambda p: calls.append(p) or clip(str(p), dur=1.5))
    cache = tmp_path / "cache.json"
    first = s.probe_clips(files, cache)
    assert len(calls) == 3 and cache.is_file()
    second = s.probe_clips(files, cache)
    assert len(calls) == 3  # served from the cache
    assert second == first and [c.path for c in second] == files
    files[1].write_bytes(b"changed")
    s.probe_clips(files, cache)
    assert calls[-1] == files[1] and len(calls) == 4  # only the changed file is probed again


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
