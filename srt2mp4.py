"""Render subtitles as a text video, muxed with the narration audio.

Each subtitle is drawn once as a 1920x1080 PNG (word-wrapped, centered, white
Yatra One text with a drop shadow), then ffmpeg's concat demuxer holds each
image for its subtitle's duration.

Two backgrounds are supported:

* Chroma blue (default): the cards are opaque and encoded directly.
* Video clips (``--clips DIR``): every ``*.mp4`` in DIR is concatenated in
  natural filename order, center-cropped to 16:9, scaled to 1920x1080, and the
  cards (now transparent, with a black outline for readability) are overlaid on
  top. The clips' own audio is dropped. On an NVIDIA GPU, decoding (NVDEC),
  scaling, overlay (CUDA) and encoding (NVENC) run on the GPU. Each GPU route is
  first checked on one second of video against a CPU render, and skipped if it
  leaves edge artifacts (some setups show a green line at the bottom); the
  fallback does crop/scale/overlay on all CPU cores, then everything on the CPU.
  All clips are read through one concat demuxer input, so thousands of clips
  are fine. Clips that differ from the majority format are re-encoded to match,
  and a still of the last frame is appended if the clips run short.

Usage:
    uv run python -m srt2mp4 subtitles.srt narration.wav out.mp4
    uv run python -m srt2mp4 subtitles.srt narration.wav out.mp4 --clips clips/
    uv run python -m srt2mp4 --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple, Protocol

import srt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from ffmpeg_progress import run_ffmpeg

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "YatraOne-Regular.ttf"
ENCODERS = ("auto", "h264_nvenc", "hevc_nvenc", "libx264")
NVENC_ENCODERS = ("hevc_nvenc", "h264_nvenc")


@dataclass(frozen=True)
class Style:
    font_size: int = 58
    line_height: int = 86  # baseline-to-baseline spacing in px
    max_width: int = 1520  # wrap width in px
    width: int = 1920
    height: int = 1080
    fps: int = 24
    bg: tuple[int, int, int] = (0x00, 0x47, 0xBB)  # chroma blue
    shadow_offset: int = 4
    shadow_blur: float = 2.0
    shadow_opacity: float = 0.85
    stroke_width: int = 0  # black outline around the glyphs, in px
    pix_fmt: str = "yuv420p"


STYLE = Style()
# Over busy footage a soft shadow alone is not enough contrast, so add a black outline.
VIDEO_STYLE = replace(STYLE, stroke_width=4, shadow_offset=5, shadow_blur=3.0, shadow_opacity=0.9)


class TimedText(Protocol):
    start: float
    end: float
    text: str


class Cue(NamedTuple):
    start: float
    end: float
    text: str


def parse_srt(path: Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    return [
        Cue(sub.start.total_seconds(), sub.end.total_seconds(), " ".join(sub.content.split()))
        for sub in srt.parse(raw)
    ]


def wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_card(
    text: str, font: ImageFont.FreeTypeFont, style: Style = STYLE, transparent: bool = False
) -> Image.Image:
    """Draw one subtitle card. ``transparent`` returns RGBA text over a clear background."""
    w, h = style.width, style.height
    lines = wrap(text, font, style.max_width - 2 * style.stroke_width) if text else []

    # Glyphs (fill) and glyphs-plus-outline (body) are drawn as alpha masks, then layered.
    fill_mask = Image.new("L", (w, h), 0)
    body_mask = Image.new("L", (w, h), 0)
    fill_draw, body_draw = ImageDraw.Draw(fill_mask), ImageDraw.Draw(body_mask)
    ascent, descent = font.getmetrics()
    block_h = (len(lines) - 1) * style.line_height + ascent + descent
    y = (h - block_h) / 2
    for line in lines:
        x = (w - font.getlength(line)) / 2
        fill_draw.text((x, y), line, font=font, fill=255)
        if style.stroke_width:
            body_draw.text((x, y), line, font=font, fill=255, stroke_width=style.stroke_width, stroke_fill=255)
        y += style.line_height
    if not style.stroke_width:
        body_mask = fill_mask

    shadow_mask = Image.new("L", (w, h), 0)
    shadow_mask.paste(body_mask, (style.shadow_offset, style.shadow_offset))
    if style.shadow_blur > 0:
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(style.shadow_blur))
    shadow_mask = shadow_mask.point(lambda v: int(v * style.shadow_opacity))

    def layer(color: tuple[int, int, int], mask: Image.Image) -> Image.Image:
        img = Image.new("RGBA", (w, h), (*color, 0))
        img.putalpha(mask)
        return img

    base = Image.new("RGBA", (w, h), (0, 0, 0, 0) if transparent else (*style.bg, 255))
    img = Image.alpha_composite(base, layer((0, 0, 0), shadow_mask))
    if style.stroke_width:
        img = Image.alpha_composite(img, layer((0, 0, 0), body_mask))
    img = Image.alpha_composite(img, layer((255, 255, 255), fill_mask))
    return img if transparent else img.convert("RGB")


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def concat_line(path: Path) -> str:
    """One ``file`` line for an ffconcat list, with single quotes escaped."""
    escaped = path.as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


# --------------------------------------------------------------------------- cards


def build_card_timeline(
    cues: Sequence[TimedText],
    fps: float | Fraction,
    total: float,
    work: Path,
    draw: Callable[[str], Image.Image],
) -> tuple[Path, int]:
    """Render one PNG per cue into ``work`` and write an ffconcat list holding each
    for its duration. Every boundary is snapped to a frame index so durations never
    drift from the audio. Returns the list path and the total frame count."""
    fps_f = float(fps)

    def frame(t: float) -> int:
        return round(t * fps_f)

    blank = work / "blank.png"
    jobs = [(blank, "")]
    entries, cursor = [], 0
    for i, cue in enumerate(cues, 1):
        f0, f1 = max(frame(cue.start), cursor), frame(cue.end)
        if f1 <= f0:
            continue
        if f0 > cursor:
            entries.append((blank, f0 - cursor))
        card = work / f"{i:05d}.png"
        jobs.append((card, cue.text))
        entries.append((card, f1 - f0))
        cursor = f1

    # Drawing and PNG compression mostly release the GIL, so threads help. These
    # files are temporary, so favor speed over size.
    def save(job: tuple[Path, str]) -> None:
        job_path, text = job
        draw(text).save(job_path, compress_level=1)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        list(tqdm(pool.map(save, jobs), total=len(jobs), desc="Rendering subtitle frames", unit="cue"))
    end_frame = max(frame(total), cursor)
    if end_frame > cursor:
        entries.append((blank, end_frame - cursor))
    if not entries:
        raise RuntimeError("Nothing to render: no subtitles and no audio duration.")

    concat = work / "cards.txt"
    with concat.open("w", encoding="utf-8") as fh:
        fh.write("ffconcat version 1.0\n")
        for path, frames in entries:
            fh.write(concat_line(path))
            fh.write(f"duration {frames / fps_f:.6f}\n")
        fh.write(concat_line(entries[-1][0]))  # concat demuxer needs the last file repeated
    return concat, end_frame


# --------------------------------------------------------------------------- clips


def natural_key(path: Path) -> list[int | str]:
    """Sort key so ``clip2`` comes before ``clip10``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_clips(clips: Path | Sequence[Path]) -> list[Path]:
    """A directory becomes its ``*.mp4`` files in natural order; a list is kept as given."""
    if isinstance(clips, (str, Path)):
        clips = Path(clips)
        if clips.is_dir():
            found = sorted((p for p in clips.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"), key=natural_key)
        else:
            found = [clips]
    else:
        found = [Path(p) for p in clips]
    if not found:
        raise FileNotFoundError(f"No .mp4 clips found in {clips}")
    missing = [p for p in found if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Clip not found: {missing[0]}")
    return found


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    codec: str
    width: int
    height: int
    fps: Fraction
    pix_fmt: str
    duration: float

    @property
    def signature(self) -> tuple[str, int, int, Fraction, str]:
        return self.codec, self.width, self.height, self.fps, self.pix_fmt


def probe_clip(path: Path) -> ClipInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream in {path}")
    s = data["streams"][0]
    return ClipInfo(
        path=Path(path),
        codec=s["codec_name"],
        width=int(s["width"]),
        height=int(s["height"]),
        fps=Fraction(s["r_frame_rate"]),
        pix_fmt=s.get("pix_fmt", ""),
        duration=float(data["format"]["duration"]),
    )


def _clip_to_dict(info: ClipInfo) -> dict:
    return {"codec": info.codec, "width": info.width, "height": info.height,
            "fps": str(info.fps), "pix_fmt": info.pix_fmt, "duration": info.duration}


def _clip_from_dict(path: Path, d: dict) -> ClipInfo:
    return ClipInfo(Path(path), d["codec"], int(d["width"]), int(d["height"]),
                    Fraction(d["fps"]), d["pix_fmt"], float(d["duration"]))


def probe_clips(paths: Sequence[Path], cache_path: Path | None = None, workers: int = 16) -> list[ClipInfo]:
    """Probe many clips in parallel (network drives are slow per file but fine in
    parallel). Results are cached in ``cache_path`` by file name, size and mtime,
    so a rerun over the same clips is instant."""
    cache: dict[str, dict] = {}
    if cache_path and Path(cache_path).is_file():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    def key(p: Path) -> str:
        st = p.stat()
        return f"{p.name}|{st.st_size}|{st.st_mtime_ns}"

    results: list[ClipInfo | None] = [None] * len(paths)
    todo = []
    for i, p in enumerate(paths):
        k = key(Path(p))
        if k in cache:
            results[i] = _clip_from_dict(p, cache[k])
        else:
            todo.append((i, Path(p), k))

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool, \
                tqdm(total=len(todo), desc="Checking clips", unit="clip") as bar:
            for (i, _, k), info in zip(todo, pool.map(lambda t: probe_clip(t[1]), todo)):
                results[i] = info
                cache[k] = _clip_to_dict(info)
                bar.update()
        if cache_path:
            try:
                Path(cache_path).write_text(json.dumps(cache), encoding="utf-8")
            except OSError as e:
                print(f"Could not save clip cache {cache_path}: {e}", flush=True)
    return results  # type: ignore[return-value]


def crop_box(width: int, height: int, out_w: int = 1920, out_h: int = 1080) -> tuple[int, int, int, int]:
    """Largest centered crop of ``width``x``height`` with the output's aspect ratio.

    Returns ``(crop_w, crop_h, x, y)`` with even sizes and offsets (4:2:0 chroma and
    NVDEC's crop need them). 1280x704 -> (1252, 704, 14, 0); 960x512 -> (910, 512, 24, 0).
    """
    target = Fraction(out_w, out_h)
    if Fraction(width, height) > target:
        cw, ch = 2 * round(height * target / 2), height
    else:
        cw, ch = width, 2 * round(width / target / 2)
    cw, ch = min(cw, width), min(ch, height)
    return cw, ch, (width - cw) // 4 * 2, (height - ch) // 4 * 2


def fps_arg(fps: Fraction) -> str:
    return str(fps.numerator) if fps.denominator == 1 else f"{fps.numerator}/{fps.denominator}"


# --------------------------------------------------------------------------- ffmpeg capabilities


@dataclass(frozen=True)
class Caps:
    encoders: frozenset[str]
    decoders: frozenset[str]
    filters: frozenset[str]
    scale_cuda_opts: frozenset[str]
    gpu: bool

    def cuda_filters(self) -> bool:
        """CUDA scale/overlay plus an NVENC encoder (needed by both GPU filter routes)."""
        return (
            self.gpu
            and {"scale_cuda", "overlay_cuda", "hwupload"} <= self.filters
            and "format" in self.scale_cuda_opts
            and any(e in self.encoders for e in NVENC_ENCODERS)
        )

    def full_gpu(self, codec: str) -> bool:
        """Can the clip path run end to end on the GPU for clips of ``codec``?"""
        return (
            self.gpu
            and f"{codec}_cuvid" in self.decoders
            and {"scale_cuda", "overlay_cuda", "hwupload"} <= self.filters
            and "format" in self.scale_cuda_opts
            and any(e in self.encoders for e in NVENC_ENCODERS)
        )


def _ffmpeg_list(kind: str) -> frozenset[str]:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Rows look like " V....D h264_nvenc  NVIDIA..." (filters: " ... scale_cuda  V->V  ...").
        if len(parts) >= 2 and re.fullmatch(r"[A-Z.]{3,6}", parts[0]):
            names.add(parts[1])
    return frozenset(names)


def _filter_options(name: str) -> frozenset[str]:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"filter={name}"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    return frozenset(m.group(1) for m in re.finditer(r"^\s{2,4}(\w+)\s+<", out, re.MULTILINE))


def _has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def ffmpeg_caps() -> Caps:
    return Caps(
        encoders=_ffmpeg_list("encoders"),
        decoders=_ffmpeg_list("decoders"),
        filters=_ffmpeg_list("filters"),
        scale_cuda_opts=_filter_options("scale_cuda"),
        gpu=_has_nvidia_gpu(),
    )


def describe_caps(caps: Caps) -> str:
    def mark(ok: bool) -> str:
        return "yes" if ok else "NO"

    rows = [
        ("NVIDIA GPU visible", caps.gpu),
        ("hevc_nvenc", "hevc_nvenc" in caps.encoders),
        ("h264_nvenc", "h264_nvenc" in caps.encoders),
        ("h264_cuvid (NVDEC)", "h264_cuvid" in caps.decoders),
        ("scale_cuda", "scale_cuda" in caps.filters),
        ("scale_cuda format=", "format" in caps.scale_cuda_opts),
        ("scale_cuda lanczos", "interp_algo" in caps.scale_cuda_opts),
        ("overlay_cuda", "overlay_cuda" in caps.filters),
    ]
    lines = [f"  {name:<22} {mark(ok)}" for name, ok in rows]
    lines.append(f"  {'full GPU clip path':<22} {mark(caps.full_gpu('h264'))}  (for H.264 clips)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- encoding


def encoder_args(encoder: str, style: Style) -> list[str]:
    """Encoder settings for the bluescreen (still-image) video."""
    args = ["-c:v", encoder]
    if "nvenc" in encoder:
        args += ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        if style.pix_fmt == "yuv444p" and encoder == "h264_nvenc":
            args += ["-profile:v", "high444p"]
    else:
        args += ["-crf", "18", "-tune", "stillimage"]
    return args


def youtube_encoder_args(encoder: str, fps: Fraction) -> list[str]:
    """Encoder settings for footage: high-quality VBR sized for YouTube 1080p uploads,
    a keyframe every 2 seconds, B-frames, BT.709 tags and AAC-LC 48 kHz stereo audio."""
    gop = str(max(1, round(2 * fps)))
    if encoder == "hevc_nvenc":
        video = [
            "-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-profile:v", "main",
            "-rc", "vbr", "-cq", "23", "-b:v", "10M", "-maxrate", "15M", "-bufsize", "20M",
            "-g", gop, "-bf", "3", "-b_ref_mode", "middle", "-spatial_aq", "1", "-temporal_aq", "1",
            "-tag:v", "hvc1",
        ]
    elif encoder == "h264_nvenc":
        video = [
            "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-profile:v", "high",
            "-rc", "vbr", "-cq", "21", "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
            "-g", gop, "-bf", "3", "-b_ref_mode", "middle", "-spatial_aq", "1", "-temporal_aq", "1",
        ]
    elif encoder == "libx264":
        video = [
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high",
            "-maxrate", "18M", "-bufsize", "24M", "-g", gop, "-bf", "2", "-flags", "+cgop",
        ]
    else:
        raise ValueError(f"Unsupported encoder for the clip background: {encoder}")
    color = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    audio = ["-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2"]
    return video + color + audio


@dataclass(frozen=True)
class ClipPlan:
    clips: list[ClipInfo]
    fps: Fraction
    end_frame: int
    out_w: int = 1920
    out_h: int = 1080

    @property
    def uniform(self) -> bool:
        return len({c.signature for c in self.clips}) == 1

    @property
    def duration(self) -> float:
        return self.end_frame / float(self.fps)


def plan_clips(clips: Sequence[ClipInfo], narration_seconds: float, style: Style = VIDEO_STYLE) -> ClipPlan:
    fps = clips[0].fps
    return ClipPlan(list(clips), fps, round(narration_seconds * fps), style.width, style.height)


def majority_signature(clips: Sequence[ClipInfo]) -> tuple[str, int, int, Fraction, str]:
    return Counter(c.signature for c in clips).most_common(1)[0][0]


def hold_seconds(clips: Sequence[ClipInfo], narration_seconds: float) -> float:
    """How long to hold the last frame so the video reaches the narration's end (0 if
    the clips are long enough, give or take a frame)."""
    gap = narration_seconds - sum(c.duration for c in clips)
    return gap if gap > 1 / float(clips[0].fps) else 0.0


# Encoders that can write clips matching the majority's codec, fastest first.
CONFORM_ENCODERS = {
    "h264": ("h264_nvenc", "libx264"),
    "hevc": ("hevc_nvenc", "libx265"),
    "av1": ("av1_nvenc", "libsvtav1"),
}


def conform_encoders(codec: str, caps: Caps) -> list[str]:
    names = [e for e in CONFORM_ENCODERS.get(codec, ()) if e in caps.encoders and (caps.gpu or "nvenc" not in e)]
    if not names:
        raise RuntimeError(f"Can't re-encode clips to {codec} to match the others; convert them to one format first.")
    return names


def conform_encoder_args(encoder: str) -> list[str]:
    """Near-transparent settings for the few clips that are re-encoded to match."""
    if "nvenc" in encoder:
        return ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
    if encoder == "libsvtav1":
        return ["-c:v", encoder, "-preset", "8", "-crf", "24"]
    return ["-c:v", encoder, "-preset", "fast", "-crf", "16" if encoder == "libx264" else "18"]


def _encode_matching(inputs: list[str], vf: str, sig: tuple, out: Path, caps: Caps, extra: Sequence[str] = ()) -> None:
    codec = sig[0]
    error = None
    for enc in conform_encoders(codec, caps):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-an", "-vf", vf,
               *extra, *conform_encoder_args(enc), str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            error = e
    raise RuntimeError(f"Could not re-encode {inputs[-1]}:\n{error.stderr[-1500:] if error else ''}")


def conform_clip(clip: ClipInfo, sig: tuple, out: Path, caps: Caps) -> ClipInfo:
    """Re-encode ``clip`` to the majority's codec, size, frame rate and pixel format
    (cropping to their aspect ratio) so it can be joined with the others."""
    codec, w, h, fps, pix = sig
    cw, ch, x, y = crop_box(clip.width, clip.height, w, h)
    vf = f"crop={cw}:{ch}:{x}:{y},scale={w}:{h}:flags=lanczos,setsar=1,fps={fps_arg(fps)},format={pix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    _encode_matching(["-i", str(clip.path)], vf, sig, out, caps)
    return ClipInfo(out, codec, w, h, fps, pix, media_duration(out))


def hold_clip(last: ClipInfo, seconds: float, sig: tuple, work: Path, caps: Caps) -> ClipInfo:
    """A clip of ``last``'s final frame, ``seconds`` long, matching the majority format."""
    codec, w, h, fps, pix = sig
    still = work / "last_frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-sseof", "-1", "-i", str(last.path),
         "-update", "1", str(still)],
        check=True, capture_output=True, text=True,
    )
    frames = max(1, round(seconds * fps))
    out = work / "hold.mp4"
    _encode_matching(
        ["-loop", "1", "-framerate", fps_arg(fps), "-i", str(still)],
        f"scale={w}:{h}:flags=lanczos,setsar=1,format={pix}", sig, out, caps, ["-frames:v", str(frames)],
    )
    return ClipInfo(out, codec, w, h, fps, pix, frames / float(fps))


def prepare_clips(clips: Sequence[ClipInfo], narration_seconds: float, work: Path, caps: Caps) -> list[ClipInfo]:
    """Make every clip share one format (re-encoding only the odd ones out) and append
    a still of the last frame if the clips run short, so the whole list can be read
    by one concat demuxer and stay on the full GPU path."""
    sig = majority_signature(clips)
    prepared = list(clips)
    odd = [i for i, c in enumerate(clips) if c.signature != sig]
    if odd:
        codec, w, h, fps, pix = sig
        print(f"{len(odd)} of {len(clips)} clips differ from the rest ({w}x{h} {codec} @ {float(fps):g} fps); "
              "re-encoding those to match", flush=True)
        for i in tqdm(odd, desc="Matching clips", unit="clip"):
            prepared[i] = conform_clip(clips[i], sig, work / "conformed" / f"{i:05d}.mp4", caps)
    hold = hold_seconds(prepared, narration_seconds)
    if hold:
        print(f"Clips are {hold:.2f}s shorter than the narration; holding the last frame", flush=True)
        prepared.append(hold_clip(prepared[-1], hold, sig, work, caps))
    return prepared


ROUTE_LABELS = {
    "gpu": "full GPU",
    "gpu_upload": "GPU decode/scale/overlay/encode, CPU crop",
    "hybrid": "GPU decode/encode + CPU filters",
    "cpu": "CPU",
}
GPU_ROUTES = ("gpu", "gpu_upload")


def build_clip_command(
    plan: ClipPlan,
    mode: str,
    encoder: str,
    clips_list: Path,
    cards_list: Path,
    audio_path: Path,
    output_path: Path,
    caps: Caps | None = None,
) -> list[str]:
    """ffmpeg command for the clip-background video. The clips must share one format
    (see ``prepare_clips``). ``mode`` is one of:

    * ``gpu``: NVDEC decode with its built-in crop, CUDA scale and overlay, NVENC.
    * ``gpu_upload``: NVDEC decode, crop on the CPU (just a pointer offset on the small
      source frame), then CUDA scale and overlay and NVENC.
    * ``hybrid``: NVDEC decode and NVENC encode, crop/scale/overlay on all CPU cores.
    * ``cpu``: everything on the CPU.
    """
    if not plan.uniform:
        raise ValueError("Clips must share one format; run prepare_clips first.")
    w, h, fps = plan.out_w, plan.out_h, plan.fps
    first = plan.clips[0]
    cw, ch, x, y = crop_box(first.width, first.height, w, h)
    clips_input = ["-f", "concat", "-safe", "0", "-i", str(clips_list)]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    # One shared CUDA device for the decoder and the filters (hwupload, overlay_cuda).
    cuda_device = ["-init_hw_device", "cuda=gpu:0", "-filter_hw_device", "gpu"]
    lanczos = ":interp_algo=lanczos" if caps is None or "interp_algo" in caps.scale_cuda_opts else ""
    cuda_tail = (
        f"scale_cuda=w={w}:h={h}{lanczos}:format=yuv420p[bg];"
        f"[1:v]format=yuva420p,hwupload[subs];"
        f"[bg][subs]overlay_cuda=x=0:y=0[v]"
    )

    if mode == "gpu":
        crop = f"{y}x{first.height - ch - y}x{x}x{first.width - cw - x}"  # top x bottom x left x right
        cmd += [
            *cuda_device,
            "-hwaccel", "cuda", "-hwaccel_device", "gpu", "-hwaccel_output_format", "cuda",
            "-c:v", f"{first.codec}_cuvid", "-crop", crop, *clips_input,
        ]
        graph = "[0:v]" + cuda_tail
    elif mode == "gpu_upload":
        cmd += [*cuda_device, "-hwaccel", "cuda", "-hwaccel_device", "gpu", *clips_input]
        graph = f"[0:v]crop={cw}:{ch}:{x}:{y},format=nv12,hwupload," + cuda_tail
    else:
        # Crop, Lanczos scale and overlay are frame-parallel on the CPU, so use every core.
        cmd += ["-filter_complex_threads", str(os.cpu_count() or 1)]
        cmd += [*(["-hwaccel", "cuda"] if mode == "hybrid" else []), *clips_input]
        graph = (
            f"[0:v]crop={cw}:{ch}:{x}:{y},scale={w}:{h}:flags=lanczos,setsar=1,fps={fps_arg(fps)}[bg];"
            f"[1:v]format=rgba[subs];[bg][subs]overlay=0:0:format=auto,format=yuv420p[v]"
        )

    cmd += [
        "-f", "concat", "-safe", "0", "-i", str(cards_list),
        "-i", str(audio_path),
        "-filter_complex", graph,
        "-map", "[v]", "-map", "2:a:0",
        "-r", fps_arg(fps), "-fps_mode", "cfr",
        *youtube_encoder_args(encoder, fps),
        "-t", f"{plan.duration:.6f}",
        "-movflags", "+faststart", str(output_path),
    ]
    return cmd


def clip_attempts(
    plan: ClipPlan, encoder: str, caps: Caps, cpu_filters: bool = False
) -> list[tuple[str, str]]:
    """Ordered (mode, encoder) attempts, fastest first; later ones are fallbacks.
    GPU filter routes are checked against a CPU render before use (``verify_routes``)."""
    if encoder == "libx264":
        return [("cpu", "libx264")]
    if encoder == "auto":
        nvencs = [e for e in NVENC_ENCODERS if e in caps.encoders] if caps.gpu else []
    else:
        nvencs = [encoder]
    attempts = []
    if not cpu_filters and nvencs and plan.uniform and caps.cuda_filters():
        if caps.full_gpu(plan.clips[0].codec):
            attempts += [("gpu", e) for e in nvencs]
        attempts += [("gpu_upload", e) for e in nvencs]
    attempts += [("hybrid", e) for e in nvencs]
    if encoder == "auto":
        attempts.append(("cpu", "libx264"))
    return attempts


def decode_frame(path: Path, index: int, w: int, h: int) -> bytes:
    """Frame ``index`` of ``path`` as raw yuv420p bytes, decoded on the CPU."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True, check=True,
    ).stdout
    if len(out) != w * h * 3 // 2:
        raise RuntimeError(f"Could not decode frame {index} of {path}")
    return out


def edge_defect(a: bytes, b: bytes, w: int, h: int) -> float:
    """How much worse two yuv420p frames agree along their edges than in their interior
    (mean absolute difference, 8-bit units). Different scalers agree to within a few
    units everywhere; a green or garbage line at an edge shows up as a large excess."""
    worst = 0.0
    planes = [(0, w, h), (w * h, w // 2, h // 2), (w * h * 5 // 4, w // 2, h // 2)]
    for off, pw, ph in planes:
        def row(r: int) -> float:
            s = off + r * pw
            return sum(abs(p - q) for p, q in zip(a[s:s + pw], b[s:s + pw])) / pw

        def col(c: int) -> float:
            return sum(abs(a[off + r * pw + c] - b[off + r * pw + c]) for r in range(ph)) / ph

        interior = sorted([row(ph * k // 10) for k in range(2, 9)] + [col(pw * k // 10) for k in range(2, 9)])
        baseline = interior[len(interior) // 2]
        edges = [row(0), row(1), row(ph - 2), row(ph - 1), col(0), col(1), col(pw - 2), col(pw - 1)]
        worst = max(worst, max(edges) - baseline)
    return worst


EDGE_DEFECT_LIMIT = 20.0


def verify_routes(
    attempts: list[tuple[str, str]], plan: ClipPlan, work: Path, audio_path: Path, caps: Caps
) -> list[tuple[str, str]]:
    """Render one second of the first clip with each GPU filter route and compare it to
    a CPU render. Routes that fail or leave edge artifacts (e.g. a green line along the
    bottom) are dropped. Stops at the first good route; the rest are dropped too, so the
    fallbacks are the CPU-filter routes, which are known to be clean."""
    if not any(mode in GPU_ROUTES for mode, _ in attempts):
        return attempts
    first = plan.clips[0]
    frames = max(1, min(round(plan.fps), int(first.duration * float(plan.fps))))
    probe = ClipPlan([first], plan.fps, frames, plan.out_w, plan.out_h)
    vdir = work / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    clips_list, cards_list = vdir / "clips.txt", vdir / "cards.txt"
    clips_list.write_text("ffconcat version 1.0\n" + concat_line(first.path.resolve()), encoding="utf-8")
    blank = vdir / "blank.png"
    Image.new("RGBA", (plan.out_w, plan.out_h), (0, 0, 0, 0)).save(blank)
    cards_list.write_text(
        "ffconcat version 1.0\n" + concat_line(blank) + f"duration {probe.duration:.6f}\n" + concat_line(blank),
        encoding="utf-8",
    )
    index = frames // 2

    def render(mode: str, enc: str, name: str) -> bytes:
        out = vdir / name
        cmd = build_clip_command(probe, mode, enc, clips_list, cards_list, audio_path, out, caps)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return decode_frame(out, index, plan.out_w, plan.out_h)

    print("Checking the GPU routes against a CPU render of the first clip", flush=True)
    try:
        reference = render("cpu", "libx264", "reference.mp4")
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"  could not make the CPU reference ({e}); using the CPU-filter routes", flush=True)
        return [a for a in attempts if a[0] not in GPU_ROUTES]

    verdicts: dict[str, bool] = {}
    for mode, enc in attempts:
        if mode not in GPU_ROUTES or mode in verdicts or any(verdicts.values()):
            continue
        label = ROUTE_LABELS[mode]
        try:
            defect = edge_defect(render(mode, enc, f"{mode}.mp4"), reference, plan.out_w, plan.out_h)
        except subprocess.CalledProcessError as e:
            tail = " | ".join((e.stderr or "").strip().splitlines()[-2:])
            print(f"  {label}: not available ({tail or 'ffmpeg failed'})", flush=True)
            verdicts[mode] = False
            continue
        except RuntimeError as e:
            print(f"  {label}: {e}", flush=True)
            verdicts[mode] = False
            continue
        verdicts[mode] = defect <= EDGE_DEFECT_LIMIT
        print(f"  {label}: {'OK' if verdicts[mode] else 'edge artifact, skipping'} (edge score {defect:.1f})",
              flush=True)
    return [a for a in attempts if a[0] not in GPU_ROUTES or verdicts.get(a[0])]


def render_clip_video(
    cues: Sequence[TimedText],
    audio_path: Path,
    output_path: Path,
    clips: Path | Sequence[Path],
    encoder: str = "auto",
    work: Path | None = None,
    style: Style = VIDEO_STYLE,
    probe_cache: Path | None = None,
    cpu_filters: bool = False,
) -> Path:
    clip_paths = find_clips(clips)
    infos = probe_clips(clip_paths, probe_cache)
    narration = media_duration(audio_path)
    caps = ffmpeg_caps()
    codec, cw, ch, fps, _ = majority_signature(infos)
    print(
        f"{len(infos)} clip(s), {cw}x{ch} {codec} @ {float(fps):g} fps, "
        f"{sum(c.duration for c in infos):.2f}s total; narration {narration:.2f}s",
        flush=True,
    )

    font = ImageFont.truetype(str(FONT_PATH), style.font_size)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(work) if work else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        prepared = prepare_clips(infos, narration, work, caps)
        plan = plan_clips(prepared, narration, style)
        cards_list, _ = build_card_timeline(
            cues, plan.fps, narration, work, lambda text: render_card(text, font, style, transparent=True)
        )
        clips_list = work / "clips.txt"
        with clips_list.open("w", encoding="utf-8") as fh:
            fh.write("ffconcat version 1.0\n")
            for clip in prepared:
                fh.write(concat_line(clip.path.resolve()))

        attempts = clip_attempts(plan, encoder, caps, cpu_filters)
        attempts = verify_routes(attempts, plan, work, audio_path, caps)
        if not attempts:
            raise RuntimeError(f"Encoder {encoder} is not usable here.")
        for n, (mode, enc) in enumerate(attempts, 1):
            label = ROUTE_LABELS[mode]
            print(f"Encoding MP4 with {enc} ({label})", flush=True)
            cmd = build_clip_command(plan, mode, enc, clips_list, cards_list, audio_path, output_path, caps)
            try:
                run_ffmpeg(cmd, plan.duration, f"Encoding ({enc})")
                break
            except subprocess.CalledProcessError:
                if n == len(attempts):
                    raise
                print(f"{enc} ({label}) failed; trying the next option", flush=True)
    return output_path


# --------------------------------------------------------------------------- entry points


def render_video(
    cues: Iterable[TimedText],
    audio_path: Path,
    output_path: Path,
    encoder: str = "auto",
    keep_frames: bool = False,
    style: Style | None = None,
    background_clips: Path | Sequence[Path] | None = None,
    probe_cache: Path | None = None,
    cpu_filters: bool = False,
) -> Path:
    """Render cues over the audio of ``audio_path`` (audio or video) into ``output_path``.

    With ``background_clips`` (a directory of ``*.mp4`` or a list of files) the cards
    are overlaid on the concatenated clips; otherwise on chroma blue.
    """
    cues = list(cues)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.with_name(output_path.stem + "_frames") if keep_frames else None

    if background_clips is not None:
        return render_clip_video(
            cues, audio_path, output_path, background_clips, encoder, frames_dir, style or VIDEO_STYLE, probe_cache,
            cpu_filters,
        )

    style = style or STYLE
    font = ImageFont.truetype(str(FONT_PATH), style.font_size)
    total = media_duration(audio_path)
    fps = style.fps

    with tempfile.TemporaryDirectory() as tmp:
        work = frames_dir or Path(tmp)
        work.mkdir(exist_ok=True)
        concat, end_frame = build_card_timeline(cues, fps, total, work, lambda text: render_card(text, font, style))

        def encode(enc: str) -> None:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-i", str(audio_path),
                "-map", "0:v", "-map", "1:a:0",
                "-r", str(fps), "-fps_mode", "cfr", "-pix_fmt", style.pix_fmt,
                *encoder_args(enc, style),
                "-c:a", "aac", "-b:a", "192k", "-t", f"{end_frame / fps:.6f}",
                "-movflags", "+faststart", str(output_path),
            ]
            run_ffmpeg(cmd, end_frame / fps, f"Encoding ({enc})")

        if encoder == "auto":
            # NVENC is much faster but needs an NVIDIA GPU and an ffmpeg built with it.
            try:
                print("Encoding MP4 with h264_nvenc", flush=True)
                encode("h264_nvenc")
            except subprocess.CalledProcessError:
                print("h264_nvenc unavailable; falling back to libx264", flush=True)
                encode("libx264")
        else:
            print(f"Encoding MP4 with {encoder}", flush=True)
            encode(encoder)

    return output_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srt", type=Path, nargs="?")
    ap.add_argument("audio", type=Path, nargs="?", help="Narration audio, or a video whose audio track is used.")
    ap.add_argument("output", type=Path, nargs="?", help="Defaults to the .srt path with an .mp4 extension.")
    ap.add_argument("--clips", type=Path, help="Directory of .mp4 clips to use as the background.")
    ap.add_argument(
        "--cpu-filters", action="store_true",
        help="Skip the GPU filter routes: crop, scale and overlay on the CPU (GPU still decodes/encodes).",
    )
    ap.add_argument(
        "--encoder", choices=ENCODERS, default="auto",
        help="auto: h264_nvenc then libx264 (bluescreen); hevc_nvenc, h264_nvenc, then libx264 (--clips).",
    )
    ap.add_argument("--keep-frames", action="store_true", help="Keep the rendered PNGs next to the output.")
    ap.add_argument("--check", action="store_true", help="Print which GPU features this ffmpeg supports and exit.")
    args = ap.parse_args(argv)

    if args.check:
        print(describe_caps(ffmpeg_caps()))
        return 0
    if not args.srt or not args.audio:
        ap.error("srt and audio are required (unless --check)")

    cues = parse_srt(args.srt)
    output = render_video(
        cues, args.audio, args.output or args.srt.with_suffix(".mp4"), args.encoder, args.keep_frames,
        background_clips=args.clips, cpu_filters=args.cpu_filters,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
