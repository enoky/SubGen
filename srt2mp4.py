"""Render subtitles as a bluescreen text video, muxed with the narration audio.

Each subtitle is drawn once as a 1920x1080 PNG (word-wrapped, centered, white
Yatra One text with a drop shadow on chroma blue), then ffmpeg's concat demuxer
holds each image for its subtitle's duration and encodes the result.

Usage:
    uv run python -m srt2mp4 subtitles.srt narration.wav out.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import srt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "YatraOne-Regular.ttf"
ENCODERS = ("auto", "h264_nvenc", "hevc_nvenc", "libx264")


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
    pix_fmt: str = "yuv420p"


STYLE = Style()


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


def render_card(text: str, font: ImageFont.FreeTypeFont, style: Style = STYLE) -> Image.Image:
    w, h = style.width, style.height
    lines = wrap(text, font, style.max_width) if text else []

    # Text and shadow are drawn as alpha masks, then composited over the key color.
    text_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(text_mask)
    ascent, descent = font.getmetrics()
    block_h = (len(lines) - 1) * style.line_height + ascent + descent
    y = (h - block_h) / 2
    for line in lines:
        x = (w - font.getlength(line)) / 2
        draw.text((x, y), line, font=font, fill=255)
        y += style.line_height

    shadow_mask = Image.new("L", (w, h), 0)
    shadow_mask.paste(text_mask, (style.shadow_offset, style.shadow_offset))
    if style.shadow_blur > 0:
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(style.shadow_blur))
    shadow_mask = shadow_mask.point(lambda v: int(v * style.shadow_opacity))

    img = Image.new("RGB", (w, h), style.bg)
    img.paste(Image.new("RGB", (w, h), (0, 0, 0)), mask=shadow_mask)
    img.paste(Image.new("RGB", (w, h), (255, 255, 255)), mask=text_mask)
    return img


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def encoder_args(encoder: str, style: Style) -> list[str]:
    args = ["-c:v", encoder]
    if "nvenc" in encoder:
        args += ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        if style.pix_fmt == "yuv444p" and encoder == "h264_nvenc":
            args += ["-profile:v", "high444p"]
    else:
        args += ["-crf", "18", "-tune", "stillimage"]
    return args


def render_video(
    cues: Iterable[TimedText],
    audio_path: Path,
    output_path: Path,
    encoder: str = "auto",
    keep_frames: bool = False,
    style: Style = STYLE,
) -> Path:
    """Render cues over the audio of ``audio_path`` (audio or video) into ``output_path``."""
    cues = list(cues)
    font = ImageFont.truetype(str(FONT_PATH), style.font_size)
    total = media_duration(audio_path)
    fps = style.fps
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if keep_frames:
            work = output_path.with_name(output_path.stem + "_frames")
            work.mkdir(exist_ok=True)
        else:
            work = Path(tmp)

        blank = work / "blank.png"
        render_card("", font, style).save(blank)

        # Snap every boundary to a frame index so durations never drift from the audio.
        def frame(t: float) -> int:
            return round(t * fps)

        entries, cursor = [], 0
        for i, cue in enumerate(tqdm(cues, desc="Rendering subtitle frames", unit="cue"), 1):
            f0, f1 = max(frame(cue.start), cursor), frame(cue.end)
            if f1 <= f0:
                continue
            if f0 > cursor:
                entries.append((blank, f0 - cursor))
            card = work / f"{i:05d}.png"
            render_card(cue.text, font, style).save(card)
            entries.append((card, f1 - f0))
            cursor = f1
        end_frame = max(frame(total), cursor)
        if end_frame > cursor:
            entries.append((blank, end_frame - cursor))
        if not entries:
            raise RuntimeError("Nothing to render: no subtitles and no audio duration.")

        concat = work / "list.txt"
        with concat.open("w", encoding="utf-8") as fh:
            fh.write("ffconcat version 1.0\n")
            for path, frames in entries:
                fh.write(f"file '{path.as_posix()}'\nduration {frames / fps:.6f}\n")
            fh.write(f"file '{entries[-1][0].as_posix()}'\n")  # concat demuxer needs the last file repeated

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
            subprocess.run(cmd, check=True)

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
    ap.add_argument("srt", type=Path)
    ap.add_argument("audio", type=Path, help="Narration audio, or a video whose audio track is used.")
    ap.add_argument("output", type=Path, nargs="?", help="Defaults to the .srt path with an .mp4 extension.")
    ap.add_argument("--encoder", choices=ENCODERS, default="auto", help="auto tries h264_nvenc, then libx264.")
    ap.add_argument("--keep-frames", action="store_true", help="Keep the rendered PNGs next to the output.")
    args = ap.parse_args(argv)

    cues = parse_srt(args.srt)
    output = render_video(cues, args.audio, args.output or args.srt.with_suffix(".mp4"), args.encoder, args.keep_frames)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
