"""Mix a background music bed under the narration.

The music is cut to the narration's length (starting at its beginning, so any
fade-in it has is kept), faded out at the end, and set to a loudness relative to
the narration: by default 20 LU below it, the usual range for music under
voice-over being about 18-25 LU. A limiter keeps the mix under the true-peak
ceiling. Music shorter than the narration is looped.

Usage:
    uv run python -m background_music narration_norm.wav music.m4a mix.wav
    uv run python -m background_music narration_norm.wav music.m4a mix.wav --level -24 --fade 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ffmpeg_progress import run_ffmpeg
from normalize_audio import TARGET_I, TARGET_TP, probe_audio

DEFAULT_LEVEL = -20.0  # music loudness relative to the narration, in LU
DEFAULT_FADE = 8.0  # fade-out at the end, in seconds
# YouTube Shorts (15-60 s typically): start past the track's own slow intro and use
# short fades, so the music is there from the first line and ends cleanly.
SHORTS_OFFSET = 10.0
SHORTS_FADE_IN = 2.0
SHORTS_FADE = 2.0
SAMPLE_RATE = 48000


def measure_loudness(path: Path, seconds: float | None = None, loop: bool = False, offset: float = 0.0) -> float:
    """Integrated loudness (LUFS) of ``path``, or of ``seconds`` of it starting at ``offset``."""
    args = ["ffmpeg", "-hide_banner", *(["-stream_loop", "-1"] if loop else [])]
    args += [*(["-ss", f"{offset:.3f}"] if offset else []), "-i", str(path), *(["-t", f"{seconds:.3f}"] if seconds else []),
             "-map", "0:a:0", "-af", "ebur128=framelog=quiet", "-f", "null", "-"]
    stderr = run_ffmpeg(args, seconds or probe_audio(path)[0], f"    measuring {path.name}")
    found = re.findall(r"\bI:\s+(-?[\d.]+|-inf) LUFS", stderr)
    if not found or found[-1] == "-inf":
        raise RuntimeError(f"Could not measure the loudness of {path}")
    return float(found[-1])


def music_gain(music_lufs: float, narration_lufs: float, level: float) -> float:
    """dB to apply to the music so it sits ``level`` LU relative to the narration."""
    return narration_lufs + level - music_lufs


def build_mix_command(
    narration: Path, music: Path, output: Path, duration: float, gain_db: float,
    fade: float = DEFAULT_FADE, true_peak: float = TARGET_TP, loop: bool = False,
    offset: float = 0.0, fade_in: float = 0.0,
) -> list[str]:
    """``offset``: start this many seconds into the music. ``fade_in``/``fade``: fade
    the music in at the start and out at the end (seconds)."""
    fade = max(0.0, min(fade, duration))
    fade_in = max(0.0, min(fade_in, duration - fade))
    limit = 10 ** (true_peak / 20)
    graph = (
        f"[0:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo[n];"
        f"[1:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"atrim=0:{duration:.6f},asetpts=PTS-STARTPTS,volume={gain_db:.2f}dB"
        + (f",afade=t=in:st=0:d={fade_in:.6f}" if fade_in else "")
        + (f",afade=t=out:st={duration - fade:.6f}:d={fade:.6f}" if fade else "")
        + "[m];"
        f"[n][m]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit={limit:.4f}:level=false[a]"
    )
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(narration),
        *(["-stream_loop", "-1"] if loop else []), *(["-ss", f"{offset:.3f}"] if offset else []),
        "-i", str(music),
        "-filter_complex", graph, "-map", "[a]",
        "-t", f"{duration:.6f}", "-c:a", "pcm_s24le", "-ar", str(SAMPLE_RATE), str(output),
    ]


def mix(
    narration: Path, music: Path, output: Path, level: float = DEFAULT_LEVEL,
    fade: float = DEFAULT_FADE, narration_lufs: float | None = None, true_peak: float = TARGET_TP,
    offset: float = 0.0, fade_in: float = 0.0,
) -> Path:
    """Write ``output``: ``narration`` with ``music`` underneath (see module docs).
    ``narration_lufs`` skips measuring the narration when its loudness is known
    (e.g. it was just normalized)."""
    duration, _ = probe_audio(narration)
    music_len, _ = probe_audio(music)
    if not duration:
        raise RuntimeError(f"Could not read the duration of {narration}")
    if music_len and offset >= music_len:
        print(f"    note: {music.name} is only {music_len:.0f}s long; starting it from the beginning", flush=True)
        offset = 0.0
    loop = bool(music_len and music_len - offset < duration)
    if loop:
        print(f"    note: {music.name} ({music_len:.0f}s) is shorter than the narration; looping it", flush=True)
    if narration_lufs is None:
        narration_lufs = measure_loudness(narration)
    music_lufs = measure_loudness(music, duration, loop, offset)
    gain = music_gain(music_lufs, narration_lufs, level)
    print(f"    music {music_lufs:.1f} LUFS -> {music_lufs + gain:.1f} LUFS ({gain:+.1f} dB), "
          f"{level:g} LU under the narration ({narration_lufs:.1f} LUFS)"
          + (f"; starts {offset:g}s in" if offset else "")
          + (f"; {fade_in:g}s fade-in" if fade_in else "") + f"; {fade:g}s fade-out", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(build_mix_command(narration, music, output, duration, gain, fade, true_peak, loop, offset, fade_in),
               duration, "    mixing")
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("narration", type=Path)
    ap.add_argument("music", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                    help=f"music loudness relative to the narration in LU (default {DEFAULT_LEVEL:g})")
    ap.add_argument("--fade", type=float, default=DEFAULT_FADE,
                    help=f"fade-out length in seconds (default {DEFAULT_FADE:g})")
    ap.add_argument("--fade-in", type=float, default=0.0, help="fade-in length in seconds (default 0)")
    ap.add_argument("--offset", type=float, default=0.0, help="start this many seconds into the music")
    args = ap.parse_args(argv)
    mix(args.narration, args.music, args.output, args.level, args.fade, offset=args.offset, fade_in=args.fade_in)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
