"""Two-pass EBU R128 loudness normalization for narration, using ffmpeg's loudnorm.

Defaults suit YouTube dialog: -14 LUFS integrated (YouTube's playback reference),
-1.5 dBTP true peak (headroom for AAC encoding), 11 LU loudness range, a single
linear gain when possible, 48 kHz 24-bit PCM WAV.

Mono stays mono. When the video is encoded, ffmpeg upmixes it to stereo at -3 dB
per channel, so the stereo track measures the same loudness as the mono file.

Usage:
    uv run python -m normalize_audio input.wav                 # writes input_norm.wav
    uv run python -m normalize_audio input.wav output.wav
    uv run python -m normalize_audio *.wav                     # batch, each -> *_norm.wav
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from ffmpeg_progress import run_ffmpeg as run_ffmpeg_with_bar

TARGET_I = -14.0  # integrated loudness (LUFS)
TARGET_TP = -1.5  # true peak ceiling (dBTP)
TARGET_LRA = 11.0  # loudness range (LU)
SAMPLE_RATE = 48000
CODEC = "pcm_s24le"
SUFFIX = "_norm"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def probe_audio(src: Path) -> tuple[float | None, int | None]:
    """Return (duration in seconds, channel count) via ffprobe; None where unknown."""
    if not shutil.which("ffprobe"):
        return None, None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels:format=duration", "-of", "json", str(src)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout
        data = json.loads(out or "{}")
    except (ValueError, OSError):
        return None, None
    duration = data.get("format", {}).get("duration")
    streams = data.get("streams") or [{}]
    channels = streams[0].get("channels")
    return (float(duration) if duration else None), (int(channels) if channels else None)


def run_ffmpeg(args: list[str], label: str, duration: float | None) -> str:
    """Run ffmpeg with a progress bar; return its stderr (where loudnorm prints JSON)."""
    start = time.time()
    try:
        stderr = run_ffmpeg_with_bar(["ffmpeg", "-hide_banner", *args], duration, f"    {label}")
    except subprocess.CalledProcessError as e:
        log(f"    {label}: FAILED")
        raise RuntimeError(f"ffmpeg failed:\n{(e.stderr or '')[-2000:]}") from e
    log(f"    {label}: done in {time.time() - start:.1f}s")
    return stderr


def parse_loudnorm_json(stderr: str) -> dict:
    """Extract the last JSON block loudnorm printed to stderr."""
    matches = re.findall(r"\{[^{}]*\}", stderr, re.DOTALL)
    if not matches:
        raise RuntimeError("Could not find loudnorm JSON in ffmpeg output.")
    return json.loads(matches[-1])


def normalize(
    src: Path,
    dst: Path,
    target_i: float = TARGET_I,
    target_tp: float = TARGET_TP,
    target_lra: float = TARGET_LRA,
    sample_rate: int = SAMPLE_RATE,
    codec: str = CODEC,
    mono: bool = False,
) -> dict:
    """Normalize ``src`` into ``dst`` and return loudnorm's pass-2 report.

    ``mono=True`` downmixes to one channel; otherwise the channel layout is kept.
    """
    src, dst = Path(src), Path(dst)
    duration, _ = probe_audio(src)
    if duration:
        log(f"    duration: {fmt_time(duration)}")
    base = f"I={target_i}:TP={target_tp}:LRA={target_lra}"
    downmix = ["-ac", "1"] if mono else []

    # Pass 1: measure
    stderr = run_ffmpeg(
        ["-i", str(src), *downmix, "-af", f"loudnorm={base}:print_format=json", "-f", "null", "-"],
        "pass 1/2 (measuring)", duration,
    )
    m = parse_loudnorm_json(stderr)
    log(f"    measured: {m['input_i']} LUFS, {m['input_tp']} dBTP, LRA {m['input_lra']}")

    # Pass 2: apply using measured values
    filt = (
        f"loudnorm={base}"
        f":measured_I={m['input_i']}"
        f":measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}"
        f":measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}"
        f":linear=true:print_format=json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    stderr = run_ffmpeg(
        # loudnorm works at 192 kHz internally, so the output rate must be set explicitly.
        ["-y", "-i", str(src), "-af", filt, *downmix, "-ar", str(sample_rate), "-c:a", codec, str(dst)],
        "pass 2/2 (normalizing)", duration,
    )
    out = parse_loudnorm_json(stderr)

    log(f"    result:   {out['output_i']} LUFS, {out['output_tp']} dBTP, LRA {out['output_lra']}")
    if out.get("normalization_type", "").lower() != "linear":
        log("    note: ffmpeg used dynamic normalization instead of a single linear gain "
            "(usually because linear gain would exceed the true-peak limit)")
    log(f"    saved:    {dst}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="input audio file(s)")
    parser.add_argument("-o", "--output", type=Path, help="output path (only valid with a single input)")
    parser.add_argument("--lufs", type=float, default=TARGET_I, help=f"integrated loudness target (default {TARGET_I})")
    parser.add_argument("--true-peak", type=float, default=TARGET_TP, help=f"true peak ceiling in dBTP (default {TARGET_TP})")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help=f"output sample rate (default {SAMPLE_RATE})")
    parser.add_argument("--mono", action="store_true", help="downmix to mono")
    args = parser.parse_args(argv)

    # Allow "normalize_audio in.wav out.wav" as a shorthand
    if args.output is None and len(args.inputs) == 2 and not args.inputs[1].exists():
        args.output = args.inputs.pop()
    if args.output and len(args.inputs) > 1:
        parser.error("--output can only be used with a single input file")

    # In batch mode, skip files this script already produced
    inputs = args.inputs
    if not args.output:
        skipped = [p for p in inputs if p.stem.endswith(SUFFIX)]
        inputs = [p for p in inputs if not p.stem.endswith(SUFFIX)]
        if skipped:
            log(f"Skipping {len(skipped)} already-normalized file(s) (*{SUFFIX}.wav)")

    total = len(inputs)
    if total == 0:
        log("Nothing to do.")
        return 0

    log(f"Normalizing {total} file(s) to {args.lufs} LUFS, {args.true_peak} dBTP, {args.sample_rate} Hz\n")
    batch_start = time.time()
    failed = []
    for i, src in enumerate(inputs, 1):
        dst = args.output or src.with_name(f"{src.stem}{SUFFIX}.wav")
        log(f"[{i}/{total}] {src.name}")
        file_start = time.time()
        try:
            if not src.is_file():
                raise FileNotFoundError(f"{src} not found")
            normalize(src, dst, args.lufs, args.true_peak, sample_rate=args.sample_rate, mono=args.mono)
            log(f"    finished in {time.time() - file_start:.1f}s\n")
        except Exception as e:  # noqa: BLE001 - keep going through the batch
            failed.append(src.name)
            log(f"    ERROR: {e}\n")

    log(f"Done: {total - len(failed)}/{total} succeeded in {fmt_time(time.time() - batch_start)}")
    if failed:
        log("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
