"""One-shot SubGen pipeline for a Google Drive folder (made for Colab).

Expected layout under ``--root`` (Colab: /content/drive/MyDrive/subgen):

    narration.wav       narration audio (required)
    clips/*.mp4         background clips, joined in natural filename order (required)
    script.txt          original script, used for cue wording if present (optional)
    output/             narration.srt, narration.mp4 and narration_norm.wav are written here

The narration is first loudness-normalized for YouTube (-14 LUFS, -1.5 dBTP,
48 kHz); the normalized audio is what gets transcribed and put in the video.

The clips are read straight from Drive while the video encodes, so reading
overlaps with GPU work instead of waiting for a full copy first. --copy-clips
copies them to local disk first instead (in parallel; clips already copied
earlier in the session are skipped). Clip details are cached in
output/.clip_info_cache.json, so reruns over the same clips start right away.
The narration and outputs go through local disk, then outputs are copied to Drive.

Usage (Colab, after mounting Drive):
    !python -m drive_pipeline
    !python -m drive_pipeline --model large-v3-turbo     # extra options go to subgen
    !python -m drive_pipeline --srt-only                 # just the subtitles
    !python -m drive_pipeline --reuse-srt                # re-render output/narration.srt (e.g. after editing it)
    !python -m drive_pipeline --lufs -16                 # different loudness target
    !python -m drive_pipeline --no-normalize             # use narration.wav as is
    !python -m drive_pipeline --copy-clips               # copy clips locally before encoding
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

import normalize_audio
import srt2mp4

DEFAULT_ROOT = Path("/content/drive/MyDrive/subgen")


def copy_in(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return dst


def copy_clips(srcs: list[Path], dst_dir: Path, workers: int = 8) -> list[Path]:
    """Copy clips to ``dst_dir`` in parallel, keeping names and modification times.
    Files already there with the same size and time are skipped; others are removed."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.name for s in srcs}
    for stale in dst_dir.iterdir():
        if stale.name not in wanted:
            stale.unlink()

    def copy(src: Path) -> int:
        dst = dst_dir / src.name
        st = src.stat()
        if dst.exists():
            dt = dst.stat()
            if dt.st_size == st.st_size and dt.st_mtime_ns == st.st_mtime_ns:
                return st.st_size
        shutil.copy2(src, dst)
        return st.st_size

    total = sum(s.stat().st_size for s in srcs)
    with ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=total, desc="Copying clips", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for size in pool.map(copy, srcs):
            bar.update(size)
    return [dst_dir / s.name for s in srcs]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Drive folder. Defaults to {DEFAULT_ROOT}.")
    ap.add_argument("--audio", default="narration.wav", help="Narration file name inside --root.")
    ap.add_argument("--clips", default="clips", help="Clips folder name inside --root.")
    ap.add_argument("--script", default="script.txt", help="Optional script file name inside --root.")
    ap.add_argument("--output", default="output", help="Output folder name inside --root.")
    ap.add_argument("--work", type=Path, help="Local working directory. Defaults to /content/subgen-work on Colab.")
    ap.add_argument("--encoder", choices=srt2mp4.ENCODERS, default="auto", help="Video encoder (auto = best available).")
    ap.add_argument(
        "--copy-clips", action="store_true",
        help="Copy clips to local disk (in parallel) before encoding instead of reading them from Drive.",
    )
    ap.add_argument(
        "--cpu-filters", action="store_true",
        help="Skip the GPU filter routes (crop/scale/overlay on the CPU; slower, GPU still decodes/encodes).",
    )
    ap.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization of the narration.")
    ap.add_argument(
        "--lufs", type=float, default=normalize_audio.TARGET_I,
        help=f"Narration loudness target in LUFS (default {normalize_audio.TARGET_I:g}, YouTube's reference).",
    )
    ap.add_argument(
        "--true-peak", type=float, default=normalize_audio.TARGET_TP,
        help=f"Narration true-peak ceiling in dBTP (default {normalize_audio.TARGET_TP:g}).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--srt-only", action="store_true", help="Only write the SRT; skip the video.")
    mode.add_argument("--reuse-srt", action="store_true", help="Skip transcription and render the existing output SRT.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args, subgen_args = build_parser().parse_known_args(argv)
    started = time.monotonic()

    root = args.root
    if not root.is_dir():
        raise SystemExit(
            f"{root} not found. On Colab, mount Drive first (from google.colab import drive; "
            "drive.mount('/content/drive')) and create MyDrive/subgen."
        )
    audio_src = root / args.audio
    if not audio_src.is_file():
        raise SystemExit(f"Narration not found: {audio_src}")
    clips_src = srt2mp4.find_clips(root / args.clips) if not args.srt_only else []
    script_src = root / args.script
    out_dir = root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_name, mp4_name = f"{audio_src.stem}.srt", f"{audio_src.stem}.mp4"

    tmp = None
    if args.work:
        work = args.work
    elif Path("/content").is_dir():
        work = Path("/content/subgen-work")
    else:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)
    if work.exists():
        for name in ("narration", "script"):  # keep input/clips so --copy-clips can skip them next run
            for old in (work / "input").glob(f"{name}*"):
                old.unlink()
    local_out = work / "output"
    local_out.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Copying narration from {root} to {work}", flush=True)
        audio = copy_in(audio_src, work / "input")
        if not args.no_normalize:
            print(f"Normalizing narration to {args.lufs:g} LUFS, {args.true_peak:g} dBTP", flush=True)
            normalized = audio.with_name(f"{audio.stem}{normalize_audio.SUFFIX}.wav")
            normalize_audio.normalize(audio, normalized, args.lufs, args.true_peak)
            audio = normalized
            shutil.copyfile(audio, out_dir / audio.name)
            print(f"Saved {out_dir / audio.name}", flush=True)
        clips = copy_clips(clips_src, work / "input" / "clips") if args.copy_clips and clips_src else clips_src
        srt_path = local_out / srt_name
        mp4_path = local_out / mp4_name

        if args.reuse_srt:
            existing = out_dir / srt_name
            if not existing.is_file():
                raise SystemExit(f"--reuse-srt: {existing} does not exist yet.")
            shutil.copyfile(existing, srt_path)
            print(f"Reusing {existing}", flush=True)
        else:
            import subgen  # imported here so --reuse-srt doesn't load torch/WhisperX

            cmd = [str(audio), "-o", str(srt_path), *subgen_args]
            if script_src.is_file():
                print(f"Using script {script_src}", flush=True)
                cmd += ["--script", str(copy_in(script_src, work / "input"))]
            subgen.main(cmd)
            subgen.free_torch_memory()
            shutil.copyfile(srt_path, out_dir / srt_name)
            print(f"Saved {out_dir / srt_name}", flush=True)

        if not args.srt_only:
            srt2mp4.render_video(
                srt2mp4.parse_srt(srt_path), audio, mp4_path, encoder=args.encoder, background_clips=clips,
                probe_cache=out_dir / ".clip_info_cache.json", cpu_filters=args.cpu_filters,
            )
            print(f"Copying video to {out_dir / mp4_name}", flush=True)
            shutil.copyfile(mp4_path, out_dir / mp4_name)
            size_mb = mp4_path.stat().st_size / 1e6
            print(f"Saved {out_dir / mp4_name} ({size_mb:.0f} MB)", flush=True)
    finally:
        if tmp is not None:
            tmp.cleanup()

    print(f"Done in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
