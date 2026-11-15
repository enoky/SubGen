"""One-shot SubGen pipeline for a Google Drive folder (made for Colab).

Expected layout under ``--root`` (Colab: /content/drive/MyDrive/subgen):

    narration.wav       narration audio (required)
    clips/*.mp4         background clips, joined in natural filename order (required)
    script.txt          original script, used for cue wording if present (optional)
    output/             narration.srt and narration.mp4 are written here

Inputs are copied to local disk first and outputs are written locally, then copied
to Drive, because reading and writing through the Drive mount is slow.

Usage (Colab, after mounting Drive):
    !python -m drive_pipeline
    !python -m drive_pipeline --model large-v3-turbo     # extra options go to subgen
    !python -m drive_pipeline --srt-only                 # just the subtitles
    !python -m drive_pipeline --reuse-srt                # re-render output/narration.srt (e.g. after editing it)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import srt2mp4

DEFAULT_ROOT = Path("/content/drive/MyDrive/subgen")


def copy_in(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return dst


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Drive folder. Defaults to {DEFAULT_ROOT}.")
    ap.add_argument("--audio", default="narration.wav", help="Narration file name inside --root.")
    ap.add_argument("--clips", default="clips", help="Clips folder name inside --root.")
    ap.add_argument("--script", default="script.txt", help="Optional script file name inside --root.")
    ap.add_argument("--output", default="output", help="Output folder name inside --root.")
    ap.add_argument("--work", type=Path, help="Local working directory. Defaults to /content/subgen-work on Colab.")
    ap.add_argument("--encoder", choices=srt2mp4.ENCODERS, default="auto", help="Video encoder (auto = best available).")
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
        shutil.rmtree(work / "input", ignore_errors=True)
    local_out = work / "output"
    local_out.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Copying inputs from {root} to {work}", flush=True)
        audio = copy_in(audio_src, work / "input")
        clips = [copy_in(c, work / "input" / "clips") for c in clips_src]
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
                srt2mp4.parse_srt(srt_path), audio, mp4_path, encoder=args.encoder, background_clips=clips
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
