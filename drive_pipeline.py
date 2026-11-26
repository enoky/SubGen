"""One-shot SubGen pipeline for a Google Drive folder (made for Colab).

Expected layout under ``--root`` (Colab: /content/drive/MyDrive/subgen):

    narration.wav       narration audio (required)
    clips/*.mp4         background clips, joined in natural filename order (required)
    script.txt          original script, used for cue wording if present (optional)
    528Hz_ambient.m4a   background music under the narration, if present (optional;
                        if missing, the one in MyDrive/subgen is used)
    output/             narration.srt, narration.mp4 and narration_norm.wav are written here

The narration is first loudness-normalized for YouTube (-14 LUFS, -1.5 dBTP,
48 kHz); the normalized audio is what gets transcribed and put in the video.
The result is reused on later runs while narration.wav and the loudness settings
are unchanged (checked against output/.narration_norm.json).

The clips are first copied to local disk, 16 at a time: a single Drive read
stream is slower than the GPU encode (it capped full runs at ~2.6x realtime),
but parallel reads add up. Clips already copied earlier in the session are
skipped. --stream-clips reads them straight from Drive during the encode instead. Clip details are cached in
output/.clip_info_cache.json, so reruns over the same clips start right away.
The narration and outputs go through local disk, then outputs are copied to Drive.

Usage (Colab, after mounting Drive):
    !python -m drive_pipeline
    !python -m drive_pipeline --model large-v3-turbo     # extra options go to subgen
    !python -m drive_pipeline --srt-only                 # just the subtitles
    !python -m drive_pipeline --reuse-srt                # re-render output/narration.srt (e.g. after editing it)
    !python -m drive_pipeline --lufs -16                 # different loudness target
    !python -m drive_pipeline --no-normalize             # use narration.wav as is
    !python -m drive_pipeline --music-level -24          # quieter background music
    !python -m drive_pipeline --no-music                 # no background music
    !python -m drive_pipeline --tint-opacity 0.3         # lighter indigo tint over the clips (0 = off)
    !python -m drive_pipeline --no-retime                # keep subtitle times as WhisperX gave them
    !python -m drive_pipeline --root /content/drive/MyDrive/subgen_shorts   # a YouTube Short
        (Shorts show short phrases; --sentences shows whole sentences)
        (portrait clips make a 1080x1920 Short automatically; --format shorts/landscape forces it)
    !python -m drive_pipeline --stream-clips             # read clips from Drive during the encode
    !python -m drive_pipeline --check-routes             # just check which video route works (~1 min)
    !python -m drive_pipeline --benchmark                # time each video route and the Drive read speed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

from tqdm import tqdm

import background_music
import normalize_audio
import phrases
import retime
import srt2mp4

DEFAULT_ROOT = Path("/content/drive/MyDrive/subgen")
MAIN_FOLDER_NAME = "subgen"  # a sibling folder with this name is checked for shared background music


def on_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def default_root() -> Path | None:
    """Colab: the Drive folder. Elsewhere: the current folder, if it holds a narration."""
    if on_colab():
        return DEFAULT_ROOT
    return Path.cwd() if (Path.cwd() / "narration.wav").is_file() else None


def default_work_dir(root: Path) -> Path:
    """Where intermediate files and caches live between runs. On Colab, local disk
    under /content. Elsewhere a per-project folder on local disk (never next to the
    project, which may be a Google Drive for desktop folder that would sync it)."""
    if on_colab():
        return Path("/content/subgen-work")
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "SubGen" / "work"
    tag = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    return base / f"{root.resolve().name or 'project'}-{tag}"


def music_candidates(root: Path, name: str) -> list[Path]:
    """Where to look for the background music: the project folder, then the main
    ``subgen`` folder next to it (so a Shorts folder shares its music), then the
    Colab Drive folder."""
    out: list[Path] = []
    for cand in (root / name, root.parent / MAIN_FOLDER_NAME / name, DEFAULT_ROOT / name):
        if cand not in out:
            out.append(cand)
    return out


def copy_in(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return dst


NORM_META = ".narration_norm.json"


def normalized_narration(
    audio_src: Path, out_dir: Path, work_in: Path, lufs: float, true_peak: float
) -> Path:
    """Loudness-normalized copy of ``audio_src`` on local disk.

    Normalizing a long narration takes minutes, so the result is reused while the
    source file (name, size, modification time) and the settings are unchanged:
    first from local disk (same Colab session), then from the copy saved on Drive.
    Each copy has a small JSON file next to it describing what it was made from."""
    st = audio_src.stat()
    key = {"source": audio_src.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
           "lufs": lufs, "true_peak": true_peak, "sample_rate": normalize_audio.SAMPLE_RATE,
           "codec": normalize_audio.CODEC}
    name = f"{audio_src.stem}{normalize_audio.SUFFIX}.wav"
    local, local_meta = work_in / name, work_in / NORM_META
    saved, saved_meta = out_dir / name, out_dir / NORM_META

    def matches(meta: Path, wav: Path) -> bool:
        try:
            return wav.is_file() and json.loads(meta.read_text(encoding="utf-8")) == key
        except (OSError, ValueError):
            return False

    work_in.mkdir(parents=True, exist_ok=True)
    if matches(local_meta, local):
        print(f"Reusing normalized narration {local} (narration.wav unchanged)", flush=True)
        return local
    if matches(saved_meta, saved):
        print(f"Reusing normalized narration {saved} (narration.wav unchanged)", flush=True)
        local_meta.unlink(missing_ok=True)
        shutil.copyfile(saved, local)
        local_meta.write_text(json.dumps(key), encoding="utf-8")
        return local

    print(f"Copying narration from {audio_src.parent}", flush=True)
    raw = copy_in(audio_src, work_in)
    print(f"Normalizing narration to {lufs:g} LUFS, {true_peak:g} dBTP", flush=True)
    local_meta.unlink(missing_ok=True)
    saved_meta.unlink(missing_ok=True)
    normalize_audio.normalize(raw, local, lufs, true_peak)
    local_meta.write_text(json.dumps(key), encoding="utf-8")
    shutil.copyfile(local, saved)
    saved_meta.write_text(json.dumps(key), encoding="utf-8")  # written last: marks the Drive copy complete
    print(f"Saved {saved}", flush=True)
    return local


MIX_META = ".narration_mix.json"


def mixed_narration(
    narration: Path, music_src: Path, work_in: Path, level: float, fade: float,
    narration_lufs: float | None, true_peak: float,
) -> Path:
    """``narration`` with the background music mixed under it, reused within the
    session while the narration, the music and the settings are unchanged. (The mix
    is large, so it is kept on local disk only; making it takes about a minute.)"""
    def stamp(p: Path) -> dict:
        st = p.stat()
        return {"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}

    key = {"narration": stamp(narration), "music": stamp(music_src), "level": level, "fade": fade,
           "narration_lufs": narration_lufs, "true_peak": true_peak}
    out, meta = work_in / f"{narration.stem.removesuffix(normalize_audio.SUFFIX)}_mix.wav", work_in / MIX_META
    try:
        if out.is_file() and json.loads(meta.read_text(encoding="utf-8")) == key:
            print(f"Reusing background-music mix {out} (nothing changed)", flush=True)
            return out
    except (OSError, ValueError):
        pass
    meta.unlink(missing_ok=True)
    print(f"Mixing {music_src.name} under the narration", flush=True)
    music = copy_in(music_src, work_in)
    background_music.mix(narration, music, out, level, fade, narration_lufs, true_peak)
    meta.write_text(json.dumps(key), encoding="utf-8")
    return out


def copy_clips(srcs: list[Path], dst_dir: Path, workers: int = 16) -> list[Path]:
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
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=total, desc="Copying clips", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for size in pool.map(copy, srcs):
            bar.update(size)
    took = time.monotonic() - start
    print(f"Clips ready locally: {total / 1e9:.1f} GB in {took:.0f}s ({total / 1e6 / max(took, 1e-6):.0f} MB/s)",
          flush=True)
    return [dst_dir / s.name for s in srcs]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help=(
        f"Project folder. Defaults to {DEFAULT_ROOT} on Colab, or the current folder "
        "if it contains narration.wav."))
    ap.add_argument("--audio", default="narration.wav", help="Narration file name inside --root.")
    ap.add_argument("--clips", default="clips", help="Clips folder name inside --root.")
    ap.add_argument("--script", default="script.txt", help="Optional script file name inside --root.")
    ap.add_argument("--output", default="output", help="Output folder name inside --root.")
    ap.add_argument("--work", type=Path, help=(
        "Working folder for intermediate files and caches. Defaults to /content/subgen-work on "
        "Colab, or a per-project folder under %%LOCALAPPDATA%%\\SubGen\\work (~/.cache/SubGen/work)."))
    ap.add_argument("--encoder", choices=srt2mp4.ENCODERS, default="auto", help="Video encoder (auto = best available).")
    clips = ap.add_mutually_exclusive_group()
    clips.add_argument(
        "--stream-clips", action="store_true",
        help="Read clips in place during the encode (the default away from Colab).",
    )
    clips.add_argument(
        "--copy-clips", action="store_true",
        help="Copy clips to the working folder first, 16 at a time (the default on Colab, where Drive "
             "reads are slow; useful locally for a Google Drive for desktop folder that isn't offline).",
    )
    ap.add_argument(
        "--benchmark", action="store_true",
        help="Time each video route and the Drive read speed on ~40 s of your clips (a few minutes), then stop.",
    )
    ap.add_argument(
        "--check-routes", action="store_true",
        help="Only check which video route works on this machine (about a minute), then stop.",
    )
    ap.add_argument(
        "--cpu-filters", action="store_true",
        help="Skip the GPU filter routes (crop/scale/overlay on the CPU; slower, GPU still decodes/encodes).",
    )
    ap.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization of the narration.")
    ap.add_argument(
        "--music", default="528Hz_ambient.m4a",
        help="Background music file name inside --root, used if it exists (default 528Hz_ambient.m4a).",
    )
    ap.add_argument("--no-music", action="store_true", help="Don't add background music.")
    ap.add_argument(
        "--music-level", type=float, default=background_music.DEFAULT_LEVEL,
        help=f"Music loudness relative to the narration in LU (default {background_music.DEFAULT_LEVEL:g}; "
             "about -18 to -25 suits music under voice-over).",
    )
    ap.add_argument(
        "--music-fade", type=float, default=background_music.DEFAULT_FADE,
        help=f"Music fade-out at the end of the video, in seconds (default {background_music.DEFAULT_FADE:g}).",
    )
    ap.add_argument(
        "--tint", type=srt2mp4.parse_color, default=srt2mp4.VIDEO_STYLE.tint,
        help="Colour washed over the clips behind the text to darken them (default #1E1248, deep indigo).",
    )
    ap.add_argument(
        "--tint-opacity", type=float, default=srt2mp4.VIDEO_STYLE.tint_opacity,
        help=f"Strength of the tint, 0-1 (default {srt2mp4.VIDEO_STYLE.tint_opacity:g}; 0 turns it off).",
    )
    ap.add_argument(
        "--format", dest="fmt", choices=srt2mp4.FORMATS, default="auto",
        help="landscape (1920x1080), shorts (1080x1920 portrait for YouTube Shorts), or auto "
             "(default): portrait clips make a Short, landscape clips a regular video.",
    )
    ap.add_argument(
        "--sentences", action="store_true",
        help="In Shorts, show whole sentences instead of short phrases.",
    )
    ap.add_argument(
        "--no-retime", action="store_true",
        help="Keep subtitle times as they are. By default they are snapped to the speech, "
             "including a reused SRT (the original is kept once as narration.srt.bak).",
    )
    ap.add_argument("--retime-lead", type=float, default=retime.LEAD,
                    help=f"Seconds a subtitle appears before its speech (default {retime.LEAD:g}).")
    ap.add_argument("--retime-tail", type=float, default=retime.TAIL,
                    help=f"Seconds a subtitle stays after its speech (default {retime.TAIL:g}).")
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

    root = args.root or default_root()
    if root is None:
        raise SystemExit(
            "Which project folder? Pass --root FOLDER (holding narration.wav, clips/ and output/), "
            "or run this from inside that folder."
        )
    if not root.is_dir():
        hint = (" On Colab, mount Drive first (from google.colab import drive; "
                "drive.mount('/content/drive')) and create MyDrive/subgen.") if on_colab() else ""
        raise SystemExit(f"{root} not found.{hint}")
    audio_src = root / args.audio
    if not audio_src.is_file():
        raise SystemExit(f"Narration not found: {audio_src}")
    clips_src = srt2mp4.find_clips(root / args.clips) if not args.srt_only else []
    script_src = root / args.script
    out_dir = root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_name, mp4_name = f"{audio_src.stem}.srt", f"{audio_src.stem}.mp4"
    words_name, phrases_name = f"{audio_src.stem}.words.json", f"{audio_src.stem}_phrases.srt"

    work = args.work or default_work_dir(root)
    work.mkdir(parents=True, exist_ok=True)
    if not on_colab():
        print(f"Working folder: {work}", flush=True)
    if work.exists():
        # Keep input/clips and the normalized narration so the next run can reuse them.
        for old in (work / "input").glob("script*"):
            old.unlink()
    local_out = work / "output"
    local_out.mkdir(parents=True, exist_ok=True)

    if args.benchmark or args.check_routes or args.no_normalize:
        print(f"Copying narration from {root} to {work}", flush=True)
        audio = copy_in(audio_src, work / "input")
    else:
        audio = normalized_narration(audio_src, out_dir, work / "input", args.lufs, args.true_peak)
    if args.benchmark:
        srt2mp4.benchmark_routes(
            clips_src, audio, work, probe_cache=out_dir / ".clip_info_cache.json",
        )
        return 0
    if args.check_routes:
        srt2mp4.render_clip_video(
            [], audio, local_out / "route_check.mp4", clips_src, encoder=args.encoder,
            work=work / "route_check", probe_cache=out_dir / ".clip_info_cache.json",
            cpu_filters=args.cpu_filters, check_only=True, fmt=args.fmt,
        )
        return 0
    copy = args.copy_clips or (on_colab() and not args.stream_clips)
    clips = copy_clips(clips_src, work / "input" / "clips") if clips_src and copy else clips_src
    srt_path = local_out / srt_name
    mp4_path = local_out / mp4_name

    if args.reuse_srt:
        existing = out_dir / srt_name
        if not existing.is_file():
            raise SystemExit(f"--reuse-srt: {existing} does not exist yet.")
        shutil.copyfile(existing, srt_path)
        print(f"Reusing {existing}", flush=True)
        (local_out / words_name).unlink(missing_ok=True)
        if (out_dir / words_name).is_file():  # word timings from the original transcription
            shutil.copyfile(out_dir / words_name, local_out / words_name)
        if not args.no_retime:
            backup = existing.with_name(existing.name + ".bak")
            if not backup.exists():
                shutil.copyfile(existing, backup)
            print("Snapping subtitle times to the speech", flush=True)
            retime.retime_srt(audio, srt_path, lead=args.retime_lead, tail=args.retime_tail)
            shutil.copyfile(srt_path, existing)
            print(f"Saved {existing} (original kept as {backup.name})", flush=True)
    else:
        import subgen  # imported here so --reuse-srt doesn't load torch/WhisperX

        cmd = [str(audio), "-o", str(srt_path), *subgen_args,
               "--retime-lead", str(args.retime_lead), "--retime-tail", str(args.retime_tail)]
        if args.no_retime:
            cmd.append("--no-retime")
        if script_src.is_file():
            print(f"Using script {script_src}", flush=True)
            cmd += ["--script", str(copy_in(script_src, work / "input"))]
        subgen.main(cmd)
        subgen.free_torch_memory()
        shutil.copyfile(srt_path, out_dir / srt_name)
        print(f"Saved {out_dir / srt_name}", flush=True)
        if (local_out / words_name).is_file():
            shutil.copyfile(local_out / words_name, out_dir / words_name)

    if not args.srt_only:
        # The music goes only into the video; transcription above used the clean narration.
        video_audio = audio
        # A Shorts project in its own folder can share the main folder's music.
        music_src = next((m for m in music_candidates(root, args.music) if m.is_file()),
                         root / args.music)
        if args.no_music:
            pass
        elif music_src.is_file():
            video_audio = mixed_narration(
                audio, music_src, work / "input", args.music_level, args.music_fade,
                None if args.no_normalize else args.lufs, args.true_peak,
            )
        else:
            print(f"No background music ({music_src} not found)", flush=True)
        style = replace(srt2mp4.VIDEO_STYLE, tint=args.tint, tint_opacity=args.tint_opacity)
        video_srt = srt_path
        short = args.fmt == "shorts" or (
            args.fmt == "auto" and srt2mp4.clips_are_portrait(clips, out_dir / ".clip_info_cache.json"))
        if short and not args.sentences:
            # Shorts: a few words at a time, bigger. The sentence SRT stays as it is
            # (it is the one to upload as YouTube captions).
            video_srt = local_out / phrases_name
            phrases.phrase_srt(audio, srt_path, video_srt, local_out / words_name)
            shutil.copyfile(video_srt, out_dir / phrases_name)
            print(f"Saved {out_dir / phrases_name}", flush=True)
            style = srt2mp4.shorts_phrase_style(style)
        srt2mp4.render_video(
            srt2mp4.parse_srt(video_srt), video_audio, mp4_path, encoder=args.encoder, background_clips=clips,
            probe_cache=out_dir / ".clip_info_cache.json", cpu_filters=args.cpu_filters,
            style=style,
            fmt=args.fmt,
        )
        print(f"Copying video to {out_dir / mp4_name}", flush=True)
        shutil.copyfile(mp4_path, out_dir / mp4_name)
        size_mb = mp4_path.stat().st_size / 1e6
        mp4_path.unlink()  # the copy in output/ is the one to keep; don't hold gigabytes twice
        print(f"Saved {out_dir / mp4_name} ({size_mb:.0f} MB)", flush=True)

    print(f"Done in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
