"""Run ffmpeg with a tqdm progress bar (percent, elapsed, ETA, speed).

ffmpeg's ``-progress pipe:1`` reports the output position as key=value lines on
stdout; that drives the bar. stderr is collected in the background (loudnorm
prints its JSON there) and shown if ffmpeg fails. tqdm bars render in a
terminal and in Colab cell output alike.
"""

from __future__ import annotations

import subprocess
import sys
import threading

from tqdm import tqdm

BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}{postfix}]"


def run_ffmpeg(cmd: list[str], total_seconds: float | None, desc: str) -> str:
    """Run ``cmd`` (starting with ``ffmpeg``) and return its stderr.

    Raises ``subprocess.CalledProcessError`` on failure, after printing the tail of
    ffmpeg's stderr.
    """
    if not cmd or cmd[0] != "ffmpeg":
        raise ValueError("cmd must start with 'ffmpeg'")
    # -stats would interleave ffmpeg's own status line with the bar.
    args = [a for a in cmd[1:] if a != "-stats"]
    full = ["ffmpeg", "-nostdin", "-nostats", "-progress", "pipe:1", *args]

    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    stderr_chunks: list[str] = []
    reader = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    reader.start()

    total = round(total_seconds, 1) if total_seconds and total_seconds > 0 else None
    bar = tqdm(total=total, desc=desc, unit="s", bar_format=BAR_FORMAT if total else None, dynamic_ncols=True)
    speed = ""
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "speed" and value.strip() not in ("", "N/A", "0x"):
                speed = value.strip()
            elif key in ("out_time_us", "out_time_ms") and value.isdigit():  # both are microseconds
                pos = int(value) / 1_000_000
                if total:
                    pos = min(pos, total)
                if pos > bar.n:
                    bar.update(pos - bar.n)
            elif key == "progress":
                if speed:
                    bar.set_postfix_str(f"speed {speed}", refresh=False)
                if value == "end" and total and bar.n < total:
                    bar.update(total - bar.n)
                bar.refresh()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        bar.close()
        reader.join()

    stderr = "".join(stderr_chunks)
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        print(f"ffmpeg failed (exit {proc.returncode}):\n{tail}", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(proc.returncode, full, stderr=stderr)
    return stderr
