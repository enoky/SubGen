"""Snap subtitle start and end times to where speech actually starts and stops.

WhisperX's word alignment often stretches a sentence's first word back into the
silence before it, or its last word forward into the pause after it, so subtitles
appear too soon or linger too long. This finds the pauses in the narration with
ffmpeg's silencedetect and adjusts each cue:

* a start inside a pause moves to where speech resumes;
* an end inside a pause moves back to where speech stopped;
* a cue overlapping its neighbour is pulled back to the pause between them;
* then a little padding is added (a short lead-in, a longer hold so the text can be
  finished), never running into the neighbouring cues or the next sentence's speech.

Run it on clean narration (not mixed with music).

Usage:
    uv run python -m retime narration.wav narration.srt               # rewrites narration.srt
    uv run python -m retime narration.wav narration.srt -o fixed.srt
"""

from __future__ import annotations

import argparse
import bisect
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import srt

from ffmpeg_progress import run_ffmpeg

NOISE_DB = -40.0  # below this level counts as a pause (narration is normalized to about -14 LUFS)
MIN_PAUSE = 0.15  # shorter dips are just gaps between words
LEAD = 0.05  # show the text this long before speech starts
TAIL = 0.30  # keep it this long after speech stops
MIN_GAP = 0.04  # always leave this much blank between two cues
OVERLAP_REACH = 0.6  # how far around an overlap to look for the pause between two sentences


@dataclass(frozen=True)
class Change:
    index: int
    old: tuple[float, float]
    new: tuple[float, float]

    @property
    def start_shift(self) -> float:
        return self.new[0] - self.old[0]

    @property
    def end_shift(self) -> float:
        return self.new[1] - self.old[1]


def detect_pauses(audio: Path, noise_db: float = NOISE_DB, min_pause: float = MIN_PAUSE,
                  duration: float | None = None) -> list[tuple[float, float]]:
    """(start, end) of every stretch quieter than ``noise_db`` lasting ``min_pause`` or longer."""
    stderr = run_ffmpeg(
        ["ffmpeg", "-hide_banner", "-i", str(audio), "-map", "0:a:0",
         "-af", f"silencedetect=noise={noise_db}dB:d={min_pause}", "-f", "null", "-"],
        duration, "    finding pauses",
    )
    pauses, start = [], None
    for kind, value in re.findall(r"silence_(start|end): (-?[\d.]+)", stderr):
        if kind == "start":
            start = max(0.0, float(value))
        elif start is not None:
            pauses.append((start, float(value)))
            start = None
    if start is not None:  # silence running to the end of the file
        pauses.append((start, float("inf")))
    return pauses


class Pauses:
    def __init__(self, pauses: list[tuple[float, float]]):
        self.pauses = sorted(pauses)
        self.starts = [a for a, _ in self.pauses]

    def containing(self, t: float) -> tuple[float, float] | None:
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.pauses[i][0] <= t <= self.pauses[i][1]:
            return self.pauses[i]
        return None

    def first_after(self, t: float) -> tuple[float, float] | None:
        """First pause starting at or after ``t``."""
        i = bisect.bisect_left(self.starts, t)
        return self.pauses[i] if i < len(self.pauses) else None

    def last_before(self, t: float) -> tuple[float, float] | None:
        """Last pause ending at or before ``t``."""
        i = bisect.bisect_right(self.starts, t) - 1
        while i >= 0 and self.pauses[i][1] > t:
            i -= 1
        return self.pauses[i] if i >= 0 else None


def retime(
    times: list[tuple[float, float]],
    pauses: list[tuple[float, float]],
    lead: float = LEAD,
    tail: float = TAIL,
    min_gap: float = MIN_GAP,
) -> list[tuple[float, float]]:
    """New (start, end) for each cue (see module docs). Cues must be in time order."""
    p = Pauses(pauses)

    # 1. Snap each edge that sits inside a pause to the speech next to it.
    speech = []
    for start, end in times:
        s, e = start, end
        hit = p.containing(s)
        if hit and hit[1] < e:
            s = hit[1]  # speech resumes here
        hit = p.containing(e)
        if hit and hit[0] > s:
            e = hit[0]  # speech stopped here
        speech.append([s, e])

    # 2. Untangle neighbours whose speech still overlaps (a word stolen from the
    #    other sentence): split them at the pause between them, if there is one.
    #    The gap between sentences is normally the longest pause near the overlap
    #    (commas inside a sentence give shorter ones).
    for i in range(1, len(speech)):
        prev, cur = speech[i - 1], speech[i]
        if prev[1] > cur[0]:
            lo, hi = min(prev[1], cur[0]) - OVERLAP_REACH, max(prev[1], cur[0]) + OVERLAP_REACH
            near = [(a, b) for a, b in p.pauses
                    if prev[0] < a and b < cur[1] and a <= hi and b >= lo]
            if near:
                prev[1], cur[0] = max(near, key=lambda ab: ab[1] - ab[0])
            else:
                mid = (prev[1] + cur[0]) / 2
                prev[1], cur[0] = mid, mid

    # 3. Padding, kept clear of neighbouring cues and the next stretch of speech.
    out = []
    for i, (s, e) in enumerate(speech):
        prev_end = out[-1][1] if out else 0.0
        before = p.last_before(s)  # the pause just before this speech
        new_s = max(s - lead, before[0] if before else 0.0, prev_end + min_gap if out else 0.0)
        after = p.containing(e) or p.first_after(e)
        limit = after[1] if after else e + tail
        if i + 1 < len(speech):
            limit = min(limit, speech[i + 1][0] - lead - min_gap)
        new_e = max(e, min(e + tail, limit))
        if new_e <= new_s:
            new_s, new_e = s, max(e, s + 0.001)
        out.append((round(new_s, 3), round(new_e, 3)))
    return out


def retime_with_report(
    times: list[tuple[float, float]], pauses: list[tuple[float, float]], **kw
) -> tuple[list[tuple[float, float]], list[Change]]:
    new = retime(times, pauses, **kw)
    changes = [Change(i, old, nw) for i, (old, nw) in enumerate(zip(times, new))
               if abs(old[0] - nw[0]) > 0.15 or abs(old[1] - nw[1]) > 0.15]
    return new, changes


def summarize(changes: list[Change], total: int, texts: list[str] | None = None, show: int = 5) -> str:
    if not changes:
        return f"Subtitle timing: all {total} cues already match the speech"
    starts = [c for c in changes if abs(c.start_shift) > 0.15]
    ends = [c for c in changes if abs(c.end_shift) > 0.15]
    lines = [f"Subtitle timing: adjusted {len(changes)} of {total} cues to the speech "
             f"({len(starts)} starts, {len(ends)} ends moved by more than 0.15s)"]
    worst = sorted(changes, key=lambda c: -max(abs(c.start_shift), abs(c.end_shift)))[:show]
    for c in worst:
        text = (texts[c.index] if texts else "")[:50]
        lines.append(f"    #{c.index + 1}: start {c.start_shift:+.2f}s, end {c.end_shift:+.2f}s  {text}")
    return "\n".join(lines)


def retime_srt(audio: Path, srt_in: Path, srt_out: Path | None = None, **kw) -> list[Change]:
    """Retime an SRT file against ``audio``; writes ``srt_out`` (default: in place)."""
    subs = list(srt.parse(Path(srt_in).read_text(encoding="utf-8-sig")))
    times = [(s.start.total_seconds(), s.end.total_seconds()) for s in subs]
    pauses = detect_pauses(audio, duration=times[-1][1] if times else None)
    new, changes = retime_with_report(times, pauses, **kw)
    for sub, (a, b) in zip(subs, new):
        sub.start = srt.timedelta(seconds=a)
        sub.end = srt.timedelta(seconds=b)
    print(summarize(changes, len(subs), [s.content.replace("\n", " ") for s in subs]), flush=True)
    Path(srt_out or srt_in).write_text(srt.compose(subs), encoding="utf-8")
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="Clean narration (no music).")
    ap.add_argument("srt", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="Defaults to rewriting the input SRT (a .bak copy is kept).")
    ap.add_argument("--lead", type=float, default=LEAD, help=f"seconds shown before speech (default {LEAD})")
    ap.add_argument("--tail", type=float, default=TAIL, help=f"seconds held after speech (default {TAIL})")
    args = ap.parse_args(argv)
    if not args.output:
        shutil.copyfile(args.srt, args.srt.with_suffix(args.srt.suffix + ".bak"))
    retime_srt(args.audio, args.srt, args.output, lead=args.lead, tail=args.tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
