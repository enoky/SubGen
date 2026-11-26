"""Split sentence subtitles into short phrases for YouTube Shorts.

Each sentence cue becomes a few short phrases (a handful of words, one or two lines
on a phone), shown one after another: each phrase stays until the next replaces it,
and the sentence's own start and end are kept.

Phrase timing, best first:
* the WhisperX word timings saved next to the SRT (``narration.words.json``): the
  subtitle text is matched to the transcript word by word, so this works for script
  wording and lightly edited SRTs too;
* otherwise the sentence's text is spread over the stretches of actual speech in
  its time slot (pauses found in the audio are skipped), so a phrase that follows a
  pause starts when the speech does.

Usage:
    uv run python -m phrases narration.wav narration.srt -o narration_phrases.srt
    uv run python -m phrases narration.wav narration.srt --words narration.words.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import srt

MAX_CHARS = 24  # about two short lines at the Shorts phrase size
MIN_CHARS = 8  # don't close a phrase at a comma before it has this much
OVERFLOW = 8  # ...but a phrase may run this much longer to end at punctuation
PAUSE_REACH = 0.7  # how far from the estimate a phrase start may snap to the end of a pause
MAX_LOOKUP_SHIFT = 0.6  # a matched word timed further than this outside its cue is ignored
# Words that read badly at the end of a phrase ("the / veil"), so the break moves before them.
WEAK_ENDINGS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from", "with", "and", "or",
    "but", "nor", "as", "that", "than", "into", "onto", "upon", "its", "his", "her", "their",
    "our", "your", "my", "this", "these", "those", "is", "are", "was", "were", "be", "if",
}
BREAK_AFTER = re.compile(r"[,;:.!?…—–-][\"'”’»)\]]*$")
KEY_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    text: str


def word_key(word: str) -> str:
    return "".join(m.group(0) for m in KEY_RE.finditer(word.casefold().replace("’", "'"))).replace("'", "")


def split_phrases(text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[str]:
    """Break one subtitle's text into short phrases, preferring punctuation."""
    words = text.split()
    phrases: list[list[str]] = []
    cur: list[str] = []

    def length(ws: list[str]) -> int:
        return len(" ".join(ws))

    for word in words:
        # a little over the limit is fine if it lets the phrase end at punctuation
        if cur and BREAK_AFTER.search(word) and length(cur + [word]) <= max_chars + OVERFLOW:
            cur.append(word)
            phrases.append(cur)
            cur = []
            continue
        if cur and length(cur + [word]) > max_chars:
            carry: list[str] = []
            # don't end on "the", "of", ...: move those words to the next phrase
            while len(cur) > 1 and word_key(cur[-1]) in WEAK_ENDINGS and not BREAK_AFTER.search(cur[-1]):
                carry.insert(0, cur.pop())
            phrases.append(cur)
            cur = carry
        cur.append(word)
        if BREAK_AFTER.search(word) and length(cur) >= min_chars:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    # a lone word at the end joins the phrase before it, if it is short or still fits
    if len(phrases) > 1 and len(phrases[-1]) == 1 and (
        length(phrases[-1]) < min_chars or length(phrases[-2] + phrases[-1]) <= max_chars + OVERFLOW
    ):
        tail = phrases.pop()
        phrases[-1] = phrases[-1] + tail
    return [" ".join(p) for p in phrases]


def load_words(path: Path) -> list[tuple[str, float | None, float | None]]:
    return [(w, s, e) for w, s, e in json.loads(Path(path).read_text(encoding="utf-8"))]


def word_times_from_transcript(
    cues: list[tuple[float, float, str]], words: list[tuple[str, float | None, float | None]]
) -> list[list[tuple[float | None, float | None]]]:
    """(start, end) of every word in every cue, matched to the transcript's timed words
    (None where a word has no trustworthy match)."""
    cue_keys, owners = [], []
    for ci, (_, _, text) in enumerate(cues):
        for wi, w in enumerate(text.split()):
            k = word_key(w)
            if k:
                cue_keys.append(k)
                owners.append((ci, wi))
    tr_keys = [word_key(w) for w, _, _ in words]
    out: list[list[tuple[float | None, float | None]]] = [[(None, None)] * len(t.split()) for _, _, t in cues]
    matcher = difflib.SequenceMatcher(None, cue_keys, tr_keys, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a1 - a0):
            ci, wi = owners[a0 + k]
            _, s, e = words[b0 + k]
            cs, ce, _ = cues[ci]
            if s is None or e is None or s < cs - MAX_LOOKUP_SHIFT or e > ce + MAX_LOOKUP_SHIFT:
                continue
            out[ci][wi] = (s, e)
    return out


def speech_map(start: float, end: float, pauses: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
    """The stretches of speech between ``start`` and ``end`` (pauses removed) and their total length."""
    segs, t = [], start
    for a, b in sorted(pauses):
        if b <= t or a >= end:
            continue
        if a > t:
            segs.append((t, a))
        t = max(t, b)
    if t < end:
        segs.append((t, end))
    if not segs:
        segs = [(start, end)]
    return segs, sum(b - a for a, b in segs)


def at_speech_fraction(segs: list[tuple[float, float]], total: float, frac: float, resume: bool) -> float:
    """Time at which ``frac`` of the speech time has passed. At a pause boundary,
    ``resume`` gives the time speech resumes rather than when it stopped."""
    left = frac * total
    for i, (a, b) in enumerate(segs):
        d = b - a
        if left < d - 1e-6 or i == len(segs) - 1:
            return a + min(left, d)
        left -= d
        if resume and abs(left) < 1e-6:
            return segs[i + 1][0]
    return segs[-1][1]


def phrase_cues(
    cues: list[tuple[float, float, str]],
    pauses: list[tuple[float, float]] | None = None,
    words: list[tuple[str, float | None, float | None]] | None = None,
    max_chars: int = MAX_CHARS,
    min_duration: float = 0.3,
) -> list[Phrase]:
    """Split each (start, end, text) cue into timed phrases."""
    word_times = word_times_from_transcript(cues, words) if words else None
    out: list[Phrase] = []
    for ci, (start, end, text) in enumerate(cues):
        parts = split_phrases(text, max_chars)
        if len(parts) <= 1:
            out.append(Phrase(start, end, text.strip()))
            continue
        counts = [len(p.split()) for p in parts]
        letters = [sum(len(word_key(w)) for w in p.split()) or 1 for p in parts]
        segs, total = speech_map(start, end, pauses or [])

        # estimated start of each phrase from the speech-time share of its text
        starts, done = [], 0
        for k in range(len(parts)):
            starts.append(at_speech_fraction(segs, total, done / sum(letters), resume=True))
            done += letters[k]
        # after punctuation the speaker usually pauses: start when speech resumes
        for k in range(1, len(parts)):
            if BREAK_AFTER.search(parts[k - 1]):
                near = [b for a, b in (pauses or []) if start < a and b < end and abs(b - starts[k]) <= PAUSE_REACH]
                if near:
                    starts[k] = min(near, key=lambda b: abs(b - starts[k]))
        # best: the first word's own timing, when the transcript has it
        if word_times:
            wi = 0
            for k, n in enumerate(counts):
                s, _ = word_times[ci][wi] if wi < len(word_times[ci]) else (None, None)
                if k and s is not None and start < s < end:
                    starts[k] = s
                wi += n
        starts[0] = start

        # keep phrases in order and on screen long enough to read
        for k in range(1, len(starts)):
            starts[k] = max(starts[k], starts[k - 1] + min_duration)
        for k in range(len(starts) - 1, 0, -1):
            limit = (end if k == len(starts) - 1 else starts[k + 1]) - min_duration
            starts[k] = min(starts[k], limit)
        if any(b <= a for a, b in zip(starts, starts[1:] + [end])):
            out.append(Phrase(start, end, text.strip()))  # too short to split sensibly
            continue
        for k, p in enumerate(parts):
            out.append(Phrase(round(starts[k], 3), round(starts[k + 1] if k + 1 < len(parts) else end, 3), p))
    return out


def write_phrase_srt(phrases: list[Phrase], path: Path) -> None:
    subs = [srt.Subtitle(i, srt.timedelta(seconds=p.start), srt.timedelta(seconds=p.end), p.text)
            for i, p in enumerate(phrases, 1)]
    Path(path).write_text(srt.compose(subs), encoding="utf-8")


def phrase_srt(audio: Path, srt_in: Path, srt_out: Path, words_json: Path | None = None) -> list[Phrase]:
    """Write ``srt_out``: ``srt_in`` split into phrases, timed with the word timings in
    ``words_json`` when given (and present), otherwise from the speech in ``audio``."""
    from retime import detect_pauses

    subs = list(srt.parse(Path(srt_in).read_text(encoding="utf-8-sig")))
    cues = [(s.start.total_seconds(), s.end.total_seconds(), " ".join(s.content.split())) for s in subs]
    words = load_words(words_json) if words_json and Path(words_json).is_file() else None
    pauses = detect_pauses(audio, duration=cues[-1][1] if cues else None)
    phrases = phrase_cues(cues, pauses, words)
    write_phrase_srt(phrases, srt_out)
    how = "word timings" if words else "the speech in the audio"
    print(f"Split {len(cues)} subtitles into {len(phrases)} short phrases (timed from {how})", flush=True)
    return phrases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="Clean narration (no music).")
    ap.add_argument("srt", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="Defaults to <srt stem>_phrases.srt.")
    ap.add_argument("--words", type=Path, help="Word timings JSON written by subgen (<srt stem>.words.json).")
    args = ap.parse_args(argv)
    out = args.output or args.srt.with_name(args.srt.stem + "_phrases.srt")
    words = args.words or args.srt.with_suffix(".words.json")
    phrase_srt(args.audio, args.srt, out, words)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
