from __future__ import annotations

import argparse
import difflib
import math
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "project"
DEFAULT_INPUT_SRT = PROJECT_DIR / "autogen-subs.srt"
DEFAULT_SCRIPT = PROJECT_DIR / "original-script.txt"
DEFAULT_OUTPUT_SRT = PROJECT_DIR / "edited-subs.srt"

TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
TIMECODE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}"
)
OPENING_PUNCTUATION = set("\"'([{<\u201c\u2018")

MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u00a6": "\u2026",
    "\u00e2\u20ac\u00a2": "\u2022",
    "\u00c2\u00a0": " ",
    "\u00c2": "",
}


@dataclass(frozen=True)
class SubtitleBlock:
    index: int
    timecode: str
    content: str


@dataclass(frozen=True)
class Token:
    key: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve timings from an auto-generated SRT while replacing caption "
            "text with aligned slices from the original script."
        )
    )
    parser.add_argument("--input-srt", type=Path, default=DEFAULT_INPUT_SRT)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--output-srt", type=Path, default=DEFAULT_OUTPUT_SRT)
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=0,
        help="Wrap subtitle lines to this width. Leave as 0 to keep each subtitle unwrapped.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.70,
        help="Abort if token alignment similarity is below this ratio.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alignment statistics without writing the output SRT.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def repair_mojibake(text: str) -> str:
    repaired = text
    for broken, fixed in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(broken, fixed)
    return repaired


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_token(token: str) -> str:
    token = token.casefold()
    token = token.replace("\u2018", "'").replace("\u2019", "'")
    token = unicodedata.normalize("NFKD", token)
    token = "".join(ch for ch in token if not unicodedata.combining(ch))
    return re.sub(r"['`]", "", token)


def tokenize(text: str) -> list[Token]:
    tokens = []
    for match in TOKEN_RE.finditer(text):
        key = normalize_token(match.group(0))
        if key:
            tokens.append(Token(key=key, start=match.start(), end=match.end()))
    return tokens


def parse_srt(text: str) -> list[SubtitleBlock]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks = []
    for chunk in re.split(r"\n{2,}", normalized):
        lines = [line.rstrip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue

        if lines[0].strip().isdigit():
            index = int(lines[0].strip())
            timecode_index = 1
        else:
            index = len(blocks) + 1
            timecode_index = 0

        if timecode_index >= len(lines) or not TIMECODE_RE.match(lines[timecode_index]):
            raise ValueError(f"Invalid SRT block near block {index}: {chunk[:120]!r}")

        blocks.append(
            SubtitleBlock(
                index=index,
                timecode=lines[timecode_index].strip(),
                content="\n".join(lines[timecode_index + 1 :]).strip(),
            )
        )

    return blocks


def caption_token_ranges(blocks: list[SubtitleBlock]) -> tuple[list[str], list[tuple[int, int]]]:
    caption_keys = []
    ranges = []

    for block in blocks:
        start = len(caption_keys)
        caption_keys.extend(token.key for token in tokenize(block.content))
        ranges.append((start, len(caption_keys)))

    return caption_keys, ranges


def map_caption_tokens_to_script(
    caption_keys: list[str], script_keys: list[str]
) -> tuple[list[float | None], float, int]:
    matcher = difflib.SequenceMatcher(None, caption_keys, script_keys, autojunk=False)
    mapped: list[float | None] = [None] * len(caption_keys)
    equal_tokens = 0

    for tag, cap_start, cap_end, script_start, script_end in matcher.get_opcodes():
        cap_len = cap_end - cap_start
        script_len = script_end - script_start

        if tag == "equal":
            equal_tokens += cap_len
            for offset in range(cap_len):
                mapped[cap_start + offset] = float(script_start + offset)
        elif tag == "replace" and cap_len:
            if script_len:
                for offset in range(cap_len):
                    relative = (offset + 0.5) / cap_len
                    script_offset = min(script_len - 1, int(relative * script_len))
                    mapped[cap_start + offset] = float(script_start + script_offset)
            else:
                for offset in range(cap_len):
                    mapped[cap_start + offset] = float(script_start)
        elif tag == "delete":
            for offset in range(cap_len):
                mapped[cap_start + offset] = float(script_start) - 0.5

    last_seen = -0.5
    for index, value in enumerate(mapped):
        if value is None:
            continue
        value = max(value, last_seen)
        mapped[index] = value
        last_seen = value

    return mapped, matcher.ratio(), equal_tokens


def fallback_boundary(
    caption_start: int,
    total_caption_tokens: int,
    total_script_tokens: int,
) -> int:
    if not total_caption_tokens:
        return 0
    return round((caption_start / total_caption_tokens) * total_script_tokens)


def compute_script_boundaries(
    block_ranges: list[tuple[int, int]],
    mapped_positions: list[float | None],
    script_token_count: int,
) -> list[int]:
    block_first_positions: list[float | None] = []
    block_last_positions: list[float | None] = []

    for start, end in block_ranges:
        positions = [pos for pos in mapped_positions[start:end] if pos is not None]
        block_first_positions.append(positions[0] if positions else None)
        block_last_positions.append(positions[-1] if positions else None)

    boundaries = [0] * (len(block_ranges) + 1)
    boundaries[-1] = script_token_count

    total_caption_tokens = len(mapped_positions)
    for index in range(1, len(block_ranges)):
        previous_last = block_last_positions[index - 1]
        current_first = block_first_positions[index]

        if previous_last is not None and current_first is not None:
            boundary = math.floor((previous_last + current_first + 1.0) / 2.0)
        elif current_first is not None:
            boundary = math.floor(current_first)
        elif previous_last is not None:
            boundary = math.floor(previous_last) + 1
        else:
            boundary = fallback_boundary(
                block_ranges[index][0], total_caption_tokens, script_token_count
            )

        boundaries[index] = max(0, min(script_token_count, boundary))

    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1])

    return boundaries


def slice_script_text(
    script_text: str,
    script_tokens: list[Token],
    start_token: int,
    end_token: int,
) -> str:
    if start_token >= end_token or start_token >= len(script_tokens):
        return ""

    end_token = min(end_token, len(script_tokens))
    start_char = script_tokens[start_token].start
    while start_char > 0 and script_text[start_char - 1] in OPENING_PUNCTUATION:
        start_char -= 1

    end_char = script_tokens[end_token - 1].end
    next_token_start = (
        script_tokens[end_token].start if end_token < len(script_tokens) else len(script_text)
    )
    trailing = script_text[end_char:next_token_start].rstrip()
    while trailing and trailing[-1] in OPENING_PUNCTUATION:
        trailing = trailing[:-1].rstrip()
    end_char += len(trailing)

    return normalize_whitespace(script_text[start_char:end_char])


def wrap_caption(text: str, max_line_length: int) -> str:
    if max_line_length <= 0:
        return text

    lines = textwrap.wrap(
        text,
        width=max_line_length,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


def compose_srt(blocks: list[SubtitleBlock], contents: list[str]) -> str:
    rendered_blocks = []
    for index, (block, content) in enumerate(zip(blocks, contents), start=1):
        rendered_blocks.append(f"{index}\n{block.timecode}\n{content.strip()}")
    return "\n\n".join(rendered_blocks) + "\n"


def build_edited_contents(
    blocks: list[SubtitleBlock],
    script_text: str,
    max_line_length: int,
) -> tuple[list[str], float, int, int, int]:
    script_tokens = tokenize(script_text)
    script_keys = [token.key for token in script_tokens]
    caption_keys, block_ranges = caption_token_ranges(blocks)

    mapped_positions, ratio, equal_tokens = map_caption_tokens_to_script(
        caption_keys, script_keys
    )
    boundaries = compute_script_boundaries(
        block_ranges, mapped_positions, len(script_tokens)
    )

    edited_contents = []
    for block_index, block in enumerate(blocks):
        start_token = boundaries[block_index]
        end_token = boundaries[block_index + 1]
        content = slice_script_text(script_text, script_tokens, start_token, end_token)
        if not content:
            content = normalize_whitespace(block.content)
        edited_contents.append(wrap_caption(content, max_line_length))

    return edited_contents, ratio, equal_tokens, len(caption_keys), len(script_tokens)


def main() -> None:
    args = parse_args()

    input_srt = args.input_srt.resolve()
    script_path = args.script.resolve()
    output_srt = args.output_srt.resolve()

    blocks = parse_srt(read_text(input_srt))
    script_text = repair_mojibake(read_text(script_path))

    if not blocks:
        raise SystemExit(f"No subtitle blocks found in {input_srt}")

    edited_contents, ratio, equal_tokens, caption_tokens, script_tokens = build_edited_contents(
        blocks, script_text, args.max_line_length
    )

    print(f"Subtitle blocks: {len(blocks)}")
    print(f"Auto subtitle tokens: {caption_tokens}")
    print(f"Script tokens: {script_tokens}")
    print(f"Exact aligned tokens: {equal_tokens}")
    print(f"Alignment ratio: {ratio:.3f}")

    if ratio < args.min_ratio:
        raise SystemExit(
            f"Alignment ratio {ratio:.3f} is below --min-ratio {args.min_ratio:.3f}; "
            "check that the SRT and script describe the same audio."
        )

    if args.dry_run:
        print("Dry run only; no output written.")
        return

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text(compose_srt(blocks, edited_contents), encoding="utf-8")
    print(f"Wrote {output_srt}")


if __name__ == "__main__":
    main()
