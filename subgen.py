from __future__ import annotations

import argparse
import bisect
import difflib
import gc
import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import spacy
import srt
import torch


def _preload_cuda12_cublas() -> None:
    """Load cuBLAS 12 for ctranslate2 (faster-whisper), which is built against CUDA 12
    while torch ships CUDA 13 DLLs. Preloading by full path lets its by-name load succeed."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        import nvidia.cublas
    except ImportError:
        return
    bin_dir = Path(nvidia.cublas.__path__[0]) / "bin"
    for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
        dll = bin_dir / name
        if dll.exists():
            ctypes.CDLL(str(dll))


_preload_cuda12_cublas()

import whisperx  # noqa: E402


DEFAULT_INPUT = Path("audio.wav")
LANGUAGE = "en"
SPACY_MODEL = "en_core_web_lg"
DEFAULT_PUNCTUATION_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
# GPUs below this much VRAM get int8 weights and smaller batches so large-v3 fits (e.g. 4 GB laptops).
LOW_VRAM_GIB = 6.0

# Labels the restoration model emits; only these signal a sentence boundary.
SENTENCE_END_PUNCT = {".", "?", "!"}
RESTORABLE_PUNCT = {".", ",", "?", "-", ":"}
# Trailing punctuation stripped from a raw word before the model re-punctuates it.
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?…]+$")

# Word tokenizer used to align the transcript against the original script. Mirrors
# edit_subs_from_script.py so the two tools agree on what counts as a word.
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)

# Common mojibake sequences seen when a UTF-8 script is mis-decoded as Latin-1.
MOJIBAKE_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "â€”": "—",
    "â€“": "–",
    "â€¦": "…",
    "â€¢": "•",
    "Â ": " ",
    "Â": "",
}


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float | None
    end: float | None
    fallback_start: float | None
    fallback_end: float | None
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SentenceCue:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Token:
    key: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sentence-level SRT subtitle file from English speech using WhisperX."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Audio file to transcribe. Defaults to audio.wav.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="SRT output path. Defaults to ./project/autogen-subs.srt.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name understood by faster-whisper/WhisperX.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory for downloaded or cached WhisperX models.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached models.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index used when --device cuda.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "Whisper compute type. Defaults to float16 on CUDA (int8_float16 on GPUs "
            f"under {LOW_VRAM_GIB:g} GB VRAM) and int8 on CPU."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"WhisperX transcription batch size. Defaults to 16, or 4 on GPUs under {LOW_VRAM_GIB:g} GB VRAM.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for torch and faster-whisper. Use 0 to keep torch defaults.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Maximum VAD speech chunk size in seconds.",
    )
    parser.add_argument(
        "--vad-method",
        choices=("pyannote", "silero"),
        default="silero",
        help="Voice activity detector used by WhisperX.",
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="VAD onset threshold.",
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.363,
        help="VAD offset threshold.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help=(
            "Original script text file (e.g. ./project/original-script.txt). When "
            "given, sentence boundaries are taken from the script's real punctuation "
            "and the script's exact wording is used for each cue; WhisperX only "
            "supplies the timings. This bypasses punctuation restoration entirely."
        ),
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.70,
        help=(
            "When --script is used, warn if transcript/script token similarity falls "
            "below this ratio (a sign the script and audio do not match)."
        ),
    )
    parser.add_argument(
        "--punctuation-model",
        default=DEFAULT_PUNCTUATION_MODEL,
        help=(
            "Hugging Face model used to restore sentence punctuation before "
            "segmentation. This is what lets subgen recover sentence boundaries "
            "when Whisper drops them, keeping one sentence per subtitle."
        ),
    )
    parser.add_argument(
        "--no-restore-punctuation",
        dest="restore_punctuation",
        action="store_false",
        help="Skip punctuation restoration and segment Whisper's raw output directly.",
    )
    parser.set_defaults(restore_punctuation=True)
    parser.add_argument(
        "--max-sentence-seconds",
        type=float,
        default=12.0,
        help=(
            "A cue longer than this (seconds) is probed for a dropped sentence "
            "boundary. Shorter cues keep Whisper's punctuation untouched."
        ),
    )
    parser.add_argument(
        "--max-sentence-words",
        type=int,
        default=48,
        help="A cue with more words than this is also probed for hidden sentences.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=0.5,
        help=(
            "When a cue begins mid-sentence (a lowercase word Whisper did not "
            "capitalize) and follows the previous cue within this gap, the two "
            "are merged into one sentence. A larger gap is treated as a real "
            "sentence break and the word is simply capitalized."
        ),
    )
    parser.add_argument(
        "--no-repair-oversplits",
        dest="repair_oversplits",
        action="store_false",
        help="Disable merging/recapitalizing of sentences split across cues.",
    )
    parser.set_defaults(repair_oversplits=True)
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Print WhisperX transcription and alignment progress.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print WhisperX segment transcripts while processing.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def is_low_vram(device: str, device_index: int) -> bool:
    if device != "cuda":
        return False
    total = torch.cuda.get_device_properties(device_index).total_memory
    return total / 1024**3 < LOW_VRAM_GIB


def resolve_compute_type(requested: str, device: str, low_vram: bool) -> str:
    if requested != "auto":
        return requested
    if device != "cuda":
        return "int8"
    return "int8_float16" if low_vram else "float16"


def resolve_batch_size(requested: int | None, low_vram: bool) -> int:
    if requested is not None:
        return requested
    return 4 if low_vram else 16


def normalize_text(text: str) -> str:
    text = text.replace("-->", "->")
    return re.sub(r"\s+", " ", text).strip()


def load_sentence_segmenter() -> spacy.Language:
    """Prefer the trained pipeline's parser, which segments sentences using
    statistical cues (casing, syntax) rather than punctuation alone. Fall back
    to a punctuation-only sentencizer if the model is not installed."""
    try:
        nlp = spacy.load(SPACY_MODEL, exclude=["ner", "lemmatizer"])
    except OSError:
        nlp = spacy.blank("en")
    if not nlp.has_pipe("parser") and not nlp.has_pipe("senter"):
        nlp.add_pipe("sentencizer")
    return nlp


def run_whisperx(args: argparse.Namespace, audio_path: Path) -> dict[str, Any]:
    patch_legacy_torchaudio_for_pyannote_import()

    device = resolve_device(args.device)
    low_vram = is_low_vram(device, args.device_index)
    compute_type = resolve_compute_type(args.compute_type, device, low_vram)
    batch_size = resolve_batch_size(args.batch_size, low_vram)
    model_dir = str(args.model_dir) if args.model_dir else None

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    audio = whisperx.load_audio(str(audio_path))
    model = whisperx.load_model(
        args.model,
        device=device,
        device_index=args.device_index,
        compute_type=compute_type,
        language=LANGUAGE,
        vad_method=args.vad_method,
        vad_options={
            "chunk_size": args.chunk_size,
            "vad_onset": args.vad_onset,
            "vad_offset": args.vad_offset,
        },
        download_root=model_dir,
        local_files_only=args.local_files_only,
        threads=args.threads if args.threads > 0 else 4,
    )

    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=args.chunk_size,
        print_progress=args.print_progress,
        verbose=args.verbose,
    )
    del model
    free_torch_memory()

    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=device,
        model_dir=model_dir,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        print_progress=args.print_progress,
    )
    del align_model
    free_torch_memory()

    return aligned


def patch_legacy_torchaudio_for_pyannote_import() -> None:
    # WhisperX imports pyannote unconditionally through whisperx.vads, even when
    # Silero VAD is selected. pyannote.audio 3.4 still references torchaudio APIs
    # removed from torchaudio 2.11 during import, so provide the import-time names.
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = object

    if not hasattr(torchaudio, "list_audio_backends"):

        def list_audio_backends() -> list[str]:
            return ["soundfile"]

        torchaudio.list_audio_backends = list_audio_backends


def free_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_timed_words(aligned: dict[str, Any]) -> tuple[str, list[TimedWord]]:
    transcript_parts: list[str] = []
    timed_words: list[TimedWord] = []
    cursor = 0

    for segment in aligned.get("segments", []):
        fallback_start = coerce_float(segment.get("start"))
        fallback_end = coerce_float(segment.get("end"))
        raw_words = segment.get("words", [])
        if raw_words:
            for raw_word in raw_words:
                cursor = append_timed_word(
                    transcript_parts=transcript_parts,
                    timed_words=timed_words,
                    cursor=cursor,
                    text=str(raw_word.get("word", "")),
                    start=coerce_float(raw_word.get("start")),
                    end=coerce_float(raw_word.get("end")),
                    fallback_start=fallback_start,
                    fallback_end=fallback_end,
                )
        else:
            for word_text in normalize_text(str(segment.get("text", ""))).split():
                cursor = append_timed_word(
                    transcript_parts=transcript_parts,
                    timed_words=timed_words,
                    cursor=cursor,
                    text=word_text,
                    start=None,
                    end=None,
                    fallback_start=fallback_start,
                    fallback_end=fallback_end,
                )

    transcript = "".join(transcript_parts)
    return transcript, timed_words


def append_timed_word(
    transcript_parts: list[str],
    timed_words: list[TimedWord],
    cursor: int,
    text: str,
    start: float | None,
    end: float | None,
    fallback_start: float | None,
    fallback_end: float | None,
) -> int:
    word_text = normalize_text(text)
    if not word_text:
        return cursor

    if transcript_parts:
        transcript_parts.append(" ")
        cursor += 1

    char_start = cursor
    transcript_parts.append(word_text)
    cursor += len(word_text)
    timed_words.append(
        TimedWord(
            text=word_text,
            start=start,
            end=end,
            fallback_start=fallback_start,
            fallback_end=fallback_end,
            char_start=char_start,
            char_end=cursor,
        )
    )
    return cursor


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_punctuation_model(model_name: str) -> Any:
    """Build the Hugging Face token-classification pipeline once and wrap it in a
    PunctuationModel so we can reuse its chunking/word-mapping in predict().

    We construct the pipeline directly because the library's own constructor
    still passes the removed `grouped_entities` kwarg and hardcodes GPU device 0.
    """
    from deepmultilingualpunctuation import PunctuationModel
    from transformers import pipeline

    model = PunctuationModel.__new__(PunctuationModel)
    model.pipe = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="none",
        device=0 if torch.cuda.is_available() else -1,
    )
    return model


def restore_word_punctuation(
    timed_words: list[TimedWord], model: Any
) -> tuple[str, list[TimedWord]]:
    """Re-punctuate and re-case a word stream so sentence boundaries survive even
    when Whisper emitted a long, unpunctuated run of speech.

    Whisper's word timings are kept; only each word's text is rewritten, and the
    character offsets are recomputed (local to this slice) so the segmenter still
    maps spans back to words. Returns the rebuilt transcript and word list.
    """
    cores = [_TRAILING_PUNCT_RE.sub("", word.text).strip() or word.text for word in timed_words]
    predictions = model.predict(cores)

    rebuilt: list[TimedWord] = []
    parts: list[str] = []
    cursor = 0
    start_of_sentence = True
    for word, core, prediction in zip(timed_words, cores, predictions):
        label = prediction[1] if isinstance(prediction[1], str) else ""
        text = core
        if start_of_sentence and text[:1].isalpha() and text[0].islower():
            text = text[0].upper() + text[1:]
        if label in RESTORABLE_PUNCT:
            text += label
        start_of_sentence = label in SENTENCE_END_PUNCT

        if parts:
            parts.append(" ")
            cursor += 1
        char_start = cursor
        parts.append(text)
        cursor += len(text)
        rebuilt.append(replace(word, text=text, char_start=char_start, char_end=cursor))

    return "".join(parts), rebuilt


def segment_sentences(
    nlp: spacy.Language, transcript: str, words: list[TimedWord]
) -> list[tuple[str, list[TimedWord]]]:
    """Split a transcript into sentences, pairing each with its timed words. The
    word char offsets must already be consistent with `transcript`."""
    nlp.max_length = max(nlp.max_length, len(transcript) + 1)
    out: list[tuple[str, list[TimedWord]]] = []
    for sentence in nlp(transcript).sents:
        text = normalize_text(sentence.text)
        if not text:
            continue
        span = words_for_span(words, sentence.start_char, sentence.end_char)
        if span:
            out.append((text, span))
    return out


def cue_duration(words: list[TimedWord]) -> float | None:
    start = first_time(words, "start", "fallback_start")
    end = last_time(words, "end", "fallback_end")
    if start is None or end is None:
        return None
    return end - start


def is_runon_candidate(words: list[TimedWord], args: argparse.Namespace) -> bool:
    """A cue long enough that it may hide more than one sentence behind missing
    Whisper punctuation, and is therefore worth probing with the model."""
    if len(words) > args.max_sentence_words:
        return True
    duration = cue_duration(words)
    return duration is not None and duration > args.max_sentence_seconds


def make_cue(text: str, words: list[TimedWord]) -> SentenceCue | None:
    start = first_time(words, "start", "fallback_start")
    end = last_time(words, "end", "fallback_end")
    if start is None or end is None:
        return None
    if end <= start:
        end = start + 0.001
    return SentenceCue(text=text, start=start, end=end)


def sentence_cues(aligned: dict[str, Any], args: argparse.Namespace) -> list[SentenceCue]:
    transcript, timed_words = build_timed_words(aligned)
    if not transcript or not timed_words:
        return []

    nlp = load_sentence_segmenter()
    base = segment_sentences(nlp, transcript, timed_words)

    # Hybrid restoration: keep Whisper's own punctuation for every cue, and only
    # re-punctuate the long cues that may hide a dropped sentence boundary. A
    # probed cue is replaced only when the model actually finds extra sentences;
    # otherwise its original Whisper text is left untouched.
    model = None
    if args.restore_punctuation and any(is_runon_candidate(words, args) for _, words in base):
        try:
            model = load_punctuation_model(args.punctuation_model)
        except Exception as error:  # noqa: BLE001 - degrade gracefully, never crash here
            print(
                f"Warning: punctuation restoration unavailable ({error}); "
                "keeping Whisper's raw punctuation."
            )

    groups: list[tuple[str, list[TimedWord]]] = []
    for text, words in base:
        if model is not None and is_runon_candidate(words, args):
            split = resplit_runon(nlp, words, model)
            if len(split) > 1:
                groups.extend(split)
                continue
        groups.append((text, words))

    if args.repair_oversplits:
        groups = repair_oversplit_sentences(groups, args)

    return [cue for cue in (make_cue(text, words) for text, words in groups) if cue]


def sentence_cues_from_script(aligned: dict[str, Any], args: argparse.Namespace) -> list[SentenceCue]:
    """Build one cue per script sentence, timed by aligning WhisperX's word stream
    to the original script.

    The script supplies the wording and the (correctly punctuated) sentence
    boundaries; WhisperX supplies only the timings. This sidesteps punctuation
    restoration and the over-split repairs the audio-only path needs.
    """
    script_path: Path = args.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")
    script_text = repair_mojibake(script_path.read_text(encoding="utf-8-sig"))

    _, timed_words = build_timed_words(aligned)
    if not timed_words or not script_text.strip():
        return []

    nlp = load_sentence_segmenter()
    cues, ratio = build_script_cues(nlp, script_text, timed_words)
    if not cues:
        return []

    print(f"Script alignment ratio: {ratio:.3f}")
    if ratio < args.min_ratio:
        print(
            f"Warning: script alignment ratio {ratio:.3f} is below --min-ratio "
            f"{args.min_ratio:.3f}; check that the script matches the audio. "
            "Subtitles were still written, but timings may be unreliable."
        )
    return cues


def build_script_cues(
    nlp: spacy.Language, script_text: str, timed_words: list[TimedWord]
) -> tuple[list[SentenceCue], float]:
    """Align timed transcript words to the script, then time each script sentence."""
    # Flatten timed words into alignment keys. A word may yield zero keys (pure
    # punctuation) or several; each key remembers which timed word it came from so
    # its timing can be recovered after alignment.
    transcript_keys: list[str] = []
    key_to_word: list[int] = []
    for index, word in enumerate(timed_words):
        for token in tokenize(word.text):
            transcript_keys.append(token.key)
            key_to_word.append(index)

    script_tokens = tokenize(script_text)
    script_keys = [token.key for token in script_tokens]
    if not transcript_keys or not script_keys:
        return [], 0.0

    mapped, ratio, _ = map_caption_tokens_to_script(transcript_keys, script_keys)
    sentences = script_sentence_token_ranges(nlp, script_text, script_tokens)

    cues: list[SentenceCue] = []
    last_end = 0.0
    for text, lo, hi in sentences:
        start: float | None = None
        end: float | None = None
        for key_index, position in enumerate(mapped):
            if position is None or not (lo <= position < hi):
                continue
            word = timed_words[key_to_word[key_index]]
            word_start = word.start if word.start is not None else word.fallback_start
            word_end = word.end if word.end is not None else word.fallback_end
            if word_start is not None:
                start = word_start if start is None else min(start, word_start)
            if word_end is not None:
                end = word_end if end is None else max(end, word_end)

        if start is None or end is None:
            # No transcript words aligned to this sentence (e.g. a line the speaker
            # skipped). Drop it rather than emit a cue with no timing.
            continue

        start = max(start, last_end)
        if end <= start:
            end = start + 0.001
        cues.append(SentenceCue(text=normalize_text(text), start=start, end=end))
        last_end = end

    return cues, ratio


def script_sentence_token_ranges(
    nlp: spacy.Language, script_text: str, script_tokens: list[Token]
) -> list[tuple[str, int, int]]:
    """Segment the script into sentences and express each as a half-open span of
    script-token indices, so aligned transcript positions can be tested against it."""
    nlp.max_length = max(nlp.max_length, len(script_text) + 1)
    token_starts = [token.start for token in script_tokens]
    ranges: list[tuple[str, int, int]] = []
    for sentence in nlp(script_text).sents:
        text = normalize_text(sentence.text)
        if not text:
            continue
        lo = bisect.bisect_left(token_starts, sentence.start_char)
        hi = bisect.bisect_left(token_starts, sentence.end_char)
        if hi > lo:
            ranges.append((text, lo, hi))
    return ranges


def normalize_token(token: str) -> str:
    token = token.casefold().replace("‘", "'").replace("’", "'")
    token = unicodedata.normalize("NFKD", token)
    token = "".join(char for char in token if not unicodedata.combining(char))
    return re.sub(r"['`]", "", token)


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(text):
        key = normalize_token(match.group(0))
        if key:
            tokens.append(Token(key=key, start=match.start(), end=match.end()))
    return tokens


def repair_mojibake(text: str) -> str:
    for broken, fixed in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(broken, fixed)
    return text


def map_caption_tokens_to_script(
    caption_keys: list[str], script_keys: list[str]
) -> tuple[list[float | None], float, int]:
    """Map each transcript token to a (fractional) script-token position via a
    difflib alignment. Replacements interpolate across the matched span; deletions
    are pinned just before the next script token. Positions are forced non-decreasing
    so sentence spans can be tested with a simple range check. Same logic as
    edit_subs_from_script.py."""
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


def repair_oversplit_sentences(
    groups: list[tuple[str, list[TimedWord]]], args: argparse.Namespace
) -> list[tuple[str, list[TimedWord]]]:
    """Repair single sentences that Whisper split across two cues.

    A cue that begins with a lowercase word is one Whisper did not treat as a
    sentence start. If it also follows the previous cue within a short gap, it is
    a genuine continuation and the two are merged; otherwise it is a real new
    sentence whose leading word is simply recapitalized. Merges never exceed the
    run-on length limits, so long sentences stay split for readability.
    """
    repaired: list[tuple[str, list[TimedWord]]] = []
    for text, words in groups:
        if repaired and starts_lowercase(text):
            prev_text, prev_words = repaired[-1]
            gap = inter_cue_gap(prev_words, words)
            combined = prev_words + words
            if (
                gap is not None
                and gap <= args.merge_gap_seconds
                and len(combined) <= args.max_sentence_words
                and not is_runon_candidate(combined, args)
            ):
                repaired[-1] = (join_continuation(prev_text, text), combined)
                continue
            text = capitalize_first_letter(text)
        repaired.append((text, words))
    return repaired


def starts_lowercase(text: str) -> bool:
    for char in text:
        if char.isalpha():
            return char.islower()
    return False


def capitalize_first_letter(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def join_continuation(first: str, second: str) -> str:
    first = first.rstrip()
    if first and first[-1] in ".?!…":
        first = first[:-1].rstrip()
        separator = " " if first and first[-1] in ",;:—-" else ", "
    elif first and first[-1] in ",;:—-":
        separator = " "
    else:
        separator = ", "
    return f"{first}{separator}{second}"


def inter_cue_gap(
    prev_words: list[TimedWord], words: list[TimedWord]
) -> float | None:
    prev_end = last_time(prev_words, "end", "fallback_end")
    start = first_time(words, "start", "fallback_start")
    if prev_end is None or start is None:
        return None
    return start - prev_end


def resplit_runon(
    nlp: spacy.Language, words: list[TimedWord], model: Any
) -> list[tuple[str, list[TimedWord]]]:
    """Re-punctuate one long cue's words and re-segment them. Returns the new
    sentence groups, or an empty list if restoration fails."""
    try:
        transcript, rebuilt = restore_word_punctuation(words, model)
        return segment_sentences(nlp, transcript, rebuilt)
    except Exception as error:  # noqa: BLE001 - keep the original cue on any failure
        print(f"Warning: could not re-punctuate a long cue ({error}); keeping it as-is.")
        return []


def words_for_span(
    timed_words: list[TimedWord], start_char: int, end_char: int
) -> list[TimedWord]:
    return [
        word
        for word in timed_words
        if word.char_end > start_char and word.char_start < end_char
    ]


def first_time(
    words: list[TimedWord], primary_attr: str, fallback_attr: str
) -> float | None:
    for word in words:
        value = getattr(word, primary_attr)
        if value is not None:
            return value
    for word in words:
        value = getattr(word, fallback_attr)
        if value is not None:
            return value
    return None


def last_time(
    words: list[TimedWord], primary_attr: str, fallback_attr: str
) -> float | None:
    for word in reversed(words):
        value = getattr(word, primary_attr)
        if value is not None:
            return value
    for word in reversed(words):
        value = getattr(word, fallback_attr)
        if value is not None:
            return value
    return None


def write_srt(cues: list[SentenceCue], output_path: Path) -> None:
    subtitles = [
        srt.Subtitle(
            index=index,
            start=seconds_to_timedelta(cue.start),
            end=seconds_to_timedelta(cue.end),
            content=cue.text,
        )
        for index, cue in enumerate(cues, start=1)
    ]
    output_path.write_text(srt.compose(subtitles), encoding="utf-8")


def seconds_to_timedelta(seconds: float) -> timedelta:
    return timedelta(milliseconds=round(max(seconds, 0.0) * 1000))


def main() -> int:
    args = parse_args()
    audio_path = args.input
    output_path = args.output or Path("./project/autogen-subs.srt")

    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    aligned = run_whisperx(args, audio_path)
    if args.script:
        cues = sentence_cues_from_script(aligned, args)
    else:
        cues = sentence_cues(aligned, args)
    if not cues:
        raise RuntimeError("WhisperX did not produce any sentence-level subtitle cues.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(cues, output_path)
    print(f"Wrote {len(cues)} sentence-level subtitles to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
