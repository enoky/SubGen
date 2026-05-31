from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import spacy
import srt
import torch
import whisperx


DEFAULT_INPUT = Path("audio.wav")
LANGUAGE = "en"


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
        help="SRT output path. Defaults to <input stem>.srt.",
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
        help="Whisper compute type. Defaults to float16 on CUDA and int8 on CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX transcription batch size.",
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


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def normalize_text(text: str) -> str:
    text = text.replace("-->", "->")
    return re.sub(r"\s+", " ", text).strip()


def load_sentence_segmenter() -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def run_whisperx(args: argparse.Namespace, audio_path: Path) -> dict[str, Any]:
    patch_legacy_torchaudio_for_pyannote_import()

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
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
        batch_size=args.batch_size,
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


def sentence_cues(aligned: dict[str, Any]) -> list[SentenceCue]:
    transcript, timed_words = build_timed_words(aligned)
    if not transcript or not timed_words:
        return []

    nlp = load_sentence_segmenter()
    nlp.max_length = max(nlp.max_length, len(transcript) + 1)
    doc = nlp(transcript)

    cues: list[SentenceCue] = []
    for sentence in doc.sents:
        text = normalize_text(sentence.text)
        if not text:
            continue

        words = words_for_span(timed_words, sentence.start_char, sentence.end_char)
        if not words:
            continue

        start = first_time(words, "start", "fallback_start")
        end = last_time(words, "end", "fallback_end")
        if start is None or end is None:
            continue
        if end <= start:
            end = start + 0.001

        cues.append(SentenceCue(text=text, start=start, end=end))

    return cues


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
    cues = sentence_cues(aligned)
    if not cues:
        raise RuntimeError("WhisperX did not produce any sentence-level subtitle cues.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(cues, output_path)
    print(f"Wrote {len(cues)} sentence-level subtitles to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
