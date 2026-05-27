import re
import torch
import spacy
import whisper
import whisper_timestamped as whisper_ts
import srt
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")
PAUSE_SPLIT_SECONDS = 0.9


def load_spacy_model():
    """
    Loads the spaCy English model for sentence segmentation.
    If the model isn't found, it attempts to download it.
    """
    try:
        nlp = spacy.load("en_core_web_lg")
        logging.info("spaCy model 'en_core_web_lg' loaded successfully.")
        return nlp
    except OSError:
        logging.warning("spaCy model 'en_core_web_lg' not found.")
        logging.info("Attempting to download 'en_core_web_lg'...")
        try:
            from spacy.cli import download
            download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
            logging.info("spaCy model 'en_core_web_lg' downloaded and loaded successfully.")
            return nlp
        except Exception as e:
            logging.error(f"Failed to download or load spaCy model: {e}")
            raise


def transcribe_audio(audio_path, model_id="large-v3-turbo"):
    """
    Transcribes an audio file using whisper-timestamped with word-level timestamps and VAD.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        model = whisper.load_model(model_id)
        transcription = whisper_ts.transcribe(model, audio_path, vad=True)
        logging.info("Audio transcription completed.")
        return transcription
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise


def _flatten_words(transcription):
    words = []
    for segment in transcription.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            words.append({"text": text, "start": float(start), "end": float(end)})
    return words


def _is_sentence_end(word_text):
    return bool(SENTENCE_END_RE.search(word_text))


def _finalize_sentence(buffer_words):
    sentence_text = " ".join(w["text"] for w in buffer_words).strip()
    sentence_text = re.sub(r"\s+", " ", sentence_text)
    if sentence_text and not _is_sentence_end(sentence_text):
        sentence_text += "."
    return sentence_text


def generate_srt_subtitles(transcription, nlp):
    """
    Generates SRT subtitles with reliable timestamps by forcing one complete sentence per subtitle.

    Sentence boundaries are taken from punctuation first and from long pauses as a fallback,
    which keeps each subtitle aligned to spoken timing.
    """
    words = _flatten_words(transcription)
    if not words:
        logging.warning("No word-level transcription to process.")
        return []

    subtitles = []
    sentence_buffer = []

    for idx, word in enumerate(words):
        sentence_buffer.append(word)

        next_word = words[idx + 1] if idx + 1 < len(words) else None
        pause_to_next = (next_word["start"] - word["end"]) if next_word else 0.0

        ends_with_punctuation = _is_sentence_end(word["text"])
        long_pause_boundary = next_word is not None and pause_to_next >= PAUSE_SPLIT_SECONDS
        last_word = next_word is None

        should_close_sentence = ends_with_punctuation or long_pause_boundary or last_word
        if not should_close_sentence:
            continue

        sentence_text = _finalize_sentence(sentence_buffer)
        if sentence_text:
            subtitles.append(
                srt.Subtitle(
                    index=len(subtitles) + 1,
                    start=timedelta(seconds=sentence_buffer[0]["start"]),
                    end=timedelta(seconds=sentence_buffer[-1]["end"]),
                    content=sentence_text,
                )
            )

        sentence_buffer = []

    return subtitles


def save_srt_file(subtitles, output_path="output.srt"):
    """
    Saves the generated subtitles to an SRT file.
    """
    try:
        composed_srt = srt.compose(subtitles)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(composed_srt)
        logging.info(f"SRT file saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save SRT file: {e}")
        raise


def main(audio_path, output_srt_path="autogen-subs.srt"):
    """
    Main function to transcribe audio and generate a precise SRT file.
    """
    try:
        nlp = load_spacy_model()
        transcription = transcribe_audio(audio_path)
        subtitles = generate_srt_subtitles(transcription, nlp)
        save_srt_file(subtitles, output_srt_path)
        logging.info("Script finished successfully.")
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise


if __name__ == "__main__":
    # Replace "audio.wav" with your audio file path
    audio_file = "audio.wav"
    main(audio_file)
