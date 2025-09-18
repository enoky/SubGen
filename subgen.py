import torch
import spacy
import whisper
import whisper_timestamped as whisper_ts
import srt
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def generate_srt_subtitles(transcription, nlp):
    """
    Generates SRT subtitles with precise timestamps using word-level data from whisper-timestamped.
    """
    if not transcription.get("segments"):
        logging.warning("No transcription segments to process.")
        return []

    # Build full text and track word positions
    full_text = ""
    word_list = []
    for segment in transcription["segments"]:
        for word in segment["words"]:
            word_text = word["text"]
            start_char = len(full_text)
            full_text += word_text + " "
            end_char = len(full_text) - 1
            word_list.append({
                "text": word_text,
                "start_time": word["start"],
                "end_time": word["end"],
                "start_char": start_char,
                "end_char": end_char
            })
    full_text = full_text.strip()

    # Segment text into sentences using spaCy
    doc = nlp(full_text)
    subtitles = []
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        start_char = sent.start_char
        end_char = sent.end_char
        # Find words corresponding to the sentence
        sentence_words = [w for w in word_list if w["start_char"] >= start_char and w["end_char"] <= end_char]
        if sentence_words:
            start_time = sentence_words[0]["start_time"]
            end_time = sentence_words[-1]["end_time"]
            subtitle = srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=sentence_text
            )
            subtitles.append(subtitle)
        else:
            logging.warning(f"No words found for sentence: {sentence_text}")
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