# Subgen: Automatic Subtitle Generator

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Subgen is a Python script that automatically generates accurate, sentence-level `.srt` subtitle files from any audio or video file. It leverages the power of OpenAI's Whisper for transcription, `whisper-timestamped` for word-level accuracy, and spaCy for intelligent sentence segmentation.

The result is perfectly timed subtitles that are grouped into natural, readable sentences, saving you hours of manual transcription and timing work.



## Key Features

-   **High-Accuracy Transcription**: Utilizes the large-v3 Whisper model for state-of-the-art speech-to-text conversion.
-   **Precise Timestamps**: Achieves word-level timestamp accuracy, ensuring subtitles appear exactly when the words are spoken.
-   **Intelligent Segmentation**: Uses NLP (Natural Language Processing) via spaCy to break the transcription into grammatically correct sentences, rather than awkward, fixed-length chunks.
-   **Voice Activity Detection (VAD)**: Automatically detects and removes silent segments from the audio to improve transcription efficiency and accuracy.
-   **Simple & Automated**: A single command is all it takes to go from an audio file to a polished `.srt` subtitle file.

## How It Works

The script follows a sophisticated pipeline to ensure high-quality output:

1.  **Transcription with Word Timestamps**: The audio file is first processed by `whisper-timestamped`. This provides not only the full transcription but also the precise start and end times for every single word.
2.  **Full-Text Reconstruction**: The script reconstructs the full, continuous text from the individual transcribed words.
3.  **NLP Sentence Segmentation**: The full text is passed to a spaCy language model (`en_core_web_lg`), which analyzes the text and accurately identifies sentence boundaries.
4.  **Timestamp Mapping**: The script then maps the start time of the first word and the end time of the last word of each identified sentence.
5.  **SRT File Generation**: Finally, it generates an `.srt` file where each entry corresponds to a complete sentence with its calculated start and end times.

## Requirements

-   Python 3.11+
-   PyTorch
-   [FFmpeg](https://ffmpeg.org/download.html) (must be installed and available in your system's PATH)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/enoky/subgen.git
    cd subgen
    ```

2.  **Install Python dependencies:**
    Create a `requirements.txt` file with the necessary packages and install them.
    ```bash
    pip install -r requirements.txt
    ```
    *(See the provided `requirements.txt` file in this repository for the correct package list.)*

3.  **Download the spaCy NLP model:**
    This is a one-time download for the English language model used for sentence detection.
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Usage

1.  Place your audio file (e.g., `my_podcast.mp3`, `lecture.wav`, `video.mp4`) in the project directory.

2.  Open the `subgen.py` script and modify the main execution block at the bottom of the file to point to your audio file and specify the desired output path.

    ```python
    if __name__ == "__main__":
        # --- CONFIGURATION ---
        audio_file = "my_podcast.mp3"  # <-- Your audio/video file here
        output_file = "my_podcast.srt" # <-- Desired output SRT file name
        # -------------------

        main(audio_file, output_srt_path=output_file)
    ```

3.  Run the script from your terminal:
    ```bash
    python subgen.py
    ```

The script will process the audio, log its progress in the console, and generate the `.srt` file in the location you specified.

### Example Output (`my_podcast.srt`)

```srt
1
00:00:01,234 --> 00:00:04,567
Hello everyone, and welcome back to the show.

2
00:00:04,890 --> 00:00:07,123
Today, we have a very special guest with us.

3
00:00:07,456 --> 00:00:10,789
We're going to be talking about the future of artificial intelligence.
