# SubGen

SubGen creates sentence-level `.srt` subtitle files from English speech audio using
WhisperX transcription and word alignment. It then groups the aligned words into
sentences with spaCy so subtitle cues are easier to read than raw chunk-level
transcripts.

## Features

- Generates standard SRT subtitle files.
- Uses WhisperX for transcription and word-level timing alignment.
- Produces sentence-level subtitle cues.
- Defaults to CUDA when available, with CPU fallback.
- Supports custom Whisper models, cache directories, device selection, VAD
  settings, batch size, and local-only model loading.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- `ffmpeg` available on your `PATH`.
- A CUDA-capable GPU is recommended for large models. CPU mode works, but can be
  slow.

The project pins PyTorch packages from the CUDA 12.8 wheel index in
`pyproject.toml`.

## Installation

Clone the repository, then install dependencies with uv:

```powershell
uv sync
```

The first transcription run may download WhisperX, alignment, VAD, and spaCy
model assets unless they are already cached.

## Usage

Generate subtitles for the default `audio.wav` file:

```powershell
uv run python -m subgen
```

Generate subtitles for a specific audio file:

```powershell
uv run python -m subgen path\to\audio.wav
```

Write the SRT file to a specific path:

```powershell
uv run python -m subgen path\to\audio.wav -o path\to\subtitles.srt
```

Use CUDA explicitly:

```powershell
uv run python -m subgen path\to\audio.wav --device cuda
```

Use locally cached models only:

```powershell
uv run python -m subgen path\to\audio.wav --model-dir .\models --local-files-only
```

By default, the output file is written next to the input file using the same stem
and an `.srt` extension.

### MP4 export

Add `--mp4` to also render a 1920x1080, 24 fps bluescreen video of the subtitles
(white Yatra One text with a drop shadow on chroma blue `#0047BB`) muxed with the
input's audio. It is written next to the `.srt` unless you pass `--mp4-output`:

```powershell
uv run python -m subgen path	oudio.wav --mp4
```

Encoding uses NVENC when available and falls back to `libx264`; force one with
`--mp4-encoder`. To re-render a video from an existing (e.g. hand-edited) `.srt`
without transcribing again:

```powershell
uv run python -m srt2mp4 subtitles.srt narration.wav out.mp4
```

The font is bundled in `fonts/`, so the output looks the same locally and on Colab.

## Web Interface

Launch a Gradio web UI to upload audio (or video), optionally add the original
script, and download the generated `.srt` (tick **Also export MP4** for the
subtitle video too):

```powershell
uv run python subgen_gradio.py
```

Add `--share` for a public `gradio.live` link.

### Google Colab

Open `SubGen_Colab.ipynb` in Colab, switch the runtime to a GPU, and run the
cells. The last cell prints a public link to the web UI.

## CLI Options

```text
input                     Audio file to transcribe. Defaults to audio.wav.
-o, --output              SRT output path. Defaults to <input stem>.srt.
--model                   Whisper model name. Defaults to large-v3.
--model-dir               Directory for downloaded or cached WhisperX models.
--local-files-only        Only use locally cached models.
--device                  Inference device: auto, cpu, or cuda. Defaults to auto.
--device-index            CUDA device index used with --device cuda. Defaults to 0.
--compute-type            Whisper compute type. Defaults to float16 on CUDA and int8 on CPU.
--batch-size              WhisperX transcription batch size. Defaults to 16.
--threads                 CPU threads for torch and faster-whisper. Defaults to 4.
--chunk-size              Maximum VAD speech chunk size in seconds. Defaults to 30.
--vad-method              Voice activity detector: silero or pyannote. Defaults to silero.
--vad-onset               VAD onset threshold. Defaults to 0.5.
--vad-offset              VAD offset threshold. Defaults to 0.363.
--print-progress          Print WhisperX progress.
--mp4                     Also render a bluescreen subtitle video with the input's audio.
--mp4-output              MP4 output path (implies --mp4). Defaults to <srt stem>.mp4.
--mp4-encoder             auto, h264_nvenc, hevc_nvenc, or libx264. Defaults to auto.
--verbose                 Print WhisperX segment transcripts while processing.
```

## Development

Run tests with:

```powershell
uv run pytest
```

Run the module directly during development:

```powershell
uv run python -m subgen --help
```

## Notes

- The current pipeline is configured for English speech.
- Audio files and generated subtitle files are ignored by Git by default.
- If you select `--vad-method pyannote`, you may need Hugging Face access for
  pyannote models, depending on your local cache and model choices.
