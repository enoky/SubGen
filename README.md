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

### Video clips as the background

Pass a folder of `.mp4` clips to use them instead of the blue background:

```powershell
uv run python -m subgen path\to\audio.wav --background-clips path\to\clips
uv run python -m srt2mp4 subtitles.srt narration.wav out.mp4 --clips path\to\clips
```

- The clips are joined in natural filename order (`clip2` before `clip10`), then
  center-cropped to 16:9 and scaled to 1920x1080 with Lanczos. For example,
  1280x704 is cropped by 14 px on each side and 960x512 by about 25 px (offsets are
  kept even for 4:2:0 video and the GPU decoder), then scaled. Their audio is dropped.
- The output uses the clips' frame rate. If the clips are shorter than the
  narration, the last frame is held; if longer, they are cut to the narration's length.
- Thousands of clips are fine: they are checked in parallel and read through a single
  ffmpeg input. Any clip whose format differs from the rest (size, frame rate, codec)
  is re-encoded to match first, so the join stays seamless and on the GPU.
- The text gets a black outline as well as the shadow, so it stays readable on footage.
- The encoding is tuned for YouTube 1080p uploads: HEVC (NVENC) quality-targeted VBR
  averaging around 10 Mbps, a keyframe every 2 seconds, BT.709 tags, AAC-LC 48 kHz
  stereo at 384 kbps, and `+faststart`.
- On an NVIDIA GPU, decoding (NVDEC), scaling and the subtitle overlay (CUDA) and
  encoding (NVENC) all run on the GPU, and frames never leave it. The GPU scales 1.5%
  larger than 1080p and the extra 8 px border is cut off on the GPU, so anything the
  GPU decoder or scaler leaves along the frame edges (some setups produce a thin green
  line at the bottom) never reaches the video. As insurance, each GPU route first renders one second of the first clip
  and its edges are compared with a CPU render; a route that leaves artifacts is
  skipped. The fallback keeps GPU decode and encode with crop/scale/overlay on all CPU
  cores (about 3x slower), then libx264 on the CPU. `--cpu-filters` goes straight to
  that fallback. Run `uv run python -m srt2mp4 --check` to see what your ffmpeg
  supports.

## Web Interface

Launch a Gradio web UI to upload audio (or video), optionally add the original
script, and download the generated `.srt` (tick **Also export MP4** for the
subtitle video too):

```powershell
uv run python subgen_gradio.py
```

Add `--share` for a public `gradio.live` link.

### Google Colab

Open `SubGen_Colab.ipynb` in Colab, switch the runtime to an L4 (or any) GPU, and
run the setup cells. Then either:

- **Drive pipeline:** put `narration.wav`, `clips/*.mp4` and (optionally)
  `script.txt` in `MyDrive/subgen/`, then run the Drive cells. `narration.srt` and
  `narration.mp4` are saved to `MyDrive/subgen/output/`. Add `--reuse-srt` to
  re-render the video from an SRT you edited on Drive, or `--srt-only` to skip the
  video. Any other options are passed to subgen (e.g. `--model large-v3-turbo`).
- **Web UI:** the last cell prints a public link to the Gradio app.

Clips are read straight from Drive while the video encodes, rather than copied to
the Colab machine first, so reading overlaps with GPU work. Clip details are cached in
`output/.clip_info_cache.json`, so reruns over the same clips start right away. If
streaming from Drive turns out slow, `--copy-clips` copies them to local disk first
(in parallel; clips already copied earlier in the session are skipped).

Before transcription, the narration is loudness-normalized for YouTube: two-pass
EBU R128 to -14 LUFS integrated and -1.5 dBTP true peak, saved as 48 kHz 24-bit WAV.
The normalized audio is what gets transcribed and put in the video, and it is also
saved to `output/narration_norm.wav`. Change the target with `--lufs` / `--true-peak`,
or skip this step with `--no-normalize`.

The same pipeline runs anywhere with `uv run python -m drive_pipeline --root DIR`.

### Loudness normalization on its own

```powershell
uv run python -m normalize_audio narration.wav            # writes narration_norm.wav
uv run python -m normalize_audio *.wav --lufs -16
```

### Google Cloud Run

The `Dockerfile` packages the web UI for Cloud Run on an NVIDIA L4 GPU, with
the Whisper, alignment, VAD, punctuation and spaCy models baked into the image
so cold starts don't download them. With `gcloud` signed in to your project,
run from the repo root:

```bash
bash cloudrun/deploy.sh
```

It builds the image with Cloud Build, then deploys a private service in
`us-east4` (8 CPU, 32 GiB, 1 L4, scale to zero, max 1 instance, 60-minute
request timeout). Override `REGION`, `SERVICE` or `WHISPER_MODELS` as
environment variables, e.g. `WHISPER_MODELS="large-v3 large-v3-turbo"`. Open
the private service through an authenticated tunnel:

```bash
gcloud run services proxy subgen --region us-east4 --port 8080
```

GPU instances are billed for as long as they run, so keep `--min-instances 0`
unless you need the app always warm.

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
--background-clips        Folder of .mp4 clips to use as the video background (implies --mp4).
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
