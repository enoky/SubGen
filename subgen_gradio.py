"""Gradio web interface for subgen, usable locally or on Google Colab."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

import subgen

WHISPER_MODELS = ["large-v3", "large-v3-turbo", "large-v2", "medium.en", "small.en", "base.en", "tiny.en"]


def generate(
    audio_file: str | None,
    script_file: str | None,
    script_text: str,
    model: str,
    device: str,
    compute_type: str,
    batch_size: float,
    vad_method: str,
    restore_punctuation: bool,
    repair_oversplits: bool,
    model_dir: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str | None, str, str]:
    if not audio_file:
        raise gr.Error("Upload an audio or video file first.")

    audio_path = Path(audio_file)
    out_dir = Path(tempfile.mkdtemp(prefix="subgen-"))
    output_path = out_dir / f"{audio_path.stem}.srt"

    argv = [str(audio_path), "-o", str(output_path), "--model", model, "--device", device]
    argv += ["--compute-type", compute_type, "--vad-method", vad_method]
    if batch_size:
        argv += ["--batch-size", str(int(batch_size))]
    if model_dir.strip():
        argv += ["--model-dir", model_dir.strip()]
    if not restore_punctuation:
        argv.append("--no-restore-punctuation")
    if not repair_oversplits:
        argv.append("--no-repair-oversplits")

    # Pasted script text wins over an uploaded file; both feed --script.
    if script_text.strip():
        script_path = out_dir / "script.txt"
        script_path.write_text(script_text, encoding="utf-8")
        argv += ["--script", str(script_path)]
    elif script_file:
        argv += ["--script", script_file]

    args = subgen.parse_args(argv)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            aligned = subgen.run_whisperx(args, audio_path)
            if args.script:
                cues = subgen.sentence_cues_from_script(aligned, args)
            else:
                cues = subgen.sentence_cues(aligned, args)
            if not cues:
                raise RuntimeError("WhisperX did not produce any sentence-level subtitle cues.")
            subgen.write_srt(cues, output_path)
            print(f"Wrote {len(cues)} sentence-level subtitles to {output_path.name}")
    except Exception as error:
        traceback.print_exc(file=log)
        print(log.getvalue(), file=sys.stderr)
        raise gr.Error(f"{type(error).__name__}: {error}") from error
    finally:
        subgen.free_torch_memory()

    return str(output_path), output_path.read_text(encoding="utf-8"), log.getvalue()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SubGen") as app:
        gr.Markdown(
            "# SubGen\nSentence-level `.srt` subtitles from English speech, using WhisperX "
            "for transcription and word timing."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.File(label="Audio or video", file_types=["audio", "video"], type="filepath")
                with gr.Accordion("Original script (optional)", open=False):
                    gr.Markdown(
                        "If you have the exact script, subtitles use its wording and punctuation; "
                        "WhisperX only supplies the timings."
                    )
                    script_file = gr.File(label="Script .txt", file_types=[".txt"], type="filepath")
                    script_text = gr.Textbox(label="…or paste the script", lines=6)
                with gr.Accordion("Settings", open=False):
                    model = gr.Dropdown(WHISPER_MODELS, value="large-v3", label="Whisper model", allow_custom_value=True)
                    device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                    compute_type = gr.Dropdown(
                        ["auto", "float16", "int8_float16", "int8", "float32"],
                        value="auto",
                        label="Compute type",
                    )
                    batch_size = gr.Number(value=0, precision=0, label="Batch size (0 = auto)")
                    vad_method = gr.Radio(["silero", "pyannote"], value="silero", label="VAD method")
                    restore_punctuation = gr.Checkbox(value=True, label="Restore dropped sentence punctuation")
                    repair_oversplits = gr.Checkbox(value=True, label="Repair sentences split across cues")
                    model_dir = gr.Textbox(
                        label="Model cache directory (optional)",
                        placeholder="e.g. /content/drive/MyDrive/subgen-models",
                    )
                run = gr.Button("Generate subtitles", variant="primary")
            with gr.Column():
                srt_file = gr.File(label="Subtitles (.srt)")
                preview = gr.Textbox(label="Preview", lines=16)
                log = gr.Textbox(label="Log", lines=6)

        run.click(
            generate,
            inputs=[
                audio,
                script_file,
                script_text,
                model,
                device,
                compute_type,
                batch_size,
                vad_method,
                restore_punctuation,
                repair_oversplits,
                model_dir,
            ],
            outputs=[srt_file, preview, log],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the SubGen web interface.")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    parser.add_argument("--port", type=int, default=None, help="Port to serve on.")
    cli = parser.parse_args()
    # A public link is the only way to reach the app from a Colab notebook.
    share = cli.share or "google.colab" in sys.modules
    build_app().queue().launch(share=share, server_port=cli.port, debug="google.colab" in sys.modules)


if __name__ == "__main__":
    main()
