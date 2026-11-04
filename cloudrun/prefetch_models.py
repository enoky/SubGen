"""Download the models SubGen uses into HF_HOME / TORCH_HOME at image build time.

Usage: uv run python cloudrun/prefetch_models.py [whisper-model ...]   (default: large-v3)
Runs on CPU, so it works on a build machine without a GPU.
"""

from __future__ import annotations

import sys

import spacy

import subgen  # applies SubGen's import-time CUDA/torch tweaks before WhisperX loads
import whisperx


def main(models: list[str]) -> None:
    subgen.patch_legacy_torchaudio_for_pyannote_import()

    for name in models:
        # Loading the ASR model also caches the Silero VAD model via torch.hub.
        print(f"Fetching Whisper model {name} and Silero VAD", flush=True)
        model = whisperx.load_model(
            name, device="cpu", compute_type="int8", language=subgen.LANGUAGE, vad_method="silero"
        )
        del model

    print("Fetching English alignment model", flush=True)
    align_model, _ = whisperx.load_align_model(language_code=subgen.LANGUAGE, device="cpu")
    del align_model

    print(f"Fetching punctuation model {subgen.DEFAULT_PUNCTUATION_MODEL}", flush=True)
    subgen.load_punctuation_model(subgen.DEFAULT_PUNCTUATION_MODEL)

    # Installed as a wheel by uv sync; load it here so a missing model fails the build
    # instead of SubGen silently falling back to punctuation-only sentence splitting.
    spacy.load(subgen.SPACY_MODEL)
    print("All models cached.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["large-v3"])
