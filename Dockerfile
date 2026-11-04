# SubGen web UI for Google Cloud Run with an NVIDIA GPU.
# Put this file at the root of the SubGen repository.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# ffmpeg loads audio for WhisperX and encodes the optional MP4 export.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

# Dependencies first, so code edits don't reinstall several GB of wheels.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# faster-whisper's ctranslate2 needs cuBLAS 12 even though torch ships CUDA 13.
# The project only installs it on Windows, so add the locked version for Linux here.
RUN uv pip install --python /app/.venv/bin/python --no-deps nvidia-cublas-cu12==12.9.2.10
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib

# Bake the models into the image so instances don't download GBs on every cold start.
# Override with: --build-arg WHISPER_MODELS="large-v3 large-v3-turbo"
ARG WHISPER_MODELS="large-v3"
COPY subgen.py ./
COPY cloudrun/prefetch_models.py ./cloudrun/
RUN uv run --no-sync python cloudrun/prefetch_models.py $WHISPER_MODELS

COPY . .

# Cloud Run sends traffic to $PORT (8080 by default).
EXPOSE 8080
CMD ["sh", "-c", "exec uv run --no-sync python subgen_gradio.py --port ${PORT:-8080}"]
