# ── Seed-VC RunPod Serverless Worker ──────────────────────────────
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Clone Seed-VC ────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc

# ── Install Python dependencies (torch already in base image) ────
# Removed: boto3, resemblyzer, jiwer, descript-audio-codec, hydra-core, pydub
RUN pip install --no-cache-dir \
    runpod \
    requests \
    scipy \
    librosa \
    "huggingface-hub>=0.28.1" \
    munch \
    einops \
    transformers \
    soundfile \
    pyyaml \
    accelerate \
    demucs \
    pedalboard

# ── Pre-generate matplotlib font cache (saves 23s at runtime) ────
RUN python -c "import matplotlib; print('Font cache generated')"

# ── Pre-download model weights (free during build, not at runtime) ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Plachta/Seed-VC', local_dir='/app/seed-vc/checkpoints/Seed-VC'); \
print('Seed-VC weights downloaded')"

RUN python -c "\
from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-small'); \
WhisperFeatureExtractor.from_pretrained('openai/whisper-small'); \
print('Whisper cached')"

# ── Copy handler ─────────────────────────────────────────────────
COPY handler.py /app/handler.py

# ── Entry point ──────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
