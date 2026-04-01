# ── Seed-VC RunPod Serverless Worker ──────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Clone Seed-VC ────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc

# ── Install Python dependencies (torch already in base image) ────
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    requests \
    scipy==1.13.1 \
    librosa==0.10.2 \
    "huggingface-hub>=0.28.1" \
    munch==4.0.0 \
    einops==0.8.0 \
    descript-audio-codec==1.0.0 \
    pydub==0.25.1 \
    resemblyzer \
    jiwer==3.0.3 \
    transformers==4.46.3 \
    soundfile==0.12.1 \
    numpy==1.26.4 \
    hydra-core==1.3.2 \
    pyyaml \
    accelerate

# ── Pre-download model weights (free during build, not at runtime) ──
# NOTE: HuggingFace username is "Plachta" (single a), not "Plachtaa"

# Seed-VC checkpoints (~2GB)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Plachta/Seed-VC', local_dir='/app/seed-vc/checkpoints/Seed-VC'); \
print('Seed-VC weights downloaded')"

# Whisper-small content encoder (~500MB)
RUN python -c "\
from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-small'); \
WhisperFeatureExtractor.from_pretrained('openai/whisper-small'); \
print('Whisper cached')"

# ── Copy handler ─────────────────────────────────────────────────
COPY handler.py /app/handler.py

# ── Entry point ──────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
