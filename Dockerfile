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
# Removed: boto3, resemblyzer, jiwer, pydub
RUN pip install --no-cache-dir \
    runpod \
    requests \
    scipy \
    librosa \
    "huggingface-hub==0.28.1" \
    munch \
    einops \
    descript-audio-codec \
    hydra-core \
    transformers \
    soundfile \
    pyyaml \
    accelerate \
    demucs \
    pedalboard

# ── Pre-generate matplotlib font cache (saves 23s at runtime) ────
RUN pip install --no-cache-dir matplotlib && python -c "import matplotlib; print('Font cache generated')"

# ── Pre-download model weights (free during build, not at runtime) ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Plachta/Seed-VC', local_dir='/app/seed-vc/checkpoints/Seed-VC'); \
print('Seed-VC weights downloaded')"

RUN python -c "\
from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-small'); \
WhisperFeatureExtractor.from_pretrained('openai/whisper-small'); \
print('Whisper cached')" || echo "Whisper download skipped, will download at runtime"

# ── Pre-download Demucs htdemucs model (~80MB) ──
RUN python -c "\
import torch; \
from demucs.pretrained import get_model; \
get_model('htdemucs'); \
print('htdemucs model downloaded')"

# ── Clone Music-Source-Separation-Training (for BS Roformer inference) ──
RUN git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git /app/msst
# Install ALL MSST dependencies (skip torch/torchaudio to keep base image versions)
RUN grep -v -E "^torch==|^torchaudio==|^torchvision==" /app/msst/requirements.txt > /tmp/msst_reqs.txt \
    && pip install --no-cache-dir -r /tmp/msst_reqs.txt

# ── Download BS Roformer vocal model (viperx edition, SDR 10.87, ~400MB) ──
RUN wget -q -O /app/msst/bs_roformer_vocals.ckpt \
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
    && wget -q -O /app/msst/bs_roformer_vocals.yaml \
    "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml" \
    && echo "BS Roformer vocals downloaded (SDR 10.87)" \
    && ls -lh /app/msst/bs_roformer_vocals.*

# ── Download BS Roformer Karaoke model (~204MB, lead/backing separation) ──
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('becruily/bs-roformer-karaoke', 'bs_roformer_karaoke_frazer_becruily.ckpt', local_dir='/app/msst'); \
hf_hub_download('becruily/bs-roformer-karaoke', 'config_karaoke_frazer_becruily.yaml', local_dir='/app/msst'); \
print('Karaoke model downloaded')"

# ── Copy handler ─────────────────────────────────────────────────
COPY handler.py /app/handler.py

# ── Entry point ──────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
