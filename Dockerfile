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
    oss2 \
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

# ── 关键：把 HF cache 指向 Seed-VC inference.py 期望的位置 ──
# inference.py 顶部硬编码了 os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
# cwd = /app/seed-vc → 实际路径 /app/seed-vc/checkpoints/hf_cache
# 所有 HF 预下载必须落到这里，否则 runtime 找不到要联网重下
ENV HF_HUB_CACHE=/app/seed-vc/checkpoints/hf_cache

# ── Pre-download Seed-VC DiT 模型 + 配置（用绝对路径，handler 用 --checkpoint 传） ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Plachta/Seed-VC', local_dir='/app/seed-vc/checkpoints/Seed-VC'); \
print('Seed-VC weights downloaded')"

# ── Pre-download Whisper-small（content encoder） ──
RUN python -c "\
from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-small'); \
WhisperFeatureExtractor.from_pretrained('openai/whisper-small'); \
print('Whisper cached')"

# ── Pre-download Demucs htdemucs（用 torch hub cache，不走 HF） ──
RUN python -c "\
import torch; \
from demucs.pretrained import get_model; \
get_model('htdemucs'); \
print('htdemucs model downloaded')"

# ── Pre-download BigVGAN vocoder（~120MB） ──
RUN python -c "\
import sys; sys.path.insert(0, '/app/seed-vc'); \
from modules.bigvgan import bigvgan; \
bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False); \
print('BigVGAN cached')"

# ── Pre-download RMVPE + CAMPPlus（hf_utils.py 硬编码 cache_dir="./checkpoints"，
# 绕过 HF_HUB_CACHE，必须显式下到 /app/seed-vc/checkpoints/）──
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='lj1995/VoiceConversionWebUI', filename='rmvpe.pt', cache_dir='/app/seed-vc/checkpoints'); \
hf_hub_download(repo_id='funasr/campplus', filename='campplus_cn_common.bin', cache_dir='/app/seed-vc/checkpoints'); \
print('RMVPE + CAMPPlus cached at /app/seed-vc/checkpoints')"

# ── Pre-download 声学性别分类模型（ECAPA-TDNN 59MB，pitch_shift 决策用）──
# handler.classify_gender 从 /app/gender-classifier 本地加载，运行时不联网
RUN pip install --no-cache-dir safetensors && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('JaesungHuh/voice-gender-classifier', local_dir='/app/gender-classifier'); \
print('Gender classifier cached at /app/gender-classifier')"

# ── Clone Music-Source-Separation-Training (for BS Roformer inference) ──
RUN git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git /app/msst
# Install MSST inference dependencies (skip GUI/training-only packages)
RUN pip install --no-cache-dir \
    ml_collections \
    beartype==0.14.1 \
    rotary-embedding-torch==0.3.5 \
    einops==0.8.1 \
    segmentation_models_pytorch==0.3.3 \
    timm==0.9.2 \
    omegaconf \
    wandb \
    loralib \
    spafe==0.3.2 \
    auraloss \
    torchseg \
    prodigyopt \
    hyper_connections==0.1.11 \
    torch_log_wmse \
    torch_l1_snr

# ── Download BS Roformer Karaoke model (~204MB, lead/backing separation) ──
RUN wget -q -O /app/msst/bs_roformer_karaoke_frazer_becruily.ckpt \
    "https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt" \
    && wget -q -O /app/msst/config_karaoke_frazer_becruily.yaml \
    "https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/config_karaoke_frazer_becruily.yaml" \
    && echo "Karaoke model downloaded" \
    && ls -lh /app/msst/bs_roformer_karaoke_frazer_becruily.*

# ── 强制 HF/transformers 离线（所有模型已预下，禁止 runtime 联网检查/下载）──
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ── Copy handler ─────────────────────────────────────────────────
COPY handler.py /app/handler.py
COPY gender_model.py /app/gender_model.py

# ── Entry point ──────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
