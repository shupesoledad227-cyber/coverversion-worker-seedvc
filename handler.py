"""
Seed-VC Cover Song Worker for RunPod Serverless.

Full pipeline:
  1. Separate vocals from instrumental (demucs)
  2. Convert vocals with Seed-VC (zero-shot, model preloaded in GPU)
  3. Mix converted vocals + original instrumental
  4. Upload and return result URL

Optimization: Seed-VC model loaded ONCE at startup, stays in GPU memory.
"""

import os
import sys
import tempfile
import time
import subprocess
import traceback
import shutil

import requests
import runpod
import torch
import torchaudio
import numpy as np
import yaml
import librosa
import soundfile as sf

# ── Add Seed-VC to path ──────────────────────────────────────────
SEED_VC_DIR = "/app/seed-vc"
sys.path.insert(0, SEED_VC_DIR)

# ── Seed-VC imports ──────────────────────────────────────────────
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.rmvpe import RMVPE
from transformers import WhisperModel, WhisperFeatureExtractor

# ── Global model state (loaded once, reused across requests) ─────
DEVICE = None
DTYPE = None
SR = 44100  # Singing model sample rate

# Model components
dit_model = None
dit_config = None
campplus_model = None
vocoder = None
rmvpe_model = None
whisper_model = None
whisper_feature_extractor = None

# Config values
dit_model_config = None
mel_fn_args = None
to_mel = None
overlap_wave_len = None
max_context_window = None
overlap_frame_len = None
bitrate = None


def load_all_models():
    """Load all models into GPU memory once at startup."""
    global DEVICE, DTYPE
    global dit_model, dit_config, dit_model_config
    global campplus_model, vocoder, rmvpe_model
    global whisper_model, whisper_feature_extractor
    global mel_fn_args, to_mel, overlap_wave_len, max_context_window, overlap_frame_len, bitrate

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

    print(f"[Init] Device: {DEVICE}, Dtype: {DTYPE}")

    # ── Load Seed-VC config ──────────────────────────────────────
    config_path = os.path.join(SEED_VC_DIR, "configs", "presets", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    with open(config_path, "r") as f:
        dit_config = yaml.safe_load(f)

    dit_model_config = recursive_munch(dit_config["model_params"])
    mel_fn_args = dit_config["preprocess_params"]["spect_params"]
    overlap_wave_len = dit_config["preprocess_params"].get("overlap_wave_len", 16 * SR)
    max_context_window = dit_config["preprocess_params"].get("max_context_window", 30 * SR)
    overlap_frame_len = 16
    bitrate = dit_config.get("vocoder_params", {}).get("bitrate", "320k")

    # ── Load DiT model ───────────────────────────────────────────
    print("[Init] Loading DiT model...")
    dit_model = build_model(dit_model_config, stage="DiT")

    # Find checkpoint
    ckpt_dir = os.path.join(SEED_VC_DIR, "checkpoints", "Seed-VC")
    ckpt_candidates = [
        os.path.join(ckpt_dir, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth"),
        os.path.join(ckpt_dir, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned.pth"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break

    if ckpt_path is None:
        # List what's available
        if os.path.exists(ckpt_dir):
            files = os.listdir(ckpt_dir)
            pth_files = [f for f in files if f.endswith('.pth')]
            if pth_files:
                ckpt_path = os.path.join(ckpt_dir, pth_files[0])
                print(f"[Init] Using checkpoint: {ckpt_path}")
            else:
                print(f"[Init] No .pth found in {ckpt_dir}, files: {files}")
        else:
            print(f"[Init] Checkpoint dir not found: {ckpt_dir}")

    if ckpt_path:
        load_checkpoint(dit_model, None, ckpt_path,
                        load_only_params=True, ignore_modules=[], is_distributed=False)
        print(f"[Init] DiT loaded from {ckpt_path}")

    dit_model = dit_model.to(DEVICE).to(DTYPE)
    dit_model.eval()

    # ── Load CAMPPlus speaker encoder ────────────────────────────
    print("[Init] Loading CAMPPlus...")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_ckpt = os.path.join(ckpt_dir, "campplus.pth")
    if os.path.exists(campplus_ckpt):
        campplus_model.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
    else:
        print(f"[Init] campplus.pth not found at {campplus_ckpt}")
    campplus_model = campplus_model.to(DEVICE).eval()

    # ── Load RMVPE (pitch extractor) ─────────────────────────────
    print("[Init] Loading RMVPE...")
    rmvpe_ckpt = os.path.join(ckpt_dir, "rmvpe.pt")
    if os.path.exists(rmvpe_ckpt):
        rmvpe_model = RMVPE(rmvpe_ckpt, is_half=True, device=DEVICE)
    else:
        print(f"[Init] rmvpe.pt not found at {rmvpe_ckpt}")

    # ── Load BigVGAN vocoder ─────────────────────────────────────
    print("[Init] Loading BigVGAN vocoder...")
    vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_44khz_128band_512x", use_cuda_kernel=False)
    vocoder = vocoder.to(DEVICE).eval()
    vocoder.remove_weight_norm()

    # ── Load Whisper (content encoder) ───────────────────────────
    print("[Init] Loading Whisper...")
    whisper_name = dit_model_config.speech_tokenizer_params.get("name", "openai/whisper-small")
    whisper_model = WhisperModel.from_pretrained(whisper_name).to(DEVICE).to(DTYPE).eval()
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_name)

    print("[Init] All models loaded successfully!")


def download_file(url: str, dest_path: str):
    """Download a file from URL to local path."""
    print(f"[Download] {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[Download] Done: {size_mb:.1f} MB")


def upload_file(file_path: str, filename: str, max_retries: int = 3) -> str:
    """Upload file to tmpfiles.org with retry, return direct download URL."""
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"[Upload] Uploading {filename} ({size_mb:.1f} MB)...")

    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (filename, f, "audio/wav")},
                    timeout=120,
                )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Response not success: {data}")
            url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            print(f"[Upload] Done: {url}")
            return url
        except Exception as e:
            print(f"[Upload] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)  # 等 3 秒重试
            else:
                raise RuntimeError(f"Upload failed after {max_retries} attempts: {e}")


def separate_vocals(song_path: str, output_dir: str):
    """Separate vocals and instrumental using demucs."""
    print(f"[Demucs] Separating vocals...")
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        song_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")

    song_name = os.path.splitext(os.path.basename(song_path))[0]
    separated_dir = os.path.join(output_dir, "htdemucs", song_name)
    vocals_path = os.path.join(separated_dir, "vocals.wav")
    instrumental_path = os.path.join(separated_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Vocals not found: {os.listdir(separated_dir)}")

    print(f"[Demucs] Done.")
    return vocals_path, instrumental_path


# 可选模型版本
MODEL_VERSIONS = {
    "standard": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth",
    "fine_tuned": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
    "fine_tuned_v2": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
}


def run_seed_vc_direct(source_path: str, target_path: str, output_path: str,
                       pitch_shift: int = 0, diffusion_steps: int = 25,
                       cfg_rate: float = 0.7, model_version: str = "standard"):
    """
    Run Seed-VC inference via subprocess.
    model_version: "standard" / "fine_tuned" / "fine_tuned_v2"
    """
    # Resolve checkpoint path
    ckpt_name = MODEL_VERSIONS.get(model_version, MODEL_VERSIONS["standard"])
    ckpt_path = os.path.join(SEED_VC_DIR, "checkpoints", "Seed-VC", ckpt_name)
    config_path = os.path.join(SEED_VC_DIR, "configs", "presets", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    print(f"[Inference] Model: {model_version} → {ckpt_name}")

    cmd = [
        "python", os.path.join(SEED_VC_DIR, "inference.py"),
        "--source", source_path,
        "--target", target_path,
        "--output", os.path.dirname(output_path),
        "--checkpoint", ckpt_path,
        "--config", config_path,
        "--diffusion-steps", str(diffusion_steps),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", str(cfg_rate),
        "--f0-condition", "True",
        "--auto-f0-adjust", "True",
        "--semi-tone-shift", str(pitch_shift),
        "--fp16", "True",
    ]

    print(f"[Inference] Starting Seed-VC...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=SEED_VC_DIR)
    elapsed = time.time() - start

    print(f"[Inference] Done in {elapsed:.1f}s, exit={result.returncode}")
    if result.stderr:
        # Extract RTF line
        for line in result.stderr.split('\n'):
            if 'RTF' in line:
                print(f"[Inference] {line.strip()}")

    if result.returncode != 0:
        raise RuntimeError(f"Seed-VC failed: {result.stderr[-300:]}")

    # Find output file
    out_dir = os.path.dirname(output_path)
    wav_files = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError(f"No output .wav found")

    generated = os.path.join(out_dir, wav_files[0])
    if generated != output_path:
        shutil.move(generated, output_path)

    return output_path


def mix_audio(vocals_path: str, instrumental_path: str, output_path: str,
              vocal_volume: float = 1.0, instrumental_volume: float = 1.0,
              reverb: float = 0.0):
    """
    Mix converted vocals with original instrumental using ffmpeg.
    vocal_volume: 人声音量 (1.0=原始, 1.3=突出人声)
    instrumental_volume: 伴奏音量 (1.0=原始, 0.8=压低伴奏)
    reverb: 混响强度 (0.0=无, 0.3=轻微KTV感, 0.6=强混响)
    """
    print(f"[Mix] Mixing: vocal_vol={vocal_volume}, inst_vol={instrumental_volume}, reverb={reverb}")

    # 构建人声滤镜链
    vocal_filters = [f"volume={vocal_volume}"]

    # 添加混响效果（KTV 感）
    if reverb > 0:
        # aecho: in_gain|out_gain|delays(ms)|decays
        # 轻度混响模拟 KTV 效果
        delay1 = 60    # 短延迟
        delay2 = 120   # 中延迟
        decay1 = round(reverb * 0.5, 2)   # 衰减系数
        decay2 = round(reverb * 0.3, 2)
        vocal_filters.append(f"aecho=0.8:0.75:{delay1}|{delay2}:{decay1}|{decay2}")
        # 加一点高频提升，让声音更亮
        vocal_filters.append("equalizer=f=3000:t=q:w=1.5:g=2")
        # 加一点低频温暖感
        vocal_filters.append("equalizer=f=200:t=q:w=1:g=1")

    vocal_chain = ",".join(vocal_filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-i", instrumental_path,
        "-filter_complex",
        f"[0:a]{vocal_chain}[v];[1:a]volume={instrumental_volume}[i];[v][i]amix=inputs=2:duration=longest",
        "-ac", "2", "-ar", "44100",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr[-300:]}")
    print(f"[Mix] Done.")


def handler(job):
    """RunPod Serverless handler — full cover song pipeline."""
    job_input = job["input"]

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    diffusion_steps = int(job_input.get("diffusion_steps", 25))
    cfg_rate = float(job_input.get("cfg_rate", 0.7))           # 音色还原度
    vocal_volume = float(job_input.get("vocal_volume", 1.1))    # 人声音量（默认略突出）
    instrumental_volume = float(job_input.get("instrumental_volume", 0.9))  # 伴奏音量
    reverb = float(job_input.get("reverb", 0.25))              # 混响（默认轻微KTV感）
    model_version = job_input.get("model_version", "standard")   # 模型版本

    print(f"\n{'='*60}")
    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={diffusion_steps}")
    print(f"[Job] cfg_rate={cfg_rate}, vocal_vol={vocal_volume}, inst_vol={instrumental_volume}, reverb={reverb}")
    print(f"[Job] model={model_version}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            total_start = time.time()

            # ── Stage 1: Download ────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "downloading", "progress": 0.05
            })

            song_path = os.path.join(tmpdir, "song_input.wav")
            voice_path = os.path.join(tmpdir, "voice_ref.wav")
            vc_output_dir = os.path.join(tmpdir, "vc_output")
            demucs_output_dir = os.path.join(tmpdir, "demucs_output")
            os.makedirs(vc_output_dir, exist_ok=True)

            t = time.time()
            download_file(song_url, song_path)
            download_file(voice_url, voice_path)
            download_time = time.time() - t

            song_info = torchaudio.info(song_path)
            song_duration = song_info.num_frames / song_info.sample_rate
            print(f"[Job] Song: {song_duration:.1f}s, Download: {download_time:.1f}s")

            # ── Stage 2: Vocal separation ────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "separating", "progress": 0.1
            })

            t = time.time()
            vocals_path, instrumental_path = separate_vocals(song_path, demucs_output_dir)
            separation_time = time.time() - t
            print(f"[Job] Separation: {separation_time:.1f}s")

            # ── Stage 3: Voice conversion ────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "converting", "progress": 0.3
            })

            t = time.time()
            converted_vocals = os.path.join(vc_output_dir, "converted.wav")
            run_seed_vc_direct(
                vocals_path, voice_path, converted_vocals,
                pitch_shift=pitch_shift,
                diffusion_steps=diffusion_steps,
                cfg_rate=cfg_rate,
                model_version=model_version
            )
            conversion_time = time.time() - t
            print(f"[Job] Conversion: {conversion_time:.1f}s")

            # ── Stage 4: Mix ─────────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "mixing", "progress": 0.85
            })

            t = time.time()
            final_output = os.path.join(tmpdir, "final_cover.wav")
            mix_audio(converted_vocals, instrumental_path, final_output,
                      vocal_volume=vocal_volume,
                      instrumental_volume=instrumental_volume,
                      reverb=reverb)
            mix_time = time.time() - t

            # ── Stage 5: Upload ──────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "uploading", "progress": 0.95
            })

            output_info = torchaudio.info(final_output)
            output_duration = output_info.num_frames / output_info.sample_rate
            output_size_mb = os.path.getsize(final_output) / (1024 * 1024)

            t = time.time()
            output_url = upload_file(final_output, f"cover_{task_id}.wav")
            upload_time = time.time() - t

            total_time = time.time() - total_start

            print(f"\n[Job] === SUMMARY ===")
            print(f"[Job] Download:   {download_time:.1f}s")
            print(f"[Job] Separation: {separation_time:.1f}s")
            print(f"[Job] Conversion: {conversion_time:.1f}s")
            print(f"[Job] Mix:        {mix_time:.1f}s")
            print(f"[Job] Upload:     {upload_time:.1f}s")
            print(f"[Job] TOTAL:      {total_time:.1f}s")
            print(f"[Job] Output:     {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "output_url": output_url,
                "duration": round(output_duration, 2),
                "download_time": round(download_time, 2),
                "separation_time": round(separation_time, 2),
                "conversion_time": round(conversion_time, 2),
                "mix_time": round(mix_time, 2),
                "upload_time": round(upload_time, 2),
                "total_time": round(total_time, 2),
                "output_format": "wav",
                "sample_rate": output_info.sample_rate,
                "size_mb": round(output_size_mb, 2),
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
            }


# ── Startup: Load models once ────────────────────────────────────
if __name__ == "__main__":
    print("[Init] Seed-VC Cover Song Worker v2")
    print("[Init] Loading all models into GPU memory...")
    try:
        load_all_models()
    except Exception as e:
        print(f"[Init] WARNING: Model preload failed: {e}")
        print("[Init] Models will be loaded on first inference via subprocess")

    runpod.serverless.start({"handler": handler})
