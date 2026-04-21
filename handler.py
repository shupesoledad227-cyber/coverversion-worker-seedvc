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
import tempfile
import time
import subprocess
import traceback
import shutil

import requests
import runpod
import torchaudio

# Seed-VC inference 由子进程执行（cwd=SEED_VC_DIR），父进程不需要导入其模块
SEED_VC_DIR = "/app/seed-vc"


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


def separate_vocals(song_path: str, output_dir: str, shifts: int = 0):
    """Separate vocals and instrumental using demucs."""
    print(f"[Demucs] Separating vocals (shifts={shifts})...")
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
    ]
    if shifts > 0:
        cmd += ["--shifts", str(shifts)]
    cmd += ["-o", output_dir, song_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


def separate_karaoke(vocals_path: str, output_dir: str):
    """Separate lead vocals from backing vocals using BS Roformer Karaoke model."""
    print(f"[Karaoke] Separating lead/backing vocals...")
    os.makedirs(output_dir, exist_ok=True)

    # Copy vocals to a temp input folder (MSST reads from folder, not single file)
    karaoke_input = os.path.join(output_dir, "input")
    os.makedirs(karaoke_input, exist_ok=True)
    shutil.copy(vocals_path, os.path.join(karaoke_input, os.path.basename(vocals_path)))

    cmd = [
        "python", "/app/msst/inference.py",
        "--model_type", "bs_roformer",
        "--config_path", "/app/msst/config_karaoke_frazer_becruily.yaml",
        "--start_check_point", "/app/msst/bs_roformer_karaoke_frazer_becruily.ckpt",
        "--input_folder", karaoke_input,
        "--store_dir", output_dir,
        "--extract_instrumental",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd="/app/msst")
    if result.stdout:
        print(f"[Karaoke] STDOUT: {result.stdout[-300:]}")
    if result.returncode != 0:
        print(f"[Karaoke] STDERR: {result.stderr[-300:]}")
        raise RuntimeError(f"Karaoke failed: {result.stderr[-300:]}")

    # Find lead and backing vocals (MSST creates subdirectories)
    lead_path = None
    backing_path = None
    all_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if not f.endswith('.wav'):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, output_dir)
            all_files.append(rel)
            lower = f.lower()
            if 'instrumental' in lower or 'other' in lower:
                backing_path = full
            elif 'vocal' in lower:
                lead_path = full
    print(f"[Karaoke] Output files: {all_files}")

    if not lead_path:
        raise RuntimeError(f"Karaoke lead vocals not found in: {all_files}")

    print(f"[Karaoke] Done: lead={os.path.basename(lead_path)}, backing={os.path.basename(backing_path) if backing_path else 'none'}")
    return lead_path, backing_path


# 可选模型版本
MODEL_VERSIONS = {
    "fine_tuned_v2": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
}


def run_seed_vc_direct(source_path: str, target_path: str, output_path: str,
                       pitch_shift: int = 0, diffusion_steps: int = 25,
                       cfg_rate: float = 0.7, model_version: str = "fine_tuned_v2",
                       auto_f0_adjust: bool = False):
    """
    Run Seed-VC inference via subprocess.
    model_version: 仅支持 "fine_tuned_v2"
    """
    # Resolve checkpoint path
    ckpt_name = MODEL_VERSIONS.get(model_version, MODEL_VERSIONS["fine_tuned_v2"])
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
        "--auto-f0-adjust", str(auto_f0_adjust),
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


def analyze_vocal_f0(vocals_path: str) -> dict:
    """
    Analyze F0 (fundamental frequency) of a clean vocal track using librosa.pyin.
    Returns median/mean F0 and musical note name.
    """
    try:
        import librosa
        import numpy as np
        # Load full audio to find the loudest 30s segment
        y, sr = librosa.load(vocals_path, sr=16000, mono=True)
        duration = len(y) / sr

        # Only analyze the loudest 30s — that's where vocals are densest (usually chorus)
        analyze_len = 30
        if duration > analyze_len:
            frame_len = sr  # 1 second per frame
            rms_per_sec = []
            for i in range(0, len(y) - frame_len, frame_len):
                rms_per_sec.append(float(np.sqrt(np.mean(y[i:i+frame_len] ** 2))))
            # Sliding window to find best 30s
            window = analyze_len
            best_start = 0
            best_energy = 0
            for i in range(len(rms_per_sec) - window + 1):
                energy = sum(rms_per_sec[i:i+window])
                if energy > best_energy:
                    best_energy = energy
                    best_start = i
            y = y[best_start * sr : (best_start + window) * sr]
            print(f"[F0] Using loudest 30s segment starting at {best_start}s (total {duration:.0f}s)")
        else:
            print(f"[F0] Audio is {duration:.1f}s, analyzing full track")

        # pyin: probabilistic YIN, higher quality than plain YIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65Hz (男低音下限)
            fmax=librosa.note_to_hz('C6'),   # ~1047Hz (女高音上限)
            sr=sr,
            frame_length=2048,
        )
        valid = f0[~np.isnan(f0)]
        if len(valid) == 0:
            return {"ok": False, "error": "no voiced frames"}

        f0_median = float(np.median(valid))
        f0_mean = float(np.mean(valid))
        # 截尾均值 5%-95%,去极端值
        lo, hi = np.percentile(valid, [5, 95])
        trimmed = valid[(valid >= lo) & (valid <= hi)]
        f0_trimmed_mean = float(np.mean(trimmed)) if len(trimmed) > 0 else f0_mean

        note = librosa.hz_to_note(f0_median)
        print(f"[F0] median={f0_median:.1f}Hz mean={f0_mean:.1f}Hz trimmed={f0_trimmed_mean:.1f}Hz note={note} valid_frames={len(valid)}")
        return {
            "ok": True,
            "f0_median": round(f0_median, 1),
            "f0_mean": round(f0_mean, 1),
            "f0_trimmed_mean": round(f0_trimmed_mean, 1),
            "note": note,
            "valid_frames": int(len(valid)),
        }
    except Exception as e:
        print(f"[F0] Analysis failed: {e}")
        return {"ok": False, "error": str(e)}


def mix_audio(vocals_path: str, instrumental_path: str, output_path: str,
              vocal_volume: float = 1.0, instrumental_volume: float = 1.0,
              reverb: float = 0.0):
    """
    Mix converted vocals with original instrumental.
    Uses pedalboard for professional reverb/EQ, ffmpeg for final mix.

    vocal_volume: 人声音量 (1.0=原始, 3.0=最大)
    instrumental_volume: 伴奏音量 (1.0=原始)
    reverb: 混响强度 (0.0=干声, 0.3=KTV包厢, 0.6=大厅, 0.8=教堂)
    """
    import numpy as np
    from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain
    from pedalboard.io import AudioFile

    print(f"[Mix] Processing: vocal_vol={vocal_volume}, inst_vol={instrumental_volume}, reverb={reverb}")

    # ── Step 1: 处理人声音效（pedalboard）──────────────────────
    # 读取人声
    with AudioFile(vocals_path) as f:
        vocal_sr = f.samplerate
        vocal_audio = f.read(f.frames)

    # Fade-in/out：消除起始"噗"声和结尾"咔嗒"声
    # Seed-VC diffusion 模型的第一帧经常有瞬态杂波，fade-in 可以彻底消除
    fade_samples = int(vocal_sr * 0.5)  # 500ms
    if vocal_audio.shape[-1] > fade_samples * 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        for ch in range(vocal_audio.shape[0]):
            vocal_audio[ch, :fade_samples] *= fade_in
            vocal_audio[ch, -fade_samples:] *= fade_out

    # 构建人声效果链
    vocal_effects = []

    # 高通滤波：去掉低频噪音（<80Hz）
    vocal_effects.append(HighpassFilter(cutoff_frequency_hz=80))

    # 压缩器：让人声更稳定，动态更一致（模拟 KTV 硬件压缩）
    vocal_effects.append(Compressor(
        threshold_db=-20,
        ratio=3.0,
        attack_ms=10,
        release_ms=100,
    ))

    # 音量调整
    if vocal_volume != 1.0:
        gain_db = 20 * np.log10(max(vocal_volume, 0.01))
        vocal_effects.append(Gain(gain_db=gain_db))

    # 混响（真正的卷积混响，模拟 AU 的"沉闷的卡拉OK酒吧"效果）
    if reverb > 0:
        # room_size: 0.0=小房间, 1.0=大教堂
        # damping: 高频衰减，越高越温暖（KTV 偏温暖）
        # wet_level: 混响量（对应 AU 的"湿声"）
        # dry_level: 原声量（对应 AU 的"干声"）
        # width: 立体声宽度
        room = min(0.2 + reverb * 0.8, 0.95)       # 0.2-0.84
        damp = min(0.5 + reverb * 0.3, 0.85)       # 0.5-0.74
        wet = min(0.15 + reverb * 0.55, 0.55)      # 0.15-0.59
        dry = max(0.7 - reverb * 0.3, 0.45)        # 0.7-0.46

        vocal_effects.append(Reverb(
            room_size=room,
            damping=damp,
            wet_level=wet,
            dry_level=dry,
            width=0.8,
        ))

    # 应用效果链到人声
    vocal_board = Pedalboard(vocal_effects)
    processed_vocal = vocal_board(vocal_audio, vocal_sr)

    # 保存处理后的人声
    processed_vocal_path = vocals_path.replace('.wav', '_fx.wav')
    with AudioFile(processed_vocal_path, 'w', vocal_sr, processed_vocal.shape[0]) as f:
        f.write(processed_vocal)

    print(f"[Mix] Vocal effects applied: {len(vocal_effects)} effects")

    # ── Step 2: 处理伴奏（模拟 AU 的"夜总会楼下" EQ）────────
    # 读取伴奏
    with AudioFile(instrumental_path) as f:
        inst_sr = f.samplerate
        inst_audio = f.read(f.frames)

    inst_effects = []

    # 伴奏音量
    if instrumental_volume != 1.0:
        gain_db = 20 * np.log10(max(instrumental_volume, 0.01))
        inst_effects.append(Gain(gain_db=gain_db))

    # 给伴奏也加一点轻微混响（模拟同一空间）
    if reverb > 0.2:
        inst_effects.append(Reverb(
            room_size=min(0.15 + reverb * 0.4, 0.5),
            damping=0.7,
            wet_level=min(reverb * 0.2, 0.15),   # 伴奏混响很轻
            dry_level=0.85,
            width=1.0,
        ))

    if inst_effects:
        inst_board = Pedalboard(inst_effects)
        processed_inst = inst_board(inst_audio, inst_sr)
    else:
        processed_inst = inst_audio

    processed_inst_path = instrumental_path.replace('.wav', '_fx.wav')
    with AudioFile(processed_inst_path, 'w', inst_sr, processed_inst.shape[0]) as f:
        f.write(processed_inst)

    print(f"[Mix] Instrumental effects applied: {len(inst_effects)} effects")

    # ── Step 3: 混合人声+伴奏（ffmpeg）────────────────────────
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_vocal_path,
        "-i", processed_inst_path,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 1:normalize=0",
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

    # Warmup mode: Worker 已启动，直接返回
    if job_input.get("mode") == "warmup":
        print("[Warmup] Worker is warm and ready.")
        return {"status": "warm", "message": "Worker is ready"}

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    user_f0 = float(job_input.get("user_f0", 0))  # 用户声音 F0 (Hz)，> 0 时自动算 pitch_shift
    diffusion_steps = int(job_input.get("diffusion_steps", 25))
    cfg_rate = float(job_input.get("cfg_rate", 0.7))           # 音色还原度
    vocal_volume = float(job_input.get("vocal_volume", 1.1))    # 人声音量（默认略突出）
    instrumental_volume = float(job_input.get("instrumental_volume", 0.9))  # 伴奏音量
    reverb = float(job_input.get("reverb", 0.25))              # 混响（默认轻微KTV感）
    auto_f0_adjust = bool(job_input.get("auto_f0_adjust", False))    # 自动音高适配（歌声转换建议关闭）
    output_format = job_input.get("output_format", "mp3_320")       # wav / mp3_320 / mp3_192
    cover_image = job_input.get("cover_image", "")                  # 封面图名称（如 img_cover_default_01）
    artist_name = job_input.get("artist_name", "")                  # 歌手名（嵌入 MP3 metadata）
    song_title = job_input.get("song_title", "")                    # 歌曲名（嵌入 MP3 metadata）
    demucs_shifts = int(job_input.get("demucs_shifts", 0))           # Demucs TTA shifts（0=最快，2=更干净，3=最干净）
    karaoke_enabled = bool(job_input.get("karaoke_enabled", False))  # 是否分离主唱和和声

    # 固定使用 fine_tuned_v2（最佳）
    model_version = "fine_tuned_v2"

    print(f"\n{'='*60}")
    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={diffusion_steps}")
    print(f"[Job] cfg_rate={cfg_rate}, vocal_vol={vocal_volume}, inst_vol={instrumental_volume}, reverb={reverb}")
    print(f"[Job] auto_f0={auto_f0_adjust}, karaoke={karaoke_enabled}")
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
            vocals_path, instrumental_path = separate_vocals(
                song_path, demucs_output_dir, shifts=demucs_shifts)
            separation_engine = "demucs"  # 固定 demucs
            separation_time = time.time() - t
            print(f"[Job] Separation ({separation_engine}): {separation_time:.1f}s")

            # ── Stage 2.1: Karaoke separation (optional) ────────
            backing_vocals_path = None
            karaoke_time = 0
            if karaoke_enabled:
                t = time.time()
                karaoke_out_dir = os.path.join(tmpdir, "karaoke_out")
                lead_path, backing_path = separate_karaoke(vocals_path, karaoke_out_dir)
                vocals_path = lead_path  # 只转换主唱
                backing_vocals_path = backing_path
                karaoke_time = time.time() - t
                print(f"[Job] Karaoke: {karaoke_time:.1f}s")

            # ── Stage 2.5: Analyze original vocal F0 ─────────────
            t = time.time()
            song_vocal_f0 = analyze_vocal_f0(vocals_path)
            f0_analysis_time = time.time() - t
            print(f"[Job] F0 Analysis: {f0_analysis_time:.1f}s")

            # ── Stage 2.6: Auto pitch_shift (if user_f0 provided) ──
            import math
            original_pitch_shift = pitch_shift
            if user_f0 > 0 and song_vocal_f0.get("ok") and song_vocal_f0["f0_median"] > 0:
                song_f0 = song_vocal_f0["f0_median"]
                raw_shift = 12 * math.log2(user_f0 / song_f0)
                if raw_shift < 0:
                    # 负值：直接用，最小 -12
                    pitch_shift = max(-12, round(raw_shift))
                else:
                    # 正值：除以 3 再四舍五入，最大 +12
                    pitch_shift = min(12, round(raw_shift / 3))
                print(f"[Job] Auto pitch_shift: user_f0={user_f0:.1f}Hz, song_f0={song_f0:.1f}Hz, "
                      f"raw={raw_shift:.2f}, applied={pitch_shift} (original={original_pitch_shift})")
            else:
                print(f"[Job] Manual pitch_shift: {pitch_shift} (user_f0={'%.1f' % user_f0 if user_f0 > 0 else 'not provided'})")
            print(f"[Job] song_vocal_f0={song_vocal_f0}")
            print(f"[Job] original_pitch_shift={original_pitch_shift}, applied_pitch_shift={pitch_shift}")

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
                model_version=model_version,
                auto_f0_adjust=auto_f0_adjust
            )
            conversion_time = time.time() - t
            print(f"[Job] Conversion: {conversion_time:.1f}s")

            # ── Stage 4: Mix ─────────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "mixing", "progress": 0.85
            })

            t = time.time()
            # If karaoke enabled, first mix backing vocals into instrumental
            if karaoke_enabled and backing_vocals_path and os.path.exists(backing_vocals_path):
                inst_with_backing = os.path.join(tmpdir, "inst_with_backing.wav")
                mix_cmd = [
                    "ffmpeg", "-y",
                    "-i", instrumental_path,
                    "-i", backing_vocals_path,
                    "-filter_complex",
                    "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 1:normalize=0",
                    "-ac", "2", "-ar", "44100",
                    inst_with_backing,
                ]
                subprocess.run(mix_cmd, capture_output=True, timeout=120)
                if os.path.exists(inst_with_backing):
                    instrumental_path = inst_with_backing
                    print(f"[Mix] Backing vocals merged into instrumental")

            final_output = os.path.join(tmpdir, "final_cover.wav")
            mix_audio(converted_vocals, instrumental_path, final_output,
                      vocal_volume=vocal_volume,
                      instrumental_volume=instrumental_volume,
                      reverb=reverb)
            mix_time = time.time() - t

            # ── Stage 5: Format conversion ───────────────────────
            t = time.time()
            output_info = torchaudio.info(final_output)
            output_duration = output_info.num_frames / output_info.sample_rate

            # Convert to target format if not WAV
            if output_format in ("mp3_320", "mp3_192"):
                bitrate = "320k" if output_format == "mp3_320" else "192k"
                mp3_output = final_output.replace(".wav", ".mp3")

                # 下载封面图（如果指定了）
                cover_path = None
                if cover_image:
                    cover_url = f"https://raw.githubusercontent.com/WhistleB/coverversion-worker/main/assets/covers/{cover_image}.png"
                    cover_path = os.path.join(tmpdir, "cover.png")
                    try:
                        download_file(cover_url, cover_path)
                    except Exception:
                        cover_path = None
                        print(f"[Cover] 封面下载失败，跳过")

                # 构建 metadata 参数
                metadata_args = []
                if artist_name:
                    metadata_args += ["-metadata", f"artist={artist_name}"]
                if song_title:
                    metadata_args += ["-metadata", f"title={song_title}"]
                metadata_args += ["-metadata", "album=AI Cover"]

                # 转 MP3 + 嵌入封面 + metadata
                if cover_path and os.path.exists(cover_path):
                    convert_cmd = [
                        "ffmpeg", "-y",
                        "-i", final_output,
                        "-i", cover_path,
                        "-map", "0:a", "-map", "1",
                        "-c:a", "libmp3lame", "-b:a", bitrate,
                        "-c:v", "png",
                        "-disposition:v", "attached_pic",
                        "-id3v2_version", "3",
                    ] + metadata_args + [mp3_output]
                    print(f"[Format] MP3 {bitrate} + cover + metadata: artist={artist_name}, title={song_title}")
                else:
                    convert_cmd = ["ffmpeg", "-y", "-i", final_output, "-b:a", bitrate] + metadata_args + [mp3_output]
                    print(f"[Format] MP3 {bitrate} + metadata (no cover)")

                subprocess.run(convert_cmd, capture_output=True, timeout=60)
                if os.path.exists(mp3_output):
                    final_output = mp3_output

            output_size_mb = os.path.getsize(final_output) / (1024 * 1024)
            file_ext = os.path.splitext(final_output)[1]  # .wav or .mp3
            format_time = time.time() - t
            print(f"[Job] Format:     {format_time:.1f}s")

            # ── Stage 6: Upload ──────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "uploading", "progress": 0.95
            })

            t = time.time()
            output_url = upload_file(final_output, f"cover_{task_id}{file_ext}")
            upload_time = time.time() - t

            total_time = time.time() - total_start

            print(f"\n[Job] === SUMMARY ===")
            print(f"[Job] 1.Download:       {download_time:.1f}s")
            print(f"[Job] 2.Separation:     {separation_time:.1f}s ({separation_engine})")
            if karaoke_enabled:
                print(f"[Job] 3.Karaoke:        {karaoke_time:.1f}s (lead/backing split)")
            print(f"[Job] 4.F0 Analyze:     {f0_analysis_time:.1f}s")
            print(f"[Job] 5.Conversion:     {conversion_time:.1f}s (Seed-VC)")
            print(f"[Job] 6.Mix:            {mix_time:.1f}s")
            print(f"[Job] 7.Format:         {format_time:.1f}s")
            print(f"[Job] 8.Upload:         {upload_time:.1f}s (final)")
            print(f"[Job] ──────────────────")
            print(f"[Job] TOTAL:            {total_time:.1f}s")
            print(f"[Job] Output:     {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "output_url": output_url,
                "duration": round(output_duration, 2),
                "download_time": round(download_time, 2),
                "separation_time": round(separation_time, 2),
                "f0_analysis_time": round(f0_analysis_time, 2),
                "conversion_time": round(conversion_time, 2),
                "mix_time": round(mix_time, 2),
                "format_time": round(format_time, 2),
                "upload_time": round(upload_time, 2),
                "total_time": round(total_time, 2),
                "output_format": output_format,
                "sample_rate": output_info.sample_rate,
                "size_mb": round(output_size_mb, 2),
                "song_vocal_f0": song_vocal_f0,
                "applied_pitch_shift": pitch_shift,
                "original_pitch_shift": original_pitch_shift,
                "separation_engine": separation_engine,
                "karaoke_enabled": karaoke_enabled,
                "karaoke_time": round(karaoke_time, 2) if karaoke_enabled else 0,
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
            }


if __name__ == "__main__":
    print("[Init] Seed-VC Cover Song Worker v2")
    runpod.serverless.start({"handler": handler})
