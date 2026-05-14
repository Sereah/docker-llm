"""
utils/audio_encoder.py — 音频编码工具（单一职责：PCM ↔ WAV ↔ Base64）

说明：
  - Qwen3-TTS 推理输出为 numpy float32 PCM 数组
  - 本模块负责将其封装为 WAV bytes，再 base64 编码后通过网络传输
  - 保持无状态，所有函数为纯函数，便于单元测试
"""

import base64
import io
import struct
import wave
from typing import Optional

import numpy as np


def pcm_to_wav_bytes(
    pcm: np.ndarray,
    sample_rate: int = 22050,
    n_channels: int = 1,
) -> bytes:
    """
    将 float32 PCM 数组转换为标准 WAV 格式字节流。

    Args:
        pcm:         float32 numpy array，值域 [-1.0, 1.0]
        sample_rate: 采样率（Hz）
        n_channels:  声道数（Qwen3-TTS 输出为单声道）

    Returns:
        完整 WAV bytes（含 RIFF 头）
    """
    # float32 → int16（16-bit PCM）
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return buf.getvalue()


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    """
    WAV bytes → Base64 字符串（无换行，直接嵌入 JSON）。
    """
    return base64.b64encode(wav_bytes).decode("utf-8")


def pcm_to_base64(
    pcm: np.ndarray,
    sample_rate: int = 22050,
) -> str:
    """
    便捷函数：PCM → WAV → Base64，一步完成。
    """
    wav = pcm_to_wav_bytes(pcm, sample_rate)
    return wav_bytes_to_base64(wav)


def estimate_duration_ms(n_samples: int, sample_rate: int = 22050) -> float:
    """估算音频时长（毫秒），用于客户端播放调度。"""
    return (n_samples / sample_rate) * 1000.0
