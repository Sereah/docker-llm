"""
config.py — 全局配置（单一职责：只管配置读取与校验）
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # ── 服务器 ──────────────────────────────────────────────
    host: str = Field(default="::", description="监听地址")
    port: int = Field(default=8000, description="监听端口")
    workers: int = Field(default=1, description="Uvicorn worker 数（GPU 服务建议 1）")
    log_level: str = Field(default="info")

    # ── 模型 ──────────────────────────────────────────────
    model_name: str = Field(
        default="/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        description="Qwen3-TTS 模型路径（HuggingFace id 或本地路径）",
    )
    device: str = Field(default="cuda:0", description="cuda:0 / cuda / cpu / mps")
    torch_dtype: str = Field(default="bfloat16", description="float16 / bfloat16 / float32")
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="注意力实现，flash_attention_2 / sdpa / eager",
    )

    # ── 音色 / 语言默认 ────────────────────────────────────
    speaker: str = Field(
        default="Vivian",
        description="默认音色",
    )
    language: str = Field(
        default="Auto",
        description="默认语言，Auto 为自动检测",
    )

    # ── 音频输出 ──────────────────────────────────────────
    audio_format: Literal["wav", "mp3", "opus"] = Field(
        default="wav", description="输出音频格式"
    )
    sample_rate: int = Field(default=22050)

    # ── 文本分块策略 ──────────────────────────────────────
    short_text_max_chars: int = Field(
        default=15,
        description="短文本上限（字），≤阈值一次性合成；>阈值分块流式返回",
    )
    stream_chunk_max_chars: int = Field(
        default=25,
        description="流式分块每段最大字符数",
    )
    stream_prefetch: int = Field(
        default=3,
        description="流式预取块数，平衡延迟与吞吐",
    )

    # ── WebSocket ────────────────────────────────────────
    ws_ping_interval: float = Field(default=20.0, description="WebSocket 心跳间隔（秒）")
    ws_ping_timeout: float = Field(default=10.0, description="WebSocket 心跳超时（秒）")

    # ── 限流 ─────────────────────────────────────────────
    max_concurrent_sessions: int = Field(
        default=4, description="最大并发合成会话数"
    )

    class Config:
        env_file = ".env"
        env_prefix = "TTS_"


settings = Settings()
