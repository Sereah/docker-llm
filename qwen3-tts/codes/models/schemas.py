"""
models/schemas.py — 请求/响应数据模型（单一职责）

类比 Android 中的 data class / State：
  这些 Pydantic 模型定义了 API 契约，handler 只负责校验输入 → 调用 service → 格式化输出。
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TTSRequest(BaseModel):
    request_id: str = Field(default="", description="客户端请求 ID，响应中原样带回")
    text: str = Field(..., description="待合成文本", max_length=50000)
    speaker: Optional[str] = Field(None, description="音色名称")
    language: Optional[str] = Field(None, description="合成语言")
    instruct: Optional[str] = Field(None, description="风格指令")


class ChunkResult(BaseModel):
    index: int
    text: str
    audio_b64: str
    duration_ms: int
    sample_rate: int


class TTSResponse(BaseModel):
    request_id: str = ""
    total_chunks: int
    chunks: List[ChunkResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_sessions: int
    synthesizing_sessions: int
