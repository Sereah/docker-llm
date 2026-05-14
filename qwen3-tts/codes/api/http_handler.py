"""
api/http_handler.py — HTTP REST TTS 接口（单一职责：HTTP 协议层）

端点：
  POST /api/tts
    Body: { "request_id": "req-001", "text": "...", "speaker": "Vivian", "language": "Chinese", "instruct": null }
    Response: { "request_id": "req-001", "total_chunks": N, "chunks": [...] }

  GET  /api/health
    Response: { "status": "ok", "model_loaded": true, "active_sessions": N }
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from loguru import logger

from dependencies import get_engine, get_session_manager, get_service
from models.schemas import TTSRequest, ChunkResult, TTSResponse, HealthResponse
from services.tts_service import TTSService
from utils.audio_encoder import estimate_duration_ms, pcm_to_base64


router = APIRouter(prefix="/api", tags=["TTS"])


@router.post(
    "/tts",
    response_model=TTSResponse,
    summary="TTS 语音合成",
)
async def http_tts(
    req: TTSRequest,
    service: TTSService = Depends(get_service),
) -> TTSResponse:
    if not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文本为空",
        )

    logger.info(f"[HTTP] request_id={req.request_id}")

    synthesized = await service.synthesize(
        text=req.text,
        speaker=req.speaker or "",
        language=req.language or "",
        instruct=req.instruct,
    )

    results: List[ChunkResult] = []
    for chunk in synthesized:
        audio_b64 = pcm_to_base64(chunk.pcm, chunk.sample_rate)
        duration_ms = int(estimate_duration_ms(len(chunk.pcm), chunk.sample_rate))
        results.append(
            ChunkResult(
                index=chunk.index,
                text=chunk.text,
                audio_b64=audio_b64,
                duration_ms=duration_ms,
                sample_rate=chunk.sample_rate,
            )
        )

    return TTSResponse(
        request_id=req.request_id,
        total_chunks=len(results),
        chunks=results,
    )


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(
    engine=Depends(get_engine),
    manager=Depends(get_session_manager),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=engine.loaded,
        active_sessions=manager.active_count(),
        synthesizing_sessions=manager.synthesizing_count(),
    )
