"""
api/ws_handler.py — WebSocket 流式 TTS 处理器（单一职责：WS 协议层）

协议设计（JSON 帧）：
─────────────────────────────────────────────────────────────────────────────
客户端 → 服务端：
  { "type": "text", "request_id": "req-001", "content": "...", "speaker": "Vivian", "language": "Chinese", "instruct": null }
  { "type": "cancel", "request_id": "req-001" }
  { "type": "ping" }

服务端 → 客户端：
  { "type": "session_id",   "session_id": "uuid" }
  { "type": "chunk_start",  "request_id": "req-001", "chunk_index": 0, "total": 5, "text": "..." }
  { "type": "audio",        "request_id": "req-001", "chunk_index": 0, "audio_b64": "...", "duration_ms": 1200, "is_last": false }
  { "type": "done",         "request_id": "req-001", "total_chunks": 5 }
  { "type": "error",        "request_id": "req-001", "message": "..." }
  { "type": "pong" }
─────────────────────────────────────────────────────────────────────────────
"""

import json

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from config import settings
from core.session_manager import SessionState
from dependencies import get_engine, get_session_manager
from services.tts_service import TTSService
from utils.audio_encoder import estimate_duration_ms, pcm_to_base64


async def handle_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    engine = get_engine()
    manager = get_session_manager()
    service = TTSService(engine=engine, settings=settings)

    session = await manager.create_session()
    await _send(websocket, {"type": "session_id", "session_id": session.session_id})
    logger.info(f"[WS] 新连接 session={session.session_id[:8]}…")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            session.touch()

            if msg_type == "ping":
                await _send(websocket, {"type": "pong"})
                continue

            if msg_type == "cancel":
                request_id = msg.get("request_id", "")
                await manager.cancel_session(session.session_id)
                await _send(websocket, {
                    "type": "done", "request_id": request_id, "cancelled": True,
                })
                break

            if msg_type == "text":
                request_id = msg.get("request_id", "")
                text = msg.get("content", "").strip()
                if not text:
                    continue

                speaker = msg.get("speaker", "")
                language = msg.get("language", "")
                instruct = msg.get("instruct", None)

                session.reset_cancel()
                session.state = SessionState.SYNTHESIZING

                chunk_count = 0

                async for chunk in service.synthesize_stream(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    cancel_event=session.cancel_event,
                ):
                    if session.cancel_event.is_set():
                        logger.info(f"[WS] {session.session_id[:8]}… 合成已取消")
                        await _send(websocket, {
                            "type": "done", "request_id": request_id, "cancelled": True,
                        })
                        break

                    audio_b64 = pcm_to_base64(chunk.pcm, chunk.sample_rate)
                    duration_ms = estimate_duration_ms(len(chunk.pcm), chunk.sample_rate)

                    if chunk.is_last:
                        chunk_count = chunk.index + 1

                    total = chunk_count if chunk_count > 0 else 0

                    await _send(websocket, {
                        "type": "chunk_start",
                        "request_id": request_id,
                        "chunk_index": chunk.index,
                        "total": total,
                        "text": chunk.text,
                    })
                    await _send(websocket, {
                        "type": "audio",
                        "request_id": request_id,
                        "chunk_index": chunk.index,
                        "audio_b64": audio_b64,
                        "duration_ms": round(duration_ms),
                        "sample_rate": chunk.sample_rate,
                        "is_last": chunk.is_last,
                    })

                    session.chunk_index = chunk.index
                    session.state = SessionState.STREAMING

                else:
                    session.state = SessionState.IDLE
                    await _send(websocket, {
                        "type": "done",
                        "request_id": request_id,
                        "total_chunks": chunk_count,
                    })

            else:
                await _send(websocket, {
                    "type": "error", "message": f"未知消息类型: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("[WS] 客户端断开连接")
    except Exception as e:
        logger.exception(f"[WS] 处理异常: {e}")
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await manager.close_session(session.session_id)


async def _send(ws: WebSocket, data: dict) -> None:
    await ws.send_text(json.dumps(data, ensure_ascii=False))
