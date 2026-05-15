"""
api/server.py — 原生 HTTP + WebSocket 混合服务器（单端口，零框架依赖）

架构：
  - 同一端口同时处理 HTTP 和 WebSocket
  - TCP 调度层：解析 HTTP 请求行/头部，按 Upgrade 头分发
  - HTTP 路由：GET /api/health → 健康检查  POST /api/tts → TTS 合成
  - WebSocket：纯 websockets 库，协议与原 ws_handler 完全兼容

依赖链：main.py → server.py → tts_service.py → tts_engine.py

日志关键字规范（方便 grep 过滤）：
  [RECV]  接收到的（客户端 → 服务端）
  [SEND]  下发的（服务端 → 客户端）
  [PROC]  内部处理的
"""

import asyncio
import json
import time

import websockets
from loguru import logger

from config import settings
from core.session_manager import SessionState
from dependencies import get_engine, get_session_manager
from services.tts_service import TTSService
from utils.audio_encoder import estimate_duration_ms, pcm_to_base64


def _log_truncate(s: str, keep: int = 40) -> str:
    if len(s) <= keep:
        return s
    return s[:keep] + f"...({len(s)}字)"


# ── WebSocket Server Mock ──────────────────────────────────────────────────


class _MockWSServer(asyncio.Server):
    def __init__(self):
        self.websockets = set()
        self._serving = True

    def register(self, protocol):
        self.websockets.add(protocol)

    def unregister(self, protocol):
        self.websockets.discard(protocol)

    def is_serving(self):
        return self._serving


_ws_server = _MockWSServer()


# ── HTTP 路由处理 ──────────────────────────────────────────────────────────


async def _handle_http(path: str, method: str, body: bytes) -> tuple[int, dict, str]:
    logger.debug(f"[PROC] _handle_http {method} {path} body_len={len(body)}")

    if path == "/api/health" and method == "GET":
        engine = get_engine()
        manager = get_session_manager()
        data = {
            "status": "ok",
            "model_loaded": engine.loaded,
            "active_sessions": manager.active_count(),
            "synthesizing_sessions": manager.synthesizing_count(),
        }
        logger.info(f"[PROC] health model_loaded={engine.loaded} active={manager.active_count()}")
        return 200, {"Content-Type": "application/json"}, json.dumps(data, ensure_ascii=False)

    if path == "/api/tts" and method == "POST":
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("[RECV] POST /api/tts body 不是合法 JSON")
            return 400, {"Content-Type": "application/json"}, json.dumps({"error": "Invalid JSON"})

        text = req.get("text", "").strip()
        if not text:
            logger.warning("[RECV] POST /api/tts text 为空")
            return 400, {"Content-Type": "application/json"}, json.dumps({"error": "文本为空"})

        request_id = req.get("request_id", "")
        speaker = req.get("speaker", "")
        language = req.get("language", "")
        instruct = req.get("instruct")
        text_len = len(text)

        logger.info(
            f"[RECV] POST /api/tts request_id={request_id} "
            f"text_len={text_len} speaker={speaker} language={language}"
        )
        logger.info(
            f"[RECV] POST /api/tts 完整请求 request_id={request_id} "
            f"json={json.dumps(req, ensure_ascii=False)}"
        )

        engine = get_engine()
        service = TTSService(engine=engine, settings=settings)

        t0 = time.time()
        try:
            chunks = await service.synthesize(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
            )
        except Exception as e:
            logger.exception(f"[PROC] 合成失败 request_id={request_id}")
            return 500, {"Content-Type": "application/json"}, json.dumps({"error": str(e)})

        elapsed = time.time() - t0
        logger.info(f"[PROC] 合成完成 request_id={request_id} chunks={len(chunks)} elapsed={elapsed:.2f}s")

        results = []
        total_audio_bytes = 0
        for chunk in chunks:
            audio_b64 = pcm_to_base64(chunk.pcm, chunk.sample_rate)
            duration_ms = int(estimate_duration_ms(len(chunk.pcm), chunk.sample_rate))
            total_audio_bytes += len(audio_b64)
            chunk_item = {
                "index": chunk.index,
                "text": chunk.text,
                "audio_b64": audio_b64,
                "duration_ms": duration_ms,
                "sample_rate": chunk.sample_rate,
            }
            results.append(chunk_item)
            logger.info(
                f"[SEND] chunk request_id={request_id} "
                f"index={chunk.index}/{len(chunks)} "
                f"text_len={len(chunk.text)} dur={duration_ms}ms "
                f"audio_b64_len={len(audio_b64)} is_last={chunk.is_last} "
                f"text={chunk.text}"
            )

        resp = {
            "request_id": request_id,
            "total_chunks": len(results),
            "chunks": results,
        }
        resp_json = json.dumps(resp, ensure_ascii=False)

        resp_log = {
            "request_id": request_id,
            "total_chunks": len(results),
            "chunks": [
                {
                    "index": c["index"],
                    "text": c["text"],
                    "audio_b64": _log_truncate(c["audio_b64"]),
                    "duration_ms": c["duration_ms"],
                    "sample_rate": c["sample_rate"],
                }
                for c in results
            ],
        }
        logger.info(
            f"[SEND] 完整响应 request_id={request_id} "
            f"chunks={len(results)} resp_size={len(resp_json)} "
            f"audio_total={total_audio_bytes} "
            f"json={json.dumps(resp_log, ensure_ascii=False)}"
        )
        return 200, {"Content-Type": "application/json"}, resp_json

    logger.warning(f"[RECV] 未知路由 {method} {path}")
    return 404, {"Content-Type": "application/json"}, json.dumps({"error": "Not found"})


# ── TCP 调度器 ─────────────────────────────────────────────────────────────


async def _dispatch(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    peer_str = f"{peer[0]}:{peer[1]}" if peer else "unknown"
    logger.debug(f"[PROC] dispatch 新连接 {peer_str}")

    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not request_line:
            logger.debug(f"[PROC] dispatch {peer_str} 空请求关闭")
            writer.close()
            return

        line = request_line.decode("utf-8", errors="replace").strip()
        parts = line.split(" ")
        if len(parts) < 2:
            logger.warning(f"[PROC] dispatch {peer_str} 非法请求行: {line}")
            writer.close()
            return

        method = parts[0]
        path = parts[1]

        headers: dict[str, str] = {}
        content_length = 0
        is_websocket = False

        while True:
            header_line = await asyncio.wait_for(reader.readline(), timeout=5)
            if not header_line or header_line == b"\r\n":
                break
            hdr = header_line.decode("utf-8", errors="replace").strip()
            if ":" in hdr:
                key, val = hdr.split(":", 1)
                key_lower = key.strip().lower()
                val = val.strip()
                headers[key_lower] = val
                if key_lower == "content-length":
                    content_length = int(val)
                if key_lower == "upgrade" and val.lower() == "websocket":
                    is_websocket = True

        if is_websocket:
            logger.info(f"[PROC] dispatch {method} {path} → WebSocket ({peer_str})")
            await _handle_ws_upgrade(reader, writer, request_line, headers, path)
        else:
            logger.info(f"[PROC] dispatch {method} {path} → HTTP content_length={content_length} ({peer_str})")
            body = b""
            if content_length > 0:
                if content_length > 10 * 1024 * 1024:
                    logger.warning(f"[PROC] dispatch {peer_str} body 过大 ({content_length} bytes)")
                    writer.write(b"HTTP/1.1 413 Payload Too Large\r\n\r\n")
                    await writer.drain()
                    writer.close()
                    return
                body = await asyncio.wait_for(reader.readexactly(content_length), timeout=30)

            status, resp_headers, resp_body = await _handle_http(path, method, body)
            resp_body_bytes = resp_body.encode("utf-8")

            status_text_map = {
                200: "OK", 400: "Bad Request", 404: "Not Found",
                413: "Payload Too Large", 500: "Internal Server Error",
            }
            status_text = status_text_map.get(status, "Unknown")
            writer.write(f"HTTP/1.1 {status} {status_text}\r\n".encode())
            for k, v in resp_headers.items():
                writer.write(f"{k}: {v}\r\n".encode())
            writer.write(f"Content-Length: {len(resp_body_bytes)}\r\n".encode())
            writer.write(b"\r\n")
            writer.write(resp_body_bytes)
            await writer.drain()
            logger.info(f"[PROC] dispatch {method} {path} → {status} resp_body={len(resp_body_bytes)}")

    except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError):
        logger.debug(f"[PROC] dispatch {peer_str} 连接中断/超时")
    except Exception as e:
        logger.exception(f"[PROC] dispatch {peer_str} 异常: {e}")
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle_ws_upgrade(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_line: bytes,
    headers: dict[str, str],
    path: str,
) -> None:
    from websockets.legacy.server import WebSocketServerProtocol

    transport = writer.transport

    logger.debug(f"[PROC] ws_upgrade 创建协议 path={path}")
    protocol = WebSocketServerProtocol(
        _handle_ws,
        _ws_server,
        max_size=2 * 1024 * 1024,
        ping_interval=settings.ws_ping_interval,
        ping_timeout=settings.ws_ping_timeout,
    )

    protocol.connection_made(transport)

    raw_data = request_line
    header_block = b""
    for key, val in headers.items():
        header_block += f"{key}: {val}\r\n".encode()
    raw_data += header_block + b"\r\n"

    if hasattr(reader, '_buffer'):
        raw_data += bytes(reader._buffer)
        reader._buffer.clear()

    protocol.data_received(raw_data)
    logger.debug(f"[PROC] ws_upgrade 已注入 {len(raw_data)} bytes")

    try:
        while True:
            data = await asyncio.wait_for(reader.read(65536), timeout=300)
            if not data:
                logger.debug(f"[PROC] ws_upgrade 对端关闭")
                break
            protocol.data_received(data)
    except asyncio.TimeoutError:
        logger.debug(f"[PROC] ws_upgrade 超时关闭")
    except (ConnectionResetError, BrokenPipeError):
        logger.debug(f"[PROC] ws_upgrade 连接中断")
    except Exception:
        pass

    try:
        protocol.connection_lost(None)
    except Exception:
        pass


# ── WebSocket 处理器 ───────────────────────────────────────────────────────


async def _send(websocket, data: dict) -> None:
    await websocket.send(json.dumps(data, ensure_ascii=False))


async def _handle_ws(websocket, path=None) -> None:
    engine = get_engine()
    manager = get_session_manager()
    service = TTSService(engine=engine, settings=settings)

    session = await manager.create_session()
    sid = session.session_id[:8]
    await _send(websocket, {"type": "session_id", "session_id": session.session_id})
    logger.info(f"[PROC] {sid}… 新连接 active={manager.active_count()}")

    try:
        async for raw in websocket:
            logger.debug(f"[RECV] {sid}… 消息 len={len(raw)}")

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"[RECV] {sid}… 非法 JSON")
                continue

            msg_type = msg.get("type")
            logger.debug(f"[RECV] {sid}… type={msg_type}")
            session.touch()

            if msg_type == "ping":
                logger.debug(f"[RECV] {sid}… ping")
                await _send(websocket, {"type": "pong"})
                continue

            if msg_type == "cancel":
                request_id = msg.get("request_id", "")
                logger.info(f"[RECV] {sid}… cancel request_id={request_id}")
                await manager.cancel_session(session.session_id)
                await _send(websocket, {
                    "type": "done", "request_id": request_id, "cancelled": True,
                })
                break

            if msg_type == "text":
                request_id = msg.get("request_id", "")
                text = msg.get("content", "").strip()
                if not text:
                    logger.warning(f"[RECV] {sid}… text 内容为空")
                    continue

                speaker = msg.get("speaker", "")
                language = msg.get("language", "")
                instruct = msg.get("instruct", None)
                text_len = len(text)

                logger.info(
                    f"[RECV] {sid}… text request_id={request_id} "
                    f"len={text_len} speaker={speaker} language={language}"
                )
                logger.info(
                    f"[RECV] {sid}… 完整 text 消息 request_id={request_id} "
                    f"json={json.dumps(msg, ensure_ascii=False)}"
                )

                session.reset_cancel()
                session.state = SessionState.SYNTHESIZING

                t0 = time.time()

                async for chunk in service.synthesize_stream(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    cancel_event=session.cancel_event,
                ):
                    if session.cancel_event.is_set():
                        logger.info(f"[PROC] {sid}… 合成已取消 request_id={request_id}")
                        await _send(websocket, {
                            "type": "done", "request_id": request_id, "cancelled": True,
                        })
                        break

                    audio_b64 = pcm_to_base64(chunk.pcm, chunk.sample_rate)
                    duration_ms = estimate_duration_ms(len(chunk.pcm), chunk.sample_rate)

                    total = chunk.total

                    chunk_start_msg = {
                        "type": "chunk_start",
                        "request_id": request_id,
                        "chunk_index": chunk.index,
                        "total": total,
                        "text": chunk.text,
                    }
                    audio_msg = {
                        "type": "audio",
                        "request_id": request_id,
                        "chunk_index": chunk.index,
                        "audio_b64": audio_b64,
                        "duration_ms": round(duration_ms),
                        "sample_rate": chunk.sample_rate,
                        "is_last": chunk.is_last,
                    }

                    logger.info(
                        f"[SEND] {sid}… chunk_start request_id={request_id} "
                        f"chunk_index={chunk.index}"
                        f"{'/' + str(total - 1) if total > 0 else ''} "
                        f"text_len={len(chunk.text)} "
                        f"json={json.dumps(chunk_start_msg, ensure_ascii=False)}"
                    )
                    logger.info(
                        f"[SEND] {sid}… audio request_id={request_id} "
                        f"chunk_index={chunk.index} "
                        f"audio_b64_len={len(audio_b64)} "
                        f"dur={round(duration_ms)}ms "
                        f"is_last={chunk.is_last} "
                        f"json={json.dumps({k: v for k, v in audio_msg.items() if k != 'audio_b64'}, ensure_ascii=False)}"
                    )

                    await _send(websocket, chunk_start_msg)
                    await _send(websocket, audio_msg)

                    session.chunk_index = chunk.index
                    session.state = SessionState.STREAMING

                else:
                    session.state = SessionState.IDLE
                    elapsed = time.time() - t0
                    logger.info(
                        f"[SEND] {sid}… done request_id={request_id} "
                        f"chunks={chunk.total} elapsed={elapsed:.2f}s"
                    )
                    await _send(websocket, {
                        "type": "done",
                        "request_id": request_id,
                        "total_chunks": chunk.total,
                    })

            else:
                logger.warning(f"[RECV] {sid}… 未知消息类型: {msg_type}")
                await _send(websocket, {
                    "type": "error", "message": f"未知消息类型: {msg_type}",
                })

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[PROC] {sid}… 客户端断开")
    except Exception as e:
        logger.exception(f"[PROC] {sid}… 处理异常: {e}")
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await manager.close_session(session.session_id)
        logger.info(f"[PROC] {sid}… 会话关闭 active={manager.active_count()}")


async def start_server(host: str, port: int) -> None:
    logger.info(f"TTS Server (HTTP + WebSocket) starting on [{host}]:{port}")
    logger.info(f"  HTTP health:  GET  http://{host}:{port}/api/health")
    logger.info(f"  HTTP TTS:     POST http://{host}:{port}/api/tts")
    logger.info(f"  WebSocket:    ws://{host}:{port}")

    server = await asyncio.start_server(_dispatch, host, port)
    logger.info("✅ Server ready. Waiting for connections...")

    async with server:
        await server.serve_forever()
