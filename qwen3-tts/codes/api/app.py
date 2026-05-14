"""
api/app.py — FastAPI 应用工厂（单一职责：组装路由 + 生命周期）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.http_handler import router as http_router
from api.ws_handler import handle_websocket
from dependencies import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动，开始加载 TTS 模型…")
    engine = get_engine()
    engine.load()
    logger.info("TTS 模型加载完毕，服务就绪")
    yield
    logger.info("服务关闭，清理资源…")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Qwen3-TTS Service",
        description="基于 Qwen3-TTS 的语音合成服务，支持 HTTP 和 WebSocket 两种接入方式。",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(http_router)

    @app.websocket("/ws/tts")
    async def ws_tts(websocket):
        await handle_websocket(websocket)

    return app
