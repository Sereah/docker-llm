"""
main.py — 服务入口

用法：
  python main.py

环境变量（可通过 .env 文件或系统环境变量设置，前缀 TTS_）：
  TTS_MODEL_NAME=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
  TTS_DEVICE=cuda
  TTS_PORT=8000
  TTS_MAX_CONCURRENT_SESSIONS=4
"""

import asyncio
import sys

from loguru import logger

from config import settings
from dependencies import get_engine
from api.server import start_server


def main() -> None:
    logger.info(
        f"Qwen3-TTS 服务\n"
        f"  HTTP + WebSocket: [{settings.host}]:{settings.port}\n"
        f"  模型: {settings.model_name}\n"
        f"  设备: {settings.device}"
    )

    logger.info("加载 TTS 模型...")
    engine = get_engine()
    engine.load()
    logger.info("模型加载完成")

    try:
        asyncio.run(start_server(settings.host, settings.port))
    except KeyboardInterrupt:
        logger.info("服务已停止")
    except Exception as e:
        logger.exception(f"服务异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
