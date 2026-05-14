"""
main.py — 服务入口（单一职责：启动 Uvicorn）

用法：
  python main.py
  或
  uvicorn main:app --host 0.0.0.0 --port 8000

环境变量（可通过 .env 文件或系统环境变量设置，前缀 TTS_）：
  TTS_MODEL_NAME=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
  TTS_DEVICE=cuda
  TTS_PORT=8000
  TTS_MAX_CONCURRENT_SESSIONS=4
  TTS_SHORT_TEXT_MAX_CHARS=120
  TTS_STREAM_CHUNK_MAX_CHARS=200
  TTS_STREAM_PREFETCH=3
"""

import uvicorn
from loguru import logger

from api.app import create_app
from config import settings

# ── 供 uvicorn 直接引用（uvicorn main:app）─────────────────────────────────────
app = create_app()


if __name__ == "__main__":
    logger.info(
        f"启动 Qwen3-TTS 服务  host={settings.host}:{settings.port}  "
        f"model={settings.model_name}  device={settings.device}"
    )
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,   # GPU 服务建议 workers=1，避免多进程争抢显存
        log_level=settings.log_level,
        # WebSocket 需要 reload=False（reload 与 workers>1 不兼容）
        reload=False,
    )
