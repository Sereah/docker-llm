"""
dependencies.py — 依赖注入 / 单例管理

  - TTSEngine / SessionManager：应用级单例（重型对象，启动时创建）
  - TTSService：请求级实例（轻量编排层，每次请求新建）
"""

from config import settings
from core.tts_engine import TTSEngine
from core.session_manager import SessionManager
from services.tts_service import TTSService

_engine: TTSEngine | None = None
_session_manager: SessionManager | None = None


def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_service() -> TTSService:
    return TTSService(engine=get_engine(), settings=settings)
