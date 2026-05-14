"""
dependencies.py — 依赖注入容器（FastAPI Depends）

类比 Android Hilt 的 @Module / @Provides：
  所有需要注入的对象集中在这里管理，handler 通过 Depends() 获取。

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
    """获取 TTSEngine 单例。由 lifespan 调用 load() 完成初始化。"""
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


def get_session_manager() -> SessionManager:
    """获取 SessionManager 单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_service() -> TTSService:
    """每次请求创建一个新的 TTSService（无状态，轻量）。"""
    return TTSService(engine=get_engine(), settings=settings)
