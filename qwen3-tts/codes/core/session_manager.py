"""
core/session_manager.py — WebSocket 会话管理（单一职责：会话生命周期）

应用级单例，通过 dependencies.py 管理生命周期。
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from loguru import logger


class SessionState(str, Enum):
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    STREAMING = "streaming"
    CLOSED = "closed"


@dataclass
class Session:
    session_id: str
    state: SessionState = SessionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    chunk_index: int = 0
    total_chunks: int = 0

    def touch(self) -> None:
        self.last_active = time.time()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.state = SessionState.CLOSED

    def reset_cancel(self) -> None:
        self.cancel_event.clear()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Session:
        session = Session(session_id=str(uuid.uuid4()))
        async with self._lock:
            self._sessions[session.session_id] = session
        logger.debug(f"[Session] 创建 {session.session_id[:8]}…")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.state = SessionState.CLOSED
            logger.debug(f"[Session] 关闭 {session_id[:8]}…")

    async def cancel_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.cancel()
            logger.info(f"[Session] 取消 {session_id[:8]}…")

    def active_count(self) -> int:
        return sum(
            1
            for s in self._sessions.values()
            if s.state not in (SessionState.CLOSED,)
        )

    def synthesizing_count(self) -> int:
        return sum(
            1
            for s in self._sessions.values()
            if s.state == SessionState.SYNTHESIZING
        )
