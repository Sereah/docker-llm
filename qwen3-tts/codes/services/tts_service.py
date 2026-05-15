"""
services/tts_service.py — TTS 业务编排（单一职责：协调分块 + 合成）

类比 Android 中的 UseCase / ViewModel：
  接收 raw 参数，编排多个底层调用，返回上层可直接使用的结果。
  handler 只做参数解析和输出格式化，所有业务逻辑都在这里。
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

import numpy as np
from loguru import logger

from config import Settings
from core.tts_engine import TTSEngine
from utils.text_splitter import split_text


@dataclass
class SynthesizedChunk:
    index: int
    text: str
    pcm: np.ndarray
    sample_rate: int
    is_last: bool
    total: int = 0


class TTSService:
    """
    TTS 合成服务。

    职责：
      - 文本分块（短文本不分块，长文本分块）
      - 调用 TTSEngine 逐块合成
      - 流式场景下通过 producer-consumer 管线预取

    生命周期：无状态，每次请求创建新实例。
    """

    def __init__(self, engine: TTSEngine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings

    async def synthesize(
        self,
        text: str,
        speaker: str = "",
        language: str = "",
        instruct: Optional[str] = None,
    ) -> List[SynthesizedChunk]:
        """
        同步式合成（HTTP 用）：收集所有 chunk 后一起返回。
        """
        chunks = split_text(
            text,
            short_max=self._settings.short_text_max_chars,
            chunk_max=self._settings.stream_chunk_max_chars,
        )
        logger.info(
            f"[PROC] synthesize 分块完成 total={len(chunks)} "
            f"text_len={len(text)} short_max={self._settings.short_text_max_chars} "
            f"chunk_max={self._settings.stream_chunk_max_chars}"
        )
        for idx, chunk_text in enumerate(chunks):
            logger.info(
                f"[PROC] synthesize chunk[{idx}/{len(chunks) - 1}] "
                f"len={len(chunk_text)} text={chunk_text}"
            )

        results: List[SynthesizedChunk] = []
        total = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            pcm, sr = await self._engine.synthesize_chunk(
                chunk_text, speaker, language, instruct,
            )
            results.append(
                SynthesizedChunk(
                    index=idx, text=chunk_text, pcm=pcm,
                    sample_rate=sr, is_last=(idx == total - 1),
                    total=total,
                )
            )
        return results

    async def synthesize_stream(
        self,
        text: str,
        speaker: str = "",
        language: str = "",
        instruct: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[SynthesizedChunk]:
        """
        流式合成（WebSocket 用）：
          - 短文本：直接合成 1 块后 yield
          - 长文本：分块 + producer-consumer 预取管线

        Yields:
            SynthesizedChunk 按 block index 顺序。
        """
        chunks = split_text(
            text,
            short_max=self._settings.short_text_max_chars,
            chunk_max=self._settings.stream_chunk_max_chars,
        )
        total = len(chunks)

        logger.info(
            f"[PROC] synthesize_stream 分块完成 total={total} "
            f"text_len={len(text)} short_max={self._settings.short_text_max_chars} "
            f"chunk_max={self._settings.stream_chunk_max_chars}"
        )
        for idx, chunk_text in enumerate(chunks):
            logger.info(
                f"[PROC] synthesize_stream chunk[{idx}/{total - 1}] "
                f"len={len(chunk_text)} text={chunk_text}"
            )

        if total == 1:
            pcm, sr = await self._engine.synthesize_chunk(
                chunks[0], speaker, language, instruct,
            )
            yield SynthesizedChunk(
                index=0, text=chunks[0], pcm=pcm, sample_rate=sr, is_last=True,
                total=1,
            )
            return

        prefetch = self._settings.stream_prefetch
        logger.info(f"[PROC] 流式调度 total={total} prefetch={prefetch}")

        queue: asyncio.Queue[Optional[SynthesizedChunk]] = asyncio.Queue(
            maxsize=prefetch + 1
        )

        async def producer() -> None:
            for idx, chunk_text in enumerate(chunks):
                if cancel_event and cancel_event.is_set():
                    logger.info(f"[PROC] 块 {idx} 收到取消信号")
                    break
                is_last = idx == total - 1
                try:
                    pcm, sr = await self._engine.synthesize_chunk(
                        chunk_text, speaker, language, instruct,
                    )
                    chunk = SynthesizedChunk(
                        index=idx, text=chunk_text, pcm=pcm,
                        sample_rate=sr, is_last=is_last,
                        total=total,
                    )
                    await queue.put(chunk)
                    logger.debug(f"[PROC] 合成完成 {idx}/{total - 1}")
                except Exception as e:
                    logger.error(f"[PROC] 块 {idx} 合成异常: {e}")
                    sr = self._settings.sample_rate
                    silent_pcm = np.zeros(sr // 4, dtype=np.float32)
                    await queue.put(
                        SynthesizedChunk(
                            index=idx, text=chunk_text, pcm=silent_pcm,
                            sample_rate=sr, is_last=is_last,
                            total=total,
                        )
                    )
            await queue.put(None)

        producer_task = asyncio.create_task(producer())

        delivered = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
            delivered += 1
            if cancel_event and cancel_event.is_set():
                producer_task.cancel()
                break

        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

        logger.info(f"[PROC] 流式管线结束，共输出 {delivered} 块")
