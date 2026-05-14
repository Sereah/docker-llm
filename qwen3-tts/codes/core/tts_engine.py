"""
core/tts_engine.py — TTS 推理引擎（单一职责：模型加载 + 推理）

架构说明：
  - 应用级单例，通过 dependencies.py 管理生命周期
  - asyncio.Semaphore 控制并发推理数，防止显存 OOM
  - synthesize_chunk() 为核心推理方法，接收单块文本，返回 PCM array
  - synthesize_stream() 为异步生成器，驱动流式合成管线

Qwen3-TTS 调用方式参考：
  https://github.com/QwenLM/Qwen3-TTS
  使用 qwen_tts 包的 Qwen3TTSModel 加载，generate_custom_voice() 推理
  模型输出为 (wavs, sample_rate)，wavs 为 list of float32 numpy array
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from config import settings


class TTSEngine:
    """
    Qwen3-TTS 推理引擎封装。

    线程安全说明：
      synthesize_chunk 内部运行在 executor（线程池），不阻塞 event loop。
      Semaphore 保证同时只有 max_concurrent_sessions 个推理任务并行。
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Semaphore(settings.max_concurrent_sessions)
        self._loaded = False

    def load(self) -> None:
        """同步加载模型（在启动事件中调用一次）。"""
        try:
            from qwen_tts import Qwen3TTSModel

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(settings.torch_dtype, torch.bfloat16)

            logger.info(f"正在加载模型 {settings.model_name}，device={settings.device}，dtype={settings.torch_dtype}")
            t0 = time.time()

            load_kwargs = {
                "pretrained_model_name_or_path": settings.model_name,
                "device_map": settings.device,
                "dtype": torch_dtype,
                "attn_implementation": settings.attn_implementation,
            }

            self._model = Qwen3TTSModel.from_pretrained(**load_kwargs)
            self._loaded = True
            logger.info(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

        except ImportError as e:
            logger.error(f"依赖未安装: {e}")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _infer(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        同步推理（在线程池中执行）。

        Returns:
            (pcm_array, sample_rate)
            pcm_array: float32 numpy array，值域 [-1.0, 1.0]
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        audio = wavs[0] if isinstance(wavs, list) else wavs
        if hasattr(audio, "cpu"):
            audio = audio.float().cpu().numpy()
        return audio, sr

    async def synthesize_chunk(
        self,
        text: str,
        speaker: str = "",
        language: str = "",
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        异步包装：通过 run_in_executor 避免阻塞 event loop。
        Semaphore 限制并发推理数。
        """
        _speaker = speaker or settings.speaker
        _language = language or settings.language

        async with self._lock:
            loop = asyncio.get_running_loop()
            pcm, sr = await loop.run_in_executor(
                None,
                lambda: self._infer(text, _speaker, _language, instruct),
            )
        return pcm, sr

    async def synthesize_stream(
        self,
        chunks: list[str],
        speaker: str = "",
        language: str = "",
        instruct: Optional[str] = None,
        prefetch: int = 1,
    ) -> AsyncIterator[Tuple[int, np.ndarray, int, bool]]:
        """
        流式合成异步生成器，支持预取以填满播放缓冲区。

        Yields:
            (chunk_index, pcm_array, sample_rate, is_last)
        """
        total = len(chunks)
        _speaker = speaker or settings.speaker
        _language = language or settings.language

        queue: asyncio.Queue[Tuple[int, np.ndarray, int, bool]] = asyncio.Queue(
            maxsize=prefetch + 1
        )

        async def producer():
            for idx, text in enumerate(chunks):
                is_last = idx == total - 1
                try:
                    pcm, sr = await self.synthesize_chunk(text, _speaker, _language, instruct)
                    await queue.put((idx, pcm, sr, is_last))
                except Exception as e:
                    logger.error(f"合成第 {idx} 块失败: {e}")
                    await queue.put((idx, np.zeros(sr // 10, dtype=np.float32), sr, is_last))

        producer_task = asyncio.create_task(producer())

        delivered = 0
        while delivered < total:
            item = await queue.get()
            yield item
            delivered += 1

        await producer_task
