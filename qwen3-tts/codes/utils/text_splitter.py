"""
utils/text_splitter.py — 文本分块策略（单一职责：只做文本分割）

统一策略：按句末标点切割，贪心合并短句成段。
  - 短文本（≤ short_text_max_chars）：直接返回原文本，不切割
  - 长文本：按句末标点切分，贪心合并成段（每段 ≤ stream_chunk_max_chars）
"""
import re
from typing import List


_SENTENCE_END_CH = re.compile(r'([。！？!?…])')
_SENTENCE_END_EN = re.compile(r'([.!?]\s+)')
_CLAUSE_BREAK = re.compile(r'([，,；;：:]\s*)')


def _merge_parts(parts: list) -> list:
    """
    re.split 带捕获组时返回 [text, sep, text, sep, ...]。
    将每个 text 与紧跟的 sep 合并，返回句子列表。
    """
    result = []
    i = 0
    while i < len(parts):
        text = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        result.append((text + sep).strip())
        i += 2
    return [r for r in result if r]


def _split_by_sentence(text: str) -> List[str]:
    lines = text.split("\n")
    result: List[str] = []

    for line in lines:
        if not line.strip():
            continue

        parts_ch = _SENTENCE_END_CH.split(line)
        merged_ch = _merge_parts(parts_ch)

        for ch in merged_ch:
            parts_en = _SENTENCE_END_EN.split(ch)
            merged_en = _merge_parts(parts_en)
            result.extend(merged_en)

    return [r for r in result if r.strip()]


def _split_by_clause(text: str) -> List[str]:
    parts = _CLAUSE_BREAK.split(text)
    return _merge_parts(parts)


def _force_split(text: str, max_chars: int) -> List[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def split_text(text: str, short_max: int = 120, chunk_max: int = 200) -> List[str]:
    """
    统一分块入口。

    规则：
      1. 若 len(text) ≤ short_max → 返回 [text]（不切割）
      2. 否则按句末标点切割，贪心合并短句直到达到 chunk_max，超长句子按从句/强制截断。

    Returns:
        分块列表，至少包含 1 个元素。
    """
    if len(text) <= short_max:
        return [text]

    sentences = _split_by_sentence(text)
    result: List[str] = []
    current_buf: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len > chunk_max and current_buf:
            result.append(" ".join(current_buf))
            current_buf = []
            current_len = 0

        if sent_len > chunk_max:
            if current_buf:
                result.append(" ".join(current_buf))
                current_buf = []
                current_len = 0

            clauses = _split_by_clause(sent)
            for clause in clauses:
                if len(clause) <= chunk_max:
                    result.append(clause)
                else:
                    result.extend(_force_split(clause, chunk_max))
        else:
            current_buf.append(sent)
            current_len += sent_len

    if current_buf:
        result.append(" ".join(current_buf))

    return [c for c in result if c]
