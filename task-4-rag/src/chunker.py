"""PDF 文本 chunking：段落感知、固定上限、可控重叠。"""
from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """按段落尽量拼接文本，并保证每个 chunk 不超过 ``chunk_size``。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    def emit(value: str) -> None:
        value = value.strip()
        if value:
            chunks.append(value[:chunk_size])

    def split_long(value: str) -> None:
        stride = chunk_size - overlap
        start = 0
        while start < len(value):
            emit(value[start:start + chunk_size])
            if start + chunk_size >= len(value):
                break
            start += stride

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                emit(current)
                current = ""
            split_long(paragraph)
            continue

        if not current:
            current = paragraph
            continue

        candidate = current + "\n" + paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        previous = current
        emit(previous)
        tail = previous[-overlap:] if overlap else ""
        candidate = (tail + "\n" + paragraph).strip() if tail else paragraph
        current = candidate if len(candidate) <= chunk_size else paragraph

    if current:
        emit(current)

    # 保留原有的短片段过滤语义，但明确写在实现中。
    return [chunk for chunk in chunks if len(chunk.strip()) >= 20]
