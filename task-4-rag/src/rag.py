"""端到端 RAG 流水线：检索 → rerank → 生成。"""
from __future__ import annotations

import re
from typing import Dict

_RETRIEVER = None
_RERANKER = None


def answer(query: str) -> Dict:
    """RAG 主入口，返回与生成上下文一致的 sources。"""
    global _RETRIEVER, _RERANKER
    from src.generator import build_rag_prompt, context_excerpt, generate
    from src.retriever import Retriever
    from src.reranker import Reranker

    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    if _RERANKER is None:
        _RERANKER = Reranker()

    results = _RETRIEVER.retrieve(query, k=20)
    reranked = _RERANKER.rerank(query, results, top_k=5)
    prompt = build_rag_prompt(query, reranked)
    response = generate(prompt, max_new_tokens=384, temperature=0.3)

    sources = []
    for result in reranked:
        raw_text = result.get("text", "")
        page_match = re.search(r"\[PAGE\s+(\d+)\]", raw_text)
        sources.append({
            "text": context_excerpt(raw_text),
            "score": result.get("rerank_score", result.get("score")),
            "source": result.get("source"),
            "page": int(page_match.group(1)) if page_match else None,
        })
    return {"answer": response, "sources": sources}
