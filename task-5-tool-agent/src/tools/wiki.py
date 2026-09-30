"""维基百科查询工具。"""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "查询维基百科获取百科知识",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}


def _query_candidates(query: str):
    import re
    query = str(query).strip()
    candidates = [query]
    cleaned = re.sub(r"[「」『』？?。！!，,：:、]", " ", query)
    cleaned = re.sub(r"^(查维基百科|查一下|查一查|查询|查|请查询|请查|在维基百科里|在维基百科中|维基百科里|维基百科中|维基百科)\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for suffix in ["出生年份", "发表年份", "发明者是谁", "的发明者是谁", "出生年份是多少", "是哪一年发表的", "的发表年份"]:
        if suffix in cleaned:
            cleaned = cleaned.replace(suffix, "").strip()
    cleaned = re.sub(r"的$", "", cleaned).strip()
    aliases = {
        "图灵机": ["Turing machine"],
        "Transformer": ["Transformer (machine learning model)", "Transformer model"],
        "Transformer 机器学习模型": ["Transformer (machine learning model)"],
        "Geoffrey Hinton": ["Geoffrey Hinton"],
        "Transformer (机器学习模型)": ["Transformer (machine learning model)", "Transformer"],
    }
    candidates += aliases.get(cleaned, [])
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)
    # 去重并保留顺序
    return list(dict.fromkeys(candidates))


def run(args: dict) -> str:
    import wikipediaapi
    query = str(args["query"]).strip()
    candidates = _query_candidates(query)

    for lang in ["zh", "en"]:
        try:
            wiki = wikipediaapi.Wikipedia(
                user_agent="LLMBeginnerAgent/1.0 (research; contact@example.com)",
                language=lang, timeout=3.0, max_retries=0,
            )
            for candidate in candidates:
                page = wiki.page(candidate)
                if page.exists():
                    summary = page.summary[:600]
                    return f"[{lang.upper()}] {page.title}\n\n{summary}"
        except wikipediaapi.WikipediaException:
            raise
        except Exception:
            raise

    return f"未在维基百科中找到 '{query}' 的条目。请尝试其他关键词。"
