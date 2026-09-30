"""只读代码审查 Subagent。"""
from __future__ import annotations

from typing import Dict


class ReviewSubagent:
    def __init__(self, call_model):
        self.call_model = call_model

    def review(self, issue: str, source_content: Dict[str, str], question: str) -> str:
        files = "\n\n".join(
            f"=== {path} ===\n{content[:12000]}" for path, content in source_content.items()
        )
        messages = [
            {"role": "system", "content": "You are a read-only code reviewer. Return concise findings, no code edits."},
            {"role": "user", "content": f"Issue:\n{issue}\n\nQuestion:\n{question}\n\nSource:\n{files}"},
        ]
        return self.call_model(messages, max_new_tokens=512)
