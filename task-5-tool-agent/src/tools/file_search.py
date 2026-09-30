"""本地文件名/内容搜索工具。"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在指定目录中按文件名模式或内容关键词搜索文件",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索关键词（匹配文件名或内容）"},
                "dir": {"type": "string", "description": "搜索目录"},
            },
            "required": ["pattern", "dir"],
        },
    },
}


def _resolve_dir(raw_dir: str) -> Path:
    """解析目录路径；目录授权边界由上层任务单独控制。"""
    p = Path(raw_dir).expanduser()
    if p.is_absolute() and p.is_dir():
        return p.resolve()
    for candidate in ((ROOT / raw_dir), (ROOT.parent / raw_dir), (ROOT / "data" / raw_dir)):
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    return p.resolve() if p.exists() else (ROOT / raw_dir).resolve()


def _filename_match(pattern: str, filename: str) -> bool:
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(filename.lower(), pattern.lower())
    return pattern.lower() in filename.lower()


def run(args: dict) -> str:
    try:
        pattern = str(args["pattern"])
        search_dir = _resolve_dir(str(args.get("dir", ROOT / "data")))
        if not search_dir.is_dir():
            return f"Error: 目录不存在: {search_dir}"

        matches = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for filename in files:
                filepath = Path(root) / filename
                relative = filepath.relative_to(search_dir)
                display = str(relative)
                if _filename_match(pattern, filename):
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        content = ""
                    preview = content[:500].replace("\x00", " ")
                    matches.append(
                        f"[文件名匹配] {display}\n内容片段:\n{preview}"
                    )
                    continue

                if filepath.suffix.lower() not in {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv"}:
                    continue
                try:
                    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue
                for line_number, line in enumerate(lines, 1):
                    if pattern in line:
                        start = max(0, line_number - 2)
                        end = min(len(lines), line_number + 1)
                        context = "\n".join(
                            f"{idx + 1}: {lines[idx]}" for idx in range(start, end)
                        )
                        matches.append(
                            f"[内容匹配] {display}:{line_number}\n内容片段:\n{context}"
                        )
                        break

        if not matches:
            return f"在 {search_dir} 中未找到匹配 '{pattern}' 的文件"
        return f"匹配文件数: {len(matches)}\n" + "\n\n".join(matches[:20])
    except Exception as exc:
        return f"Error: {exc}"
