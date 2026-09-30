"""读取带 YAML front-matter 的 Skill。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
        if not match:
            return {}, text.strip()
        raw, body = match.groups()
        try:
            import yaml  # type: ignore
            meta = yaml.safe_load(raw) or {}
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            # PyYAML 不可用时仍支持本项目使用的简单标量 front-matter。
            meta = {}
            for line in raw.splitlines():
                if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                meta[key.strip()] = value
        return {str(k): v for k, v in meta.items()}, body.strip()

    def _iter_skill_files(self):
        if not self.skills_dir.is_dir():
            return
        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            if skill_file.is_file():
                yield skill_file

    def list_skills(self) -> List[Dict[str, str]]:
        skills: List[Dict[str, str]] = []
        for skill_file in self._iter_skill_files() or ():
            meta, body = self._parse_front_matter(skill_file.read_text(encoding="utf-8"))
            name = str(meta.get("name") or skill_file.parent.name)
            description = str(meta.get("description") or "")
            self._cache[name] = {
                "name": name,
                "description": description,
                "meta": meta,
                "body": body,
                "path": skill_file,
            }
            skills.append({"name": name, "description": description})
        return skills

    def load(self, name: str) -> Optional[str]:
        if name not in self._cache:
            self.list_skills()
        entry = self._cache.get(name)
        return None if entry is None else str(entry["body"])

    def load_relevant(self, names: Optional[List[str]] = None) -> Dict[str, str]:
        """加载指定 Skill；不传 names 时加载全部可用 Skill。"""
        self.list_skills()
        selected = names or list(self._cache)
        return {name: body for name in selected if (body := self.load(name)) is not None}
