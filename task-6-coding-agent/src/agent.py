"""基于 MCP 工具循环的 Mini Coding Agent。"""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mcp_server import StdioMCPClient
from src.skill_loader import SkillLoader
from src.subagents import ReviewSubagent

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Qwen2.5-Coder-7B-Instruct"


class CodingAgent:
    """通过 model → tool → observation 循环修复仓库中的代码问题。"""

    def __init__(self, max_steps: int = 12, client_factory=None):
        self.max_steps = max_steps
        self._model = None
        self._tokenizer = None
        self.skills = SkillLoader(str(ROOT / "src" / "skills"))
        self.client_factory = client_factory or (lambda: StdioMCPClient(str(ROOT)))

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("加载 Qwen2.5-Coder-7B-Instruct (FP16)...")
        self._tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            device_map="auto",
            max_memory={0: "20GiB", "cpu": "64GiB"},
            torch_dtype=torch.float16,
        )
        self._model.eval()
        print("模型加载完成")

    def _call(self, messages: list, max_new_tokens: int = 384) -> str:
        import torch

        self._load_model()
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 192),
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """从模型输出提取一个完整 JSON 对象，支持代码块和嵌套字符串。"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        decoder = json.JSONDecoder()
        candidates = [cleaned]
        for match in re.finditer(r"\{", cleaned):
            candidates.append(cleaned[match.start():])
        for candidate in candidates:
            try:
                value, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @classmethod
    def _parse_action(cls, text: str) -> Dict[str, Any]:
        action = cls._extract_json(text)
        if action is not None:
            if "action" not in action:
                if "tool" in action:
                    action["action"] = "tool_call"
                elif "final" in action or "answer" in action:
                    action["action"] = "final"
            return action
        if re.search(r"final\s+answer", text, re.I):
            return {"action": "final", "summary": text.strip()}
        return {"action": "invalid", "error": "模型输出不是有效 JSON action", "raw": text}

    @staticmethod
    def _text_from_result(result: Dict[str, Any]) -> str:
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
        return "\n".join(item.get("text", "") for item in result.get("content", []))

    @staticmethod
    def _test_passed(result: Dict[str, Any]) -> bool:
        data = result.get("structuredContent") or {}
        return data.get("returncode") == 0 and not data.get("timed_out", False)

    @staticmethod
    def _is_test_path(path: str) -> bool:
        parts = Path(path).parts
        return any(part == "tests" for part in parts) or Path(path).name.startswith("test_")

    @staticmethod
    def _normalize_tool_arguments(tool: str, arguments: Dict[str, Any], repo_path: str) -> Dict[str, Any]:
        normalized = dict(arguments)
        if "file_path" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("file_path")
        if tool in {"read_file", "write_file"} and "path" not in normalized:
            normalized["path"] = normalized.get("filename", "")
        if tool in {"read_file", "write_file", "run_tests", "git_diff", "git_apply", "list_files"}:
            # 工作仓库由 run() 指定；模型常把文件名错填进 repo_path。
            normalized["repo_path"] = repo_path
        return normalized

    def _system_prompt(self, tools: List[Dict[str, Any]], skills: Dict[str, str]) -> str:
        tool_names = [t.get("name", "") for t in tools]
        skill_text = "\n\n".join(f"## Skill: {name}\n{body}" for name, body in skills.items())
        return f"""You are a coding agent. Work only through the available MCP tools.

Available tools: {', '.join(tool_names)}

Every response MUST be exactly one JSON object and nothing else.
Include a one-sentence decision summary in the "thought" field of every response.
For a tool call use:
{{"thought":"Read the source relevant to the issue.","action":"tool_call","tool":"read_file","arguments":{{...}}}}
For a final response use:
{{"thought":"Tests pass.","action":"final","summary":"..."}}
For a read-only reviewer use:
{{"thought":"Need an independent review.","action":"delegate","role":"reviewer","question":"..."}}

Rules:
- A workspace snapshot is included in the user message. For a simple issue, use it directly instead of repeatedly delegating or rereading the same file.
- First identify the actual target file, then make the smallest correct change.
- For a direct code fix, call write_file with path, content, and repo_path, then call run_tests.
- Never modify tests.
- Select the actual target file; do not assume the first source file is the target.
- After editing, run tests and use the test observation to decide whether to continue.
- Do not invent tool observations or claim success before tests pass.

{skill_text}"""

    def _delegate_reviewer(self, issue: str, source_content: Dict[str, str], question: str) -> str:
        reviewer = ReviewSubagent(self._call)
        return reviewer.review(issue, source_content, question)

    def _make_patch(self, repo_path: str, snapshots: Dict[str, str], client: StdioMCPClient) -> str:
        diffs: List[str] = []
        for path, original in snapshots.items():
            result = client.call_tool("read_file", {"path": path, "repo_path": repo_path})
            current_data = result.get("structuredContent") or {}
            current = current_data.get("content")
            if current is None:
                continue
            if current != original:
                diff = difflib.unified_diff(
                    original.splitlines(), current.splitlines(),
                    fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
                )
                diffs.append("\n".join(diff))
        return "\n".join(diffs)

    def run(self, repo_path: str, issue: str) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        snapshots: Dict[str, str] = {}
        tests_passed = False
        final_summary = ""
        final_status = "max_steps_exceeded"
        source_content: Dict[str, str] = {}
        last_test_result: Dict[str, Any] | None = None

        with self.client_factory() as client:
            started = time.perf_counter()
            tools = client.list_tools()
            steps.append({
                "step": 0,
                "type": "mcp_handshake",
                "tools": [tool.get("name") for tool in tools],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            })

            # 初始目录观察只用于给模型提供工作区地图，之后所有动作仍由模型选择。
            result = client.call_tool("list_files", {"repo_path": repo_path})
            steps.append({"step": 1, "tool": "list_files", "arguments": {"repo_path": repo_path}, "result": result})
            file_data = result.get("structuredContent") or {}
            files = file_data.get("files", [])
            for path in files:
                if not path.endswith(".py") or self._is_test_path(path):
                    continue
                read_result = client.call_tool("read_file", {"path": path, "repo_path": repo_path})
                data = read_result.get("structuredContent") or {}
                content = data.get("content")
                if isinstance(content, str):
                    source_content[path] = content
                    snapshots[path] = content
                    steps.append({"step": len(steps), "tool": "read_file", "arguments": {"path": path, "repo_path": repo_path}, "result": read_result})

            skill_map = self.skills.load_relevant()
            messages: List[Dict[str, str]] = [{
                "role": "system",
                "content": self._system_prompt(tools, skill_map),
            }, {
                "role": "user",
                "content": (
                    f"Issue:\n{issue}\n\nWorkspace files:\n{json.dumps(files, ensure_ascii=False)}\n\n"
                    "Relevant source snapshot:\n" +
                    "\n\n".join(f"=== {p} ===\n{c[:12000]}" for p, c in source_content.items())
                ),
            }]

            for loop_step in range(self.max_steps):
                call_started = time.perf_counter()
                response = self._call(messages, max_new_tokens=384)
                action = self._parse_action(response)
                step_record: Dict[str, Any] = {
                    "step": len(steps),
                    "type": "model_decision",
                    "response": response,
                    "action": action,
                    "thought": action.get("thought"),
                    "elapsed_ms": round((time.perf_counter() - call_started) * 1000, 2),
                }
                steps.append(step_record)
                messages.append({"role": "assistant", "content": response})

                kind = action.get("action")
                if kind in {item.get("name") for item in tools} and isinstance(action.get("arguments", {}), dict):
                    # 模型有时把工具名直接放在 action 字段，并省略无参工具的 arguments。
                    action = {**action, "action": "tool_call", "tool": kind,
                              "arguments": action.get("arguments", {})}
                    kind = "tool_call"
                if kind == "final":
                    final_summary = str(action.get("summary") or action.get("answer") or "")
                    if tests_passed:
                        final_status = "success"
                        break
                    messages.append({
                        "role": "user",
                        "content": '{"action":"tool_call","tool":"run_tests","arguments":{"repo_path":"%s"}}' % repo_path,
                    })
                    continue

                if kind == "delegate":
                    review = self._delegate_reviewer(issue, source_content, str(action.get("question", "")))
                    observation = {"role": action.get("role", "reviewer"), "review": review}
                    steps.append({"step": len(steps), "type": "subagent", "request": action, "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue

                if kind != "tool_call":
                    observation = {"error": action.get("error", "invalid action"), "raw": action.get("raw", "")}
                    steps.append({"step": len(steps), "type": "protocol_error", "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue

                tool = action.get("tool")
                arguments = action.get("arguments")
                if isinstance(tool, str) and isinstance(arguments, dict):
                    arguments = self._normalize_tool_arguments(tool, arguments, repo_path)
                if not isinstance(tool, str) or not isinstance(arguments, dict):
                    observation = {"error": "tool_call 必须包含 tool 和对象 arguments"}
                    steps.append({"step": len(steps), "type": "protocol_error", "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue
                if tool not in {t.get("name") for t in tools}:
                    observation = {"error": f"未知工具: {tool}"}
                    steps.append({"step": len(steps), "type": "tool_error", "tool": tool, "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue
                if tool == "write_file" and self._is_test_path(str(arguments.get("path", ""))):
                    observation = {"error": "禁止修改测试文件"}
                    steps.append({"step": len(steps), "type": "tool_error", "tool": tool, "arguments": arguments, "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue
                if tool == "write_file" and str(arguments.get("path", "")) not in snapshots:
                    observation = {"error": "目标必须是已读取的现有源文件"}
                    steps.append({"step": len(steps), "type": "tool_error", "tool": tool, "arguments": arguments, "result": observation})
                    messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
                    continue

                tool_started = time.perf_counter()
                try:
                    result = client.call_tool(tool, arguments)
                except Exception as exc:
                    result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
                step_record = {
                    "step": len(steps),
                    "tool": tool,
                    "arguments": arguments,
                    "result": result,
                    "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 2),
                }
                steps.append(step_record)
                if tool == "run_tests":
                    last_test_result = result
                    tests_passed = self._test_passed(result)
                messages.append({
                    "role": "user",
                    "content": json.dumps({"observation": result}, ensure_ascii=False),
                })

            if final_status != "success" and tests_passed:
                final_status = "tests_passed"

            patch = self._make_patch(repo_path, snapshots, client)

        return {
            "steps": steps,
            "patch": patch,
            "tests_passed": tests_passed,
            "final_status": final_status,
            "summary": final_summary,
            "last_test_result": last_test_result,
        }
