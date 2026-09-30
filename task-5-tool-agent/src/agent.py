"""手写 ReAct Agent：通过严格解析的工具循环解决任务。"""
from __future__ import annotations

import json
import re
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT.parent / "task-4-rag" / "models" / "Qwen2.5-7B-Instruct-AWQ"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct-AWQ"
FIXTURES_DIR = str(ROOT / "data" / "agent-fixtures")

_tools_registry: Dict[str, Any] = {}


def _register_tools():
    if _tools_registry:
        return
    from src.tools import calculator, file_search, python_sandbox, wiki
    _tools_registry.update({
        "calculator": calculator,
        "python_sandbox": python_sandbox,
        "file_search": file_search,
        "wiki": wiki,
    })


class ReActAgent:
    def __init__(self, max_steps: int = 12):
        _register_tools()
        self.max_steps = max_steps
        self._inject_call_number: Optional[int] = None
        self._error_msg = "Error: 工具内部异常，结果不可用，请重试或换一种方法。"
        self._model = None
        self._tokenizer = None
        self._wiki_unavailable = False
        self._active_task = ""

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), device_map="auto", dtype=torch.float16
        )
        self._model.eval()

    def _build_system_prompt(self) -> str:
        return """你是一个使用工具解决问题的助手。
可用工具：
- calculator: 参数 {"expression": "数学表达式"}；支持 sqrt(2026)、round(sqrt(2026), 6)、len(str(456831))；不支持 math.sqrt 或对象属性访问
- python_sandbox: 参数 {"code": "Python代码"}；适合列表、字符串、编程逻辑与统计。每次调用是独立进程，不保留上次定义的函数或变量，结果必须 print 输出
- file_search: 参数 {"pattern": "关键词或文件名", "dir": "目录"}
- wiki: 参数 {"query": "查询关键词"}

每次回复必须只包含一种动作，并严格使用以下格式：
Thought: 简短说明
Action: 工具名
Action Input: 一个合法 JSON 对象

收到 Observation 后继续。如果已经有足够信息，回复：
Final Answer: 最终答案

规则：
1. 不要自己伪造 Observation。
2. Action Input 必须是合法 JSON，字符串中的大括号和换行必须正确转义，例如多行代码写成 \n。
3. 一次回复不能同时包含 Action 和 Final Answer。
4. 最终答案必须包含题目要求的数值、文件名和关键词。如果题目同时要计算结果及派生值（如位数），两个值都要写。要求“小数点后 N 位”时，用 calculator 计算完整值，再用 python_sandbox 验证格式。
5. 判断真假时保留工具输出的 True/False，不只写“是”或“不是”。
6. 只有明确要求百科资料的任务才调用 wiki；纯数学、Python 和本地文件题不需要 wiki。wiki 报网络错误时，明确说明无法核实来源，不要声称已查到维基百科结果。
7. 工具失败时分析错误并重试或改用其他工具。如果已有足够工具证据，立即输出 Final Answer，不要反复查询同一个目标。
"""

    def _call_model(self, messages: list) -> str:
        import torch
        self._load_model()
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=384, do_sample=False,
                repetition_penalty=1.1, pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

    @staticmethod
    def _escape_string_controls(text: str) -> str:
        """仅修复 JSON 字符串中模型输出的原样换行/制表符，保留嵌套括号。"""
        result = []
        quoted = False
        escaped = False
        for char in text:
            if escaped:
                result.append(char)
                escaped = False
            elif char == "\\" and quoted:
                result.append(char)
                escaped = True
            elif char == '"':
                result.append(char)
                quoted = not quoted
            elif quoted and char in "\n\r\t":
                result.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
            else:
                result.append(char)
        return "".join(result)

    @classmethod
    def _decode_json_after(cls, text: str, marker_end: int) -> Optional[dict]:
        decoder = json.JSONDecoder()
        start = text.find("{", marker_end)
        if start < 0:
            return None
        # 不扫描内部的 {：如果外层 JSON 坏了，绝不能把字符串中的对象当作参数。
        candidate = text[start:]
        for value in (candidate, cls._escape_string_controls(candidate)):
            try:
                parsed, _ = decoder.raw_decode(value)
            except JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _missing_numeric_detail(task: str, answer: str, steps: List[Dict[str, Any]]) -> Optional[str]:
        """需要同时汇报数值和位数时，阻止模型遗漏已计算出的原数值。"""
        if not re.search(r"位数|几位|digit", task, re.IGNORECASE):
            return None
        # 以最后一次成功的计算作为最终值，不能把前面的中间量误当结果。
        values = [str(step.get("observation", "")).strip() for step in steps
                  if step.get("type") == "tool_call" and step.get("tool") == "calculator"]
        for value in reversed(values):
            if re.fullmatch(r"-?\d{2,}", value):
                return value if value not in answer else None
        return None

    @staticmethod
    def _missing_boolean_detail(answer: str, steps: List[Dict[str, Any]]) -> Optional[str]:
        """有实际布尔工具输出时，不只用含糊的自然语言代替。"""
        outputs = [str(step.get("observation", "")) for step in steps
                   if step.get("type") == "tool_call" and step.get("tool") == "python_sandbox"
                   and not step.get("error_injected")]
        if not outputs:
            return None
        tokens = {token for output in outputs for token in re.findall(r"\b(?:True|False)\b", output)}
        missing = sorted(token for token in tokens if token not in answer)
        return ", ".join(missing) if missing else None

    @staticmethod
    def _missing_decimal_result(task: str, answer: str, steps: List[Dict[str, Any]]) -> Optional[str]:
        """问小数结果时，不能只返回末尾数字而丢掉整数部分。"""
        if not re.search(r"小数|decimal|\d+\s*位结果", task, re.IGNORECASE):
            return None
        for step in reversed(steps):
            if step.get("type") != "tool_call" or step.get("tool") not in ("calculator", "python_sandbox"):
                continue
            observation = str(step.get("observation", "")).strip()
            if re.fullmatch(r"-?\d+\.\d+", observation):
                integer, fraction = observation.split(".", 1)
                if not re.search(r"(?<!\d)" + re.escape(integer) + r"\.\d+", answer):
                    return observation
                return None
        return None

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """按出现位置解析单个动作，避免 Final Answer 抢先吞掉 Action。"""
        final_match = re.search(r"Final\s+Answer\s*:\s*", text, re.IGNORECASE)
        action_match = re.search(r"Action\s*:\s*([A-Za-z_][\w-]*)", text, re.IGNORECASE)
        input_match = re.search(r"Action\s+Input\s*:\s*", text, re.IGNORECASE)

        if action_match and final_match:
            return {"type": "parse_error", "error": "一次响应不能同时包含 Action 和 Final Answer", "raw": text}

        if action_match and input_match:
            action_name = action_match.group(1)
            parsed_input = self._decode_json_after(text, input_match.end())
            if parsed_input is None:
                return {"type": "parse_error", "error": "Action Input 不是完整 JSON", "raw": text}
            if not isinstance(parsed_input, dict):
                return {"type": "parse_error", "error": "Action Input 必须是 JSON 对象", "raw": text}
            return {"type": "action", "name": action_name, "input": parsed_input}

        if final_match:
            answer = text[final_match.end():].strip()
            return {"type": "final", "answer": answer}
        return None

    @staticmethod
    def _task_needs_wiki(task: str) -> bool:
        return bool(re.search(r"维基|百科|出生|发明者|发表年份|发布年份|哪一年", task))

    def _execute_tool(self, name: str, input_data: dict, task: str = "") -> str:
        if name == "wiki" and task and not self._task_needs_wiki(task):
            return "Error: 当前任务不需要百科检索，请继续使用题目相关工具或直接回答。"
        if name not in _tools_registry:
            return f"Error: 未知工具 '{name}'，可用: {list(_tools_registry)}"
        if name == "wiki" and self._wiki_unavailable:
            return "Error: Wikipedia 网络当前不可用；不要将模型记忆当作检索结果。"
        try:
            result = _tools_registry[name].run(input_data)
            return str(result)[:4000]
        except Exception as exc:
            if name == "wiki":
                self._wiki_unavailable = True
            return f"Error: 工具执行失败: {type(exc).__name__}: {exc}"

    def run(
        self,
        task: str,
        inject_error_at: int = None,
        error_message: str = None,
    ) -> Dict[str, Any]:
        """运行 ReAct 循环。

        inject_error_at 使用 1-based 工具调用序号，而不是循环 step 序号。
        """
        _register_tools()
        self._wiki_unavailable = False
        self._active_task = task
        self._inject_call_number = inject_error_at
        self._error_msg = error_message or "Error: 工具内部异常，结果不可用，请重试或换一种方法。"
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"任务：{task}\n请开始。"},
        ]
        steps: List[Dict[str, Any]] = []
        final_answer = ""
        tool_call_count = 0
        injected = False
        last_observations: List[str] = []
        revision_requested = False

        for step_num in range(self.max_steps):
            started = time.perf_counter()
            response = self._call_model(messages)
            parsed = self._parse_action(response)
            steps.append({
                "step": step_num,
                "type": "model_response",
                "response": response,
                "parsed": parsed,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            messages.append({"role": "assistant", "content": response})

            if parsed is None:
                messages.append({"role": "user", "content": "格式错误：请严格输出一个 Action 或 Final Answer。"})
                continue
            if parsed.get("type") == "parse_error":
                messages.append({"role": "user", "content": f"格式错误：{parsed['error']}。请重试。"})
                continue
            if parsed["type"] == "final":
                candidate = parsed["answer"]
                numeric = self._missing_numeric_detail(task, candidate, steps)
                boolean = self._missing_boolean_detail(candidate, steps)
                decimal = self._missing_decimal_result(task, candidate, steps)
                if (numeric or boolean or decimal) and not revision_requested:
                    revision_requested = True
                    steps.append({"step": step_num, "type": "answer_revision_requested",
                                  "missing_numeric": numeric, "missing_boolean": boolean,
                                  "missing_decimal": decimal})
                    hint = (f"请保留最终计算结果 {numeric}，不要误用中间数值。" if numeric else "")
                    hint += (f"请按工具输出保留布尔值 {boolean}。" if boolean else "")
                    hint += (f"请保留完整的小数 {decimal}（包括整数部分），不要只给小数位。" if decimal else "")
                    messages.append({"role": "user", "content":
                        f"最终答案遗漏必要的工具证据：{hint}请重新给出完整的 Final Answer。"})
                    continue
                final_answer = candidate
                steps.append({"step": step_num, "type": "final_answer", "answer": final_answer})
                break

            action_name = parsed["name"]
            action_input = parsed["input"]
            tool_call_count += 1
            tool_started = time.perf_counter()
            observation = self._execute_tool(action_name, action_input)
            if self._inject_call_number == tool_call_count:
                observation = self._error_msg
                injected = True
            last_observations.append(observation)
            steps.append({
                "step": step_num,
                "type": "tool_call",
                "tool": action_name,
                "input": action_input,
                "observation": observation,
                "tool_call_number": tool_call_count,
                "error_injected": self._inject_call_number == tool_call_count,
                "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 2),
            })
            messages.append({"role": "user", "content": f"Observation: {observation}\n请继续。"})

        if not final_answer:
            obs_str = "\n".join(f"工具结果：{obs}" for obs in last_observations[-6:]) or "(无工具结果)"
            messages.append({
                "role": "user",
                "content": f"以下是工具结果：\n{obs_str}\n请直接回复 Final Answer: 最终答案。",
            })
            response = self._call_model(messages)
            final_match = re.search(r"Final\s+Answer\s*:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
            final_answer = final_match.group(1).strip() if final_match else response.strip()
            missing_decimal = self._missing_decimal_result(task, final_answer, steps)
            if missing_decimal:
                # 步数耗尽也不能把小数的末尾数字错称为完整结果。
                steps.append({"step": self.max_steps, "type": "forced_final_revision",
                              "rejected_answer": final_answer, "required_decimal": missing_decimal})
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    f"之前的回答遗漏了整数部分。计算器已经返回完整值 {missing_decimal}。"
                    "请直接给出包含整数和小数部分的 Final Answer；不要只给末尾数字。"})
                response = self._call_model(messages)
                final_match = re.search(r"Final\s+Answer\s*:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
                final_answer = final_match.group(1).strip() if final_match else response.strip()
                if self._missing_decimal_result(task, final_answer, steps):
                    final_answer = ""  # 宁可明确失败，不把不完整数字作为成功返回。
            steps.append({"step": self.max_steps, "type": "forced_final", "answer": final_answer, "response": response})

        return {
            "steps": steps,
            "final_answer": final_answer,
            "success": bool(final_answer),
            "tool_calls": tool_call_count,
            "error_injected": injected,
            "error_recovery": bool(injected and final_answer and any(
                step.get("type") == "tool_call" and step.get("tool_call_number", 0) > (inject_error_at or 0)
                and not step.get("error_injected")
                and not str(step.get("observation", "")).startswith("Error:")
                for step in steps
            )),
        }
