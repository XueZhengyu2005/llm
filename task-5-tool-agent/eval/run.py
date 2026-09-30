"""任务五自检：工具单元测试 + 多工具任务成功率 + 错误恢复。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests

SUCCESS_RATE_PASS = 0.6   # 多工具任务成功率通过线


def normalize_answer(text):
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    # “位数为6”和“6位”语义相同；不改 gold，只统一等价表达。
    text = re.sub(r"(?:其)?位数(?:为|是)?\s*(\d+)", r"\1位", text)
    return re.sub(r"\s+", "", text)


def answer_matches(answer, expected_keywords):
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(keyword) in norm_answer
                       for keyword in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


def expected_tools_succeeded(trace, expected_tools, *, after_injection=False):
    """每个预期工具至少有一次成功且未被注入覆盖的真实 Observation。"""
    steps = trace.get("steps", []) if isinstance(trace, dict) else []
    if after_injection:
        indices = [i for i, step in enumerate(steps) if step.get("error_injected")]
        steps = steps[indices[0] + 1:] if indices else []
    return bool(expected_tools) and all(
        any(step.get("type") == "tool_call" and step.get("tool") == tool
            and not step.get("error_injected")
            and not str(step.get("observation", "")).startswith("Error:")
            for step in steps)
        for tool in expected_tools
    )


def extract_used_tools(trace):
    used = []
    for step in trace.get("steps", []) if isinstance(trace, dict) else []:
        if not isinstance(step, dict):
            continue
        name = step.get("tool") or step.get("tool_name") or step.get("action")
        if isinstance(name, str) and name:
            used.append(name)
    return used


def test_tools_individual():
    """每个工具单元测试。"""
    results = {}
    try:
        from src.tools import calculator, python_sandbox, file_search, wiki
    except ImportError as e:
        return {"test": "tools_individual", "pass": False,
                "error": f"工具导入失败：{e}"}

    # (工具名, 模块, 调用参数, 预期子串, 是否依赖网络)：
    # expected=None 表示只检查响应长度 > 50（如 wiki 网络回包）；
    # network=True 的工具若抛异常按“跳过”处理（多半是离线/被墙），不拖累其余工具判定。
    checks = [
        ("calculator",     calculator,     {"expression": "2 + 3 * 4"},                  "14",        False),
        ("python_sandbox", python_sandbox, {"code": "print(sum(range(10)))"},            "45",        False),
        ("file_search",    file_search,    {"pattern": "README.md", "dir": str(ROOT)},   "README.md", False),
        ("wiki",           wiki,           {"query": "Alan Turing"},                     None,        True),
    ]
    network_skipped = []
    for name, mod, args, expected, network in checks:
        try:
            out = str(mod.run(args))
            results[name] = (expected in out) if expected is not None else (len(out) > 50)
        except Exception as e:
            if network:
                results[name] = f"skip(网络不可用？): {e}"
                network_skipped.append(name)
            else:
                results[name] = f"error: {e}"

    gated = [v for k, v in results.items() if k not in network_skipped]
    all_pass = bool(gated) and all(v is True for v in gated)
    return {"test": "tools_individual", "pass": all_pass, "results": results,
            "network_skipped": network_skipped or None}


def test_multi_tool_success_rate():
    from src.agent import ReActAgent
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not tasks:
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 为空；重跑 data/download.py 生成"}
    agent = ReActAgent()
    answer_success = 0
    tool_coverage = 0
    tool_execution_success = 0
    grounded_success = 0
    details = []
    traces_path = ROOT / "eval" / "agent_traces.jsonl"
    traces_path.write_text("", encoding="utf-8")
    for t in tasks:
        try:
            trace = agent.run(t["task"])
            with traces_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": t["id"], "trace": trace}, ensure_ascii=False) + "\n")
            final_answer = trace.get("final_answer", "") if isinstance(trace, dict) else ""
            expected = t.get("expected_answer_contains", [])
            answer_ok = answer_matches(final_answer, expected)
            answer_success += int(answer_ok)
            used_tools = extract_used_tools(trace)
            expected_tools = t.get("expected_tools", [])
            expected_tools_ok = all(
                any(expected_tool == used for used in used_tools)
                for expected_tool in expected_tools
            ) if expected_tools else True
            tool_errors = any(
                step.get("type") == "tool_call" and str(step.get("observation", "")).startswith("Error:")
                for step in trace.get("steps", [])
            ) if isinstance(trace, dict) else True
            tool_ok = expected_tools_ok and not tool_errors
            expected_tool_success = expected_tools_succeeded(trace, expected_tools)
            grounded = answer_ok and expected_tool_success
            grounded_success += int(grounded)
            tool_coverage += int(expected_tools_ok)
            tool_execution_success += int(tool_ok)
            details.append({
                "id": t["id"],
                "answer_correct": answer_ok,
                "tool_execution_success": tool_ok,
                "expected_tools_succeeded": expected_tool_success,
                "grounded_success": grounded,
                "final_answer_preview": str(final_answer)[:120],
                "expected_answer_contains": expected,
                "expected_tools": expected_tools,
                "used_tools": used_tools,
                "expected_tools_covered": expected_tools_ok,
            })
        except Exception as e:
            details.append({"id": t["id"], "answer_correct": False, "tool_execution_success": False, "error": str(e)})
    answer_rate = answer_success / len(tasks)
    return {"test": "multi_tool_success_rate", "pass": answer_rate > SUCCESS_RATE_PASS,
            "rate": round(answer_rate, 3),
            "answer_correct_rate": round(answer_rate, 3),
            "tool_execution_success_rate": round(tool_execution_success / len(tasks), 3),
            "expected_tool_coverage_rate": round(tool_coverage / len(tasks), 3),
            "grounded_success_rate": round(grounded_success / len(tasks), 3),
            "n": len(tasks), "details": details}


def test_error_recovery():
    """注入错误工具响应，检查 agent 能否恢复并完成任务。"""
    from src.agent import ReActAgent
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "error_recovery", "pass": None, "skip": "data/tasks.json 不存在"}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    # 选前 5 题做错误恢复测试（节省时间）
    test_tasks = tasks[:5]
    agent = ReActAgent(max_steps=12)
    recovered = 0
    verified_recovered = 0
    injected_count = 0
    details = []
    traces_path = ROOT / "eval" / "error_recovery_traces.jsonl"
    traces_path.write_text("", encoding="utf-8")
    for t in test_tasks:
        try:
            trace = agent.run(t["task"], inject_error_at=1)
            with traces_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": t["id"], "trace": trace}, ensure_ascii=False) + "\n")
            final_answer = trace.get("final_answer", "") if isinstance(trace, dict) else ""
            injected = bool(trace.get("error_injected")) if isinstance(trace, dict) else False
            ok = injected and answer_matches(final_answer, t.get("expected_answer_contains", []))
            injected_count += int(injected)
            recovered += int(ok)
            verified = ok and expected_tools_succeeded(trace, t.get("expected_tools", []), after_injection=True)
            verified_recovered += int(verified)
            details.append({"id": t["id"], "injected": injected, "recovered": ok,
                            "tool_verified_recovery": verified, "answer": str(final_answer)[:100]})
        except Exception as e:
            details.append({"id": t["id"], "injected": False, "recovered": False, "error": str(e)[:100]})
    rate = recovered / injected_count if injected_count else 0.0
    return {"test": "error_recovery",
            "pass": injected_count == len(test_tasks) and rate >= 0.4,
            "recovery_rate": round(rate, 3),
            "tool_verified_recovery_rate": round(verified_recovered / injected_count, 3)
                                             if injected_count else 0.0,
            "injected": injected_count, "n": len(test_tasks), "details": details}


if __name__ == "__main__":
    run_tests([test_tools_individual, test_multi_tool_success_rate,
               test_error_recovery], ROOT)
