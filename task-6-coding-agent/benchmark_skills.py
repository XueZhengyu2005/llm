"""Task 6 S3: 纯 prompt vs 加 Skill（工具+结构化流程）的修复成功率对比。"""
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # task-6-coding-agent/
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent
from src.mcp_server import run_tool


def reset_repo():
    """恢复 buggy 版本。"""
    import shutil
    repo = ROOT / "data" / "toy-repo"
    shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")
    return str(repo)


def run_with_skill(agent):
    """正常流程：读代码 → 分析 → 修复 → 测试（Skill 流程）。"""
    repo = reset_repo()
    issue = (repo + "/ISSUE.md")
    issue_text = Path(issue).read_text(encoding="utf-8")
    start = time.perf_counter()
    trace = agent.run(repo, issue_text)
    elapsed = time.perf_counter() - start
    return {"mode": "with_skill", "success": trace["tests_passed"],
            "steps": len(trace["steps"]), "time_s": round(elapsed, 1)}


def run_without_skill(agent):
    """纯 prompt：不给源代码，直接让模型猜 bug 并修复。"""
    import torch, re
    repo = reset_repo()
    
    issue_text = Path(repo + "/ISSUE.md").read_text(encoding="utf-8")
    # 只给 issue，不给源代码
    prompt = f"You are a coding agent. Fix the bug.\n\nIssue: {issue_text}\n\nOutput the corrected calculator.py in a ```python code block."
    messages = [{"role": "user", "content": prompt}]
    
    start = time.perf_counter()
    response = agent._call(messages, max_new_tokens=512)
    elapsed = time.perf_counter() - start
    
    # 提取代码并写入
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).rstrip()
        run_tool("write_file", {"path": "calculator.py", "content": code + "\n", "repo_path": repo})
    
    # 跑测试
    test_result = run_tool("run_tests", {"repo_path": repo})
    success = "exit_code=0" in test_result
    return {"mode": "without_skill", "success": success,
            "steps": 1, "time_s": round(elapsed, 1)}


if __name__ == "__main__":
    print("=" * 50)
    print("S3: 纯 prompt vs 加 Skill 对比")
    print("=" * 50)
    
    agent = CodingAgent()
    agent._load_model()  # 只加载一次模型
    
    r1 = run_with_skill(agent)
    print(f"\n有 Skill（工具流程）: success={r1['success']}, steps={r1['steps']}, time={r1['time_s']}s")
    
    r2 = run_without_skill(agent)
    print(f"无 Skill（纯 prompt）: success={r2['success']}, steps={r2['steps']}, time={r2['time_s']}s")
    
    import json
    out = ROOT / "eval" / "skill_comparison.json"
    out.write_text(json.dumps([r1, r2], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存到 {out}")
