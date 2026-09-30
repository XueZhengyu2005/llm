"""受限 Python 沙箱（教学级，带子进程超时，不等同于安全隔离）。"""
from __future__ import annotations

import ast
import builtins
import io
import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "在受限沙箱中执行 Python 代码，返回 stdout",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
            "required": ["code"],
        },
    },
}

_MAX_OUTPUT = 8000
_TIMEOUT_SECONDS = 3
_ALLOWED_MODULES = {"math", "random", "string", "collections", "itertools", "re", "json"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"不允许导入模块 {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    values = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "divmod": divmod, "enumerate": enumerate, "filter": filter,
        "float": float, "format": format, "frozenset": frozenset,
        "int": int, "isinstance": isinstance, "len": len, "list": list,
        "map": map, "max": max, "min": min, "print": print, "range": range,
        "repr": repr, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip, "True": True, "False": False, "None": None,
        "__import__": _safe_import,
    }
    return values


def _worker(code: str, queue) -> None:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] not in _ALLOWED_MODULES:
                        raise ImportError(f"不允许导入模块 {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if not node.module or node.module.split(".", 1)[0] not in _ALLOWED_MODULES:
                    raise ImportError(f"不允许导入模块 {node.module}")

        stdout, stderr = io.StringIO(), io.StringIO()
        safe_globals = {"__builtins__": _safe_builtins()}
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(tree, "<python_sandbox>", "exec"), safe_globals, safe_globals)
        output = stdout.getvalue()
        if stderr.getvalue().strip():
            output += f"\n[stderr] {stderr.getvalue().strip()}"
        queue.put({"ok": True, "output": output[:_MAX_OUTPUT] or "(无输出)"})
    except BaseException as exc:
        queue.put({"ok": False, "output": f"Error: {type(exc).__name__}: {exc}"[:_MAX_OUTPUT]})


def run(args: dict) -> str:
    code = args.get("code")
    if not isinstance(code, str):
        return "Error: code 必须是字符串"
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_worker, args=(code, queue))
    process.start()
    process.join(_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return f"Error: 执行超时（>{_TIMEOUT_SECONDS}s）"
    try:
        result = queue.get(timeout=0.5)
    except Exception:
        return "Error: 沙箱进程没有返回结果"
    return str(result.get("output", "Error: 空结果"))
