"""Coding Agent 使用的轻量 MCP stdio server。

实现 MCP 的 JSON-RPC 方法：initialize、tools/list、tools/call，并保留
run_tool/list_tools 作为本地兼容接口，方便课程自检和单元测试。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_tool_defs: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "读取仓库内文件内容",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "repo_path": {"type": "string"},
            },
            "required": ["path", "repo_path"],
        },
    },
    {
        "name": "write_file",
        "description": "写入仓库内文件内容，不应修改测试文件",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "repo_path": {"type": "string"},
            },
            "required": ["path", "content", "repo_path"],
        },
    },
    {
        "name": "run_tests",
        "description": "在仓库内运行 pytest",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_diff",
        "description": "查看当前代码修改",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_apply",
        "description": "在仓库内应用 unified diff",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "patch": {"type": "string"},
            },
            "required": ["repo_path", "patch"],
        },
    },
    {
        "name": "list_files",
        "description": "列出仓库中的源代码、测试和文档文件",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": ["repo_path"],
        },
    },
]


def list_tools() -> List[Dict[str, Any]]:
    """返回 MCP tools/list 使用的工具定义。"""
    return _tool_defs


def _safe_resolve(repo_path: str, rel_path: str) -> Path:
    """将相对路径解析到仓库内。

    这里保留课程原有的 repo_path 参数模型；路径策略后续可单独强化。
    """
    root = Path(repo_path).resolve()
    target = (root / rel_path).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"路径越界: {rel_path}")
    return target


def _result(text: str = "", *, data: Any = None, is_error: bool = False) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "isError": is_error,
        "content": [{"type": "text", "text": text}],
    }
    if data is not None:
        value["structuredContent"] = data
    return value


def _dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "read_file":
        fp = _safe_resolve(args["repo_path"], args["path"])
        text = fp.read_text(encoding="utf-8", errors="replace")
        return _result(text, data={"path": args["path"], "content": text})

    if name == "write_file":
        fp = _safe_resolve(args["repo_path"], args["path"])
        fp.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        # 先写临时文件再替换，避免写入过程中留下半个文件。
        tmp = fp.with_name(fp.name + ".agent-tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, fp)
        return _result(f"OK: 写入 {args['path']}", data={"path": args["path"], "written": True})

    if name == "run_tests":
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short"],
                cwd=args["repo_path"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            data = {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            data = {
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "timed_out": True,
            }
        text = json.dumps(data, ensure_ascii=False)
        return _result(text, data=data, is_error=bool(data["timed_out"]))

    if name == "git_diff":
        completed = subprocess.run(
            ["git", "diff"], cwd=args["repo_path"], capture_output=True, text=True, timeout=15
        )
        diff = completed.stdout or "(no diff)"
        return _result(diff, data={"returncode": completed.returncode, "diff": diff})

    if name == "git_apply":
        completed = subprocess.run(
            ["git", "apply", "-"],
            input=args["patch"],
            cwd=args["repo_path"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        return _result(json.dumps(data, ensure_ascii=False), data=data, is_error=completed.returncode != 0)

    if name == "list_files":
        root = Path(args["repo_path"]).resolve()
        files: List[str] = []
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            relative_parts = fp.relative_to(root).parts
            if "__pycache__" in relative_parts or any(part.startswith(".") for part in relative_parts):
                continue
            files.append(str(fp.relative_to(root)))
        files.sort()
        return _result("\n".join(files), data={"files": files})

    raise ValueError(f"未知工具: {name}")


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """调用工具并返回结构化 MCP 工具结果。"""
    try:
        return _dispatch(name, args)
    except Exception as exc:  # 工具错误必须回到 observation，而不是杀死 Agent
        return _result(f"Error: {exc}", data={"error": str(exc)}, is_error=True)


def run_tool(name: str, args: Dict[str, Any]) -> str:
    """兼容旧自检的文本接口。新 Agent 应使用 call_tool。"""
    response = call_tool(name, args)
    if response.get("structuredContent") is not None:
        data = response["structuredContent"]
        if name == "read_file":
            return str(data.get("content", ""))
        if name == "list_files":
            return "\n".join(data.get("files", []))
        if name == "run_tests":
            return json.dumps(data, ensure_ascii=False)
    return "\n".join(item.get("text", "") for item in response.get("content", []))


def _jsonrpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_request(request: Dict[str, Any]) -> Dict[str, Any] | None:
    """处理一条 MCP JSON-RPC 请求；通知消息不返回响应。"""
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "llm-beginner-coding-agent", "version": "1.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list_tools()}}
    if method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        if not isinstance(name, str):
            return _jsonrpc_error(request_id, -32602, "tools/call 缺少 name")
        return {"jsonrpc": "2.0", "id": request_id, "result": call_tool(name, params.get("arguments", {}))}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    return _jsonrpc_error(request_id, -32601, f"未知方法: {method}")


def serve_stdio() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps(_jsonrpc_error(None, -32603, str(exc)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    serve_stdio()


class StdioMCPClient:
    """最小 MCP stdio client，供 CodingAgent 通过 JSON-RPC 调用工具。"""

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd or str(Path(__file__).resolve().parents[1])
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server"],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "llm-beginner-agent", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized", {})

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP client 未启动")
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("MCP client 未启动")
        request_id = self._next_id
        self._next_id += 1
        self.process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params
        }) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"MCP server 提前退出: {stderr}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "MCP error"))
        return response

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list", {}).get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}).get("result", {})

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            self.process.kill()
        finally:
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
