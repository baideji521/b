"""浏览器扩展对接用的本地 HTTP Bridge。

用途：把「合并导出」那份文本 + 高光筛选提示词交给浏览器扩展，扩展在真实浏览器里
驱动网页版 AI（Gemini）问一次，把 AI 回的 JSON 回传，GUI 直接拿去剪高光。
好处是复用浏览器里已登录的会话，不需要 API key。

协议（扩展侧见 视频好帮手_浏览器扩展/src/ai-task.js，沿用它 bridge-client.js 的
endpoint/token 契约）：

    GET  /v1/health                     无需鉴权，扩展探测端口用
    GET  /v1/pair                       仅在 GUI 点了「配对扩展」后的窗口期内返回 token
    GET  /v1/ai/next?types=gemini_json  领一个任务（没有就返回 task=null）
    GET  /v1/ai/file?task_id=..&index=0 取任务附带的 txt（扩展要把它上传到网页）
    POST /v1/ai/progress                回报阶段，响应里带 cancelled 让扩展及时中断
    POST /v1/ai/result                  回传结果（AI 原文 + 解析出的 JSON）

只监听 127.0.0.1，且除 /v1/health、/v1/pair 外都要 Bearer token —— 本机上的任意
网页都能访问 localhost，没有 token 就等于把「打开任意网页并抓取内容」的能力
暴露给了所有站点。

/v1/ai/file 只认「任务 id + 下标」，不接受客户端传路径：文件清单是 GUI 提交任务时
登记的，扩展只能取这几个，拿不到任意读盘的能力。
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..logging_setup import get_logger

logger = get_logger("bridge")

PROTOCOL_VERSION = 1
# 配对窗口：GUI 点一次「配对扩展」开一个短窗口，扩展轮询到就把 token 领走
PAIR_WINDOW_SECONDS = 120.0
# 单个请求体上限：提示词 + 合并导出大概几百 KB，留足余量同时挡住异常大的包
MAX_BODY_BYTES = 32 * 1024 * 1024

EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class Task:
    task_id: str
    type: str
    payload: dict[str, Any]
    # 要让扩展上传到网页的本地文件（绝对路径）。只有这几个能被 /v1/ai/file 读到
    files: list[Path] = field(default_factory=list)
    status: str = "queued"      # queued / running / done / failed / cancelled
    stage: str = ""
    message: str = ""
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)


def extract_json(text: str) -> dict[str, Any] | None:
    """从 AI 的回答里抠出 JSON 对象：优先 ```json 围栏，其次第一个配平的 {...}。

    AI 经常在 JSON 前后写解释，直接 json.loads 整段会失败。
    """
    if not text:
        return None
    fence = text.find("```")
    while fence >= 0:
        newline = text.find("\n", fence)
        close = text.find("```", newline + 1) if newline > 0 else -1
        if newline > 0 and close > 0:
            body = text[newline + 1:close].strip()
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        fence = text.find("```", fence + 3)

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


class _Server(ThreadingHTTPServer):
    """Windows 下 SO_REUSEADDR 允许绑到别人已经在听的端口，两个实例会抢同一个端口、
    请求被随机一个接走。关掉复用，占用就老实报错，交给 start() 顺延到下一个端口。"""

    allow_reuse_address = False


class BridgeServer:

    """本机 HTTP 服务 + 一条任务队列。线程安全，GUI 侧只碰这个类。"""

    def __init__(self, port: int = 5998, fallbacks: int = 9, token: str = "",
                 on_event: EventFn | None = None) -> None:
        self.port = int(port)
        self.fallbacks = max(0, int(fallbacks))
        self.token = token or secrets.token_urlsafe(24)
        self.on_event = on_event
        self._lock = threading.RLock()
        self._tasks: list[Task] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bound_port = 0
        self._pair_until = 0.0
        self._last_seen = 0.0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> int:
        """起服务，返回实际监听的端口。首选端口被占就往后顺延。"""
        if self._httpd is not None:
            return self._bound_port
        last_error: OSError | None = None
        for offset in range(self.fallbacks + 1):
            port = self.port + offset
            try:
                httpd = _Server(("127.0.0.1", port), _Handler)
            except OSError as exc:
                last_error = exc
                continue
            httpd.bridge = self  # type: ignore[attr-defined]
            httpd.daemon_threads = True
            self._httpd = httpd
            self._bound_port = port
            self._thread = threading.Thread(target=httpd.serve_forever,
                                            name="bridge-http", daemon=True)
            self._thread.start()
            logger.info("Bridge 监听 %s", self.url)
            self._emit("listening", {"url": self.url, "port": port})
            return port
        raise OSError(f"{self.port}-{self.port + self.fallbacks} 全被占用：{last_error}")

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        self._thread = None
        self._bound_port = 0
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
            logger.info("Bridge 已停止")
            self._emit("stopped", {})

    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._bound_port}" if self._bound_port else ""

    # ------------------------------------------------------------------ 对 GUI
    def open_pair_window(self, seconds: float = PAIR_WINDOW_SECONDS) -> float:
        """开一个配对窗口。窗口内扩展 GET /v1/pair 就能拿到 token（拿完即关）。"""
        with self._lock:
            self._pair_until = time.time() + float(seconds)
        self._emit("pair_window", {"seconds": seconds})
        return self._pair_until

    def submit(self, task_type: str, payload: dict[str, Any],
               files: list[Path] | None = None) -> str:
        """入队一个任务，返回 task_id。同类型的旧任务直接作废，只留最新那条。

        files 是要让扩展上传到网页的本地 txt；扩展只能按下标从 /v1/ai/file 取，
        任务里回给扩展的清单只有文件名和大小，不带路径。
        """
        picked = [Path(p) for p in (files or [])]
        payload = dict(payload)
        payload["files"] = [
            {"index": index, "name": path.name,
             "size": path.stat().st_size if path.is_file() else 0,
             "url": f"/v1/ai/file?task_id=%(task_id)s&index={index}"}
            for index, path in enumerate(picked)
        ]
        task = Task(task_id=secrets.token_hex(8), type=task_type, payload=payload,
                    files=picked)
        # url 里要带上真正的 task_id
        for item in task.payload["files"]:
            item["url"] = item["url"] % {"task_id": task.task_id}
        with self._lock:
            for old in self._tasks:
                if old.type == task_type and old.status in ("queued", "running"):
                    old.status = "cancelled"
                    old.message = "被新任务取代"
            self._tasks.append(task)
        self._emit("queued", {"task_id": task.task_id, "type": task_type,
                              "files": [p.name for p in picked]})
        return task.task_id

    def cancel(self, task_id: str | None = None) -> None:
        """取消指定任务；不给 id 就取消所有排队中/进行中的。"""
        with self._lock:
            for task in self._tasks:
                if task_id and task.task_id != task_id:
                    continue
                if task.status in ("queued", "running"):
                    task.status = "cancelled"
                    task.message = "用户停止"
        self._emit("cancelled", {"task_id": task_id or ""})

    def state(self) -> dict[str, Any]:
        """给界面显示用的快照。"""
        with self._lock:
            active = next((t for t in self._tasks if t.status in ("queued", "running")), None)
            return {
                "running": self.running,
                "url": self.url,
                "paired_at": self._last_seen,
                "extension_online": bool(self._last_seen
                                         and time.time() - self._last_seen < 15.0),
                "pair_window_left": max(0.0, self._pair_until - time.time()),
                "task": None if active is None else {
                    "task_id": active.task_id, "type": active.type,
                    "status": active.status, "stage": active.stage,
                    "message": active.message,
                },
            }

    # ------------------------------------------------------------------ 内部
    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001 - 回调是 GUI 的事，别让它掀翻 HTTP 线程
            logger.exception("Bridge 事件回调出错：%s", kind)

    def _authorized(self, header: str | None) -> bool:
        if not header or not header.startswith("Bearer "):
            return False
        return secrets.compare_digest(header[7:].strip(), self.token)

    def _touch(self) -> None:
        with self._lock:
            self._last_seen = time.time()

    def _take_token(self) -> str | None:
        """配对窗口内取一次 token，取完立刻关窗（单次有效）。"""
        with self._lock:
            if time.time() > self._pair_until:
                return None
            self._pair_until = 0.0
        self._emit("paired", {})
        return self.token

    def _claim(self, types: list[str]) -> Task | None:
        with self._lock:
            for task in self._tasks:
                if task.status == "queued" and (not types or task.type in types):
                    task.status = "running"
                    task.stage = "claimed"
                    self._emit("claimed", {"task_id": task.task_id, "type": task.type})
                    return task
        return None

    def _find(self, task_id: str) -> Task | None:
        with self._lock:
            return next((t for t in self._tasks if t.task_id == task_id), None)

    def _task_file(self, task_id: str, index: int) -> Path | None:
        """按任务 id + 下标取登记过的文件。越界或任务不存在都返回 None。"""
        task = self._find(task_id)
        if task is None or index < 0 or index >= len(task.files):
            return None
        path = task.files[index]
        return path if path.is_file() else None

    def _progress(self, task_id: str, stage: str, message: str) -> bool:
        """更新阶段，返回该任务是否已被取消（扩展据此中断）。"""
        task = self._find(task_id)
        if task is None:
            return True
        with self._lock:
            if task.status == "cancelled":
                return True
            task.stage = stage
            task.message = message
        self._emit("progress", {"task_id": task_id, "stage": stage, "message": message})
        return False

    def _finish(self, task_id: str, body: dict[str, Any]) -> None:
        task = self._find(task_id)
        status = str(body.get("status") or "completed")
        text = str(body.get("text") or "")
        parsed = body.get("json")
        if not isinstance(parsed, dict):
            parsed = extract_json(text)
        result = {"status": status, "text": text, "json": parsed,
                  "error": body.get("error") or ""}
        if task is not None:
            with self._lock:
                task.result = result
                task.status = "done" if status == "completed" and parsed else "failed"
                task.stage = "finished"
        self._emit("result", {"task_id": task_id, **result})


class _Handler(BaseHTTPRequestHandler):
    server_version = "VidScribeBridge/1"
    protocol_version = "HTTP/1.1"

    # --- 基础设施 ---
    @property
    def bridge(self) -> BridgeServer:
        return self.server.bridge  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # 扩展的 service worker 带 chrome-extension:// 源发请求，放行简单跨源读取
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, filename: str) -> None:
        """回一个 txt 文件本体，给扩展拿去当附件上传。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _auth_ok(self) -> bool:
        if self.bridge._authorized(self.headers.get("Authorization")):
            self.bridge._touch()
            return True
        self._send(401, {"ok": False, "message": "缺少或错误的配对令牌"})
        return False

    # --- 路由 ---
    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/v1/health":
            self._send(200, {"ok": True, "version": PROTOCOL_VERSION, "app": "vidscribe"})
            return
        if route == "/v1/pair":
            token = self.bridge._take_token()
            if not token:
                self._send(403, {"ok": False, "message": "没有打开配对窗口"})
                return
            self.bridge._touch()
            self._send(200, {"ok": True, "token": token})
            return
        if route == "/v1/ai/next":
            if not self._auth_ok():
                return
            wanted = [t for t in (parse_qs(parsed.query).get("types") or [""])[0].split(",") if t]
            task = self.bridge._claim(wanted)
            if task is None:
                self._send(200, {"ok": True, "task": None})
                return
            self._send(200, {"ok": True, "task": {"task_id": task.task_id,
                                                  "type": task.type, **task.payload}})
            return
        if route == "/v1/ai/file":
            if not self._auth_ok():
                return
            query = parse_qs(parsed.query)
            task_id = (query.get("task_id") or [""])[0]
            try:
                index = int((query.get("index") or ["-1"])[0])
            except ValueError:
                index = -1
            path = self.bridge._task_file(task_id, index)
            if path is None:
                self._send(404, {"ok": False, "message": "任务或文件不存在"})
                return
            self._send_bytes(path.read_bytes(), path.name)
            return
        self._send(404, {"ok": False, "message": "未知路径"})

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route not in ("/v1/ai/progress", "/v1/ai/result"):
            self._send(404, {"ok": False, "message": "未知路径"})
            return
        if not self._auth_ok():
            return
        body = self._body()
        task_id = str(body.get("task_id") or "")
        if not task_id:
            self._send(400, {"ok": False, "message": "缺少 task_id"})
            return
        if route == "/v1/ai/progress":
            cancelled = self.bridge._progress(task_id, str(body.get("stage") or ""),
                                              str(body.get("message") or ""))
            self._send(200, {"ok": True, "cancelled": cancelled})
            return
        self.bridge._finish(task_id, body)
        self._send(200, {"ok": True})
