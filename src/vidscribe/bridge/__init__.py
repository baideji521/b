"""浏览器扩展对接：本机 HTTP Bridge（详见 server.py 的模块说明）。"""

from .server import BridgeServer, Task, extract_json

__all__ = ["BridgeServer", "Task", "extract_json"]
