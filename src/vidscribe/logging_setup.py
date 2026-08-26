"""日志：同时输出到控制台和 logs/ 目录，强制 UTF-8 避免 Windows 控制台乱码。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: Path, name: str = "run", level: int = logging.INFO) -> Path:
    global _CONFIGURED
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    root = logging.getLogger()
    root.setLevel(level)
    if _CONFIGURED:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream = sys.stdout
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    console = logging.StreamHandler(stream)
    console.setFormatter(fmt)
    root.addHandler(console)

    for noisy in ("urllib3", "filelock", "huggingface_hub", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
