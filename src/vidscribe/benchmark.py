"""Benchmark：环境信息 + 各阶段耗时 + 峰值显存。"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from typing import Any

from .logging_setup import get_logger

logger = get_logger(__name__)


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["name"] = props.name
            info["total_vram_mb"] = round(props.total_memory / 1024 ** 2)
            info["capability"] = f"{props.major}.{props.minor}"
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:
        info["error"] = str(exc)[:200]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode == 0:
            info["driver"] = proc.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: dict[str, str | None] = {}
    for pkg in ("torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils",
                "faster-whisper", "ctranslate2", "opencv-python", "av", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def reset_peak_vram() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_vram_mb() -> dict[str, float] | None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            "reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
        }
    except Exception:
        return None


class Timer:
    """阶段计时器，支持累加（断点续跑时旧阶段耗时记为 0）。"""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start = time.perf_counter()

    def stage(self, name: str):
        return _StageContext(self, name)

    def record(self, name: str, seconds: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + seconds, 3)

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self._start, 3)


class _StageContext:
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.record(self.name, time.perf_counter() - self._t0)
        return False


def environment_snapshot() -> dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpu": gpu_info(),
    }
