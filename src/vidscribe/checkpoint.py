"""断点续跑：每个视频在缓存目录下一份 `cache/videos/<视频标识>/`，各阶段产物独立落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache import video_dir_in
from .logging_setup import get_logger

logger = get_logger(__name__)

STAGES = ("probe", "visual", "speech", "timeline")


class Checkpoint:
    def __init__(self, cache_root: Path, video_path: Path):
        self.dir = video_dir_in(cache_root, video_path)
        self.state_file = self.dir / "state.json"
        self.state: dict[str, Any] = {"video": str(video_path.resolve()), "stages": {}}
        if self.state_file.is_file():
            try:
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if loaded.get("video") == str(video_path.resolve()):
                    self.state = loaded
            except Exception as exc:
                logger.warning("checkpoint 损坏，重新开始: %s", exc)

    # --- 阶段状态 ---
    def done(self, stage: str) -> bool:
        return bool(self.state["stages"].get(stage, {}).get("done")) and self.artifact(stage).is_file()

    def artifact(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def load(self, stage: str) -> Any:
        with open(self.artifact(stage), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, stage: str, payload: Any, meta: dict[str, Any] | None = None) -> None:
        with open(self.artifact(stage), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        entry = {"done": True}
        entry.update(meta or {})
        self.state["stages"][stage] = entry
        self._flush()

    def mark_failed(self, stage: str, error: str) -> None:
        self.state["stages"][stage] = {"done": False, "error": error}
        self._flush()

    def reset(self) -> None:
        self.state["stages"] = {}
        for stage in STAGES:
            self.artifact(stage).unlink(missing_ok=True)
        self.window_cache_file().unlink(missing_ok=True)
        self._flush()

    # --- 窗口级续跑（长视频视觉分析用）---
    def window_cache_file(self) -> Path:
        return self.dir / "visual_windows.json"

    def load_window_cache(self) -> dict[str, Any]:
        path = self.window_cache_file()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def save_window_cache(self, cache: dict[str, Any]) -> None:
        with open(self.window_cache_file(), "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)

    def _flush(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)
