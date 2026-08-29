"""断点续跑：每个视频在缓存目录下一份 `cache/videos/<视频标识>/`，各阶段产物独立落盘。

产物照旧是 JSON 文件，格式一个字没改。变的只有"这份旧结果还能不能用"的判断：
构造时可以传一个 `reuse_gate(stage) -> bool`，由数据库回答（模型/配置哈希对不上就别复用）。
不传就是老规矩——文件在就复用。

注意窗口级续跑（`visual_windows.json`）不走 stage gate：那是长视频跑一半的中间态，
跟"整套结果能不能当缓存用"是两回事，必须一直可用，否则长视频每次都得从第一个窗口重来。
窗口能不能复用只由一件事决定：这份缓存是不是同一套视觉配置产的（`visual_config_hash`）。
配置没变就照常续跑，配置变了整份都不算命中——但旧文件一个字节都不动。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cache import video_dir_in
from .logging_setup import get_logger

logger = get_logger(__name__)

STAGES = ("probe", "visual", "speech", "timeline")

# 窗口缓存文件格式版本。v2 起是 {"version","visual_config_hash","windows"} 三键信封；
# v1（没有 version 的裸 dict）是历史格式，读得出来但一律不算命中：
# 它没记自己是哪套配置产的，猜不得。
WINDOW_CACHE_VERSION = 2


class Checkpoint:
    def __init__(self, cache_root: Path, video_path: Path,
                 reuse_gate: Callable[[str], bool] | None = None):
        self.dir = video_dir_in(cache_root, video_path)
        self.state_file = self.dir / "state.json"
        self._gate = reuse_gate
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
    def files_ready(self, stage: str) -> bool:
        """盘上有没有这个阶段的结果。不问"能不能用"，只问"在不在"。"""
        return bool(self.state["stages"].get(stage, {}).get("done")) and self.artifact(stage).is_file()

    def done(self, stage: str) -> bool:
        """能不能直接复用这个阶段的旧结果：文件要在，数据库那边也得认。"""
        if not self.files_ready(stage):
            return False
        if self._gate is not None and not self._gate(stage):
            return False
        return True

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

    def _read_window_file(self) -> dict[str, Any]:
        path = self.window_cache_file()
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
        except Exception as exc:
            logger.warning("窗口缓存读不了，这次从第一个窗口重来：%s", exc)
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def load_window_cache(self, config_hash: str | None = None) -> dict[str, Any]:
        """读窗口缓存，返回 {window_key: {...}}。

        给了 `config_hash` 就只认同一套视觉配置产出的窗口：对不上（包括没记过配置的
        v1 老文件）一律当没命中，重新问模型。**旧文件不删、不改**，只是这次不用它。
        不给 `config_hash` 是老规矩（有什么用什么），留给不关心配置的调用方。
        """
        raw = self._read_window_file()
        if not raw:
            return {}
        version = raw.get("version")
        if version != WINDOW_CACHE_VERSION:
            # 没有 version 的裸 dict = v1：它不记得自己是哪套配置跑出来的，不能猜
            logger.info("窗口缓存是老格式（version=%s），保留原文件但这次不复用", version)
            return {}
        if config_hash is not None and raw.get("visual_config_hash") != config_hash:
            logger.info("视觉配置变了（缓存 %s，本次 %s），窗口缓存整份不复用，旧文件保留",
                        str(raw.get("visual_config_hash"))[:12] or "无", config_hash[:12])
            return {}
        windows = raw.get("windows")
        return windows if isinstance(windows, dict) else {}

    def save_window_cache(self, cache: dict[str, Any], config_hash: str | None = None) -> None:
        """把窗口缓存整份写成 v2 信封。

        先写 `.tmp` 再 `os.replace`：崩在写盘中途时，盘上要么是上一份完整缓存、
        要么是这一份完整缓存，不会留下半截 JSON 把几十个窗口的推理白扔掉。
        """
        payload = {
            "version": WINDOW_CACHE_VERSION,
            "visual_config_hash": config_hash,
            "windows": cache,
        }
        target = self.window_cache_file()
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)

    def _flush(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

