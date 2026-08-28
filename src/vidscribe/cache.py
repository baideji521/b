"""缓存目录：固定位置、统一命名、超过 3 天自动清一次。

目录结构（`config.json` 的 `paths.cache_dir`，默认 `<项目>/cache`）：

    cache/
      cleanup.json                  清理记录（上次清理时间、删了什么）
      videos/<视频标识>/            每个视频一份，删掉只是下次重跑慢一点
          state.json                断点状态
          probe.json speech.json visual.json timeline.json
          visual_windows.json       视觉窗口级缓存（长视频续跑用）
          preview_audio.wav         GUI 预览音轨
          translate_request.json / translate_result.json

`<视频标识>` = 清洗过的文件名（最多 40 字符）+ `-` + 绝对路径 sha1 前 8 位。
带哈希是因为不同目录下可能有同名视频，光用文件名会互相覆盖；
留可读前缀是为了在文件管理器里还能认出来是哪个视频。

哪些**绝不**碰：`output/`（分析结果，是交付物）、`models/`（几十 GB 权重）、`input/`（原始视频）。

清理策略：按最后修改时间，超过 `max_age_days` 天的整份删掉；
距上次清理不满 `interval_days` 天就跳过。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .logging_setup import get_logger

logger = get_logger(__name__)

VIDEOS_SUBDIR = "videos"
CLEANUP_FILE = "cleanup.json"
LEGACY_CLEANUP_FILE = ".cache_state.json"
LEGACY_DIR_NAME = "work"
SLUG_MAX_CHARS = 40
DEFAULT_INTERVAL_DAYS = 3.0
DEFAULT_MAX_AGE_DAYS = 3.0
DAY = 86400.0

_UNSAFE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def slug_for(video: str | Path) -> str:
    """视频 -> 缓存子目录名：可读前缀 + 路径哈希，唯一且不含奇怪字符。"""
    path = Path(video)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    name = _UNSAFE.sub("_", path.stem).strip("._-") or "video"
    return f"{name[:SLUG_MAX_CHARS]}-{digest}"


def videos_root(root: str | Path) -> Path:
    return Path(root) / VIDEOS_SUBDIR


def video_dir_in(root: str | Path, video: str | Path, create: bool = True) -> Path:
    """某个视频的缓存目录。root 是缓存根目录（cfg.path("cache_dir")）。"""
    path = videos_root(root) / slug_for(video)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(cfg: Any) -> Path:
    """缓存根目录（config.paths.cache_dir，默认 <项目>/cache）。"""
    return cfg.path("cache_dir")


def log_dir(cfg: Any) -> Path:
    return cfg.path("log_dir")


def _inside(path: Path, root: Path) -> bool:
    """删除前的护栏：路径必须真的在缓存目录里面。"""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _video_of(dir_path: Path) -> str | None:
    """从断点文件里读出这份缓存属于哪个视频（用来算规范目录名）。"""
    state = dir_path / "state.json"
    if not state.is_file():
        return None
    try:
        with open(state, "r", encoding="utf-8") as fh:
            value = json.load(fh).get("video")
        return str(value) if value else None
    except Exception:
        return None


def migrate_layout(cfg: Any) -> dict[str, Any]:
    """把旧布局搬到新布局，并把目录名统一成 `<视频名>-<哈希>`。

    旧布局是 `work/<视频文件名>/`：名字又长又容易重名。这里
    1) 把 `work/` 和 `cache/` 根下的散目录搬进 `cache/videos/`；
    2) 按断点文件里记的视频路径重算规范名，名字不对的就改名。
    纯移动/改名，不删任何缓存内容；目标已存在就跳过（保留已有那份）。
    """
    root = cache_dir(cfg)
    target_root = videos_root(root)
    moved: list[str] = []
    renamed: list[str] = []

    legacy_root = Path(cfg.root) / LEGACY_DIR_NAME
    for src in [p for p in (legacy_root, root) if p.is_dir()]:
        for child in sorted(src.iterdir()):
            if child.name in (VIDEOS_SUBDIR, CLEANUP_FILE):
                continue
            if child.name == LEGACY_CLEANUP_FILE:  # 清理记录改名
                target = root / CLEANUP_FILE
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    child.unlink(missing_ok=True)
                else:
                    shutil.move(str(child), str(target))
                continue
            if not child.is_dir():
                continue
            video = _video_of(child)
            name = slug_for(video) if video else f"{_UNSAFE.sub('_', child.name)[:SLUG_MAX_CHARS]}"
            target = target_root / name
            target_root.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            try:
                shutil.move(str(child), str(target))
            except Exception as exc:
                logger.warning("迁移缓存 %s 失败：%s", child.name, exc)
                continue
            moved.append(child.name)

    # 已经在 videos/ 里、但名字还是老样子的，按视频路径重算
    if target_root.is_dir():
        for child in sorted(target_root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            video = _video_of(child)
            if not video:
                continue
            proper = slug_for(video)
            if child.name == proper or (target_root / proper).exists():
                continue
            try:
                child.rename(target_root / proper)
            except Exception as exc:
                logger.warning("规范化缓存目录名 %s 失败：%s", child.name, exc)
                continue
            renamed.append(f"{child.name} -> {proper}")

    if legacy_root.is_dir() and not any(legacy_root.iterdir()):
        legacy_root.rmdir()
    if moved or renamed:
        logger.info("缓存目录已规范化：搬入 %d 项、改名 %d 项 -> %s",
                    len(moved), len(renamed), target_root)
    return {"moved": moved, "renamed": renamed, "videos_dir": str(target_root)}



def _newest_mtime(path: Path) -> float:
    """目录里最新一个文件的 mtime；空目录用目录自己的。"""
    newest = 0.0
    if path.is_file():
        return path.stat().st_mtime
    for item in path.rglob("*"):
        if item.is_file():
            try:
                newest = max(newest, item.stat().st_mtime)
            except OSError:
                continue
    if newest == 0.0:
        try:
            newest = path.stat().st_mtime
        except OSError:
            newest = time.time()
    return newest


def _size_of(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit in ("B", "KB") else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GB"


def _entries(cfg: Any) -> list[dict[str, Any]]:
    """列出所有缓存单元：cache/videos 下每个视频一个目录，logs 下每个日志一个文件。"""
    now = time.time()
    items: list[dict[str, Any]] = []
    vd = videos_root(cache_dir(cfg))
    if vd.is_dir():
        for child in sorted(vd.iterdir()):
            mtime = _newest_mtime(child)
            items.append({
                "path": child, "kind": "video", "name": child.name,
                "bytes": _size_of(child), "mtime": mtime,
                "age_days": round((now - mtime) / DAY, 2),
            })
    ld = log_dir(cfg)
    if ld.is_dir():
        for child in sorted(ld.glob("*.log")):
            mtime = _newest_mtime(child)
            items.append({
                "path": child, "kind": "log", "name": child.name,
                "bytes": _size_of(child), "mtime": mtime,
                "age_days": round((now - mtime) / DAY, 2),
            })
    return items


def read_state(cfg: Any) -> dict[str, Any]:
    path = cache_dir(cfg) / CLEANUP_FILE
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_state(cfg: Any, state: dict[str, Any]) -> None:
    root = cache_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with open(root / CLEANUP_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("写缓存清理记录失败：%s", exc)


def status(cfg: Any, interval_days: float = DEFAULT_INTERVAL_DAYS,
           max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> dict[str, Any]:
    """扫描缓存：总量、过期量、上次清理时间、是否该清了。只读，不删东西。"""
    items = _entries(cfg)
    state = read_state(cfg)
    last = float(state.get("last_cleanup_ts") or 0.0)
    now = time.time()
    stale = [it for it in items if it["age_days"] >= max_age_days]
    return {
        "dir": str(cache_dir(cfg)),
        "videos_dir": str(videos_root(cache_dir(cfg))),
        "log_dir": str(log_dir(cfg)),
        "items": len(items),
        "bytes": sum(it["bytes"] for it in items),
        "stale_items": len(stale),
        "stale_bytes": sum(it["bytes"] for it in stale),
        "stale_names": [it["name"] for it in stale],
        "last_cleanup": state.get("last_cleanup"),
        "days_since_cleanup": round((now - last) / DAY, 2) if last else None,
        # 从没清过也算到期：第一次打开就把陈旧缓存收掉
        "due": (last <= 0.0) or (now - last) >= interval_days * DAY,
        "interval_days": interval_days,
        "max_age_days": max_age_days,
    }


def cleanup(cfg: Any, max_age_days: float = DEFAULT_MAX_AGE_DAYS,
            dry_run: bool = False) -> dict[str, Any]:
    """删掉超过 max_age_days 天没动过的缓存单元，返回删了什么、腾出多少空间。"""
    roots = {"video": videos_root(cache_dir(cfg)), "log": log_dir(cfg)}
    removed: list[str] = []
    failed: list[str] = []
    freed = 0
    for item in _entries(cfg):
        if item["age_days"] < max_age_days:
            continue
        path: Path = item["path"]
        if not _inside(path, roots[item["kind"]]):  # 护栏：不在缓存目录里的一概不动
            logger.warning("跳过不在缓存目录内的路径：%s", path)
            continue
        if dry_run:
            removed.append(item["name"])
            freed += item["bytes"]
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as exc:
            failed.append(f"{item['name']}: {type(exc).__name__}")
            continue
        removed.append(item["name"])
        freed += item["bytes"]

    if not dry_run:
        state = read_state(cfg)
        state["last_cleanup_ts"] = time.time()
        state["last_cleanup"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["last_removed"] = removed
        state["last_freed_bytes"] = freed
        write_state(cfg, state)
    return {"removed": removed, "failed": failed, "freed_bytes": freed,
            "dry_run": dry_run, "max_age_days": max_age_days}


def check_on_start(cfg: Any, interval_days: float = DEFAULT_INTERVAL_DAYS,
                   max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> dict[str, Any]:
    """开软件时调一次：先把目录布局规范化，再报告现状，到期就清。"""
    migrate_layout(cfg)
    info = status(cfg, interval_days=interval_days, max_age_days=max_age_days)
    info["cleanup"] = None
    if info["due"] and info["stale_items"]:
        info["cleanup"] = cleanup(cfg, max_age_days=max_age_days)
        logger.info("缓存自动清理：删掉 %d 项，腾出 %s",
                    len(info["cleanup"]["removed"]), human_size(info["cleanup"]["freed_bytes"]))
    elif info["due"]:
        # 到期但没有过期内容：也把时间戳往前推，避免每次开都重扫
        state = read_state(cfg)
        state["last_cleanup_ts"] = time.time()
        state["last_cleanup"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["last_removed"] = []
        state["last_freed_bytes"] = 0
        write_state(cfg, state)
    return info


def summary_line(info: dict[str, Any]) -> str:
    """给日志/界面用的一句话。"""
    text = (f"缓存目录 {info['dir']}：{info['items']} 项 / {human_size(info['bytes'])}"
            f"，其中超过 {info['max_age_days']:g} 天的 {info['stale_items']} 项 / "
            f"{human_size(info['stale_bytes'])}")
    done = info.get("cleanup")
    if done:
        text += f"；已自动清理 {len(done['removed'])} 项，腾出 {human_size(done['freed_bytes'])}"
        if done["failed"]:
            text += f"，{len(done['failed'])} 项删除失败（可能被占用）"
    elif info.get("last_cleanup"):
        text += f"；上次清理 {info['last_cleanup']}"
    return text
