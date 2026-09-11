"""GUI 参数持久化：退出时存，启动时自动加载。

存哪儿：项目根目录的 `gui_settings.json`。刻意**不放在 cache/**，因为那是缓存目录，
「高级选项 -> 缓存管理」里一键清空会把里面的东西整份删掉，设置放进去会莫名丢失。

存什么：界面上用户会调的东西——视觉模型、重要性过滤、置信度门槛、播放声音、
分析后自动翻译、导出目录、上次打开视频的目录、窗口大小位置与是否最大化、
三个分隔条的分区尺寸（播放器/时间轴、语音/日志、上下）、事件表各列宽度。
不存分析结果，也不碰 config.json（那份是手写配置）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

logger = get_logger(__name__)

FILE_NAME = "gui_settings.json"


def path_for(cfg: Any) -> Path:
    return Path(cfg.root) / FILE_NAME


def load(cfg: Any) -> dict[str, Any]:
    path = path_for(cfg)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # 设置文件坏了不该让软件打不开
        logger.warning("读取 %s 失败，用默认设置：%s", FILE_NAME, exc)
        return {}


def update(cfg: Any, patch: dict[str, Any]) -> None:
    """**只改自己那几个键**：先把磁盘上最新的一份读回来，再盖上 `patch`。

    主界面和卡点舞是同一个进程里的两个窗口，各自在启动时 `load()` 了一份**整份**
    快照。谁最后整份 `save()` 回去，就会拿自己启动时的老快照把对方后来写的东西
    抹掉（「手动切分」关掉窗口又变回「等间切分」就是这么丢的）。
    所以各窗口落盘一律走这里，只写自己名下的键。
    """
    data = load(cfg)
    data.update(patch)
    save(cfg, data)


def save(cfg: Any, data: dict[str, Any]) -> None:
    """先写 .part 再原子替换：直接覆盖时若有另一个进程同时在写，
    会留下前一份的尾巴，下次读就报 "Extra data"，整份设置作废。"""
    target = path_for(cfg)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(target)
    except Exception as exc:
        logger.warning("写 %s 失败：%s", FILE_NAME, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
