"""GUI 参数持久化：退出时存，启动时自动加载。

存哪儿：项目根目录的 `gui_settings.json`。刻意**不放在 cache/**，因为那是缓存目录，
超过 3 天会被自动清掉（见 vidscribe/cache.py），设置放进去会莫名丢失。

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
