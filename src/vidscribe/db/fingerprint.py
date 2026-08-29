"""视频身份与配置指纹。

视频身份不用文件名也不用路径：改个名、换个目录，缓存还得认得出是同一个视频。
也不用全文件 sha256——几百兆的 mp4 每次开面板都整盘读一遍，界面会顿。
折中：**文件大小 + 头/中/尾各 1MB 的 sha256**。同一个文件永远一样，
不同文件撞车的概率可以忽略（除了刻意构造）。真需要全文件哈希再惰性补 `sha256` 列。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024  # 头/中/尾各取 1MB


def fingerprint(video: str | Path) -> str:
    """视频指纹：`<字节数>-<头中尾 1MB 的 sha1 前 16 位>`。文件读不了就抛 OSError。"""
    path = Path(video)
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(str(size).encode("ascii"))
    with open(path, "rb") as fh:
        digest.update(fh.read(CHUNK))
        if size > CHUNK * 2:
            fh.seek(max(0, size // 2 - CHUNK // 2))
            digest.update(fh.read(CHUNK))
        if size > CHUNK:
            fh.seek(max(0, size - CHUNK))
            digest.update(fh.read(CHUNK))
    return f"{size}-{digest.hexdigest()[:16]}"


def full_sha256(video: str | Path) -> str:
    """全文件 sha256。慢，只在明确需要时调（比如查重、归档）。"""
    digest = hashlib.sha256()
    with open(Path(video), "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def config_hash(payload: Any) -> str:
    """配置指纹：配置变了就该重新分析，靠这个值比对。

    只认内容不认书写顺序（sort_keys），所以手改 config.json 调换字段顺序不会误判。
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
