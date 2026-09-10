"""选文件对话框 + **记住上次选到哪儿**。

两件事：

1. **默认用系统自带的对话框。** 它认得"快速访问 / 最近使用 / 网盘 / 这台电脑"，
   翻素材比 Qt 自绘那个顺手太多。
   （曾经一度改成 Qt 自绘，理由是怀疑原生对话框把界面拖死了 —— 后来查清楚那次卡死
   是对齐面板读了一个不存在的列名抛 IndexError，跟对话框无关。所以改回来。
   但退路留着：`config.json` 里 `dance.native_dialogs = false` 就换成 Qt 自绘，
   万一哪天真被系统 shell 扩展卡住，那种卡死卡在系统代码里、日志上一个字都没有。)

2. **每个用途各自记住上次的目录。** "选源视频"和"选目标歌"通常不在一个盘上，
   共用一个"最近目录"反而更烦。所以按 key 分开记（`dance.source` / `dance.song` / …），
   存进 `gui_settings.json`，下次开界面直接落到那儿。

记忆由窗口在启动时 `install_memory()` 装进来：这一层不认识配置文件，也不负责落盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PyQt5.QtWidgets import QFileDialog

from ...logging_setup import get_logger

logger = get_logger("dance.gui.dialog")

VIDEO_FILTER = "视频 (*.mp4 *.mov *.mkv *.avi *.flv);;所有文件 (*)"
AUDIO_FILTER = "音频/视频 (*.wav *.mp3 *.m4a *.flac *.aac *.mp4 *.mov);;所有文件 (*)"

#: 用系统对话框还是 Qt 自绘。由 `configure()` 按 `dance.native_dialogs` 设置
_native = True
#: `{用途: 上次的目录}`。窗口装进来的那个字典就是设置文件里的一块，改了它就等于改设置
_recent: dict[str, Any] = {}
_flush: Callable[[], None] | None = None


def configure(native: bool = True) -> None:
    global _native                       # noqa: PLW0603 - 模块级开关，全界面共用一个
    _native = bool(native)


def install_memory(store: dict[str, Any], flush: Callable[[], None] | None = None) -> None:
    """把"上次选的目录"这块记忆挂上来。`flush` 用于选完之后立刻落盘。"""
    global _recent, _flush               # noqa: PLW0603
    _recent = store if isinstance(store, dict) else {}
    _flush = flush


def options():
    """`DontResolveSymlinks` 两种模式都加：解析快捷方式可能去访问失效的网络路径。"""
    flags = QFileDialog.DontResolveSymlinks
    if not _native:
        flags |= QFileDialog.DontUseNativeDialog
    return flags


def start_dir(folder, key: str = "") -> str:
    """起始目录：先看这个用途上次去过哪儿，再看调用方给的默认目录，最后退回主目录。

    空字符串要当"没给"处理 —— `Path("")` 会变成 `.`（当前工作目录），
    那是启动程序时的随机目录，对用户毫无意义。
    """
    for candidate in (_recent.get(key) if key else None, folder):
        text = str(candidate or "").strip()
        if not text:
            continue
        try:
            target = Path(text)
            if target.is_dir():
                return str(target)
        except OSError as exc:
            logger.debug("起始目录探不动 %s：%s", candidate, exc)
    return str(Path.home())


def remember(key: str, path: str | Path, *, is_dir: bool = False) -> None:
    """把这次选的位置记下来（文件记它所在的目录）。"""
    if not key or not path:
        return
    try:
        target = Path(path)
        folder = target if is_dir else target.parent
        if not folder.is_dir():
            return
        if _recent.get(key) == str(folder):
            return
        _recent[key] = str(folder)
    except OSError as exc:
        logger.debug("记不住目录 %s：%s", path, exc)
        return
    if _flush is not None:
        _flush()


def open_file(parent, title: str, folder, filters: str, key: str = "") -> str:
    logger.info("[选文件] %s：从 %s 开始", title, start_dir(folder, key))
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(folder, key), filters,
                                          options=options())
    logger.info("[选文件] %s：%s", title, path or "（取消）")
    remember(key, path)
    return path


def open_files(parent, title: str, folder, filters: str, key: str = "") -> list[str]:
    logger.info("[选文件] %s：从 %s 开始", title, start_dir(folder, key))
    paths, _ = QFileDialog.getOpenFileNames(parent, title, start_dir(folder, key), filters,
                                            options=options())
    logger.info("[选文件] %s：选了 %d 个", title, len(paths))
    if paths:
        remember(key, paths[0])
    return list(paths)


def open_dir(parent, title: str, folder, key: str = "") -> str:
    logger.info("[选目录] %s：从 %s 开始", title, start_dir(folder, key))
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(folder, key),
                                            options=options() | QFileDialog.ShowDirsOnly)
    logger.info("[选目录] %s：%s", title, path or "（取消）")
    remember(key, path, is_dir=True)
    return path


def save_file(parent, title: str, folder, filters: str, key: str = "") -> str:
    logger.info("[存文件] %s：从 %s 开始", title, start_dir(folder, key))
    path, _ = QFileDialog.getSaveFileName(parent, title, start_dir(folder, key), filters,
                                          options=options())
    logger.info("[存文件] %s：%s", title, path or "（取消）")
    remember(key, path)
    return path


__all__ = ["VIDEO_FILTER", "AUDIO_FILTER", "configure", "install_memory", "options",
           "start_dir", "remember", "open_file", "open_files", "open_dir", "save_file"]
