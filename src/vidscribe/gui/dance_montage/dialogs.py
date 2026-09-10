"""选文件对话框：**一律用 Qt 自己画的那个**，不用 Windows 原生的。

原因很实在：原生对话框会加载系统 shell 扩展（缩略图提供程序、网盘/杀软的右键菜单
插件、"最近使用"里失效的网络路径……），任何一个卡住，整个界面就跟着一起没响应，
而且卡在系统代码里，日志上一个字都看不到。Qt 自己那个只读文件系统，慢也慢不到哪儿去。
代价是长得朴素一点 —— 用得动比好看重要。

抽成模块是因为「混剪」和「音频对齐 / 卡点测试」两个面板都要选文件，
而这条规矩不能有第二份实现：漏一处就是一次界面卡死。
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QFileDialog

from ...logging_setup import get_logger

logger = get_logger("dance.gui.dialog")

VIDEO_FILTER = "视频 (*.mp4 *.mov *.mkv *.avi *.flv);;所有文件 (*)"
AUDIO_FILTER = "音频/视频 (*.wav *.mp3 *.m4a *.flac *.aac *.mp4 *.mov);;所有文件 (*)"


def options():
    return QFileDialog.DontUseNativeDialog | QFileDialog.DontResolveSymlinks


def start_dir(folder) -> str:
    """起始目录：不存在就退回用户主目录，别把对话框指到一个死路径上。

    空字符串要当"没给"处理 —— `Path("")` 会变成 `.`（当前工作目录），
    那是启动程序时的随机目录，对用户毫无意义。
    """
    text = str(folder or "").strip()
    if text:
        try:
            target = Path(text)
            if target.is_dir():
                return str(target)
        except OSError as exc:
            logger.debug("起始目录探不动 %s：%s", folder, exc)
    return str(Path.home())


def open_file(parent, title: str, folder, filters: str) -> str:
    logger.info("[选文件] %s：从 %s 开始", title, folder)
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(folder), filters,
                                          options=options())
    logger.info("[选文件] %s：%s", title, path or "（取消）")
    return path


def open_files(parent, title: str, folder, filters: str) -> list[str]:
    logger.info("[选文件] %s：从 %s 开始", title, folder)
    paths, _ = QFileDialog.getOpenFileNames(parent, title, start_dir(folder), filters,
                                            options=options())
    logger.info("[选文件] %s：选了 %d 个", title, len(paths))
    return list(paths)


def open_dir(parent, title: str, folder) -> str:
    logger.info("[选目录] %s：从 %s 开始", title, folder)
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(folder),
                                            options=options() | QFileDialog.ShowDirsOnly)
    logger.info("[选目录] %s：%s", title, path or "（取消）")
    return path


def save_file(parent, title: str, folder, filters: str) -> str:
    logger.info("[存文件] %s：从 %s 开始", title, folder)
    path, _ = QFileDialog.getSaveFileName(parent, title, start_dir(folder), filters,
                                          options=options())
    logger.info("[存文件] %s：%s", title, path or "（取消）")
    return path


__all__ = ["VIDEO_FILTER", "AUDIO_FILTER", "options", "start_dir",
           "open_file", "open_files", "open_dir", "save_file"]
