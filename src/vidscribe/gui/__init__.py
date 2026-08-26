"""GUI 子包（PyQt5）。

导入时先把 Qt 插件路径钉到 PyQt5 自带的目录：这台机器上系统里存在其他 Qt5，
不指定的话 QApplication 构造会直接崩（0xC0000409）。opencv-python 也会在 import
时改写 QT_QPA_PLATFORM_PLUGIN_PATH，所以 GUI 进程里刻意不导入 cv2。
"""

from __future__ import annotations

import os
from pathlib import Path


def configure_qt_plugin_path() -> None:
    try:
        import PyQt5  # noqa: PLC0415  # 只取路径，不会加载 Qt DLL
    except ImportError:
        return
    base = Path(PyQt5.__file__).resolve().parent / "Qt5"
    plugins = base / "plugins"
    if not plugins.is_dir():
        return
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")
    os.environ["QT_PLUGIN_PATH"] = str(plugins)
    bin_dir = base / "bin"
    if bin_dir.is_dir():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass


configure_qt_plugin_path()
