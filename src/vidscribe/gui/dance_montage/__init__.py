"""AI_卡点舞 独立界面。

入口只有一个：`launch(cfg)`。它自己建 QApplication（如果还没有），
所以既能被 `vidscribe dance-montage gui` 直接开，也能被主界面按钮拉起来。

**独立**是有意的（技术指导第二十节）：这个窗口和主界面不共享 QThread、
不共享库连接、不共享任何全局状态。哪边出问题都不牵连另一边。
"""

from __future__ import annotations

from ...logging_setup import get_logger

logger = get_logger("dance.gui")


def launch(cfg, parent=None) -> int:
    """开窗。`parent` 非空表示是被主界面拉起来的，那就不再自己起事件循环。"""
    from PyQt5.QtWidgets import QApplication

    from .main_page import DanceMontageWindow

    app = QApplication.instance()
    owns = app is None
    if owns:
        import sys

        app = QApplication(sys.argv[:1])
        try:
            from .. import theme

            theme.apply(app)
        except Exception as exc:  # noqa: BLE001 - 没主题也要能开
            logger.debug("主题没加载上：%s", exc)
        _install_excepthook()

    window = DanceMontageWindow(cfg, parent)
    window.show()
    if not owns:
        # 被主界面拉起来：挂在 parent 上防止被 GC，事件循环用现成的那个
        if parent is not None:
            setattr(parent, "_dance_window", window)
        return 0
    return int(app.exec_())


def _install_excepthook() -> None:
    """槽函数里抛出的异常必须能看见。

    PyQt5 里槽抛异常会直接 abort 整个进程；而 `run_kadian.bat` 用的是
    `pythonw.exe`（没有控制台），于是界面"啪"一下就没了，什么都看不到 ——
    这类故障根本没法查。装上钩子之后：完整堆栈进日志文件，界面上弹一个框，
    窗口继续开着。主界面（`gui/main_window.launch`）早就这么干了，这里补齐。
    """
    import sys
    import traceback

    def on_error(kind, value, trace) -> None:
        text = "".join(traceback.format_exception(kind, value, trace))
        logger.error("界面异常：%s", text)
        try:
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.critical(None, "AI_卡点舞 出错了",
                                 f"{kind.__name__}: {value}\n\n"
                                 f"完整堆栈已写进日志文件。\n\n{text[-1500:]}")
        except Exception:  # noqa: BLE001 - 弹不出框也不能再抛，否则一样静默死
            logger.error("连错误提示框都弹不出来")

    sys.excepthook = on_error



__all__ = ["launch"]
