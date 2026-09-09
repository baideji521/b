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

    window = DanceMontageWindow(cfg, parent)
    window.show()
    if not owns:
        # 被主界面拉起来：挂在 parent 上防止被 GC，事件循环用现成的那个
        if parent is not None:
            setattr(parent, "_dance_window", window)
        return 0
    return int(app.exec_())


__all__ = ["launch"]
