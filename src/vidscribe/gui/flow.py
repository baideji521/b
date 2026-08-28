"""横向自动换行的布局：窗口拖窄时工具栏按钮往下折行，而不是把窗口最小宽度顶死。

为什么需要它：工具栏那一排按钮 + 下拉框用 QHBoxLayout 时，最小宽度是所有控件宽度之和
（实测 2333px），窗口根本拉不窄。换成这个布局后最小宽度只取最宽的那一个控件。

实现照搬 Qt 官方 FlowLayout 例子：逐个摆放，摆不下就换行；heightForWidth 让外层布局
知道换行后需要多高。
"""

from __future__ import annotations

from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin: int = 0, spacing: int = 8):
        super().__init__(parent)
        self._items: list = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _arrange(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        space = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + space
            if next_x - space > area.right() and line_height > 0:
                x = area.x()
                y = y + line_height + space
                next_x = x + hint.width() + space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def wrap(layout: FlowLayout):
    """把 FlowLayout 装进容器控件，方便塞进 QVBoxLayout。"""
    return FlowBar(layout)


class FlowBar(QWidget):
    """FlowLayout 的容器：最小高度按“当前宽度”算，而不是按“最小宽度”算。

    为什么要这么写：QVBoxLayout 求最小高度时会调 heightForWidth(最小宽度)，工具栏在最窄
    状态下会折成十几行，于是窗口最小高度被顶到 800+，窗口反而变矮不了。这里按实际宽度算，
    再在 resizeEvent 里 updateGeometry()，宽度变窄时高度才会跟着涨、按钮不会被裁掉。
    """

    def __init__(self, layout: FlowLayout):
        super().__init__()
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def _hint(self) -> QSize:
        layout = self.layout()
        width = self.width() or layout.minimumSize().width()
        return QSize(layout.minimumSize().width(), layout.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._hint()

    def sizeHint(self) -> QSize:  # noqa: N802
        return self._hint()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.updateGeometry()
