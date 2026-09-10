"""视频位置：**一条**总览带，画已对齐视频在主音频上的覆盖范围。

用户明确不要"每个 mp4 一条时间轴"那种界面 —— 十个视频十条轴，屏幕全占满、
还得一条条对着看。这里只有一条带子：横轴还是主音频那条唯一时间轴，
每个已对齐视频在它上面盖出一段（`offset` 决定从哪儿盖到哪儿），叠在一起就能
一眼看出"这首歌的哪一段有素材、哪一段没人拍到"。

它只画，不改任何数据；点一下报告点到的秒数（`seeked`），交给上层去 seek。
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .. import theme

PAD = 8.0
BAR_HEIGHT = 34


class VideoCoverageBar(QWidget):
    """一条带子看完所有已对齐视频的位置。`set_coverage()` 喂数据。"""

    seeked = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(BAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("视频位置：每个已对齐视频在主音频上盖住的范围（点一下可以跳过去）")
        self._duration = 0.0
        self._items: list[tuple[float, float, str]] = []
        self._playhead: float | None = None

    def set_coverage(self, duration: float, items) -> None:
        """`items` = `[(start, end, 名字), …]`，秒数都在**主音频**的时间轴上。"""
        self._duration = max(0.0, float(duration))
        self._items = [(float(a), float(b), str(c)) for a, b, c in items or ()]
        self.update()

    def set_playhead(self, moment: float | None) -> None:
        self._playhead = None if moment is None else float(moment)
        self.update()

    def _x_of(self, moment: float) -> float:
        span = self._duration if self._duration > 0 else 1.0
        usable = max(1.0, self.width() - 2 * PAD)
        return PAD + max(0.0, min(span, float(moment))) / span * usable

    def mousePressEvent(self, event) -> None:                # noqa: N802 - Qt 的名字
        if self._duration <= 0 or event.button() != Qt.LeftButton:
            return
        usable = max(1.0, self.width() - 2 * PAD)
        moment = (event.x() - PAD) / usable * self._duration
        self.seeked.emit(round(max(0.0, min(self._duration, moment)), 3))

    def paintEvent(self, event) -> None:                     # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.VIDEO_BG))
        box = QRectF(0, 1, float(self.width()), float(self.height() - 2))
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        if self._duration <= 0 or not self._items:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box, Qt.AlignCenter, "视频位置：还没有对齐好的源视频")
            return
        # 每个视频盖一层半透明：叠得越厚 = 这一段候选越多
        for start, end, name in self._items:
            x0, x1 = self._x_of(start), self._x_of(end)
            painter.fillRect(QRectF(x0, box.top() + 4, max(2.0, x1 - x0), box.height() - 9),
                             QColor(90, 150, 220, 60))
            if x1 - x0 > 60:
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(QRectF(x0 + 3, box.top(), x1 - x0 - 6, box.height()),
                                 Qt.AlignVCenter | Qt.AlignLeft, name)
        if self._playhead is not None:
            x = self._x_of(self._playhead)
            painter.setPen(QPen(QColor(theme.DONE), 2))
            painter.drawLine(int(x), int(box.top()), int(x), int(box.bottom()))


__all__ = ["VideoCoverageBar", "BAR_HEIGHT"]
