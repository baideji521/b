"""Source ↔ Target 双时间轴 + 缩略波形。Qt 自己画，不引入 PyQtGraph。

这个控件回答一个问题，而且只回答这一个：

    **两条音轨是怎么错开的，错开之后目标歌的这一格落在源视频的哪一段。**

画法：横轴统一是**目标歌时间**（它才是那把尺子）。源视频那条轨按
`source_time = target_time - offset` 反向摆放 —— 也就是整条源波形往右平移 offset 秒。
于是"两条波形上下对齐"这件事在视觉上就等价于"offset 求对了"，
用户不用理解互相关也能一眼判断。

只画包络（`dsp.envelope` 算好的 0~1 数组），不做 DAW 那种可缩放波形编辑器：
这里要的是"能看出错开多少"，不是逐采样修剪。
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .. import theme

#: 每条横带的高度（像素）。轨名、刻度、offset 说明各占一条，谁也不压谁 ——
#: 之前把它们画在同一片区域里，文字互相叠成一团，等于白画
LABEL_HEIGHT = 15
TICK_BAND = 20
CAPTION_HEIGHT = 18


class DualTimeline(QWidget):
    """上：目标歌。下：源视频（已按 offset 平移）。点/拖 → `moved(目标秒)`。"""

    moved = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(2 * LABEL_HEIGHT + TICK_BAND + CAPTION_HEIGHT + 2 * 40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.CrossCursor)

        self._offset = 0.0
        self._target_duration = 0.0
        self._source_duration = 0.0
        self._target_env: list[float] = []
        self._source_env: list[float] = []
        self._marker = 0.0                 # 当前点（目标歌时间）
        self._span: tuple[float, float] | None = None   # 正在测试的那一格
        self._playing: float | None = None  # 播放头（源视频时间）

    # ------------------------------------------------------------------ 输入
    def set_alignment(self, offset: float, target_duration: float,
                      source_duration: float) -> None:
        self._offset = float(offset)
        self._target_duration = max(0.0, float(target_duration))
        self._source_duration = max(0.0, float(source_duration))
        self.update()

    def set_envelopes(self, target: list[float], source: list[float]) -> None:
        self._target_env = [float(v) for v in (target or [])]
        self._source_env = [float(v) for v in (source or [])]
        self.update()

    def set_marker(self, target_time: float) -> None:
        self._marker = max(0.0, float(target_time))
        self.update()

    def set_span(self, target_start: float | None, target_end: float | None = None) -> None:
        self._span = (None if target_start is None
                      else (float(target_start), float(target_end or target_start)))
        self.update()

    def set_playhead(self, source_time: float | None) -> None:
        """播放头给的是**源视频**时间（播放器只认源），画的时候再换算回目标轴。"""
        self._playing = None if source_time is None else float(source_time)
        self.update()

    def clear(self) -> None:
        self._offset = 0.0
        self._target_duration = self._source_duration = 0.0
        self._target_env = []
        self._source_env = []
        self._marker = 0.0
        self._span = None
        self._playing = None
        self.update()

    # ------------------------------------------------------------------ 坐标
    def _axis_span(self) -> tuple[float, float]:
        """横轴范围（目标歌时间）。要能同时装下目标歌和平移后的源视频。"""
        left = min(0.0, self._offset)
        right = max(self._target_duration, self._source_duration + self._offset)
        return left, right if right > left else left + 1.0

    def _x_of(self, moment: float) -> float:
        left, right = self._axis_span()
        pad = 6.0
        usable = max(1.0, self.width() - 2 * pad)
        return pad + (float(moment) - left) / (right - left) * usable

    def _time_of(self, x: float) -> float:
        left, right = self._axis_span()
        pad = 6.0
        usable = max(1.0, self.width() - 2 * pad)
        return left + (float(x) - pad) / usable * (right - left)

    # ------------------------------------------------------------------ 交互
    def mousePressEvent(self, event) -> None:                # noqa: N802 - Qt 的名字
        self._emit_at(event.x())

    def mouseMoveEvent(self, event) -> None:                 # noqa: N802
        if event.buttons() & Qt.LeftButton:
            self._emit_at(event.x())

    def _emit_at(self, x: float) -> None:
        if self._target_duration <= 0:
            return
        moment = min(max(0.0, self._time_of(x)), self._target_duration)
        self.set_marker(moment)
        self.moved.emit(round(moment, 3))

    # ------------------------------------------------------------------ 画
    def paintEvent(self, event) -> None:                     # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(theme.VIDEO_BG))
        if self._target_duration <= 0 and self._source_duration <= 0:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "对齐之后这里会画出两条音轨")
            return

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        # 每样东西一条独立的横带，谁也不许压在谁上面：
        # 轨名 / 波形 / 刻度 / 轨名 / 波形 / offset 说明
        width = float(self.width())
        lane = max(26.0, (self.height() - 2 * LABEL_HEIGHT
                          - TICK_BAND - CAPTION_HEIGHT - 8) / 2.0)
        top_label = QRectF(0, 2, width, LABEL_HEIGHT)
        top = QRectF(0, top_label.bottom(), width, lane)
        tick_top, tick_bottom = top.bottom(), top.bottom() + TICK_BAND
        bottom_label = QRectF(0, tick_bottom, width, LABEL_HEIGHT)
        bottom = QRectF(0, bottom_label.bottom(), width, lane)
        caption = QRectF(0, bottom.bottom() + 2, width, CAPTION_HEIGHT)

        self._label(painter, top_label, "TARGET 目标歌", QColor(theme.ACCENT))
        self._lane(painter, top, self._target_env, 0.0, self._target_duration,
                   QColor(theme.ACCENT))
        self._label(painter, bottom_label,
                    "SOURCE 源视频（已按 offset 平移）", QColor(theme.PLAYING))
        self._lane(painter, bottom, self._source_env, self._offset,
                   self._source_duration, QColor(theme.PLAYING))
        self._ticks(painter, top, bottom, tick_top, tick_bottom)
        self._overlay(painter, top, bottom)

        painter.setPen(QColor(theme.TEXT))
        painter.drawText(caption, Qt.AlignCenter,
                         f"OFFSET {self._offset:+.3f}s　　source = target − offset")

    def _label(self, painter: QPainter, box: QRectF, text: str, color: QColor) -> None:
        painter.setPen(color)
        painter.drawText(box.adjusted(6, 0, -6, 0), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _lane(self, painter: QPainter, box: QRectF, env: list[float], shift: float,
              duration: float, color: QColor) -> None:
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(0, 0, -1, -1))
        if duration <= 0:
            return
        begin, end = self._x_of(shift), self._x_of(shift + duration)
        painter.fillRect(QRectF(begin, box.top() + 1, max(1.0, end - begin),
                                box.height() - 2), QColor(theme.PANEL))
        if not env:
            return
        painter.setPen(QPen(color, 1))
        middle = box.center().y()
        half = box.height() / 2.0 - 3.0
        step = (end - begin) / float(len(env))
        for index, value in enumerate(env):
            x = begin + index * step
            tall = max(1.0, value * half)
            painter.drawLine(int(x), int(middle - tall), int(x), int(middle + tall))

    def _ticks(self, painter: QPainter, top: QRectF, bottom: QRectF,
               band_top: float, band_bottom: float) -> None:
        """刻度：竖线穿过两条轨，秒数写在中间那条带里（带自己的底色，不和竖线糊在一起）。"""
        left, right = self._axis_span()
        span = right - left
        step = 1.0
        for candidate in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0):
            step = candidate
            if span / candidate <= 12:
                break
        moment = float(int(left / step) * step)
        while moment <= right:
            x = self._x_of(moment)
            painter.setPen(QPen(QColor(theme.LINE), 1))
            painter.drawLine(int(x), int(top.top()), int(x), int(bottom.bottom()))
            box = QRectF(x - 26, band_top + 1, 52, band_bottom - band_top - 2)
            painter.fillRect(box, QColor(theme.VIDEO_BG))
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box, Qt.AlignCenter, f"{moment:g}s")
            moment = round(moment + step, 6)


    def _overlay(self, painter: QPainter, top: QRectF, bottom: QRectF) -> None:
        """当前点、正在测试的那一格、播放头。"""
        if self._span is not None:
            begin, end = self._span
            x0, x1 = self._x_of(begin), self._x_of(end)
            painter.fillRect(QRectF(x0, top.top(), max(1.0, x1 - x0),
                                    bottom.bottom() - top.top()),
                             QColor(217, 164, 65, 48))
            painter.setPen(QPen(QColor(theme.ACCENT), 1, Qt.DashLine))
            painter.drawRect(QRectF(x0, top.top(), max(1.0, x1 - x0),
                                    bottom.bottom() - top.top()))
        if self._target_duration > 0:
            x = self._x_of(self._marker)
            painter.setPen(QPen(QColor(theme.TEXT), 1))
            painter.drawLine(int(x), int(top.top()), int(x), int(bottom.bottom()))
        if self._playing is not None:
            # 播放头是源视频时间，换回目标轴要**加**回 offset（口径的反向）
            x = self._x_of(self._playing + self._offset)
            painter.setPen(QPen(QColor(theme.DONE), 2))
            painter.drawLine(int(x), int(bottom.top()), int(x), int(bottom.bottom()))


__all__ = ["LABEL_HEIGHT", "TICK_BAND", "CAPTION_HEIGHT", "DualTimeline"]
