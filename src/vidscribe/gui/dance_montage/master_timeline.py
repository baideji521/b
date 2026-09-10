"""MASTER AUDIO 编辑区的时间轴：音谱图 + 波形 + 人声带 + ⏸ 停顿 + 段落边界 + 播放头。

一条横轴，几条互不重叠的横带（这条规矩和 `align_timeline` 一样：谁也不许压在谁上面）：

    刻度 ── 音谱图 ── 波形 ── 人声/停顿 ── 段落 S1|S2|S3 ── （播放头贯穿全部）

**它只负责画和报告，不改任何数据。** 拖动边界时发 `boundary_dragged`，
松手发 `boundary_committed`，真正改模板的是面板（走 `segment_template.move_boundary`，
拖不过去就报错）。这样"合法的分段"只有一个判定处，控件里不会藏第二套规则。

音谱图是 `dsp.spectrogram_image` 算好的 uint8 小图，这里只做一次 QImage 包装 + 缩放；
逐帧 FFT 一律在后台线程算完再送进来，主线程一次都不碰音频。
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .. import theme

#: 各条横带的高度（像素）
TICK_BAND = 20
SPECTRUM_HEIGHT = 92
WAVE_HEIGHT = 44
VOCAL_HEIGHT = 30
ZONE_HEIGHT = 22
SEGMENT_HEIGHT = 36
#: 左右留白：边界拖到最边上也得有地方下手
PAD = 8.0
#: 鼠标离边界这么近（像素）就算"抓住了它"
GRAB_PIXELS = 6.0
#: 缩放能看到的最短窗口（秒）。再细下去一屏放不下一个字
MIN_VIEW_SECONDS = 2.0


class MasterTimeline(QWidget):
    """主音频时间轴。点空白 → `seeked`；拖段落边界 → `boundary_dragged` / `boundary_committed`。"""

    seeked = pyqtSignal(float)
    boundary_dragged = pyqtSignal(int, float)
    boundary_committed = pyqtSignal(int, float)
    segment_clicked = pyqtSignal(int)
    view_changed = pyqtSignal(float, float)      # 可见窗口 (起点, 跨度)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(TICK_BAND + SPECTRUM_HEIGHT + WAVE_HEIGHT
                              + VOCAL_HEIGHT + SEGMENT_HEIGHT + 10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._duration = 0.0
        self._image: QImage | None = None
        self._raw = b""                     # QImage 不持有数据，得自己留着
        self._env: list[float] = []
        self._vocal: list[tuple[float, float, str]] = []
        self._pauses: list[tuple[float, float, float]] = []   # start, end, score
        self._spans: list[tuple[float, float, str]] = []
        self._zones: list[tuple[float, float, float, str]] = []
        self._marks: list[float] = []
        self._beats: list[float] = []
        self._snap: list[float] = []
        self._playhead: float | None = None
        self._cursor_time = 0.0
        self._dragging = -1                 # 正在拖第几个内部分割点，-1 = 没拖
        self._panning: tuple[int, float] | None = None   # 抓着音轨挪：(按下时的 x, 那会儿的窗口起点)
        self._view_start = 0.0              # 可见窗口起点（秒）
        self._view_span = 0.0               # 可见窗口跨度；0 = 看全曲


    # ------------------------------------------------------------------ 输入
    def set_song(self, duration: float, envelope: list[float] | None = None) -> None:
        self._duration = max(0.0, float(duration))
        if envelope is not None:
            self._env = [float(v) for v in envelope]
        self.update()

    def set_spectrogram(self, image) -> None:
        """`image` 是 `dsp.spectrogram_image` 的 (rows, columns) uint8 数组，或 None。"""
        if image is None or getattr(image, "size", 0) == 0:
            self._image, self._raw = None, b""
        else:
            rows, columns = int(image.shape[0]), int(image.shape[1])
            # ascontiguousarray：QImage 按行读裸内存，非连续数组会画出斜纹
            data = np.ascontiguousarray(image, dtype=np.uint8)
            self._raw = data.tobytes()
            self._image = QImage(self._raw, columns, rows, columns,
                                 QImage.Format_Grayscale8)
        self.update()

    def set_vocal(self, spans, pauses) -> None:
        """`spans` 是 `VocalSpan` 序列，`pauses` 是 `VocalPause` 序列。"""
        self._vocal = [(float(s.start), float(s.end), str(s.kind)) for s in spans or ()]
        self._pauses = [(float(p.start), float(p.end), float(p.score))
                        for p in pauses or ()]
        self._refresh_snap()
        self.update()

    def set_template(self, template) -> None:
        """`template` 是 `SegmentTemplate`，None 表示还没分段。"""
        self._spans = ([(float(s.start), float(s.end), s.name) for s in template.spans]
                       if template is not None else [])
        self.update()

    def set_beats(self, beats) -> None:
        self._beats = [float(b) for b in beats or ()]
        self._refresh_snap()
        self.update()

    def set_zones(self, zones) -> None:
        """可取区间：`[(start, end, score, level), …]`，来自 `vocal_activity.cut_zones`。"""
        self._zones = [(float(a), float(b), float(s), str(t)) for a, b, s, t in zones or ()]
        self.update()

    def set_marks(self, moments) -> None:
        """用户「⭐ 标记为可取」的时刻（库里读出来的）。"""
        self._marks = sorted(float(m) for m in moments or ())
        self.update()

    # ------------------------------------------------------------ 缩放/滚动
    def set_view(self, start: float, span: float) -> None:
        """设可见窗口。`span<=0` 或大于全曲 = 看全曲。所有横带共用它，绝不各画一套。"""
        total = self._duration
        span = 0.0 if span <= 0 or (total > 0 and span >= total) else max(MIN_VIEW_SECONDS,
                                                                          float(span))
        if span <= 0:
            self._view_start, self._view_span = 0.0, 0.0
        else:
            self._view_span = span
            self._view_start = max(0.0, min(float(start), max(0.0, total - span)))
        self.view_changed.emit(self._view_start, self._view_span)
        self.update()

    def view(self) -> tuple[float, float]:
        """`(起点, 跨度)`；跨度 0 表示全曲。"""
        return self._view_start, self._view_span

    def visible_span(self) -> tuple[float, float]:
        """真正画出来的那一段 `(左, 右)`。"""
        if self._view_span <= 0:
            return 0.0, self._duration if self._duration > 0 else 1.0
        return self._view_start, self._view_start + self._view_span

    def zoom(self, factor: float, center: float | None = None) -> None:
        """缩放。`factor<1` 放大（窗口变短），围绕 `center`（默认当前光标）不动。"""
        left, right = self.visible_span()
        span = (right - left) * max(0.05, float(factor))
        pivot = self._cursor_time if center is None else float(center)
        self.set_view(pivot - span / 2.0, span)

    def scroll_to(self, start: float) -> None:
        self.set_view(float(start), self._view_span)

    def ensure_visible(self, moment: float) -> None:
        """播放头跑出可见范围时把窗口挪过去（自动跟随）。全曲视图什么都不用做。"""
        if self._view_span <= 0:
            return
        left, right = self.visible_span()
        if left <= float(moment) <= right:
            return
        self.set_view(float(moment) - self._view_span * 0.25, self._view_span)

    def set_playhead(self, moment: float | None) -> None:
        self._playhead = None if moment is None else max(0.0, float(moment))
        self.update()

    def set_cursor_time(self, moment: float) -> None:
        self._cursor_time = max(0.0, float(moment))
        self.update()

    def clear(self) -> None:
        self._duration = 0.0
        self._image, self._raw = None, b""
        self._env, self._vocal, self._pauses, self._spans = [], [], [], []
        self._zones, self._marks = [], []
        self._beats, self._snap = [], []
        self._playhead, self._cursor_time, self._dragging = None, 0.0, -1
        self._panning = None
        self._view_start, self._view_span = 0.0, 0.0
        self.update()

    def _refresh_snap(self) -> None:
        """吸附参考点 = 拍点 + 每个停顿的正中间。拖边界时往这些点上贴。"""
        self._snap = sorted(self._beats + [(s + e) / 2.0 for s, e, _ in self._pauses])

    @property
    def snap_points(self) -> list[float]:
        return list(self._snap)

    # ------------------------------------------------------------------ 坐标
    def _x_of(self, moment: float) -> float:
        left, right = self.visible_span()
        span = max(1e-6, right - left)
        usable = max(1.0, self.width() - 2 * PAD)
        return PAD + (float(moment) - left) / span * usable

    def _time_of(self, x: float) -> float:
        left, right = self.visible_span()
        span = max(1e-6, right - left)
        usable = max(1.0, self.width() - 2 * PAD)
        moment = left + (float(x) - PAD) / usable * span
        top = self._duration if self._duration > 0 else right
        return min(top, max(0.0, moment))

    def _lanes(self) -> dict[str, QRectF]:
        """从上到下切带。多出来的高度按 6:2:1:1 分给音谱图/波形/人声/段落 ——
        音谱图最值得看，但段落条也得跟着长高一点，否则边界细得抓不住。
        """
        width = float(self.width())
        spare = max(0.0, self.height() - TICK_BAND - SPECTRUM_HEIGHT - WAVE_HEIGHT
                    - VOCAL_HEIGHT - ZONE_HEIGHT - SEGMENT_HEIGHT - 12)
        top = 2.0
        tick = QRectF(0, top, width, TICK_BAND)
        spectrum = QRectF(0, tick.bottom(), width, SPECTRUM_HEIGHT + spare * 0.6)
        wave = QRectF(0, spectrum.bottom() + 2, width, WAVE_HEIGHT + spare * 0.2)
        vocal = QRectF(0, wave.bottom() + 2, width, VOCAL_HEIGHT + spare * 0.1)
        zone = QRectF(0, vocal.bottom() + 2, width, ZONE_HEIGHT)
        segment = QRectF(0, zone.bottom() + 2, width, SEGMENT_HEIGHT + spare * 0.1)
        return {"tick": tick, "spectrum": spectrum, "wave": wave,
                "vocal": vocal, "zone": zone, "segment": segment}

    def _boundary_at(self, x: float, y: float) -> int:
        """鼠标底下是第几个内部分割点？不在段落带上、或没抓到就返回 -1。"""
        lanes = self._lanes()
        if not lanes["segment"].adjusted(0, -4, 0, 4).contains(x, y):
            return -1
        for index in range(1, len(self._spans)):
            if abs(self._x_of(self._spans[index][0]) - x) <= GRAB_PIXELS:
                return index - 1
        return -1

    # ------------------------------------------------------------------ 交互
    def pan_by(self, pixels: float) -> None:
        """把音轨横着挪 `pixels` 个像素（正数 = 内容往右走，看到的是更早的地方）。

        缩放倍率不动，只搬窗口起点 —— 看长卷轴就是这个手感。全曲视图没得可挪。
        """
        if self._view_span <= 0 or self._duration <= 0:
            return
        left, right = self.visible_span()
        per_pixel = (right - left) / max(1, self.width())
        self.scroll_to(self._view_start - float(pixels) * per_pixel)

    def mousePressEvent(self, event) -> None:                # noqa: N802 - Qt 的名字
        if self._duration <= 0:
            return
        if (event.button() == Qt.MiddleButton
                or (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier)):
            # 中键拖（或 Alt+左键拖）＝ 抓着音轨左右挪。左键单独用来定位/拖边界，
            # 所以挪动得另给一个键，不然一拖就跳播放位置
            self._panning = (event.x(), self._view_start)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        grabbed = self._boundary_at(event.x(), event.y())
        if grabbed >= 0:
            self._dragging = grabbed
            return
        moment = self._time_of(event.x())
        lanes = self._lanes()
        if lanes["segment"].contains(event.x(), event.y()):
            for index, (start, end, _name) in enumerate(self._spans):
                if start <= moment < end:
                    self.segment_clicked.emit(index)
                    break
        self.set_cursor_time(moment)
        self.seeked.emit(round(moment, 3))

    def mouseMoveEvent(self, event) -> None:                 # noqa: N802
        if self._panning is not None:
            grabbed_x, _start = self._panning
            self.pan_by(event.x() - grabbed_x)
            self._panning = (event.x(), self._view_start)   # 一段一段挪，免得越拖越飘
            return
        if self._dragging >= 0:
            self.boundary_dragged.emit(self._dragging, round(self._time_of(event.x()), 3))
            return
        near = self._boundary_at(event.x(), event.y()) >= 0
        self.setCursor(Qt.SplitHCursor if near else Qt.CrossCursor)

    def mouseReleaseEvent(self, event) -> None:              # noqa: N802
        if self._panning is not None:
            self._panning = None
            self.setCursor(Qt.CrossCursor)
            return
        if self._dragging >= 0:
            index, self._dragging = self._dragging, -1
            self.boundary_committed.emit(index, round(self._time_of(event.x()), 3))


    # ------------------------------------------------------------------ 画
    def paintEvent(self, event) -> None:                     # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(theme.VIDEO_BG))
        if self._duration <= 0:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "选一首主音频，这里会画出音谱图、波形、人声和分段")
            return

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        lanes = self._lanes()
        self._draw_ticks(painter, lanes)
        self._draw_spectrum(painter, lanes["spectrum"])
        self._draw_wave(painter, lanes["wave"])
        self._draw_vocal(painter, lanes["vocal"])
        self._draw_zones(painter, lanes["zone"])
        self._draw_segments(painter, lanes["segment"])
        self._draw_heads(painter, lanes)

    def wheelEvent(self, event) -> None:                      # noqa: N802 - Qt 的名字
        """滚轮 = 缩放（围绕鼠标那一刻），Shift+滚轮 = 横向滚动。"""
        if self._duration <= 0:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        if event.modifiers() & Qt.ShiftModifier:
            left, right = self.visible_span()
            self.scroll_to(self._view_start - steps * (right - left) * 0.2)
            return
        self.zoom(0.8 if steps > 0 else 1.25, self._time_of(event.x()))

    def _draw_ticks(self, painter: QPainter, lanes: dict[str, QRectF]) -> None:
        left, right = self.visible_span()
        span = max(1e-6, right - left)
        step = 1.0
        for candidate in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0):
            step = candidate
            if span / candidate <= 14:
                break
        top, bottom = lanes["spectrum"].top(), lanes["segment"].bottom()
        moment = float(int(left / step) * step)
        while moment <= right:
            x = self._x_of(moment)
            painter.setPen(QPen(QColor(theme.LINE), 1))
            painter.drawLine(int(x), int(top), int(x), int(bottom))
            box = QRectF(x - 30, lanes["tick"].top(), 60, lanes["tick"].height())
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box, Qt.AlignCenter, _clock(moment))
            moment = round(moment + step, 6)

    def _draw_spectrum(self, painter: QPainter, box: QRectF) -> None:
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        inner = box.adjusted(PAD + 1, 1, -PAD - 1, -1)
        if self._image is None:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(inner, Qt.AlignCenter, "音谱图算好之后画在这里")
            return
        # 缩放时只画可见那一段的列：图是整曲的，源矩形按时间比例裁
        left, right = self.visible_span()
        total = self._duration if self._duration > 0 else max(1e-6, right)
        columns = float(self._image.width())
        x0 = max(0.0, min(columns, left / total * columns))
        x1 = max(x0 + 1.0, min(columns, right / total * columns))
        source = QRectF(x0, 0.0, x1 - x0, float(self._image.height()))
        painter.drawImage(inner, self._image, source)
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.drawText(inner.adjusted(4, 2, -4, 0), Qt.AlignLeft | Qt.AlignTop,
                         "音谱图（上=高频）")

    def _draw_zones(self, painter: QPainter, box: QRectF) -> None:
        """可取区间 + 用户的 ⭐ 标记。**只是参考**，不改任何分段。"""
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        if not self._zones and not self._marks:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box.adjusted(PAD + 4, 0, -PAD, 0),
                             Qt.AlignLeft | Qt.AlignVCenter, "可取区间：还没分析")
            return
        for start, end, score, level in self._zones:
            x0, x1 = self._x_of(start), self._x_of(end)
            alpha = 40 + int(min(1.0, max(0.0, score)) * 90)
            painter.fillRect(QRectF(x0, box.top() + 3, max(1.0, x1 - x0),
                                    box.height() - 6), QColor(90, 190, 120, alpha))
            if x1 - x0 > 46:
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(QRectF(x0, box.top(), x1 - x0, box.height()),
                                 Qt.AlignCenter, level)
        for moment in self._marks:
            x = self._x_of(moment)
            painter.setPen(QPen(QColor(theme.ACCENT), 2))
            painter.drawLine(int(x), int(box.top() + 1), int(x), int(box.bottom() - 1))
            painter.drawText(QRectF(x - 12, box.top(), 24, box.height()),
                             Qt.AlignCenter, "⭐")

    def _draw_wave(self, painter: QPainter, box: QRectF) -> None:
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        if not self._env:
            return
        painter.setPen(QPen(QColor(theme.ACCENT), 1))
        middle = box.center().y()
        half = box.height() / 2.0 - 3.0
        total = self._duration if self._duration > 0 else 1.0
        left, right = self.visible_span()
        buckets = len(self._env)
        # 只画可见那一段的桶：缩放之后照样是同一条时间轴，不会和别的带错开
        first = max(0, int(left / total * buckets) - 1)
        last = min(buckets, int(right / total * buckets) + 2)
        for index in range(first, last):
            moment = (index + 0.5) / buckets * total
            x = self._x_of(moment)
            tall = max(1.0, float(self._env[index]) * half)
            painter.drawLine(int(x), int(middle - tall), int(x), int(middle + tall))

    def _draw_vocal(self, painter: QPainter, box: QRectF) -> None:
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        if not self._vocal:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box.adjusted(PAD + 4, 0, -PAD, 0),
                             Qt.AlignLeft | Qt.AlignVCenter, "人声/停顿：还没分析")
            return
        for start, end, kind in self._vocal:
            if kind != "vocal":
                continue
            x0, x1 = self._x_of(start), self._x_of(end)
            painter.fillRect(QRectF(x0, box.top() + 4, max(1.0, x1 - x0),
                                    box.height() - 8), QColor(theme.PLAYING))
        # 停顿：画一条竖标 + ⏸，推荐度高的画实心（值得优先考虑的换人点）
        for start, end, score in self._pauses:
            middle = self._x_of((start + end) / 2.0)
            strong = score >= 0.6
            painter.setPen(QPen(QColor(theme.DONE if strong else theme.TEXT_DIM),
                                2 if strong else 1))
            painter.drawLine(int(middle), int(box.top() + 1),
                             int(middle), int(box.bottom() - 1))
            painter.drawText(QRectF(middle - 14, box.top(), 28, box.height()),
                             Qt.AlignCenter, "⏸")

    def _draw_segments(self, painter: QPainter, box: QRectF) -> None:
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawRect(box.adjusted(PAD, 0, -PAD, -1))
        if not self._spans:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(box.adjusted(PAD + 4, 0, -PAD, 0),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             "还没分段 —— 先「等间隔起步」或者「照停顿分」")
            return
        for index, (start, end, name) in enumerate(self._spans):
            x0, x1 = self._x_of(start), self._x_of(end)
            cell = QRectF(x0, box.top() + 2, max(1.0, x1 - x0), box.height() - 4)
            painter.fillRect(cell, QColor(217, 164, 65, 40 if index % 2 else 68))
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(cell, Qt.AlignCenter, name)
        # 边界手柄：粗一点，让人看出"这条线是能拖的"
        for index in range(1, len(self._spans)):
            x = self._x_of(self._spans[index][0])
            painter.setPen(QPen(QColor(theme.ACCENT), 3 if self._dragging == index - 1 else 2))
            painter.drawLine(int(x), int(box.top()), int(x), int(box.bottom()))

    def _draw_heads(self, painter: QPainter, lanes: dict[str, QRectF]) -> None:
        top, bottom = lanes["spectrum"].top(), lanes["segment"].bottom()
        x = self._x_of(self._cursor_time)
        painter.setPen(QPen(QColor(theme.TEXT), 1, Qt.DashLine))
        painter.drawLine(int(x), int(top), int(x), int(bottom))
        if self._playhead is not None:
            x = self._x_of(self._playhead)
            painter.setPen(QPen(QColor(theme.DONE), 2))
            painter.drawLine(int(x), int(top), int(x), int(bottom))


def _clock(seconds: float) -> str:
    """秒 → `分:秒.毫秒`。歌是分钟量级，纯秒数看着累。"""
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    return f"{minutes}:{total - minutes * 60:06.3f}"


__all__ = ["TICK_BAND", "SPECTRUM_HEIGHT", "WAVE_HEIGHT", "VOCAL_HEIGHT",
           "ZONE_HEIGHT", "SEGMENT_HEIGHT", "GRAB_PIXELS", "MIN_VIEW_SECONDS",
           "MasterTimeline"]




