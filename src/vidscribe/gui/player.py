"""基于 OpenCV 的逐帧播放器。

为什么不用 QMediaPlayer：这台机器上 Qt 的 WMF/DirectShow 后端对 H.264 返回
InvalidMedia（ASCII 路径、8.3 短路径都试过），播放器完全不可用。
自己解码渲染虽然没有声音，但定位是帧级精确的，而语音内容本来就在下方面板里。
"""

from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class FramePlayer(QWidget):
    positionChanged = pyqtSignal(float)   # 秒
    durationChanged = pyqtSignal(float)   # 秒
    stateChanged = pyqtSignal(bool)       # 是否正在播放

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.view = QLabel("打开视频后在这里预览")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background:#111; color:#888;")
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setMinimumSize(320, 240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self._cap = None
        self._cv2 = None
        self._fps = 25.0
        self._duration = 0.0
        self._position = 0.0
        self._image: QImage | None = None
        self._playing = False
        self._last_tick = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------ 打开
    def open(self, path: str | Path) -> bool:
        # cv2 只能在 QApplication 创建之后再导入：它会改写 QT_QPA_PLATFORM_PLUGIN_PATH
        if self._cv2 is None:
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
        cv2 = self._cv2

        self.close_video()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            self.view.setText("无法解码这个视频")
            return False
        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 0.1 else 25.0
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self._duration = round(frames / self._fps, 3) if frames > 0 else 0.0
        self._timer.setInterval(max(10, int(1000.0 / self._fps)))
        self.durationChanged.emit(self._duration)
        self.seek(0.0)
        return True

    def close_video(self) -> None:
        self.pause()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image = None
        self._position = 0.0

    # ------------------------------------------------------------------ 控制
    def is_playing(self) -> bool:
        return self._playing

    def duration(self) -> float:
        return self._duration

    def position(self) -> float:
        return self._position

    def play(self) -> None:
        if self._cap is None or self._playing:
            return
        self._playing = True
        self._last_tick = time.perf_counter()
        self._timer.start()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.stateChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek(self, seconds: float) -> None:
        if self._cap is None or self._cv2 is None:
            return
        cv2 = self._cv2
        seconds = max(0.0, seconds if self._duration <= 0 else min(seconds, max(self._duration - 0.05, 0.0)))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * self._fps)))
        self._last_tick = time.perf_counter()
        self._render_next()

    # ------------------------------------------------------------------ 渲染
    def _on_tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now
        # 按真实时间推进，落后时用 grab() 跳帧，保持音画之外的时间同步
        steps = max(1, min(int(round(elapsed * self._fps)), 5))
        for _ in range(steps - 1):
            if self._cap is not None:
                self._cap.grab()
        self._render_next()

    def _render_next(self) -> bool:
        if self._cap is None or self._cv2 is None:
            return False
        cv2 = self._cv2
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.pause()
            return False
        # 注意：seek 之后再读 POS_MSEC 之前的值是脏的，必须在 read() 之后按帧号算，
        # read() 之后 POS_FRAMES 指向下一帧，所以当前帧是 POS_FRAMES-1。
        next_index = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_index = max(next_index - 1.0, 0.0)
        self._position = round(current_index / self._fps, 3)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._repaint()
        self.positionChanged.emit(self._position)
        return True

    def _repaint(self) -> None:
        if self._image is None:
            return
        pix = QPixmap.fromImage(self._image).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint()
