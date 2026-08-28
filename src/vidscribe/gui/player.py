"""基于 OpenCV 的逐帧播放器（画面）+ winsound（声音）。

为什么不用 QMediaPlayer 播视频：这台机器上 Qt 的 WMF/DirectShow 后端对 H.264 返回
InvalidMedia（ASCII 路径、8.3 短路径都试过），播放器完全不可用。
自己解码渲染的好处是定位帧级精确。

为什么声音也不用 QMediaPlayer：实测在这台机器上 `QMediaPlayer()` 会让进程直接崩掉
（无输出、退出码 1），不是返回错误码，没法用 try/except 兜住。所以声音走
`winsound.PlaySound`（Windows 自带，只认 PCM WAV，进程内不加载任何多媒体后端）。
代价是它只能"从头播"、没有音量/定位接口，于是：
- 音轨先用 PyAV 解成 wav（见 vidscribe/audio.py）
- 每次 play/seek 都从当前秒切出剩余片段再播（切一次毫秒级）
- 真实时间（perf_counter）是主时钟：画面每次 tick 按「起点 + 已过真实秒数」算出该显示第几帧，
  落后就 grab 跳帧。声音也按真实时间走，两边共用一个钟才不会越播越偏
"""

from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..audio import slice_wav
from . import theme


class FramePlayer(QWidget):
    positionChanged = pyqtSignal(float)   # 秒
    durationChanged = pyqtSignal(float)   # 秒
    stateChanged = pyqtSignal(bool)       # 是否正在播放
    audioFailed = pyqtSignal(str)         # 声音不可用时带原因

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.view = QLabel("把视频拖进来，或点左上角“打开视频”")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(
            f"background:{theme.VIDEO_BG}; color:{theme.TEXT_DIM};"
            f"border:1px solid {theme.LINE}; border-radius:6px;"
        )
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setMinimumSize(160, 120)  # 给小一点，左侧画面才能被拖窄
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
        self._frame_index = 0          # 当前显示的是第几帧
        self._clock_origin = 0.0       # 本次播放的真实时间起点
        self._clock_base = 0.0         # 起点对应的视频位置（秒）
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._audio_wav: Path | None = None
        self._audio_cut: Path | None = None
        self._audio_on = False



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
        # 每帧时长的一半跑一次：定时器抖动就不会攒成掉帧，实际推进由真实时间决定
        self._timer.setInterval(max(5, int(500.0 / self._fps)))
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
        self._frame_index = 0
        self._clear_audio()

    # ------------------------------------------------------------------ 声音
    @staticmethod
    def _winsound():
        try:
            import winsound  # noqa: PLC0415
        except Exception:
            return None
        return winsound

    def _stop_audio(self) -> None:
        sound = self._winsound()
        if sound is None:
            return
        try:
            sound.PlaySound(None, sound.SND_PURGE)
        except Exception:
            pass

    def _start_audio(self, position: float) -> None:
        """从 position 秒开始放声音：先切片再异步播。"""
        sound = self._winsound()
        if sound is None or self._audio_wav is None or not self._audio_on:
            return
        cut = self._audio_cut or self._audio_wav.with_name(self._audio_wav.stem + "_cut.wav")
        if slice_wav(self._audio_wav, cut, position) is None:
            self._audio_on = False
            self.audioFailed.emit("音轨切分失败")
            return
        self._audio_cut = cut
        try:
            sound.PlaySound(str(cut), sound.SND_FILENAME | sound.SND_ASYNC | sound.SND_NODEFAULT)
        except Exception as exc:
            self._audio_on = False
            self.audioFailed.emit(f"{type(exc).__name__}: {exc}")

    def _clear_audio(self) -> None:
        self._stop_audio()
        self._audio_wav = None
        if self._audio_cut is not None:
            try:
                self._audio_cut.unlink(missing_ok=True)
            except OSError:
                pass
            self._audio_cut = None

    def set_audio_file(self, path: str | Path | None) -> bool:
        """挂上音轨 wav。返回是否可用（不可用时调用方应禁用勾选框）。"""
        self._clear_audio()
        if not path or not Path(path).is_file():
            return False
        if self._winsound() is None:
            self.audioFailed.emit("这个系统没有 winsound，预览无法出声")
            return False
        self._audio_wav = Path(path)
        return True

    def audio_available(self) -> bool:
        return self._audio_wav is not None

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_on = bool(enabled) and self._audio_wav is not None
        if not self._audio_on:
            self._stop_audio()
        elif self._playing:
            self._start_audio(self._position)

    def audio_enabled(self) -> bool:
        return self._audio_on



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
        self._start_audio(self._position)
        # 声音切片要花几毫秒，等它真正开播之后再对表，画面才不会一上来就超前
        self._clock_origin = time.perf_counter()
        self._clock_base = self._position
        self._timer.start()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self._stop_audio()
        self.stateChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek(self, seconds: float) -> None:
        if self._cap is None or self._cv2 is None:
            return
        cv2 = self._cv2
        seconds = max(0.0, seconds if self._duration <= 0 else min(seconds, max(self._duration - 0.05, 0.0)))
        # 用 floor：跳到「这一秒正在显示的那一帧」，和 video_io.plan_frame_indices
        # 以及高光剪辑的起剪帧号用同一套换算，点时间轴才不会差一帧
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * self._fps))
        self._render_next()
        # 声音没有定位接口，只能停掉再从新位置切片重播
        self._stop_audio()
        if self._playing:
            self._start_audio(self._position)
        self._clock_origin = time.perf_counter()
        self._clock_base = self._position

    # ------------------------------------------------------------------ 渲染
    def _on_tick(self) -> None:
        if self._cap is None:
            return
        # 画面对齐真实时间的绝对钟：目标帧 = (播放起点位置 + 已过真实秒数) * fps。
        # 不能按「上一次 tick 到现在过了几帧」四舍五入推进——余下的零头会被丢掉，
        # 定时器每次晚一点就攒成系统性慢放，而声音是独立按真实时间走的，于是越播越不同步。
        target = int((self._clock_base + (time.perf_counter() - self._clock_origin)) * self._fps)
        if target <= self._frame_index:
            return  # 还没到下一帧，这轮不动画面
        skip = target - self._frame_index - 1
        if skip > 30:  # 卡了太久，直接跳过去，别一帧一帧啃
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, target)
        else:
            for _ in range(skip):
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
        self._frame_index = int(current_index)
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
