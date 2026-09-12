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

**片段预解码**（`preload`）：卡点要看的是"这一刀到底压在鼓点上没有"，几秒的片段来回拖
十几遍，每次都让 cv2 重新定位 + 从关键帧啃回来 —— 这就是卡的来源。所以对已经知道
起止的片段，先把这几秒**逐帧解成内存里的 QImage 列表**，之后播放/定位/逐帧都只是
数组下标，主线程一帧都不解码。内存靠"解码时先缩到 `CACHE_EDGE` 以内 + 总量封顶"控制，
放不下就整段退回流式播放（不做半段缓存 —— 半段的行为没法跟用户解释）。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..audio import slice_wav
from ..logging_setup import get_logger
from . import theme

logger = get_logger(__name__)

#: 预取队列最多排这么多活。超了就丢旧的 —— 用户已经翻到别处去了
PREFETCH_QUEUE = 4

#: 预解码时把画面缩到最长边不超过这么多像素。3:4 竖屏 1080×1440 的一帧 RGB888 是 4.7MB，
#: 原尺寸缓存 2 秒（60 帧）就是 280MB —— 直接把机器吃穿。缩到 540 长边后一帧约 0.64MB
#: （405×540），而实时播放那个框本身也就三四百像素高，肉眼看不出差别。
#: 缩放用 INTER_NEAREST：实测 1080p→540 用 INTER_AREA 要 4.7ms/帧，NEAREST 只要 0.45ms，
#: 一个 2 秒片段就差 250ms —— 这点画质换的是"切段落时不卡一下"。
CACHE_EDGE = 540
#: 单段最多缓存多少帧（30fps 下 ≈ 20 秒）。卡点片段通常 1~5 秒，超过这个数说明
#: 调用方传的不是"一个片段"，那就别缓存了
CACHE_MAX_FRAMES = 600
#: 缓存总字节上限。到顶就整段放弃缓存，退回流式播放
CACHE_MAX_BYTES = 192 * 1024 * 1024


def _shrunk_image(cv2, frame) -> QImage:
    """BGR ndarray → 缩小过的 QImage。**先缩后转色**（转色在 1080p 上做要贵一倍）。

    必须 `.copy()`：QImage 只是包了 ndarray 的内存，帧一被回收画面就成花屏。
    """
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > CACHE_EDGE:
        scale = CACHE_EDGE / float(longest)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()


def decode_window(path, begin: float, end: float) -> tuple[list[QImage], int, float]:
    """把 `path` 的 [begin, end) 逐帧解成内存里的图，返回 (帧, 起始帧号, fps)。

    **播放器和后台预取线程共用这一份**：两边解出来的东西必须一模一样，
    否则"预取的和现场解的对不上一帧"这种问题根本查不出来。
    自己开自己的 `VideoCapture` —— cv2 的读写头不是线程安全的，不许共享。
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return ([], 0, 0.0)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps if fps > 0.1 else 25.0
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = total / fps if total > 0 else 0.0
        begin = max(0.0, float(begin))
        end = min(float(end), duration) if duration > 0 else float(end)
        if end - begin <= 0.0:
            return ([], 0, fps)
        first = int(begin * fps)                 # 和 seek() 用同一套 floor 换算
        count = int(round((end - begin) * fps)) + 1
        if count > CACHE_MAX_FRAMES:
            return ([], first, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames: list[QImage] = []
        budget = CACHE_MAX_BYTES
        for _ in range(count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break                            # 到文件尾了，有多少算多少
            image = _shrunk_image(cv2, frame)
            budget -= image.byteCount()
            if budget < 0:
                return ([], first, fps)          # 放不下就整段放弃，不做半段缓存
            frames.append(image)
        return (frames, first, fps)
    finally:
        cap.release()


class FramePrefetcher(QThread):
    """后台把片段解成内存帧。**解码绝不能在 GUI 线程上做。**

    实测 1080×1440（3:4，素材基本都是这个）一帧要 8ms，一个 2 秒片段 60 帧就是 0.5 秒 ——
    在段落切换那一刻现场解，界面就是实打实卡半秒。所以这条线程提前解好，
    播放器到点直接 `adopt_cache()` 装上，GUI 线程一帧都不解。
    """

    ready = pyqtSignal(str, float, float, object)     # path, begin, end, (帧, 起始帧号, fps)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._jobs: list[tuple[str, float, float]] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False

    def request(self, path, begin: float, end: float) -> None:
        """排一个活。重复的活不排第二遍；新活插到队头（当前要播的最急）。"""
        job = (str(path), round(float(begin), 3), round(float(end), 3))
        with self._lock:
            if job in self._jobs:
                return
            self._jobs.insert(0, job)
            del self._jobs[PREFETCH_QUEUE:]      # 排太多说明没人要了，丢掉旧的
        self._wake.set()
        if not self.isRunning():
            self.start()

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()
        self.wait(3000)

    def run(self) -> None:                       # noqa: D102 - QThread 的入口
        while not self._stop:
            with self._lock:
                job = self._jobs.pop(0) if self._jobs else None
            if job is None:
                self._wake.wait(0.2)
                self._wake.clear()
                continue
            path, begin, end = job
            try:
                bundle = decode_window(path, begin, end)
            except Exception as exc:             # noqa: BLE001 - 线程里炸了要报出来，别静默死掉
                logger.warning("片段预解码失败 %s [%.3f, %.3f]：%s", path, begin, end, exc)
                continue
            if bundle[0]:
                self.ready.emit(path, begin, end, bundle)


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
        #: 帧映射：`秒 → 秒`。给卡帧抖动这类"只改取哪一帧"的效果用。
        #: **只影响显示哪一帧，不影响 position()** —— 位置还是那口绝对钟说的，
        #: 否则跟主音频的纠偏会把它当成漂移，一路来回拽
        self._frame_map = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._audio_wav: Path | None = None
        self._audio_cut: Path | None = None
        self._audio_on = False

        # 片段预解码：整段的帧都在这个列表里，第 0 个对应文件里的第 `_cache_first` 帧。
        # 非空 = 当前处于"内存播放"模式，播放/定位/逐帧都不再碰 cv2
        self._cache: list[QImage] = []
        self._cache_first = 0
        self._path = ""                # 当前打开的文件（预取的帧要拿它核对是不是同一个）



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
        self._path = str(path)
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
        """关掉视频。**Windows 上这一步就是"解除文件占用"** —— 想删文件先调它。"""
        self.pause()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._drop_cache()
        self._path = ""
        self._image = None
        self._position = 0.0
        self._frame_index = 0
        self._clear_audio()

    def path(self) -> str:
        """当前打开的是哪个文件（没开就是空串）。"""
        return self._path

    def holds(self, path) -> bool:
        """是不是正占着这个文件。删文件之前拿它判断该不该先松手。"""
        if not self._path or not path:
            return False
        try:
            return Path(str(path)) == Path(self._path)
        except (TypeError, ValueError):
            return False

    # -------------------------------------------------------------- 片段预解码
    def _drop_cache(self) -> None:
        self._cache = []
        self._cache_first = 0

    def is_cached(self) -> bool:
        """当前是不是在放内存里的帧。"""
        return bool(self._cache)

    def cached_span(self) -> tuple[float, float]:
        """缓存覆盖的文件时间区间（左闭右开）。没缓存时是 (0, 0)。"""
        if not self._cache:
            return (0.0, 0.0)
        return (round(self._cache_first / self._fps, 3),
                round((self._cache_first + len(self._cache)) / self._fps, 3))

    def preload(self, begin: float, end: float) -> bool:
        """把 [begin, end) 逐帧解到内存（**在当前线程上解**，会卡住界面）。

        只在"用户主动等一下也认"的地方用，比如双击单独看一格。跟着主音频播的
        那条路走 `FramePrefetcher` + `adopt_cache`，绝不在 GUI 线程上解码。

        失败（片段太长、内存放不下、文件读不动）就**保持流式播放**并返回 False ——
        调用方不需要写两套播放逻辑，播放/定位接口在两种模式下行为一致，
        差别只有"卡不卡"。
        """
        self._drop_cache()
        if self._cap is None or not self._path:
            return False
        frames, first, _fps = decode_window(self._path, begin, end)
        return self._install(frames, first)

    def adopt_cache(self, path, begin: float, end: float, bundle) -> bool:
        """装上**别人（预取线程）解好的**帧。文件对不上就不收。

        这是"不卡"的关键一步：段落切换时 GUI 线程只做一次列表赋值，
        真正的 0.6 秒解码早在后台线程干完了。
        """
        if not self._path or Path(str(path)) != Path(self._path):
            return False
        frames, first, fps = bundle
        if fps > 0.1 and abs(fps - self._fps) > 0.01:
            return False                          # fps 都不一样，说明不是同一个文件
        span = self.cached_span()
        if self._cache and abs(span[0] - float(begin)) < 0.001:
            return True                           # 已经装着同一段了，别白折腾
        return self._install(list(frames), int(first))

    def _install(self, frames: list[QImage], first: int) -> bool:
        """把一段帧真正装进播放器，并把画面/时钟落到这一段的开头。"""
        if len(frames) < 2:
            self._drop_cache()
            return False
        keep_playing = self._playing
        self._cache = frames
        self._cache_first = int(first)
        self.seek(round(first / self._fps, 3))
        if keep_playing:
            self._clock_origin = time.perf_counter()
            self._clock_base = self._position
        return True

    def fps(self) -> float:
        """当前文件的帧率。抖动这类"按帧算"的效果要拿它换算。"""
        return float(self._fps)

    def set_frame_map(self, mapper) -> None:
        """装一个 `秒 → 秒` 的帧映射（None = 拆掉）。

        卡帧抖动就靠它做预览：窗口内把时间"按档保持"，显示的是同一帧，
        但 `position()` 照旧按绝对钟走 —— 位置不动手脚，跟主音频的纠偏才不会打架。
        只在**内存播放**下生效（流式播放是顺着解的，硬回跳等于重解关键帧）。
        """
        self._frame_map = mapper
        if self._cache:
            self._show_cached(self._position)

    def _show_cached(self, seconds: float) -> None:

        """显示缓存里对应 `seconds` 的那一帧（越界就贴到最近的一端）。

        **同一帧就直接回**：跟着主音频走时每 50ms 会来纠一次偏，而一帧有 33ms，
        大半次纠偏落在已经显示着的那一帧上。一次缩放 + setPixmap 要 3ms，
        白做的话每秒就白烧几十毫秒 —— 那就是"内存播放了还觉得有点顿"的来源。

        装了 `set_frame_map()` 的话，**取哪一帧**按映射走（卡帧抖动就是这么预览的），
        但 `position()` 仍然是传进来的那个时间：位置一动手脚，外面的纠偏就会
        把它当成漂移，一路来回拽（那正是之前那个"来回跳"的成因）。
        """
        clock = round(float(seconds), 3)
        want = int(seconds * self._fps) - self._cache_first
        last = len(self._cache) - 1
        if want < 0 or want > last:
            # 越界：贴到最近一端，**位置也跟着贴** —— 外面靠它知道"只能到这儿了"
            clock = round((self._cache_first + max(0, min(want, last))) / self._fps, 3)
        shown = seconds
        if self._frame_map is not None:
            try:
                shown = float(self._frame_map(seconds))
            except Exception:                    # noqa: BLE001 - 效果坏了不许带走播放
                shown = seconds
        index = max(0, min(int(shown * self._fps) - self._cache_first, last))
        if want >= last and self._playing:
            self.pause()                         # 片段播完就停，不往后溢到下一段
        frame = self._cache_first + index
        if frame == self._frame_index and self._image is not None:
            if clock != self._position:
                self._position = clock           # 抖动"卡住"的是画面，位置照旧往前走
                self.positionChanged.emit(self._position)
            return
        self._frame_index = frame
        self._position = clock
        self._image = self._cache[index]
        self._repaint()
        self.positionChanged.emit(self._position)



    def step_frame(self, delta: int = 1) -> None:
        """逐帧走。**核卡点就靠它**：停下来一帧一帧比画面和鼓点。

        缓存模式下只是换个下标（零解码），没缓存时退化成按帧长 seek。
        """
        self.pause()
        step = int(delta)
        if not step:
            return
        if self._cache:
            self._show_cached((self._frame_index + step) / self._fps)
            self._clock_base = self._position
            self._clock_origin = time.perf_counter()
            return
        self.seek(max(0.0, (self._frame_index + step) / self._fps))

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
        if not self._audio_wav.is_file():
            # 预览音轨被缓存清理删了。这条消息主界面认得，会自动重新解一次音轨
            self._audio_on = False
            self.audioFailed.emit("预览音轨已被清理，重新解一次")
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
        if self._cache:
            # 内存模式：定位就是算个下标，不碰 cv2。跟随主音频时每 50ms 纠一次偏
            # 也照样不掉帧 —— 卡顿的根子（重复 seek → 关键帧重解）就是在这儿断掉的
            self._show_cached(seconds)
        else:
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
        moment = self._clock_base + (time.perf_counter() - self._clock_origin)
        if self._cache:
            self._show_cached(moment)
            return
        target = int(moment * self._fps)
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
        self._image = self._to_image(frame)
        self._repaint()
        self.positionChanged.emit(self._position)
        return True

    def _to_image(self, frame, shrink: bool = False) -> QImage:
        """BGR ndarray → QImage。`shrink` 用于预解码：先缩小再存，内存才压得住。

        必须 `.copy()`：QImage 只是包了 ndarray 的内存，帧一被回收画面就成花屏。
        """
        cv2 = self._cv2
        if shrink:
            return _shrunk_image(cv2, frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _repaint(self) -> None:
        if self._image is None:
            return
        # 播放时用 Fast：3:4 竖屏 1080×1440 每帧做一次双线性平滑缩放，30fps 就能把
        # 主线程吃掉一大块（解码/转色/缩放全在这条线程上），画面反而顿。
        # 停下来看单帧时再用 Smooth —— 那时候要的是清楚，不是帧率。
        mode = Qt.FastTransformation if self._playing else Qt.SmoothTransformation
        pix = QPixmap.fromImage(self._image).scaled(
            self.view.size(), Qt.KeepAspectRatio, mode
        )
        self.view.setPixmap(pix)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint()
