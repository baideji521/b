"""媒体后端抽象层。**业务层禁止直接 `subprocess.run(["ffmpeg", ...])`**。

技术指导第十七节的要求：如果以后要加 FFmpeg / NVENC，必须抽象成 `MediaBackend`，
业务代码只认这个接口。所以哪怕现在只有一个实现，这层也要先立起来 ——
等真要换编码器时，改的是这里，不是切片和混剪的业务逻辑。

第一版只提供 `PyAVBackend`（复用项目既有的 cv2 抓帧 + PyAV/libx264 编码栈，
和 `highlight/clip.py` 同一套习惯，不引入外部 ffmpeg.exe 依赖）。
`FFmpegCPUBackend` / `FFmpegNVENCBackend` 的位置留在 `resolve()` 里，
现在会明确报"不可用"而不是假装能跑。

两个必须做对的归一化（多源拼接的经典坑）：

1. **画布归一化**：不同源的分辨率/宽高比不同，而输出流的宽高在写第一帧前就定死了。
   做法是 scale-to-cover + 居中裁切（`fit_frame`）—— 卡点舞成片一律竖屏满画面，
   加黑边比裁掉边缘更难看。
2. **帧率归一化**：输出是**恒定** fps，所以取帧必须按"输出第 k 帧 ← 源时间
   start + k/fps_out"去 seek，**不能**顺序 `cap.read()` 计数。
   30fps 的源顺序读进 25fps 的输出会让动作快 20%，而卡点舞对速度极其敏感。
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..logging_setup import get_logger

logger = get_logger("dance.media")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]

#: 渲染中的临时后缀，同 highlight/clip.py：故意不是视频后缀
PART_SUFFIX = ".part"
#: 默认输出画布（竖屏 9:16）与帧率
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_FPS = 30.0
#: x264 参数。crf 20 比高光那条线（18）略省，素材片数量多、单片只有两秒
CRF = "20"
PRESET = "medium"

@dataclass(frozen=True)
class Canvas:
    """输出画布。所有素材都被归一到这一套宽高帧率，多源才能拼在一起。"""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS

    def frames_for(self, seconds: float) -> int:
        """一段时长对应多少输出帧。`round` 而不是 `floor`：
        2.0 秒 @30fps 必须正好 60 帧，floor 会因浮点误差变成 59，一版下来累积成半秒。"""
        return max(1, int(round(max(0.0, float(seconds)) * self.fps)))

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "fps": round(self.fps, 4)}


@dataclass(frozen=True)
class MediaMeta:
    """探测结果。`fps <= 0` 表示探不到，调用方必须自己兜底。"""

    path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    has_audio: bool = False
    audio_rate: int = 0
    video_streams: int = 0
    audio_streams: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "width": self.width, "height": self.height,
                "fps": round(self.fps, 4), "duration": round(self.duration, 3),
                "has_audio": self.has_audio, "audio_rate": self.audio_rate,
                "video_streams": self.video_streams, "audio_streams": self.audio_streams}


def fit_frame(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """把一帧归一到画布：scale-to-cover + 居中裁切。返回 `(height, width, 3)` uint8。

    cover 而不是 contain：竖屏卡点舞成片加黑边比裁掉左右边缘难看得多，
    而舞者基本都在画面中央，裁边损失很小。
    """
    import cv2  # noqa: PLC0415 - GUI 进程不能在主线程 import cv2，见 gui/__init__.py

    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"期望 (h, w, 3) 的 RGB 帧，收到 {frame.shape}")
    src_h, src_w = frame.shape[0], frame.shape[1]
    if src_h == height and src_w == width:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    scale = max(width / float(src_w), height / float(src_h))
    new_w = max(width, int(math.ceil(src_w * scale)))
    new_h = max(height, int(math.ceil(src_h * scale)))
    # 放大用 INTER_LINEAR、缩小用 INTER_AREA：缩小时 LINEAR 会有明显摩尔纹
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return np.ascontiguousarray(resized[top:top + height, left:left + width], dtype=np.uint8)

class _Writer:
    """一个输出容器：视频 libx264，可选音频 aac。

    所有流必须在写第一个 packet 之前建好（写过 packet 再 add_stream 会报
    "Cannot rebase to zero time."）—— 和 `highlight/clip.py:_Writer` 同一个坑，
    所以画布和音轨采样率都得在构造前决定好。
    """

    def __init__(self, target: Path, canvas: Canvas, audio_rate: int | None = None,
                 container_format: str | None = None) -> None:
        import av  # noqa: PLC0415

        target.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(target), mode="w", format=container_format)
        rate = Fraction(canvas.fps).limit_denominator(60000)
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = canvas.width
        self.stream.height = canvas.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = Fraction(rate.denominator, rate.numerator)
        self.stream.options = {"crf": CRF, "preset": PRESET}
        self.audio: Any | None = None
        if audio_rate:
            self.audio = self.container.add_stream("aac", rate=int(audio_rate))
            self.audio.codec_context.layout = "stereo"
        self.index = 0

    def write_rgb(self, rgb: np.ndarray) -> None:
        import av  # noqa: PLC0415

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self.index
        self.index += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        """正常收尾：flush 编码器再关容器（moov 在这一步写进去）。重复调用安全。"""
        container = self.container
        if container is None:
            return
        self.container = None
        try:
            for packet in self.stream.encode():
                container.mux(packet)
            if self.audio is not None:
                for packet in self.audio.encode():
                    container.mux(packet)
        finally:
            container.close()

    def abort(self) -> None:
        """出错时收尾：只放开句柄，不保证封装完整，也绝不再抛异常盖掉原始错误。"""
        container = self.container
        if container is None:
            return
        self.container = None
        try:
            container.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("放弃渲染时关容器失败：%s", exc)

class _FrameReader:
    """按**帧号**取帧的读取器，带顺序优化。用完必须 `close()`。

    `frame_at(index)` 的语义是"第 index 帧的画面"。顺序前进时直接 `read()`，
    跳跃时才 `set(CAP_PROP_POS_FRAMES)` —— seek 在长视频上很贵，
    而卡点舞切片正好是"一段连续两秒"，绝大多数取帧都是顺序的。
    """

    def __init__(self, path: str | Path) -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV 打不开视频：{self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0.1:
            self.fps = DEFAULT_FPS
            logger.warning("%s 读不到帧率，按 %.1f 兜底", self.path, self.fps)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cursor = -1
        self._last: np.ndarray | None = None

    def frame_at(self, index: int) -> np.ndarray | None:
        index = max(0, int(index))
        if self._last is not None and index == self._cursor:
            return self._last                      # 同一帧被连续要（输出 fps 高于源）
        if index != self._cursor + 1:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, index)
            self._cursor = index - 1
        ok, frame = self.cap.read()
        if not ok:
            self._cursor = -1                      # 失败后强制下一次重新 seek
            return self._last
        self._cursor = index
        self._last = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return self._last

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None                        # type: ignore[assignment]


class MediaBackend:
    """媒体后端接口。业务层只认这几个方法。

    子类必须实现 `probe` / `render_spans` / `mux_audio`；`available()` 决定它能不能用。
    """

    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def probe(self, path: str | Path) -> MediaMeta:
        raise NotImplementedError

    def render_spans(self, spans: Sequence[tuple[str, float, float]], target: Path,
                     canvas: Canvas, *, on_log: LogFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def mux_audio(self, video: Path, audio: Path, target: Path, *,
                  audio_start: float = 0.0, duration: float | None = None,
                  on_log: LogFn | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def extract_clip(self, source: str | Path, start: float, end: float, target: Path,
                     canvas: Canvas, *, on_log: LogFn | None = None) -> dict[str, Any]:
        """切一段并归一到画布。默认实现就是"只有一段的 render_spans"。"""
        return self.render_spans([(str(source), float(start), float(end))], target, canvas,
                                 on_log=on_log)

def _copy_stream(container: Any, template: Any) -> Any:
    """建一条"照抄模板"的输出流，用于视频 stream copy（不重编码）。

    PyAV 各版本的写法不一样，实测本机 13.x 只认新写法：
      13.x：`add_stream_from_template(template)`
      旧版：`add_stream(template=template)`
    所以按能力探测，两条都留着 —— 只写一种会在另一批环境上直接 TypeError。
    """
    if hasattr(container, "add_stream_from_template"):
        return container.add_stream_from_template(template)
    return container.add_stream(template=template)


class PyAVBackend(MediaBackend):

    """cv2 抓帧 + PyAV/libx264 编码。项目既有的栈，不需要外部 ffmpeg.exe。"""

    name = "pyav"

    def available(self) -> bool:
        try:
            import av  # noqa: F401, PLC0415
            import cv2  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def probe(self, path: str | Path) -> MediaMeta:
        """容器元信息走 PyAV（准），宽高帧率缺失时用 cv2 补（PyAV 偶尔读不到 fps）。"""
        import av  # noqa: PLC0415

        target = Path(path)
        meta = MediaMeta(path=str(target))
        try:
            with av.open(str(target)) as container:
                videos = list(container.streams.video)
                audios = list(container.streams.audio)
                duration = (float(container.duration) / 1_000_000.0
                            if container.duration else 0.0)
                width = height = 0
                fps = 0.0
                if videos:
                    stream = videos[0]
                    width = int(stream.codec_context.width or 0)
                    height = int(stream.codec_context.height or 0)
                    fps = float(stream.average_rate) if stream.average_rate else 0.0
                    if duration <= 0 and stream.duration and stream.time_base:
                        duration = float(stream.duration * stream.time_base)
                rate = int(audios[0].rate or 0) if audios else 0
                if duration <= 0 and audios and audios[0].duration and audios[0].time_base:
                    duration = float(audios[0].duration * audios[0].time_base)
                meta = MediaMeta(path=str(target), width=width, height=height, fps=fps,
                                 duration=round(duration, 3), has_audio=bool(audios),
                                 audio_rate=rate, video_streams=len(videos),
                                 audio_streams=len(audios))
        except Exception as exc:  # noqa: BLE001 - 探不到就返回空壳，由调用方判
            logger.debug("PyAV 探测失败 %s：%s", target, exc)
        return meta

    def render_spans(self, spans: Sequence[tuple[str, float, float]], target: Path,
                     canvas: Canvas, *, on_log: LogFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
        """把多个 `(源路径, 起, 止)` 按顺序渲成**一条无声视频**，归一到 canvas。

        无声是有意的：目标歌是唯一正式音轨（技术指导第十六节），
        源舞蹈视频的原声在这一步就被丢掉，`mux_audio` 再把目标歌接上。

        落地方式：全程写 `.part`，完整收尾后 `os.replace` —— 崩溃只会留下 .part。
        """
        log = on_log or (lambda line: logger.info("%s", line))
        report = on_progress or (lambda done, total, stage: None)
        pieces = [(str(p), float(a), float(b)) for p, a, b in spans if float(b) > float(a)]
        if not pieces:
            raise ValueError("render_spans 收到空的片段列表")

        total_frames = sum(canvas.frames_for(b - a) for _p, a, b in pieces)
        part = target.with_name(target.name + PART_SUFFIX)
        writer = _Writer(part, canvas,
                         container_format=target.suffix.lstrip(".").lower() or "mp4")
        written = 0
        try:
            report(0, total_frames, "拼接画面")
            for order, (path, start, end) in enumerate(pieces, 1):
                reader = _FrameReader(path)
                try:
                    frames = canvas.frames_for(end - start)
                    for k in range(frames):
                        # 帧率归一化的关键：按**时间**取源帧，不顺序计数
                        at = start + k / canvas.fps
                        rgb = reader.frame_at(int(math.floor(at * reader.fps)))
                        if rgb is None:
                            log(f"[段 {order}/{len(pieces)}] {Path(path).name} "
                                f"在 {at:.3f}s 取不到帧，这一段提前收尾")
                            break
                        writer.write_rgb(fit_frame(rgb, canvas.width, canvas.height))
                        written += 1
                        report(written, total_frames, f"拼接第 {order}/{len(pieces)} 段")
                finally:
                    reader.close()
            if written <= 0:
                raise RuntimeError("一帧都没写出来，检查源文件与区间是否有效")
            writer.close()
        except BaseException:
            writer.abort()
            raise
        os.replace(part, target)
        duration = round(written / canvas.fps, 4)
        log(f"[画面] {len(pieces)} 段共 {written} 帧，{duration:.3f}s → {target.name}")
        return {"output": str(target), "clips": len(pieces), "frames": written,
                "duration": duration, "canvas": canvas.to_dict(), "backend": self.name}

    def mux_audio(self, video: Path, audio: Path, target: Path, *,
                  audio_start: float = 0.0, duration: float | None = None,
                  on_log: LogFn | None = None) -> dict[str, Any]:
        """把目标歌接到无声成片上，输出**只有一条音轨**的最终成品。

        画面走 **stream copy**（`add_stream(template=...)`）：无声成片刚刚才由我们自己
        用 libx264 编出来，再解码重编一遍纯属自损画质又费时间。

        音频按画面时长对齐：歌比画面长就截断，短就补静音 ——
        成片的时长由**画面**说了算，音轨不许把成片拖长出一截黑屏。
        """
        import av  # noqa: PLC0415

        log = on_log or (lambda line: logger.info("%s", line))
        target = Path(target)
        part = target.with_name(target.name + PART_SUFFIX)
        want = float(duration) if duration else self.probe(video).duration
        rate = 44100

        with av.open(str(video)) as src:
            if not src.streams.video:
                raise ValueError(f"{video} 里没有视频流，没法 mux")
            part.parent.mkdir(parents=True, exist_ok=True)
            out = av.open(str(part), mode="w",
                          format=target.suffix.lstrip(".").lower() or "mp4")
            try:
                in_video = src.streams.video[0]
                out_video = _copy_stream(out, in_video)

                out_audio = out.add_stream("aac", rate=rate)
                out_audio.codec_context.layout = "stereo"
                for packet in src.demux(in_video):
                    if packet.dts is None:
                        continue
                    packet.stream = out_video
                    out.mux(packet)
                samples = self._write_song(out, out_audio, audio, audio_start, want, rate, log)
                for packet in out_audio.encode():
                    out.mux(packet)
            except BaseException:
                try:
                    out.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mux 失败时关容器出错：%s", exc)
                raise
            out.close()
        os.replace(part, target)
        meta = self.probe(target)
        log(f"[成品] {target.name}｜{meta.duration:.3f}s｜"
            f"视频流 {meta.video_streams} 条、音频流 {meta.audio_streams} 条")
        return {"output": str(target), "duration": meta.duration,
                "audio_samples": samples, "audio_streams": meta.audio_streams,
                "video_streams": meta.video_streams, "backend": self.name}

    def _write_song(self, container: Any, stream: Any, audio: Path, start: float,
                    want: float, rate: int, log: LogFn) -> int:
        """把目标歌 [start, start+want) 编成 aac 写进容器，返回写了多少样本。"""
        from .audio_fingerprint import AudioReadError, extract_analysis_audio  # noqa: PLC0415

        need = max(1, int(round(max(0.0, want) * rate)))
        try:
            mono = extract_analysis_audio(audio, start=start, duration=want, sample_rate=rate)
        except AudioReadError as exc:
            log(f"[音轨] 读目标歌失败，成品将是静音：{exc}")
            mono = np.zeros(0, dtype=np.float32)
        if mono.size < need:
            mono = np.pad(mono, (0, need - mono.size))      # 歌比画面短 → 补静音
        pcm = np.vstack([mono[:need], mono[:need]])          # mono → stereo
        return self._encode_audio(container, stream, pcm, rate)

    @staticmethod
    def _encode_audio(container: Any, stream: Any, pcm: np.ndarray, rate: int) -> int:
        """按 aac 的 frame_size 分块喂，pts 按累计样本数排（同 highlight/clip.py 的写法）。"""
        import av  # noqa: PLC0415

        size = int(stream.frame_size or 1024)
        fifo = av.AudioFifo()
        fed = 0

        def flush(final: bool = False) -> None:
            while True:
                chunk = fifo.read(size, partial=final)
                if chunk is None:
                    break
                for packet in stream.encode(chunk):
                    container.mux(packet)

        for begin in range(0, pcm.shape[1], size):
            block = np.ascontiguousarray(pcm[:, begin:begin + size])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.time_base = Fraction(1, rate)
            frame.pts = fed
            fed += frame.samples
            fifo.write(frame)
            flush()
        flush(final=True)
        return fed

class _FFmpegBackend(MediaBackend):
    """FFmpeg 系后端的占位实现（CPU / NVENC 各一个子类）。

    技术指导第十七节：**不要假设当前项目已经存在完整 NVENC pipeline**，第一版继续用
    PyAV/libx264。所以这里只把接缝留出来 —— `available()` 老实按 `ffmpeg` 在不在 PATH
    上判，不在就返回 False，`resolve()` 会自动退回 PyAV。

    刻意**不**给一个半成品实现：一个只在部分参数下正确的编码路径，比没有这个后端更危险。
    真要接的时候，改的是这个类，业务层（material_slice / montage_render）一行都不用动 ——
    这正是抽象这一层的全部意义。
    """

    binary = "ffmpeg"

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def probe(self, path: str | Path) -> MediaMeta:
        # 探测不涉及编码器，直接复用 PyAV 那份，避免再养一套 ffprobe 解析
        return PyAVBackend().probe(path)

    def render_spans(self, spans, target, canvas, *, on_log=None, on_progress=None):
        raise NotImplementedError(
            f"{self.name} 后端还没实现（第一版只用 pyav）。"
            f"要接的话在 media_backend.py 里实现这个类，业务层不需要改动。")

    def mux_audio(self, video, audio, target, *, audio_start=0.0, duration=None, on_log=None):
        raise NotImplementedError(
            f"{self.name} 后端还没实现（第一版只用 pyav）。")


class FFmpegCPUBackend(_FFmpegBackend):
    name = "ffmpeg_cpu"


class FFmpegNVENCBackend(_FFmpegBackend):
    name = "ffmpeg_nvenc"


#: 可选后端。业务层用名字取，不 import 具体类
BACKENDS: dict[str, type[MediaBackend]] = {
    PyAVBackend.name: PyAVBackend,
    FFmpegCPUBackend.name: FFmpegCPUBackend,
    FFmpegNVENCBackend.name: FFmpegNVENCBackend,
}


def resolve(name: str = "") -> MediaBackend:
    """按名字取后端；`auto` / 空 / 取不到就用 PyAV。

    显式指定了一个不可用的后端时**报错**而不是悄悄换 —— 用户明确选了 NVENC 却拿到
    libx264 的输出，事后只会困惑于"为什么这么慢"。只有 `auto` 才允许自动回退。
    """
    key = (name or "auto").strip().lower()
    if key in ("", "auto"):
        for candidate in (PyAVBackend(),):
            if candidate.available():
                return candidate
        raise RuntimeError("没有可用的媒体后端（PyAV 与 opencv 都不可用）")
    if key not in BACKENDS:
        raise ValueError(f"未知的媒体后端 {name!r}，可选：{', '.join(sorted(BACKENDS))}")
    backend = BACKENDS[key]()
    if not backend.available():
        raise RuntimeError(f"媒体后端 {key} 当前不可用（依赖或可执行文件缺失）")
    return backend


def is_complete_video(path: str | Path) -> bool:
    """封装是否完整。渲染中途崩掉的残片必须挡在登记之外。

    判据只有三条（同 `video_io.is_complete_video` 的口径）：文件非空、能打开、
    有视频流且时长为正。缺 moov 的未写完容器会在 `av.open` 那一步抛出来。
    """
    import av  # noqa: PLC0415

    target = Path(path)
    try:
        if not target.is_file() or target.stat().st_size <= 0:
            return False
    except OSError:
        return False
    try:
        with av.open(str(target)) as container:
            if not container.streams.video:
                return False
            seconds = (float(container.duration) / 1_000_000.0 if container.duration else 0.0)
            if seconds <= 0:
                stream = container.streams.video[0]
                if stream.duration and stream.time_base:
                    seconds = float(stream.duration * stream.time_base)
            return seconds > 0
    except Exception:  # noqa: BLE001 - 打不开就是不完整
        return False


__all__ = [
    "PART_SUFFIX", "DEFAULT_WIDTH", "DEFAULT_HEIGHT", "DEFAULT_FPS",
    "Canvas", "MediaMeta", "fit_frame",
    "MediaBackend", "PyAVBackend", "FFmpegCPUBackend", "FFmpegNVENCBackend",
    "BACKENDS", "resolve", "is_complete_video",
]






