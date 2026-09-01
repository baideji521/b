"""高光剪辑核心：起剪点和结束点严格取自 AI JSON 的 clip.start / clip.end，绝不推算。

    clip.start = 起始剪辑位置（配合界面「起始 加减秒数」）
    clip.end   = 结束剪辑位置（配合界面「结束 加减秒数」）

成品结构只有两段（没有冻帧、没有字幕、没有转场特效）：

    clip.start ── 原速播放原视频（带原声）──> clip.end ── 1 秒纯红背景（静音）──> 结束

执行时间轴（以 clip.start=20.68 / clip.end=24.00 为例）：

    20.68 ~ 24.00  正常播放，音频用原声
    24.00 ~ 25.00  纯红背景，静音
    最终时长 = (clip.end - clip.start) + 1 秒。

实现上复用项目既有栈：cv2 抓帧（同 video_io 的 seek 习惯）、PyAV 编码封装
（同 audio.py 的用法）。不依赖外部 ffmpeg 可执行文件。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np

from ..logging_setup import get_logger
from ..video_io import probe_video

logger = get_logger("highlight")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]   # (已完成帧, 总帧, 当前阶段)

# 渲染中的临时后缀。故意不是视频后缀：db/importer 的成品扫描按后缀过滤，
# 所以 .part 天然进不了 artifacts，崩溃留下的残片不会被当成成品。
PART_SUFFIX = ".part"

# 片尾固定追加的纯红背景：长度和颜色都不给外面调，成品一律带这一段
RED_TAIL_SECONDS = 1.0

RED_TAIL_RGB = (255, 0, 0)


FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑 Bold
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


# ====================================================================== JSON
@dataclass
class HighlightSpec:
    video_name: str
    clip_start: float
    clip_end: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.clip_end - self.clip_start

    def shifted(self, start_delta: float, end_delta: float) -> HighlightSpec:
        """按界面上填的两个加减秒数算出剪辑区间。

        起剪点 = clip.start + 起始加减
        结束点 = clip.end   + 结束加减（片尾那 1 秒红屏是额外追加的，不算在这里）
        """
        start = round(self.clip_start + start_delta, 3)
        end = round(self.clip_end + end_delta, 3)
        if start < 0:
            raise ValueError(f"start 偏移后变成负数：{start}")
        if end <= start:
            raise ValueError(f"偏移后结束点({end})不大于起剪点({start})")
        return replace(self, clip_start=start, clip_end=end)


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} 不是数字：{value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数字：{value!r}") from exc


def parse_spec(payload: dict[str, Any]) -> HighlightSpec:
    """解析 AI JSON：只认 clip.start / clip.end 两个时间，原样使用不做推算。

    overlays 一概不看——字幕功能已经去掉了，JSON 里有没有 overlays 都不影响剪辑。
    """
    if not isinstance(payload, dict):
        raise ValueError("JSON 根节点必须是对象")
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    if not isinstance(clip, dict) or "start" not in clip or "end" not in clip:
        raise ValueError("JSON 里缺少 clip.start / clip.end")

    start = _as_float(clip["start"], "clip.start")
    end = _as_float(clip["end"], "clip.end")
    if end <= start:
        raise ValueError(f"clip.end({end}) 必须大于 clip.start({start})")

    return HighlightSpec(
        video_name=str(payload.get("video") or clip.get("video") or ""),
        clip_start=start, clip_end=end, raw=payload,
    )



def resolve_video(spec: HighlightSpec, output_root: Path, input_root: Path | None = None,
                  fallback: Path | None = None) -> Path:
    """按 JSON 的 video 字段找源视频：绝对路径 -> input/ -> 已分析结果里的原始路径 -> 当前打开的视频。"""
    name = spec.video_name.strip()
    if name:
        direct = Path(name)
        if direct.is_file():
            return direct.resolve()
        stem = direct.stem
        if input_root is not None:
            hit = input_root / direct.name
            if hit.is_file():
                return hit.resolve()
        meta = output_root / stem / "video_metadata.json"
        if meta.is_file():
            import json  # noqa: PLC0415
            try:
                recorded = json.loads(meta.read_text(encoding="utf-8")).get("video", {}).get("path")
            except Exception:
                recorded = None
            if recorded and Path(recorded).is_file():
                return Path(recorded).resolve()
        if fallback is not None and fallback.is_file() and fallback.stem == stem:
            return fallback.resolve()
    if fallback is not None and fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"找不到源视频：{name or '(JSON 里没有 video 字段)'}")


def default_target(directory: Path, video: Path) -> Path:
    """输出到给定目录（GUI 的导出目录），文件名固定 <视频名>_高光时刻.mp4，同名直接覆盖。"""
    return directory / f"{video.stem}_高光时刻.mp4"


def part_target(target: Path) -> Path:
    """渲染中的临时文件：<成品名>.part，和成品同目录（`os.replace` 不许跨盘）。

    成品路径在整个渲染过程中都是空的，只有封装完整收尾之后才由 `os.replace` 一次性
    出现——这样"崩在渲染中途"留下的永远是 .part 残片，不会被当成成品登记。
    """
    return target.with_name(target.name + PART_SUFFIX)


# ==================================================================== 编码输出
def _fps_fraction(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(60000)


class _Writer:
    """一个输出容器：视频用 libx264，音频用 aac（原声按 clip 区间搬过来）。

    注意：所有流必须在写第一个 packet 之前建好，写过 packet 再 add_stream 会报
    "Cannot rebase to zero time."，所以音轨采样率要提前探出来。
    """

    def __init__(self, target: Path, width: int, height: int, fps: float,
                 audio_rate: int | None = None,
                 sample_aspect_ratio: Fraction | None = None,
                 container_format: str | None = None) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # 写的是 .part，扩展名猜不出容器格式，所以由调用方按成品扩展名显式给（None = 让 PyAV 猜）
        self.container = av.open(str(target), mode="w", format=container_format)
        rate = _fps_fraction(fps)
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = Fraction(rate.denominator, rate.numerator)
        if sample_aspect_ratio is not None:
            # 源视频是非方形像素时把 SAR 一起搬过来，播放器才会按原比例显示
            self.stream.codec_context.sample_aspect_ratio = sample_aspect_ratio
        self.stream.options = {"crf": "18", "preset": "medium"}
        self.audio: Any | None = None
        if audio_rate:
            self.audio = self.container.add_stream("aac", rate=int(audio_rate))
            self.audio.codec_context.layout = "stereo"
        self.index = 0

    def write_rgb(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self.index
        self.index += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        """正常收尾：把编码器里剩的帧吐完，再关容器（moov 就在这一步写进去）。

        重复调用是安全的（第二次直接返回）；中途出错会抛出去，调用方绝不能把这次
        渲染当成功。
        """
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
        """出错时收尾：只把文件句柄放开，不保证封装完整，也绝不再抛异常。

        真正的错误由调用方往上抛，这儿再抛就会把原因盖掉。
        """
        container = self.container
        if container is None:
            return
        self.container = None
        try:
            container.close()
        except Exception as exc:  # noqa: BLE001 - 放弃这次渲染，关不上也只能记一笔
            logger.debug("放弃渲染时关容器失败：%s", exc)


def _sample_aspect_ratio(video: Path) -> Fraction | None:
    """探源视频的像素长宽比（SAR）；方形像素或探不到就返回 None。"""
    try:
        with av.open(str(video)) as container:
            sar = container.streams.video[0].sample_aspect_ratio
    except Exception:  # noqa: BLE001
        return None
    if not sar or sar <= 0 or sar == 1:
        return None
    return Fraction(sar)


def _audio_rate(video: Path) -> int | None:
    """探源视频音轨采样率；没有音轨返回 None。"""
    try:
        with av.open(str(video)) as container:
            if not container.streams.audio:
                return None
            return int(container.streams.audio[0].rate or 44100)
    except Exception:  # noqa: BLE001
        return None


def _decode_pcm(video: Path, start: float, end: float, rate: int) -> np.ndarray:
    """取原声 [start, end) 的 PCM（fltp/stereo），按样本精确对齐，返回 (2, n) float32。"""
    with av.open(str(video)) as src:
        src_stream = src.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        src.seek(max(int((start - 0.5) / av.time_base), 0), stream=src_stream)
        chunks: list[np.ndarray] = []
        base: float | None = None
        for frame in src.decode(src_stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * src_stream.time_base)
            if t + float(frame.samples) / rate <= start:
                continue
            if t >= end:
                break
            for resampled in resampler.resample(frame):
                if base is None:
                    base = t
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):  # 冲掉重采样器里剩的样本
            if resampled is not None and resampled.samples:
                chunks.append(resampled.to_ndarray())
    want = max(0, int(round((end - start) * rate)))
    if not chunks or base is None:
        return np.zeros((2, want), dtype=np.float32)
    pcm = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    offset = max(0, int(round((start - base) * rate)))
    pcm = pcm[:, offset:offset + want]
    if pcm.shape[1] < want:  # 尾部不够就补静音，保证和视频一样长
        pcm = np.pad(pcm, ((0, 0), (0, want - pcm.shape[1])))
    return pcm


def _write_audio(writer: _Writer, video: Path, audio_start: float, live_seconds: float,
                 total_seconds: float, on_log: LogFn) -> None:
    """音频完全跟着画面的帧网格走，避免音画错位。

    audio_start   = 输出第一帧对应的源时间（floor(clip.start*fps)/fps，不是 clip.start）
    live_seconds  = 正常播放段时长（play_frames/fps），到这里画面冻结，音频同时静音
    total_seconds = 输出总时长（含冻帧段和片尾红屏），音频补静音补到和视频一样长

    冻帧段和片尾一律纯静音：高光成品不混任何音效。
    """
    if writer.audio is None:
        on_log("[AUDIO] 源视频没有可用音轨，输出为无声")
        return
    out_stream = writer.audio
    rate = int(out_stream.rate or 44100)
    live_samples = max(0, int(round(live_seconds * rate)))
    total_samples = max(live_samples, int(round(total_seconds * rate)))
    try:
        live = _decode_pcm(video, audio_start, audio_start + live_seconds, rate)
    except Exception as exc:  # noqa: BLE001
        on_log(f"[AUDIO] 读音轨失败，输出为无声：{exc}")
        live = np.zeros((2, live_samples), dtype=np.float32)
    live = live[:, :live_samples]
    if live.shape[1] < live_samples:
        live = np.pad(live, ((0, 0), (0, live_samples - live.shape[1])))
    silence = np.zeros((2, total_samples - live_samples), dtype=np.float32)
    pcm = np.concatenate([live, silence], axis=1) if silence.size else live
    if pcm.size == 0:
        on_log("[AUDIO] 没有可写的音频样本")
        return

    frame_size = out_stream.frame_size or 1024

    fifo = av.AudioFifo()
    fed = 0

    def flush(final: bool = False) -> None:
        while True:
            chunk = fifo.read(frame_size, partial=final)
            if chunk is None:
                break
            for packet in out_stream.encode(chunk):
                writer.container.mux(packet)

    for offset in range(0, pcm.shape[1], frame_size):
        block = np.ascontiguousarray(pcm[:, offset:offset + frame_size])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
        frame.sample_rate = rate
        # AudioFifo 要求送进去的 pts 从 0 起连续，所以按累计样本数排
        frame.time_base = Fraction(1, rate)
        frame.pts = fed
        fed += frame.samples
        fifo.write(frame)
        flush()
    flush(final=True)
    on_log(f"[AUDIO] 原声 {audio_start:.4f} → {audio_start + live_seconds:.4f}"
           f"（{live_samples / rate:.3f}s，与画面同一帧网格）"
           f" + 冻帧和片尾静音 {silence.shape[1] / rate:.3f}s，合计 {fed / rate:.3f}s")


# ====================================================================== 主流程
def render_highlight(video: Path, spec: HighlightSpec, target: Path,
                     on_log: LogFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
    """按 spec 生成高光 MP4，返回统计信息。on_progress 每写一帧回报一次进度。

    成品只有两段：原速播放段（带原声）+ 1 秒纯红背景（静音）。
    没有冻帧、没有字幕、没有转场特效，也**一律不混音效**。
    片尾那 1 秒（`RED_TAIL_SECONDS`）是额外追加的，不占 spec 的时长预算。

    落地方式是「先写 .part，完整收尾后 os.replace 成 target」：崩溃 / 被杀 / 断电
    只会留下 .part，target 要么是上一次的完整成品、要么根本不存在，绝不会是残片。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    report = on_progress or (lambda done, total, stage: None)
    info = probe_video(video)
    fps = float(info.fps)
    if fps <= 0:
        raise ValueError(f"读不到有效帧率：{info.fps}")
    if spec.clip_end > info.duration + 1e-3:
        raise ValueError(f"结束点({spec.clip_end}) 超过视频时长({info.duration})")

    # 帧网格：起剪帧 = clip.start 这一刻正在显示的那一帧，结束帧同理
    start_index = int(math.floor(spec.clip_start * fps))
    end_index = int(math.floor(spec.clip_end * fps))
    play_frames = max(1, min(int(round(spec.duration * fps)), end_index - start_index))
    grid_start = start_index / fps          # 输出第 0 帧对应的源时间
    live_seconds = play_frames / fps        # 播放段时长，画面和音频共用
    # 片尾红屏是额外追加的，不吃 spec.duration 的预算；进度和音频长度都得算上它
    tail_frames = max(1, int(round(fps * RED_TAIL_SECONDS)))
    out_frames = play_frames + tail_frames
    total_seconds = out_frames / fps

    log("[HIGHLIGHT]")
    log(f"Clip Start : {spec.clip_start:.2f}")
    log(f"Clip End   : {spec.clip_end:.2f}")
    log("")
    log(f"[VIDEO] {video.name}  {info.width}x{info.height}  {fps:g} fps  音轨={info.has_audio}")
    sar = _sample_aspect_ratio(video)
    log(f"[SIZE] 输出保持原分辨率 {info.width}x{info.height}"
        + (f"，像素比 SAR={sar} 一并沿用" if sar else "，方形像素，比例与源一致"))
    log(f"[ALIGN] 起剪第 {start_index} 帧（源 {grid_start:.4f}s）｜"
        f"结束第 {end_index} 帧（源 {end_index / fps:.4f}s）｜"
        f"音频起点与画面同为 {grid_start:.4f}s，片尾红屏静音")

    part = part_target(target)
    writer = _Writer(part, info.width, info.height, fps, audio_rate=_audio_rate(video),
                     sample_aspect_ratio=sar,
                     container_format=target.suffix.lstrip(".").lower() or "mp4")
    # 从这里到 writer.close() 全程只写 .part。出任何岔子都只放开句柄、把原始错误抛出去，
    # 成品路径在这期间始终是空的——所以"崩在渲染中途"绝不可能留下一个半成品成品文件。
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 打不开视频：{video}")

        written_play = 0
        report(0, out_frames, "正常播放段")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            log(f"[{spec.clip_start:.2f}] START NORMAL PLAYBACK")
            for _ in range(play_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                written_play += 1
                report(written_play, out_frames, "正常播放段")
            if written_play <= 0:
                raise RuntimeError("一帧都没读到，检查起剪点 / 结束点是否落在视频范围内")
        finally:
            cap.release()

        # 片尾：1 秒纯红背景（无字、无声），固定追加在成品最后
        log(f"[TAIL] 片尾追加 {RED_TAIL_SECONDS:.2f}s 纯红背景（{tail_frames} 帧，静音）")
        red = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        red[:, :] = RED_TAIL_RGB
        for i in range(tail_frames):
            writer.write_rgb(red)
            report(written_play + i + 1, out_frames, "片尾红屏")

        report(out_frames, out_frames, "写音频并封装")
        _write_audio(writer, video, grid_start, live_seconds, total_seconds, log)
        writer.close()          # moov 写进 .part：到这一刻 .part 才是一份完整视频
    except BaseException:       # 含 KeyboardInterrupt：句柄必须放开，错误原样上抛
        writer.abort()
        raise
    # 只有完整收尾之后才把成品搬到最终位置：同目录 rename，原子替换旧成品（不先删）
    try:
        os.replace(part, target)
    except OSError as exc:
        raise RuntimeError(f"成品提交失败，{target.name} 可能正被占用（残片留在 "
                           f"{part.name}）：{exc}") from exc

    log(f"[{spec.clip_end:.2f}] END CLIP")

    out_duration = (written_play + tail_frames) / fps
    log(f"[OUTPUT] {target}")
    log(f"[OUTPUT] 帧数 {written_play + tail_frames}"
        f"（正常播放 {written_play} + 片尾红屏 {tail_frames}）"
        f"，时长 {out_duration:.3f}s，目标 {spec.duration:.2f}s"
        f" + 片尾 {RED_TAIL_SECONDS:.2f}s")
    return {
        "output": str(target),
        "clip_start": spec.clip_start,
        "clip_end": spec.clip_end,
        "fps": fps,
        "total_frames": written_play + tail_frames,
        "play_frames": written_play,
        "red_tail_frames": tail_frames,
        "red_tail_seconds": RED_TAIL_SECONDS,
        "duration_seconds": round(out_duration, 3),
        "target_duration_seconds": round(spec.duration, 3),
        "grid_start_seconds": round(grid_start, 4),
    }
