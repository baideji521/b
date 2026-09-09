"""高光剪辑核心：起剪点和结束点严格取自 AI JSON 的 segments[0].sa / .end，绝不推算。

    segments[0].sa  = 起始剪辑位置（配合界面「起始 加减秒数」），原视频时间
    segments[0].end = 结束剪辑位置（配合界面「结束 加减秒数」），原视频时间

协议只认一种写法，字段含义见 vidscribe/ai_protocol.py；`dst` 和 timeline.duration
是成片时间域，这一层不看。

成品结构最多两段（没有字幕、没有转场特效）：

    sa ── 原速播放原视频（带原声）──> end ── 末帧冻结 N 秒（静音）──> 结束

一条素材内部那些"没人说话干等着"的长静音要剪掉时，调用方把**要保留的片段列表**
（原视频绝对秒，按时间排好）通过 `render_highlight(..., keep_spans=[...])` 递进来，
渲染就逐段解码、按顺序写进同一个输出文件，段与段之间直接接上（不做转场、不做淡入淡出）：

    保留段1 ─ 保留段2 ─ … ─ 保留段n ── 最后一段的末帧冻结 N 秒（静音）──> 结束

片段怎么算是上一层的事（见 highlight/clip_engine.py 的 `trim_plan`），这一层只管照单渲染，
所以这里**不 import clip_engine**：渲染不需要知道静音是怎么判出来的。

冻帧秒数由 config.json 的 `highlight.freeze_tail_seconds` 控制（默认 2 秒，0 = 不冻），

开关在「剪辑高光」窗里。AI 报的 `timeline.duration` 已经把冻帧那几秒算进去了，
所以剪辑区间只按 sa→end 走，冻帧是渲染时额外追加回去的。

执行时间轴（以 sa=20.68 / end=24.00、冻帧 2 秒为例）：

    20.68 ~ 24.00  正常播放，音频用原声
    24.00 ~ 26.00  末帧冻结，静音
    最终时长 = (end - sa) + 冻帧秒数。

实现上复用项目既有栈：cv2 抓帧（同 video_io 的 seek 习惯）、PyAV 编码封装
（同 audio.py 的用法）。不依赖外部 ffmpeg 可执行文件。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import av
import cv2
import numpy as np

from ..logging_setup import get_logger
from ..video_io import probe_video
from .. import ai_protocol

logger = get_logger("highlight")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]   # (已完成帧, 总帧, 当前阶段)

# 渲染中的临时后缀。故意不是视频后缀：db/importer 的成品扫描按后缀过滤，
# 所以 .part 天然进不了 artifacts，崩溃留下的残片不会被当成成品。
PART_SUFFIX = ".part"


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

        起剪点 = segments[0].sa  + 起始加减
        结束点 = segments[0].end + 结束加减（末帧冻结是额外追加的，不算在这里）
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
    """解析 AI JSON：只认 `segments[0].sa` / `segments[0].end` 两个时间，原样使用不推算。

    sa / end 是**原视频时间**；`dst`、`timeline.duration`、`o` / `s` / `w` / `a` / `e`
    都是成片时间域，剪辑一概不看（字幕、贴纸这些功能现在没有渲染端）。
    老协议（`clip.start` / `clip.end`）由 `ai_protocol.payload_of` 先升级成 segments
    再解析，所以这里只有一套形状要管，见 vidscribe/ai_protocol.py。
    """
    if not isinstance(payload, dict):
        raise ValueError("JSON 根节点必须是对象")
    why = ai_protocol.validate(payload)
    if why is not None:
        raise ValueError(why)
    segment = ai_protocol.raw_segments(payload)[0]
    start = _as_float(segment["sa"], "segments[0].sa")
    end = _as_float(segment["end"], "segments[0].end")
    if end <= start:
        raise ValueError(f"segments[0].end({end}) 必须大于 sa({start})")

    return HighlightSpec(
        video_name=str(payload.get("video") or ""),
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


def duration_tag(seconds: float) -> str:
    """把时长写成文件名安全的样子：6.89 秒 → `689`。真源在 `ai_protocol.duration_tag`。"""
    return ai_protocol.duration_tag(seconds)


def default_target(directory: Path, video: Path, seconds: float | None = None, *,
                   index: int = 1) -> Path:
    """成品路径：`<视频名>_<时长>.mp4`，例 `xxx_689.mp4`（成片 6.89 秒）。

    时长放在最后，一眼就能看出这条成品多长（排版/发布挑素材全靠它）。
    `seconds` 不给就退回老名字 `<视频名>_高光时刻.mp4`——`--out` 没指定的 CLI
    和只想要个落点的调用方还走这条。

    `index > 1` 再加 `_2` / `_3`：一份 JSON 剪出多段、时长又刚好相同也不会互相覆盖。
    """
    stem = video.stem
    stem += f"_{duration_tag(seconds)}" if seconds else "_高光时刻"
    if index > 1:
        stem += f"_{index}"
    return directory / f"{stem}.mp4"


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


def _write_audio(writer: _Writer, video: Path, pieces: Sequence[tuple[float, float]],
                 total_seconds: float, on_log: LogFn) -> None:
    """音频完全跟着画面的帧网格走，避免音画错位。

    pieces        = 画面实际写出去的每一段 `(源起点秒, 时长秒)`，顺序和写帧顺序一模一样
                    源起点 = 该段第一帧对应的源时间（floor(段起点*fps)/fps，不是段起点本身）
                    时长   = 该段真写了多少帧 / fps
    total_seconds = 输出总时长（含冻帧段），音频补静音补到和视频一样长

    多段拼接时这里逐段抽原声再首尾相接：画面剪掉的那几秒，音频在同一处也被跳过，
    所以"画面少 2 秒、音频只少 1 秒"这种错位不可能发生 —— 两边用的是同一份 pieces。
    每段样本数各自 round(时长*rate) 后再拼，不用"总时长一次算"：那样每段的舍入误差会
    累积到后面的段上，越往后越飘。

    冻帧段一律纯静音：高光成品不混任何音效。
    """
    if writer.audio is None:
        on_log("[AUDIO] 源视频没有可用音轨，输出为无声")
        return
    out_stream = writer.audio
    rate = int(out_stream.rate or 44100)
    blocks: list[np.ndarray] = []
    for piece_start, piece_seconds in pieces:
        want = max(0, int(round(piece_seconds * rate)))
        try:
            pcm = _decode_pcm(video, piece_start, piece_start + piece_seconds, rate)
        except Exception as exc:  # noqa: BLE001
            on_log(f"[AUDIO] 读音轨失败（{piece_start:.4f}s 起），这一段输出为无声：{exc}")
            pcm = np.zeros((2, want), dtype=np.float32)
        pcm = pcm[:, :want]
        if pcm.shape[1] < want:
            pcm = np.pad(pcm, ((0, 0), (0, want - pcm.shape[1])))
        blocks.append(pcm)
    live = (np.concatenate(blocks, axis=1) if len(blocks) > 1
            else (blocks[0] if blocks else np.zeros((2, 0), dtype=np.float32)))
    live_samples = live.shape[1]
    total_samples = max(live_samples, int(round(total_seconds * rate)))
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
    if len(pieces) == 1:
        piece_start, piece_seconds = pieces[0]
        on_log(f"[AUDIO] 原声 {piece_start:.4f} → {piece_start + piece_seconds:.4f}"
               f"（{live_samples / rate:.3f}s，与画面同一帧网格）"
               f" + 冻帧静音 {silence.shape[1] / rate:.3f}s，合计 {fed / rate:.3f}s")
    else:
        joined = " + ".join(f"{start:.2f}→{start + seconds:.2f}" for start, seconds in pieces)
        on_log(f"[AUDIO] 原声 {len(pieces)} 段拼接 {joined}"
               f"（{live_samples / rate:.3f}s，与画面同一帧网格、同一份分段）"
               f" + 冻帧静音 {silence.shape[1] / rate:.3f}s，合计 {fed / rate:.3f}s")



# ====================================================================== 分段规划
def _keep_spans_or_clip(spec: HighlightSpec,
                        keep_spans: Sequence[tuple[float, float]] | None,
                        ) -> list[tuple[float, float]]:
    """定下这次到底渲哪几段：给了 keep_spans 就只认它，没给就是 spec 的 sa→end 一整段。

    keep_spans 里的时间是**原视频绝对秒**、已按时间排好（上游 `trim_plan` 的口径），
    这里只做"能不能渲"的把关，不重排也不合并 —— 谁算的谁负责，渲染端擅自调整只会
    让日志里的段号和调用方手上的片段列表对不上。
    """
    if keep_spans is None:
        return [(spec.clip_start, spec.clip_end)]
    spans = [(float(start), float(end)) for start, end in keep_spans]
    if not spans:
        raise ValueError("keep_spans 给了空列表：要么别给（None），要么至少一段")
    for index, (start, end) in enumerate(spans):
        if start < 0:
            raise ValueError(f"keep_spans[{index}] 起点是负数：{start}")
        if end <= start:
            raise ValueError(f"keep_spans[{index}] 结束点({end})不大于起点({start})")
    return spans


def _frame_pieces(spans: Sequence[tuple[float, float]], fps: float) -> list[tuple[int, int]]:
    """把每段落到帧网格上：返回 `(起始帧号, 要写多少帧)`，顺序原样保留。

    每段都用和单段渲染完全相同的那把尺子（起点取 floor(t*fps) 这一刻正在显示的帧，
    帧数再被 floor 出来的帧号差夹一次），所以**只有一段时算出来的东西和以前一模一样**，
    多段只是把同一套算法重复 n 次。
    """
    pieces: list[tuple[int, int]] = []
    for start, end in spans:
        start_index = int(math.floor(start * fps))
        end_index = int(math.floor(end * fps))
        frames = max(1, min(int(round((end - start) * fps)), end_index - start_index))
        pieces.append((start_index, frames))
    return pieces


# ====================================================================== 主流程
def render_highlight(video: Path, spec: HighlightSpec, target: Path,
                     on_log: LogFn | None = None,
                     on_progress: ProgressFn | None = None,
                     freeze_seconds: float = 0.0,
                     keep_spans: Sequence[tuple[float, float]] | None = None) -> dict[str, Any]:
    """按 spec 生成高光 MP4，返回统计信息。on_progress 每写一帧回报一次进度。

    成品按顺序是：

        原速播放段（带原声）→ 最后一帧冻结 freeze_seconds 秒（静音）

    `keep_spans` 给了就**只认它**（原视频绝对秒，按时间排好，形如
    `[(80.48, 81.68), (83.48, 87.80)]`，一般来自 `clip_engine.trim_plan`），spec 的
    sa→end 这时只用来算"少掉了多少秒"；不给（None）走老路，就是 sa→end 一整段。
    多段时逐段解码、按顺序写进同一个文件，段与段之间直接接上（不做转场、不做淡入淡出），
    冻的是**最后一段的最后一帧**。

    冻帧是**额外追加**的，不占 spec 的时长预算：AI 报的 `timeline.duration` 已经把
    冻帧那几秒算进去了，所以剪辑区间只按 sa→end 走，冻帧在这里加回来。没有字幕、
    没有转场特效，也**一律不混音效**。

    freeze_seconds=0 就没有冻帧段。这个值来自 config.json 的
    `highlight.freeze_tail_seconds`，由调用方读进来。

    返回里 `play_frames` / `freeze_frames` / `total_frames` / `duration_seconds` 说的都是
    「最终成品里实际有多少」，也就是拼接之后的结果；另有 `pieces`（实际渲了几段，单段=1）
    和 `trimmed_seconds`（spec 区间总长 − keep_spans 总长，没给 keep_spans 时是 0.0）。


    落地方式是「先写 .part，完整收尾后 os.replace 成 target」：崩溃 / 被杀 / 断电
    只会留下 .part，target 要么是上一次的完整成品、要么根本不存在，绝不会是残片。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    report = on_progress or (lambda done, total, stage: None)
    info = probe_video(video)
    fps = float(info.fps)
    if fps <= 0:
        raise ValueError(f"读不到有效帧率：{info.fps}")
    spans = _keep_spans_or_clip(spec, keep_spans)
    for span_end in (end for _, end in spans):
        if span_end > info.duration + 1e-3:
            raise ValueError(f"结束点({span_end}) 超过视频时长({info.duration})")

    # 帧网格：起剪帧 = clip.start 这一刻正在显示的那一帧，结束帧同理
    pieces = _frame_pieces(spans, fps)
    start_index = pieces[0][0]
    end_index = int(math.floor(spans[-1][1] * fps))
    # play_frames 说的是"成品里到底有多少播放帧"，多段就是各段之和（拼接后的结果）
    play_frames = sum(frames for _, frames in pieces)
    grid_start = start_index / fps          # 输出第 0 帧对应的源时间
    # 少掉的秒数只对 keep_spans 有意义：没给的时候整段照渲，一秒都没剪

    trimmed_seconds = (max(0.0, spec.duration - sum(end - start for start, end in spans))
                       if keep_spans is not None else 0.0)
    # 冻帧段是额外追加的，不吃 spec.duration 的预算；
    # 进度和音频长度都得算上它。秒数填 0 就是 0 帧
    freeze_frames = max(0, int(round(fps * max(0.0, float(freeze_seconds)))))
    freeze_span = freeze_frames / fps
    out_frames = play_frames + freeze_frames

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
        f"音频起点与画面同为 {grid_start:.4f}s，"
        + (f"末帧冻结 {freeze_span:.2f}s（静音）" if freeze_frames else "不冻帧"))
    if len(pieces) > 1:
        detail = "｜".join(
            f"#{order} {span[0]:.2f}→{span[1]:.2f}（第 {piece[0]} 帧起 {piece[1]} 帧）"
            for order, (span, piece) in enumerate(zip(spans, pieces), 1))
        log(f"[JOIN] 保留 {len(pieces)} 段，按顺序直接接上（无转场、无淡入淡出）：{detail}")
        log(f"[JOIN] 区间 {spec.duration:.2f}s 里剪掉 {trimmed_seconds:.2f}s；"
            f"音频用同一份分段拼，画面少哪几秒音频就少哪几秒")

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
        last_rgb = None          # 播放段最后一帧：冻帧段就是把它重复写几秒
        # 音频要用的是"画面**实际**写了哪几段"，不是计划里的那几段：
        # 某段读帧提前断了也不会错位，两边始终是同一份清单
        audio_pieces: list[tuple[float, float]] = []
        report(0, out_frames, "正常播放段")
        try:
            for order, (piece_start_index, piece_frames) in enumerate(pieces, 1):
                # 进度按总帧数一路往前推，不在段边界上重置：用户看到的进度条不能来回跳
                stage = "正常播放段" if len(pieces) == 1 else f"拼接第 {order}/{len(pieces)} 段"
                cap.set(cv2.CAP_PROP_POS_FRAMES, piece_start_index)
                if len(pieces) == 1:
                    log(f"[{spans[0][0]:.2f}] START NORMAL PLAYBACK")
                else:
                    log(f"[{spans[order - 1][0]:.2f}] JOIN PIECE {order}/{len(pieces)}"
                        f" → {spans[order - 1][1]:.2f}（{piece_frames} 帧）")
                written_here = 0
                for _ in range(piece_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    last_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.write_rgb(last_rgb)
                    written_play += 1
                    written_here += 1
                    report(written_play, out_frames, stage)
                if written_here:
                    audio_pieces.append((piece_start_index / fps, written_here / fps))
                if written_here < piece_frames:
                    log(f"[JOIN] 第 {order} 段只读到 {written_here}/{piece_frames} 帧，"
                        f"音频按实际帧数跟着截短")
            if written_play <= 0:
                raise RuntimeError("一帧都没读到，检查起剪点 / 结束点是否落在视频范围内")
        finally:
            cap.release()

        # 冻帧：把播放段最后一帧原样重复几秒（静音）。多段时它就是**最后一段的最后一帧**。
        # AI 报的 duration 已经含这一段，剪辑区间只按 sa→end 走，这几秒在这里加回去
        written_freeze = 0
        if freeze_frames and last_rgb is not None:
            log(f"[FREEZE] 末帧冻结 {freeze_span:.2f}s（{freeze_frames} 帧，静音）")
            for _ in range(freeze_frames):
                writer.write_rgb(last_rgb)
                written_freeze += 1
                report(written_play + written_freeze, out_frames, "末帧冻结")

        report(out_frames, out_frames, "写音频并封装")
        # 画面可能比计划短，音频长度一律跟着**已写帧数**算，不用计划值
        _write_audio(writer, video, audio_pieces,
                     (written_play + written_freeze) / fps, log)
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

    out_frames_written = written_play + written_freeze
    out_duration = out_frames_written / fps
    log(f"[OUTPUT] {target}")
    log(f"[OUTPUT] 帧数 {out_frames_written}"
        f"（正常播放 {written_play} + 末帧冻结 {written_freeze}）"
        f"，时长 {out_duration:.3f}s，剪辑区间 {spec.duration:.2f}s"
        + (f"（{len(pieces)} 段拼接，剪掉 {trimmed_seconds:.2f}s）" if len(pieces) > 1 else "")
        + f" + 冻帧 {written_freeze / fps:.2f}s")
    return {
        "output": str(target),
        "clip_start": spec.clip_start,
        "clip_end": spec.clip_end,
        "fps": fps,
        "total_frames": out_frames_written,
        "play_frames": written_play,
        "freeze_frames": written_freeze,
        "freeze_seconds": round(written_freeze / fps, 3),
        "duration_seconds": round(out_duration, 3),
        "target_duration_seconds": round(spec.duration, 3),
        "grid_start_seconds": round(grid_start, 4),
        # 多段拼接的两笔账：渲了几段、因为拼接少掉了多少秒（单段渲染是 1 段 / 0.0 秒）
        "pieces": len(pieces),
        "trimmed_seconds": round(trimmed_seconds, 3),
    }

