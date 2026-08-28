"""高光剪辑核心：起剪点和冻帧点严格取自 AI JSON 的 clip.start / clip.end，绝不推算。

    clip.start   = 起始剪辑位置（配合界面「起始 加减秒数」）
    clip.end     = 高光冻帧位置（配合界面「结束 加减秒数」）
    overlay.time = 不参与时间计算，只从 overlay 里取要显示的文本
    片尾         = 冻帧点 + 界面「文本 加减秒数」，也就是冻帧+字幕这一段的时长

执行时间轴（以 clip.start=20.68 / clip.end=24.00 / 文本加减=1.50 为例）：

    20.68 ── 正常播放原视频 ──> 24.00 ── 抓取该时刻最后有效帧 ──>
    24.00 ~ 25.50 全程使用这一张冻结帧：Zoom Punch + Flash + 画面增强 + 逐字弹出字幕
    音频跟着画面一起冻：原声只到 24.00，24.00 ~ 25.50 是静音
    25.50 结束，最终时长 = (冻帧点 - 起剪点) + 文本加减秒数。
    冻帧段是拿一张静态帧合成出来的，不读源视频，所以片尾允许超出原视频时长。


实现上复用项目既有栈：cv2 抓帧（同 video_io 的 seek 习惯）、PIL 画字与特效、
PyAV 编码封装（同 audio.py 的用法）。不依赖外部 ffmpeg 可执行文件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from ..logging_setup import get_logger
from ..video_io import probe_video

logger = get_logger("highlight")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]   # (已完成帧, 总帧, 当前阶段)

# --- 字幕样式（第八节）---
TEXT_COLOR = (0x16, 0x83, 0xFF)      # #1683FF 蓝
STROKE_COLOR = (0, 0, 0)             # 黑色描边
TEXT_HEIGHT_RATIO = 0.075            # 字号 ≈ 画面高度的 7.5%
TEXT_CENTER_Y_RATIO = 0.80           # 纵向 75%~85% 区间的中线
TEXT_SAFE_WIDTH_RATIO = 0.84         # 左右各留 8% 安全边距

# --- 冻帧特效（第九节）---
ZOOM_KEYS = ((0.00, 1.00), (0.10, 1.06), (0.18, 1.04), (0.25, 1.05))
FLASH_SECONDS = 0.08
FLASH_STRENGTH = 0.22
POP_SECONDS = 0.18                   # 单个字的弹出时长
CHAR_STEP_SECONDS = 0.12             # 字与字之间的间隔

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑 Bold
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


# ====================================================================== JSON
@dataclass
class Overlay:
    """AI JSON 里 overlays 下的一项。只用 text；time 不参与时间计算，留着仅供日志核对。"""

    name: str
    time: float
    text: str
    kind: str = ""


@dataclass
class HighlightSpec:
    video_name: str
    clip_start: float
    clip_end: float
    freeze_time: float
    freeze_text: str
    freeze_overlays: list[Overlay] = field(default_factory=list)
    other_overlays: list[Overlay] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.clip_end - self.clip_start

    @property
    def freeze_duration(self) -> float:
        return self.clip_end - self.freeze_time

    def shifted(self, start_delta: float, end_delta: float, text_delta: float) -> HighlightSpec:
        """按界面上填的加减秒数算出三个时间。

        起剪点 = clip.start + 起始加减
        冻帧点 = clip.end   + 结束加减
        片尾   = 冻帧点     + 文本加减（冻帧+字幕这段的时长，0 表示只留一帧）
        """
        start = round(self.clip_start + start_delta, 3)
        freeze = round(self.freeze_time + end_delta, 3)
        end = round(freeze + text_delta, 3)
        if start < 0:
            raise ValueError(f"start 偏移后变成负数：{start}")
        if freeze <= start:
            raise ValueError(f"偏移后冻帧点({freeze})不大于起剪点({start})")
        if end < freeze:
            raise ValueError(f"文本加减秒数({text_delta})为负，片尾({end})落在冻帧点({freeze})之前")
        return replace(self, clip_start=start, clip_end=end, freeze_time=freeze)



def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} 不是数字：{value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数字：{value!r}") from exc


def parse_spec(payload: dict[str, Any]) -> HighlightSpec:
    """解析 AI JSON。clip.start = 起剪点，clip.end = 冻帧点，两个时间原样使用不做推算。

    overlay.time 不参与时间计算（AI 给的这个值不可靠），只从 overlay 里取字幕文本。
    片尾在 shifted() 里由「文本 加减秒数」决定，这里先让 clip_end 等于冻帧点。
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

    overlays: list[Overlay] = []
    raw_overlays = clip.get("overlays")
    if isinstance(raw_overlays, dict):
        items = raw_overlays.items()
    elif isinstance(raw_overlays, list):  # 也接受数组写法
        items = [(str(i), v) for i, v in enumerate(raw_overlays)]
    else:
        items = []
    for name, item in items:
        if not isinstance(item, dict) or item.get("time") is None:
            continue  # evaluation 这种说明性字段直接跳过
        text = str(item.get("text") or "")
        overlays.append(Overlay(name=str(name), time=_as_float(item["time"], f"overlays.{name}.time"),
                                text=text, kind=str(item.get("kind") or "")))
    if not overlays:
        raise ValueError("overlays 里没有可用条目，无法取到字幕文本")

    # 字幕文本：优先 kind=comment，其次最后一条有文字的；time 一律不看
    with_text = [o for o in overlays if o.text.strip()]
    if not with_text:
        raise ValueError("overlays 里没有 text，无法生成字幕")
    comments = [o for o in with_text if o.kind == "comment"]
    chosen = (comments or with_text)[-1]
    freeze_text = chosen.text.strip()

    other = [o for o in overlays if o is not chosen]
    return HighlightSpec(
        video_name=str(payload.get("video") or clip.get("video") or ""),
        clip_start=start, clip_end=end, freeze_time=end, freeze_text=freeze_text,
        freeze_overlays=[chosen], other_overlays=other, raw=payload,
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


# ==================================================================== 字幕排版
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:  # noqa: BLE001 - 字体损坏就换下一个
                continue
    logger.warning("找不到可用的粗体字体，退回 PIL 默认字体（中文可能显示为方块）")
    return ImageFont.load_default()


@dataclass
class _Glyph:
    char: str
    image: Image.Image   # 已带描边的 RGBA 小图
    center_x: float
    center_y: float
    appear_at: float     # 相对冻帧点的秒数


def _fit_font(text: str, width: int, height: int) -> tuple[ImageFont.FreeTypeFont, int]:
    size = max(16, int(height * TEXT_HEIGHT_RATIO))
    limit = width * TEXT_SAFE_WIDTH_RATIO
    font = _load_font(size)
    while size > 16:
        tracking = size * 0.06
        total = sum(font.getlength(ch) for ch in text) + tracking * max(len(text) - 1, 0)
        if total <= limit:
            break
        size -= 2
        font = _load_font(size)
    return font, size


def _build_glyphs(text: str, width: int, height: int, char_step: float) -> list[_Glyph]:
    """把整句话排成横向居中、纵向靠底部的一排字，每个字单独出一张带描边的小图。"""
    font, size = _fit_font(text, width, height)
    stroke = max(3, min(6, size // 14))
    tracking = size * 0.06
    advances = [font.getlength(ch) for ch in text]
    total = sum(advances) + tracking * max(len(text) - 1, 0)
    pen = (width - total) / 2.0

    # 纵向：整行文字的外框中心对准 height*TEXT_CENTER_Y_RATIO，
    # 这样字块落在 75%~85% 区间里，而不是被 bbox 偏移拖到画面更下方
    boxes = [font.getbbox(ch, stroke_width=stroke) for ch in text]
    block_top = min(b[1] for b in boxes)
    block_bottom = max(b[3] for b in boxes)
    block_center = (block_top + block_bottom) / 2.0
    baseline_y = height * TEXT_CENTER_Y_RATIO - block_center

    glyphs: list[_Glyph] = []
    for i, ch in enumerate(text):
        box = boxes[i]
        pad = stroke + 4
        w = max(int(box[2] - box[0]) + pad * 2, 2)
        h = max(int(box[3] - box[1]) + pad * 2, 2)
        tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((pad - box[0], pad - box[1]), ch, font=font, fill=TEXT_COLOR,
                                  stroke_width=stroke, stroke_fill=STROKE_COLOR)
        glyphs.append(_Glyph(
            char=ch, image=tile,
            center_x=pen + advances[i] / 2.0,
            center_y=baseline_y + (box[1] + box[3]) / 2.0,
            appear_at=i * char_step,
        ))
        pen += advances[i] + tracking
    return glyphs


def _pop_scale(elapsed: float, pop: float) -> float:
    """0.8 -> 1.15 -> 0.95 -> 1.0 的快速弹出，超过 pop 秒后固定 1.0。"""
    if elapsed >= pop:
        return 1.0
    r = elapsed / pop
    if r < 0.35:
        return 0.80 + (1.15 - 0.80) * (r / 0.35)
    if r < 0.70:
        return 1.15 + (0.95 - 1.15) * ((r - 0.35) / 0.35)
    return 0.95 + (1.00 - 0.95) * ((r - 0.70) / 0.30)


def _zoom_scale(elapsed: float) -> float:
    if elapsed >= ZOOM_KEYS[-1][0]:
        return ZOOM_KEYS[-1][1]
    prev_t, prev_v = ZOOM_KEYS[0]
    for t, v in ZOOM_KEYS[1:]:
        if elapsed <= t:
            span = max(t - prev_t, 1e-6)
            return prev_v + (v - prev_v) * ((elapsed - prev_t) / span)
        prev_t, prev_v = t, v
    return ZOOM_KEYS[-1][1]


def _zoom(image: Image.Image, scale: float) -> Image.Image:
    if scale <= 1.0001:
        return image
    w, h = image.size
    cw, ch = int(round(w / scale)), int(round(h / scale))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


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
                 sample_aspect_ratio: Fraction | None = None) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(target), mode="w")
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
        for packet in self.stream.encode():
            self.container.mux(packet)
        if self.audio is not None:
            for packet in self.audio.encode():
                self.container.mux(packet)
        self.container.close()


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
    total_seconds = 输出总时长（total_frames/fps），音频补静音补到和视频一样长
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
           f" + 冻帧静音 {silence.shape[1] / rate:.3f}s，合计 {fed / rate:.3f}s")


# ====================================================================== 主流程
def render_highlight(video: Path, spec: HighlightSpec, target: Path,
                     on_log: LogFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
    """按 spec 生成高光 MP4，返回统计信息。on_progress 每写一帧回报一次进度。"""
    log = on_log or (lambda line: logger.info("%s", line))
    report = on_progress or (lambda done, total, stage: None)
    info = probe_video(video)
    fps = float(info.fps)
    if fps <= 0:
        raise ValueError(f"读不到有效帧率：{info.fps}")
    # 冻帧段是合成出来的，不读源视频，所以只校验冻帧点必须落在视频里，片尾可以超出
    if spec.freeze_time > info.duration + 1e-3:
        raise ValueError(f"冻帧点({spec.freeze_time}) 超过视频时长({info.duration})")

    # 帧网格：起剪帧 = clip.start 这一刻正在显示的那一帧；冻帧 = 冻帧点这一刻正在显示的那一帧

    start_index = int(math.floor(spec.clip_start * fps))
    freeze_index = int(math.floor(spec.freeze_time * fps))
    total_frames = max(1, int(round(spec.duration * fps)))
    play_frames = max(0, min(total_frames, freeze_index - start_index))
    freeze_frames = total_frames - play_frames
    if freeze_frames <= 0:  # freeze 恰好等于 end：至少留一帧冻帧，仍不超总长
        freeze_frames = 1
        play_frames = total_frames - 1
    grid_start = start_index / fps          # 输出第 0 帧对应的源时间
    live_seconds = play_frames / fps        # 正常播放段时长，画面和音频共用
    total_seconds = total_frames / fps

    log("[HIGHLIGHT]")
    log(f"Clip Start : {spec.clip_start:.2f}")
    log(f"Freeze Point: {spec.freeze_time:.2f}")
    log(f"Clip End   : {spec.clip_end:.2f}")
    log("")
    log(f"[VIDEO] {video.name}  {info.width}x{info.height}  {fps:g} fps  音轨={info.has_audio}")
    sar = _sample_aspect_ratio(video)
    log(f"[SIZE] 输出保持原分辨率 {info.width}x{info.height}"
        + (f"，像素比 SAR={sar} 一并沿用" if sar else "，方形像素，比例与源一致"))
    log(f"[ALIGN] 起剪第 {start_index} 帧（源 {grid_start:.4f}s）｜"
        f"冻帧第 {freeze_index} 帧（源 {freeze_index / fps:.4f}s）｜"
        f"音频起点与画面同为 {grid_start:.4f}s，冻帧处同时静音")
    for overlay in spec.other_overlays:
        log(f"OVERLAY 未使用: {overlay.name} = {overlay.text}"
            f"（只取一条字幕，overlay.time 不参与时间计算）")


    # ---- 逐字动画时间表：全部压在 freeze -> end 之内 ----
    avail = spec.freeze_duration
    text = spec.freeze_text
    pop = min(POP_SECONDS, avail * 0.9)
    if len(text) > 1:
        step = min(CHAR_STEP_SECONDS, max(0.0, avail - pop) / (len(text) - 1))
    else:
        step = 0.0
    glyphs = _build_glyphs(text, info.width, info.height, step)

    writer = _Writer(target, info.width, info.height, fps, audio_rate=_audio_rate(video),
                     sample_aspect_ratio=sar)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        writer.close()
        raise RuntimeError(f"OpenCV 打不开视频：{video}")

    freeze_bgr: np.ndarray | None = None
    written_play = 0
    report(0, total_frames, "正常播放段")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        log(f"[{spec.clip_start:.2f}] START NORMAL PLAYBACK")
        last_bgr: np.ndarray | None = None
        for _ in range(play_frames):
            ok, frame = cap.read()
            if not ok:
                break
            last_bgr = frame
            writer.write_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            written_play += 1
            report(written_play, total_frames, "正常播放段")
        # 冻帧点这一刻的最后有效帧
        ok, frame = cap.read()
        freeze_bgr = frame if ok else last_bgr
        if freeze_bgr is None:
            raise RuntimeError("抓不到高光冻帧，检查起剪点 / 冻帧点是否落在视频范围内")

    finally:
        cap.release()

    log(f"[{spec.freeze_time:.2f}] FREEZE HIGH LIGHT FRAME"
        f"（原视频第 {start_index + written_play} 帧）")
    log(f"[{spec.freeze_time:.2f} → {spec.clip_end:.2f}] FREEZE EFFECT")

    base = Image.fromarray(cv2.cvtColor(freeze_bgr, cv2.COLOR_BGR2RGB))
    base = ImageEnhance.Color(base).enhance(1.06)      # 轻微画面增强
    base = ImageEnhance.Contrast(base).enhance(1.04)
    base = ImageEnhance.Brightness(base).enhance(1.02)

    for glyph in glyphs:
        log(f"[{spec.freeze_time + glyph.appear_at:.2f}] TEXT: {glyph.char}")

    zoom_cache: dict[int, Image.Image] = {}
    white = Image.new("RGB", base.size, (255, 255, 255))
    for i in range(freeze_frames):
        elapsed = i / fps
        scale = _zoom_scale(elapsed)
        key = int(round(scale * 1000))
        canvas = zoom_cache.get(key)
        if canvas is None:
            canvas = _zoom(base, scale)
            zoom_cache[key] = canvas
        canvas = canvas.copy()
        if elapsed < FLASH_SECONDS:
            alpha = FLASH_STRENGTH * (1.0 - elapsed / FLASH_SECONDS)
            canvas = Image.blend(canvas, white, alpha)
        for glyph in glyphs:
            if elapsed + 1e-9 < glyph.appear_at:
                continue  # 还没到这个字
            tile = glyph.image
            char_scale = _pop_scale(elapsed - glyph.appear_at, pop)
            if abs(char_scale - 1.0) > 1e-3:
                size = (max(1, int(round(tile.width * char_scale))),
                        max(1, int(round(tile.height * char_scale))))
                tile = tile.resize(size, Image.LANCZOS)
            canvas.paste(tile, (int(round(glyph.center_x - tile.width / 2)),
                                int(round(glyph.center_y - tile.height / 2))), tile)
        writer.write_rgb(np.asarray(canvas))
        report(written_play + i + 1, total_frames, "冻帧特效")

    report(total_frames, total_frames, "写音频并封装")
    _write_audio(writer, video, grid_start, live_seconds, total_seconds, log)
    writer.close()
    log(f"[{spec.clip_end:.2f}] END CLIP")

    out_duration = total_frames / fps
    log(f"[OUTPUT] {target}")
    log(f"[OUTPUT] 帧数 {total_frames}（正常播放 {written_play} + 冻帧 {freeze_frames}）"
        f"，时长 {out_duration:.3f}s，目标 {spec.duration:.2f}s")
    return {
        "output": str(target),
        "clip_start": spec.clip_start,
        "freeze_time": spec.freeze_time,
        "clip_end": spec.clip_end,
        "text": text,
        "fps": fps,
        "total_frames": total_frames,
        "play_frames": written_play,
        "freeze_frames": freeze_frames,
        "duration_seconds": round(out_duration, 3),
        "target_duration_seconds": round(spec.duration, 3),
        "grid_start_seconds": round(grid_start, 4),
        "freeze_at_output_seconds": round(live_seconds, 4),
        "char_step_seconds": round(step, 3),
        "pop_seconds": round(pop, 3),
    }
