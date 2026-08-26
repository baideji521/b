"""视频探测 + 内存帧采样 + 镜头切换检测。

原则：不落地 JPG 中间文件，全部在内存中完成；帧时间戳一律取解码器返回的真实位置。
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .constants import PIXEL_FACTOR, VIDEO_SUFFIXES
from .logging_setup import get_logger

logger = get_logger(__name__)

__all__ = [
    "PIXEL_FACTOR", "VIDEO_SUFFIXES", "VideoInfo", "FrameBatch", "list_videos", "probe_video",
    "smart_size", "plan_frame_indices", "sample_frames", "detect_scene_cuts", "plan_windows",
]


@dataclass
class VideoInfo:
    path: str
    name: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameBatch:
    """一次窗口采样的结果，全部在内存中。"""

    images: list[Image.Image] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)  # 真实秒数
    frame_indices: list[int] = field(default_factory=list)  # 原视频帧号
    sample_fps: float = 0.0
    resized_width: int = 0
    resized_height: int = 0

    def __len__(self) -> int:
        return len(self.images)


def list_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def _probe_with_av(path: Path) -> dict:
    """用 PyAV（faster-whisper 的依赖，无需外部 ffmpeg.exe）读取容器元信息。"""
    result: dict = {}
    try:
        import av  # noqa: PLC0415
    except Exception:
        return result
    try:
        with av.open(str(path)) as container:
            if container.duration:
                result["duration"] = float(container.duration) / 1_000_000.0
            vstreams = [s for s in container.streams if s.type == "video"]
            astreams = [s for s in container.streams if s.type == "audio"]
            result["has_audio"] = len(astreams) > 0
            if astreams and astreams[0].codec_context:
                result["audio_codec"] = astreams[0].codec_context.name
            if vstreams:
                vs = vstreams[0]
                if vs.codec_context:
                    result["video_codec"] = vs.codec_context.name
                if vs.average_rate:
                    result["fps"] = float(vs.average_rate)
                if vs.frames:
                    result["total_frames"] = int(vs.frames)
                if vs.duration and vs.time_base:
                    result.setdefault("duration", float(vs.duration * vs.time_base))
                if vs.codec_context is not None:
                    result["width"] = int(vs.codec_context.width or 0)
                    result["height"] = int(vs.codec_context.height or 0)
    except Exception as exc:  # 损坏或不支持的容器
        logger.debug("PyAV 探测失败 %s: %s", path.name, exc)
    return result


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")

    meta = _probe_with_av(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        if not meta:
            raise RuntimeError(f"无法打开视频: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    fps = fps if fps > 0.1 else float(meta.get("fps") or 0.0)
    width = width or int(meta.get("width") or 0)
    height = height or int(meta.get("height") or 0)
    total_frames = total_frames if total_frames > 0 else int(meta.get("total_frames") or 0)

    duration = float(meta.get("duration") or 0.0)
    if duration <= 0 and fps > 0 and total_frames > 0:
        duration = total_frames / fps
    if total_frames <= 0 and fps > 0 and duration > 0:
        total_frames = int(round(duration * fps))
    if fps <= 0.1:
        fps = 25.0
        logger.warning("%s 未能读到 FPS，回退为 25", path.name)

    return VideoInfo(
        path=str(path.resolve()),
        name=path.name,
        duration=round(duration, 3),
        fps=round(fps, 5),
        width=width,
        height=height,
        total_frames=total_frames,
        has_audio=bool(meta.get("has_audio", _has_audio_fallback(path))),
        video_codec=meta.get("video_codec"),
        audio_codec=meta.get("audio_codec"),
    )


def _has_audio_fallback(path: Path) -> bool:
    """PyAV 不可用时用 ffprobe 兜底；都不可用则保守认为有音频，交给 ASR 阶段判定。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return True


def smart_size(width: int, height: int, max_pixels: int, min_pixels: int = 4 * PIXEL_FACTOR ** 2) -> tuple[int, int]:
    """把分辨率缩放到像素预算内，并且宽高都对齐到 32 的倍数（Qwen3-VL 要求）。"""
    if width <= 0 or height <= 0:
        return PIXEL_FACTOR, PIXEL_FACTOR
    scale = 1.0
    pixels = width * height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    w = max(PIXEL_FACTOR, int(round(width * scale / PIXEL_FACTOR)) * PIXEL_FACTOR)
    h = max(PIXEL_FACTOR, int(round(height * scale / PIXEL_FACTOR)) * PIXEL_FACTOR)
    while w * h > max_pixels and (w > PIXEL_FACTOR or h > PIXEL_FACTOR):
        if w >= h and w > PIXEL_FACTOR:
            w -= PIXEL_FACTOR
        elif h > PIXEL_FACTOR:
            h -= PIXEL_FACTOR
        else:
            break
    return w, h


def plan_frame_indices(info: VideoInfo, start: float, end: float, fps: float,
                       min_frames: int, max_frames: int) -> list[int]:
    """按目标 fps 在 [start, end) 内均匀取帧号；帧数强制为偶数（temporal_patch_size=2）。"""
    start = max(0.0, start)
    end = min(end, info.duration if info.duration > 0 else end)
    span = max(end - start, 1.0 / max(info.fps, 1.0))
    want = int(round(span * fps))
    want = max(min_frames, min(max_frames, want))
    want = max(2, want - want % 2)

    first = int(math.floor(start * info.fps))
    last = int(math.ceil(end * info.fps)) - 1
    upper = (info.total_frames - 1) if info.total_frames > 0 else last
    last = min(last, upper)
    first = min(first, last)
    if last <= first:
        return [max(0, first)] * 2

    positions = np.linspace(first, last, want)
    indices = sorted({int(round(p)) for p in positions})
    if len(indices) % 2 == 1 and len(indices) > 1:
        indices = indices[:-1]
    if len(indices) < 2:
        indices = [indices[0], min(indices[0] + 1, last)]
    return indices


def sample_frames(info: VideoInfo, frame_indices: Iterable[int], max_pixels: int) -> FrameBatch:
    """按帧号在内存中抓帧，返回 RGB PIL 图像 + 解码器报告的真实时间戳。"""
    wanted = list(frame_indices)
    batch = FrameBatch()
    if not wanted:
        return batch

    target_w, target_h = smart_size(info.width, info.height, max_pixels)
    batch.resized_width, batch.resized_height = target_w, target_h

    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频进行采样: {info.path}")
    try:
        current = -1
        for idx in wanted:
            if idx != current + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            ok, frame = cap.read()
            if not ok or frame is None:
                current = -1
                continue
            current = pos_frames
            ts = pos_msec / 1000.0 if pos_msec > 0 else pos_frames / max(info.fps, 1e-6)
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            batch.images.append(image)
            batch.timestamps.append(round(ts, 3))
            batch.frame_indices.append(pos_frames)
    finally:
        cap.release()

    if len(batch) > 1:
        span = batch.timestamps[-1] - batch.timestamps[0]
        batch.sample_fps = round((len(batch) - 1) / span, 4) if span > 0.01 else 1.0
    else:
        batch.sample_fps = 1.0
    return batch


def detect_scene_cuts(info: VideoInfo, sample_fps: float = 3.0, threshold: float = 0.35) -> list[float]:
    """基于 HSV 直方图差异的镜头切换检测，返回真实秒数的切点列表（不含 0 和 duration）。

    这一步的时间戳完全来自解码器，是 timeline 里 frame_based 边界的来源。
    """
    if info.duration <= 0:
        return []
    step = max(1, int(round(info.fps / max(sample_fps, 0.1))))
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        cap.release()
        return []

    cuts: list[float] = []
    prev_hist = None
    frame_no = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_no % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    if prev_hist is not None:
                        diff = 1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                        if diff >= threshold:
                            ts = frame_no / max(info.fps, 1e-6)
                            if not cuts or ts - cuts[-1] > 0.6:
                                cuts.append(round(ts, 3))
                    prev_hist = hist
            frame_no += 1
    finally:
        cap.release()
    logger.info("镜头切换检测：%d 个切点", len(cuts))
    return cuts


def plan_windows(duration: float, cuts: list[float], window_seconds: float,
                 overlap_seconds: float, long_threshold: float) -> list[tuple[float, float]]:
    """规划分析窗口。

    - 短视频（<= long_threshold）：整段一个窗口。
    - 中/长视频：以镜头切点为优先边界，窗口长度上限 window_seconds，窗口之间保留 overlap。
    """
    if duration <= 0:
        return [(0.0, 0.0)]
    if duration <= long_threshold:
        return [(0.0, round(duration, 3))]

    stride = max(window_seconds - overlap_seconds, window_seconds / 2.0)
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.05:
        end = min(start + window_seconds, duration)
        # 若窗口尾部附近有镜头切点，把边界对齐到切点，避免事件被硬切
        candidates = [c for c in cuts if start + window_seconds * 0.6 < c < end]
        if candidates and end < duration:
            end = candidates[-1]
        windows.append((round(start, 3), round(end, 3)))
        if end >= duration - 0.05:
            break
        start = max(end - overlap_seconds, start + stride * 0.5)
    return windows
