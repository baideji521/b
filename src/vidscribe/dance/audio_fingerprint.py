"""音频读取与指纹：对齐层的地基。

三件事：
1. `extract_analysis_audio()` —— 用 **PyAV**（项目既有解码栈，不引入外部 ffmpeg.exe）
   把任意媒体的音轨解成 mono / float32 / 固定采样率的 numpy 数组。
   技术指导第五节第 1 条明确要求"不要因为参考项目用 FFmpeg 就改变 b 的基础解码架构"。
2. `peak_ratio_confidence()` —— 主峰与次强峰的比值。这是判断"这个 offset 到底可不可信"
   的核心：主峰高但次峰同样高，说明这段音频自己在周期性重复（副歌、循环鼓点），
   offset 完全可能整拍错位。
3. 指纹与缓存键 —— 目标歌只算一次，重复运行直接复用（技术指导第二十四节）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_setup import get_logger
from .dsp import DEFAULT_SR

logger = get_logger("dance.audio")

#: 指纹只读文件的头/中/尾各这么多字节，和 db/fingerprint.py 同一套思路
_CHUNK = 1 << 20
#: 峰值比的护栏半径（秒）：主峰左右这么宽的范围不算"次强峰"，
#: 否则主峰自己的裙边会被当成竞争者，比值恒等于 1
GUARD_SECONDS = 0.25


class AudioReadError(RuntimeError):
    """音轨读不出来。**不返回空数组** —— 静默返回空会让上层算出一个假 offset。"""


def _resample(resampler: Any, frame: Any) -> list[Any]:
    """PyAV 各版本 resample() 有的返回单帧、有的返回列表，统一成列表（同 audio.py）。"""
    out = resampler.resample(frame)
    if out is None:
        return []
    return out if isinstance(out, list) else [out]


def media_audio_duration(media_path: str | Path) -> float:
    """音轨时长（秒）。没有音轨返回 0.0，不抛异常 —— 调用方要用它做"能不能对齐"的预检。"""
    import av  # noqa: PLC0415

    try:
        with av.open(str(media_path)) as src:
            if not src.streams.audio:
                return 0.0
            stream = src.streams.audio[0]
            if stream.duration and stream.time_base:
                return round(float(stream.duration * stream.time_base), 3)
            if src.duration:
                return round(float(src.duration) / 1_000_000.0, 3)
    except Exception as exc:  # noqa: BLE001 - 探测失败当没有音轨，由调用方决定怎么办
        logger.debug("探音轨时长失败 %s：%s", media_path, exc)
    return 0.0


def extract_analysis_audio(media_path: str | Path, start: float = 0.0,
                           duration: float | None = None,
                           sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """把媒体音轨解成 mono / float32 / `sample_rate` 的一维数组。

    `start` / `duration` 是**源媒体时间**（秒）；`duration=None` 表示读到结尾。
    seek 只用来粗定位（往前多留 1 秒），真正的裁切按解码出来的 pts 算，
    所以返回数组的第 0 个采样严格对应 `start` 那一刻 —— 对齐算法的精度全押在这上面。

    读不到音轨 / 解码失败一律抛 `AudioReadError`。
    """
    import av  # noqa: PLC0415

    path = Path(media_path)
    if not path.is_file():
        raise AudioReadError(f"文件不存在：{path}")
    start = max(0.0, float(start))
    want = None if duration is None else max(0.0, float(duration))
    if want is not None and want <= 0:
        return np.zeros(0, dtype=np.float32)

    chunks: list[np.ndarray] = []
    base: float | None = None
    try:
        with av.open(str(path)) as src:
            if not src.streams.audio:
                raise AudioReadError(f"{path.name} 没有音轨，无法做音乐对齐")
            stream = src.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=int(sample_rate))
            if start > 1.0 and stream.time_base:
                # seek 的偏移量在**流自己的 time_base** 里，所以是「秒 ÷ time_base」。
                # 写成「秒 × av.time_base」或「秒 ÷ av.time_base」都是错的：前者对 WAV
                # （time_base=1/22050）会跳到几百秒之外，后者恒等于 0 等于没 seek。
                # 往前多留 1 秒，精确起点靠后面按 pts 裁。
                src.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream)

            stop = None if want is None else start + want
            for frame in src.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                at = float(frame.pts * stream.time_base)
                if at + float(frame.samples) / float(frame.sample_rate or sample_rate) <= start:
                    continue
                if stop is not None and at >= stop:
                    break
                for piece in _resample(resampler, frame):
                    if base is None:
                        base = at
                    chunks.append(piece.to_ndarray().reshape(-1))
            for piece in _resample(resampler, None):      # 冲掉重采样器里剩的样本
                if piece is not None and piece.samples:
                    chunks.append(piece.to_ndarray().reshape(-1))
    except AudioReadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AudioReadError(f"{path.name} 解码音轨失败：{type(exc).__name__}: {exc}") from exc

    if not chunks or base is None:
        raise AudioReadError(f"{path.name} 在 {start:.3f}s 起没解出任何音频样本")
    pcm = np.concatenate(chunks).astype(np.float32, copy=False)
    offset = max(0, int(round((start - base) * sample_rate)))
    pcm = pcm[offset:]
    if want is not None:
        pcm = pcm[:int(round(want * sample_rate))]
    return np.ascontiguousarray(pcm, dtype=np.float32)


# ================================================================== 峰值置信度
def peak_ratio_confidence(correlation: np.ndarray, peak_index: int,
                          guard_samples: int) -> float:
    """主峰 / 最强次峰 的比值，压到 0~1 当置信度。

    思路来自 `audio-video-sync/sync.py` 的 peak ratio，在此重新实现并补齐边界：

    - 空数组 / `peak_index` 越界        → 0.0（什么都没法判断）
    - 主峰 ≈ 0（整条相关都是噪声）      → 0.0
    - 次峰 ≈ 0（护栏外一片平地）        → 1.0（主峰独一无二，最可信）
    - `guard_samples` 大到把整条都盖住   → 1.0，并且**不**报错：护栏比数据长
                                          说明这段太短，只有一个峰可比

    映射用 `1 - 1/ratio`：ratio=1（主次一样高）→ 0，ratio=2 → 0.5，ratio=5 → 0.8，
    ratio→∞ → 1。比线性截断更符合"次峰只要还在一个量级，就别太自信"的直觉。
    """
    corr = np.abs(np.asarray(correlation, dtype=np.float64).reshape(-1))
    if corr.size == 0:
        return 0.0
    index = int(peak_index)
    if index < 0 or index >= corr.size:
        return 0.0
    primary = float(corr[index])
    if primary <= 1e-12:
        return 0.0
    guard = max(0, int(guard_samples))
    lo = max(0, index - guard)
    hi = min(corr.size, index + guard + 1)
    outside = np.concatenate([corr[:lo], corr[hi:]])
    if outside.size == 0:
        return 1.0
    secondary = float(outside.max())
    if secondary <= 1e-12:
        return 1.0
    ratio = primary / secondary
    if ratio <= 1.0:
        return 0.0
    return round(float(min(1.0, 1.0 - 1.0 / ratio)), 4)


def peak_of(correlation: np.ndarray) -> int:
    """相关序列的主峰下标。空数组返回 -1（调用方必须判）。"""
    corr = np.asarray(correlation, dtype=np.float64).reshape(-1)
    if corr.size == 0:
        return -1
    return int(np.argmax(corr))


# ==================================================================== 指纹
def file_fingerprint(path: str | Path) -> str:
    """文件指纹 `<字节数>-<sha1前16>`，只读头/中/尾各 1MB。

    和 `db/fingerprint.py` 同一套思路（改名换目录仍认得出是同一份），但独立实现：
    那边算的是"视频身份"，这边算的是"对齐缓存键的一部分"，两者不该互相牵连。
    """
    target = Path(path)
    size = target.stat().st_size
    digest = hashlib.sha1()
    digest.update(str(size).encode("ascii"))
    with open(target, "rb") as fh:
        digest.update(fh.read(_CHUNK))
        if size > _CHUNK * 2:
            fh.seek(max(0, size // 2 - _CHUNK // 2))
            digest.update(fh.read(_CHUNK))
        if size > _CHUNK:
            fh.seek(max(0, size - _CHUNK))
            digest.update(fh.read(_CHUNK))
    return f"{size}-{digest.hexdigest()[:16]}"


def audio_fingerprint(pcm: np.ndarray, sample_rate: int = DEFAULT_SR) -> str:
    """波形指纹：直接对采样字节求 sha1。

    用在"目标歌只算一次"上：同一首歌无论从哪个容器里解出来，只要解码结果一致，
    指纹就一致，对齐缓存就能命中。
    """
    pcm = np.ascontiguousarray(np.asarray(pcm, dtype=np.float32).reshape(-1))
    digest = hashlib.sha1()
    digest.update(f"{sample_rate}:{pcm.size}:".encode("ascii"))
    digest.update(pcm.tobytes())
    return f"{pcm.size}-{digest.hexdigest()[:16]}"


def config_hash(payload: Any) -> str:
    """配置指纹（sha1 前 16）。`sort_keys` 是关键：手改配置调换键顺序不该让缓存失效。"""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def alignment_cache_key(source_fp: str, target_fp: str, algorithm_version: str,
                        cfg_hash: str) -> str:
    """对齐缓存键 = 源指纹 + 目标指纹 + 算法版本 + 配置指纹（技术指导第二十四节）。

    四项缺一不可：少了算法版本，改完算法还在吃旧结果；少了配置指纹，
    改了窗口数/采样率也不会重算。
    """
    return config_hash([source_fp, target_fp, algorithm_version, cfg_hash])


__all__ = [
    "AudioReadError", "GUARD_SECONDS",
    "media_audio_duration", "extract_analysis_audio",
    "peak_ratio_confidence", "peak_of",
    "file_fingerprint", "audio_fingerprint", "config_hash", "alignment_cache_key",
]



