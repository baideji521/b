"""高光冻帧的音效：从 assets/sfx/<类别>/ 里挑一条，混进冻帧那一刻。

冻帧段的音频原本是纯静音（原声只到冻帧点，见 clip._write_audio），Zoom Punch + Flash +
逐字弹字全程无声。音效就填这段静音，不动原声，也不做压侧链。

类别由表情轨（timeline.json 的 expression_track，人脸 HSEmotion）决定：冻帧点落在哪个
span 就取那个 span 的 emotion_en，查 highlight.sfx.emotion_map 得到类别目录。表情轨没覆盖
到冻帧点（没检到人脸）就用 fallback_category，不猜。

素材是 CC0（kenney.nl），用 tools/fetch_sfx.py 下载归类；自己往类别目录里丢文件也认，
扫描只看目录不看文件名。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from ..logging_setup import get_logger

logger = get_logger("highlight.sfx")

AUDIO_SUFFIXES = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
# 冻帧只有一两帧时音效会被截断，截断处补个淡出免得爆音
TAIL_FADE_SECONDS = 0.06
# 起点也补一小段淡入，避免样本从 0 直接跳到峰值
HEAD_FADE_SECONDS = 0.004


@dataclass
class SfxPlan:
    """一次渲染要混的音效。path 为 None 表示不混。"""

    path: Path | None
    category: str = ""
    emotion: str = ""
    gain_db: float = -6.0
    offset_seconds: float = 0.0
    reason: str = ""


def library(root: Path) -> dict[str, list[Path]]:
    """扫音效库：类别目录名 -> 文件列表（按文件名排序，保证挑选可复现）。"""
    found: dict[str, list[Path]] = {}
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        files = sorted(p for p in entry.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if files:
            found[entry.name] = files
    return found


def emotion_at(expression_track: Any, moment: float) -> str:
    """冻帧点落在表情轨的哪个 span 里。没有覆盖到就返回空串（不外推、不取最近）。"""
    if not isinstance(expression_track, list):
        return ""
    for span in expression_track:
        if not isinstance(span, dict):
            continue
        try:
            start, end = float(span["start"]), float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= moment <= end:
            return str(span.get("emotion_en") or "")
    return ""


def pick(files: list[Path], key: str) -> Path:
    """同类别里按 key 稳定轮换：同一个片段重复渲染音效一致，换片段会换音效。"""
    digest = hashlib.sha1(key.encode("utf-8")).digest()  # noqa: S324 - 只做取模，不做安全用途
    return files[int.from_bytes(digest[:4], "big") % len(files)]


def plan(sfx_config: dict[str, Any], root: Path, timeline: dict[str, Any] | None,
         freeze_time: float, key: str, category: str = "") -> SfxPlan:
    """决定这次渲染混哪条音效。

    category 非空 = 界面上手动指定了类别，跳过表情判断；"none" 表示不加音效。
    """
    gain_db = float(sfx_config.get("gain_db", -6.0))
    offset = float(sfx_config.get("offset_seconds", 0.0))
    if not sfx_config.get("enabled", True) or category == "none":
        return SfxPlan(path=None, reason="音效已关闭")

    packs = library(root)
    if not packs:
        return SfxPlan(path=None, reason=f"音效库为空（{root}），跑 tools/fetch_sfx.py 下载")

    emotion = ""
    if category:
        chosen = category
        reason = f"界面指定类别 {category}"
    else:
        mapping = sfx_config.get("emotion_map") or {}
        fallback = str(sfx_config.get("fallback_category") or "punch")
        emotion = emotion_at((timeline or {}).get("expression_track"), freeze_time)
        chosen = str(mapping.get(emotion) or fallback)
        reason = (f"冻帧点表情 {emotion} -> 类别 {chosen}" if emotion
                  else f"冻帧点没有表情数据，用兜底类别 {chosen}")

    files = packs.get(chosen)
    if not files:
        return SfxPlan(path=None, category=chosen, emotion=emotion,
                       reason=f"类别目录 {chosen} 不存在或没文件（有：{'/'.join(packs)}）")
    return SfxPlan(path=pick(files, key), category=chosen, emotion=emotion,
                   gain_db=gain_db, offset_seconds=offset, reason=reason)


def load(path: Path, rate: int) -> np.ndarray:
    """解码音效文件为输出音轨的格式，返回 (2, n) float32。ogg/wav/mp3 都走 PyAV。"""
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as src:
        stream = src.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        for frame in src.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):  # 冲掉重采样器里剩的样本
            if resampled is not None and resampled.samples:
                chunks.append(resampled.to_ndarray())
    if not chunks:
        return np.zeros((2, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)


def mix(pcm: np.ndarray, sfx: np.ndarray, at_sample: int, gain_db: float,
        rate: int) -> tuple[np.ndarray, int, bool]:
    """把音效叠加到 pcm 的 at_sample 位置。返回 (混好的 pcm, 实际用掉的样本数, 是否被截断)。

    超出 pcm 长度的部分直接丢掉（视频已经结束了），截断处补淡出。叠加后整体峰值超过 1.0
    就按峰值等比压回来，avoid 削波（宁可整体轻一点，也不要爆音）。
    """
    if sfx.shape[1] == 0 or at_sample >= pcm.shape[1]:
        return pcm, 0, False
    at_sample = max(0, at_sample)
    used = min(pcm.shape[1] - at_sample, sfx.shape[1])
    truncated = used < sfx.shape[1]
    piece = sfx[:, :used] * (10.0 ** (gain_db / 20.0))

    head = min(int(HEAD_FADE_SECONDS * rate), used)
    if head > 1:
        piece[:, :head] *= np.linspace(0.0, 1.0, head, dtype=np.float32)
    if truncated:
        tail = min(int(TAIL_FADE_SECONDS * rate), used)
        if tail > 1:
            piece[:, used - tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)

    pcm = pcm.copy()
    pcm[:, at_sample:at_sample + used] += piece
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak > 1.0:
        pcm /= peak
    return pcm, used, truncated
