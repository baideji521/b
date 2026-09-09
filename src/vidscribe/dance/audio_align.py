"""音乐对齐主流程：波形互相关 + chroma 互相关 + 多窗口验证 → `DanceAlignment`。

参考 `sanjeed5/audio-video-sync` 的 `sync.py` 里 `find_offset()` 的思路，但**不是复制**：
那边是"取一段、算一次互相关、返回峰值"，这里按技术指导第五节第 5 条升级成

    source ─┬─ waveform correlation ─┐
            └─ chroma correlation ───┴─→ 两个 offset / 两个 confidence
                                        ↓ 两法互证 agreement
                                        ↓ 多窗口验证（开头/中间/结尾 + 等距补齐）
                                        ↓ 合成 final confidence
                                        ↓ decide_status
                                        → DanceAlignment

全系统唯一的时间口径：**`source_time = target_time - offset`**。
实现上就是 `cross_correlate(目标歌, 源音轨)` 的峰值 lag 除以采样率（见 `dsp.cross_correlate`
的说明），所以这一层不需要再做任何符号翻转 —— 符号搞反是这类系统最常见的 bug。
"""

from __future__ import annotations

import numpy as np

from ..logging_setup import get_logger
from . import ALIGNMENT_ALGORITHM_VERSION
from . import alignment_validation as validate
from . import dsp
from .audio_fingerprint import (
    GUARD_SECONDS,
    AudioReadError,
    extract_analysis_audio,
    media_audio_duration,
    peak_of,
    peak_ratio_confidence,
)
from .types import DanceAlignment, WindowResult

logger = get_logger("dance.align")

#: 单窗口至少要这么多采样才值得算（0.5 秒）。更短的窗口互相关全是噪声
MIN_SAMPLES = dsp.DEFAULT_SR // 2


def correlate_waveform(target: np.ndarray, source: np.ndarray,
                       sample_rate: int = dsp.DEFAULT_SR,
                       guard_seconds: float = GUARD_SECONDS,
                       ) -> tuple[float | None, float]:
    """波形互相关，返回 `(offset 秒, 置信度)`。算不出来返回 `(None, 0.0)`。

    参数顺序是 `(target, source)` 而不是反过来 —— 这样峰值 lag 直接就是 offset，
    满足 `source_time = target_time - offset`，不需要在这一层翻符号。

    去均值 + 全长 FFT 互相关 + 峰值检测；置信度走 `peak_ratio_confidence`
    （主峰 / 护栏外最强次峰），所以"副歌重复导致多个同样高的峰"会被如实反映成低分。
    """
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if target.size < MIN_SAMPLES or source.size < MIN_SAMPLES:
        return None, 0.0
    corr, zero = dsp.cross_correlate(target, source)
    index = peak_of(np.abs(corr))
    if index < 0:
        return None, 0.0
    guard = max(1, int(round(float(guard_seconds) * sample_rate)))
    confidence = peak_ratio_confidence(corr, index, guard)
    return round((index - zero) / float(sample_rate), 6), confidence


def correlate_chroma(target: np.ndarray, source: np.ndarray,
                     sample_rate: int = dsp.DEFAULT_SR,
                     hop: int = dsp.HOP,
                     guard_seconds: float = GUARD_SECONDS,
                     ) -> tuple[float | None, float]:
    """Chroma 逐带互相关，返回 `(offset 秒, 置信度)`。

    波形互相关对"同一首歌的两次不同录制/转码"很脆弱（相位、EQ、压缩都会削峰），
    而 chroma 只看和声走向，对音量、音色、轻微失真都不敏感 —— 两路互为独立证人。

    时间分辨率是一个 hop（512/22050 ≈ 23ms），所以它的 offset 天生比波形粗，
    上层用 `METHOD_TOLERANCE`（0.12s）而不是 `OFFSET_TOLERANCE` 来比这两路。
    """
    probe = chroma_correlation(target, source, sample_rate, hop)
    if probe is None:
        return None, 0.0
    corr, zero, frame_rate = probe
    index = peak_of(np.abs(corr))
    if index < 0:
        return None, 0.0
    guard = max(1, int(round(float(guard_seconds) * frame_rate)))
    return round((index - zero) / frame_rate, 6), peak_ratio_confidence(corr, index, guard)


def chroma_correlation(target: np.ndarray, source: np.ndarray,
                       sample_rate: int = dsp.DEFAULT_SR, hop: int = dsp.HOP,
                       ) -> tuple[np.ndarray, int, float] | None:
    """算出 chroma 互相关曲线本体，返回 `(corr, 零延迟下标, 帧率)`；算不了返回 None。

    单独拿出来是因为上层要问的不只是"峰在哪"，还有"波形给的那个 offset 处
    chroma 支持不支持" —— 后者需要整条曲线，见 `chroma_support`。
    """
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if target.size < dsp.N_FFT * 2 or source.size < dsp.N_FFT * 2:
        return None
    target_chroma = dsp.chromagram(target, sample_rate, dsp.N_FFT, hop)
    source_chroma = dsp.chromagram(source, sample_rate, dsp.N_FFT, hop)
    if target_chroma.shape[0] < 4 or source_chroma.shape[0] < 4:
        return None
    corr, zero = dsp.correlate_matrix(target_chroma, source_chroma)
    if corr.size == 0:
        return None
    return corr, zero, float(sample_rate) / float(hop)


def chroma_support(corr: np.ndarray, zero: int, frame_rate: float, offset: float,
                   radius_seconds: float = 0.25) -> float:
    """波形给的 offset 处，chroma 有多支持它 —— 0~1。

    取该 offset 附近 ±radius 的 chroma 相关峰值，除以全局最大值。

    为什么不直接比"两路 offset 是否相等"：和声本身常常是周期性的（4 小节一循环），
    这时 chroma 相关在 offset、offset±循环周期 处会出现**同样高**的峰，全局 argmax
    落在哪个纯属偶然。那种情况下"两路 offset 不等"不等于矛盾，只等于 chroma
    分辨不出来 —— 而"chroma 在波形那个位置上根本没有能量"才是真矛盾。
    所以这里问的是后者，正好也是技术指导要的"两路一致则加分、不一致则降分"。
    """
    corr = np.abs(np.asarray(corr, dtype=np.float64).reshape(-1))
    if corr.size == 0:
        return 0.0
    peak = float(corr.max())
    if peak <= 1e-12:
        return 0.0
    center = int(round(float(offset) * frame_rate)) + int(zero)
    span = max(1, int(round(float(radius_seconds) * frame_rate)))
    lo, hi = max(0, center - span), min(corr.size, center + span + 1)
    if lo >= hi:
        return 0.0
    return round(float(min(1.0, float(corr[lo:hi].max()) / peak)), 4)



# ================================================================== 主流程
def align_arrays(target: np.ndarray, source: np.ndarray,
                 sample_rate: int = dsp.DEFAULT_SR,
                 window_seconds: float = validate.WINDOW_SECONDS,
                 window_count: int = validate.WINDOW_COUNT,
                 algorithm_version: str = ALIGNMENT_ALGORITHM_VERSION,
                 ) -> DanceAlignment:
    """在内存数组上做对齐。测试和 GUI 预览都走这条，不需要磁盘文件。

    每个源窗口都拿去和**整条目标歌**比：offset 是全局量，只要源里任意一段能在
    目标歌里定位，就得到一个 offset 候选。多个窗口给出的候选一致，才算真的对上了。

    最终 offset 取多窗口中位数（`robust_offset`），不取"置信度最高那个窗口" ——
    中位数抗离群，一个撞上副歌重复的窗口拽不动结果。
    """
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    target_duration = round(target.size / float(sample_rate), 3)
    source_duration = round(source.size / float(sample_rate), 3)
    blank = DanceAlignment(
        offset=0.0, confidence=0.0, method="hybrid", status="rejected",
        algorithm_version=algorithm_version, sample_rate=int(sample_rate),
        source_duration=source_duration, target_duration=target_duration,
    )
    if target.size < MIN_SAMPLES or source.size < MIN_SAMPLES:
        return _with_notes(blank, "音频太短，无法对齐")


    plan = validate.window_plan(source_duration, window_seconds, window_count)
    windows: list[WindowResult] = []
    offsets: list[float] = []
    confidences: list[float] = []
    for index, (start, length) in enumerate(plan, 1):
        begin = int(round(start * sample_rate))
        end = min(source.size, begin + int(round(length * sample_rate)))
        chunk = source[begin:end]
        if chunk.size < MIN_SAMPLES:
            continue
        offset, confidence = correlate_waveform(target, chunk, sample_rate)
        if offset is None:
            continue
        # 窗口里的 offset 是"这一窗相对目标歌"的，要减掉窗口自身在源里的起点
        # 才能还原成整条源视频的 offset
        global_offset = round(offset - start, 6)
        windows.append(WindowResult(index=index, window_start=round(start, 3),
                                    window_seconds=round(length, 3),
                                    offset=global_offset, confidence=confidence,
                                    method="waveform"))
        offsets.append(global_offset)
        confidences.append(confidence)

    if not offsets:
        return _with_notes(blank, "所有验证窗口都算不出峰值，无法对齐")

    agreement, deviation = validate.agreement_of(offsets)
    offset = validate.robust_offset(offsets, confidences)
    waveform_confidence = round(float(sum(confidences) / len(confidences)), 4)

    # chroma 只在"最能代表整条"的那一段上算一次：逐窗都算 chroma 会把耗时翻好几倍，
    # 而 chroma 的作用是给波形结论找个独立证人，一个证人就够
    chroma_offset, chroma_confidence, support = _chroma_probe(
        target, source, sample_rate, source_duration, window_seconds, offset)
    # 两法互证取"offset 直接相等"和"chroma 在波形 offset 处有支持"里更宽容的那个：
    # 前者在和声不重复时成立，后者在和声周期性重复时才是唯一有意义的判据
    methods_agree = max(validate.method_agreement(offset, chroma_offset), support)
    confidence = validate.combine_confidence(waveform_confidence, chroma_confidence,
                                             methods_agree, agreement, len(offsets))
    status, reasons = validate.decide_status(confidence, deviation, offset,
                                            source_duration, methods_agree,
                                            target_duration=target_duration)

    if chroma_offset is not None and abs(chroma_offset - offset) > validate.METHOD_TOLERANCE:
        reasons.append(f"chroma 全局峰在 {chroma_offset:.3f}s（和声重复所致），"
                       f"但波形 offset {offset:.3f}s 处 chroma 支持度 {support:.3f}")

    return DanceAlignment(
        offset=offset,
        confidence=confidence,
        method="hybrid" if chroma_offset is not None else "waveform",
        waveform_offset=offset,
        waveform_confidence=waveform_confidence,
        chroma_offset=chroma_offset,
        chroma_confidence=chroma_confidence if chroma_offset is not None else None,
        window_count=len(offsets),
        max_deviation=deviation,
        agreement=agreement,
        status=status,
        algorithm_version=algorithm_version,
        source_duration=source_duration,
        target_duration=target_duration,
        sample_rate=int(sample_rate),
        windows=tuple(windows),
        notes=tuple(reasons),
    )


def _with_notes(alignment: DanceAlignment, note: str) -> DanceAlignment:
    from dataclasses import replace  # noqa: PLC0415

    return replace(alignment, notes=alignment.notes + (note,))


def _chroma_probe(target: np.ndarray, source: np.ndarray, sample_rate: int,
                  source_duration: float, window_seconds: float, expected_offset: float,
                  ) -> tuple[float | None, float, float]:
    """取源视频中段的一个窗口做 chroma 对齐。

    返回 `(还原到整条的 chroma offset, 峰值置信度, 对波形 offset 的支持度)`。
    取中段而不是开头：舞蹈视频开头常有静止起手式、口播、片头音乐，中段一定在跳。
    """
    window = max(validate.MIN_WINDOW_SECONDS, float(window_seconds))
    start = max(0.0, (source_duration - window) / 2.0) if source_duration > window else 0.0
    begin = int(round(start * sample_rate))
    end = min(source.size, begin + int(round(window * sample_rate)))
    probe = chroma_correlation(target, source[begin:end], sample_rate, dsp.HOP)
    if probe is None:
        return None, 0.0, 0.5          # 算不出来 = 没有证人，不奖不罚
    corr, zero, frame_rate = probe
    index = peak_of(np.abs(corr))
    if index < 0:
        return None, 0.0, 0.5
    guard = max(1, int(round(GUARD_SECONDS * frame_rate)))
    confidence = peak_ratio_confidence(corr, index, guard)
    free_offset = round((index - zero) / frame_rate - start, 6)
    # 支持度问的是"波形那个 offset 处 chroma 有没有能量"，所以要把窗口起点加回去
    support = chroma_support(corr, zero, frame_rate, float(expected_offset) + start)
    return free_offset, confidence, support



def find_alignment(source_path: str, target_path: str, *,
                   sample_rate: int = dsp.DEFAULT_SR,
                   window_seconds: float = validate.WINDOW_SECONDS,
                   window_count: int = validate.WINDOW_COUNT,
                   target_pcm: np.ndarray | None = None,
                   algorithm_version: str = ALIGNMENT_ALGORITHM_VERSION,
                   ) -> DanceAlignment:
    """从磁盘读两个媒体做对齐。这是业务层唯一该调的入口。

    `target_pcm` 给了就直接用，不重复解码目标歌 —— 批量对齐几十个源视频时，
    目标歌只解一次（技术指导第二十四节：Target song fingerprint 只计算一次）。

    读不出音轨抛 `AudioReadError`，**不返回一个假的 offset=0**：
    静默返回 0 会让下游切出一整批错位素材，而且事后无从追查。
    """
    if target_pcm is None:
        target_pcm = extract_analysis_audio(target_path, sample_rate=sample_rate)
    source_pcm = extract_analysis_audio(source_path, sample_rate=sample_rate)
    result = align_arrays(target_pcm, source_pcm, sample_rate=sample_rate,
                          window_seconds=window_seconds, window_count=window_count,
                          algorithm_version=algorithm_version)
    logger.info("对齐 %s → offset %.3fs 置信度 %.3f 结论 %s（%d 窗口，最大偏差 %.3fs）",
                source_path, result.offset, result.confidence, result.status,
                result.window_count, result.max_deviation)
    return result


def load_target(target_path: str, sample_rate: int = dsp.DEFAULT_SR) -> np.ndarray:
    """解目标歌，供批量对齐复用。单独开一个函数是为了让"只解一次"这件事显式可见。"""
    return extract_analysis_audio(target_path, sample_rate=sample_rate)


__all__ = [
    "MIN_SAMPLES", "AudioReadError",
    "correlate_waveform", "correlate_chroma", "chroma_correlation", "chroma_support",
    "align_arrays", "find_alignment", "load_target", "media_audio_duration",
]





