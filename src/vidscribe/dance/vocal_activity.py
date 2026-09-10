"""人声活动分析：哪儿在唱、哪儿是空白、哪些空白值得当成切段的参考点。

**先把话说清楚，免得后面有人当它是源分离**：这里没有模型、没有人声分离，
只有频谱启发式 —— 人声主要能量落在 300~3400 Hz，而且是谐波结构（比 hi-hat 干净、
比底鼓高）。所以判断规则是「人声带能量占比高 + 该带谱平坦度低」。
它对「纯乐器段 / 唱歌段」这种大块差别判得住，对「和声垫底」「说唱夹白」会犯错。

正因为会犯错，这一层的定位是 **参考层**：
    算出停顿 → 在时间轴上标一个 ⏸ → 告诉用户这里空了 0.67 秒
    → 用户自己决定要不要在这儿切段（`segment_template`）
一律不许拿它自动切段、自动改素材 —— 那是把一个会犯错的启发式变成既成事实。

依赖只有 numpy + `dance.dsp`，与 `music_structure` 平级，互不调用。
"""

from __future__ import annotations

import numpy as np

from ..logging_setup import get_logger
from . import dsp
from .types import VocalActivity, VocalPause, VocalSpan

logger = get_logger("dance.vocal")

#: 人声活动分析版本。改了算法就改它，缓存据此失效
VOCAL_VERSION = "vocal-v1"

#: 人声主频带（Hz）。取 300~3400 是电话带宽那一段：基频 + 前几个谐波都在里面，
#: 底鼓（<250）和 hi-hat（>6k）都被挡在外面
VOCAL_BAND = (300.0, 3400.0)
#: 参照带：低频（鼓/贝斯）和高频（镲/空气声），用来算"人声带占了多少"
LOW_BAND = (20.0, 250.0)
HIGH_BAND = (5000.0, 14000.0)

#: 判定门槛（0~1 强度曲线上）。进入人声要超过 ENTER，退出要跌破 EXIT ——
#: 两个门槛（滞回）是为了别在门槛附近抖出一堆碎片
ENTER = 0.55
EXIT = 0.40
#: 强度曲线的平滑窗口（帧）。约 0.25 秒，够抹掉单帧毛刺又不糊掉换句
SMOOTH_FRAMES = 11
#: 比这还短的人声段当噪声丢掉（秒）
MIN_VOCAL_SECONDS = 0.35
#: 比这还短的空白不算停顿 —— 唱句之间的换气不是"可以换人的地方"（秒）
MIN_PAUSE_SECONDS = 0.30
#: 停顿推荐度里"够长了"的参照（秒）：到这个长度推荐度的长度分就满了
GOOD_PAUSE_SECONDS = 1.20


def _flatness(mag: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """频带内的谱平坦度（几何均值 / 算术均值），逐帧。

    噪声型声音（镲、掌声、气声）平坦度接近 1，谐波型（人声、弦乐）明显低。
    用 log 域算几何均值，免得几百个分量连乘直接下溢成 0。
    """
    band = mag[:, picks].astype(np.float64) + 1e-10
    geometric = np.exp(np.log(band).mean(axis=1))
    arithmetic = band.mean(axis=1)
    return (geometric / np.maximum(arithmetic, 1e-10)).astype(np.float32)


def vocal_strength(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR,
                   n_fft: int = dsp.N_FFT, hop: int = dsp.HOP) -> tuple[np.ndarray, np.ndarray]:
    """逐帧人声强度（0~1）+ 帧时间。

    强度 = 人声带能量占比 × 谐波性 × 有声音
    三项都是 0~1，相乘的意思是「三个条件都得成立」：
    能量确实集中在人声带、这段声音是谐波型的、而且这一帧不是静音。
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    mag = dsp.stft_magnitude(pcm, n_fft=n_fft, hop=hop)
    times = dsp.frame_times(mag.shape[0], sample_rate, hop)
    if mag.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), times

    vocal = dsp.band_energy(mag, sample_rate, n_fft, *VOCAL_BAND)
    low = dsp.band_energy(mag, sample_rate, n_fft, *LOW_BAND)
    high = dsp.band_energy(mag, sample_rate, n_fft, *HIGH_BAND)
    share = vocal / np.maximum(vocal + low + high, 1e-10)

    freqs = dsp.fft_frequencies(sample_rate, n_fft)
    picks = (freqs >= VOCAL_BAND[0]) & (freqs <= VOCAL_BAND[1])
    harmonic = 1.0 - _flatness(mag, picks) if picks.any() else np.ones_like(share)

    loud = dsp.robust_normalize01(np.sqrt(np.maximum(vocal, 0.0)))
    raw = dsp.normalize01(share) * dsp.normalize01(harmonic) * loud
    return dsp.smooth(raw, SMOOTH_FRAMES), times


def _spans(strength: np.ndarray, times: np.ndarray, duration: float) -> list[VocalSpan]:
    """滞回门槛把强度曲线切成 vocal / pause 交替的区间。"""
    if strength.size == 0:
        return []
    speaking = False
    marks: list[tuple[float, bool]] = [(0.0, False)]
    for value, moment in zip(strength, times):
        if not speaking and value >= ENTER:
            speaking = True
            marks.append((float(moment), True))
        elif speaking and value < EXIT:
            speaking = False
            marks.append((float(moment), False))

    spans: list[VocalSpan] = []
    for i, (start, is_vocal) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else float(duration)
        if end - start <= 0:
            continue
        window = (times >= start) & (times < end)
        mean = float(strength[window].mean()) if window.any() else 0.0
        spans.append(VocalSpan(index=len(spans), start=round(start, 3), end=round(end, 3),
                               kind="vocal" if is_vocal else "pause",
                               strength=round(mean, 4)))
    return _tidy(spans)


def _tidy(spans: list[VocalSpan]) -> list[VocalSpan]:
    """丢掉太短的人声碎片（并进前后的空白），再把相邻同类合并。"""
    kept = [s for s in spans
            if not (s.kind == "vocal" and s.duration < MIN_VOCAL_SECONDS)]
    merged: list[VocalSpan] = []
    for span in kept:
        if merged and merged[-1].kind == span.kind:
            last = merged[-1]
            weight = last.duration + span.duration
            strength = ((last.strength * last.duration + span.strength * span.duration)
                        / weight) if weight > 0 else last.strength
            merged[-1] = VocalSpan(index=last.index, start=last.start, end=span.end,
                                   kind=last.kind, strength=round(strength, 4))
        else:
            merged.append(VocalSpan(index=len(merged), start=span.start, end=span.end,
                                    kind=span.kind, strength=span.strength))
    # 中间被丢掉的碎片会留下时间缝，这里把每段的 start 拉到上一段的 end
    stitched: list[VocalSpan] = []
    for span in merged:
        start = stitched[-1].end if stitched else span.start
        stitched.append(VocalSpan(index=len(stitched), start=start, end=span.end,
                                  kind=span.kind, strength=span.strength))
    return stitched


def _nearest_beat(moment: float, beats: tuple[float, ...]) -> float:
    if not beats:
        return -1.0
    return round(min(beats, key=lambda b: abs(b - moment)), 3)


def pause_score(pause_duration: float, strength: float, beat_gap: float) -> float:
    """停顿的推荐度（0~1）：够长 + 够干净 + 贴着拍点。

    三项加权（长度 0.5 / 干净 0.3 / 贴拍 0.2）。贴拍这一项只在有拍网格时才算，
    没有网格就把这 0.2 按前两项的表现补回去，而不是白扣分。
    """
    length = min(1.0, max(0.0, pause_duration) / GOOD_PAUSE_SECONDS)
    clean = 1.0 - min(1.0, max(0.0, strength) / EXIT) if EXIT > 0 else 1.0
    core = 0.5 * length + 0.3 * clean
    if beat_gap < 0:
        return round(core / 0.8, 4)
    on_beat = max(0.0, 1.0 - min(1.0, beat_gap / 0.25))
    return round(core + 0.2 * on_beat, 4)


#: 可取区间：以停顿为中心往两边各留这么久（秒）。切点落在这一带里都算"附近"
ZONE_RADIUS = 0.60
#: 推荐等级的分档（按停顿推荐度）
ZONE_LEVELS = ((0.75, "⭐⭐⭐⭐⭐"), (0.55, "高"), (0.35, "中"), (0.0, "低"))


def zone_level(score: float) -> str:
    """推荐度 → 人看得懂的等级。只是措辞，不参与任何判定。"""
    for floor, text in ZONE_LEVELS:
        if float(score) >= floor:
            return text
    return "低"


def cut_zones(activity: VocalActivity, *, radius: float = ZONE_RADIUS,
              min_score: float = 0.0) -> tuple[tuple[float, float, float, str], ...]:
    """「可取区间」：每个停顿周围一段"这里附近适合换人"的区域。

    返回 `((start, end, score, level), …)`。它比"这里有个停顿"更好用：
    切点不必正正落在停顿里，落在停顿附近同样合适。
    **仍然只是参考** —— 不改 Segment，也不自动切。
    """
    out: list[tuple[float, float, float, str]] = []
    span = max(0.0, float(radius))
    for pause in activity.pauses:
        if pause.score < float(min_score):
            continue
        start = max(0.0, pause.start - span)
        end = min(float(activity.duration or pause.end + span), pause.end + span)
        if end <= start:
            continue
        if out and start <= out[-1][1]:            # 挨着的两带合成一带，取更高那个分
            last = out[-1]
            score = max(last[2], pause.score)
            out[-1] = (last[0], max(last[1], end), score, zone_level(score))
            continue
        out.append((round(start, 3), round(end, 3), pause.score, zone_level(pause.score)))
    return tuple(out)


def analyze_vocal_activity(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR,
                           beats: tuple[float, ...] = ()) -> VocalActivity:
    """一首歌的人声活动 + 停顿导航点。`beats` 给了就顺手算"这个停顿贴不贴拍"。"""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    duration = round(pcm.size / float(sample_rate or dsp.DEFAULT_SR), 3)
    strength, times = vocal_strength(pcm, sample_rate)
    spans = _spans(strength, times, duration)

    pauses: list[VocalPause] = []
    for span in spans:
        if span.kind != "pause" or span.duration < MIN_PAUSE_SECONDS:
            continue
        if span.start <= 0.0:            # 开头那段前奏不是"停顿"，它就是还没开始唱
            continue
        beat = _nearest_beat(span.start, beats)
        gap = abs(beat - span.start) if beat >= 0 else -1.0
        pauses.append(VocalPause(index=len(pauses), start=span.start, end=span.end,
                                 score=pause_score(span.duration, span.strength, gap),
                                 nearest_beat=beat))

    logger.info("人声活动：%d 段（其中 %d 段有人声），可用停顿 %d 处",
                len(spans), sum(1 for s in spans if s.kind == "vocal"), len(pauses))
    return VocalActivity(duration=duration, sample_rate=int(sample_rate),
                         hop_seconds=round(dsp.HOP / float(sample_rate or dsp.DEFAULT_SR), 6),
                         frame_times=tuple(round(float(t), 4) for t in times),
                         strength=tuple(round(float(v), 4) for v in strength),
                         threshold=ENTER, spans=tuple(spans), pauses=tuple(pauses),
                         version=VOCAL_VERSION)


__all__ = ["VOCAL_VERSION", "VOCAL_BAND", "ENTER", "EXIT", "ZONE_RADIUS",
           "vocal_strength", "pause_score", "zone_level", "cut_zones",
           "analyze_vocal_activity"]
