"""目标音乐结构分析：节拍网格、全局特征、四条节奏带、段落切分与分类。

参考 BeatSync-Engine 的 stage1~stage4 的**思路**（onset → 平滑 → 归一 → beat_track、
RMS/centroid/flux/novelty、chroma+MFCC 聚类分段），全部在这里用 numpy 重新实现，
一行代码都不复制，也不引入 librosa（理由见 `dance/__init__.py`）。

一条重要的边界：段落类型（intro/verse/hook/chorus/…）**只是辅助特征**。
技术指导第六节第 5 条明确禁止把 section 类型写死成最终素材选择规则 ——
这里算出来的 `type` 只用于界面展示和评分里的一个小权重项，
选素材的决定权在 `material_score` + `combination_search`，不在这儿。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..logging_setup import get_logger
from . import dsp
from .types import BeatGrid, MusicSection, RhythmBands, TargetMusicFeatures, TargetPosition

logger = get_logger("dance.music")

#: 音乐结构分析版本。改了算法就 +1，落库时一起存，缓存据此失效
ANALYSIS_VERSION = "music-v1"

#: 四条节奏带的频率范围（Hz）。kick 和 bass 刻意重叠：
#: kick 是"底鼓那一下"（更低更窄），bass 是"贝斯线"（更宽），两者在编曲里各有作用
BANDS: dict[str, tuple[float, float]] = {
    "kick": (20.0, 120.0),
    "bass": (60.0, 250.0),
    "clap": (1500.0, 4000.0),
    "hihat": (6000.0, 14000.0),
}
#: 每条带的击点判定：相对该带最大值的门槛 + 最小间隔（秒）
BAND_THRESHOLD = 0.30
BAND_MIN_GAP = 0.08

#: 段落切分：特征降采样到这个帧率再算新颖度（2Hz 足够，越细越碎）
SECTION_RATE = 2.0
#: 新颖度的前后对比窗口（秒）。一个乐句大约 2 秒，取 4 秒能跨过乐句看到段落变化
NOVELTY_WINDOW = 4.0
#: 一个段落至少这么长（秒）。比它短的并进相邻段
MIN_SECTION_SECONDS = 6.0
#: 段落聚类的相似度门槛：余弦相似度高于此算同一类（副歌重复会落在同一类）
SECTION_SIMILARITY = 0.90


# ============================================================== 1. 节拍网格
def _synthetic_grid(duration: float, bpm: float = dsp.PRIOR_BPM) -> BeatGrid:
    """第三级兜底：一条等间距的编造网格，`confidence=0` 明说它是编的。

    宁可给一条假网格也不给空的：界面上"这首歌有几拍"、评分里"这个位置在第几拍"
    都得有个数，返回空会让每个调用方各自写一遍 if 判断，而且很容易忘。
    `source="grid"` + `confidence=0` 已经把真相说清楚了。
    """
    period = 60.0 / float(bpm)
    count = max(2, int(max(0.0, float(duration)) / period))
    beats = tuple(round(i * period, 4) for i in range(count))
    return BeatGrid(bpm=float(bpm), beats=beats,
                    downbeats=tuple(beats[::4]) if len(beats) >= 4 else beats[:1],
                    duration=round(float(duration), 3), beats_per_bar=4,
                    source="grid", confidence=0.0)


def detect_target_beat_grid(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR) -> BeatGrid:

    """节拍网格：onset 包络 → tempo → 动态规划跟踪 → 降级 onset 峰值。

    三级降级，任何一级失败都往下退，绝不返回空网格骗人：
      1. `beat_track`（DP）：正常路径，能跟上前奏空拍和轻微渐快
      2. `onset_fallback`：DP 出来的拍太少（不足理论拍数的一半）时改用 onset 峰值，
         再按中位间隔反算 BPM —— 拍点不等间距，但至少每个都落在真实响声上
      3. `grid`：连 onset 峰都挑不出来（纯环境音/极安静）时给一条等间距网格并
         把 confidence 记 0，让上层知道"这份网格是编出来的"

    downbeat 取每 `beats_per_bar` 拍一个；这里不做真正的小节检测（那需要和声分析），
    对卡点舞够用 —— 用户真正在意的是"每一拍在哪"。
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    duration = round(pcm.size / float(sample_rate), 3)
    if pcm.size < dsp.N_FFT * 2:
        return _synthetic_grid(duration)

    env, env_rate = dsp.onset_envelope(pcm, sample_rate)
    bpm, tempo_confidence = dsp.estimate_tempo(env, env_rate)
    if bpm <= 0:
        return _synthetic_grid(duration)


    frames = dsp.beat_track(env, env_rate, bpm)
    beats = [round(float(f) * dsp.HOP / sample_rate, 4) for f in frames]
    expected = int(duration / (60.0 / bpm)) if bpm > 0 else 0
    source = "beat_track"
    if len(beats) < max(2, expected // 2):
        peaks = dsp.pick_peaks(env, min_distance=max(1, int(env_rate * 60.0 / bpm * 0.5)))
        if peaks.size >= 2:
            beats = [round(float(p) * dsp.HOP / sample_rate, 4) for p in peaks]
            gaps = np.diff(np.asarray(beats))
            median = float(np.median(gaps)) if gaps.size else 0.0
            if median > 0:
                bpm = round(60.0 / median, 3)
            source = "onset_fallback"
        else:
            period = 60.0 / bpm
            count = max(2, int(duration / period))
            beats = [round(i * period, 4) for i in range(count)]
            source = "grid"
            tempo_confidence = 0.0

    beats = [b for b in beats if 0.0 <= b <= duration]
    downbeats = tuple(beats[::4]) if len(beats) >= 4 else tuple(beats[:1])
    return BeatGrid(bpm=bpm, beats=tuple(beats), downbeats=downbeats, duration=duration,
                    beats_per_bar=4, source=source, confidence=tempo_confidence)


# ============================================================== 2. 全局特征
def analyze_target_music_features(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR,
                                  ) -> TargetMusicFeatures:
    """逐帧特征 + 三个全局分数。所有逐帧曲线都归一到 0~1，方便直接当权重用。

    - `rms` / `energy_wave`：响度。energy_wave 是 rms 平滑归一后的版本，
      给"这个音乐位置有多带劲"用
    - `centroid` / `brightness`：频谱重心与它的归一版本，亮 = 高频多 = 通常更"炸"
    - `flux` / `onset`：谱通量与 onset 包络。onset 直接当"冲击力"用
    - `novelty`：变化率（onset 的一阶差分再整流），段落边界就在它的峰上
    - `rhythm_score`：节奏性强不强 = onset 包络自相关的峰值突出度
    - `impact_score`：全曲最强冲击的平均水平 = onset 前 10% 分位的均值
    - `arc`：把整曲能量压成 16 段的走向曲线，给界面画缩略图和"编排弧线"用
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    duration = round(pcm.size / float(sample_rate), 3)
    hop_seconds = dsp.HOP / float(sample_rate)
    if pcm.size < dsp.N_FFT * 2:
        return TargetMusicFeatures(duration=duration, sample_rate=int(sample_rate),
                                   hop_seconds=hop_seconds)

    mag = dsp.stft_magnitude(pcm, dsp.N_FFT, dsp.HOP)
    frames = mag.shape[0]
    times = dsp.frame_times(frames, sample_rate, dsp.HOP)
    rms = dsp.frame_rms(pcm, dsp.N_FFT, dsp.HOP)[:frames]
    centroid = dsp.spectral_centroid(mag, sample_rate, dsp.N_FFT)
    mel_db = dsp.power_to_db((mag ** 2) @ dsp.mel_filterbank(sample_rate, dsp.N_FFT).T)
    onset = dsp.onset_strength(mel_db)
    flux = np.concatenate([[0.0], np.maximum(np.diff(rms), 0.0)]).astype(np.float32)
    novelty = np.concatenate([[0.0], np.abs(np.diff(onset))]).astype(np.float32)

    energy_wave = dsp.normalize01(dsp.smooth(rms, 9))
    brightness = dsp.normalize01(dsp.smooth(centroid, 9))
    env_rate = float(sample_rate) / float(dsp.HOP)
    _, rhythm_score = dsp.estimate_tempo(onset, env_rate)
    strong = np.sort(onset)[int(onset.size * 0.9):] if onset.size else np.zeros(1)
    impact_score = round(float(strong.mean()) if strong.size else 0.0, 4)
    arc = _arc(energy_wave, 16)

    return TargetMusicFeatures(
        duration=duration, sample_rate=int(sample_rate), hop_seconds=hop_seconds,
        frame_times=tuple(float(t) for t in times),
        rms=tuple(float(v) for v in dsp.normalize01(rms)),
        centroid=tuple(float(v) for v in centroid),
        flux=tuple(float(v) for v in dsp.normalize01(flux)),
        onset=tuple(float(v) for v in onset),
        novelty=tuple(float(v) for v in dsp.normalize01(novelty)),
        energy_wave=tuple(float(v) for v in energy_wave),
        brightness=tuple(float(v) for v in brightness),
        rhythm_score=rhythm_score, impact_score=impact_score, arc=arc,
    )


def _arc(track: np.ndarray, buckets: int) -> tuple[float, ...]:
    """把一条曲线压成 `buckets` 段的均值，给界面画走向缩略图。"""
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return ()
    edges = np.linspace(0, track.size, buckets + 1).astype(int)
    out: list[float] = []
    for i in range(buckets):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out.append(round(float(track[lo:hi].mean()), 4))
    return tuple(out)


# ============================================================== 3. 节奏带
def analyze_rhythm_bands(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR) -> RhythmBands:
    """kick / bass / clap / hihat 四条带的逐帧能量与击点时刻。

    先保证 CPU 正确性，不做 GPU 化（技术指导第六节第 3 条）。
    每条带各自归一：底鼓的绝对能量比 hihat 高一两个数量级，不各自归一的话
    hihat 那条曲线在图上就是一条直线。
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if pcm.size < dsp.N_FFT * 2:
        return RhythmBands()
    mag = dsp.stft_magnitude(pcm, dsp.N_FFT, dsp.HOP)
    times = dsp.frame_times(mag.shape[0], sample_rate, dsp.HOP)
    frame_rate = float(sample_rate) / float(dsp.HOP)
    min_gap = max(1, int(round(BAND_MIN_GAP * frame_rate)))

    tracks: dict[str, tuple[float, ...]] = {}
    hits: dict[str, tuple[float, ...]] = {}
    for name, (low, high) in BANDS.items():
        energy = dsp.band_energy(mag, sample_rate, dsp.N_FFT, low, high)
        # 击点看的是"能量上冲"而不是"能量高"：持续的低音垫能量也高，但不是击点
        rise = np.concatenate([[0.0], np.maximum(np.diff(energy), 0.0)]).astype(np.float32)
        curve = dsp.normalize01(dsp.smooth(energy, 3))
        peaks = dsp.pick_peaks(dsp.normalize01(rise), min_gap, BAND_THRESHOLD)
        tracks[name] = tuple(float(v) for v in curve)
        hits[name] = tuple(round(float(times[p]), 4) for p in peaks if p < times.size)

    return RhythmBands(
        frame_times=tuple(float(t) for t in times),
        kick=tracks["kick"], bass=tracks["bass"], clap=tracks["clap"], hihat=tracks["hihat"],
        kick_hits=hits["kick"], bass_hits=hits["bass"],
        clap_hits=hits["clap"], hihat_hits=hits["hihat"],
    )


# ============================================================== 4. 段落切分
def _section_features(pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """分段用的特征矩阵，返回 `(特征 (帧, 维), 帧时刻)`。

    chroma（和声，12 维）+ MFCC 的 1~12 号系数（音色，跳过 c0 因为它只是总能量）。
    两者拼起来的原因：副歌和主歌常常和声相同而配器不同（音色变），
    也常常配器相同而和声不同 —— 只看一个会漏掉一半的段落边界。
    再降采样到 `SECTION_RATE`（2Hz）：段落是几十秒级的东西，43Hz 的分辨率纯属浪费。
    """
    chroma = dsp.chromagram(pcm, sample_rate, dsp.N_FFT, dsp.HOP)
    mfcc = dsp.mfcc(pcm, sample_rate, 13, dsp.N_FFT, dsp.HOP)
    frames = min(chroma.shape[0], mfcc.shape[0])
    if frames < 4:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0)
    timbre = mfcc[:frames, 1:]
    scale = np.maximum(np.abs(timbre).max(axis=0, keepdims=True), 1e-6)
    matrix = np.hstack([chroma[:frames], (timbre / scale).astype(np.float32)])

    step = max(1, int(round((float(sample_rate) / dsp.HOP) / SECTION_RATE)))
    kept = frames // step
    if kept < 4:
        return matrix, dsp.frame_times(frames, sample_rate, dsp.HOP)
    trimmed = matrix[:kept * step].reshape(kept, step, matrix.shape[1]).mean(axis=1)
    times = np.arange(kept) * step * dsp.HOP / float(sample_rate)
    return trimmed.astype(np.float32), times


def _novelty(matrix: np.ndarray, rate: float, window_seconds: float) -> np.ndarray:
    """新颖度：每一帧前后两个窗口的平均特征之间的余弦距离。

    这是 checkerboard-kernel 自相似矩阵法的轻量等价物：段落边界处"前面像前面、
    后面像后面、前后互不像"，正是这个距离最大的地方。直接算 SSM 要 O(n²) 内存，
    而我们只需要对角线附近那一条带。
    """
    if matrix.shape[0] < 4:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    span = max(2, int(round(window_seconds * rate)))
    out = np.zeros(matrix.shape[0], dtype=np.float32)
    for index in range(matrix.shape[0]):
        lo = max(0, index - span)
        hi = min(matrix.shape[0], index + span)
        if index - lo < 2 or hi - index < 2:
            continue
        before = matrix[lo:index].mean(axis=0)
        after = matrix[index:hi].mean(axis=0)
        denom = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denom <= 1e-9:
            continue
        out[index] = float(1.0 - float(before @ after) / denom)
    return dsp.normalize01(dsp.smooth(out, 3))


def _boundaries(novelty: np.ndarray, times: np.ndarray, duration: float) -> list[float]:
    """新颖度峰值 → 段落边界（秒），含 0 和 duration，且相邻至少隔 MIN_SECTION_SECONDS。"""
    if novelty.size == 0 or times.size == 0:
        return [0.0, duration]
    rate = 1.0 / max(float(times[1] - times[0]), 1e-6) if times.size > 1 else SECTION_RATE
    min_gap = max(1, int(round(MIN_SECTION_SECONDS * rate)))
    peaks = dsp.pick_peaks(novelty, min_gap, threshold_ratio=0.35)
    marks = [0.0]
    for peak in peaks:
        if peak >= times.size:
            continue
        at = round(float(times[peak]), 3)
        if at - marks[-1] >= MIN_SECTION_SECONDS and duration - at >= MIN_SECTION_SECONDS:
            marks.append(at)
    marks.append(round(duration, 3))
    return marks


def _cluster(vectors: list[np.ndarray]) -> list[int]:
    """把段落按特征相似度归类，返回每段的类号（从 0 起）。

    单趟贪心聚类：和已有某一类的代表向量余弦相似度 ≥ SECTION_SIMILARITY 就归入它，
    否则自成一类。之所以够用：段落数量个位数，而"副歌重复"这件事在特征上非常明显，
    上 k-means 只会引入"k 取多少"这个新问题。
    """
    labels: list[int] = []
    centers: list[np.ndarray] = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        best, best_score = -1, -1.0
        for index, center in enumerate(centers):
            denom = norm * float(np.linalg.norm(center))
            score = float(vector @ center) / denom if denom > 1e-9 else 0.0
            if score > best_score:
                best, best_score = index, score
        if best >= 0 and best_score >= SECTION_SIMILARITY:
            labels.append(best)
            centers[best] = (centers[best] + vector) / 2.0
        else:
            labels.append(len(centers))
            centers.append(vector.copy())
    return labels


def classify_music_section(index: int, total: int, energy: float, impact: float,
                           brightness: float, label: int, label_counts: dict[int, int],
                           energy_rank: float) -> str:
    """给一个段落定类型。返回值见 `types.SECTION_TYPES`。

    **这只是辅助特征**（技术指导第六节第 5 条）：类型不参与素材选择的硬规则，
    只用于界面展示和评分里一个很小的权重项。所以这里用简单可解释的规则，
    不上分类模型 —— 一个说不清为什么的段落标签，比没有标签更糟。

    判定顺序（命中即止）：
      1. 第一段且能量偏低      → intro
      2. 最后一段且能量偏低    → outro
      3. 最后一段且能量很高    → finale
      4. 能量排名进前 25% 且冲击力高 → drop（爆点），否则 chorus（副歌）
      5. 能量很低（<0.3）      → breakdown（掉下来喘口气）
      6. 只出现一次的类且亮度高 → bridge（过渡段）
      7. 能量中上、时长偏短    → hook（钩子）
      8. 其余                  → verse（主歌）
    """
    last = index == total - 1
    if index == 0 and energy_rank < 0.5:
        return "intro"
    if last and energy_rank < 0.5:
        return "outro"
    if last and energy_rank >= 0.75:
        return "finale"
    if energy_rank >= 0.75:
        return "drop" if impact >= 0.55 else "chorus"
    if energy < 0.30:
        return "breakdown"
    if label_counts.get(label, 0) <= 1 and brightness >= 0.6:
        return "bridge"
    if energy_rank >= 0.55:
        return "hook"
    return "verse"


def analyze_music_sections(pcm: np.ndarray, sample_rate: int = dsp.DEFAULT_SR,
                           features: TargetMusicFeatures | None = None,
                           rhythm: RhythmBands | None = None,
                           ) -> tuple[MusicSection, ...]:
    """段落切分 + 分类。chroma+MFCC → 新颖度 → 边界 → 合并 → 聚类 → 定类型。

    素材太短（放不下两个 `MIN_SECTION_SECONDS`）时老实返回一整段，不硬切 ——
    编出来的段落边界比没有段落更有害，下游会拿它当真。
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    duration = round(pcm.size / float(sample_rate), 3)
    if duration <= 0:
        return ()
    if features is None:
        features = analyze_target_music_features(pcm, sample_rate)
    if rhythm is None:
        rhythm = analyze_rhythm_bands(pcm, sample_rate)
    if duration < MIN_SECTION_SECONDS * 2:
        return (_section(0, 0.0, duration, features, rhythm, 0, {0: 1}, 0.5),)

    matrix, times = _section_features(pcm, sample_rate)
    if matrix.shape[0] < 4:
        return (_section(0, 0.0, duration, features, rhythm, 0, {0: 1}, 0.5),)
    rate = (1.0 / float(times[1] - times[0])) if times.size > 1 else SECTION_RATE
    novelty = _novelty(matrix, rate, NOVELTY_WINDOW)
    marks = _boundaries(novelty, times, duration)

    spans = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    spans = [(a, b) for a, b in spans if b - a > 0.5]
    if not spans:
        spans = [(0.0, duration)]

    vectors: list[np.ndarray] = []
    for start, end in spans:
        picks = [i for i, t in enumerate(times) if start <= t < end]
        vectors.append(matrix[picks].mean(axis=0) if picks else matrix.mean(axis=0))
    labels = _cluster(vectors)
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    energies = [features.energy_at(a, b) for a, b in spans]
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    ranks = [0.0] * len(spans)
    for position, index in enumerate(order):
        ranks[index] = position / max(1, len(spans) - 1)

    return tuple(
        _section(index, start, end, features, rhythm, labels[index], counts, ranks[index])
        for index, (start, end) in enumerate(spans))


def _section(index: int, start: float, end: float, features: TargetMusicFeatures,
             rhythm: RhythmBands, label: int, counts: dict[int, int],
             energy_rank: float) -> MusicSection:
    energy = features.energy_at(start, end)
    impact = features.impact_at(start, end)
    brightness = features.brightness_at(start, end)
    total = sum(counts.values())
    kind = classify_music_section(index, total, energy, impact, brightness,
                                  label, counts, energy_rank)
    return MusicSection(index=index, start=round(start, 3), end=round(end, 3),
                        duration=round(end - start, 3), type=kind,
                        energy=energy, impact=impact, brightness=brightness,
                        dominant_pattern=rhythm.dominant_at(start, end))


# ========================================================== 5. 固定音乐位置
def target_positions(duration: float, slice_duration: float) -> tuple[TargetPosition, ...]:
    """把目标歌切成等长的固定音乐位置：0-2 / 2-4 / 4-6 …（`slice_duration=2.0`）。

    这是整个系统与普通 BeatSync 的分界线（技术指导第七节）：位置是**目标歌的属性**，
    与任何源视频无关，所有源都往这同一把尺子上贴。所以这里只做除法，不看任何素材。

    末尾不足一整格的那一段**丢掉**：留一个 0.4 秒的碎片进成片只会露馅，
    而且它没法和任何一个源的完整动作对应。
    """
    duration = max(0.0, float(duration))
    step = float(slice_duration)
    if step <= 0:
        raise ValueError(f"slice_duration 必须为正，收到 {slice_duration}")
    count = int(duration // step)
    return tuple(TargetPosition(index=i, start=round(i * step, 6),
                                end=round((i + 1) * step, 6)) for i in range(count))


# ============================================================== 6. 一站式
def analyze_song(source: str | Path | np.ndarray, sample_rate: int = dsp.DEFAULT_SR,
                 ) -> dict[str, Any]:
    """把一首目标歌一次分析完，返回可直接落 `dance_target_songs` 的字典。

    传路径就自己解码，传数组就直接用（测试和 GUI 预览走后者）。
    四份结果一起返回是有意的：它们共用同一份 STFT 输入，分开调会把耗时翻三倍，
    而且分开缓存容易出现"节拍是新版算的、段落还是旧版"的错配。
    """
    if isinstance(source, np.ndarray):
        pcm = np.asarray(source, dtype=np.float32).reshape(-1)
    else:
        from .audio_fingerprint import extract_analysis_audio  # noqa: PLC0415

        pcm = extract_analysis_audio(source, sample_rate=sample_rate)

    grid = detect_target_beat_grid(pcm, sample_rate)
    features = analyze_target_music_features(pcm, sample_rate)
    rhythm = analyze_rhythm_bands(pcm, sample_rate)
    sections = analyze_music_sections(pcm, sample_rate, features, rhythm)
    logger.info("目标歌分析完成：%.2fs｜BPM %.2f（%s）｜%d 拍｜%d 段落",
                features.duration, grid.bpm, grid.source, len(grid.beats), len(sections))
    return {
        "duration": features.duration,
        "sample_rate": int(sample_rate),
        "bpm": grid.bpm,
        "beat_count": len(grid.beats),
        "analysis_version": ANALYSIS_VERSION,
        "beat_grid": grid,
        "features": features,
        "rhythm": rhythm,
        "sections": sections,
        "beats_json": grid.to_dict(),
        "sections_json": [s.to_dict() for s in sections],
        "features_json": features.to_dict(),
        "rhythm_json": rhythm.to_dict(),
    }


def section_at(sections: tuple[MusicSection, ...], moment: float) -> MusicSection | None:
    """某一刻落在哪个段落里。给评分层问"这个音乐位置属于什么段落"用。"""
    for section in sections:
        if section.contains(moment):
            return section
    return sections[-1] if sections and moment >= sections[-1].end else None


__all__ = [
    "ANALYSIS_VERSION", "BANDS", "MIN_SECTION_SECONDS", "SECTION_SIMILARITY",
    "detect_target_beat_grid", "analyze_target_music_features", "analyze_rhythm_bands",
    "analyze_music_sections", "classify_music_section",
    "target_positions", "analyze_song", "section_at",
]







