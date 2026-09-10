"""舞蹈子系统的全部数据结构。

只放 dataclass 和纯函数（`to_dict` / `from_row` 这类搬字段的），**不 import av / cv2 /
torch**，也不 import 数据库 —— GUI 进程、测试、CLI 都要能安全 `import` 这一层。

命名对照（技术指导 → 本文件）：
    DanceAlignment          对齐结果，落 dance_audio_alignments
    BeatGrid                目标歌节拍网格
    TargetMusicFeatures     目标歌全局特征
    RhythmBands             kick / bass / clap / hihat 四条节奏带
    MusicSection            音乐段落（intro/verse/hook/...）
    TargetPosition          目标歌上的固定音乐位置（0-2 / 2-4 / …）
    SliceSpec / SlicePlan   固定音乐位置切片计划
    DanceMaterial           素材资产，落 dance_materials
    MaterialScore           单条素材的评分 + score_breakdown
    FilterSpec              多维筛选条件
    CandidatePool           候选池（一个 target position 一组候选）
    RecommendationRun/Item  推荐运行与条目
    MontageStrategy         评分权重 + 组合约束 + 搜索预算
    DanceMontageClip        编辑计划里的一个片段
    DanceMontageContext     一次混剪的完整上下文（纯计划）
    RepeatStats             七种重复率
    RenderResult            渲染 + mux 的结果
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

# ====================================================================== 状态枚举
#: 对齐结论。ok = 可以直接用；low_confidence = 建议人工确认；rejected = 不可用
ALIGNMENT_STATUS = ("ok", "low_confidence", "disagree", "rejected", "manual")
#: 素材状态。历史素材一律软状态流转，不物理删除。
#: missing = 文件在盘上找不到了（记录留着，历史成品还引用它）
MATERIAL_STATUS = ("ready", "disabled", "missing", "invalid", "regenerated")

#: 素材使用事件。candidate 只是「进过候选池」，不等于用过
USAGE_EVENTS = ("candidate", "selected", "montage", "render_success", "render_failed", "rejected")
#: 推荐策略种类
STRATEGY_KINDS = ("cold_start", "rule_based", "history_based", "exploration", "hybrid")
#: 音乐段落类型。只是辅助特征，**禁止**写死成最终素材选择规则
SECTION_TYPES = ("intro", "verse", "hook", "chorus", "bridge",
                 "breakdown", "drop", "finale", "outro")
#: 排序键。技术指导第十二节要求至少这六种
SORT_KEYS = ("score_desc", "use_count_asc", "output_count_asc",
             "last_used_at_asc", "confidence_desc", "freshness_desc")

def _round(value: Any, digits: int = 4) -> Any:
    """浮点统一保留位数：落库和断言都靠它，避免同一份数据两次算出不同尾数。"""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _dict(obj: Any, digits: int = 4) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in asdict(obj).items():
        if isinstance(value, (list, tuple)):
            out[key] = [_round(item, digits) for item in value]
        elif isinstance(value, dict):
            out[key] = {k: _round(v, digits) for k, v in value.items()}
        else:
            out[key] = _round(value, digits)
    return out


def _from_row(cls: Any, row: Mapping[str, Any]) -> Any:
    """按字段名从 sqlite3.Row 建 dataclass；表里没有的键一律走默认值。"""
    keys = set(row.keys()) if hasattr(row, "keys") else set(row)
    data = {name: row[name] for name in cls.__dataclass_fields__
            if name in keys and row[name] is not None}
    return cls(**data)


# ================================================================== 音频对齐
@dataclass(frozen=True)
class WindowResult:
    """单个验证窗口的对齐结果。多窗口一致性就是拿这些 offset 比出来的。"""

    index: int
    window_start: float
    window_seconds: float
    offset: float
    confidence: float
    method: str = "waveform"

    def to_dict(self) -> dict[str, Any]:
        return _dict(self)

@dataclass(frozen=True)
class DanceAlignment:
    """source 舞蹈视频 → target 目标歌的时间对齐。

    口径只有一条，全系统统一：`source_time = target_time - offset`。
    也就是说 offset > 0 表示「源视频比目标歌晚开始」，源里的动作要往前找。

    `status` 见 `ALIGNMENT_STATUS`；`manual_offset` 非 None 表示人工改过，
    此时 `offset` 已经是人工值，`original_offset` 保留算法原值 —— 禁止静默覆盖。
    """

    offset: float
    confidence: float
    method: str = "hybrid"
    waveform_offset: float | None = None
    waveform_confidence: float | None = None
    chroma_offset: float | None = None
    chroma_confidence: float | None = None
    window_count: int = 0
    max_deviation: float = 0.0
    agreement: float = 0.0
    status: str = "ok"
    algorithm_version: str = ""
    source_duration: float = 0.0
    target_duration: float = 0.0
    sample_rate: int = 0
    windows: tuple[WindowResult, ...] = ()
    notes: tuple[str, ...] = ()
    original_offset: float | None = None
    original_confidence: float | None = None
    manual_offset: float | None = None
    manual_reason: str = ""
    manual_at: str = ""
    manual_operator: str = ""

    @property
    def manual(self) -> bool:
        return self.manual_offset is not None

    def source_time(self, target_time: float) -> float:
        """目标歌时刻 → 源视频时刻。整个切片系统只认这一个换算。"""
        return round(float(target_time) - self.offset, 6)

    def target_time(self, source_time: float) -> float:
        return round(float(source_time) + self.offset, 6)

    def to_dict(self) -> dict[str, Any]:
        data = _dict(replace(self, windows=()))
        data["windows"] = [w.to_dict() for w in self.windows]
        data["notes"] = list(self.notes)
        data["manual"] = self.manual
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

# ============================================================== 目标音乐结构
@dataclass(frozen=True)
class BeatGrid:
    """目标歌的节拍网格。`beats` 是绝对秒，严格递增。

    `source` 记这一版是怎么来的：`beat_track` = 走了完整节拍跟踪，
    `onset_fallback` = 节拍跟踪失败退回 onset 峰值，`grid` = 只有等间距网格。
    """

    bpm: float
    beats: tuple[float, ...] = ()
    downbeats: tuple[float, ...] = ()
    duration: float = 0.0
    beats_per_bar: int = 4
    source: str = "beat_track"
    confidence: float = 0.0

    @property
    def period(self) -> float:
        return round(60.0 / self.bpm, 6) if self.bpm > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": _round(self.bpm, 3),
            "period": self.period,
            "beats": [round(float(b), 3) for b in self.beats],
            "downbeats": [round(float(b), 3) for b in self.downbeats],
            "duration": _round(self.duration, 3),
            "beats_per_bar": self.beats_per_bar,
            "source": self.source,
            "confidence": _round(self.confidence),
        }


@dataclass(frozen=True)
class RhythmBands:
    """四条节奏带的逐帧能量（0~1 归一化）+ 每带的击点时刻。

    kick 20~120Hz / bass 60~250Hz / clap 1.5~4kHz / hihat 6~14kHz。
    先保证 CPU 正确性，不做 GPU 化。
    """

    frame_times: tuple[float, ...] = ()
    kick: tuple[float, ...] = ()
    bass: tuple[float, ...] = ()
    clap: tuple[float, ...] = ()
    hihat: tuple[float, ...] = ()
    kick_hits: tuple[float, ...] = ()
    bass_hits: tuple[float, ...] = ()
    clap_hits: tuple[float, ...] = ()
    hihat_hits: tuple[float, ...] = ()

    def dominant_at(self, start: float, end: float) -> str:
        """区间内哪条带最强。给 MusicSection.dominant_pattern 用。"""
        names = ("kick", "bass", "clap", "hihat")
        tracks = (self.kick, self.bass, self.clap, self.hihat)
        if not self.frame_times:
            return ""
        picks = [i for i, t in enumerate(self.frame_times) if start <= t < end]
        if not picks:
            return ""
        best, best_value = "", -1.0
        for name, track in zip(names, tracks):
            if not track:
                continue
            value = sum(track[i] for i in picks if i < len(track)) / len(picks)
            if value > best_value:
                best, best_value = name, value
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "kick_hits": [round(float(x), 3) for x in self.kick_hits],
            "bass_hits": [round(float(x), 3) for x in self.bass_hits],
            "clap_hits": [round(float(x), 3) for x in self.clap_hits],
            "hihat_hits": [round(float(x), 3) for x in self.hihat_hits],
            "frames": len(self.frame_times),
        }

@dataclass(frozen=True)
class MusicSection:
    """目标歌的一个段落。`type` 见 `SECTION_TYPES`，只是辅助特征。"""

    index: int
    start: float
    end: float
    duration: float
    type: str = "verse"
    energy: float = 0.0
    impact: float = 0.0
    brightness: float = 0.0
    dominant_pattern: str = ""

    def contains(self, moment: float) -> bool:
        return self.start <= moment < self.end

    def to_dict(self) -> dict[str, Any]:
        return _dict(self, 3)


@dataclass(frozen=True)
class TargetMusicFeatures:
    """目标歌的全局特征。逐帧曲线用于按位置取能量/亮度/冲击力。"""

    duration: float
    sample_rate: int
    hop_seconds: float
    frame_times: tuple[float, ...] = ()
    rms: tuple[float, ...] = ()
    centroid: tuple[float, ...] = ()
    flux: tuple[float, ...] = ()
    onset: tuple[float, ...] = ()
    novelty: tuple[float, ...] = ()
    energy_wave: tuple[float, ...] = ()
    brightness: tuple[float, ...] = ()
    rhythm_score: float = 0.0
    impact_score: float = 0.0
    arc: tuple[float, ...] = ()

    def _mean(self, track: Sequence[float], start: float, end: float) -> float:
        if not track or not self.frame_times:
            return 0.0
        picks = [track[i] for i, t in enumerate(self.frame_times)
                 if start <= t < end and i < len(track)]
        if not picks:
            # 区间比一帧还短时取最近那一帧，不返回 0（0 会被下游当成"这里很安静"）
            nearest = min(range(len(self.frame_times)),
                          key=lambda i: abs(self.frame_times[i] - start))
            return float(track[nearest]) if nearest < len(track) else 0.0
        return float(sum(picks) / len(picks))

    def energy_at(self, start: float, end: float) -> float:
        return round(self._mean(self.energy_wave, start, end), 4)

    def impact_at(self, start: float, end: float) -> float:
        return round(self._mean(self.onset, start, end), 4)

    def brightness_at(self, start: float, end: float) -> float:
        return round(self._mean(self.brightness, start, end), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": _round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "hop_seconds": _round(self.hop_seconds, 6),
            "frames": len(self.frame_times),
            "rhythm_score": _round(self.rhythm_score),
            "impact_score": _round(self.impact_score),
            "arc": [round(float(x), 4) for x in self.arc],
        }

# ========================================================== 固定音乐位置切片
@dataclass(frozen=True)
class TargetPosition:
    """目标歌上的一个固定音乐位置。`slice_duration=2.0` 时就是 0-2 / 2-4 / 4-6 …

    位置是**目标歌的属性**，与任何源视频无关 —— 所有源视频都往这同一把尺子上贴。
    """

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": _round(self.start, 3),
                "end": _round(self.end, 3), "duration": _round(self.duration, 3)}


@dataclass(frozen=True)
class SliceSpec:
    """一条切片计划：目标位置 + 换算出来的源区间。

    `source_start = target_start - offset`，`source_end = target_end - offset`。
    源区间越界时**必须报错**，禁止静默 clamp（clamp 会让素材内容和音乐位置错开，
    而这个系统的全部价值就建立在「素材已经绑定到该音乐位置」这一条上）。
    """

    segment_index: int
    target_start: float
    target_end: float
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        return round(self.target_end - self.target_start, 6)

    def to_dict(self) -> dict[str, Any]:
        data = _dict(self, 6)
        data["duration"] = _round(self.duration, 6)
        return data


@dataclass(frozen=True)
class SlicePlan:
    """一个源视频对一首目标歌的完整切片计划。

    `skipped` 记「为什么这个位置切不出来」（源区间越界 / 源视频不够长），
    一条都不许静默丢掉 —— GUI 要能把原因显示给用户。
    """

    source_video_id: int
    target_song_id: int
    alignment_offset: float
    slice_duration: float
    specs: tuple[SliceSpec, ...] = ()
    skipped: tuple[tuple[int, str], ...] = ()
    generation_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_video_id": self.source_video_id,
            "target_song_id": self.target_song_id,
            "alignment_offset": _round(self.alignment_offset, 6),
            "slice_duration": _round(self.slice_duration, 3),
            "specs": [s.to_dict() for s in self.specs],
            "skipped": [{"segment_index": i, "reason": why} for i, why in self.skipped],
            "generation_version": self.generation_version,
        }

# ================================================================== 素材资产
@dataclass
class DanceMaterial:
    """一条素材资产（dance_materials 的一行）。

    素材是**长期资产**：切一次，之后无数次混剪都从库里挑它，用一次记一次账。
    `use_count` / `montage_count` / `output_count` 全部可以从事件账本重算
    （见 `history.recount_material`），库里的数只是加速用的缓存。
    """

    id: int = 0
    source_video_id: int = 0
    alignment_id: int = 0
    target_song_id: int = 0
    segment_index: int = 0
    target_start: float = 0.0
    target_end: float = 0.0
    source_start: float = 0.0
    source_end: float = 0.0
    duration: float = 0.0
    file_path: str = ""
    file_hash: str = ""
    alignment_confidence: float = 0.0
    generation_version: str = ""
    person: str = ""
    source_group: str = ""
    quality: float = 0.0
    candidate_count: int = 0
    use_count: int = 0
    montage_count: int = 0
    output_count: int = 0
    first_used_at: str = ""
    last_used_at: str = ""
    last_output_at: str = ""
    status: str = "ready"
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    # 以下不落 dance_materials，是 JOIN / 现算出来的展示与评分辅助字段
    source_name: str = ""
    section_type: str = ""

    @property
    def never_used(self) -> bool:
        return self.use_count <= 0

    @property
    def never_output(self) -> bool:
        return self.output_count <= 0

    @property
    def usable(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return _dict(self, 6)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DanceMaterial":
        return _from_row(cls, row)

# ============================================================== 评分与候选池
@dataclass(frozen=True)
class MaterialScore:
    """一条素材在某个目标位置上的得分。

    `breakdown` 是**必须**输出的：评分是黑盒的话，用户看到"为什么这条被选中"
    就只能猜。键名与技术指导第十节一致。
    """

    material_id: int
    segment_index: int
    final_score: float
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "segment_index": self.segment_index,
            "final_score": _round(self.final_score),
            "breakdown": {k: _round(v) for k, v in self.breakdown.items()},
        }

    def breakdown_json(self) -> str:
        return json.dumps({k: _round(v) for k, v in self.breakdown.items()},
                          ensure_ascii=False, sort_keys=True)


@dataclass
class FilterSpec:
    """多维筛选条件。全部可选，`None` / 空 = 这一维不筛。

    这一层是纯声明，翻译成 SQL 的活儿在 `material_selection` 里干。
    """

    target_song_id: int | None = None
    segment_index: int | None = None
    segment_indexes: tuple[int, ...] = ()
    section_type: str = ""
    min_confidence: float | None = None
    max_confidence: float | None = None
    min_use_count: int | None = None
    max_use_count: int | None = None
    min_montage_count: int | None = None
    max_montage_count: int | None = None
    min_output_count: int | None = None
    max_output_count: int | None = None
    never_used: bool = False
    never_output: bool = False
    recently_used_days: float | None = None
    long_unused_days: float | None = None
    source_video_ids: tuple[int, ...] = ()
    persons: tuple[str, ...] = ()
    source_groups: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ("ready",)
    generation_versions: tuple[str, ...] = ()
    search: str = ""
    sort: str = "score_desc"
    limit: int = 500

    def to_dict(self) -> dict[str, Any]:
        return _dict(self)


@dataclass(frozen=True)
class CandidatePool:
    """一个目标位置的候选池：已评分、已排序的素材列表。

    进候选池只记 `candidate` 事件，**不算用过** —— use_count 只在真的进了 montage
    才 +1（技术指导第九节）。
    """

    segment_index: int
    target_start: float
    target_end: float
    scored: tuple[MaterialScore, ...] = ()
    materials: dict[int, DanceMaterial] = field(default_factory=dict)

    @property
    def top(self) -> MaterialScore | None:
        return self.scored[0] if self.scored else None

    def head(self, k: int) -> tuple[MaterialScore, ...]:
        return self.scored[:max(0, int(k))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "target_start": _round(self.target_start, 3),
            "target_end": _round(self.target_end, 3),
            "count": len(self.scored),
            "scored": [s.to_dict() for s in self.scored],
        }

# ==================================================================== 推荐
@dataclass(frozen=True)
class RecommendationItem:
    """推荐结果里的一条。`rank` 从 1 起。"""

    segment_index: int
    material_id: int
    rank: int
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "material_id": self.material_id,
            "rank": self.rank,
            "score": _round(self.score),
            "breakdown": {k: _round(v) for k, v in self.breakdown.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecommendationRun:
    """一次推荐运行。同样的 seed + 同样的库状态 + 同样的策略 = 同样的结果。

    可重现是硬要求：`random_seed` 必须落库，随机数一律走
    `recommendation.stable_rng(seed, *parts)`，不许用全局 `random`。
    """

    id: int = 0
    target_song_id: int = 0
    montage_id: int = 0
    strategy_id: int = 0
    strategy_kind: str = "hybrid"
    strategy_version: str = ""
    random_seed: int = 0
    candidate_count: int = 0
    recommended_count: int = 0
    algorithm_version: str = ""
    created_at: str = ""
    items: tuple[RecommendationItem, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_song_id": self.target_song_id,
            "montage_id": self.montage_id,
            "strategy_id": self.strategy_id,
            "strategy_kind": self.strategy_kind,
            "strategy_version": self.strategy_version,
            "random_seed": self.random_seed,
            "candidate_count": self.candidate_count,
            "recommended_count": self.recommended_count,
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
            "items": [i.to_dict() for i in self.items],
            "notes": self.notes,
        }


# ==================================================================== 策略
@dataclass
class MontageStrategy:
    """一份混剪策略：评分权重 + 组合约束 + 搜索预算，整份可落库、可复用、可版本化。

    LLM 可以帮用户"生成一份策略"，但**最终选择必须由确定性评分/搜索完成**
    （技术指导第二十一节第 4 条）—— 所以策略里只有数字，没有"让模型自己挑"这种口子。
    """

    id: int = 0
    name: str = "默认策略"
    kind: str = "hybrid"
    version: str = "v1"
    weights: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    is_default: int = 0
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "version": self.version,
            "weights": {k: _round(v) for k, v in self.weights.items()},
            "constraints": dict(self.constraints),
            "search": dict(self.search),
            "is_default": int(self.is_default), "note": self.note,
        }

# ================================================================== 编辑计划
@dataclass(frozen=True)
class DanceMontageClip:
    """编辑计划里的一个片段。

    引用的是 `material_id`，**不是** source path —— 素材才是资产，路径只是它现在
    恰好躺在哪儿（技术指导第十五节）。
    """

    order_index: int
    segment_index: int
    material_id: int
    target_start: float
    target_end: float
    source_start: float
    source_end: float
    file_path: str = ""
    selection_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    person: str = ""
    source_video_id: int = 0

    @property
    def duration(self) -> float:
        return round(self.target_end - self.target_start, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_index": self.order_index,
            "segment_index": self.segment_index,
            "material_id": self.material_id,
            "target_start": _round(self.target_start, 6),
            "target_end": _round(self.target_end, 6),
            "source_start": _round(self.source_start, 6),
            "source_end": _round(self.source_end, 6),
            "duration": _round(self.duration, 6),
            "file_path": self.file_path,
            "selection_score": _round(self.selection_score),
            "score_breakdown": {k: _round(v) for k, v in self.score_breakdown.items()},
            "person": self.person,
            "source_video_id": self.source_video_id,
        }


@dataclass(frozen=True)
class RepeatStats:
    """一版混剪的重复率。七个口径全部落 dance_montage_versions。

    每个值都是 0~1：0 = 一点不重复，1 = 全在重复。算法见 `statistics.repeat_stats`。
    """

    material_repeat: float = 0.0
    position_repeat: float = 0.0
    person_repeat: float = 0.0
    source_repeat: float = 0.0
    pair_repeat: float = 0.0
    combination_repeat: float = 0.0
    overall_repeat: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_repeat": _round(self.material_repeat),
            "position_repeat": _round(self.position_repeat),
            "person_repeat": _round(self.person_repeat),
            "source_repeat": _round(self.source_repeat),
            "pair_repeat": _round(self.pair_repeat),
            "combination_repeat": _round(self.combination_repeat),
            "overall_repeat": _round(self.overall_repeat),
            "detail": dict(self.detail),
        }

@dataclass
class DanceMontageContext:
    """一次混剪的完整上下文 —— **纯编辑计划**。

    Timeline 这一层禁止再做推荐/选择（技术指导第十五节）：进来的时候片段已经定了，
    这里只负责排顺序、算成片时间、报重复率、给渲染层一份不需要再思考的清单。
    """

    target_song_id: int = 0
    target_song_path: str = ""
    slice_duration: float = 2.0
    clips: list[DanceMontageClip] = field(default_factory=list)
    strategy_id: int = 0
    recommendation_run_id: int = 0
    montage_id: int = 0
    version_index: int = 1
    signature: str = ""
    repeat: RepeatStats = field(default_factory=RepeatStats)
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(sum(c.duration for c in self.clips), 6)

    @property
    def material_ids(self) -> tuple[int, ...]:
        return tuple(c.material_id for c in self.clips)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_song_id": self.target_song_id,
            "target_song_path": self.target_song_path,
            "slice_duration": _round(self.slice_duration, 3),
            "duration": _round(self.duration, 3),
            "montage_id": self.montage_id,
            "version_index": self.version_index,
            "signature": self.signature,
            "strategy_id": self.strategy_id,
            "recommendation_run_id": self.recommendation_run_id,
            "clips": [c.to_dict() for c in self.clips],
            "repeat": self.repeat.to_dict(),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# ==================================================================== 渲染
@dataclass
class RenderResult:
    """渲染 + mux 的结果。`audio_streams == 1` 是硬验收：目标歌必须是唯一音轨。"""

    output: str = ""
    ok: bool = False
    clips: int = 0
    frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    audio_duration: float = 0.0
    audio_streams: int = 0
    video_streams: int = 0
    backend: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dict(self, 3)


# ==================================================== 人声活动 / 停顿导航点
@dataclass(frozen=True)
class VocalSpan:
    """一段"有人声"或"没人声"的区间。`kind` 只有 `vocal` / `pause` 两种。

    说明白它是什么：这是**频谱启发式**的判断（人声频带能量 + 谐波性），
    不是源分离，也没有模型。所以它只是给人看的参考层，
    最终在哪儿切段永远由用户点下去决定（见 `segment_template`）。
    """

    index: int
    start: float
    end: float
    kind: str = "vocal"
    strength: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    def to_dict(self) -> dict[str, Any]:
        data = _dict(self, 3)
        data["duration"] = _round(self.duration, 3)
        return data


@dataclass(frozen=True)
class VocalPause:
    """一个人声停顿 —— 界面左边那份"导航点"列表里的一行。

    `score` 是推荐度（0~1）：停得越久、越干净、越贴着拍点，就越可能适合换人。
    `nearest_beat` 是最近的拍点（秒），`-1` 表示这首歌没有可用的拍网格。
    """

    index: int
    start: float
    end: float
    score: float = 0.0
    nearest_beat: float = -1.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    @property
    def middle(self) -> float:
        """停顿正中间 —— 「跳到这个停顿」时播放头落在这里最稳。"""
        return round((self.start + self.end) / 2.0, 6)

    def to_dict(self) -> dict[str, Any]:
        data = _dict(self, 3)
        data["duration"] = _round(self.duration, 3)
        data["middle"] = _round(self.middle, 3)
        return data


@dataclass(frozen=True)
class VocalActivity:
    """一首歌的人声活动分析结果：逐帧强度 + 区间 + 停顿导航点。

    `strength` 是 0~1 的逐帧曲线，`threshold` 是当时用的判定门槛 ——
    两个都留着，界面才能把门槛画成一条横线，让用户看出"为什么这里算有人声"。
    """

    duration: float
    sample_rate: int
    hop_seconds: float
    frame_times: tuple[float, ...] = ()
    strength: tuple[float, ...] = ()
    threshold: float = 0.0
    spans: tuple[VocalSpan, ...] = ()
    pauses: tuple[VocalPause, ...] = ()
    version: str = ""

    def at(self, moment: float) -> float:
        """某一刻的人声强度。越界返回 0。"""
        if not self.frame_times or not self.strength:
            return 0.0
        nearest = min(range(len(self.frame_times)),
                      key=lambda i: abs(self.frame_times[i] - float(moment)))
        return round(float(self.strength[nearest]), 4)

    def speaking_at(self, moment: float) -> bool:
        """某一刻是否落在人声区间里 —— 状态栏那句"人声中 / 停顿中"。"""
        return any(s.kind == "vocal" and s.start <= moment < s.end for s in self.spans)

    def next_pause(self, moment: float) -> VocalPause | None:
        """下一处人声停顿。没有了返回 None（界面显示「—」，不要编一个）。"""
        for pause in self.pauses:
            if pause.start > moment:
                return pause
        return None

    def previous_pause(self, moment: float) -> VocalPause | None:
        for pause in reversed(self.pauses):
            if pause.end < moment:
                return pause
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": _round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "hop_seconds": _round(self.hop_seconds, 6),
            "frames": len(self.frame_times),
            "threshold": _round(self.threshold, 4),
            "spans": [s.to_dict() for s in self.spans],
            "pauses": [p.to_dict() for p in self.pauses],
            "version": self.version,
        }


# ======================================================== 用户确定的段落模板
@dataclass(frozen=True)
class SegmentSpan:
    """段落模板里的一格：S1 / S2 / S3 …

    和 `TargetPosition` 的区别很重要，别混：
    `TargetPosition` 是「等间隔的尺子」，机器算的；
    `SegmentSpan` 是「用户拍板的段落」，长度可以各不相同。
    所有源视频都继承同一份模板，不会因为某个视频的人声不同而各自重新分段。
    """

    index: int
    start: float
    end: float
    label: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    @property
    def name(self) -> str:
        """显示名：没起名字就叫 S1 / S2 …（下标从 0 开始，名字从 1 开始）。"""
        return self.label or f"S{self.index + 1}"

    def contains(self, moment: float) -> bool:
        return self.start <= float(moment) < self.end

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": _round(self.start, 3),
                "end": _round(self.end, 3), "duration": _round(self.duration, 3),
                "label": self.label, "note": self.note}


@dataclass(frozen=True)
class SegmentTemplate:
    """一首歌的段落模板：一串首尾相接、不重叠、不留缝的 `SegmentSpan`。

    `source` 记它是怎么来的：`uniform`（等间隔生成）/ `pause`（照人声停顿生成）/
    `manual`（用户拖过边界）—— 界面要能告诉用户"这份分段是谁定的"。
    """

    target_song_id: int = 0
    duration: float = 0.0
    spans: tuple[SegmentSpan, ...] = ()
    source: str = "uniform"
    name: str = ""
    note: str = ""
    version: str = ""

    @property
    def boundaries(self) -> tuple[float, ...]:
        """内部分割点（不含 0 和结尾）—— 拖动的就是这些点。"""
        return tuple(span.start for span in self.spans[1:])

    def span_at(self, moment: float) -> SegmentSpan | None:
        for span in self.spans:
            if span.contains(moment):
                return span
        return self.spans[-1] if self.spans and moment >= self.spans[-1].end else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_song_id": self.target_song_id,
            "duration": _round(self.duration, 3),
            "spans": [s.to_dict() for s in self.spans],
            "source": self.source,
            "name": self.name,
            "note": self.note,
            "version": self.version,
        }


__all__ = [
    "ALIGNMENT_STATUS", "MATERIAL_STATUS", "USAGE_EVENTS", "STRATEGY_KINDS",
    "SECTION_TYPES", "SORT_KEYS",
    "WindowResult", "DanceAlignment",
    "BeatGrid", "RhythmBands", "MusicSection", "TargetMusicFeatures",
    "TargetPosition", "SliceSpec", "SlicePlan",
    "VocalSpan", "VocalPause", "VocalActivity",
    "SegmentSpan", "SegmentTemplate",
    "DanceMaterial", "MaterialScore", "FilterSpec", "CandidatePool",
    "RecommendationItem", "RecommendationRun", "MontageStrategy",
    "DanceMontageClip", "RepeatStats", "DanceMontageContext", "RenderResult",
]










