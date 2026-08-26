"""事件数据结构 + 合并/去重逻辑（视觉事件的"什么时候"由程序决定）。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

IMPORTANCE_ORDER = {"low": 0, "normal": 1, "high": 2, "critical": 3}
TIMESTAMP_SOURCES = ("frame_based", "hybrid", "model_estimated")


@dataclass
class VisualEvent:
    id: int
    start: float
    end: float
    event: str
    description: str
    confidence: float = 0.5
    importance: str = "normal"
    timestamp_source: str = "model_estimated"
    source_frames: list[int] = field(default_factory=list)
    ocr_text: str | None = None
    window: list[float] | None = None
    # --- 内部结构化事实：固定英文，不随最终输出语言变化（用于合并/去重/兜底渲染）---
    action: str | None = None
    scene: str | None = None
    subjects: list[str] = field(default_factory=list)
    # --- 最终自然语言层的记录 ---
    description_language: str | None = None
    language_fallback: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def importance_rank(self) -> int:
        return IMPORTANCE_ORDER.get(self.importance, 1)


@dataclass
class SpeechWord:
    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class SpeechEvent:
    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None
    language: str | None = None
    # 原始语音识别结果永不被覆盖：最终输出语言变了也要能拿回原话
    original_text: str | None = None
    original_language: str | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    words: list[SpeechWord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.original_text is None:
            self.original_text = self.text
        if self.original_language is None:
            self.original_language = self.language

    def to_dict(self) -> dict:
        return asdict(self)


def confidence_level(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


_NORM_RE = re.compile(r"[\s，。、,.!！?？:：;；\"'“”‘’()（）\-—_]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def text_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _event_similarity(a: VisualEvent, b: VisualEvent) -> float:
    return max(
        text_similarity(a.event, b.event),
        0.9 * text_similarity(a.description, b.description),
    )


def _overlap(a: VisualEvent, b: VisualEvent) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def _absorb(keep: VisualEvent, other: VisualEvent) -> VisualEvent:
    """把 other 合并进 keep：时间区间取并集，元数据取更可靠的一方。"""
    keep.start = min(keep.start, other.start)
    keep.end = max(keep.end, other.end)
    keep.confidence = round(max(keep.confidence, other.confidence), 3)
    if other.importance_rank > keep.importance_rank:
        keep.importance = other.importance
    if len(other.description) > len(keep.description):
        keep.description = other.description
    if other.ocr_text and not keep.ocr_text:
        keep.ocr_text = other.ocr_text
    # 内部英文事实：动作/场景保留先出现的，主体取并集（跨窗口更完整）
    if not keep.action and other.action:
        keep.action = other.action
    if not keep.scene and other.scene:
        keep.scene = other.scene
    if other.subjects:
        keep.subjects = sorted(set(keep.subjects) | set(other.subjects))
    keep.language_fallback = keep.language_fallback or other.language_fallback
    frames = sorted(set(keep.source_frames) | set(other.source_frames))
    keep.source_frames = frames
    ranks = {s: i for i, s in enumerate(TIMESTAMP_SOURCES)}
    keep.timestamp_source = min(
        (keep.timestamp_source, other.timestamp_source), key=lambda s: ranks.get(s, 9)
    )
    return keep


def dedupe_across_windows(events: list[VisualEvent], similarity: float = 0.72) -> list[VisualEvent]:
    """消除窗口 overlap 区域产生的重复事件。"""
    result: list[VisualEvent] = []
    for ev in sorted(events, key=lambda e: (e.start, e.end)):
        merged = False
        for kept in result:
            if _overlap(kept, ev) <= 0 and min(abs(kept.end - ev.start), abs(ev.end - kept.start)) > 1.0:
                continue
            if _event_similarity(kept, ev) >= similarity:
                _absorb(kept, ev)
                merged = True
                break
        if not merged:
            result.append(ev)
    return sorted(result, key=lambda e: (e.start, e.end))


def merge_adjacent(events: list[VisualEvent], similarity: float = 0.82,
                   max_gap: float = 1.5) -> list[VisualEvent]:
    """相邻的同一状态事件合并成一个长事件（"男人一直站着" 只输出一条）。"""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: (e.start, e.end))
    merged: list[VisualEvent] = [ordered[0]]
    for ev in ordered[1:]:
        prev = merged[-1]
        gap = ev.start - prev.end
        if gap <= max_gap and _event_similarity(prev, ev) >= similarity:
            _absorb(prev, ev)
        else:
            merged.append(ev)
    return merged


def drop_noise(events: list[VisualEvent], min_seconds: float = 0.4) -> list[VisualEvent]:
    """过滤过短的碎片事件，但保留 high/critical。"""
    out = []
    for ev in events:
        if ev.duration < min_seconds and ev.importance_rank < 2:
            continue
        out.append(ev)
    return out


def resolve_overlaps(events: list[VisualEvent], min_seconds: float = 0.4) -> list[VisualEvent]:
    """消除相邻事件的时间重叠：窗口 overlap 会让不同窗口给出交叉的时间段。

    事件已按 start 排序，把前一个事件的 end 裁到后一个的 start；
    如果前一个把后一个完全包住，保留更具体（时间段更短）的那个。
    """
    if len(events) < 2:
        return events
    ordered = sorted(events, key=lambda e: (e.start, e.end))
    result: list[VisualEvent] = []
    for ev in ordered:
        if not result:
            result.append(ev)
            continue
        prev = result[-1]
        if ev.start >= prev.end - 1e-3:
            result.append(ev)
            continue
        if ev.end <= prev.end + 1e-3 and prev.importance_rank > ev.importance_rank:
            continue  # 被包住且没那么重要，丢掉
        prev.end = round(max(prev.start + min_seconds, ev.start), 3)
        if prev.end - prev.start < min_seconds - 1e-6 and prev.importance_rank < 2:
            result.pop()
        result.append(ev)
    return result


def finalize(events: list[VisualEvent], duration: float, *, dedup_similarity: float,
             merge_similarity: float, min_seconds: float) -> list[VisualEvent]:
    for ev in events:
        ev.start = max(0.0, round(min(ev.start, duration if duration > 0 else ev.start), 3))
        ev.end = round(min(max(ev.end, ev.start), duration if duration > 0 else ev.end), 3)
    events = dedupe_across_windows(events, dedup_similarity)
    events = merge_adjacent(events, merge_similarity)
    events = resolve_overlaps(events, min_seconds)
    events = drop_noise(events, min_seconds)
    events = sorted(events, key=lambda e: (e.start, e.end))
    for i, ev in enumerate(events, start=1):
        ev.id = i
    return events
