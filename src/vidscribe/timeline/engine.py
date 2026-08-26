"""Timeline Engine：把视觉事件和语音片段按时间区间对齐合并。

不是按相同 start/end 拼接，而是做区间重叠匹配：
- 视觉事件是时间轴主干；
- 与视觉事件有实质重叠的语音挂到该事件上；
- 没有被任何视觉事件覆盖的语音单独成条，保证语音不丢失。
"""

from __future__ import annotations

from typing import Any

from ..events import IMPORTANCE_ORDER, SpeechEvent, VisualEvent, confidence_level
from ..logging_setup import get_logger

logger = get_logger(__name__)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return min(a_end, b_end) - max(a_start, b_start)


def build_timeline(visual_events: list[VisualEvent], speech_events: list[SpeechEvent],
                   min_overlap: float = 0.2) -> list[dict[str, Any]]:
    # 每条语音只能挂到一个视觉事件上（重叠最多的那个），否则窗口 overlap 会让同一句话重复出现
    attach: dict[int, list[SpeechEvent]] = {ev.id: [] for ev in visual_events}
    unassigned: list[SpeechEvent] = []
    for sp in speech_events:
        best_ev: VisualEvent | None = None
        best_score = 0.0
        for ev in visual_events:
            ov = _overlap(ev.start, ev.end, sp.start, sp.end)
            center = (sp.start + sp.end) / 2.0
            inside = ev.start - 0.01 <= center <= ev.end + 0.01
            span = max(sp.end - sp.start, 1e-6)
            score = max(ov, 0.0) / span + (0.5 if inside else 0.0)
            if ov < min_overlap and not inside and ov < 0.5 * span:
                continue
            if score > best_score:
                best_ev, best_score = ev, score
        if best_ev is None:
            unassigned.append(sp)
        else:
            attach[best_ev.id].append(sp)

    entries: list[dict[str, Any]] = [_entry(ev, attach[ev.id]) for ev in visual_events]
    entries += [_entry(None, [sp]) for sp in unassigned]

    entries.sort(key=lambda e: (e["start"], e["end"]))
    for i, entry in enumerate(entries, start=1):
        entry["id"] = i
    return entries


def _entry(ev: VisualEvent | None, speech: list[SpeechEvent]) -> dict[str, Any]:
    speech = sorted(speech, key=lambda s: s.start)
    if ev is not None:
        start, end = ev.start, ev.end
    else:
        start, end = speech[0].start, speech[-1].end

    speech_text = " ".join(s.text.strip() for s in speech).strip() or None
    speech_conf = None
    confs = [s.confidence for s in speech if s.confidence is not None]
    if confs:
        speech_conf = round(sum(confs) / len(confs), 3)

    return {
        "id": 0,
        "start": round(start, 3),
        "end": round(end, 3),
        "visual": ev.description if ev else None,
        "visual_event": ev.event if ev else None,
        "speech": speech_text,
        "visual_event_id": ev.id if ev else None,
        "speech_event_ids": [s.id for s in speech],
        "importance": ev.importance if ev else "normal",
        "ocr_text": ev.ocr_text if ev else None,
        "timestamp_source": ev.timestamp_source if ev else "speech_based",
        "source_frames": ev.source_frames if ev else [],
        "visual_confidence": ev.confidence if ev else None,
        "speech_confidence": speech_conf,
        "quality": confidence_level(ev.confidence if ev else speech_conf),
    }


def filter_timeline(entries: list[dict[str, Any]], importance: str = "low",
                    min_confidence: float = 0.0) -> list[dict[str, Any]]:
    threshold = IMPORTANCE_ORDER.get(importance, 0)
    out = []
    for entry in entries:
        if entry["visual"] is None:  # 语音条目永不因重要性被丢弃
            out.append(entry)
            continue
        if IMPORTANCE_ORDER.get(entry["importance"], 1) < threshold:
            continue
        if (entry["visual_confidence"] or 0.0) < min_confidence:
            continue
        out.append(entry)
    return out
