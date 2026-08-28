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

    # 一条时间轴可能挂着好几句语音，取情绪最强的那句代表这一条
    speech_emotion = speech_emotion_en = speech_intensity = None
    scored = [s for s in speech if s.emotion and s.emotion_intensity is not None]
    if scored:
        top = max(scored, key=lambda s: float(s.emotion_intensity))
        speech_emotion = top.emotion
        speech_emotion_en = top.emotion_en
        speech_intensity = round(float(top.emotion_intensity), 3)

    # 说话人：一条时间轴可能挂着好几句，按出现顺序去重记下来（可能真是两个人对话）
    speakers: list[int] = []
    for s in speech:
        if s.speaker is not None and int(s.speaker) not in speakers:
            speakers.append(int(s.speaker))

    return {
        "id": 0,
        "start": round(start, 3),
        "end": round(end, 3),
        "visual": ev.description if ev else None,
        "visual_event": ev.event if ev else None,
        # 结构化事实（固定英文小写标签）：description 是自然语言，按动作/场景检索得靠这三个
        "action": ev.action if ev else None,
        "scene": ev.scene if ev else None,
        "subjects": list(ev.subjects) if ev and ev.subjects else [],
        "speech": speech_text,
        "visual_event_id": ev.id if ev else None,
        "speech_event_ids": [s.id for s in speech],
        "importance": ev.importance if ev else "normal",
        "ocr_text": ev.ocr_text if ev else None,
        "timestamp_source": ev.timestamp_source if ev else "speech_based",
        "source_frames": ev.source_frames if ev else [],
        "visual_confidence": ev.confidence if ev else None,
        "speech_confidence": speech_conf,
        "speech_speakers": speakers,
        # 情绪分两路：语音情绪来自 emotion2vec，画面情绪来自视觉模型，各自独立。
        # 同时给英文标签：切到译文视图时显示名要按译文语言重渲，不能拿写死的显示名。
        "speech_emotion": speech_emotion,
        "speech_emotion_en": speech_emotion_en,
        "speech_emotion_intensity": speech_intensity,
        "visual_emotion": ev.emotion if ev else None,
        "visual_emotion_en": ev.emotion_en if ev else None,
        "visual_emotion_intensity": ev.emotion_intensity if ev else None,
        "quality": confidence_level(ev.confidence if ev else speech_conf),
    }


def action_track(events: list[VisualEvent], max_gap: float = 0.5) -> list[dict[str, Any]]:
    """逐动作时间戳：相邻事件的 action 标签相同就并成一段。

    纯归并，不额外推理——动作标签是视觉模型这一趟已经给出的。粒度就是事件粒度
    （本机实测平均 8~9s 一段），要更细必须加帧或上骨架模型，那会实打实加推理时间。
    """
    track: list[dict[str, Any]] = []
    for ev in events:
        label = ev.action or None
        if not label:
            continue
        last = track[-1] if track else None
        if last and last["action"] == label and ev.start - last["end"] <= max_gap:
            last["end"] = round(max(last["end"], ev.end), 3)
            last["event_ids"].append(ev.id)
            continue
        track.append({"start": round(ev.start, 3), "end": round(ev.end, 3),
                      "action": label, "scene": ev.scene or None, "event_ids": [ev.id]})
    return track


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
