"""导出 timeline.json / timeline.txt / timeline.srt。

JSON 是唯一完整的数据源；TXT 和 SRT 只是导出格式。
所有时间都保留真实秒数（浮点），保证播放器可以精确定位。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

logger = get_logger(__name__)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes:02d}:{sec:05.2f}"


def fmt_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_timeline_txt(path: Path, video_name: str, duration: float, language: str | None,
                       entries: list[dict[str, Any]], output_language: str = "zh") -> None:
    """最终用户可见文本。段落标签跟随 output_language（英文音频 -> Visual/Speech）。"""
    from ..language import labels_for, normalize_code

    labels = labels_for(output_language)
    lang = normalize_code(output_language) or "zh"
    if lang == "zh":
        header = [
            f"视频: {video_name}",
            f"时长: {duration:.2f}s   语音语言: {language or labels['no_speech']}   输出语言: {lang}",
            f"条目: {len(entries)}",
        ]
    else:
        header = [
            f"Video: {video_name}",
            f"Duration: {duration:.2f}s   Audio language: {language or labels['no_speech']}   Output language: {lang}",
            f"Entries: {len(entries)}",
        ]
    lines = [*header, "=" * 60, ""]
    for entry in entries:
        lines.append(f"[{fmt_time(entry['start'])} - {fmt_time(entry['end'])}]")
        lines.append("")
        if entry.get("visual"):
            lines.append(f"{labels['visual']}：" if lang == "zh" else f"{labels['visual']}:")
            lines.append(entry["visual"])
        if entry.get("ocr_text"):
            sep = "：" if lang == "zh" else ": "
            lines.append(f"{labels['ocr']}{sep}{entry['ocr_text']}")
        if entry.get("speech"):
            lines.append("")
            lines.append(f"{labels['speech']}：" if lang == "zh" else f"{labels['speech']}:")
            lines.append(entry["speech"])
        lines.append("")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def write_srt(path: Path, speech_segments: list[dict[str, Any]],
              visual_events: list[dict[str, Any]] | None = None) -> str:
    """优先输出语音字幕；完全没有语音时才退化为视觉事件字幕。"""
    if speech_segments:
        items = [(s["start"], s["end"], s["text"]) for s in speech_segments if s.get("text")]
        kind = "speech"
    else:
        items = [(e["start"], e["end"], e.get("event") or e.get("description") or "")
                 for e in (visual_events or [])]
        items = [i for i in items if i[2]]
        kind = "visual_fallback"
        if items:
            logger.info("没有语音，timeline.srt 使用视觉事件作为兜底字幕")

    blocks = []
    for i, (start, end, text) in enumerate(items, start=1):
        end = max(end, start + 0.2)
        blocks.append(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{text.strip()}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(blocks))
    return kind
