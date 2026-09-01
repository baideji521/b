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


def fmt_secs(seconds: float) -> str:
    """绝对秒数，两位小数。

    合并导出（喂给大模型那份）专用：mm:ss.cc 会被模型顺手读成数字——`[02:04.27]` 变成
    204.27 秒，比视频还长。改成纯秒就没有这个转换环节了。
    """
    return f"{max(0.0, float(seconds)):.2f}"


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
                       entries: list[dict[str, Any]], output_language: str = "zh",
                       actions: list[dict[str, Any]] | None = None,
                       emotions: list[dict[str, Any]] | None = None) -> None:
    """最终用户可见文本。段落标签跟随 output_language（英文音频 -> Visual/Speech）。

    末尾附两条独立时间戳轨：动作（事件粒度归并）和表情（人脸模型 2fps 归并），
    两者都是已算好的结果重新排一遍，不额外推理。
    """
    from ..emotions import display_name  # noqa: PLC0415
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
    multi_speaker = len({s for e in entries for s in (e.get("speech_speakers") or [])}) > 1
    for entry in entries:
        lines.append(f"[{fmt_time(entry['start'])} - {fmt_time(entry['end'])}]")
        lines.append("")
        if entry.get("visual"):
            # 画面行跟画面情绪，语音行跟语音情绪，两路各自标注不混
            tag = emotion_tag(entry.get("visual_emotion_en"),
                              entry.get("visual_emotion_intensity"), output_language,
                              entry.get("visual_emotion"))
            head = f"{labels['visual']}{tag}"
            lines.append(f"{head}：" if lang == "zh" else f"{head}:")
            lines.append(entry["visual"])
            facts = " / ".join(x for x in (entry.get("action"), entry.get("scene"),
                                           ", ".join(entry.get("subjects") or [])) if x)
            if facts:
                sep = "：" if lang == "zh" else ": "
                lines.append(f"{labels['facts']}{sep}{facts}")
        if entry.get("ocr_text"):
            sep = "：" if lang == "zh" else ": "
            lines.append(f"{labels['ocr']}{sep}{entry['ocr_text']}")
        if entry.get("speech"):
            lines.append("")
            tag = emotion_tag(entry.get("speech_emotion_en"),
                              entry.get("speech_emotion_intensity"), output_language,
                              entry.get("speech_emotion"))
            # 只有真判出 2 人以上才标说话人：单人素材每行挂个"说话人1"是纯噪声
            who = speaker_tag(entry.get("speech_speakers"), output_language) if multi_speaker else ""
            head = f"{labels['speech']}{who}{tag}"
            lines.append(f"{head}：" if lang == "zh" else f"{head}:")
            lines.append(entry["speech"])
        lines.append("")
        lines.append("")
    sep = "：" if lang == "zh" else ": "
    if actions:
        lines.append("=" * 60)
        lines.append("动作轨（逐动作时间戳）" if lang == "zh" else "Action track")
        lines.append("")
        for span in actions:
            scene = f" @ {span['scene']}" if span.get("scene") else ""
            lines.append(f"[{fmt_time(span['start'])} - {fmt_time(span['end'])}]"
                         f"{sep}{span['action']}{scene}")
        lines.append("")
    if emotions:
        lines.append("=" * 60)
        lines.append("表情轨（逐表情时间戳）" if lang == "zh" else "Expression track")
        lines.append("")
        for span in emotions:
            name = display_name(span.get("emotion_en"), None, output_language) or span.get("emotion_en")
            lines.append(f"[{fmt_time(span['start'])} - {fmt_time(span['end'])}]"
                         f"{sep}{name} {span.get('intensity', 0):.2f}")
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


# ------------------------------------------------------- GUI 侧的单项/合并导出
MIN_SRT_SECONDS = 0.3
# 合并导出里语音情绪的最低置信度：低于这个分就不打标签（详见 write_merged_txt）
MIN_SPEECH_EMOTION_SCORE = 0.30
# 逐词时间轴里超过这个秒数的静音会显式标一行，免得下游跨过洞取片段边界
MIN_WORD_GAP_SECONDS = 1.5
# 表情轨里超过这个秒数的空档会显式标一行（没检到人脸，不是 neutral）
MIN_FACE_GAP_SECONDS = 1.0

# 导出文本里的固定用词：整份文件跟着内容语言走，不能中文表头配英文正文
_TXT_WORDS: dict[str, dict[str, str]] = {
    "zh": {"video": "视频", "speech_count": "语音段", "event_count": "画面事件",
           "duration": "时长",
           "merged_count": "画面事件 {events} 条，语音 {speech} 段",
           "translated": "（译文）", "ocr": "画面文字", "sep": "：",
           "speech_file": "语音", "events_file": "事件", "merged_file": "合并",
           "words_file": "逐词", "word_count": "词数",
           "words_section": "逐词时间轴（一个词一个时间戳，原文，不含译文）",
           "action_section": "动作轨（逐动作时间戳，事件粒度归并）",
           "expression_section": "表情轨（逐表情时间戳，人脸模型 2fps 归并）",
           "timeline_section": "交错时间线（画面事件 + 语音，按时间排序）",
           "sections_index": "本文件共 4 段：1 交错时间线 | 2 动作轨 | 3 表情轨 | 4 逐词时间轴",
           "authority_note": "时间基准 = SECTION 4；情绪基准 = SECTION 3（人脸模型）；"
                             "语音行括号里的情绪来自音频模型，仅作参考。",
           "timestamp_note": "所有时间戳都是从视频开头算起的绝对秒数（十进制秒，不是 mm:ss），"
                             "取值范围 0 到上面那个时长；直接照抄，不要做任何换算。",
           "legend_visual": "行格式：[起 - 止] 画面（表情 置信度）[重要度]：描述",
           "legend_ocr": "行格式：    画面文字：…（缩进行，属于上面那条画面行）",
           "legend_speech": "行格式：[起 - 止] 语音（说话人 N）（音频情绪 置信度，仅参考；"
                            "置信度低于 0.30 的直接不打）：内容",
           "legend_action": "行格式：[起 - 止]：动作 @ 场景",
           "legend_expression": "行格式：[起 - 止]：表情 强度(0-1) —— 情绪判定以本段为准；"
                                "只代表画面里最大的那张脸",
           "face_gap": "--- 中间 {gap:.2f}s 没检到人脸（不等于 neutral）---",
           "expression_none": "--- 全片没有检测到有效人脸，表情轨为空（不等于 neutral）---",
           "expression_legacy": "--- 这次分析在表情落库之前完成，数据库里没有表情轨数据；"
                                "重新分析该视频后本段才会有内容（不要当成没有表情）---",
           "legend_words": "行格式：[起 - 止] 词 —— clip.start / clip.end 一律取自本段",
           "word_gap": "--- 中间 {gap:.2f}s 没有说话 ---",
           "pieces_file": "合并素材",
           "pieces_count": "共 {count} 段素材（相邻素材之间的 1 秒纯红画面只作分界，"
                           "不属于任何一段）",
           "pieces_item": "  素材 {order}：{start} - {end}（{span:.2f}s）",
           "pieces_note": "下面按素材逐段给数据，每段内部就是一份完整的合并导出（4 个 SECTION）。"
                          "所有时间戳仍是合并视频的绝对秒数，直接照抄；"
                          "某段素材内的相对秒数 = 绝对秒数 - 该段起点。",
           "pieces_head": "素材 {order} / {count}    {start} - {end}"
                          "（{span:.2f}s，本段 0.00 对应合并视频 {start}）",
           "pieces_empty": "--- 这一段里没有任何画面事件和语音 ---",
           "translated_file": "译文"},
    "en": {"video": "Video", "speech_count": "Speech segments", "event_count": "Visual events",
           "duration": "Duration",
           "merged_count": "{events} visual events, {speech} speech segments",
           "translated": " (translated)", "ocr": "On-screen text", "sep": ": ",
           "speech_file": "speech", "events_file": "events", "merged_file": "merged",
           "words_file": "words", "word_count": "Words",
           "words_section": "Word-by-word timeline (one timestamp per word, source language)",
           "action_section": "Action track (one timestamp per action, event granularity)",
           "expression_section": "Expression track (one timestamp per expression, face model @2fps)",
           "timeline_section": "Interleaved timeline (visual events + speech, sorted by time)",
           "sections_index": "This file has 4 sections: 1 interleaved timeline | 2 action track "
                             "| 3 expression track | 4 word-by-word timeline",
           "authority_note": "Timing authority = SECTION 4. Emotion authority = SECTION 3 "
                             "(face model). The emotion on Speech lines is audio-based: "
                             "reference only.",
           "timestamp_note": "All timestamps are absolute seconds from the start of the video "
                             "(decimal seconds, NOT mm:ss), between 0 and the Duration above. "
                             "Copy them verbatim; never convert them.",
           "legend_visual": "Row: [start - end] Visual (expression score) [importance]: "
                            "description",
           "legend_ocr": "Row:     On-screen text: ... (indented, belongs to the Visual row "
                         "above)",
           "legend_speech": "Row: [start - end] Speech (speaker N) (audio emotion score, "
                            "reference only; omitted when the score is below 0.30): text",
           "legend_action": "Row: [start - end]: action @ scene",
           "legend_expression": "Row: [start - end]: expression intensity(0-1) - decisive "
                                "emotional evidence; largest face on screen only",
           "face_gap": "--- {gap:.2f}s with no face detected (this is NOT neutral) ---",
           "expression_none": "--- No valid face detected anywhere in this video; the "
                              "expression track is empty (this is NOT neutral) ---",
           "expression_legacy": "--- This analysis predates expression persistence, so the "
                                "database holds no expression track; re-analyse the video to "
                                "fill this section (do NOT read it as 'no expression') ---",
           "legend_words": "Row: [start - end] word - clip.start / clip.end must come from here",
           "word_gap": "--- {gap:.2f}s with no speech ---",
           "pieces_file": "pieces",
           "pieces_count": "{count} source pieces (the 1-second pure red frames between "
                           "neighbouring pieces are boundaries only and belong to no piece)",
           "pieces_item": "  Piece {order}: {start} - {end} ({span:.2f}s)",
           "pieces_note": "The data below is grouped per piece; each block is a complete merged "
                          "export (4 sections). Timestamps stay absolute seconds of the merged "
                          "video - copy them verbatim; a piece-relative second = absolute second "
                          "- that piece's start.",
           "pieces_head": "Piece {order} / {count}    {start} - {end}"
                          " ({span:.2f}s, 0.00 of this piece = {start} of the merged video)",
           "pieces_empty": "--- nothing (no visual event, no speech) inside this piece ---",
           "translated_file": "translated"},

}


def txt_words(language: str | None) -> dict[str, str]:
    """导出文本的用词表。非中文一律走英文，避免中英混排。"""
    from ..language import normalize_code  # noqa: PLC0415

    return _TXT_WORDS.get(normalize_code(language) or "zh", _TXT_WORDS["en"])



def normalize_srt_items(items: list[tuple[float, float, str]],
                        min_duration: float = MIN_SRT_SECONDS) -> list[tuple[float, float, str]]:
    """整理成剪映/CapCut 能直接导入的字幕序列。

    剪映对时间轴很挑：块必须按时间升序、不能重叠、不能零长度、不能空文本，
    否则整份 SRT 导入失败或只进来前几条。这里统一修掉这些问题：
    - 丢掉空文本
    - 按 start 升序
    - end 不足 min_duration 时补齐
    - 与下一条重叠时把 end 压到下一条 start 之前 1ms
    """
    cleaned: list[tuple[float, float, str]] = []
    for start, end, text in items:
        body = " ".join(str(text or "").split())
        if not body:
            continue
        s = max(0.0, float(start))
        e = max(float(end), s + min_duration)
        cleaned.append((s, e, body))
    cleaned.sort(key=lambda x: (x[0], x[1]))

    out: list[tuple[float, float, str]] = []
    for i, (s, e, body) in enumerate(cleaned):
        if i + 1 < len(cleaned):
            next_start = cleaned[i + 1][0]
            if e > next_start:
                e = max(next_start - 0.001, s + 0.001)
        if out and s < out[-1][1]:  # 起点被前一条盖住，往后挪
            s = min(out[-1][1] + 0.001, e - 0.001) if e - 0.001 > out[-1][1] else out[-1][1] + 0.001
            e = max(e, s + 0.001)
        out.append((round(s, 3), round(e, 3), body))
    return out


def write_capcut_srt(path: Path, items: list[tuple[float, float, str]]) -> int:
    """写 SRT。UTF-8 不带 BOM（剪映和其它播放器都认），块间一个空行，返回块数。"""
    blocks = []
    for i, (start, end, text) in enumerate(normalize_srt_items(items), start=1):
        blocks.append(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{text}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(blocks))
    return len(blocks)


def speech_text_of(seg: dict[str, Any], translated: bool = False) -> str:
    if translated and seg.get("text_translated"):
        return str(seg["text_translated"])
    return str(seg.get("text") or "")


def event_text_of(event: dict[str, Any], translated: bool = False) -> str:
    if translated and event.get("description_translated"):
        return str(event["description_translated"])
    return str(event.get("description") or event.get("event") or "")


def emotion_tag(emotion_en: Any, intensity: Any, language: str = "zh",
                stored: Any = None) -> str:
    """情绪后缀：`（开心 0.80）` / ` (happy 0.80)`。没判到情绪就返回空串。

    按当前文本语言现渲显示名，所以切到译文视图导出时情绪也跟着变；老结果没有
    英文标签时从存下来的显示名反查（认中英两种写法）。
    括号跟着语言走，别在英文文件里插全角括号。
    """
    from ..emotions import display_name  # noqa: PLC0415
    from ..language import normalize_code  # noqa: PLC0415

    name = display_name(emotion_en, stored, language)
    if not name:
        return ""
    body = str(name)
    if isinstance(intensity, (int, float)):
        body = f"{body} {float(intensity):.2f}"
    return f"（{body}）" if (normalize_code(language) or "zh") == "zh" else f" ({body})"


def multi_speaker(segments: list[dict[str, Any]]) -> bool:
    """这批句子里是不是真的判出了 2 个人以上。

    只有 2 人以上才值得在每行标说话人——单人素材每行挂个"说话人1"是纯噪声。
    """
    return len({seg.get("speaker") for seg in segments if seg.get("speaker")}) > 1


def speaker_tag(speakers: Any, language: str = "zh") -> str:
    """说话人后缀：`（说话人2）` / ` (speaker 2)`。没做声纹或判不出时返回空串。

    传单个编号或一串编号都行——一条 timeline 条目可能挂着两个人的对话。
    只有一个人说话的素材也会标 `说话人1`：这时候它没有区分意义，
    所以调用方（比如只有 1 人的整片）可以自己选择不标。
    """
    from ..language import normalize_code  # noqa: PLC0415

    if isinstance(speakers, (int, float)):
        ids = [int(speakers)]
    else:
        ids = [int(s) for s in (speakers or [])]
    if not ids:
        return ""
    zh = (normalize_code(language) or "zh") == "zh"
    if zh:
        return f"（说话人{'、'.join(str(i) for i in ids)}）"
    word = "speaker" if len(ids) == 1 else "speakers"
    return f" ({word} {', '.join(str(i) for i in ids)})"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def words_of(segments: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """把句级结果摊成逐词：一个词一条，时间用 whisper 的 word_timestamps。

    只出原文——逐词没有译文这回事（一个个词单独翻译出来是词表，不是句子）。
    拿不到词级时间的段（`time_estimated` 那类）整段出一条，不按字数编时间。
    """
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        words = [w for w in (seg.get("words") or [])
                 if w.get("start") is not None and w.get("end") is not None]
        if not words:
            body = str(seg.get("text") or "").strip()
            if body:
                out.append((float(seg.get("start") or 0.0), float(seg.get("end") or 0.0), body))
            continue
        for word in words:
            body = str(word.get("word") or "").strip()
            if body:
                out.append((float(word["start"]), float(word["end"]), body))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def write_words_txt(path: Path, video_name: str, segments: list[dict[str, Any]],
                    language: str = "zh") -> int:
    """逐词文本：一行一个词，带自己的时间区间。"""
    w = txt_words(language)
    items = words_of(segments)
    lines = [f"{w['video']}{w['sep']}{video_name}",
             f"{w['word_count']}{w['sep']}{len(items)}",
             "=" * 60, ""]
    for start, end, body in items:
        lines.append(f"[{fmt_time(start)} - {fmt_time(end)}] {body}")
    _write_lines(path, lines)
    return len(items)


def write_speech_txt(path: Path, video_name: str, segments: list[dict[str, Any]],

                     translated: bool = False, language: str = "zh") -> int:
    """只导出语音文本。表头用词跟着 language 走（英文内容 -> 英文表头）。"""
    w = txt_words(language)
    multi = multi_speaker(segments)
    lines = [f"{w['video']}{w['sep']}{video_name}",
             f"{w['speech_count']}{w['sep']}{len(segments)}" + (w["translated"] if translated else ""),
             "=" * 60, ""]
    count = 0
    for seg in segments:
        text = speech_text_of(seg, translated).strip()
        if not text:
            continue
        who = speaker_tag(seg.get("speaker"), language) if multi else ""
        lines.append(f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}]{who} {text}")
        count += 1
    _write_lines(path, lines)
    return count


def write_events_txt(path: Path, video_name: str, events: list[dict[str, Any]],
                     translated: bool = False, language: str = "zh") -> int:
    """只导出画面事件文本。"""
    w = txt_words(language)
    lines = [f"{w['video']}{w['sep']}{video_name}",
             f"{w['event_count']}{w['sep']}{len(events)}" + (w["translated"] if translated else ""),
             "=" * 60, ""]
    count = 0
    for ev in events:
        text = event_text_of(ev, translated).strip()
        if not text:
            continue
        head = f"[{fmt_time(ev['start'])} - {fmt_time(ev['end'])}]"
        tag = ev.get("event") or ""
        importance = ev.get("importance") or ""
        lines.append(f"{head} ({importance}) {tag}".rstrip())
        lines.append(f"    {text}")
        if ev.get("ocr_text"):
            lines.append(f"    {w['ocr']}{w['sep']}{ev['ocr_text']}")
        lines.append("")
        count += 1
    _write_lines(path, lines)
    return count


def merged_lines(video_name: str, segments: list[dict[str, Any]],
                 events: list[dict[str, Any]], translated: bool = False,
                 language: str = "zh",
                 actions: list[dict[str, Any]] | None = None,
                 emotions: list[dict[str, Any]] | None = None,
                 duration: float = 0.0,
                 expression_state: str = "ok") -> tuple[list[str], int]:
    """拼出合并导出的正文，返回 (行列表, 时间线条数)。

    写文件走 write_merged_txt；直接要文本的（比如把这份喂给 AI）用这个函数，
    免得为了拿字符串先落一个临时文件。

    expression_state 只在 emotions 为空时起作用（见 db.schema.EXPRESSION_STATES）：
    "no_face" / "legacy_missing" 都会照常输出 SECTION 3 表头，再写一行说明为什么是空的。
    SECTION 3 既不许静默消失，也不许拿假数据填。
    """
    from ..emotions import display_name  # noqa: PLC0415
    from ..language import labels_for  # noqa: PLC0415

    w = txt_words(language)
    labels = labels_for(language)
    multi = multi_speaker(segments)
    rows: list[tuple[float, float, str, str, list[str]]] = []
    for ev in events:
        text = event_text_of(ev, translated).strip()
        if text:
            kind = labels["visual"] + emotion_tag(ev.get("emotion_en"),
                                                 ev.get("emotion_intensity"), language,
                                                 ev.get("emotion"))
            # importance 是判定用的既有标签（摔倒/碰撞/场景剧变 -> high/critical），带出来省得下游重推
            if ev.get("importance"):
                kind += f" [{ev['importance']}]"
            extra = [f"    {labels['ocr']}{w['sep']}{ev['ocr_text']}"] if ev.get("ocr_text") else []
            rows.append((float(ev["start"]), float(ev["end"]), kind, text, extra))
    for seg in segments:
        text = speech_text_of(seg, translated).strip()
        if text:
            who = speaker_tag(seg.get("speaker"), language) if multi else ""
            # 置信度太低的音频情绪不打出来：emotion2vec 逐句判，低分那批基本是噪声
            # （"Yeah, it's green" 判 sad 0.99 这种错也有，但 0.0x 的更是纯干扰），
            # 而高光判定本来就以表情轨为准，少给模型一堆假证据。
            score = seg.get("emotion_intensity")
            weak = isinstance(score, (int, float)) and float(score) < MIN_SPEECH_EMOTION_SCORE
            mood = "" if weak else emotion_tag(seg.get("emotion_en"), score, language,
                                               seg.get("emotion"))
            rows.append((float(seg["start"]), float(seg["end"]),
                         labels["speech"] + who + mood, text, []))
    rows.sort(key=lambda r: (r[0], r[2]))

    lines = [
        f"{w['video']}{w['sep']}{video_name}",
    ]
    if duration:
        lines.append(f"{w['duration']}{w['sep']}{duration:.2f}s")
    lines += [
        w["merged_count"].format(events=len(events), speech=len(segments))
        + (w["translated"] if translated else ""),
        # 表头先把四段结构和"谁说了算"讲清楚：下游的高光筛选提示词按 SECTION 编号引用本文件
        w["sections_index"],
        w["authority_note"],
        w["timestamp_note"],
        "",
        "=" * 60,
        f"SECTION 1 - {w['timeline_section']}",
        w["legend_visual"],
        w["legend_ocr"],
        w["legend_speech"],
        "=" * 60,
        "",
    ]
    for start, end, kind, text, extra in rows:
        lines.append(f"[{fmt_secs(start)} - {fmt_secs(end)}] {kind}{w['sep']}{text}")
        lines += extra

    # 两条独立时间戳轨：动作（事件粒度归并）和表情（人脸模型 2fps 归并）。
    # actions 没传就从事件里现算；表情段只能由调用方给（来自 visual meta 的 face.segments）。
    from .engine import action_track  # noqa: PLC0415

    spans = actions if actions is not None else action_track(events)
    if spans:
        lines += ["", "=" * 60, f"SECTION 2 - {w['action_section']}",
                  w["legend_action"], "=" * 60, ""]
        for span in spans:
            scene = f" @ {span['scene']}" if span.get("scene") else ""
            lines.append(f"[{fmt_secs(span['start'])} - {fmt_secs(span['end'])}]"
                         f"{w['sep']}{span['action']}{scene}")
    if emotions:
        lines += ["", "=" * 60, f"SECTION 3 - {w['expression_section']}",
                  w["legend_expression"], "=" * 60, ""]
        prev_face_end: float | None = None
        for span in emotions:
            # 表情轨的空档是"没检到人脸"，不是情绪变 neutral。不标出来，
            # 下游只能靠相邻两段的时间差自己猜，通常猜成中性。
            if prev_face_end is not None and float(span["start"]) - prev_face_end > MIN_FACE_GAP_SECONDS:
                lines.append(w["face_gap"].format(gap=float(span["start"]) - prev_face_end))
            name = display_name(span.get("emotion_en"), None, language) or span.get("emotion_en")
            lines.append(f"[{fmt_secs(span['start'])} - {fmt_secs(span['end'])}]"
                         f"{w['sep']}{name} {span.get('intensity', 0):.2f}")
            prev_face_end = float(span["end"])
    elif expression_state in ("no_face", "legacy_missing"):
        # 没有表情段也要把 SECTION 3 摆出来，并说清是"真没脸"还是"库里没这份数据"。
        # 静默省掉这一段，下游会当成"这个视频没有情绪信息"；编一条假的更糟。
        lines += ["", "=" * 60, f"SECTION 3 - {w['expression_section']}",
                  w["legend_expression"], "=" * 60, "",
                  w["expression_none"] if expression_state == "no_face"
                  else w["expression_legacy"]]

    # 末尾附一段逐词时间轴：一个词一个时间戳。说明文字跟着内容语言走（英文内容出英文表头），
    # 逐词只出原文——逐词翻译是词表不是句子。
    items = words_of(segments)
    if items:
        lines += ["", "=" * 60, f"SECTION 4 - {w['words_section']}",
                  f"{w['word_count']}{w['sep']}{len(items)}",
                  w["legend_words"], "=" * 60, ""]
        prev_end: float | None = None
        for start, end, body in items:
            # 词与词之间的静音显式标出来：ASR 的对齐会在一句话中间留洞
            # （实测 "If" 105.50 -> "this" 114.06，中间 8.5s），
            # 下游按"第一个词的 start 到最后一个词的 end"取边界就会把静音包进片段。
            if prev_end is not None and start - prev_end > MIN_WORD_GAP_SECONDS:
                lines.append(w["word_gap"].format(gap=start - prev_end))
            lines.append(f"[{fmt_secs(start)} - {fmt_secs(end)}] {body}")
            prev_end = end
    return lines, len(rows)


def write_merged_txt(path: Path, video_name: str, segments: list[dict[str, Any]],
                     events: list[dict[str, Any]], translated: bool = False,
                     language: str = "zh",
                     actions: list[dict[str, Any]] | None = None,
                     emotions: list[dict[str, Any]] | None = None,
                     duration: float = 0.0,
                     expression_state: str = "ok") -> int:
    """合并导出：按时间把画面事件和语音段穿插在一条时间线上。

    每行带各自来源的情绪：画面行跟画面情绪，语音行跟语音情绪，两者不混。
    末尾依次附动作轨、表情轨、逐词时间轴，供高光筛选时按时间对齐三路证据。
    """
    lines, count = merged_lines(video_name, segments, events, translated, language,
                               actions, emotions, duration, expression_state)
    _write_lines(path, lines)
    return count


# ============================================================ 按素材分段的合并导出
def _overlaps(item: dict[str, Any], start: float, end: float) -> bool:
    """这条数据跟 [start, end) 有没有交集（端点相碰不算）。"""
    try:
        a = float(item["start"])
        b = float(item["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return b > start and a < end


def slice_docs(start: float, end: float,
               segments: list[dict[str, Any]],
               events: list[dict[str, Any]],
               actions: list[dict[str, Any]] | None = None,
               emotions: list[dict[str, Any]] | None = None,
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                          list[dict[str, Any]], list[dict[str, Any]]]:
    """把四路数据裁到 [start, end) 这一段，时间戳保持原样（绝对秒）。

    只按时间取交集，不改任何数值：下游拿到的时间戳仍然是合并视频里的绝对秒数，
    照抄就能定位。语音段里的 words 也一并按时间过滤，免得 SECTION 4 串到别的素材。
    """
    kept_segments: list[dict[str, Any]] = []
    for seg in segments:
        if not _overlaps(seg, start, end):
            continue
        item = dict(seg)
        words = seg.get("words")
        if isinstance(words, list):
            item["words"] = [dict(word) for word in words if _overlaps(word, start, end)]
        kept_segments.append(item)
    kept_events = [dict(ev) for ev in events if _overlaps(ev, start, end)]
    kept_actions = ([dict(span) for span in actions if _overlaps(span, start, end)]
                    if actions is not None else [])
    kept_emotions = ([dict(span) for span in emotions if _overlaps(span, start, end)]
                     if emotions is not None else [])
    return kept_segments, kept_events, kept_actions, kept_emotions


def grouped_merged_lines(video_name: str, spans: list[tuple[float, float]],
                         segments: list[dict[str, Any]],
                         events: list[dict[str, Any]], translated: bool = False,
                         language: str = "zh",
                         actions: list[dict[str, Any]] | None = None,
                         emotions: list[dict[str, Any]] | None = None,
                         duration: float = 0.0,
                         expression_state: str = "ok") -> tuple[list[str], int]:
    """按素材分段的合并导出，返回 (行列表, 时间线总条数)。

    内容跟 write_merged_txt 完全一致，只是按 spans（红屏分界切出来的素材区间）
    分组：先给一份素材清单，然后每段素材一份完整的四段式合并导出。
    每段的时间戳都还是合并视频的绝对秒数，块头会写明「本段 0.00 对应哪个绝对秒」。
    """
    from .engine import action_track  # noqa: PLC0415

    w = txt_words(language)
    # 动作轨先整片算一次再切：按素材切完事件再算，跨段归并的动作会被算成两条。
    all_actions = actions if actions is not None else action_track(events)
    count = len(spans)
    lines = [f"{w['video']}{w['sep']}{video_name}"]
    if duration:
        lines.append(f"{w['duration']}{w['sep']}{duration:.2f}s")
    lines.append(w["pieces_count"].format(count=count))
    for order, (start, end) in enumerate(spans, 1):
        lines.append(w["pieces_item"].format(order=order, start=fmt_secs(start),
                                            end=fmt_secs(end), span=end - start))
    lines += [w["pieces_note"], ""]

    total = 0
    for order, (start, end) in enumerate(spans, 1):
        part_segments, part_events, part_actions, part_emotions = slice_docs(
            start, end, segments, events, all_actions, emotions)
        lines += ["#" * 60,
                  w["pieces_head"].format(order=order, count=count, start=fmt_secs(start),
                                          end=fmt_secs(end), span=end - start),
                  "#" * 60, ""]
        if not part_segments and not part_events:
            lines += [w["pieces_empty"], ""]
            continue
        body, rows = merged_lines(video_name, part_segments, part_events, translated, language,
                                  part_actions, part_emotions, 0.0, expression_state)
        lines += body
        lines.append("")
        total += rows
    return lines, total


def write_grouped_merged_txt(path: Path, video_name: str, spans: list[tuple[float, float]],
                             segments: list[dict[str, Any]],
                             events: list[dict[str, Any]], translated: bool = False,
                             language: str = "zh",
                             actions: list[dict[str, Any]] | None = None,
                             emotions: list[dict[str, Any]] | None = None,
                             duration: float = 0.0,
                             expression_state: str = "ok") -> int:
    """把按素材分段的合并导出写成文件，返回时间线总条数。"""
    lines, count = grouped_merged_lines(video_name, spans, segments, events, translated,
                                        language, actions, emotions, duration, expression_state)
    _write_lines(path, lines)
    return count



# ============================================================ 从数据库重建剧本
def export_events(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """时间轴条目 -> 导出用的画面事件（只有画面条目算事件）。

    GUI 那条路（内存里的 timeline）和数据库重建那条路共用这一份映射，
    否则两边字段一旦对不上，同一个视频导出的剧本就会不一样。
    """
    out = []
    for e in entries:
        if not e.get("visual"):
            continue
        out.append({
            "start": e["start"], "end": e["end"],
            "description": e.get("visual"),
            "description_translated": e.get("visual_translated"),
            "event": "", "importance": e.get("importance") or "",
            "ocr_text": e.get("ocr_text"),
            # 结构化事实：动作轨要靠 action 归并（老结果里 timeline.json 没有轨时的兜底）
            "action": e.get("action"),
            "scene": e.get("scene"),
            "subjects": e.get("subjects") or [],
            # 画面事件只带画面情绪，语音情绪由语音段自己带，导出时不会串行
            "emotion": e.get("visual_emotion"),
            "emotion_en": e.get("visual_emotion_en"),
            "emotion_intensity": e.get("visual_emotion_intensity"),
        })
    return out


def _rebuild(cls: Any, data: dict[str, Any], **extra: Any) -> Any:
    """按 dataclass 现有字段过滤后再构造：老库里的行可能多字段也可能少字段。"""
    from dataclasses import fields  # noqa: PLC0415

    names = {f.name for f in fields(cls)} - set(extra)
    return cls(**{k: v for k, v in data.items() if k in names}, **extra)


def script_lines(payload: dict[str, Any], *,
                 translated: bool = False,
                 language: str | None = None,
                 pieces: list[tuple[float, float]] | None = None) -> tuple[list[str], int]:
    """把 `db.repo.script_inputs()` 取出的库数据渲染成完整剧本，返回 (行列表, 时间线条数)。

    数据库是权威来源：这里一个文件都不读（不碰 output/timeline.json，也不碰 cache/visual.json）。
    时间线仍旧由 build_timeline / filter_timeline 合并、动作轨仍旧由 action_track 归并、
    正文仍旧由 merged_lines 排版——三处都是原来那份实现，这里只负责把库里的形状喂进去。

    pieces 给了（这个视频是拼起来的合并视频，红屏分界见 timeline.red_split）就按素材分段排版，
    正文内容不变，只是多一层归类。

    过滤参数一律用当次分析存下来的 render_config：用户后来改了 GUI 配置，
    同一个视频重新生成的剧本也必须和当初逐行一致。老记录（v5 之前跑的）没有存过
    render_config / output_language，这时才退回默认值和调用方给的 language。
    """
    from ..events import SpeechEvent, SpeechWord, VisualEvent  # noqa: PLC0415
    from .engine import action_track, build_timeline, filter_timeline  # noqa: PLC0415

    segments = payload.get("segments") or []
    if translated and not any(s.get("text_translated") for s in segments):
        # 译文没落库就明确报错，绝不拿原文冒充译文
        raise ValueError("译文未落库，请使用内存结果导出或重新翻译。")

    cfg = payload.get("render_config") or {}
    visual = [_rebuild(VisualEvent, e) for e in (payload.get("events") or [])]
    speech = [_rebuild(SpeechEvent, s,
                       words=[_rebuild(SpeechWord, w) for w in (s.get("words") or [])])
              for s in segments]
    entries = build_timeline(visual, speech,
                             min_overlap=float(cfg.get("min_overlap_seconds", 0.2)))
    filtered = filter_timeline(entries,
                               importance=str(cfg.get("importance_filter", "low")),
                               min_confidence=float(cfg.get("confidence_filter", 0.0)))
    video_name = str(payload.get("video_name") or "")
    events = export_events(filtered)
    language_code = str(payload.get("output_language") or language or "zh")
    common = dict(actions=action_track(visual),
                  emotions=payload.get("emotions") or [],
                  duration=float(payload.get("duration") or 0.0),
                  expression_state=str(payload.get("expression_state") or "ok"))
    if pieces:
        return grouped_merged_lines(video_name, list(pieces), segments, events,
                                    translated, language_code, **common)
    return merged_lines(video_name, segments, events, translated, language_code, **common)


def write_script_txt(path: Path, payload: dict[str, Any], *,
                     translated: bool = False,
                     language: str | None = None,
                     pieces: list[tuple[float, float]] | None = None) -> int:
    """把库里的分析结果直接写成一份完整剧本 TXT，返回时间线条数。"""
    lines, count = script_lines(payload, translated=translated, language=language,
                                pieces=pieces)
    _write_lines(path, lines)
    return count



