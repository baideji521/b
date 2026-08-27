"""逐帧行格式：小模型不会在窗口内切分时间，就不让它切。

实测（MiniCPM-V 4.6, 1.3B）：
- 嵌套 JSON 它写不合法（括号错位、字段跑到对象外面），解析成功率 0；
- 管道分隔的一行一事件它能写对，但每一行的 start/end 都填成整个窗口，
  甚至出现 10.0|10.0 这种零长度段 —— 也就是它没有时间切分能力；
- 给它真实时间戳列表让它抄，它一个都抄不对，还把模板里的占位词
  （scene / visible_text）原样吐回来 —— 也就是它不会抄浮点数。

所以这里换一种分工：只让它按**帧编号**逐行描述"这一刻画面里是什么"，
时间边界完全由主进程的真实帧时间戳决定，行号即帧序号。
连续相同动作的行自动合并成一个事件段。模型只管"发生什么"，程序管"什么时候"。
"""

from __future__ import annotations

import re

from ..events import VisualEvent

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.M)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_INT = re.compile(r"\d+")

# 模型爱把模板里的字段名当内容吐回来，这些词一律当空值
_PLACEHOLDERS = {
    "people", "person", "persons", "action", "actions", "scene", "text",
    "visible_text", "on_screen_text", "description", "timestamp", "frame",
    "none", "nobody", "n_a", "na", "null", "unknown", "no_text", "no_one",
    "人物", "动作", "场景", "画面文字", "描述", "时间戳", "无",
}

_PROMPT_EN = """\
You are given {n} still frames from one video, numbered 1 to {n} in time order.

Describe each frame on its own line. Output exactly {n} lines and nothing else.
Each line: number|who is visible|what they are doing|text readable on screen

Example of the style (do not copy its content):
1|woman, man|woman holds a bag of candy|PASCALL
2|woman, man|man points at the bag|NONE

Rules:
- Start every line with its frame number, then three fields separated by |.
- Name only people and objects actually visible. Do not guess names.
- Put NONE in the last field when no text is readable.
- No JSON, no markdown, no headings, no commentary.
"""

_PROMPT_ZH = """\
下面是同一段视频里的 {n} 张画面，按时间顺序编号 1 到 {n}。

每张画面写一行，总共 {n} 行，不要输出别的内容。
每行格式：编号|画面里有谁|他们在做什么|画面上能读到的文字

风格示例（不要照抄内容）：
1|女性, 男性|女性拿着一袋糖|PASCALL
2|女性, 男性|男性指向那袋糖|NONE

要求：
- 每行开头是画面编号，后面三个字段用竖线分隔。
- 只写画面里真实出现的人和物，不要猜名字。
- 最后一个字段没有可读文字就写 NONE。
- 不要 JSON、不要 markdown、不要标题、不要多余说明。
"""

_SYSTEM = {
    "en": "You are a precise video annotator. Follow the requested line format exactly.",
    "zh": "你是严谨的视频画面标注器，严格按要求的逐行格式输出。",
}


def system_prompt(output_language: str) -> str:
    return _SYSTEM.get((output_language or "zh").lower(), _SYSTEM["en"])


def build_prompt(window_start: float, window_end: float, timestamps: list[float],
                 previous_summary: str | None = None, output_language: str = "zh") -> str:
    is_zh = (output_language or "zh").lower().startswith("zh")
    template = _PROMPT_ZH if is_zh else _PROMPT_EN
    text = template.format(n=len(timestamps))
    if previous_summary:
        prefix = "已知前文（不要重复其中已经描述过的内容）：\n" if is_zh \
            else "Context from earlier windows (do not repeat it):\n"
        text = f"{prefix}{previous_summary}\n\n{text}"
    return text


def _clean(text: str) -> str:
    text = _THINK.sub("", text or "")
    text = text.replace("</think>", " ").replace("<think>", " ")
    return _FENCE.sub("", text).strip()


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", (value or "").strip().lower()).strip("_")


def _clean_field(value: str | None) -> str:
    """去掉模型回吐的占位词，返回可用文本（可能为空串）。"""
    raw = (value or "").strip().strip("<>[]（）()").strip()
    if not raw or _norm(raw) in _PLACEHOLDERS:
        return ""
    return raw


def _split_people(value: str | None) -> list[str]:
    text = _clean_field(value)
    if not text:
        return []
    out: list[str] = []
    for part in re.split(r"[,，、/]+|\band\b|\bwith\b|\u548c", text):
        token = _norm(part)
        if token and token not in _PLACEHOLDERS:
            out.append(token)
    return out


def _slot_of(head: str, order: int, timestamps: list[float]) -> int:
    """行首数字优先当帧编号（1-based），像时间戳就对齐到最近的真实帧，都不像就按行序。"""
    n = len(timestamps)
    m = _NUM.search(head or "")
    if not m:
        return min(order, n - 1)
    value = float(m.group())
    if "." not in m.group() and 1 <= value <= n:
        return int(value) - 1
    if timestamps:
        span_ok = timestamps[0] - 0.5 <= value <= timestamps[-1] + 0.5
        if span_ok:
            return min(range(n), key=lambda i: abs(timestamps[i] - value))
    if float(value).is_integer() and 1 <= value <= n:
        return int(value) - 1
    return min(order, n - 1)


def parse_frame_lines(raw: str, timestamps: list[float], frame_indices: list[int],
                      window_start: float, window_end: float) -> list[VisualEvent]:
    """把逐帧行解析成事件段：时间边界只来自真实帧时间戳。"""
    if not timestamps:
        return []
    text = _clean(raw)
    records: list[dict] = []
    order = 0
    for line in text.splitlines():
        line = line.strip().lstrip("-*•# ").strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p != ""] if parts and parts[0] == "" else parts
        if len(parts) < 2:
            continue
        head = parts[0]
        # 表头行（"编号|人物|动作|文字"）：整行都是占位词，丢掉
        if not _NUM.search(head) and all(_norm(p) in _PLACEHOLDERS for p in parts):
            continue
        slot = _slot_of(head, order, timestamps)
        fields = parts[1:]
        while len(fields) < 3:
            fields.append("")
        people, action = fields[0], fields[1]
        # 逐字段过滤，避免把 "scene"、"visible_text" 这类回吐的占位词拼进 OCR
        visible = " ".join(x for x in (_clean_field(f) for f in fields[2:]) if x).strip()
        if not _clean_field(action) and not _clean_field(people):
            continue
        records.append({
            "slot": slot,
            "people": _split_people(people),
            "action": _clean_field(action),
            "ocr": _clean_field(visible) or None,
        })
        order += 1
    if not records:
        return []

    # 同一帧多行只保留第一条，然后按帧顺序排列
    by_slot: dict[int, dict] = {}
    for rec in records:
        by_slot.setdefault(rec["slot"], rec)
    ordered = [by_slot[k] for k in sorted(by_slot)]

    # 连续相同动作合并成一段
    groups: list[list[dict]] = []
    for rec in ordered:
        key = _norm(rec["action"])[:48]
        if groups and _norm(groups[-1][0]["action"])[:48] == key:
            groups[-1].append(rec)
        else:
            groups.append([rec])

    events: list[VisualEvent] = []
    for gi, group in enumerate(groups):
        first, last = group[0], group[-1]
        start = timestamps[first["slot"]]
        if gi + 1 < len(groups):
            end = timestamps[groups[gi + 1][0]["slot"]]
        else:
            end = max(window_end, timestamps[last["slot"]])
        if end <= start:
            end = min(window_end, start + 0.2)
        subjects: list[str] = []
        for rec in group:
            for person in rec["people"]:
                if person not in subjects:
                    subjects.append(person)
        ocr = next((rec["ocr"] for rec in group if rec["ocr"]), None)
        action = first["action"]
        events.append(VisualEvent(
            id=len(events) + 1,
            start=round(float(start), 3),
            end=round(float(end), 3),
            event=_norm(action)[:60] or "unknown",
            description=action or _norm(action),
            confidence=0.6 if len(group) > 1 else 0.5,
            importance="normal",
            timestamp_source="frame_based",   # 边界全部来自真实帧时间
            source_frames=[frame_indices[rec["slot"]] for rec in group
                           if rec["slot"] < len(frame_indices)],
            ocr_text=ocr,
            window=[round(window_start, 3), round(window_end, 3)],
            action=_norm(action)[:60] or None,
            scene=None,     # 单帧行格式不问场景，场景由窗口级摘要给
            subjects=subjects,
        ))
    return events

