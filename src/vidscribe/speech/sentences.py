"""把 whisper 的一段切成一句一行，让所有输出都做到"一句话一个时间戳"。

whisper 的 segment 边界是按解码窗口来的，一段里常常塞了好几句话。断句直接看语音：
两个词之间静音超过 `min_pause` 秒就断行，时间取 word_timestamps
（`speech.word_timestamps=true`），所以每行都是 whisper 原生精度，不是按字数估的。

为什么不按标点：拿 output 下 20 份真实结果统计了 3156 个词间隔——标点后的停顿中位数
只有 0.100s，也就是 whisper 经常在根本没停顿的地方点逗号；反过来 0.30s 以上的真实停顿
里有 166 处压根没标点。照标点切会把一句话切碎、又漏掉真正该断的地方。想按标点切的话
传 `break_punct=True`。

拿不到词级时间时才退回按标点切文本，那种行会打上 `time_estimated`。
函数是幂等的：已经一句一行的输入原样返回，所以复用断点缓存时可以放心再调一次。
"""



from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

# 句末标点（全角半角都认）。半角句点 `.` 故意不在这里：缩写（e.g. / Mr.）和序号
# 会被切碎，用户选择"有点就不换行"。英文句子照样靠逗号和 whisper 自己的段边界分开。
_END_CHARS = "。！？!?…‼⁇⁈⁉"
# 短句分隔符：逗号、顿号、分号、冒号，全角半角都认。默认也断行——用户要求"一句话一行"
_CLAUSE_CHARS = "，,、；;：:"


# 结句符后面可能还跟引号/括号，属于同一句
_TRAILING = "\"'”’）)》」』】]"
# 判断一段文字里有没有实际内容（只剩标点的碎片要并回上一句）
_CONTENT = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# 中日韩字符：这类语言没有空格分词，长度要按字数算
_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# 这些词后面的点是缩写不是句末。只用于"一行里是不是塞了好几句"的判断，
# 不影响断行主规则（主规则看停顿）。Mr. Smith 这种后接大写专名的必须靠它兜住。
_ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "mt.",
           "vs.", "etc.", "e.g.", "i.e.", "no.", "inc.", "ltd.", "co.", "approx."}



# 碎片并行的停顿上限（秒）。"说个 If 吊你一下再接下半句"这种桥段停顿常在 2~3.5s，
# 并起来才是一句话；超过这个值就不并了，否则会做出十秒长的字幕、字幕早早出现人还没开口。
MAX_MERGE_GAP = 4.0

# 切出来的句子被情绪/翻译污染的字段：父段的值套到子句上是错的，清掉让下游重算

_DROP_KEYS = ("text_translated", "translated_language", "emotion", "emotion_en",
              "emotion_confidence", "emotion_intensity", "emotion_scores")


def _has_content(text: str) -> bool:
    return bool(_CONTENT.search(text))


def _ends_sentence(token: str, break_clauses: bool = True) -> bool:

    body = token.strip().rstrip(_TRAILING)
    if not body:
        return False
    tail = body[-1]
    if break_clauses and tail in _CLAUSE_CHARS:
        return True
    return tail in _END_CHARS



def _cut_text(text: str, break_clauses: bool = True) -> list[str]:

    """没有词级时间时，纯按标点切文本。"""
    cuts = _END_CHARS + (_CLAUSE_CHARS if break_clauses else "")
    pieces: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in cuts or (buf and ch in _TRAILING and buf[:-1].rstrip()[-1:] in cuts):
            if _has_content(buf):
                pieces.append(buf)
                buf = ""
    if buf.strip():
        if pieces and not _has_content(buf):
            pieces[-1] += buf
        else:
            pieces.append(buf)
    return pieces


def _sentence_end_here(token: str, next_token: str) -> bool:
    """这个词是不是真的结了一句。

    全角 。！？ 和半角 ! ? 没有歧义，直接算。半角 `.` 有歧义（e.g. / Mr. / 3.5），
    只有当下一个词以大写字母开头时才算结句——"some. I have" 算，"e.g. apples" 不算。
    """
    body = token.strip().rstrip(_TRAILING)
    if not body:
        return False
    tail = body[-1]
    if tail in _END_CHARS:
        return True
    if tail != ".":
        return False
    if body.lower() in _ABBREV:
        return False            # Mr. Smith / e.g. Apple 这种不算结句
    nxt = next_token.strip()
    return bool(nxt) and nxt[0].isupper()



def _split_multi_sentence(groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """一行里塞了好几句话就拆开。

    语速快的时候句子之间没有停顿，光靠停顿断不开，会出现一行念完两三句。这里在
    真正的句末（见 `_sentence_end_here`）再补一刀，逗号一律不动，免得把一句话切碎。
    """
    out: list[list[dict[str, Any]]] = []
    for group in groups:
        buf: list[dict[str, Any]] = []
        for index, word in enumerate(group):
            buf.append(word)
            nxt = str(group[index + 1].get("word") or "") if index + 1 < len(group) else ""
            if nxt and _sentence_end_here(str(word.get("word") or ""), nxt):
                out.append(buf)
                buf = []
        if buf:
            out.append(buf)
    return out


def _text_word_count(text: str) -> int:
    text = text.strip()
    if _CJK.search(text):
        return len(_CJK.findall(text))     # 中文按字算
    return len([t for t in text.split() if t])


def _word_count(group: list[dict[str, Any]]) -> int:

    return _text_word_count("".join(str(w.get("word") or "") for w in group))


def _merge_fragments(groups: list[list[dict[str, Any]]],
                     max_merge_gap: float = MAX_MERGE_GAP) -> list[list[dict[str, Any]]]:
    """把"说半句就停"留下的碎片并到下一行。

    实拍里很常见：说 "If" 然后停一下再接 "this is pink,"。纯按停顿断会留下只有一两个词、
    又没有任何标点收尾的孤行。判定为碎片就并进下一行（末尾的并进上一行）。

    但停顿超过 `max_merge_gap` 秒就不并了：那种"说个 if 吊你七秒"的桥段并起来会做出一条
    十几秒的字幕，字幕早早出现、人还没开口。宁可让它单独成行，时间才对得上声音。
    """
    out: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    for group in groups:
        if pending:
            gap = float(group[0]["start"]) - float(pending[-1]["end"])
            if gap <= max_merge_gap:
                group = pending + group
            else:
                out.append(pending)      # 隔太久，不并
            pending = []
        text = "".join(str(w.get("word") or "") for w in group).strip()
        tail = text.rstrip(_TRAILING)[-1:]
        # 半角句点也算收尾（它不在 _END_CHARS 里，那是为了不按 `.` 主动断行）
        incomplete = tail not in (_END_CHARS + _CLAUSE_CHARS + ".")
        if incomplete and _word_count(group) < 4:
            pending = group          # 留着并进下一行
            continue
        out.append(group)
    if pending:
        # 最后一个碎片没有下一行可并。往前并要先看上一行是不是已经说完一句了：
        # 已经以 。！？. 收尾的话，往前并会做出"…home alone. Oh"这种一行两句，宁可让它单独成行。
        prev = "".join(str(w.get("word") or "") for w in out[-1]).strip() if out else ""
        if out and prev.rstrip(_TRAILING)[-1:] not in (_END_CHARS + "."):
            out[-1].extend(pending)
        else:
            out.append(pending)
    return out




def _group_words(words: list[dict[str, Any]], min_pause: float = 0.30,
                 break_punct: bool = False,
                 break_clauses: bool = True) -> list[list[dict[str, Any]]]:

    groups: list[list[dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        buf.append(word)
        cut = False
        if min_pause > 0 and index + 1 < len(words):
            # 直接看语音：这个词结束到下个词开始之间静了多久
            gap = float(words[index + 1]["start"]) - float(word["end"])
            if gap >= min_pause:
                cut = True
        if not cut and break_punct and _ends_sentence(str(word.get("word") or ""), break_clauses):
            cut = True
        if cut:
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)
    # 只剩标点的碎片并进上一句
    merged: list[list[dict[str, Any]]] = []
    for group in groups:
        joined = "".join(str(w.get("word") or "") for w in group)
        if merged and not _has_content(joined):
            merged[-1].extend(group)
        else:
            merged.append(group)
    # 语速快时句子之间没停顿，补一刀把"一行好几句"拆开；再把说半句就停的碎片并回去
    return _merge_fragments(_split_multi_sentence(merged))






def _child(seg: dict[str, Any], text: str, start: float, end: float,
           words: list[dict[str, Any]] | None) -> dict[str, Any]:
    child = deepcopy(seg)
    for key in _DROP_KEYS:
        child.pop(key, None)
    child["start"] = round(float(start), 3)
    child["end"] = round(float(end), 3)
    child["text"] = text
    child["original_text"] = text
    child["words"] = words if words is not None else []
    return child


def _split_one(seg: dict[str, Any], min_pause: float = 0.30,
               break_punct: bool = False,
               break_clauses: bool = True) -> list[dict[str, Any]]:

    text = str(seg.get("text") or "").strip()
    if not text:
        return [seg]

    words = [w for w in (seg.get("words") or [])
             if w.get("start") is not None and w.get("end") is not None]
    if words:
        groups = _group_words(words, min_pause, break_punct, break_clauses)
        if len(groups) <= 1:
            return [seg]
        return [_child(seg, "".join(str(w.get("word") or "") for w in group).strip(),
                       float(group[0]["start"]), float(group[-1]["end"]), group)
                for group in groups]

    # 没有词级时间戳，只能退回按标点切文本
    pieces = _cut_text(text, break_clauses)


    if len(pieces) <= 1:
        return [seg]
    # 没有词级时间，只能按字数比例摊时间；标出来免得下游误当成原生精度
    start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
    total = sum(len(p) for p in pieces) or 1
    out: list[dict[str, Any]] = []
    cursor = 0
    for piece in pieces:
        head = start + (end - start) * cursor / total
        cursor += len(piece)
        tail = start + (end - start) * cursor / total
        child = _child(seg, piece.strip(), head, tail, [])
        child["time_estimated"] = True
        out.append(child)
    return out


def _line_is_fragment(line: dict[str, Any]) -> bool:
    """这行是不是"说半句就停"的碎片（判定同 `_merge_fragments`，只是作用在成行的结果上）。"""
    text = str(line.get("text") or "").strip()
    if not text:
        return True
    tail = text.rstrip(_TRAILING)[-1:]
    incomplete = tail not in (_END_CHARS + _CLAUSE_CHARS + ".")
    return incomplete and _text_word_count(text) < 4


def _join_lines(head: dict[str, Any], tail: dict[str, Any]) -> dict[str, Any]:
    """把碎片行 head 并进 tail：文字接起来，时间取 head.start ~ tail.end。"""
    merged = deepcopy(tail)
    for key in _DROP_KEYS:      # 两行的情绪/译文不能混用，清掉让下游重算
        merged.pop(key, None)
    left = str(head.get("text") or "").strip()
    right = str(merged.get("text") or "").strip()
    sep = "" if (_CJK.search(left[-1:]) or _CJK.search(right[:1])) else " "
    text = (left + sep + right).strip()
    merged["text"] = text
    merged["original_text"] = text
    merged["start"] = round(float(head.get("start") or 0.0), 3)
    merged["words"] = list(head.get("words") or []) + list(tail.get("words") or [])
    if head.get("time_estimated") or tail.get("time_estimated"):
        merged["time_estimated"] = True
    return merged


def _merge_across_segments(lines: list[dict[str, Any]],
                          max_merge_gap: float = MAX_MERGE_GAP) -> list[dict[str, Any]]:
    """跨 whisper 段并碎片。

    `_merge_fragments` 只在一段之内work，可 whisper 经常把"If ……（停 3 秒）……
    this is pink, then you have to jump in" 切成两个 segment，碎片和下半句根本不在同一段里，
    段内那道合并够不着。这里在成行之后再扫一遍，把跨段的碎片并回它所属的那句话。
    """
    out: list[dict[str, Any]] = []
    for line in lines:
        if out and _line_is_fragment(out[-1]):
            gap = float(line.get("start") or 0.0) - float(out[-1].get("end") or 0.0)
            if gap <= max_merge_gap:
                out[-1] = _join_lines(out[-1], line)
                continue
        out.append(line)
    # 末尾的碎片没有下一行可并，贴回上一行——但上一行已经说完一句就别贴，理由同 _merge_fragments
    if len(out) > 1 and _line_is_fragment(out[-1]):
        prev = str(out[-2].get("text") or "").strip()
        gap = float(out[-1].get("start") or 0.0) - float(out[-2].get("end") or 0.0)
        if gap <= max_merge_gap and prev.rstrip(_TRAILING)[-1:] not in (_END_CHARS + "."):
            tail = out.pop()
            out[-1] = _join_lines(out[-1], tail)
            out[-1]["start"] = round(float(out[-1].get("start") or 0.0), 3)
    return out



def _split_pass(segments: list[dict[str, Any]], min_pause: float,
                break_punct: bool, break_clauses: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.extend(_split_one(seg, min_pause, break_punct, break_clauses))
    return _merge_across_segments(out)


def _shape(lines: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    return [(float(s.get("start") or 0.0), float(s.get("end") or 0.0),
             str(s.get("text") or "").strip()) for s in lines]


def split_sentences(segments: list[dict[str, Any]], min_pause: float = 0.30,
                    break_punct: bool = False,
                    break_clauses: bool = True) -> list[dict[str, Any]]:
    """一句一行并重排 id。返回新列表；没有可切的段就原样带过。

    min_pause   直接看语音：两个词之间静音超过这么多秒就断行（默认 0.30s）。设 0 关掉。
    break_punct 是否也按标点断行。默认关——实测标点后的停顿中位数只有 0.1s，
                whisper 经常在没有停顿的地方点逗号，照标点切会把一句话切碎。
    break_clauses 仅在 break_punct=True 时有意义：逗号一类是否也算断点。
    """

    out = _split_pass(segments, min_pause, break_punct, break_clauses)
    # 跑到不动点。合并过的行里带着大停顿，再切一次会从别的地方断开，一遍下来结果不稳定；
    # 这里自己先收敛，返回值就是不动点，三个入口重复调用才真正幂等。
    for _ in range(4):
        again = _split_pass(out, min_pause, break_punct, break_clauses)
        if _shape(again) == _shape(out):
            break
        out = again

    for i, seg in enumerate(out, start=1):
        seg["id"] = i
    return out


