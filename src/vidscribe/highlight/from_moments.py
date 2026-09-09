"""结果清单 → 高光 JSON：AI 只指两个时间点，区间和挂字落位由程序算。

为什么有这一层：

AI 擅长回答「这条视频里哪些时刻有看点」，不擅长「精确到词的区间边界」。
实测下来它反复犯同一批错：起点切进被静音打断的句子、把结果甩在区间外、
`Speech text` 翻进区间外的半句、挂字全堆在片尾同一秒。

这些恰好是程序最擅长的 —— 逐词时间戳就在库里。

所以分工改成：

    AI  →  一行一个结果，只给 `from`（这件事从哪开始）和 `at`（结果在哪）
           两个点都允许指不准，外加文字内容（type / reason / 挂字文案 / 场景动作）

    程序 →  `plan_from_span` 把两个点吸到句边界，算出 start / end
            挂字时间按结果时刻自动排开
            `Speech text` 按词轨从区间里原样拼出来

产出的是**标准老协议**形状（`clip.start/end/duration` + `overlays` + `Trimclip`），
`ai_protocol._from_legacy` 照旧把它升级成新形状，渲染、提取、血缘一律走原路，
协议和下游一个字都不用改。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .. import ai_protocol
from .clip_engine import Segment, plan_from_span, voiced_between

#: 三个挂字之间至少错开多久（同时弹出会互相打架）
OVERLAY_STEP = 0.50
#: 挂字不许正好压在区间末尾：留这么一点，免得只闪一帧
OVERLAY_MARGIN = 0.10

#: 聊天界面复制出来的噪声：``` 围栏、行内的 `[cite: 1, 2]` 引用标记
_FENCE = re.compile(r"^\s*```.*$", re.MULTILINE)
_CITE = re.compile(r"\[cite[^\]]*\]")


def rows_from_text(text: str) -> list[dict[str, Any]]:
    """AI 回复原文 → 结果清单的行。

    先把聊天界面带出来的噪声抹掉再拆：``` 围栏和行内 `[cite: 1, 2]` 标记。
    这一步不能省 —— JSONL 是按行解的，一行末尾多个 `[cite: 1]` 那行就整行废掉，
    干净的行照旧进来、脏行悄悄消失，比整份解不开更难发现。

    拆分本身复用协议那一套（整段 JSON / JSONL / 并排 `{...}`），但**不做协议升级**：
    清单行里没有 `clip`，升级只会把它判成无效。
    """
    clean = _CITE.sub("", _FENCE.sub("", text or ""))
    return ai_protocol.split_objects(clean)



def video_of_row(row: dict[str, Any], fallback: str = "") -> str:
    """这一行清单是哪个视频的：行里写了就用行里的，没写才用兜底。

    清单一行只有两个时间点，区间得拿这个视频的逐词时间戳算 —— 所以「是哪个视频」
    是这一行**最要紧**的一个字段，不是可选装饰。提示词要求把剧本第一行那个文件名
    照抄进 `video`：这样一份清单丢到哪儿（界面粘贴、命令行导入、隔几天再导）都还认得出
    自己属于谁，不用靠界面上刚好选中了哪个视频去猜。

    只取文件名（AI 可能抄成整个路径），路径怎么找源视频是 `resolve_video` 的事。
    """
    from pathlib import Path  # noqa: PLC0415 - 只为取个文件名，不值得放模块头

    name = str(row.get("video") or "").strip().strip('"').replace("\\", "/")
    return Path(name).name if name else fallback


def overlay_times(start: float, end: float, at: float) -> list[float]:
    """三个挂字的**绝对时间**，返回顺序是 `[word, emoji, comment]`。

    `word` 钉在结果时刻 —— 它是冲击字，就该跟着结果一起出现。

    `emoji` 和 `comment` 依次往**前**错开。这个方向是有意的：`comment` 是一句话，
    要读的，出现得早一点才有阅读时间；而且三个挂字分开出现，视觉上不打架。

    区间太短时间隔自动缩小，三个时间一律不越出 `[start, end]`。
    """
    span = max(0.0, end - start)
    step = OVERLAY_STEP if span >= 3 * OVERLAY_STEP else max(0.0, span / 3.0)
    word = min(max(at, start), max(start, end - OVERLAY_MARGIN))
    emoji = max(word - step, start)
    comment = max(word - 2 * step, start)
    return [round(value, 2) for value in (word, emoji, comment)]


def speech_between(segments: Sequence[Segment], start: float, end: float) -> str:
    """区间里**实际说出**的原文，按时间顺序拼起来。

    逐词优先：只把落在 `[start, end]` 内的词收进来，所以不可能带出区间外的半句。
    某句没有逐词就退回整句文本，但**必须整句都在区间里**才算 —— 宁可少一句，
    也不把区间外的话拼进来。

    拉丁字母的语言词之间补空格（不补的话 `If this is` 会连成 `Ifthisis`，
    翻译那一步就废了）；中日韩这类不用空格的原样拼。
    句与句之间用 " - " 分隔，和 PRM 对 `Speech text` 的要求一致。
    """
    parts: list[str] = []
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        if seg.words:
            inside = [w.text.strip() for w in seg.words if w.start >= start - 0.001
                      and w.end <= end + 0.001]
            inside = [one for one in inside if one]
            glue = " " if any(any("a" <= ch.lower() <= "z" for ch in one)
                              for one in inside) else ""
            text = glue.join(inside).strip()
        elif seg.start >= start - 0.001 and seg.end <= end + 0.001:
            text = seg.text.strip()
        else:
            text = ""                  # 只蹭到一半又没有逐词：不猜，整句丢掉
        if text:
            parts.append(text)
    return " - ".join(parts)



#: AI 打的主观维度：每维 0-5，**互不合成**
SCORE_KEYS = ("surprise", "standalone", "emotion_power", "caption")


def scores_of(row: dict[str, Any]) -> dict[str, int]:
    """清单行里的主观评分，缺的维度不补默认值。

    四维各管一件事，谁都替不了谁：

        surprise       结果有多意外（爆点强度）
        standalone     不看前文能不能懂 —— 混剪的硬门槛，低了就只能当过场
        emotion_power  反应有多大（表情 / 语气 / 动作的幅度）
        caption        挂字文案潜力（有没有话可写）

    **不做加权合成**：合成分一旦算出来，第二轮混剪就只能按它排序，
    「要爆点」和「要凑时长」这两种需求会被同一个数字压平。
    每维独立存着，混剪那一关自己按当次目标加权。

    **四维全部由 AI 打，程序一个都不代算。** 程序能算的客观数字（发声时长、
    静音、间隔）摆在剧本里给 AI 当依据看，不另外写成产物字段 ——
    同一个数在两处各算一遍，早晚不同步。

    兼容老写法：只给了 `score`（0-100）就原样留在 `clip.score` 里，不硬换算成四维。
    """
    out: dict[str, int] = {}
    for key in SCORE_KEYS:
        value = ai_protocol.num(row.get(key))
        if value is not None:
            out[key] = max(0, min(5, int(round(value))))
    return out



def span_without_speech(segments: Sequence[Segment], frm: float, at: float, *,
                        min_sec: float, duration: float | None = None
                        ) -> tuple[tuple[float, float], list[str]]:
    """几乎没语音的视频怎么定区间：不吸句边界，从 `result_at` 反推。

    * 终点 = `result_at`。要是有句话正好压在这个点上（刚开口、或者说到一半），
      就延到那句话说完 —— 区间里的语音原文必须留住：入库时它是方案备注，
      后面翻译填 `Speech text` 也全靠它。
    * 起点 = `setup_at`；离终点不足 `min_sec` 就从终点继续往前反推，最多退到 0。
    * 知道视频时长就不让终点越过片子结尾。

    只给「整段几乎没人说话」的视频用（滤镜挑战那类，全片语音零点几秒）。有话说的
    视频照旧走 `plan_from_span` 吸句边界 —— 那条路的成品对齐句子，这条不对齐。
    """
    start, end = (min(frm, at), max(frm, at))
    notes: list[str] = []
    limit = float(duration) if duration and duration > 0 else None
    for one in segments:
        if float(one.start) <= end + 0.01 and float(one.end) > end:
            end = float(one.end)
            notes.append(f"结果点上压着一句话，终点延到它说完 {end:.2f}s，语音原文留住")
            break
    if limit is not None:
        end = min(end, limit)
    if end - start < min_sec:
        start = max(0.0, end - min_sec)
        notes.append(f"两点之间不足 {min_sec:.2f}s，从终点往前反推到 {start:.2f}s")
    return (start, end), notes


def payload_from_moment(segments: Sequence[Segment], row: dict[str, Any], *,
                        video_name: str, min_sec: float = 3.0,
                        max_sec: float = 20.0,
                        max_voiced: float = 12.0,
                        duration: float | None = None
                        ) -> tuple[dict[str, Any], str, list[str]] | None:
    """一行结果清单 → `(高光 JSON, 区间内原文, 说明)`；这一行用不了就返回 None。

    `row` 认这些键（缺的用空值兜底，只有两个时间点是必须的）：

        video            —— 这一条是哪个视频的（照抄剧本第一行那个文件名）。
                            给了就以它为准，`video_name` 只当兜底 —— 一份清单里混着
                            两个视频的行也不会串（谁的行进谁的方案）
        setup_at         —— 这件事从哪开始（条件 / 铺垫 / 动作起手）
        result_at        —— 结果或反应发生在哪一刻
        surprise / standalone / emotion_power / caption  —— 四维评分，各 0-5
        score            —— 老写法的 0-100 总分（给了就留着）
        type / reason    —— 中文类型和原因
        word             —— 挂字冲击字，emoji 就写在里面（`WHAT?! 🐶`）
        emoji            —— 老清单单独给 emoji 的写法，给了非空就单独成一条轨
        comment / evaluation  —— 挂字长文案和中文评价
        scene / action   —— Trimclip 的场景和动作描述

    两个时间点也认早期的 `from` / `at` 写法，新名字优先。

    返回的 `Speech text` 是**空字符串**，区间内原文单独返回 —— 那是源语言，
    要翻译成中文才能填进去，翻译不是这一层的事。
    """
    frm = ai_protocol.num(row.get("setup_at"))
    if frm is None:
        frm = ai_protocol.num(row.get("from"))
    at = ai_protocol.num(row.get("result_at"))
    if at is None:
        at = ai_protocol.num(row.get("at"))
    if frm is None or at is None:
        return None
    # 整段几乎没人说话的视频：句边界这条路必然算不出够长的区间（全片语音可能只有
    # 零点几秒，吸完还是零点几秒）。这时候改成从 result_at 反推，见
    # `span_without_speech`。有话说的视频一个字都不变，照旧吸句边界。
    quiet = sum(max(0.0, float(one.end) - float(one.start)) for one in segments)
    plan = plan_from_span(segments, frm, at, min_sec=min_sec, max_sec=max_sec,
                          max_voiced=max_voiced)
    notes: list[str] = list(plan[2]) if plan is not None else []
    span = (plan[0], plan[1]) if plan is not None else None
    if (span is None or span[1] - span[0] < min_sec) and quiet < min_sec:
        span, extra = span_without_speech(segments, frm, at, min_sec=min_sec,
                                          duration=duration)
        notes = [f"整段语音只有 {quiet:.2f}s，不吸句边界，从 result_at 反推", *extra]
    if span is None:
        return None
    start, end = span
    if end - start < min_sec:
        notes.append(f"算出来只有 {end - start:.2f}s，不足 {min_sec:.2f}s，这一条不要")
        return None
    # 考核有效时长（刨掉静音），不是跨度 —— 静音可以剪掉，不该占时长配额
    voiced = voiced_between(segments, start, end)
    if voiced > max_voiced:
        notes.append(f"有效时长 {voiced:.2f}s，超过 {max_voiced:.2f}s，这一条不要")
        return None
    if end - start > max_sec:
        notes.append(f"跨度 {end - start:.2f}s，超过硬上限 {max_sec:.2f}s，这一条不要")
        return None

    marks = overlay_times(start, end, at)
    score = ai_protocol.num(row.get("score"))
    # 挂字轨：`word` 里 emoji 已经跟冲击字写在一起（`WHAT?! 🐶`），所以正常只有两条轨。
    # 空文本的轨一条都不造 —— 造出来下游会当成"这里有个挂字，只是没词"。
    # 老清单把 emoji 单独成字段的写法照旧认：给了非空就还原成第三条轨。
    overlays: dict[str, Any] = {"evaluation": str(row.get("evaluation") or "")}
    for key, moment in (("word", marks[0]), ("emoji", marks[1]), ("comment", marks[2])):
        text = str(row.get(key) or "").strip()
        if text:
            overlays[key] = {"time": moment, "text": text, "kind": key}
    clip: dict[str, Any] = {
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "scores": scores_of(row),
        "type": str(row.get("type") or ""),
        "reason": str(row.get("reason") or ""),
        "overlays": overlays,
        "Trimclip": {
            "Scene": str(row.get("scene") or ""),
            "Action": str(row.get("action") or ""),
            "Speech text": "",
        },
    }
    # 老写法的总分：给了才留。没给就不要塞个 0 进去 ——
    # 四维评分时代的 `score: 0` 会被下游当成"这条被判 0 分"，而不是"没打这个分"
    if score is not None:
        clip["score"] = int(score)
    payload = {"video": video_of_row(row, video_name), "clip": clip}
    return payload, speech_between(segments, start, end), notes


def payloads_from_rows(segments: Sequence[Segment], rows: Sequence[dict[str, Any]], *,
                       video_name: str, min_sec: float = 3.0,
                       max_sec: float = 20.0, max_voiced: float = 12.0,
                       duration: float | None = None
                       ) -> tuple[list[tuple[dict[str, Any], str, dict[str, Any]]], list[str]]:
    """整份清单 → `([(高光 JSON, 区间内原文, 清单原行), ...], 日志)`。

    原文跟着 payload 一起出来而不是只写进日志：入库那一步要把它记成方案备注，
    翻译环节回头就靠它填 `Speech text`，从日志里再抠一遍太脆。

    清单原行也带出来：入库时它就是 `raw_json`（AI 原话），而算出来的 payload 是
    `current_json`。少了这一路，血缘里就只剩程序算出的区间，追不回 AI 当初指的两个点。

    算不出合法区间的行直接跳过并记一句日志 —— 一份清单里有一行不合规，
    不该让整批都进不来。

    区间重叠的后来者也丢掉：同一次事件被指了两遍时，留先出现的那个
    （清单按 score 从高到低给，先出现的就是分高的那条）。
    """
    out: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    logs: list[str] = []
    taken: list[tuple[float, float]] = []
    for index, row in enumerate(rows, start=1):
        made = payload_from_moment(segments, row, video_name=video_name,
                                   min_sec=min_sec, max_sec=max_sec,
                                   max_voiced=max_voiced, duration=duration)
        if made is None:
            logs.append(f"第 {index} 行算不出合法区间，跳过")
            continue
        payload, speech, notes = made
        start = float(payload["clip"]["start"])
        end = float(payload["clip"]["end"])
        if any(start < old_end and end > old_start for old_start, old_end in taken):
            logs.append(f"第 {index} 行（{start:.2f}-{end:.2f}）和前面的区间重叠，跳过")
            continue
        taken.append((start, end))
        out.append((payload, speech, dict(row)))
        detail = "；".join(notes) if notes else "边界正好落在句边界上"
        logs.append(f"第 {index} 行 → {start:.2f}-{end:.2f}（{end - start:.2f}s）：{detail}"
                    f"｜原文：{speech or '（区间内没有语音）'}")
    return out, logs

