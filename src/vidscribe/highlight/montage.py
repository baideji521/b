"""第二轮混剪的守门人：AI 交回来的 montage JSON 合不合规，程序说了算。

为什么要有这一层：混剪 PRM 里能机器判定的规则占了绝大多数（素材够不够分、
同一组里有没有同源、Rank 1 有没有闭嘴、旁白说不说得完），而这些恰恰是 AI 最容易
算错的地方 —— 实测一批输出里 16 处违规**全部**出在「旁白时间算得对不对」上，
一处主观判断的错都没有。

所以分工是：**能算的程序算，只把真正要人判断的留给人。**
这个模块只负责前者，输出违规清单；后者（故事/节奏/情绪好不好）由评分表人工填，
两边合起来才是一条 montage 的最终成绩。

口径必须和混剪 PRM 逐字一致 —— PRM 改了这里也要改，不然会出现
「PRM 说合规、程序说违规」这种谁都不敢动的僵局。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from .. import ai_protocol


#: 素材准入：四维之和的下限。PRM 第 5b 节 —— 17/20 就是 85 分
SCORE_FLOOR = 17
#: 四维满分（换算成百分制用）
SCORE_FULL = 20

#: 旁白落在空隙里时，前后各留这么多秒不说话（PRM 第 28 节）。
#: 不留的话起说贴着上一句的尾音、收尾撞上下一句的起音，听着就是抢话
GAP_LEAD = 0.15
GAP_TAIL = 0.15

#: Rank 1 的高潮保护区从「结果时刻」往前留这么久（PRM 第 3 节）
CLIMAX_GUARD = 1.0
#: 其余 rank **用空隙插的** 旁白要避开挂字前后这么久。
#: 末尾硬插不受这条限制 —— 短素材的挂字本来就靠尾巴，不豁免就无解
WORD_KEEPOUT = 0.5

#: 一条旁白最多几个词（PRM 自检清单）
COMMENT_MAX_WORDS = 5
#: 一条 montage 至少几条旁白（开场白算 1 条，Rank 1 免除）
COMMENT_MIN_TOTAL = 4
#: 一条 montage 恰好几个素材
PICK_COUNT = 5

#: 输出的 `overlays` 里只允许这些 type。
#: `word` 允许出现，但**必须是输入 `o` 里那条的原样照抄**（时间和文案一个字不改）——
#: 挂字文案是第一轮定的、已经烧进画面，这一轮不重新想、不润色、不换词。
#: `emoji` 不单独成条：它本来就写在 `word` 的文案里（`RECORD 📈`）
ALLOWED_KINDS = ("comment", "word")



#: TTS 时长查表（PRM 第 29 节）。**不许换成「每词 X 秒」的线性公式** ——
#: 短句有起停开销，1 词和 5 词不成正比，线性算出来 4-5 词那档会偏短，
#: 末尾硬插就会把话的尾巴甩到素材外面
TTS_BASE = {1: 0.60, 2: 0.90, 3: 1.20, 4: 1.60, 5: 1.90}
#: 超过 5 词每多一个词加这么多（PRM 不许超 5 词，这里只为算出「超了多少」）
TTS_EXTRA = 0.35
#: 句内每个逗号 / 问号 / 叹号 / 省略号加这么多（结尾句号不加）
TTS_PUNCT = 0.15

#: 时间比对容差：两位小数的输入，差一分钱不算错
EPS = 0.02

_TOKEN = re.compile(r"[A-Za-z0-9']+")
_PUNCT = re.compile(r"\.\.\.|[,?!]")
_TIKTOK_ID = re.compile(r"tiktok_(\d+)")


def word_count(text: Any) -> int:
    """一条旁白有几个词。emoji 和标点不算词 —— TTS 不读它们。"""
    return len(_TOKEN.findall(str(text or "")))


def tts_seconds(text: Any) -> float:
    """这句话念出来要多久（秒）。查 `TTS_BASE` 那张表，句内标点各加 `TTS_PUNCT`。

    结尾的句号不加：那是自然收尾，不是句中停顿。
    """
    words = word_count(text)
    if words <= 0:
        return 0.0
    base = TTS_BASE.get(words)
    if base is None:
        base = TTS_BASE[5] + (words - 5) * TTS_EXTRA
    body = re.sub(r"[.!?]+$", "", str(text or ""))
    return round(base + TTS_PUNCT * len(_PUNCT.findall(body)), 2)


def total_score(scores: Any) -> int | None:
    """四维之和（0-20）；给不出数就返回 None，绝不当成 0。

    当成 0 会让「没打分的老数据」直接被判不合格，而它可能只是字段缺失。
    """
    if not isinstance(scores, dict):
        return None
    out = 0
    seen = False
    for key in ("surprise", "standalone", "emotion_power", "caption"):
        value = scores.get(key)
        if isinstance(value, (int, float)):
            out += int(value)
            seen = True
    return out if seen else None


def percent(scores: Any) -> float | None:
    """四维换算成百分制（PRM 第 5b 节的公式）。"""
    got = total_score(scores)
    return None if got is None else round(got / SCORE_FULL * 100, 1)


def source_id(source: Any) -> str:
    """`source` 里那串数字 ID。取不到就退回整个文件名当标识。

    为什么要单独取 ID：同一条原片下载两次会得到两个不同的 `source`
    （日期前缀不一样），但内容是同一条 —— 只比 `source` 字符串抓不到这种。
    """
    text = str(source or "")
    found = _TIKTOK_ID.search(text)
    return found.group(1) if found else text


def load_pool(lines: Iterable[str]) -> dict[str, dict[str, Any]]:
    """「提取数据」那份 JSONL → `{成品文件名: 那一行}`。读不懂的行跳过，不报错。"""
    pool: dict[str, dict[str, Any]] = {}
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            one = json.loads(text)
        except (TypeError, ValueError):
            continue
        name = one.get("video")
        if isinstance(name, str) and name:
            pool[name] = one
    return pool


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _overlays(pick: Any, kind: str) -> list[tuple[float, str]]:
    """挑出某一类挂字，返回 `[(时间, 文案)]`，按时间排好。"""
    out: list[tuple[float, str]] = []
    for item in (pick.get("overlays") or ()):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        if str(item[1]) != kind:
            continue
        at = _num(item[0])
        if at is not None:
            out.append((at, str(item[2])))
    out.sort()
    return out


def _result_word(ref: dict[str, Any]) -> tuple[float, str] | None:
    """输入 `o` 里那条 `word`：`(时间, 文案)`；没有就 None。

    召回那一轮把挂字**钉在结果时刻**上，所以它的时间就是高潮秒数
    —— 这是输入里唯一标出高潮位置的东西（PRM 第 3 节）。
    文案也一并带出来：输出必须原样照抄，得有个东西可比。
    """
    for item in (ref.get("o") or ()):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        if str(item[1]) != "word":
            continue
        at = _num(item[0])
        if at is not None:
            return at, str(item[2])
    return None


def _result_moment(ref: dict[str, Any]) -> float | None:
    """这条素材的结果 / 高潮时刻（秒）。"""
    found = _result_word(ref)
    return None if found is None else found[0]



def _gaps(ref: dict[str, Any]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in (ref.get("gaps") or ()):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lo, hi = _num(item[0]), _num(item[1])
        if lo is not None and hi is not None and hi > lo:
            out.append((lo, hi))
    return out


def fits_words(room: float) -> int:
    """这么长的说话时间最多塞几个词（查 `TTS_BASE` 那张表）。

    报错时直接给出「最多 N 词」，比只说「差 0.13 秒」有用 ——
    差多少秒改不动，知道只能说 1 个词才知道怎么改。
    """
    out = 0
    for words in sorted(TTS_BASE):
        if TTS_BASE[words] <= room + 0.005:
            out = words
    return out


def check_pick(pick: dict[str, Any], ref: dict[str, Any] | None, *,
               first: bool, opening: str) -> list[str]:

    """一条素材的违规清单（空列表 = 这一条没问题）。

    `first` = 它排在 `play_order` 第一位（要带开场白）；
    `opening` = 这条 montage 的 `opening.text`（开场白必须逐字一致）。
    """
    name = str(pick.get("video") or "?")
    rank = pick.get("rank")
    tag = f"rank{rank} {name}"
    if ref is None:
        return [f"{tag}：不在输入素材里，凭空冒出来的"]

    bad: list[str] = []
    timeline = ref.get("timeline") or {}
    real = _num(timeline.get("duration"))
    got = total_score(timeline.get("scores"))

    if got is not None and got < SCORE_FLOOR:
        bad.append(f"{tag}：{got}/{SCORE_FULL} = {percent(timeline.get('scores'))} 分，"
                   f"低于 {SCORE_FLOOR}/{SCORE_FULL}（85 分）门槛")

    said = _num(pick.get("duration"))
    if real is not None and (said is None or abs(said - real) > 0.001):
        bad.append(f"{tag}：duration 写 {said}，输入是 {real} —— 照抄，不许改")

    source = _result_word(ref)
    moment = None if source is None else source[0]
    marks = _overlays(pick, "word")
    if source is None:
        for at, text in marks:
            bad.append(f"{tag}：输入的 o 里没有 word，输出却写了 {text!r}（{at}）"
                       f" —— 挂字只能照抄，不许自己造")
    elif not marks:
        bad.append(f"{tag}：漏了挂字 —— 输入的 o 里有 "
                   f"[{source[0]}, 'word', {source[1]!r}]，照抄一条出来")
    else:
        if len(marks) > 1:
            bad.append(f"{tag}：写了 {len(marks)} 条挂字，输入只有 1 条 —— 照抄那一条就行")
        for at, text in marks:
            if abs(at - source[0]) > 0.001:
                bad.append(f"{tag}：挂字时间写 {at}，输入是 {source[0]} —— 照抄，不许改")
            if text != source[1]:
                bad.append(f"{tag}：挂字文案写 {text!r}，输入是 {source[1]!r} —— "
                           f"照抄，不许润色、不许换词、不许动 emoji")
    for item in (pick.get("overlays") or ()):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            kind = str(item[1])
            if kind not in ALLOWED_KINDS:
                bad.append(f"{tag}：overlays 里出现了 {kind!r} —— 只允许 "
                           f"{' / '.join(ALLOWED_KINDS)}"
                           + ("（emoji 写在 word 的文案里，不单独成条）"
                              if kind == "emoji" else ""))



    gaps = _gaps(ref)
    comments = _overlays(pick, "comment")
    if rank != 1 and not comments:
        bad.append(f"{tag}：一条 comment 都没有（除 Rank 1 每段至少 1 条）")

    for at, text in comments:
        words = word_count(text)
        need = tts_seconds(text)
        head = f"{tag} comment {at:.2f} {text!r}"
        if words > COMMENT_MAX_WORDS:
            bad.append(f"{head}：{words} 个词，超过上限 {COMMENT_MAX_WORDS}")

        if first and abs(at) <= EPS:
            if text != opening:                     # 开场白钉在 0.0，不用找空隙
                bad.append(f"{head}：开场白和 opening.text {opening!r} 不一致")
            continue

        hit = [one for one in gaps if abs(at - (one[0] + GAP_LEAD)) <= 0.06]
        if hit:
            lo, hi = hit[0]
            room = round(hi - lo - GAP_LEAD - GAP_TAIL, 2)
            if need > room + 0.005:
                # 分两种说法：真压到原声，和只是收尾缓冲不够。
                # 混着说会让人以为「差 0.02 秒也算压住原声」，下次照旧改错
                if at + need > hi + 0.005:
                    why = (f"压到原声 {at + need - hi:.2f}s"
                           f"（说到 {at + need:.2f}，原声 {hi:.2f} 就回来了）")
                else:
                    why = (f"收尾只剩 {hi - at - need:.2f}s，不够 {GAP_TAIL} 的缓冲"
                           f"（说到 {at + need:.2f}，原声 {hi:.2f} 回来）")
                bad.append(f"{head}：空隙 {hi - lo:.2f}s 只容得下 {room:.2f}s"
                           f"（最多 {fits_words(room)} 个词），这句要 {need:.2f}s —— {why}")
            if rank != 1 and moment is not None and abs(at - moment) < WORD_KEEPOUT:
                bad.append(f"{head}：压在结果时刻 {moment:.2f} 的 ±{WORD_KEEPOUT} 秒内")

        elif real is not None:
            want = round(real - need, 2)            # 末尾硬插：在素材播完那一刻正好说完
            if abs(at - want) > EPS:
                bad.append(f"{head}：既不在 gaps 里，按末尾硬插算应该是 {want:.2f}"
                           f"（{real:.2f} − {need:.2f}），差 {at - want:+.2f}")

        if real is not None and at + need > real + 0.01:
            bad.append(f"{head}：说不完 —— {at:.2f} + {need:.2f} = {at + need:.2f} > {real:.2f}")
        if rank == 1 and moment is not None and at >= moment - CLIMAX_GUARD:
            bad.append(f"{head}：进了 Rank 1 的高潮保护区 "
                       f"[{moment - CLIMAX_GUARD:.2f}, {real if real else 0:.2f}]，"
                       f"这一段必须闭嘴让原声爆发")
    return bad


def check_montage(one: dict[str, Any], pool: dict[str, dict[str, Any]]) -> list[str]:
    """一条 montage 的违规清单（空列表 = 合规）。"""
    bad: list[str] = []
    picks = list(one.get("selected") or ())
    order = [str(x) for x in (one.get("play_order") or ())]
    opening = str(((one.get("opening") or {}).get("text")) or "")

    if len(picks) != PICK_COUNT:
        bad.append(f"selected 有 {len(picks)} 条，应该恰好 {PICK_COUNT} 条")
    names = [str(p.get("video") or "?") for p in picks]
    if sorted(names) != sorted(order):
        bad.append("play_order 和 selected 的 video 对不上")
    if len(set(names)) != len(names):
        bad.append("同一条 montage 里有重复的 video")

    groups: dict[str, list[str]] = {}
    for pick, name in zip(picks, names):
        ref = pool.get(name)
        if ref is not None:
            groups.setdefault(source_id(ref.get("source")), []).append(name)
    for key, members in groups.items():
        if len(members) > 1:
            bad.append(f"同源撞车（ID {key}）：{'、'.join(members)} —— 同一组里只能留一条")

    tops = [p for p in picks if p.get("rank") == 1]
    if len(tops) != 1:
        bad.append(f"Rank 1 有 {len(tops)} 条，应该恰好 1 条")
    elif order and order[-1] != str(tops[0].get("video")):
        bad.append(f"play_order 最后是 {order[-1]}，应该是 Rank 1 那条")

    if not opening:
        bad.append("opening.text 是空的")

    total = 0
    for pick, name in zip(picks, names):
        first = bool(order) and name == order[0]
        bad.extend(check_pick(pick, pool.get(name), first=first, opening=opening))
        total += len(_overlays(pick, "comment"))
    if total < COMMENT_MIN_TOTAL:
        bad.append(f"整条只有 {total} 条 comment，下限是 {COMMENT_MIN_TOTAL} 条")
    return bad


def montages_of(payload: Any) -> list[dict[str, Any]]:
    """AI 的回复 → montage 列表。认这几种写法：

      * **并排的多个对象**（现在要求的写法）：`{...}` `{...}` `{...}`，
        中间有没有换行、逗号、```json 围栏都行；
      * 一行一个对象（JSONL）；
      * 只有一条时的单个对象；
      * 老写法 `{"montages": [...]}` —— 照旧认，不然历史回复就读不了了。

    拆分这件事交给 `ai_protocol.split_payloads` —— 那是「一次回复拆成一份份 JSON」
    的唯一入口，入库那条路走的也是它。自己再写一遍括号扫描就会出现两份口径，
    一边能读一边读不了。

    整个回复是空的（一组都没凑出来，PRM 第 0 节要求什么都不输出）→ 空列表。
    """
    if isinstance(payload, str) and not payload.strip():
        return []
    out: list[dict[str, Any]] = []
    for one in ai_protocol.split_payloads(payload):
        many = one.get("montages")
        if isinstance(many, list):          # 老写法：外面包了一层
            out.extend(x for x in many if isinstance(x, dict))
        elif one.get("selected"):
            out.append(one)
    return out



def check_all(payload: Any, pool: dict[str, dict[str, Any]]) -> list[tuple[int, list[str]]]:
    """整份回复 → `[(第几条, 违规清单)]`。跨 montage 复用是允许的，不查。"""
    return [(i, check_montage(one, pool))
            for i, one in enumerate(montages_of(payload), start=1)]


def report(checked: Sequence[tuple[int, list[str]]]) -> str:
    """违规清单的人话报告。一条都不违规就只说一句「全部合规」。"""
    if not checked:
        return "没有可检查的 montage（回复是空的 —— 按 PRM，这就是「一组都凑不出来」）"
    lines: list[str] = []
    total = 0
    for index, bad in checked:
        if not bad:
            lines.append(f"montage {index}：合规")
            continue
        total += len(bad)
        lines.append(f"montage {index}：{len(bad)} 处不合规")
        lines.extend(f"  - {one}" for one in bad)
    lines.append(f"合计 {total} 处不合规" if total else "全部合规")
    return "\n".join(lines)


#: 评分表里**人工**填的那几项：机器判不了的才留在这。
#: 砍掉的三项和原因：
#:   「主题收敛」—— 一批素材往往同一个题材，任何组合都收敛，没有区分度；
#:   「旁白空隙可用」—— 这里已经逐条算过了，人再打一遍是重复；
#:   「高潮前静默」—— 和 Rank 1 保护区那条硬规则是同一件事，会重复计分。
#: 情绪那三项（类型变化 / 期待建立 / 梯度向上）实操里判不开，并成一项。
SCORE_ITEMS: tuple[tuple[str, int], ...] = (
    ("故事 · 叙事链闭环", 15),
    ("故事 · 信息梯度", 10),
    ("故事 · 话题潜力", 10),
    ("节奏 · 开篇强 Hook", 15),
    ("节奏 · 爆点错落", 15),
    ("节奏 · 时长搭配", 10),
    ("节奏 · 视觉多样性", 10),
    ("情绪 · 情绪梯度向上", 15),
)


def scorecard(checked: Sequence[tuple[int, list[str]]],
              payload: Any, pool: dict[str, dict[str, Any]]) -> str:
    """评分表 CSV：机器判定的部分填好，主观项留空等人填。

    为什么是 CSV 不是 xlsx：Excel 直接打得开，而且不用为一张表拉一个新依赖。
    第一行是表头，一列一条 montage。
    """
    many = montages_of(payload)
    heads = ["维度 / 评分项", "满分"] + [f"montage{i}" for i, _ in enumerate(many, 1)]
    rows: list[list[str]] = [heads]

    verdict = ["机器判定（不合规=淘汰，不看总分）", "-"]
    for (_i, bad), _one in zip(checked, many):
        verdict.append("通过" if not bad else f"淘汰（{len(bad)} 处）")
    rows.append(verdict)

    detail = ["  不合规明细", "-"]
    for _i, bad in checked:
        detail.append("；".join(bad) if bad else "")
    rows.append(detail)

    picks = ["  素材（rank1 在最后）", "-"]
    for one in many:
        order = [str(x) for x in (one.get("play_order") or ())]
        picks.append(" → ".join(order))
    rows.append(picks)

    floor = ["  最低素材分（四维/20）", "-"]
    for one in many:
        got = [total_score(((pool.get(str(p.get('video'))) or {}).get("timeline") or {})
                           .get("scores"))
               for p in (one.get("selected") or ())]
        have = [x for x in got if x is not None]
        floor.append(str(min(have)) if have else "")
    rows.append(floor)

    rows.append([""] * len(heads))
    for name, full in SCORE_ITEMS:
        rows.append([name, str(full)] + [""] * len(many))
    total = ["人工总分（0-100）", "100"]
    span = len(SCORE_ITEMS)
    start = len(rows) - span + 1                    # CSV 第 1 行是表头，行号从 1 数
    for col in range(len(many)):
        letter = chr(ord("C") + col)
        total.append(f"=SUM({letter}{start}:{letter}{start + span - 1})")
    rows.append(total)
    grade = ["评级", "-"]
    for col in range(len(many)):
        letter = chr(ord("C") + col)
        cell = f"{letter}{len(rows)}"
        grade.append(f'=IF({letter}2<>"通过","淘汰",'
                     f'IF({cell}>=85,"优秀",IF({cell}>=70,"良好",'
                     f'IF({cell}>=60,"及格","淘汰"))))')
    rows.append(grade)

    def cell(text: str) -> str:
        return '"' + text.replace('"', '""') + '"' if any(
            ch in text for ch in ',"\n') else text

    return "\n".join(",".join(cell(x) for x in row) for row in rows)
