"""剪辑决策引擎：AI 的高光区间 + 逐词时间戳 -> 确定性的 ClipPlan。

这一层**只算时间**，不解码、不渲染、不碰数据库、不问 AI：

    高光 JSON（clip.start / clip.end / type / reason / score）
  + 逐词时间戳（speech_segments + speech_words）
  + 视频时长（可选）
        ↓
    plan_clips()  ← 纯函数：同样的输入永远得到同样的输出
        ↓
    ClipPlan（最终 start / end + 调整原因 + 用到哪些词）
        ↓
    highlight/clip.py 的 parse_spec + render_highlight（既有 PyAV 渲染，原样复用）

两条硬规则（都是之前踩过的坑）：

  1. 不从一句话中间开始——落在句中就回溯到整句起点（太远则退到词边界）。
  2. 结束必须在**下一次说话之前**——AI 给的 end 越到下一句里，就提前到上一句说完那一刻。

片段多长不由这一层管：时长是 PRM（提示词）里对 AI 提的要求，AI 给多长就剪多长。
以前这里写死「普通片段 ≤ 15 秒」，那是把提示词里的口径搬进了代码，PRM 一改就打架。

时间一律按毫秒（3 位小数）取整，避免浮点尾差让"同样的输入"算出不同结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# 结束点与下一句开口之间至少留这么多，免得把下一句的第一个音带进来
SPEECH_GUARD = 0.08
# 判断"已经在整句起点/整句收尾"的容差
EPS = 0.06
# 整句起点比 AI 起点早超过这么多，就不整句回溯了（那不叫修边界，叫换片段）
MAX_BACKTRACK = 3.0
# 两个高光之间的空隙小于这个值，且中间没有别的句子开口，才考虑合并
MERGE_GAP = 0.60
# 后续句子要被算进本片段，至少得被 AI 区间覆盖这么多（比例 / 秒数取大者），
# 否则就当"AI 多给了一点尾巴"，不把这句拉进来
MIN_COVER_RATIO = 0.5
MIN_COVER_SECONDS = 0.8



# ============================================================== 输入数据结构
@dataclass(frozen=True)
class Word:
    """一个词的时间戳。`text` 原样保留（含标点），引擎不改写文字。"""

    start: float
    end: float
    text: str = ""


@dataclass(frozen=True)
class Segment:
    """一句话（whisper 断句后的 speech_segment）及其逐词。"""

    start: float
    end: float
    text: str = ""
    words: tuple[Word, ...] = ()


# ============================================================== 输出数据结构
@dataclass(frozen=True)
class ClipPlan:
    """一条可以直接交给渲染的剪辑计划。全部字段都是算完的结果，不含随机成分。"""

    source_video: str
    start: float
    end: float
    duration: float
    ai_start: float
    ai_end: float
    reason: str = ""
    score: float | None = None
    type: str = ""
    words: tuple[Word, ...] = ()
    next_speech_start: float | None = None
    notes: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """这段话的原文（逐词拼起来，只为日志/核对用）。"""
        return "".join(w.text for w in self.words).strip()


@dataclass(frozen=True)
class PlanResult:
    """一次决策的完整结果：能剪的 + 被拒的（附中文原因）。"""

    plans: tuple[ClipPlan, ...] = ()
    rejected: tuple[tuple[dict[str, Any], str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.plans)


# ============================================================== 逐词数据整理
def segments_from_payload(rows: Iterable[Any]) -> tuple[Segment, ...]:
    """把内存里的段（`seg["words"]` 这种 dict）整理成引擎用的 Segment。

    容忍缺字段：没有 words 的段照旧当整句用（`time_estimated` 那类），
    段起止缺失就用逐词的首尾兜底；两样都没有的段直接丢掉。
    """
    out: list[Segment] = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        words: list[Word] = []
        for item in (row.get("words") or ()):
            if not isinstance(item, dict):
                continue
            start, end = _num(item.get("start")), _num(item.get("end"))
            if start is None or end is None or end < start:
                continue
            words.append(Word(round(start, 3), round(end, 3), str(item.get("word") or "")))
        words.sort(key=lambda w: (w.start, w.end))
        seg_start = _num(row.get("start"))
        seg_end = _num(row.get("end"))
        if seg_start is None and words:
            seg_start = words[0].start
        if seg_end is None and words:
            seg_end = words[-1].end
        if seg_start is None or seg_end is None or seg_end <= seg_start:
            continue
        out.append(Segment(round(seg_start, 3), round(seg_end, 3),
                           str(row.get("text") or ""), tuple(words)))
    out.sort(key=lambda s: (s.start, s.end))
    return tuple(out)


def segments_for_video(db: Any, video_id: int) -> tuple[Segment, ...]:
    """从库里取这个视频最近一次成功分析的句 + 逐词。取不到就返回空。

    这是引擎唯一一处"对外要数据"的地方，核心决策函数完全不认识数据库。
    """
    from ..db import repo  # noqa: PLC0415 - 只有走库这条路才需要

    run = repo.latest_analysis(db, video_id)
    if run is None:
        return ()
    analysis_id = int(run["id"])
    words_by_segment: dict[int, list[Word]] = {}
    for row in repo.get_speech_words(db, analysis_id):
        start, end = _num(row["start_time"]), _num(row["end_time"])
        if start is None or end is None or end < start:
            continue
        words_by_segment.setdefault(int(row["segment_id"]), []).append(
            Word(round(start, 3), round(end, 3), str(row["word"] or "")))
    segments: list[Segment] = []
    for row in repo.get_speech_segments(db, analysis_id):
        words = sorted(words_by_segment.get(int(row["id"]), []), key=lambda w: (w.start, w.end))
        start, end = _num(row["start_time"]), _num(row["end_time"])
        if start is None and words:
            start = words[0].start
        if end is None and words:
            end = words[-1].end
        if start is None or end is None or end <= start:
            continue
        segments.append(Segment(round(start, 3), round(end, 3),
                                str(row["text"] or ""), tuple(words)))
    segments.sort(key=lambda s: (s.start, s.end))
    return tuple(segments)


def _num(value: Any) -> float | None:
    """能当秒数用就返回 float，否则 None。bool / NaN / inf 一律不算数字。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


# ============================================================== AI JSON 读取
def clips_in_payload(payload: Any) -> list[dict[str, Any]]:
    """从 AI JSON 里取出所有候选片段，**不改协议**：

      {"clip": {...}}      单条（现行协议，最常见）
      {"clips": [{...}]}   多条
      {...start,end...}    根上直接是 clip

    字段原样带出（含中文 reason / evaluation），引擎绝不改写文字内容。
    """
    if isinstance(payload, str):
        import json  # noqa: PLC0415

        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    found: list[dict[str, Any]] = []
    raw_list = payload.get("clips")
    if isinstance(raw_list, list):
        found.extend(item for item in raw_list if isinstance(item, dict))
    single = payload.get("clip")
    if isinstance(single, dict):
        found.append(single)
    if not found and ("start" in payload or "end" in payload):
        found.append(payload)
    # 顶层的 video 字段补给每一条（多条写法通常只在顶层写一次）
    video = payload.get("video")
    out: list[dict[str, Any]] = []
    for item in found:
        merged = dict(item)
        merged.setdefault("video", video)
        out.append(merged)
    return out


# ============================================================== 边界算法
def _segment_at(segments: Sequence[Segment], moment: float) -> Segment | None:
    """哪一句正好覆盖这个时刻（含端点）。没有就是 None（此刻没人说话）。"""
    for seg in segments:
        if seg.start - EPS <= moment <= seg.end + EPS:
            return seg
    return None


def _word_at(seg: Segment, moment: float) -> Word | None:
    for word in seg.words:
        if word.start - EPS <= moment <= word.end + EPS:
            return word
    return None


def _next_speech_start(segments: Sequence[Segment], after: float) -> float | None:
    """`after` 之后最近一次开口的时间。没有下一句就是 None。"""
    for seg in segments:
        if seg.start > after + EPS:
            return seg.start
    return None


def _snap_start(ai_start: float, segments: Sequence[Segment]) -> tuple[float, list[str]]:
    """起点不许落在一句话中间：落在句中就回溯到整句起点。"""
    notes: list[str] = []
    seg = _segment_at(segments, ai_start)
    if seg is None:
        if segments:
            notes.append("起点落在没人说话的间隙，按 AI 原点开始")
        return round(ai_start, 3), notes
    if ai_start - seg.start <= EPS:
        return round(max(seg.start, 0.0), 3), notes
    if ai_start - seg.start <= MAX_BACKTRACK:
        notes.append(f"起点从句中回溯到整句起点（{ai_start:.2f} → {seg.start:.2f}）")
        return round(max(seg.start, 0.0), 3), notes
    # 整句起点太远：那就退一步，至少不要把一个词切成两半
    word = _word_at(seg, ai_start)
    if word is not None and ai_start - word.start > EPS:
        notes.append(f"整句起点过远（>{MAX_BACKTRACK:.0f}s），退到词边界"
                     f"（{ai_start:.2f} → {word.start:.2f}）")
        return round(max(word.start, 0.0), 3), notes
    return round(max(ai_start, 0.0), 3), notes


def _clip_sentences(segments: Sequence[Segment], start: float,
                    ai_end: float) -> list[Segment]:
    """这一条高光真正说了哪几句。

    关键在最后一句的取舍：AI 的 end 常常越过静音、伸进下一句的头上一点点。
    第一句（片段开头那句）永远算自己的；后面的句子只有在**被 AI 区间覆盖得够多**
    （整句吃下，或至少 `MIN_COVER_RATIO` / `MIN_COVER_SECONDS`）时才算，
    否则就认定"AI 只是多给了一点尾巴"，这句不属于本片段。
    """
    chain: list[Segment] = []
    for seg in segments:
        if seg.end <= start + EPS:
            continue
        if seg.start > ai_end + EPS:
            break
        if not chain:
            chain.append(seg)
            continue
        if ai_end >= seg.end - EPS:          # 整句都在区间里
            chain.append(seg)
            continue
        covered = min(ai_end, seg.end) - seg.start
        span = seg.end - seg.start
        if covered + EPS >= max(MIN_COVER_SECONDS, span * MIN_COVER_RATIO):
            chain.append(seg)
        else:
            break                            # 只蹭到一个头，这句不算
    return chain


def _snap_end(start: float, ai_end: float, segments: Sequence[Segment], *,
              video_duration: float | None) -> tuple[float, float | None, list[str]]:
    """结束点：别越到下一句里，也别越出视频。返回 (end, 下一句起点, 说明)。

    多长不管——时长的要求写在 PRM 里，由 AI 决定，这一层只修边界。
    """
    notes: list[str] = []
    end = round(ai_end, 3)
    inside = _clip_sentences(segments, start, ai_end)
    last = inside[-1] if inside else _segment_at(segments, start)

    # --- 规则一：AI 的 end 不许越过下一次开口 ---
    next_start: float | None = None
    if last is not None:
        next_start = _next_speech_start(segments, last.end)
        if end > last.end + EPS and next_start is not None and end > next_start - SPEECH_GUARD:
            end = round(min(last.end, next_start - SPEECH_GUARD), 3)
            notes.append(f"结束提前到下一段说话之前（{ai_end:.2f} → {end:.2f}，"
                         f"下一句 {next_start:.2f} 开口）")
        elif last.start - EPS <= end <= last.end - EPS and end < last.end:
            # AI 把这句话切了一半：补到整句说完
            notes.append(f"结束延到整句说完（{ai_end:.2f} → {last.end:.2f}）")
            end = round(last.end, 3)
    elif segments:
        next_start = _next_speech_start(segments, start)
        if next_start is not None and end > next_start - SPEECH_GUARD:
            end = round(next_start - SPEECH_GUARD, 3)
            notes.append(f"结束提前到下一段说话之前（{ai_end:.2f} → {end:.2f}）")

    # --- 规则二：不许超出视频本身 ---
    if video_duration is not None and end > video_duration:
        end = round(video_duration, 3)
        notes.append(f"结束点超出视频时长，收到 {end:.2f}")
    return end, next_start, notes


def _words_between(segments: Sequence[Segment], start: float, end: float) -> tuple[Word, ...]:
    """落在 [start, end] 里的词（用于日志和核对，不参与时间决策）。"""
    out = [word for seg in segments for word in seg.words
           if word.start >= start - EPS and word.end <= end + EPS]
    out.sort(key=lambda w: (w.start, w.end))
    return tuple(out)


# ============================================================== 主入口
def plan_clips(payload: Any, segments: Sequence[Segment] | None = None, *,
               video_duration: float | None = None,
               source_video: str = "") -> PlanResult:
    """把 AI JSON 变成一组 ClipPlan。纯函数：不看时间、不读盘、不问 AI、不随机。

    `segments` 为空（没跑过分析 / 没有逐词）时不瞎猜：保持 AI 原区间，只做
    合法性校验和"不超出视频时长"，并在 notes 里写明"没有逐词数据"。
    """
    segs = tuple(segments or ())
    plans: list[ClipPlan] = []
    rejected: list[tuple[dict[str, Any], str]] = []

    for clip in clips_in_payload(payload):
        ai_start, ai_end = _num(clip.get("start")), _num(clip.get("end"))
        if ai_start is None or ai_end is None:
            rejected.append((clip, "clip.start / clip.end 不是有效数字"))
            continue
        if ai_start < 0 or ai_end < 0:
            rejected.append((clip, f"时间不能是负数（start={ai_start}, end={ai_end}）"))
            continue
        if ai_end <= ai_start:
            rejected.append((clip, f"clip.end({ai_end}) 必须大于 clip.start({ai_start})"))
            continue
        if video_duration is not None and ai_start >= video_duration:
            rejected.append((clip, f"起点 {ai_start} 已经超出视频时长 {video_duration}"))
            continue

        notes: list[str] = []
        if segs:
            start, start_notes = _snap_start(ai_start, segs)
            notes += start_notes
        else:
            start = round(ai_start, 3)
            notes.append("没有逐词数据，起点保持 AI 原值")
        end, next_start, end_notes = _snap_end(
            start, ai_end, segs, video_duration=video_duration)
        notes += end_notes
        if not segs:
            notes.append("没有逐词数据，结束点只受视频时长约束")
        if end - start <= 0:
            rejected.append((clip, f"修正后区间不成立（{start} → {end}）"))
            continue

        plans.append(ClipPlan(
            source_video=str(clip.get("video") or source_video or ""),
            start=start, end=end, duration=round(end - start, 3),
            ai_start=round(ai_start, 3), ai_end=round(ai_end, 3),
            reason=str(clip.get("reason") or ""),      # 中文原样保留
            score=_num(clip.get("score")),
            type=str(clip.get("type") or ""),
            words=_words_between(segs, start, end),
            next_speech_start=next_start,
            notes=tuple(notes), raw=dict(clip),
        ))

    return PlanResult(plans=_dedupe(plans, segs), rejected=tuple(rejected))


def _dedupe(plans: list[ClipPlan], segments: Sequence[Segment]) -> tuple[ClipPlan, ...]:
    """多高光整理：排序 -> 去完全重复 -> 重叠留高分 -> 同句相邻才合并。

    排序键写全（start, end, -score, type, reason），所以顺序不依赖输入顺序，
    也不依赖字典/文件遍历顺序。
    """
    ordered = sorted(plans, key=lambda p: (p.start, p.end,
                                           -(p.score if p.score is not None else -1.0),
                                           p.type, p.reason))
    kept: list[ClipPlan] = []
    for plan in ordered:
        if any(abs(k.start - plan.start) <= EPS and abs(k.end - plan.end) <= EPS for k in kept):
            continue                      # 完全重复的一条，丢掉
        if kept:
            prev = kept[-1]
            if plan.start < prev.end - EPS:          # 重叠：留分高的那条
                if _score(plan) > _score(prev):
                    kept[-1] = plan
                continue
            merged = _merge(prev, plan, segments)
            if merged is not None:
                kept[-1] = merged
                continue
        kept.append(plan)
    return tuple(kept)


def _score(plan: ClipPlan) -> float:
    return plan.score if plan.score is not None else -1.0


def _merge(first: ClipPlan, second: ClipPlan,
           segments: Sequence[Segment]) -> ClipPlan | None:
    """相邻两段能不能并成一段：空隙很小、中间没有别人开口。"""
    gap = second.start - first.end
    if gap < 0 or gap > MERGE_GAP:
        return None
    between = [seg for seg in segments if first.end + EPS < seg.start < second.start - EPS]
    if between:
        return None                       # 中间还夹着一整句，不是"同一段话"
    winner = first if _score(first) >= _score(second) else second
    notes = tuple(first.notes) + tuple(second.notes) + (
        f"与相邻高光合并（{first.start:.2f}→{first.end:.2f} + "
        f"{second.start:.2f}→{second.end:.2f}，中间只隔 {gap:.2f}s 且同一段话）",)
    return ClipPlan(
        source_video=winner.source_video,
        start=first.start, end=second.end, duration=round(second.end - first.start, 3),
        ai_start=min(first.ai_start, second.ai_start),
        ai_end=max(first.ai_end, second.ai_end),
        reason=winner.reason, score=winner.score, type=winner.type,
        words=_words_between(segments, first.start, second.end),
        next_speech_start=second.next_speech_start,
        notes=notes, raw=dict(winner.raw),
    )


# ============================================================== dry-run 输出
def describe(plan: ClipPlan, index: int = 1, total: int = 1) -> list[str]:
    """中文 dry-run 报告：不渲染，只把这条计划怎么算出来的讲清楚。"""
    lines = [
        f"[剪辑引擎] 第 {index}/{total} 段"
        + (f"（视频：{plan.source_video}）" if plan.source_video else ""),
        f"[剪辑引擎] AI区间：{plan.ai_start:.2f} → {plan.ai_end:.2f}"
        f"（{plan.ai_end - plan.ai_start:.2f} 秒）",
        f"[剪辑引擎] 修正后：{plan.start:.2f} → {plan.end:.2f}",
        f"[剪辑引擎] 时长：{plan.duration:.2f} 秒",
        f"[剪辑引擎] 下一段说话起点："
        + (f"{plan.next_speech_start:.2f}" if plan.next_speech_start is not None else "没有下一段"),
        f"[剪辑引擎] 用到 {len(plan.words)} 个词"
        + (f"：{plan.text[:60]}" if plan.words else "（这段里没有逐词数据）"),
    ]
    if plan.type or plan.score is not None:
        lines.append(f"[剪辑引擎] 类型：{plan.type or '未标注'}｜评分："
                     + (f"{plan.score:g}" if plan.score is not None else "未给"))
    if plan.reason:
        lines.append(f"[剪辑引擎] AI 理由：{plan.reason}")
    for note in plan.notes:
        lines.append(f"[剪辑引擎] 调整：{note}")
    if not plan.notes:
        lines.append("[剪辑引擎] 调整：无（AI 区间本身就落在语义边界上）")
    return lines


def describe_result(result: PlanResult) -> list[str]:
    """整份结果的 dry-run 报告（含被拒的片段和原因）。"""
    lines: list[str] = []
    total = len(result.plans)
    for i, plan in enumerate(result.plans, start=1):
        lines += describe(plan, i, total)
        lines.append("")
    for clip, why in result.rejected:
        lines.append(f"[剪辑引擎] 已拒绝：{why}（原始 start={clip.get('start')!r}"
                     f" end={clip.get('end')!r}）")
    if not result.plans:
        lines.append("[剪辑引擎] 没有可剪的片段，不启动渲染")
    return lines


def first_clip_payload(payload: Any) -> dict[str, Any] | None:
    """把多条写法（`clips: [...]`）折成单条，交给既有 `parse_spec` 定位源视频。

    `parse_spec` 只认 `clip`，多条写法会在校验这一步就被拒；这里只做"取第一条"，
    时间和文案原样不动，真正的决策仍然由 `plan_clips` 负责。
    """
    clips = clips_in_payload(payload)
    if not clips:
        return None
    clip = dict(clips[0])
    video = clip.get("video")
    out: dict[str, Any] = {"clip": clip}
    if video:
        out["video"] = video
    return out


def payload_for(plan: ClipPlan) -> dict[str, Any]:
    """把修正后的时间写回 AI JSON 形状，交给既有 `clip.parse_spec` 渲染。

    只改 clip.start / clip.end / clip.duration 三个时间；overlays、reason、
    evaluation、score、type 全部原样带走（中文文案绝不动）。
    """
    clip = dict(plan.raw)
    clip["start"] = plan.start
    clip["end"] = plan.end
    clip["duration"] = plan.duration
    out: dict[str, Any] = {"clip": clip}
    if plan.source_video:
        out["video"] = plan.source_video
    return out
