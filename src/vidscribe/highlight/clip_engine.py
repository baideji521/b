"""剪辑决策引擎：AI 的高光区间 + 逐词时间戳 -> 确定性的 ClipPlan。

这一层**只算时间**，不解码、不渲染、不碰数据库、不问 AI：

    高光 JSON（segments[].sa / .end + timeline.type / reason / score）
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

from .. import ai_protocol

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
# 报给下游混剪的原声空隙至少这么长：比这更短的空隙塞不进一句配音
#: 静音压缩：一刀剪掉的量少于这个秒数就不剪了。
#: 每剪一刀多一个拼接点，为 0.06 秒收益去换一次画面接缝不划算
MIN_TRIM_SECONDS = 0.30



#: 报给下游混剪的原声空隙至少这么长：比这更短的空隙塞不进一句配音。
#: 0.5 秒是「一句最短旁白」的下限（第二轮按这个数决定哪里能插 TTS）。
#: 注意它和静音压缩的关系：`silence_keep` 比这个数小的话，压完之后
#: 所有空隙都只剩 `silence_keep` 秒，一处都进不了这张表 —— 那就等于自己
#: 把插旁白的位置全剪没了。要留旁白位就把 `silence_keep` 设成 >= 这个数
GAP_FLOOR = 0.5

# 从「结果时刻」往后并反应时：能被并进来的句子**实际发声**最长多久
# （语气词、单字反应）。按发声时长算而不是句子跨度，见 `voiced_seconds`
REACTION_MAX_LEN = 1.5
# 反应和结果之间最大间隔：超过这个距离就不是同一件事的收尾了
REACTION_GAP = 2.5



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


def silent_gaps(segments: Iterable[Segment], start: float, end: float, *,
                floor: float = GAP_FLOOR) -> list[list[float]]:
    """`[start, end]` 这段素材内部没有任何语音的空隙，换成**成片坐标**（0 起）返回。

    第二轮混剪要在原声空隙里插 TTS，空隙由这里算死，不让 AI 估（AI 手上没有词级时间戳，
    只能靠 overlays 那几个点猜，猜出来的 gap 偏大，配音就压在原声上）。

    口径：
      * 语音区间以**逐词**为准，词与词之间的呼吸也算空隙；某句没有逐词才退回整句区间
        （宁可少报一个空隙，也不谎报静音）；
      * 含首尾——开头和结尾的静音同样是可用的落位点；
      * 只报 >= `floor` 秒的空隙，比这更短塞不进一句 TTS；
      * 两位小数，单位秒。
    """
    left, right = float(start), float(end)
    span = round(right - left, 3)
    if span <= 0:
        return []
    speech: list[tuple[float, float]] = []
    for seg in segments:
        spans = [(w.start, w.end) for w in seg.words] or [(seg.start, seg.end)]
        for one_start, one_end in spans:
            lo, hi = max(left, float(one_start)), min(right, float(one_end))
            if hi > lo:
                speech.append((lo, hi))
    speech.sort()
    gaps: list[list[float]] = []
    cursor = left
    for lo, hi in speech:
        if lo - cursor >= floor:
            gaps.append([round(cursor - left, 2), round(lo - left, 2)])
        cursor = max(cursor, hi)
    if right - cursor >= floor:
        gaps.append([round(cursor - left, 2), round(span, 2)])
    return gaps


def trim_plan(segments: Iterable[Segment], start: float, end: float, *,
              keep: float = 0.0) -> list[tuple[float, float]]:
    """区间里超时的静音剪掉，返回**要保留的片段**（原视频绝对秒，按时间排好）。

    AI 只给两个时间点，区间由 `plan_from_span` 算；这里再往区间里面看一层：
    那些"没人说话、干等着"的静音，超过 `keep` 秒就把多出来的砍掉，前后两段接起来。

    静音位置全部由词级时间戳推出来（口径和 `silent_gaps` 完全一致），所以静音长在
    **开头、中间还是结尾都一样处理** —— 不需要谁告诉程序它在哪。

    剪的位置取静音**中间**，两端各留一半：说完话留一点收尾，下一句开口前留一点准备。
    只留开头那一截会显得话音刚落就被掐断。

    `keep <= 0`（没填这个数）或者区间里没有超时静音 → 返回一整段 `[(start, end)]`，
    调用方照旧按单段渲染，行为和以前一个样。

    **这是固定档**：不管区间多长，一律按 `keep` 剪。想按成品时长倒推该剪多少，
    走 `trim_to_target`（或统一入口 `trim_for`）。

    返回多段时，调用方要按顺序把每一段解码出来接起来 —— 中间被剪掉的部分不出现在成片里。
    挂字、空隙这些跟着的时间点用 `shift_time()` 换算，别自己减。
    """
    left, right = float(start), float(end)
    if right <= left:
        return []
    if keep <= 0:
        return [(round(left, 3), round(right, 3))]
    # silent_gaps 给的是成片坐标（0 起），这里换回绝对秒。
    # floor 直接用 keep：短于 keep 的静音本来就一分一秒都不用剪，让它压根不进来
    gaps = [(left + a, left + b)
            for a, b in silent_gaps(segments, left, right, floor=keep)]
    keeps: list[tuple[float, float]] = []
    cursor = left
    for lo, hi in gaps:
        if (hi - lo) - keep < MIN_TRIM_SECONDS:
            continue                      # 剪掉的量太少，不值得多一个拼接点
        half = keep / 2.0
        cut_from, cut_to = lo + half, hi - (keep - half)
        if cut_to - cut_from <= EPS:
            continue
        keeps.append((cursor, cut_from))
        cursor = cut_to
    keeps.append((cursor, right))
    return [(round(a, 3), round(b, 3)) for a, b in keeps if b - a > EPS]


def _keep_candidates(gaps: Sequence[float], need: float, floor: float) -> list[float]:
    """「留多少静音」的候选值，从大到小。

    剪掉的总量 `cut(keep) = Σ max(0, gap - keep)` 是 keep 的单调不增函数，最优解
    （刚好剪够 `need` 的那个最大 keep）一定落在两类点上：某个 gap 的长度本身，
    或者「前 k 长的 gap 各剪到同一水平」时的解 `(S_k - need) / k`。两类都收进来，
    再由调用方按实测挑第一个达标的 —— 不直接用公式定案，因为 `trim_plan` 里还有
    `MIN_TRIM_SECONDS`（剪不够 0.3 秒就不剪）这条规则，公式算不到它。
    """
    picks = {round(float(floor), 3)}
    ordered = sorted((float(g) for g in gaps), reverse=True)
    total = 0.0
    for index, gap in enumerate(ordered, start=1):
        picks.add(round(gap, 3))
        total += gap
        even = (total - need) / index          # 前 index 个都剪到同一水平
        if even > 0:
            picks.add(round(even, 3))
    return sorted((p for p in picks if p >= float(floor) - EPS), reverse=True)


def trim_to_target(segments: Iterable[Segment], start: float, end: float, *,
                   target: float, min_keep: float) -> list[tuple[float, float]]:
    """把这一段压到 `target` 秒以内，**静音尽量少剪** —— 剪到刚够就停。

    流程和「先看超没超，超了才动手」一致：

      1. 跨度没超 `target` → 一刀不剪，整段照渲；
      2. 超了 → 算还差多少，然后找**最大**的「静音留多少」，剪到总长刚好压进 `target`；
      3. 连 `min_keep`（最紧敢留的静音）都压不进去 → 就按 `min_keep` 剪，
         也就是**尽力剪到最紧**，成品仍然超一点 —— 那说明说话内容本身就有这么长，
         再往下剪就得动区间或者动语音了，不是静音这一层能解决的。

    第 3 步为什么不退回一个更松的档（比如"静音最多留 2 秒"）：那样会出现
    「目标定得越紧、成品反而越长」的怪事 —— 越松的档剪得越少。压不进去时至少要
    保证「剪得不比不设上限时少」，时长才是单调的。

    `target <= 0` 就是不设上限：这条路一刀不剪，要压静音请直接用 `trim_plan`。
    """
    left, right = float(start), float(end)
    if right <= left:
        return []
    whole = [(round(left, 3), round(right, 3))]
    if target <= 0 or (right - left) <= float(target) + EPS:
        return whole
    floor = max(0.0, float(min_keep))
    if floor <= 0:
        return whole                     # 没说最紧能留多少，不擅自剪
    need = (right - left) - float(target)
    gaps = [b - a for a, b in silent_gaps(segments, left, right, floor=floor)]
    for keep in _keep_candidates(gaps, need, floor):
        spans = trim_plan(segments, left, right, keep=keep)
        if sum(hi - lo for lo, hi in spans) <= float(target) + EPS:
            return spans
    return trim_plan(segments, left, right, keep=floor)   # 压不进去：剪到最紧就是了


def trim_for(segments: Iterable[Segment], start: float, end: float, *,
             keep: float = 0.0, target: float = 0.0) -> list[tuple[float, float]]:
    """静音压缩的**唯一入口**：填了目标时长就按目标压，没填就按固定「留多少」压。

    渲染（手动 / AI 自动 / 命令行）和「提取数据」都走这里，所以四条路算出来的
    片段逐字一致 —— 提取侧要是和渲染侧差一刀，报给第二轮的 `gaps` 就整体偏了。

    **注意**：这是「该剪成什么样」。要问「这个已经渲好的成品当初剪成了什么样」，
    走 `trim_as_made` —— 那条路认成品的真实时长，不认当前配置。
    """
    if float(target) > 0:
        return trim_to_target(segments, start, end, target=float(target),
                              min_keep=float(keep))
    return trim_plan(segments, start, end, keep=float(keep))


#: 反推时收集静音的下限。比这更短的静音任何档都不会去剪，收进来只会让候选爆炸。
#: 不跟 `GAP_FLOOR` 走：那是「报给下游的空隙至少多长」，和「当初可能用过多小的 keep」
#: 是两件事 —— 历史成品用过 0.3 这种档，下限定成 0.5 就反推不出来了
PROBE_FLOOR = 0.2
#: 反推认定「对上了」的容差。渲染按帧对齐，算出 5.52 渲成 5.533 属正常
PROBE_TOLERANCE = 0.15


def trim_as_made(segments: Iterable[Segment], start: float, end: float, *,
                 made: float, tolerance: float = PROBE_TOLERANCE,
                 ) -> list[tuple[float, float]] | None:
    """成品的**真实时长**是 `made` 秒 → 反推它当初剪成了哪几段（原视频绝对秒）。

    为什么要反推，不直接按当前配置现算：`silence_keep` 一改，现算出来的片段描述的
    就是「一个还没渲出来的成品」—— 报给第二轮的 `gaps` 整体偏掉，而且从数据上
    完全看不出来。成品的真实时长在库里（渲染当时写进 `clips.duration` 的），
    拿它反推出的片段才和盘上那个文件对得上。

    这样改配置不必重剪一遍旧成品：口径由**成品自己**说了算，不由配置说了算。

    做法：剪掉的总量 `cut(keep)` 对 keep 单调不增，所以按 `_keep_candidates` 给的
    候选从大到小试 `trim_plan`，第一个总长落进容差的就是它 —— 从大到小意味着
    命中的是「剪得最少的那个能对上的档」，和渲染时「按固定档剪」的语义一致。

    返回 None 的几种情况，调用方一律按「没剪过静音」处理：
    时长和区间长度本来就一样（压根没剪）、没有够长的静音、`made` 不合法、
    以及**反推不出来**（差得太远）—— 最后这种宁可说「不知道」也不猜一个。
    """
    left, right = float(start), float(end)
    if right <= left:
        return None
    span = right - left
    goal = float(made)
    if goal <= 0 or goal >= span - tolerance:
        return None                       # 没剪过：整段就是成品
    need = span - goal
    gaps = [b - a for a, b in silent_gaps(segments, left, right, floor=PROBE_FLOOR)]
    if not gaps:
        return None                       # 区间里没有可剪的静音，时长对不上另有原因
    for keep in _keep_candidates(gaps, need, PROBE_FLOOR):
        spans = trim_plan(segments, left, right, keep=keep)
        total = sum(hi - lo for lo, hi in spans)
        if abs(total - goal) <= tolerance:
            return spans if len(spans) > 1 else None
    return None                           # 反推不出来，不猜



def shift_time(keeps: Sequence[tuple[float, float]], moment: float) -> float | None:
    """原视频里的某一刻 → **剪完之后**的成片坐标（0 起）。

    `keeps` 是 `trim_plan()` 给的保留片段。剪掉多少、后面的东西该往前移多少，
    只在这里算一次 —— 挂字、空隙、任何跟着走的时间点都走这个函数。
    自己拿"剪掉的总秒数"去减，早晚会漏掉某一路，那就是成片里字和画面对不上。

    这一刻正好落在被剪掉的静音里 → 返回 None，调用方自己决定丢掉还是挪到边界。
    """
    passed = 0.0
    for lo, hi in keeps:
        if moment < lo - EPS:
            return None                   # 落在被剪掉的那一段空档里
        if moment <= hi + EPS:
            return round(passed + (moment - lo), 3)
        passed += hi - lo
    return None


def voiced_seconds(seg: Segment) -> float:
    """这一句**实际发声**多久（把句内静音刨掉）。

    句子的 `end - start` 会被句内停顿骗过去：「Thank God.」两个词加起来只发声
    0.44 秒，但中间夹了 3.70 秒静音，整句跨度就成了 4.14 秒。拿跨度去判断
    「这是不是一声短反应」会直接判错，所以按逐词累加。
    没有逐词数据才退回整句跨度。
    """
    if not seg.words:
        return max(0.0, seg.end - seg.start)
    return sum(max(0.0, w.end - w.start) for w in seg.words)


def voiced_between(segments: Iterable[Segment], start: float, end: float) -> float:
    """`[start, end]` 里**实际发声**多少秒（静音一律不算）。

    这是「有效时长」的唯一口径：判断一条素材够不够长、超不超上限都看它，
    不看 `end - start`。理由很简单 —— 静音是可以剪掉的（第二轮混剪会挑掉
    多余的干等），它不该占用这条素材的时长配额。

    实测过的极端例子：某条素材跨度 8.04 秒，里面 5.76 秒没人说话，
    真正发声只有 2.28 秒。按跨度算它逼近上限，按发声算它还很短。
    """
    total = 0.0
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        if seg.words:
            total += sum(max(0.0, min(w.end, end) - max(w.start, start))
                         for w in seg.words if w.end > start and w.start < end)
        else:
            total += max(0.0, min(seg.end, end) - max(seg.start, start))
    return round(total, 3)


def plan_from_span(segments: Sequence[Segment], frm: float, at: float, *,
                   min_sec: float = 3.0,
                   max_sec: float = 20.0,
                   max_voiced: float = 12.0) -> tuple[float, float, list[str]] | None:
    """两个时间点 → 一个落在句边界上的区间。返回 `(start, end, 说明)`。

    AI 只需要指两个位置：`frm` 这件事从哪开始（条件 / 铺垫），
    `at` 结果或反应发生在哪一刻。**区间由这里算**，所以：

      * `start` 必然是某一句的第一个词 —— 不会切进被静音打断的句子中间；
      * `end` 必然是某一句的最后一个词 —— 不会把结果或反应甩在区间外面；
      * 结果之后紧跟的短反应（语气词那种）自动收进来。

    两个点都允许指得不精确，落在句子里任何位置都会被吸到该句的边界上。
    段里一句都对不上（整段没有语音）就返回 None，交调用方决定。

    **两个上限管两件事，别混**：

      `max_voiced`  有效时长上限（发声秒数）—— 这是真正的考核。
                    中间静音多久都不影响，因为静音可以剪掉。
      `max_sec`     跨度硬上限 —— 只防极端，连静音一起算才用它。

    一个事件中间空了 8 秒但发声只有 4 秒，两个上限都不碰，整件事完整留下来；
    换成按跨度考核的老口径，它会被从前面砍掉铺垫，条件就没了。
    """
    notes: list[str] = []
    head = _segment_at(segments, frm)
    tail = _segment_at(segments, at)
    if tail is None:                     # 结果时刻落在没人说话的地方：取它之前最近一句
        before = [seg for seg in segments if seg.end <= at + EPS]
        tail = before[-1] if before else None
        if tail is not None:
            notes.append(f"结果时刻 {at:.2f} 落在静音里，锚到上一句（{tail.end:.2f} 结束）")
    if head is None:
        after = [seg for seg in segments if seg.start >= frm - EPS]
        head = after[0] if after else tail
        if head is not None and head is not tail:
            notes.append(f"起点 {frm:.2f} 落在静音里，顺到下一句（{head.start:.2f} 开口）")
    if head is None or tail is None:
        return None
    if head.start > tail.start:           # 两个点给反了，按时间顺序纠正
        head, tail = tail, head
        notes.append("frm 和 at 给反了，已按时间顺序纠正")

    chain = [seg for seg in segments if head.start - EPS <= seg.start <= tail.start + EPS]
    if not chain:
        chain = [tail]

    def _over(begin: float, finish: float) -> bool:
        """这个区间是不是已经超标了（发声超考核上限，或连静音一起超硬上限）。"""
        return (voiced_between(segments, begin, finish) > max_voiced
                or finish - begin > max_sec)

    # --- 往后并短反应：结果说完之后那一声「哦 / 好的 / 黄色」属于同一件事 ---
    end = chain[-1].end
    for seg in segments:
        if seg.start <= end + EPS:
            continue
        if seg.start - end > REACTION_GAP:
            break
        if voiced_seconds(seg) > REACTION_MAX_LEN:
            break                         # 是新的一句话，不是反应
        if _over(chain[0].start, seg.end):
            break
        notes.append(f"收进结果之后的短反应（到 {seg.end:.2f}）")
        chain.append(seg)
        end = seg.end

    # --- 超上限：从最前面砍铺垫，永远保住结果和反应那几句 ---
    while len(chain) > 1 and _over(chain[0].start, end):
        dropped = chain.pop(0)
        notes.append(f"超上限（发声 > {max_voiced:.2f}s 或跨度 > {max_sec:.2f}s），"
                     f"去掉最前面那句铺垫（{dropped.start:.2f} 起）")

    start = chain[0].start
    # --- 不足下限：往前再要一句（要不到就照实返回，让调用方决定） ---
    while end - start < min_sec:
        earlier = [seg for seg in segments if seg.end <= start + EPS]
        if not earlier:
            notes.append(f"不足 {min_sec:.2f}s，前面已经没有可并的句子")
            break
        prev = earlier[-1]
        if _over(prev.start, end):
            notes.append(f"不足 {min_sec:.2f}s，但再往前并就会超上限")
            break
        notes.append(f"不足 {min_sec:.2f}s，往前并一句（{prev.start:.2f} 起）")
        start = prev.start

    return round(max(start, 0.0), 3), round(end, 3), notes


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
    """从 AI JSON 里取出候选片段。协议只有一种，解析全在 `ai_protocol.clips`：

      {"timeline": {...}, "segments": [{"sa": 48.55, "end": 60.38, "dst": [...]}], ...}

    返回的是内部规范形状（`start` / `end` / `duration` / `score` / `type` / `reason`
    / `video` / `dst` + 原样带着的 o、s、w、a、e、t）。现阶段只取第一条 segment，
    多给的段丢掉并在 `dropped_segments` 里记数。文字内容一个字都不改。
    """
    return ai_protocol.clips(payload)


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


def _first_segment(payload: Any) -> dict[str, Any]:
    """报错时带上第一条 segment 的原文（一条都没有就是空 dict），方便看错在哪。"""
    items = ai_protocol.raw_segments(payload)
    return dict(items[0]) if items else {}


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

    found = clips_in_payload(payload)
    if not found:
        # 协议层就没抠出片段（缺 segments、sa/end 写坏、end<=sa……）。原因在
        # ai_protocol.validate 里是现成的，得带出来，别让日志只剩一句「没有可剪的片段」
        why = ai_protocol.validate(payload) or "JSON 里没有可用的 segments"
        return PlanResult(plans=(), rejected=((_first_segment(payload), why),))

    for clip in found:
        ai_start, ai_end = _num(clip.get("start")), _num(clip.get("end"))
        if ai_start is None or ai_end is None:
            rejected.append((clip, "segments[0].sa / segments[0].end 不是有效数字"))
            continue
        if ai_start < 0 or ai_end < 0:
            rejected.append((clip, f"时间不能是负数（sa={ai_start}, end={ai_end}）"))
            continue
        if ai_end <= ai_start:
            rejected.append((clip, f"segments[0].end({ai_end}) 必须大于 sa({ai_start})"))
            continue
        if video_duration is not None and ai_start >= video_duration:
            rejected.append((clip, f"起点 {ai_start} 已经超出视频时长 {video_duration}"))
            continue

        notes: list[str] = []
        dropped = int(clip.get("dropped_segments") or 0)
        if dropped:
            notes.append(f"JSON 里有 {dropped + 1} 段 segments，多段拼接还没做，"
                         f"这次只剪第一段")
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
        lines.append(f"[剪辑引擎] 已拒绝：{why}（原始 sa={clip.get('sa', clip.get('start'))!r}"
                     f" end={clip.get('end')!r}）")
    if not result.plans:
        lines.append("[剪辑引擎] 没有可剪的片段，不启动渲染")
    return lines


def first_clip_payload(payload: Any) -> dict[str, Any] | None:
    """把第一条 segment 单独折成一份协议 JSON，交给 `parse_spec` 定位源视频。

    时间和文案原样不动（`dst` 按 timeline 时长重算），真正的决策仍由 `plan_clips` 负责。
    """
    found = clips_in_payload(payload)
    if not found:
        return None
    clip = found[0]
    return ai_protocol.build_payload(clip, start=float(clip["start"]),
                                     end=float(clip["end"]),
                                     duration=_num(clip.get("duration")))


def payload_for(plan: ClipPlan) -> dict[str, Any]:
    """把修正后的时间写回协议形状，交给既有 `clip.parse_spec` 渲染。

    只改 `segments[0].sa` / `.end` / `timeline.duration`（`dst` 跟着重算成
    `[0, duration]`）；type、reason、t、o 这些文案全部原样带走，中文一个字都不动。
    末帧冻结是渲染时追加的，不写进 dst。
    """
    clip = dict(plan.raw)
    clip.setdefault("score", plan.score)
    clip.setdefault("type", plan.type)
    clip.setdefault("reason", plan.reason)
    if plan.source_video:
        clip["video"] = plan.source_video
    return ai_protocol.build_payload(clip, start=plan.start, end=plan.end,
                                     duration=plan.duration)
