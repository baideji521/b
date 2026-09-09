"""组合搜索：Top-K 候选池 + beam search + 有界回溯。**绝不做全排列。**

技术指导第十三节的要求：不能简单地"每个位置各取最高分"，因为那样会连着出现同一个人、
把历史上出现过的搭配又拼一遍。但也不能穷举 —— 12 个位置 × 每个位置 30 条候选
就是 30^12，本地机器算到明天也算不完。

所以走 beam search，三个预算全部有硬上限（都可由策略覆盖）：

    candidate_k   每个位置只看前 K 条（8）
    beam_width    同时保留几条部分解（6）
    max_nodes     总扩展节点上限（20000），到了就收工返回当前最好的

约束分两类，这个区分很重要：

    **硬约束**（violates）→ 直接拒，不进 beam。比如"连着同一个人"。
    **软惩罚**（dynamic_delta）→ 扣分但仍可选。比如"这个人已经出现两次了"。

把该硬的做成软的，结果就是分数一高就破功；把该软的做成硬的，
素材不够时会直接搜不出解。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..logging_setup import get_logger
from .material_score import ScoreContext, dynamic_delta, merge_breakdown, pair_key
from .types import CandidatePool, DanceMaterial, MaterialScore, MontageStrategy

logger = get_logger("dance.search")


def stable_rng(seed: int, *parts: Any) -> float:
    """稳定的伪随机数 0~1。**不用全局 `random`。**

    同样的 `(seed, parts)` 永远得到同样的值，所以推荐结果可重现
    （技术指导第十一节的硬要求）。用 sha1 而不是 `random.Random(seed)` 的原因是
    这里需要"按内容取值"而不是"按调用顺序取值"—— 后者一改遍历顺序结果就变了。
    """
    payload = "|".join(str(p) for p in (seed, *parts)).encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


@dataclass
class SearchResult:
    """一次组合搜索的结果。`complete=False` 表示有位置没填上（素材不够）。"""

    picks: list[MaterialScore] = field(default_factory=list)
    materials: list[DanceMaterial] = field(default_factory=list)
    score: float = 0.0
    nodes: int = 0
    complete: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "nodes": self.nodes,
                "complete": self.complete, "clips": len(self.picks),
                "notes": list(self.notes)}

@dataclass
class _State:
    """beam 里的一条部分解。"""

    score: float = 0.0
    picks: tuple[MaterialScore, ...] = ()
    materials: tuple[DanceMaterial, ...] = ()

    @property
    def key(self) -> tuple[int, ...]:
        """去重键：选了哪几条素材（**按顺序**）。顺序不同就是不同的解。"""
        return tuple(m.id for m in self.materials)


def violates(material: DanceMaterial, chosen: Sequence[DanceMaterial],
             context: ScoreContext, constraints: Mapping[str, Any],
             total_positions: int) -> str:
    """硬约束检查。通过返回空串，否则返回中文原因（会进 notes，方便排查"为什么没选它"）。

    占比类约束（人物 / 来源）在搜索**过程中**就按上限帧数折算成硬上限来判，
    而不是等拼完再检查：等拼完再拒等于白搜一遍，而且 beam 里可能一条合法解都不剩。
    """
    person = str(material.person or "")
    # 1. 连续同一个人
    limit = int(constraints.get("max_consecutive_same_person", 1) or 1)
    if person and limit >= 1:
        run = 0
        for previous in reversed(chosen):
            if str(previous.person or "") == person:
                run += 1
            else:
                break
        if run >= limit:
            return f"连续同一个人（{person}）已达上限 {limit}"
    # 2. 同一条素材重复
    repeat_limit = int(constraints.get("max_material_repeat", 1) or 1)
    if sum(1 for c in chosen if int(c.id) == int(material.id)) >= repeat_limit:
        return f"素材 #{material.id} 重复次数已达上限 {repeat_limit}"
    # 3. 单人占比
    share = float(constraints.get("max_person_share", 1.0) or 1.0)
    if person and 0 < share < 1.0:
        cap = max(1, int(share * max(1, total_positions)))
        if sum(1 for c in chosen if str(c.person or "") == person) >= cap:
            return f"{person} 的占比已达上限 {share:.0%}（{cap} 格）"
    # 4. 单源占比
    share = float(constraints.get("max_source_share", 1.0) or 1.0)
    if 0 < share < 1.0:
        cap = max(1, int(share * max(1, total_positions)))
        same = sum(1 for c in chosen
                   if int(c.source_video_id) == int(material.source_video_id))
        if same >= cap:
            return f"源视频 #{material.source_video_id} 的占比已达上限 {share:.0%}"
    # 5. 相邻对历史出现次数
    pair_cap = int(constraints.get("max_pair_history", 999) or 999)
    if chosen:
        seen = int(context.pair_usage.get(pair_key(chosen[-1].id, material.id), 0))
        if seen >= pair_cap:
            return f"这一对以前已经挨着出现过 {seen} 次，超过上限 {pair_cap}"
    return ""

def search(pools: Sequence[CandidatePool], context: ScoreContext, strategy: MontageStrategy,
           *, seed: int = 0, avoid: Mapping[int, float] | None = None) -> SearchResult:
    """beam search 找一套组合。返回得分最高的那条完整（或尽可能完整的）解。

    `avoid` 是"这一轮想避开的素材" `{material_id: 额外扣分}` ——
    生成第 2、第 3 个版本时把第 1 版用过的素材传进来，就能得到明显不同的版本，
    而不需要另写一套"多版本"逻辑。

    `seed` 只用于**打破平分**（`stable_rng`），不用于随机选择：
    同样的输入 + 同样的 seed 必须得到同样的结果（技术指导第十一节）。
    平分时不引入扰动会让结果完全由字典/id 顺序决定，那样多版本之间会长得一模一样。
    """
    budget = strategy.search or {}
    constraints = strategy.constraints or {}
    weights = strategy.weights or {}
    candidate_k = max(1, int(budget.get("candidate_k", 8) or 8))
    beam_width = max(1, int(budget.get("beam_width", 6) or 6))
    max_nodes = max(100, int(budget.get("max_nodes", 20000) or 20000))
    penalties = dict(avoid or {})

    ordered = sorted(pools, key=lambda p: p.segment_index)
    filled = [p for p in ordered if p.scored]
    result = SearchResult()
    for pool in ordered:
        if not pool.scored:
            result.complete = False
            result.notes.append(f"位置 #{pool.segment_index} 没有候选素材，这一格空着")
    if not filled:
        result.notes.append("所有位置都没有候选素材，搜不出任何组合")
        return result

    total_positions = len(filled)
    beam: list[_State] = [_State()]
    nodes = 0
    stopped = False

    for pool in filled:
        candidates = pool.head(candidate_k)
        nxt: list[_State] = []
        seen: set[tuple[int, ...]] = set()
        rejected: list[str] = []
        for state in beam:
            for scored in candidates:
                material = pool.materials.get(scored.material_id)
                if material is None:
                    continue
                nodes += 1
                if nodes > max_nodes:
                    stopped = True
                    break
                why = violates(material, state.materials, context, constraints,
                               total_positions)
                if why:
                    rejected.append(f"位置 #{pool.segment_index} 拒了素材 "
                                   f"#{material.id}：{why}")
                    continue
                delta, parts = dynamic_delta(material, context, state.materials, weights)
                merged = merge_breakdown(scored, delta, parts)
                extra = float(penalties.get(int(material.id), 0.0))
                # 平分时用稳定随机数抖动一丝：量级 1e-4，不会翻转真实的分数差距，
                # 但能让不同 seed 给出不同的（同样合法的）组合
                jitter = stable_rng(seed, pool.segment_index, material.id) * 1e-4
                total = round(state.score + merged.final_score - extra + jitter, 6)
                fresh = _State(score=total,
                               picks=state.picks + (merged,),
                               materials=state.materials + (material,))
                if fresh.key in seen:
                    continue
                seen.add(fresh.key)
                nxt.append(fresh)
            if stopped:
                break
        if not nxt:
            # 这个位置一条都合法不了：整格跳过，并把原因带出去
            result.complete = False
            result.notes.append(f"位置 #{pool.segment_index} 在当前约束下无解，这一格空着")
            result.notes.extend(rejected[:3])
            continue
        nxt.sort(key=lambda s: -s.score)
        beam = nxt[:beam_width]
        if stopped:
            result.notes.append(f"搜索到达节点上限 {max_nodes}，返回当前最好的组合")
            break

    best = max(beam, key=lambda s: s.score)
    result.picks = list(best.picks)
    result.materials = list(best.materials)
    result.score = best.score
    result.nodes = nodes
    if len(result.picks) < total_positions:
        result.complete = False
    logger.info("组合搜索：%d/%d 格填上，得分 %.3f，扩展 %d 个节点",
                len(result.picks), len(ordered), result.score, nodes)
    return result

def search_versions(pools: Sequence[CandidatePool], context: ScoreContext,
                    strategy: MontageStrategy, *, count: int = 0,
                    seed: int = 0) -> list[SearchResult]:
    """连续搜出 `count` 套**互不相同**的组合，这就是"多版本混剪"。

    做法是每搜出一版，就把它用过的素材放进下一轮的 `avoid` 里额外扣分。
    扣分而不是禁用：素材少的时候禁用会直接搜不出第二版，而扣分只是"尽量换掉"，
    实在没有替代品时仍然可以复用。

    去重靠"素材集合完全相同"判：真的搜不出新组合时提前收工，返回已有的几版
    （给用户三个一模一样的版本，比给一个更糟）。
    """
    budget = strategy.search or {}
    want = max(1, int(count or budget.get("versions", 3) or 3))
    step = float(budget.get("diversity_penalty", 0.35) or 0.35)
    out: list[SearchResult] = []
    avoid: dict[int, float] = {}
    seen: set[tuple[int, ...]] = set()

    for index in range(want):
        result = search(pools, context, strategy, seed=seed + index * 1013, avoid=avoid)
        if not result.picks:
            break
        fingerprint = tuple(sorted(m.id for m in result.materials))
        if fingerprint in seen:
            logger.info("第 %d 版和已有版本完全相同，提前收工（共 %d 版）", index + 1, len(out))
            break
        seen.add(fingerprint)
        out.append(result)
        for material in result.materials:
            avoid[int(material.id)] = avoid.get(int(material.id), 0.0) + step
    return out


def describe(result: SearchResult, limit: int = 12) -> list[str]:
    """把搜索结果写成中文行，CLI 与界面共用。"""
    lines = [f"[组合] {len(result.picks)} 格｜总分 {result.score:.3f}"
             f"｜扩展 {result.nodes} 个节点"
             + ("" if result.complete else "｜**有位置没填上**")]
    for pick, material in zip(result.picks[:limit], result.materials[:limit]):
        person = material.person or "(未标注)"
        lines.append(f"  位置 {pick.segment_index:>3}  素材 #{material.id:<5}"
                     f" {person:<12} 分 {pick.final_score:+.3f}")

    if len(result.picks) > limit:
        lines.append(f"  …… 还有 {len(result.picks) - limit} 格")
    lines.extend(f"  说明：{note}" for note in result.notes[:5])
    return lines


__all__ = ["stable_rng", "SearchResult", "violates", "search", "search_versions", "describe"]



