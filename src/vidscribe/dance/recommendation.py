"""推荐：从候选池里挑出"这一格建议用哪几条"，结果**必须可重现**。

技术指导第十一节的五种策略全部实现：

    cold_start    库里还没有使用历史 → 只看对齐质量与先验质量，不做历史去重
    rule_based    纯按静态评分排，不加任何随机性 —— 同样的库永远给同样的答案
    history_based 加重历史项（用户反馈、位置使用、轮换），"用过的往后排"
    exploration   加大稳定扰动，故意翻出平时排不上来的素材
    hybrid        rule_based + 轻微扰动 + 历史，默认档

可重现怎么做到的（三件事缺一不可）：
1. 随机数一律走 `combination_search.stable_rng(seed, *parts)`，**不用全局 `random`**
2. `random_seed` / 策略 id / 策略版本 / 算法版本全部落库
3. 排序的比较键里带上 `material_id` 兜底，绝不依赖字典或集合的遍历顺序

技术指导第二十一节第 3 条明确禁止拿 BeatSync 那套"随机素材选择"替代历史资产推荐 ——
所以这里的"随机"只用于**打破平分**和 exploration 档的探索，never 用于决定谁被选中。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from . import RECOMMENDATION_ALGORITHM_VERSION
from . import material_repository as repo
from . import material_selection, strategy as strategy_mod
from .combination_search import stable_rng
from .material_score import ScoreContext, explain
from .types import (
    CandidatePool,
    FilterSpec,
    MaterialScore,
    MontageStrategy,
    RecommendationItem,
    RecommendationRun,
)

logger = get_logger("dance.recommend")

#: 每档策略的扰动幅度。rule_based 严格为 0 —— 它的全部价值就是"完全确定"
JITTER: dict[str, float] = {
    "cold_start": 0.00,
    "rule_based": 0.00,
    "history_based": 0.02,
    "hybrid": 0.05,
    "exploration": 0.35,
}

#: 每档策略在默认权重之上的覆盖。只列**这一档特有**的偏好
KIND_WEIGHTS: dict[str, dict[str, float]] = {
    "cold_start": {"alignment_confidence": 1.30, "quality": 0.70,
                   "position_usage_penalty": 0.0, "recent_use_penalty": 0.0,
                   "recent_output_penalty": 0.0, "feedback": 0.0},
    "rule_based": {},
    "history_based": {"feedback": 0.80, "position_usage_penalty": -0.70,
                      "recent_use_penalty": -0.80, "recent_output_penalty": -0.95,
                      "never_output_bonus": 0.85},
    "exploration": {"never_used_bonus": 0.90, "freshness": 0.90},
    "hybrid": {},
}

def weights_for(kind: str, base: Mapping[str, float] | None = None) -> dict[str, float]:
    """某一档策略实际用的权重 = 默认值 ← 策略自定义 ← 这一档的特有偏好。

    顺序是有讲究的：档位偏好放最后，所以选了 `cold_start` 就一定不看历史，
    哪怕策略里把 `recent_use_penalty` 调得很重 —— 冷启动时那一项本来就没有数据。
    """
    from .material_score import DEFAULT_WEIGHTS  # noqa: PLC0415

    merged = {**DEFAULT_WEIGHTS, **(base or {})}
    merged.update(KIND_WEIGHTS.get(kind, {}))
    return merged


def rank_pool(pool: CandidatePool, kind: str, seed: int, top_n: int) -> list[MaterialScore]:
    """给一个位置的候选排序并取前 N。

    排序键 `(-分数, material_id)`：分数相同时按 id 定序，**绝不**依赖
    列表/字典的原始顺序 —— 那样换一次 SQL 的 ORDER BY 结果就变了，可重现性就没了。
    """
    jitter = float(JITTER.get(kind, 0.0))
    ranked: list[tuple[float, int, MaterialScore]] = []
    for scored in pool.scored:
        noise = (stable_rng(seed, "rank", pool.segment_index, scored.material_id) * jitter
                 if jitter > 0 else 0.0)
        ranked.append((-(scored.final_score + noise), int(scored.material_id), scored))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max(1, int(top_n))]]


def recommend(db: Database, target_song_id: int, segment_indexes: Sequence[int], *,
              strategy: MontageStrategy | None = None, seed: int = 0, top_n: int = 5,
              spec: FilterSpec | None = None, montage_id: int = 0,
              persist: bool = True, context: ScoreContext | None = None,
              ) -> RecommendationRun:
    """跑一次推荐。默认落库（`persist=False` 用于界面预览，不留痕）。

    `seed=0` 时按 `(歌id, 策略id, 位置数)` 派生一个**确定的** seed ——
    这样"不填 seed"也仍然可重现，而不是每次都不一样。真要随机就自己传一个变化的 seed。
    """
    plan = strategy or strategy_mod.resolve(db)
    kind = str(plan.kind or "hybrid")
    if kind not in JITTER:
        logger.warning("未知策略档位 %r，按 hybrid 处理", kind)
        kind = "hybrid"
    if not seed:
        seed = int(stable_rng(0, target_song_id, plan.id, len(segment_indexes)) * 1_000_000)

    weights = weights_for(kind, plan.weights)
    ctx = context or material_selection.score_context(db, target_song_id)
    pools = [material_selection.build_pool(db, target_song_id, index, spec=spec,
                                           context=ctx, weights=weights)
             for index in sorted(set(int(i) for i in segment_indexes))]

    items: list[RecommendationItem] = []
    candidates = 0
    for pool in pools:
        candidates += len(pool.scored)
        for rank, scored in enumerate(rank_pool(pool, kind, seed, top_n), 1):
            material = pool.materials.get(scored.material_id)
            reason = explain(scored, limit=3)
            items.append(RecommendationItem(
                segment_index=pool.segment_index, material_id=scored.material_id,
                rank=rank, score=scored.final_score, breakdown=dict(scored.breakdown),
                reason=(f"{material.person or '(未标注)'}｜{reason}"
                        if material is not None else reason)))

    run = RecommendationRun(
        target_song_id=int(target_song_id), montage_id=int(montage_id),
        strategy_id=int(plan.id), strategy_kind=kind,
        strategy_version=str(plan.version or ""), random_seed=int(seed),
        candidate_count=candidates, recommended_count=len(items),
        algorithm_version=RECOMMENDATION_ALGORITHM_VERSION,
        created_at=repo.now(), items=tuple(items),
        notes=f"{len(pools)} 个位置，每格取前 {top_n}")
    if persist:
        run_id = repo.save_recommendation_run(db, run, spec.to_dict() if spec else None)
        from dataclasses import replace  # noqa: PLC0415

        run = replace(run, id=run_id)
    logger.info("推荐 #%d：策略 %s（seed=%d）｜%d 个位置｜候选 %d 条｜推荐 %d 条",
                run.id, kind, seed, len(pools), candidates, len(items))
    return run

def top_pick(run: RecommendationRun) -> dict[int, int]:
    """`{位置: rank 1 的素材 id}`。「一键采纳推荐」就是它。"""
    out: dict[int, int] = {}
    for item in run.items:
        if item.rank == 1:
            out[int(item.segment_index)] = int(item.material_id)
    return out


def items_for(run: RecommendationRun, segment_index: int) -> list[RecommendationItem]:
    """某个位置的推荐条目，按 rank 升序。界面上一格的下拉列表就是它。"""
    return sorted((i for i in run.items if int(i.segment_index) == int(segment_index)),
                  key=lambda i: i.rank)


def reproduce(db: Database, run_id: int, *, spec: FilterSpec | None = None,
              top_n: int = 5) -> tuple[RecommendationRun, RecommendationRun, bool]:
    """重跑一次历史推荐，返回 `(原始, 重跑, 是否一致)`。

    这是"推荐可重现"这条要求的**可执行证明** —— CLI 的 `dance-montage recommend
    --verify <run_id>` 直接调它。库状态变了（新切了素材、用过几次）结果当然会不同，
    所以不一致不一定是 bug；但库没动过时它必须一致，测试正是这么验的。
    """
    original = repo.get_recommendation_run(db, run_id)
    if original is None:
        raise ValueError(f"推荐记录 #{run_id} 不存在")
    plan = strategy_mod.resolve(db, original.strategy_id)
    positions = sorted({int(i.segment_index) for i in original.items})
    again = recommend(db, original.target_song_id, positions, strategy=plan,
                      seed=original.random_seed, top_n=top_n, spec=spec, persist=False)
    same = _same_ranking(original, again)
    return original, again, same


def _same_ranking(one: RecommendationRun, two: RecommendationRun) -> bool:
    """只比 `(位置, rank, 素材id)` 三元组序列 —— 分数会因为浮点尾数差一点，
    但"谁排第几"必须一模一样。"""
    def shape(run: RecommendationRun) -> list[tuple[int, int, int]]:
        return sorted((int(i.segment_index), int(i.rank), int(i.material_id))
                      for i in run.items)

    return shape(one) == shape(two)


def describe(run: RecommendationRun, limit: int = 10) -> list[str]:
    """把推荐结果写成中文行。"""
    lines = [f"[推荐] #{run.id}｜策略 {run.strategy_kind}"
             f"（{run.strategy_version or '-'}）｜seed {run.random_seed}"
             f"｜算法 {run.algorithm_version}",
             f"  候选 {run.candidate_count} 条 → 推荐 {run.recommended_count} 条"
             f"｜{run.created_at}"]
    for item in run.items[:limit]:
        lines.append(f"  位置 {item.segment_index:>3} #{item.rank}  素材 "
                     f"#{item.material_id:<5} 分 {item.score:+.3f}  {item.reason}")
    if len(run.items) > limit:
        lines.append(f"  …… 还有 {len(run.items) - limit} 条")
    return lines


__all__ = [
    "JITTER", "KIND_WEIGHTS",
    "weights_for", "rank_pool", "recommend", "top_pick", "items_for",
    "reproduce", "describe",
]


