"""混剪策略：评分权重 + 组合约束 + 搜索预算，整份可落库、可复用、可版本化。

一条硬边界（技术指导第二十一节第 4 条）：**LLM 可以产生策略，但最终选择必须由
确定性评分/搜索完成。** 所以这一层里只有数字和阈值，没有任何"让模型自己挑"的口子。
把策略做成数据而不是代码分支，正是为了让"AI 帮你调参"这件事安全 ——
模型能改的只是几个权重，改不了选择逻辑本身。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..db.db import Database
from ..logging_setup import get_logger
from . import material_repository as repo
from .material_score import DEFAULT_WEIGHTS
from .types import MontageStrategy

logger = get_logger("dance.strategy")

#: 默认组合约束。技术指导第十三节要求的那几项都在这里
DEFAULT_CONSTRAINTS: dict[str, Any] = {
    # 同一个人最多连着出现几次。1 = 绝不允许连着（卡点舞里连着同一个人最刺眼）
    "max_consecutive_same_person": 1,
    # 一个人最多占全片多少比例
    "max_person_share": 0.45,
    # 同一个源视频最多占全片多少比例
    "max_source_share": 0.50,
    # 同一条素材在一版里最多出现几次。1 = 不许重复
    "max_material_repeat": 1,
    # 相邻对历史上出现过多少次以上就直接拒（不是扣分，是拒）
    "max_pair_history": 3,
    # 整套组合和历史某一版的相似度上限，超过就拒
    "max_combination_similarity": 0.85,
}

#: 默认搜索预算。**必须**有上限，保证 Windows + RTX 3060 本地跑得动
DEFAULT_SEARCH: dict[str, Any] = {
    "candidate_k": 8,          # 每个位置只取前 K 个候选进搜索
    "beam_width": 6,           # beam 宽度
    "max_nodes": 20000,        # 搜索节点上限，到了就收工返回当前最好的
    "versions": 3,             # 一次生成几个版本
    "diversity_penalty": 0.35,  # 生成第 2/3 个版本时，对已出现过的素材额外扣分
}

def default_strategy() -> MontageStrategy:
    """一份开箱可用的默认策略。用户第一次打开界面时不该看到一堆 0 分。"""
    return MontageStrategy(
        name="默认策略", kind="hybrid", version="v1",
        weights=dict(DEFAULT_WEIGHTS), constraints=dict(DEFAULT_CONSTRAINTS),
        search=dict(DEFAULT_SEARCH), is_default=1,
        note="轮换优先 + 人物/来源多样性 + 历史去重。权重与约束都可改。")


PRESETS: dict[str, dict[str, Any]] = {
    # 冷启动：库里还没有历史，那些历史项全是 0，不如把权重压到对齐质量上
    "cold_start": {
        "name": "冷启动（新库）", "kind": "cold_start",
        "weights": {"alignment_confidence": 1.30, "quality": 0.70,
                    "never_output_bonus": 0.20, "never_used_bonus": 0.15,
                    "recent_use_penalty": 0.0, "recent_output_penalty": 0.0,
                    "position_usage_penalty": 0.0, "pair_penalty": 0.0,
                    "combination_penalty": 0.0},
        "note": "库里还没有使用历史时用它：只看对齐质量和多样性，不做历史去重。",
    },
    # 轮换优先：把没出过片的素材推上来，适合"素材攒了很多但只用过几个"的库
    "rotation": {
        "name": "轮换优先", "kind": "rule_based",
        "weights": {"never_output_bonus": 1.20, "never_used_bonus": 0.80,
                    "freshness": 0.90, "output_penalty": -0.90,
                    "recent_output_penalty": -1.10},
        "note": "尽量把从没出过片的素材推出去。适合素材多、出片少的库。",
    },
    # 求稳：只用高置信度素材，多样性权重降低，适合要交付的正式成片
    "safe": {
        "name": "求稳（高置信度）", "kind": "rule_based",
        "weights": {"alignment_confidence": 1.60, "quality": 0.80,
                    "person_diversity": 0.30, "source_diversity": 0.25},
        "constraints": {"max_consecutive_same_person": 2, "max_person_share": 0.60},
        "note": "只偏向对齐可信的素材，约束放松一点，适合要交付的正式成片。",
    },
    # 探索：加大随机扰动，用来"看看还有什么别的剪法"
    "exploration": {
        "name": "探索（多试新组合）", "kind": "exploration",
        "weights": {"never_used_bonus": 1.00, "freshness": 1.00,
                    "pair_penalty": -1.20, "combination_penalty": -1.20},
        "search": {"candidate_k": 12, "beam_width": 10, "versions": 5,
                   "diversity_penalty": 0.60},
        "note": "刻意避开历史上出现过的搭配，用来找新剪法。候选池和版本数都放大。",
    },
}


def preset(name: str) -> MontageStrategy:
    """按预设名造策略。预设只覆盖它关心的那几个权重，其余走默认值。"""
    if name not in PRESETS:
        raise ValueError(f"未知的策略预设 {name!r}，可选：{', '.join(sorted(PRESETS))}")
    spec = PRESETS[name]
    return MontageStrategy(
        name=str(spec["name"]), kind=str(spec["kind"]), version="v1",
        weights={**DEFAULT_WEIGHTS, **spec.get("weights", {})},
        constraints={**DEFAULT_CONSTRAINTS, **spec.get("constraints", {})},
        search={**DEFAULT_SEARCH, **spec.get("search", {})},
        note=str(spec.get("note", "")))


def resolve(db: Database, strategy_id: int = 0) -> MontageStrategy:
    """取要用的策略：指定了就用它，没指定用默认那份，库里一份都没有就现建一份。

    "库里没有就建一份"是有意的：用户第一次点"生成混剪"时不该先被要求去建策略。
    """
    if strategy_id:
        found = repo.get_strategy(db, strategy_id)
        if found is not None:
            return _filled(found)
        logger.warning("策略 #%d 不存在，退回默认策略", strategy_id)
    found = repo.default_strategy(db)
    if found is not None:
        return _filled(found)
    fresh = default_strategy()
    fresh.id = repo.save_strategy(db, fresh)
    logger.info("库里还没有策略，已建默认策略 #%d", fresh.id)
    return fresh


def _filled(strategy: MontageStrategy) -> MontageStrategy:
    """给从库里读出来的策略补齐缺省项。

    老策略只存了几个权重时，缺的那些必须回落到默认值而不是 0 ——
    权重为 0 意味着"这一项完全不看"，和"没配过"是两回事。
    """
    strategy.weights = {**DEFAULT_WEIGHTS, **(strategy.weights or {})}
    strategy.constraints = {**DEFAULT_CONSTRAINTS, **(strategy.constraints or {})}
    strategy.search = {**DEFAULT_SEARCH, **(strategy.search or {})}
    return strategy


def ensure_presets(db: Database) -> list[int]:
    """把四个预设策略灌进库（已存在同名的就跳过），返回新建的 id 列表。

    GUI 第一次打开时调一次，用户就有几档现成的可选，不用从零调权重。
    """
    existing = {s.name for s in repo.list_strategies(db, include_deleted=True)}
    created: list[int] = []
    default = default_strategy()
    if default.name not in existing:
        default.id = repo.save_strategy(db, default)
        created.append(default.id)
        existing.add(default.name)
    for key in PRESETS:
        candidate = preset(key)
        if candidate.name in existing:
            continue
        candidate.id = repo.save_strategy(db, candidate)
        created.append(candidate.id)
    return created


def describe(strategy: MontageStrategy) -> list[str]:
    """把策略写成中文行。只打**和默认值不同**的权重 —— 全打 19 项没人看。"""
    lines = [f"[策略] #{strategy.id} {strategy.name}（{strategy.kind} / {strategy.version}）"]
    diff = {k: v for k, v in (strategy.weights or {}).items()
            if abs(float(v) - float(DEFAULT_WEIGHTS.get(k, 0.0))) > 1e-9}
    lines.append("  权重改动：" + ("；".join(f"{k}={v:+.2f}" for k, v in sorted(diff.items()))
                                if diff else "（全部为默认值）"))
    search = strategy.search or {}
    lines.append(f"  搜索预算：候选 K={search.get('candidate_k')}"
                 f"｜beam={search.get('beam_width')}"
                 f"｜节点上限={search.get('max_nodes')}"
                 f"｜版本数={search.get('versions')}")
    constraints = strategy.constraints or {}
    lines.append(f"  组合约束：连续同人 <= {constraints.get('max_consecutive_same_person')}"
                 f"｜单人占比 <= {constraints.get('max_person_share')}"
                 f"｜单源占比 <= {constraints.get('max_source_share')}")
    if strategy.note:
        lines.append(f"  说明：{strategy.note}")
    return lines


__all__ = [
    "DEFAULT_CONSTRAINTS", "DEFAULT_SEARCH", "PRESETS",
    "default_strategy", "preset", "resolve", "ensure_presets", "describe",
]

