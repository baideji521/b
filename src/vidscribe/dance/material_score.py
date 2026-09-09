"""素材评分。技术指导第十节的 19 项因子，全部输出到 `score_breakdown`。

参考 BeatSync 的 `_score_candidate()` 的**思路**（多因子加权 + 近期使用惩罚），
但那是"一次性随机混剪"的评分，这里要的是**长期资产系统**的评分 ——
差别在于这里的每一项惩罚都来自数据库里真实的使用历史，而不是本次运行内的临时计数。

评分刻意拆成两半，这是让 beam search 跑得动的关键：

    静态分  static_score()    只看素材自己 + 历史 + 音乐位置，与"旁边选了谁"无关
    动态分  dynamic_delta()   只看"和已选的那些放在一起"，与素材自身属性无关

候选池只算静态分（每个位置几十条素材，算一遍就够）；组合搜索在扩展每个节点时
只补算动态分（很便宜）。如果混在一起，每换一个邻居就得把 19 项全部重算，
搜索规模直接乘上一个常数，本地机器就跑不动了。

**所有权重都可以被策略覆盖**（`MontageStrategy.weights`），但默认值必须自成一套
能用的方案 —— 用户第一次打开界面时不该看到一堆 0 分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from ..logging_setup import get_logger
from .types import DanceMaterial, MaterialScore

logger = get_logger("dance.score")

#: 默认权重。正数是加分项，负数是扣分项；每一项的原始值都归一到 0~1，
#: 所以权重之间可以直接比大小。
DEFAULT_WEIGHTS: dict[str, float] = {
    # --- 素材本身好不好 ---
    "position_match": 1.00,       # 素材是不是**就是**为这个音乐位置切出来的
    "alignment_confidence": 0.90,  # 对齐可信度：不可信的素材内容和音乐位置是错开的
    "quality": 0.40,              # 切片时算的先验质量
    # --- 轮换：优先用没用过的 ---
    "never_output_bonus": 0.60,   # 从没出过片 —— 这是资产系统最想推的一类
    "never_used_bonus": 0.35,     # 从没用过
    "freshness": 0.45,            # 越久没用越新鲜
    "use_penalty": -0.50,         # 用得越多越扣
    "montage_penalty": -0.30,
    "output_penalty": -0.45,      # 出过片的比只进过混剪的扣得更多
    "recent_use_penalty": -0.55,  # 最近刚用过，扣重一点（观众会认出来）
    "recent_output_penalty": -0.65,
    "position_usage_penalty": -0.40,  # 这条素材在**这个位置**上已经被用过
    # --- 音乐匹配（辅助，权重刻意小） ---
    "energy_match": 0.20,
    "impact_match": 0.15,
    "section_match": 0.10,        # 段落类型只是辅助特征，绝不主导选择
    # --- 用户反馈 ---
    "feedback": 0.35,
    # --- 组合层（动态，由 dynamic_delta 使用） ---
    "person_diversity": 0.50,
    "source_diversity": 0.40,
    "pair_penalty": -0.70,        # 这两条素材以前挨着出现过
    "combination_penalty": -0.60,  # 这一整套组合以前出现过
    "same_person_penalty": -0.85,  # 连续同一个人
}

#: "最近"的口径（天）。比这更久以前用过的就不算最近了
RECENT_DAYS = 7.0
#: 新鲜度饱和天数：超过这么久没用，freshness 就满分了
FRESH_SATURATION_DAYS = 60.0
#: 使用次数惩罚的饱和点：用了这么多次之后再多用也不会更扣
USE_SATURATION = 8.0
OUTPUT_SATURATION = 5.0

def _days_since(stamp: str, now: datetime) -> float | None:
    """距今多少天。解析不了返回 None（"没有记录"和"很久以前"不是一回事）。"""
    text = str(stamp or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return max(0.0, (now - datetime.strptime(text, fmt)).total_seconds() / 86400.0)
        except ValueError:
            continue
    return None


def _saturate(value: float, ceiling: float) -> float:
    """把计数压到 0~1。超过 ceiling 就当满 —— 用了 8 次和 80 次在体感上没差别。"""
    if ceiling <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(value)) / float(ceiling)), 4)


@dataclass
class ScoreContext:
    """评分要用的一切外部事实。**全部预先算好传进来**，评分函数自己不查库。

    这样做的两个理由：一是评分能在测试里用纯字典驱动、不需要数据库；
    二是组合搜索要对同一批素材算成千上万次分，每次都回库查会慢几个数量级。
    """

    now: datetime = field(default_factory=datetime.now)
    #: 音乐位置的能量 / 冲击力 / 段落类型：`{segment_index: 值}`
    energy_at: Mapping[int, float] = field(default_factory=dict)
    impact_at: Mapping[int, float] = field(default_factory=dict)
    section_at: Mapping[int, str] = field(default_factory=dict)
    #: 历史：这条素材在**这个位置**上被用过几次 `{(segment_index, material_id): 次数}`
    position_usage: Mapping[tuple[int, int], int] = field(default_factory=dict)
    #: 历史：两条素材挨着出现过几次 `{(前, 后): 次数}`，无向（两个方向都记）
    pair_usage: Mapping[tuple[int, int], int] = field(default_factory=dict)
    #: 历史：整套组合签名出现过几次 `{signature: 次数}`
    combination_usage: Mapping[str, int] = field(default_factory=dict)
    #: 历史：这个人 / 这个源出过多少片，用来算多样性基线
    person_usage: Mapping[str, int] = field(default_factory=dict)
    source_usage: Mapping[int, int] = field(default_factory=dict)
    #: 用户反馈 `{material_id: {verdict: 次数}}`
    feedback: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    #: 段落类型 → 偏好分 0~1。留空就不看段落（默认就是留空，段落只是辅助特征）
    section_preference: Mapping[str, float] = field(default_factory=dict)

    def energy(self, segment_index: int) -> float:
        return float(self.energy_at.get(int(segment_index), 0.0) or 0.0)

    def impact(self, segment_index: int) -> float:
        return float(self.impact_at.get(int(segment_index), 0.0) or 0.0)

    def section(self, segment_index: int) -> str:
        return str(self.section_at.get(int(segment_index), "") or "")

def static_score(material: DanceMaterial, segment_index: int, context: ScoreContext,
                 weights: Mapping[str, float] | None = None) -> MaterialScore:
    """素材在某个音乐位置上的**静态**得分（与旁边选了谁无关）。

    `breakdown` 里每一项都是"已经乘过权重"的贡献值，加起来正好等于 `final_score` ——
    这样用户在界面上看到的每一行数字都能直接回答"这一项帮了/害了多少分"，
    不需要心算权重。原始的 0~1 特征值另放在 `raw_*` 键里。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    now = context.now
    parts: dict[str, float] = {}
    raw: dict[str, float] = {}

    # 1. 位置匹配。素材是按目标位置切出来的，所以正常都是 1.0；
    #    不等就是调用方拿错了素材，给 0 让它自然沉底而不是崩掉
    raw["position_match"] = 1.0 if int(material.segment_index) == int(segment_index) else 0.0
    # 2. 对齐可信度
    raw["alignment_confidence"] = max(0.0, min(1.0, float(material.alignment_confidence)))
    # 3. 先验质量
    raw["quality"] = max(0.0, min(1.0, float(material.quality or 0.0)))

    # 4~9. 轮换：没用过的、很久没用的优先；用得多的、刚用过的扣分
    raw["never_output_bonus"] = 1.0 if material.never_output else 0.0
    raw["never_used_bonus"] = 1.0 if material.never_used else 0.0
    raw["use_penalty"] = _saturate(material.use_count, USE_SATURATION)
    raw["montage_penalty"] = _saturate(material.montage_count, USE_SATURATION)
    raw["output_penalty"] = _saturate(material.output_count, OUTPUT_SATURATION)

    used_days = _days_since(material.last_used_at, now)
    output_days = _days_since(material.last_output_at, now)
    if used_days is None:
        raw["freshness"] = 1.0                 # 从没用过 = 最新鲜
        raw["recent_use_penalty"] = 0.0
    else:
        raw["freshness"] = _saturate(used_days, FRESH_SATURATION_DAYS)
        raw["recent_use_penalty"] = round(max(0.0, 1.0 - used_days / RECENT_DAYS), 4)
    raw["recent_output_penalty"] = (
        0.0 if output_days is None else round(max(0.0, 1.0 - output_days / RECENT_DAYS), 4))

    # 10. 这条素材在**这个位置**上用过几次（换个位置用同一条素材，观感上是新的）
    seen_here = int(context.position_usage.get((int(segment_index), int(material.id)), 0))
    raw["position_usage_penalty"] = _saturate(seen_here, 3.0)

    # 11~13. 音乐匹配。能量高的位置偏好高能量素材（用 quality 当代理），
    #        段落只有在策略明确给了偏好表时才起作用
    raw["energy_match"] = round(context.energy(segment_index) * raw["quality"], 4)
    raw["impact_match"] = round(context.impact(segment_index) * raw["alignment_confidence"], 4)
    raw["section_match"] = float(
        context.section_preference.get(context.section(segment_index), 0.0) or 0.0)

    # 14. 用户反馈：采纳过加分、否决过扣分，都压到 -1~1 再乘权重
    votes = context.feedback.get(int(material.id), {}) or {}
    accepted = int(votes.get("accepted", 0))
    rejected = int(votes.get("rejected", 0)) + int(votes.get("replaced", 0))
    total_votes = accepted + rejected
    raw["feedback"] = (round((accepted - rejected) / total_votes, 4) if total_votes else 0.0)

    for key, value in raw.items():
        parts[key] = round(float(w.get(key, 0.0)) * float(value), 6)
    final = round(sum(parts.values()), 6)
    breakdown = dict(parts)
    breakdown.update({f"raw_{k}": round(v, 4) for k, v in raw.items()})
    breakdown["final_score"] = final
    return MaterialScore(material_id=int(material.id), segment_index=int(segment_index),
                         final_score=final, breakdown=breakdown)

def combination_signature(material_ids: Iterable[int]) -> str:
    """一套组合的签名。**顺序无关**（排序后再拼）—— 把同一批素材换个顺序
    当成"新组合"是自欺欺人，观众看到的还是那几个人那几段。

    用作 `dance_montage_versions.signature`，也是 `combination_usage` 的键。
    """
    ids = sorted({int(i) for i in material_ids})
    return "-".join(str(i) for i in ids)


def pair_key(first: int, second: int) -> tuple[int, int]:
    """相邻对的键。无向：`(A,B)` 和 `(B,A)` 是同一件事。"""
    a, b = int(first), int(second)
    return (a, b) if a <= b else (b, a)


def dynamic_delta(material: DanceMaterial, context: ScoreContext,
                  chosen: Sequence[DanceMaterial],
                  weights: Mapping[str, float] | None = None) -> tuple[float, dict[str, float]]:
    """把这条素材接在 `chosen` 后面的**增量**得分，返回 `(delta, breakdown)`。

    只看四件和"邻居"有关的事，全都来自技术指导第十三节的组合约束：

    - `same_person_penalty`  紧邻同一个人 —— 观感上最刺眼的重复，扣最重
    - `person_diversity`     这个人在本次组合里出现得越少越加分
    - `source_diversity`     同一源视频反复出现要扣（同一个人也可能有多个源）
    - `pair_penalty`         这两条素材**历史上**挨着出现过
    - `combination_penalty`  加上它之后的整套组合**历史上**出现过

    `chosen` 为空（第一个位置）时只有历史项起作用。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts: dict[str, float] = {}
    person = str(material.person or "")
    used = len(chosen)

    same_person = 0.0
    if chosen and person and str(chosen[-1].person or "") == person:
        same_person = 1.0
    parts["same_person_penalty"] = round(float(w["same_person_penalty"]) * same_person, 6)

    if used and person:
        appeared = sum(1 for c in chosen if str(c.person or "") == person)
        # 出现比例越低越加分：0 次 → 1.0，占满 → 0.0
        parts["person_diversity"] = round(
            float(w["person_diversity"]) * (1.0 - appeared / float(used)), 6)
    else:
        parts["person_diversity"] = round(float(w["person_diversity"]), 6)

    if used:
        same_source = sum(1 for c in chosen
                          if int(c.source_video_id) == int(material.source_video_id))
        parts["source_diversity"] = round(
            float(w["source_diversity"]) * (1.0 - same_source / float(used)), 6)
    else:
        parts["source_diversity"] = round(float(w["source_diversity"]), 6)

    pair_seen = 0
    if chosen:
        pair_seen = int(context.pair_usage.get(pair_key(chosen[-1].id, material.id), 0))
    parts["pair_penalty"] = round(
        float(w["pair_penalty"]) * _saturate(pair_seen, 3.0), 6)

    signature = combination_signature([c.id for c in chosen] + [material.id])
    combo_seen = int(context.combination_usage.get(signature, 0))
    parts["combination_penalty"] = round(
        float(w["combination_penalty"]) * _saturate(combo_seen, 2.0), 6)

    return round(sum(parts.values()), 6), parts


def merge_breakdown(static: MaterialScore, delta: float,
                    parts: Mapping[str, float]) -> MaterialScore:
    """把动态项并进静态 breakdown，得到最终这一格的完整解释。"""
    from dataclasses import replace  # noqa: PLC0415

    breakdown = dict(static.breakdown)
    breakdown.update({k: round(float(v), 6) for k, v in parts.items()})
    final = round(static.final_score + float(delta), 6)
    breakdown["final_score"] = final
    return replace(static, final_score=final, breakdown=breakdown)


def explain(score: MaterialScore, limit: int = 6) -> str:
    """把 breakdown 写成一行中文，给日志和界面 tooltip 用。

    只挑绝对值最大的几项：19 项全打出来没人看，"是哪三件事决定的"才有用。
    """
    items = [(k, v) for k, v in score.breakdown.items()
             if not k.startswith("raw_") and k != "final_score"]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    head = "、".join(f"{k} {v:+.3f}" for k, v in items[:max(1, limit)] if abs(v) > 1e-9)
    return f"素材 #{score.material_id} @位置{score.segment_index} 得分 {score.final_score:.3f}（{head}）"


__all__ = [
    "DEFAULT_WEIGHTS", "RECENT_DAYS", "FRESH_SATURATION_DAYS",
    "USE_SATURATION", "OUTPUT_SATURATION",
    "ScoreContext", "static_score", "dynamic_delta", "merge_breakdown",
    "combination_signature", "pair_key", "explain",
]



