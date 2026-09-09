"""多维筛选 + 候选池。技术指导第十二节的全部筛选维度与六种排序。

分层：`FilterSpec`（声明"要什么"）→ 这里翻成 SQL → `material_score` 打分 →
`CandidatePool`（一个音乐位置一组候选）。

一条约定：**进候选池不等于用过**。这里只记 `candidate` 事件，
`use_count` 要等真的进了 montage 才 +1（技术指导第九节）。
所以 `build_pool()` 默认**不写任何事件** —— 写不写由调用方显式决定，
避免"用户在界面上翻了几页素材，库里的使用次数就涨了"这种荒唐事。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from . import material_repository as repo
from .material_score import ScoreContext, static_score
from .types import CandidatePool, DanceMaterial, FilterSpec, MaterialScore

logger = get_logger("dance.select")

#: 排序键 → SQL ORDER BY 片段。`score_desc` 不在这里 —— 得分是内存里算的，
#: SQL 排不了，所以那一档由 Python 排
_ORDER_SQL: dict[str, str] = {
    "use_count_asc": "m.use_count ASC, m.output_count ASC, m.id ASC",
    "output_count_asc": "m.output_count ASC, m.use_count ASC, m.id ASC",
    # NULL（从没用过）排最前：那正是"最该拿出来用"的一批
    "last_used_at_asc": "m.last_used_at IS NOT NULL, m.last_used_at ASC, m.id ASC",
    "confidence_desc": "m.alignment_confidence DESC, m.id ASC",
    "freshness_desc": "m.last_used_at IS NOT NULL, m.last_used_at ASC, m.use_count ASC",
    # `score_desc` 的真正排序在 Python 里（得分要读历史，SQL 排不了）。
    # 但这条 SQL 仍然决定**哪一批行会被 LIMIT 留下来**，所以绝不能写成 `m.id ASC`：
    # 那等于"库里最早入库的 N 条"，一旦某个位置的素材超过 limit，
    # 新切进来的、一次没用过的素材会在打分之前就被悄悄扔掉（候选池过早截断）。
    # 这里按"最该被考虑"预取：没出过片 → 没用过 → 出片少 → 用得少 → 对齐好。
    "score_desc": ("m.output_count ASC, m.use_count ASC, "
                   "m.alignment_confidence DESC, m.id ASC"),
}

#: `FilterSpec.limit` 的兜底值。这是**预取上限**而不是候选池上限：
#: 打分和组合搜索都在这一批之内进行，所以宁可大一点。
#: 真正进 beam search 的是 `strategy.search["candidate_k"]`。
DEFAULT_POOL_LIMIT = 500


#: 一期第三十三节的七个筛选方案（A~G）。GUI 直接照这个建下拉，
#: 用户点一下就是一整套条件，不用自己拼
PRESETS: dict[str, dict[str, Any]] = {
    "never_output": {"label": "A 未出片优先", "never_output": True},
    "never_used": {"label": "B 从未使用", "never_used": True},
    "low_use": {"label": "C 低频使用（<= N 次）", "max_use_count": 2},
    "long_unused": {"label": "D 长期未使用（> 7 天）", "long_unused_days": 7.0},
    "high_confidence": {"label": "E 高质量对齐（>= 90%）", "min_confidence": 0.90},
    "by_person": {"label": "F 指定人物", "persons": ()},
    "exclude_recent": {"label": "G 排除最近 3 天用过", "recently_used_days": 3.0,
                       "exclude_recent": True},
}

def build_query(spec: FilterSpec, *, now: datetime | None = None,
                exclude_recent: bool = False) -> tuple[str, list[Any]]:
    """把 `FilterSpec` 翻成 `(SQL, 参数)`。返回的 SQL 已经带好 JOIN 和 ORDER BY。

    全部用占位符，**一个值都不拼进 SQL 字符串** —— 人物名、搜索词都来自用户输入。

    `long_unused_days` 的语义要注意：「超过 N 天没用」必须**包含"从来没用过"**
    （`last_used_at IS NULL`）。漏掉这一半就会出现"筛长期未使用，结果最该用的
    全新素材一条都不出现"这种反直觉结果。
    """
    stamp = now or datetime.now()
    where: list[str] = []
    params: list[Any] = []

    if spec.target_song_id is not None:
        where.append("m.target_song_id = ?")
        params.append(int(spec.target_song_id))
    if spec.segment_index is not None:
        where.append("m.segment_index = ?")
        params.append(int(spec.segment_index))
    if spec.segment_indexes:
        marks = ",".join("?" for _ in spec.segment_indexes)
        where.append(f"m.segment_index IN ({marks})")
        params.extend(int(i) for i in spec.segment_indexes)
    if spec.statuses:
        marks = ",".join("?" for _ in spec.statuses)
        where.append(f"m.status IN ({marks})")
        params.extend(spec.statuses)
    if spec.generation_versions:
        marks = ",".join("?" for _ in spec.generation_versions)
        where.append(f"m.generation_version IN ({marks})")
        params.extend(spec.generation_versions)
    if spec.source_video_ids:
        marks = ",".join("?" for _ in spec.source_video_ids)
        where.append(f"m.source_video_id IN ({marks})")
        params.extend(int(i) for i in spec.source_video_ids)
    if spec.persons:
        marks = ",".join("?" for _ in spec.persons)
        where.append(f"m.person IN ({marks})")
        params.extend(spec.persons)
    if spec.source_groups:
        marks = ",".join("?" for _ in spec.source_groups)
        where.append(f"m.source_group IN ({marks})")
        params.extend(spec.source_groups)

    for column, low, high in (("alignment_confidence", spec.min_confidence, spec.max_confidence),
                              ("use_count", spec.min_use_count, spec.max_use_count),
                              ("montage_count", spec.min_montage_count, spec.max_montage_count),
                              ("output_count", spec.min_output_count, spec.max_output_count)):
        if low is not None:
            where.append(f"m.{column} >= ?")
            params.append(low)
        if high is not None:
            where.append(f"m.{column} <= ?")
            params.append(high)

    if spec.never_used:
        where.append("m.use_count <= 0")
    if spec.never_output:
        where.append("m.output_count <= 0")
    if spec.recently_used_days is not None:
        cutoff = (stamp - timedelta(days=float(spec.recently_used_days))).strftime(
            "%Y-%m-%dT%H:%M:%S")
        if exclude_recent:
            # 方案 G：把最近用过的**排除**掉（从没用过的必须留下）
            where.append("(m.last_used_at IS NULL OR m.last_used_at < ?)")
        else:
            where.append("(m.last_used_at IS NOT NULL AND m.last_used_at >= ?)")
        params.append(cutoff)
    if spec.long_unused_days is not None:
        cutoff = (stamp - timedelta(days=float(spec.long_unused_days))).strftime(
            "%Y-%m-%dT%H:%M:%S")
        where.append("(m.last_used_at IS NULL OR m.last_used_at < ?)")
        params.append(cutoff)
    if spec.search.strip():
        where.append("(m.person LIKE ? OR m.source_group LIKE ? OR v.file_name LIKE ?)")
        like = f"%{spec.search.strip()}%"
        params.extend([like, like, like])

    sql = ("SELECT m.*, v.file_name AS source_name "
           "FROM dance_materials m JOIN videos v ON v.id = m.source_video_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY " + _ORDER_SQL.get(spec.sort, _ORDER_SQL["score_desc"])
    sql += " LIMIT ?"
    params.append(max(1, int(spec.limit)))
    return sql, params


def find_materials(db: Database, spec: FilterSpec, *, now: datetime | None = None,
                   exclude_recent: bool = False) -> list[DanceMaterial]:
    """按筛选条件查素材。这是 GUI 素材库列表和候选池共用的唯一入口。"""
    sql, params = build_query(spec, now=now, exclude_recent=exclude_recent)
    rows = db.connect().execute(sql, tuple(params))
    return [DanceMaterial.from_row(row) for row in rows]


def preset_spec(name: str, base: FilterSpec | None = None, **overrides: Any) -> FilterSpec:
    """按一期的 A~G 方案造 `FilterSpec`。`overrides` 用来填方案里的可调参数
    （比如方案 C 的 N、方案 F 的人物）。"""
    from dataclasses import replace  # noqa: PLC0415

    if name not in PRESETS:
        raise ValueError(f"未知的筛选方案 {name!r}，可选：{', '.join(sorted(PRESETS))}")
    # 方案里写成空元组的字段（比如 F 方案的 persons）是**占位**，意思是"这一项由调用方填"。
    # 不能让它反过来把 base 里已经填好的人物名清空 —— 那样"指定人物"就永远筛不出人。
    fields = {k: v for k, v in PRESETS[name].items()
              if k not in ("label", "exclude_recent") and v != ()}
    fields.update(overrides)
    return replace(base or FilterSpec(), **fields)


def score_context(db: Database, target_song_id: int, *, now: datetime | None = None,
                  sections: Sequence[Mapping[str, Any]] | None = None,
                  section_preference: Mapping[str, float] | None = None) -> ScoreContext:
    """把评分要用的历史事实一次性从库里捞齐。

    一次查完而不是评分时逐条查：一个位置几十条候选、十几个位置，逐条回库
    就是上千次查询；这里四条聚合 SQL 就够，之后全在内存里算。
    """
    from . import history, statistics  # noqa: PLC0415 - 避免模块间循环导入

    stamp = now or datetime.now()
    energy: dict[int, float] = {}
    impact: dict[int, float] = {}
    section_type: dict[int, str] = {}
    rows = sections if sections is not None else repo.song_sections(db, target_song_id)
    song = repo.get_song(db, target_song_id)
    slice_duration = 0.0
    if song is not None and song["duration"]:
        # 位置长度从素材反推：素材是按同一个 slice_duration 切的
        first = db.connect().execute(
            "SELECT duration FROM dance_materials WHERE target_song_id=? LIMIT 1",
            (target_song_id,)).fetchone()
        slice_duration = float(first["duration"]) if first is not None else 0.0
    if slice_duration > 0 and rows:
        count = int(float(song["duration"]) // slice_duration)
        for index in range(count):
            middle = index * slice_duration + slice_duration / 2.0
            for section in rows:
                try:
                    if float(section["start"]) <= middle < float(section["end"]):
                        energy[index] = float(section.get("energy") or 0.0)
                        impact[index] = float(section.get("impact") or 0.0)
                        section_type[index] = str(section.get("type") or "")
                        break
                except (KeyError, TypeError, ValueError):
                    continue
    return ScoreContext(
        now=stamp, energy_at=energy, impact_at=impact, section_at=section_type,
        position_usage=history.position_usage(db, target_song_id),
        pair_usage=statistics.pair_history(db, target_song_id),
        combination_usage=statistics.combination_history(db, target_song_id),
        person_usage=statistics.person_history(db, target_song_id),
        source_usage=statistics.source_history(db, target_song_id),
        feedback=repo.feedback_counts(db, target_song_id),
        section_preference=dict(section_preference or {}))


def build_pool(db: Database, target_song_id: int, segment_index: int, *,
               spec: FilterSpec | None = None, context: ScoreContext | None = None,
               weights: Mapping[str, float] | None = None,
               exclude_recent: bool = False, limit: int = 0) -> CandidatePool:
    """一个音乐位置的候选池：查素材 → 打静态分 → 按分排序。

    `exclude_recent=True` 时 `spec.recently_used_days` 的语义从"只要最近用过的"
    翻转成"排除最近用过的"（一期方案 G）。两种都要：G 用来避免撞车，
    而"只看最近用过的"在排查"上一版为什么这么像"时有用。

    **不写任何使用事件**（技术指导第九节：candidate ≠ use）。要记 candidate
    事件的话由调用方显式调 `history.note_candidates()` —— 用户在界面上翻素材
    不该让库里的使用次数往上涨。
    """
    from dataclasses import replace  # noqa: PLC0415

    base = spec or FilterSpec()
    query = replace(base, target_song_id=target_song_id, segment_index=segment_index)
    ctx = context or score_context(db, target_song_id)
    materials = find_materials(db, query, now=ctx.now, exclude_recent=exclude_recent)
    scored = [static_score(m, segment_index, ctx, weights) for m in materials]
    scored.sort(key=lambda s: (-s.final_score, s.material_id))
    if limit > 0:
        scored = scored[:int(limit)]
    keep = {s.material_id for s in scored}
    start, end = 0.0, 0.0
    if materials:
        first = materials[0]
        start, end = float(first.target_start), float(first.target_end)
    return CandidatePool(segment_index=int(segment_index), target_start=start, target_end=end,
                         scored=tuple(scored),
                         materials={m.id: m for m in materials if m.id in keep})


def build_pools(db: Database, target_song_id: int, segment_indexes: Sequence[int], *,
                spec: FilterSpec | None = None, weights: Mapping[str, float] | None = None,
                exclude_recent: bool = False, limit: int = 0) -> list[CandidatePool]:
    """一次给一批位置建候选池，共用同一份 `ScoreContext`（历史只查一次）。"""
    ctx = score_context(db, target_song_id)
    pools = [build_pool(db, target_song_id, index, spec=spec, context=ctx, weights=weights,
                        exclude_recent=exclude_recent, limit=limit)
             for index in segment_indexes]
    total = sum(len(p.scored) for p in pools)
    logger.info("候选池：%d 个位置共 %d 条候选（歌 #%d）", len(pools), total, target_song_id)
    return pools



def empty_positions(pools: Sequence[CandidatePool]) -> list[int]:
    """哪些位置一条候选都没有。GUI 要把这些位置标红 —— 它们会让成片出现空档。"""
    return [p.segment_index for p in pools if not p.scored]


__all__ = [
    "PRESETS", "build_query", "find_materials", "preset_spec",
    "score_context", "build_pool", "build_pools", "empty_positions",
]


