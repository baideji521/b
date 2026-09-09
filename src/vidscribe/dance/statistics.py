"""重复率分析与统计。技术指导第十四节的七个口径，全部落 `dance_montage_versions`。

七个数字回答的是同一个问题的七个侧面：**这一版和别的版本、和历史，有多像。**
每个值都是 0~1，0 = 一点不重复，1 = 全在重复。

一条设计取向：重复率既看**版内**（这一版自己有没有把同一个人连着塞五次），
也看**历史**（这一版是不是把上周那一版又剪了一遍）。只看版内是不够的 ——
一个长期运行的素材资产系统，最容易犯的错就是"每一版单独看都很好，但十版下来都一样"。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from .material_score import combination_signature, pair_key
from .types import RepeatStats

logger = get_logger("dance.stats")

#: overall_repeat 的合成权重。素材重复和人物重复最刺眼，权重最高
OVERALL_WEIGHTS: dict[str, float] = {
    "material_repeat": 0.25,
    "person_repeat": 0.22,
    "position_repeat": 0.18,
    "source_repeat": 0.13,
    "pair_repeat": 0.12,
    "combination_repeat": 0.10,
}


# ============================================================== 历史聚合
def pair_history(db: Database, target_song_id: int) -> dict[tuple[int, int], int]:
    """`{(素材A, 素材B): 历史上挨着出现过几次}`，无向。

    从 `dance_montage_materials` 按 `order_index` 相邻推出来：同一版里
    order_index 差 1 的两条就是一对邻居。
    """
    rows = db.connect().execute(
        "SELECT a.version_id AS vid, a.material_id AS first, b.material_id AS second "
        "FROM dance_montage_materials a "
        "JOIN dance_montage_materials b "
        "  ON b.version_id = a.version_id AND b.order_index = a.order_index + 1 "
        "JOIN dance_montage_versions v ON v.id = a.version_id "
        "JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ?", (int(target_song_id),))
    out: dict[tuple[int, int], int] = {}
    for row in rows:
        key = pair_key(int(row["first"]), int(row["second"]))
        out[key] = out.get(key, 0) + 1
    return out

def version_material_sets(db: Database, target_song_id: int, *,
                          exclude_version: int = 0) -> dict[int, set[int]]:
    """`{version_id: 这一版用到的素材 id 集合}`。算组合相似度要用它。"""
    rows = db.connect().execute(
        "SELECT vm.version_id AS vid, vm.material_id AS mid "
        "FROM dance_montage_materials vm "
        "JOIN dance_montage_versions v ON v.id = vm.version_id "
        "JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ?", (int(target_song_id),))
    out: dict[int, set[int]] = {}
    for row in rows:
        version = int(row["vid"])
        if exclude_version and version == int(exclude_version):
            continue
        out.setdefault(version, set()).add(int(row["mid"]))
    return out


def combination_history(db: Database, target_song_id: int) -> dict[str, int]:
    """`{组合签名: 出现过几次}`。签名顺序无关（见 `combination_signature`）。

    注意这里存的是**完整版本**的签名。评分里的 `combination_penalty` 查的是
    "已选前缀 + 候选"的签名，正常不会命中完整签名 —— 只有在搜索已经拼到
    和某个历史版本一模一样的时候才会命中，那恰恰是最该扣分的时刻。
    """
    out: dict[str, int] = {}
    for materials in version_material_sets(db, target_song_id).values():
        signature = combination_signature(materials)
        out[signature] = out.get(signature, 0) + 1
    return out


def person_history(db: Database, target_song_id: int) -> dict[str, int]:
    """`{人物: 在历史版本里出现过几次}`。"""
    rows = db.connect().execute(
        "SELECT m.person AS person, COUNT(*) AS n FROM dance_montage_materials vm "
        "JOIN dance_materials m ON m.id = vm.material_id "
        "JOIN dance_montage_versions v ON v.id = vm.version_id "
        "JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ? AND m.person IS NOT NULL AND m.person <> '' "
        "GROUP BY m.person", (int(target_song_id),))
    return {str(r["person"]): int(r["n"]) for r in rows}


def source_history(db: Database, target_song_id: int) -> dict[int, int]:
    """`{源视频 id: 在历史版本里出现过几次}`。"""
    rows = db.connect().execute(
        "SELECT m.source_video_id AS sid, COUNT(*) AS n FROM dance_montage_materials vm "
        "JOIN dance_materials m ON m.id = vm.material_id "
        "JOIN dance_montage_versions v ON v.id = vm.version_id "
        "JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ? GROUP BY m.source_video_id", (int(target_song_id),))
    return {int(r["sid"]): int(r["n"]) for r in rows}


# ============================================================== 七个重复率
def _ratio(repeated: int, total: int) -> float:
    return round(repeated / float(total), 4) if total > 0 else 0.0


def repeat_stats(clips: Sequence[Any], *,
                 position_usage: Mapping[tuple[int, int], int] | None = None,
                 pair_usage: Mapping[tuple[int, int], int] | None = None,
                 history_sets: Mapping[int, set[int]] | None = None) -> RepeatStats:
    """算一版混剪的七个重复率。

    `clips` 是 `DanceMontageClip` 序列（已按 order_index 排好）。
    三个历史参数都可以不给 —— 不给就只算版内重复，历史项为 0。

    逐项口径：

    - `material_repeat`  版内：同一条素材被用了几次。`1 - 去重后/总数`
    - `position_repeat`  历史：有多少格用的素材**在这一格**以前就用过
    - `person_repeat`    版内：同一个人占了多少格
    - `source_repeat`    版内：同一个源视频占了多少格
    - `pair_repeat`      历史：相邻对里有多少对以前就挨着出现过
    - `combination_repeat` 历史：这一整套素材和历史某一版的最大 Jaccard 相似度。
      用 Jaccard 而不是"签名是否完全相同"：完全相同才算重复太宽松了，
      换掉一格就"不重复"，但观众看到的还是同一支视频
    - `overall_repeat`   上面六项的加权平均（权重见 `OVERALL_WEIGHTS`）
    """
    total = len(clips)
    if total == 0:
        return RepeatStats()

    material_ids = [int(c.material_id) for c in clips]
    persons = [str(c.person or "") for c in clips]
    sources = [int(c.source_video_id) for c in clips]

    material_repeat = _ratio(total - len(set(material_ids)), total)
    # 人物重复只在"有人物标签的那些格"之间算：素材没打人物标签时把它算成
    # "和别人重复"是凭空冤枉，算成"不重复"又会掩盖真实重复率
    named = [p for p in persons if p]
    person_repeat = _ratio(len(named) - len(set(named)), len(named))
    source_repeat = _ratio(total - len(set(sources)), total)


    seen_here = 0
    usage = position_usage or {}
    for clip in clips:
        if usage.get((int(clip.segment_index), int(clip.material_id)), 0) > 0:
            seen_here += 1
    position_repeat = _ratio(seen_here, total)

    pairs = pair_usage or {}
    repeated_pairs = 0
    pair_total = max(0, total - 1)
    for index in range(pair_total):
        if pairs.get(pair_key(material_ids[index], material_ids[index + 1]), 0) > 0:
            repeated_pairs += 1
    pair_repeat = _ratio(repeated_pairs, pair_total)

    combination_repeat = 0.0
    current = set(material_ids)
    best_version = 0
    for version_id, others in (history_sets or {}).items():
        union = current | others
        if not union:
            continue
        score = len(current & others) / float(len(union))
        if score > combination_repeat:
            combination_repeat, best_version = round(score, 4), int(version_id)

    values = {
        "material_repeat": material_repeat,
        "person_repeat": person_repeat,
        "position_repeat": position_repeat,
        "source_repeat": source_repeat,
        "pair_repeat": pair_repeat,
        "combination_repeat": combination_repeat,
    }
    overall = round(sum(values[k] * w for k, w in OVERALL_WEIGHTS.items()), 4)
    detail = {
        "clips": total,
        "unique_materials": len(set(material_ids)),
        "unique_persons": len({p for p in persons if p}),
        "unique_sources": len(set(sources)),
        "repeated_pairs": repeated_pairs,
        "positions_seen_before": seen_here,
        "closest_version_id": best_version,
        "signature": combination_signature(material_ids),
    }
    return RepeatStats(overall_repeat=overall, detail=detail, **values)

# ============================================================== 面板统计
def song_overview(db: Database, target_song_id: int) -> dict[str, Any]:
    """一首歌的整体统计，给统计面板铺第一屏。全部来自 SQL，不扫盘。"""
    conn = db.connect()
    materials = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN status='ready' THEN 1 ELSE 0 END) AS ready, "
        "SUM(CASE WHEN use_count<=0 THEN 1 ELSE 0 END) AS never_used, "
        "SUM(CASE WHEN output_count<=0 THEN 1 ELSE 0 END) AS never_output, "
        "COUNT(DISTINCT source_video_id) AS sources, "
        "COUNT(DISTINCT person) AS persons, "
        "COUNT(DISTINCT segment_index) AS positions, "
        "AVG(alignment_confidence) AS avg_confidence "
        "FROM dance_materials WHERE target_song_id = ?", (int(target_song_id),)).fetchone()
    versions = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN v.render_status='rendered' THEN 1 ELSE 0 END) AS rendered, "
        "SUM(CASE WHEN v.render_status='failed' THEN 1 ELSE 0 END) AS failed, "
        "AVG(v.overall_repeat) AS avg_repeat, MIN(v.overall_repeat) AS best_repeat "
        "FROM dance_montage_versions v JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ?", (int(target_song_id),)).fetchone()
    alignments = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok, "
        "SUM(CASE WHEN status='low_confidence' THEN 1 ELSE 0 END) AS low, "
        "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected, "
        "SUM(CASE WHEN manual_offset IS NOT NULL THEN 1 ELSE 0 END) AS manual "
        "FROM dance_audio_alignments WHERE target_song_id = ?",
        (int(target_song_id),)).fetchone()

    def _row(row: Any, keys: Sequence[str]) -> dict[str, Any]:
        return {k: (0 if row[k] is None else
                    (round(float(row[k]), 4) if isinstance(row[k], float) else int(row[k])))
                for k in keys}

    from . import history  # noqa: PLC0415 - 只在这里用，避免模块级循环

    return {
        "target_song_id": int(target_song_id),
        "materials": _row(materials, ("total", "ready", "never_used", "never_output",
                                      "sources", "persons", "positions", "avg_confidence")),
        "versions": _row(versions, ("total", "rendered", "failed", "avg_repeat",
                                    "best_repeat")),
        "alignments": _row(alignments, ("total", "ok", "low", "rejected", "manual")),
        "events": history.event_summary(db, target_song_id),
    }


def position_coverage(db: Database, target_song_id: int) -> list[dict[str, Any]]:
    """每个音乐位置有多少条素材、多少个人、用过多少次。

    这是一期第十六节那个「音乐 4~6 秒有哪些人的素材」的数据源，
    也是界面上"哪些位置素材不够"的唯一依据。
    """
    rows = db.connect().execute(
        "SELECT segment_index AS pos, COUNT(*) AS materials, "
        "COUNT(DISTINCT person) AS persons, SUM(use_count) AS uses, "
        "SUM(output_count) AS outputs, MIN(target_start) AS start, MAX(target_end) AS end "
        "FROM dance_materials WHERE target_song_id = ? AND status = 'ready' "
        "GROUP BY segment_index ORDER BY segment_index", (int(target_song_id),))
    return [{"segment_index": int(r["pos"]), "materials": int(r["materials"]),
             "persons": int(r["persons"]), "uses": int(r["uses"] or 0),
             "outputs": int(r["outputs"] or 0),
             "start": round(float(r["start"] or 0.0), 3),
             "end": round(float(r["end"] or 0.0), 3)} for r in rows]


def person_breakdown(db: Database, target_song_id: int) -> list[dict[str, Any]]:
    """按人物汇总：素材数、使用次数、出片次数、覆盖多少个位置。"""
    rows = db.connect().execute(
        "SELECT COALESCE(NULLIF(person, ''), '(未标注)') AS person, COUNT(*) AS materials, "
        "SUM(use_count) AS uses, SUM(output_count) AS outputs, "
        "COUNT(DISTINCT segment_index) AS positions, AVG(alignment_confidence) AS confidence "
        "FROM dance_materials WHERE target_song_id = ? "
        "GROUP BY COALESCE(NULLIF(person, ''), '(未标注)') ORDER BY materials DESC",
        (int(target_song_id),))
    return [{"person": str(r["person"]), "materials": int(r["materials"]),
             "uses": int(r["uses"] or 0), "outputs": int(r["outputs"] or 0),
             "positions": int(r["positions"]),
             "confidence": round(float(r["confidence"] or 0.0), 4)} for r in rows]


def describe(stats: RepeatStats) -> list[str]:
    """把重复率写成中文行，CLI 和界面共用同一份措辞。"""
    return [
        f"[重复率] 综合 {stats.overall_repeat:.3f}",
        f"  素材 {stats.material_repeat:.3f}｜人物 {stats.person_repeat:.3f}"
        f"｜位置 {stats.position_repeat:.3f}｜来源 {stats.source_repeat:.3f}",
        f"  相邻对 {stats.pair_repeat:.3f}｜整套组合 {stats.combination_repeat:.3f}",
        f"  明细：{stats.detail.get('unique_materials', 0)} 条不同素材 / "
        f"{stats.detail.get('clips', 0)} 格，"
        f"{stats.detail.get('unique_persons', 0)} 个人，"
        f"{stats.detail.get('repeated_pairs', 0)} 对邻居以前出现过",
    ]


__all__ = [
    "OVERALL_WEIGHTS",
    "pair_history", "version_material_sets", "combination_history",
    "person_history", "source_history", "repeat_stats",
    "song_overview", "position_coverage", "person_breakdown", "describe",
]


