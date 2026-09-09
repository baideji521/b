"""编辑计划（Timeline）。**纯计划层，禁止在这里做推荐或选择。**

技术指导第十五节的分层：Selection → Timeline → Render。
进到这一层的时候"每个位置用哪条素材"已经定了（不管是推荐定的还是人工点的），
这里只负责四件事：

    1. 排顺序、算成片时间（按 target_start 升序，成片时间从 0 连续铺）
    2. 引用 `material_id` 而**不是**文件路径 —— 素材才是资产，路径只是它现在躺在哪儿
    3. 算七个重复率
    4. 校验：有没有空档、素材文件还在不在、时长对不对

`DanceMontageContext` 出去之后，渲染层拿到的是一份**不需要再思考**的清单。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from . import material_repository as repo, statistics
from .material_score import combination_signature
from .types import (
    DanceMaterial,
    DanceMontageClip,
    DanceMontageContext,
    MaterialScore,
    RepeatStats,
)

logger = get_logger("dance.timeline")


def build_timeline(db: Database, target_song_id: int,
                   picks: Sequence[MaterialScore],
                   materials: Mapping[int, DanceMaterial] | Sequence[DanceMaterial], *,
                   song_path: str = "", slice_duration: float = 2.0,
                   montage_id: int = 0, version_index: int = 1,
                   strategy_id: int = 0, recommendation_run_id: int = 0,
                   ) -> DanceMontageContext:
    """把"选好的素材"整理成编辑计划。

    成片时间**按顺序连续铺**（第一格从 0 开始），而不是照抄素材的 `target_start`：
    某些位置可能没素材（空档），照抄就会在成片里留下真空。铺连续之后成片会比
    目标歌短一点，`mux` 那边按画面时长截音轨，两边始终对齐。

    素材原本对应的音乐位置仍然记在 `segment_index` 里 —— 那是它的身份，不能丢。
    """
    lookup: dict[int, DanceMaterial] = (
        dict(materials) if isinstance(materials, Mapping)
        else {int(m.id): m for m in materials})
    ordered = sorted(picks, key=lambda p: (p.segment_index, p.material_id))
    clips: list[DanceMontageClip] = []
    notes: list[str] = []
    cursor = 0.0
    for order, pick in enumerate(ordered):
        material = lookup.get(int(pick.material_id))
        if material is None:
            notes.append(f"位置 #{pick.segment_index} 的素材 #{pick.material_id} 查不到，跳过")
            continue
        duration = float(material.duration) or float(slice_duration)
        clips.append(DanceMontageClip(
            order_index=order, segment_index=int(pick.segment_index),
            material_id=int(material.id),
            target_start=round(cursor, 6), target_end=round(cursor + duration, 6),
            source_start=float(material.source_start), source_end=float(material.source_end),
            file_path=str(material.file_path or ""),
            selection_score=float(pick.final_score),
            score_breakdown=dict(pick.breakdown),
            person=str(material.person or ""),
            source_video_id=int(material.source_video_id)))
        cursor += duration

    context = DanceMontageContext(
        target_song_id=int(target_song_id), target_song_path=str(song_path),
        slice_duration=float(slice_duration), clips=clips,
        strategy_id=int(strategy_id), recommendation_run_id=int(recommendation_run_id),
        montage_id=int(montage_id), version_index=int(version_index),
        signature=combination_signature(c.material_id for c in clips), notes=notes)
    context.repeat = repeat_of(db, context)
    logger.info("编辑计划：%d 格｜%.3fs｜签名 %s｜综合重复率 %.3f",
                len(clips), context.duration, context.signature[:40],
                context.repeat.overall_repeat)
    return context

def repeat_of(db: Database, context: DanceMontageContext) -> RepeatStats:
    """算这一版的七个重复率（版内 + 和历史比）。"""
    from . import history  # noqa: PLC0415

    song = context.target_song_id
    return statistics.repeat_stats(
        context.clips,
        position_usage=history.position_usage(db, song),
        pair_usage=statistics.pair_history(db, song),
        history_sets=statistics.version_material_sets(db, song))


def from_manual(db: Database, target_song_id: int, selection: Mapping[int, int], *,
                song_path: str = "", slice_duration: float = 2.0, montage_id: int = 0,
                version_index: int = 1) -> DanceMontageContext:
    """纯手动选择 → 编辑计划。`selection` 是 `{音乐位置: 素材id}`。

    技术指导第二十节要求"智能推荐关闭后仍可以纯手动选择"，这就是那条路。
    手动挑的素材没有推荐分，`selection_score` 记 0、breakdown 里标明来源是手动 ——
    绝不伪造一个分数出来，否则统计里会分不清"人挑的"和"算法挑的"。
    """
    ids = [int(v) for v in selection.values()]
    materials = repo.get_materials(db, ids)
    picks = [MaterialScore(material_id=int(mid), segment_index=int(pos), final_score=0.0,
                           breakdown={"manual": 1.0, "final_score": 0.0})
             for pos, mid in sorted(selection.items(), key=lambda kv: int(kv[0]))
             if int(mid) in materials]
    context = build_timeline(db, target_song_id, picks, materials, song_path=song_path,
                             slice_duration=slice_duration, montage_id=montage_id,
                             version_index=version_index)
    context.notes.append("这一版是纯手动选择（未使用智能推荐）")
    return context


def validate(context: DanceMontageContext, *, expected_positions: int = 0) -> list[str]:
    """校验编辑计划，返回问题清单（空 = 可以渲染）。

    只报**会让成片出问题**的事，不报审美意见：
      - 一格都没有
      - 素材文件不在盘上（渲染必然失败，提前说清楚是哪一条）
      - 成片时间不连续或有重叠（说明排序算错了）
      - 位置数比预期少（成片会比目标歌短，用户该知道）
    """
    problems: list[str] = []
    if not context.clips:
        problems.append("编辑计划里一格都没有，没法渲染")
        return problems
    for clip in context.clips:
        if not clip.file_path:
            problems.append(f"位置 #{clip.segment_index} 的素材 #{clip.material_id} 没有文件路径")
        elif not Path(clip.file_path).is_file():
            problems.append(f"位置 #{clip.segment_index} 的素材文件不在盘上："
                           f"{Path(clip.file_path).name}")
        if clip.target_end <= clip.target_start:
            problems.append(f"位置 #{clip.segment_index} 的成片区间非法："
                           f"{clip.target_start} → {clip.target_end}")
    for previous, current in zip(context.clips, context.clips[1:]):
        if abs(current.target_start - previous.target_end) > 1e-3:
            problems.append(f"成片时间不连续：第 {previous.order_index} 格结束于 "
                           f"{previous.target_end:.3f}s，第 {current.order_index} 格却从 "
                           f"{current.target_start:.3f}s 开始")
    if expected_positions and len(context.clips) < int(expected_positions):
        problems.append(f"只填上 {len(context.clips)}/{int(expected_positions)} 个位置，"
                       f"成片会比目标歌短 "
                       f"{(int(expected_positions) - len(context.clips)) * context.slice_duration:.1f}s")
    return problems


def save(db: Database, context: DanceMontageContext, *, montage_name: str = "") -> int:
    """把编辑计划落库成一个**新版本**，返回 version_id。

    历史版本永不覆盖（`version_index` 自增 + 唯一约束挡着）—— 技术指导第十四节。
    同时记 `montage` 事件并推 use_count：到这一步素材才算"真的被用了"。
    """
    from . import history  # noqa: PLC0415

    montage_id = context.montage_id
    if not montage_id:
        song = repo.get_song(db, context.target_song_id)
        name = montage_name or f"{(song['title'] if song else '混剪')}"
        montage_id = repo.ensure_montage(db, target_song_id=context.target_song_id,
                                         name=name, slice_duration=context.slice_duration)
        context.montage_id = montage_id
    context.version_index = repo.next_version_index(db, montage_id)
    version_id = repo.save_version(
        db, montage_id=montage_id, version_index=context.version_index,
        signature=context.signature, strategy_id=context.strategy_id or None,
        recommendation_run_id=context.recommendation_run_id or None,
        clips=context.clips, duration=context.duration, repeat=context.repeat,
        timeline_json=context.to_dict())
    history.note_montage(db, version_id, montage_id, context.clips)
    logger.info("编辑计划已存为版本 #%d（混剪 #%d 第 %d 版）",
                version_id, montage_id, context.version_index)
    return version_id


def load(db: Database, version_id: int) -> DanceMontageContext | None:
    """从库里读回一版编辑计划。历史版本要能重新渲染、重新比对。"""
    row = repo.get_version(db, version_id)
    if row is None:
        return None
    montage = repo.get_montage(db, int(row["montage_id"]))
    song = repo.get_song(db, int(montage["target_song_id"])) if montage is not None else None
    clips = [DanceMontageClip(
        order_index=int(r["order_index"]), segment_index=int(r["segment_index"]),
        material_id=int(r["material_id"]),
        target_start=float(r["target_start"]), target_end=float(r["target_end"]),
        source_start=float(r["source_start"]), source_end=float(r["source_end"]),
        file_path=str(r["file_path"] or ""),
        selection_score=float(r["selection_score"] or 0.0),
        score_breakdown=repo._loads(r["score_breakdown_json"]) or {},   # noqa: SLF001
        person=str(r["person"] or ""), source_video_id=int(r["source_video_id"] or 0))
        for r in repo.version_materials(db, version_id)]
    stats = RepeatStats(
        material_repeat=float(row["material_repeat"] or 0.0),
        position_repeat=float(row["position_repeat"] or 0.0),
        person_repeat=float(row["person_repeat"] or 0.0),
        source_repeat=float(row["source_repeat"] or 0.0),
        pair_repeat=float(row["pair_repeat"] or 0.0),
        combination_repeat=float(row["combination_repeat"] or 0.0),
        overall_repeat=float(row["overall_repeat"] or 0.0),
        detail=repo._loads(row["repeat_detail_json"]) or {})            # noqa: SLF001
    return DanceMontageContext(
        target_song_id=int(montage["target_song_id"]) if montage is not None else 0,
        target_song_path=str(song["file_path"]) if song is not None else "",
        slice_duration=float(montage["slice_duration"]) if montage is not None else 2.0,
        clips=clips, strategy_id=int(row["strategy_id"] or 0),
        recommendation_run_id=int(row["recommendation_run_id"] or 0),
        montage_id=int(row["montage_id"]), version_index=int(row["version_index"]),
        signature=str(row["signature"] or ""), repeat=stats)


def describe(context: DanceMontageContext, limit: int = 12) -> list[str]:
    """把编辑计划写成中文行。"""
    lines = [f"[计划] 混剪 #{context.montage_id} 第 {context.version_index} 版"
             f"｜{len(context.clips)} 格｜{context.duration:.3f}s",
             f"  签名 {context.signature[:60]}"]
    for clip in context.clips[:limit]:
        lines.append(f"  {clip.target_start:>7.2f}→{clip.target_end:<7.2f} "
                     f"位置 {clip.segment_index:>3}  素材 #{clip.material_id:<5} "
                     f"{(clip.person or '(未标注)'):<12} 分 {clip.selection_score:+.3f}")
    if len(context.clips) > limit:
        lines.append(f"  …… 还有 {len(context.clips) - limit} 格")
    lines.extend(statistics.describe(context.repeat))
    lines.extend(f"  说明：{note}" for note in context.notes)
    return lines


__all__ = ["build_timeline", "repeat_of", "from_manual", "validate", "save", "load", "describe"]

