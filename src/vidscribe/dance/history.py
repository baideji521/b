"""使用事件账本。**所有计数都必须能由这张表重算**（技术指导第九节）。

`dance_materials` 上的 `use_count / montage_count / output_count` 只是加速用的缓存，
唯一的事实来源是 `dance_material_usage_events` —— 它只追加、不修改、不删除。

三个语义层级必须严格分开，混了这个系统就没有价值了：

    candidate      进过候选池          → 不算用过，任何计数都不动
    selected       被人/推荐挑中了      → 也还不算用过（可能最后没进这一版）
    montage        真的进了某一版混剪   → use_count + 1、montage_count + 1
    render_success 那一版真的出片了     → output_count + 1
    render_failed  渲染失败             → **绝不**增加出片次数
    rejected       被否决               → 只记账，给下一次推荐做参考
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from . import material_repository as repo

logger = get_logger("dance.history")

#: 会让 use_count / montage_count 往上走的事件
COUNT_AS_USE = ("montage",)
#: 会让 output_count 往上走的事件。**render_failed 不在里面**
COUNT_AS_OUTPUT = ("render_success",)


def note_event(db: Database, material_id: int, event: str, *, montage_id: int | None = None,
               version_id: int | None = None, segment_index: int | None = None,
               detail: Mapping[str, Any] | None = None) -> int:
    """记一条事件。**只追加**，不动任何计数缓存 —— 计数由 `apply_counts` 统一算。"""
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_material_usage_events(material_id, montage_id, version_id, "
            "event, segment_index, detail_json, created_at) VALUES(?,?,?,?,?,?,?)",
            (int(material_id), montage_id, version_id, str(event), segment_index,
             repo._dumps(dict(detail)) if detail else None, repo.now()))   # noqa: SLF001
        return int(cursor.lastrowid)

def note_events(db: Database, material_ids: Iterable[int], event: str, *,
                montage_id: int | None = None, version_id: int | None = None,
                segment_index: int | None = None) -> int:
    """批量记同一种事件，返回条数。整批一个事务，别开几十个。"""
    ids = [int(i) for i in material_ids]
    if not ids:
        return 0
    stamp = repo.now()
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO dance_material_usage_events(material_id, montage_id, version_id, "
            "event, segment_index, detail_json, created_at) VALUES(?,?,?,?,?,NULL,?)",
            [(i, montage_id, version_id, str(event), segment_index, stamp) for i in ids])
    return len(ids)


def note_candidates(db: Database, pools: Sequence[Any]) -> int:
    """把候选池里的素材记成 `candidate` 事件，并更新 `candidate_count`。

    `candidate_count` 是**唯一**由 candidate 事件推动的计数 —— 它回答的是
    "这条素材被考虑过多少次"，和"用过多少次"是两件不同的事。
    """
    stamp = repo.now()
    rows: list[tuple[Any, ...]] = []
    for pool in pools:
        for scored in pool.scored:
            rows.append((int(scored.material_id), None, None, "candidate",
                         int(pool.segment_index), None, stamp))
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO dance_material_usage_events(material_id, montage_id, version_id, "
            "event, segment_index, detail_json, created_at) VALUES(?,?,?,?,?,?,?)", rows)
        ids = sorted({row[0] for row in rows})
        marks = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE dance_materials SET candidate_count = candidate_count + 1, updated_at=? "
            f"WHERE id IN ({marks})", (stamp, *ids))
    return len(rows)


def note_montage(db: Database, version_id: int, montage_id: int, clips: Sequence[Any]) -> int:
    """一版混剪定稿：记 `montage` 事件 + 推 `use_count` / `montage_count` / 时间戳。

    这是 use_count 唯一会 +1 的地方（技术指导第九节）。
    `first_used_at` 只在第一次写入（`COALESCE`），`last_used_at` 每次都刷。
    """
    stamp = repo.now()
    rows = [(int(c.material_id), montage_id, version_id, "montage",
             int(c.segment_index), None, stamp) for c in clips]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO dance_material_usage_events(material_id, montage_id, version_id, "
            "event, segment_index, detail_json, created_at) VALUES(?,?,?,?,?,?,?)", rows)
        # 同一版里同一条素材可能出现在多个位置，计数按**出现次数**加
        for material_id in (row[0] for row in rows):
            conn.execute(
                "UPDATE dance_materials SET use_count = use_count + 1, "
                "montage_count = montage_count + 1, "
                "first_used_at = COALESCE(first_used_at, ?), last_used_at = ?, updated_at = ? "
                "WHERE id = ?", (stamp, stamp, stamp, material_id))
    logger.info("版本 #%d 记 montage 事件 %d 条", version_id, len(rows))
    return len(rows)


def note_render(db: Database, version_id: int, montage_id: int, clips: Sequence[Any], *,
                ok: bool, detail: Mapping[str, Any] | None = None) -> int:
    """渲染收尾：成功记 `render_success` 并推 `output_count`；失败记 `render_failed`。

    **失败绝不增加出片次数**（技术指导第九节写死的一条）。失败也要记事件 ——
    "这条素材参与过一次失败的渲染"是有用的信息，静默丢掉就查不出"为什么这条老是失败"。
    """
    stamp = repo.now()
    event = "render_success" if ok else "render_failed"
    payload = repo._dumps(dict(detail)) if detail else None       # noqa: SLF001
    rows = [(int(c.material_id), montage_id, version_id, event,
             int(c.segment_index), payload, stamp) for c in clips]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO dance_material_usage_events(material_id, montage_id, version_id, "
            "event, segment_index, detail_json, created_at) VALUES(?,?,?,?,?,?,?)", rows)
        if ok:
            for material_id in (row[0] for row in rows):
                conn.execute(
                    "UPDATE dance_materials SET output_count = output_count + 1, "
                    "last_output_at = ?, updated_at = ? WHERE id = ?",
                    (stamp, stamp, material_id))
    logger.info("版本 #%d 记 %s 事件 %d 条", version_id, event, len(rows))
    return len(rows)


def note_rejected(db: Database, material_ids: Iterable[int], *,
                  segment_index: int | None = None) -> int:
    """记 `rejected`：用户把推荐出来的素材换掉了。下一次推荐会读它。"""
    return note_events(db, material_ids, "rejected", segment_index=segment_index)

# ============================================================== 重算与查询
def recount_material(db: Database, material_id: int) -> dict[str, Any]:
    """从事件账本重算这条素材的全部计数并写回，返回新旧对照。

    这是技术指导第九节那条"所有计数必须可以从 event ledger 重算"的兑现。
    用途有三个：库被手工改过之后自愈、升级算法后统一口径、以及在测试里
    证明"缓存的数字和账本一致"。
    """
    conn = db.connect()
    before = conn.execute(
        "SELECT candidate_count, use_count, montage_count, output_count "
        "FROM dance_materials WHERE id = ?", (int(material_id),)).fetchone()
    if before is None:
        return {}
    counts = {row["event"]: int(row["n"]) for row in conn.execute(
        "SELECT event, COUNT(*) AS n FROM dance_material_usage_events "
        "WHERE material_id = ? GROUP BY event", (int(material_id),))}
    times = conn.execute(
        "SELECT MIN(created_at) AS first_used, MAX(created_at) AS last_used "
        "FROM dance_material_usage_events WHERE material_id = ? AND event = 'montage'",
        (int(material_id),)).fetchone()
    last_output = conn.execute(
        "SELECT MAX(created_at) AS at FROM dance_material_usage_events "
        "WHERE material_id = ? AND event = 'render_success'", (int(material_id),)).fetchone()

    fresh = {
        "candidate_count": counts.get("candidate", 0),
        "use_count": counts.get("montage", 0),
        "montage_count": counts.get("montage", 0),
        "output_count": counts.get("render_success", 0),
    }
    with db.tx() as write:
        write.execute(
            "UPDATE dance_materials SET candidate_count=?, use_count=?, montage_count=?, "
            "output_count=?, first_used_at=?, last_used_at=?, last_output_at=?, updated_at=? "
            "WHERE id=?",
            (fresh["candidate_count"], fresh["use_count"], fresh["montage_count"],
             fresh["output_count"], times["first_used"], times["last_used"],
             last_output["at"], repo.now(), int(material_id)))
    return {"material_id": int(material_id),
            "before": {k: int(before[k]) for k in fresh},
            "after": fresh,
            "changed": any(int(before[k]) != v for k, v in fresh.items())}


def recount_song(db: Database, target_song_id: int) -> dict[str, Any]:
    """把这首歌下所有素材的计数重算一遍。返回改了几条。"""
    rows = db.connect().execute(
        "SELECT id FROM dance_materials WHERE target_song_id = ?", (int(target_song_id),))
    changed = 0
    total = 0
    for row in rows.fetchall():
        total += 1
        if recount_material(db, int(row["id"])).get("changed"):
            changed += 1
    logger.info("重算歌 #%d 的素材计数：%d 条里改了 %d 条", target_song_id, total, changed)
    return {"target_song_id": int(target_song_id), "materials": total, "changed": changed}


def position_usage(db: Database, target_song_id: int) -> dict[tuple[int, int], int]:
    """`{(音乐位置, 素材id): 在这个位置被用过几次}`。

    一期第三十九节要求的"位置级使用统计"就是它 —— 一条素材总共用了 8 次，
    但如果全在第 3 段，那第 5 段用它仍然是"新的"。评分里的
    `position_usage_penalty` 吃的就是这份数据。
    """
    rows = db.connect().execute(
        "SELECT e.segment_index AS pos, e.material_id AS mid, COUNT(*) AS n "
        "FROM dance_material_usage_events e "
        "JOIN dance_materials m ON m.id = e.material_id "
        "WHERE m.target_song_id = ? AND e.event = 'montage' AND e.segment_index IS NOT NULL "
        "GROUP BY e.segment_index, e.material_id", (int(target_song_id),))
    return {(int(r["pos"]), int(r["mid"])): int(r["n"]) for r in rows}


def material_events(db: Database, material_id: int, *, limit: int = 200) -> list[Any]:
    """一条素材的事件流水，最近的在前。GUI 的"素材历史"面板直接铺这个。"""
    return list(db.connect().execute(
        "SELECT * FROM dance_material_usage_events WHERE material_id = ? "
        "ORDER BY id DESC LIMIT ?", (int(material_id), int(limit))))


def song_events(db: Database, target_song_id: int, *, limit: int = 500) -> list[Any]:
    """整首歌的事件流水（带素材信息），给历史面板用。"""
    return list(db.connect().execute(
        "SELECT e.*, m.person, m.segment_index AS material_segment, m.file_path "
        "FROM dance_material_usage_events e "
        "JOIN dance_materials m ON m.id = e.material_id "
        "WHERE m.target_song_id = ? ORDER BY e.id DESC LIMIT ?",
        (int(target_song_id), int(limit))))


def event_summary(db: Database, target_song_id: int) -> dict[str, int]:
    """这首歌各类事件各多少条。统计面板的第一行数字。"""
    rows = db.connect().execute(
        "SELECT e.event, COUNT(*) AS n FROM dance_material_usage_events e "
        "JOIN dance_materials m ON m.id = e.material_id "
        "WHERE m.target_song_id = ? GROUP BY e.event", (int(target_song_id),))
    return {str(r["event"]): int(r["n"]) for r in rows}


__all__ = [
    "COUNT_AS_USE", "COUNT_AS_OUTPUT",
    "note_event", "note_events", "note_candidates", "note_montage", "note_render",
    "note_rejected", "recount_material", "recount_song",
    "position_usage", "material_events", "song_events", "event_summary",
]


