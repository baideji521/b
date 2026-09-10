"""舞蹈子系统的数据库读写层。业务层只调这里，**不自己写 SQL**。

复用 `b` 已有的 `Database`（同一个 video.db、同一套 WAL + 写锁 + 外键），
不新建第二个 SQLite（技术指导第二十一节第 5 条）。

分工和 `db/repo.py` 一致：这一层只搬数据，不做任何决策 ——
"该不该重算对齐"、"这条素材该不该进候选池"都不是这里的事。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from ..db.db import Database
from ..logging_setup import get_logger
from .types import (
    DanceAlignment,
    DanceMaterial,
    MontageStrategy,
    RecommendationItem,
    RecommendationRun,
    SegmentTemplate,
)

logger = get_logger("dance.repo")


def now() -> str:
    """统一时间戳，和 `db/repo.py:now()` 同一格式，方便两边的时间放一起比。"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(text: Any) -> Any:
    """读库里的 JSON 文本。坏 JSON 返回 None，不让一条脏数据把整个界面打不开。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        logger.warning("解析库里的 JSON 失败：%s", exc)
        return None

# ================================================================== 目标歌
def upsert_song(db: Database, *, fingerprint: str, file_path: str, file_name: str,
                title: str = "", duration: float | None = None,
                sample_rate: int | None = None) -> int:
    """按指纹登记/更新一首目标歌，返回 id。改名换目录仍是同一首。"""
    stamp = now()
    with db.tx() as conn:
        row = conn.execute("SELECT id FROM dance_target_songs WHERE fingerprint = ?",
                           (fingerprint,)).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE dance_target_songs SET file_path=?, file_name=?, "
                "title=COALESCE(NULLIF(?, ''), title), duration=COALESCE(?, duration), "
                "sample_rate=COALESCE(?, sample_rate), exists_on_disk=1, updated_at=? "
                "WHERE id=?",
                (str(file_path), file_name, title, duration, sample_rate, stamp, row["id"]))
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO dance_target_songs(fingerprint, file_path, file_name, title, "
            "duration, sample_rate, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (fingerprint, str(file_path), file_name, title or file_name,
             duration, sample_rate, stamp, stamp))
        return int(cursor.lastrowid)


def save_song_analysis(db: Database, song_id: int, analysis: dict[str, Any]) -> None:
    """把 `music_structure.analyze_song()` 的结果整份缓存进库。

    四份 JSON 一起写：分开写会出现"节拍是新版算的、段落还是旧版"的错配，
    而这两者一旦口径不一致，切出来的素材就会和界面显示的段落对不上。
    """
    stamp = now()
    with db.tx() as conn:
        conn.execute(
            "UPDATE dance_target_songs SET duration=?, sample_rate=?, bpm=?, beat_count=?, "
            "beats_json=?, sections_json=?, features_json=?, rhythm_json=?, "
            "analysis_version=?, updated_at=? WHERE id=?",
            (analysis.get("duration"), analysis.get("sample_rate"), analysis.get("bpm"),
             analysis.get("beat_count"), _dumps(analysis.get("beats_json")),
             _dumps(analysis.get("sections_json")), _dumps(analysis.get("features_json")),
             _dumps(analysis.get("rhythm_json")), analysis.get("analysis_version"),
             stamp, song_id))


def get_song(db: Database, song_id: int):
    return db.connect().execute("SELECT * FROM dance_target_songs WHERE id = ?",
                                (song_id,)).fetchone()


def get_song_by_fingerprint(db: Database, fingerprint: str):
    return db.connect().execute("SELECT * FROM dance_target_songs WHERE fingerprint = ?",
                                (fingerprint,)).fetchone()


def get_song_by_path(db: Database, file_path: str):
    return db.connect().execute("SELECT * FROM dance_target_songs WHERE file_path = ?",
                                (str(file_path),)).fetchone()


def list_songs(db: Database, *, include_retired: bool = False) -> list[Any]:
    """目标歌清单。默认**不含已下架的** —— 界面下拉框只该看到还在用的那几首。"""
    sql = ("SELECT s.*, r.retired_at, r.reason AS retired_reason "
           "FROM dance_target_songs s "
           "LEFT JOIN dance_song_retirement r ON r.target_song_id = s.id ")
    if not include_retired:
        sql += "WHERE r.target_song_id IS NULL "
    sql += "ORDER BY s.updated_at DESC, s.id DESC"
    return list(db.connect().execute(sql))


# ------------------------------------------------------------ 下架 / 恢复 / 删除
#: `song_usage()` 各项的中文名。返回的字典键保持英文（给代码用、稳定），
#: 摆给用户看的时候一律走这张表 —— CLI、界面、异常信息共用同一份措辞
USAGE_LABELS = {
    "materials": "素材",
    "alignments": "对齐",
    "montages": "混剪",
    "versions": "成片版本",
    "events": "使用事件",
}


def describe_usage(usage: Mapping[str, int], *, only_nonzero: bool = True) -> str:
    """把 `song_usage()` 的结果写成一句中文，比如"素材 6、混剪 1、成片版本 1"。"""
    parts = [f"{USAGE_LABELS.get(key, key)} {usage.get(key, 0)}"
             for key in USAGE_LABELS
             if not only_nonzero or usage.get(key, 0)]
    return "、".join(parts) or "没有任何附属数据"


def song_usage(db: Database, song_id: int) -> dict[str, int]:

    """这首歌下面挂着多少东西。删之前必须先把这几个数摆给用户看。"""
    conn = db.connect()

    def count(sql: str) -> int:
        return int(conn.execute(sql, (int(song_id),)).fetchone()[0])

    materials = count("SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ?")
    return {
        "materials": materials,
        "alignments": count(
            "SELECT COUNT(*) FROM dance_audio_alignments WHERE target_song_id = ?"),
        "montages": count("SELECT COUNT(*) FROM dance_montages WHERE target_song_id = ?"),
        "versions": count(
            "SELECT COUNT(*) FROM dance_montage_versions v "
            "JOIN dance_montages m ON m.id = v.montage_id WHERE m.target_song_id = ?"),
        "events": count(
            "SELECT COUNT(*) FROM dance_material_usage_events e "
            "JOIN dance_materials m ON m.id = e.material_id WHERE m.target_song_id = ?"),
    }


def retire_song(db: Database, song_id: int, *, reason: str, operator: str = "") -> bool:
    """把目标歌**下架**：从界面上消失，但素材/对齐/历史成片一条都不动。

    这是"删除目标歌"的默认答案。为什么不直接删：`dance_materials` /
    `dance_audio_alignments` / `dance_montages` 都是 `ON DELETE CASCADE`，
    真删下去会连带把素材、对齐、历史成片和它们的事件流水全部抹掉 ——
    而这个项目的整个立足点就是"素材是长期资产、历史永不删除"。

    `reason` 留空直接抛 ValueError：和人工修正 offset 一样，
    任何让东西从界面上消失的操作都不许静默进行。
    """
    if not str(reason).strip():
        raise ValueError("下架目标歌必须写理由（禁止静默隐藏）")
    if get_song(db, int(song_id)) is None:
        raise ValueError(f"目标歌 #{song_id} 不存在")
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO dance_song_retirement(target_song_id, reason, operator, retired_at) "
            "VALUES(?,?,?,?) ON CONFLICT(target_song_id) DO UPDATE SET "
            "reason=excluded.reason, operator=excluded.operator, retired_at=excluded.retired_at",
            (int(song_id), str(reason).strip(), str(operator), now()))
    logger.info("目标歌 #%d 已下架：%s", int(song_id), reason)
    return True


def restore_song(db: Database, song_id: int) -> bool:
    """撤销下架。下架本来就是可逆的，这就是那条回头路。"""
    with db.tx() as conn:
        changed = conn.execute(
            "DELETE FROM dance_song_retirement WHERE target_song_id = ?",
            (int(song_id),)).rowcount
    if changed:
        logger.info("目标歌 #%d 已恢复", int(song_id))
    return bool(changed)


def song_retirement(db: Database, song_id: int):
    """这首歌的下架记录，没下架就返回 None。"""
    return db.connect().execute(
        "SELECT * FROM dance_song_retirement WHERE target_song_id = ?",
        (int(song_id),)).fetchone()


def delete_song(db: Database, song_id: int, *, confirm: bool = False) -> dict[str, int]:
    """**彻底删除**目标歌，连带级联删掉它的素材/对齐/混剪/事件。返回删掉了多少东西。

    只在一种情况下适合用它：歌是误加进来的（选错文件、重复导入），
    下面那些素材本来就不该存在。

    `confirm=False` 时，只要下面还挂着任何东西就**拒绝执行**并抛 ValueError ——
    调用方必须先把 `song_usage()` 的数字摆给用户看、拿到明确同意，再带 `confirm=True`
    回来。素材文件本身**不删**：盘上的东西留着，用户想清理可以自己去目录里删。
    """
    row = get_song(db, int(song_id))
    if row is None:
        raise ValueError(f"目标歌 #{song_id} 不存在")
    usage = song_usage(db, int(song_id))
    attached = {k: v for k, v in usage.items() if v}
    if attached and not confirm:
        raise ValueError(
            f"目标歌 #{song_id}《{row['title']}》下面还挂着 {describe_usage(usage)}；"
            "彻底删除会把这些连带删掉。确认要删就带 confirm=True 再来一次，"
            "只想让它从界面上消失请用 retire_song()")
    with db.tx() as conn:
        conn.execute("DELETE FROM dance_target_songs WHERE id = ?", (int(song_id),))
    logger.warning("目标歌 #%d《%s》已彻底删除，连带 %s（磁盘上的素材文件保留）",
                   int(song_id), row["title"], describe_usage(usage))
    return usage




def song_sections(db: Database, song_id: int) -> list[dict[str, Any]]:
    """读缓存的段落。没分析过返回空列表 —— 上层据此决定"要不要先分析"。"""
    row = get_song(db, song_id)
    data = _loads(row["sections_json"]) if row is not None else None
    return data if isinstance(data, list) else []


def song_beats(db: Database, song_id: int) -> dict[str, Any]:
    row = get_song(db, song_id)
    data = _loads(row["beats_json"]) if row is not None else None
    return data if isinstance(data, dict) else {}

# ============================================================ 段落模板（v13）
#: 没起名字的模板都叫这个 —— 一首歌一份"默认分段"，够用而且好找
DEFAULT_TEMPLATE_NAME = "默认"


def save_segment_template(db: Database, template: SegmentTemplate, *,
                          name: str = "", make_active: bool = True) -> int:
    """存一份段落模板，按 `(歌, 名字)` 幂等（同名就更新那一份）。

    存进去之前先 `validate` —— 半份模板（有缝/重叠/越界）绝不许落库，
    否则读出来的每一处都得再判一遍，而且总会有人忘了判。
    `make_active` 只影响"这首歌默认用哪一份"，一首歌同一时刻只有一份是活的。
    """
    from . import segment_template as editor      # 只为校验/序列化，避免顶层循环导入

    editor.validate(template)
    label = (name or template.name or DEFAULT_TEMPLATE_NAME).strip() or DEFAULT_TEMPLATE_NAME
    stamp = now()
    payload = editor.to_json(template)
    with db.tx() as conn:
        row = conn.execute(
            "SELECT id FROM dance_segment_templates "
            "WHERE target_song_id = ? AND name = ?",
            (int(template.target_song_id), label)).fetchone()
        if row is None:
            cursor = conn.execute(
                "INSERT INTO dance_segment_templates (target_song_id, name, duration, "
                "segment_count, spans_json, source, template_version, is_active, note, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (int(template.target_song_id), label, float(template.duration),
                 len(template.spans), payload, template.source,
                 template.version or editor.TEMPLATE_VERSION,
                 1 if make_active else 0, template.note, stamp, stamp))
            template_id = int(cursor.lastrowid)
        else:
            template_id = int(row["id"])
            conn.execute(
                "UPDATE dance_segment_templates SET duration=?, segment_count=?, "
                "spans_json=?, source=?, template_version=?, note=?, updated_at=? "
                "WHERE id=?",
                (float(template.duration), len(template.spans), payload, template.source,
                 template.version or editor.TEMPLATE_VERSION, template.note, stamp,
                 template_id))
        if make_active:
            conn.execute(
                "UPDATE dance_segment_templates SET is_active = CASE WHEN id = ? "
                "THEN 1 ELSE 0 END, updated_at = CASE WHEN id = ? THEN ? "
                "ELSE updated_at END WHERE target_song_id = ?",
                (template_id, template_id, stamp, int(template.target_song_id)))
    logger.info("段落模板「%s」已存：歌 #%d，%d 段，来源 %s",
                label, int(template.target_song_id), len(template.spans), template.source)
    return template_id


def _template(row: Any) -> SegmentTemplate:
    from . import segment_template as editor

    built = editor.from_json(row["spans_json"], target_song_id=int(row["target_song_id"]))
    return SegmentTemplate(target_song_id=built.target_song_id, duration=built.duration,
                           spans=built.spans, source=built.source,
                           name=str(row["name"] or ""), note=built.note,
                           version=str(row["template_version"] or built.version))


def get_segment_template(db: Database, template_id: int) -> SegmentTemplate | None:
    row = db.connect().execute("SELECT * FROM dance_segment_templates WHERE id = ?",
                               (int(template_id),)).fetchone()
    return _template(row) if row is not None else None


def active_segment_template(db: Database, song_id: int) -> SegmentTemplate | None:
    """这首歌当前生效的段落模板。**没有就返回 None** —— 上层据此回落成等间隔，
    不要在这儿现编一份：编出来的东西会被当成"用户定过的分段"。
    """
    row = db.connect().execute(
        "SELECT * FROM dance_segment_templates WHERE target_song_id = ? "
        "ORDER BY is_active DESC, updated_at DESC LIMIT 1", (int(song_id),)).fetchone()
    return _template(row) if row is not None else None


def list_segment_templates(db: Database, song_id: int) -> list[Any]:
    return list(db.connect().execute(
        "SELECT id, name, duration, segment_count, source, is_active, note, updated_at "
        "FROM dance_segment_templates WHERE target_song_id = ? "
        "ORDER BY is_active DESC, updated_at DESC", (int(song_id),)))


def set_active_segment_template(db: Database, template_id: int) -> bool:
    """切换"这首歌用哪一份分段"。模板不存在返回 False，不抛。"""
    row = db.connect().execute(
        "SELECT target_song_id FROM dance_segment_templates WHERE id = ?",
        (int(template_id),)).fetchone()
    if row is None:
        return False
    with db.tx() as conn:
        conn.execute(
            "UPDATE dance_segment_templates SET is_active = CASE WHEN id = ? "
            "THEN 1 ELSE 0 END WHERE target_song_id = ?",
            (int(template_id), int(row["target_song_id"])))
    return True


def delete_segment_template(db: Database, template_id: int) -> bool:
    """删掉一份模板。已经按它切出来的素材一条都不动 —— 素材自己记着
    绑在哪一段（`dance_materials.segment_index/target_start/target_end`），
    删模板不等于删素材，这一点不许含糊。
    """
    with db.tx() as conn:
        cursor = conn.execute("DELETE FROM dance_segment_templates WHERE id = ?",
                              (int(template_id),))
    return cursor.rowcount > 0


# ================================================== 人工编排（v14：最终选择等）
class SelectionError(ValueError):
    """最终选择非法：素材不存在，或者它绑的段落和要放的段落不是一个。

    界面上已经拦了一层（拖拽时不给落），这里是**第二道**：任何走到库里的路
    都得过这一关。只靠界面拦不住脚本、CLI、以后新写的页面。
    """


def set_final_selection(db: Database, target_song_id: int, segment_index: int,
                        material_id: int, *, note: str = "") -> None:
    """定下"这一段用哪条素材"。素材的 segment_index 必须等于这一段，否则抛。

    一段只有一个最终选择（主键就是 `(歌, 段落)`），重复设置就是覆盖。
    """
    row = db.connect().execute(
        "SELECT segment_index, target_song_id FROM dance_materials WHERE id = ?",
        (int(material_id),)).fetchone()
    if row is None:
        raise SelectionError(f"素材 #{material_id} 不在库里")
    if int(row["segment_index"]) != int(segment_index):
        raise SelectionError(
            f"素材 #{material_id} 绑在 S{int(row['segment_index']) + 1}，"
            f"不能放到 S{int(segment_index) + 1} —— 素材和音乐位置是绑死的")
    if int(row["target_song_id"]) != int(target_song_id):
        raise SelectionError(f"素材 #{material_id} 属于另一首歌，不能用在这里")
    stamp = now()
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO dance_final_selections (target_song_id, segment_index, "
            "material_id, note, selected_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(target_song_id, segment_index) DO UPDATE SET "
            "material_id=excluded.material_id, note=excluded.note, "
            "selected_at=excluded.selected_at",
            (int(target_song_id), int(segment_index), int(material_id), note, stamp))


def final_selections(db: Database, target_song_id: int) -> dict[int, int]:
    """`{段落: 素材id}`。FINAL TIMELINE 重新打开就是靠它恢复的。"""
    rows = db.connect().execute(
        "SELECT segment_index, material_id FROM dance_final_selections "
        "WHERE target_song_id = ? ORDER BY segment_index", (int(target_song_id),))
    return {int(r["segment_index"]): int(r["material_id"]) for r in rows}


def clear_final_selection(db: Database, target_song_id: int, segment_index: int) -> bool:
    with db.tx() as conn:
        cursor = conn.execute(
            "DELETE FROM dance_final_selections WHERE target_song_id = ? "
            "AND segment_index = ?", (int(target_song_id), int(segment_index)))
    return cursor.rowcount > 0


# 「⭐ 可取标记」这套读写已经删掉了：编排台上不再有标记按钮、标记点和标记数据，
# 段落边界由用户直接切分决定，不需要第二套"以后也许有用"的记号。
# v14 建的 `dance_cut_marks` 表留在建表脚本里不动 —— 迁移这条链只许 CREATE，
# 删表会让"升级上来的库"和"新建的库"schema 文本不一致（v4 踩过的坑）。



def save_candidate_order(db: Database, target_song_id: int, segment_index: int,
                         material_ids: Sequence[int]) -> None:
    """存这一段候选素材的排列顺序（用户拖出来的那份）。"""
    stamp = now()
    payload = _dumps([int(m) for m in material_ids]) or "[]"
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO dance_candidate_order (target_song_id, segment_index, "
            "order_json, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(target_song_id, segment_index) DO UPDATE SET "
            "order_json=excluded.order_json, updated_at=excluded.updated_at",
            (int(target_song_id), int(segment_index), payload, stamp))


def candidate_order(db: Database, target_song_id: int, segment_index: int) -> list[int]:
    """存过的顺序；没存过返回空列表（上层就按默认排序显示）。"""
    row = db.connect().execute(
        "SELECT order_json FROM dance_candidate_order WHERE target_song_id = ? "
        "AND segment_index = ?", (int(target_song_id), int(segment_index))).fetchone()
    data = _loads(row["order_json"]) if row is not None else None
    return [int(x) for x in data] if isinstance(data, list) else []


def order_materials(materials: Sequence[Any], order: Sequence[int]) -> list[Any]:
    """按存过的顺序重排素材；存过之后新入库的素材排在后面，一条都不丢。"""
    rank = {int(m): i for i, m in enumerate(order)}
    tail = len(rank)
    return sorted(materials, key=lambda m: (rank.get(int(m.id), tail), int(m.id)))


# ============================================== 片段仓库：按「歌 + 段落」隔离的候选池
#
# 这一节是自动编排的地基，只有一条硬规则：
#
#     一个段落只能用属于这个段落的素材。
#
# 所以查询入口就叫 `get_candidates(song, segment)` —— 想拿素材必须同时说清
# 「哪首歌」和「第几段」，**没有** `get_all_clips(song)` 这种把所有段落混在一起
# 让人随机挑的口子。素材本身在 `dance_materials` 里就带着 segment_index
# （切片那一步写的），这里不再另建一份平行的仓库表，免得两份真相打架。
def get_candidates(db: Database, target_song_id: int, segment_index: int,
                   *, statuses: Sequence[str] = ("ready",)) -> list[DanceMaterial]:
    """第 `segment_index` 段的候选池。**跨段落取素材在这一层就不可能。**

    以后自动编排就走这个口子：`score(get_candidates(song, seg))` 选最佳，
    而不是 `random(all_clips(song))` —— 后者会让画面和音乐错开。
    """
    return materials_at(db, target_song_id, segment_index, statuses=statuses)


def candidate_counts(db: Database, target_song_id: int,
                     *, statuses: Sequence[str] = ("ready",)) -> dict[int, int]:
    """每一段各有几条候选：`{段落: 条数}`。界面上"已生成片段"就是它的和。"""
    marks = ",".join("?" for _ in statuses) or "''"
    rows = db.connect().execute(
        f"SELECT segment_index, COUNT(*) AS n FROM dance_materials "  # noqa: S608
        f"WHERE target_song_id = ? AND status IN ({marks}) "
        "GROUP BY segment_index ORDER BY segment_index",
        (int(target_song_id), *[str(s) for s in statuses])).fetchall()
    return {int(r["segment_index"]): int(r["n"]) for r in rows}


def song_identity(db: Database, target_song_id: int) -> dict[str, Any]:
    """歌和音频文件的映射，两者**分开**存放，不揉成一个字段。

    `audio_*` 说的是硬盘上那个文件（never_trust_01.mp3），
    `song_*` 说的是这首歌（Never Trust Prediction）。
    片段要能追溯到这两样，以后才分得清"同一首歌的不同音频版本"。
    """
    row = get_song(db, int(target_song_id))
    if row is None:
        return {"song_id": 0, "song_name": "", "audio_id": 0,
                "audio_name": "", "audio_path": ""}
    title = str(row["title"] or "").strip()
    name = str(row["file_name"] or "").strip()
    return {"song_id": int(row["id"]),
            # 没填标题就退回文件名（去掉后缀）：宁可重复，也不要空着没法追溯
            "song_name": title or Path(name).stem,
            "audio_id": int(row["id"]),          # 一首歌当前只挂一个音频文件
            "audio_name": name,
            "audio_path": str(row["file_path"] or "")}


def repository_summary(db: Database, target_song_id: int) -> dict[str, Any]:
    """片段仓库那一栏要显示的东西：音频名、歌名、每段几条、一共几条。"""
    counts = candidate_counts(db, int(target_song_id))
    info = song_identity(db, int(target_song_id))
    info.update({"per_segment": counts, "clip_count": sum(counts.values()),
                 "segment_count": len(counts)})
    return info


def log_manual_selection(db: Database, *, target_song_id: int, segment_index: int,
                         material_id: int, segment_start: float = 0.0,
                         segment_end: float = 0.0, decided_by: str = "manual_selection",
                         note: str = "") -> int:
    """记一笔"人工把这条素材拖进了这一段"。**只追加，不覆盖。**

    `dance_final_selections` 只留"现在用谁"，会被下一次拖拽盖掉；这张流水账
    留的是人做过的每一次判断 —— 以后统计规律、训练自动排序全靠它。
    源视频、片段路径、歌名/音频名都在写入时定好，素材以后被删也还追得回来。
    """
    material = get_material(db, int(material_id))
    if material is None:
        raise SelectionError(f"素材 #{material_id} 不在库里")
    if int(material.segment_index) != int(segment_index):
        raise SelectionError(
            f"素材 #{material_id} 绑在 S{int(material.segment_index) + 1}，"
            f"不能记成 S{int(segment_index) + 1} 的选择")
    info = song_identity(db, int(target_song_id))
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_manual_selections("
            "target_song_id, segment_index, material_id, source_video_id, audio_name,"
            " song_name, clip_path, segment_start, segment_end, decided_by, note,"
            " decided_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(target_song_id), int(segment_index), int(material_id),
             int(getattr(material, "source_video_id", 0) or 0),
             info["audio_name"], info["song_name"],
             str(getattr(material, "file_path", "") or ""),
             round(float(segment_start), 3), round(float(segment_end), 3),
             str(decided_by), str(note), now()))
        return int(cursor.lastrowid)


def manual_selections(db: Database, target_song_id: int, *, segment_index: int | None = None,
                      limit: int = 500) -> list[Any]:
    """人工选择流水账，最近的在前。给统计和以后的自动编排看。"""
    sql = "SELECT * FROM dance_manual_selections WHERE target_song_id = ?"
    args: list[Any] = [int(target_song_id)]
    if segment_index is not None:
        sql += " AND segment_index = ?"
        args.append(int(segment_index))
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    return list(db.connect().execute(sql, tuple(args)).fetchall())



# ==================================================================== 对齐
def save_alignment(db: Database, *, source_video_id: int, target_song_id: int,
                   cache_key: str, alignment: DanceAlignment, config_hash: str = "") -> int:
    """存一条对齐结果，按 `cache_key` 幂等（同一套输入重复跑就更新那一条）。

    人工修正过的记录**不会**被算法结果覆盖：`manual_offset IS NOT NULL` 时
    只更新算法侧那几列，`offset_seconds` 和 `status` 保持人工值 ——
    否则用户改完 offset、下次重跑对齐就白改了。
    """
    stamp = now()
    detail = _dumps(alignment.to_dict())
    with db.tx() as conn:
        row = conn.execute(
            "SELECT id, manual_offset FROM dance_audio_alignments WHERE cache_key = ?",
            (cache_key,)).fetchone()
        if row is not None:
            if row["manual_offset"] is not None:
                conn.execute(
                    "UPDATE dance_audio_alignments SET waveform_offset=?, "
                    "waveform_confidence=?, chroma_offset=?, chroma_confidence=?, "
                    "window_count=?, max_deviation=?, agreement=?, detail_json=?, "
                    "updated_at=? WHERE id=?",
                    (alignment.waveform_offset, alignment.waveform_confidence,
                     alignment.chroma_offset, alignment.chroma_confidence,
                     alignment.window_count, alignment.max_deviation, alignment.agreement,
                     detail, stamp, row["id"]))
                logger.info("对齐 #%d 已被人工改过，只更新算法侧字段", int(row["id"]))
                return int(row["id"])
            conn.execute(
                "UPDATE dance_audio_alignments SET offset_seconds=?, confidence=?, method=?, "
                "waveform_offset=?, waveform_confidence=?, chroma_offset=?, "
                "chroma_confidence=?, window_count=?, max_deviation=?, agreement=?, "
                "status=?, algorithm_version=?, config_hash=?, source_duration=?, "
                "target_duration=?, detail_json=?, updated_at=? WHERE id=?",
                (alignment.offset, alignment.confidence, alignment.method,
                 alignment.waveform_offset, alignment.waveform_confidence,
                 alignment.chroma_offset, alignment.chroma_confidence,
                 alignment.window_count, alignment.max_deviation, alignment.agreement,
                 alignment.status, alignment.algorithm_version, config_hash,
                 alignment.source_duration, alignment.target_duration, detail,
                 stamp, row["id"]))
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO dance_audio_alignments(source_video_id, target_song_id, cache_key, "
            "offset_seconds, confidence, method, waveform_offset, waveform_confidence, "
            "chroma_offset, chroma_confidence, window_count, max_deviation, agreement, "
            "status, algorithm_version, config_hash, source_duration, target_duration, "
            "detail_json, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (source_video_id, target_song_id, cache_key, alignment.offset,
             alignment.confidence, alignment.method, alignment.waveform_offset,
             alignment.waveform_confidence, alignment.chroma_offset,
             alignment.chroma_confidence, alignment.window_count, alignment.max_deviation,
             alignment.agreement, alignment.status, alignment.algorithm_version,
             config_hash, alignment.source_duration, alignment.target_duration,
             detail, stamp, stamp))
        return int(cursor.lastrowid)


def alignment_by_key(db: Database, cache_key: str):
    """按缓存键取对齐。命中就不用重算（技术指导第二十四节）。"""
    return db.connect().execute(
        "SELECT * FROM dance_audio_alignments WHERE cache_key = ?", (cache_key,)).fetchone()


def get_alignment(db: Database, alignment_id: int):
    return db.connect().execute("SELECT * FROM dance_audio_alignments WHERE id = ?",
                                (alignment_id,)).fetchone()


def alignments_for_song(db: Database, target_song_id: int, *,
                        statuses: Sequence[str] = ()) -> list[Any]:
    sql = ("SELECT a.*, v.file_path AS source_path, v.file_name AS source_name "
           "FROM dance_audio_alignments a JOIN videos v ON v.id = a.source_video_id "
           "WHERE a.target_song_id = ?")
    params: list[Any] = [target_song_id]
    if statuses:
        sql += " AND a.status IN (%s)" % ",".join("?" for _ in statuses)
        params.extend(statuses)
    sql += " ORDER BY a.confidence DESC, a.id"
    return list(db.connect().execute(sql, tuple(params)))


def set_manual_offset(db: Database, alignment_id: int, *, offset: float, reason: str,
                      operator: str = "") -> bool:
    """人工改 offset 并留痕。理由为空直接拒 —— 禁止静默覆盖（技术指导第五节第 7 条）。"""
    if not str(reason).strip():
        raise ValueError("人工修正必须写理由（禁止静默覆盖）")
    stamp = now()
    with db.tx() as conn:
        row = conn.execute("SELECT offset_seconds, confidence, original_offset "
                           "FROM dance_audio_alignments WHERE id=?", (alignment_id,)).fetchone()
        if row is None:
            return False
        # original_* 只在第一次写入：留住的必须是**算法**原值，不是上一次的人工值
        first_offset = (row["original_offset"] if row["original_offset"] is not None
                        else row["offset_seconds"])
        conn.execute(
            "UPDATE dance_audio_alignments SET offset_seconds=?, status='manual', "
            "original_offset=?, original_confidence=COALESCE(original_confidence, ?), "
            "manual_offset=?, manual_reason=?, manual_operator=?, manual_at=?, updated_at=? "
            "WHERE id=?",
            (float(offset), first_offset, row["confidence"], float(offset),
             reason.strip(), str(operator or ""), stamp, stamp, alignment_id))
        return True

#: 查素材时统一带上的 JOIN 与派生列，保证 `DanceMaterial.from_row` 拿到的字段一致
_MATERIAL_SELECT = (
    "SELECT m.*, v.file_name AS source_name "
    "FROM dance_materials m JOIN videos v ON v.id = m.source_video_id ")


# ==================================================================== 素材
def upsert_material(db: Database, *, source_video_id: int, alignment_id: int,
                    target_song_id: int, segment_index: int, target_start: float,
                    target_end: float, source_start: float, source_end: float,
                    duration: float, generation_version: str,
                    file_path: str = "", file_hash: str = "",
                    alignment_confidence: float = 0.0, person: str = "",
                    source_group: str = "", quality: float | None = None) -> int:
    """登记/更新一条素材，按 `(歌, 源, 位置, 切片版本)` 幂等。

    **不动使用计数**：重切一遍素材不该把它的历史使用记录清零 ——
    那些数字是资产的价值所在（技术指导第八节：不要物理删除历史素材）。
    """
    stamp = now()
    with db.tx() as conn:
        row = conn.execute(
            "SELECT id FROM dance_materials WHERE target_song_id=? AND source_video_id=? "
            "AND segment_index=? AND generation_version=?",
            (target_song_id, source_video_id, segment_index, generation_version)).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE dance_materials SET alignment_id=?, target_start=?, target_end=?, "
                "source_start=?, source_end=?, duration=?, file_path=?, file_hash=?, "
                "alignment_confidence=?, person=COALESCE(NULLIF(?, ''), person), "
                "source_group=COALESCE(NULLIF(?, ''), source_group), "
                "quality=COALESCE(?, quality), status='ready', updated_at=? WHERE id=?",
                (alignment_id, target_start, target_end, source_start, source_end, duration,
                 str(file_path), file_hash, alignment_confidence, person, source_group,
                 quality, stamp, row["id"]))
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO dance_materials(source_video_id, alignment_id, target_song_id, "
            "segment_index, target_start, target_end, source_start, source_end, duration, "
            "file_path, file_hash, alignment_confidence, generation_version, person, "
            "source_group, quality, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (source_video_id, alignment_id, target_song_id, segment_index, target_start,
             target_end, source_start, source_end, duration, str(file_path), file_hash,
             alignment_confidence, generation_version, person, source_group, quality,
             stamp, stamp))
        return int(cursor.lastrowid)


def get_material(db: Database, material_id: int) -> DanceMaterial | None:
    row = db.connect().execute(_MATERIAL_SELECT + "WHERE m.id = ?", (material_id,)).fetchone()
    return DanceMaterial.from_row(row) if row is not None else None


def get_materials(db: Database, material_ids: Iterable[int]) -> dict[int, DanceMaterial]:
    ids = [int(i) for i in material_ids]
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    rows = db.connect().execute(_MATERIAL_SELECT + f"WHERE m.id IN ({marks})", tuple(ids))
    return {int(row["id"]): DanceMaterial.from_row(row) for row in rows}


def materials_at(db: Database, target_song_id: int, segment_index: int, *,
                 statuses: Sequence[str] = ("ready",)) -> list[DanceMaterial]:
    """某个固定音乐位置上所有可用素材。候选池就是从这里起步的。"""
    sql = _MATERIAL_SELECT + "WHERE m.target_song_id=? AND m.segment_index=?"
    params: list[Any] = [target_song_id, segment_index]
    if statuses:
        sql += " AND m.status IN (%s)" % ",".join("?" for _ in statuses)
        params.extend(statuses)
    sql += " ORDER BY m.id"
    return [DanceMaterial.from_row(row) for row in db.connect().execute(sql, tuple(params))]


def set_material_status(db: Database, material_id: int, status: str) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE dance_materials SET status=?, updated_at=? WHERE id=?",
                     (status, now(), material_id))


def mark_regenerated(db: Database, target_song_id: int, source_video_id: int,
                     keep_version: str) -> int:
    """把这个源在这首歌上的**旧版本**素材标 regenerated，返回条数。

    只改状态不删行：历史成品还引用着它们，删掉就断了血缘
    （技术指导第八节：不要物理删除历史素材）。
    """
    with db.tx() as conn:
        cursor = conn.execute(
            "UPDATE dance_materials SET status='regenerated', updated_at=? "
            "WHERE target_song_id=? AND source_video_id=? AND generation_version<>? "
            "AND status='ready'", (now(), target_song_id, source_video_id, keep_version))
        return int(cursor.rowcount or 0)

# ============================================================== 混剪与版本
def create_montage(db: Database, *, target_song_id: int, name: str,
                   slice_duration: float, note: str = "") -> int:
    stamp = now()
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_montages(target_song_id, name, slice_duration, note, "
            "created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (target_song_id, name, float(slice_duration), note, stamp, stamp))
        return int(cursor.lastrowid)


def get_montage(db: Database, montage_id: int):
    return db.connect().execute("SELECT * FROM dance_montages WHERE id = ?",
                                (montage_id,)).fetchone()


def list_montages(db: Database, target_song_id: int | None = None) -> list[Any]:
    sql = "SELECT * FROM dance_montages WHERE deleted_at IS NULL"
    params: tuple[Any, ...] = ()
    if target_song_id is not None:
        sql += " AND target_song_id = ?"
        params = (target_song_id,)
    return list(db.connect().execute(sql + " ORDER BY updated_at DESC, id DESC", params))


def ensure_montage(db: Database, *, target_song_id: int, name: str,
                   slice_duration: float) -> int:
    """同名的就复用，没有才新建。GUI 反复点"生成"不该攒出一堆空任务。"""
    row = db.connect().execute(
        "SELECT id FROM dance_montages WHERE target_song_id=? AND name=? "
        "AND deleted_at IS NULL", (target_song_id, name)).fetchone()
    if row is not None:
        return int(row["id"])
    return create_montage(db, target_song_id=target_song_id, name=name,
                          slice_duration=slice_duration)


def next_version_index(db: Database, montage_id: int) -> int:
    """下一版的版本号。历史版本永不覆盖，所以只会往上加。"""
    row = db.connect().execute(
        "SELECT COALESCE(MAX(version_index), 0) AS top FROM dance_montage_versions "
        "WHERE montage_id = ?", (montage_id,)).fetchone()
    return int(row["top"]) + 1


def save_version(db: Database, *, montage_id: int, version_index: int, signature: str,
                 strategy_id: int | None, recommendation_run_id: int | None,
                 clips: Sequence[Any], duration: float, repeat: Any,
                 timeline_json: dict[str, Any] | None = None) -> int:
    """存一版混剪（版本行 + 编辑计划明细），返回 version_id。

    两张表在**同一个事务**里写完：版本行存在而明细缺失的话，
    这一版就变成一条查不出内容的孤儿记录，重复率数字也失去了依据。
    """
    stamp = now()
    stats = repeat.to_dict() if hasattr(repeat, "to_dict") else dict(repeat or {})
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_montage_versions(montage_id, version_index, signature, "
            "strategy_id, recommendation_run_id, clip_count, duration, material_repeat, "
            "position_repeat, person_repeat, source_repeat, pair_repeat, "
            "combination_repeat, overall_repeat, repeat_detail_json, timeline_json, "
            "render_status, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'planned',?,?)",
            (montage_id, int(version_index), signature, strategy_id, recommendation_run_id,
             len(clips), float(duration),
             stats.get("material_repeat", 0.0), stats.get("position_repeat", 0.0),
             stats.get("person_repeat", 0.0), stats.get("source_repeat", 0.0),
             stats.get("pair_repeat", 0.0), stats.get("combination_repeat", 0.0),
             stats.get("overall_repeat", 0.0), _dumps(stats.get("detail")),
             _dumps(timeline_json), stamp, stamp))
        version_id = int(cursor.lastrowid)
        for clip in clips:
            conn.execute(
                "INSERT INTO dance_montage_materials(version_id, material_id, order_index, "
                "segment_index, target_start, target_end, source_start, source_end, "
                "selection_score, score_breakdown_json, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (version_id, clip.material_id, clip.order_index, clip.segment_index,
                 clip.target_start, clip.target_end, clip.source_start, clip.source_end,
                 clip.selection_score, _dumps(clip.score_breakdown), stamp))
        conn.execute("UPDATE dance_montages SET updated_at=? WHERE id=?", (stamp, montage_id))
    return version_id


def set_render_status(db: Database, version_id: int, status: str, *,
                     output_path: str = "", detail: dict[str, Any] | None = None,
                     error: str = "") -> None:
    with db.tx() as conn:
        conn.execute(
            "UPDATE dance_montage_versions SET render_status=?, "
            "output_path=COALESCE(NULLIF(?, ''), output_path), render_detail_json=?, "
            "error=?, updated_at=? WHERE id=?",
            (status, str(output_path), _dumps(detail), error[:2000], now(), version_id))


def get_version(db: Database, version_id: int):
    return db.connect().execute("SELECT * FROM dance_montage_versions WHERE id = ?",
                                (version_id,)).fetchone()


def list_versions(db: Database, montage_id: int) -> list[Any]:
    return list(db.connect().execute(
        "SELECT * FROM dance_montage_versions WHERE montage_id = ? "
        "ORDER BY version_index DESC", (montage_id,)))


def version_materials(db: Database, version_id: int) -> list[Any]:
    return list(db.connect().execute(
        "SELECT vm.*, m.file_path, m.person, m.source_video_id "
        "FROM dance_montage_materials vm JOIN dance_materials m ON m.id = vm.material_id "
        "WHERE vm.version_id = ? ORDER BY vm.order_index", (version_id,)))


def versions_for_song(db: Database, target_song_id: int, *, limit: int = 200) -> list[Any]:
    """这首歌下所有版本（跨混剪任务）。重复率统计要跟历史比，靠它取历史。"""
    return list(db.connect().execute(
        "SELECT v.* FROM dance_montage_versions v "
        "JOIN dance_montages g ON g.id = v.montage_id "
        "WHERE g.target_song_id = ? ORDER BY v.id DESC LIMIT ?",
        (target_song_id, int(limit))))

# ==================================================================== 策略
def _strategy(row: Any) -> MontageStrategy:
    return MontageStrategy(
        id=int(row["id"]), name=str(row["name"]), kind=str(row["kind"]),
        version=str(row["version"]),
        weights=_loads(row["weights_json"]) or {},
        constraints=_loads(row["constraints_json"]) or {},
        search=_loads(row["search_json"]) or {},
        is_default=int(row["is_default"] or 0), note=str(row["note"] or ""),
        created_at=str(row["created_at"] or ""), updated_at=str(row["updated_at"] or ""),
        deleted_at=str(row["deleted_at"] or ""))


def save_strategy(db: Database, strategy: MontageStrategy) -> int:
    """存/更新一份策略。`is_default=1` 时先把别人的默认让出来（部分唯一索引挡着）。"""
    stamp = now()
    with db.tx() as conn:
        if strategy.is_default:
            conn.execute("UPDATE dance_montage_strategies SET is_default=0, updated_at=? "
                         "WHERE is_default=1 AND deleted_at IS NULL", (stamp,))
        if strategy.id:
            conn.execute(
                "UPDATE dance_montage_strategies SET name=?, kind=?, version=?, "
                "weights_json=?, constraints_json=?, search_json=?, is_default=?, note=?, "
                "updated_at=? WHERE id=?",
                (strategy.name, strategy.kind, strategy.version, _dumps(strategy.weights),
                 _dumps(strategy.constraints), _dumps(strategy.search),
                 int(strategy.is_default), strategy.note, stamp, strategy.id))
            return int(strategy.id)
        cursor = conn.execute(
            "INSERT INTO dance_montage_strategies(name, kind, version, weights_json, "
            "constraints_json, search_json, is_default, note, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (strategy.name, strategy.kind, strategy.version, _dumps(strategy.weights),
             _dumps(strategy.constraints), _dumps(strategy.search),
             int(strategy.is_default), strategy.note, stamp, stamp))
        return int(cursor.lastrowid)


def get_strategy(db: Database, strategy_id: int) -> MontageStrategy | None:
    row = db.connect().execute("SELECT * FROM dance_montage_strategies WHERE id=?",
                               (strategy_id,)).fetchone()
    return _strategy(row) if row is not None else None


def default_strategy(db: Database) -> MontageStrategy | None:
    """默认策略；没设过就取最早那一份（和 PRM 那边同一套习惯）。"""
    row = db.connect().execute(
        "SELECT * FROM dance_montage_strategies WHERE is_default=1 AND deleted_at IS NULL"
    ).fetchone()
    if row is None:
        row = db.connect().execute(
            "SELECT * FROM dance_montage_strategies WHERE deleted_at IS NULL "
            "ORDER BY id LIMIT 1").fetchone()
    return _strategy(row) if row is not None else None


def list_strategies(db: Database, *, include_deleted: bool = False) -> list[MontageStrategy]:
    sql = "SELECT * FROM dance_montage_strategies"
    if not include_deleted:
        sql += " WHERE deleted_at IS NULL"
    return [_strategy(row) for row in db.connect().execute(sql + " ORDER BY id")]


def delete_strategy(db: Database, strategy_id: int) -> None:
    """软删。历史版本还记着 strategy_id，硬删就查不出"这一版当初用的什么策略"了。"""
    with db.tx() as conn:
        conn.execute("UPDATE dance_montage_strategies SET deleted_at=?, is_default=0, "
                     "updated_at=? WHERE id=?", (now(), now(), strategy_id))

# ==================================================================== 推荐
def save_recommendation_run(db: Database, run: RecommendationRun,
                            filter_spec: dict[str, Any] | None = None) -> int:
    """存一次推荐（run 行 + items 明细），返回 run_id。

    seed / 策略 / 算法版本全部落库，同一套输入重跑必须得到同样的结果
    （技术指导第十一节：推荐结果必须可重现）。run 和 items 同一事务写完。
    """
    stamp = run.created_at or now()
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_recommendation_runs(target_song_id, montage_id, strategy_id, "
            "strategy_kind, strategy_version, random_seed, candidate_count, "
            "recommended_count, algorithm_version, filter_json, notes, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.target_song_id, run.montage_id or None, run.strategy_id or None,
             run.strategy_kind, run.strategy_version, int(run.random_seed),
             int(run.candidate_count), int(run.recommended_count), run.algorithm_version,
             _dumps(filter_spec), run.notes, stamp))
        run_id = int(cursor.lastrowid)
        for item in run.items:
            conn.execute(
                "INSERT INTO dance_recommendation_items(run_id, material_id, segment_index, "
                "rank, score, score_breakdown_json, reason, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (run_id, item.material_id, item.segment_index, item.rank, item.score,
                 _dumps(item.breakdown), item.reason, stamp))
    return run_id


def get_recommendation_run(db: Database, run_id: int) -> RecommendationRun | None:
    row = db.connect().execute("SELECT * FROM dance_recommendation_runs WHERE id=?",
                               (run_id,)).fetchone()
    if row is None:
        return None
    items = tuple(
        RecommendationItem(
            segment_index=int(r["segment_index"]), material_id=int(r["material_id"]),
            rank=int(r["rank"]), score=float(r["score"] or 0.0),
            breakdown=_loads(r["score_breakdown_json"]) or {}, reason=str(r["reason"] or ""))
        for r in db.connect().execute(
            "SELECT * FROM dance_recommendation_items WHERE run_id=? "
            "ORDER BY segment_index, rank", (run_id,)))
    return RecommendationRun(
        id=int(row["id"]), target_song_id=int(row["target_song_id"]),
        montage_id=int(row["montage_id"] or 0), strategy_id=int(row["strategy_id"] or 0),
        strategy_kind=str(row["strategy_kind"]), strategy_version=str(row["strategy_version"] or ""),
        random_seed=int(row["random_seed"] or 0), candidate_count=int(row["candidate_count"] or 0),
        recommended_count=int(row["recommended_count"] or 0),
        algorithm_version=str(row["algorithm_version"] or ""),
        created_at=str(row["created_at"] or ""), items=items, notes=str(row["notes"] or ""))


def list_recommendation_runs(db: Database, target_song_id: int, *,
                             limit: int = 50) -> list[Any]:
    return list(db.connect().execute(
        "SELECT * FROM dance_recommendation_runs WHERE target_song_id=? "
        "ORDER BY id DESC LIMIT ?", (target_song_id, int(limit))))


def save_feedback(db: Database, *, run_id: int, verdict: str, item_id: int | None = None,
                  material_id: int | None = None, comment: str = "") -> int:
    """记一条推荐反馈。`verdict` ∈ accepted / rejected / replaced。"""
    with db.tx() as conn:
        cursor = conn.execute(
            "INSERT INTO dance_recommendation_feedback(run_id, item_id, material_id, "
            "verdict, comment, created_at) VALUES(?,?,?,?,?,?)",
            (run_id, item_id, material_id, verdict, comment, now()))
        return int(cursor.lastrowid)


def feedback_counts(db: Database, target_song_id: int) -> dict[int, dict[str, int]]:
    """`{material_id: {verdict: 次数}}`。history_based 推荐读它当"用户偏好"。"""
    rows = db.connect().execute(
        "SELECT f.material_id, f.verdict, COUNT(*) AS n "
        "FROM dance_recommendation_feedback f "
        "JOIN dance_recommendation_runs r ON r.id = f.run_id "
        "WHERE r.target_song_id = ? AND f.material_id IS NOT NULL "
        "GROUP BY f.material_id, f.verdict", (target_song_id,))
    out: dict[int, dict[str, int]] = {}
    for row in rows:
        out.setdefault(int(row["material_id"]), {})[str(row["verdict"])] = int(row["n"])
    return out


__all__ = [
    "now",
    "upsert_song", "save_song_analysis", "get_song", "get_song_by_fingerprint",
    "get_song_by_path", "list_songs", "song_sections", "song_beats",
    "DEFAULT_TEMPLATE_NAME", "save_segment_template", "get_segment_template",
    "active_segment_template", "list_segment_templates",
    "set_active_segment_template", "delete_segment_template",
    "SelectionError", "set_final_selection", "final_selections",
    "clear_final_selection", "add_cut_mark", "cut_marks", "remove_cut_mark",
    "save_candidate_order", "candidate_order", "order_materials",
    "save_alignment", "alignment_by_key", "get_alignment", "alignments_for_song",
    "set_manual_offset",
    "upsert_material", "get_material", "get_materials", "materials_at",
    "set_material_status", "mark_regenerated",
    "create_montage", "get_montage", "list_montages", "ensure_montage",
    "next_version_index", "save_version", "set_render_status", "get_version",
    "list_versions", "version_materials", "versions_for_song",
    "save_strategy", "get_strategy", "default_strategy", "list_strategies",
    "delete_strategy",
    "save_recommendation_run", "get_recommendation_run", "list_recommendation_runs",
    "save_feedback", "feedback_counts",
]






