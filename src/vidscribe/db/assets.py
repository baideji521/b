"""高光方案（highlight_assets）与 PRM 档案（prm_profiles）的读写。

这一层解决的是 Batch 8 之前的一个缺口：**AI 回的高光 JSON 只是"一次性结果"**，
新的一份进来就把旧的盖掉，事后既查不到、也没法拿旧 JSON 重剪。

现在的模型：

    视频 ──┬── 方案 A（gemini / gemini-2.5-flash）
           ├── 方案 B（gemini / gemini-2.5-flash）
           ├── 方案 C（qwen / Qwen3-VL-4B-Instruct）
           └── 方案 D（manual）

    方案 X ─┬── + PRM V1 → 成品 1
            ├── + PRM V2 → 成品 2
            └── + PRM V3 → 成品 3

几条硬规则（都写进了测试）：

  1. **新结果永远是新的一行**，绝不 UPDATE 旧方案的 JSON。
  2. `raw_json` 是当时那份原始输出，任何编辑都不碰它；编辑默认另开一条
     （`source_type='edited'` + `parent_id` 指回来），AI 原话永远追得到。
  3. 删除一律软删（`deleted_at`），已经剪出来的成品**不跟着删**，
     成品详情照旧能显示"原方案：方案 A（已删除）"。
  4. PRM 只登记名字/文件名/语言，**提示词内容仍然只存在文件里**；
     成品记 `prm_id`，所以以后文件改名也查得到当时用的是哪一版。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .db import Database
from .repo import now

logger = get_logger(__name__)

# 方案默认名：方案 A、方案 B……用完 26 个字母就退回数字（方案 27）
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ============================================================== 小工具
def _dumps(payload: Any) -> str:
    """JSON 一律存文本。已经是字符串就原样留着（不重新格式化 AI 原文）。"""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


def loads(text: Any) -> Any:
    """把库里的 JSON 文本读回对象；读不回来就返回 None（坏 JSON 不当好的用）。"""
    if isinstance(text, (dict, list)):
        return text
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def summarize(payload: Any) -> tuple[int, float | None]:
    """这份 JSON 里有几个可用片段、最高分多少。判定沿用剪辑引擎，不另写一套。"""
    from ..highlight import clip_engine  # noqa: PLC0415 - 纯函数模块，无重依赖

    data = loads(payload)
    clips = clip_engine.clips_in_payload(data)
    scores: list[float] = []
    for clip in clips:
        value = clip.get("score")
        if isinstance(value, bool) or value is None:
            continue
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return len(clips), (max(scores) if scores else None)


def _next_name(db: Database, video_id: int) -> str:
    """这个视频的下一个方案名。软删掉的也占名额，免得出现两个"方案 B"。"""
    used = int(db.value("SELECT COUNT(*) FROM highlight_assets WHERE video_id = ?",
                        (video_id,), default=0) or 0)
    if used < len(_LETTERS):
        return f"方案 {_LETTERS[used]}"
    return f"方案 {used + 1}"


# ============================================================== 高光方案
def create_asset(db: Database, video_id: int, payload: Any, *,
                 provider: str | None = None, model: str | None = None,
                 source_type: str = "ai", name: str | None = None,
                 analysis_id: int | None = None, source_task_id: int | None = None,
                 ai_result_id: int | None = None, prm_id: int | None = None,
                 parent_id: int | None = None, version: int = 1,
                 raw_payload: Any = None, note: str | None = None,
                 make_current: bool = True) -> int:
    """新登记一份高光方案，返回 id。**只 INSERT，从不覆盖已有方案。**

    `raw_payload` 不给就等于 `payload`（AI 直接回来的那种情况）；
    编辑/复制出来的方案要把原始那份传进来，`raw_json` 才能一直是 AI 原话。
    """
    current_text = _dumps(payload)
    raw_text = _dumps(raw_payload if raw_payload is not None else payload)
    count, best = summarize(current_text)
    stamp = now()
    title = name or _next_name(db, video_id)
    with db.tx() as conn:
        if make_current:      # 「当前方案」全视频唯一，先把旧的让出来
            conn.execute("UPDATE highlight_assets SET is_current = 0, updated_at = ? "
                         "WHERE video_id = ? AND is_current = 1", (stamp, video_id))
        cur = conn.execute(
            """
            INSERT INTO highlight_assets(video_id, analysis_id, source_task_id, ai_result_id,
                                         prm_id, parent_id, provider, model, source_type,
                                         name, version, raw_json, current_json, clip_count,
                                         best_score, is_current, note, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, analysis_id, source_task_id, ai_result_id, prm_id, parent_id,
             provider or None, model or None, source_type, title, int(version),
             raw_text, current_text, count, best, 1 if make_current else 0, note,
             stamp, stamp))
        return int(cur.lastrowid)


def get_asset(db: Database, asset_id: int) -> sqlite3.Row | None:
    """按 id 取方案。软删掉的照样返回——成品要靠它显示"（已删除）"。"""
    return db.one("SELECT * FROM highlight_assets WHERE id = ?", (asset_id,))


def list_assets(db: Database, video_id: int, *,
                include_deleted: bool = False) -> list[sqlite3.Row]:
    """一个视频的所有方案，按登记顺序。默认不含已删除的。"""
    sql = "SELECT * FROM highlight_assets WHERE video_id = ?"
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    return db.all(sql + " ORDER BY id", (video_id,))


def assets_by_ai(db: Database, *, provider: str | None = None, model: str | None = None,
                 include_deleted: bool = False) -> list[sqlite3.Row]:
    """按 AI 来源查方案（provider 必给、model 可选）。"""
    sql = "SELECT * FROM highlight_assets WHERE 1 = 1"
    params: list[Any] = []
    if provider:
        sql += " AND provider = ?"
        params.append(provider)
    if model:
        sql += " AND model = ?"
        params.append(model)
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    return db.all(sql + " ORDER BY id", tuple(params))


def assets_by_prm(db: Database, prm_id: int, *,
                  include_deleted: bool = False) -> list[sqlite3.Row]:
    """哪些方案是用这一版 PRM 换回来的。"""
    sql = "SELECT * FROM highlight_assets WHERE prm_id = ?"
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    return db.all(sql + " ORDER BY id", (prm_id,))


def asset_counts(db: Database, video_ids: list[int]) -> dict[int, int]:
    """每个视频有几份可用方案（数据管理表格那一列）。"""
    out = {int(v): 0 for v in video_ids}
    if not out:
        return out
    marks = ", ".join("?" for _ in out)
    rows = db.all(
        f"SELECT video_id, COUNT(*) AS n FROM highlight_assets "
        f"WHERE deleted_at IS NULL AND video_id IN ({marks}) GROUP BY video_id",
        tuple(out))
    for row in rows:
        out[int(row["video_id"])] = int(row["n"])
    return out


def videos_with_assets(db: Database, video_ids: list[int]) -> set[int]:
    """这批视频里，哪些已经有可用高光方案（自动剪辑筛"有 JSON / 没 JSON"就看它）。

    只认 `clip_count > 0`：抠不出片段的 JSON 剪不出东西，不能算"有方案"。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    rows = db.all(
        f"SELECT DISTINCT video_id FROM highlight_assets "
        f"WHERE deleted_at IS NULL AND clip_count > 0 AND video_id IN ({marks})",
        tuple(int(v) for v in video_ids))
    return {int(r["video_id"]) for r in rows}


def current_asset(db: Database, video_id: int) -> sqlite3.Row | None:
    """这个视频的「当前方案」；没有标记过就退回最近登记的那一份。"""
    row = db.one("SELECT * FROM highlight_assets WHERE video_id = ? AND is_current = 1 "
                 "AND deleted_at IS NULL", (video_id,))
    if row is not None:
        return row
    return db.one("SELECT * FROM highlight_assets WHERE video_id = ? AND deleted_at IS NULL "
                  "ORDER BY id DESC LIMIT 1", (video_id,))


def set_current_asset(db: Database, asset_id: int) -> bool:
    """把某个方案设为当前。已删除的不许设。"""
    row = get_asset(db, asset_id)
    if row is None or row["deleted_at"]:
        return False
    stamp = now()
    with db.tx() as conn:
        conn.execute("UPDATE highlight_assets SET is_current = 0, updated_at = ? "
                     "WHERE video_id = ? AND is_current = 1", (stamp, int(row["video_id"])))
        conn.execute("UPDATE highlight_assets SET is_current = 1, updated_at = ? WHERE id = ?",
                     (stamp, asset_id))
    return True


def copy_asset(db: Database, asset_id: int, *, name: str | None = None,
               make_current: bool = False) -> int | None:
    """复制一份方案（副本可以随便改，原件不动）。"""
    row = get_asset(db, asset_id)
    if row is None:
        return None
    return create_asset(
        db, int(row["video_id"]), row["current_json"],
        provider=row["provider"], model=row["model"], source_type="copied",
        name=name or f"{row['name']} 副本", analysis_id=row["analysis_id"],
        source_task_id=row["source_task_id"], ai_result_id=row["ai_result_id"],
        prm_id=row["prm_id"], parent_id=asset_id, version=1,
        raw_payload=row["raw_json"], note=f"复制自 {row['name']}",
        make_current=make_current)


def edit_asset(db: Database, asset_id: int, payload: Any, *,
               in_place: bool = False, name: str | None = None,
               make_current: bool = True) -> int | None:
    """编辑方案。默认**另开一条**（原方案一个字都不动），返回新方案 id。

    `in_place=True` 才会改原行的 `current_json`——即便如此 `raw_json` 也不动，
    所以"AI 当时给了什么"永远查得到。
    """
    row = get_asset(db, asset_id)
    if row is None or row["deleted_at"]:
        return None
    if in_place:
        count, best = summarize(payload)
        with db.tx() as conn:
            conn.execute("UPDATE highlight_assets SET current_json = ?, clip_count = ?, "
                         "best_score = ?, updated_at = ? WHERE id = ?",
                         (_dumps(payload), count, best, now(), asset_id))
        return asset_id
    return create_asset(
        db, int(row["video_id"]), payload,
        provider=row["provider"], model=row["model"], source_type="edited",
        name=name or f"{row['name']}-编辑{int(row['version']) + 1}",
        analysis_id=row["analysis_id"], source_task_id=row["source_task_id"],
        ai_result_id=row["ai_result_id"], prm_id=row["prm_id"], parent_id=asset_id,
        version=int(row["version"]) + 1, raw_payload=row["raw_json"],
        note=f"在 {row['name']} 上编辑", make_current=make_current)


def delete_asset(db: Database, asset_id: int) -> bool:
    """软删方案。**成品一个都不动**：artifacts.highlight_asset_id 照旧指着它。"""
    row = get_asset(db, asset_id)
    if row is None or row["deleted_at"]:
        return False
    stamp = now()
    with db.tx() as conn:
        conn.execute("UPDATE highlight_assets SET deleted_at = ?, is_current = 0, "
                     "updated_at = ? WHERE id = ?", (stamp, stamp, asset_id))
    return True


def restore_asset(db: Database, asset_id: int) -> bool:
    """把软删掉的方案捞回来。"""
    row = get_asset(db, asset_id)
    if row is None or not row["deleted_at"]:
        return False
    with db.tx() as conn:
        conn.execute("UPDATE highlight_assets SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                     (now(), asset_id))
    return True


def asset_payload(db: Database, asset_id: int) -> Any:
    """拿这个方案当前的 JSON 对象（交给剪辑引擎的就是它）。"""
    row = get_asset(db, asset_id)
    if row is None:
        return None
    return loads(row["current_json"])


# ============================================================== PRM 档案
def create_prm(db: Database, name: str, filename: str | Path, *,
               description: str | None = None, language: str | None = None,
               version: str | None = None, make_default: bool = False) -> int:
    """登记一份 PRM。只记名字/文件名/语言，**内容仍旧只在文件里**。"""
    stamp = now()
    with db.tx() as conn:
        if make_default:
            conn.execute("UPDATE prm_profiles SET is_default = 0, updated_at = ? "
                         "WHERE is_default = 1", (stamp,))
        cur = conn.execute(
            """
            INSERT INTO prm_profiles(name, filename, description, language, version,
                                     is_default, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, str(filename), description, language, version,
             1 if make_default else 0, stamp, stamp))
        return int(cur.lastrowid)


def get_prm(db: Database, prm_id: int) -> sqlite3.Row | None:
    """按 id 取 PRM。软删掉的照样返回，成品要靠它显示"（已删除）"。"""
    return db.one("SELECT * FROM prm_profiles WHERE id = ?", (prm_id,))


def list_prms(db: Database, *, include_deleted: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM prm_profiles"
    if not include_deleted:
        sql += " WHERE deleted_at IS NULL"
    return db.all(sql + " ORDER BY id", ())


def find_prm_by_file(db: Database, filename: str | Path) -> sqlite3.Row | None:
    """按文件路径找 PRM（同一份文件不重复登记）。"""
    target = str(filename)
    row = db.one("SELECT * FROM prm_profiles WHERE filename = ? AND deleted_at IS NULL",
                 (target,))
    if row is not None:
        return row
    name = Path(target).name
    return db.one("SELECT * FROM prm_profiles WHERE filename LIKE ? AND deleted_at IS NULL "
                  "ORDER BY id LIMIT 1", (f"%{name}",))


def update_prm(db: Database, prm_id: int, *, name: str | None = None,
               filename: str | Path | None = None, description: str | None = None,
               language: str | None = None, version: str | None = None) -> bool:
    sets, params = [], []
    for column, value in (("name", name), ("filename", str(filename) if filename else None),
                          ("description", description), ("language", language),
                          ("version", version)):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(value)
    if not sets:
        return False
    sets.append("updated_at = ?")
    params += [now(), prm_id]
    with db.tx() as conn:
        cur = conn.execute(f"UPDATE prm_profiles SET {', '.join(sets)} WHERE id = ?", params)
        return bool(cur.rowcount)


def delete_prm(db: Database, prm_id: int) -> bool:
    """软删 PRM。历史成品的 prm_id 不动，照旧查得到当时用的哪一版。"""
    row = get_prm(db, prm_id)
    if row is None or row["deleted_at"]:
        return False
    stamp = now()
    with db.tx() as conn:
        conn.execute("UPDATE prm_profiles SET deleted_at = ?, is_default = 0, updated_at = ? "
                     "WHERE id = ?", (stamp, stamp, prm_id))
    return True


def set_default_prm(db: Database, prm_id: int) -> bool:
    row = get_prm(db, prm_id)
    if row is None or row["deleted_at"]:
        return False
    stamp = now()
    with db.tx() as conn:
        conn.execute("UPDATE prm_profiles SET is_default = 0, updated_at = ? WHERE is_default = 1",
                     (stamp,))
        conn.execute("UPDATE prm_profiles SET is_default = 1, updated_at = ? WHERE id = ?",
                     (stamp, prm_id))
    return True


def default_prm(db: Database) -> sqlite3.Row | None:
    """当前默认 PRM；没设过就用最早登记的那一份。"""
    row = db.one("SELECT * FROM prm_profiles WHERE is_default = 1 AND deleted_at IS NULL", ())
    if row is not None:
        return row
    return db.one("SELECT * FROM prm_profiles WHERE deleted_at IS NULL ORDER BY id LIMIT 1", ())


def prm_file(row: sqlite3.Row | None, root: str | Path | None = None) -> Path | None:
    """PRM 行 -> 实际文件路径（相对路径按项目根拼）。"""
    if row is None:
        return None
    path = Path(str(row["filename"]))
    if not path.is_absolute() and root:
        path = Path(root) / path
    return path


def ensure_prm(db: Database, filename: str | Path, *, name: str | None = None,
               language: str | None = None, make_default: bool = False) -> int:
    """确保这份提示词文件在库里有登记，返回 prm_id。已经有了就直接用。"""
    found = find_prm_by_file(db, filename)
    if found is not None:
        if make_default and not int(found["is_default"] or 0):
            set_default_prm(db, int(found["id"]))
        return int(found["id"])
    stem = Path(str(filename)).stem
    return create_prm(db, name or stem, filename, language=language,
                      make_default=make_default)


# ============================================================== 成品溯源
def link_artifact(db: Database, artifact_id: int, *, asset_id: int | None = None,
                  prm_id: int | None = None) -> bool:
    """给成品补上"哪个方案 + 哪个 PRM"。只写给了值的那一列。"""
    sets, params = [], []
    if asset_id is not None:
        sets.append("highlight_asset_id = ?")
        params.append(int(asset_id))
    if prm_id is not None:
        sets.append("prm_id = ?")
        params.append(int(prm_id))
    if not sets:
        return False
    sets.append("updated_at = ?")
    params += [now(), artifact_id]
    with db.tx() as conn:
        cur = conn.execute(f"UPDATE artifacts SET {', '.join(sets)} WHERE id = ?", params)
        return bool(cur.rowcount)


def products_for_asset(db: Database, asset_id: int) -> list[sqlite3.Row]:
    """这个方案剪出过哪些成品（一个 JSON 配不同 PRM 就会有多条）。"""
    return db.all("SELECT * FROM artifacts WHERE highlight_asset_id = ? ORDER BY id",
                  (asset_id,))


def products_for_prm(db: Database, prm_id: int) -> list[sqlite3.Row]:
    """这一版 PRM 剪出过哪些成品。"""
    return db.all("SELECT * FROM artifacts WHERE prm_id = ? ORDER BY id", (prm_id,))


def product_counts(db: Database, video_ids: list[int], kind: str = "final_video") -> dict[int, int]:
    """每个视频有几个成品（数据管理表格那一列）。"""
    out = {int(v): 0 for v in video_ids}
    if not out:
        return out
    marks = ", ".join("?" for _ in out)
    rows = db.all(
        f"SELECT video_id, COUNT(*) AS n FROM artifacts "
        f"WHERE type = ? AND video_id IN ({marks}) GROUP BY video_id",
        (kind, *out))
    for row in rows:
        out[int(row["video_id"])] = int(row["n"])
    return out


def artifact_lineage(db: Database, artifact_id: int) -> dict[str, Any] | None:
    """成品反查：视频 / analysis / 方案 / AI / 模型 / PRM / 任务 / 时间。

    方案或 PRM 被软删了也照样查得到，只是多一个 `asset_deleted` / `prm_deleted` 标记。
    """
    art = db.one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
    if art is None:
        return None
    out: dict[str, Any] = {
        "artifact_id": int(art["id"]),
        "path": str(art["path"]),
        "type": str(art["type"]),
        "size": art["size"],
        "exists_on_disk": bool(art["exists_on_disk"]),
        "created_at": art["created_at"],
        "video": None, "analysis_id": None,
        "asset": None, "asset_deleted": False,
        "provider": None, "model": None,
        "task_id": None, "prm": None, "prm_deleted": False,
    }
    video = db.one("SELECT * FROM videos WHERE id = ?", (int(art["video_id"]),))
    if video is not None:
        out["video"] = {"id": int(video["id"]), "file_name": video["file_name"],
                        "file_path": video["file_path"], "duration": video["duration"]}
    asset_id = art["highlight_asset_id"]
    if asset_id is not None:
        asset = get_asset(db, int(asset_id))
        if asset is not None:
            out["asset"] = {"id": int(asset["id"]), "name": asset["name"],
                            "source_type": asset["source_type"],
                            "clip_count": int(asset["clip_count"] or 0),
                            "best_score": asset["best_score"],
                            "created_at": asset["created_at"]}
            out["asset_deleted"] = bool(asset["deleted_at"])
            out["analysis_id"] = asset["analysis_id"]
            out["provider"] = asset["provider"]
            out["model"] = asset["model"]
            out["task_id"] = asset["source_task_id"]
    prm_id = art["prm_id"]
    if prm_id is not None:
        prm = get_prm(db, int(prm_id))
        if prm is not None:
            out["prm"] = {"id": int(prm["id"]), "name": prm["name"],
                          "filename": prm["filename"], "version": prm["version"],
                          "language": prm["language"]}
            out["prm_deleted"] = bool(prm["deleted_at"])
    return out


def video_overview(db: Database, video_id: int) -> dict[str, Any]:
    """一个视频的全景：分析、方案清单、成品清单（每条带溯源）。"""
    from . import repo  # noqa: PLC0415 - 只是复用现成的读函数

    video = db.one("SELECT * FROM videos WHERE id = ?", (video_id,))
    analysis = repo.latest_analysis(db, video_id)
    words = 0
    if analysis is not None:
        words = int(db.value("SELECT COUNT(*) FROM speech_words WHERE analysis_id = ?",
                             (int(analysis["id"]),), default=0) or 0)
    assets = list_assets(db, video_id)
    products = [artifact_lineage(db, int(r["id"]))
                for r in db.all("SELECT id FROM artifacts WHERE video_id = ? AND type = ? "
                                "ORDER BY id", (video_id, "final_video"))]
    return {
        "video": video,
        "analysis": analysis,
        "word_count": words,
        "assets": assets,
        "products": [p for p in products if p],
    }
