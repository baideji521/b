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
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .. import ai_protocol
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
    """这份 JSON 里有几个可用片段、最高分多少。协议解析只有 ai_protocol 一处真源。"""
    from .. import ai_protocol  # noqa: PLC0415 - 纯解析模块，无重依赖

    data = loads(payload)
    clips = ai_protocol.clips(data)
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


def _taken_names(db: Database, video_id: int) -> set[str]:
    """这个视频已经占掉的方案名。软删掉的也算占着，免得出现两个"方案 B"。"""
    return {str(row["name"] or "") for row in
            db.all("SELECT name FROM highlight_assets WHERE video_id = ?", (video_id,))}


def _next_name(db: Database, video_id: int) -> str:
    """这个视频的下一个方案名：方案 A、方案 B、方案 C……用完 26 个字母退回数字。

    同一个视频可以有任意多份方案，名字自动往后排、而且**保证不重名**：先按已有份数
    猜一个（正常情况一猜就中），要是这个名字已经被人占了（比如调用方之前显式传过
    「方案 B」），就继续往后挪一格，直到空出来为止。软删掉的也占名额。
    一次 AI 回复拆出好几份 JSON 连续入库，也不会冒出两个一模一样的「方案 B」。
    """
    taken = _taken_names(db, video_id)
    step = len(taken)
    limit = step + len(_LETTERS) + 1000        # 兜底上限，撞不了这么多次，绝不死循环
    while step < limit:
        candidate = f"方案 {_LETTERS[step]}" if step < len(_LETTERS) else f"方案 {step + 1}"
        if candidate not in taken:
            return candidate
        step += 1
    # 真挪到上限了（几乎不可能）：退回带后缀那套，反正也不能让入库卡住
    return _unique_name(db, video_id, f"方案 {len(taken) + 1}")


def _unique_name(db: Database, video_id: int, wanted: str) -> str:
    """把调用方想要的名字，变成这个视频下没人用过的名字。

    重名不许并存——资产中心里摆着两个一模一样的「方案 B」，用户根本分不清哪个是哪个。
    所以撞上了就在后面挂个区分后缀：`方案 B (2)`、`方案 B (3)`……软删的照样算占用。
    最多试到 999，够用了还不会死循环；真撞满了就用原名（宁可重名也不能把入库卡死）。
    """
    base = str(wanted or "").strip() or "方案"
    taken = _taken_names(db, video_id)
    if base not in taken:
        return base
    for index in range(2, 1000):
        candidate = f"{base} ({index})"
        if candidate not in taken:
            return candidate
    return base


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

    名字这块：同一个视频可以有任意多份方案，不给 `name` 就自动往后排（方案 A、
    方案 B、方案 C……），**排出来的名字保证不和已有方案重名**；显式给了 `name`
    又正好撞上已有的，就加个区分后缀（`方案 B (2)`）——一次 AI 回复拆出好几份 JSON
    连续入库也不会出现两个一模一样的名字，而且谁都不会被覆盖掉。
    """
    current_text = _dumps(payload)
    raw_text = _dumps(raw_payload if raw_payload is not None else payload)
    count, best = summarize(current_text)
    stamp = now()
    # 显式给的名字也要过一遍防重：重名方案并存的话，用户在资产中心分不清哪个是哪个
    title = _unique_name(db, video_id, name) if name else _next_name(db, video_id)
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


def assets_without_product(db: Database, video_id: int,
                           kind: str = "final_video") -> list[sqlite3.Row]:
    """这个视频还**没出成品**的高光 JSON，按登记顺序排（旧的先剪）。

    自动剪辑按「一份 JSON 一个成品」跑，靠这个函数决定还剩哪几份要剪：

      * 软删的不算（`deleted_at`）、抠不出片段的不算（`clip_count = 0`）；
      * 「已经出过成品」= `artifacts` 里有挂着这份 asset 且**还在盘上**的成品，
        所以成品文件被删掉之后，这份 JSON 会重新排到待剪里。
    """
    return db.all(
        """
        SELECT h.* FROM highlight_assets h
         WHERE h.video_id = ? AND h.deleted_at IS NULL AND COALESCE(h.clip_count, 0) > 0
           AND NOT EXISTS (SELECT 1 FROM artifacts f
                            WHERE f.highlight_asset_id = h.id AND f.type = ?
                              AND f.exists_on_disk = 1)
         ORDER BY h.id
        """, (video_id, kind))



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


def fork_with_offsets(db: Database, asset_id: int, startframe: float,
                      freeze: float, *,
                      freezeframe: float | None = None) -> tuple[int | None, bool]:
    """把这一次剪辑用的参数盖进方案，返回 (要用的方案 id, 是不是新派生的)。

    盖的是「剪辑高光」窗里的当前配置：`startframe` / `freeze` / `freezeframe`，
    有就替换、没有就新增。规则只有一条：**已经有成品挂着的方案不许被改**，改了它那些成品
    的溯源就指向一份「已经不是当时那份」的 JSON。所以

      - 这份方案还没出过成品 → 原地盖章（`in_place`）。一个成品对应一个 JSON，
        刚从 AI 拿到就开剪的那种情况不该凭空多出一版没有成品的 JSON。
      - 已经有成品挂着 → 派生新的一版（parent_id 指原方案、`raw_json` 仍是 AI 原话、
        成为当前方案）。老成品继续指着老那版，新成品指新的这版，各自查得清。
      - 三个参数和库里一模一样 → 什么都不做，沿用原方案。
    """
    from .. import ai_protocol  # noqa: PLC0415 - 轻量解析模块

    row = get_asset(db, asset_id)
    if row is None or row["deleted_at"]:
        return None, False
    payload = loads(row["current_json"])
    stamped = ai_protocol.stamp_offsets(payload, startframe, freeze,
                                        freezeframe=freezeframe)
    if stamped is None:
        return int(row["id"]), False
    if isinstance(payload, dict) and stamped == payload:
        return int(row["id"]), False
    if not products_for_asset(db, asset_id):
        # 没有成品挂着：原地盖，血缘没有任何东西会被这一改弄错
        edit_asset(db, asset_id, stamped, in_place=True)
        return int(row["id"]), False
    name = (f"{row['name']}-偏移{round(float(startframe), 2):+g}"
            f"/{round(float(freeze), 2):+g}")
    new_id = edit_asset(db, asset_id, stamped, name=name)
    if new_id is None:
        return int(row["id"]), False
    return int(new_id), True


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


def purge_assets_for_video(db: Database, video_id: int) -> int:
    """把这个视频的高光 JSON 记录**真删掉**（硬删），返回删了几条。软删的一并清。

    和 `delete_asset`（软删）是两回事：软删只是打个 `deleted_at`，成品详情里还能显示
    「原方案：方案 A（已删除）」；这里是资产中心「批量删 JSON」用的——JSON 攒太多了
    要整个清掉重来，所以连行都不留。

    成品文件一个都不动，只把指着它们的 `artifacts.highlight_asset_id` 先手工置空
    （不靠 `ON DELETE SET NULL`：`PRAGMA foreign_keys` 不一定开着，留下悬空 id
    会让成品的血缘查出一个不存在的方案）。清完成品就是「溯源不到 JSON」的状态。
    """
    ids = [int(row["id"]) for row in
           db.all("SELECT id FROM highlight_assets WHERE video_id = ?", (video_id,))]
    if not ids:
        return 0
    marks = ", ".join("?" for _ in ids)
    with db.tx() as conn:
        conn.execute(f"UPDATE artifacts SET highlight_asset_id = NULL, updated_at = ? "
                     f"WHERE highlight_asset_id IN ({marks})", (now(), *ids))
        conn.execute(f"DELETE FROM highlight_assets WHERE id IN ({marks})", tuple(ids))
    logger.info("硬删高光 JSON：视频 #%s 清掉 %d 份（成品文件没动）", video_id, len(ids))
    return len(ids)



def asset_payload(db: Database, asset_id: int) -> Any:
    """拿这个方案当前的 JSON 对象（交给剪辑引擎的就是它）。"""
    row = get_asset(db, asset_id)
    if row is None:
        return None
    return loads(row["current_json"])


# ============================================================== PRM 档案
def create_prm(db: Database, name: str, filename: str | Path, *,
               description: str | None = None, language: str | None = None,
               version: str | None = None, make_default: bool = False,
               enabled: bool = True, content: str | None = None) -> int:
    """登记一份 PRM。**提示词正文存在库里**（content），filename 只记来源文件。

    新登记的默认是「使用中」：发 AI 时会跟着一起发，不想发就在 PRM 管理页停用。
    `content` 不给就先留空，第一次要用的时候 `prm_text` 会按 filename 把文件读进来补上。
    """
    stamp = now()
    with db.tx() as conn:
        if make_default:
            conn.execute("UPDATE prm_profiles SET is_default = 0, updated_at = ? "
                         "WHERE is_default = 1", (stamp,))
        cur = conn.execute(
            """
            INSERT INTO prm_profiles(name, filename, description, language, version,
                                     content, is_default, enabled, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, str(filename), description, language, version, content,
             1 if make_default else 0, 1 if enabled else 0, stamp, stamp))
        return int(cur.lastrowid)


def get_prm(db: Database, prm_id: int) -> sqlite3.Row | None:
    """按 id 取 PRM。软删掉的照样返回，成品要靠它显示"（已删除）"。"""
    return db.one("SELECT * FROM prm_profiles WHERE id = ?", (prm_id,))


def list_prms(db: Database, *, include_deleted: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM prm_profiles"
    if not include_deleted:
        sql += " WHERE deleted_at IS NULL"
    return db.all(sql + " ORDER BY id", ())


def enabled_prms(db: Database) -> list[sqlite3.Row]:
    """当前「使用中」的 PRM（按 id 从小到大）。

    发 AI 时**这里面每一份都会当附件带上**：两份都在用就发两份，
    停用的一份都不发，一份都没启用就整条不发（由界面那边决定怎么记）。
    """
    return db.all("SELECT * FROM prm_profiles WHERE deleted_at IS NULL AND enabled = 1 "
                  "ORDER BY id", ())


def set_prm_enabled(db: Database, prm_id: int, enabled: bool) -> bool:
    """把一份 PRM 标成使用中 / 停用。软删掉的不给改（先恢复再说）。"""
    with db.tx() as conn:
        cur = conn.execute("UPDATE prm_profiles SET enabled = ?, updated_at = ? "
                           "WHERE id = ? AND deleted_at IS NULL",
                           (1 if enabled else 0, now(), prm_id))
        return cur.rowcount > 0



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


def prm_name_taken(db: Database, name: str, *, except_id: int | None = None) -> bool:
    """这个名字有没有被别的在库档案占了（名字在库里唯一：idx_prm_name_live）。

    界面改名前先问一句，免得撞上唯一索引直接抛异常、看着像「改名不生效」。
    """
    title = str(name).strip()
    if not title:
        return False
    if except_id is None:
        row = db.one("SELECT id FROM prm_profiles WHERE name = ? AND deleted_at IS NULL",
                     (title,))
    else:
        row = db.one("SELECT id FROM prm_profiles WHERE name = ? AND deleted_at IS NULL "
                     "AND id <> ?", (title, int(except_id)))
    return row is not None


def update_prm(db: Database, prm_id: int, *, name: str | None = None,
               filename: str | Path | None = None, description: str | None = None,
               language: str | None = None, version: str | None = None,
               content: str | None = None) -> bool:
    sets, params = [], []
    for column, value in (("name", name), ("filename", str(filename) if filename else None),
                          ("description", description), ("language", language),
                          ("version", version), ("content", content)):
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


def restore_prm(db: Database, prm_id: int) -> bool:
    """把软删的 PRM 捞回来（不自动恢复默认标记）。"""
    row = get_prm(db, prm_id)
    if row is None or not row["deleted_at"]:
        return False
    with db.tx() as conn:
        conn.execute("UPDATE prm_profiles SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                     (now(), prm_id))
    return True


def copy_prm(db: Database, prm_id: int, *, name: str | None = None,
             filename: str | Path | None = None) -> int | None:
    """复制一份 PRM 档案（正文一起复制过去）。原档案一个字不动。

    复制出来的那份是独立的：改它的正文不会动到原件。
    """
    row = get_prm(db, prm_id)
    if row is None:
        return None
    base = str(row["name"])
    title = name or f"{base} 副本"
    if db.one("SELECT id FROM prm_profiles WHERE name = ? AND deleted_at IS NULL",
              (title,)) is not None:
        title = f"{title} {now()[11:19]}"      # 撞名了就加个时间尾巴，避免唯一索引报错
    return create_prm(db, title, filename or row["filename"],
                      description=row["description"], language=row["language"],
                      version=row["version"], content=row["content"])


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
    """PRM 行 -> 当初导入它的那个文件路径（相对路径按项目根拼）。

    **只用于"从哪儿导进来的"这类溯源和一次性导入**：正文的权威来源是库里的 content。
    """
    if row is None:
        return None
    path = Path(str(row["filename"]))
    if not path.is_absolute() and root:
        path = Path(root) / path
    return path


def prm_text(db: Database, prm_id: int, root: str | Path | None = None) -> str | None:
    """这份 PRM 的提示词正文。库里没有（老库 / 刚登记）就按 filename 读文件补进库。

    自愈导入只发生一次：读到内容立刻写回 content，之后文件删了也照样发得出去。
    正文真的空（文件也不在）返回 None，调用方据此判断"这一份发不了"。
    """
    row = get_prm(db, prm_id)
    if row is None:
        return None
    text = row["content"]
    if text:
        return str(text)
    path = prm_file(row, root)
    if path is None or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    update_prm(db, prm_id, content=text)      # 一次性导入：以后就以库为准
    return text


def ensure_prm(db: Database, filename: str | Path, *, name: str | None = None,
               language: str | None = None, make_default: bool = False,
               root: str | Path | None = None) -> int:
    """确保这份提示词在库里有登记，返回 prm_id。已经有了就直接用。

    新登记时顺手把文件正文读进库（读不到就先留空，`prm_text` 以后还会再试一次）。
    """
    found = find_prm_by_file(db, filename)
    if found is not None:
        if make_default and not int(found["is_default"] or 0):
            set_default_prm(db, int(found["id"]))
        return int(found["id"])
    path = Path(str(filename))
    if not path.is_absolute() and root:
        path = Path(root) / path
    content = None
    if path.is_file():
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = None
    return create_prm(db, name or path.stem, filename, language=language,
                      make_default=make_default, content=content)


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


def clip_spec_for(plan: Any, start: float, end: float) -> dict[str, Any]:
    """把一条 ClipPlan + **实际剪出来的**区间，变成 clips 表要的一行。

    文案字段（score / type / reason / evaluation）走既有的 `repo.clips_from_payload`
    解析 AI 原始片段，一个字不改写；时间用实际值（引擎修正 + 加减秒之后的）。
    AI 原始区间不写进 clips——它一直在 `highlight_assets.raw_json` 里，那份永不改。
    """
    from .. import ai_protocol  # noqa: PLC0415 - 轻量，放函数里保持 import 顺序干净
    from . import repo  # noqa: PLC0415 - 同包，放函数里避免循环导入

    raw = dict(getattr(plan, "raw", None) or {})
    parsed: list[dict[str, Any]] = []
    if raw and raw.get("start") is not None and raw.get("end") is not None:
        # plan.raw 已经是规范化后的片段，回写成协议形状再解一遍，口径与入库一致
        parsed = repo.clips_from_payload(
            ai_protocol.build_payload(raw, start=float(raw["start"]),
                                      end=float(raw["end"]),
                                      duration=ai_protocol.num(raw.get("duration"))))
    spec: dict[str, Any] = dict(parsed[0]) if parsed else {}
    spec.update({"start": round(float(start), 3), "end": round(float(end), 3),
                 "duration": round(float(end) - float(start), 3)})
    for key, value in (("score", getattr(plan, "score", None)),
                       ("type", getattr(plan, "type", "") or ""),
                       ("reason", getattr(plan, "reason", "") or "")):
        spec.setdefault(key, value)
    return spec


def record_clips(db: Database, video_id: int, output_path: str | Path,
                 specs: list[dict[str, Any]] | None = None, *,
                 ai_result_id: int | None = None) -> list[int]:
    """把**实际剪出来的**片段写进 clips，一段一行，直接就是 rendered。

    只有渲染成功（`os.replace` 落地 + `video_io.is_complete_video()` 通过）之后
    才允许调用——这里不做"算不算成品"的判断，那是渲染那一层的闸门。
    """
    from . import repo  # noqa: PLC0415

    target = Path(output_path)
    return [repo.create_clip(db, video_id, spec, ai_result_id=ai_result_id,
                             status="rendered", output_path=target)
            for spec in (specs or ())]


def record_product(db: Database, video_id: int, output_path: str | Path, *,
                   specs: list[dict[str, Any]] | None = None,
                   asset_id: int | None = None, prm_id: int | None = None,
                   ai_result_id: int | None = None,
                   kind: str = "final_video") -> dict[str, Any]:
    """成品记账：写 clips → 登记 artifact → 挂方案 / PRM。

    调用方的义务：只有在渲染成功并通过完整性检查之后才允许调这里。
    CLI（`assets --render`）走这一个函数；GUI 的成品登记走
    `_register_artifact` + `_link_final_video` + `record_clips`，
    写 clips 的规则两边共用同一段代码（`record_clips`）。
    """
    from . import repo  # noqa: PLC0415

    target = Path(output_path)
    clip_ids = record_clips(db, video_id, target, specs, ai_result_id=ai_result_id)
    artifact_id = repo.register_artifact(db, video_id, kind, target)
    if asset_id is not None or prm_id is not None:
        link_artifact(db, artifact_id, asset_id=asset_id, prm_id=prm_id)
    return {"artifact_id": artifact_id, "clip_ids": clip_ids}


def clips_for_product(db: Database, video_id: int, path: str | Path) -> list[sqlite3.Row]:
    """这个成品文件对应的 clips 行（实际剪辑区间的来源）。"""
    from . import repo  # noqa: PLC0415

    want = str(path)
    return [row for row in repo.get_clips(db, video_id)
            if str(row["output_path"] or "") == want]


def asset_layers(db: Database, asset_id: int, *, artifact_id: int | None = None,
                 row: sqlite3.Row | None = None) -> dict[str, list[dict[str, Any]]]:
    """三层区间**一次算完**：AI 原始 / Clip Engine 修正（含原因）/ 实际渲染。

    这是 `asset_spans()` + `lineage_spans()` 的合并版：同一次刷新里视频时长、
    分析批次、逐词时间戳只查一遍（以前两个函数各查一遍，界面一次刷新要 20 多条 SQL）。
    规则来源仍然只有 `highlight/clip_engine.plan_clips` 一个，算法一个字没改。

    `row` 可以把已经查出来的 highlight_assets 行传进来（界面刷新时手上就有），省一次查询。
    `artifact_id` 给了就带上这个成品的实际渲染区间（来自 `clips`）。
    """
    from . import repo  # noqa: PLC0415

    out: dict[str, list[dict[str, Any]]] = {"ai": [], "engine": [], "actual": []}
    if row is None or int(row["id"]) != int(asset_id):
        row = get_asset(db, asset_id)
    if row is None:
        return out
    payload = loads(row["current_json"])
    if not payload:
        return out
    out["ai"] = [{"start": clip.get("start"), "end": clip.get("end"),
                  "score": clip.get("score"), "type": clip.get("type") or "",
                  "evaluation": clip.get("evaluation") or ""}
                 for clip in repo.clips_from_payload(payload)]
    video_id = int(row["video_id"])
    try:
        from ..highlight import clip_engine  # noqa: PLC0415 - 重依赖，用到才导

        duration = db.value("SELECT duration FROM videos WHERE id = ?", (video_id,), None)
        segments = clip_engine.segments_for_video(db, video_id)
        for plan in clip_engine.plan_clips(payload, segments,
                                           video_duration=float(duration) if duration
                                           else None).plans:
            out["engine"].append({"start": plan.start, "end": plan.end,
                                  "duration": plan.duration, "notes": list(plan.notes),
                                  "ai_start": plan.ai_start, "ai_end": plan.ai_end})
    except Exception as exc:  # noqa: BLE001 - 算不出来不影响看 AI 原始那一层
        logger.warning("复算 Clip Engine 区间失败：%s", exc)
    if artifact_id is not None:
        art = db.one("SELECT video_id, path FROM artifacts WHERE id = ?", (int(artifact_id),))
        if art is not None:
            out["actual"] = [{"start": clip["start_time"], "end": clip["end_time"],
                              "duration": clip["duration"]}
                             for clip in clips_for_product(db, int(art["video_id"]),
                                                           str(art["path"]))]
    return out


def asset_spans(db: Database, asset_id: int) -> dict[str, list[dict[str, Any]]]:
    """一份高光 JSON 的两层区间：AI 原始 + Clip Engine 修正（含原因）。

    Engine 那一层是现算的，规则来源只有 `highlight/clip_engine.plan_clips` 一个
    （纯函数，同输入同输出），所以"现算"就是真剪的时候会用的边界。
    界面拿它做「剪之前先看看会剪成什么样」，不另写一套算法。
    """
    layers = asset_layers(db, asset_id)
    return {"ai": layers["ai"], "engine": layers["engine"]}


def lineage_spans(db: Database, artifact_id: int) -> dict[str, list[dict[str, Any]]]:
    """一个成品的三层区间：AI 原始 / Clip Engine 修正（含原因）/ 实际渲染。

      * AI 原始 + Engine：来自 `asset_layers()`（同一个规则源）；
      * 实际渲染：来自 `clips`（渲染成功、完整性检查通过之后写进去的）。

    GUI 的血缘面板和 CLI 的 `assets --trace` 都拿这一个函数的结果显示，不各写一套。
    """
    out: dict[str, list[dict[str, Any]]] = {"ai": [], "engine": [], "actual": []}
    art = db.one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
    if art is None:
        return out
    video_id = int(art["video_id"])
    out["actual"] = [{"start": row["start_time"], "end": row["end_time"],
                      "duration": row["duration"]}
                     for row in clips_for_product(db, video_id, str(art["path"]))]
    asset_id = art["highlight_asset_id"]
    if asset_id is None:
        return out
    spans = asset_layers(db, int(asset_id))
    out["ai"] = spans["ai"]
    out["engine"] = spans["engine"]
    return out



def products_for_asset(db: Database, asset_id: int) -> list[sqlite3.Row]:
    """这个方案剪出过哪些成品（一个 JSON 配不同 PRM 就会有多条）。"""
    return db.all("SELECT * FROM artifacts WHERE highlight_asset_id = ? ORDER BY id",
                  (asset_id,))


def products_for_prm(db: Database, prm_id: int) -> list[sqlite3.Row]:
    """这一版 PRM 剪出过哪些成品。"""
    return db.all("SELECT * FROM artifacts WHERE prm_id = ? ORDER BY id", (prm_id,))


def sync_product_presence(db: Database, kind: str = "final_video") -> int:
    """成品对账：库里记的成品文件还在不在盘上，把 `exists_on_disk` 拨正，返回改了几条。

    成品是**用户会在资源管理器里直接删的东西**（清盘、挪走、重剪前手动清理），
    而所有"有几个成品"的计数都只查库。不对账的话删完文件列表照旧显示有成品，
    筛「无成品」也永远筛不出它来。

    只看 `type = 'final_video'` 这一类：成品行数远少于全部 artifacts（每个视频
    的音频、字幕、切片都在同一张表里），所以刷新时顺手 stat 一遍不会卡。
    需要连视频文件一起对齐的，走 `importer.reconcile()`。
    """
    changed = 0
    stamp = now()
    for row in db.all("SELECT id, path, exists_on_disk, size FROM artifacts WHERE type = ?",
                      (kind,)):
        target = Path(str(row["path"]))
        here = target.is_file()
        if bool(row["exists_on_disk"]) == here:
            continue
        size = target.stat().st_size if here else None
        with db.tx() as conn:
            conn.execute("UPDATE artifacts SET exists_on_disk = ?, size = ?, updated_at = ? "
                         "WHERE id = ?", (int(here), size, stamp, int(row["id"])))
        changed += 1
    if changed:
        logger.info("成品对账：%d 条在盘状态被拨正", changed)
    return changed


def product_counts(db: Database, video_ids: list[int], kind: str = "final_video") -> dict[int, int]:
    """每个视频有几个成品（数据管理表格那一列）。只算**还在盘上**的。"""
    out = {int(v): 0 for v in video_ids}
    if not out:
        return out
    marks = ", ".join("?" for _ in out)
    rows = db.all(
        f"SELECT video_id, COUNT(*) AS n FROM artifacts "
        f"WHERE type = ? AND exists_on_disk = 1 AND video_id IN ({marks}) GROUP BY video_id",
        (kind, *out))
    for row in rows:
        out[int(row["video_id"])] = int(row["n"])
    return out


def list_products(db: Database, video_id: int,
                  kind: str = "final_video") -> list[sqlite3.Row]:
    """这个视频的成品清单（新的排前面）。**界面不再自己写 SELECT。**"""
    return db.all("SELECT * FROM artifacts WHERE video_id = ? AND type = ? ORDER BY id DESC",
                  (video_id, kind))


def product_path(db: Database, artifact_id: int) -> Path | None:
    """成品 id → 文件路径（「打开成品」这类动作用它，界面不查库）。"""
    row = db.one("SELECT path FROM artifacts WHERE id = ?", (artifact_id,))
    return None if row is None else Path(str(row["path"]))


def product_counts_for_assets(db: Database, video_id: int,
                              kind: str = "final_video") -> dict[int, int]:
    """一次查完：这个视频里每份高光 JSON 各剪出了几个成品（`asset_id → 个数`）。

    JSON 表那一列以前是逐行 `products_for_asset()`，20 份 JSON 就是 20+ 次查询；
    这里一条 GROUP BY 解决。
    """
    rows = db.all(
        "SELECT highlight_asset_id AS asset_id, COUNT(*) AS n FROM artifacts "
        "WHERE video_id = ? AND type = ? AND exists_on_disk = 1 "
        "AND highlight_asset_id IS NOT NULL "
        "GROUP BY highlight_asset_id", (video_id, kind))
    return {int(row["asset_id"]): int(row["n"]) for row in rows}


def product_counts_for_prms(db: Database, kind: str = "final_video") -> dict[int, int]:
    """一次查完：每一版 PRM 各剪出了几个成品（`prm_id → 个数`）。

    PRM 管理页那一列以前是逐行 `products_for_prm()`，几十份 PRM 就是几十次查询；
    这里一条 GROUP BY 解决（Phase 16 收 N+1）。
    """
    rows = db.all(
        "SELECT prm_id, COUNT(*) AS n FROM artifacts "
        "WHERE type = ? AND exists_on_disk = 1 AND prm_id IS NOT NULL GROUP BY prm_id", (kind,))
    return {int(row["prm_id"]): int(row["n"]) for row in rows}


def _really_gone(path: Path) -> bool:
    """文件真的被删了，而不是「盘没挂上 / 目录被整个挪走」。

    判据是**父目录还在**：目录在、文件没了 = 用户删了它，清记录是对的；
    目录也不在 = 外接盘掉线、路径变了这类，这时候删记录会把血缘白丢掉。
    """
    return not path.is_file() and path.parent.is_dir()


def forget_missing_products(db: Database, video_id: int,
                            kind: str = "final_video") -> int:
    """把这个视频**文件已经不在盘上**的成品记录删掉，返回删了几条。

    成品文件被手工删掉之后，库里那条记录除了让界面显示一个「⚠ 文件不在盘上」之外
    没有任何用处（血缘指向的是一个不存在的文件）。这是唯一一处真删 artifacts 行的
    地方，**只删真的丢了的**：文件没了但所在目录还在（`_really_gone`）——外接盘
    没挂上、目录被整个挪走的那种一条都不动。
    clips 里指着这个文件的 output_path 一并清掉，免得留下指向空气的渲染记录。
    """
    rows = [row for row in db.all(
        "SELECT id, path FROM artifacts "
        "WHERE video_id = ? AND type = ? AND exists_on_disk = 0", (video_id, kind))
        if _really_gone(Path(str(row["path"])))]
    if not rows:
        return 0
    with db.tx() as conn:
        for row in rows:
            conn.execute("UPDATE clips SET output_path = NULL, status = 'planned' "
                         "WHERE video_id = ? AND output_path = ?",
                         (video_id, str(row["path"])))
            conn.execute("DELETE FROM artifacts WHERE id = ?", (int(row["id"]),))
    logger.info("清理丢失的成品记录：视频 #%s 删掉 %d 条", video_id, len(rows))
    return len(rows)


def purge_missing_products(db: Database, kind: str = "final_video") -> int:
    """全库版：把所有**真的丢了**的成品记录清掉，返回删了几条。

    资产中心每次刷新列表都会跑一次（先 `sync_product_presence` 对账，再清）：
    成品是用户会在资源管理器里直接删的东西，删完还留着记录只会让「成品」数、
    提取数据的跳过清单里全是幽灵。护栏和 `forget_missing_products` 一样——
    只清「目录还在、文件没了」的那些。
    """
    gone: dict[int, list[tuple[int, str]]] = {}
    for row in db.all("SELECT id, video_id, path FROM artifacts "
                      "WHERE type = ? AND exists_on_disk = 0", (kind,)):
        if _really_gone(Path(str(row["path"]))):
            gone.setdefault(int(row["video_id"]), []).append(
                (int(row["id"]), str(row["path"])))
    if not gone:
        return 0
    total = 0
    with db.tx() as conn:
        for video_id, items in gone.items():
            for artifact_id, path in items:
                conn.execute("UPDATE clips SET output_path = NULL, status = 'planned' "
                             "WHERE video_id = ? AND output_path = ?", (video_id, path))
                conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
                total += 1
    logger.info("清理丢失的成品记录：%d 个视频、共 %d 条", len(gone), total)
    return total



def product_progress(db: Database, video_ids: list[int],
                     kind: str = "final_video") -> dict[int, tuple[int, int]]:
    """每个视频「出了几个成品 / 一共该出几个」——一份 JSON 一个成品的进度口径。

    `该出几个` = 未软删且抠得出片段的高光 JSON 份数；`出了几个` = 其中已经有
    **还在盘上**的成品的那几份。两条 GROUP BY 查完，视频再多也不会 N+1。
    没有任何 JSON 的视频返回 `(0, 0)`——它的进度还不由 JSON 决定（等 AI 回话）。
    """
    out = {int(v): (0, 0) for v in video_ids}
    if not out:
        return out
    marks = ", ".join("?" for _ in out)
    ids = tuple(int(v) for v in out)
    total = db.all(
        f"SELECT video_id, COUNT(*) AS n FROM highlight_assets "
        f"WHERE deleted_at IS NULL AND COALESCE(clip_count, 0) > 0 "
        f"AND video_id IN ({marks}) GROUP BY video_id", ids)
    made = db.all(
        f"""
        SELECT h.video_id AS video_id, COUNT(*) AS n FROM highlight_assets h
         WHERE h.deleted_at IS NULL AND COALESCE(h.clip_count, 0) > 0
           AND h.video_id IN ({marks})
           AND EXISTS (SELECT 1 FROM artifacts f
                        WHERE f.highlight_asset_id = h.id AND f.type = ?
                          AND f.exists_on_disk = 1)
         GROUP BY h.video_id
        """, (*ids, kind))
    for row in total:
        out[int(row["video_id"])] = (0, int(row["n"]))
    for row in made:
        vid = int(row["video_id"])
        out[vid] = (int(row["n"]), out[vid][1])
    return out


def detach_product_clips(db: Database, video_id: int, output_path: str | Path) -> int:
    """把库里指向这个成品路径的**旧**渲染记录退回待剪，返回退了几条。

    成品刚落地时先清一遍：文件被覆盖 / 删了重剪之后，之前指着这个路径的 clips 一律过期
    （同名重剪时旧记录会假装成新成品的渲染区间，血缘里就出现「⚠ 不一致」）。
    退回 planned 而不是删掉：它可能是别的高光 JSON 的待剪片段，删了就丢了。
    """
    want = str(output_path)
    rows = db.all("SELECT id FROM clips WHERE video_id = ? AND output_path = ?",
                  (video_id, want))
    if not rows:
        return 0
    with db.tx() as conn:
        for row in rows:
            conn.execute("UPDATE clips SET output_path = NULL, status = 'planned' "
                         "WHERE id = ?", (int(row["id"]),))
    return len(rows)


def repair_product_clips(db: Database, video_id: int,

                         kind: str = "final_video") -> int:
    """把「被别的 JSON 认领」的实际渲染记录退回去，返回修了几条。

    历史脏数据的修复入口：一个视频有好几份 JSON 时，早期版本渲染第一个成品会把**这个
    视频所有** planned 片段都标成 rendered 并写上这次的成品路径，于是别的 JSON 的区间
    被这个成品认领了——血缘里的「实际渲染」就显示成另一份 JSON 的区间（⚠ 不一致）。

    判据是 `ai_result_id`：planned 片段和高光方案都记着自己出自哪次 AI 结果。
    指着某个成品、但 `ai_result_id` 明确属于**别次**结果的片段，一律清掉 output_path
    并退回 planned。`ai_result_id` 为空的不动——那是渲染完按实际区间补建的，
    天然属于这个成品。
    """
    fixed = 0
    for art in db.all("SELECT id, path, highlight_asset_id FROM artifacts "
                      "WHERE video_id = ? AND type = ?", (video_id, kind)):
        if art["highlight_asset_id"] is None:
            continue
        row = db.one("SELECT ai_result_id FROM highlight_assets WHERE id = ?",
                     (int(art["highlight_asset_id"]),))
        result_id = None if row is None else row["ai_result_id"]
        bad = db.all(
            "SELECT id FROM clips WHERE video_id = ? AND output_path = ? "
            "AND ai_result_id IS NOT NULL AND ai_result_id IS NOT ?",
            (video_id, str(art["path"]), result_id))
        if not bad:
            continue
        with db.tx() as conn:
            for clip in bad:
                conn.execute("UPDATE clips SET output_path = NULL, status = 'planned' "
                             "WHERE id = ?", (int(clip["id"]),))
        fixed += len(bad)
    if fixed:
        logger.info("修正实际渲染记录：视频 #%s 退回 %d 条被张冠李戴的片段", video_id, fixed)
    return fixed


def product_rename_plan(db: Database, video_ids: list[int] | None = None,
                        kind: str = "final_video") -> list[dict[str, Any]]:
    """按现在的命名口径给成品算一份改名计划（**只读**，一个文件都不动）。

    口径和渲染时一致：名字 = `<原视频名>_<实际剪进去的区间长度>.mp4`。时长从哪来：

      1. 这个成品的实际渲染区间（`clips` 按 output_path 归组）——最准，引擎修正过
         也算在里面；
      2. 没有 clips 就按它挂的高光 JSON 现算 `(end + freeze) − (sa + startframe)`；
      3. 两条都没有就跳过，`skip` 里写原因。

    每条返回：`artifact_id` / `video_id` / `old`（现路径）/ `new`（目标路径）/
    `seconds` / `skip`（None = 可以改）。名字已经对的返回 `skip="名字已经对了"`。
    """
    marks = ""
    args: list[Any] = [kind]
    if video_ids:
        marks = " AND f.video_id IN (%s)" % ", ".join("?" for _ in video_ids)
        args += [int(v) for v in video_ids]
    rows = db.all(
        f"""
        SELECT f.id, f.video_id, f.path, f.highlight_asset_id, f.exists_on_disk,
               v.file_name
          FROM artifacts f JOIN videos v ON v.id = f.video_id
         WHERE f.type = ?{marks}
         ORDER BY f.video_id, f.id
        """, tuple(args))
    out: list[dict[str, Any]] = []
    taken: set[str] = set()      # 这一批已经分配掉的目标名，撞了就是覆盖
    for row in rows:
        old = Path(str(row["path"]))
        item: dict[str, Any] = {"artifact_id": int(row["id"]),
                               "video_id": int(row["video_id"]),
                               "old": old, "new": None, "seconds": None,
                               "overwrite": False, "skip": None}
        if not old.is_file():
            item["skip"] = "文件不在盘上"
            out.append(item)
            continue
        span = db.value("SELECT duration FROM clips WHERE video_id = ? AND output_path = ? "
                        "ORDER BY start_time LIMIT 1",
                        (int(row["video_id"]), str(row["path"])), None)
        seconds = ai_protocol.num(span)
        if seconds is None and row["highlight_asset_id"] is not None:
            payload = asset_payload(db, int(row["highlight_asset_id"]))
            found = ai_protocol.clips(payload)
            if found:
                clip = found[0]
                seconds = ai_protocol.play_seconds(
                    clip, startframe=ai_protocol.num(clip.get("startframe")) or 0.0,
                    freeze=ai_protocol.num(clip.get("freeze")) or 0.0)
        if seconds is None or seconds <= 0:
            item["skip"] = "既没有渲染区间也没有来源 JSON，算不出时长"
            out.append(item)
            continue
        item["seconds"] = round(seconds, 2)
        stem = Path(str(row["file_name"])).stem
        item["new"] = old.with_name(
            f"{stem}_{ai_protocol.duration_tag(seconds)}{old.suffix}")
        if item["new"] == old:
            item["skip"] = "名字已经对了"
        elif str(item["new"]) in taken or item["new"].exists():
            # 同名就覆盖（同一视频两个成品剪出同样时长时会撞）。被顶掉的那条记录
            # 在 apply 时一并删掉，免得库里两行指着同一个文件
            item["overwrite"] = True
        taken.add(str(item["new"]))
        out.append(item)
    return out


def apply_product_rename(db: Database, items: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """执行改名计划：**先改文件、再改库**，返回 (改了几个, 没改成的原因)。

    库里跟着走两处：`artifacts.path` 和 `clips.output_path`（血缘就是按这两个串起来的，
    只改文件不改库等于把血缘弄断）。文件改成功但库写失败时把文件名改回去——
    宁可什么都没变，也不留下"文件和库对不上"的中间态。

    目标名撞了就**覆盖**（同一视频两个成品剪出同样时长时会撞）：`os.replace` 顶掉那个
    文件之后，把库里指着这个路径的**别的** artifacts 行一并删掉、它们的 clips 退回
    planned——盘上只剩一个文件，库里就只该剩一条记录。
    """
    done = 0
    failed: list[str] = []
    stamp = now()
    for item in items:
        if item.get("skip") or item.get("new") is None:
            continue
        old, new = Path(item["old"]), Path(item["new"])
        try:
            os.replace(old, new)
        except OSError as exc:
            failed.append(f"{old.name}：{exc}")
            continue
        try:
            with db.tx() as conn:
                # 被顶掉的那个文件对应的记录清掉：它的内容已经是这一条的了
                for other in db.all(
                        "SELECT id, video_id FROM artifacts WHERE path = ? AND id != ?",
                        (str(new), int(item["artifact_id"]))):
                    conn.execute("UPDATE clips SET output_path = NULL, status = 'planned' "
                                 "WHERE video_id = ? AND output_path = ?",
                                 (int(other["video_id"]), str(new)))
                    conn.execute("DELETE FROM artifacts WHERE id = ?", (int(other["id"]),))
                conn.execute("UPDATE artifacts SET path = ?, updated_at = ? WHERE id = ?",
                             (str(new), stamp, int(item["artifact_id"])))
                conn.execute("UPDATE clips SET output_path = ? "
                             "WHERE video_id = ? AND output_path = ?",
                             (str(new), int(item["video_id"]), str(old)))
        except Exception as exc:  # noqa: BLE001 - 库写不进去就把文件名改回去
            try:
                os.replace(new, old)
            except OSError:
                failed.append(f"{new.name}：库没更新，文件名也没能改回去（{exc}）")
            else:
                failed.append(f"{old.name}：库写不进去，已还原文件名（{exc}）")
            continue
        done += 1
    if done:
        logger.info("成品批量改名：%d 个（文件 + artifacts.path + clips.output_path）", done)
    return done, failed


def products_overview(db: Database, video_id: int,
                      kind: str = "final_video") -> list[dict[str, Any]]:
    """当前视频的成品全景：**两条 SQL** 就带出成品表要显示的一切。

    每条包含 artifact / 来源高光 JSON / PRM / 生成时间 / 文件状态 /
    实际渲染区间（`spans`，来自 clips，按 output_path 归组）。
    以前成品表是逐行 `artifact_lineage()` + `clips_for_product()`，60 行要 300 多次查询。
    """
    rows = db.all(
        """
        SELECT f.id, f.path, f.created_at, f.exists_on_disk,
               f.highlight_asset_id, f.prm_id,
               h.name AS asset_name, h.deleted_at AS asset_deleted,
               p.name AS prm_name,   p.deleted_at AS prm_deleted
          FROM artifacts f
          LEFT JOIN highlight_assets h ON h.id = f.highlight_asset_id
          LEFT JOIN prm_profiles     p ON p.id = f.prm_id
         WHERE f.video_id = ? AND f.type = ?
         ORDER BY f.id DESC
        """, (video_id, kind))
    spans: dict[str, list[dict[str, Any]]] = {}
    for clip in db.all("SELECT output_path, start_time, end_time, duration FROM clips "
                       "WHERE video_id = ? AND output_path IS NOT NULL "
                       "ORDER BY start_time", (video_id,)):
        spans.setdefault(str(clip["output_path"]), []).append(
            {"start": clip["start_time"], "end": clip["end_time"],
             "duration": clip["duration"]})
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "artifact_id": int(row["id"]),
            "path": str(row["path"]),
            "created_at": row["created_at"],
            "exists_on_disk": bool(row["exists_on_disk"]),
            "asset_id": None if row["highlight_asset_id"] is None
            else int(row["highlight_asset_id"]),
            "asset_name": row["asset_name"],
            "asset_deleted": bool(row["asset_deleted"]),
            "prm_id": None if row["prm_id"] is None else int(row["prm_id"]),
            "prm_name": row["prm_name"],
            "prm_deleted": bool(row["prm_deleted"]),
            "spans": spans.get(str(row["path"]), []),
        })
    return out


def artifact_lineage(db: Database, artifact_id: int) -> dict[str, Any] | None:
    """成品反查：视频 / analysis / 方案 / AI / 模型 / PRM / 任务 / 时间。

    方案或 PRM 被软删了也照样查得到，只是多一个 `asset_deleted` / `prm_deleted` 标记。
    **一条 JOIN 查完**（以前 artifacts / videos / highlight_assets / prm_profiles 各查一次）。
    """
    art = db.one(
        """
        SELECT f.id, f.path, f.type, f.size, f.exists_on_disk, f.created_at,
               f.highlight_asset_id, f.prm_id,
               v.id AS v_id, v.file_name AS v_name, v.file_path AS v_path,
               v.duration AS v_duration,
               h.id AS h_id, h.name AS h_name, h.source_type AS h_source,
               h.clip_count AS h_clips, h.best_score AS h_score,
               h.created_at AS h_created, h.deleted_at AS h_deleted,
               h.analysis_id AS h_analysis, h.provider AS h_provider,
               h.model AS h_model, h.source_task_id AS h_task,
               p.id AS p_id, p.name AS p_name, p.filename AS p_file,
               p.version AS p_version, p.language AS p_lang, p.deleted_at AS p_deleted
          FROM artifacts f
          LEFT JOIN videos          v ON v.id = f.video_id
          LEFT JOIN highlight_assets h ON h.id = f.highlight_asset_id
          LEFT JOIN prm_profiles     p ON p.id = f.prm_id
         WHERE f.id = ?
        """, (artifact_id,))
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
    if art["v_id"] is not None:
        out["video"] = {"id": int(art["v_id"]), "file_name": art["v_name"],
                        "file_path": art["v_path"], "duration": art["v_duration"]}
    if art["h_id"] is not None:
        out["asset"] = {"id": int(art["h_id"]), "name": art["h_name"],
                        "source_type": art["h_source"],
                        "clip_count": int(art["h_clips"] or 0),
                        "best_score": art["h_score"],
                        "created_at": art["h_created"]}
        out["asset_deleted"] = bool(art["h_deleted"])
        out["analysis_id"] = art["h_analysis"]
        out["provider"] = art["h_provider"]
        out["model"] = art["h_model"]
        out["task_id"] = art["h_task"]
    if art["p_id"] is not None:
        out["prm"] = {"id": int(art["p_id"]), "name": art["p_name"],
                      "filename": art["p_file"], "version": art["p_version"],
                      "language": art["p_lang"]}
        out["prm_deleted"] = bool(art["p_deleted"])
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


# ====================================================== 资产中心的列表（只读）
CENTER_STATUS = ("all", "analysed", "not_analysed", "has_json", "no_json",
                 "has_product", "no_product")
CENTER_ORDER = ("recent", "processed", "name", "duration", "json", "highlight", "product")
# 「有没有 JSON」「有没有成品」是两个独立维度，各自 any / has / none
CENTER_HAS = ("any", "has", "none")
# AI 筛选里的「未知」：有高光 JSON，但库里没记下 provider（多半是手工导入的）
NO_PROVIDER = "__none__"


def _under(path: str | None, folder: str | None) -> bool:
    """`path` 在 `folder`（含子目录）里吗。folder 留空 = 不筛，一律算命中。

    Windows 上同一个目录会以 `F:/a`、`F:\\a`、大小写不同的写法出现（8.3 短名也算），
    所以两边都 normcase + normpath 再比前缀，不做字符串裸比较。
    """
    if not folder:
        return True
    if not path:
        return False
    base = os.path.normcase(os.path.normpath(str(folder)))
    here = os.path.normcase(os.path.normpath(str(path)))
    return here == base or here.startswith(base.rstrip(os.sep) + os.sep)


def known_dirs(db: Database) -> tuple[list[str], list[str]]:
    """库里出现过的目录：（原视频目录，成品目录），各自去重排序，给筛选下拉用。

    扫描目录下面常常有几十个子目录，所以这里给的是**每个文件真正所在的那一层**，
    选哪个就只看哪个。文件在不在盘上不影响它出现在列表里（库是权威）。
    """
    videos = {str(Path(str(r["file_path"])).parent)
              for r in db.all("SELECT file_path FROM videos", ())
              if r["file_path"]}
    products = {str(Path(str(r["path"])).parent)
                for r in db.all("SELECT path FROM artifacts WHERE type = 'final_video'", ())
                if r["path"]}
    return sorted(videos), sorted(products)


def center_rows(db: Database, *, search: str | None = None, provider: str | None = None,
                status: str = "all", json: str = "any", product: str = "any",
                video_dir: str | None = None, product_dir: str | None = None,
                order: str = "recent", limit: int = 500) -> list[dict[str, Any]]:
    """资产中心主列表：一个视频一行，JSON 数 / 高光数 / 成品数 / 最近用的 AI 全在里面。

    全部靠 SQL 聚合，**GUI 不需要扫磁盘也不需要逐个视频查库**——几千个视频也是一次查询。
    `search` 同时匹配文件名、路径和视频 ID；`provider` 按 AI 过滤（`NO_PROVIDER`
    = 只看没记下 AI 的）。

    筛选维度三个，可以随便组合（这才能一步问出「有 JSON 但没成品」）：

      * `status`：`all` / `analysed` / `not_analysed`
      * `json`：`any` / `has` / `none`
      * `product`：`any` / `has` / `none`

    再加两个**目录**维度（扫描目录下常常有几十个子目录，得能只看其中一个）：

      * `video_dir`：只看原视频落在这个目录（含子目录）里的
      * `product_dir`：按**成品**所在目录收窄。它只收窄「有成品」的那部分，
        **不会把没成品的视频顺手筛掉**（列表显示什么完全由上面三个维度决定，
        目录不许偷偷加一条"必须有成品"）。三种组合分别是：

          - `product="has"`：严格——必须有成品，且至少一个成品落在这个目录下
          - `product="any"`：有成品的按目录筛（成品全在别处的排掉），
            `product_count == 0` 的视频照样留着
          - `product="none"`：这一条**直接不生效**——只看没成品的视频，
            再按成品目录筛就自相矛盾了

    老调用写的是 `status="has_json"` 这类合并档，仍然照旧支持：显式给了
    `json=` / `product=` 时以它们为准，没给才把老 `status` 翻译过来。
    筛选和排序只有这一处，GUI 不再自己过一遍。
    """



    want_json = str(json or "any")
    want_product = str(product or "any")
    keep_status = str(status or "all")
    legacy = {"has_json": ("has", "any"), "no_json": ("none", "any"),
              "has_product": ("any", "has"), "no_product": ("any", "none")}
    if keep_status in legacy:                      # 老的合并档：翻译成两个维度
        was_json, was_product = legacy[keep_status]
        if want_json == "any":
            want_json = was_json
        if want_product == "any":
            want_product = was_product
        keep_status = "all"

    where, params = [], []
    if search:
        text = str(search).strip()
        clause_or = ["v.file_name LIKE ?", "v.file_path LIKE ?"]
        params.extend([f"%{text}%", f"%{text}%"])
        if text.isdigit():
            clause_or.append("v.id = ?")
            params.append(int(text))
        where.append("(" + " OR ".join(clause_or) + ")")
    if provider and provider != NO_PROVIDER:
        where.append("EXISTS (SELECT 1 FROM highlight_assets p WHERE p.video_id = v.id "
                     "AND p.deleted_at IS NULL AND p.provider = ?)")
        params.append(provider)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.all(
        f"""
        SELECT v.id, v.file_name, v.file_path, v.duration,
               (SELECT COUNT(*) FROM analysis_runs a
                 WHERE a.video_id = v.id AND a.status = 'completed')            AS analysed,
               (SELECT COUNT(*) FROM highlight_assets h
                 WHERE h.video_id = v.id AND h.deleted_at IS NULL)              AS json_count,
               (SELECT COALESCE(SUM(h.clip_count), 0) FROM highlight_assets h
                 WHERE h.video_id = v.id AND h.deleted_at IS NULL)              AS highlight_count,
               (SELECT MAX(h.updated_at) FROM highlight_assets h
                 WHERE h.video_id = v.id AND h.deleted_at IS NULL)              AS asset_updated,
               (SELECT COUNT(*) FROM artifacts f
                 WHERE f.video_id = v.id AND f.type = 'final_video'
                   AND f.exists_on_disk = 1)                                    AS product_count,
               (SELECT MAX(f.created_at) FROM artifacts f
                 WHERE f.video_id = v.id AND f.type = 'final_video'
                   AND f.exists_on_disk = 1)                                    AS product_updated

          FROM videos v{clause}
        """, tuple(params))

    latest: dict[int, tuple[str, str]] = {}
    for row in db.all("SELECT video_id, provider, model FROM highlight_assets "
                      "WHERE deleted_at IS NULL ORDER BY id", ()):
        if row["provider"] or row["model"]:
            latest[int(row["video_id"])] = (str(row["provider"] or ""), str(row["model"] or ""))

    # 成品目录筛选要知道每个视频的成品都落在哪儿：一次查完，不逐行发 SQL
    product_paths: dict[int, list[str]] = {}
    if product_dir:
        for row in db.all("SELECT video_id, path FROM artifacts "
                          "WHERE type = 'final_video' AND exists_on_disk = 1", ()):
            product_paths.setdefault(int(row["video_id"]), []).append(str(row["path"] or ""))

    out: list[dict[str, Any]] = []
    for row in rows:
        vid = int(row["id"])
        item = {
            "id": vid,
            "file_name": str(row["file_name"]),
            "file_path": str(row["file_path"]),
            "duration": row["duration"],
            "analysed": bool(int(row["analysed"] or 0)),
            "json_count": int(row["json_count"] or 0),
            "highlight_count": int(row["highlight_count"] or 0),
            "product_count": int(row["product_count"] or 0),
            "updated_at": row["asset_updated"] or "",
            "product_updated": row["product_updated"] or "",
            "provider": latest.get(vid, ("", ""))[0],
            "model": latest.get(vid, ("", ""))[1],
        }
        if provider == NO_PROVIDER and (item["provider"] or not item["json_count"]):
            continue                      # 「未知」= 有 JSON 但没记下 AI
        if keep_status == "analysed" and not item["analysed"]:
            continue
        if keep_status == "not_analysed" and item["analysed"]:
            continue
        if want_json == "has" and not item["json_count"]:
            continue
        if want_json == "none" and item["json_count"]:
            continue
        if want_product == "has" and not item["product_count"]:
            continue
        if want_product == "none" and item["product_count"]:
            continue
        if not _under(item["file_path"], video_dir):
            continue
        # 成品目录只用来收窄「有成品」的那一部分。没成品的视频在这个目录下当然一个
        # 文件都没有，但它不是"不符合筛选"，只是还没剪出来——真按目录把它们筛掉，
        # 列表就变成了隐式的"只显示有成品的视频"，而用户压根没这么要求
        # （这个目录还被记在 config 里，重启也还在，一选就再也看不到待剪的视频）。
        # 所以：只看无成品（`none`）时这一条整条跳过；其余情况只筛有成品的那些。
        if product_dir and want_product != "none" and item["product_count"]:
            if not any(_under(p, product_dir) for p in product_paths.get(vid, ())):
                continue
        out.append(item)


    keys = {
        "recent": lambda r: (r["updated_at"] or "", r["id"]),
        "processed": lambda r: (r["product_updated"] or "", r["id"]),
        "name": lambda r: r["file_name"].lower(),
        "duration": lambda r: (float(r["duration"] or 0.0), r["id"]),
        "json": lambda r: (r["json_count"], r["id"]),
        "highlight": lambda r: (r["highlight_count"], r["id"]),
        "product": lambda r: (r["product_count"], r["id"]),
    }

    out.sort(key=keys.get(order, keys["recent"]), reverse=order != "name")
    return out[:max(1, int(limit))]


def known_providers(db: Database) -> list[str]:
    """库里出现过的 AI 来源，给筛选下拉用。"""
    return [str(r["provider"]) for r in
            db.all("SELECT DISTINCT provider FROM highlight_assets "
                   "WHERE provider IS NOT NULL AND provider <> '' ORDER BY provider", ())]
