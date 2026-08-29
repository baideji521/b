"""数据库管理：体检、备份、恢复、瘦身、孤儿检查。

分工没有变——**文件是文件，数据库是状态**。这里只管数据库本身好不好，
不碰业务流程，也不删任何历史记录（哪怕文件早没了，记录仍然是历史事实）。

几条原则：
* 备份不用 `copy video.db`。WAL 模式下光拷主库会丢掉还在 WAL 里的那部分，
  拷出来的库逻辑上是残的。这里走 SQLite 官方 backup API，落一个自洽的单文件。
* 恢复不覆盖文件，而是反向用 backup API 把备份写进当前库：
  Windows 上库文件正被打开时改名/替换很容易失败，页级写入没这个问题。
  写之前先给当前库留一份安全备份，写完再 integrity_check。
* VACUUM 只在用户主动叫的时候跑，并且先确认没有正在跑的任务。
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .db import Database
from .schema import SCHEMA_VERSION, TABLES, TASK_ACTIVE

logger = get_logger(__name__)

# 从建表语句里现取表名/索引名，别在这儿再抄一份清单（抄了就会跟 schema.py 走散）
EXPECTED_TABLES: tuple[str, ...] = tuple(
    m.group(1) for stmt in TABLES
    if (m := re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", stmt)))
EXPECTED_INDEXES: tuple[str, ...] = tuple(
    m.group(1) for stmt in TABLES
    if (m := re.search(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)", stmt)))

# 外键关系：孤儿检查按这张表逐条 LEFT JOIN
_ORPHAN_RULES: tuple[tuple[str, str, str, str], ...] = (
    # (子表, 子表外键列, 父表, 说明)
    ("ai_results", "task_id", "ai_tasks", "AI 结果找不到对应任务"),
    ("clips", "ai_result_id", "ai_results", "片段找不到对应 AI 结果"),
    ("artifacts", "video_id", "videos", "文件记录找不到对应视频"),
    ("analysis_runs", "video_id", "videos", "分析批次找不到对应视频"),
    ("speech_segments", "analysis_id", "analysis_runs", "语音段找不到对应分析批次"),
    ("speech_words", "segment_id", "speech_segments", "逐词找不到对应语音段"),
    ("visual_events", "analysis_id", "analysis_runs", "视觉事件找不到对应分析批次"),
    ("ai_tasks", "video_id", "videos", "AI 任务找不到对应视频"),
    ("ai_results", "video_id", "videos", "AI 结果找不到对应视频"),
    ("clips", "video_id", "videos", "片段找不到对应视频"),
)


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ==================================================================== 体检
def health_check(db: Database) -> dict[str, Any]:
    """数据库体检。返回 ok / 各项结论 / problems 列表，CLI 据此定 exit code。"""
    conn = db.connect()
    result: dict[str, Any] = {
        "path": str(db.path),
        "problems": [],
        "integrity": "?",
        "foreign_keys": "?",
        "fk_violations": [],
        "version": None,
        "expected_version": SCHEMA_VERSION,
        "missing_tables": [],
        "missing_indexes": [],
        "journal_mode": "?",
        "foreign_keys_pragma": 0,
        "writable": False,
        "size_bytes": 0,
    }

    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        verdict = [str(r[0]) for r in rows]
        result["integrity"] = "OK" if verdict == ["ok"] else "；".join(verdict[:5])
        if verdict != ["ok"]:
            result["problems"].append(f"integrity_check 不干净：{result['integrity']}")
    except sqlite3.Error as exc:
        result["integrity"] = f"查不了（{exc}）"
        result["problems"].append(f"integrity_check 执行失败：{exc}")

    try:
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        result["fk_violations"] = [tuple(r) for r in bad[:20]]
        result["foreign_keys"] = "OK" if not bad else f"{len(bad)} 处不一致"
        if bad:
            result["problems"].append(f"foreign_key_check 发现 {len(bad)} 处不一致")
    except sqlite3.Error as exc:
        result["foreign_keys"] = f"查不了（{exc}）"
        result["problems"].append(f"foreign_key_check 执行失败：{exc}")

    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    result["version"] = version
    if version != SCHEMA_VERSION:
        result["problems"].append(f"结构版本是 v{version}，程序要的是 v{SCHEMA_VERSION}")

    have_tables = {str(r[0]) for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    have_indexes = {str(r[0]) for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'")}
    result["missing_tables"] = [t for t in EXPECTED_TABLES if t not in have_tables]
    result["missing_indexes"] = [i for i in EXPECTED_INDEXES if i not in have_indexes]
    if result["missing_tables"]:
        result["problems"].append("缺表：" + "、".join(result["missing_tables"]))
    if result["missing_indexes"]:
        result["problems"].append("缺索引：" + "、".join(result["missing_indexes"]))

    result["journal_mode"] = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    if result["journal_mode"].lower() != "wal":
        result["problems"].append(f"日志模式是 {result['journal_mode']}，本项目要 WAL")
    result["foreign_keys_pragma"] = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if not result["foreign_keys_pragma"]:
        result["problems"].append("这条连接没开 foreign_keys")

    # 真写一下再撤，只有能写才算健康（库文件被只读挂载、被占用都能在这里暴露）
    try:
        with db.tx() as tx:
            tx.execute("INSERT INTO schema_meta(key, value) VALUES('health_probe', ?) "
                       "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (_stamp(),))
        result["writable"] = True
    except sqlite3.Error as exc:
        result["problems"].append(f"写不进去：{exc}")

    try:
        result["size_bytes"] = db.path.stat().st_size
    except OSError:
        pass

    result["ok"] = not result["problems"]
    return result


def verify_file(path: str | Path) -> dict[str, Any]:
    """单独检查一个库文件（备份文件也走这条），不建表不升级。"""
    path = Path(path)
    out: dict[str, Any] = {"path": str(path), "problems": [], "counts": {},
                           "integrity": "?", "foreign_keys": "?", "version": None}
    if not path.is_file():
        out["problems"].append("文件不存在")
        out["ok"] = False
        return out
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        verdict = [str(r[0]) for r in conn.execute("PRAGMA integrity_check").fetchall()]
        out["integrity"] = "OK" if verdict == ["ok"] else "；".join(verdict[:5])
        if verdict != ["ok"]:
            out["problems"].append(f"integrity_check 不干净：{out['integrity']}")
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        out["foreign_keys"] = "OK" if not bad else f"{len(bad)} 处不一致"
        if bad:
            out["problems"].append(f"foreign_key_check 发现 {len(bad)} 处不一致")
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        out["version"] = version
        if version > SCHEMA_VERSION:
            out["problems"].append(f"备份的结构版本 v{version} 比程序 v{SCHEMA_VERSION} 还新")
        have = {str(r[0]) for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = [t for t in EXPECTED_TABLES if t not in have]
        if missing:
            out["problems"].append("缺表：" + "、".join(missing))
        for table in EXPECTED_TABLES:
            if table in have:
                out["counts"][table] = int(
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        out["problems"].append(f"打不开或不是 SQLite 库：{exc}")
    finally:
        conn.close()
    out["ok"] = not out["problems"]
    return out


# ==================================================================== 备份
def backup(db: Database, dest: str | Path | None = None) -> Path:
    """用 SQLite backup API 落一份自洽的库文件，返回备份路径。

    默认放 `<库目录>/backups/video_<时间戳>.db`。WAL 里的内容会一起进去，
    所以这份文件可以单独打开、单独恢复。
    """
    if dest is None:
        folder = db.path.parent / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{db.path.stem}_{_stamp()}.db"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == db.path.resolve():
        raise ValueError("备份路径不能就是当前库文件")
    target = sqlite3.connect(str(dest))
    try:
        db.connect().backup(target)
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        target.close()
    logger.info("已备份到 %s", dest)
    return dest


def restore(db: Database, source: str | Path) -> dict[str, Any]:
    """从备份恢复。坏备份绝不允许覆盖当前库。

    顺序：验备份 -> 给当前库留一份安全备份 -> 反向 backup 写回 -> 再验当前库。
    """
    source = Path(source)
    report: dict[str, Any] = {"source": str(source), "restored": False}
    checked = verify_file(source)
    report["backup_check"] = checked
    if not checked["ok"]:
        report["error"] = "备份本身不合格：" + "；".join(checked["problems"])
        return report

    safety = backup(db)
    report["safety_backup"] = str(safety)

    src = sqlite3.connect(str(source))
    try:
        src.backup(db.connect())
    finally:
        src.close()
    report["restored"] = True

    # 先数行数，再体检——体检会往 schema_meta 写一次探针，先数就不会把探针算进恢复结果
    report["counts"] = {t: int(db.value(f"SELECT COUNT(*) FROM {t}", default=0))
                        for t in EXPECTED_TABLES}
    after = health_check(db)
    report["after_check"] = after
    if not after["ok"]:
        report["error"] = "恢复后体检不过：" + "；".join(after["problems"])
    logger.info("已从 %s 恢复（当前库先备份到 %s）", source, safety)
    return report


# ==================================================================== 瘦身
def vacuum(db: Database, *, force: bool = False) -> dict[str, Any]:
    """整理库文件。有任务在跑就不做，除非 force。VACUUM 之后确认还是 WAL。"""
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    busy_tasks = int(db.value(f"SELECT COUNT(*) FROM ai_tasks WHERE status IN ({marks})",
                              tuple(TASK_ACTIVE), default=0))
    busy_runs = int(db.value("SELECT COUNT(*) FROM analysis_runs WHERE status = 'running'",
                             default=0))
    out: dict[str, Any] = {"busy_tasks": busy_tasks, "busy_runs": busy_runs, "done": False,
                           "size_before": 0, "size_after": 0}
    try:
        out["size_before"] = db.path.stat().st_size
    except OSError:
        pass
    if (busy_tasks or busy_runs) and not force:
        out["error"] = f"还有 {busy_tasks} 个 AI 任务、{busy_runs} 条分析在跑，先等它们结束"
        return out
    conn = db.connect()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.execute("PRAGMA journal_mode=WAL")  # VACUUM 不该改模式，还是确认一遍
    out["journal_mode"] = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    out["integrity"] = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    try:
        out["size_after"] = db.path.stat().st_size
    except OSError:
        pass
    out["done"] = out["integrity"] == "ok"
    if not out["done"]:
        out["error"] = f"VACUUM 之后 integrity_check 是 {out['integrity']}"
    return out


# ================================================================== 孤儿
def orphans(db: Database, *, sample: int = 5) -> dict[str, Any]:
    """只报告不删。库里对不上的关联、登记了但文件没了的产物，都列出来。"""
    found: list[dict[str, Any]] = []
    for child, column, parent, why in _ORPHAN_RULES:
        sql = (f"SELECT c.id FROM {child} c "
               f"LEFT JOIN {parent} p ON p.id = c.{column} "
               f"WHERE c.{column} IS NOT NULL AND p.id IS NULL")
        ids = [int(r[0]) for r in db.all(sql)]
        if ids:
            found.append({"table": child, "column": column, "parent": parent,
                          "why": why, "count": len(ids), "sample": ids[:sample]})
    missing = db.all(
        "SELECT a.id, a.type, a.path FROM artifacts a WHERE a.exists_on_disk = 1")
    gone = [{"id": int(r["id"]), "type": str(r["type"]), "path": str(r["path"])}
            for r in missing if not Path(str(r["path"])).exists()]
    return {
        "relations": found,
        "relation_total": sum(item["count"] for item in found),
        "artifacts_missing": gone[:50],
        "artifacts_missing_total": len(gone),
    }


def unregistered_files(cfg: Any, db: Database) -> dict[str, Any]:
    """盘上有、库里没登记的视频。只数不写——真要补登记走 `db --reconcile`。"""
    from .importer import VIDEO_SUFFIXES, _known_video, _scan_dirs  # noqa: PLC0415

    names: list[str] = []
    for folder in _scan_dirs(cfg):
        for item in sorted(folder.rglob("*")):
            if not item.is_file() or item.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            if _known_video(db, item) is None:
                names.append(str(item))
    return {"count": len(names), "sample": names[:20]}
