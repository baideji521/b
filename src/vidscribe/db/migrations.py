"""表结构升级：`PRAGMA user_version` 记版本，一步一步往上走。

第一次打开就是 v0 -> 建全部表 -> 写成当前版本。
以后加字段/加表就在 `_STEPS` 里追加一段，不要直接改 schema.py 里已有的表定义，
否则老库升不上来。升级只做加法（ADD COLUMN / CREATE TABLE），不删列不改类型。
"""

from __future__ import annotations

import sqlite3

from ..logging_setup import get_logger
from .schema import SCHEMA_VERSION, TABLES

logger = get_logger(__name__)


def _create_all(conn: sqlite3.Connection) -> None:
    for statement in TABLES:
        conn.execute(statement)


# 版本 N 的升级脚本：从 N-1 升到 N 要干什么。v1 就是建全套表。
_STEPS: dict[int, list[str]] = {
    # v2：ai_tasks 变成真正的持久化队列——任务种类、优先级、尝试次数上限、
    # 是谁在跑、最后一次改动时间；再加一个「同一视频同一任务只能有一条没跑完」的唯一索引。
    2: [
        "ALTER TABLE ai_tasks ADD COLUMN task_type TEXT NOT NULL DEFAULT 'auto_clip'",
        "ALTER TABLE ai_tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 100",
        "ALTER TABLE ai_tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE ai_tasks ADD COLUMN worker_id TEXT",
        "ALTER TABLE ai_tasks ADD COLUMN updated_at TEXT",
        # 老库里万一已经有重复的未完成任务，先留最早那条，其余标 cancelled，
        # 否则下面这个唯一索引建不起来。
        """
        UPDATE ai_tasks SET status = 'cancelled', finished_at = COALESCE(finished_at, created_at)
         WHERE status IN ('pending', 'uploading', 'waiting', 'processing')
           AND id NOT IN (
               SELECT MIN(id) FROM ai_tasks
                WHERE status IN ('pending', 'uploading', 'waiting', 'processing')
                GROUP BY video_id, task_type, mode)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_open_unique
            ON ai_tasks(video_id, task_type, mode)
         WHERE status IN ('pending', 'uploading', 'waiting', 'processing')
        """,
    ],
    # v3：记下每次真正发给 AI 的那份提示词文件的指纹（内容不进库）。
    # 只加列、不动索引不动外键：老数据全留着，新列一律 NULL。
    3: [
        "ALTER TABLE ai_tasks ADD COLUMN prompt_hash TEXT",
        "ALTER TABLE ai_tasks ADD COLUMN prompt_path TEXT",
        "ALTER TABLE ai_tasks ADD COLUMN prompt_size INTEGER",
        "ALTER TABLE ai_results ADD COLUMN prompt_hash TEXT",
        "ALTER TABLE ai_results ADD COLUMN prompt_path TEXT",
        "ALTER TABLE ai_results ADD COLUMN prompt_size INTEGER",
    ],
    # v4：高光 JSON 变成"资产"（highlight_assets）+ PRM 档案（prm_profiles），
    # 成品多两列指回它们。只建表 / 只加列，老数据一行不动，新列全是 NULL。
    4: [
        """
        CREATE TABLE IF NOT EXISTS prm_profiles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            filename    TEXT    NOT NULL,
            description TEXT,
            language    TEXT,
            version     TEXT,
            is_default  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            deleted_at  TEXT
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prm_name_live
            ON prm_profiles(name) WHERE deleted_at IS NULL
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prm_default_live
            ON prm_profiles(is_default) WHERE is_default = 1 AND deleted_at IS NULL
        """,
        """
        CREATE TABLE IF NOT EXISTS highlight_assets (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id       INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            analysis_id    INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
            source_task_id INTEGER REFERENCES ai_tasks(id) ON DELETE SET NULL,
            ai_result_id   INTEGER REFERENCES ai_results(id) ON DELETE SET NULL,
            prm_id         INTEGER REFERENCES prm_profiles(id) ON DELETE SET NULL,
            parent_id      INTEGER REFERENCES highlight_assets(id) ON DELETE SET NULL,
            provider       TEXT,
            model          TEXT,
            source_type    TEXT    NOT NULL DEFAULT 'ai',
            name           TEXT    NOT NULL,
            version        INTEGER NOT NULL DEFAULT 1,
            raw_json       TEXT    NOT NULL,
            current_json   TEXT    NOT NULL,
            clip_count     INTEGER NOT NULL DEFAULT 0,
            best_score     REAL,
            is_current     INTEGER NOT NULL DEFAULT 0,
            note           TEXT,
            created_at     TEXT    NOT NULL,
            updated_at     TEXT    NOT NULL,
            deleted_at     TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_assets_video ON highlight_assets(video_id, deleted_at)",
        "CREATE INDEX IF NOT EXISTS idx_assets_ai ON highlight_assets(provider, model)",
        "CREATE INDEX IF NOT EXISTS idx_assets_prm ON highlight_assets(prm_id)",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_current_live
            ON highlight_assets(video_id)
         WHERE is_current = 1 AND deleted_at IS NULL
        """,
        "ALTER TABLE artifacts ADD COLUMN highlight_asset_id INTEGER",
        "ALTER TABLE artifacts ADD COLUMN prm_id INTEGER",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_asset ON artifacts(highlight_asset_id)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_prm ON artifacts(prm_id)",
    ],
}


def apply(conn: sqlite3.Connection) -> int:
    """把库升到当前版本，返回升级后的版本号。已经是最新就什么都不做。"""
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current == SCHEMA_VERSION:
        return current
    if current > SCHEMA_VERSION:
        logger.warning("库的版本(%d)比程序(%d)还新，先不动它", current, SCHEMA_VERSION)
        return current

    conn.execute("BEGIN IMMEDIATE")
    try:
        if current == 0:
            # 新库：schema.py 里的建表语句本身就是当前版本，不用再走升级脚本
            _create_all(conn)
        else:
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for statement in _STEPS.get(version, []):
                    conn.execute(statement)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    logger.info("数据库结构 v%d -> v%d", current, SCHEMA_VERSION)
    return SCHEMA_VERSION
