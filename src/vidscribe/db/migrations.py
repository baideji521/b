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
_STEPS: dict[int, list[str]] = {}


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
            _create_all(conn)
        for version in range(max(current, 1) + 1, SCHEMA_VERSION + 1):
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
