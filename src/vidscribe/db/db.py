"""SQLite 连接管理：全项目只从这儿拿连接，业务代码不自己 sqlite3.connect。

为什么要集中：GUI 主线程、分析线程、AI 线程、Bridge 都会读写库。
sqlite3 的连接不能跨线程用，所以这里按线程各给一条连接，
再统一开 WAL（读写不互相挡）、`synchronous=NORMAL`（够安全、快很多）、外键约束。

写操作一律走 `with db.tx():`——短事务、要么全成要么全滚，
不要在事务里干耗时的活（跑模型、发网络请求），否则会长时间占着写锁。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from . import migrations

logger = get_logger(__name__)

DB_NAME = "video.db"
_INSTANCES: dict[str, "Database"] = {}
_INSTANCES_LOCK = threading.Lock()


class Database:
    """一个库文件对应一个实例；每个线程一条连接。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()  # 只保护本进程内的写，跨进程靠 WAL + busy_timeout
        with self.connect() as conn:  # 建表 / 升级，只在第一条连接上做一次
            migrations.apply(conn)

    # ------------------------------------------------------------ 连接
    def connect(self) -> sqlite3.Connection:
        """当前线程的连接，没有就建一条。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """关掉当前线程那条连接（线程结束前调，或者程序退出时）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------ 事务
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """一次写事务。中途抛异常就整段回滚，不会留半套数据。"""
        conn = self.connect()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # ------------------------------------------------------------ 查询
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.connect().executemany(sql, rows)

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def value(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.one(sql, params)
        return default if row is None else row[0]


def db_path(cfg: Any) -> Path:
    """库文件位置：`paths.db_dir` 下的 video.db（默认 <项目>/database/video.db）。"""
    raw = str(cfg.data["paths"].get("db_dir") or "database").strip() or "database"
    root = Path(raw)
    if not root.is_absolute():
        root = Path(cfg.root) / root
    return root / DB_NAME


def open_db(cfg: Any) -> Database:
    """拿数据库句柄。同一个库文件全进程共用一个实例（内部按线程分连接）。"""
    path = str(db_path(cfg).resolve())
    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(path)
        if inst is None:
            inst = Database(path)
            _INSTANCES[path] = inst
            logger.info("数据库就绪：%s", path)
        return inst
