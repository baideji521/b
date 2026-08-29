"""SQLite 统一管理缓存与任务状态。

对外就这几件事：
    from .db import open_db, db_path      # 拿库
    from . import repo                    # 查/写（业务层只用 repo，不写 SQL）
    from .importer import import_all, reconcile   # 旧缓存导入 / 文件与库对账

设计前提：**文件是文件，数据库是状态。** 分析结果、AI 回复、成品照旧落盘，
库里记的是"有什么、属于哪个视频、走到哪一步"，别处不用再扫目录猜。
"""

from __future__ import annotations

from .db import DB_NAME, Database, db_path, open_db
from .fingerprint import config_hash, fingerprint, full_sha256
from .schema import ARTIFACT_TYPES, SCHEMA_VERSION, TASK_STATES

__all__ = [
    "ARTIFACT_TYPES",
    "DB_NAME",
    "SCHEMA_VERSION",
    "TASK_STATES",
    "Database",
    "config_hash",
    "db_path",
    "fingerprint",
    "full_sha256",
    "open_db",
]
