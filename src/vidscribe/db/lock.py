"""数据库目录里的运行时独占锁：用来回答「现在还有别的实例在用这个库吗」。

判据只有一条：能不能拿到 `<库文件所在目录>/queue.lock` 的**独占锁**。

- 不看文件在不在：上次崩溃留下的空文件不是锁，锁是"句柄 + 锁区间"，进程死了
  操作系统自己就放开了（断电、任务管理器结束进程都一样）。
- 不看 PID：Windows 上 PID 会被复用，新进程完全可能撞上上次那个号，
  拿 PID 判断"上次那个我"是错的。
- 拿不到锁不是错误，是"现在不确定有没有别人在跑"，调用方必须按保守路线走。

锁路径由**真实库文件路径**派生（`Database.path`），不从 cache 目录、
AI 目录或当前工作目录推——两个配置指向同一个库时，必须争同一把锁。
"""

from __future__ import annotations

import os
from pathlib import Path

from ..logging_setup import get_logger
from .db import Database

logger = get_logger(__name__)

LOCK_NAME = "queue.lock"

if os.name == "nt":  # pragma: no cover - 平台分支
    import msvcrt

    def _try_lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:  # pragma: no cover - 平台分支
    import fcntl

    def _try_lock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def queue_lock_path(db: Database) -> Path:
    """这个库对应的锁文件：库文件旁边的 queue.lock。"""
    return Path(db.path).resolve().parent / LOCK_NAME


class RuntimeLock:
    """一把进程级独占锁。拿到就一直握着，直到 release() 或进程结束。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """拿锁。拿到返回 True；被别人占着、目录不可写、网络盘锁语义异常都返回 False。"""
        if self._handle is not None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(self.path, "a+b")  # noqa: SIM115 - 句柄要活到 release()
        except OSError as exc:
            logger.info("运行时锁打不开（%s），按「可能还有别的实例」处理：%s", self.path, exc)
            return False
        try:
            _try_lock(handle)
        except OSError as exc:
            handle.close()
            logger.info("运行时锁被占着（%s）：%s", self.path, exc)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        """放锁。异常退出时不用管，操作系统会替我们放。"""
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            _unlock(handle)
        except OSError:  # noqa: S110 - 放不掉也只能交给操作系统
            pass
        try:
            handle.close()
        except OSError:  # noqa: S110
            pass
