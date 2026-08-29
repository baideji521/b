"""数据库 API：业务层只调这里的函数，不写 SQL。

命名对着实体走：video / analysis / events / task / result / clip / artifact / statistics。
所有写操作内部包在 `db.tx()` 里，一次调用就是一个完整事务。
时间统一存 ISO 字符串（本地时区），方便直接给界面显示。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .db import Database
from .fingerprint import config_hash, fingerprint, full_sha256
from .schema import ANALYSIS_STATES, AUTO_TASK_TYPE, TASK_ACTIVE, TASK_OPEN, TASK_STATES

logger = get_logger(__name__)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _dumps(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


# ====================================================================== 视频
def upsert_video(db: Database, video: str | Path, *, info: dict[str, Any] | None = None,
                 cache_slug: str | None = None, in_library: bool | None = None,
                 status: str | None = None) -> int:
    """登记/更新一个视频，返回 video_id。

    身份靠指纹（大小 + 头中尾 1MB 哈希），所以改名、搬目录都还是同一条记录，
    只是把 file_path / file_name 更新过来。文件读不了就抛 OSError，调用方自己决定跳过还是报错。
    """
    path = Path(video).resolve()
    fp = fingerprint(path)
    stat = path.stat()
    stamp = now()
    meta = info or {}
    with db.tx() as conn:
        row = conn.execute("SELECT id FROM videos WHERE fingerprint = ?", (fp,)).fetchone()
        if row is None:
            # 同一路径以前登记过（比如导入旧缓存时视频还不在盘上，用的是占位指纹），
            # 现在能算真指纹了就把那条补上，不要再开一条新记录
            row = conn.execute("SELECT id FROM videos WHERE file_path = ?",
                               (str(path),)).fetchone()
            if row is not None:
                conn.execute("UPDATE videos SET fingerprint = ? WHERE id = ?", (fp, row["id"]))
        fields = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size": stat.st_size,
            "duration": meta.get("duration"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "cache_slug": cache_slug,
            "exists_on_disk": 1,
            "in_library": None if in_library is None else int(in_library),
            "updated_at": stamp,
        }
        if row is None:
            fields.update({"fingerprint": fp, "status": status or "new", "created_at": stamp})
            columns = ", ".join(fields)
            marks = ", ".join("?" for _ in fields)
            cur = conn.execute(f"INSERT INTO videos({columns}) VALUES({marks})",
                               tuple(fields.values()))
            return int(cur.lastrowid)
        # 已有记录：只覆盖拿得到的值，别用 None 把之前探测到的时长/分辨率抹掉
        sets, params = [], []
        for key, value in fields.items():
            if value is None and key in ("duration", "width", "height", "fps",
                                         "cache_slug", "in_library"):
                continue
            sets.append(f"{key} = ?")
            params.append(value)
        if status:
            sets.append("status = ?")
            params.append(status)
        params.append(row["id"])
        conn.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id = ?", params)
        return int(row["id"])


def upsert_missing_video(db: Database, path: str | Path, *, cache_slug: str | None = None,
                         info: dict[str, Any] | None = None) -> int:
    """登记一个「盘上已经没有」的视频（导入旧缓存时常见：缓存还在，原片被删了）。

    算不出指纹，就用路径派生一个占位指纹；等这个视频哪天又出现在原路径上，
    `upsert_video()` 会把占位指纹换成真指纹，不会多出一条记录。
    """
    target = Path(path)
    fp = f"missing-{config_hash(str(target))}"
    stamp = now()
    meta = info or {}
    with db.tx() as conn:
        row = conn.execute(
            "SELECT id FROM videos WHERE file_path = ? OR fingerprint = ?",
            (str(target), fp)).fetchone()
        if row is not None:
            conn.execute(
                """
                UPDATE videos SET exists_on_disk = 0, cache_slug = COALESCE(?, cache_slug),
                                  duration = COALESCE(?, duration), updated_at = ?
                 WHERE id = ?
                """, (cache_slug, meta.get("duration"), stamp, row["id"]))
            return int(row["id"])
        cur = conn.execute(
            """
            INSERT INTO videos(fingerprint, file_path, file_name, file_size, duration, width,
                               height, fps, cache_slug, exists_on_disk, status,
                               created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'missing', ?, ?)
            """,
            (fp, str(target), target.name, meta.get("size"), meta.get("duration"),
             meta.get("width"), meta.get("height"), meta.get("fps"), cache_slug, stamp, stamp))
        return int(cur.lastrowid)


def get_video(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.one("SELECT * FROM videos WHERE id = ?", (video_id,))


def get_video_by_path(db: Database, video: str | Path) -> sqlite3.Row | None:
    return db.one("SELECT * FROM videos WHERE file_path = ?", (str(Path(video).resolve()),))


def get_video_by_fingerprint(db: Database, fp: str) -> sqlite3.Row | None:
    return db.one("SELECT * FROM videos WHERE fingerprint = ?", (fp,))


def find_video(db: Database, video: str | Path) -> sqlite3.Row | None:
    """先按路径找，找不到再按指纹找（视频被改名/搬走的情况）。都没有返回 None。"""
    hit = get_video_by_path(db, video)
    if hit is not None:
        return hit
    try:
        return get_video_by_fingerprint(db, fingerprint(video))
    except OSError:
        return None


def set_video_status(db: Database, video_id: int, status: str) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE videos SET status = ?, updated_at = ? WHERE id = ?",
                     (status, now(), video_id))


def set_video_presence(db: Database, video_id: int, *, exists: bool,
                       in_library: bool | None = None) -> None:
    """对账用：视频还在不在盘上、在不在视频库里。"""
    with db.tx() as conn:
        conn.execute(
            "UPDATE videos SET exists_on_disk = ?, in_library = ?, updated_at = ? WHERE id = ?",
            (int(exists), None if in_library is None else int(in_library), now(), video_id))


def list_videos(db: Database, *, only_existing: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM videos"
    if only_existing:
        sql += " WHERE exists_on_disk = 1"
    return db.all(sql + " ORDER BY file_name")


# ================================================================== 分析批次
def signature(cfg: Any) -> dict[str, Any]:
    """当前配置下的分析签名：模型 + 配置哈希。缓存命中就靠它比对。"""
    visual = dict(cfg.visual)
    speech = dict(cfg.speech)
    return {
        "vision_model": str(visual.get("model_id") or ""),
        "vision_config": _dumps(visual),
        "vision_config_hash": config_hash(visual),
        "asr_model": str(speech.get("model_size") or ""),
        "asr_config": _dumps(speech),
        "asr_config_hash": config_hash(speech),
    }


def create_analysis(db: Database, video_id: int, sig: dict[str, Any],
                    source: str = "pipeline") -> int:
    """开一条分析记录（running）。同一个视频可以有很多条，历史不覆盖。"""
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO analysis_runs(
                video_id, status, started_at, vision_model, vision_config, vision_config_hash,
                asr_model, asr_config, asr_config_hash, source, created_at)
            VALUES(?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, stamp, sig.get("vision_model"), sig.get("vision_config"),
             sig.get("vision_config_hash"), sig.get("asr_model"), sig.get("asr_config"),
             sig.get("asr_config_hash"), source, stamp))
        return int(cur.lastrowid)


def finish_analysis(db: Database, analysis_id: int, *, scene_count: int | None = None,
                    speech_count: int | None = None, output_dir: str | Path | None = None) -> None:
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE analysis_runs
               SET status = 'completed', finished_at = ?, scene_count = ?, speech_count = ?,
                   output_dir = ?, error = NULL
             WHERE id = ?
            """,
            (now(), scene_count, speech_count,
             str(output_dir) if output_dir else None, analysis_id))


def fail_analysis(db: Database, analysis_id: int, error: str) -> None:
    with db.tx() as conn:
        conn.execute(
            "UPDATE analysis_runs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
            (now(), str(error)[:2000], analysis_id))


def find_cached_analysis(db: Database, video_id: int, sig: dict[str, Any]) -> sqlite3.Row | None:
    """缓存命中判据：同一个视频 + 同样的模型 + 同样的配置哈希 + 跑成功过。"""
    return db.one(
        """
        SELECT * FROM analysis_runs
         WHERE video_id = ? AND status = 'completed'
           AND vision_model = ? AND vision_config_hash = ?
           AND asr_model = ? AND asr_config_hash = ?
         ORDER BY id DESC LIMIT 1
        """,
        (video_id, sig.get("vision_model"), sig.get("vision_config_hash"),
         sig.get("asr_model"), sig.get("asr_config_hash")))


def find_stage_cache(db: Database, video_id: int, sig: dict[str, Any],
                     stage: str) -> sqlite3.Row | None:
    """按阶段查缓存：语音只比 ASR 那两项，视觉只比视觉那两项。

    为什么分开：改了 whisper 的参数不该把跑了几分钟的 Qwen 结果一起作废，反之也一样。
    整体「这次分析算命中」仍然要求五项全对（见 find_cached_analysis）。
    导入的老记录哈希是 'imported'，跟任何真实哈希都不相等，所以永远不会被当成命中。
    """
    if stage == "speech":
        return db.one(
            """
            SELECT * FROM analysis_runs
             WHERE video_id = ? AND status = 'completed'
               AND asr_model = ? AND asr_config_hash = ?
             ORDER BY id DESC LIMIT 1
            """, (video_id, sig.get("asr_model"), sig.get("asr_config_hash")))
    if stage == "visual":
        return db.one(
            """
            SELECT * FROM analysis_runs
             WHERE video_id = ? AND status = 'completed'
               AND vision_model = ? AND vision_config_hash = ?
             ORDER BY id DESC LIMIT 1
            """, (video_id, sig.get("vision_model"), sig.get("vision_config_hash")))
    return None


def stage_cache_ok(db: Database, video_id: int, sig: dict[str, Any], stage: str) -> bool:
    """这个阶段的旧结果还能不能用。probe/timeline 跟模型无关，一律可以复用。"""
    if stage not in ("speech", "visual"):
        return True
    return find_stage_cache(db, video_id, sig, stage) is not None


def analysis_by_source(db: Database, video_id: int, source: str) -> sqlite3.Row | None:
    """按来源找分析记录。导入旧缓存时用它保证「导一次就够」，反复跑不会重复插。"""
    return db.one(
        "SELECT * FROM analysis_runs WHERE video_id = ? AND source = ? ORDER BY id DESC LIMIT 1",
        (video_id, source))


def latest_analysis(db: Database, video_id: int, *, only_completed: bool = True) -> sqlite3.Row | None:
    sql = "SELECT * FROM analysis_runs WHERE video_id = ?"
    if only_completed:
        sql += " AND status = 'completed'"
    return db.one(sql + " ORDER BY id DESC LIMIT 1", (video_id,))


def recover_stale_analyses(db: Database, timeout_minutes: float = 180.0) -> int:
    """把崩溃留下的 running 记录标成 failed，免得永远挂着。返回处理了几条。"""
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                           time.localtime(time.time() - timeout_minutes * 60))
    with db.tx() as conn:
        cur = conn.execute(
            """
            UPDATE analysis_runs
               SET status = 'failed', finished_at = ?, error = '程序异常退出，未跑完'
             WHERE status = 'running' AND COALESCE(started_at, created_at) < ?
            """,
            (now(), cutoff))
        return int(cur.rowcount or 0)


# ================================================================== 分析产物
def save_visual_events(db: Database, analysis_id: int, events: list[dict[str, Any]]) -> int:
    """存视觉事件。重存会先清掉这条分析下的旧事件（同一条分析只该有一套）。"""
    rows = []
    for i, event in enumerate(events, start=1):
        rows.append((
            analysis_id,
            event.get("start"), event.get("end"),
            event.get("description") or event.get("text") or "",
            event.get("type") or event.get("event_type") or "",
            event.get("confidence"),
            event.get("id") or i,
            _dumps(event),
        ))
    with db.tx() as conn:
        conn.execute("DELETE FROM visual_events WHERE analysis_id = ?", (analysis_id,))
        conn.executemany(
            """
            INSERT INTO visual_events(analysis_id, start_time, end_time, description,
                                      event_type, confidence, sequence, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
    return len(rows)


def get_visual_events(db: Database, analysis_id: int) -> list[sqlite3.Row]:
    return db.all("SELECT * FROM visual_events WHERE analysis_id = ? ORDER BY sequence",
                  (analysis_id,))


def save_speech_segments(db: Database, analysis_id: int,
                         segments: list[dict[str, Any]]) -> tuple[int, int]:
    """存语音段 + 逐词时间戳，一个事务里做完。返回（段数，词数）。

    逐词时间戳是精确剪辑的数据源，一个都不能丢；段级原始 JSON 也留一份在 raw_json。
    """
    words_total = 0
    with db.tx() as conn:
        conn.execute("DELETE FROM speech_segments WHERE analysis_id = ?", (analysis_id,))
        for i, seg in enumerate(segments, start=1):
            cur = conn.execute(
                """
                INSERT INTO speech_segments(analysis_id, start_time, end_time, text, speaker,
                                            emotion, confidence, sequence, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (analysis_id, seg.get("start"), seg.get("end"), seg.get("text") or "",
                 seg.get("speaker"), _speech_emotion(seg), seg.get("confidence"),
                 seg.get("id") or i, _dumps(seg)))
            segment_id = int(cur.lastrowid)
            words = seg.get("words") or []
            rows = []
            for index, word in enumerate(words):
                if not isinstance(word, dict):
                    continue
                rows.append((segment_id, analysis_id, index, word.get("word"),
                             word.get("start"), word.get("end"),
                             word.get("probability", word.get("confidence")),
                             word.get("speaker") or seg.get("speaker")))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO speech_words(segment_id, analysis_id, word_index, word,
                                             start_time, end_time, confidence, speaker)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows)
                words_total += len(rows)
    return len(segments), words_total


def _speech_emotion(seg: dict[str, Any]) -> str | None:
    """段里的情绪写法有好几种（emotion 字符串 / emotion.label），统一取标签。"""
    value = seg.get("emotion")
    if isinstance(value, dict):
        return value.get("label") or value.get("emotion") or None
    return str(value) if value else None


def get_speech_segments(db: Database, analysis_id: int) -> list[sqlite3.Row]:
    return db.all("SELECT * FROM speech_segments WHERE analysis_id = ? ORDER BY sequence",
                  (analysis_id,))


def get_speech_words(db: Database, analysis_id: int) -> list[sqlite3.Row]:
    return db.all(
        "SELECT * FROM speech_words WHERE analysis_id = ? ORDER BY start_time, word_index",
        (analysis_id,))


# =================================================================== AI 任务
def prompt_fingerprint(path: str | Path) -> dict[str, Any]:
    """提示词文件的指纹：全文件 sha256（64 位）、绝对路径、字节数。

    只读不写，内容不进库——库里存指纹就够回答「当时用的是哪一版」。
    读不了文件抛 OSError，由调用方决定要不要吞。
    """
    target = Path(path)
    return {"prompt_hash": full_sha256(target),
            "prompt_path": str(target.resolve()),
            "prompt_size": target.stat().st_size}


def note_task_prompt(db: Database, task_id: int, path: str | Path) -> dict[str, Any]:
    """记下这条任务**真正发出去的**那份提示词。返回写进去的三个值。

    必须在 dispatch（真要上传/请求）那一刻调，不能在入队时调：
    一批任务可能排两小时，中间提示词文件被改过，入队时算的指纹就是假的。
    """
    info = prompt_fingerprint(path)
    with db.tx() as conn:
        conn.execute(
            "UPDATE ai_tasks SET prompt_hash = ?, prompt_path = ?, prompt_size = ?, "
            "updated_at = ? WHERE id = ?",
            (info["prompt_hash"], info["prompt_path"], info["prompt_size"], now(), task_id))
    return info


def create_ai_task(db: Database, video_id: int, *, mode: str = "full",
                   provider: str | None = None, model: str | None = None,
                   prompt_version: str | None = None,
                   input_txt: str | Path | None = None,
                   task_type: str = AUTO_TASK_TYPE, priority: int = 100,
                   max_attempts: int = 1) -> int:
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_tasks(video_id, mode, provider, model, status, prompt_version,
                                 input_txt, created_at, task_type, priority, max_attempts,
                                 updated_at)
            VALUES(?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, mode, provider, model, prompt_version,
             str(input_txt) if input_txt else None, stamp, task_type, priority,
             max_attempts, stamp))
        return int(cur.lastrowid)


# ------------------------------------------------------------------ 持久化队列
def open_ai_task(db: Database, video_id: int, *, mode: str,
                 task_type: str = AUTO_TASK_TYPE) -> sqlite3.Row | None:
    """这个视频这一种任务有没有还没跑完的（pending / 跑着一半）。"""
    marks = ", ".join("?" for _ in TASK_OPEN)
    return db.one(
        f"""
        SELECT * FROM ai_tasks
         WHERE video_id = ? AND task_type = ? AND mode = ? AND status IN ({marks})
         ORDER BY id LIMIT 1
        """,
        (video_id, task_type, mode, *TASK_OPEN))


def enqueue_ai_task(db: Database, video_id: int, *, mode: str,
                    task_type: str = AUTO_TASK_TYPE, provider: str | None = None,
                    model: str | None = None, prompt_version: str | None = None,
                    input_txt: str | Path | None = None, priority: int = 100,
                    max_attempts: int = 1) -> tuple[int, bool]:
    """入队，幂等。返回 (task_id, 是不是新建的)。

    同一个视频 + 同一种任务 + 同一种模式已经有没跑完的任务，就复用那一条：
    连点五次「自动剪辑」不会变成五条任务。并发下靠唯一索引兜底（撞了就回头拿现成那条）。
    """
    existing = open_ai_task(db, video_id, mode=mode, task_type=task_type)
    if existing is not None:
        return int(existing["id"]), False
    try:
        task_id = create_ai_task(db, video_id, mode=mode, provider=provider, model=model,
                                 prompt_version=prompt_version, input_txt=input_txt,
                                 task_type=task_type, priority=priority,
                                 max_attempts=max_attempts)
    except sqlite3.IntegrityError:
        # 别的线程/进程刚好插进去了：唯一索引挡住我们，那就用它那条
        existing = open_ai_task(db, video_id, mode=mode, task_type=task_type)
        if existing is None:
            raise
        return int(existing["id"]), False
    return task_id, True


def task_with_video(db: Database, task_id: int) -> sqlite3.Row | None:
    """任务连着视频路径一起取（队列要拿路径去跑）。"""
    return db.one(
        """
        SELECT t.*, v.file_path, v.file_name FROM ai_tasks t
          JOIN videos v ON v.id = t.video_id
         WHERE t.id = ?
        """,
        (task_id,))


# 「等 AI」和「正在剪」是从 TASK_ACTIVE 里切出来的两段，不新增状态、不动 schema：
#   uploading  正在准备/提交给 AI      -> 界面上的「等待 AI」
#   waiting    已提交，等 AI 回话       -> 界面上的「等待 AI」
#   processing AI JSON 到手，正在渲染   -> 界面上的「剪辑中」
TASK_WAITING_AI = ("uploading", "waiting")
TASK_RENDERING = ("processing",)


def claim_next_ai_task(db: Database, *, mode: str | None = None,

                       task_type: str = AUTO_TASK_TYPE, worker_id: str | None = None,
                       status: str = "uploading") -> sqlite3.Row | None:
    """从队列里领一条任务：挑 pending 里优先级最高（数字最小）、最早的那条占下来。

    整个"挑 + 占"在一个 BEGIN IMMEDIATE 事务里，`UPDATE ... WHERE status='pending'`
    的 rowcount 决定谁抢到——两个 worker 同时来，只有一个能把某条从 pending 改走，
    另一个 rowcount=0 就往下一条试。不会出现同一条任务被领两次。

    领到手先落 `uploading`（= 已经归我、正在准备/提交给 AI）。往后由
    `mark_ai_task_waiting`（提交完，等 AI 回话）和 `mark_ai_task_rendering`
    （JSON 到手，真正开剪）推进，所以 `processing` 只表示"正在渲染"。
    调用方要模拟别的阶段可以显式传 `status`。
    """

    sql = "SELECT id FROM ai_tasks WHERE status = 'pending' AND task_type = ?"
    params: list[Any] = [task_type]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    stamp = now()
    claimed: int | None = None
    with db.tx() as conn:
        rows = conn.execute(sql + " ORDER BY priority, id", params).fetchall()
        for row in rows:
            cur = conn.execute(
                """
                UPDATE ai_tasks
                   SET status = ?, worker_id = ?, started_at = COALESCE(started_at, ?),
                       heartbeat_at = ?, updated_at = ?, finished_at = NULL
                 WHERE id = ? AND status = 'pending'
                """,
                (status, worker_id, stamp, stamp, stamp, int(row["id"])))
            if cur.rowcount:
                claimed = int(row["id"])
                break
    if claimed is None:
        return None
    return task_with_video(db, claimed)


def mark_ai_task_waiting(db: Database, task_id: int) -> bool:
    """提交给 AI 之后：uploading -> waiting（东西发出去了，现在等 AI 回话）。

    只认 uploading：completed / failed / cancelled 碰不着，被孤儿恢复退回 pending 的
    也碰不着（状态不能倒退回 active）。已经在 waiting 就返回 False，重复调用无副作用。
    顺手刷心跳——等 AI 可能很久，这条时间戳是恢复逻辑判活的唯一依据。
    """
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            """
            UPDATE ai_tasks SET status = 'waiting', heartbeat_at = ?, updated_at = ?
             WHERE id = ? AND status = 'uploading'
            """,
            (stamp, stamp, task_id))
        return bool(cur.rowcount)


def mark_ai_task_rendering(db: Database, task_id: int) -> bool:
    """真正开剪之前：uploading / waiting -> processing（`processing` = 正在渲染）。

    调用方必须**先**确认手上这份 AI JSON 能直接开剪（解得开、是 dict、抠得出片段），
    否则不许调这里——`processing` 的含义就是"素材齐了，正在出片"。
    脚本剪辑那一串不问 AI，领到手（uploading）就直接进渲染，所以 uploading 也算合法起点。

    `processing` 自己也在允许的旧状态里：历史遗留的 processing 记录（拆分之前写下的，
    分不清当时是在等 AI 还是在渲染）继续按 processing 处理，重复调用也不会失败。
    pending / completed / failed / cancelled 一概不动：状态只能往前走。
    """
    marks = ", ".join("?" for _ in (*TASK_WAITING_AI, *TASK_RENDERING))
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_tasks SET status = 'processing', heartbeat_at = ?, updated_at = ?
             WHERE id = ? AND status IN ({marks})
            """,
            (stamp, stamp, task_id, *TASK_WAITING_AI, *TASK_RENDERING))
        return bool(cur.rowcount)


def fail_or_requeue_ai_task(db: Database, task_id: int, error: str) -> str:

    """任务失败：尝试次数 +1，还没到 max_attempts 就退回 pending，到了就定格 failed。

    默认 max_attempts=1（跟改造前一样：失败就跳过，不自动重跑）。
    崩溃退回（recover_stale_ai_tasks）不算失败、不吃这个额度，否则关一次程序任务就废了。
    """
    row = get_ai_task(db, task_id)
    if row is None:
        return "missing"
    attempts = int(row["retry_count"] or 0) + 1
    limit = max(1, int(row["max_attempts"] or 1))
    status = "pending" if attempts < limit else "failed"
    stamp = now()
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE ai_tasks
               SET status = ?, error = ?, retry_count = ?, updated_at = ?, heartbeat_at = ?,
                   worker_id = NULL,
                   finished_at = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
             WHERE id = ?
            """,
            (status, str(error)[:2000], attempts, stamp, stamp, status, stamp, task_id))
    return status


def cancel_ai_task(db: Database, task_id: int, reason: str | None = None) -> None:
    """单独取消一条任务（人工点停止时手上那条）。状态名只在这一层出现。"""
    stamp = now()
    with db.tx() as conn:
        conn.execute(
            "UPDATE ai_tasks SET status = 'cancelled', finished_at = ?, updated_at = ?, "
            "heartbeat_at = ?, worker_id = NULL, error = ? WHERE id = ?",
            (stamp, stamp, stamp, str(reason)[:2000] if reason else None, task_id))


def cancel_open_ai_tasks(db: Database, *, mode: str | None = None,
                         task_type: str = AUTO_TASK_TYPE, video_id: int | None = None,
                         exclude_id: int | None = None) -> int:
    """手动「停止」用：把还没跑完的任务标 cancelled，不留一堆假的「跑着」。"""
    marks = ", ".join("?" for _ in TASK_OPEN)
    sql = (f"UPDATE ai_tasks SET status = 'cancelled', finished_at = ?, updated_at = ?, "
           f"worker_id = NULL WHERE status IN ({marks}) AND task_type = ?")
    stamp = now()
    params: list[Any] = [stamp, stamp, *TASK_OPEN, task_type]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
    if exclude_id is not None:
        sql += " AND id <> ?"
        params.append(exclude_id)
    with db.tx() as conn:
        return int(conn.execute(sql, params).rowcount or 0)


def queue_counts(db: Database, *, mode: str | None = None,
                 task_type: str = AUTO_TASK_TYPE) -> dict[str, int]:
    """队列里各状态各有多少条。界面显示进度、判断还有没有活都用它。"""
    sql = "SELECT status, COUNT(*) AS n FROM ai_tasks WHERE task_type = ?"
    params: list[Any] = [task_type]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    counts = {state: 0 for state in ("pending", "uploading", "waiting", "processing",
                                     "completed", "failed", "cancelled")}
    for row in db.all(sql + " GROUP BY status", params):
        counts[str(row["status"])] = int(row["n"])
    counts["active"] = sum(counts[s] for s in TASK_ACTIVE)
    counts["open"] = counts["pending"] + counts["active"]
    return counts



def claim_ai_task(db: Database, task_id: int, status: str = "uploading") -> bool:
    """把一条 pending 任务占下来。抢到返回 True，被别人抢了或状态不对返回 False。"""
    with db.tx() as conn:
        stamp = now()
        cur = conn.execute(
            """
            UPDATE ai_tasks SET status = ?, started_at = COALESCE(started_at, ?), heartbeat_at = ?
             WHERE id = ? AND status IN ('pending', 'failed')
            """,
            (status, stamp, stamp, task_id))
        return bool(cur.rowcount)


def update_ai_task(db: Database, task_id: int, status: str | None = None,
                   error: str | None = None) -> None:
    """推进任务状态，同时刷心跳（崩溃恢复靠心跳判超时）。"""
    sets = ["heartbeat_at = ?", "updated_at = ?"]
    params: list[Any] = [now(), now()]
    if status:
        sets.append("status = ?")
        params.append(status)
    if error is not None:
        sets.append("error = ?")
        params.append(str(error)[:2000])
    params.append(task_id)
    with db.tx() as conn:
        conn.execute(f"UPDATE ai_tasks SET {', '.join(sets)} WHERE id = ?", params)


def complete_ai_task(db: Database, task_id: int) -> None:
    with db.tx() as conn:
        conn.execute(
            "UPDATE ai_tasks SET status = 'completed', finished_at = ?, updated_at = ?, "
            "worker_id = NULL, error = NULL WHERE id = ?",
            (now(), now(), task_id))


def fail_ai_task(db: Database, task_id: int, error: str, *, retry: bool = False) -> None:
    """失败。retry=True 就退回 pending 并把重试次数 +1，等下一轮再捞。"""
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE ai_tasks
               SET status = ?, finished_at = ?, error = ?, retry_count = retry_count + 1
             WHERE id = ?
            """,
            ("pending" if retry else "failed", now(), str(error)[:2000], task_id))


def cancel_ai_tasks(db: Database, video_id: int | None = None) -> int:
    """停止时把还没跑完的任务标成 cancelled，别留一堆假的「跑着」。"""
    sql = ("UPDATE ai_tasks SET status = 'cancelled', finished_at = ? "
           "WHERE status IN ('pending', 'uploading', 'waiting', 'processing')")
    params: list[Any] = [now()]
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
    with db.tx() as conn:
        return int(conn.execute(sql, params).rowcount or 0)


def pending_ai_tasks(db: Database, *, mode: str | None = None,
                     task_type: str | None = None,
                     limit: int = 500) -> list[sqlite3.Row]:
    sql = ("SELECT t.*, v.file_path, v.file_name FROM ai_tasks t "
           "JOIN videos v ON v.id = t.video_id WHERE t.status = 'pending'")
    params: list[Any] = []
    if mode:
        sql += " AND t.mode = ?"
        params.append(mode)
    if task_type:
        sql += " AND t.task_type = ?"
        params.append(task_type)
    params.append(limit)
    return db.all(sql + " ORDER BY t.priority, t.id LIMIT ?", params)


def get_ai_task(db: Database, task_id: int) -> sqlite3.Row | None:
    return db.one("SELECT * FROM ai_tasks WHERE id = ?", (task_id,))


def touch_ai_task(db: Database, task_id: int) -> bool:
    """只刷心跳，不动状态。

    长活（Qwen、Whisper、FFmpeg、等 AI 回话）随便跑多久都可能超过
    `ai_task_timeout_minutes`；只要还在定期刷这个时间戳，恢复逻辑就不会把它当死任务。
    只对还在跑的状态生效——已经 completed/failed 的别被心跳弄活。
    """
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            f"UPDATE ai_tasks SET heartbeat_at = ?, updated_at = ? "
            f"WHERE id = ? AND status IN ({marks})",
            (stamp, stamp, task_id, *TASK_ACTIVE))
        return bool(cur.rowcount)


def recover_stale_ai_tasks(db: Database, timeout_minutes: float = 30.0) -> int:
    """捞回卡死的任务：uploading/waiting/processing 且心跳超时的，退回 pending。

    没有这一步，程序在 processing 状态被强杀之后那条任务会永远卡着。
    """
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                           time.localtime(time.time() - timeout_minutes * 60))
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_tasks
               SET status = 'pending', error = '上次异常退出，已退回等待',
                   worker_id = NULL, updated_at = ?, finished_at = NULL
             WHERE status IN ({marks})
               AND COALESCE(heartbeat_at, started_at, created_at) < ?
            """,
            (stamp, *TASK_ACTIVE, cutoff))
        return int(cur.rowcount or 0)


def release_ai_task(db: Database, task_id: int, reason: str) -> bool:
    """把一条还在跑的任务放回队列（正常关程序时用）。退回去了返回 True。

    只对 active 状态生效：completed/failed/cancelled 碰不着，不会被"弄活"。
    这不是失败，所以 `retry_count` 一动不动，也不消耗 `max_attempts`。
    心跳不刷——pending 不看心跳，刷了反而模糊了"上次跑到哪"这个事实。
    """
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_tasks
               SET status = 'pending', error = ?, worker_id = NULL,
                   updated_at = ?, finished_at = NULL
             WHERE id = ? AND status IN ({marks})
            """,
            (str(reason)[:2000], stamp, task_id, *TASK_ACTIVE))
        return bool(cur.rowcount)


def recover_orphaned_ai_tasks(db: Database, before: str) -> int:
    """本次进程启动之前就挂在 active 的任务，全部退回 pending。返回退回了几条。

    调用方必须先拿到库目录里的运行时独占锁（`lock.RuntimeLock`）——锁在手上才能断定
    "没有别的实例在跑"，那些还挂着 active 的任务只可能是上一次没退干净留下的。
    判据是"最后一次有动静的时间早于 `before`（本次进程启动时刻）"，所以本进程自己
    刚领的任务不会被误退。

    不限 mode、不限 task_type：唯一索引 idx_tasks_open_unique 把所有 active 都算 open，
    只放 full 会让 collect/script 的孤儿继续挡着重新入队。
    跟崩溃退回一样，这不算失败：`retry_count` 不加，`max_attempts` 不消耗。
    只改 ai_tasks 这几列，不碰 ai_results / clips / artifacts。
    """
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_tasks
               SET status = 'pending', error = '上次没有正常退出，已退回等待',
                   worker_id = NULL, updated_at = ?, finished_at = NULL
             WHERE status IN ({marks})
               AND COALESCE(heartbeat_at, started_at, created_at) < ?
            """,
            (stamp, *TASK_ACTIVE, str(before)))
        return int(cur.rowcount or 0)


# =================================================================== AI 结果
def save_ai_result(db: Database, video_id: int, *, task_id: int | None = None,
                   raw_response: str | None = None, json_data: Any = None,
                   candidate_count: int | None = None, winner_score: float | None = None,
                   validated: bool = False, validation_error: str | None = None,
                   prompt_hash: str | None = None, prompt_path: str | None = None,
                   prompt_size: int | None = None) -> int:
    """存 AI 结果。raw_response 一定要存原文，以后要追溯 AI 当时到底回了什么。

    prompt_* 三个是本次实际发出去的提示词指纹：自动任务从任务那儿带过来，
    手工单发没有任务行，就直接记在这儿，两条路都能回答「这份 JSON 用的哪版提示词」。
    """
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_results(task_id, video_id, raw_response, json_data, candidate_count,
                                   winner_score, validated, validation_error, created_at,
                                   prompt_hash, prompt_path, prompt_size)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, video_id, raw_response, _dumps(json_data), candidate_count,
             winner_score, int(validated), validation_error, now(),
             prompt_hash, prompt_path, prompt_size))
        return int(cur.lastrowid)


def get_ai_result(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.one("SELECT * FROM ai_results WHERE video_id = ? ORDER BY id DESC LIMIT 1",
                  (video_id,))


def ai_result_for_task(db: Database, task_id: int) -> sqlite3.Row | None:
    """**这一条任务**最新的那份 AI 结果。只读。

    崩溃续跑要用它回答「这条任务是不是已经问过 AI 了」，所以只认 task_id 对得上的：
    按 video_id 取（get_ai_result）会捞到同一个视频以前那些任务的结果，
    拿旧结果去剪新任务是错的。同一条任务重发过几次就取 id 最大的那份。
    """
    return db.one("SELECT * FROM ai_results WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                  (task_id,))


# ===================================================================== 片段
def clips_from_payload(payload: Any) -> list[dict[str, Any]]:
    """从 AI JSON 里抠出片段。字段原样取，不做任何推算或改写。

    兼容两种写法：根上直接是 clip 对象，或者 {"clip": {...}}；
    clip.duration 没给就用 end - start 补一个，别的字段一律照抄。
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    if not isinstance(clip, dict) or clip.get("start") is None or clip.get("end") is None:
        return []
    try:
        start = float(clip["start"])
        end = float(clip["end"])
    except (TypeError, ValueError):
        return []
    duration = clip.get("duration")
    try:
        duration = float(duration) if duration is not None else round(end - start, 3)
    except (TypeError, ValueError):
        duration = round(end - start, 3)
    evaluation = clip.get("evaluation") or payload.get("evaluation")
    if not evaluation and isinstance(clip.get("overlays"), dict):
        # 现行提示词把中文评价放在 clip.overlays.evaluation 里，别漏掉
        evaluation = clip["overlays"].get("evaluation")
    return [{
        "start": start,
        "end": end,
        "duration": duration,
        "score": _as_float(clip.get("score", payload.get("score"))),
        "type": clip.get("type") or payload.get("type") or "",
        "reason": clip.get("reason") or payload.get("reason") or "",
        "evaluation": _dumps(evaluation) if not isinstance(evaluation, str) else evaluation,
    }]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_clip(db: Database, video_id: int, spec: dict[str, Any], *,
                ai_result_id: int | None = None, status: str = "planned",
                output_path: str | Path | None = None) -> int:
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO clips(video_id, ai_result_id, start_time, end_time, duration, score,
                              clip_type, reason, evaluation, status, output_path, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, ai_result_id, spec.get("start"), spec.get("end"), spec.get("duration"),
             spec.get("score"), spec.get("type"), spec.get("reason"), spec.get("evaluation"),
             status, str(output_path) if output_path else None, now()))
        return int(cur.lastrowid)


def update_clip(db: Database, clip_id: int, *, status: str | None = None,
                output_path: str | Path | None = None,
                start: float | None = None, end: float | None = None,
                duration: float | None = None) -> None:
    """改片段行。start/end/duration 用来把**实际剪出来的**区间回写。

    AI 给的区间和真正剪的区间可能不一样（剪辑引擎会把边界挪到语义位置上），
    库里得留实际值，不然以后按 clips 复盘会和成片对不上。
    """
    sets, params = [], []
    if status:
        sets.append("status = ?")
        params.append(status)
    if output_path is not None:
        sets.append("output_path = ?")
        params.append(str(output_path))
    for column, value in (("start_time", start), ("end_time", end), ("duration", duration)):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(float(value))
    if not sets:
        return
    params.append(clip_id)
    with db.tx() as conn:
        conn.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id = ?", params)


def get_clips(db: Database, video_id: int) -> list[sqlite3.Row]:
    return db.all("SELECT * FROM clips WHERE video_id = ? ORDER BY id", (video_id,))


# ===================================================================== 文件
def register_artifact(db: Database, video_id: int, kind: str, path: str | Path, *,
                      sha256: str | None = None) -> int:
    """登记一个实际文件（存在与否、多大）。同一 video+type+path 重复登记就更新。"""
    target = Path(path)
    exists = target.is_file()
    size = target.stat().st_size if exists else None
    stamp = now()
    with db.tx() as conn:
        conn.execute(
            """
            INSERT INTO artifacts(video_id, type, path, size, sha256, exists_on_disk,
                                  created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, type, path) DO UPDATE SET
                size = excluded.size,
                sha256 = COALESCE(excluded.sha256, artifacts.sha256),
                exists_on_disk = excluded.exists_on_disk,
                updated_at = excluded.updated_at
            """,
            (video_id, kind, str(target), size, sha256, int(exists), stamp, stamp))
        row = conn.execute(
            "SELECT id FROM artifacts WHERE video_id = ? AND type = ? AND path = ?",
            (video_id, kind, str(target))).fetchone()
        return int(row["id"])


def update_artifact(db: Database, artifact_id: int, *, exists: bool | None = None,
                    size: int | None = None, sha256: str | None = None) -> None:
    sets, params = ["updated_at = ?"], [now()]
    if exists is not None:
        sets.append("exists_on_disk = ?")
        params.append(int(exists))
    if size is not None:
        sets.append("size = ?")
        params.append(size)
    if sha256 is not None:
        sets.append("sha256 = ?")
        params.append(sha256)
    params.append(artifact_id)
    with db.tx() as conn:
        conn.execute(f"UPDATE artifacts SET {', '.join(sets)} WHERE id = ?", params)


def get_artifacts(db: Database, video_id: int, kind: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM artifacts WHERE video_id = ?"
    params: list[Any] = [video_id]
    if kind:
        sql += " AND type = ?"
        params.append(kind)
    return db.all(sql + " ORDER BY type, id", params)


def artifact_path(db: Database, video_id: int, kind: str) -> Path | None:
    """某种产物的路径（只认还在盘上的那条）。没有就 None。

    同一秒内登记的两条 updated_at 会一模一样（now() 只到秒），只按它排序时选谁全看
    SQLite 心情。补一个 id DESC 兜底：同时间就认后登记的那条，结果才是确定的。
    """
    row = db.one(
        """
        SELECT path FROM artifacts
         WHERE video_id = ? AND type = ? AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
         ORDER BY updated_at DESC, id DESC LIMIT 1
        """, (video_id, kind))
    return Path(row["path"]) if row else None


def has_artifact(db: Database, video_id: int, kind: str) -> bool:
    return artifact_path(db, video_id, kind) is not None


# ===================================================================== 统计
def _folder_like(folder: str | Path) -> str:
    """把目录变成 LIKE 用的前缀。带上分隔符，免得 AI_输入 顺手把 AI_输入_old 也捞进来。"""
    prefix = str(Path(folder).resolve())
    if not prefix.endswith(os.sep):
        prefix += os.sep
    # 文件名里可能真带 % 或 _，得转义，否则 LIKE 会把它们当通配符
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def videos_under(db: Database, folder: str | Path | None) -> list[sqlite3.Row]:
    """某个目录（含子目录）里、还在盘上的视频。界面列任务表就用这个，不去翻目录。"""
    if folder is None:
        return []
    return db.all(
        "SELECT * FROM videos WHERE exists_on_disk = 1 AND file_path LIKE ? ESCAPE '\\' "
        "ORDER BY file_name",
        (_folder_like(folder),))


def missing_input_videos(db: Database, folder: str | Path | None) -> list[sqlite3.Row]:
    """登记过、但现在盘上找不着的输入视频（同一个目录口径）。

    和 `videos_under` 正好互补：那边是 exists_on_disk = 1，这边是 0。文件被挪走或删掉时，
    视频不会从统计总数里凭空少掉——它只是不在 `videos_under` 的范围里了，这里能查出来。
    """
    if folder is None:
        return []
    return db.all(
        "SELECT * FROM videos WHERE exists_on_disk = 0 AND file_path LIKE ? ESCAPE '\\' "
        "ORDER BY file_name",
        (_folder_like(folder),))



def states_for_videos(db: Database, video_ids: list[int],
                      sig: dict[str, Any] | None = None) -> dict[int, dict[str, bool]]:
    """一批视频的四个状态，四条聚合 SQL 出来，不按视频逐个查、更不扫磁盘。

    界面刷新就靠这个：40 个视频也是四次查询，不会因为数据库化反而变慢。
    四个状态各自独立：
    - analysed  analysis_runs 里有 completed（给了 sig 还要模型/配置哈希对得上）
    - txt       artifacts 里有还在盘上的 merged_txt
    - json      ai_results 有记录，或 artifacts 里有 ai_script
    - clipped   artifacts 里有还在盘上的 final_video（clips 的 rendered 只算历史）
    """
    empty = {"analysed": False, "txt": False, "json": False, "clipped": False}
    if not video_ids:
        return {}
    marks = ", ".join("?" for _ in video_ids)
    params = list(video_ids)
    out: dict[int, dict[str, bool]] = {vid: dict(empty) for vid in video_ids}

    if sig is None:
        rows = db.all(
            f"SELECT DISTINCT video_id FROM analysis_runs "
            f"WHERE status = 'completed' AND video_id IN ({marks})", params)
    else:
        rows = db.all(
            f"""
            SELECT DISTINCT video_id FROM analysis_runs
             WHERE status = 'completed' AND video_id IN ({marks})
               AND vision_model = ? AND vision_config_hash = ?
               AND asr_model = ? AND asr_config_hash = ?
            """,
            [*params, sig.get("vision_model"), sig.get("vision_config_hash"),
             sig.get("asr_model"), sig.get("asr_config_hash")])
    for row in rows:
        out[int(row["video_id"])]["analysed"] = True

    for row in db.all(
            f"""
            SELECT video_id, type FROM artifacts
             WHERE video_id IN ({marks}) AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
               AND type IN ('merged_txt', 'ai_script', 'final_video')
            """, params):
        kind = row["type"]
        key = {"merged_txt": "txt", "ai_script": "json", "final_video": "clipped"}[kind]
        out[int(row["video_id"])][key] = True

    for row in db.all(
            f"SELECT DISTINCT video_id FROM ai_results WHERE video_id IN ({marks})", params):
        out[int(row["video_id"])]["json"] = True

    # 「成品」只认还在盘上的 final_video 产物：clips.status='rendered' 是历史（当时确实剪出来了），
    # 成品后来被删掉了就不该继续显示完成——这跟队列跳过的判断（artifacts）保持同一个口径
    return out


def statistics_for(db: Database, video_ids: list[int], *,
                   done_key: str = "clipped",
                   sig: dict[str, Any] | None = None) -> dict[str, int]:
    """【遗留】旧四格统计：总任务 / 未剪辑（未完成）/ 已获取 JSON / 成品。

    自动剪辑总览已经统一走 `video_queue_statistics`（按视频互斥分桶），这里的 "todo"
    是 `总数 - 完成` 的减法口径，一个视频可能同时算进好几个含义，不要再用于新的展示。
    留着是因为口径和 `states_for_videos` 绑在一起，历史行为还有测试在盯。

    `done_key` 决定"算完成"的口径：剪辑成片和脚本剪辑看成品（clipped），
    收取脚本只要拿到 JSON 就算完成（json）。
    """
    states = states_for_videos(db, video_ids, sig)
    done = sum(1 for s in states.values() if s.get(done_key))
    return {
        "total": len(video_ids),
        "todo": len(video_ids) - done,
        "json": sum(1 for s in states.values() if s["json"]),
        "done": done,
    }


def artifact_videos(db: Database, video_ids: list[int], kind: str) -> set[int]:
    """这些视频里，哪些有一份**还在盘上且非空**的该类产物。一条 SQL，按视频去重。

    口径跟 `artifact_path` / `states_for_videos` 完全一致（exists_on_disk = 1 且 size > 0），
    所以同一个视频有好几条历史 artifacts（改过名、换过输出目录）也只算一次。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    return {int(row["video_id"]) for row in db.all(
        f"""
        SELECT DISTINCT video_id FROM artifacts
         WHERE video_id IN ({marks}) AND type = ?
           AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
        """, [*video_ids, kind])}


def reusable_json_videos(db: Database, video_ids: list[int]) -> set[int]:
    """这些视频里，哪些手上有一份**能直接开剪**的 AI JSON。按视频去重。

    判据跟崩溃续跑（`main_window._resume_existing_ai_json`）逐条一致：
    json_data 解得开 → 是 dict → `clips_from_payload` 至少抠出一个片段。
    故意不看 `validated` / `candidate_count`：那两个是写入当时的自我声明，
    真正决定「这份结果能不能拿去剪」的是 json_data 本身，两边必须同一个口径，
    否则界面说「已获取 JSON」而队列又去重新问一遍 AI。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    ok: set[int] = set()
    for row in db.all(
            f"""
            SELECT video_id, json_data FROM ai_results
             WHERE video_id IN ({marks}) AND json_data IS NOT NULL AND json_data <> ''
             ORDER BY id DESC
            """, list(video_ids)):
        vid = int(row["video_id"])
        if vid in ok:
            continue
        try:
            parsed = json.loads(str(row["json_data"]))
        except (TypeError, ValueError):
            continue  # 存着的是坏 JSON，跟没有一样
        if isinstance(parsed, dict) and clips_from_payload(parsed):
            ok.add(vid)
    return ok


def task_videos(db: Database, video_ids: list[int], states: tuple[str, ...], *,
                mode: str | None = None, task_type: str = AUTO_TASK_TYPE) -> set[int]:
    """这些视频里，哪些有处于给定状态的任务。按视频去重（一个视频有几条历史任务都只算一次）。"""
    if not video_ids or not states:
        return set()
    vmarks = ", ".join("?" for _ in video_ids)
    smarks = ", ".join("?" for _ in states)
    sql = (f"SELECT DISTINCT video_id FROM ai_tasks WHERE task_type = ? "
           f"AND video_id IN ({vmarks}) AND status IN ({smarks})")
    params: list[Any] = [task_type, *video_ids, *states]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    return {int(row["video_id"]) for row in db.all(sql, params)}


def video_queue_statistics(db: Database, video_ids: list[int], *,
                           mode: str | None = None, done_key: str = "clipped",
                           task_type: str = AUTO_TASK_TYPE) -> dict[str, int]:
    """自动剪辑总览：**按视频**统计，每个视频只落进一个桶。

    输入就是界面那份视频集合（`videos_under` 给的，已经限定"还在盘上 + 在 AI_输入目录里"），
    所有数字都从这同一个集合推导，不会各自用不同范围。

    六组集合运算，不写巨型 JOIN：
      done       有效成品（done_key='json' 的收取脚本口径下改成"有可复用 JSON"）
      json_ok    有可复用 AI JSON
      rendering  有 processing 的任务（JSON 到手，正在出片）
      waiting_ai 有 uploading / waiting 的任务（正在提交 / 等 AI 回话）
      failed     有 failed 的任务
      cancelled  有 cancelled 的任务

    互斥优先级（成品优先，历史任务状态不能盖过实际产物；同一视频两条活任务时取走得更远的）：
      有成品                 -> done
      否则在渲染             -> rendering（界面：剪辑中）
      否则在等 AI            -> waiting_ai（界面：等待 AI）
      否则有可复用 JSON      -> failed / cancelled / pending_render
      否则                   -> no_json

    所以 done + rendering + waiting_ai + pending_render + failed + cancelled + no_json
    == total 恒成立。`json` 是一个横切指标（可能和任何桶重叠），单独给出来方便界面显示。
    """
    ids = {int(v) for v in video_ids}
    ordered = sorted(ids)
    json_ok = reusable_json_videos(db, ordered) & ids
    done = (json_ok if done_key == "json"
            else artifact_videos(db, ordered, "final_video") & ids)
    # 旧成品还在就算完成，别让重排的任务把它抢走
    rendering = (task_videos(db, ordered, TASK_RENDERING,
                             mode=mode, task_type=task_type) & ids) - done
    waiting_ai = (task_videos(db, ordered, TASK_WAITING_AI, mode=mode, task_type=task_type)
                  & ids) - done - rendering
    failed_ids = task_videos(db, ordered, ("failed",), mode=mode, task_type=task_type)
    cancelled_ids = task_videos(db, ordered, ("cancelled",), mode=mode, task_type=task_type)

    pending_render = failed = cancelled = no_json = 0
    for vid in ids - done - rendering - waiting_ai:
        if vid not in json_ok:
            no_json += 1          # 基础数据都还没到位，历史上失败过也不归到"失败"
        elif vid in failed_ids:
            failed += 1
        elif vid in cancelled_ids:
            cancelled += 1
        else:
            pending_render += 1
    return {
        "total": len(ids),
        "json": len(json_ok),
        "no_json": no_json,
        "pending_render": pending_render,
        "waiting_ai": len(waiting_ai),
        "rendering": len(rendering),

        "done": len(done),
        "failed": failed,
        "cancelled": cancelled,
    }


def video_state(db: Database, video_id: int, sig: dict[str, Any] | None = None) -> dict[str, bool]:
    """一个视频的四个状态，各自独立判定，全部来自数据库。

    - analysed：有跑成功的分析（给了 sig 就还要模型/配置对得上）
    - txt：merged_txt 产物在
    - json：有 AI 结果，或者登记过 ai_script 文件
    - clipped：有还在盘上的 final_video 产物（clips 里的 rendered 只是历史，
      成品被删了就不该继续算完成——跟队列跳过的判断同一个口径）
    """
    if sig is None:
        analysed = latest_analysis(db, video_id) is not None
    else:
        analysed = find_cached_analysis(db, video_id, sig) is not None
    txt = has_artifact(db, video_id, "merged_txt")
    json_ok = (get_ai_result(db, video_id) is not None
               or has_artifact(db, video_id, "ai_script"))
    clipped = has_artifact(db, video_id, "final_video")
    return {"analysed": analysed, "txt": txt, "json": json_ok, "clipped": clipped}


def get_statistics(db: Database, video_ids: list[int] | None = None) -> dict[str, int]:
    """【遗留】整库四格：总任务 / 未剪辑 / 已获取 JSON / 成品。一条 SQL 聚合出来。

    口径是"整库所有视频"，而且 json 只要有 ai_results 行就算（不看能不能开剪），
    跟自动剪辑总览（`video_queue_statistics`）不是一回事。新的展示一律用后者，
    这里只服务于"看一眼整库大概多少东西"。
    """
    scope = ""
    params: list[Any] = []
    if video_ids is not None:
        if not video_ids:
            return {"total": 0, "todo": 0, "json": 0, "done": 0}
        marks = ", ".join("?" for _ in video_ids)
        scope = f" WHERE v.id IN ({marks})"
        params = list(video_ids)
    row = db.one(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN done.n > 0 THEN 1 ELSE 0 END)  AS done,
            SUM(CASE WHEN js.n  > 0 THEN 1 ELSE 0 END)   AS js
        FROM videos v
        LEFT JOIN (
            SELECT video_id, COUNT(*) AS n FROM artifacts
             WHERE type = 'final_video' AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
             GROUP BY video_id
        ) done ON done.video_id = v.id
        LEFT JOIN (
            SELECT video_id, COUNT(*) AS n FROM (
                SELECT video_id FROM ai_results
                UNION ALL
                SELECT video_id FROM artifacts
                 WHERE type = 'ai_script' AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
            ) GROUP BY video_id
        ) js ON js.video_id = v.id
        {scope}
        """, params)
    total = int(row["total"] or 0)
    done = int(row["done"] or 0)
    return {"total": total, "todo": total - done, "json": int(row["js"] or 0), "done": done}


def counts(db: Database) -> dict[str, int]:
    """整库体检数字，给 CLI 和缓存管理看。"""
    tables = ("videos", "analysis_runs", "visual_events", "speech_segments", "speech_words",
              "ai_tasks", "ai_results", "clips", "artifacts")
    return {name: int(db.value(f"SELECT COUNT(*) FROM {name}", default=0)) for name in tables}


def full_stats(db: Database) -> dict[str, Any]:
    """整库现状，全部来自 SQL——不扫目录数文件，业务数字只认库。"""
    def _by_status(table: str, states: tuple[str, ...]) -> dict[str, int]:
        out = {state: 0 for state in states}
        for row in db.all(f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status"):
            out[str(row["status"])] = int(row["n"])
        return out

    videos_total = int(db.value("SELECT COUNT(*) FROM videos", default=0))
    videos_here = int(db.value("SELECT COUNT(*) FROM videos WHERE exists_on_disk = 1", default=0))
    artifacts_total = int(db.value("SELECT COUNT(*) FROM artifacts", default=0))
    artifacts_here = int(db.value("SELECT COUNT(*) FROM artifacts WHERE exists_on_disk = 1",
                                  default=0))
    clips = _by_status("clips", ("planned", "rendered", "failed"))
    return {
        "schema_version": int(db.value("PRAGMA user_version", default=0) or 0),
        "videos": {"total": videos_total, "on_disk": videos_here,
                   "missing": videos_total - videos_here},
        "analysis": _by_status("analysis_runs", ANALYSIS_STATES),
        "tasks": _by_status("ai_tasks", TASK_STATES),
        "ai_results": int(db.value("SELECT COUNT(*) FROM ai_results", default=0)),
        "clips": {**clips, "total": int(db.value("SELECT COUNT(*) FROM clips", default=0))},
        "artifacts": {"total": artifacts_total, "on_disk": artifacts_here,
                      "missing": artifacts_total - artifacts_here},
        "artifacts_by_type": {str(r["type"]): int(r["n"]) for r in db.all(
            "SELECT type, COUNT(*) AS n FROM artifacts GROUP BY type ORDER BY type")},
        "speech_words": int(db.value("SELECT COUNT(*) FROM speech_words", default=0)),
        "speech_segments": int(db.value("SELECT COUNT(*) FROM speech_segments", default=0)),
        "visual_events": int(db.value("SELECT COUNT(*) FROM visual_events", default=0)),
    }


def cache_stats(db: Database) -> dict[str, Any]:
    """分析次数与模型分布。

    说明清楚口径：库里记的是「真的跑过一次分析」（每次一条 analysis_runs），
    **命中缓存那次什么都不写**，所以没法从库里数出准确的命中次数。
    这里只给能确凿算出来的：跑过多少次、其中多少次是同一视频同一套配置又跑了一遍
    （`reruns`，说明那次没吃到缓存），以及按模型的分布。
    """
    total = int(db.value("SELECT COUNT(*) FROM analysis_runs", default=0))
    completed = int(db.value("SELECT COUNT(*) FROM analysis_runs WHERE status = 'completed'",
                             default=0))
    combos = int(db.value(
        """
        SELECT COUNT(*) FROM (
            SELECT video_id, vision_model, vision_config_hash, asr_model, asr_config_hash
              FROM analysis_runs WHERE status = 'completed'
             GROUP BY video_id, vision_model, vision_config_hash, asr_model, asr_config_hash)
        """, default=0))
    return {
        "runs_total": total,
        "runs_completed": completed,
        "runs_failed": int(db.value("SELECT COUNT(*) FROM analysis_runs WHERE status = 'failed'",
                                    default=0)),
        "runs_running": int(db.value("SELECT COUNT(*) FROM analysis_runs WHERE status = 'running'",
                                     default=0)),
        "distinct_configs": combos,
        "reruns": max(completed - combos, 0),
        "by_vision_model": {str(r["m"] or "(没记)"): int(r["n"]) for r in db.all(
            "SELECT vision_model AS m, COUNT(*) AS n FROM analysis_runs "
            "GROUP BY vision_model ORDER BY n DESC")},
        "by_asr_model": {str(r["m"] or "(没记)"): int(r["n"]) for r in db.all(
            "SELECT asr_model AS m, COUNT(*) AS n FROM analysis_runs "
            "GROUP BY asr_model ORDER BY n DESC")},
        "hit_events_recorded": False,
    }
