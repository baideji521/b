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
from .fingerprint import config_hash, fingerprint
from .schema import TASK_ACTIVE

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
def create_ai_task(db: Database, video_id: int, *, mode: str = "full",
                   provider: str | None = None, model: str | None = None,
                   prompt_version: str | None = None,
                   input_txt: str | Path | None = None) -> int:
    stamp = now()
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_tasks(video_id, mode, provider, model, status, prompt_version,
                                 input_txt, created_at)
            VALUES(?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (video_id, mode, provider, model, prompt_version,
             str(input_txt) if input_txt else None, stamp))
        return int(cur.lastrowid)


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
    sets = ["heartbeat_at = ?"]
    params: list[Any] = [now()]
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
            "UPDATE ai_tasks SET status = 'completed', finished_at = ?, error = NULL WHERE id = ?",
            (now(), task_id))


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
                     limit: int = 500) -> list[sqlite3.Row]:
    sql = ("SELECT t.*, v.file_path, v.file_name FROM ai_tasks t "
           "JOIN videos v ON v.id = t.video_id WHERE t.status = 'pending'")
    params: list[Any] = []
    if mode:
        sql += " AND t.mode = ?"
        params.append(mode)
    params.append(limit)
    return db.all(sql + " ORDER BY t.id LIMIT ?", params)


def get_ai_task(db: Database, task_id: int) -> sqlite3.Row | None:
    return db.one("SELECT * FROM ai_tasks WHERE id = ?", (task_id,))


def recover_stale_ai_tasks(db: Database, timeout_minutes: float = 30.0) -> int:
    """捞回卡死的任务：uploading/waiting/processing 且心跳超时的，退回 pending。

    没有这一步，程序在 processing 状态被强杀之后那条任务会永远卡着。
    """
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                           time.localtime(time.time() - timeout_minutes * 60))
    marks = ", ".join("?" for _ in TASK_ACTIVE)
    with db.tx() as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_tasks
               SET status = 'pending', error = '上次异常退出，已退回等待',
                   retry_count = retry_count + 1
             WHERE status IN ({marks})
               AND COALESCE(heartbeat_at, started_at, created_at) < ?
            """,
            (*TASK_ACTIVE, cutoff))
        return int(cur.rowcount or 0)


# =================================================================== AI 结果
def save_ai_result(db: Database, video_id: int, *, task_id: int | None = None,
                   raw_response: str | None = None, json_data: Any = None,
                   candidate_count: int | None = None, winner_score: float | None = None,
                   validated: bool = False, validation_error: str | None = None) -> int:
    """存 AI 结果。raw_response 一定要存原文，以后要追溯 AI 当时到底回了什么。"""
    with db.tx() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_results(task_id, video_id, raw_response, json_data, candidate_count,
                                   winner_score, validated, validation_error, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, video_id, raw_response, _dumps(json_data), candidate_count,
             winner_score, int(validated), validation_error, now()))
        return int(cur.lastrowid)


def get_ai_result(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.one("SELECT * FROM ai_results WHERE video_id = ? ORDER BY id DESC LIMIT 1",
                  (video_id,))


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
                output_path: str | Path | None = None) -> None:
    sets, params = [], []
    if status:
        sets.append("status = ?")
        params.append(status)
    if output_path is not None:
        sets.append("output_path = ?")
        params.append(str(output_path))
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
    """某种产物的路径（只认还在盘上的那条）。没有就 None。"""
    row = db.one(
        """
        SELECT path FROM artifacts
         WHERE video_id = ? AND type = ? AND exists_on_disk = 1 AND COALESCE(size, 0) > 0
         ORDER BY updated_at DESC LIMIT 1
        """, (video_id, kind))
    return Path(row["path"]) if row else None


def has_artifact(db: Database, video_id: int, kind: str) -> bool:
    return artifact_path(db, video_id, kind) is not None


# ===================================================================== 统计
def videos_under(db: Database, folder: str | Path | None) -> list[sqlite3.Row]:
    """某个目录（含子目录）里、还在盘上的视频。界面列任务表就用这个，不去翻目录。"""
    if folder is None:
        return []
    prefix = str(Path(folder).resolve())
    if not prefix.endswith(os.sep):
        prefix += os.sep
    # 文件名里可能真带 % 或 _，得转义，否则 LIKE 会把它们当通配符
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return db.all(
        "SELECT * FROM videos WHERE exists_on_disk = 1 AND file_path LIKE ? ESCAPE '\\' "
        "ORDER BY file_name",
        (escaped + "%",))


def states_for_videos(db: Database, video_ids: list[int],
                      sig: dict[str, Any] | None = None) -> dict[int, dict[str, bool]]:
    """一批视频的四个状态，四条聚合 SQL 出来，不按视频逐个查、更不扫磁盘。

    界面刷新就靠这个：40 个视频也是四次查询，不会因为数据库化反而变慢。
    四个状态各自独立：
    - analysed  analysis_runs 里有 completed（给了 sig 还要模型/配置哈希对得上）
    - txt       artifacts 里有还在盘上的 merged_txt
    - json      ai_results 有记录，或 artifacts 里有 ai_script
    - clipped   artifacts 里有 final_video，或 clips 里有 rendered
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

    for row in db.all(
            f"SELECT DISTINCT video_id FROM clips "
            f"WHERE status = 'rendered' AND video_id IN ({marks})", params):
        out[int(row["video_id"])]["clipped"] = True
    return out


def statistics_for(db: Database, video_ids: list[int], *,
                   done_key: str = "clipped",
                   sig: dict[str, Any] | None = None) -> dict[str, int]:
    """AI 面板四格统计：总任务 / 未剪辑（未完成）/ 已获取 JSON / 成品。

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


def video_state(db: Database, video_id: int, sig: dict[str, Any] | None = None) -> dict[str, bool]:
    """一个视频的四个状态，各自独立判定，全部来自数据库。

    - analysed：有跑成功的分析（给了 sig 就还要模型/配置对得上）
    - txt：merged_txt 产物在
    - json：有 AI 结果，或者登记过 ai_script 文件
    - clipped：有 final_video 产物，或者 clips 里有 rendered 的
    """
    if sig is None:
        analysed = latest_analysis(db, video_id) is not None
    else:
        analysed = find_cached_analysis(db, video_id, sig) is not None
    txt = has_artifact(db, video_id, "merged_txt")
    json_ok = (get_ai_result(db, video_id) is not None
               or has_artifact(db, video_id, "ai_script"))
    clipped = (has_artifact(db, video_id, "final_video")
               or db.one("SELECT 1 FROM clips WHERE video_id = ? AND status = 'rendered'",
                         (video_id,)) is not None)
    return {"analysed": analysed, "txt": txt, "json": json_ok, "clipped": clipped}


def get_statistics(db: Database, video_ids: list[int] | None = None) -> dict[str, int]:
    """AI 面板那四格：总任务 / 未剪辑 / 已获取 JSON / 成品。一条 SQL 聚合出来。"""
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
             WHERE type = 'final_video' AND exists_on_disk = 1 GROUP BY video_id
        ) done ON done.video_id = v.id
        LEFT JOIN (
            SELECT video_id, COUNT(*) AS n FROM (
                SELECT video_id FROM ai_results
                UNION ALL
                SELECT video_id FROM artifacts WHERE type = 'ai_script' AND exists_on_disk = 1
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
