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
from .schema import (
    ANALYSIS_STATES,
    AUTO_TASK_TYPE,
    EXPRESSION_LEGACY_MISSING,
    EXPRESSION_NO_FACE,
    EXPRESSION_OK,
    TASK_ACTIVE,
    TASK_OPEN,
    TASK_STATES,
)

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


def ensure_duration(db: Database, video_id: int, video: str | Path) -> float | None:
    """库里没有时长就现探一次写回去；已经有值不动，探不到就保持 NULL。

    探测复用 `video_io.probe_video`（剪辑引擎本来就要探这一下），所以这里不是第二套
    获取逻辑，只是把探到的结果顺手落库——资产中心显示时长只查库，不扫盘。
    """
    row = db.one("SELECT duration, width, height, fps FROM videos WHERE id = ?", (video_id,))
    if row is None:
        return None
    if row["duration"] is not None and float(row["duration"]) > 0:
        return float(row["duration"])          # 已有值就用库里的，绝不覆盖
    path = Path(video)
    if not path.is_file():
        return None
    try:
        from ..video_io import probe_video  # noqa: PLC0415 - cv2/av 很重，用到才导

        info = probe_video(path)
    except Exception as exc:  # noqa: BLE001 - 探不到就保持空，绝不写假数据
        logger.warning("探不到 %s 的时长（%s），duration 保持空", path.name, exc)
        return None
    duration = float(info.duration or 0.0)
    if duration <= 0:
        return None
    fields: dict[str, Any] = {"duration": duration}
    for key, value in (("width", info.width), ("height", info.height), ("fps", info.fps)):
        if row[key] is None and value:
            fields[key] = value
    sets = ", ".join(f"{key} = ?" for key in fields)
    with db.tx() as conn:
        conn.execute(f"UPDATE videos SET {sets}, updated_at = ? WHERE id = ?",
                     (*fields.values(), now(), video_id))
    return duration


def fill_missing_durations(db: Database, *, limit: int | None = None) -> dict[str, int]:
    """给老数据补时长的**显式**入口（`run.py db --fill-duration`）。

    只碰 duration 为空、文件还在盘上的视频。GUI 不会自动调这个，免得一开界面
    就把整个视频库探一遍。
    """
    rows = db.all("SELECT id, file_path FROM videos WHERE duration IS NULL "
                  "AND exists_on_disk = 1 ORDER BY id", ())
    if limit:
        rows = rows[:limit]
    stats = {"checked": 0, "filled": 0, "failed": 0}
    for row in rows:
        stats["checked"] += 1
        if ensure_duration(db, int(row["id"]), str(row["file_path"])) is None:
            stats["failed"] += 1
        else:
            stats["filled"] += 1
    return stats


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


def set_blocked_language(db: Database, video_id: int, language: str | None) -> None:
    """记下「这条视频的语言不在允许范围」。非空之后自动剪辑不再排它。

    只记语言码（'id'、'ko'…），不记时间：要重跑就把这条登记删了重新导入。
    """
    code = (str(language).strip().lower().split("-")[0] or None) if language else None
    with db.tx() as conn:
        conn.execute("UPDATE videos SET blocked_language = ?, updated_at = ? WHERE id = ?",
                     (code, now(), video_id))


def blocked_language(db: Database, video_id: int) -> str | None:
    """这条视频被哪个语言拦下了；没拦过返回 None。"""
    row = db.one("SELECT blocked_language FROM videos WHERE id = ?", (video_id,))
    if row is None:
        return None
    code = row["blocked_language"]
    return str(code) if code else None


def blocked_language_videos(db: Database, folder: str | Path | None = None) -> list[sqlite3.Row]:
    """被语言拦下的视频（`blocked_language` 非空）。给目录就只看那个目录（含子目录）。

    盘上还在不在都算：文件已经被挪走的，库里那条登记也一样该清掉。
    """
    if folder is None:
        return db.all("SELECT * FROM videos WHERE blocked_language IS NOT NULL "
                      "AND blocked_language <> '' ORDER BY file_name")
    return db.all(
        "SELECT * FROM videos WHERE blocked_language IS NOT NULL AND blocked_language <> '' "
        "AND file_path LIKE ? ESCAPE '\\' ORDER BY file_name",
        (_folder_like(folder),))


def set_video_presence(db: Database, video_id: int, *, exists: bool,
                       in_library: bool | None = None) -> None:
    """对账用：视频还在不在盘上、在不在视频库里。"""
    with db.tx() as conn:
        conn.execute(
            "UPDATE videos SET exists_on_disk = ?, in_library = ?, updated_at = ? WHERE id = ?",
            (int(exists), None if in_library is None else int(in_library), now(), video_id))


def rename_video(db: Database, video_id: int, target: str | Path) -> None:
    """视频文件改名之后，把库里的路径 / 文件名跟着改（指纹和缓存目录不动）。

    调用方负责先把磁盘上的文件挪好。`fingerprint` 是按内容算的，改名不影响；
    `cache_slug` 也保持原值，免得已经算好的分析缓存全部失联。
    """
    path = Path(target)
    with db.tx() as conn:
        conn.execute("UPDATE videos SET file_path = ?, file_name = ?, updated_at = ? "
                     "WHERE id = ?", (str(path), path.name, now(), video_id))


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


# --- 人脸表情轨（剧本 SECTION 3 的唯一权威来源）------------------------------
def save_expression_spans(db: Database, analysis_id: int,
                          spans: list[dict[str, Any]]) -> int:
    """存人脸表情轨。重存会先清掉这条分析下的旧段（同一条分析只该有一套）。

    拆出来的列是给查询用的，`raw_json` 存整段原样——以后 face 模型多给字段，
    不用再改表也不会丢数据。
    """
    rows = []
    for i, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            continue
        rows.append((
            analysis_id, i,
            span.get("start"), span.get("end"),
            span.get("emotion_en"), span.get("intensity"), span.get("samples"),
            _dumps(span),
        ))
    with db.tx() as conn:
        conn.execute("DELETE FROM expression_spans WHERE analysis_id = ?", (analysis_id,))
        conn.executemany(
            """
            INSERT INTO expression_spans(analysis_id, sequence, start_time, end_time,
                                         emotion_en, intensity, samples, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
    return len(rows)


def get_expression_spans(db: Database, analysis_id: int) -> list[sqlite3.Row]:
    return db.all("SELECT * FROM expression_spans WHERE analysis_id = ? ORDER BY sequence",
                  (analysis_id,))


def note_render(db: Database, analysis_id: int, *, output_language: str | None = None,
                render_config: dict[str, Any] | None = None,
                face_available: bool | None = None) -> None:
    """记下这次分析**当时**的渲染事实：输出语言、timeline 过滤参数、有没有检到人脸。

    用户后来改了 GUI 配置，同一个视频从库里重建出的剧本也必须和当初逐行一致，
    所以这三样必须跟着分析记录存，不能等到导出时再去读"现在的配置"。
    """
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE analysis_runs
               SET output_language = COALESCE(?, output_language),
                   render_config   = COALESCE(?, render_config),
                   face_available  = COALESCE(?, face_available)
             WHERE id = ?
            """,
            (output_language, _dumps(render_config),
             None if face_available is None else int(face_available), analysis_id))


def expression_state(db: Database, analysis_id: int, *,
                     face_available: Any = None, span_count: int | None = None) -> str:
    """表情轨的三种状态，见 schema.EXPRESSION_STATES。

    传了 face_available / span_count 就不再查库（调用方已经拿到行的时候省一次往返）。
    判定只看库：**不去摸 timeline.json 或 cache**，那两个是派生文件。
    """
    if face_available is None:
        row = db.one("SELECT face_available FROM analysis_runs WHERE id = ?", (analysis_id,))
        face_available = None if row is None else row["face_available"]
    if span_count is None:
        row = db.one("SELECT COUNT(*) AS n FROM expression_spans WHERE analysis_id = ?",
                     (analysis_id,))
        span_count = 0 if row is None else int(row["n"])
    if span_count > 0:
        return EXPRESSION_OK
    if face_available is None:
        # 这条分析是表情落库之前跑的：库里没有这份数据，只能重新分析，绝不能当成"没有表情"
        return EXPRESSION_LEGACY_MISSING
    return EXPRESSION_NO_FACE


def _loads(raw: Any) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def script_inputs(db: Database, video_id: int, *,
                  analysis_id: int | None = None) -> dict[str, Any] | None:
    """把库里的分析结果还原成"生成剧本要用的原始数据"，供 timeline/exporters 使用。

    这里**只取数、只还原形状**：不合并时间线、不算动作轨、不排版。
    时间线仍旧由 timeline.engine 的 build_timeline / filter_timeline 算，
    剧本正文仍旧由 exporters.merged_lines 生成——那两处是唯一的实现，不在这里复制。

    找不到可用的 completed 分析就返回 None（调用方自己决定报错还是回退）。
    """
    if analysis_id is None:
        run = latest_analysis(db, video_id)
        if run is None:
            return None
        analysis_id = int(run["id"])
    else:
        run = db.one("SELECT * FROM analysis_runs WHERE id = ?", (analysis_id,))
        if run is None:
            return None
    video = db.one("SELECT * FROM videos WHERE id = ?", (video_id,))
    if video is None:
        return None

    keys = run.keys()
    # 老库刚升上来时这三列可能还没被任何一次分析写过：读不到就是没存过，不猜
    output_language = run["output_language"] if "output_language" in keys else None
    render_config = _loads(run["render_config"] if "render_config" in keys else None) or {}
    face_available = run["face_available"] if "face_available" in keys else None

    segments = [_loads(row["raw_json"]) or {} for row in get_speech_segments(db, analysis_id)]
    events = [_loads(row["raw_json"]) or {} for row in get_visual_events(db, analysis_id)]
    spans = [_loads(row["raw_json"]) or {} for row in get_expression_spans(db, analysis_id)]
    return {
        "analysis_id": analysis_id,
        "video_id": video_id,
        "video_name": video["file_name"],
        "video_path": video["file_path"],
        "duration": float(video["duration"] or 0.0),
        "segments": segments,
        "events": events,
        "emotions": spans,
        "output_language": output_language,
        "render_config": render_config,
        "expression_state": expression_state(db, analysis_id,
                                             face_available=face_available,
                                             span_count=len(spans)),
    }



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
def video_footprint(db: Database, video_id: int) -> dict[str, int]:
    """这个视频在库里占了多少行：分析 / 事件 / 语音 / AI 任务 / AI 回复 / 高光 JSON /
    片段 / 文件登记。

    删之前先给用户看清楚要没掉什么，别让人蒙着眼点「确定」。
    """
    counts = {
        "analyses": "SELECT COUNT(*) FROM analysis_runs WHERE video_id = ?",
        "events": "SELECT COUNT(*) FROM visual_events WHERE analysis_id IN "
                  "(SELECT id FROM analysis_runs WHERE video_id = ?)",
        "segments": "SELECT COUNT(*) FROM speech_segments WHERE analysis_id IN "
                    "(SELECT id FROM analysis_runs WHERE video_id = ?)",
        "tasks": "SELECT COUNT(*) FROM ai_tasks WHERE video_id = ?",
        "results": "SELECT COUNT(*) FROM ai_results WHERE video_id = ?",
        "assets": "SELECT COUNT(*) FROM highlight_assets WHERE video_id = ?",
        "clips": "SELECT COUNT(*) FROM clips WHERE video_id = ?",
        "artifacts": "SELECT COUNT(*) FROM artifacts WHERE video_id = ?",
    }
    return {key: int(db.value(sql, (video_id,)) or 0) for key, sql in counts.items()}


def forget_video(db: Database, video_id: int) -> dict[str, int] | None:
    """把这个视频从库里删掉（**磁盘上的文件一个都不动**）。返回删掉了多少行；不存在返回 None。

    `videos` 是所有表的外键根，全部是 `ON DELETE CASCADE`，所以这一条 DELETE 会带走
    它的分析、画面事件、语音段、表情段、AI 任务、AI 结果、片段、高光 JSON 资产、
    成品登记、文件登记。**不可恢复**，调用方必须先让用户确认。
    """
    if db.one("SELECT id FROM videos WHERE id = ?", (video_id,)) is None:
        return None
    gone = video_footprint(db, video_id)
    with db.tx() as conn:
        conn.execute("PRAGMA foreign_keys=ON")      # 级联靠它，别指望默认
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    return gone


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
    """一批视频的业务阶段，几条聚合 SQL 出来，不按视频逐个查、更不扫磁盘。

    界面刷新就靠这个。**数据库是唯一权威**，各状态互相独立：
    - analysed  analysis_runs 里有 completed（给了 sig 还要模型/配置哈希对得上）
    - script    库里能生成完整剧本（script_ready_videos）—— 这才是"有剧本"的判据
    - attempted 已经做过高光分析（highlight_attempted_videos）
    - json_ok   库里有一份能直接开剪的高光 JSON（reusable_json_videos）
    - rendered  剪辑这一步做过（clips 里有 rendered，成品后来被删也算做过）
    - clipped   artifacts 里有还在盘上的 final_video（clips 的 rendered 只算历史）
    - txt/json  **只表示"盘上有这个文件"**（merged_txt / ai_script），仅供显示与兼容，
                任何决策都不许用它们：TXT 是传输文件、_脚本.json 是导出文件。
    """
    empty = {"analysed": False, "script": False, "attempted": False, "json_ok": False,
             "rendered": False, "txt": False, "json": False, "clipped": False}
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

    # 真正参与决策的三个：能不能出剧本 / 问过 AI 没有 / 手上的 JSON 能不能直接开剪。
    # 三者全部只查库，和上面那两个"文件在不在"是不同的东西。
    for vid in script_ready_videos(db, video_ids):
        if vid in out:
            out[vid]["script"] = True
    for vid in highlight_attempted_videos(db, video_ids):
        if vid in out:
            out[vid]["attempted"] = True
    for vid in reusable_json_videos(db, video_ids):
        if vid in out:
            out[vid]["json_ok"] = True
    for vid in rendered_clip_videos(db, video_ids):
        if vid in out:
            out[vid]["rendered"] = True

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


def rendered_clip_videos(db: Database, video_ids: list[int]) -> set[int]:
    """这些视频里，哪些**剪过**（clips 里有 rendered 记录）。按视频去重。

    这是"剪辑这一步做过没有"，跟"成品还在不在盘上"（artifact_videos final_video）
    是两件事：成品被删掉了，剪辑仍然发生过——面板要能把这两列分开显示。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    return {int(row["video_id"]) for row in db.all(
        f"SELECT DISTINCT video_id FROM clips "
        f"WHERE video_id IN ({marks}) AND status = 'rendered'", list(video_ids))}


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


def script_ready_videos(db: Database, video_ids: list[int]) -> set[int]:
    """这些视频里，哪些**能从库里生成完整剧本**。一条 SQL，按视频去重。

    判据只看库：有 completed 的分析，且那次分析下面至少落过画面事件或语音段。
    表情轨（expression_spans）不作为门槛——历史分析没存过它，剧本照样出四段，
    只是 SECTION 3 会明确写"库里没有这份数据"（见 exporters 的 expression_state）。
    **绝不看 AI_输入目录里有没有 TXT**：那是传输文件，不是剧本。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    return {int(row["video_id"]) for row in db.all(
        f"""
        SELECT DISTINCT r.video_id FROM analysis_runs r
         WHERE r.status = 'completed' AND r.video_id IN ({marks})
           AND (EXISTS(SELECT 1 FROM visual_events e WHERE e.analysis_id = r.id)
                OR EXISTS(SELECT 1 FROM speech_segments s WHERE s.analysis_id = r.id))
        """, list(video_ids))}


def highlight_attempted_videos(db: Database, video_ids: list[int], *,
                               mode: str | None = None,
                               task_type: str = AUTO_TASK_TYPE) -> set[int]:
    """这些视频里，哪些**已经做过高光分析**（把 PRM + 完整剧本交给 AI 过）。

    判据：ai_tasks 里有非 pending 的任务（pending 只是排着队，还没提交），
    或者 ai_results 里已经有结果。跟"结果能不能用"无关——能不能用看
    `reusable_json_videos`，这里回答的是"问过没有"。
    """
    if not video_ids:
        return set()
    marks = ", ".join("?" for _ in video_ids)
    params: list[Any] = [task_type, *video_ids]
    sql = (f"SELECT DISTINCT video_id FROM ai_tasks WHERE task_type = ? "
           f"AND video_id IN ({marks}) AND status <> 'pending'")
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    out = {int(row["video_id"]) for row in db.all(sql, params)}
    out |= {int(row["video_id"]) for row in db.all(
        f"SELECT DISTINCT video_id FROM ai_results WHERE video_id IN ({marks})",
        list(video_ids))}
    return out


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
    另外四个横切指标按真实业务链给：`analysed`（分析完成）、`script`（库里能出完整剧本）、
    `attempted`（做过高光分析）、`clipped`（有有效成品）——它们和九个桶无关，只用于顶部数字。
    """
    ids = {int(v) for v in video_ids}
    ordered = sorted(ids)
    json_ok = reusable_json_videos(db, ordered) & ids
    clipped_ids = artifact_videos(db, ordered, "final_video") & ids
    analysed_ids = {vid for vid, state in states_for_videos(db, ordered).items()
                    if state["analysed"]}
    script_ids = script_ready_videos(db, ordered) & ids
    attempted_ids = highlight_attempted_videos(db, ordered, mode=mode,
                                               task_type=task_type) & ids
    done = json_ok if done_key == "json" else clipped_ids
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
        # 横切指标：跟九个互斥桶无关，界面顶部按真实业务链显示用的
        "analysed": len(analysed_ids & ids),
        "script": len(script_ids & ids),
        "attempted": len(attempted_ids & ids),
        "rendered": len(rendered_clip_videos(db, ordered) & ids),
        "clipped": len(clipped_ids & ids),
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
