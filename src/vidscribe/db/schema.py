"""建表语句与表结构版本。

分层就一句话：**文件是文件，数据库是状态。**
分析结果、AI 回复、剪辑片段照旧落盘（output/、cache/、AI_输出目录），
这里只记录"有什么、属于哪个视频、现在是什么状态"，让别处不用再扫目录猜。

一个视频（videos）下面挂：
  analysis_runs   每跑一次分析一条，模型/配置换了就是新的一条，不覆盖历史
    visual_events   视觉事件（Qwen 只负责"发生了什么"）
    speech_segments 语音段
      speech_words  逐词时间戳（精确剪辑靠它，不能退化成只存句子）
  ai_tasks        一次 AI 请求，状态机 pending -> ... -> completed/failed
    ai_results    AI 原文（raw_response）+ 解析后的 JSON
      clips       AI 选中的片段，start/end/score/type/reason 原样存
  artifacts       实际文件：原片、merged txt、srt、预览音轨、成品 mp4 ...
"""

from __future__ import annotations

# 表结构版本。加/改表就 +1，并在 migrations.py 里补一段升级脚本。
SCHEMA_VERSION = 3

# AI 任务的状态机。别再用「TXT 存不存在」推断任务走到哪了。
TASK_STATES = ("pending", "uploading", "waiting", "processing",
               "completed", "failed", "cancelled")
# 跑着一半的状态：程序崩了要靠超时把它们捞回 pending
TASK_ACTIVE = ("uploading", "waiting", "processing")
# 还没跑完的状态（含 pending）：幂等判重、取消剩余任务都看这一组
TASK_OPEN = ("pending", *TASK_ACTIVE)
# 任务种类：自动剪辑队列一种，手工单发一种（人工操作不进队列，互不干扰）
AUTO_TASK_TYPE = "auto_clip"
MANUAL_TASK_TYPE = "manual"

ANALYSIS_STATES = ("running", "completed", "failed")

ARTIFACT_TYPES = ("source_video", "merged_txt", "words_srt", "translated_txt",
                  "preview_audio", "final_video", "thumbnail", "ai_script")

TABLES: tuple[str, ...] = (
    # --- 视频主表 ---------------------------------------------------------
    # fingerprint 是主键式的身份：文件大小 + 头/中/尾各 1MB 的 sha256。
    # 改名、搬目录都还能认出是同一个视频；全文件 sha256 太慢，留列惰性补算。
    """
    CREATE TABLE IF NOT EXISTS videos (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        fingerprint    TEXT    NOT NULL UNIQUE,
        sha256         TEXT,
        file_path      TEXT    NOT NULL,
        file_name      TEXT    NOT NULL,
        file_size      INTEGER,
        duration       REAL,
        width          INTEGER,
        height         INTEGER,
        fps            REAL,
        cache_slug     TEXT,
        exists_on_disk INTEGER NOT NULL DEFAULT 1,
        in_library     INTEGER,
        status         TEXT    NOT NULL DEFAULT 'new',
        created_at     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_videos_path ON videos(file_path)",
    "CREATE INDEX IF NOT EXISTS idx_videos_slug ON videos(cache_slug)",

    # --- 分析批次 ---------------------------------------------------------
    # 缓存命中不再看「json 在不在」，而是看有没有 completed 且
    # vision_model / vision_config_hash / asr_model / asr_config_hash 全对得上的一条。
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id           INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        status             TEXT    NOT NULL DEFAULT 'running',
        started_at         TEXT,
        finished_at        TEXT,
        vision_model       TEXT,
        vision_config      TEXT,
        vision_config_hash TEXT,
        asr_model          TEXT,
        asr_config         TEXT,
        asr_config_hash    TEXT,
        scene_count        INTEGER,
        speech_count       INTEGER,
        output_dir         TEXT,
        source             TEXT    NOT NULL DEFAULT 'pipeline',
        error              TEXT,
        created_at         TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_video ON analysis_runs(video_id, status)",
    """
    CREATE INDEX IF NOT EXISTS idx_runs_hit
        ON analysis_runs(video_id, status, vision_model, vision_config_hash,
                         asr_model, asr_config_hash)
    """,

    # --- 视觉事件 ---------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS visual_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        start_time  REAL,
        end_time    REAL,
        description TEXT,
        event_type  TEXT,
        confidence  REAL,
        sequence    INTEGER,
        raw_json    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_visual_analysis ON visual_events(analysis_id, sequence)",

    # --- 语音段 + 逐词 ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS speech_segments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        start_time  REAL,
        end_time    REAL,
        text        TEXT,
        speaker     TEXT,
        emotion     TEXT,
        confidence  REAL,
        sequence    INTEGER,
        raw_json    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_segments_analysis ON speech_segments(analysis_id, sequence)",
    """
    CREATE TABLE IF NOT EXISTS speech_words (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        segment_id  INTEGER NOT NULL REFERENCES speech_segments(id) ON DELETE CASCADE,
        analysis_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        word_index  INTEGER,
        word        TEXT,
        start_time  REAL,
        end_time    REAL,
        confidence  REAL,
        speaker     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_words_segment ON speech_words(segment_id, word_index)",
    "CREATE INDEX IF NOT EXISTS idx_words_analysis ON speech_words(analysis_id, start_time)",

    # --- AI 任务 / 结果 / 片段 --------------------------------------------
    # 队列落库，关掉程序再开还在；processing 卡死靠 heartbeat_at 超时捞回来。
    """
    CREATE TABLE IF NOT EXISTS ai_tasks (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id       INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        mode           TEXT    NOT NULL DEFAULT 'full',
        provider       TEXT,
        model          TEXT,
        status         TEXT    NOT NULL DEFAULT 'pending',
        prompt_version TEXT,
        input_txt      TEXT,
        created_at     TEXT    NOT NULL,
        started_at     TEXT,
        finished_at    TEXT,
        heartbeat_at   TEXT,
        retry_count    INTEGER NOT NULL DEFAULT 0,
        error          TEXT,
        task_type      TEXT    NOT NULL DEFAULT 'auto_clip',
        priority       INTEGER NOT NULL DEFAULT 100,
        max_attempts   INTEGER NOT NULL DEFAULT 1,
        worker_id      TEXT,
        updated_at     TEXT,
        -- 这次真正发给 AI 的那份提示词文件（内容不进库，只留指纹/路径/大小）：
        -- 事后能回答「这条任务当时用的是哪一版 prm_en.txt」
        prompt_hash    TEXT,
        prompt_path    TEXT,
        prompt_size    INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON ai_tasks(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_video ON ai_tasks(video_id, status)",
    # 幂等的底线：同一个视频 + 同一种任务 + 同一种模式，同时只能有一条没跑完的。
    # 连点五次「自动剪辑」也只会有一条 pending，靠数据库拦，不靠界面自觉。
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_open_unique
        ON ai_tasks(video_id, task_type, mode)
     WHERE status IN ('pending', 'uploading', 'waiting', 'processing')
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_results (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id          INTEGER REFERENCES ai_tasks(id) ON DELETE SET NULL,
        video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        raw_response     TEXT,
        json_data        TEXT,
        candidate_count  INTEGER,
        winner_score     REAL,
        validated        INTEGER NOT NULL DEFAULT 0,
        validation_error TEXT,
        created_at       TEXT    NOT NULL,
        -- 这份结果是拿哪一版提示词换回来的（手工单发没有任务行，就靠这三列追溯）
        prompt_hash      TEXT,
        prompt_path      TEXT,
        prompt_size      INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_results_video ON ai_results(video_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS clips (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id     INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        ai_result_id INTEGER REFERENCES ai_results(id) ON DELETE SET NULL,
        start_time   REAL,
        end_time     REAL,
        duration     REAL,
        score        REAL,
        clip_type    TEXT,
        reason       TEXT,
        evaluation   TEXT,
        status       TEXT    NOT NULL DEFAULT 'planned',
        output_path  TEXT,
        created_at   TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id, status)",

    # --- 实际文件 ---------------------------------------------------------
    # 同一个视频的同一种产物同一个路径只留一条，重复登记就更新。
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id       INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        type           TEXT    NOT NULL,
        path           TEXT    NOT NULL,
        size           INTEGER,
        sha256         TEXT,
        exists_on_disk INTEGER NOT NULL DEFAULT 1,
        created_at     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL,
        UNIQUE(video_id, type, path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_video ON artifacts(video_id, type)",

    # --- 元信息 -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
)
