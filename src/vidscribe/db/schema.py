"""建表语句与表结构版本。

分层就一句话：**数据库是分析结果的唯一权威来源，文件只是导出/临时/兼容。**
本地分析产生的一切（语音段、逐词、画面事件、表情轨、当次渲染参数）都必须进库，
删掉 output/ 与 cache/ 之后仍要能从库里重建出完整剧本；
落盘的 JSON/TXT 只是给人看、给扩展 AI 传输、给老版本兼容用的派生物。

一个视频（videos）下面挂：
  analysis_runs   每跑一次分析一条，模型/配置换了就是新的一条，不覆盖历史
    visual_events     视觉事件（Qwen 只负责"发生了什么"）
    speech_segments   语音段
      speech_words    逐词时间戳（精确剪辑靠它，不能退化成只存句子）
    expression_spans  人脸表情轨（剧本 SECTION 3 的唯一权威来源）
  ai_tasks        一次 AI 请求，状态机 pending -> ... -> completed/failed
    ai_results    AI 原文（raw_response）+ 解析后的 JSON
      clips       AI 选中的片段，start/end/score/type/reason 原样存
  artifacts       实际文件：原片、merged txt、srt、预览音轨、成品 mp4 ...
"""

from __future__ import annotations

# 表结构版本。加/改表就 +1，并在 migrations.py 里补一段升级脚本。
SCHEMA_VERSION = 8

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

# 表情轨（剧本 SECTION 3）的三种状态，靠 analysis_runs.face_available + 表内行数区分：
#   ok             这次分析检到了人脸，expression_spans 里有段
#   no_face        这次分析跑过人脸模型，但全片没有有效人脸（合法的空）
#   legacy_missing 这条分析在表情落库之前完成，库里根本没有这份数据 -> 只能重新分析
# 三者必须严格区分：不能拿 no_face 冒充 ok，也不能把 legacy_missing 当成"没有表情"。
EXPRESSION_OK = "ok"
EXPRESSION_NO_FACE = "no_face"
EXPRESSION_LEGACY_MISSING = "legacy_missing"
EXPRESSION_STATES = (EXPRESSION_OK, EXPRESSION_NO_FACE, EXPRESSION_LEGACY_MISSING)

ARTIFACT_TYPES = ("source_video", "merged_txt", "words_srt", "translated_txt",
                  "preview_audio", "final_video", "thumbnail", "ai_script")

# 高光方案（highlight_assets）的来源：AI 回的 / 手工写的 / 从盘上导入的 /
# 在已有方案上编辑出来的 / 复制出来的。原始 AI 结果永远留在 raw_json 里。
ASSET_SOURCES = ("ai", "manual", "imported", "edited", "copied")

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
        -- 语言预检判出来、又不在 speech.allowed_languages 里的那个语言码（比如 'id'）。
        -- 非空 = 这条视频以后不再自动跑（手动点分析仍会重新预检并当场终止）
        blocked_language TEXT,
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
        created_at         TEXT    NOT NULL,
        -- 当次分析的渲染事实：换了 GUI 配置也不该让同一个视频重新生成出不一样的剧本。
        -- output_language 决定表头与情绪显示名，render_config 是 timeline 的三个过滤参数。
        output_language    TEXT,
        render_config      TEXT,
        -- 1 = 跑过人脸模型且检到脸，0 = 跑过但全片无脸，NULL = 这条分析没存过表情轨
        face_available     INTEGER
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

    # --- 人脸表情轨 -------------------------------------------------------
    # 剧本 SECTION 3 的唯一权威来源。视觉事件上的 emotion_* 是"事件粒度的覆盖值"，
    # 这里是人脸模型 2fps 采样归并出的独立时间轴，两者粒度和语义都不同，不能互相推算。
    # raw_json 存整段原始 span：以后 face 模型多给字段，不用再动 schema。
    """
    CREATE TABLE IF NOT EXISTS expression_spans (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        sequence    INTEGER,
        start_time  REAL,
        end_time    REAL,
        emotion_en  TEXT,
        intensity   REAL,
        samples     INTEGER,
        raw_json    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_expression_analysis "
    "ON expression_spans(analysis_id, sequence)",


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

    # --- PRM（提示词档案）-------------------------------------------------
    # **提示词正文存在库里（content）**，库就是唯一权威；filename 只记"当初从哪个
    # 文件导进来的"，发 AI 时不再读它。成品记 prm_id 而不是文件名：以后改名/换目录，
    # 历史依然查得到。enabled 是「使用状况」：发 AI 时启用的每一份都当附件带上。
    """
    CREATE TABLE IF NOT EXISTS prm_profiles (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        filename    TEXT    NOT NULL,
        description TEXT,
        language    TEXT,
        version     TEXT,
        content     TEXT,
        is_default  INTEGER NOT NULL DEFAULT 0,
        enabled     INTEGER NOT NULL DEFAULT 1,
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

    # --- 高光方案（资产）--------------------------------------------------
    # 一个视频可以有任意多份高光 JSON，谁也不覆盖谁：新结果永远是新的一行。
    # raw_json 是当时那份原始输出，一个字都不改；人工编辑落在 current_json，
    # 而且编辑默认另开一条（source_type='edited' + parent_id 指回来），
    # 所以「AI 当时到底给了什么」永远追得到。
    # 删除一律软删（deleted_at），已经剪出来的成品绝不跟着消失。
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
    # 每个视频同时只能有一个「当前方案」（软删掉的不算）
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_current_live
        ON highlight_assets(video_id)
     WHERE is_current = 1 AND deleted_at IS NULL
    """,

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
        -- 成品溯源（v4）：这份文件是拿哪个高光方案、哪个 PRM 剪出来的。
        -- 方案/PRM 以后被软删也不影响这里：存的是 id，历史照旧查得到。
        highlight_asset_id INTEGER REFERENCES highlight_assets(id) ON DELETE SET NULL,
        prm_id             INTEGER REFERENCES prm_profiles(id) ON DELETE SET NULL,
        UNIQUE(video_id, type, path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_video ON artifacts(video_id, type)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_asset ON artifacts(highlight_asset_id)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_prm ON artifacts(prm_id)",

    # --- 元信息 -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
)
