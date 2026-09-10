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
SCHEMA_VERSION = 14


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
        -- 音轨预检的结论：1 = 这个文件里根本没有音轨，0 = 有，NULL = 还没探过。
        -- 1 的不再排进自动剪辑（没声音就没剧本、没高光），也能被「清空无声音视频」清走
        no_audio       INTEGER,
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
    # confidence 是分类置信度（段内 top-1 softmax 概率的平均），不是"表情强弱"。
    # raw_json 存整段原始 span：以后 face 模型多给字段，不用再动 schema。
    """
    CREATE TABLE IF NOT EXISTS expression_spans (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        sequence    INTEGER,
        start_time  REAL,
        end_time    REAL,
        emotion_en  TEXT,
        confidence  REAL,
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


# ======================================================================== 舞蹈
# v11 起：独立的舞蹈素材资产子系统（见 vidscribe/dance/）。
# 和高光那一套**完全分开**：不共用 highlight_assets / clips / artifacts，
# 也不新建第二个 SQLite —— 全部落在同一个 video.db 里，表名一律 dance_ 前缀。
#
# 一首目标歌（dance_target_songs）下面挂：
#   dance_audio_alignments        每个源舞蹈视频 → 这首歌的时间对齐，一条一版
#     dance_materials             按固定音乐位置切出来的素材，**长期资产**，不物理删
#       dance_material_usage_events   使用事件账本，所有计数都能由它重算
#   dance_montages                一次混剪任务
#     dance_montage_versions      每次 remix 一版，历史版本不因重新生成而丢失
#       dance_montage_materials   这一版的编辑计划（引用 material_id，不是路径）
#   dance_recommendation_runs     每次推荐一条，带 seed 所以可重现
#     dance_recommendation_items  推荐出来的条目
#     dance_recommendation_feedback 用户对推荐的反馈（采纳/否决）
#   dance_montage_strategies      评分权重 + 组合约束 + 搜索预算，可复用可版本化

#: 对齐结论。manual = 人工改过（原值仍在 original_offset 里）
DANCE_ALIGNMENT_STATUS = ("ok", "low_confidence", "disagree", "rejected", "manual")
#: 素材状态。历史素材一律软状态流转，不物理删除。
#: missing = 文件在盘上找不到了（记录留着，历史成品还引用它）
DANCE_MATERIAL_STATUS = ("ready", "disabled", "missing", "invalid", "regenerated")

#: 素材使用事件。candidate ≠ use：只有真进了 montage 才算用过，
#: 只有真渲染成功才算出过片，render_failed 绝不增加出片次数
DANCE_USAGE_EVENTS = ("candidate", "selected", "montage",
                      "render_success", "render_failed", "rejected")
#: 推荐策略种类
DANCE_STRATEGY_KINDS = ("cold_start", "rule_based", "history_based",
                        "exploration", "hybrid")
#: 一版混剪的渲染状态
DANCE_RENDER_STATES = ("planned", "rendering", "rendered", "failed")

DANCE_TABLES: tuple[str, ...] = (
    # --- 目标歌 -----------------------------------------------------------
    # 指纹口径和 videos.fingerprint 一样（大小 + 头/中/尾），所以换目录也认得出。
    # 节拍/段落/特征都缓存成 JSON：算一次几秒钟，但每次开界面都重算就没法用了。
    """
    CREATE TABLE IF NOT EXISTS dance_target_songs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        fingerprint      TEXT    NOT NULL UNIQUE,
        file_path        TEXT    NOT NULL,
        file_name        TEXT    NOT NULL,
        title            TEXT,
        duration         REAL,
        sample_rate      INTEGER,
        bpm              REAL,
        beat_count       INTEGER,
        beats_json       TEXT,
        sections_json    TEXT,
        features_json    TEXT,
        rhythm_json      TEXT,
        analysis_version TEXT,
        exists_on_disk   INTEGER NOT NULL DEFAULT 1,
        created_at       TEXT    NOT NULL,
        updated_at       TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_songs_path ON dance_target_songs(file_path)",

    # --- 对齐 -------------------------------------------------------------
    # cache_key = 源指纹 + 目标指纹 + 算法版本 + 配置指纹（技术指导第二十四节）。
    # 做成 UNIQUE：同一套输入重复跑直接命中，换算法版本自动重算而不是吃旧结果。
    # 算法原值（original_*）和人工值（manual_*）分列存着，禁止静默覆盖。
    """
    CREATE TABLE IF NOT EXISTS dance_audio_alignments (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        source_video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        target_song_id       INTEGER NOT NULL
                             REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        cache_key            TEXT    NOT NULL UNIQUE,
        offset_seconds       REAL    NOT NULL,
        confidence           REAL    NOT NULL DEFAULT 0,
        method               TEXT,
        waveform_offset      REAL,
        waveform_confidence  REAL,
        chroma_offset        REAL,
        chroma_confidence    REAL,
        window_count         INTEGER NOT NULL DEFAULT 0,
        max_deviation        REAL    NOT NULL DEFAULT 0,
        agreement            REAL    NOT NULL DEFAULT 0,
        status               TEXT    NOT NULL DEFAULT 'ok',
        algorithm_version    TEXT,
        config_hash          TEXT,
        source_duration      REAL,
        target_duration      REAL,
        detail_json          TEXT,
        original_offset      REAL,
        original_confidence  REAL,
        manual_offset        REAL,
        manual_reason        TEXT,
        manual_operator      TEXT,
        manual_at            TEXT,
        created_at           TEXT    NOT NULL,
        updated_at           TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_align_song "
    "ON dance_audio_alignments(target_song_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_dance_align_video "
    "ON dance_audio_alignments(source_video_id, target_song_id)",

    # --- 素材（长期资产） --------------------------------------------------
    # UNIQUE(target_song_id, source_video_id, segment_index, generation_version)：
    # 同一首歌、同一个源、同一个音乐位置、同一个切片版本，只能有一条。
    # 重切（换 slice_duration / 换算法）走新的 generation_version，旧素材标 regenerated
    # 但**留着** —— 历史成品还引用着它们。
    """
    CREATE TABLE IF NOT EXISTS dance_materials (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        source_video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        alignment_id         INTEGER
                             REFERENCES dance_audio_alignments(id) ON DELETE SET NULL,
        target_song_id       INTEGER NOT NULL
                             REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        segment_index        INTEGER NOT NULL,
        target_start         REAL    NOT NULL,
        target_end           REAL    NOT NULL,
        source_start         REAL    NOT NULL,
        source_end           REAL    NOT NULL,
        duration             REAL    NOT NULL,
        file_path            TEXT,
        file_hash            TEXT,
        alignment_confidence REAL    NOT NULL DEFAULT 0,
        generation_version   TEXT    NOT NULL DEFAULT '',
        person               TEXT,
        source_group         TEXT,
        quality              REAL,
        candidate_count      INTEGER NOT NULL DEFAULT 0,
        use_count            INTEGER NOT NULL DEFAULT 0,
        montage_count        INTEGER NOT NULL DEFAULT 0,
        output_count         INTEGER NOT NULL DEFAULT 0,
        first_used_at        TEXT,
        last_used_at         TEXT,
        last_output_at       TEXT,
        status               TEXT    NOT NULL DEFAULT 'ready',
        note                 TEXT,
        created_at           TEXT    NOT NULL,
        updated_at           TEXT    NOT NULL,
        UNIQUE(target_song_id, source_video_id, segment_index, generation_version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_mat_position "
    "ON dance_materials(target_song_id, segment_index, status)",
    "CREATE INDEX IF NOT EXISTS idx_dance_mat_source "
    "ON dance_materials(source_video_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_dance_mat_usage "
    "ON dance_materials(target_song_id, use_count, output_count)",
    "CREATE INDEX IF NOT EXISTS idx_dance_mat_person ON dance_materials(person)",

    # --- 使用事件账本 ------------------------------------------------------
    # 所有计数（use_count / montage_count / output_count）都必须能由这张表重算
    # （技术指导第九节）。dance_materials 上那几个数只是加速用的缓存。
    # 事件只追加，永不修改、永不删除 —— 这是唯一的事实来源。
    """
    CREATE TABLE IF NOT EXISTS dance_material_usage_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id  INTEGER NOT NULL REFERENCES dance_materials(id) ON DELETE CASCADE,
        montage_id   INTEGER REFERENCES dance_montages(id) ON DELETE SET NULL,
        version_id   INTEGER REFERENCES dance_montage_versions(id) ON DELETE SET NULL,
        event        TEXT    NOT NULL,
        segment_index INTEGER,
        detail_json  TEXT,
        created_at   TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_events_material "
    "ON dance_material_usage_events(material_id, event)",
    "CREATE INDEX IF NOT EXISTS idx_dance_events_version "
    "ON dance_material_usage_events(version_id, event)",
    "CREATE INDEX IF NOT EXISTS idx_dance_events_time "
    "ON dance_material_usage_events(created_at)",

    # --- 策略 -------------------------------------------------------------
    # 一份策略 = 评分权重 + 组合约束 + 搜索预算，整份可复用可版本化。
    # LLM 可以帮用户生成一份策略，但最终选择必须由确定性评分/搜索完成，
    # 所以这张表里只有数字，没有"让模型自己挑"这种口子（技术指导第二十一节第 4 条）。
    """
    CREATE TABLE IF NOT EXISTS dance_montage_strategies (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        name             TEXT    NOT NULL,
        kind             TEXT    NOT NULL DEFAULT 'hybrid',
        version          TEXT    NOT NULL DEFAULT 'v1',
        weights_json     TEXT    NOT NULL DEFAULT '{}',
        constraints_json TEXT    NOT NULL DEFAULT '{}',
        search_json      TEXT    NOT NULL DEFAULT '{}',
        is_default       INTEGER NOT NULL DEFAULT 0,
        note             TEXT,
        created_at       TEXT    NOT NULL,
        updated_at       TEXT    NOT NULL,
        deleted_at       TEXT
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dance_strategy_name_live "
    "ON dance_montage_strategies(name) WHERE deleted_at IS NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dance_strategy_default_live "
    "ON dance_montage_strategies(is_default) WHERE is_default = 1 AND deleted_at IS NULL",

    # --- 混剪任务 / 版本 ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dance_montages (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        target_song_id INTEGER NOT NULL
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        name           TEXT    NOT NULL,
        slice_duration REAL    NOT NULL DEFAULT 2.0,
        note           TEXT,
        created_at     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL,
        deleted_at     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_montage_song "
    "ON dance_montages(target_song_id, deleted_at)",

    # 「这一次混剪用了哪些**完整源视频**」。粒度是源视频，不是素材 ——
    # 素材级明细在 dance_montage_materials 里。两张都要：
    # 用户在界面上先勾"这轮用 A/B/D 三个人"，再由推荐在这三个人的素材里挑，
    # 没有这张表就没法表达"这一轮把 C 排除在外"这件事。
    """
    CREATE TABLE IF NOT EXISTS dance_montage_sources (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        montage_id   INTEGER NOT NULL REFERENCES dance_montages(id) ON DELETE CASCADE,
        video_id     INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        alignment_id INTEGER REFERENCES dance_audio_alignments(id) ON DELETE SET NULL,
        order_index  INTEGER NOT NULL DEFAULT 0,
        enabled      INTEGER NOT NULL DEFAULT 1,
        note         TEXT,
        created_at   TEXT    NOT NULL,
        UNIQUE(montage_id, video_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_msrc_montage "
    "ON dance_montage_sources(montage_id, enabled, order_index)",
    "CREATE INDEX IF NOT EXISTS idx_dance_msrc_video "
    "ON dance_montage_sources(video_id)",


    # 每次 remix 都是一个 version，历史版本不能因为重新生成而丢失
    # （技术指导第十四节）。七个重复率就存在这里。
    """
    CREATE TABLE IF NOT EXISTS dance_montage_versions (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        montage_id            INTEGER NOT NULL
                              REFERENCES dance_montages(id) ON DELETE CASCADE,
        version_index         INTEGER NOT NULL,
        signature             TEXT    NOT NULL DEFAULT '',
        strategy_id           INTEGER
                              REFERENCES dance_montage_strategies(id) ON DELETE SET NULL,
        recommendation_run_id INTEGER
                              REFERENCES dance_recommendation_runs(id) ON DELETE SET NULL,
        clip_count            INTEGER NOT NULL DEFAULT 0,
        duration              REAL    NOT NULL DEFAULT 0,
        material_repeat       REAL    NOT NULL DEFAULT 0,
        position_repeat       REAL    NOT NULL DEFAULT 0,
        person_repeat         REAL    NOT NULL DEFAULT 0,
        source_repeat         REAL    NOT NULL DEFAULT 0,
        pair_repeat           REAL    NOT NULL DEFAULT 0,
        combination_repeat    REAL    NOT NULL DEFAULT 0,
        overall_repeat        REAL    NOT NULL DEFAULT 0,
        repeat_detail_json    TEXT,
        timeline_json         TEXT,
        output_path           TEXT,
        render_status         TEXT    NOT NULL DEFAULT 'planned',
        render_detail_json    TEXT,
        error                 TEXT,
        created_at            TEXT    NOT NULL,
        updated_at            TEXT    NOT NULL,
        UNIQUE(montage_id, version_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_version_montage "
    "ON dance_montage_versions(montage_id, version_index)",
    "CREATE INDEX IF NOT EXISTS idx_dance_version_signature "
    "ON dance_montage_versions(signature)",

    # 一版混剪的编辑计划。引用 material_id 而不是文件路径 —— 素材才是资产，
    # 路径只是它现在恰好躺在哪儿（技术指导第十五节）。
    """
    CREATE TABLE IF NOT EXISTS dance_montage_materials (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id           INTEGER NOT NULL
                             REFERENCES dance_montage_versions(id) ON DELETE CASCADE,
        material_id          INTEGER NOT NULL
                             REFERENCES dance_materials(id) ON DELETE CASCADE,
        order_index          INTEGER NOT NULL,
        segment_index        INTEGER NOT NULL,
        target_start         REAL    NOT NULL,
        target_end           REAL    NOT NULL,
        source_start         REAL    NOT NULL,
        source_end           REAL    NOT NULL,
        selection_score      REAL,
        score_breakdown_json TEXT,
        created_at           TEXT    NOT NULL,
        UNIQUE(version_id, order_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_vm_version "
    "ON dance_montage_materials(version_id, order_index)",
    "CREATE INDEX IF NOT EXISTS idx_dance_vm_material "
    "ON dance_montage_materials(material_id)",

    # --- 推荐 -------------------------------------------------------------
    # 推荐必须可重现（技术指导第十一节）：seed + 策略 + 算法版本全部落库，
    # 同样的输入重跑一定得到同样的结果。随机数一律走 stable_rng(seed, *parts)，
    # 禁止用全局 random。
    """
    CREATE TABLE IF NOT EXISTS dance_recommendation_runs (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        target_song_id    INTEGER NOT NULL
                          REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        montage_id        INTEGER REFERENCES dance_montages(id) ON DELETE SET NULL,
        strategy_id       INTEGER
                          REFERENCES dance_montage_strategies(id) ON DELETE SET NULL,
        strategy_kind     TEXT    NOT NULL DEFAULT 'hybrid',
        strategy_version  TEXT,
        random_seed       INTEGER NOT NULL DEFAULT 0,
        candidate_count   INTEGER NOT NULL DEFAULT 0,
        recommended_count INTEGER NOT NULL DEFAULT 0,
        algorithm_version TEXT,
        filter_json       TEXT,
        notes             TEXT,
        created_at        TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_recrun_song "
    "ON dance_recommendation_runs(target_song_id, created_at)",

    """
    CREATE TABLE IF NOT EXISTS dance_recommendation_items (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id               INTEGER NOT NULL
                             REFERENCES dance_recommendation_runs(id) ON DELETE CASCADE,
        material_id          INTEGER NOT NULL
                             REFERENCES dance_materials(id) ON DELETE CASCADE,
        segment_index        INTEGER NOT NULL,
        rank                 INTEGER NOT NULL,
        score                REAL    NOT NULL DEFAULT 0,
        score_breakdown_json TEXT,
        reason               TEXT,
        created_at           TEXT    NOT NULL,
        UNIQUE(run_id, segment_index, rank)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_recitem_run "
    "ON dance_recommendation_items(run_id, segment_index, rank)",
    "CREATE INDEX IF NOT EXISTS idx_dance_recitem_material "
    "ON dance_recommendation_items(material_id)",

    # 用户对推荐的反馈：采纳 / 否决 / 换掉。下一次 history_based 推荐会读它。
    """
    CREATE TABLE IF NOT EXISTS dance_recommendation_feedback (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL
                    REFERENCES dance_recommendation_runs(id) ON DELETE CASCADE,
        item_id     INTEGER
                    REFERENCES dance_recommendation_items(id) ON DELETE SET NULL,
        material_id INTEGER REFERENCES dance_materials(id) ON DELETE SET NULL,
        verdict     TEXT    NOT NULL,
        comment     TEXT,
        created_at  TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_feedback_run "
    "ON dance_recommendation_feedback(run_id, verdict)",
    "CREATE INDEX IF NOT EXISTS idx_dance_feedback_material "
    "ON dance_recommendation_feedback(material_id, verdict)",
)

#: 推荐反馈的结论
DANCE_FEEDBACK_VERDICTS = ("accepted", "rejected", "replaced")

# 新建库时一次把舞蹈那几张表也建好；老库靠 migrations 的 v11 走同一批语句，
# 两条路径**用的是同一份 SQL**，不会出现"升级上来的表和新建的表不一样"。
TABLES = TABLES + DANCE_TABLES


#: v12：目标歌的"下架"记录。
#:
#: 为什么单开一张表、而不是给 `dance_target_songs` 加一列：加列只能靠
#: `ALTER TABLE ADD COLUMN`，而新建库那条路会把这一列写在 CREATE TABLE 里，
#: 于是两条路径的 `sqlite_master.sql` 不再逐字相同（v4 就是这么走散的，
#: `test_dance_migration.py` 现在专门盯着这件事）。新建一张表则天然一致 ——
#: 两条路径跑的是**同一个**建表语句。
#:
#: 顺带把"为什么下架、谁下架的、什么时候"一并记下：和人工修正 offset 一样，
#: 这个项目里任何"让东西从界面上消失"的操作都不许静默进行。
DANCE_V12_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS dance_song_retirement (
        target_song_id INTEGER PRIMARY KEY
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        reason         TEXT    NOT NULL DEFAULT '',
        operator       TEXT    NOT NULL DEFAULT '',
        retired_at     TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_song_retirement_at "
    "ON dance_song_retirement(retired_at DESC)",
)

TABLES = TABLES + DANCE_V12_TABLES


#: v13：用户拍板的段落模板（S1/S2/S3…）。同样**只建新表**，理由和 v12 一样。
#:
#: 为什么这东西必须落库：整个卡点混剪的前提是"所有源视频贴同一把尺子"。
#: 以前那把尺子是 `时长 + 格长` 两个数现算的，用户改不了；现在用户可以拖边界，
#: 那这份边界就成了必须留住的资产 —— 重启还在、下次开同一首歌还是它，
#: 否则所有素材"绑在哪一段"这件事就没有依据了。
#:
#: 只存 `spans_json`（边界 + 标签）而不是一行一段：模板是**整体**才有意义，
#: 半份模板（有缝/重叠）是非法状态，一条 JSON 存进去、读出来整体校验，
#: 天然没有"写了一半"的中间态。校验在 `dance.segment_template.validate`。
#: `source` 记它怎么来的（uniform/pause/manual），界面要能说清"这份分段是谁定的"。
DANCE_V13_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS dance_segment_templates (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        target_song_id INTEGER NOT NULL
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        name           TEXT    NOT NULL DEFAULT '',
        duration       REAL    NOT NULL DEFAULT 0,
        segment_count  INTEGER NOT NULL DEFAULT 0,
        spans_json     TEXT    NOT NULL,
        source         TEXT    NOT NULL DEFAULT 'manual',
        template_version TEXT  NOT NULL DEFAULT '',
        is_active      INTEGER NOT NULL DEFAULT 0,
        note           TEXT    NOT NULL DEFAULT '',
        created_at     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL,
        UNIQUE(target_song_id, name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_segment_templates_song "
    "ON dance_segment_templates(target_song_id, is_active DESC, updated_at DESC)",
)

TABLES = TABLES + DANCE_V13_TABLES


#: v14：人工编排的三份"用户决定"。同样**只建新表**。
#:
#: 为什么必须落库而不是留在界面里：这三样都是用户拍板的结果，关掉窗口就没了等于白干。
#:
#:   dance_final_selections   FINAL TIMELINE 上"这一段用哪条素材"。
#:                            主键是 (歌, 段落编号)：一段只能有一个最终选择。
#:                            **写入前要校验素材的 segment_index 必须等于这个段落**
#:                            （见 `material_repository.set_final_selection`）——
#:                            跨段落的选择不只在界面上拦，业务层再拦一次。
#:   dance_cut_marks          用户「⭐ 标记为可取」的时刻。它是**参考**，
#:                            不改 Segment、不改素材，只是留个记号下次好找。
#:   dance_candidate_order    每个段落里候选素材的排列顺序（拖出来的那份）。
#:                            存整份 order_json 而不是给 dance_materials 加
#:                            sort_order 列：加列会让"升级上来的表"和"新建的表"
#:                            SQL 文本不一致（v4 的老坑），而且顺序天然是**整段**的属性。
DANCE_V14_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS dance_final_selections (
        target_song_id INTEGER NOT NULL
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        segment_index  INTEGER NOT NULL,
        material_id    INTEGER NOT NULL
                       REFERENCES dance_materials(id) ON DELETE CASCADE,
        note           TEXT    NOT NULL DEFAULT '',
        selected_at    TEXT    NOT NULL,
        PRIMARY KEY (target_song_id, segment_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_final_selections_material "
    "ON dance_final_selections(material_id)",
    """
    CREATE TABLE IF NOT EXISTS dance_cut_marks (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        target_song_id INTEGER NOT NULL
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        moment         REAL    NOT NULL,
        kind           TEXT    NOT NULL DEFAULT 'usable',
        note           TEXT    NOT NULL DEFAULT '',
        created_at     TEXT    NOT NULL,
        UNIQUE(target_song_id, moment)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dance_cut_marks_song "
    "ON dance_cut_marks(target_song_id, moment)",
    """
    CREATE TABLE IF NOT EXISTS dance_candidate_order (
        target_song_id INTEGER NOT NULL
                       REFERENCES dance_target_songs(id) ON DELETE CASCADE,
        segment_index  INTEGER NOT NULL,
        order_json     TEXT    NOT NULL,
        updated_at     TEXT    NOT NULL,
        PRIMARY KEY (target_song_id, segment_index)
    )
    """,
)

TABLES = TABLES + DANCE_V14_TABLES






