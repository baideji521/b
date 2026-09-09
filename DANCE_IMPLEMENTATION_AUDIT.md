# Dance 子系统实施审计（DANCE_IMPLEMENTATION_AUDIT）

审计对象：当前仓库 `baideji521/b` 实际源码（不是设计文档、不是聊天记录）。
审计日期：2026-09-10。审计后本轮已落地的修改列在第 5 节。

四个参考项目的源码是**实际拉取 raw 文件读过的**，函数名与常量都从源码抄录，
没有凭记忆编造。许可证逐个核对过，结论见第 3 节 —— 其中两个项目**不允许复制代码**。

---

## 1. 当前 Dance 实际文件清单

`src/vidscribe/dance/`（20 个模块，5568 行）

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `__init__.py` | 40 | 三个算法版本常量：`ALIGNMENT_ALGORITHM_VERSION` / `MATERIAL_GENERATION_VERSION` / `RECOMMENDATION_ALGORITHM_VERSION` |
| `types.py` | 784 | 23 个 dataclass，不 import 任何重模块（av/cv2/db），GUI 和测试可安全导入 |
| `dsp.py` | 476 | 纯 numpy DSP：STFT / mel / MFCC / chroma / onset / 自相关 / tempo / `beat_track` / `cross_correlate` |
| `audio_fingerprint.py` | 238 | 音频解码、`file_fingerprint`、`audio_fingerprint`、`config_hash`、`alignment_cache_key`、`peak_ratio_confidence` |
| `audio_align.py` | 311 | `correlate_waveform` / `correlate_chroma` / `chroma_support` / `align_arrays` / `find_alignment` |
| `alignment_validation.py` | 272 | 阈值常量 + `window_plan` / `robust_offset` / `method_agreement` / `combine_confidence` / `decide_status` / `manual_override` |
| `music_structure.py` | 501 | `detect_target_beat_grid` / `analyze_target_music_features` / `analyze_rhythm_bands` / `analyze_music_sections` / `target_positions` / `analyze_song` |
| `material_slice.py` | 210 | `map_to_source` / `plan_slices` / `coverage` / `material_filename` / `render_material` / `render_plan` / `SourceRangeError` |
| `media_backend.py` | 566 | `Canvas` / `MediaBackend` 抽象 / `PyAVBackend` / FFmpeg 骨架 / `is_complete_video` |
| `material_repository.py` | 626 | 歌 / 对齐 / 素材 / 混剪 / 版本 / 策略 / 推荐的全部 CRUD |
| `material_ingest.py` | 460 | `register_song` / `align_source` / `align_batch` / `slice_and_register` |
| `history.py` | 253 | 事件账本：`note_candidates` / `note_montage` / `note_render` / `recount_*` / `position_usage` / `event_summary` |
| `material_score.py` | 295 | 19 个权重 / `static_score` / `dynamic_delta` / `combination_signature` / `explain` |
| `material_selection.py` | 285 | `build_query`（全占位符）/ `find_materials` / 七套 `PRESETS` / `build_pool` / `build_pools` |
| `strategy.py` | 184 | `DEFAULT_CONSTRAINTS` / `DEFAULT_SEARCH` / 四套预设 / `resolve` / `ensure_presets` |
| `recommendation.py` | 213 | `weights_for` / `rank_pool` / `recommend` / `reproduce` / `items_for` |
| `combination_search.py` | 272 | `stable_rng` / `violates` / `search`（beam）/ `search_versions` |
| `montage_timeline.py` | 249 | `build_timeline` / `repeat_of` / `from_manual` / `validate` / `save` / `load` |
| `montage_render.py` | 274 | `render`（只执行定稿计划）/ `remix`（编排）/ `output_name` |
| `statistics.py` | 301 | 七个重复率 / `song_overview` / `position_coverage` / `person_breakdown` |

`src/vidscribe/gui/dance_montage/`（11 个文件，2174 行）：`__init__`(launch) /
`main_page`(DanceMontageWindow) / `worker`(DanceMontageWorker，九阶段) /
`alignment_panel` / `filter_panel` / `material_library` / `candidate_panel` /
`recommendation_panel` / `remix_panel` / `history_panel` / `statistics_panel`。

**没有** `pipeline.py` / `workflow.py` / `context.py` / `gui.py` —— 也不需要：
编排在 `montage_render.remix()` 里，GUI 在独立包里。这是有意的，不打算再加一层"大 Pipeline"。

## 2. 数据库真实状态

`SCHEMA_VERSION = 11`（`src/vidscribe/db/schema.py`）。
`migrations.py` 的 `_STEPS[11] = list(DANCE_TABLES)` —— v11 **只有 CREATE，没有一条
ALTER/DROP**，所以新建库和升级库的建表 SQL 逐字相同（历史上 v4 用 `ADD COLUMN` 带不上
`REFERENCES`，两条路径外键不等价，这个坑有测试盯着不许再犯）。

12 张 `dance_*` 表 + 约 20 个索引：`dance_target_songs` / `dance_audio_alignments`
（`cache_key` UNIQUE）/ `dance_materials`（UNIQUE(target_song_id, source_video_id,
segment_index, generation_version)）/ `dance_material_usage_events` /
`dance_montage_strategies` / `dance_montages` / `dance_montage_sources` /
`dance_montage_versions`（UNIQUE(montage_id, version_index)）/ `dance_montage_materials` /
`dance_recommendation_runs` / `dance_recommendation_items` / `dance_recommendation_feedback`。

源视频身份复用主项目的 `videos` 表（外键指过去），**没有第二个 SQLite**。

## 3. 参考项目对照表（函数名均从源码抄录）

### 3.1 许可证结论（决定了能不能抄代码）

| 项目 | 许可证 | 能否复制代码 | 本项目做法 |
| --- | --- | --- | --- |
| sanjeed5/audio-video-sync | MIT | 可以（保留版权声明即可） | **仍然没抄**：它用 scipy + librosa，本项目是纯 numpy 独立实现 |
| Merserk/BeatSync-Engine | AGPL-3.0 | 不可以（会传染整个项目） | 只吸收算法思想，独立实现 |
| interrupt21h/tubeviz | Apache-2.0 | 可以（需保留声明） | 只吸收资产库设计思想，未复制代码 |
| mfahsold/montage-ai | PolyForm Noncommercial 1.0.0 | **不可以**（非开源，禁商用） | 只作为架构分层的旁证，未复制任何代码 |

### 3.2 audio-video-sync → 本项目对齐层

| 参考文件 | 参考函数 | 当前文件 | 当前函数 | 状态 |
| --- | --- | --- | --- | --- |
| `src/audio_video_sync/sync.py` | `_correlate_raw` | `dance/audio_align.py` | `correlate_waveform` | 已有，等价且更强 |
| 同上 | `_correlate_chroma` | 同上 | `correlate_chroma` + `chroma_support` | 已有，多了支持度判据 |
| 同上 | `_peak_ratio_confidence` | `dance/audio_fingerprint.py` | `peak_ratio_confidence` | 已有，口径不同（见下） |
| 同上 | `_extract_audio_ffmpeg` | `dance/audio_fingerprint.py` | `extract_analysis_audio` | 已有，走 PyAV 不起子进程 |
| 同上 | `find_offset` | `dance/audio_align.py` | `align_arrays` / `find_alignment` | 已有，多窗口 |
| `src/audio_video_sync/ffmpeg.py` | `merge` | `dance/media_backend.py` | `PyAVBackend.mux_audio` | 已有 |

逐条对齐结论：

- **互相关**：它 `scipy.signal.correlate(a, b, mode="full")`，本项目 `dsp.cross_correlate`
  用 FFT 手写（`irfft(rfft(a) * conj(rfft(b)))` 再拼成"负 lag 在前"）。两者数学等价，
  本项目快一个量级且不依赖 scipy。
- **符号口径**：它的 `offset > 0` = 音乐在视频里更晚出现；本项目钉死
  `source_time = target_time - offset`，靠 `cross_correlate(目标歌, 源音轨)` 的峰值 lag
  直接得到，**全系统只有这一个口径**（`DanceAlignment.source_time()` 是唯一换算入口）。
- **去均值**：两边都做（本项目在 `dsp.cross_correlate` 里 `a -= a.mean()`）。
- **极性容错**：它 `argmax(np.abs(correlation))`；本项目 `peak_of(np.abs(corr))`。一致。
- **置信度**：它返回**裸比值**（可能 `inf`），阈值 2.0；本项目映射成 `1 - 1/ratio` 压到 0~1，
  再乘一个门限 `min(1, peak_quality / 0.15)`。有界的好处是能和"多窗口一致度""两法互证"
  加权成一个分，界面上也能当百分比显示。
- **多窗口**：**它没有**（单个 40 秒窗口）。本项目 5 个窗口覆盖头/中/尾 + 等距补齐，
  取中位数做 `robust_offset`，最大偏差超 0.5s 直接判 rejected。这是本项目相对它的实质增强。
- **边缘伪峰**：它按方向分别用两个文件的长度做上限；本项目原来只拿源时长做**对称**上限
  —— 这是本轮查出并修掉的真 bug，见第 5 节 F1。

### 3.3 BeatSync-Engine → 本项目音乐分析层

| 参考文件 | 参考函数 | 当前文件 | 当前函数 | 状态 |
| --- | --- | --- | --- | --- |
| `src/auto_mode/stage1_audio.py` | `detect_master_beat_grid` | `dance/music_structure.py` | `detect_target_beat_grid` | 已有，独立实现（不用 librosa） |
| `src/auto_mode/stage2_features.py` | `analyze_wave_features` | 同上 | `analyze_target_music_features` | 已有 |
| 同上 | `analyze_rhythm_bands` | 同上 | `analyze_rhythm_bands` | 已有，频带划分不同（见下） |
| 同上 | `_normalize`（2/98 分位） | `dance/dsp.py` | `robust_normalize01` | **本轮新增**（见 F4） |
| `src/auto_mode/stage3_sections.py` | `analyze_sections` / `classify_section` | `dance/music_structure.py` | `analyze_music_sections` / `classify_music_section` | 已有 |
| `src/auto_mode/stage4_select.py` | `compute_cut_scores` / `adaptive_beat_step` | — | — | **不采用**（见下） |
| `src/auto_mode/stage6_av_planner.py` | `_choose_candidate` | `dance/material_score.py` + `combination_search.py` | `static_score` / `dynamic_delta` / `search` | 已有，且是长期历史版 |
| `src/auto_mode/stage5_qwen_scene_worker.py` | 全部 | — | — | **明确不采用** |

- **节拍**：它用 `librosa.beat.beat_track(onset_envelope=..., start_bpm=120, tightness=120)`；
  本项目手写 Ellis 2007 动态规划（`dsp.beat_track`，`TIGHTNESS=100`）+ 对数正态先验
  （`PRIOR_BPM=120`）。**它没有真正的 downbeat 检测**（`is_bar_anchor = [::4]` 是几何假设），
  本项目的 `downbeats = beats[::4]` 是同一个诚实的近似，文档里写明了。
- **频带**：它 kick 35–145 / bass 35–220 / clap 150–4200 / hihat ≥4200 —— kick 完全落在
  bass 里，clap 宽到覆盖整个中频。本项目用 kick 20–120 / bass 60–250 / clap 1500–4000 /
  hihat 6000–14000，**故意不照抄**：四条带要能互相区分才有意义。
- **归一化**：它统一用 2/98 分位裁剪拉伸，这一点比 min-max 稳健得多，本轮采纳（F4）。
- **段落**：它 `librosa.segment.agglomerative(k=clip(duration/32, 3, 8))` + novelty 峰补充，
  最短段 10 秒；本项目用 chroma+MFCC 的新颖度峰 + 聚类，最短段 6 秒（卡点舞的歌普遍更短）。
  段落类型两边都是九类左右，本项目多一个 `prechorus`，少 `hook`/`finale`。
- **stage4 不采用的理由**：它的选点是"在音乐里挑剪切点"，本项目的位置是**固定的**
  （`target_positions` 只做除法），要挑的是"这个位置放谁"。而且它 `stage4` 无任何跨运行历史
  （报告原文：不读写任何磁盘状态），本项目的核心恰恰是长期使用历史。
- **一个值得记下的事实**：它 `stage4` **完全没有随机**（全文无 `random`/`seed`），
  这和"BeatSync 是随机选素材"的传闻不符 —— 随机只出现在 stage6 无候选时的兜底采样。
  本项目 `stable_rng` 只用于打破平分，同样不做随机选择。

### 3.4 tubeviz → 本项目素材资产层

| 参考文件 | 参考类/函数 | 当前文件 | 当前函数 | 状态 |
| --- | --- | --- | --- | --- |
| `src/tubeviz/library.py` | `ClipLibrary` | `dance/material_repository.py` | 整个模块 | 已有，思想一致 |
| 同上 | `ClipRecord` | `dance/types.py` | `DanceMaterial` | 已有，本项目多了四个计数 |
| 同上 | `SceneCandidate` | `dance/types.py` | `MaterialScore` + `CandidatePool` | 已有 |
| 同上 | `sha256_file` | `dance/audio_fingerprint.py` | `file_fingerprint` | 已有，头/中/尾 1MB + sha1 |
| 同上 | `find_by_original_sha256` / `mark_duplicate` | `dance/material_repository.py` | `upsert_material` 的唯一键 | 部分（见下） |
| 同上 | `set_clip_trim` / `usable_start`/`usable_end` | `dance/types.py` | `source_start` / `source_end` | 已有 |
| 同上 | `reject_clip` / `restore_clip` / `status` | `dance/material_repository.py` | `set_material_status` | 已有 |
| `src/tubeviz/beat_warp.py` | 全部 | — | — | **明确不采用**（变速会改动作节奏） |

- 它也是 **SQLite + WAL**、也用 `AUTOINCREMENT` 主键 + 业务唯一键、也把 JSON 当列存
  ——和本项目的做法撞了个正着，说明这条路是对的。
- **它没有任何使用计数**（报告原文：schema 里没有 `use_count`/`last_used_at`，
  `stats()` 也不统计"这条素材出现在几个成片里"）。本项目的
  `candidate_count` / `use_count` / `montage_count` / `output_count` +
  `dance_material_usage_events` 事件账本是**本项目独有**的部分，也正是"避免永远重复
  同一批素材"这个需求的地基。
- **重复检测**：它靠文件 sha256 精确去重 + `duplicate_of_clip_id` 指向 canonical。
  本项目目前靠 `(歌, 源, 位置, 切片版本)` 唯一键防重复生成，`file_hash` 列已经在写，
  但**还没有跨源的"两个不同源视频切出了一模一样的画面"检测**。这是已知缺口，
  见第 6 节 —— 没有为它提前重构存储，因为当前唯一键已经挡住了真正会发生的重复。
- **它把路径存成相对库根**（`str(path.relative_to(self.root))`），整库搬目录不会失效。
  本项目存绝对路径，靠 `status='missing'` + 重新切片修复。这是**有意的取舍**：
  素材目录在配置里可改，相对化之后"配置改了但库没动"会更难解释。

### 3.5 montage-ai → 本项目分层与渲染

| 参考文件 | 参考类 | 当前文件 | 当前对应物 | 状态 |
| --- | --- | --- | --- | --- |
| `src/montage_ai/core/context.py` | `MontageContext` | `dance/types.py` | `DanceMontageContext` | 已有，更小更专 |
| 同上 | `MontageResult` | `dance/types.py` | `RenderResult` | 已有 |
| 同上 | `ClipMetadata` | `dance/types.py` | `DanceMontageClip` | 已有 |
| `src/montage_ai/core/workflow.py` | `VideoWorkflow.execute` 的七阶段 | `dance/montage_render.py` + `gui/dance_montage/worker.py` | `remix()` + `STAGES` 九阶段 | 已有 |
| `src/montage_ai/core/render_engine.py` | `RenderEngine.render_output` | `dance/montage_render.py` | `render()` | 已有 |
| `src/montage_ai/segment_writer.py` | 最终 mux（`-map 0:v -map 1:a`） | `dance/media_backend.py` | `PyAVBackend.mux_audio` | 已有，独立实现 |
| `src/montage_ai/vlm_clip_selector.py` | LLM 选片 | — | — | **明确不采用** |

- **音频处理两边完全同构**：它在每一步都加 `-an` 剥掉源音频，最后一次
  `-map 0:v -map 1:a -c:a aac -shortest` 把音乐 mux 进去。本项目素材切片阶段就无声，
  最后 `mux_audio` 用视频流拷贝 + 新编 AAC + 按画面时长截音轨。**独立实现，结论一致**
  —— 这条路径是对的。
- **它的 context 没有 error 字段**，任一 stage 抛异常整个 workflow 直接 FAILED，
  且异常路径下 `cleanup()` 不会执行。本项目 `DanceMontageContext.notes` +
  `SlicePlan.skipped` + `render_failed` 事件把失败**记进账本**，单条素材失败不中断整批。
- **它没有真正的断点续跑**（只有"输出文件已存在就跳过"+ 分析缓存）。本项目靠
  对齐缓存 + `render_plan(skip_existing=True)` + `upsert_material` 幂等实现真续跑，
  本轮补了测试（F5）。
- 顺带记一条：它 `montage_builder.py:260` 引用了 `context.py` 里不存在的
  `timeline.clip_sequence`，任何路径都会 `AttributeError`。这类"上下文字段散落各处"的
  故障正是本项目把 `types.py` 收成单一定义处的原因。

---

## 4. 真实调用链（以代码为准）

```
CLI: run.py → cli.main → cmd_dance_montage → _dance_dispatch
GUI: run_kadian.bat / 主界面「AI_卡点舞」按钮 → gui/dance_montage/launch
       → DanceMontageWindow → DanceMontageWorker.run()

两条入口最终都调同一批函数，没有第二套实现：

  material_ingest.register_song
        ↓  audio_fingerprint.extract_analysis_audio → music_structure.analyze_song
        ↓  material_repository.upsert_song / save_song_analysis
  material_ingest.align_batch                      （ThreadPool 4~6，只算不写库）
        ↓  audio_align.find_alignment → align_arrays
        ↓      dsp.cross_correlate（波形）+ correlate_chroma / chroma_support
        ↓      alignment_validation.window_plan / robust_offset / combine_confidence
        ↓      alignment_validation.decide_status
        ↓  material_repository.save_alignment      （主线程，保留人工 offset）
  material_ingest.slice_and_register
        ↓  music_structure.target_positions        （位置只由歌长 + 切片时长决定）
        ↓  material_slice.plan_slices → map_to_source（越界抛 SourceRangeError）
        ↓  material_slice.render_plan → render_material → media_backend.PyAVBackend
        ↓  material_repository.upsert_material     （幂等，不动计数）
  montage_render.remix
        ↓  material_selection.score_context / build_pool   （预取上限来自配置）
        ↓  history.note_candidates                          （candidate 事件）
        ↓  recommendation.recommend                          （种子落库）
        ↓  combination_search.search_versions → search → violates / dynamic_delta
        ↓  montage_timeline.build_timeline → repeat_of → save → history.note_montage
        ↓  montage_render.render → media_backend.render_spans → mux_audio
        ↓  history.note_render                               （render_success/failed）
        ↓  statistics.song_overview / position_coverage      （界面与 CLI 共用）
```

---

## 5. 本轮实际改的代码

### F1 `alignment_validation.decide_status` —— 边缘伪峰上限按方向分

原来：`abs(offset) >= source_duration * 0.95` 一律判 rejected。
问题：200 秒的歌 + 20 秒的源，源合法地对在歌的第 19 秒上（offset=+19），
`19 >= 20*0.95` 成立 → **正确结果被判成伪峰**。短素材越容易踩。
改法：`offset >= 0` 时上限用**目标歌**时长，`offset < 0` 时用**源**时长；
不传 `target_duration` 时退回旧行为。参考 audio-video-sync 的方向性上限思路。
测试：`test_dance_audio_align.py::test_edge_limit_is_direction_aware`。

### F2 `material_selection._ORDER_SQL["score_desc"]` —— 候选池不再按入库顺序砍

原来：`score_desc` 档的 SQL 是 `ORDER BY m.id ASC ... LIMIT 500`。
问题：得分是内存里算的，SQL 排不了，所以这条 ORDER BY 决定**哪 500 行会被留下**。
某个位置的素材一超过上限，留下的就是"库里最早的 500 条"，
新切进来、一次没用过的素材在打分之前就被悄悄扔掉 —— 正是"候选池过早截断"。
改法：预取顺序改成 `output_count ASC, use_count ASC, alignment_confidence DESC, id ASC`
（最该被考虑的优先），并加 `DEFAULT_POOL_LIMIT = 500` 常量。
测试：`test_dance_selection.py::test_candidate_pool_is_not_truncated_by_insertion_order`
（改之前这条测试会失败）。

### F3 候选池大小配置化

`config.py` 新增 `dance.candidate_pool_size = 500`（注释说明它是**预取**上限，
不是最终选几条；真正进 beam search 的是 `candidate_k`）。
`montage_render.remix(pool_size=...)` 新增参数并把它写进 `FilterSpec.limit`，
还会打一行日志说明"预取上限多少、实际取到多少"。
CLI 新增 `--pool N`；GUI 的候选池改用 `_pool_spec()`（配置值），
不再借用素材库列表那个"最多显示 300 行"的显示上限。

### F4 `dsp.robust_normalize01` —— 特征曲线改用分位数归一化

min-max 归一化会被单个爆音毁掉：一首歌里有一下削波，整条能量曲线被压到 0.1 以下，
`position_match` 评分全变成 0。新增 `robust_normalize01`（2/98 分位裁剪后拉伸），
用在 `energy_wave` / `brightness` / 频带曲线上。
**故意不用在 onset 包络上** —— 节拍跟踪要的正是那些尖峰，裁掉最高 2% 会更难跟拍。
思想来自 BeatSync 的 `_normalize`，独立实现（AGPL 项目不复制代码）。

### F5 补齐三类缺测试

- `test_dance_material.py::test_resume_only_redoes_what_is_missing`
  断点续跑：重跑不重编（比对 `st_mtime_ns`）、不产生新素材行；删掉一条只补那一条。
- `test_dance_selection.py::test_history_heavy_material_ranks_below_fresh_one`
  用烂的素材必须排在全新素材之后，且 breakdown 里能看出扣分来源。
- `test_dance_selection.py::test_soft_penalties_punish_same_person_and_same_source`
  同人/同源连着出现要**扣分**（软惩罚），和硬约束 `violates` 分开验。

### F6 `montage_timeline.validate` 增加源区间校验

历史版本重渲染这条路上，库里的行有可能被人手改过、或是旧算法切的。
新增三项检查：`source_start` 为负、源区间首尾颠倒、源时长和成片格子对不上（差 >0.05s）。

### F7 `material_slice.render_plan` 不再裸吞异常

`except OSError: pass`（删不完整残片失败）→ 改成 `logger.debug` 留痕。
删不掉不影响正确性（下一轮还会判它不完整并重渲），但"磁盘满/文件被占用"必须查得到。

### F8 `montage_render.describe` —— `--plan-only` 不再报"渲染失败"

只出计划时 `RenderResult` 是空的，原来会被 describe 成"失败：未知原因"，
现在明确说"N 格已定稿，本次没有渲染"。

---

## 6. 审计发现但**故意不改**的地方

| 项 | 结论 | 理由 |
| --- | --- | --- |
| `audio_align.py` 的互相关与置信度 | 保持不变 | 已经正确：去均值、abs 峰、护栏次峰、5 窗口中位数。比参考项目更强 |
| `material_repository.upsert_material` | 保持不变 | 已经按 (歌,源,位置,切片版本) 幂等，且**不动计数**，重切不清历史 |
| `montage_render.render` | 保持不变 | 输入已经是定稿 Timeline，层内不推荐/不选择/不随机；音轨硬验收 `audio_streams==1` |
| 频带 Hz 划分 | 保持本项目的 | BeatSync 的 kick⊂bass、clap 150–4200 太糊，四条带要能分开 |
| 素材路径存绝对路径 | 保持不变 | tubeviz 存相对路径更耐搬家，但本项目素材目录在配置里可改，相对化会让"配置改了库没动"更难解释 |
| 跨源重复画面检测（感知哈希） | 不做 | tubeviz 也只做字节级 sha256 去重。本项目唯一键已挡住真正会发生的重复；感知去重要引入新依赖，不值 |
| `cache.py` 复用 | 不改 | 对齐结果落 `dance_audio_alignments`（带 `cache_key` 唯一键）比落文件缓存更符合"DB 是唯一事实来源"；`cache.py` 是给分析产物用的，两者不混 |
| FFmpeg CPU/NVENC 后端 | 仍是骨架 | 第一版 PyAV 够用；显式指定不可用后端会**报错**而不是悄悄降级 |
| 人脸聚类自动分人 | 不做 | 会引入大模型，违背"不为 Dance 引入 AI GPU 模型" |

---

## 7. 测试结论

见 `AI_卡点舞_实施报告.md` 第 O 节与本轮终端输出。本轮新增/修改的测试：
`test_dance_audio_align.py`（17 项）、`test_dance_selection.py`（13 项）、
`test_dance_material.py`（8 项）。全部 27 个套件通过，其中 18 个是原有套件。
