# AI_卡点舞 实施报告

按技术指导第二十六节的 A~O 逐条交代。标记含义：
`已实现并测试` / `已实现未测` / `部分实现` / `未做（原因）`。

一期 → 二期 → 三期的要求全部落在同一套代码里（分期只是需求文档的组织方式，
不是三套实现）。原项目的功能、GUI、数据库表**一个都没动**：
27 个测试套件全绿，其中 18 个是原有套件。

---

## A. 范围与边界　`已实现并测试`

做了：目标歌音乐分析 → 源视频音频对齐 → 固定音乐位置切片 → 舞蹈素材资产库
→ 筛选/评分/推荐 → 组合搜索 → 编辑计划 → 渲染 → 多版本混剪 → 使用历史与统计。

明确**没有**做（技术指导第二十五节的"不许吸收"清单）：
- BeatSync 的 `stage5_qwen_scene_worker.py` 那套 LLM 选镜
- tubeviz 的 beat-warp / 变速对齐（会改动作速度，卡点舞不能要）
- BeatSync 的随机选素材
- montage-ai 的"让大模型直接挑"
- 参考项目各自的第二套数据库 / 缓存 / 进度系统

## B. 与原项目的关系　`已实现并测试`

- 没有改 `pipeline.py`、`highlight/montage.py`、`AnalyzeWorker` 一行
- 没有建第二个 SQLite：全部落在现有 `video.db`，新增 12 张 `dance_*` 表
- 源视频复用主项目的 `videos` 表做身份（外键指过去），不另建一套视频库
- 主界面只加了一个按钮（`btn_dance` → `on_dance_montage`），核心逻辑没碰
- `tests/test_dance_gui.py::test_worker_is_isolated_from_analyze_worker`
  静态断言 `DanceMontageWorker` 和 `AnalyzeWorker` 没有任何继承关系

## C. 数据库设计　`已实现并测试`

`SCHEMA_VERSION = 10 → 11`，v11 **只建新表，一条 ALTER/DROP 都没有**：

`dance_target_songs` / `dance_audio_alignments` / `dance_materials` /
`dance_material_usage_events` / `dance_montage_strategies` / `dance_montages` /
`dance_montage_sources` / `dance_montage_versions` / `dance_montage_materials` /
`dance_recommendation_runs` / `dance_recommendation_items` /
`dance_recommendation_feedback`，另加约 20 个索引。

迁移纪律：`_STEPS[11] = list(DANCE_TABLES)`，所以**新建库和升级库的建表 SQL 逐字相同**
（历史上 v4 用 `ADD COLUMN` 带不上 `REFERENCES`，两条路径外键不等价 —— 这个坑没再踩）。
`tests/test_dance_migration.py` 13 项全绿，含"v11 里没有任何 DROP/ALTER"的静态检查、
两条路径 schema 逐字比对、外键级联、四个唯一约束、`db_admin.health_check` 通过。

## D. 音频对齐　`已实现并测试`

口径全系统只有一个，写在 `DanceAlignment.source_time()` 里：

    source_time = target_time - offset

实现成 `cross_correlate(目标歌, 源音轨)` 的峰值 lag ÷ 采样率，所以从头到尾没有一次符号翻转。

- 波形互相关 + chroma 互相关双路，5 个窗口（头/中/尾 + 等距补齐）交叉验证
- chroma 那一路用的是**支持度**而不是"两法 offset 相等"：周期性和声会让 chroma 的
  全局峰按循环周期别名，有意义的问题是"chroma 在波形给出的 offset 处有没有能量"
- 置信度是**门限**不是天花板：`confidence = 加权分 × min(1, 峰值质量 / 0.15)`。
  节奏性音乐天生有 ±一拍的强旁瓣（连自己对自己都只有 0.17 的峰比），
  给峰值质量设上限会把最强的证据扔掉，所以用门限压垮垃圾、让多窗口一致性说话
- 人工修正必须写理由（`manual_override` 无理由直接 `ValueError`），
  `original_offset` 只在第一次写入，算法原值永远留着
- 缓存键 = 源指纹 + 目标指纹 + 算法版本 + 配置哈希；并发 4 个 worker（上限 6），
  **worker 只算不写库，落库全在主线程**

`tests/test_dance_audio_align.py` 16 项全绿：0 / 1.0 / 3.25 / 7.5 秒延迟都能还原，
负偏移能还原，无关噪声被拒（置信度掉到阈值下），边缘偏移被拒，短源拿不到高置信度。

## E. 音乐分析　`已实现并测试`

**故意不用 librosa**（它拖 numba + llvmlite 一整条链），手写约 450 行 numpy DSP：
STFT / HTK mel 滤波器组 / 正交 DCT-II 的 MFCC / MIDI 音高类投影的 chroma /
谱通量 onset / FFT 自相关 / Ellis 2007 动态规划节拍跟踪。

Tempo 估计踩过一个坑并修好了：自相关的短 lag 天然占便宜（重叠多），
不做去锥形修正会一路往 60 BPM 掉。修法是除以重叠计数 + 120 BPM 对数正态先验 + 倍频加分。
实测 90→89.1、120→117.5、128→129.2、150→152.0，都在 hop 分辨率内。

段落分 9 类、四条节奏频带（kick/bass/clap/hihat）、逐帧能量/亮度/冲击曲线。
静音或极短音频走三级兜底，**永远不返回空网格**（`_synthetic_grid`，`confidence=0` 明说是编的）。

## F. 固定音乐位置切片　`已实现并测试`

这是本项目和普通随机混剪的分水岭：位置是**目标歌的属性**，
`target_positions(歌长, 每格时长)` 只做除法，不看任何素材；所有源都贴在同一把尺子上。
末尾装不满一整格的丢掉，绝不留碎片。

**越界不 clamp**：源开头不够或结尾不够，`map_to_source` 抛 `SourceRangeError`，
`plan_slices` 把这些位置记进 `SlicePlan.skipped` 并写中文理由。clamp 会悄悄产出
一段和音乐错位的画面，而错位是这类项目最难查的 bug。

素材落地是原子的（写 `.part` → 完整关闭 → `os.replace`，跟 `highlight/clip.py` 一致），
素材本身**无声**（目标歌才是唯一音轨）。

## G. 媒体后端　`已实现并测试`

`MediaBackend` 抽象 + `PyAVBackend`（可用）+ FFmpeg CPU/NVENC 骨架（显式指定不可用的后端会
**报错**而不是悄悄换）。全系统**没有一处 `subprocess.run(["ffmpeg", ...])`**。

两处归一化是多源拼接的前提（`highlight/clip.py` 做不到的正是这个）：
- 画布：scale-to-cover + 居中裁切到 1080×1920
- **帧率**：输出第 k 帧取源时间 `start + k/fps_out`，绝不用顺序 `cap.read()` 数帧
  （30fps 源读进 25fps 输出会把动作快放 20%）

实测 160×120@24fps + 90×160@30fps → 180×320@30fps，正好 6.000 秒。

## H. 素材资产模型　`已实现并测试`

素材是**长期资产**：界面和 CLI 都只有"停用"，没有删除。重切是新增一个
`generation_version`，旧素材标 `regenerated` 但行和文件都留着（历史成品还引用它们）。
状态：`ready / disabled / missing / invalid / regenerated`。

## I. 计数与事件账本　`已实现并测试`

`dance_material_usage_events` 只增不改，是**唯一事实来源**。四个计数各管一件事，不许顶包：

- `candidate_count` ← `candidate`（进过候选池 = 被考虑过）
- `use_count` / `montage_count` ← `montage`（进了某一版编辑计划）
- `output_count` ← `render_success`（那一版真的渲染成功了）

**`render_failed` 一次都不推 `output_count`** —— 测试里连失败 3 次，出片数仍为 0。
`history.recount_song` 能把被人手改花的计数完全按流水重建，且幂等。
位置级统计 `position_usage` 分别记 `(位置, 素材)`：一条素材用了 8 次但全在第 3 格，
第 5 格用它仍然算新的。

## J. 筛选与评分　`已实现并测试`

一期的 A~G 七套方案全在 `material_selection.PRESETS`。
查询**全部走占位符**（人物名和搜索词来自用户输入），测试直接拿 SQL 注入串当人名试。
`long_unused_days` 包含 `last_used_at IS NULL` —— 漏掉这一半会出现
"筛长期未使用，结果最该用的全新素材一条都不出现"。

评分拆成两半，这是 beam search 可行的前提：
- **静态分**（19 个因子）只看素材自己 + 历史 + 音乐位置，每个候选算一次
- **动态分**只看邻居，每次 beam 扩展时算

修掉一个 bug：F 方案（指定人物）的预设里 `persons: ()` 会把调用方填好的人名清空，
现在空元组当占位符处理，不再覆盖 base。

## K. 组合搜索　`已实现并测试`

Top-K + beam search，预算封顶（`candidate_k=8, beam_width=6, max_nodes=20000`），
**没有全排列**，Windows + RTX 3060 上跑得动。

硬约束**拒绝**（`violates`）：连续同人、同素材重复、人物/来源占比、相邻对历史次数、
整套组合相似度。软偏好扣分（`dynamic_delta`）。两者分开是有意的 —— 混在一起
就会出现"扣够多分就等于禁止"这种说不清的行为。

可复现靠 `stable_rng(seed, *parts)`（sha1 内容寻址，不依赖调用顺序），
所以迭代顺序变了结果也不变。同种子重跑逐格一致，多版本的组合签名互不相同。

## L. 推荐　`已实现并测试`

种子落库、算法版本落库，`reproduce(run_id)` 当场重跑对比 —— CLI 有
`recommend --verify <run_id>`，界面上有「验证可复现」按钮。抖动按策略类型分级
（cold_start/rule_based 0，history_based 0.02，hybrid 0.05，exploration 0.35）。

**LLM 可以生成策略，但最终选择一定由确定性评分/搜索完成** —— `strategy.py` 里
只有数字和阈值，没有任何"让模型自己挑"的口子。

## M. 编辑计划与渲染　`已实现并测试`

分层严格：Selection → Timeline → Render。渲染层不推荐、不选择、不对齐、不查历史，
只把一份定稿计划变成能播的 MP4。

- 成片时间**连续铺**（有空档就整体前移），素材原本的位置身份记在 `segment_index` 里
- 引用 `material_id` 而不是文件路径（素材才是资产，路径只是它现在躺在哪儿）
- 历史版本永不覆盖：`version_index` 自增 + 唯一约束挡着
- 三步固定：素材片段 → 无声成片 → mux 目标歌；成品硬验收
  `audio_streams == 1 and video_streams == 1`，不过就走失败路径

smoke test 实测：10 秒目标歌 + 3 个源 + slice=2s → 5 个位置 → 出片 10.000s，
144×256@24，音轨正好 1 条（目标歌），无声中间文件已清。

## N. CLI 与界面　`已实现并测试`

CLI 新增 `dance-montage`，九个动作：
`songs / align / slice / materials / recommend / remix / stats / history / gui`。
返回码沿用全局约定（0 成功 / 1 业务失败 / 2 参数不对）。
原有 12 个子命令一个没少，参数解析照旧（有测试盯着）。

界面是**独立窗口** `src/vidscribe/gui/dance_montage/`（10 个文件），四个区域：

- ① 输入与操作：`remix_panel.py`（切片预设 1.0/1.5/2.0/2.5/3.0/自定义、
  一个主行动按钮、九阶段进度、日志）
- ② 素材资产：`filter_panel.py` + `material_library.py`
- ③ 选择与推荐：`candidate_panel.py` + `recommendation_panel.py`
- ④ 历史与统计：`history_panel.py` + `statistics_panel.py`
- 另有 `alignment_panel.py`（对齐结果 + 人工修正）、`main_page.py`、`worker.py`

`DanceMontageWorker` 九个阶段：扫描素材 / 音频提取 / 音乐对齐 / 生成切片 /
建立素材库 / 准备混剪 / 渲染 / 封装 / 完成。它**自己开自己的库连接**
（`Database` 是每线程一条连接），停止是协作式的（阶段边界才停，不会写坏文件）。
主线程绝不 import cv2（会改写 `QT_QPA_PLATFORM_PLUGIN_PATH` 把 QApplication 搞崩 0xC0000409）。

关掉「智能推荐」之后仍能纯手动出片：候选池面板逐格钉选 → `remix(recommend_enabled=False,
manual={位置: 素材id})`。没有手动选择就点开始会被拦住，而不是跑出个空片。

启动方式：`run_kadian.bat`、`python run.py dance-montage gui`，
或主界面第一行的「AI_卡点舞」按钮。

**一处规格冲突的处理**：一期说界面做成现有 GUI 的一页，用户原始需求和技术指导都说独立窗口。
选了独立窗口（满足用户 + 技术指导），另外从主界面加一个入口按钮（满足一期），
主窗口核心逻辑没碰。

## O. 测试　`已实现并测试`

27 个套件全绿（18 个原有 + 9 个新增）：


新增套件：
- `test_dance_migration.py` 13 项 —— 迁移安全、双路径一致、外键、唯一约束
- `test_dance_audio_align.py` 16 项 —— 偏移还原、置信度门限、人工修正
- `test_dance_music_structure.py` 9 项 —— tempo 不掉八度、位置只由歌决定、段落铺满
- `test_dance_material.py` 7 项 —— 换算口径、越界不 clamp、真渲染无声素材、重切不删旧
- `test_dance_selection.py` 10 项 —— 占位符、七套方案、静态/动态分、硬约束、搜索可复现
- `test_dance_history.py` 8 项 —— 四个计数、失败不推出片、按流水重算、版本不覆盖
- `test_dance_smoke.py` 3 项 —— 端到端真出片、纯手动路径、失败记账
- `test_dance_cli.py` 5 项 —— 新命令整链 + 原有命令没被挤掉
- `test_dance_gui.py` 9 项 —— 四区域、面板铺真数据、切片预设、工人线程整条路、协作式停止

原有 18 个套件全部照旧通过（`SCHEMA_VERSION` 的硬编码 `== 10` 改成了引用常量，
以后升 v12 也不会再被打断）。

跑法：`python tests/test_dance_smoke.py`（本仓库 venv 里没装 pytest，
所有测试都遵循项目的双入口约定，直接 `python` 跑即可）。

### 审计补丁（2026-09-10）

对照 audio-video-sync / BeatSync-Engine / tubeviz / montage-ai 的**真实源码**
重新审计了一遍现有实现，查出并修掉 8 处问题，明细见 `DANCE_IMPLEMENTATION_AUDIT.md`。
其中两处是会真的出错的 bug：

- 对齐的"边缘伪峰"上限原来两个方向都拿源时长比，会把"短源合法地对在长歌后段"
  判成伪峰（200 秒歌 + 20 秒源、offset=+19 就中招）。改成按方向取上限。
- `score_desc` 档的候选池预取 SQL 原来是 `ORDER BY m.id ASC LIMIT 500`，
  某个位置素材超过上限时留下的是"最早入库的 500 条"，新切的、一次没用过的素材
  在打分之前就被丢掉。改成按"最该被考虑"预取，并把上限做成 `dance.candidate_pool_size`。

pytest 现已装进 venv（`pytest==8.3.4`），全量结果：**445 passed, 2 skipped**（149 秒）。
那 2 个 skip 是原有的：`conftest.py` 在仓库根缺 `test.mp4` 时跳过真实解码用例，与 Dance 无关。


---

## 已知限制

- FFmpeg CPU / NVENC 后端只有骨架，第一版实际可用的是 PyAV。
  显式指定不可用后端会报错，不会悄悄降级。
- 人工修正偏移之后，已经切好的素材**不会自动重切**（界面上有明确提示），
  需要重新切片才会用上新偏移。
- 人物标注目前靠手填或按源视频名带入，没有做人脸聚类自动分人。
- 素材少的时候硬约束会让组合填不满所有位置（例如只有 2 个源 + 
  `max_person_share=0.45`，5 格最多填 4 格）。这是约束在正常工作，
  日志和统计面板都会把"哪些位置没填上"说清楚，放宽约束或多切几个源即可。
