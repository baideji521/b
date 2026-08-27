# 最终推荐（按实测数据，不按论文 / 参数量 / README）

- 视频：`test.mp4` 810×1080（3:4）49.167s 30fps 英文有声
- 机器：Windows 11 + RTX 3060 12GB，driver 591.86，torch 2.8.0+cu126
- 质量权威：`benchmark/ground_truth.json`（人工看帧标注）
- 打分权重：稳定性 30% / 速度 30% / 时间定位 20% / 画面理解 10% / 显存 10%

## 一、逐条结论

- **最快**：Qwen3-VL-4B-Instruct / 配置C —— 视觉 41.1s（benchmark 里 batch=2）；
  换成最终配置 batch=4 后 **38.6s**（三次全流程实测 38.556 / 38.559 / 38.847）
- **最稳**：Qwen3-VL-4B-Instruct / 配置C —— 稳定性 1.000。
  3 次正式运行事件数 [4,4,4]、人数众数 [2,2,2]、描述一致度 1.000；
  最终配置再跑 3 次全流程，4 条事件的起止时间、事件名、subjects 完全逐字一致
- **画面理解最好**：Qwen3-VL-4B-Instruct —— 人数众数 2（真值 2）、人数一致性 1.000、
  幻觉率 0.000、覆盖率 0.999。4.6 覆盖率相近但把两个人认成一个，事件切成 24~44 条碎片
- **人物处理最好**：Qwen3-VL-4B-Instruct —— **18 次实测全部人数正确且一致**。
  2B 在配置 C 掉到 1 人，4.6 在两个配置下都只认出 1 人
- **时间定位最好**：Qwen3-VL-4B-Instruct / 配置C —— 帧校准比例 1.000、时间戳合法率 1.000。
  最终配置下 4 条事件全部 `timestamp_source=hybrid`，覆盖 0.00~49.10s（视频 49.167s）
- **显存最低**：MiniCPM-V-4.6 —— 峰值 3770MB（两个配置都一样）。
  但它只认出 1 个人、事件碎片化，不能作默认
- **英文视频最合适**：Qwen3-VL-4B-Instruct —— 语言匹配 1.000，事件名/描述全英文，
  与"最终语言跟随原始音频语言"的策略一致（Whisper 判定 en，置信度可用）
- **RTX 3060 12GB 综合第一 / 建议默认**：**Qwen3-VL-4B-Instruct / 配置C + batch=4**
  —— 总分 0.8991，视觉 RTF 0.79，Total RTF 1.12，峰值 9650MB（离 12GB 有余量）

## 二、排名（总分，benchmark 口径 batch=2）

1. Qwen/Qwen3-VL-4B-Instruct / 配置C 0.8991（稳定 1.000｜速度 1.000｜时间 1.000｜理解 0.600｜显存 0.391）41.1s / 9636MB
2. Qwen/Qwen3-VL-4B-Instruct / 配置A 0.8785（稳定 1.000｜速度 0.947｜时间 1.000｜理解 0.587｜显存 0.356）43.4s / 10580MB
3. Qwen/Qwen3-VL-2B-Instruct / 配置C 0.8499（稳定 1.000｜速度 0.835｜时间 1.000｜理解 0.200｜显存 0.794）49.3s / 4746MB
4. openbmb/MiniCPM-V-4.6 / 配置C 0.8410（稳定 1.000｜速度 0.709｜时间 1.000｜理解 0.282｜显存 1.000）58.0s / 3770MB
5. Qwen/Qwen3-VL-2B-Instruct / 配置A 0.7586（稳定 1.000｜速度 0.425｜时间 1.000｜理解 0.600｜显存 0.712）96.8s / 5298MB
6. openbmb/MiniCPM-V-4.6 / 配置A 0.7326（稳定 1.000｜速度 0.388｜时间 0.955｜理解 0.254｜显存 1.000）106.0s / 3770MB
7. openbmb/MiniCPM-V-4_5-int4 / 配置C 0.7176（稳定 1.000｜速度 0.399｜时间 1.000｜理解 0.545｜显存 0.435）103.1s / 8670MB
8. openbmb/MiniCPM-V-4_5-int4 / 配置A 0.6937（稳定 1.000｜速度 0.323｜时间 1.000｜理解 0.586｜显存 0.381）127.2s / 9884MB

四个模型 × 两个配置 × (1 cold + 2 正式) = 24 次运行，OOM 0，降级 0，重试 0。

## 三、最终配置表（已写入 config.json）

- 视觉模型：`Qwen/Qwen3-VL-4B-Instruct`（backend `qwen3vl`）
- 采样 FPS：`0.75`
- 每窗口帧数：`max_frames 8` / `min_frames 6`
- 分辨率：`max_pixels_tokens 112`（288×384）/ `total_pixels_tokens 2048`
- max_new_tokens：`192`
- 窗口：`15.0s`，重叠 `3.0s`
- Batch（一次 generate 处理几个窗口）：`4`
- dtype：`bfloat16`
- Attention：`sdpa`
- Tracking：**关闭**（代码已删除，实测无收益且有害）
- OCR：**不做专门优化**（`ocr_text` 字段保留但不参与打分；四个模型召回率均为 0）
- ASR：`faster-whisper large-v3`，`float16`，备选 `medium` / `small`
- 帧来源：`frame_source "opencv"`（内存中 OpenCV → PIL → 模型，不落盘 JPG）
- 输出语言：跟随原始音频语言（本视频 en）

备选（低显存场景）：`Qwen/Qwen3-VL-2B-Instruct` + 配置A（峰值 5.3GB，人数正确，但 96.8s）。
不推荐 2B + 配置C：快但人数错、覆盖率 0.565。

## 四、OOM 降级阶梯（`VisualParams.degrade()`，每步都写进 degrade_history）

`batch_size` → `max_frames`（下限 6）→ `resolution`（下限 64 token/帧）→
`max_new_tokens`（512 → 384 → 256 → 192 → 128，下限 128）。
每次只动一条轴，返回新对象（原配置不被污染），降级原因与历史都会落日志。
本轮 24 次 benchmark + 3 次全流程共 27 次运行，`degrade_attempts` 全部为 0。

## 五、验收清单（最终配置，test.mp4 连续 3 次全流程）

- [x] 3 次运行全部 exit 0，wall 59.9 / 59.8 / 59.9 s
- [x] 产物齐全：`timeline.json` / `timeline.srt` / `timeline.txt` /
      `visual_events.json` / `speech_events.json` / `video_metadata.json` / `benchmark.json`
- [x] 事件数一致：4 / 4 / 4
- [x] 事件内容逐字一致（起止时间、事件名、subjects 三次完全相同）
- [x] 时间戳全部合法：单调、不重叠、都在 0~49.167s 内，无 None
- [x] 视觉时间来自真实帧：4 条事件 `timestamp_source` 全部 `hybrid`（帧号 + fps 换算），
      覆盖 0.00~49.10s
- [x] 人数正确：每条事件 subjects 都是 man + woman（真值 2 人），三次一致
- [x] 幻觉率 0.000（按 ground_truth 的分类判定，无 locomotion / shot_change / extra_person 冲突）
- [x] 语音时间轴正确：20 段，首段 0.00~3.24s，末段 47.04~48.50s，SRT 20 个块
- [x] 语言正确：detected=en，original=en，output=en（英文视频出英文）
- [x] 显存：峰值 reserved 9650MB < 12287MB，未溢出共享内存
- [x] 稳定性：generated_tokens 三次都是 616，prompt_tokens_max 都是 882
- [x] OOM 0 次，降级 0 次
- [x] 回归测试 `tests/test_frame_sampling.py` 6/6 通过
- [x] 分阶段耗时全部记录：probe 1.7 / speech 14.7~15.0 / visual 38.6~38.8
      （其中 frame_decode 4.2、chat_template 0.03~0.05、processor 0.1、generate 23.0~23.2、
      text_decode 0.0）/ timeline 0.003 / total 54.9~55.3
- [x] RTF：视觉 0.79，ASR 0.30，Total 1.12

## 六、明确不做的事

- 不接入 SupoClip、不接入 VideoMind（只借鉴窗口切分与时间定位思路，不引入代码和依赖）
- 不做人物检测 / ByteTrack（实测无收益，详见 `quality_report.md`）
- 不提高分辨率（480×640 起显存 11.8GB，576×768 起溢出共享内存，OCR 仍读错）
- 不把 VLM 当 OCR 用
- 不改写 pipeline / Whisper / 场景检测 / Timeline 结构，不改 Event schema
