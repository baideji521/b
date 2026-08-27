# 失败与已修复问题报告

Benchmark 本身没有失败项：4 个模型 × 2 个配置 × (1 cold + 2 正式) = 24 次运行全部 OK，
OOM 0 次，降级 0 次，重试 0 次。

下面记录的是本轮**在真实源码里查出并修掉的 Bug**，以及**测量后被否决的方案**。

## 1. 帧时间戳全部塌成 0（影响所有后端，最严重）

`video_io.sample_frames` 在 `cap.set(CAP_PROP_POS_FRAMES, idx)` 之后、`cap.read()` 之前
读取 `CAP_PROP_POS_MSEC`。OpenCV 在 seek 之后这个值是垃圾（实测第 43 帧返回 2.79ms），
导致所有帧的时间戳都变成 0.0~0.02s。

这条是整个项目"时间从真实帧来"的地基：视觉事件的 start/end 全靠它。

修法：`read()` 之后再读 POS_MSEC，并减去一帧时长（read 之后它指向的是**下一帧**的时间），
不可用时回退 `idx / fps`。

```python
next_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
this_msec = next_msec - frame_msec
ts = max(this_msec, 0.0) / 1000.0 if (next_msec > 0.0 and this_msec > -1.0) else pos_frames / info.fps
```

已用 `tests/test_frame_sampling.py::test_timestamps_are_real` 锁住：
单调递增、跨度 > 窗口的 60%、与 `idx/fps` 误差 < 0.15s。

## 2. `plan_frame_indices` 参数顺序在部分调用点写反

签名是 `(..., min_frames, max_frames)`，有调用点按 `(max, min)` 传，
结果 `max_frames` 改成 6/8/12/16 时实际帧数不变。

已修全部调用点，并用两条测试锁住：
`test_signature_order`（签名顺序）、`test_call_sites_use_min_then_max`（AST 扫调用点）、
`test_max_frames_takes_effect`（6/8/12/16 → 实际 6/8/12/16 帧）。

## 3. `VisualParams.degrade()` 的 max_new_tokens 卡在 384，降不下去

原实现每次降级都从同一个上限重算，实际永远到不了 256/192/128，
OOM 时"降级"是无效动作，而且不留痕迹。

改成**三轴轮转阶梯**，每次只动一条轴，并把每一步写进 `degrade_history`：

- 轴顺序：`max_frames` → `resolution` → `max_new_tokens`
- token 阶梯：512 → 384 → 256 → 192 → 128（下限 128）
- 帧数下限 6，单帧像素预算下限 64 token
- 原对象不被修改（返回新实例），降级过程可追溯、可打日志

`test_degrade_reaches_128` 断言 5 级 token 全部被访问到且能收敛。

## 4. MiniCPM-V 4.6 无法输出合法嵌套 JSON（1.3B 模型能力限制）

4.6 有两个硬限制：不能稳定输出嵌套 JSON，也不会从帧推算秒数。
第一版提示词让它抄浮点时间戳，它直接把模板占位符回吐出来
（`person|woman|points|scene|visible_text`），解析结果 0 条事件。

改成 `frame_lines` 格式：`帧号|谁在画面|在做什么|屏幕上的文字`，
时间完全由程序按帧号 → 真实帧时间戳换算。配套加了占位符黑名单
（`person` / `action` / `场景` / `画面文字` 等）和三级兜底
（帧号优先 → 时间戳对齐 → 行序）。

4.6 现在能出事件了，但仍然把人数认成 1（真值 2），事件切得过碎
（配置 A 44 条、配置 C 24 条，重复率 0.30 / 0.04），所以不做默认。

## 5. MiniCPM-V 4.5-int4 默认 num_beams=3

`model.chat(sampling=False)` 在官方实现里等价于 `num_beams=3`，
耗时和显存都是 3 倍，与 Qwen 的 greedy 不可比。已显式传 `num_beams=1`。
（即使如此，4.5-int4 仍是最慢的：配置 C 103.1s，RTF 2.10。）

## 6. CLI `--backend` 选项漏了 minicpm46

`--backend` 的 choices 是手写字符串列表，新加的 `minicpm46` 不在里面，
命令行根本无法选到。已改成 `["auto", *BACKENDS]`（`run` 和 `download` 都改）。

## 7. 幻觉指标不可信（详见 quality_report.md）

旧算法子串匹配，8 个组合全部 1.0，没有区分度。已改成分类判定 + 例外 + evidence。

## 8. 被测量否决的方案

- **人物检测 + 跟踪**：4B 本来人数就全对（18/18 次人数众数 2、一致性 1.0），
  开启后模型把提示词回吐进事件名，命名变差。已删除相关代码。详见 quality_report.md。
- **提高分辨率**：480×640 起显存 11.8GB，576×768 起溢出共享内存、耗时涨到 8 倍，
  而 OCR 在任何分辨率下都读错。不上调。
- **SupoClip / VideoMind**：只借鉴思路（窗口切分、时间定位），不引入代码和依赖。
- **`RESOLUTION_RECOMMENDATION.txt` 的 "推荐 800x1088" 结论**：已删除该文件。
  它的 quality_score 存在自参考偏差（参考分辨率就是 800×1088 自己），
  而实际 RTF 23.6，在这台机器上完全不可用。原始测量数据保留在 `benchmark/resolution_sweep.json`。
