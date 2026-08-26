# 本地视频事件 + 语音时间轴

Windows 11 本地离线视频理解工具：输入一个视频，输出「什么时候发生了什么」和「什么时候说了什么」的统一时间轴，时间戳可直接回跳原视频。

- 画面：Qwen3-VL-4B-Instruct（只负责"发生了什么"，不做语音识别、不负责计时）
- 语音：faster-whisper large-v3（词级时间戳，中英混说自动识别，不翻译）
- 时间轴引擎：按区间重叠把画面事件与语音段配对，相邻事件合并、跨窗口去重
- 时间戳来源显式标注：`frame_based` / `hybrid` / `model_estimated`，不伪造精度

## 最终语言

最终用户可见的自然语言由**原始音频语言**决定，由程序判定，不让模型自己选：

1. 先跑 faster-whisper，拿到 `detected_language` + 置信度，按时长加权投票得出 `dominant_language`
2. 程序确定 `output_language`，再开始视觉分析——英文音频输出英文，中文音频输出中文
3. 中文夹英文单词不会把整段切成英文（判的是主导语言，`secondary_languages` 单独记录）
4. 无音频/无有效语音 → 用 `config.json` 里的 `language.default_language`（默认中文），
   并在 `FINAL_REPORT.txt` 中明确写出 `Audio: NONE` / `Language: DEFAULT`
5. 内部结构化事实（`action` / `scene` / `subjects`）固定英文小写标签，不随输出语言变化
6. 原始对白永不被覆盖：`speech_events.json` 保留 `original_text` / `original_language`


## 环境要求

- Windows 11 + NVIDIA GPU（CUDA 12，实测 24GB 显存峰值约 10.3GB）
- Python 3.10+
- 首次运行会从国内镜像（ModelScope / 清华 / 阿里 / 南大）自动下载模型与依赖

## 使用

```bat
setup_and_test.bat   :: 建虚拟环境、装依赖、下模型、冒烟测试（无人值守）
run_auto.bat         :: 批量分析 input\ 下所有视频，产出 FINAL_REPORT.txt
run_gui.bat          :: 打开 GUI
```

命令行：

```bat
python run.py check              :: 环境自检
python run.py download           :: 只下载模型
python run.py run <视频路径>     :: 分析单个视频（--force 忽略缓存）
python run.py gui                :: 打开 GUI
```

## 输出

`output\<视频名>\` 下：

- `timeline.json` / `timeline.txt` / `timeline.srt`
- `visual_events.json` / `speech_events.json`
- `video_metadata.json` / `benchmark.json`

`timeline.txt` 的段落标签跟随 `output_language`：

```
[00:00.00 - 00:06.80] 画面：两人展示巧克力并互动 / 语音：If this is pink, Blake's not allowed to drink beer.
[00:12.00 - 00:17.00] Visual: A man eats a chocolate egg while a woman smiles at the camera.
```

[00:00.00 - 00:06.80] 画面：两人展示巧克力并互动 / 语音：If this is pink, Blake's not allowed to drink beer anymore.
[00:12.00 - 00:17.00] 画面：男子吃巧克力蛋，女子微笑看镜头
```

## GUI

左侧播放器、右侧事件时间轴、底部语音文本与运行日志；点击时间轴行或语音行即跳转。支持重要性与置信度过滤，分析在子进程中运行、日志实时回传。

注意：内置播放器只有画面没有声音——Qt Multimedia 在本机无法解码 H.264，因此改用 OpenCV 逐帧渲染，定位是帧级精确的。

## 其他

- 模型只加载一次并复用；阶段级与窗口级断点续跑（已完成的窗口不重算）
- 长视频按窗口 + 重叠切分，不会只分析前 N 秒
- 无音轨时 `speech = null`，任务仍然成功
- 只使用官方模型仓库，镜像仅用于加速下载
