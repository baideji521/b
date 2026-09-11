"""配置加载：默认值 + config.json 覆盖 + 命令行覆盖。"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "paths": {
        "input_dir": "input",
        "output_dir": "output",
        # 集中管理用的视频库根目录（递归扫）：设了它，缓存管理就知道哪些缓存还有对应视频，
        # 不在库里的（视频删了/搬走了）会被标出来，可以一键清掉。留空＝不做这个判断
        "video_dir": "",
        # 缓存根目录（断点、窗口缓存、预览音轨），见 vidscribe/cache.py
        "cache_dir": "cache",
        # SQLite 库放哪（库文件固定叫 video.db）：缓存状态、分析结果、AI 任务都记在里面
        "db_dir": "database",
        "log_dir": "logs",
        "model_dir": "models",
    },
    "visual": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "fallback_model_ids": ["Qwen/Qwen3-VL-2B-Instruct"],
        # auto = 按 model_id 猜后端；也可显式写 qwen3vl / minicpm
        "backend": "auto",
        # GUI / CLI 可切换的模型清单（label 只影响界面显示）
        "models": [
            {"label": "Qwen3-VL-4B-Instruct (默认)", "model_id": "Qwen/Qwen3-VL-4B-Instruct",
             "backend": "qwen3vl"},
            {"label": "Qwen3-VL-2B-Instruct (更省显存)", "model_id": "Qwen/Qwen3-VL-2B-Instruct",
             "backend": "qwen3vl"},
            {"label": "MiniCPM-V-4.5 int4 (8.7B/4bit)", "model_id": "openbmb/MiniCPM-V-4_5-int4",
             "backend": "minicpm"},
            {"label": "MiniCPM-V-4.6 (1.3B/BF16, 隔离依赖 tf57)", "model_id": "openbmb/MiniCPM-V-4.6",
             "backend": "minicpm46"},
        ],
        # MiniCPM 专属参数：3D-Resampler 把 packing_nums 帧压成 64 token
        "minicpm": {
            "packing_nums": 1,
            "max_slice_nums": 1,
            "use_image_id": False,
            # 官方 sampling=False 默认 num_beams=3（3 倍耗时/显存），压回 1 才和 Qwen 可比
            "num_beams": 1,
        },
        # 4.6 换成 downsample_mode（4x 更细/16x 更省）+ window merger，没有 temporal_ids
        "minicpm46": {
            "downsample_mode": "16x",
            "max_slice_nums": 1,
            "use_image_id": False,
            "stack_frames": 1,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        # opencv | official | auto
        # 默认 opencv：官方 qwen-vl-utils 在 Windows 上走 torchvision 后端，
        # 它返回的 frames_indices 是窗口内相对帧号，且整段解码内存开销大；
        # 自己做内存帧采样可以拿到原视频绝对帧号（时间戳更可靠），且长视频不爆内存。
        "frame_source": "opencv",
        # 以下默认值来自 2026-08-27 在 RTX 3060 12GB 上的实测（benchmark/speed_report.md）：
        # 49s 3:4 英文视频，4 个窗口，Qwen3-VL-4B。
        # fps/帧数再往上加只增加耗时，事件数不增加；再往下减会把事件合并成一条。
        "fps": 0.75,
        "max_frames": 8,
        "min_frames": 6,
        # 单帧 / 全部帧的像素预算，单位是 token（1 token = 32*32 像素）
        # 112 -> 288x384。实测 336(512x672) 峰值 11.6GB、448(576x768) 峰值 12.7GB 会溢出到
        # 共享内存，耗时从 42s 涨到 115s，而 OCR 仍然读错，所以不上调。
        "max_pixels_tokens": 112,
        "total_pixels_tokens": 2048,
        "max_new_tokens": 192,
        # 一次 generate 同时处理几个窗口：单步解码受 CPU/launch 开销主导，
        # 实测 batch 1/2/4 -> 43.4s / 41.9s / 27.7s，batch=4 峰值 9.7GB 仍在 12GB 内。
        "batch_size": 4,
        "window_seconds": 15.0,
        "window_overlap_seconds": 3.0,
        "long_video_threshold": 18.0,
        "scene_detect": True,
        "scene_threshold": 0.35,
        "scene_sample_fps": 3.0,
        "snap_tolerance_seconds": 1.0,
        "min_event_seconds": 0.4,
        # 合并后的事件上限：再像也不连成一条 36 秒的"一直在吃"，动作轨得留分辨率
        "max_event_seconds": 12.0,
        "merge_similarity": 0.82,
        "dedup_similarity": 0.72,
        # 画面情绪：视觉模型在同一次推理里顺便判人物情绪，只多两个输出字段，不额外加载模型
        "emotion_enabled": True,
        # 人脸表情：在原始帧上单独跑 YuNet + HSEmotion，覆盖视觉模型给的情绪
        # （视觉模型那边人脸只有 70~85 像素、一个窗口只看 8 帧，判表情不可靠）
        "face_emotion": {
            "enabled": True,
            "sample_fps": 2.0,
            "detect_size": 640,
            "detect_score": 0.6,
            "min_face_px": 60,
            "max_faces": 2,
            "min_score": 0.35,
        },
    },
    "speech": {
        "model_size": "large-v3",
        "fallback_model_sizes": ["medium", "small"],
        "device": "auto",
        "compute_type": "float16",
        "language": None,
        # 只跑这几种语言的视频（语言预检那一步就判，不跑完整识别）。
        # 手动分析：语言不在这里面就终止并弹窗；自动剪辑：跳过并把视频标记成以后不再跑
        # （videos.blocked_language）。留空 = 谁都跑，不拦
        "allowed_languages": ["en", "zh"],
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "emotion": {
            # 语音情绪识别（FunASR emotion2vec+），按 whisper 的句子边界逐段判
            "enabled": True,
            "model_id": "iic/emotion2vec_plus_large",
            "fallback_model_ids": ["iic/emotion2vec_plus_base"],
            "device": "auto",
            "batch_size": 8,
            "min_segment_seconds": 0.3,   # 更短的段声学特征不够，不判
            "top_k": 3,                   # 每段保留概率最高的几类
            "peak_top_n": 5,              # 给高光剪辑推荐几个冻帧点
            "peak_min_intensity": 0.5,    # 情绪强度低于此不算高光候选
        },
    },
    "language": {
        # 最终自然语言由原始音频语言决定；这里只配置兜底与判定门槛
        "default_language": "zh",       # 无音频/无语音时使用
        "min_language_confidence": 0.4,  # 低于此置信度时改用默认语言
        # 描述语种与 output_language 不符时，用视觉模型做一次文本改写
        "rewrite_mismatch_with_model": True,
    },
    "timeline": {
        "min_overlap_seconds": 0.2,
        "importance_filter": "low",
        "confidence_filter": 0.0,
    },
    "highlight": {
        # 成品一律无音效：原声只到剪辑区间结束，冻帧段是静音。
        # 末帧冻结几秒（0 = 不冻）：AI 报的 timeline.duration 已经把这几秒算进去了，
        # 剪辑区间只按 segments[0].sa → end 走，冻帧在渲染时加回去。
        # 例：sa/end 跨度 7.64 + 冻帧 2 = 成片 9.64，duration 报的也是 9.64
        "freeze_tail_seconds": 2.0,
        # 剪辑引擎（highlight/clip_engine.py）**总开关**，默认关。
        # 开：用库里的逐词时间戳把 AI 的粗区间修到词 / 整句边界（起点可能前移、结束可能
        # 延到整句说完），成品不断句，代价是区间和 AI 给的 sa/end 不再一致。
        # 关：完全按 AI 的 segments[0].sa → end 剪，只做合法性校验和「不超出视频时长」，
        # 所以成品名和「提取数据」的 duration 就等于 AI 那个区间长度。
        # 全局生效：手动剪辑、AI 自动剪辑、CLI 的 highlight / assets --render 都读这一个键
        # （CLI 的 --no-engine 可以在开着的时候临时关掉这一次）。
        "clip_engine": False,
        # 区间**内部**那些「没人说话干等着」的静音，渲染时最多留这么多秒（0 = 不剪，老行为）。
        # 这个数是「剪完之后**留多少**」，不是「剪掉多少」：填 0.8 表示一处 3.8 秒的静音
        # 剪成 0.8 秒，剪口取静音正中间，前后各留一半 —— 话音刚落不至于被掐断。
        # 这个数也决定「多小的缝不算静音」：短于它的空隙一刀都不剪。所以填大了等于没开
        # （词与词之间的缝大多不到 1 秒，填 2.0 时它们全都留着，成品照样很长）。
        # **不要填到 clip_engine.GAP_FLOOR（0.5）以下**：压完之后每处静音只剩这么多，
        # 比 0.5 还短的话，第二轮就一个能插 TTS 旁白的空隙都找不到了。
        # 静音长在区间的开头、中间还是结尾一视同仁，位置全由词级时间戳推出来，
        # 不需要谁指定在哪，也不需要额外开关：默认 0.8 就是默认生效。
        # 全局生效：手动剪辑、AI 自动剪辑、CLI 三条路读同一个键，成品口径只有一份。
        # 和 freeze_tail_seconds 互不影响：冻结是往成片**后面加**几秒静帧，
        # 这个是往区间**里面剪**，一个加一个减，各算各的。
        "silence_keep": 0.8,
        # 成品时长上限，单位秒（0 = 不设上限）。填了就按这个数倒推该剪多少静音：
        #   跨度没超 -> 一刀不剪；超了 -> 找**最大**的「静音留多少」，剪到刚好压进来；
        #   连 silence_keep 那么紧都压不进去 -> 就按 silence_keep 剪到最紧，成品仍会超
        #   一点（那说明说话内容本身就有这么长，再往下剪得动区间，不是静音这一层的事）。
        # 不退回更松的档是有意的：那样会出现「目标定得越紧、成品反而越长」的怪事。
        # 所以 silence_keep 在这个模式下是「最紧敢留多少」，不再是固定值。
        # 全局生效：手动剪辑、AI 自动剪辑、CLI、提取数据读同一个键
        "silence_target": 0.0,
    },

    # 视频资产中心自己的两个目录：和 AI 面板的 bridge.ai_input_dir / ai_output_dir
    # **完全分开**，谁也不覆盖谁。中心只用它们扫盘登记，不参与自动剪辑的排队口径
    "assets": {
        # 扫原始视频：这个目录里的 mp4 登记进库，成为中心列表里的「视频」
        "input_dir": "",
        # 扫高光成品：这个目录里的 mp4 认成已有成品，挂回对应视频
        "output_dir": "",
        # 列表的两个目录筛选（手动选，选完即存，重启还在；点「清掉筛选」会一起清掉）：
        # 扫描目录下常常有几十个子目录，得能只看其中一个
        "filter_video_dir": "",
        "filter_product_dir": "",
    },

    # 舞蹈素材资产 + 音乐对齐 + 自动混剪（见 vidscribe/dance/，schema v11 起）。
    # 和高光那一套**完全分开**：目录、库表、GUI 都是独立的，互不影响。
    "dance": {
        # 目标歌所在目录（用户往里丢 mp3/wav/m4a）
        "song_dir": "dance/songs",
        # 源舞蹈视频所在目录（递归扫）
        "source_dir": "dance/sources",
        # 切出来的素材落哪儿。素材是**长期资产**，这个目录只增不减；
        # 它刻意不在 cache/ 下面 —— 「缓存管理」一键清空绝不能把素材库删掉
        "material_dir": "dance/materials",
        # 混剪成品落哪儿
        "output_dir": "dance/output",
        # 固定音乐位置的长度（秒）。改它等于换一套素材版本：旧素材会被标 regenerated
        # 但**保留**，因为历史成品还引用着它们
        "slice_duration": 2.0,
        # 对齐并发。技术指导要求 4~6、不要 100：每个 worker 都在做几百万点的 FFT，
        # 开太多只会互相抢内存带宽
        "align_workers": 4,
        # 单个验证窗口的长度与个数。5 个窗口覆盖开头/中间/结尾 + 等距补齐，
        # 挡住「只看前 40 秒就相信结果」这个坑
        "window_seconds": 20.0,
        "window_count": 5,
        # 置信度低于此的对齐不切片（0 = 都切，由 status 自己判）。
        # 想严一点就填 0.55（= alignment_validation.LOW_CONFIDENCE）
        "min_confidence": 0.0,
        # 首段 / 尾段**允许缺多少秒**（0 = 差一点就整段不要）。
        # 很多录屏素材开头少半秒、结尾早停一秒，于是首段映射到源的负数、尾段超过源时长，
        # 那两段就一条素材都没有。给了余量之后这两头改成取交集：源夹到 [0, 时长]，
        # 缺的那一截切片时用**边界帧**补足 —— 素材时长仍然精确等于段落长度，
        # 成片不会因此前移、音乐不漂，代价只是那一截画面是静帧。
        # 默认 2 秒：足够吃下常见的"开头缺一点 / 结尾早停一点"。补的量再多也过不了
        # material_slice.MAX_PAD_SHARE（一格里静帧不许超过一半）
        "slice_head_room": 2.0,
        "slice_tail_room": 2.0,
        # 输出画布与帧率。**默认跟素材走**（canvas_auto）：素材是 1080×1440 就出
        # 1080×1440，一个像素都不裁。关掉 canvas_auto 才用下面这套固定尺寸，
        # 那时所有源被 scale-to-cover + 居中裁切归一过去（比例不同就会裁掉边）
        "canvas_auto": True,
        "canvas_width": 1080,
        "canvas_height": 1920,
        "canvas_fps": 30.0,
        # 媒体后端：auto / pyav / ffmpeg_cpu / ffmpeg_nvenc。第一版只有 pyav 可用；
        # 显式指定不可用的后端会**报错**而不是悄悄换（见 media_backend.resolve）
        "media_backend": "auto",
        # 组合搜索预算。保证 Windows + RTX 3060 本地跑得动，绝不做全排列
        "candidate_k": 8,
        "beam_width": 6,
        "max_search_nodes": 20000,
        # 每个音乐位置**预取**多少条素材进候选池再打分。这不是最终选几条，
        # 而是"允许算法考虑多少条" —— 调小了会在打分之前就把新素材扔掉
        # （候选池过早截断），所以默认放宽。真正进 beam search 的是 candidate_k
        "candidate_pool_size": 500,

        # 一次 remix 生成几个版本（多版本混剪）
        "versions_per_run": 3,
        # 智能推荐总开关。关掉之后 GUI 仍然可以纯手动选素材（技术指导第二十节）
        "recommend_enabled": True,
        # 选文件用系统自带的对话框（默认开）。它认得"快速访问/最近使用/网盘"，
        # 找素材比 Qt 自绘那个顺手得多。
        # 关掉就退回 Qt 自己画的：万一系统 shell 扩展（缩略图、网盘、杀软插件）
        # 把对话框拖死，这里是唯一的退路 —— 那种卡死卡在系统代码里，日志上看不到
        "native_dialogs": True,
    },


    "bridge": {

        # 浏览器扩展对接（见 vidscribe/bridge/server.py）：GUI 起一个只监听
        # 127.0.0.1 的小 HTTP 服务，扩展轮询领任务、驱动网页版 AI、回传 JSON
        "enabled": True,
        "port": 5998,
        # 端口被占时往后顺延几个（扩展那边也按这个范围探测）
        "port_fallbacks": 9,
        # 任务类型标识，扩展按它筛任务（名字是历史遗留，两家提供方都走这一种）
        "task_type": "gemini_json",
        # 找哪家 AI：gemini / deepseek。接口直连和网页版扩展都看这个
        "provider": "gemini",
        "ai_url": "https://gemini.google.com/app",
        # DeepSeek 的键单独一节，Gemini 的还平铺在 bridge 下（老配置照旧能跑）
        "deepseek": {
            "api_key": "",
            "api_key_env": "DEEPSEEK_API_KEY",
            "api_model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "ai_url": "https://chat.deepseek.com/",
        },
        # AI 自己的输入/输出目录，跟 GUI 的「导入文件」「导出目录」互不相干。
        # 留空＝按老规矩来：合并导出落 cache/，AI 自动剪的成品落导出目录
        "ai_input_dir": "",
        "ai_output_dir": "",
        # 「自动剪辑」按钮干哪一串（GUI 的 AI 选项里选）：
        # full   剪辑成片：扫 AI_输入目录，库里没有分析结果就先分析，再把 PRM + 完整剧本
        #        发 AI，回来的高光 JSON 入库后按主界面高光配置出片，落 AI_输出目录
        # collect 收取高光 JSON：同上，但高光 JSON 入库就算干完，不剪
        # script 高光 JSON 剪辑：只用**库里已有的**高光 JSON 开剪，一次 AI 都不问
        # analyze 只解析视频：本地分析入库就算完，不问 AI 不剪辑（语音 / 视觉 / 表情
        #        按主界面配置跑，结果进库这一条就干完了，AI_输出目录不会多东西）
        "ai_job": "full",
        # 自动剪辑这一轮挑哪些视频（AI 面板里选）：
        # all      全部：库里已有高光方案的直接开剪，没有的问 AI
        # existing 只挑已经有高光方案的——这一档一次 AI 都不调
        # missing  只挑还没有方案的，全部走 AI
        "highlight_source": "all",
        # 「不跑成品」（AI 面板里那个勾选框，默认勾上）：成品库里已经有这个视频的有效成品
        # 就整条跳过，不重新分析、不重新问 AI、不重新剪。取消勾选＝已有成品也照样重跑一遍
        "skip_done_products": True,
        # 发 AI 时用哪一份 PRM 档案（prm_profiles.id）。0 = 按 prompt_file 那条老路找
        "prm_id": 0,


        # 高光筛选提示词：相对项目根目录。这份和合并导出都是当附件上传给网页版 AI
        "prompt_file": "prm/prm_en.txt",
        # 合并导出临时落在项目根目录，任务结束就删（想留档改成 true）
        "keep_merged_file": False,
        # 两个 txt 上传后跟着发的那句话。规则都在 prm_en.txt 里，这里只说清干什么
        "message": ("Follow the rules in prm_en.txt and analyze the attached "
                    "*_merged.txt. Reply with the JSON object only."),
        # AI 回了可用 JSON 就直接按它剪，不再等我点一次「剪辑高光」
        "auto_clip": True,
    },
    "runtime": {
        "max_auto_retries": 3,
        "keep_models_loaded": True,
        # 语音跑完就释放 whisper 显存再加载视觉模型（12GB 卡上必须开，否则会换页）
        "unload_speech_before_visual": True,
        # 缓存（cache/ 断点与预览音频 + logs/ 日志）：开软件只扫一眼报现状，绝不自动删。
        # 这个天数只用来在清单里标"多久没动过"，清理都从「高级选项 -> 缓存管理」手动来
        "cache_max_age_days": 3,
        # 分析完就把这个视频的 preview_audio.wav 删掉：cache 里只剩 json，省几百兆。
        # 代价是下次要看波形/听预览得重新解一遍音轨（几秒到几十秒）
        "drop_preview_audio": False,
        # 崩溃恢复：卡在 uploading/waiting/processing 的 AI 任务超过这么多分钟就退回 pending，
        # 跑了一半的分析记录超过这么久就标 failed。别设太短，长视频分析本身就慢
        "ai_task_timeout_minutes": 30,
        "analysis_timeout_minutes": 180,
        # 开程序时把上次没跑完的自动剪辑任务捞回来接着跑。
        # 关掉就只是把它们退回 pending 等你手点「自动剪辑」，不自动开工
        "auto_resume_queue": True,

    },
    "mirrors": {
        # 优先国内镜像，只加速下载，仓库仍是官方仓库
        "pypi": [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://mirrors.aliyun.com/pypi/simple",
            "https://pypi.org/simple",
        ],
        "pytorch_index": [
            "https://mirror.nju.edu.cn/pytorch/whl/cu126",
            "https://download.pytorch.org/whl/cu126",
        ],
        "pytorch_find_links": ["https://mirrors.aliyun.com/pytorch-wheels/cu126/"],
        "hf_endpoint": "https://hf-mirror.com",
        "model_sources": ["modelscope", "hf_mirror", "hf"],
        "modelscope_map": {
            "Qwen/Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
            "openbmb/MiniCPM-V-4_5-int4": "OpenBMB/MiniCPM-V-4_5-int4",
            "openbmb/MiniCPM-V-4_5": "OpenBMB/MiniCPM-V-4_5",
            # 4.6 世代改成点号命名，且没有 -int4（4bit 叫 -BNB / -AWQ / -GPTQ）
            "openbmb/MiniCPM-V-4.6": "OpenBMB/MiniCPM-V-4.6",
            "openbmb/MiniCPM-V-4.6-BNB": "OpenBMB/MiniCPM-V-4.6-BNB",
        },
        "whisper_sources": ["modelscope", "hf_mirror", "hf"],
        "whisper_modelscope_map": {
            # ModelScope 上的 Systran 官方镜像仓库（faster-whisper 的 CTranslate2 权重）
            "Systran/faster-whisper-large-v3": "Systran/faster-whisper-large-v3",
            "Systran/faster-whisper-medium": "Systran/faster-whisper-medium",
            "Systran/faster-whisper-small": "Systran/faster-whisper-small",
        },
        "emotion_modelscope_map": {
            # FunASR 语音情绪模型（权重是 model.pt + config.yaml）
            "iic/emotion2vec_plus_large": "iic/emotion2vec_plus_large",
            "iic/emotion2vec_plus_base": "iic/emotion2vec_plus_base",
            "iic/SenseVoiceSmall": "iic/SenseVoiceSmall",
        },
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class Config:
    root: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path, config_file: str | Path | None = None) -> "Config":
        root = Path(root).resolve()
        data = copy.deepcopy(DEFAULTS)
        path = Path(config_file) if config_file else root / "config.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                _deep_update(data, json.load(fh))
        # 兼容老配置：paths.work_dir 是 cache_dir 的旧名字
        legacy = data["paths"].pop("work_dir", None)
        if legacy and data["paths"].get("cache_dir") == DEFAULTS["paths"]["cache_dir"]:
            data["paths"]["cache_dir"] = legacy
        return cls(root=root, data=data)

    # --- 分节访问 ---
    @property
    def visual(self) -> dict[str, Any]:
        return self.data["visual"]

    @property
    def speech(self) -> dict[str, Any]:
        return self.data["speech"]

    @property
    def language(self) -> dict[str, Any]:
        return self.data["language"]

    @property
    def timeline(self) -> dict[str, Any]:
        return self.data["timeline"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.data["runtime"]

    @property
    def highlight(self) -> dict[str, Any]:
        return self.data["highlight"]

    @property
    def bridge(self) -> dict[str, Any]:
        return self.data["bridge"]

    @property
    def assets(self) -> dict[str, Any]:
        """视频资产中心自己的目录（input_dir / output_dir）和两个目录筛选，跟 bridge 无关。"""
        return self.data.setdefault("assets", {"input_dir": "", "output_dir": "",
                                               "filter_video_dir": "",
                                               "filter_product_dir": ""})

    @property
    def mirrors(self) -> dict[str, Any]:
        return self.data["mirrors"]

    @property
    def dance(self) -> dict[str, Any]:
        """舞蹈子系统的配置（见 vidscribe/dance/）。`setdefault` 是为了兼容
        没有这一节的老 config.json —— 缺了也能跑，走 DEFAULTS 那份。"""
        return self.data.setdefault("dance", dict(DEFAULTS["dance"]))

    def dance_path(self, key: str) -> Path:
        """舞蹈的四个目录（song_dir / source_dir / material_dir / output_dir）。

        单独开一个函数而不是塞进 `paths`：那一节是主流程的目录，混进来会让
        「缓存管理」之类按 `paths` 遍历的功能连带扫到素材库。
        """
        value = Path(str(self.dance.get(key) or "").strip() or f"dance/{key}")
        return value if value.is_absolute() else self.root / value

    def ensure_dance_dirs(self) -> None:
        """建舞蹈那四个目录。**不**在 `ensure_dirs` 里做 —— 没用舞蹈功能的用户
        不该因为开一次主界面就多出四个空目录。"""
        for key in ("song_dir", "source_dir", "material_dir", "output_dir"):
            self.dance_path(key).mkdir(parents=True, exist_ok=True)


    def path(self, key: str) -> Path:
        value = Path(self.data["paths"][key])
        return value if value.is_absolute() else self.root / value

    def ensure_dirs(self) -> None:
        for key, value in self.data["paths"].items():
            if not str(value).strip():  # 留空的（比如 video_dir）不建目录
                continue
            self.path(key).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def save_patch(self, patch: dict[str, Any], config_file: str | Path | None = None) -> Path:
        """把界面上改的几项深度合并回 config.json，并同步到内存里的 self.data。

        只写传进来的那几个键，文件里其它内容（注释性字段、手改过的值）原样保留——
        所以是"读盘 -> 合并 -> 写回"，不是拿 self.data 整体覆盖（那会把默认值也写进去）。
        """
        path = Path(config_file) if config_file else self.root / "config.json"
        on_disk: dict[str, Any] = {}
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                on_disk = json.load(fh)
        _deep_update(on_disk, patch)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(on_disk, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        _deep_update(self.data, patch)
        return path
