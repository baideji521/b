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
        # 冻帧音效：原本冻帧段是纯静音（原声只到冻帧点），这里往那段里混一条音效
        "sfx": {
            "enabled": True,
            # 音效库根目录，下面一层子目录就是类别（tools/fetch_sfx.py 下载归类）
            "dir": "assets/sfx",
            # 原声不动，音效压低一点混进去；正数会更响，注意别削波
            "gain_db": -6.0,
            # 相对冻帧点的偏移，负数=提前一点点起（配合 Flash 更有力）
            "offset_seconds": 0.0,
            # 表情轨没覆盖到冻帧点时用哪个类别
            "fallback_category": "punch",
            # 冻帧点的表情（timeline.json 的 expression_track）-> 类别目录
            # 标签集合见 visual/face.py 的 AFFECTNET
            "emotion_map": {
                "happy": "funny",
                "excited": "funny",
                "surprised": "punch",
                "angry": "punch",
                "sad": "riser",
                "fearful": "riser",
                "disgusted": "fail",
                "contempt": "fail",
                "neutral": "ding",
                "calm": "ding",
            },
        },
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
        # full   剪辑成片：扫 AI_输入目录，缺 <视频名>.txt 就先分析生成，再发 AI，
        #        拿到 JSON 按主界面高光配置直接出成片，落 AI_输出目录
        # collect 收取脚本：只把 AI 回的 JSON 存成 <视频名>_脚本.json，不剪
        # script 脚本剪辑：直接读 AI_输入目录里现成的脚本 JSON 开剪，不问 AI
        "ai_job": "full",

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
    def mirrors(self) -> dict[str, Any]:
        return self.data["mirrors"]

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
