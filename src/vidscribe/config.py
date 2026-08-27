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
        "work_dir": "work",
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
        "merge_similarity": 0.82,
        "dedup_similarity": 0.72,
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
    "runtime": {
        "max_auto_retries": 3,
        "keep_models_loaded": True,
        # 语音跑完就释放 whisper 显存再加载视觉模型（12GB 卡上必须开，否则会换页）
        "unload_speech_before_visual": True,
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
    def mirrors(self) -> dict[str, Any]:
        return self.data["mirrors"]

    def path(self, key: str) -> Path:
        value = Path(self.data["paths"][key])
        return value if value.is_absolute() else self.root / value

    def ensure_dirs(self) -> None:
        for key in self.data["paths"]:
            self.path(key).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)
