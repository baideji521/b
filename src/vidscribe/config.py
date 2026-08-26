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
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        # opencv | official | auto
        # 默认 opencv：官方 qwen-vl-utils 在 Windows 上走 torchvision 后端，
        # 它返回的 frames_indices 是窗口内相对帧号，且整段解码内存开销大；
        # 自己做内存帧采样可以拿到原视频绝对帧号（时间戳更可靠），且长视频不爆内存。
        "frame_source": "opencv",
        "fps": 1.5,
        "max_frames": 16,
        "min_frames": 6,
        # 单帧 / 全部帧的像素预算，单位是 token（1 token = 32*32 像素）
        "max_pixels_tokens": 112,
        "total_pixels_tokens": 2048,
        "max_new_tokens": 512,
        # 一次 generate 同时处理几个窗口：单步解码受 CPU/launch 开销主导，
        # 批量可以显著提升总吞吐（实测 batch=4 约为 batch=1 的 3 倍）
        "batch_size": 2,
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
