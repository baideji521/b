"""MiniCPM-V 4.5 视觉事件分析器（int4 / bf16 都走这条路径）。

与 Qwen3-VL 后端保持同一套契约：同样的窗口、同样的 JSON schema、同样的
`parse_events` / `calibrate_events`，所以时间戳仍然由程序用真实帧时间校准，
模型只负责"发生了什么"。

与 Qwen3-VL 的关键差异（来自官方模型卡，不是猜的）：
- 接口是 `model.chat(msgs=..., tokenizer=..., use_image_id=False, max_slice_nums=1,
  temporal_ids=frame_ts_id_group)`，不是 processor + generate。
- 没有 `<x.x seconds>` 原生时间戳注入；时间信息通过 `temporal_ids`（0.1s 为单位的整数）
  传给 3D-Resampler，提示词里再把每帧真实秒数显式列出来。
- 3D-Resampler 把 packing_nums 帧压成 64 个 token，所以帧要按 packing_nums 分组。
- int4 版本是 bitsandbytes nf4 预量化权重，不能再 `.cuda()` 搬运，必须用 device_map。
"""

from __future__ import annotations

import time
from typing import Any

from ..events import VisualEvent
from ..logging_setup import get_logger
from ..video_io import VideoInfo, plan_frame_indices, sample_frames
from . import prompts
# 解析与时间戳校准是与后端无关的，复用同一份实现，避免两套 JSON 修复逻辑漂移
from .qwen_vl import VisualOOM, VisualParams, _is_oom, _parse_rewrite, calibrate_events, parse_events

logger = get_logger(__name__)

TIME_SCALE = 0.1  # 官方示例：temporal_ids 以 0.1 秒为单位的整数


class MiniCPMAnalyzer:
    """MiniCPM-V 4.5 后端，接口与 QwenVLAnalyzer 一致。"""

    backend = "minicpm"

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model_id: str = cfg["model_id"]
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.load_seconds = 0.0
        self.model_path: str | None = None
        self.output_language: str = "zh"
        mc = cfg.get("minicpm") or {}
        # packing_nums 有效范围 1-6：越大越省 token 也越快，但时序分辨率越粗
        self.packing_nums = max(1, min(6, int(mc.get("packing_nums", 1))))
        self.max_slice_nums = int(mc.get("max_slice_nums", 1))
        self.use_image_id = bool(mc.get("use_image_id", False))
        self.num_beams = max(1, int(mc.get("num_beams", 1)))

    def set_output_language(self, code: str) -> None:
        self.output_language = code or "zh"

    # ------------------------------------------------------------------ 模型
    def load(self, model_id: str | None = None) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        target = model_id or self.model_id
        if self.model is not None and target == self.model_id:
            return
        if self.model is not None:
            self.unload()

        source = target
        if self.model_dir:
            from ..mirrors import resolve_model  # noqa: PLC0415

            source = resolve_model(target, self.model_dir, self.mirrors, kind="visual")

        is_int4 = "int4" in str(target).lower() or "int4" in str(source).lower()
        dtype_name = self.cfg.get("dtype", "bfloat16")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            dtype_name, torch.bfloat16
        )
        if is_int4:
            # bnb nf4 的 compute dtype 是 float16，dtype 跟着走，避免 bf16/fp16 混用报错
            dtype = torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "attn_implementation": self.cfg.get("attn_implementation", "sdpa"),
        }
        if torch.cuda.is_available():
            # 预量化权重不能事后 .cuda()，用 device_map 一次放好
            kwargs["device_map"] = {"": 0}

        logger.info("加载视觉模型 %s (backend=minicpm, dtype=%s, int4=%s, attn=%s)",
                    source, dtype, is_int4, kwargs["attn_implementation"])
        started = time.perf_counter()
        try:
            model = AutoModel.from_pretrained(source, **kwargs)
        except TypeError as exc:
            # 老版本 transformers 只认 torch_dtype
            if "dtype" not in str(exc):
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModel.from_pretrained(source, **kwargs)
        model.eval()
        if torch.cuda.is_available() and not is_int4 and next(model.parameters()).device.type != "cuda":
            model = model.cuda()

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        # 自己加载 processor 并显式传给 chat()：否则 chat() 会用 config._name_or_path
        # 再去 from_pretrained 一次，可能触发联网。
        from transformers import AutoProcessor  # noqa: PLC0415

        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        self.model_id = target
        self.model_path = source
        self.load_seconds = round(time.perf_counter() - started, 2)
        logger.info("视觉模型就绪：%s，耗时 %.1fs", target, self.load_seconds)

    def unload(self) -> None:
        import gc  # noqa: PLC0415

        self.model = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _free(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ 推理
    def _temporal_ids(self, timestamps: list[float]) -> list[list[int]]:
        """真实秒数 -> 0.1s 整数 id，再按 packing_nums 分组（3D-Resampler 需要）。"""
        ids = [int(round(t / TIME_SCALE)) for t in timestamps]
        size = self.packing_nums
        return [ids[i:i + size] for i in range(0, len(ids), size)]

    def analyze_window(self, info: VideoInfo, start: float, end: float, params: VisualParams,
                       scene_cuts: list[float] | None = None,
                       previous_summary: str | None = None) -> tuple[list[VisualEvent], dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("视觉模型未加载")
        import torch  # noqa: PLC0415

        indices = plan_frame_indices(info, start, end, params.fps, params.max_frames, params.min_frames)
        batch = sample_frames(info, indices, params.max_pixels)
        if not batch.images:
            logger.warning("窗口 %.1f-%.1fs 没有取到帧，跳过", start, end)
            return [], {"frames": 0, "window": [start, end]}

        prompt = prompts.build_user_prompt(
            start, end, batch.timestamps, previous_summary,
            output_language=self.output_language, timestamp_mode="list",
        )
        system = prompts.system_prompt(self.output_language)
        # 官方 chat() 里 assert role in ["user","assistant"]，system 必须走 system_prompt 参数
        msgs = [{"role": "user", "content": list(batch.images) + [prompt]}]
        temporal_ids = self._temporal_ids(batch.timestamps)

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                raw = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    system_prompt=system,
                    use_image_id=self.use_image_id,
                    max_slice_nums=self.max_slice_nums,
                    temporal_ids=temporal_ids,
                    sampling=False,               # greedy，保证可复现
                    # sampling=False 时官方默认 num_beams=3：显存和耗时都约 3 倍，
                    # 这里压回 1 才和 Qwen3-VL 的贪心解码可比。
                    num_beams=self.num_beams,
                    max_new_tokens=params.max_new_tokens,
                )
        except Exception as exc:
            if _is_oom(exc):
                self._free()
                raise VisualOOM(str(exc)) from exc
            raise
        infer_seconds = time.perf_counter() - started
        self._free()

        text = raw if isinstance(raw, str) else str(raw)
        events = parse_events(text)
        events = calibrate_events(
            events, batch, scene_cuts or [], start, end,
            tolerance=float(self.cfg.get("snap_tolerance_seconds", 1.0)),
        )
        meta = {
            "frames": len(batch.images),
            "window": [round(start, 3), round(end, 3)],
            "resolution": [batch.resized_width, batch.resized_height],
            "sample_fps": batch.sample_fps,
            "infer_seconds": round(infer_seconds, 2),
            "batch_size": 1,
            "backend": "minicpm",
            "packing_nums": self.packing_nums,
            "num_beams": self.num_beams,
            "temporal_groups": len(temporal_ids),
            "raw_chars": len(text),
        }
        return events, meta

    def analyze_windows(self, info: VideoInfo, windows: list[tuple[float, float]], params: VisualParams,
                        scene_cuts: list[float] | None = None,
                        previous_summary: str | None = None) -> list[tuple[list[VisualEvent], dict[str, Any]]]:
        """MiniCPM 的 chat() 一次只处理一个对话，这里顺序执行（batch 恒为 1）。"""
        results = []
        summary = previous_summary
        for start, end in windows:
            events, meta = self.analyze_window(info, start, end, params, scene_cuts, summary)
            if events:
                summary = prompts.build_context_summary(events)
            results.append((events, meta))
        return results

    # ------------------------------------------------------ 语言改写（纯文本）
    def rewrite_texts(self, texts: list[str], output_language: str,
                      max_new_tokens: int | None = None) -> list[str | None]:
        if not texts:
            return []
        if self.model is None or self.tokenizer is None:
            logger.warning("视觉模型已卸载，无法做语言改写")
            return [None] * len(texts)
        import torch  # noqa: PLC0415

        system, user = prompts.build_rewrite_prompt(texts, output_language)
        msgs = [{"role": "user", "content": [user]}]
        try:
            with torch.inference_mode():
                raw = self.model.chat(
                    msgs=msgs, tokenizer=self.tokenizer, processor=self.processor,
                    system_prompt=system, sampling=False, num_beams=self.num_beams,
                    max_new_tokens=max_new_tokens or min(1024, 64 * len(texts) + 64),
                )
        except Exception as exc:
            logger.warning("语言改写调用失败：%s", str(exc)[:200])
            return [None] * len(texts)
        return _parse_rewrite(raw if isinstance(raw, str) else str(raw), len(texts))
