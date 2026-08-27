"""Qwen3-VL 视觉事件分析器。

- 模型只加载一次，多视频复用。
- 帧采样在内存中完成，不写 JPG 中间文件。
- 模型只回答"发生了什么"；"什么时候发生"由程序用真实帧时间校准。
- OOM 时自动降级参数，最后降级到更小的模型。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from ..events import VisualEvent
from ..logging_setup import get_logger
from ..video_io import PIXEL_FACTOR, FrameBatch, VideoInfo, plan_frame_indices, sample_frames
from . import prompts

logger = get_logger(__name__)


class VisualOOM(RuntimeError):
    """显存不足，需要降级参数重试。"""


@dataclass
class VisualParams:
    fps: float
    max_frames: int
    min_frames: int
    max_pixels_tokens: int
    total_pixels_tokens: int
    max_new_tokens: int
    degrade_level: int = 0
    degrade_history: tuple[str, ...] = ()

    @property
    def max_pixels(self) -> int:
        return int(self.max_pixels_tokens) * PIXEL_FACTOR * PIXEL_FACTOR

    @property
    def total_pixels(self) -> int:
        return int(self.total_pixels_tokens) * PIXEL_FACTOR * PIXEL_FACTOR

    # OOM 降级顺序（spec §34）：batch 由调用方先砍，然后 frames -> resolution -> max_new_tokens
    _AXES = ("max_frames", "resolution", "max_new_tokens")
    # 生成长度阶梯：必须真的能降到 128，旧实现的 max(384, x*0.75) 让 512 只能降到 384
    _TOKEN_LADDER = (512, 384, 256, 192, 128)
    _MIN_FRAMES_FLOOR = 6
    _MIN_PIXELS_TOKENS = 64
    _MIN_NEW_TOKENS = 128

    def _lower_tokens(self) -> int:
        lower = [t for t in self._TOKEN_LADDER if t < self.max_new_tokens]
        return max(lower) if lower else self._MIN_NEW_TOKENS

    def can_degrade(self) -> bool:
        """三条轴是否还有下降空间；没有就别再重试，直接换更小的模型或报错。"""
        return (self.max_frames > self._MIN_FRAMES_FLOOR
                or self.max_pixels_tokens > self._MIN_PIXELS_TOKENS
                or self.max_new_tokens > self._MIN_NEW_TOKENS)

    def degrade(self, reason: str = "cuda_oom") -> "VisualParams":
        """返回降一级的新参数（原对象不变，用户配置不被静默修改）。

        每次只动一条轴，按 frames -> resolution -> max_new_tokens 轮转，
        并把动作写进 degrade_history，meta/报告里能追溯到每一步。
        """
        fps, frames, min_frames = self.fps, self.max_frames, self.min_frames
        pixels, total, tokens = self.max_pixels_tokens, self.total_pixels_tokens, self.max_new_tokens
        note = "no_room"
        for offset in range(len(self._AXES)):
            axis = self._AXES[(self.degrade_level + offset) % len(self._AXES)]
            if axis == "max_frames" and frames > self._MIN_FRAMES_FLOOR:
                frames = max(self._MIN_FRAMES_FLOOR, int(frames * 0.75) // 2 * 2)
                min_frames = min(min_frames, frames)
                fps = max(0.5, round(fps * 0.75, 3))
                note = f"max_frames {self.max_frames}->{frames}, fps {self.fps}->{fps}"
                break
            if axis == "resolution" and pixels > self._MIN_PIXELS_TOKENS:
                pixels = max(self._MIN_PIXELS_TOKENS, int(pixels * 0.7))
                total = max(pixels * max(frames, 1), int(total * 0.7))
                note = f"max_pixels_tokens {self.max_pixels_tokens}->{pixels}"
                break
            if axis == "max_new_tokens" and tokens > self._MIN_NEW_TOKENS:
                tokens = self._lower_tokens()
                note = f"max_new_tokens {self.max_new_tokens}->{tokens}"
                break
        return VisualParams(
            fps=fps,
            max_frames=frames,
            min_frames=max(2, min_frames),
            max_pixels_tokens=pixels,
            total_pixels_tokens=total,
            max_new_tokens=tokens,
            degrade_level=self.degrade_level + 1,
            degrade_history=self.degrade_history + (f"L{self.degrade_level + 1} {reason}: {note}",),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "fps": self.fps,
            "max_frames": self.max_frames,
            "min_frames": self.min_frames,
            "max_pixels_tokens": self.max_pixels_tokens,
            "total_pixels_tokens": self.total_pixels_tokens,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.degrade_level:
            data["degrade_level"] = self.degrade_level
            data["degrade_history"] = list(self.degrade_history)
        return data



def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "alloc" in text and "cuda" in text


class QwenVLAnalyzer:
    backend = "qwen3vl"

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model_id: str = cfg["model_id"]
        self.model = None
        self.processor = None
        self._frame_source = cfg.get("frame_source", "auto")
        self._patch_size = PIXEL_FACTOR // 2
        self.load_seconds = 0.0
        self.model_path: str | None = None
        # 最终自然语言由程序决定（见 language.decide_output_language），模型不自己选
        self.output_language: str = "zh"

    def set_output_language(self, code: str) -> None:
        self.output_language = code or "zh"

    # ------------------------------------------------------------------ 模型
    def load(self, model_id: str | None = None) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        target = model_id or self.model_id
        if self.model is not None and target == self.model_id:
            return
        if self.model is not None:
            self.unload()

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass  # noqa: PLC0415
        except ImportError:  # transformers < 4.57
            from transformers import AutoModelForImageTextToText as ModelClass  # noqa: PLC0415
            logger.warning("未找到 Qwen3VLForConditionalGeneration，回退到 AutoModelForImageTextToText")

        dtype_name = self.cfg.get("dtype", "bfloat16")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            dtype_name, torch.bfloat16
        )
        if not torch.cuda.is_available():
            dtype = torch.float32

        # device_map={"":0} 比 "auto" 快很多（实测加载 9.9s vs 39.1s），
        # 而且避免 accelerate 在显存紧张时偷偷把层放到 CPU 拖慢推理。
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map}
        source = target
        if self.model_dir:
            from pathlib import Path  # noqa: PLC0415

            from ..mirrors import resolve_model  # noqa: PLC0415

            source = resolve_model(target, Path(self.model_dir), self.mirrors, kind="visual")
            if source == target:  # 镜像下载失败，交给 transformers 自己拉取
                kwargs["cache_dir"] = self.model_dir
        attn = self.cfg.get("attn_implementation")
        if attn:
            kwargs["attn_implementation"] = attn

        logger.info("加载视觉模型 %s (dtype=%s, attn=%s)", source, dtype, attn)
        started = time.perf_counter()
        try:
            self.model = ModelClass.from_pretrained(source, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            self.model = ModelClass.from_pretrained(source, **kwargs)
        self.model.eval()
        proc_kwargs = {"cache_dir": self.model_dir} if kwargs.get("cache_dir") else {}
        self.processor = AutoProcessor.from_pretrained(source, **proc_kwargs)
        self.load_seconds = round(time.perf_counter() - started, 2)
        self.model_id = target
        self.model_path = source

        patch = getattr(getattr(self.processor, "image_processor", None), "patch_size", None)
        if isinstance(patch, int) and patch > 0:
            self._patch_size = patch
        # 批量推理必须左侧 padding，否则生成会从 pad 位置继续
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        logger.info("视觉模型就绪：%s，耗时 %.1fs，patch_size=%d", target, self.load_seconds, self._patch_size)

    def unload(self) -> None:
        import gc  # noqa: PLC0415

        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------- 推理主入口
    def analyze_window(self, info: VideoInfo, start: float, end: float, params: VisualParams,
                       scene_cuts: list[float], previous_summary: str | None = None) -> tuple[list[VisualEvent], dict]:
        assert self.model is not None and self.processor is not None, "模型未加载"
        import torch  # noqa: PLC0415

        source = self._frame_source
        batch: FrameBatch | None = None
        raw_text = ""
        meta: dict[str, Any] = {"window": [start, end], "params": params.to_dict()}
        stage_timing: dict[str, float] = {}

        if source in ("auto", "official"):
            try:
                t0 = time.perf_counter()
                inputs, batch = self._build_inputs_official(info, start, end, params, previous_summary)
                stage_timing["prepare_seconds"] = round(time.perf_counter() - t0, 3)
                meta["frame_source"] = "official"
            except Exception as exc:
                if _is_oom(exc):
                    raise VisualOOM(str(exc)) from exc
                if source == "official":
                    raise
                logger.warning("官方视频输入流程不可用（%s），切换为 OpenCV 内存采样", exc)
                self._frame_source = "opencv"
                inputs = batch = None
        if batch is None:
            inputs, batch, stage_timing = self._build_inputs_opencv(info, start, end, params, previous_summary)
            meta["frame_source"] = "opencv"

        meta["frames"] = len(batch)
        meta["frame_timestamps"] = batch.timestamps
        meta["frame_indices"] = batch.frame_indices
        meta["resolution"] = [batch.resized_width, batch.resized_height]
        meta.update(stage_timing)
        prompt_len = int(inputs["input_ids"].shape[1])

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=params.max_new_tokens,
                    do_sample=False,
                )
        except Exception as exc:
            if _is_oom(exc):
                self._free()
                raise VisualOOM(str(exc)) from exc
            raise
        generate_seconds = time.perf_counter() - started
        t1 = time.perf_counter()
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        raw_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        meta["text_decode_seconds"] = round(time.perf_counter() - t1, 3)
        meta["generate_seconds"] = round(generate_seconds, 3)
        meta["infer_seconds"] = round(generate_seconds, 3)
        meta["batch_size"] = 1
        meta["prompt_tokens"] = prompt_len
        meta["generated_tokens"] = int(len(trimmed[0])) if trimmed else 0
        meta["raw_output"] = raw_text


        events = parse_events(raw_text)
        events = calibrate_events(events, batch, scene_cuts, start, end,
                                 tolerance=float(self.cfg.get("snap_tolerance_seconds", 1.0)))
        meta["event_count"] = len(events)
        del inputs
        self._free()
        return events, meta

    # ------------------------------------------------------------- 输入构建
    def _messages(self, video_content: dict, start: float, end: float,
                  timestamps: list[float], previous_summary: str | None) -> list[dict]:
        return [
            {"role": "system",
             "content": [{"type": "text", "text": prompts.system_prompt(self.output_language)}]},
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text",
                     "text": prompts.build_user_prompt(start, end, timestamps, previous_summary,
                                                       output_language=self.output_language)},
                ],
            },
        ]

    def _build_inputs_official(self, info: VideoInfo, start: float, end: float,
                               params: VisualParams, previous_summary: str | None):
        """官方流程：qwen_vl_utils.process_vision_info 直接读视频（无 JPG 中间文件）。"""
        from qwen_vl_utils import process_vision_info  # noqa: PLC0415

        video_content: dict[str, Any] = {
            "type": "video",
            "video": _file_uri(info.path),
            "fps": params.fps,
            "min_frames": params.min_frames,
            "max_frames": params.max_frames,
            "max_pixels": params.max_pixels,
            "total_pixels": params.total_pixels,
        }
        if end - start < info.duration - 0.05:
            video_content["video_start"] = start
            video_content["video_end"] = end

        probe_messages = self._messages(video_content, start, end, [], previous_summary)
        images, videos, video_kwargs = process_vision_info(
            probe_messages,
            image_patch_size=self._patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not videos:
            raise RuntimeError("process_vision_info 未返回视频输入")

        video_metadatas: list[Any] | None
        first = videos[0]
        if isinstance(first, (tuple, list)) and len(first) == 2 and not hasattr(first, "shape"):
            videos, video_metadatas = [list(x) for x in zip(*videos)]
        else:
            video_metadatas = None

        batch = _batch_from_metadata(videos[0], video_metadatas[0] if video_metadatas else None, info,
                                     offset=start if "video_start" in video_content else 0.0)

        messages = self._messages(video_content, start, end, batch.timestamps, previous_summary)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs: dict[str, Any] = {
            "text": [text],
            "images": images,
            "videos": videos,
            "return_tensors": "pt",
            "do_resize": False,
        }
        if video_metadatas is not None:
            proc_kwargs["video_metadata"] = video_metadatas
        proc_kwargs.update(video_kwargs or {})
        inputs = self.processor(**proc_kwargs)
        inputs = inputs.to(self.model.device)
        return inputs, batch

    def _prepare_opencv_window(self, info: VideoInfo, start: float, end: float,
                               params: VisualParams, previous_summary: str | None):
        """OpenCV 内存采样 -> (提示词文本, 帧列表, video_metadata, FrameBatch, 分阶段耗时)。"""
        t0 = time.perf_counter()
        indices = plan_frame_indices(info, start, end, params.fps, params.min_frames, params.max_frames)
        per_frame_budget = max(params.total_pixels // max(len(indices), 1), 64 * PIXEL_FACTOR * PIXEL_FACTOR)
        max_pixels = min(params.max_pixels, per_frame_budget)
        batch = sample_frames(info, indices, max_pixels)
        decode_seconds = time.perf_counter() - t0
        if len(batch) == 0:
            raise RuntimeError(f"窗口 {start:.2f}-{end:.2f}s 未能采到任何帧")

        t1 = time.perf_counter()
        video_content = {"type": "video", "video": batch.images}
        messages = self._messages(video_content, start, end, batch.timestamps, previous_summary)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        template_seconds = time.perf_counter() - t1



        metadata = {
            # 关键：fps 必须是原视频 fps、frames_indices 必须是原视频绝对帧号，
            # Qwen3VLProcessor 会用 frames_indices / fps 生成注入到提示词里的 <x.x seconds>，
            # 这样模型看到的时间戳就是真实的绝对视频时间。
            "fps": info.fps,
            "frames_indices": batch.frame_indices,
            "total_num_frames": info.total_frames or len(batch),
            "duration": info.duration or (batch.timestamps[-1] if batch.timestamps else 0.0),
            "width": batch.resized_width,
            "height": batch.resized_height,
            "video_backend": "opencv",
        }
        timing = {
            "frame_decode_seconds": round(decode_seconds, 3),
            "chat_template_seconds": round(template_seconds, 3),
        }
        return text, batch.images, metadata, batch, timing


    def _processor_call(self, texts: list[str], videos: list[list], metadatas: list[dict]):
        base_kwargs: dict[str, Any] = {
            "text": texts,
            "videos": videos,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
        }
        if len(texts) > 1:
            base_kwargs["padding"] = True
        try:
            inputs = self.processor(**base_kwargs, video_metadata=metadatas)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            logger.debug("video_metadata 不被当前 processor 接受(%s)，退回不带 metadata 的调用", exc)
            inputs = self.processor(**base_kwargs)
        return inputs.to(self.model.device)

    def _build_inputs_opencv(self, info: VideoInfo, start: float, end: float,
                             params: VisualParams, previous_summary: str | None):
        text, images, metadata, batch, timing = self._prepare_opencv_window(
            info, start, end, params, previous_summary
        )
        t0 = time.perf_counter()
        inputs = self._processor_call([text], [images], [metadata])
        timing["processor_seconds"] = round(time.perf_counter() - t0, 3)
        return inputs, batch, timing

    # ------------------------------------------------------------ 批量窗口推理
    def analyze_windows(self, info: VideoInfo, windows: list[tuple[float, float]], params: VisualParams,
                        scene_cuts: list[float], previous_summary: str | None = None
                        ) -> list[tuple[list[VisualEvent], dict]]:
        """把多个窗口拼成一个 batch 一次生成。

        这台机器上单步解码由 CPU/kernel launch 开销主导（GPU 利用率只有 30% 左右），
        batch 起来可以把每步开销摊薄，实测 batch=4 总吞吐约为 batch=1 的 3 倍。
        """
        assert self.model is not None and self.processor is not None, "模型未加载"
        import torch  # noqa: PLC0415

        prepared = [self._prepare_opencv_window(info, s, e, params, previous_summary) for s, e in windows]
        texts = [p[0] for p in prepared]
        videos = [p[1] for p in prepared]
        metadatas = [p[2] for p in prepared]
        batches = [p[3] for p in prepared]
        decode_seconds = sum(p[4]["frame_decode_seconds"] for p in prepared)
        template_seconds = sum(p[4]["chat_template_seconds"] for p in prepared)

        t0 = time.perf_counter()
        inputs = self._processor_call(texts, videos, metadatas)
        processor_seconds = time.perf_counter() - t0
        prompt_len = inputs["input_ids"].shape[1]

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=params.max_new_tokens,
                    do_sample=False,
                )
        except Exception as exc:
            if _is_oom(exc):
                self._free()
                raise VisualOOM(str(exc)) from exc
            raise
        elapsed = time.perf_counter() - started
        t1 = time.perf_counter()
        new_tokens = int(generated[:, prompt_len:].shape[1])
        raw_texts = self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        decode_text_seconds = time.perf_counter() - t1
        del inputs, generated
        self._free()

        n = max(len(windows), 1)
        tolerance = float(self.cfg.get("snap_tolerance_seconds", 1.0))
        results: list[tuple[list[VisualEvent], dict]] = []
        for (start, end), batch, raw in zip(windows, batches, raw_texts):
            events = parse_events(raw)
            events = calibrate_events(events, batch, scene_cuts, start, end, tolerance=tolerance)
            meta = {
                "window": [start, end],
                "params": params.to_dict(),
                "frame_source": "opencv",
                "frames": len(batch),
                "frame_timestamps": batch.timestamps,
                "frame_indices": batch.frame_indices,
                "resolution": [batch.resized_width, batch.resized_height],
                "batch_size": len(windows),
                "infer_seconds": round(elapsed / n, 3),
                # 分阶段耗时（batch 内按窗口数均摊），用来定位真实瓶颈
                "frame_decode_seconds": round(decode_seconds / n, 3),
                "chat_template_seconds": round(template_seconds / n, 3),
                "processor_seconds": round(processor_seconds / n, 3),
                "generate_seconds": round(elapsed / n, 3),
                "text_decode_seconds": round(decode_text_seconds / n, 3),
                "prompt_tokens": int(prompt_len),
                "generated_tokens": new_tokens,
                "raw_output": raw,
                "event_count": len(events),
            }
            results.append((events, meta))
        return results


    # ------------------------------------------------------ 语言改写（纯文本）
    def rewrite_texts(self, texts: list[str], output_language: str,
                      max_new_tokens: int | None = None) -> list[str | None]:
        """把语种不符的描述批量改写成目标语言。

        复用已在显存里的视觉模型做一次纯文本调用，不额外加载翻译模型；
        返回与输入等长的列表，失败位置为 None（由调用方决定兜底策略）。
        """
        if not texts:
            return []
        if self.model is None or self.processor is None:
            logger.warning("视觉模型已卸载，无法做语言改写")
            return [None] * len(texts)
        import torch  # noqa: PLC0415

        system, user = prompts.build_rewrite_prompt(texts, output_language)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        budget = max_new_tokens or min(1024, 64 * len(texts) + 64)
        try:
            inputs = self.processor(text=[prompt], return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            raw = self.processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            del inputs, generated
            self._free()
        except Exception as exc:
            logger.warning("语言改写调用失败：%s", str(exc)[:200])
            return [None] * len(texts)

        return _parse_rewrite(raw, len(texts))

    def _free(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _file_uri(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", normalized):
        return "file:///" + normalized
    return normalized if normalized.startswith("file://") else "file://" + normalized


def _batch_from_metadata(video_tensor: Any, metadata: Any, info: VideoInfo,
                         offset: float = 0.0) -> FrameBatch:
    """把官方流程返回的帧数据换算成真实时间戳。

    注意：torchvision/decord 后端在指定 video_start/video_end 时，frames_indices 是
    "窗口内的相对帧号"，所以要加上窗口起点 offset 才是原视频时间。
    """
    batch = FrameBatch()
    meta_dict: dict[str, Any] = {}
    if isinstance(metadata, dict):
        meta_dict = metadata
    elif metadata is not None:
        for key in ("fps", "frames_indices", "total_num_frames", "duration", "video_backend"):
            if hasattr(metadata, key):
                meta_dict[key] = getattr(metadata, key)

    frame_count = 0
    shape = getattr(video_tensor, "shape", None)
    if shape is not None and len(shape) >= 1:
        frame_count = int(shape[0])
        if len(shape) >= 3:
            batch.resized_height = int(shape[-2])
            batch.resized_width = int(shape[-1])
    elif isinstance(video_tensor, (list, tuple)):
        frame_count = len(video_tensor)

    indices = meta_dict.get("frames_indices") or []
    native_fps = float(meta_dict.get("fps") or info.fps or 25.0)
    if indices:
        batch.frame_indices = [int(i) for i in indices]
        batch.timestamps = [round(offset + int(i) / max(native_fps, 1e-6), 3) for i in indices]
    else:
        batch.frame_indices = list(range(frame_count))
        batch.timestamps = [round(offset + i / max(native_fps, 1e-6), 3) for i in range(frame_count)]

    batch.images = [None] * len(batch.timestamps)  # 官方流程已张量化，无需保留 PIL
    if len(batch.timestamps) > 1:
        span = batch.timestamps[-1] - batch.timestamps[0]
        batch.sample_fps = round((len(batch.timestamps) - 1) / span, 4) if span > 0.01 else 1.0
    else:
        batch.sample_fps = 1.0
    return batch


# ------------------------------------------------------------------ 输出解析
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _loads_lenient(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("模型输出中找不到 JSON")
    candidate = _TRAILING_COMMA.sub(r"\1", match.group(0))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 截断输出：逐个字符补齐括号
        opens = candidate.count("{") - candidate.count("}")
        brackets = candidate.count("[") - candidate.count("]")
        repaired = candidate.rstrip().rstrip(",")
        repaired += "]" * max(0, brackets) + "}" * max(0, opens)
        return json.loads(_TRAILING_COMMA.sub(r"\1", repaired))


def _label(value: Any) -> str | None:
    """内部结构化标签：统一成英文小写 snake_case，非英文内容直接丢弃。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", text)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", text)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        try:
            return float(re.sub(r"[^0-9.\-]", "", text))
        except ValueError:
            return None
    return None


def parse_events(raw_text: str) -> list[VisualEvent]:
    try:
        data = _loads_lenient(raw_text)
    except Exception as exc:
        logger.warning("视觉输出解析失败: %s | 原始输出前200字: %s", exc, raw_text[:200].replace("\n", " "))
        return []

    items = data.get("events") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []

    events: list[VisualEvent] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        start = _to_float(item.get("start"))
        end = _to_float(item.get("end"))
        label = str(item.get("event") or item.get("title") or "").strip()
        desc = str(item.get("description") or item.get("desc") or "").strip()
        if start is None or not (label or desc):
            continue
        if end is None or end < start:
            end = start
        importance = str(item.get("importance") or "normal").strip().lower()
        if importance not in ("low", "normal", "high", "critical"):
            importance = "normal"
        conf = _to_float(item.get("confidence"))
        conf = 0.5 if conf is None else max(0.0, min(1.0, conf))
        ocr = item.get("ocr_text")
        ocr = None if ocr in (None, "", "null", "None", "无") else str(ocr).strip()
        action = _label(item.get("action"))
        scene = _label(item.get("scene"))
        subjects_raw = item.get("subjects") or item.get("participants") or []
        if isinstance(subjects_raw, str):
            subjects_raw = re.split(r"[,;/]| and ", subjects_raw)
        subjects = [s for s in (_label(x) for x in subjects_raw if x) if s][:5]
        events.append(
            VisualEvent(
                id=i + 1,
                start=round(start, 3),
                end=round(end, 3),
                event=label or desc[:20],
                description=desc or label,
                confidence=round(conf, 3),
                importance=importance,
                ocr_text=ocr,
                action=action,
                scene=scene,
                subjects=subjects,
            )
        )
    return events


def _parse_rewrite(raw: str, expected: int) -> list[str | None]:
    """解析改写结果：优先 JSON 数组，退化成按行/编号解析。"""
    out: list[str | None] = [None] * expected
    try:
        data = _loads_lenient(raw if raw.strip().startswith(("[", "{")) else f"[{raw}]")
    except Exception:
        data = None
    if isinstance(data, dict):
        data = data.get("texts") or data.get("result") or list(data.values())
    if isinstance(data, list) and data:
        for i, item in enumerate(data[:expected]):
            if isinstance(item, str) and item.strip():
                out[i] = item.strip()
        if any(out):
            return out

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for i, line in enumerate(lines[:expected]):
        out[i] = re.sub(r'^\s*[\d]+[.、)]\s*|^["\'\[\s]+|["\'\],\s]+$', "", line).strip() or None
    return out


# ------------------------------------------------------------ 时间戳校准
def _snap(value: float, anchors: list[float], tolerance: float) -> tuple[float, bool]:
    if not anchors:
        return value, False
    nearest = min(anchors, key=lambda a: abs(a - value))
    if abs(nearest - value) <= tolerance:
        return round(nearest, 3), True
    return value, False


def calibrate_events(events: list[VisualEvent], batch: FrameBatch, scene_cuts: list[float],
                     window_start: float, window_end: float, tolerance: float = 1.0) -> list[VisualEvent]:
    """把模型给出的时间吸附到真实帧时间/镜头切点上，并标注 timestamp_source。"""
    frame_ts = batch.timestamps
    cuts_in_window = [c for c in scene_cuts if window_start - 0.01 <= c <= window_end + 0.01]
    calibrated: list[VisualEvent] = []
    for ev in events:
        start = max(window_start, min(ev.start, window_end))
        end = max(start, min(ev.end if ev.end > ev.start else ev.start, window_end))

        cut_start, on_cut_start = _snap(start, cuts_in_window, min(tolerance, 0.75))
        frame_start, on_frame_start = _snap(cut_start if on_cut_start else start, frame_ts, tolerance)
        cut_end, on_cut_end = _snap(end, cuts_in_window, min(tolerance, 0.75))
        frame_end, on_frame_end = _snap(cut_end if on_cut_end else end, frame_ts, tolerance)

        if frame_end <= frame_start:
            frame_end = min(window_end, frame_start + max(0.2, 1.0 / max(batch.sample_fps, 0.5)))

        if (on_cut_start or on_frame_start) and (on_cut_end or on_frame_end):
            source = "frame_based" if (on_cut_start and on_cut_end) else "hybrid"
        elif on_cut_start or on_frame_start or on_cut_end or on_frame_end:
            source = "hybrid"
        else:
            source = "model_estimated"

        ev.start = round(frame_start, 3)
        ev.end = round(frame_end, 3)
        ev.timestamp_source = source
        ev.source_frames = [
            idx for idx, ts in zip(batch.frame_indices, frame_ts) if ev.start - 0.01 <= ts <= ev.end + 0.01
        ]
        ev.window = [round(window_start, 3), round(window_end, 3)]
        calibrated.append(ev)
    return sorted(calibrated, key=lambda e: (e.start, e.end))
