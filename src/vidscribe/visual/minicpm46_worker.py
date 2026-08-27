"""MiniCPM-V 4.6 推理 worker：在**隔离依赖**下运行的独立进程。

为什么要单独一个进程：
- MiniCPM-V 4.6 已原生进入 transformers（config 声明 transformers_version 5.7.0，
  model_type=minicpmv4_6，仓库里没有任何 modeling_*.py 可以 trust_remote_code 回退）。
- 主环境必须留在 transformers 4.57.6 给 Qwen3-VL / faster-whisper 用，不能升级。
- 所以把 transformers 5.7.0 装在 libs/tf57，只在这个 worker 进程里用
  PYTHONPATH 覆盖，torch / torchvision 仍然共用主环境的 2.8.0+cu126。

协议：stdin 一行一个 JSON 请求，stdout 一行一个 `@@RESP {json}` 响应。
帧数据不走管道，走 .npy 文件（uint8, (T,H,W,C), RGB），避免大 base64 开销。

请求：
  {"cmd":"load","model_path":...,"dtype":"bfloat16","attn":"sdpa"}
  {"cmd":"generate","npy":...,"fps":...,"duration":...,"system":...,"prompt":...,
   "max_new_tokens":512,"downsample_mode":"16x","max_slice_nums":1,
   "use_image_id":false,"stack_frames":1}
  {"cmd":"vram"} / {"cmd":"exit"}
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MARKER = "@@RESP "

_state: dict[str, object] = {"model": None, "processor": None, "video_token": None}


def _respond(payload: dict) -> None:
    sys.stdout.write(MARKER + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load(req: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoProcessor

    path = str(req["model_path"])
    dtype_name = str(req.get("dtype", "bfloat16"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_name, torch.bfloat16)

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(path)
    kwargs: dict[str, object] = {"dtype": dtype, "device_map": {"": 0}}
    attn = str(req.get("attn") or "sdpa")
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    except TypeError:
        # 老参数名兜底
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    model.eval()
    load_seconds = time.perf_counter() - started

    tok = getattr(processor, "tokenizer", None)
    video_token = getattr(processor, "video_token", None) or getattr(tok, "video_token", None)
    _state.update({"model": model, "processor": processor, "video_token": video_token})
    return {
        "ok": True,
        "load_seconds": round(load_seconds, 3),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "video_token": video_token,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1)
        if torch.cuda.is_available() else None,
    }


def _build_text(processor, system: str, prompt: str, video_token: str | None) -> tuple[str, bool]:
    """渲染 chat 模板。返回 (文本, 模板是否已插入 video 占位符)。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]})
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # 模板可能不接受空的 video 项，退化成纯文本模板再自己插占位符
        messages[-1]["content"] = [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    has_token = bool(video_token) and video_token in text
    return text, has_token


def _generate(req: dict) -> dict:
    import numpy as np
    import torch

    model = _state["model"]
    processor = _state["processor"]
    if model is None or processor is None:
        return {"ok": False, "error": "model not loaded"}

    frames = np.load(str(req["npy"]))          # (T,H,W,C) uint8 RGB
    n_frames = int(frames.shape[0])
    fps = float(req.get("fps") or 1.0)
    duration = float(req.get("duration") or max(n_frames / max(fps, 1e-6), 1.0))
    downsample_mode = str(req.get("downsample_mode") or "16x")

    video_token = _state.get("video_token")
    text, has_token = _build_text(processor, str(req.get("system") or ""),
                                  str(req.get("prompt") or ""), video_token)  # type: ignore[arg-type]
    if video_token and not has_token:
        text = text.replace(str(req.get("prompt") or ""), f"{video_token}\n" + str(req.get("prompt") or ""), 1)

    from transformers.video_utils import VideoMetadata

    metadata = VideoMetadata(total_num_frames=n_frames, fps=fps, duration=duration, video_backend="numpy")

    proc_kwargs = {
        "text": [text],
        "videos": [frames],
        "do_sample_frames": False,          # 帧是主进程按真实时间戳采好的，不让它重采
        "video_metadata": [metadata],
        "stack_frames": int(req.get("stack_frames") or 1),
        "max_slice_nums": int(req.get("max_slice_nums") or 1),
        "use_image_id": bool(req.get("use_image_id") or False),
        "downsample_mode": downsample_mode,
        "return_tensors": "pt",
    }
    t0 = time.perf_counter()
    try:
        inputs = processor(**proc_kwargs)
    except TypeError as exc:
        proc_kwargs.pop("video_metadata", None)
        try:
            inputs = processor(**proc_kwargs)
        except Exception:
            return {"ok": False, "error": f"processor: {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}
    inputs = inputs.to(model.device)
    processor_seconds = time.perf_counter() - t0
    prompt_tokens = int(inputs["input_ids"].shape[1])

    t1 = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                downsample_mode=downsample_mode,
                max_new_tokens=int(req.get("max_new_tokens") or 512),
                do_sample=False,
            )
    except Exception as exc:
        oom = "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc)
        return {"ok": False, "oom": oom, "error": f"{type(exc).__name__}: {exc}"[:600],
                "traceback": traceback.format_exc()[-2000:]}
    generate_seconds = time.perf_counter() - t1

    new_ids = generated[:, prompt_tokens:]
    out = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    peak = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1) if torch.cuda.is_available() else None
    del inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "ok": True,
        "text": out,
        "frames": n_frames,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(new_ids.shape[1]),
        "processor_seconds": round(processor_seconds, 3),
        "generate_seconds": round(generate_seconds, 3),
        "peak_reserved_mb": peak,
        "downsample_mode": downsample_mode,
    }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _respond({"ok": False, "error": f"bad request: {exc}"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "load":
                _respond(_load(req))
            elif cmd == "generate":
                _respond(_generate(req))
            elif cmd == "vram":
                import torch
                _respond({"ok": True, "peak_reserved_mb":
                          round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1)
                          if torch.cuda.is_available() else None})
            elif cmd == "exit":
                _respond({"ok": True})
                return 0
            else:
                _respond({"ok": False, "error": f"unknown cmd {cmd}"})
        except Exception as exc:
            _respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600],
                      "traceback": traceback.format_exc()[-2000:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
