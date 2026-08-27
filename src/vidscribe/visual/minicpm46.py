"""MiniCPM-V 4.6 后端：通过隔离依赖的子进程 worker 调用。

对外契约和 QwenVLAnalyzer 完全一致（load / unload / set_output_language /
analyze_window / analyze_windows / rewrite_texts / model_id / load_seconds），
Pipeline 不需要知道它其实跑在另一个进程里。

时间戳仍然由主进程用真实帧时间校准（calibrate_events），模型只负责"发生了什么"。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_setup import get_logger
from ..mirrors import resolve_model
from ..video_io import VideoInfo, plan_frame_indices, sample_frames
from . import frame_lines, prompts
from .qwen_vl import VisualOOM, VisualParams, calibrate_events, parse_events

logger = get_logger(__name__)

MARKER = "@@RESP "
ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "libs" / "tf57"
WORKER = Path(__file__).with_name("minicpm46_worker.py")


class MiniCPM46Analyzer:
    backend = "minicpm46"

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg
        self.model_dir = Path(model_dir) if model_dir else None
        self.mirrors = mirrors or {}
        self.model_id = str(cfg.get("model_id") or "openbmb/MiniCPM-V-4.6")
        mc = dict(cfg.get("minicpm46") or {})
        self.downsample_mode = str(mc.get("downsample_mode", "16x"))
        self.max_slice_nums = int(mc.get("max_slice_nums", 1))
        self.use_image_id = bool(mc.get("use_image_id", False))
        self.stack_frames = int(mc.get("stack_frames", 1))
        # 实测 1.3B 写不出合法嵌套 JSON，默认走逐帧行格式（时间由真实帧决定）
        self.output_format = str(mc.get("output_format", "frame_lines")).lower()
        self.dtype = str(mc.get("dtype", cfg.get("dtype", "bfloat16")))
        self.attn = str(mc.get("attn_implementation", "sdpa"))
        self.output_language = "zh"
        self.load_seconds = 0.0
        self.worker_info: dict[str, Any] = {}
        self._proc: subprocess.Popen | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------ 语言
    def set_output_language(self, code: str) -> None:
        self.output_language = (code or "zh").lower()

    # ---------------------------------------------------------------- 进程管理
    def _spawn(self) -> None:
        if not OVERLAY.is_dir():
            raise RuntimeError(
                f"缺少隔离依赖目录 {OVERLAY}。MiniCPM-V 4.6 需要 transformers>=5.7.0，"
                f"请先执行：.venv\\Scripts\\python.exe -m pip install --target libs/tf57 "
                f'"transformers==5.7.0" -i https://pypi.tuna.tsinghua.edu.cn/simple'
            )
        env = dict(os.environ)
        # 关键：把 5.7.0 放在 PYTHONPATH 最前面，只影响这个子进程；
        # torch / torchvision 仍然从主环境 site-packages 解析。
        env["PYTHONPATH"] = os.pathsep.join([str(OVERLAY), str(ROOT / "src")])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        if self.mirrors.get("hf_endpoint"):
            env.setdefault("HF_ENDPOINT", str(self.mirrors["hf_endpoint"]))
        self._proc = subprocess.Popen(
            [sys.executable, str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(ROOT), text=True, encoding="utf-8", errors="replace", bufsize=1,
        )

    def _request(self, payload: dict[str, Any], timeout: float = 1800.0) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("MiniCPM-V 4.6 worker 未运行")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        deadline = time.perf_counter() + timeout
        while True:
            line = self._proc.stdout.readline()
            if not line:
                stderr = ""
                if self._proc.stderr is not None:
                    stderr = self._proc.stderr.read()[-2000:]
                raise RuntimeError(f"worker 意外退出（code={self._proc.poll()}）：{stderr}")
            if line.startswith(MARKER):
                return json.loads(line[len(MARKER):])
            logger.debug("worker: %s", line.rstrip())
            if time.perf_counter() > deadline:
                raise TimeoutError("worker 响应超时")

    # ------------------------------------------------------------------ 加载
    def load(self, model_id: str | None = None) -> None:
        target = model_id or self.model_id
        source = target
        if self.model_dir is not None:
            source = resolve_model(target, self.model_dir, self.mirrors, kind="visual")
        logger.info("加载视觉模型 %s (backend=minicpm46, dtype=%s, 隔离依赖=%s)",
                    source, self.dtype, OVERLAY.name)
        started = time.perf_counter()
        self._spawn()
        resp = self._request({"cmd": "load", "model_path": str(source),
                              "dtype": self.dtype, "attn": self.attn}, timeout=1200.0)
        if not resp.get("ok"):
            detail = resp.get("error") or "unknown"
            tb = resp.get("traceback") or ""
            self.unload()
            raise RuntimeError(f"MiniCPM-V 4.6 加载失败：{detail}\n{tb}")
        self.worker_info = resp
        self.model_id = target
        self.load_seconds = round(time.perf_counter() - started, 2)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mcpm46_")
        logger.info("视觉模型就绪：%s，耗时 %.1fs（worker transformers=%s，torch=%s，参数 %.2fB）",
                    target, self.load_seconds, resp.get("transformers"), resp.get("torch"),
                    (resp.get("param_count") or 0) / 1e9)

    def unload(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._request({"cmd": "exit"}, timeout=30.0)
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=15)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ------------------------------------------------------------------ 推理
    def analyze_window(self, info: VideoInfo, start: float, end: float, params: VisualParams,
                       scene_cuts: list[float] | None = None,
                       previous_summary: str | None = None) -> tuple[list[Any], dict[str, Any]]:
        t0 = time.perf_counter()
        indices = plan_frame_indices(info, start, end, params.fps, params.min_frames, params.max_frames)
        per_frame_budget = max(params.total_pixels // max(len(indices), 1), 64 * 32 * 32)
        batch = sample_frames(info, indices, min(params.max_pixels, per_frame_budget))
        decode_seconds = time.perf_counter() - t0
        if len(batch) == 0:
            raise RuntimeError(f"窗口 {start:.2f}-{end:.2f}s 未能采到任何帧")

        # 帧走 .npy 文件而不是管道，避免 base64 开销
        assert self._tmpdir is not None, "模型未加载"
        npy_path = Path(self._tmpdir.name) / f"w{start:.3f}_{end:.3f}.npy"
        np.save(npy_path, np.stack([np.asarray(im.convert("RGB")) for im in batch.images]))

        use_lines = self.output_format == "frame_lines"
        if use_lines:
            prompt = frame_lines.build_prompt(start, end, batch.timestamps, previous_summary,
                                             output_language=self.output_language)
            system = frame_lines.system_prompt(self.output_language)
        else:
            prompt = prompts.build_user_prompt(
                start, end, batch.timestamps, previous_summary,
                output_language=self.output_language, timestamp_mode="list",
            )
            system = prompts.system_prompt(self.output_language)
        window_fps = batch.sample_fps or max(len(batch) / max(end - start, 1e-6), 0.1)
        resp = self._request({
            "cmd": "generate",
            "npy": str(npy_path),
            "fps": window_fps,
            "duration": max(end - start, 1.0),
            "system": system,
            "prompt": prompt,
            "max_new_tokens": int(params.max_new_tokens),
            "downsample_mode": self.downsample_mode,
            "max_slice_nums": self.max_slice_nums,
            "use_image_id": self.use_image_id,
            "stack_frames": self.stack_frames,
        })
        try:
            npy_path.unlink()
        except Exception:
            pass
        if not resp.get("ok"):
            if resp.get("oom"):
                raise VisualOOM(str(resp.get("error")))
            raise RuntimeError(f"MiniCPM-V 4.6 推理失败：{resp.get('error')}")

        raw = str(resp.get("text") or "")
        tolerance = float(self.cfg.get("snap_tolerance_seconds", 1.0))
        if use_lines:
            events = frame_lines.parse_frame_lines(raw, batch.timestamps, batch.frame_indices,
                                                  start, end)
            parse_mode = "frame_lines"
            if not events:
                # 行格式也没解析出来时，退一步试 JSON，避免整窗口丢空
                events = calibrate_events(parse_events(raw), batch, scene_cuts or [], start, end,
                                          tolerance=tolerance)
                parse_mode = "frame_lines_fallback_json"
        else:
            events = calibrate_events(parse_events(raw), batch, scene_cuts or [], start, end,
                                      tolerance=tolerance)
            parse_mode = "json"
        meta = {
            "window": [round(start, 3), round(end, 3)],
            "params": params.to_dict(),
            "frame_source": "opencv",
            "frames": len(batch),
            "frame_timestamps": batch.timestamps,
            "frame_indices": batch.frame_indices,
            "resolution": [batch.resized_width, batch.resized_height],
            "batch_size": 1,
            "output_format": self.output_format,
            "parse_mode": parse_mode,
            "downsample_mode": resp.get("downsample_mode"),
            "frame_decode_seconds": round(decode_seconds, 3),
            "processor_seconds": resp.get("processor_seconds"),
            "generate_seconds": resp.get("generate_seconds"),
            "infer_seconds": resp.get("generate_seconds"),
            "prompt_tokens": resp.get("prompt_tokens"),
            "generated_tokens": resp.get("generated_tokens"),
            "worker_peak_reserved_mb": resp.get("peak_reserved_mb"),
            "raw_output": raw,
            "event_count": len(events),
        }
        return events, meta

    def analyze_windows(self, info: VideoInfo, windows: list[tuple[float, float]],
                        params: VisualParams, scene_cuts: list[float],
                        previous_summary: str | None = None) -> list[tuple[list[Any], dict[str, Any]]]:
        # 4.6 走 worker，逐窗口串行；batch 由上层决定是否拆分
        return [self.analyze_window(info, s, e, params, scene_cuts, previous_summary)
                for s, e in windows]

    def rewrite_texts(self, texts: list[str], output_language: str,
                      max_new_tokens: int = 0) -> list[str]:
        """纯文本改写：worker 目前只实现了带视频的入口，这里直接返回原文。"""
        return list(texts)
