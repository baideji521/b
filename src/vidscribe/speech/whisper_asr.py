"""faster-whisper 语音识别。

- 模型只加载一次，多视频复用。
- word_timestamps=True，保留原始语言，不做翻译。
- CUDA / float16 优先，失败自动降级：compute_type -> 更小模型 -> CPU。
- 视频没有音轨时返回空结果，不让整个任务失败。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from ..events import SpeechEvent, SpeechWord
from ..logging_setup import get_logger
from ..video_io import VideoInfo

logger = get_logger(__name__)


def _register_cuda_dlls() -> None:
    """CTranslate2 需要 cuBLAS/cuDNN9 DLL；torch 的 wheel 里自带，把目录加入搜索路径。"""
    if os.name != "nt":
        return
    try:
        import torch  # noqa: PLC0415

        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib_dir):
            os.add_dll_directory(lib_dir)
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:
        logger.debug("注册 CUDA DLL 目录失败: %s", exc)


def _confidence(avg_logprob: float | None, no_speech_prob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    base = math.exp(max(min(avg_logprob, 0.0), -5.0))
    if no_speech_prob is not None:
        base *= max(0.0, 1.0 - float(no_speech_prob))
    return round(max(0.0, min(1.0, base)), 3)


class WhisperASR:
    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model = None
        self.model_size: str = cfg["model_size"]
        self.device: str = "cpu"
        self.compute_type: str = cfg.get("compute_type", "float16")
        self.load_seconds = 0.0
        self.model_path: str | None = None

    def _resolve(self, size: str) -> str:
        """优先用国内镜像把官方 Systran 权重拉到本地，返回本地目录。"""
        if not self.model_dir:
            return size
        from pathlib import Path  # noqa: PLC0415

        from ..mirrors import resolve_model, whisper_repo_id  # noqa: PLC0415

        repo = whisper_repo_id(size)
        resolved = resolve_model(repo, Path(self.model_dir), self.mirrors, kind="whisper")
        return resolved if resolved != repo else size

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        if self.model is not None:
            return
        _register_cuda_dlls()
        from faster_whisper import WhisperModel  # noqa: PLC0415

        want_device = self.cfg.get("device", "auto")
        if want_device == "auto":
            try:
                import torch  # noqa: PLC0415

                want_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                want_device = "cpu"

        sizes = [self.cfg["model_size"], *self.cfg.get("fallback_model_sizes", [])]
        attempts: list[tuple[str, str, str]] = []
        for size in sizes:
            if want_device == "cuda":
                attempts.append((size, "cuda", self.cfg.get("compute_type", "float16")))
                attempts.append((size, "cuda", "int8_float16"))
            attempts.append((size, "cpu", "int8"))

        last_error: Exception | None = None
        resolved: dict[str, str] = {}
        for size, device, compute in attempts:
            try:
                if size not in resolved:
                    resolved[size] = self._resolve(size)
                target = resolved[size]
                logger.info("加载语音模型 %s (device=%s, compute_type=%s)", size, device, compute)
                started = time.perf_counter()
                self.model = WhisperModel(
                    target,
                    device=device,
                    compute_type=compute,
                    download_root=self.model_dir,
                    cpu_threads=0,
                    num_workers=1,
                )
                self.load_seconds = round(time.perf_counter() - started, 2)
                self.model_size, self.device, self.compute_type = size, device, compute
                self.model_path = target
                logger.info("语音模型就绪：%s / %s / %s，耗时 %.1fs", size, device, compute, self.load_seconds)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("语音模型 %s (%s/%s) 加载失败：%s", size, device, compute, str(exc)[:200])
                self.model = None
        raise RuntimeError(f"所有 faster-whisper 配置均加载失败: {last_error}")

    def unload(self) -> None:
        self.model = None

    # ------------------------------------------------------------------ 识别
    def transcribe(self, info: VideoInfo) -> dict[str, Any]:
        if not info.has_audio:
            logger.warning("%s 没有音轨，跳过语音识别", info.name)
            return {"available": False, "reason": "no_audio_stream", "language": None, "segments": []}

        self.load()
        started = time.perf_counter()
        vad_filter = bool(self.cfg.get("vad_filter", True))
        kwargs: dict[str, Any] = {
            "language": self.cfg.get("language"),
            "task": "transcribe",  # 第一版不做翻译，保留原始语言
            "beam_size": int(self.cfg.get("beam_size", 5)),
            "word_timestamps": bool(self.cfg.get("word_timestamps", True)),
            "condition_on_previous_text": bool(self.cfg.get("condition_on_previous_text", False)),
            "vad_filter": vad_filter,
        }
        if vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}

        try:
            segments_iter, tr_info = self.model.transcribe(info.path, **kwargs)
            segments = list(segments_iter)  # 生成器是懒执行的，必须迭代才真正推理
        except Exception as exc:
            message = str(exc)
            if "does not contain any stream" in message or "Invalid data" in message or "no audio" in message.lower():
                logger.warning("%s 音频解码失败（可能无音轨）：%s", info.name, message[:160])
                return {"available": False, "reason": f"decode_failed: {message[:160]}", "language": None, "segments": []}
            raise

        events: list[SpeechEvent] = []
        for i, seg in enumerate(segments, start=1):
            text = (seg.text or "").strip()
            if not text:
                continue
            words = []
            for w in (getattr(seg, "words", None) or []):
                words.append(
                    SpeechWord(
                        word=w.word,
                        start=round(float(w.start), 3),
                        end=round(float(w.end), 3),
                        probability=round(float(w.probability), 4) if w.probability is not None else None,
                    )
                )
            events.append(
                SpeechEvent(
                    id=i,
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=text,
                    confidence=_confidence(getattr(seg, "avg_logprob", None), getattr(seg, "no_speech_prob", None)),
                    language=tr_info.language,
                    no_speech_prob=round(float(seg.no_speech_prob), 4) if seg.no_speech_prob is not None else None,
                    avg_logprob=round(float(seg.avg_logprob), 4) if seg.avg_logprob is not None else None,
                    words=words,
                )
            )

        elapsed = round(time.perf_counter() - started, 2)
        logger.info(
            "语音识别完成：%d 段，语言=%s(%.2f)，耗时 %.1fs",
            len(events), tr_info.language, tr_info.language_probability or 0.0, elapsed,
        )
        return {
            "available": True,
            "language": tr_info.language,
            "language_probability": round(float(tr_info.language_probability or 0.0), 4),
            "duration": round(float(tr_info.duration or info.duration), 3),
            "duration_after_vad": round(float(getattr(tr_info, "duration_after_vad", 0.0) or 0.0), 3),
            "model": {"size": self.model_size, "device": self.device, "compute_type": self.compute_type},
            "elapsed_seconds": elapsed,
            "segments": [e.to_dict() for e in events],
        }
