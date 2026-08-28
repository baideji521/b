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
from ..progress import report as report_progress
from ..video_io import VideoInfo
from .punctuate import Punctuator
from .sentences import split_sentences



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
        # 标点恢复：whisper 在口语素材上经常整段不给标点，靠 ct-punc 补，补完再断句
        self.punctuator = Punctuator(cfg.get("punctuation", {}), model_dir, mirrors)


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
        """释放语音模型。

        CTranslate2 的显存不走 torch 分配器，只有对象被回收才会还给驱动，
        所以这里必须显式 gc；12GB 卡上不释放会把视觉模型挤到共享内存里换页。
        """
        import gc  # noqa: PLC0415

        self.punctuator.unload()      # 标点模型也一起放掉，别占着显存等视觉模型
        if self.model is None:
            return

        self.model = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("已释放语音模型显存")

    # ------------------------------------------------------------------ 识别
    def _language_and_prompt(self, info: VideoInfo,
                            vad_filter: bool) -> tuple[str | None, str]:
        """定下解码语言，并挑同一语言的示范 prompt。

        `initial_prompt` 支持两种写法：一个字符串（不管什么语言都用它），或者
        `{"en": "...", "zh": "..."}` 按语言挑。**强烈建议用后者**：prompt 里混进
        另一种语言，whisper 会顺着 prompt 把语音"翻译"过去——实测一条纯英文视频
        被一段中英混排的 prompt 带出 27/43 行中文，还伴随重复幻听。

        用字典时先跑一次 `detect_language` 拿到语言，再把这个语言显式传给 transcribe，
        保证"prompt 的语言 == 解码的语言"，不给模型留切语言的口子。
        """
        raw = self.cfg.get("initial_prompt")
        language = self.cfg.get("language") or None
        if isinstance(raw, str):
            return language, raw.strip()
        if not isinstance(raw, dict) or not raw:
            return language, ""

        if not language:
            try:
                from faster_whisper.audio import decode_audio  # noqa: PLC0415

                audio = decode_audio(info.path, sampling_rate=16000)
                detected, prob, _ = self.model.detect_language(audio=audio)
                language = str(detected)
                logger.info("语言预检：%s(%.2f)，据此挑 initial_prompt", language, prob or 0.0)
            except Exception as exc:  # noqa: BLE001 - 检测失败就不给 prompt，别赌语言
                logger.warning("语言预检失败，本次不喂 initial_prompt：%s", str(exc)[:160])
                return None, ""

        code = str(language).lower().split("-")[0]
        prompt = str(raw.get(code) or raw.get("default") or "").strip()
        if not prompt:
            logger.info("语言 %s 没有对应的 initial_prompt，本次不喂", code)
        return language, prompt

    def _warn_script_mismatch(self, segments: list[dict[str, Any]],
                             language: str | None) -> None:
        """非中文语音里冒出大量汉字 = 模型在翻译或幻听，出个警告。

        不自动删——用户的素材真的有中英混说的句子，删了就是丢内容。只是把可疑比例
        报出来，好让人一眼看出"这条得重跑"。
        """
        code = str(language or "").lower().split("-")[0]
        if code in ("zh", "ja", "yue", ""):
            return
        import re  # noqa: PLC0415

        cjk = re.compile(r"[\u4e00-\u9fff]")
        hits = sum(1 for seg in segments if cjk.search(str(seg.get("text") or "")))
        if hits and hits >= max(2, len(segments) * 0.1):
            logger.warning("语言=%s 却有 %d/%d 段含汉字，疑似 initial_prompt 把模型带去翻译了，"
                           "建议检查 speech.initial_prompt 是否混了别的语言", code, hits, len(segments))

    def transcribe(self, info: VideoInfo) -> dict[str, Any]:


        if not info.has_audio:
            logger.warning("%s 没有音轨，跳过语音识别", info.name)
            return {"available": False, "reason": "no_audio_stream", "language": None, "segments": []}

        self.load()
        started = time.perf_counter()
        vad_filter = bool(self.cfg.get("vad_filter", True))
        # 先定语言，再挑对应语言的示范 prompt。混语言的 prompt 会把模型带跑：
        # 实测一段"英文+中文"的 prompt 让一条纯英文视频吐出 27/43 行中文（模型顺手翻译了）。
        language, prompt = self._language_and_prompt(info, vad_filter)
        kwargs: dict[str, Any] = {
            "language": language,
            "task": "transcribe",  # 第一版不做翻译，保留原始语言
            "beam_size": int(self.cfg.get("beam_size", 5)),
            "word_timestamps": bool(self.cfg.get("word_timestamps", True)),
            "condition_on_previous_text": bool(self.cfg.get("condition_on_previous_text", False)),
            "vad_filter": vad_filter,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt


        if vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}

        try:
            segments_iter, tr_info = self.model.transcribe(info.path, **kwargs)
            # 生成器是懒执行的，必须迭代才真正推理；顺便按已识别到的时间点上报进度
            audio_seconds = float(getattr(tr_info, "duration", 0.0) or info.duration or 0.0)
            segments = []
            for seg in segments_iter:
                segments.append(seg)
                if audio_seconds > 0:
                    report_progress("speech", min(1.0, float(seg.end) / audio_seconds),
                                    f"已识别 {seg.end:.1f}s / {audio_seconds:.1f}s（{len(segments)} 段）",
                                    video=info.name, done=round(float(seg.end), 2), total=round(audio_seconds, 2))
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
        raw = [e.to_dict() for e in events]
        self._warn_script_mismatch(raw, tr_info.language)
        # 先补标点（标点直接贴到 words 上，时间戳不动），再按句切分

        punctuation = self.punctuate(raw)
        # whisper 一段常常含好几句话，这里按停顿和句末标点切成一句一段（时间取词级时间戳）
        sentences = split_sentences(raw)
        logger.info(
            "语音识别完成：%d 段 -> %d 句，语言=%s(%.2f)，耗时 %.1fs",
            len(events), len(sentences), tr_info.language, tr_info.language_probability or 0.0, elapsed,
        )
        return {
            "available": True,
            "language": tr_info.language,
            "language_probability": round(float(tr_info.language_probability or 0.0), 4),
            "duration": round(float(tr_info.duration or info.duration), 3),
            "duration_after_vad": round(float(getattr(tr_info, "duration_after_vad", 0.0) or 0.0), 3),
            "model": {"size": self.model_size, "device": self.device, "compute_type": self.compute_type},
            "elapsed_seconds": elapsed,
            "punctuation": punctuation,
            "segments": sentences,
        }

    def punctuate(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """给缺标点的段补标点。标点模型挂了不能连累转写，所以整段兜住异常。

        也给 pipeline 复用老缓存时调——那条路上不重跑 whisper，只补标点再重切。
        """
        if not self.punctuator.enabled:
            return {"available": False, "reason": "disabled"}
        try:
            return self.punctuator.restore(segments)
        except Exception as exc:  # noqa: BLE001 - 标点是增强项，失败就退回原文
            logger.warning("标点恢复整体失败，保留原始转写：%s", str(exc)[:200])
            return {"available": False, "reason": f"error: {str(exc)[:200]}"}


