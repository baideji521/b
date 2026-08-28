"""语音情绪识别（FunASR / emotion2vec+）。

在 whisper 转写之后跑：按 whisper 已经切好的句子边界逐段判情绪，所以时间轴天然对齐，
不需要另做分句。每段输出：

    emotion            最可能的情绪（显示名，跟随 output_language：英文视频出 happy，中文出开心）
    emotion_en         同一个情绪的英文标签（稳定不变，换语言只重渲显示名，不重跑模型）
    emotion_confidence 该情绪的概率
    emotion_intensity  情绪强度 = 1 - (中立 + 其他 + unk)，用来挑高光冻帧点
    emotion_scores     概率最高的若干类，便于人工判断模型是否含糊

音频只解码一次（PyAV，和 audio.py 同一套用法），16k 单声道 float32 全量拿在内存里，
再按样本切片喂给模型——比每段单独 seek 解码快，也不用落临时 wav。

模型权重是 model.pt + config.yaml（不是 safetensors），所以 mirrors 里单开 kind="emotion"。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..emotions import label_for, to_english
from ..logging_setup import get_logger
from ..progress import report as report_progress
from ..video_io import VideoInfo

logger = get_logger(__name__)

SAMPLE_RATE = 16000
# 情绪强度里不算作"有情绪"的类别（emotion2vec+ 的标签体系）
_FLAT_LABELS = {"neutral", "other", "unk", "<unk>"}


def _decode_mono16k(video: Path) -> np.ndarray:
    """把整条音轨解成 16k 单声道 float32。没有音轨或解码失败返回空数组。"""
    import av  # noqa: PLC0415

    chunks: list[np.ndarray] = []
    with av.open(str(video)) as src:
        if not src.streams.audio:
            return np.zeros(0, dtype=np.float32)
        stream = src.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in src.decode(stream):
            for out in _resampled(resampler, frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in _resampled(resampler, None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _resampled(resampler, frame):
    """PyAV 各版本 resample() 有的返回单帧有的返回列表，统一成列表（同 audio.py）。"""
    out = resampler.resample(frame)
    if out is None:
        return []
    return out if isinstance(out, list) else [out]


def _split_label(label: str) -> tuple[str, str]:
    """emotion2vec 的标签形如 '开心/happy'，拆成 (中文, 英文)。"""
    text = str(label).strip()
    if "/" in text:
        zh, en = text.split("/", 1)
        return zh.strip(), en.strip().lower()
    return text, text.lower()


class EmotionRecognizer:
    """按语音段判情绪。模型只加载一次，多视频复用；显存用法与 WhisperASR 对齐。"""

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model = None
        self.model_id: str = str(self.cfg.get("model_id") or "iic/emotion2vec_plus_large")
        self.device: str = "cpu"
        self.load_seconds = 0.0
        self.model_path: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    # ---------------------------------------------------------------- 模型
    def _resolve(self, model_id: str) -> str:
        if not self.model_dir:
            return model_id
        from ..mirrors import resolve_model  # noqa: PLC0415

        resolved = resolve_model(model_id, Path(self.model_dir), self.mirrors, kind="emotion")
        return resolved if resolved != model_id else model_id

    def load(self) -> None:
        if self.model is not None:
            return
        import torch  # noqa: PLC0415
        from funasr import AutoModel  # noqa: PLC0415

        want = self.cfg.get("device", "auto")
        if want == "auto":
            want = "cuda:0" if torch.cuda.is_available() else "cpu"

        candidates = [self.model_id, *self.cfg.get("fallback_model_ids", [])]
        errors: list[str] = []
        for model_id in candidates:
            target = self._resolve(model_id)
            for device in ([want, "cpu"] if want != "cpu" else ["cpu"]):
                started = time.perf_counter()
                try:
                    logger.info("加载情绪模型 %s (device=%s)", target, device)
                    self.model = AutoModel(model=target, device=device, disable_update=True,
                                           disable_log=True, disable_pbar=True, hub="ms")
                except Exception as exc:  # noqa: BLE001 - 逐个候选试，全失败才放弃
                    errors.append(f"{model_id}@{device}: {str(exc)[:160]}")
                    logger.warning("情绪模型加载失败 %s@%s：%s", model_id, device, str(exc)[:160])
                    continue
                self.model_id, self.device, self.model_path = model_id, device, target
                self.load_seconds = round(time.perf_counter() - started, 2)
                logger.info("情绪模型就绪：%s / %s，耗时 %.1fs", model_id, device, self.load_seconds)
                return
        raise RuntimeError("情绪模型全部加载失败: " + "; ".join(errors))

    def unload(self) -> None:
        """和 WhisperASR.unload 一样显式回收，别让它占着显存等视觉模型。"""
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self.model is None:
            return
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("已释放情绪模型显存")

    # ---------------------------------------------------------------- 推理
    def _infer(self, clips: list[np.ndarray]) -> list[dict[str, Any] | None]:
        """一批音频片段 -> 每段的原始结果。整批失败就退回逐条，避免一段坏数据毁掉全批。"""
        try:
            out = self.model.generate(input=clips, granularity="utterance",
                                      extract_embedding=False)
            if isinstance(out, list) and len(out) == len(clips):
                return list(out)
        except Exception as exc:  # noqa: BLE001
            logger.debug("批量情绪推理失败，改逐条：%s", str(exc)[:160])

        results: list[dict[str, Any] | None] = []
        for clip in clips:
            try:
                one = self.model.generate(input=clip, granularity="utterance",
                                          extract_embedding=False)
                results.append(one[0] if one else None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("单段情绪推理失败：%s", str(exc)[:160])
                results.append(None)
        return results

    def _pack(self, raw: dict[str, Any] | None, output_language: str) -> dict[str, Any] | None:
        """把 {labels, scores} 整理成写进 speech_events.json 的字段。

        emotion 是显示名（跟随 output_language），emotion_en 是稳定的英文标签。
        """
        if not raw:
            return None
        labels = list(raw.get("labels") or [])
        scores = [float(s) for s in (raw.get("scores") or [])]
        if not labels or len(labels) != len(scores):
            return None
        pairs = sorted(zip(labels, scores), key=lambda kv: kv[1], reverse=True)
        top_en = _split_label(pairs[0][0])[1]
        flat = sum(s for label, s in pairs if _split_label(label)[1] in _FLAT_LABELS)
        top_k = max(1, int(self.cfg.get("top_k", 3)))
        return {
            "emotion": label_for(top_en, output_language),
            "emotion_en": top_en,
            "emotion_confidence": round(pairs[0][1], 3),
            "emotion_intensity": round(max(0.0, min(1.0, 1.0 - flat)), 3),
            "emotion_scores": {label_for(_split_label(label)[1], output_language): round(score, 3)
                               for label, score in pairs[:top_k]},
        }

    def annotate(self, info: VideoInfo, segments: list[dict[str, Any]],
                 output_language: str = "zh") -> dict[str, Any]:
        """给每个语音段补情绪字段，返回本次统计（也用于 speech_events.json 的 meta）。"""
        if not segments:
            return {"available": False, "reason": "no_speech_segments"}
        if not info.has_audio:
            return {"available": False, "reason": "no_audio_stream"}

        started = time.perf_counter()
        audio = _decode_mono16k(Path(info.path))
        if audio.size == 0:
            return {"available": False, "reason": "decode_failed"}
        decode_seconds = round(time.perf_counter() - started, 2)

        min_seconds = float(self.cfg.get("min_segment_seconds", 0.3))
        min_samples = max(int(min_seconds * SAMPLE_RATE), 400)
        todo: list[tuple[int, np.ndarray]] = []
        skipped = 0
        for index, seg in enumerate(segments):
            begin = max(0, int(float(seg.get("start", 0.0)) * SAMPLE_RATE))
            finish = min(audio.size, int(float(seg.get("end", 0.0)) * SAMPLE_RATE))
            clip = audio[begin:finish]
            if clip.size < min_samples:  # 太短的段声学特征不够，判了也不可信
                skipped += 1
                continue
            todo.append((index, np.ascontiguousarray(clip)))

        if not todo:
            return {"available": False, "reason": "segments_too_short", "skipped": skipped}

        self.load()
        batch = max(1, int(self.cfg.get("batch_size", 8)))
        done = 0
        for offset in range(0, len(todo), batch):
            group = todo[offset:offset + batch]
            for (index, _clip), raw in zip(group, self._infer([c for _, c in group])):
                packed = self._pack(raw, output_language)
                if packed:
                    segments[index].update(packed)
                    done += 1
            report_progress("speech", 0.9 + 0.1 * (offset + len(group)) / len(todo),
                            f"语音情绪 {offset + len(group)}/{len(todo)} 段", video=info.name)

        elapsed = round(time.perf_counter() - started, 2)
        logger.info("语音情绪识别完成：%d/%d 段（跳过 %d 段过短），耗时 %.1fs（解码 %.1fs）",
                    done, len(segments), skipped, elapsed, decode_seconds)
        return {
            "available": done > 0,
            "model": {"id": self.model_id, "device": self.device, "path": self.model_path},
            "language": output_language,
            "annotated": done,
            "skipped_short": skipped,
            "load_seconds": self.load_seconds,
            "decode_seconds": decode_seconds,
            "elapsed_seconds": elapsed,
        }


def relabel(segments: list[dict[str, Any]], output_language: str) -> int:
    """只把显示名换成另一种语言，不重跑模型。

    情绪判定结果存在 emotion_en 里（英文标签），所以换输出语言时不需要再听一遍音频。
    """
    changed = 0
    for seg in segments:
        english = seg.get("emotion_en") or to_english(seg.get("emotion"))
        if not english:
            continue
        seg["emotion"] = label_for(english, output_language)
        scores = seg.get("emotion_scores")
        if isinstance(scores, dict):
            # 分数字典的键也是显示名，跟着一起换；反查不出英文标签的键原样保留
            seg["emotion_scores"] = {label_for(to_english(k), output_language) or k: v
                                     for k, v in scores.items()}
        changed += 1
    return changed


def emotion_peaks(segments: list[dict[str, Any]], top_n: int = 5,
                  min_intensity: float = 0.5) -> list[dict[str, Any]]:
    """按情绪强度挑出最"炸"的几句，供剪辑高光时参考冻帧点。

    冻帧点取该句的结束时刻：情绪句说完那一瞬间的画面通常就是要冻住的表情。
    """
    peaks = []
    for seg in segments:
        intensity = seg.get("emotion_intensity")
        if not isinstance(intensity, (int, float)) or intensity < min_intensity:
            continue
        peaks.append({
            "start": round(float(seg.get("start", 0.0)), 2),
            "end": round(float(seg.get("end", 0.0)), 2),
            "freeze_at": round(float(seg.get("end", 0.0)), 2),
            "emotion": seg.get("emotion"),
            "emotion_en": seg.get("emotion_en"),
            "intensity": round(float(intensity), 3),
            "confidence": seg.get("emotion_confidence"),
            "text": (seg.get("text") or "")[:80],
        })
    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    return peaks[:max(1, top_n)]
