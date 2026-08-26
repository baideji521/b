"""单个视频的完整处理流程（探测 -> 视觉 -> 语音 -> Timeline -> 导出 -> benchmark）。

模型只加载一次，多视频复用；每个阶段独立落盘，支持断点续跑。
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from . import benchmark as bench
from .checkpoint import Checkpoint
from .config import Config
from .events import SpeechEvent, SpeechWord, VisualEvent, finalize
from .logging_setup import get_logger
from .speech.whisper_asr import WhisperASR
from .timeline.engine import build_timeline, filter_timeline
from .timeline.exporters import write_json, write_srt, write_timeline_txt
from .video_io import VideoInfo, detect_scene_cuts, plan_windows, probe_video
from .visual import prompts
from .visual.qwen_vl import QwenVLAnalyzer, VisualOOM, VisualParams

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        model_dir = str(cfg.path("model_dir"))
        self.analyzer = QwenVLAnalyzer(cfg.visual, model_dir, cfg.mirrors)
        self.asr = WhisperASR(cfg.speech, model_dir, cfg.mirrors)
        self.env = bench.environment_snapshot()

    # --------------------------------------------------------------- 对外入口
    def run_video(self, video_path: str | Path, force: bool = False,
                  skip_visual: bool = False, skip_speech: bool = False) -> dict[str, Any]:
        video_path = Path(video_path).resolve()
        timer = bench.Timer()
        ckpt = Checkpoint(self.cfg.path("work_dir"), video_path)
        if force:
            ckpt.reset()
        out_dir = self.cfg.path("output_dir") / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("开始处理: %s", video_path.name)

        # 1) 视频探测
        with timer.stage("probe_seconds"):
            if ckpt.done("probe"):
                info = VideoInfo(**ckpt.load("probe")["video"])
                cuts = ckpt.load("probe").get("scene_cuts", [])
                logger.info("复用已有探测结果")
            else:
                info = probe_video(video_path)
                cuts = []
                if self.cfg.visual.get("scene_detect", True):
                    cuts = detect_scene_cuts(
                        info,
                        sample_fps=float(self.cfg.visual.get("scene_sample_fps", 3.0)),
                        threshold=float(self.cfg.visual.get("scene_threshold", 0.35)),
                    )
                ckpt.save("probe", {"video": info.to_dict(), "scene_cuts": cuts})
        logger.info(
            "视频: %.2fs, %dx%d, %.3f fps, %d 帧, 音轨=%s",
            info.duration, info.width, info.height, info.fps, info.total_frames, info.has_audio,
        )
        write_json(out_dir / "video_metadata.json", {"video": info.to_dict(), "scene_cuts": cuts})

        # 2) 视觉分析
        visual_meta: dict[str, Any] = {}
        visual_events: list[VisualEvent] = []
        if skip_visual:
            logger.warning("按要求跳过视觉分析")
        elif ckpt.done("visual"):
            payload = ckpt.load("visual")
            visual_events = [VisualEvent(**e) for e in payload["events"]]
            visual_meta = payload.get("meta", {})
            logger.info("复用已有视觉分析结果：%d 个事件", len(visual_events))
        else:
            with timer.stage("visual_seconds"):
                visual_events, visual_meta = self._run_visual(info, cuts, ckpt)
            ckpt.save("visual", {"events": [e.to_dict() for e in visual_events], "meta": visual_meta})
        write_json(out_dir / "visual_events.json", {
            "video": info.name,
            "duration": info.duration,
            "meta": visual_meta,
            "events": [e.to_dict() for e in visual_events],
        })

        # 3) 语音识别
        if skip_speech:
            speech_payload = {"available": False, "reason": "skipped", "language": None, "segments": []}
            logger.warning("按要求跳过语音识别")
        elif ckpt.done("speech"):
            speech_payload = ckpt.load("speech")
            logger.info("复用已有语音识别结果：%d 段", len(speech_payload.get("segments", [])))
        else:
            with timer.stage("speech_seconds"):
                try:
                    speech_payload = self.asr.transcribe(info)
                except Exception as exc:
                    logger.error("语音识别失败：%s", exc)
                    logger.debug(traceback.format_exc())
                    speech_payload = {"available": False, "reason": f"error: {exc}"[:300],
                                      "language": None, "segments": []}
            ckpt.save("speech", speech_payload)
        write_json(out_dir / "speech_events.json", speech_payload)

        speech_events = _speech_events_from_payload(speech_payload)

        # 4) Timeline 合并 + 导出
        with timer.stage("timeline_seconds"):
            entries = build_timeline(
                visual_events, speech_events,
                min_overlap=float(self.cfg.timeline.get("min_overlap_seconds", 0.2)),
            )
            filtered = filter_timeline(
                entries,
                importance=str(self.cfg.timeline.get("importance_filter", "low")),
                min_confidence=float(self.cfg.timeline.get("confidence_filter", 0.0)),
            )
            language = speech_payload.get("language")
            timeline_doc = {
                "video": info.name,
                "video_path": info.path,
                "duration": info.duration,
                "language": language,
                "speech_available": bool(speech_payload.get("available")),
                "counts": {
                    "visual_events": len(visual_events),
                    "speech_segments": len(speech_events),
                    "timeline_entries": len(filtered),
                },
                "timeline": [
                    {
                        "start": e["start"],
                        "end": e["end"],
                        "visual": e["visual"],
                        "speech": e["speech"],
                        "importance": e["importance"],
                        "timestamp_source": e["timestamp_source"],
                        "ocr_text": e["ocr_text"],
                        "visual_event_id": e["visual_event_id"],
                        "speech_event_ids": e["speech_event_ids"],
                        "source_frames": e["source_frames"],
                        "visual_confidence": e["visual_confidence"],
                        "speech_confidence": e["speech_confidence"],
                        "quality": e["quality"],
                    }
                    for e in filtered
                ],
            }
            write_json(out_dir / "timeline.json", timeline_doc)
            write_timeline_txt(out_dir / "timeline.txt", info.name, info.duration, language, filtered)
            srt_kind = write_srt(
                out_dir / "timeline.srt",
                speech_payload.get("segments", []),
                [e.to_dict() for e in visual_events],
            )
            ckpt.save("timeline", {"entries": len(filtered), "srt_kind": srt_kind})

        # 5) Benchmark
        benchmark = {
            "video": info.to_dict(),
            "environment": self.env,
            "visual_model": {
                "model_id": self.analyzer.model_id,
                "load_seconds": self.analyzer.load_seconds,
                "frame_source": visual_meta.get("frame_source"),
                "params": visual_meta.get("params"),
                "windows": visual_meta.get("window_count"),
                "analyzed_frames": visual_meta.get("total_frames_analyzed"),
                "degrade_attempts": visual_meta.get("degrade_attempts", 0),
            },
            "speech_model": speech_payload.get("model"),
            "timings": {**timer.stages, "total_seconds": timer.total},
            "peak_vram": bench.peak_vram_mb(),
            "counts": timeline_doc["counts"],
            "srt_kind": srt_kind,
        }
        write_json(out_dir / "benchmark.json", benchmark)

        logger.info(
            "完成 %s：视觉 %d 事件 / 语音 %d 段 / timeline %d 条，总耗时 %.1fs",
            info.name, len(visual_events), len(speech_events), len(filtered), timer.total,
        )
        return {
            "video": info.name,
            "video_path": info.path,
            "output_dir": str(out_dir),
            "status": "OK",
            "timeline_entries": len(filtered),
            "visual_events": len(visual_events),
            "speech_segments": len(speech_events),
            "language": speech_payload.get("language"),
            "speech_available": bool(speech_payload.get("available")),
            "benchmark": benchmark,
        }

    # --------------------------------------------------------------- 视觉阶段
    def _run_visual(self, info: VideoInfo, cuts: list[float], ckpt: Checkpoint) -> tuple[list[VisualEvent], dict]:
        vcfg = self.cfg.visual
        params = VisualParams(
            fps=float(vcfg["fps"]),
            max_frames=int(vcfg["max_frames"]),
            min_frames=int(vcfg["min_frames"]),
            max_pixels_tokens=int(vcfg["max_pixels_tokens"]),
            total_pixels_tokens=int(vcfg["total_pixels_tokens"]),
            max_new_tokens=int(vcfg["max_new_tokens"]),
        )
        windows = plan_windows(
            info.duration, cuts,
            window_seconds=float(vcfg["window_seconds"]),
            overlap_seconds=float(vcfg["window_overlap_seconds"]),
            long_threshold=float(vcfg["long_video_threshold"]),
        )
        logger.info("视觉分析窗口：%d 个 -> %s", len(windows),
                    ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in windows[:6]) + (" ..." if len(windows) > 6 else ""))

        bench.reset_peak_vram()
        cache = ckpt.load_window_cache()
        # 所有窗口都有缓存时不必加载模型（断点续跑/只改后处理时省下几十秒）
        all_cached = all(f"{i}:{s:.3f}-{e:.3f}" in cache for i, (s, e) in enumerate(windows))
        if all_cached:
            logger.info("全部 %d 个窗口命中缓存，跳过视觉模型加载", len(windows))
        else:
            self._load_visual_model()

        all_events: list[VisualEvent] = []
        window_metas: list[dict] = []
        total_frames = 0
        degrade_attempts = 0
        max_retries = int(self.cfg.runtime.get("max_auto_retries", 3))
        fallback_ids = list(vcfg.get("fallback_model_ids", []))
        batch_size = max(1, int(vcfg.get("batch_size", 1)))

        queue = [(i, s, e) for i, (s, e) in enumerate(windows)]
        while queue:
            chunk = queue[:batch_size]
            queue = queue[batch_size:]
            todo: list[tuple[int, float, float]] = []
            for idx, start, end in chunk:
                key = f"{idx}:{start:.3f}-{end:.3f}"
                if key in cache:
                    cached = cache[key]
                    events = [VisualEvent(**e) for e in cached["events"]]
                    all_events.extend(events)
                    window_metas.append(cached["meta"])
                    total_frames += int(cached["meta"].get("frames", 0))
                    logger.info("窗口 %d/%d 复用缓存（%d 事件）", idx + 1, len(windows), len(events))
                else:
                    todo.append((idx, start, end))
            if not todo:
                continue

            summary = prompts.build_context_summary(all_events) if all_events else None
            attempt = 0
            current_batch = todo
            while True:
                try:
                    if len(current_batch) == 1:
                        idx, start, end = current_batch[0]
                        results = [self.analyzer.analyze_window(info, start, end, params, cuts, summary)]
                    else:
                        results = self.analyzer.analyze_windows(
                            info, [(s, e) for _, s, e in current_batch], params, cuts, summary
                        )
                    break
                except VisualOOM as exc:
                    attempt += 1
                    degrade_attempts += 1
                    if len(current_batch) > 1:
                        half = max(1, len(current_batch) // 2)
                        logger.warning("CUDA OOM，batch %d -> %d 后重试", len(current_batch), half)
                        current_batch = current_batch[:half]
                        continue
                    if attempt > max_retries:
                        if fallback_ids:
                            smaller = fallback_ids.pop(0)
                            logger.warning("多次 OOM，切换到更小的视觉模型：%s", smaller)
                            self.analyzer.unload()
                            self._load_visual_model(smaller)
                            attempt = 0
                            continue
                        raise RuntimeError(f"窗口显存不足且已无降级空间: {exc}") from exc
                    params = params.degrade()
                    logger.warning("CUDA OOM，降级参数后重试(%d/%d)：%s", attempt, max_retries, params.to_dict())

            for (idx, start, end), (events, meta) in zip(current_batch, results):
                logger.info(
                    "窗口 %d/%d [%.1f-%.1f s]：%d 帧 -> %d 事件，推理 %.1fs（batch=%d）",
                    idx + 1, len(windows), start, end, meta.get("frames", 0), len(events),
                    meta.get("infer_seconds", 0.0), meta.get("batch_size", 1),
                )
                total_frames += int(meta.get("frames", 0))
                all_events.extend(events)
                window_metas.append(meta)
                cache[f"{idx}:{start:.3f}-{end:.3f}"] = {
                    "events": [e.to_dict() for e in events], "meta": meta,
                }
            ckpt.save_window_cache(cache)
            # 若因 OOM 缩小了 batch，剩下的窗口放回队列下一轮处理
            leftover = todo[len(current_batch):]
            if leftover:
                queue = leftover + queue

        events = finalize(
            all_events, info.duration,
            dedup_similarity=float(vcfg["dedup_similarity"]),
            merge_similarity=float(vcfg["merge_similarity"]),
            min_seconds=float(vcfg["min_event_seconds"]),
        )
        logger.info("视觉事件：原始 %d -> 合并去重后 %d", len(all_events), len(events))

        meta = {
            "model_id": self.analyzer.model_id,
            "frame_source": window_metas[0].get("frame_source") if window_metas else None,
            "params": params.to_dict(),
            "window_count": len(windows),
            "windows": [{k: v for k, v in m.items() if k != "raw_output"} for m in window_metas],
            "total_frames_analyzed": total_frames,
            "raw_event_count": len(all_events),
            "degrade_attempts": degrade_attempts,
            "peak_vram": bench.peak_vram_mb(),
        }
        return events, meta

    def _load_visual_model(self, model_id: str | None = None) -> None:
        try:
            self.analyzer.load(model_id)
        except Exception as exc:
            if not _looks_like_oom(exc):
                raise
            fallback = list(self.cfg.visual.get("fallback_model_ids", []))
            if not fallback:
                raise
            logger.warning("视觉模型加载显存不足，改用 %s", fallback[0])
            self.analyzer.load(fallback[0])

    def close(self) -> None:
        self.analyzer.unload()
        self.asr.unload()


def _looks_like_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _speech_events_from_payload(payload: dict[str, Any]) -> list[SpeechEvent]:
    events = []
    for item in payload.get("segments", []):
        words = [SpeechWord(**w) for w in item.get("words", [])]
        data = {k: v for k, v in item.items() if k != "words"}
        events.append(SpeechEvent(words=words, **data))
    return events
