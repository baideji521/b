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
from .language import LanguageRenderer, decide_output_language, labels_for
from .logging_setup import get_logger
from .progress import report as report_progress
from .speech.whisper_asr import WhisperASR
from .timeline.engine import build_timeline, filter_timeline
from .timeline.exporters import write_json, write_srt, write_timeline_txt
from .video_io import VideoInfo, detect_scene_cuts, plan_windows, probe_video
from .visual import prompts
from .visual.factory import backend_for, create_analyzer
from .visual.qwen_vl import VisualOOM, VisualParams

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        model_dir = str(cfg.path("model_dir"))
        self.model_dir = model_dir
        self.analyzer = create_analyzer(cfg.visual, model_dir, cfg.mirrors)
        self.asr = WhisperASR(cfg.speech, model_dir, cfg.mirrors)
        self.env = bench.environment_snapshot()
        logger.info("视觉后端：%s（%s）", self.analyzer.backend, self.analyzer.model_id)


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
                report_progress("probe", 1.0, "复用已有探测结果", video=video_path.name)
            else:
                report_progress("probe", 0.02, "读取视频元信息", video=video_path.name)
                info = probe_video(video_path)
                cuts = []
                if self.cfg.visual.get("scene_detect", True):
                    cuts = detect_scene_cuts(
                        info,
                        sample_fps=float(self.cfg.visual.get("scene_sample_fps", 3.0)),
                        threshold=float(self.cfg.visual.get("scene_threshold", 0.35)),
                        on_progress=lambda f: report_progress(
                            "probe", 0.05 + 0.95 * f,
                            f"镜头切点检测 {f * 100:.0f}%", video=info.name,
                        ),
                    )
                ckpt.save("probe", {"video": info.to_dict(), "scene_cuts": cuts})
                report_progress("probe", 1.0, f"镜头切点 {len(cuts)} 个", video=info.name)
        logger.info(
            "视频: %.2fs, %dx%d, %.3f fps, %d 帧, 音轨=%s",
            info.duration, info.width, info.height, info.fps, info.total_frames, info.has_audio,
        )
        write_json(out_dir / "video_metadata.json", {"video": info.to_dict(), "scene_cuts": cuts})

        # 2) 语音识别（必须在视觉分析之前：最终输出语言由音频语言决定）
        if skip_speech:
            speech_payload = {"available": False, "reason": "skipped", "language": None, "segments": []}
            logger.warning("按要求跳过语音识别")
            report_progress("speech", 1.0, "已跳过语音识别", video=info.name)
        elif ckpt.done("speech"):
            speech_payload = ckpt.load("speech")
            logger.info("复用已有语音识别结果：%d 段", len(speech_payload.get("segments", [])))
            report_progress("speech", 1.0,
                            f"复用已有语音结果（{len(speech_payload.get('segments', []))} 段）", video=info.name)
        else:
            with timer.stage("speech_seconds"):
                try:
                    report_progress("speech", 0.01, "加载语音模型 / 解码音频", video=info.name)
                    speech_payload = self.asr.transcribe(info)
                except Exception as exc:
                    logger.error("语音识别失败：%s", exc)
                    logger.debug(traceback.format_exc())
                    speech_payload = {"available": False, "reason": f"error: {exc}"[:300],
                                      "language": None, "segments": []}
            ckpt.save("speech", speech_payload)
            report_progress("speech", 1.0,
                            f"语音识别完成（{len(speech_payload.get('segments', []))} 段）", video=info.name)

        # 3) 语言判定：程序决定 output_language，不交给视觉模型自己选
        lcfg = self.cfg.language
        decision = decide_output_language(
            speech_payload,
            default_language=str(lcfg.get("default_language", "zh")),
            min_confidence=float(lcfg.get("min_language_confidence", 0.4)),
        )
        renderer = LanguageRenderer(decision.output_language)
        speech_payload["language_decision"] = decision.to_dict()
        _ensure_speech_originals(speech_payload)
        write_json(out_dir / "speech_events.json", speech_payload)
        speech_events = _speech_events_from_payload(speech_payload)

        # 4) 视觉分析（用 output_language 生成最终描述，内部事实固定英文）
        visual_meta: dict[str, Any] = {}
        visual_events: list[VisualEvent] = []
        cached_visual = ckpt.load("visual") if ckpt.done("visual") else None
        if cached_visual is not None and cached_visual.get("output_language") != decision.output_language:
            logger.info("已有视觉结果是 %s，本次需要 %s，重新分析",
                        cached_visual.get("output_language"), decision.output_language)
            cached_visual = None
        cached_model = (cached_visual or {}).get("meta", {}).get("model_id")
        if cached_visual is not None and cached_model and cached_model != self.analyzer.model_id:
            logger.info("已有视觉结果来自 %s，本次要用 %s，重新分析", cached_model, self.analyzer.model_id)
            cached_visual = None

        if skip_visual:
            logger.warning("按要求跳过视觉分析")
            report_progress("visual", 1.0, "已跳过画面分析", video=info.name)
        elif cached_visual is not None:
            visual_events = [VisualEvent(**e) for e in cached_visual["events"]]
            visual_meta = cached_visual.get("meta", {})
            logger.info("复用已有视觉分析结果：%d 个事件", len(visual_events))
            report_progress("visual", 1.0, f"复用已有画面结果（{len(visual_events)} 事件）", video=info.name)
        else:
            with timer.stage("visual_seconds"):
                visual_events, visual_meta = self._run_visual(
                    info, cuts, ckpt, decision.output_language, renderer
                )
            ckpt.save("visual", {
                "events": [e.to_dict() for e in visual_events],
                "meta": visual_meta,
                "output_language": decision.output_language,
            })
        write_json(out_dir / "visual_events.json", {
            "video": info.name,
            "duration": info.duration,
            "output_language": decision.output_language,
            "meta": visual_meta,
            "events": [e.to_dict() for e in visual_events],
        })

        # 5) Timeline 合并 + 导出
        with timer.stage("timeline_seconds"):
            report_progress("timeline", 0.2, "合并画面事件与语音", video=info.name)
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
                # 语言字段：original_* 是音频事实，output_language 是最终自然语言
                "original_language": decision.dominant_language or decision.detected_language,
                "output_language": decision.output_language,
                "detected_language": decision.detected_language,
                "language_confidence": decision.language_confidence,
                "dominant_language": decision.dominant_language,
                "secondary_languages": decision.secondary_languages,
                "language_default_used": decision.default_used,
                "language_reason": decision.reason,
                "language_render": renderer.stats(),
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
            write_timeline_txt(out_dir / "timeline.txt", info.name, info.duration, language, filtered,
                               output_language=decision.output_language)
            srt_kind = write_srt(
                out_dir / "timeline.srt",
                speech_payload.get("segments", []),
                [e.to_dict() for e in visual_events],
            )
            ckpt.save("timeline", {"entries": len(filtered), "srt_kind": srt_kind})
            report_progress("timeline", 1.0, f"导出完成（timeline {len(filtered)} 条）", video=info.name)

        # 5) Benchmark
        benchmark = {
            "video": info.to_dict(),
            "environment": self.env,
            "visual_model": {
                "model_id": self.analyzer.model_id,
                "backend": self.analyzer.backend,
                "load_seconds": self.analyzer.load_seconds,
                "frame_source": visual_meta.get("frame_source"),
                "params": visual_meta.get("params"),
                "windows": visual_meta.get("window_count"),
                "analyzed_frames": visual_meta.get("total_frames_analyzed"),
                "degrade_attempts": visual_meta.get("degrade_attempts", 0),
            },
            "speech_model": speech_payload.get("model"),
            "language": {
                **decision.to_dict(),
                "render": renderer.stats(),
                "labels": labels_for(decision.output_language),
            },
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
            "output_language": decision.output_language,
            "language_decision": decision.to_dict(),
            "language_render": renderer.stats(),
            "speech_available": bool(speech_payload.get("available")),
            "benchmark": benchmark,
        }

    # --------------------------------------------------------------- 视觉阶段
    def _run_visual(self, info: VideoInfo, cuts: list[float], ckpt: Checkpoint,
                    output_language: str, renderer: LanguageRenderer) -> tuple[list[VisualEvent], dict]:
        vcfg = self.cfg.visual
        self.analyzer.set_output_language(output_language)
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
        # 缓存键带上输出语言 + 模型：换语言或换模型重跑时不能复用旧描述
        model_tag = self.analyzer.model_id.split("/")[-1]

        def key_of(idx: int, s: float, e: float) -> str:
            return f"{model_tag}|{output_language}|{idx}:{s:.3f}-{e:.3f}"


        all_cached = all(key_of(i, s, e) in cache for i, (s, e) in enumerate(windows))
        if all_cached:
            logger.info("全部 %d 个窗口命中缓存，跳过视觉模型加载", len(windows))
        else:
            report_progress("visual", 0.01, f"加载视觉模型（共 {len(windows)} 个窗口）", video=info.name,
                            done=0, total=len(windows))
            # 语音已经跑完，先把 whisper 的显存还回去再加载视觉模型。
            # 12GB 卡上实测：不释放会让视觉模型加载 8.6s -> 95.7s、首批推理 11.4s -> 163.6s
            # （驱动把权重换页到共享内存）。重新加载 whisper 只要 ~20s，明显划算。
            if self.cfg.runtime.get("unload_speech_before_visual", True):
                self.asr.unload()
            self._load_visual_model()

        done_windows = 0

        def report_window(detail: str) -> None:
            # 视觉阶段占总进度的大头，进度按已完成窗口数推进（含缓存命中的窗口）
            report_progress("visual", done_windows / max(len(windows), 1), detail,
                            video=info.name, done=done_windows, total=len(windows))


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
                key = key_of(idx, start, end)
                if key in cache:
                    cached = cache[key]
                    events = [VisualEvent(**e) for e in cached["events"]]
                    all_events.extend(events)
                    window_metas.append(cached["meta"])
                    total_frames += int(cached["meta"].get("frames", 0))
                    logger.info("窗口 %d/%d 复用缓存（%d 事件）", idx + 1, len(windows), len(events))
                    done_windows += 1
                    report_window(f"窗口 {idx + 1}/{len(windows)} 复用缓存")
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
                cache[key_of(idx, start, end)] = {
                    "events": [e.to_dict() for e in events], "meta": meta,
                }
                done_windows += 1
                report_window(f"窗口 {idx + 1}/{len(windows)} [{start:.1f}-{end:.1f}s] -> {len(events)} 事件")
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

        # Language Renderer：最终描述统一到 output_language（模型还在显存里，改写最便宜）
        report_window("统一最终语言描述")
        events = self._render_language(events, renderer)

        meta = {
            "model_id": self.analyzer.model_id,
            "backend": self.analyzer.backend,
            "output_language": output_language,
            "language_render": renderer.stats(),
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

    def _render_language(self, events: list[VisualEvent], renderer: LanguageRenderer) -> list[VisualEvent]:
        """最终自然语言层：语种不符的描述先让模型改写，再走模板/标记降级。"""
        bad = renderer.needs_rewrite(events)
        if bad and self.cfg.language.get("rewrite_mismatch_with_model", True):
            texts = [events[i].description or events[i].event for i in bad]
            rewritten = self.analyzer.rewrite_texts(texts, renderer.output_language)
            for i, text in zip(bad, rewritten):
                if renderer.apply_rewrite(events[i], text):
                    logger.info("事件 %d 描述已改写为 %s", events[i].id, renderer.output_language)
        events = renderer.finalize_events(events)
        stats = renderer.stats()
        if stats["template_or_kept"]:
            logger.warning("仍有 %d 个事件未能生成 %s 描述（已标记 language_fallback）",
                           stats["template_or_kept"], renderer.output_language)
        return events

    def _load_visual_model(self, model_id: str | None = None) -> None:
        self._ensure_backend(model_id)
        try:
            self.analyzer.load(model_id)
        except Exception as exc:
            if not _looks_like_oom(exc):
                raise
            fallback = list(self.cfg.visual.get("fallback_model_ids", []))
            if not fallback:
                raise
            logger.warning("视觉模型加载显存不足，改用 %s", fallback[0])
            self._ensure_backend(fallback[0])
            self.analyzer.load(fallback[0])

    def _ensure_backend(self, model_id: str | None) -> None:
        """降级/切换模型时后端可能也要换（Qwen3-VL <-> MiniCPM 接口完全不同）。"""
        if not model_id:
            return
        wanted = backend_for(self.cfg.visual, model_id)
        if wanted == self.analyzer.backend:
            return
        logger.info("视觉后端切换：%s -> %s（%s）", self.analyzer.backend, wanted, model_id)
        language = self.analyzer.output_language
        self.analyzer.unload()
        self.analyzer = create_analyzer(self.cfg.visual, self.model_dir, self.cfg.mirrors, model_id)
        self.analyzer.set_output_language(language)


    def close(self) -> None:
        self.analyzer.unload()
        self.asr.unload()


def _looks_like_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _ensure_speech_originals(payload: dict[str, Any]) -> None:
    """补齐 original_text / original_language。

    旧版本（以及断点续跑复用的 speech.json）里没有这两个字段，
    但"原始语音识别结果不可被覆盖"是硬要求，所以在写出前统一回填。
    """
    detected = payload.get("language")
    for seg in payload.get("segments", []):
        seg.setdefault("original_text", seg.get("text"))
        seg.setdefault("original_language", seg.get("language") or detected)


def _speech_events_from_payload(payload: dict[str, Any]) -> list[SpeechEvent]:
    events = []
    for item in payload.get("segments", []):
        words = [SpeechWord(**w) for w in item.get("words", [])]
        data = {k: v for k, v in item.items() if k != "words"}
        events.append(SpeechEvent(words=words, **data))
    return events
