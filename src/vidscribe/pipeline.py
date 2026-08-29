"""单个视频的完整处理流程（探测 -> 视觉 -> 语音 -> Timeline -> 导出 -> benchmark）。

模型只加载一次，多视频复用；每个阶段独立落盘，支持断点续跑。

阶段产物照旧写 `cache/videos/<视频标识>/*.json` 和 `output/<视频名>/*`，格式一个字没改。
变的是"这套旧结果还能不能当缓存用"由数据库回答（见 DbRun）：
视频指纹 + 视觉模型 + 视觉配置哈希 + ASR 模型 + ASR 配置哈希，五项全对才算命中。
数据库里的 analysis_runs 只有在这次的事件、语音段、逐词全部写进库之后才会标 completed；
中途抛异常一律标 failed 并记原因，不会留下"库里说成了、其实没跑完"的假记录。
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from . import benchmark as bench
from .cache import slug_for
from .checkpoint import Checkpoint
from .config import Config
from .db import open_db
from .db import repo as db_repo
from .emotions import label_for as emotion_label
from .events import SpeechEvent, SpeechWord, VisualEvent, finalize
from .language import LanguageRenderer, decide_output_language, labels_for
from .logging_setup import get_logger
from .progress import report as report_progress
from .speech.emotion import EmotionRecognizer, emotion_peaks, relabel
from .speech.sentences import split_sentences
from .speech.speakers import SpeakerTagger
from .speech.whisper_asr import WhisperASR

from .timeline.engine import action_track, build_timeline, filter_timeline
from .timeline.exporters import write_json, write_srt, write_timeline_txt
from .video_io import VideoInfo, detect_scene_cuts, plan_windows, probe_video
from .visual import prompts
from .visual.factory import backend_for, create_analyzer
from .visual.face import FaceEmotion
from .visual.face import annotate as annotate_faces
from .visual.qwen_vl import VisualOOM, VisualParams

logger = get_logger(__name__)


class DbRun:
    """一次分析在数据库里的落脚点。

    库里记的是状态，不是产物：JSON 照旧落盘，这里只登记"这个视频、这套配置、跑到哪一步"。
    库开不起来（文件被占、磁盘满）不该让分析跑不了——那种情况下所有方法退化成空操作，
    缓存判断也退回老规矩（文件在就复用），只在日志里说一声。
    """

    def __init__(self, cfg: Config, video_path: Path, force: bool = False):
        self.cfg = cfg
        self.video_path = video_path
        self.force = force
        self.db: Any = None
        self.video_id: int | None = None
        self.analysis_id: int | None = None
        self.reused = False
        self.sig: dict[str, Any] = {}
        try:
            self.db = open_db(cfg)
            self.sig = db_repo.signature(cfg)
            self.video_id = db_repo.upsert_video(self.db, video_path,
                                                 cache_slug=slug_for(video_path))
            hit = None if force else db_repo.find_cached_analysis(self.db, self.video_id, self.sig)
            if hit is not None:
                # 五项全对的老记录：接着写这一条，别每跑一次就堆一条一模一样的历史。
                # 状态不动（还是 completed）：万一这次中途崩了，那条老结果本来也是好的
                self.analysis_id = int(hit["id"])
                self.reused = True
            else:
                self.analysis_id = db_repo.create_analysis(self.db, self.video_id, self.sig)
        except Exception as exc:
            logger.warning("数据库用不了，这次只按文件跑（缓存判断退回老规矩）：%s", exc)
            self.db = None
            self.video_id = None
            self.analysis_id = None

    @property
    def active(self) -> bool:
        return self.db is not None and self.analysis_id is not None

    # --- 缓存判断 -------------------------------------------------------
    def reuse_gate(self, stage: str) -> bool:
        """Checkpoint 问"这个阶段的旧结果能用吗"，答案由数据库给。

        语音只看 ASR 模型与配置哈希，视觉只看视觉那两项——改了 whisper 参数不该把
        跑了几分钟的 Qwen 结果一起作废。导入的老记录哈希是 'imported'，永远不命中。
        """
        if self.force:
            return False
        if self.db is None or self.video_id is None:
            return True
        try:
            return db_repo.stage_cache_ok(self.db, self.video_id, self.sig, stage)
        except Exception as exc:
            logger.warning("查缓存状态失败，按可复用处理：%s", exc)
            return True

    # --- 写状态 ---------------------------------------------------------
    def note_probe(self, info: Any) -> None:
        """探测完把时长分辨率补进视频表（第一次登记时还没探测过）。"""
        if not self.active:
            return
        try:
            db_repo.upsert_video(self.db, self.video_path, info=info.to_dict(),
                                 cache_slug=slug_for(self.video_path))
        except Exception as exc:
            logger.warning("视频信息写库失败：%s", exc)

    def save_speech(self, payload: dict[str, Any]) -> tuple[int, int]:
        """语音段 + 逐词写库。段是空的就什么都不做，免得把上一次的好数据清掉
        （skip_speech / 语音识别失败都会给一个空 payload）。"""
        if not self.active:
            return 0, 0
        segments = [s for s in (payload.get("segments") or []) if isinstance(s, dict)]
        if not segments:
            return 0, 0
        return db_repo.save_speech_segments(self.db, self.analysis_id, segments)

    def save_visual(self, events: list[dict[str, Any]]) -> int:
        if not self.active or not events:
            return 0
        return db_repo.save_visual_events(self.db, self.analysis_id, events)

    def finish(self, *, visual_count: int, speech_count: int, out_dir: Path,
               artifacts: list[tuple[str, Path]] | None = None) -> None:
        """标 completed。**只在这里标**，而且必须是所有数据都写完之后。"""
        if not self.active:
            return
        try:
            for kind, path in (artifacts or []):
                if path.is_file() and path.stat().st_size > 0:
                    db_repo.register_artifact(self.db, self.video_id, kind, path)
            db_repo.finish_analysis(self.db, self.analysis_id, scene_count=visual_count,
                                    speech_count=speech_count, output_dir=out_dir)
        except Exception as exc:  # 库写不进去不代表分析没跑成，但状态得是真的
            logger.warning("分析状态写库失败（这条记录仍是 running）：%s", exc)

    def fail(self, exc: BaseException) -> None:
        """记失败。绝不把已经 completed 的老记录抹成失败，但这次失败也一定留痕。"""
        if not self.active:
            return
        error = f"{type(exc).__name__}: {exc}"
        try:
            if self.reused:
                # 接的是一条本来就成功的记录：那条不动，另开一条 failed 记这次的事
                failed_id = db_repo.create_analysis(self.db, self.video_id, self.sig)
                db_repo.fail_analysis(self.db, failed_id, error)
                logger.warning("这次没跑完，已另记一条 failed（复用的那条成功记录保持不变）")
                return
            db_repo.fail_analysis(self.db, self.analysis_id, error)
        except Exception as inner:
            logger.warning("失败状态写库也失败了：%s", inner)



class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        model_dir = str(cfg.path("model_dir"))
        self.model_dir = model_dir
        self.analyzer = create_analyzer(cfg.visual, model_dir, cfg.mirrors)
        self.asr = WhisperASR(cfg.speech, model_dir, cfg.mirrors)
        self.emotion = EmotionRecognizer(cfg.speech.get("emotion", {}), model_dir, cfg.mirrors)
        self.speaker = SpeakerTagger(cfg.speech.get("speaker", {}), model_dir, cfg.mirrors)
        # 人脸表情：跟视觉模型无关的独立通道（YuNet + HSEmotion，CPU onnx）
        self.face = FaceEmotion(cfg.visual.get("face_emotion", {}), model_dir)
        self.env = bench.environment_snapshot()
        logger.info("视觉后端：%s（%s）", self.analyzer.backend, self.analyzer.model_id)


    # --------------------------------------------------------------- 对外入口
    def run_video(self, video_path: str | Path, force: bool = False,
                  skip_visual: bool = False, skip_speech: bool = False,
                  force_speech: bool = False, translate: bool = False) -> dict[str, Any]:
        """入口签名跟以前一样。外面这层只管数据库里的状态：

        开一条 analysis_run（或接上五项全对的那条），跑完标 completed，抛异常标 failed。
        流程本身在 _run_video 里，一行没改。
        """
        video_path = Path(video_path).resolve()
        run = DbRun(self.cfg, video_path, force=force)
        try:
            return self._run_video(video_path, run, force=force, skip_visual=skip_visual,
                                   skip_speech=skip_speech, force_speech=force_speech,
                                   translate=translate)
        except BaseException as exc:
            run.fail(exc)
            raise

    def _run_video(self, video_path: Path, run: DbRun, force: bool = False,
                   skip_visual: bool = False, skip_speech: bool = False,
                   force_speech: bool = False, translate: bool = False) -> dict[str, Any]:
        timer = bench.Timer()
        ckpt = Checkpoint(self.cfg.path("cache_dir"), video_path, reuse_gate=run.reuse_gate)
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
        run.note_probe(info)

        # 2) 语音识别（必须在视觉分析之前：最终输出语言由音频语言决定）
        if skip_speech:
            speech_payload = {"available": False, "reason": "skipped", "language": None, "segments": []}
            logger.warning("按要求跳过语音识别")
            report_progress("speech", 1.0, "已跳过语音识别", video=info.name)
        elif ckpt.done("speech") and not force_speech:
            speech_payload = ckpt.load("speech")
            # 老缓存是"一段多句"的，这里补切一刀，不用为了断句重跑 whisper
            before = speech_payload.get("segments") or []
            # 老缓存里常常整片没标点（whisper 对口语素材就这样），先用 ct-punc 补上再切。
            # 标点直接贴到 words 上，时间戳还是 whisper 的原生精度，不用重跑识别。
            punctuation = self.asr.punctuate(before)
            after = split_sentences(before)
            if punctuation.get("available") or len(after) != len(before):
                speech_payload["segments"] = after
                if punctuation.get("available"):
                    speech_payload["punctuation"] = punctuation
                # 情绪是按老边界判的，套到子句上不对，删掉让下面重判一次
                speech_payload.pop("emotion", None)
                speech_payload.pop("emotion_peaks", None)
                # 说话人同理：声纹是按老句子边界取的，边界变了就重判
                speech_payload.pop("speaker", None)
                ckpt.save("speech", speech_payload)
                logger.info("已有语音结果重排：%d 段 -> %d 句（补标点 %d 段）", len(before),
                            len(after), punctuation.get("restored_segments") or 0)

            logger.info("复用已有语音识别结果：%d 句", len(speech_payload.get("segments", [])))
            report_progress("speech", 1.0,
                            f"复用已有语音结果（{len(speech_payload.get('segments', []))} 句）", video=info.name)

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

        # 2.5) 说话人分离：跟语言无关，紧接着语音识别做，句子边界就是刚切出来的这批
        self._annotate_speaker(info, speech_payload, ckpt, timer)

        # 3) 语言判定：程序决定 output_language，不交给视觉模型自己选
        lcfg = self.cfg.language
        decision = decide_output_language(
            speech_payload,
            default_language=str(lcfg.get("default_language", "zh")),
            min_confidence=float(lcfg.get("min_language_confidence", 0.4)),
        )
        renderer = LanguageRenderer(decision.output_language)
        # 3.5) 语音情绪：用 whisper 已经切好的句子边界，时间轴天然对齐。
        # 放在语言判定之后：情绪显示名要跟 output_language 一致（英文视频出 happy，中文出开心）。
        self._annotate_emotion(info, speech_payload, ckpt, timer, decision.output_language)
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
        # 画面情绪开关变了就得重跑：情绪是视觉模型在同一次推理里给的，缓存里补不出来。
        # 只重跑语音（skip_visual）时不做这个判断，否则会把已有画面结果白白丢掉。
        want_visual_emotion = bool(self.cfg.visual.get("emotion_enabled", True))
        if (not skip_visual and cached_visual is not None
                and bool(cached_visual.get("emotion_enabled")) != want_visual_emotion):
            logger.info("画面情绪开关变了（缓存 %s，本次 %s），重新分析",
                        cached_visual.get("emotion_enabled"), want_visual_emotion)
            cached_visual = None

        if skip_visual and cached_visual is not None:
            # "只重跑语音"这种用法不该把已有画面结果清掉：有缓存就照常复用
            visual_events = [VisualEvent(**e) for e in cached_visual["events"]]
            visual_meta = cached_visual.get("meta", {})
            logger.info("跳过视觉分析，复用已有结果：%d 个事件", len(visual_events))
            report_progress("visual", 1.0,
                            f"跳过画面分析，复用已有结果（{len(visual_events)} 事件）", video=info.name)
        elif skip_visual:
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
                "emotion_enabled": want_visual_emotion,
            })
        # 4.5) 人脸表情：在原始帧上单独判，覆盖视觉模型顺带给的情绪
        self._annotate_face_emotion(info, visual_events, visual_meta, ckpt, timer,
                                   decision.output_language, want_visual_emotion)
        # 复用缓存时情绪显示名可能还是上一次的语言：标签固定英文存着，按需重渲，不用重跑模型
        for ev in visual_events:
            if ev.emotion_en:
                ev.emotion = emotion_label(ev.emotion_en, decision.output_language)
        write_json(out_dir / "visual_events.json", {
            "video": info.name,
            "duration": info.duration,
            "output_language": decision.output_language,
            "emotion_enabled": want_visual_emotion,
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
            # 两条独立时间戳轨：动作按事件归并，表情来自人脸模型的 2fps 采样。
            # 都是已算好的结果重排一遍，不额外推理。
            actions = action_track(visual_events)
            face_spans = ((visual_meta.get("face") or {}).get("segments") or []) \
                if isinstance(visual_meta.get("face"), dict) else []
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
                        "speech_speakers": e["speech_speakers"],
                        "speech_emotion": e["speech_emotion"],
                        "speech_emotion_en": e["speech_emotion_en"],
                        "speech_emotion_intensity": e["speech_emotion_intensity"],
                        "visual_emotion": e["visual_emotion"],
                        "visual_emotion_en": e["visual_emotion_en"],
                        "visual_emotion_intensity": e["visual_emotion_intensity"],
                        "quality": e["quality"],
                    }
                    for e in filtered
                ],
                "action_track": actions,
                "expression_track": face_spans,
            }
            write_json(out_dir / "timeline.json", timeline_doc)
            write_timeline_txt(out_dir / "timeline.txt", info.name, info.duration, language, filtered,
                               output_language=decision.output_language,
                               actions=actions, emotions=face_spans)
            srt_kind = write_srt(
                out_dir / "timeline.srt",
                speech_payload.get("segments", []),
                [e.to_dict() for e in visual_events],
            )
            ckpt.save("timeline", {"entries": len(filtered), "srt_kind": srt_kind})
            report_progress("timeline", 1.0, f"导出完成（timeline {len(filtered)} 条）", video=info.name)

        # 5) 顺手翻译（可选）：模型还在显存里，这时候翻最便宜，省掉单独跑一次约 15s 的加载
        translation: dict[str, Any] | None = None
        if translate:
            with timer.stage("translate_seconds"):
                translation = self._translate_stage(out_dir, info.name)

        # 6) Benchmark

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
                "stage_seconds": visual_meta.get("stage_seconds"),
                "generated_tokens": visual_meta.get("generated_tokens"),
                "prompt_tokens_max": visual_meta.get("prompt_tokens_max"),
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

        # 数据库：所有 JSON 都写完了，这时候才把结果写进库并标 completed。
        # 顺序很关键——先写数据（事件、语音段、逐词），最后一步才改状态，
        # 中间任何一步炸了，这条记录就停在 running / failed，不会出现"库里说成了、盘上没跑完"。
        from .audio import wav_path  # noqa: PLC0415

        segments_saved, words_saved = run.save_speech(speech_payload)
        events_saved = run.save_visual([e.to_dict() for e in visual_events])
        run.finish(visual_count=len(visual_events), speech_count=len(speech_events),
                   out_dir=out_dir,
                   artifacts=[("source_video", video_path),
                              ("words_srt", out_dir / "timeline.srt"),
                              ("preview_audio",
                               wav_path(self.cfg.path("cache_dir"), video_path))])
        if run.active:
            logger.info("数据库已记：分析 #%s（%s），视觉事件 %d / 语音段 %d / 逐词 %d",
                        run.analysis_id, "接上已有记录" if run.reused else "新记录",
                        events_saved, segments_saved, words_saved)


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
            "translation": translation,
            "benchmark": benchmark,
        }

    # --------------------------------------------------------------- 翻译阶段
    def _translate_stage(self, out_dir: Path, video_name: str) -> dict[str, Any] | None:
        """分析结束、模型还在显存里的时候把没有译文的行补上（增量，原文不覆盖）。

        单独点"翻译"要重开进程加载模型（实测约 15s）；这里直接复用，
        所以只剩解码时间。翻译失败不影响已经导出的结果，只记日志。
        """
        from .translate import translate_output  # noqa: PLC0415

        def on_progress(done: int, total: int) -> None:
            # 分析部分已经 100% 了，这里只更新文字，不把总进度条拉回去
            report_progress("translate", 1.0, f"翻译 {done}/{total} 行",
                            video=video_name, done=done, total=total)

        try:
            result = translate_output(self.cfg, out_dir, analyzer=self.analyzer,
                                      on_progress=on_progress)
        except Exception as exc:
            logger.warning("顺手翻译失败（已导出的结果不受影响）：%s: %s",
                           type(exc).__name__, str(exc)[:200])
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
        if result.get("ok"):
            logger.info("翻译完成：语音 %s/%s 段、事件 %s/%s 个 -> %s",
                        result.get("speech_translated", 0), result.get("speech_total", 0),
                        result.get("event_translated", 0), result.get("event_total", 0),
                        result.get("target_language"))
        else:
            logger.warning("翻译未完成：%s %s", result.get("reason"), result.get("detail") or "")
        return result

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
        # 缓存键带上输出语言 + 模型 + 画面情绪开关：换语言、换模型、开关情绪重跑时都不能复用旧描述
        model_tag = self.analyzer.model_id.split("/")[-1]
        emotion_tag = "em1" if vcfg.get("emotion_enabled", True) else "em0"

        def key_of(idx: int, s: float, e: float) -> str:
            return f"{model_tag}|{output_language}|{emotion_tag}|{idx}:{s:.3f}-{e:.3f}"


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
                self.emotion.unload()
                self.speaker.unload()
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
                    if attempt > max_retries or not params.can_degrade():
                        if fallback_ids:
                            smaller = fallback_ids.pop(0)
                            logger.warning("多次 OOM，切换到更小的视觉模型：%s", smaller)
                            self.analyzer.unload()
                            self._load_visual_model(smaller)
                            attempt = 0
                            continue
                        raise RuntimeError(f"窗口显存不足且已无降级空间: {exc}") from exc
                    params = params.degrade(reason="cuda_oom")
                    logger.warning("CUDA OOM，降级参数后重试(%d/%d)：%s | %s",
                                   attempt, max_retries, params.to_dict(),
                                   params.degrade_history[-1] if params.degrade_history else "")

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
            max_event_seconds=float(vcfg.get("max_event_seconds") or 12.0),
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
            # 视觉阶段分项耗时：定位瓶颈用，不依赖外部 profiler
            "stage_seconds": {
                key: round(sum(float(m.get(f"{key}_seconds") or 0.0) for m in window_metas), 3)
                for key in ("frame_decode", "chat_template", "processor", "generate", "text_decode")
            },
            "generated_tokens": sum(int(m.get("generated_tokens") or 0) for m in window_metas),
            "prompt_tokens_max": max([int(m.get("prompt_tokens") or 0) for m in window_metas] or [0]),
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


    def _annotate_speaker(self, info: VideoInfo, speech_payload: dict[str, Any],
                          ckpt: Checkpoint, timer: bench.Timer) -> None:
        """给每句补 speaker / speaker_confidence（谁说的），结果一起进断点缓存。

        人数是声纹自己定的，不固定几人：cam++ 声纹 + 谱聚类扫 k，用余弦轮廓系数挑。
        分不出来时老实判成 1 人（宁可少判，不给错答案）——同卵双胞胎那种音色就是这种情况。
        用哪个声纹模型看配置（界面「声纹」下拉：英文 / 中文），换了模型缓存作废重判。
        """
        segments = speech_payload.get("segments") or []
        if not self.speaker.enabled or not segments:
            return
        language = speech_payload.get("language")
        wanted = self.speaker.model_id
        cached = speech_payload.get("speaker")
        if cached is not None:
            if (cached.get("model") or {}).get("id") == wanted:
                logger.info("复用已有说话人结果：%s 人 / %d 句", cached.get("speakers", "?"), len(segments))
                return
            # 换了声纹模型，旧结论不能留：编号和人数都可能变
            logger.info("已有说话人结果来自 %s，本次要用 %s，重新判一次",
                        (cached.get("model") or {}).get("id"), wanted)

        with timer.stage("speaker_seconds"):
            try:
                report_progress("speech", 0.85, "加载声纹模型 / 解码音频", video=info.name)
                meta = self.speaker.annotate(info, segments, language)
            except Exception as exc:  # noqa: BLE001 - 声纹是增强项，不能连累转写
                logger.error("说话人分离失败：%s", exc)
                logger.debug(traceback.format_exc())
                meta = {"available": False, "reason": f"error: {exc}"[:300]}

        speech_payload["speaker"] = meta
        ckpt.save("speech", speech_payload)
        if meta.get("available"):
            report_progress("speech", 0.9, f"说话人分离完成（{meta.get('speakers')} 人）",
                            video=info.name)
        else:
            logger.info("本条没做说话人分离：%s", meta.get("reason"))

    def _annotate_face_emotion(self, info: VideoInfo, events: list[VisualEvent],
                               visual_meta: dict[str, Any], ckpt: Checkpoint,
                               timer: bench.Timer, output_language: str,
                               emotion_enabled: bool) -> None:
        """人脸表情：在原始帧上扫全片，覆盖视觉模型顺带给的情绪。

        判过就不重判（visual_meta 里有 face 元信息当标记），跟说话人那边一个套路。
        判不出人脸的事件保留视觉模型的情绪，`emotion_source` 会写清楚是谁判的。
        """
        if not self.face.enabled or not events:
            return
        cached = visual_meta.get("face")
        # 缺 segments 的是加表情轨之前存下来的旧缓存：重算一遍（21s 上下），否则表情轨永远是空的
        if isinstance(cached, dict) and cached.get("available") and cached.get("segments"):
            logger.info("复用已有画面表情结果：%s 个事件来自人脸模型，表情段 %d",
                        cached.get("events_from_face"), len(cached.get("segments") or []))
            return

        fcfg = self.cfg.visual.get("face_emotion", {}) or {}
        with timer.stage("face_seconds"):
            try:
                report_progress("visual", 0.9, "人脸表情识别", video=info.name)
                samples = self.face.scan(
                    info,
                    on_progress=lambda p: report_progress(
                        "visual", 0.9 + 0.08 * p, "人脸表情识别", video=info.name),
                )
                meta = annotate_faces(events, samples,
                                      min_score=float(fcfg.get("min_score", 0.35)))
            except Exception as exc:  # noqa: BLE001 - 表情是增强项，不能连累画面事件
                logger.error("人脸表情识别失败：%s", exc)
                logger.debug(traceback.format_exc())
                meta = {"available": False, "reason": f"error: {exc}"[:300]}
            finally:
                self.face.unload()

        visual_meta["face"] = meta
        ckpt.save("visual", {
            "events": [e.to_dict() for e in events],
            "meta": visual_meta,
            "output_language": output_language,
            "emotion_enabled": emotion_enabled,
        })

    def _annotate_emotion(self, info: VideoInfo, speech_payload: dict[str, Any],
                          ckpt: Checkpoint, timer: bench.Timer,
                          output_language: str) -> None:
        """给每句语音补情绪标签，并挑出情绪最强的几句作为高光冻帧候选。

        情绪一起进断点缓存：复用已有语音结果时若缓存里没有情绪，这里会补判一次；
        缓存有情绪但语言变了，只重渲显示名（emotion_en 是稳定标签），不重跑模型。
        """
        segments = speech_payload.get("segments") or []
        if not self.emotion.enabled or not segments:
            return
        ecfg = self.cfg.speech.get("emotion", {})
        # 判过就不重判：注意不能看段里有没有 emotion 键——SpeechEvent 现在总会带这个字段
        # （值是 None），所以用 payload 级别的 emotion 元信息当标记。
        cached = speech_payload.get("emotion")
        if cached is not None:
            if cached.get("language") == output_language:
                logger.info("复用已有语音情绪结果：%d 段", len(segments))
                return
            count = relabel(segments, output_language)
            cached["language"] = output_language
            speech_payload["emotion_peaks"] = emotion_peaks(
                segments,
                top_n=int(ecfg.get("peak_top_n", 5)),
                min_intensity=float(ecfg.get("peak_min_intensity", 0.5)),
            )
            ckpt.save("speech", speech_payload)
            logger.info("语音情绪换成 %s 显示：%d 段（没有重跑模型）", output_language, count)
            return

        with timer.stage("emotion_seconds"):
            try:
                report_progress("speech", 0.9, "加载情绪模型 / 解码音频", video=info.name)
                meta = self.emotion.annotate(info, segments, output_language)
            except Exception as exc:
                logger.error("语音情绪识别失败：%s", exc)
                logger.debug(traceback.format_exc())
                meta = {"available": False, "reason": f"error: {exc}"[:300],
                        "language": output_language}

        speech_payload["emotion"] = meta
        speech_payload["emotion_peaks"] = emotion_peaks(
            segments,
            top_n=int(ecfg.get("peak_top_n", 5)),
            min_intensity=float(ecfg.get("peak_min_intensity", 0.5)),
        )
        ckpt.save("speech", speech_payload)
        peaks = speech_payload["emotion_peaks"]
        if peaks:
            logger.info("情绪最强的 %d 句（可作高光冻帧点）：%s", len(peaks),
                        "; ".join(f"{p['freeze_at']:.2f}s {p['emotion']}({p['intensity']:.2f})"
                                  for p in peaks))
        report_progress("speech", 1.0, f"语音情绪完成（{meta.get('annotated', 0)} 段）",
                        video=info.name)

    def close(self) -> None:
        self.analyzer.unload()
        self.asr.unload()
        self.emotion.unload()
        self.speaker.unload()


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
