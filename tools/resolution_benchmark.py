"""Qwen3-VL 视觉输入分辨率 Benchmark（只测视觉输入分辨率，不动其它任何东西）。

约束（来自需求）：
- 不改原始视频、不改 faster-whisper、不改 Timeline、不改语言判定/GUI/断点。
- 只调整喂给 Qwen3-VL 的每帧像素预算，从而得到不同的视觉输入分辨率。
- 其它参数全部固定：同一视频、同样窗口、同样 fps / max_frames / max_new_tokens、
  greedy 解码、batch=1、同一模型实例。
- 无人值守：不询问、不中断，OOM/失败只记录并继续。

分辨率对齐的硬约束：Qwen3-VL 的 patch=16、merge=2，宽高必须是 32 的倍数，
所以严格 3:4 且 32 对齐的尺寸只能是 (96k, 128k)。810x1080 / 720x960 本身不满足，
用不放大、最接近的合法尺寸替代，requested 与 actual 都会记录在结果里。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe import benchmark as bench  # noqa: E402
from vidscribe.config import Config  # noqa: E402
from vidscribe.events import VisualEvent, finalize, text_similarity  # noqa: E402
from vidscribe.logging_setup import get_logger, setup_logging  # noqa: E402
from vidscribe.video_io import detect_scene_cuts, plan_windows, probe_video, smart_size  # noqa: E402
from vidscribe.visual.qwen_vl import QwenVLAnalyzer, VisualOOM, VisualParams  # noqa: E402

logger = get_logger("resbench")

OUTPUT_LANGUAGE = "en"  # 固定，避免语言差异污染对比

PERSON_WORDS = {
    "man", "woman", "person", "people", "boy", "girl", "child", "children", "baby",
    "male", "female", "guy", "lady", "men", "women", "human", "adult", "couple",
    "person_a", "person_b", "hand", "hands",
}

# requested 来自需求；aligned 是严格 3:4 且 32 对齐的合法尺寸（不放大、不拉伸）。
# 32 对齐后仍要严格 3:4，尺寸只能取 (96k, 128k)：768x1024 / 672x896 / 576x768 / 480x640。
# 逐维四舍五入到 32 会得到 800x1088（0.735）这类被拉伸的尺寸，所以这里显式给出目标尺寸。
REQUESTED: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((810, 1080), (768, 1024)),
    ((720, 960), (672, 896)),
    ((576, 768), (576, 768)),
    ((480, 640), (480, 640)),
]


@dataclass
class RunResult:
    requested: str
    resolution: str = ""
    aspect_ratio: float = 0.0
    tokens_per_frame: int = 0
    status: str = "OK"
    error: str | None = None
    video_duration: float = 0.0
    input_frames: int = 0
    qwen_calls: int = 0
    preprocess_time: float = 0.0
    qwen_time: float = 0.0
    total_visual_time: float = 0.0
    peak_vram_allocated_mb: float = 0.0
    peak_vram_reserved_mb: float = 0.0
    peak_vram_nvidia_smi_mb: float = 0.0
    fits_in_physical_vram: bool | None = None
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0
    system_ram_used_mb: float = 0.0
    system_ram_peak_mb: float = 0.0
    events: list[dict] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    quality_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        return data


class HardwareMonitor:
    """采样 GPU 利用率 / 显存 / 系统内存。nvidia-smi 每秒一次，开销可忽略。"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self.ram_used: list[float] = []

    def _sample(self) -> None:
        try:
            import psutil  # noqa: PLC0415
        except Exception:
            psutil = None
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                line = (out.stdout or "").strip().splitlines()
                if line:
                    util, mem = [p.strip() for p in line[0].split(",")[:2]]
                    self.gpu_util.append(float(util))
                    self.gpu_mem.append(float(mem))
            except Exception:
                pass
            if psutil is not None:
                try:
                    self.ram_used.append(psutil.virtual_memory().used / 1024 / 1024)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.gpu_util, self.gpu_mem, self.ram_used = [], [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float]:
        def _mean(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 1) if xs else 0.0

        return {
            "gpu_utilization_mean": _mean(self.gpu_util),
            "gpu_utilization_max": round(max(self.gpu_util), 1) if self.gpu_util else 0.0,
            "peak_vram_nvidia_smi_mb": round(max(self.gpu_mem), 1) if self.gpu_mem else 0.0,
            "system_ram_used_mb": _mean(self.ram_used),
            "system_ram_peak_mb": round(max(self.ram_used), 1) if self.ram_used else 0.0,
        }


# --------------------------------------------------------------------- 质量指标
def quality_metrics(events: list[VisualEvent], duration: float, windows: int,
                    parsed_windows: int) -> dict[str, Any]:
    persons: set[str] = set()
    objects: set[str] = set()
    actions: set[str] = set()
    scenes: list[str] = []
    interactions = 0
    max_persons_in_event = 0
    covered = 0.0
    ts_counter: dict[str, int] = {}
    ocr: set[str] = set()

    last_scene = None
    scene_changes = 0
    for ev in sorted(events, key=lambda e: e.start):
        subj = [s for s in (ev.subjects or [])]
        p = [s for s in subj if s in PERSON_WORDS]
        o = [s for s in subj if s not in PERSON_WORDS]
        persons.update(p)
        objects.update(o)
        max_persons_in_event = max(max_persons_in_event, len(p))
        if len(subj) >= 2:
            interactions += 1
        if ev.action:
            actions.add(ev.action)
        if ev.scene:
            scenes.append(ev.scene)
            if last_scene is not None and ev.scene != last_scene:
                scene_changes += 1
            last_scene = ev.scene
        covered += max(0.0, ev.end - ev.start)
        ts_counter[ev.timestamp_source] = ts_counter.get(ev.timestamp_source, 0) + 1
        if ev.ocr_text:
            ocr.add(ev.ocr_text.strip())

    monotonic = all(
        b.start >= a.start - 1e-6
        for a, b in zip(sorted(events, key=lambda e: e.start), sorted(events, key=lambda e: e.start)[1:])
    )
    in_range = all(-0.01 <= ev.start <= duration + 0.01 and ev.start <= ev.end <= duration + 0.01
                   for ev in events)
    frameish = sum(v for k, v in ts_counter.items() if k in ("frame_based", "hybrid"))
    return {
        "event_count": len(events),
        "person_labels": sorted(persons),
        "distinct_person_labels": len(persons),
        "max_persons_in_single_event": max_persons_in_event,
        "object_labels": sorted(objects),
        "distinct_objects": len(objects),
        "action_labels": sorted(actions),
        "distinct_actions": len(actions),
        "scene_labels": sorted(set(scenes)),
        "scene_changes": scene_changes,
        "interaction_events": interactions,
        "interaction_ratio": round(interactions / len(events), 3) if events else 0.0,
        "ocr_texts": sorted(ocr),
        "timestamp_sources": ts_counter,
        "frame_or_hybrid_ratio": round(frameish / len(events), 3) if events else 0.0,
        "coverage_ratio": round(min(1.0, covered / duration), 3) if duration > 0 else 0.0,
        "timestamps_monotonic": monotonic,
        "timestamps_in_range": in_range,
        "parsed_window_ratio": round(parsed_windows / windows, 3) if windows else 0.0,
        "avg_description_chars": round(
            sum(len(e.description or "") for e in events) / len(events), 1) if events else 0.0,
    }


def description_agreement(events: list[VisualEvent], ref: list[VisualEvent]) -> float:
    """与参考分辨率的描述一致度：对每个参考事件找时间重叠的最佳匹配再取平均。"""
    if not ref or not events:
        return 0.0
    scores = []
    for r in ref:
        best = 0.0
        for e in events:
            overlap = min(r.end, e.end) - max(r.start, e.start)
            if overlap <= 0:
                continue
            best = max(best, text_similarity(r.description, e.description))
        scores.append(best)
    return round(sum(scores) / len(scores), 4)


QUALITY_WEIGHTS = {
    "description_agreement": 0.30,
    "person_count_match": 0.20,
    "action_diversity": 0.15,
    "coverage": 0.15,
    "timestamp_quality": 0.10,
    "parse_success": 0.10,
}


def score_run(run: RunResult, consensus_persons: int, max_actions: int) -> tuple[float, dict[str, float]]:
    q = run.quality
    delta = abs(q.get("distinct_person_labels", 0) - consensus_persons)
    person_match = 1.0 if delta == 0 else (0.6 if delta == 1 else 0.2)
    action_div = min(1.0, q.get("distinct_actions", 0) / max(max_actions, 1))

    components = {
        "description_agreement": q.get("description_agreement", 0.0),
        "person_count_match": person_match,
        "action_diversity": round(action_div, 4),
        "coverage": q.get("coverage_ratio", 0.0),
        "timestamp_quality": q.get("frame_or_hybrid_ratio", 0.0),
        "parse_success": q.get("parsed_window_ratio", 0.0),
    }
    score = sum(components[k] * w for k, w in QUALITY_WEIGHTS.items())
    return round(score, 4), components


# --------------------------------------------------------------------- 单次运行
def run_one(analyzer: QwenVLAnalyzer, info, windows, cuts, tokens: int,
            requested: tuple[int, int], base_params: VisualParams) -> RunResult:
    import torch  # noqa: PLC0415

    w, h = smart_size(info.width, info.height, tokens * 32 * 32)
    result = RunResult(requested=f"{requested[0]}x{requested[1]}",
                       resolution=f"{w}x{h}", aspect_ratio=round(w / max(h, 1), 4),
                       tokens_per_frame=tokens, video_duration=info.duration)

    params = VisualParams(
        fps=base_params.fps,
        max_frames=base_params.max_frames,
        min_frames=base_params.min_frames,
        max_pixels_tokens=tokens,
        # 每帧预算 = total / 帧数，保证 min(max_pixels, per_frame) 就是我们要测的分辨率
        total_pixels_tokens=tokens * base_params.max_frames,
        max_new_tokens=base_params.max_new_tokens,
    )

    monitor = HardwareMonitor()
    bench.reset_peak_vram()
    monitor.start()
    started = time.perf_counter()
    raw_events: list[VisualEvent] = []
    infer_total = 0.0
    frames_total = 0
    parsed_windows = 0
    try:
        for i, (start, end) in enumerate(windows):
            events, meta = analyzer.analyze_window(info, start, end, params, cuts, None)
            infer_total += float(meta.get("infer_seconds", 0.0))
            frames_total += int(meta.get("frames", 0))
            actual = meta.get("resolution")
            if actual and f"{actual[0]}x{actual[1]}" != result.resolution:
                logger.warning("窗口 %d 实际分辨率 %s 与目标 %s 不一致", i + 1, actual, result.resolution)
                result.resolution = f"{actual[0]}x{actual[1]}"
            if events:
                parsed_windows += 1
            raw_events.extend(events)
            result.qwen_calls += 1
            logger.info("[%s] 窗口 %d/%d：%d 帧 -> %d 事件，推理 %.1fs",
                        result.resolution, i + 1, len(windows), meta.get("frames", 0),
                        len(events), meta.get("infer_seconds", 0.0))
    except VisualOOM as exc:
        result.status = "OOM"
        result.error = str(exc)[:400]
        logger.error("[%s] CUDA OOM，记录后继续下一个分辨率", result.resolution)
    except Exception as exc:  # 模型错误：记录 FAIL 后继续
        result.status = "FAIL"
        result.error = f"{type(exc).__name__}: {exc}"[:400]
        logger.error("[%s] 失败：%s", result.resolution, result.error)
    finally:
        wall = time.perf_counter() - started
        monitor.stop()
        peak = bench.peak_vram_mb() or {}
        result.total_visual_time = round(wall, 2)
        result.qwen_time = round(infer_total, 2)
        result.preprocess_time = round(max(0.0, wall - infer_total), 2)
        result.input_frames = frames_total
        result.peak_vram_allocated_mb = peak.get("allocated_mb", 0.0)
        result.peak_vram_reserved_mb = peak.get("reserved_mb", 0.0)
        for key, value in monitor.summary().items():
            setattr(result, key, value)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if result.status == "OK":
        events = finalize(
            raw_events, info.duration,
            dedup_similarity=0.72, merge_similarity=0.82, min_seconds=0.4,
        )
        result.events = [e.to_dict() for e in events]
        result.quality = quality_metrics(events, info.duration, len(windows), parsed_windows)
        result._events_obj = events  # type: ignore[attr-defined]
    return result


# --------------------------------------------------------------------- 主流程
def main() -> int:
    cfg = Config.load(ROOT)
    cfg.ensure_dirs()
    setup_logging(cfg.path("log_dir"), name="resolution_benchmark")

    video = ROOT / "test.mp4"
    candidates = sorted(cfg.path("input_dir").glob("*.mp4")) + sorted(cfg.path("input_dir").glob("*.mov"))
    best_candidate = None
    for path in candidates:
        probe = probe_video(path)
        ratio = probe.width / max(probe.height, 1)
        if abs(ratio - 0.75) < 0.02:
            if best_candidate is None or abs(probe.duration - 40) < abs(best_candidate[1].duration - 40):
                best_candidate = (path, probe)
    if best_candidate:
        video, info = best_candidate
        logger.info("使用 input/ 里的 3:4 视频：%s", video.name)
    else:
        info = probe_video(video)
        logger.info("input/ 里没有 3:4 视频，使用项目根目录的 %s", video.name)

    logger.info("视频：%s %dx%d ratio=%.4f 时长=%.2fs",
                info.name, info.width, info.height, info.width / max(info.height, 1), info.duration)

    vcfg = cfg.visual
    cuts = detect_scene_cuts(info, sample_fps=float(vcfg["scene_sample_fps"]),
                             threshold=float(vcfg["scene_threshold"]))
    windows = plan_windows(info.duration, cuts,
                           window_seconds=float(vcfg["window_seconds"]),
                           overlap_seconds=float(vcfg["window_overlap_seconds"]),
                           long_threshold=float(vcfg["long_video_threshold"]))
    logger.info("窗口 %d 个（所有分辨率共用）：%s", len(windows),
                ", ".join(f"{s:.1f}-{e:.1f}" for s, e in windows))

    base_params = VisualParams(
        fps=float(vcfg["fps"]),
        max_frames=int(vcfg["max_frames"]),
        min_frames=int(vcfg["min_frames"]),
        max_pixels_tokens=int(vcfg["max_pixels_tokens"]),
        total_pixels_tokens=int(vcfg["total_pixels_tokens"]),
        max_new_tokens=int(vcfg["max_new_tokens"]),
    )

    analyzer = QwenVLAnalyzer(vcfg, str(cfg.path("model_dir")), cfg.mirrors)
    analyzer.set_output_language(OUTPUT_LANGUAGE)
    analyzer.load()

    # 目标分辨率：requested -> 32 对齐且严格 3:4 的合法尺寸（不放大）
    plans: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for requested, aligned in REQUESTED:
        if aligned[0] > info.width or aligned[1] > info.height:
            logger.warning("原始视频 %dx%d 低于 %dx%d，跳过（不放大）",
                           info.width, info.height, aligned[0], aligned[1])
            continue
        tokens = max(4, int(round(aligned[0] * aligned[1] / (32 * 32))))
        actual = smart_size(info.width, info.height, tokens * 32 * 32)
        if actual != aligned:
            logger.warning("目标 %dx%d 与 smart_size 结果 %dx%d 不一致，按实际值记录", *aligned, *actual)
        plans.append((requested, aligned, tokens))
        logger.info("计划：requested %dx%d -> 实际 %dx%d（%d tokens/帧，ratio=%.4f）",
                    requested[0], requested[1], actual[0], actual[1], tokens,
                    actual[0] / max(actual[1], 1))

    # 附带当前默认配置作为基线（不参与"四个分辨率"的评比，只为决策提供对照）
    baseline_tokens = min(int(vcfg["max_pixels_tokens"]),
                          int(vcfg["total_pixels_tokens"]) // int(vcfg["max_frames"]))
    baseline_size = smart_size(info.width, info.height, baseline_tokens * 32 * 32)

    results: list[RunResult] = []
    for requested, aligned, tokens in plans:
        logger.info("=" * 70)
        logger.info("开始测试 requested=%dx%d -> %dx%d（%d tokens/帧）",
                    requested[0], requested[1], aligned[0], aligned[1], tokens)
        results.append(run_one(analyzer, info, windows, cuts, tokens, requested, base_params))

    logger.info("=" * 70)
    logger.info("基线：当前默认配置 %dx%d（%d tokens/帧）", baseline_size[0], baseline_size[1], baseline_tokens)
    baseline = run_one(analyzer, info, windows, cuts, baseline_tokens, baseline_size, base_params)
    baseline.requested = f"BASELINE {baseline_size[0]}x{baseline_size[1]}"

    ok_runs = [r for r in results if r.status == "OK"]
    all_runs = results + [baseline]

    # 显存是否真的装得下：reserved 超过物理显存说明驱动在往共享内存换页，
    # 这种情况下的耗时数据不能代表健康配置，必须显式标记并排除出推荐。
    gpu_env = bench.gpu_info()
    total_vram = float(gpu_env.get("total_vram_mb") or 0.0)
    for r in all_runs:
        if r.status == "OK" and total_vram > 0:
            r.fits_in_physical_vram = r.peak_vram_reserved_mb <= total_vram * 0.98

    # 一致度用"与其它所有成功分辨率的平均相似度"，避免拿最高分辨率当参考时它自己恒为 1.0
    for r in all_runs:
        if r.status != "OK":
            continue
        others = [o for o in all_runs if o is not r and o.status == "OK"]
        scores = [description_agreement(getattr(r, "_events_obj", []), getattr(o, "_events_obj", []))
                  for o in others]
        r.quality["description_agreement"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # 人物数量取所有成功运行的众数作为共识，而不是信任某一个分辨率
    person_counts = [r.quality.get("distinct_person_labels", 0) for r in all_runs if r.status == "OK"]
    consensus_persons = max(set(person_counts), key=person_counts.count) if person_counts else 0
    consensus_actions = max((r.quality.get("distinct_actions", 0) for r in all_runs
                             if r.status == "OK"), default=1)
    for r in all_runs:
        if r.status != "OK":
            continue
        score, components = score_run(r, consensus_persons, consensus_actions)
        r.quality_score = score
        r.quality["score_components"] = components

    # ---------------- 选择最佳平衡点 ----------------
    decision: dict[str, Any] = {
        "consensus_person_count": consensus_persons,
        "max_distinct_actions": consensus_actions,
        "physical_vram_mb": total_vram,
    }
    chosen: RunResult | None = None
    if ok_runs:
        fitting = [r for r in ok_runs if r.fits_in_physical_vram is not False]
        pool_note = "全部候选都超出物理显存，只能在换页的结果里选" if not fitting else ""
        pool = fitting or ok_runs
        best_q = max(r.quality_score or 0.0 for r in pool)
        acceptable = []
        for r in pool:
            q = r.quality_score or 0.0
            checks = {
                "person_count_match_consensus":
                    r.quality.get("distinct_person_labels", 0) == consensus_persons,
                "parse_success>=0.9": r.quality.get("parsed_window_ratio", 0.0) >= 0.9,
                "coverage>=0.85": r.quality.get("coverage_ratio", 0.0) >= 0.85,
                "quality_within_0.05_of_best": q >= best_q - 0.05,
                "fits_in_physical_vram": r.fits_in_physical_vram is not False,
            }
            r.quality["accept_checks"] = checks
            if all(checks.values()):
                acceptable.append(r)
        # 质量接近时选更快的：先筛出质量达标的，再取最快
        final_pool = acceptable or pool
        chosen = min(final_pool, key=lambda r: (r.qwen_time, r.peak_vram_reserved_mb))
        slowest = max(ok_runs, key=lambda r: r.qwen_time)
        decision.update({
            "best_quality_score_in_pool": round(best_q, 4),
            "vram_fitting": [r.resolution for r in fitting],
            "acceptable": [r.resolution for r in acceptable],
            "fallback_used": not acceptable,
            "pool_note": pool_note,
            "chosen": chosen.resolution,
            "speed_gain_vs_slowest_percent": round(
                (slowest.qwen_time - chosen.qwen_time) / slowest.qwen_time * 100, 1)
            if slowest.qwen_time > 0 else 0.0,
            "vram_saving_vs_slowest_percent": round(
                (slowest.peak_vram_reserved_mb - chosen.peak_vram_reserved_mb)
                / max(slowest.peak_vram_reserved_mb, 1e-6) * 100, 1),
            "quality_delta_vs_best": round((chosen.quality_score or 0.0) - best_q, 4),
            "baseline_comparison": {
                "baseline_resolution": baseline.resolution,
                "baseline_qwen_time": baseline.qwen_time,
                "baseline_quality_score": baseline.quality_score,
                "chosen_is_faster_than_baseline": chosen.qwen_time <= baseline.qwen_time,
            },
        })


    env = bench.environment_snapshot()
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": {
            "name": info.name, "path": info.path, "duration": info.duration,
            "native_resolution": f"{info.width}x{info.height}",
            "aspect_ratio": round(info.width / max(info.height, 1), 4),
            "fps": info.fps, "has_audio": info.has_audio,
        },
        "fixed_parameters": {
            "model": analyzer.model_id,
            "dtype": vcfg.get("dtype"),
            "attn_implementation": vcfg.get("attn_implementation"),
            "sampling_fps": base_params.fps,
            "max_frames": base_params.max_frames,
            "min_frames": base_params.min_frames,
            "max_new_tokens": base_params.max_new_tokens,
            "do_sample": False,
            "temperature": "n/a (greedy)",
            "batch_size": 1,
            "prompt": "同一提示词模板",
            "output_language": OUTPUT_LANGUAGE,
            "windows": [[round(s, 3), round(e, 3)] for s, e in windows],
            "scene_cuts": cuts,
        },
        "environment": env,
        "alignment_note": (
            "Qwen3-VL 要求宽高为 32 的倍数（patch 16 × merge 2），因此严格 3:4 的合法尺寸只能是 (96k,128k)："
            "768x1024 / 672x896 / 576x768 / 480x640。810x1080 与 720x960 本身不满足 32 对齐，"
            "若逐维四舍五入会得到 800x1088(0.735) / 704x960(0.733) 这类被轻微拉伸的尺寸，"
            "所以改用不放大、不拉伸、最接近的合法尺寸 768x1024 与 672x896，requested/resolution 都已记录。"
        ),
        "quality_score_weights": QUALITY_WEIGHTS,
        "quality_score_note": (
            "quality_score 是可复现的启发式指标，全部由本次真实输出计算：与其它分辨率的平均描述一致度、"
            "人物数量是否等于各分辨率的众数共识、动作多样性、时间覆盖率、时间戳质量、解析成功率。"
            "一致度用两两平均而不是与最高分辨率对比，避免最高分辨率自己恒为 1.0 的自参考偏差；"
            "请结合各分项与 events 原文一起看。"
        ),
        "tests": [r.to_dict() for r in results],
        "baseline_current_config": baseline.to_dict(),
        "decision": decision,
    }
    for item in payload["tests"] + [payload["baseline_current_config"]]:
        item.pop("_events_obj", None)

    out_json = ROOT / "resolution_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("已写出 %s", out_json)

    write_recommendation(ROOT / "RESOLUTION_RECOMMENDATION.txt", payload, results, baseline, chosen, env)

    # ---------------- 只有全部测试都跑完才允许改默认配置 ----------------
    completed = len(results) == len(plans)
    if completed and chosen is not None and chosen.status == "OK":
        update_config(cfg, chosen, base_params)
    else:
        logger.warning("Benchmark 未全部完成或没有可用结果，保持默认配置不变")
    return 0


def write_recommendation(path: Path, payload: dict, results: list[RunResult],
                         baseline: RunResult, chosen: RunResult | None, env: dict) -> None:
    gpu = env.get("gpu", {})
    lines = [
        "=" * 40,
        "Qwen3-VL Resolution Benchmark",
        "=" * 40,
        "",
        "Video:",
        payload["video"]["name"],
        "",
        "Native Resolution:",
        payload["video"]["native_resolution"],
        "",
        "Aspect Ratio:",
        "3:4",
        "",
        "GPU:",
        f"{gpu.get('name', 'N/A')} {gpu.get('total_vram_mb', 'N/A')} MB",
        "",
        "Fixed:",
        f"model={payload['fixed_parameters']['model']}  fps={payload['fixed_parameters']['sampling_fps']}  "
        f"max_frames={payload['fixed_parameters']['max_frames']}  "
        f"max_new_tokens={payload['fixed_parameters']['max_new_tokens']}  greedy  batch=1",
        "",
        "-" * 40,
    ]
    for r in results:
        lines += [
            f"requested {r.requested}  ->  actual {r.resolution}  (ratio {r.aspect_ratio})",
            f"Status:  {r.status}" + (f"  ({r.error})" if r.error else ""),
            f"Qwen time:  {r.qwen_time} sec   (preprocess {r.preprocess_time} sec, total {r.total_visual_time} sec)",
            f"VRAM:  allocated {r.peak_vram_allocated_mb} MB / reserved {r.peak_vram_reserved_mb} MB "
            f"/ nvidia-smi {r.peak_vram_nvidia_smi_mb} MB   fits_in_physical_vram={r.fits_in_physical_vram}",
            f"GPU util:  mean {r.gpu_utilization_mean}%  max {r.gpu_utilization_max}%",
            f"System RAM:  mean {r.system_ram_used_mb} MB  peak {r.system_ram_peak_mb} MB",
            f"Frames:  {r.input_frames}   Qwen calls: {r.qwen_calls}",
        ]
        if r.status == "OK":
            q = r.quality
            lines += [
                f"Quality score:  {r.quality_score}",
                f"  events={q.get('event_count')}  persons={q.get('person_labels')}  "
                f"max_persons_in_event={q.get('max_persons_in_single_event')}",
                f"  actions={q.get('action_labels')}",
                f"  scenes={q.get('scene_labels')}  scene_changes={q.get('scene_changes')}",
                f"  objects={q.get('object_labels')}",
                f"  interaction_ratio={q.get('interaction_ratio')}  coverage={q.get('coverage_ratio')}",
                f"  timestamps={q.get('timestamp_sources')}  frame_or_hybrid={q.get('frame_or_hybrid_ratio')}  "
                f"monotonic={q.get('timestamps_monotonic')}  in_range={q.get('timestamps_in_range')}",
                f"  description_agreement_vs_reference={q.get('description_agreement')}",
                f"  ocr={q.get('ocr_texts')}",
            ]
        lines += ["-" * 40]

    lines += [
        f"BASELINE (current default) {baseline.resolution}",
        f"Status: {baseline.status}   Qwen time: {baseline.qwen_time} sec   "
        f"VRAM reserved: {baseline.peak_vram_reserved_mb} MB   quality_score: {baseline.quality_score}",
        "-" * 40,
        "",
        "=" * 40,
        "",
        "Recommended:",
        "",
        chosen.resolution if chosen else "N/A (no successful run)",
        "",
        "Reason:",
        "",
    ]
    d = payload["decision"]
    if chosen:
        lines += [
            f"速度：比最慢的成功分辨率快 {d.get('speed_gain_vs_slowest_percent')}%",
            f"显存：比最慢的成功分辨率省 {d.get('vram_saving_vs_slowest_percent')}%",
            f"视觉质量：quality_score {chosen.quality_score}（候选池内最高 {d.get('best_quality_score_in_pool')}，"
            f"差距 {d.get('quality_delta_vs_best')}）",
            f"人物数量共识：{d.get('consensus_person_count')} 人（各分辨率投票）",
            f"物理显存：{d.get('physical_vram_mb')} MB；装得下的候选：{d.get('vram_fitting')}",
            f"通过质量门槛的候选：{d.get('acceptable')}",
            f"基线对照：{d.get('baseline_comparison')}",
        ]
        if d.get("pool_note"):
            lines.append(f"注意：{d['pool_note']}")
        if d.get("fallback_used"):
            lines.append("注意：没有候选同时满足全部质量门槛，已退化为在候选池里选最快的，请人工复核 events 原文。")
    else:
        lines.append("所有分辨率都失败，未给出推荐。")
    lines += ["", "=" * 40, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("已写出 %s", path)


def update_config(cfg: Config, chosen: RunResult, base_params: VisualParams) -> None:
    """把最佳分辨率写回 config.json（只改两个像素预算字段）。"""
    path = cfg.root / "config.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    old = {
        "max_pixels_tokens": data["visual"].get("max_pixels_tokens"),
        "total_pixels_tokens": data["visual"].get("total_pixels_tokens"),
    }
    data["visual"]["max_pixels_tokens"] = chosen.tokens_per_frame
    data["visual"]["total_pixels_tokens"] = chosen.tokens_per_frame * base_params.max_frames
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    logger.info("已把默认视觉输入分辨率改为 %s：max_pixels_tokens %s -> %d，total_pixels_tokens %s -> %d",
                chosen.resolution, old["max_pixels_tokens"], chosen.tokens_per_frame,
                old["total_pixels_tokens"], chosen.tokens_per_frame * base_params.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
