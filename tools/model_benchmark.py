"""统一模型 Benchmark：同一视频 / 同一窗口 / 同一提示词，逐个视觉后端 × 参数配置实测。

设计原则（对应任务书）：
- 不改原始视频，不改 Whisper / Timeline / Checkpoint / 镜像逻辑，只是调用它们。
- 语音只跑一次并缓存（模型无关），视觉部分才是每个模型都要重复测的。
- 每个模型先做 10s Smoke Test，跑不通直接 FAILED，不浪费时间跑完整矩阵。
- Cold Start 1 次 + 正式 3 次，取平均/最快/最慢。
- OOM 自动降级（batch -> frames -> 分辨率 -> tokens）并记录，绝不静默切 CPU。
- 任何模型失败都只标记 FAILED 并继续下一个。
- 质量分数只用可客观计算的指标 + benchmark/ground_truth.json 里的人工标注，不编数字。

用法（全自动，无需交互）：
    .venv\\Scripts\\python.exe tools/model_benchmark.py
    .venv\\Scripts\\python.exe tools/model_benchmark.py --configs A,B --repeats 2
    .venv\\Scripts\\python.exe tools/model_benchmark.py --smoke-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe import benchmark as bench  # noqa: E402
from vidscribe.config import Config  # noqa: E402
from vidscribe.events import finalize  # noqa: E402
from vidscribe.language import decide_output_language, text_matches_language  # noqa: E402
from vidscribe.logging_setup import get_logger, setup_logging  # noqa: E402
from vidscribe.video_io import detect_scene_cuts, plan_windows, probe_video  # noqa: E402
from vidscribe.visual.factory import backend_for, create_analyzer  # noqa: E402
from vidscribe.visual.qwen_vl import VisualOOM, VisualParams  # noqa: E402

logger = get_logger("modelbench")

# 参数配置矩阵：只动 fps / max_frames / max_new_tokens，分辨率与窗口保持一致，
# 这样同一模型不同配置之间是单变量可比的。
CONFIGS: dict[str, dict[str, Any]] = {
    "A": {"fps": 1.5, "max_frames": 16, "max_new_tokens": 512},
    "B": {"fps": 1.0, "max_frames": 12, "max_new_tokens": 256},
    "C": {"fps": 0.75, "max_frames": 8, "max_new_tokens": 192},
    "D": {"fps": 0.5, "max_frames": 8, "max_new_tokens": 128},
}

PERSON_TOKENS = {
    "man", "men", "woman", "women", "person", "people", "boy", "girl", "guy", "lady",
    "male", "female", "man_a", "man_b", "woman_a", "woman_b", "person_a", "person_b",
    "adult", "teenager", "human",
}

# ------------------------------------------------------- 幻觉判定（分类 + 可审计）
# 旧实现是"出现 marker 词就算幻觉"，把 green leaves 的 leaves、
# 描述里顺带提到的 phone 全判成幻觉，三个模型一律 1.0，等于没有信息量。
# 现在按类别判定，每类都带 unless 例外，并保留证据原文供人工复核。
HALLUCINATION_RULES: dict[str, dict[str, Any]] = {
    # 真值：全程并排坐着，没有人进出画面
    "locomotion": {
        "fields": ["event", "action", "description"],
        "patterns": [r"\bwalk(?:s|ing)?\b", r"\brun(?:s|ning)?\b", r"\benter(?:s|ing)?\b",
                     r"\bexit(?:s|ing)?\b", r"\bleav(?:es|ing)\s+(?:the\s+)?(?:room|frame|scene|shot)\b",
                     r"\bstands?\s+up\b", r"\bgets?\s+up\b", r"\bapproach(?:es|ing)?\b"],
        "unless": [r"\b(?:green|tree|autumn)\s+leaves\b", r"\bleaves\s+of\b"],
        "why": "真值 person_enters_or_leaves_frame=false，两人全程并排就坐",
    },
    # 真值：single_continuous_shot=true，shot_cuts 为空
    "shot_change": {
        "fields": ["event", "action", "description", "scene"],
        "patterns": [r"\bshot\s+(?:cut|change)\b", r"\bscene\s+(?:cut|change|transition)\b",
                     r"\bcuts?\s+to\b", r"\bcamera\s+(?:pans?|zooms?|moves?)\b", r"\bnew\s+scene\b"],
        "unless": [],
        "why": "真值 single_continuous_shot=true，没有任何镜头切换",
    },
    # 真值：室内窗前并排坐
    "wrong_place": {
        "fields": ["scene", "description"],
        "patterns": [r"\bkitchen\b", r"\bstreet\b", r"\boutdoors?\b", r"\bpark\b", r"\bbeach\b",
                     r"\bcar\b", r"\bvehicle\b", r"\boffice\b", r"\brestaurant\b", r"\bcafe\b",
                     r"\bstore\b", r"\bshop\b", r"\bgym\b", r"\bbathroom\b", r"\bbedroom\b",
                     r"\bstage\b", r"\bclassroom\b", r"\bstadium\b"],
        "unless": [r"\bcar(?:ry|ries|rying|d|ds)\b", r"\bcandy\b"],
        "why": "真值场景：室内、窗前、两人并排坐；没有出现其它场所",
    },
    # 真值 key_objects：糖果袋 / 糖果 / 帽子 / 眼镜 / 手表
    "wrong_object": {
        "fields": ["event", "action", "description"],
        "patterns": [r"\bphone\b", r"\bsmartphone\b", r"\blaptop\b", r"\bcomputer\b",
                     r"\btelevision\b", r"\btv\b", r"\bdog\b", r"\bcat\b", r"\bbicycle\b",
                     r"\bguitar\b", r"\bdoor\b", r"\bmicrophone\b", r"\bbook\b", r"\bknife\b"],
        "unless": [r"\bsmart\s*watch\b"],
        "why": "真值 key_objects 只有糖果袋/糖果/帽子/眼镜/手表",
    },
    # 真值 person_count=2
    "extra_person": {
        "fields": ["event", "action", "description", "subjects"],
        "patterns": [r"\bthird\s+person\b", r"\bcrowd\b", r"\bgroup\s+of\s+people\b",
                     r"\b(?:three|four|several|many)\s+(?:people|persons|men|women)\b",
                     r"\bchild(?:ren)?\b", r"\bbaby\b"],
        "unless": [],
        "why": "真值 person_count=2，只有一位女性和一位男性",
    },
}


def _rules_from_gt(gt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """允许 ground_truth.json 里的 contradictions 覆盖/扩展默认规则。"""
    rules = {k: dict(v) for k, v in HALLUCINATION_RULES.items()}
    for name, spec in (gt.get("contradictions") or {}).items():
        if name.startswith("_"):
            continue
        merged = dict(rules.get(name) or {"fields": ["description"], "unless": [], "why": ""})
        merged.update({k: v for k, v in spec.items() if not k.startswith("_")})
        rules[name] = merged
    return rules


def _event_text(event: Any, fields: list[str]) -> str:
    parts: list[str] = []
    for name in fields:
        value = getattr(event, name, None)
        if isinstance(value, (list, tuple)):
            parts.append(" ".join(str(x) for x in value))
        elif value:
            parts.append(str(value))
    return " ".join(parts).replace("_", " ").lower()


def _hallucination_scores(events: list[Any], gt: dict[str, Any]) -> dict[str, Any]:
    """按类别统计与真值矛盾的事件；返回率 + 每类计数 + 证据原文。"""
    rules = _rules_from_gt(gt)
    n = len(events)
    per_category = {name: 0 for name in rules}
    evidence: list[dict[str, Any]] = []
    bad_events = 0
    for event in events:
        hits: list[str] = []
        for name, rule in rules.items():
            text = _event_text(event, list(rule.get("fields") or ["description"]))
            if not text:
                continue
            if any(re.search(p, text) for p in (rule.get("unless") or [])):
                continue
            matched = next((p for p in (rule.get("patterns") or []) if re.search(p, text)), None)
            if matched:
                per_category[name] += 1
                hits.append(f"{name}:{matched}")
        if hits:
            bad_events += 1
            if len(evidence) < 8:
                evidence.append({
                    "start": round(float(getattr(event, "start", 0.0)), 2),
                    "end": round(float(getattr(event, "end", 0.0)), 2),
                    "hits": hits,
                    "description": str(getattr(event, "description", ""))[:160],
                })
    # 人数矛盾单独算：subjects 里数出 >2 个人也是与真值矛盾
    gt_persons = int(gt.get("person_count") or 0)
    over = sum(1 for e in events
               if gt_persons and _person_count(list(getattr(e, "subjects", []) or [])) > gt_persons)
    per_category["extra_person"] += over
    return {
        "hallucination_rate": round(bad_events / n, 4) if n else 1.0,
        "hallucination_events": bad_events,
        "hallucination_by_category": {k: v for k, v in per_category.items() if v},
        "hallucination_evidence": evidence,
    }


def _ocr_scores(events: list[Any], gt: dict[str, Any]) -> dict[str, Any]:
    """OCR 既看命中真值文字，也看有没有凭空造字（invented）。"""
    spec = gt.get("on_screen_text") or {}
    expect = {t.lower() for t in (spec.get("expected_tokens") or [])}
    allowed = expect | {t.lower() for t in (spec.get("secondary_tokens") or [])}
    reported: list[str] = []
    for event in events:
        raw = str(getattr(event, "ocr_text", "") or "").strip()
        if raw:
            reported.extend(t for t in re.split(r"[^0-9a-z\u4e00-\u9fff]+", raw.lower()) if len(t) > 2)
    hit = {t for t in allowed if any(t in r or r in t for r in reported)}
    invented = [t for t in reported if not any(a in t or t in a for a in allowed)]
    return {
        "ocr_hit": 1.0 if expect and (hit & expect) else 0.0,
        "ocr_reported": 1.0 if reported else 0.0,
        "ocr_recall": round(len(hit & expect) / len(expect), 4) if expect else None,
        "ocr_precision": round(1.0 - len(invented) / len(reported), 4) if reported else None,
        "ocr_invented_samples": sorted(set(invented))[:8],
    }



# ----------------------------------------------------------------- GPU 采样
class GpuSampler:
    """后台线程轮询 nvidia-smi，拿真实 GPU 利用率（torch 只能给显存）。"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.util: list[float] = []
        self.mem: list[float] = []

    def _run(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
               "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                if proc.returncode == 0:
                    line = proc.stdout.strip().splitlines()[0]
                    util, mem = [x.strip() for x in line.split(",")[:2]]
                    self.util.append(float(util))
                    self.mem.append(float(mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> "GpuSampler":
        self.util, self.mem = [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return {
            "gpu_util_avg": round(statistics.fmean(self.util), 1) if self.util else None,
            "gpu_util_max": round(max(self.util), 1) if self.util else None,
            "gpu_mem_used_max_mb": round(max(self.mem), 1) if self.mem else None,
            "samples": len(self.util),
        }


# ----------------------------------------------------------------- 质量评分
def _person_count(subjects: list[str]) -> int:
    return sum(1 for s in subjects if str(s).strip().lower() in PERSON_TOKENS)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-z\u4e00-\u9fff]+", (text or "").lower()) if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_events(events: list[Any], gt: dict[str, Any], duration: float,
                 output_language: str) -> dict[str, Any]:
    """只算能客观判定的指标；需要人工标注的部分来自 ground_truth.json。"""
    n = len(events)
    out: dict[str, Any] = {"event_count": n}
    if n == 0:
        out.update({
            "coverage": 0.0, "person_count_mode": None, "person_count_correct": 0.0,
            "person_consistency": 0.0, "hallucination_rate": 1.0, "ocr_hit": 0.0,
            "frame_calibrated_ratio": 0.0, "timestamp_sane_ratio": 0.0,
            "language_match_ratio": 0.0, "duplicate_ratio": 0.0, "mean_confidence": 0.0,
        })
        return out

    # 覆盖率：事件时间并集 / 视频时长
    spans = sorted((float(e.start), float(e.end)) for e in events)
    merged: list[list[float]] = []
    for s, t in spans:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    covered = sum(max(0.0, b - a) for a, b in merged)
    out["coverage"] = round(min(1.0, covered / max(duration, 1e-6)), 4)

    counts = [_person_count(list(getattr(e, "subjects", []) or [])) for e in events]
    counts_nonzero = [c for c in counts if c > 0]
    mode = statistics.mode(counts_nonzero) if counts_nonzero else None
    gt_persons = int(gt.get("person_count") or 0)
    out["person_count_mode"] = mode
    out["person_count_correct"] = 1.0 if mode == gt_persons else 0.0
    out["person_count_per_event_accuracy"] = round(
        sum(1 for c in counts if c == gt_persons) / n, 4)
    out["person_consistency"] = round(
        sum(1 for c in counts_nonzero if c == mode) / len(counts_nonzero), 4
    ) if counts_nonzero else 0.0

    out.update(_hallucination_scores(events, gt))
    out.update(_ocr_scores(events, gt))


    frame_based = sum(1 for e in events
                      if str(getattr(e, "timestamp_source", "")) != "model_estimated")
    out["frame_calibrated_ratio"] = round(frame_based / n, 4)

    sane = 0
    prev_end = -1e9
    for e in sorted(events, key=lambda x: x.start):
        ok = (0.0 - 1e-6) <= e.start < e.end <= duration + 0.05 and e.start >= prev_end - 1e-6
        sane += 1 if ok else 0
        prev_end = e.end
    out["timestamp_sane_ratio"] = round(sane / n, 4)

    lang_ok = sum(1 for e in events
                  if text_matches_language(getattr(e, "description", "") or "", output_language))
    out["language_match_ratio"] = round(lang_ok / n, 4)

    dup = 0
    seen: list[set[str]] = []
    for e in events:
        toks = _tokens(getattr(e, "description", "") or "")
        if any(_jaccard(toks, prev) >= 0.85 for prev in seen):
            dup += 1
        seen.append(toks)
    out["duplicate_ratio"] = round(dup / n, 4)
    out["mean_confidence"] = round(
        statistics.fmean([float(getattr(e, "confidence", 0.0) or 0.0) for e in events]), 4)
    return out


def stability_between_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """A/B 稳定性：do_sample=False 理论上应完全一致，不一致就是真实的不稳定。"""
    ok = [r for r in runs if r.get("status") == "OK"]
    if len(ok) < 2:
        return {"runs_compared": len(ok), "desc_agreement": None,
                "event_count_identical": None, "person_count_identical": None}
    texts = [" ".join((e.get("description") or "") for e in r["events"]) for r in ok]
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    agree = statistics.fmean([_jaccard(_tokens(texts[i]), _tokens(texts[j])) for i, j in pairs])
    counts = [len(r["events"]) for r in ok]
    persons = [r["quality"].get("person_count_mode") for r in ok]
    return {
        "runs_compared": len(ok),
        "desc_agreement": round(agree, 4),
        "event_count_identical": len(set(counts)) == 1,
        "event_counts": counts,
        "person_count_identical": len(set(persons)) == 1,
        "person_count_modes": persons,
    }


# ----------------------------------------------------------------- 语音（跑一次）
def speech_reference(cfg: Config, video: Path, out_dir: Path, force: bool) -> dict[str, Any]:
    cache = out_dir / "speech_reference.json"
    if cache.is_file() and not force:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("复用已有语音参考结果：%s（whisper %.1fs）",
                    cache.name, payload.get("whisper_seconds", 0.0))
        return payload

    from vidscribe.speech.whisper_asr import WhisperASR  # noqa: PLC0415

    asr = WhisperASR(cfg.speech, str(cfg.path("model_dir")), cfg.mirrors)
    info = probe_video(video)
    t0 = time.perf_counter()
    result = asr.transcribe(info)
    whisper_seconds = time.perf_counter() - t0
    load_seconds = float(getattr(asr, "load_seconds", 0.0) or 0.0)
    asr.unload()

    payload = {
        "whisper_seconds": round(whisper_seconds, 3),
        "whisper_load_seconds": round(load_seconds, 3),
        "language": result.get("language"),
        "language_confidence": result.get("language_confidence"),
        "segments": result.get("segments", []),
        "available": result.get("available"),
        "model": result.get("model"),
        "word_count": sum(len(s.get("words") or []) for s in result.get("segments", [])),
    }
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("语音参考完成：%.1fs，%d 段，语言 %s",
                whisper_seconds, len(payload["segments"]), payload["language"])
    return payload


# ----------------------------------------------------------------- 单次视觉运行
@dataclass
class RunResult:
    status: str = "OK"
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    peak_vram: dict[str, Any] = field(default_factory=dict)
    oom_events: int = 0
    effective_params: dict[str, Any] = field(default_factory=dict)
    windows: int = 0
    frames: int = 0
    generated_tokens: int = 0
    actual_batch: int = 1


def run_visual_once(analyzer: Any, info: Any, windows: list[tuple[float, float]],
                    params: VisualParams, cuts: list[float], batch_size: int,
                    vcfg: dict[str, Any], gt: dict[str, Any], output_language: str,
                    max_retries: int = 3) -> RunResult:
    from vidscribe.visual import prompts  # noqa: PLC0415

    res = RunResult()
    bench.reset_peak_vram()
    sampler = GpuSampler().start()
    t_all = time.perf_counter()

    all_events: list[Any] = []
    metas: list[dict[str, Any]] = []
    queue = [(i, s, e) for i, (s, e) in enumerate(windows)]
    cur_params = params
    actual_batch = max(1, batch_size)
    try:
        while queue:
            chunk = queue[:actual_batch]
            queue = queue[actual_batch:]
            summary = prompts.build_context_summary(all_events) if all_events else None
            attempt = 0
            current = chunk
            while True:
                try:
                    if len(current) == 1:
                        _, s, e = current[0]
                        results = [analyzer.analyze_window(info, s, e, cur_params, cuts, summary)]
                    else:
                        results = analyzer.analyze_windows(
                            info, [(s, e) for _, s, e in current], cur_params, cuts, summary)
                    break
                except VisualOOM as exc:
                    res.oom_events += 1
                    attempt += 1
                    if len(current) > 1:
                        half = max(1, len(current) // 2)
                        logger.warning("OOM：batch %d -> %d", len(current), half)
                        actual_batch = half
                        current = current[:half]
                        continue
                    if attempt > max_retries or not cur_params.can_degrade():
                        raise
                    cur_params = cur_params.degrade(reason="cuda_oom")
                    logger.warning("OOM：参数降级 -> %s | %s", cur_params.to_dict(),
                                   cur_params.degrade_history[-1])

            for (_, s, e), (events, meta) in zip(current, results):
                all_events.extend(events)
                metas.append(meta)
            leftover = chunk[len(current):]
            if leftover:
                queue = leftover + queue
    except Exception as exc:  # 单次运行失败不影响其它配置
        sampler.stop()
        res.status = "FAILED"
        res.error = f"{type(exc).__name__}: {exc}"[:400]
        logger.error("视觉运行失败：%s", res.error)
        logger.debug(traceback.format_exc())
        return res

    visual_seconds = time.perf_counter() - t_all
    res.gpu = sampler.stop()
    res.peak_vram = bench.peak_vram_mb() or {}
    # 子进程后端（minicpm46）的显存在父进程里读不到，用 worker 自己上报的峰值
    worker_peaks = [float(m.get("worker_peak_reserved_mb") or 0.0) for m in metas]
    if worker_peaks and max(worker_peaks) > float(res.peak_vram.get("reserved_mb") or 0.0):
        res.peak_vram = {"reserved_mb": max(worker_peaks), "allocated_mb": max(worker_peaks),
                         "source": "worker"}


    merged = finalize(
        all_events, info.duration,
        dedup_similarity=float(vcfg["dedup_similarity"]),
        merge_similarity=float(vcfg["merge_similarity"]),
        min_seconds=float(vcfg["min_event_seconds"]),
    )
    res.events = [e.to_dict() for e in merged]
    res.raw_outputs = [m.get("raw_output", "") for m in metas]
    res.quality = score_events(merged, gt, info.duration, output_language)
    res.quality["raw_event_count"] = len(all_events)
    res.windows = len(windows)
    res.frames = sum(int(m.get("frames") or 0) for m in metas)
    res.generated_tokens = sum(int(m.get("generated_tokens") or 0) for m in metas)
    res.actual_batch = actual_batch
    res.effective_params = cur_params.to_dict()
    res.timing = {
        "visual_seconds": round(visual_seconds, 3),
        "frame_decode_seconds": round(sum(float(m.get("frame_decode_seconds") or 0) for m in metas), 3),
        "chat_template_seconds": round(sum(float(m.get("chat_template_seconds") or 0) for m in metas), 3),
        "processor_seconds": round(sum(float(m.get("processor_seconds") or 0) for m in metas), 3),
        "generate_seconds": round(sum(float(m.get("generate_seconds") or m.get("infer_seconds") or 0)
                                      for m in metas), 3),
        "text_decode_seconds": round(sum(float(m.get("text_decode_seconds") or 0) for m in metas), 3),
        "resolution": metas[0].get("resolution") if metas else None,
    }
    return res


def smoke_test(analyzer: Any, info: Any, cuts: list[float], vcfg: dict[str, Any],
               gt: dict[str, Any], output_language: str) -> dict[str, Any]:
    """10 秒窗口跑通 MP4 -> frames -> model -> text -> event 全链路。"""
    params = VisualParams(
        fps=1.0, max_frames=8, min_frames=4,
        max_pixels_tokens=int(vcfg["max_pixels_tokens"]),
        total_pixels_tokens=int(vcfg["total_pixels_tokens"]),
        max_new_tokens=192,
    )
    end = min(10.0, info.duration)
    t0 = time.perf_counter()
    try:
        events, meta = analyzer.analyze_window(info, 0.0, end, params, cuts, None)
    except Exception as exc:
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:400],
                "seconds": round(time.perf_counter() - t0, 2)}
    seconds = time.perf_counter() - t0
    return {
        "status": "OK" if events else "FAILED",
        "error": None if events else "模型返回可解析事件数为 0",
        "seconds": round(seconds, 2),
        "events": [e.to_dict() for e in events],
        "raw_output": (meta.get("raw_output") or "")[:1200],
        "frames": meta.get("frames"),
        "resolution": meta.get("resolution"),
        "generate_seconds": meta.get("generate_seconds"),
        "generated_tokens": meta.get("generated_tokens"),
        "worker_peak_reserved_mb": meta.get("worker_peak_reserved_mb"),
        "quality": score_events(events, gt, info.duration, output_language),
    }


# ----------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="视觉模型统一 Benchmark（全自动，无交互）")
    ap.add_argument("--video", default=None, help="测试视频，默认自动找 40s 左右的 3:4 视频")
    ap.add_argument("--models", default=None, help="逗号分隔的 model_id，默认取 config.visual.models")
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--repeats", type=int, default=3, help="正式运行次数（cold start 另算）")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--force-speech", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="超过这个分钟数就不再开始新的运行（0=不限制）")
    ap.add_argument("--out", default="benchmark")

    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="默认会读取已有 model_benchmark.json，PASS 的组合直接跳过")
    args = ap.parse_args(argv)


    cfg = Config.load(ROOT)
    cfg.ensure_dirs()


    setup_logging(cfg.path("log_dir"), name=f"modelbench_{datetime.now():%Y%m%d_%H%M%S}")
    endpoint = cfg.mirrors.get("hf_endpoint")
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("PYTHONUTF8", "1")

    out_dir = (ROOT / args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    gt_file = out_dir / "ground_truth.json"
    gt = json.loads(gt_file.read_text(encoding="utf-8")) if gt_file.is_file() else {}
    if not gt:
        logger.warning("没有 ground_truth.json，质量分只保留可客观计算的部分")

    # --- 测试视频 ---
    video = Path(args.video) if args.video else None
    if video is not None and not video.is_absolute():
        video = ROOT / video
    if video is None:
        candidates = list((cfg.path("input_dir")).glob("*.mp4")) + sorted(ROOT.glob("*.mp4"))
        if not candidates:
            logger.error("找不到测试视频")
            return 2
        video = candidates[0]
    info = probe_video(video)
    logger.info("测试视频：%s %dx%d ratio=%.4f 时长=%.2fs fps=%.2f",
                info.name, info.width, info.height, info.width / max(info.height, 1),
                info.duration, info.fps)

    env = bench.environment_snapshot()
    total_vram = float(env["gpu"].get("total_vram_mb") or 0.0)
    if not env["gpu"].get("available"):
        logger.error("CUDA 不可用，按要求不允许偷偷退到 CPU，直接退出")
        return 3

    # --- 场景切点（一次，所有模型共用）---
    t0 = time.perf_counter()
    cuts = detect_scene_cuts(info, sample_fps=float(cfg.visual.get("scene_sample_fps", 3.0)),
                             threshold=float(cfg.visual.get("scene_threshold", 0.35)))
    scene_seconds = time.perf_counter() - t0

    # --- 语音（一次）---
    speech = speech_reference(cfg, video, out_dir, args.force_speech)
    decision = decide_output_language(
        speech, default_language=str(cfg.language.get("default_language", "zh")),
        min_confidence=float(cfg.language.get("min_language_confidence", 0.4)),
    )
    output_language = decision.output_language
    logger.info("音频语言 %s -> 输出语言 %s", speech.get("language"), output_language)

    windows = plan_windows(
        info.duration, cuts,
        window_seconds=float(cfg.visual["window_seconds"]),
        overlap_seconds=float(cfg.visual["window_overlap_seconds"]),
        long_threshold=float(cfg.visual["long_video_threshold"]),
    )
    logger.info("窗口 %d 个（所有模型共用）：%s", len(windows),
                ", ".join(f"{s:.1f}-{e:.1f}" for s, e in windows))

    # --- 候选模型 ---
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_ids = [str(m["model_id"]) for m in (cfg.visual.get("models") or [])]
        if not model_ids:
            model_ids = [str(cfg.visual["model_id"])]
    config_keys = [c.strip().upper() for c in args.configs.split(",") if c.strip()]

    # --- 断点：同一视频上已经 PASS 的 (模型, 配置) 直接复用，FAILED 最多重试 3 次 ---
    prior_models: dict[str, dict[str, Any]] = {}
    prior_file = out_dir / "model_benchmark.json"
    if args.resume and prior_file.is_file():
        try:
            old = json.loads(prior_file.read_text(encoding="utf-8"))
            if (old.get("video") or {}).get("name") == info.name:
                prior_models = {str(m.get("model_id")): m for m in (old.get("models") or [])}
                (out_dir / "model_benchmark.prev.json").write_text(
                    json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("断点续跑：复用 %d 个模型的历史结果", len(prior_models))
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取历史 benchmark 失败，全部重跑：%s", exc)


    started_all = time.perf_counter()
    def budget_left() -> bool:
        if args.time_budget_min <= 0:
            return True
        return (time.perf_counter() - started_all) / 60.0 < args.time_budget_min

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": env,
        "video": info.to_dict(),
        "ground_truth_file": str(gt_file) if gt else None,
        "scene_detect_seconds": round(scene_seconds, 3),
        "scene_cuts": cuts,
        "speech": {k: v for k, v in speech.items() if k != "segments"},
        "output_language": output_language,
        "language_decision": decision.to_dict(),
        "windows": windows,
        "fixed": {
            "window_seconds": cfg.visual["window_seconds"],
            "window_overlap_seconds": cfg.visual["window_overlap_seconds"],
            "max_pixels_tokens": cfg.visual["max_pixels_tokens"],
            "total_pixels_tokens": cfg.visual["total_pixels_tokens"],
            "batch_size": cfg.visual.get("batch_size", 1),
            "repeats": args.repeats,
        },
        "configs": CONFIGS,
        "models": [],
    }

    for model_id in model_ids:
        backend = backend_for(cfg.visual, model_id)
        prior = prior_models.get(model_id) or {}
        prior_ok = {k: v for k, v in (prior.get("configs") or {}).items()
                    if v.get("status") == "OK" and k in config_keys}
        pending = [k for k in config_keys if k not in prior_ok]
        prior_retries = int(prior.get("retries") or 0)
        if prior.get("status") == "FAILED" and prior_retries >= 3:
            logger.warning("%s 历史已失败 %d 次，按规则不再重试", model_id, prior_retries)
            report["models"].append({**prior, "resumed": True})
            continue
        if not pending and prior.get("smoke"):
            logger.info("%s 全部配置已 PASS，跳过", model_id)
            report["models"].append({**prior, "resumed": True})
            continue
        entry: dict[str, Any] = {"model_id": model_id, "backend": backend, "status": "OK",
                                 "smoke": prior.get("smoke"), "load_seconds": None,
                                 "retries": prior_retries + (1 if prior.get("status") == "FAILED" else 0),
                                 "configs": dict(prior_ok)}

        logger.info("=" * 70)
        logger.info("模型 %s（backend=%s）", model_id, backend)
        analyzer = create_analyzer(cfg.visual, str(cfg.path("model_dir")), cfg.mirrors, model_id)
        analyzer.set_output_language(output_language)

        # Cold start 加载
        bench.reset_peak_vram()
        try:
            t0 = time.perf_counter()
            analyzer.load(model_id)
            entry["load_seconds"] = round(time.perf_counter() - t0, 2)
            entry["load_peak_vram"] = bench.peak_vram_mb()
        except Exception as exc:
            entry["status"] = "FAILED"
            entry["error"] = f"load: {type(exc).__name__}: {exc}"[:500]
            entry["traceback"] = traceback.format_exc()[-2500:]
            logger.error("模型加载失败：%s", entry["error"])
            report["models"].append(entry)
            try:
                analyzer.unload()
            except Exception:
                pass
            continue

        # Smoke Test（历史已 PASS 就不重复烧时间）
        smoke = entry.get("smoke") if (entry.get("smoke") or {}).get("status") == "OK" else None
        if smoke is None:
            smoke = smoke_test(analyzer, info, cuts, cfg.visual, gt, output_language)
            entry["smoke"] = smoke
            logger.info("Smoke Test：%s（%.1fs，%d 事件）", smoke["status"], smoke["seconds"],
                        len(smoke.get("events") or []))
        else:
            logger.info("Smoke Test：复用历史 PASS 结果")

        if smoke["status"] != "OK":
            entry["status"] = "FAILED"
            entry["error"] = f"smoke: {smoke.get('error')}"
            analyzer.unload()
            report["models"].append(entry)
            continue
        if args.smoke_only:
            analyzer.unload()
            report["models"].append(entry)
            continue

        for key in pending:
            if key not in CONFIGS:
                continue

            if not budget_left():
                logger.warning("时间预算用尽，跳过 %s / 配置 %s", model_id, key)
                entry["configs"][key] = {"status": "SKIPPED", "reason": "time budget"}
                continue
            spec = CONFIGS[key]
            params = VisualParams(
                fps=float(spec["fps"]),
                max_frames=int(spec["max_frames"]),
                min_frames=min(int(cfg.visual["min_frames"]), int(spec["max_frames"])),
                max_pixels_tokens=int(cfg.visual["max_pixels_tokens"]),
                total_pixels_tokens=int(cfg.visual["total_pixels_tokens"]),
                max_new_tokens=int(spec["max_new_tokens"]),
            )
            batch_size = 1 if backend == "minicpm" else max(1, int(cfg.visual.get("batch_size", 1)))
            runs: list[dict[str, Any]] = []
            labels = ["cold"] + [f"run{i}" for i in range(1, args.repeats + 1)]
            for label in labels:
                if not budget_left():
                    logger.warning("时间预算用尽，%s/%s 只完成 %d 次", model_id, key, len(runs))
                    break
                logger.info("[%s] 配置 %s %s 开始（fps=%s frames=%s tokens=%s batch=%s）",
                            model_id.split("/")[-1], key, label, spec["fps"],
                            spec["max_frames"], spec["max_new_tokens"], batch_size)
                r = run_visual_once(analyzer, info, windows, params, cuts, batch_size,
                                    cfg.visual, gt, output_language)
                row = {
                    "label": label, "status": r.status, "error": r.error,
                    "timing": r.timing, "gpu": r.gpu, "peak_vram": r.peak_vram,
                    "oom_events": r.oom_events, "effective_params": r.effective_params,
                    "windows": r.windows, "frames": r.frames,
                    "generated_tokens": r.generated_tokens, "actual_batch": r.actual_batch,
                    "quality": r.quality, "events": r.events,
                }
                runs.append(row)
                raw_name = f"{model_id.replace('/', '__')}_{key}_{label}.json"
                (out_dir / "raw" / raw_name).write_text(
                    json.dumps({"model_id": model_id, "config": key, "label": label,
                                "params": params.to_dict(), "raw_outputs": r.raw_outputs,
                                "events": r.events, "timing": r.timing,
                                "quality": r.quality}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                logger.info("[%s] 配置 %s %s -> %s，视觉 %.1fs（generate %.1fs），%d 事件，"
                            "peak %.0fMB，OOM %d",
                            model_id.split("/")[-1], key, label, r.status,
                            r.timing.get("visual_seconds", 0.0), r.timing.get("generate_seconds", 0.0),
                            len(r.events), (r.peak_vram or {}).get("reserved_mb", 0.0), r.oom_events)

            formal = [r for r in runs if r["label"] != "cold" and r["status"] == "OK"]
            times = [r["timing"]["visual_seconds"] for r in formal]
            entry["configs"][key] = {
                "status": "OK" if formal else "FAILED",
                "params": params.to_dict(),
                "batch_requested": batch_size,
                "runs": runs,
                "visual_seconds_avg": round(statistics.fmean(times), 3) if times else None,
                "visual_seconds_min": round(min(times), 3) if times else None,
                "visual_seconds_max": round(max(times), 3) if times else None,
                "cold_visual_seconds": next(
                    (r["timing"]["visual_seconds"] for r in runs
                     if r["label"] == "cold" and r["status"] == "OK"), None),
                "stability": stability_between_runs(runs),
                "oom_total": sum(r["oom_events"] for r in runs),
                "peak_vram_reserved_mb": max(
                    [float((r["peak_vram"] or {}).get("reserved_mb") or 0.0) for r in runs] or [0.0]),
                "fits_in_physical_vram": max(
                    [float((r["peak_vram"] or {}).get("reserved_mb") or 0.0) for r in runs] or [0.0]
                ) <= total_vram * 0.98,
            }
            # 每个配置跑完就落盘，中途断电也不丢数据
            (out_dir / "model_benchmark.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        analyzer.unload()
        report["models"].append(entry)
        (out_dir / "model_benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    report["total_seconds"] = round(time.perf_counter() - started_all, 2)
    (out_dir / "model_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "model_benchmark.csv", report)
    write_reports(out_dir, report, speech)
    logger.info("Benchmark 完成，总耗时 %.1f 分钟，结果在 %s",
                report["total_seconds"] / 60.0, out_dir)
    return 0


# ----------------------------------------------------------------- 报告
def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speech_seconds = float(report.get("speech", {}).get("whisper_seconds") or 0.0)
    scene_seconds = float(report.get("scene_detect_seconds") or 0.0)
    for m in report.get("models", []):
        for key, c in (m.get("configs") or {}).items():
            if c.get("status") != "OK":
                rows.append({"model_id": m["model_id"], "backend": m["backend"], "config": key,
                             "status": c.get("status", "FAILED")})
                continue
            q_runs = [r["quality"] for r in c["runs"] if r["status"] == "OK" and r["label"] != "cold"]
            def qavg(name: str) -> float | None:
                vals = [float(q.get(name)) for q in q_runs if q.get(name) is not None]
                return round(statistics.fmean(vals), 4) if vals else None
            t_runs = [r["timing"] for r in c["runs"] if r["status"] == "OK" and r["label"] != "cold"]
            def tavg(name: str) -> float | None:
                vals = [float(t.get(name) or 0.0) for t in t_runs]
                return round(statistics.fmean(vals), 3) if vals else None
            rows.append({
                "model_id": m["model_id"],
                "backend": m["backend"],
                "config": key,
                "status": "OK",
                "fps": c["params"]["fps"],
                "max_frames": c["params"]["max_frames"],
                "max_new_tokens": c["params"]["max_new_tokens"],
                "resolution": "x".join(str(x) for x in (t_runs[0].get("resolution") or [])) if t_runs else "",
                "batch_requested": c.get("batch_requested"),
                "actual_batch": c["runs"][-1].get("actual_batch"),
                "model_load_seconds": m.get("load_seconds"),
                "cold_visual_seconds": c.get("cold_visual_seconds"),
                "visual_seconds_avg": c.get("visual_seconds_avg"),
                "visual_seconds_min": c.get("visual_seconds_min"),
                "visual_seconds_max": c.get("visual_seconds_max"),
                "frame_decode_seconds": tavg("frame_decode_seconds"),
                "processor_seconds": tavg("processor_seconds"),
                "generate_seconds": tavg("generate_seconds"),
                "text_decode_seconds": tavg("text_decode_seconds"),
                "whisper_seconds": round(speech_seconds, 3),
                "scene_detect_seconds": round(scene_seconds, 3),
                "end_to_end_estimate_seconds": round(
                    (c.get("visual_seconds_avg") or 0.0) + speech_seconds + scene_seconds, 3),
                "peak_vram_reserved_mb": c.get("peak_vram_reserved_mb"),
                "gpu_util_avg": statistics.fmean(
                    [float(r["gpu"].get("gpu_util_avg") or 0.0) for r in c["runs"]
                     if r["status"] == "OK"]) if c["runs"] else None,
                "generated_tokens": statistics.fmean(
                    [float(r.get("generated_tokens") or 0) for r in c["runs"] if r["status"] == "OK"])
                if c["runs"] else None,
                "event_count_avg": qavg("event_count"),
                "coverage": qavg("coverage"),
                "person_count_mode": (q_runs[0].get("person_count_mode") if q_runs else None),
                "person_count_correct": qavg("person_count_correct"),
                "person_consistency": qavg("person_consistency"),
                "hallucination_rate": qavg("hallucination_rate"),
                "hallucination_categories": ";".join(
                    f"{k}={v}" for k, v in sorted(
                        ((q_runs[0].get("hallucination_by_category") or {}) if q_runs else {}).items())),
                "ocr_reported": qavg("ocr_reported"),
                "ocr_hit": qavg("ocr_hit"),
                "ocr_precision": qavg("ocr_precision"),
                "ocr_recall": qavg("ocr_recall"),
                "rtf_visual": round((c.get("visual_seconds_avg") or 0.0)
                                    / max(float((report.get("video") or {}).get("duration") or 0.0), 1e-6), 4)
                if c.get("visual_seconds_avg") else None,
                "rtf_total": round(((c.get("visual_seconds_avg") or 0.0) + speech_seconds + scene_seconds)
                                   / max(float((report.get("video") or {}).get("duration") or 0.0), 1e-6), 4)
                if c.get("visual_seconds_avg") else None,
                "rtf_asr": round(speech_seconds
                                 / max(float((report.get("video") or {}).get("duration") or 0.0), 1e-6), 4),

                "frame_calibrated_ratio": qavg("frame_calibrated_ratio"),
                "timestamp_sane_ratio": qavg("timestamp_sane_ratio"),
                "language_match_ratio": qavg("language_match_ratio"),
                "duplicate_ratio": qavg("duplicate_ratio"),
                "desc_agreement": c["stability"].get("desc_agreement"),
                "event_count_identical": c["stability"].get("event_count_identical"),
                "oom_total": c.get("oom_total"),
                "fits_in_physical_vram": c.get("fits_in_physical_vram"),
            })
    return rows


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = _rows(report)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按任务书权重打分：稳定性30 速度30 时间定位20 画面理解10 显存10。"""
    ok = [r for r in rows if r.get("status") == "OK" and r.get("visual_seconds_avg")]
    if not ok:
        return []
    fastest = min(float(r["visual_seconds_avg"]) for r in ok)
    lowest_vram = min(float(r.get("peak_vram_reserved_mb") or 1e9) for r in ok)
    scored = []
    for r in ok:
        speed = fastest / float(r["visual_seconds_avg"])
        agree = float(r.get("desc_agreement") or 0.0)
        identical = 1.0 if r.get("event_count_identical") else 0.0
        no_oom = 1.0 if not r.get("oom_total") else 0.5
        fits = 1.0 if r.get("fits_in_physical_vram") else 0.0
        stability = (0.5 * agree + 0.3 * identical + 0.2 * no_oom) * (1.0 if fits else 0.5)
        timing = 0.5 * float(r.get("frame_calibrated_ratio") or 0.0) + \
            0.5 * float(r.get("timestamp_sane_ratio") or 0.0)
        understanding = (
            0.4 * (1.0 - float(r.get("hallucination_rate") or 1.0))
            + 0.3 * float(r.get("person_count_correct") or 0.0)
            + 0.2 * float(r.get("coverage") or 0.0)
            + 0.1 * (1.0 - float(r.get("duplicate_ratio") or 0.0))
        )
        vram = lowest_vram / float(r.get("peak_vram_reserved_mb") or 1e9)
        total = 0.30 * stability + 0.30 * speed + 0.20 * timing + 0.10 * understanding + 0.10 * vram
        scored.append({**r, "_stability": round(stability, 4), "_speed": round(speed, 4),
                       "_timing": round(timing, 4), "_understanding": round(understanding, 4),
                       "_vram": round(vram, 4), "_total": round(total, 4)})
    return sorted(scored, key=lambda r: -r["_total"])


def write_reports(out_dir: Path, report: dict[str, Any], speech: dict[str, Any]) -> None:
    rows = _rows(report)
    scored = _score(rows)
    video = report.get("video", {})
    env = report.get("environment", {})
    gpu = env.get("gpu", {})

    def fmt(v: Any, nd: int = 1) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)):
            return f"{v:.{nd}f}"
        return str(v)

    # --- speed_report.md ---
    lines = [
        "# 速度报告（全部为本机实测）", "",
        f"- 生成时间：{report.get('generated_at')}",
        f"- 视频：{video.get('name')} {video.get('width')}x{video.get('height')} "
        f"{video.get('duration')}s {video.get('fps')}fps",
        f"- GPU：{gpu.get('name')} {gpu.get('total_vram_mb')}MB，driver {gpu.get('driver')}，"
        f"torch {env.get('packages', {}).get('torch')}，transformers {env.get('packages', {}).get('transformers')}",
        f"- 语音（所有模型共用，只跑一次）：whisper {fmt(speech.get('whisper_seconds'))}s "
        f"（加载 {fmt(speech.get('whisper_load_seconds'))}s，{len(speech.get('segments') or [])} 段，"
        f"语言 {speech.get('language')}）",
        f"- 场景切点检测：{fmt(report.get('scene_detect_seconds'))}s，切点 {len(report.get('scene_cuts') or [])} 个",
        f"- 窗口：{len(report.get('windows') or [])} 个，固定 "
        f"{report.get('fixed', {}).get('window_seconds')}s / overlap "
        f"{report.get('fixed', {}).get('window_overlap_seconds')}s", "",
        "## 视觉阶段耗时（秒，正式运行平均）", "",
    ]
    for r in rows:
        if r.get("status") != "OK":
            lines.append(f"- {r['model_id']} / {r['config']}：{r.get('status')}")
            continue
        lines += [
            f"- **{r['model_id'].split('/')[-1]} / 配置{r['config']}** "
            f"(fps={r['fps']}, frames={r['max_frames']}, tokens={r['max_new_tokens']}, "
            f"res={r['resolution']}, batch={r['actual_batch']})",
            f"  - 视觉总计 平均 {fmt(r['visual_seconds_avg'])} / 最快 {fmt(r['visual_seconds_min'])} "
            f"/ 最慢 {fmt(r['visual_seconds_max'])} / cold {fmt(r['cold_visual_seconds'])}",
            f"  - 分项：解码 {fmt(r['frame_decode_seconds'])} | processor {fmt(r['processor_seconds'])} "
            f"| generate {fmt(r['generate_seconds'])} | 文本解码 {fmt(r['text_decode_seconds'])}",
            f"  - 模型加载 {fmt(r['model_load_seconds'])}｜生成 token 均值 {fmt(r['generated_tokens'], 0)}"
            f"｜GPU 利用率均值 {fmt(r['gpu_util_avg'])}%｜峰值显存 {fmt(r['peak_vram_reserved_mb'], 0)}MB"
            f"｜OOM {r['oom_total']}",
            f"  - 端到端估算（视觉+语音+场景检测）{fmt(r['end_to_end_estimate_seconds'])}s",
        ]
    (out_dir / "speed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- quality_report.md ---
    q = [
        "# 质量报告", "",
        "评分依据：`benchmark/ground_truth.json`（人工看帧标注）+ 可客观计算的指标。",
        "需要主观判断的项目（画面美感、语义细腻度）不打分，只把原始输出放在 benchmark/raw/ 里。", "",
    ]
    for r in rows:
        if r.get("status") != "OK":
            continue
        q += [
            f"## {r['model_id'].split('/')[-1]} / 配置{r['config']}",
            f"- 事件数均值 {fmt(r['event_count_avg'])}，覆盖率 {fmt(r['coverage'], 3)}，"
            f"重复率 {fmt(r['duplicate_ratio'], 3)}",
            f"- 人数众数 {r['person_count_mode']}（真值 2）→ 正确 {fmt(r['person_count_correct'], 2)}，"
            f"人数一致性 {fmt(r['person_consistency'], 3)}",
            f"- 幻觉率 {fmt(r['hallucination_rate'], 3)}"
            f"（与真值矛盾的事件占比；分类命中：{r.get('hallucination_categories') or '无'}）",
            f"- OCR：报告了文字 {fmt(r['ocr_reported'], 2)}，命中 Pascall/Clinkers {fmt(r['ocr_hit'], 2)}，"
            f"精确率 {fmt(r.get('ocr_precision'), 3)}（1.0 = 没有凭空造字），召回 {fmt(r.get('ocr_recall'), 3)}",
            f"- RTF：视觉 {fmt(r.get('rtf_visual'), 3)}｜ASR {fmt(r.get('rtf_asr'), 3)}｜整体 {fmt(r.get('rtf_total'), 3)}",

            f"- 时间：帧校准比例 {fmt(r['frame_calibrated_ratio'], 3)}，时间戳合法比例 {fmt(r['timestamp_sane_ratio'], 3)}",
            f"- 语言匹配（应为 {report.get('output_language')}）{fmt(r['language_match_ratio'], 3)}",
            f"- A/B 稳定性：描述一致度 {fmt(r['desc_agreement'], 3)}，事件数是否完全一致 {fmt(r['event_count_identical'])}",
            "",
        ]
    (out_dir / "quality_report.md").write_text("\n".join(q) + "\n", encoding="utf-8")

    # --- failure_report.md ---
    f = ["# 失败报告", ""]
    any_fail = False
    for m in report.get("models", []):
        if m.get("status") != "OK":
            any_fail = True
            f += [f"## {m['model_id']}（backend={m['backend']}）→ FAILED",
                  f"- 原因：{m.get('error')}", ""]
            if m.get("smoke") and m["smoke"].get("status") != "OK":
                f.append(f"- Smoke Test 输出片段：`{(m['smoke'].get('raw_output') or '')[:300]}`")
            if m.get("traceback"):
                f += ["```", m["traceback"][-1500:], "```", ""]
        for key, c in (m.get("configs") or {}).items():
            if c.get("status") == "OK":
                bad = [r for r in c["runs"] if r["status"] != "OK"]
                if bad:
                    any_fail = True
                    f += [f"## {m['model_id']} / 配置{key}：{len(bad)} 次运行失败"]
                    f += [f"- {b['label']}: {b['error']}" for b in bad] + [""]
            else:
                any_fail = True
                f += [f"## {m['model_id']} / 配置{key} → {c.get('status')}",
                      f"- {c.get('reason') or c.get('error')}", ""]
    if not any_fail:
        f.append("没有失败项。")
    (out_dir / "failure_report.md").write_text("\n".join(f) + "\n", encoding="utf-8")

    # --- final_recommendation.md ---
    rec = ["# 最终推荐（按实测数据，不按论文/参数量）", ""]
    if not scored:
        rec.append("没有任何配置成功完成，无法给出推荐。见 failure_report.md。")
    else:
        best = scored[0]
        fastest = min(scored, key=lambda r: float(r["visual_seconds_avg"]))
        lowest = min(scored, key=lambda r: float(r.get("peak_vram_reserved_mb") or 1e9))
        most_stable = max(scored, key=lambda r: r["_stability"])
        best_timing = max(scored, key=lambda r: r["_timing"])
        best_quality = max(scored, key=lambda r: r["_understanding"])

        def tag(r: dict[str, Any]) -> str:
            return f"{r['model_id']} / 配置{r['config']}"

        rec += [
            "权重：稳定性 30% / 速度 30% / 时间定位 20% / 画面理解 10% / 显存 10%", "",
            f"- 最快模型：**{tag(fastest)}** — 视觉 {fmt(fastest['visual_seconds_avg'])}s",
            f"- 最稳模型：**{tag(most_stable)}** — 稳定性得分 {fmt(most_stable['_stability'], 3)}",
            f"- 时间定位最好：**{tag(best_timing)}** — 得分 {fmt(best_timing['_timing'], 3)}",
            f"- 画面理解最好：**{tag(best_quality)}** — 得分 {fmt(best_quality['_understanding'], 3)}",
            f"- 显存最低：**{tag(lowest)}** — 峰值 {fmt(lowest['peak_vram_reserved_mb'], 0)}MB",
            f"- RTX 3060 12GB 最佳综合 / 建议默认：**{tag(best)}** — 总分 {fmt(best['_total'], 4)}",
            "", "## 排名（总分）", "",
        ]
        for i, r in enumerate(scored, 1):
            rec.append(
                f"{i}. {tag(r)} 总分 {fmt(r['_total'], 4)}（稳定 {fmt(r['_stability'], 3)}"
                f"｜速度 {fmt(r['_speed'], 3)}｜时间 {fmt(r['_timing'], 3)}"
                f"｜理解 {fmt(r['_understanding'], 3)}｜显存 {fmt(r['_vram'], 3)}）"
                f" 视觉 {fmt(r['visual_seconds_avg'])}s，峰值 {fmt(r['peak_vram_reserved_mb'], 0)}MB")
        rec += ["", "## 建议写入 config.json 的默认值", "",
                "```json",
                json.dumps({"visual": {
                    "model_id": best["model_id"],
                    "backend": best["backend"],
                    "fps": best["fps"],
                    "max_frames": best["max_frames"],
                    "max_new_tokens": best["max_new_tokens"],
                }}, ensure_ascii=False, indent=2),
                "```"]
    (out_dir / "final_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
