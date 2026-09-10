"""素材入库编排：注册目标歌 → 批量对齐 → 固定位置切片 → 登记素材资产。

这一层是**编排**，本身不做算法：对齐找 `audio_align`、音乐结构找 `music_structure`、
区间算法找 `material_slice`、落库找 `material_repository`。
把编排单独拆出来的原因是它要管三件跨层的事：并发、缓存、以及"哪一步失败了继续跑哪一步"。

三条性能约定（技术指导第二十四节）：
1. 目标歌指纹与音乐分析**只算一次**，之后所有源视频复用同一份内存 PCM
2. 对齐并发默认 4~6 个 worker，**不是** 100 个 —— 每个 worker 都在做几百万点的 FFT，
   开太多只会互相抢内存带宽，还会把 12GB 的机器吃穷
3. 对齐缓存键 = 源指纹 + 目标指纹 + 算法版本 + 配置指纹，命中就跳过重算
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..db.db import Database
from ..db import repo as db_repo
from ..logging_setup import get_logger
from . import ALIGNMENT_ALGORITHM_VERSION, MATERIAL_GENERATION_VERSION
from . import audio_align, material_repository as repo, material_slice, music_structure
from . import alignment_validation as validate
from .audio_fingerprint import (
    AudioReadError,
    alignment_cache_key,
    audio_fingerprint,
    config_hash,
    extract_analysis_audio,
    file_fingerprint,
    media_audio_duration,
)
from .dsp import DEFAULT_SR
from .types import DanceAlignment, SlicePlan

logger = get_logger("dance.ingest")

LogFn = Callable[[str], None]
#: 对齐并发上限。技术指导明确"4~6 workers，不要 100"
DEFAULT_WORKERS = 4
MAX_WORKERS = 6

@dataclass
class TargetSong:
    """已注册并分析好的目标歌。`pcm` 只在内存里传，不落库。"""

    song_id: int
    path: Path
    fingerprint: str
    duration: float
    bpm: float
    sample_rate: int = DEFAULT_SR
    analysis: dict[str, Any] = field(default_factory=dict)
    pcm: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"song_id": self.song_id, "path": str(self.path),
                "fingerprint": self.fingerprint, "duration": round(self.duration, 3),
                "bpm": round(self.bpm, 3), "sample_rate": self.sample_rate}


@dataclass
class AlignOutcome:
    """一个源视频的对齐结果。`cached=True` 表示直接命中缓存，没重算。"""

    source_path: Path
    video_id: int = 0
    alignment_id: int = 0
    alignment: DanceAlignment | None = None
    cached: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.alignment is not None and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {"source": str(self.source_path), "video_id": self.video_id,
                "alignment_id": self.alignment_id, "cached": self.cached,
                "error": self.error,
                "offset": None if self.alignment is None else self.alignment.offset,
                "confidence": None if self.alignment is None else self.alignment.confidence,
                "status": None if self.alignment is None else self.alignment.status}


def align_config(*, sample_rate: int, window_seconds: float, window_count: int) -> dict[str, Any]:
    """进缓存键的那份配置。只放**会改变对齐结果**的参数。

    刻意不放并发数、日志开关这类东西：把它们放进来会让"改了个 worker 数"
    也触发全部重算，那不是缓存，那是摆设。
    """
    return {"sample_rate": int(sample_rate), "window_seconds": float(window_seconds),
            "window_count": int(window_count),
            "offset_tolerance": validate.OFFSET_TOLERANCE,
            "method_tolerance": validate.METHOD_TOLERANCE,
            "min_peak_quality": validate.MIN_PEAK_QUALITY,
            "min_cluster_share": validate.MIN_CLUSTER_SHARE}


def register_song(db: Database, path: str | Path, *, sample_rate: int = DEFAULT_SR,
                  force: bool = False, on_log: LogFn | None = None) -> TargetSong:
    """注册并分析一首目标歌。已分析过且版本一致就直接吃缓存（除非 `force`）。

    音乐分析（节拍/段落/特征/节奏带）几秒钟，但每次开界面都重算就没法用了，
    所以结果整份缓存进 `dance_target_songs`，靠 `analysis_version` 判失效。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    target = Path(path)
    if not target.is_file():
        raise AudioReadError(f"目标歌文件不存在：{target}")
    fingerprint = file_fingerprint(target)
    song_id = repo.upsert_song(db, fingerprint=fingerprint, file_path=str(target.resolve()),
                               file_name=target.name, title=target.stem)
    row = repo.get_song(db, song_id)
    fresh = (row is not None and row["analysis_version"] == music_structure.ANALYSIS_VERSION
             and row["beats_json"] and row["sections_json"])
    pcm = extract_analysis_audio(target, sample_rate=sample_rate)
    if fresh and not force:
        log(f"[目标歌] {target.name} 已分析过（{row['analysis_version']}），复用缓存")
        return TargetSong(song_id=song_id, path=target, fingerprint=fingerprint,
                          duration=float(row["duration"] or 0.0),
                          bpm=float(row["bpm"] or 0.0), sample_rate=sample_rate,
                          analysis={"cached": True}, pcm=pcm)

    analysis = music_structure.analyze_song(pcm, sample_rate)
    repo.save_song_analysis(db, song_id, analysis)
    log(f"[目标歌] {target.name}｜{analysis['duration']:.2f}s｜BPM {analysis['bpm']:.2f}"
        f"｜{analysis['beat_count']} 拍｜{len(analysis['sections'])} 段落")
    return TargetSong(song_id=song_id, path=target, fingerprint=fingerprint,
                      duration=float(analysis["duration"]), bpm=float(analysis["bpm"]),
                      sample_rate=sample_rate, analysis=analysis, pcm=pcm)

def align_source(db: Database, song: TargetSong, source: str | Path, *,
                 window_seconds: float = validate.WINDOW_SECONDS,
                 window_count: int = validate.WINDOW_COUNT,
                 force: bool = False, on_log: LogFn | None = None) -> AlignOutcome:
    """对齐一个源舞蹈视频到目标歌，结果落库。命中缓存就不重算。

    源视频顺手登记进 `videos` 表（复用主项目的视频身份体系，不另建一套）。
    读不出音轨、解码失败都记进 `AlignOutcome.error`，**不抛** ——
    批量对齐几十个文件时一条坏文件不该中断整批；但也绝不返回一个假的 offset=0。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    path = Path(source)
    outcome = AlignOutcome(source_path=path)
    if not path.is_file():
        outcome.error = f"文件不存在：{path}"
        return outcome
    try:
        outcome.video_id = db_repo.upsert_video(db, path)
    except Exception as exc:  # noqa: BLE001 - 登记失败也要继续算，只是没法落库
        outcome.error = f"登记视频失败：{type(exc).__name__}: {exc}"
        return outcome

    cfg_hash = config_hash(align_config(sample_rate=song.sample_rate,
                                        window_seconds=window_seconds,
                                        window_count=window_count))
    try:
        source_fp = file_fingerprint(path)
    except OSError as exc:
        outcome.error = f"算源指纹失败：{exc}"
        return outcome
    key = alignment_cache_key(source_fp, song.fingerprint,
                              ALIGNMENT_ALGORITHM_VERSION, cfg_hash)

    if not force:
        cached = repo.alignment_by_key(db, key)
        if cached is not None:
            outcome.alignment_id = int(cached["id"])
            outcome.alignment = _alignment_from_row(cached)
            outcome.cached = True
            log(f"[对齐] {path.name} 命中缓存：offset {outcome.alignment.offset:.3f}s"
                f"（{cached['status']}）")
            return outcome

    try:
        result = audio_align.find_alignment(
            str(path), str(song.path), sample_rate=song.sample_rate,
            window_seconds=window_seconds, window_count=window_count,
            target_pcm=song.pcm, algorithm_version=ALIGNMENT_ALGORITHM_VERSION)
    except AudioReadError as exc:
        outcome.error = str(exc)
        log(f"[对齐] {path.name} 失败：{exc}")
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome.error = f"{type(exc).__name__}: {exc}"
        log(f"[对齐] {path.name} 异常：{outcome.error}")
        return outcome

    outcome.alignment = result
    outcome.alignment_id = repo.save_alignment(
        db, source_video_id=outcome.video_id, target_song_id=song.song_id,
        cache_key=key, alignment=result, config_hash=cfg_hash)
    log(f"[对齐] {path.name}｜offset {result.offset:.3f}s｜置信度 {result.confidence:.3f}"
        f"｜{result.status}｜{result.window_count} 窗口")
    return outcome


def _alignment_from_row(row: Any) -> DanceAlignment:
    """把库里的一行还原成 `DanceAlignment`。窗口明细从 detail_json 里捞。"""
    detail: dict[str, Any] = {}
    if row["detail_json"]:
        try:
            detail = json.loads(row["detail_json"]) or {}
        except (TypeError, ValueError):
            detail = {}
    windows = tuple(
        validate.WindowResult(index=int(w.get("index", i + 1)),
                              window_start=float(w.get("window_start", 0.0)),
                              window_seconds=float(w.get("window_seconds", 0.0)),
                              offset=float(w.get("offset", 0.0)),
                              confidence=float(w.get("confidence", 0.0)),
                              method=str(w.get("method", "waveform")))
        for i, w in enumerate(detail.get("windows", []) or []) if isinstance(w, dict))
    return DanceAlignment(
        offset=float(row["offset_seconds"]), confidence=float(row["confidence"] or 0.0),
        method=str(row["method"] or "hybrid"),
        waveform_offset=row["waveform_offset"], waveform_confidence=row["waveform_confidence"],
        chroma_offset=row["chroma_offset"], chroma_confidence=row["chroma_confidence"],
        window_count=int(row["window_count"] or 0),
        max_deviation=float(row["max_deviation"] or 0.0),
        agreement=float(row["agreement"] or 0.0), status=str(row["status"] or "ok"),
        algorithm_version=str(row["algorithm_version"] or ""),
        source_duration=float(row["source_duration"] or 0.0),
        target_duration=float(row["target_duration"] or 0.0),
        windows=windows, notes=tuple(detail.get("notes", []) or []),
        original_offset=row["original_offset"], original_confidence=row["original_confidence"],
        manual_offset=row["manual_offset"], manual_reason=str(row["manual_reason"] or ""),
        manual_at=str(row["manual_at"] or ""), manual_operator=str(row["manual_operator"] or ""))

def align_batch(db: Database, song: TargetSong, sources: Sequence[str | Path], *,
                workers: int = DEFAULT_WORKERS,
                window_seconds: float = validate.WINDOW_SECONDS,
                window_count: int = validate.WINDOW_COUNT,
                force: bool = False, persist: bool = True,
                on_log: LogFn | None = None,
                on_progress: Callable[[int, int, str], None] | None = None,
                ) -> list[AlignOutcome]:
    """批量对齐。默认 4 个 worker，上限 6（技术指导第二十四节）。

    为什么线程够用而不必上进程：耗时全在 numpy FFT 和 PyAV 解码上，两者都会释放 GIL；
    而进程池要把几 MB 的目标歌 PCM 序列化给每个子进程，反而更慢。

    落库放在**主线程**（worker 只算不写）：`Database` 是每线程一条连接 + 整库写串行化，
    让 4 个 worker 同时抢写锁没有任何好处，还让"哪条先写"变得不确定。

    `persist=False` 是**只算不写**模式，给界面上的「音频对齐 / 卡点测试」用：
    那边只是拿一个视频加一首歌试试对不对，不该因此就往 `videos` 里塞一行、
    往 `dance_audio_alignments` 里留一条记录。缓存照旧读（只读不写，命中就省一次全曲 FFT），
    确认没问题之后由 `persist_alignment()` 正式落库。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    report = on_progress or (lambda done, total, stage: None)
    paths = [Path(p) for p in sources]
    if not paths:
        return []
    count = max(1, min(int(workers), MAX_WORKERS))
    cfg_hash = config_hash(align_config(sample_rate=song.sample_rate,
                                        window_seconds=window_seconds,
                                        window_count=window_count))
    log(f"[批量对齐] {len(paths)} 个源视频，{count} 个 worker，目标歌 {song.path.name}"
        + ("" if persist else "（只算不写，不进库）"))

    # 第一步（主线程）：登记视频 + 查缓存，命中的直接出结果，不进线程池
    pending: list[tuple[Path, int, str]] = []
    results: dict[str, AlignOutcome] = {}
    for path in paths:
        outcome = AlignOutcome(source_path=path)
        if not path.is_file():
            outcome.error = f"文件不存在：{path}"
            results[str(path)] = outcome
            continue
        try:
            if persist:
                outcome.video_id = db_repo.upsert_video(db, path)
            source_fp = file_fingerprint(path)
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"{type(exc).__name__}: {exc}"
            results[str(path)] = outcome
            continue

        key = alignment_cache_key(source_fp, song.fingerprint,
                                  ALIGNMENT_ALGORITHM_VERSION, cfg_hash)
        cached = None if force else repo.alignment_by_key(db, key)
        if cached is not None:
            outcome.alignment_id = int(cached["id"])
            outcome.alignment = _alignment_from_row(cached)
            outcome.cached = True
            results[str(path)] = outcome
            continue
        pending.append((path, outcome.video_id, key))
        results[str(path)] = outcome

    done = len(paths) - len(pending)
    report(done, len(paths), "复用缓存")
    if pending:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="dance-align") as pool:
            futures = {
                pool.submit(_align_worker, song, path, window_seconds, window_count):
                    (path, video_id, key)
                for path, video_id, key in pending}
            for future in as_completed(futures):
                path, video_id, key = futures[future]
                outcome = results[str(path)]
                try:
                    outcome.alignment = future.result()
                except AudioReadError as exc:
                    outcome.error = str(exc)
                except Exception as exc:  # noqa: BLE001
                    outcome.error = f"{type(exc).__name__}: {exc}"
                if outcome.alignment is not None:
                    if persist:
                        outcome.alignment_id = repo.save_alignment(
                            db, source_video_id=video_id, target_song_id=song.song_id,
                            cache_key=key, alignment=outcome.alignment, config_hash=cfg_hash)
                    log(f"[对齐] {path.name}｜offset {outcome.alignment.offset:.3f}s"
                        f"｜{outcome.alignment.confidence:.3f}｜{outcome.alignment.status}")
                else:
                    log(f"[对齐] {path.name} 失败：{outcome.error}")
                done += 1
                report(done, len(paths), "对齐")

    ordered = [results[str(path)] for path in paths]
    ok = sum(1 for r in ordered if r.ok)
    cached_count = sum(1 for r in ordered if r.cached)
    log(f"[批量对齐] 完成 {ok}/{len(paths)}（其中 {cached_count} 条命中缓存）")
    return ordered


def _align_worker(song: TargetSong, path: Path, window_seconds: float,
                  window_count: int) -> DanceAlignment:
    """线程里跑的那一段：只算，不碰数据库。"""
    return audio_align.find_alignment(
        str(path), str(song.path), sample_rate=song.sample_rate,
        window_seconds=window_seconds, window_count=window_count,
        target_pcm=song.pcm, algorithm_version=ALIGNMENT_ALGORITHM_VERSION)


def persist_alignment(db: Database, song: TargetSong, source: str | Path,
                      alignment: DanceAlignment, *,
                      window_seconds: float = validate.WINDOW_SECONDS,
                      window_count: int = validate.WINDOW_COUNT,
                      on_log: LogFn | None = None) -> AlignOutcome:
    """把一份**已经算好**的对齐结果正式落库（登记源视频 + 写 dance_audio_alignments）。

    给「音频对齐 / 卡点测试」用：那边先 `align_batch(persist=False)` 只算不写，
    人工确认过了才点「保存对齐结果」走到这里。缓存键在这里现算，和
    `align_batch` 用的是同一个 `align_config` + `alignment_cache_key`，
    所以保存之后再跑正式流程会**命中缓存**，不会为同一份输入重算一次全曲 FFT。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    path = Path(source)
    outcome = AlignOutcome(source_path=path, alignment=alignment)
    if not path.is_file():
        outcome.error = f"文件不存在：{path}"
        return outcome
    try:
        outcome.video_id = db_repo.upsert_video(db, path)
        source_fp = file_fingerprint(path)
    except Exception as exc:  # noqa: BLE001 - 登记失败就明确报错，不要留一条半截记录
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome
    cfg_hash = config_hash(align_config(sample_rate=song.sample_rate,
                                        window_seconds=window_seconds,
                                        window_count=window_count))
    key = alignment_cache_key(source_fp, song.fingerprint,
                              ALIGNMENT_ALGORITHM_VERSION, cfg_hash)
    outcome.alignment_id = repo.save_alignment(
        db, source_video_id=outcome.video_id, target_song_id=song.song_id,
        cache_key=key, alignment=alignment, config_hash=cfg_hash)
    log(f"[保存] {path.name} 的对齐已入库（#{outcome.alignment_id}）"
        f"｜offset {alignment.offset:+.3f}s｜{alignment.status}")
    return outcome


@dataclass
class SliceOutcome:
    """一个源视频切片入库的结果。"""

    source_path: Path
    video_id: int = 0
    plan: SlicePlan | None = None
    material_ids: list[int] = field(default_factory=list)
    rendered: int = 0
    skipped: int = 0
    retired: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.material_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"source": str(self.source_path), "video_id": self.video_id,
                "materials": len(self.material_ids), "rendered": self.rendered,
                "skipped": self.skipped, "retired": self.retired, "error": self.error}


def slice_and_register(db: Database, song: TargetSong, outcome: AlignOutcome, *,
                       slice_duration: float, material_dir: str | Path,
                       canvas=None, backend=None,
                       generation_version: str = MATERIAL_GENERATION_VERSION,
                       person: str = "", source_group: str = "",
                       min_confidence: float = 0.0,
                       on_log: LogFn | None = None,
                       on_progress: Callable[[int, int, str], None] | None = None,
                       ) -> SliceOutcome:
    """按固定音乐位置切片、渲染成文件、登记成素材资产。

    `min_confidence > 0` 时置信度不够的对齐直接不切 —— 一个不可信的 offset
    切出来的素材全是错位的，进了库还会被推荐出去，不如不要。

    只登记**真的落到盘上、封装完整**的素材（`render_plan` 已经把不完整的挡掉了）：
    库里记着一条文件不存在的素材，混剪时才发现，那时候已经晚了。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    result = SliceOutcome(source_path=outcome.source_path, video_id=outcome.video_id)
    if not outcome.ok or outcome.alignment is None:
        result.error = outcome.error or "没有可用的对齐结果"
        return result
    alignment = outcome.alignment
    if min_confidence > 0 and alignment.confidence < float(min_confidence):
        result.error = (f"置信度 {alignment.confidence:.3f} 低于要求的 "
                       f"{float(min_confidence):.3f}，不切片")
        log(f"[切片] {outcome.source_path.name} 跳过：{result.error}")
        return result

    source_duration = media_audio_duration(outcome.source_path)
    if source_duration <= 0:
        source_duration = float(alignment.source_duration or 0.0)
    plan = material_slice.plan_slices(
        alignment, song_duration=song.duration, source_duration=source_duration,
        slice_duration=slice_duration, source_video_id=outcome.video_id,
        target_song_id=song.song_id, generation_version=generation_version)
    result.plan = plan
    result.skipped = len(plan.skipped)
    if not plan.specs:
        reasons = "；".join(why for _i, why in plan.skipped[:2])
        result.error = f"一个位置都切不出来：{reasons}"
        log(f"[切片] {outcome.source_path.name} {result.error}")
        return result

    rendered = material_slice.render_plan(
        outcome.source_path, plan, Path(material_dir), canvas=canvas, backend=backend,
        on_log=log, on_progress=on_progress)
    result.rendered = len(rendered)

    sections = repo.song_sections(db, song.song_id)
    for spec, target in rendered:
        material_id = repo.upsert_material(
            db, source_video_id=outcome.video_id, alignment_id=outcome.alignment_id,
            target_song_id=song.song_id, segment_index=spec.segment_index,
            target_start=spec.target_start, target_end=spec.target_end,
            source_start=spec.source_start, source_end=spec.source_end,
            duration=spec.duration, generation_version=generation_version,
            file_path=str(target), file_hash=_safe_fingerprint(target),
            alignment_confidence=alignment.confidence,
            person=person or outcome.source_path.stem,
            source_group=source_group or outcome.source_path.parent.name,
            quality=_quality_of(alignment, sections, spec.target_start))
        result.material_ids.append(material_id)
    # 旧切片版本的素材标 regenerated（留着不删，历史成品还引用着）
    result.retired = repo.mark_regenerated(db, song.song_id, outcome.video_id,
                                          generation_version)
    log(f"[切片] {outcome.source_path.name}：登记 {len(result.material_ids)} 条素材"
        f"，跳过 {result.skipped} 个位置"
        + (f"，旧版 {result.retired} 条标为 regenerated" if result.retired else ""))
    return result


def _safe_fingerprint(path: Path) -> str:
    try:
        return file_fingerprint(path)
    except OSError:
        return ""


def _quality_of(alignment: DanceAlignment, sections: Sequence[dict[str, Any]],
                moment: float) -> float:
    """素材的先验质量 0~1：对齐置信度为主，落在高能量段落略加成。

    刻意做得很轻（段落只占两成）：段落类型是**辅助特征**，技术指导第六节第 5 条
    明确禁止让它主导素材选择。真正的选择权在评分层和组合搜索。
    """
    base = float(alignment.confidence)
    energy = 0.0
    for section in sections:
        try:
            if float(section["start"]) <= moment < float(section["end"]):
                energy = float(section.get("energy") or 0.0)
                break
        except (KeyError, TypeError, ValueError):
            continue
    return round(min(1.0, 0.8 * base + 0.2 * energy), 4)


__all__ = [
    "DEFAULT_WORKERS", "MAX_WORKERS",
    "TargetSong", "AlignOutcome", "SliceOutcome",
    "align_config", "register_song", "align_source", "align_batch", "persist_alignment",
    "slice_and_register",
]




