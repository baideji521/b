"""把已有的缓存/产物导进数据库。**只读文件，不删不改任何东西。**

导入源（按可靠程度排）：
1. `cache/videos/<slug>/` —— state.json 记着这份缓存属于哪个视频，
   probe.json 有时长分辨率，speech.json 有段和逐词，visual.json 有视觉事件；
2. `output/<视频名>/` —— 缓存被清过但导出还在时，从 speech_events.json / visual_events.json 补；
3. 视频所在目录 / AI_输入目录 —— `<视频名>.txt`、`<视频名>_merged.txt` 登记成 merged_txt；
4. AI_输出目录 —— `<视频名>_脚本.json` 变成 ai_task(completed) + ai_result + clips，
   `<视频名>_高光时刻.mp4` 登记成 final_video 并把片段标成 rendered。

原片已经不在盘上的（缓存还在、视频删了），照样登记，`exists_on_disk = 0`，
这样 AI 面板能显示"视频没了"，而不是把这份分析结果当不存在。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import cache as cache_mod
from ..logging_setup import get_logger
from ..video_io import is_complete_video
from . import repo
from .db import Database, open_db

logger = get_logger(__name__)

VIDEO_SUFFIXES = cache_mod.VIDEO_SUFFIXES

# 渲染中的临时后缀，和 highlight/clip.py 的 PART_SUFFIX 是同一个值。
# 这里另写一份而不是 import，是不想让 db 包为了一个字符串拖上整套渲染依赖（av/PIL/cv2）。
PART_SUFFIX = ".part"


def _ok_to_register(kind: str, path: Path) -> bool:
    """这个文件能不能登记成 artifacts。除 final_video 外还是老规矩：在盘上且非空。

    final_video 多一条：必须是一份封装完整的视频。渲染中途崩掉留下的残片、历史遗留
    的坏 mp4 都是"非空文件"，登记上去 `_auto_done_file()` 就会把任务当干完了，
    于是永远不会重剪——所以成品这一种必须严一点（见 video_io.is_complete_video）。
    """
    if not (path.is_file() and path.stat().st_size > 0):
        return False
    if kind == "final_video" and not is_complete_video(path):
        logger.warning("成品封装不完整，不登记 final_video：%s", path)
        return False
    return True


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # 半截文件、编码坏了都不该让导入整体失败
        logger.warning("读不了 %s：%s", path, exc)
        return None


def _bridge_dir(cfg: Any, key: str) -> Path | None:
    raw = str(cfg.bridge.get(key) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    path = path if path.is_absolute() else Path(cfg.root) / path
    return path if path.is_dir() else None


def _signature_from_cache(speech: Any, visual: Any) -> dict[str, Any]:
    """旧缓存里没存配置哈希，只能把当时的模型名捞出来，配置哈希留 'imported'。

    这样一来：导入的记录不会被当成"配置一致的缓存"直接命中（哈希对不上），
    但历史结果、逐词时间戳、AI 结果全都在库里查得到，也不会丢。
    """
    asr_model = ""
    if isinstance(speech, dict):
        model = speech.get("model")
        if isinstance(model, dict):
            asr_model = str(model.get("size") or "")
    vision_model = ""
    if isinstance(visual, dict):
        meta = visual.get("meta")
        if isinstance(meta, dict):
            vision_model = str(meta.get("model_id") or meta.get("model") or "")
    return {
        "vision_model": vision_model,
        "vision_config": None,
        "vision_config_hash": "imported",
        "asr_model": asr_model,
        "asr_config": None,
        "asr_config_hash": "imported",
    }


def _register_video(db: Database, path: Path, slug: str | None,
                    probe: Any, in_library: bool | None) -> int:
    info: dict[str, Any] = {}
    if isinstance(probe, dict):
        video = probe.get("video") if isinstance(probe.get("video"), dict) else probe
        if isinstance(video, dict):
            info = {k: video.get(k) for k in ("duration", "width", "height", "fps")}
    if path.is_file():
        return repo.upsert_video(db, path, info=info, cache_slug=slug, in_library=in_library)
    return repo.upsert_missing_video(db, path, cache_slug=slug, info=info)


def _import_analysis(db: Database, video_id: int, speech: Any, visual: Any,
                     out_dir: Path | None) -> tuple[int, int, int] | None:
    """把一套分析结果写成一条 analysis_run。返回（视觉事件数，语音段数，词数）。

    同一个视频只导一次：已经有 source='import' 的记录就跳过，
    否则每跑一次 `db --import` 都会多出一条一样的历史。
    """
    if repo.analysis_by_source(db, video_id, "import") is not None:
        return None
    speech_segments = []
    if isinstance(speech, dict):
        speech_segments = [s for s in (speech.get("segments") or []) if isinstance(s, dict)]
    visual_events = []
    if isinstance(visual, dict):
        visual_events = [e for e in (visual.get("events") or []) if isinstance(e, dict)]
    if not speech_segments and not visual_events:
        return None
    sig = _signature_from_cache(speech, visual)
    analysis_id = repo.create_analysis(db, video_id, sig, source="import")
    events = repo.save_visual_events(db, analysis_id, visual_events)
    segments, words = repo.save_speech_segments(db, analysis_id, speech_segments)
    repo.finish_analysis(db, analysis_id, scene_count=events, speech_count=segments,
                         output_dir=out_dir)
    return events, segments, words


def _import_artifacts(db: Database, cfg: Any, video_id: int, video: Path,
                      slug: str | None, out_dir: Path | None,
                      ai_out: Path | None) -> int:
    """登记这个视频相关的实际文件。文件不在的就不登记，免得库里塞一堆幽灵路径。"""
    count = 0
    stem = video.stem
    candidates: list[tuple[str, Path]] = []
    if video.is_file():
        candidates.append(("source_video", video))
    for name in (f"{stem}.txt", f"{stem}_merged.txt"):
        candidates.append(("merged_txt", video.parent / name))
    if out_dir is not None:
        candidates.append(("words_srt", out_dir / "timeline.srt"))
        candidates.append(("translated_txt", out_dir / "timeline_译文.txt"))
    if slug:
        wav = cache_mod.videos_root(cache_mod.cache_dir(cfg)) / slug / cache_mod.PREVIEW_AUDIO
        candidates.append(("preview_audio", wav))
    if ai_out is not None:
        candidates.append(("ai_script", ai_out / f"{stem}_脚本.json"))
        candidates.append(("final_video", ai_out / f"{stem}_高光时刻.mp4"))
    for kind, path in candidates:
        if _ok_to_register(kind, path):
            repo.register_artifact(db, video_id, kind, path)
            count += 1
    return count


def _legacy_script_json(video: Path, ai_out: Path | None) -> Path | None:
    """找这个视频的历史高光 JSON：视频旁边和 AI_输出目录都看，先到先用。

    脚本剪辑那一串的 JSON 往往就丢在视频旁边，AI 回传的落在 AI_输出目录。
    这两处是**兼容导入源**，不是判断依据——认出来就进库，之后一切查库。
    """
    for folder in (video.parent, ai_out):
        if folder is None:
            continue
        for name in (f"{video.stem}_脚本.json", f"{video.stem}.json"):
            candidate = folder / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def _import_ai(db: Database, cfg: Any, video_id: int, video: Path,
              ai_out: Path | None) -> tuple[int, int]:
    """历史高光 JSON -> ai_task(completed) + ai_result + clips。返回（结果数，片段数）。

    库里已经有这个视频的 AI 结果就不再导——库是权威，文件只是兜底来源。
    """
    script = _legacy_script_json(video, ai_out)
    if script is None:
        return 0, 0
    if repo.get_ai_result(db, video_id) is not None:  # 导过就不重复导
        return 0, 0
    raw = script.read_text(encoding="utf-8", errors="replace")
    payload = _read_json(script)
    clips = repo.clips_from_payload(payload)
    if not clips:      # 抠不出可用片段的 JSON 不进库，免得"库里有 JSON"变成假状态
        return 0, 0
    task_id = repo.create_ai_task(db, video_id, mode=str(cfg.bridge.get("ai_job") or "full"),
                                 provider=str(cfg.bridge.get("provider") or ""),
                                 input_txt=None)
    repo.claim_ai_task(db, task_id)
    repo.complete_ai_task(db, task_id)
    result_id = repo.save_ai_result(db, video_id, task_id=task_id, raw_response=raw,
                                   json_data=payload, candidate_count=len(clips),
                                   winner_score=clips[0]["score"] if clips else None,
                                   validated=bool(clips))
    final = (ai_out / f"{video.stem}_高光时刻.mp4") if ai_out is not None else None
    rendered = final is not None and final.is_file() and final.stat().st_size > 0
    for spec in clips:
        repo.create_clip(db, video_id, spec, ai_result_id=result_id,
                         status="rendered" if rendered else "planned",
                         output_path=final if rendered else None)
    return 1, len(clips)


def import_all(cfg: Any, db: Database | None = None) -> dict[str, int]:
    """扫一遍旧缓存和产物，能认出来的都导进库。可以反复跑，重复的会更新而不是重复插入。"""
    db = db or open_db(cfg)
    cache_root = cache_mod.cache_dir(cfg)
    videos_root = cache_mod.videos_root(cache_root)
    out_root = cfg.path("output_dir")
    ai_out = _bridge_dir(cfg, "ai_output_dir") or (out_root if out_root.is_dir() else None)
    library = cache_mod.library_slugs(cfg) if cache_mod.library_root(cfg) is not None else None

    stats = {"videos": 0, "analyses": 0, "visual_events": 0, "speech_segments": 0,
             "speech_words": 0, "artifacts": 0, "ai_results": 0, "clips": 0, "skipped": 0}
    seen: set[str] = set()

    # 1) 缓存目录：最完整的一份（有 state.json 才知道属于哪个视频）
    if videos_root.is_dir():
        for slug_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
            state = _read_json(slug_dir / "state.json")
            raw_path = (state or {}).get("video") if isinstance(state, dict) else None
            if not raw_path:
                stats["skipped"] += 1
                continue
            video = Path(str(raw_path))
            in_library = None if library is None else slug_dir.name in library
            video_id = _register_video(db, video, slug_dir.name,
                                       _read_json(slug_dir / "probe.json"), in_library)
            stats["videos"] += 1
            seen.add(str(video.resolve() if video.is_file() else video))
            out_dir = out_root / video.stem
            imported = _import_analysis(db, video_id,
                                        _read_json(slug_dir / "speech.json"),
                                        _read_json(slug_dir / "visual.json"),
                                        out_dir if out_dir.is_dir() else None)
            if imported is None:  # 缓存里没有段也没有事件，退回 output/ 里找
                imported = _import_analysis(db, video_id,
                                            _read_json(out_dir / "speech_events.json"),
                                            _read_json(out_dir / "visual_events.json"),
                                            out_dir if out_dir.is_dir() else None)
            if imported is not None:
                stats["analyses"] += 1
                stats["visual_events"] += imported[0]
                stats["speech_segments"] += imported[1]
                stats["speech_words"] += imported[2]
            stats["artifacts"] += _import_artifacts(db, cfg, video_id, video, slug_dir.name,
                                                    out_dir if out_dir.is_dir() else None, ai_out)
            results, clips = _import_ai(db, cfg, video_id, video, ai_out)
            stats["ai_results"] += results
            stats["clips"] += clips

    # 2) 视频库 / input / AI_输入目录里还没登记的视频：至少让它们在库里有一条
    for folder in _scan_dirs(cfg):
        for video in sorted(p for p in folder.rglob("*")
                            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
            key = str(video.resolve())
            if key in seen:
                continue
            seen.add(key)
            slug = cache_mod.slug_for(video)
            slug_dir = videos_root / slug
            in_library = None if library is None else slug in library
            video_id = _register_video(db, video, slug,
                                       _read_json(slug_dir / "probe.json"), in_library)
            stats["videos"] += 1
            out_dir = out_root / video.stem
            imported = _import_analysis(db, video_id,
                                        _read_json(slug_dir / "speech.json")
                                        or _read_json(out_dir / "speech_events.json"),
                                        _read_json(slug_dir / "visual.json")
                                        or _read_json(out_dir / "visual_events.json"),
                                        out_dir if out_dir.is_dir() else None)
            if imported is not None:
                stats["analyses"] += 1
                stats["visual_events"] += imported[0]
                stats["speech_segments"] += imported[1]
                stats["speech_words"] += imported[2]
            stats["artifacts"] += _import_artifacts(db, cfg, video_id, video, slug,
                                                    out_dir if out_dir.is_dir() else None, ai_out)
            results, clips = _import_ai(db, cfg, video_id, video, ai_out)
            stats["ai_results"] += results
            stats["clips"] += clips

    logger.info("旧缓存导入完成：%s", stats)
    return stats


def _scan_dirs(cfg: Any) -> list[Path]:
    """要扫哪些目录找视频：视频库、input、AI_输入目录（去重、只要真存在的）。"""
    found: list[Path] = []
    library = cache_mod.library_root(cfg)
    if library is not None:
        found.append(library)
    for candidate in (cfg.path("input_dir"), _bridge_dir(cfg, "ai_input_dir")):
        if candidate is not None and Path(candidate).is_dir():
            found.append(Path(candidate))
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if all(resolved != p.resolve() for p in unique):
            unique.append(resolved)
    return unique


def _known_video(db: Database, video: Path) -> int | None:
    """路径和大小都跟库里对得上就直接用那条记录，省掉重新算指纹的 IO。

    指纹要读文件头中尾各 1MB，界面每次刷新对 40 个视频都算一遍就是上百 MB 读盘；
    大小变了或者路径没见过才真去算。
    """
    row = repo.get_video_by_path(db, video)
    if row is None:
        return None
    try:
        if int(row["file_size"] or -1) != video.stat().st_size:
            return None
    except OSError:
        return None
    return int(row["id"])


def register_video_files(cfg: Any, db: Database, video: Path, video_id: int,
                         ai_out: Path | None) -> int:
    """把这个视频相关的实际文件登记进 artifacts：原片、剧本 TXT、历史高光 JSON、高光片段。

    这是磁盘扫描唯一允许出现的地方（登记/对账）。登记完之后界面上的状态判断
    一律查库，不再自己 is_file()。
    """
    stem = video.stem
    found = 0
    candidates: list[tuple[str, Path]] = [("source_video", video)]
    for name in (f"{stem}.txt", f"{stem}_merged.txt"):
        candidates.append(("merged_txt", video.parent / name))
    # 脚本剪辑那一串脚本就放在视频旁边，所以两处都看
    for name in (f"{stem}_脚本.json", f"{stem}.json"):
        candidates.append(("ai_script", video.parent / name))
        if ai_out is not None:
            candidates.append(("ai_script", ai_out / name))
    if ai_out is not None:
        exact = ai_out / f"{stem}_高光时刻.mp4"
        candidates.append(("final_video", exact))
        # 成品名字被手改过也认：AI_输出目录里同名开头的视频文件都算成品。
        # 但渲染中的 .part 残片永远不算——它连视频后缀都不是，这里再显式挡一道。
        if ai_out.is_dir():
            for item in sorted(ai_out.iterdir()):
                if item == exact or item.name.endswith(PART_SUFFIX):
                    continue
                if (item.is_file() and item.stem.startswith(stem)
                        and item.suffix.lower() in VIDEO_SUFFIXES):
                    candidates.append(("final_video", item))
    for kind, path in candidates:
        try:
            if _ok_to_register(kind, path):
                repo.register_artifact(db, video_id, kind, path)
                found += 1
        except OSError as exc:
            logger.warning("登记文件失败 %s：%s", path, exc)
    return found


def sync_inputs(cfg: Any, db: Database | None = None, folders: list[Path] | None = None,
                ai_out: Path | None = None) -> dict[str, int]:
    """扫输入目录，把新出现的视频和文件登记进库（你手动丢进去的也能认）。

    只做登记，不跑分析、不删文件。重复跑不会重复插——视频按指纹/路径去重，
    文件按 (video, type, path) 唯一约束更新。

    顺带把**历史高光 JSON**（视频旁边或 AI_输出目录的 `<视频名>_脚本.json` /
    `<视频名>.json`）导成 ai_result + clips：这样"库里有没有可复用的高光 JSON"
    这个判断永远只查库，文件退回到纯兼容导入源。
    """
    db = db or open_db(cfg)
    if folders is None:
        folders = _scan_dirs(cfg)
    if ai_out is None:
        ai_out = _bridge_dir(cfg, "ai_output_dir")
    stats = {"videos_seen": 0, "videos_new": 0, "artifacts": 0,
             "ai_results": 0, "clips": 0}
    for folder in folders:
        if not folder.is_dir():
            continue
        for video in sorted(p for p in folder.rglob("*")
                            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
            stats["videos_seen"] += 1
            video_id = _known_video(db, video)
            if video_id is None:
                try:
                    video_id = repo.upsert_video(db, video, cache_slug=cache_mod.slug_for(video))
                except OSError as exc:
                    logger.warning("登记视频失败 %s：%s", video, exc)
                    continue
                stats["videos_new"] += 1
            stats["artifacts"] += register_video_files(cfg, db, video, video_id, ai_out)
            try:
                results, clips = _import_ai(db, cfg, video_id, video, ai_out)
            except Exception as exc:  # noqa: BLE001 - 一份坏 JSON 不该拖垮整次刷新
                logger.warning("历史高光 JSON 导入失败 %s：%s", video, exc)
            else:
                stats["ai_results"] += results
                stats["clips"] += clips
    return stats


def refresh_from_disk(cfg: Any, db: Database | None = None, folders: list[Path] | None = None,
                      ai_out: Path | None = None) -> dict[str, int]:
    """界面刷新时调这一个：先登记新文件，再跟磁盘对账，然后一切查库。

    这样磁盘扫描只发生在这里一次，状态判断函数（面板的四个状态、队列的三个
    文件判断）全都只查数据库，不会每个视频再去翻目录。
    """
    db = db or open_db(cfg)
    stats = sync_inputs(cfg, db, folders=folders, ai_out=ai_out)
    stats.update(reconcile(cfg, db))
    return stats


def reconcile(cfg: Any, db: Database | None = None) -> dict[str, int]:
    """对账：库里记的文件还在不在盘上，视频还在不在视频库里。只改状态，不删记录。

    数据库是状态的来源，但文件可能被你在资源管理器里删掉/挪走——
    所以要有这么一个入口把两边对齐（面板刷新、`run.py db --reconcile` 都走它）。
    """
    db = db or open_db(cfg)
    library = cache_mod.library_slugs(cfg) if cache_mod.library_root(cfg) is not None else None
    changed = {"videos_gone": 0, "videos_back": 0, "artifacts_gone": 0, "artifacts_back": 0}
    for row in repo.list_videos(db):
        path = Path(row["file_path"])
        exists = path.is_file()
        slug = row["cache_slug"] or cache_mod.slug_for(path)
        in_library = None if library is None else slug in library
        if bool(row["exists_on_disk"]) != exists:
            changed["videos_gone" if not exists else "videos_back"] += 1
        repo.set_video_presence(db, int(row["id"]), exists=exists, in_library=in_library)
        for art in repo.get_artifacts(db, int(row["id"])):
            target = Path(art["path"])
            here = target.is_file()
            if bool(art["exists_on_disk"]) == here:
                continue
            changed["artifacts_gone" if not here else "artifacts_back"] += 1
            repo.update_artifact(db, int(art["id"]), exists=here,
                                 size=target.stat().st_size if here else None)
    logger.info("对账完成：%s", changed)
    return changed
