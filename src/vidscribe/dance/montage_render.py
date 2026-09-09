"""渲染：只执行已经定稿的 Timeline。

技术指导第十六节的分层与禁令：

    Selection → Timeline → Render

    Render **不**负责：推荐、选择、对齐、查历史。
    Render 只做一件事：把一份确定的编辑计划变成一个能播的 MP4。

流程固定三步：

    素材片段 → 无声成片（video-only montage）→ mux 目标歌 → 最终成品

**源舞蹈视频的原声绝不能进最终音轨**（素材切片时就已经丢掉了，见 `material_slice`）。
目标歌是唯一正式音轨，成品的 `audio_streams` 必须正好等于 1 —— 这是硬验收项。

编码细节一律走 `media_backend`，这一层不碰 PyAV/cv2，也绝不 `subprocess ffmpeg`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from ..db.db import Database
from ..logging_setup import get_logger
from . import material_repository as repo, montage_timeline
from .types import DanceMontageContext, RenderResult

logger = get_logger("dance.render")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]

#: 无声中间文件的后缀。放在成品旁边、渲染完就删；名字带 `.mute` 是为了
#: 万一崩在中途，用户一眼能看出这不是成品
MUTE_SUFFIX = ".mute.mp4"


def output_name(context: DanceMontageContext, song_title: str = "") -> str:
    """成品文件名：`<歌名>_v<版本号>_<格数>格.mp4`。

    版本号进文件名是有意的：多版本混剪会在同一个目录里生成好几个文件，
    看名字就能对上界面里的第几版，不用去查库。
    """
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in (song_title or "dance"))[:40] or "dance"
    return f"{stem}_v{context.version_index:02d}_{len(context.clips)}格.mp4"

def render(db: Database, context: DanceMontageContext, *, out_dir: str | Path,
           version_id: int = 0, canvas=None, backend=None,
           song_path: str = "", keep_mute: bool = False,
           on_log: LogFn | None = None,
           on_progress: ProgressFn | None = None) -> RenderResult:
    """执行一份编辑计划，产出最终成品。

    失败一律记 `render_failed` 事件并把版本标 failed，**绝不增加出片次数**
    （技术指导第九节）。成功才记 `render_success` 并推 `output_count`。

    渲染前先跑 `montage_timeline.validate`：素材文件不在盘上这种事，
    在开始编码之前就该说清楚，而不是编到第 7 格才崩。
    """
    from . import history, media_backend  # noqa: PLC0415 - 这里才碰 av/cv2

    log = on_log or (lambda line: logger.info("%s", line))
    result = RenderResult(clips=len(context.clips))
    problems = montage_timeline.validate(context)
    fatal = [p for p in problems if "不在盘上" in p or "一格都没有" in p or "非法" in p]
    for problem in problems:
        log(f"[校验] {problem}")
    if fatal:
        result.error = "；".join(fatal[:3])
        _fail(db, history, version_id, context, result, log)
        return result

    music = Path(song_path or context.target_song_path)
    if not music.is_file():
        result.error = f"目标歌文件不在盘上：{music}"
        _fail(db, history, version_id, context, result, log)
        return result

    board = canvas if canvas is not None else media_backend.Canvas()
    engine = backend if backend is not None else media_backend.resolve("auto")
    result.backend = engine.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    song = repo.get_song(db, context.target_song_id)
    target = out_dir / output_name(context, str(song["title"]) if song is not None else "")
    mute = target.with_name(target.stem + MUTE_SUFFIX)

    # 素材文件本身就是一整段（切片时已经归一到 2 秒），所以 span 是 (0, 时长)
    spans = [(clip.file_path, 0.0, float(clip.duration)) for clip in context.clips]
    if version_id:
        repo.set_render_status(db, version_id, "rendering")
    try:
        video = engine.render_spans(spans, mute, board, on_log=log, on_progress=on_progress)
        result.frames = int(video.get("frames") or 0)
        result.fps = float(board.fps)
        result.width, result.height = board.width, board.height
        muxed = engine.mux_audio(mute, music, target,
                                duration=float(video.get("duration") or 0.0), on_log=log)
        meta = engine.probe(target)
        result.output = str(target)
        result.duration = meta.duration
        result.audio_duration = float(muxed.get("audio_samples") or 0) / 44100.0
        result.audio_streams = meta.audio_streams
        result.video_streams = meta.video_streams
    except Exception as exc:  # noqa: BLE001 - 渲染失败必须走失败路径记账
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"[渲染] 失败：{result.error}")
        _fail(db, history, version_id, context, result, log)
        return result
    finally:
        if not keep_mute:
            try:
                mute.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("删中间文件失败：%s", exc)

    # 硬验收：能播 + 目标歌是唯一音轨
    checks: list[str] = []
    if not media_backend.is_complete_video(target):
        checks.append("成品封装不完整")
    if result.audio_streams != 1:
        checks.append(f"音轨数是 {result.audio_streams}，必须正好 1 条（目标歌）")
    if result.video_streams != 1:
        checks.append(f"视频流数是 {result.video_streams}，必须正好 1 条")
    if checks:
        result.error = "；".join(checks)
        log(f"[渲染] 验收不过：{result.error}")
        _fail(db, history, version_id, context, result, log)
        return result

    result.ok = True
    if version_id:
        repo.set_render_status(db, version_id, "rendered", output_path=str(target),
                              detail=result.to_dict())
        history.note_render(db, version_id, context.montage_id, context.clips,
                            ok=True, detail={"output": str(target)})
    log(f"[渲染] 成功：{target.name}｜{result.duration:.3f}s｜"
        f"{result.width}x{result.height}@{result.fps:g}｜音轨 1 条（目标歌）")
    return result


def _fail(db: Database, history: Any, version_id: int, context: DanceMontageContext,
          result: RenderResult, log: LogFn) -> None:
    """失败收尾：标 failed + 记 render_failed 事件。**不动 output_count。**"""
    result.ok = False
    if not version_id:
        return
    repo.set_render_status(db, version_id, "failed", error=result.error,
                          detail=result.to_dict())
    history.note_render(db, version_id, context.montage_id, context.clips,
                        ok=False, detail={"error": result.error})
    log(f"[渲染] 版本 #{version_id} 已标记 failed（出片次数不变）")

def remix(db: Database, target_song_id: int, *, out_dir: str | Path,
          slice_duration: float = 2.0, versions: int = 0, strategy_id: int = 0,
          seed: int = 0, spec=None, canvas=None, backend=None, name: str = "",
          recommend_enabled: bool = True, manual: dict[int, int] | None = None,
          render_video: bool = True, on_log: LogFn | None = None,
          on_progress: ProgressFn | None = None) -> list[tuple[int, RenderResult]]:
    """**编排**：候选池 → 推荐 → 组合搜索 → 编辑计划 → 渲染，生成多个版本。

    放在这个模块里只是为了给 CLI 和 GUI 一个显而易见的入口；`render()` 本身
    仍然只执行已定稿的计划，一步都不越界。这个函数做的全部事情就是**按顺序调各层**。

    `recommend_enabled=False` + `manual={位置: 素材id}` 走纯手动路径
    （技术指导第二十节：关掉智能推荐后仍可以纯手动选择）。

    返回 `[(version_id, RenderResult)]`。`render_video=False` 时只出计划不渲染 ——
    界面上"先看看这一版长什么样"用它，不浪费几分钟编码。
    """
    from . import (  # noqa: PLC0415
        combination_search, material_selection, music_structure, recommendation,
        strategy as strategy_mod,
    )

    log = on_log or (lambda line: logger.info("%s", line))
    song = repo.get_song(db, target_song_id)
    if song is None:
        raise ValueError(f"目标歌 #{target_song_id} 不存在")
    song_path = str(song["file_path"])
    duration = float(song["duration"] or 0.0)
    positions = [p.index for p in music_structure.target_positions(duration, slice_duration)]
    if not positions:
        raise ValueError(f"目标歌只有 {duration:.2f}s，放不下一个 {slice_duration}s 的位置")

    plan = strategy_mod.resolve(db, strategy_id)
    log(f"[混剪] 目标歌《{song['title']}》{duration:.2f}s｜{len(positions)} 个位置"
        f"｜策略 {plan.name}（{plan.kind}）")

    montage_id = repo.ensure_montage(db, target_song_id=target_song_id,
                                     name=name or str(song["title"] or "混剪"),
                                     slice_duration=slice_duration)
    out: list[tuple[int, RenderResult]] = []

    if not recommend_enabled:
        if not manual:
            raise ValueError("关闭智能推荐时必须给出手动选择（manual={位置: 素材id}）")
        context = montage_timeline.from_manual(
            db, target_song_id, manual, song_path=song_path,
            slice_duration=slice_duration, montage_id=montage_id)
        version_id = montage_timeline.save(db, context)
        result = (render(db, context, out_dir=out_dir, version_id=version_id, canvas=canvas,
                         backend=backend, on_log=log, on_progress=on_progress)
                  if render_video else RenderResult(clips=len(context.clips)))
        return [(version_id, result)]

    ctx = material_selection.score_context(db, target_song_id)
    weights = recommendation.weights_for(plan.kind, plan.weights)
    pools = [material_selection.build_pool(db, target_song_id, index, spec=spec,
                                           context=ctx, weights=weights)
             for index in positions]
    empty = material_selection.empty_positions(pools)
    if empty:
        log(f"[混剪] 有 {len(empty)} 个位置没有素材，成片会短一些：{empty[:10]}")
    # 真的要出片了，这一轮进候选池的素材才算"被考虑过"（界面上单纯翻素材不记）
    from . import history  # noqa: PLC0415

    log(f"[混剪] 候选池共 {history.note_candidates(db, pools)} 条素材进入本轮考虑")

    run = recommendation.recommend(db, target_song_id, positions, strategy=plan, seed=seed,
                                   spec=spec, montage_id=montage_id, context=ctx)
    want = int(versions or (plan.search or {}).get("versions", 3) or 3)
    results = combination_search.search_versions(pools, ctx, plan, count=want,
                                                 seed=run.random_seed)
    if not results:
        raise ValueError("组合搜索一版都没搜出来（素材不够，或约束太紧）")

    for index, found in enumerate(results, 1):
        lookup = {m.id: m for m in found.materials}
        context = montage_timeline.build_timeline(
            db, target_song_id, found.picks, lookup, song_path=song_path,
            slice_duration=slice_duration, montage_id=montage_id,
            strategy_id=plan.id, recommendation_run_id=run.id)
        context.notes.extend(found.notes[:5])
        version_id = montage_timeline.save(db, context)
        log(f"[混剪] 第 {index}/{len(results)} 版 → 版本 #{version_id}"
            f"｜{len(context.clips)} 格｜综合重复率 {context.repeat.overall_repeat:.3f}")
        result = (render(db, context, out_dir=out_dir, version_id=version_id, canvas=canvas,
                         backend=backend, song_path=song_path, on_log=log,
                         on_progress=on_progress)
                  if render_video else RenderResult(clips=len(context.clips)))
        out.append((version_id, result))
    return out


def describe(result: RenderResult) -> list[str]:
    """把渲染结果写成中文行。"""
    if not result.ok:
        # 只出计划不渲染时 RenderResult 是空的：这不是失败，别把用户吓一跳
        if not result.error and not result.output:
            return [f"[计划] {result.clips} 格已定稿，本次没有渲染（--plan-only）"]
        return [f"[渲染] 失败：{result.error or '未知原因'}（出片次数不变）"]

    return [
        f"[渲染] 成品 {result.output}",
        f"  {result.duration:.3f}s｜{result.width}x{result.height}@{result.fps:g}"
        f"｜{result.frames} 帧｜{result.clips} 格",
        f"  音轨 {result.audio_streams} 条（目标歌，{result.audio_duration:.3f}s）"
        f"｜视频流 {result.video_streams} 条｜后端 {result.backend}",
    ]


__all__ = ["MUTE_SUFFIX", "output_name", "render", "remix", "describe"]


