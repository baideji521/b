"""命令行入口：环境检查 / 模型下载 / 批量处理 / 最终报告。

用法（在项目根目录，已激活 venv）：
    python -m vidscribe.cli check
    python -m vidscribe.cli download
    python -m vidscribe.cli run test.mp4
    python -m vidscribe.cli run                 # 处理 input/ 下所有视频
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# 允许直接以脚本方式运行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidscribe import benchmark as bench  # noqa: E402
from vidscribe.config import Config  # noqa: E402
from vidscribe.logging_setup import get_logger, setup_logging  # noqa: E402
from vidscribe.timeline.exporters import fmt_time, write_json  # noqa: E402
from vidscribe.video_io import list_videos  # noqa: E402
from vidscribe.visual.factory import BACKENDS  # noqa: E402

logger = get_logger("cli")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_mirror(cfg: Config) -> None:
    """统一走国内镜像：pip / HuggingFace 端点。模型仓库仍是官方 repo。"""
    from vidscribe.mirrors import apply_pip_env  # noqa: PLC0415

    endpoint = cfg.mirrors.get("hf_endpoint")
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    apply_pip_env(cfg.mirrors)
    os.environ.setdefault("PYTHONUTF8", "1")


def _apply_visual_override(cfg: Config, args: argparse.Namespace) -> None:
    """命令行覆盖视觉模型 / 后端（GUI 切换模型也是走这两个参数）。"""
    from vidscribe.visual.factory import known_models, resolve_backend  # noqa: PLC0415

    model = getattr(args, "visual_model", None)
    backend = getattr(args, "backend", None)
    if model:
        # 允许只写短名，比如 minicpm / MiniCPM-V-4_5-int4
        matched = model
        for entry in known_models(cfg.visual):
            if entry["model_id"].lower() == model.lower() or entry["model_id"].split("/")[-1].lower() == model.lower():
                matched = entry["model_id"]
                break
        cfg.visual["model_id"] = matched
        logger.info("视觉模型覆盖为: %s", matched)
    if backend:
        cfg.visual["backend"] = resolve_backend(cfg.visual["model_id"], backend)
        logger.info("视觉后端覆盖为: %s", cfg.visual["backend"])


def _apply_emotion_override(cfg: Config, args: argparse.Namespace) -> None:
    """命令行覆盖两路情绪识别的开关（GUI 的两个勾选框就是走这两个参数）。

    不给参数就按 config.json 走，所以 None 和 False 必须分开判。
    """
    audio = getattr(args, "audio_emotion", None)
    visual = getattr(args, "visual_emotion", None)
    if audio is not None:
        cfg.speech.setdefault("emotion", {})["enabled"] = bool(audio)
        logger.info("语音情绪识别: %s", "开" if audio else "关")
    if visual is not None:
        cfg.visual["emotion_enabled"] = bool(visual)
        logger.info("画面情绪识别: %s", "开" if visual else "关")


# ------------------------------------------------------------------ 环境检查


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    snapshot = bench.environment_snapshot()
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    problems: list[str] = []
    gpu = snapshot["gpu"]
    if not gpu.get("available"):
        problems.append("CUDA 不可用，将回退 CPU（速度会非常慢）")
    elif gpu.get("total_vram_mb", 0) < 8000:
        problems.append(f"显存偏小: {gpu.get('total_vram_mb')} MB")
    for pkg in ("torch", "transformers", "qwen-vl-utils", "faster-whisper", "opencv-python"):
        if not snapshot["packages"].get(pkg):
            problems.append(f"缺少依赖: {pkg}")
    try:
        import transformers  # noqa: PLC0415

        if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            problems.append("transformers 版本过低，缺少 Qwen3VLForConditionalGeneration（需要 >= 4.57.0）")
    except Exception as exc:
        problems.append(f"transformers 导入失败: {exc}")

    if problems:
        print("\n[WARN] 环境问题：")
        for p in problems:
            print(f"  - {p}")
        return 1 if any(p.startswith("缺少依赖") or "transformers" in p for p in problems) else 0
    print("\n[OK] 环境检查通过")
    return 0


# ------------------------------------------------------------------ 模型下载
def cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    _apply_mirror(cfg)
    _apply_visual_override(cfg, args)
    from vidscribe.mirrors import resolve_model, whisper_repo_id  # noqa: PLC0415

    model_dir = cfg.path("model_dir")
    targets = [cfg.visual["model_id"]]
    if args.all:
        targets += list(cfg.visual.get("fallback_model_ids", []))
    whisper_sizes = [cfg.speech["model_size"]]
    if args.all:
        whisper_sizes += list(cfg.speech.get("fallback_model_sizes", []))

    ok = True
    for repo in targets:
        path = resolve_model(repo, model_dir, cfg.mirrors, kind="visual", force=args.force)
        if path == repo:
            ok = False
            logger.error("视觉模型下载失败: %s", repo)
        else:
            logger.info("视觉模型已就绪: %s -> %s", repo, path)

    for size in whisper_sizes:
        repo = whisper_repo_id(size)
        path = resolve_model(repo, model_dir, cfg.mirrors, kind="whisper", force=args.force)
        if path == repo:
            ok = False
            logger.error("语音模型下载失败: %s", repo)
        else:
            logger.info("语音模型已就绪: %s -> %s", repo, path)
    return 0 if ok else 1


def _apply_speaker_override(cfg: Config, args: argparse.Namespace) -> None:
    """命令行覆盖声纹（说话人分离）模型（GUI 的「声纹」下拉就走这个参数）。

    取值：
      - 不给 / "auto"：按 config.json 里的 speech.speaker.model_id
      - "off" / "none"：这次不做说话人分离
      - "en" / "zh"：界面上那两个选项的简写
      - 其它：当成完整模型 id 用
    """
    want = getattr(args, "speaker_model", None)
    if not want:
        return
    speaker = cfg.speech.setdefault("speaker", {})
    value = str(want).strip()
    if value.lower() in ("auto", ""):
        logger.info("声纹模型: 按 config.json 的设置")
        return
    if value.lower() in ("off", "none", "no"):
        speaker["enabled"] = False
        logger.info("声纹模型: 关闭说话人分离")
        return
    # 两个别名必须跟 config.json 的 speech.speaker.models 对齐：一份纯英文、一份纯中文。
    alias = {"en": "iic/speech_campplus_sv_en_voxceleb_16k",
             "zh": "iic/speech_campplus_sv_zh-cn_16k-common"}
    model_id = alias.get(value.lower(), value)
    speaker["enabled"] = True
    speaker["model_id"] = model_id
    logger.info("声纹模型覆盖为: %s", model_id)


# ------------------------------------------------------------------ 主流程
def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    _apply_mirror(cfg)
    _apply_visual_override(cfg, args)
    _apply_emotion_override(cfg, args)
    _apply_speaker_override(cfg, args)
    from vidscribe.pipeline import Pipeline  # noqa: PLC0415
    from vidscribe.speech.whisper_asr import LanguageNotAllowed  # noqa: PLC0415


    videos: list[Path] = []
    for item in args.videos:
        path = Path(item)
        if not path.is_absolute():
            path = cfg.root / path
        if path.is_dir():
            videos.extend(list_videos(path))
        elif path.is_file():
            videos.append(path)
        else:
            logger.error("找不到: %s", path)
    if not args.videos:
        videos = list_videos(cfg.path("input_dir"))
        if not videos:
            fallback = sorted(cfg.root.glob("*.mp4")) + sorted(cfg.root.glob("*.mkv")) \
                + sorted(cfg.root.glob("*.mov")) + sorted(cfg.root.glob("*.avi"))
            videos = fallback[: args.limit] if args.limit else fallback
            if videos:
                logger.info("input/ 为空，改用项目根目录下的视频：%s", ", ".join(v.name for v in videos))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        logger.error("没有可处理的视频。把视频放到 %s 或用命令行指定路径。", cfg.path("input_dir"))
        return 2

    logger.info("待处理视频 %d 个", len(videos))
    pipeline = Pipeline(cfg)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for i, video in enumerate(videos, start=1):
            logger.info("[%d/%d] %s", i, len(videos), video.name)
            try:
                results.append(pipeline.run_video(
                    video, force=args.force,
                    skip_visual=args.skip_visual, skip_speech=args.skip_speech,
                    force_speech=getattr(args, "force_speech", False),
                    translate=getattr(args, "translate", False),
                ))
            except LanguageNotAllowed as exc:
                # 语言不在允许范围：不是失败，是"这条别跑"。打一行机器可读的标记，
                # GUI 就靠它判断该弹窗（手动）还是跳过（自动）；库里的标记已经在
                # pipeline.run_video 里落好了
                print(f"[语言拦截] language={exc.language} video={video}", flush=True)
                logger.error("跳过 %s：音频语言 %s 不在 speech.allowed_languages 里",
                             video.name, exc.language)
                results.append({
                    "video": video.name, "video_path": str(video), "status": "SKIP_LANGUAGE",
                    "error": f"语言 {exc.language} 不在允许范围",
                })
            except Exception as exc:  # 单个视频失败不影响其它视频
                logger.error("处理 %s 失败: %s", video.name, exc)
                logger.debug(traceback.format_exc())
                results.append({
                    "video": video.name, "video_path": str(video), "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "traceback": traceback.format_exc()[-2000:],
                })
    finally:
        if not cfg.runtime.get("keep_models_loaded", True):
            pipeline.close()

    total = round(time.perf_counter() - started, 2)
    report_path = cfg.root / "FINAL_REPORT.txt"
    write_final_report(report_path, cfg, results, total)
    write_json_report(cfg.path("log_dir") / "run_summary.json", cfg, results, total)
    logger.info("最终报告: %s", report_path)

    return 0 if all(r.get("status") == "OK" for r in results) else 1


# ------------------------------------------------------------------ 翻译
def cmd_translate(cfg: Config, args: argparse.Namespace) -> int:
    """中英互译，纯文本，不解码视频。

    两种用法：
    - `translate <输出目录>`：翻译该目录里还没有译文的语音段与画面事件（增量）
    - `translate --items 请求.json --result 结果.json`：GUI 用的模式，
      只翻译请求文件里给出的那些行（也就是界面上当前显示、还没译文的行）
    """
    _apply_mirror(cfg)
    _apply_visual_override(cfg, args)
    from vidscribe import progress as progress_mod  # noqa: PLC0415
    from vidscribe.translate import translate_items, translate_output  # noqa: PLC0415

    def on_progress(done: int, total: int) -> None:
        progress_mod.report("translate", done / max(total, 1), f"{done}/{total} 行")

    # --- 模式一：只翻译请求文件里的行（GUI）---
    if args.items:
        request_path = Path(args.items)
        if not request_path.is_file():
            logger.error("找不到翻译请求文件: %s", request_path)
            return 2
        with open(request_path, "r", encoding="utf-8") as fh:
            request = json.load(fh)
        rows = request.get("items") or []
        logger.info("按界面内容翻译 %d 行（不重新分析视频）", len(rows))
        result = translate_items(cfg, rows, source=request.get("source"), on_progress=on_progress)
        if args.result:
            write_json(Path(args.result), result)
        if not result.get("ok"):
            logger.error("翻译失败：%s %s", result.get("reason"), result.get("detail") or "")
            return 1
        logger.info("翻译完成：%s -> %s，成功 %d 行，失败 %d 行，耗时 %.1fs",
                    result.get("source_language"), result.get("target_language"),
                    len(result.get("translations") or {}), len(result.get("failed") or []),
                    result.get("elapsed_seconds") or 0.0)
        return 0

    # --- 模式二：翻译一个输出目录 ---
    if not args.target:
        logger.error("请给出输出目录，或用 --items 指定翻译请求文件")
        return 2
    target = Path(args.target)
    if not target.is_absolute():
        candidate = cfg.path("output_dir") / args.target
        target = candidate if candidate.exists() else cfg.root / args.target
    if target.is_file():  # 传视频路径时自动换成它的输出目录
        target = cfg.path("output_dir") / target.stem
    if not target.is_dir():
        logger.error("找不到输出目录: %s", target)
        return 2

    # _apply_visual_override 已经把 --visual-model / --backend 写进 cfg.visual
    # （短名也在那里补全成完整 id），所以这里不再单独传 model_id
    result = translate_output(cfg, target, retranslate=args.retranslate, on_progress=on_progress)
    if not result.get("ok"):
        logger.error("翻译失败：%s %s", result.get("reason"), result.get("detail") or "")
        return 1
    if result.get("reason") == "already_translated":
        logger.info("无需翻译：%s", result.get("detail"))
        return 0
    logger.info("翻译完成：%s -> %s，语音 %d/%d，事件 %d/%d",
                result.get("source_language"), result.get("target_language"),
                result.get("speech_translated"), result.get("speech_total"),
                result.get("event_translated"), result.get("event_total"))
    return 0


# ------------------------------------------------------------------ 缓存
def cmd_cache(cfg: Config, args: argparse.Namespace) -> int:
    """看/清缓存：固定目录 cache/videos（断点、预览音频）+ logs/（日志）。

    默认只报告；`--clean` 才真删，`--dry-run` 配合 `--clean` 只列出要删什么。
    output/ 的分析结果和 models/ 的权重永远不动。
    """
    from vidscribe import cache as cache_mod  # noqa: PLC0415

    cache_mod.migrate_layout(cfg)  # 顺手把旧的 work/<视频名>/ 布局搬过来
    days = float(args.days if args.days is not None
                 else cfg.runtime.get("cache_max_age_days", 3))
    info = cache_mod.status(cfg, max_age_days=days)
    logger.info("%s", cache_mod.summary_line(info))
    logger.info("日志目录 %s；上次清理 %s（%s 天前）",
                info["log_dir"], info["last_cleanup"] or "从未",
                info["days_since_cleanup"] if info["days_since_cleanup"] is not None else "-")
    if info["stale_names"]:
        logger.info("超过 %g 天的缓存：%s", days, "，".join(info["stale_names"][:20])
                    + (" ..." if len(info["stale_names"]) > 20 else ""))
    if not args.clean:
        return 0

    result = cache_mod.cleanup(cfg, max_age_days=days, dry_run=args.dry_run)
    verb = "将删除" if args.dry_run else "已删除"
    logger.info("%s %d 项，%s %s", verb, len(result["removed"]),
                "预计腾出" if args.dry_run else "腾出",
                cache_mod.human_size(result["freed_bytes"]))
    if result["failed"]:
        logger.warning("删除失败 %d 项（可能正被占用）：%s",
                       len(result["failed"]), "，".join(result["failed"][:10]))
    return 0


# ------------------------------------------------------------------- 数据库
def _db_report_check(result: dict[str, Any]) -> int:
    """打印体检结果，返回 exit code（有问题就非 0）。"""
    logger.info("数据库检查")
    logger.info("--------------------")
    logger.info("SQLite integrity : %s", result["integrity"])
    logger.info("Foreign keys     : %s", result["foreign_keys"])
    logger.info("Schema version   : v%s（程序要 v%s）",
                result["version"], result["expected_version"])
    logger.info("Tables           : %s",
                "OK" if not result["missing_tables"] else "缺 " + "、".join(result["missing_tables"]))
    logger.info("Indexes          : %s",
                "OK" if not result["missing_indexes"]
                else "缺 " + "、".join(result["missing_indexes"]))
    logger.info("Journal mode     : %s", result["journal_mode"])
    logger.info("Database         : %s", "OK" if result["writable"] else "写不进去")
    for row in result["fk_violations"]:
        logger.warning("外键不一致：%s", row)
    if result["ok"]:
        return 0
    for problem in result["problems"]:
        logger.error("有问题：%s", problem)
    return 1


def _db_report_stats(stats: dict[str, Any], cache: dict[str, Any]) -> None:
    """整库统计。数字全部来自 SQL，不重新扫目录。"""
    videos = stats["videos"]
    logger.info("视频：总数 %d，在盘上 %d，已不在盘上 %d",
                videos["total"], videos["on_disk"], videos["missing"])
    logger.info("分析：%s", "，".join(f"{k} {v}" for k, v in stats["analysis"].items()))
    logger.info("AI 任务：%s", "，".join(f"{k} {v}" for k, v in stats["tasks"].items()))
    logger.info("AI 结果：%d", stats["ai_results"])
    clips = stats["clips"]
    logger.info("片段：总数 %d，已出片 %d，失败 %d，计划中 %d",
                clips["total"], clips["rendered"], clips["failed"], clips["planned"])
    art = stats["artifacts"]
    logger.info("文件记录：总数 %d，还在 %d，丢了 %d", art["total"], art["on_disk"], art["missing"])
    logger.info("文件按类型：%s", "，".join(f"{k} {v}" for k, v in stats["artifacts_by_type"].items())
                or "（没有）")
    logger.info("逐词 %d，语音段 %d，视觉事件 %d",
                stats["speech_words"], stats["speech_segments"], stats["visual_events"])
    logger.info("分析次数：总 %d（完成 %d / 失败 %d / 还在跑 %d），不同配置组合 %d，"
                "同配置重复跑过 %d 次",
                cache["runs_total"], cache["runs_completed"], cache["runs_failed"],
                cache["runs_running"], cache["distinct_configs"], cache["reruns"])
    logger.info("视觉模型：%s", "，".join(f"{k} {v}" for k, v in cache["by_vision_model"].items())
                or "（没有）")
    logger.info("ASR 模型：%s", "，".join(f"{k} {v}" for k, v in cache["by_asr_model"].items())
                or "（没有）")
    logger.info("说明：命中缓存那次不会写库（省下的就是没跑），所以命中次数没法从库里数出来；"
                "上面的「重复跑过」才是确凿的未命中次数")


# 四串活儿的中文名，只给日志看着舒服用（值本身由 bridge.ai_job 定，见 gui/ai_options.JOB_FLAGS）
JOB_NAMES = {"full": "剪辑成片", "collect": "收取高光 JSON", "script": "高光 JSON 剪辑",
             "analyze": "只解析视频"}


def _db_report_queue(cfg: Config, db: Any) -> None:
    """自动剪辑总览。和 GUI 那八格是同一个数：同一个函数、同一个目录、同一个 done_key。

    命令行自己不做任何加减：范围来自 `videos_under(AI_输入目录)`，分桶来自
    `repo.video_queue_statistics`，这样 `db --stats` 和面板不可能各说一套。
    """
    from vidscribe.db import repo  # noqa: PLC0415
    from vidscribe.db.importer import _bridge_dir  # noqa: PLC0415

    in_dir = _bridge_dir(cfg, "ai_input_dir")
    if in_dir is None:
        logger.info("自动剪辑总览：AI_输入目录没配或不在盘上（bridge.ai_input_dir），跳过")
        return
    ids = [int(row["id"]) for row in repo.videos_under(db, in_dir)]
    job = str(cfg.bridge.get("ai_job") or "full")
    # 每一串各自的"干完了"口径，跟 AI 面板同一行判断：收 JSON 那串拿到 JSON 就算完事，
    # 只解析视频那串本地分析入库就算完事，其余两串要出成品才算。
    # 注意 video_queue_statistics 现在只对 "json" 单独特判，别的值都按成品算，
    # 所以 analyze 这一档的「已完成」桶暂时还是成品数，真实进度看下面单独打的「已分析」
    if job == "analyze":
        done_key = "analysed"
    elif job == "collect":
        done_key = "json"
    else:
        done_key = "clipped"
    st = repo.video_queue_statistics(db, ids, mode=job, done_key=done_key)
    logger.info("自动剪辑总览（%s，干的是 %s）", in_dir, JOB_NAMES.get(job, job))
    logger.info("  总视频 %d / 已获取 JSON %d（横切指标，和下面的桶会重叠）",
                st["total"], st["json"])
    if job == "analyze":
        # 只解析视频这一档就看这一个数：本地分析进库的有多少条，剩下几步压根不跑
        logger.info("  已分析 %d / 还没分析 %d（这一档分析入库就算干完，不问 AI 也不剪）",
                    st["analysed"], st["total"] - st["analysed"])
    logger.info("  已完成 %d / 剪辑中 %d / 等待 AI %d / 待剪辑 %d / 失败 %d / 已取消 %d / "
                "未获取 JSON %d",
                st["done"], st["rendering"], st["waiting_ai"], st["pending_render"],
                st["failed"], st["cancelled"], st["no_json"])
    buckets = (st["done"] + st["rendering"] + st["waiting_ai"] + st["pending_render"]
               + st["failed"] + st["cancelled"] + st["no_json"])
    if buckets != st["total"]:
        logger.error("  分桶合计 %d ≠ 总视频 %d（统计口径出问题了）", buckets, st["total"])
    missing = repo.missing_input_videos(db, in_dir)
    if missing:
        logger.info("  另有 %d 个登记过但现在盘上找不着的输入视频（不进上面的分桶）：%s",
                    len(missing), "，".join(row["file_name"] for row in missing[:5]))


def cmd_db(cfg: Config, args: argparse.Namespace) -> int:
    """SQLite 库：建库 / 导入旧缓存 / 对账 / 体检 / 备份恢复 / 瘦身 / 查孤儿。

    库只记状态（视频、分析批次、事件、逐词时间戳、AI 任务与结果、片段、文件），
    分析结果照旧落 output/ 和 cache/。这里的命令一律不删历史记录，也不删业务文件。
    """
    from vidscribe import cache as cache_mod  # noqa: PLC0415
    from vidscribe.db import admin, db_path, open_db  # noqa: PLC0415
    from vidscribe.db import repo  # noqa: PLC0415
    from vidscribe.db.importer import import_all, reconcile  # noqa: PLC0415

    db = open_db(cfg)
    logger.info("库文件 %s", db_path(cfg))
    code = 0
    if args.init:
        logger.info("库已就绪，结构版本 v%s", db.value("PRAGMA user_version"))
    if args.do_import:
        stats = import_all(cfg, db)
        logger.info("导入：视频 %d，分析 %d，视觉事件 %d，语音段 %d，逐词 %d，"
                    "文件 %d，AI 结果 %d，片段 %d（跳过 %d 份没记视频路径的缓存）",
                    stats["videos"], stats["analyses"], stats["visual_events"],
                    stats["speech_segments"], stats["speech_words"], stats["artifacts"],
                    stats["ai_results"], stats["clips"], stats["skipped"])
    if args.reconcile:
        changed = reconcile(cfg, db)
        logger.info("对账：视频没了 %d / 又回来了 %d；文件没了 %d / 又回来了 %d",
                    changed["videos_gone"], changed["videos_back"],
                    changed["artifacts_gone"], changed["artifacts_back"])
    if args.recover:
        tasks = repo.recover_stale_ai_tasks(
            db, float(cfg.runtime.get("ai_task_timeout_minutes", 30)))
        runs = repo.recover_stale_analyses(
            db, float(cfg.runtime.get("analysis_timeout_minutes", 180)))
        logger.info("恢复：%d 个卡住的 AI 任务退回等待，%d 条没跑完的分析标成失败", tasks, runs)
    if args.fill_duration:
        stats = repo.fill_missing_durations(db)
        logger.info("补时长：查了 %d 个（duration 为空且文件还在盘上），补上 %d 个，探不到 %d 个",
                    stats["checked"], stats["filled"], stats["failed"])
    if args.check:
        code = _db_report_check(admin.health_check(db)) or code
    if args.backup is not None:
        try:
            dest = admin.backup(db, args.backup or None)
            checked = admin.verify_file(dest)
            logger.info("备份：%s（%s，integrity %s，foreign_keys %s，v%s）",
                        dest, cache_mod.human_size(dest.stat().st_size),
                        checked["integrity"], checked["foreign_keys"], checked["version"])
            if not checked["ok"]:
                logger.error("备份文件不合格：%s", "；".join(checked["problems"]))
                code = 1
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.error("备份失败：%s", exc)
            code = 1
    if args.restore:
        try:
            report = admin.restore(db, args.restore)
        except sqlite3.Error as exc:
            logger.error("恢复失败：%s", exc)
            return 1
        if not report["restored"]:
            logger.error("没恢复：%s", report.get("error", "备份不合格"))
            return 1
        logger.info("已恢复自 %s；当前库先备份到 %s", report["source"], report["safety_backup"])
        logger.info("恢复后行数：%s", "，".join(f"{k} {v}" for k, v in report["counts"].items()))
        code = _db_report_check(report["after_check"]) or code
    if args.vacuum:
        result = admin.vacuum(db, force=bool(args.force))
        if not result["done"]:
            logger.error("没做 VACUUM：%s", result.get("error", "未知原因"))
            code = 1
        else:
            freed = result["size_before"] - result["size_after"]
            logger.info("VACUUM 完成：%s -> %s（%s %s），journal_mode %s，integrity %s",
                        cache_mod.human_size(result["size_before"]),
                        cache_mod.human_size(result["size_after"]),
                        "腾出" if freed >= 0 else "多占",
                        cache_mod.human_size(abs(freed)),
                        result["journal_mode"], result["integrity"])
    if args.orphans:
        found = admin.orphans(db)
        for item in found["relations"]:
            logger.warning("%s：%d 条（%s.%s 对不上 %s），例：%s", item["why"], item["count"],
                           item["table"], item["column"], item["parent"], item["sample"])
        if not found["relations"]:
            logger.info("表之间的关联没有对不上的")
        logger.info("登记着但文件已经没了：%d 条（只报告，不删记录）",
                    found["artifacts_missing_total"])
        for item in found["artifacts_missing"][:10]:
            logger.info("  丢了：#%d %s %s", item["id"], item["type"], item["path"])
        loose = admin.unregistered_files(cfg, db)
        logger.info("盘上有、库里还没登记的视频：%d 个（跑 `db --reconcile` 或 `--import` 补登记）",
                    loose["count"])
        for name in loose["sample"][:10]:
            logger.info("  没登记：%s", name)
    if args.stats:
        _db_report_stats(repo.full_stats(db), repo.cache_stats(db))

    rows = repo.counts(db)
    logger.info("表内容：%s", "，".join(f"{k} {v}" for k, v in rows.items()))
    _db_report_queue(cfg, db)
    return code


# ------------------------------------------------------------------ 高光剪辑
def cmd_highlight(cfg: Config, args: argparse.Namespace) -> int:
    """按 AI JSON 剪高光：从 segments[0].sa 剪到 .end 原速播放，片尾按配置冻住末帧几秒。"""

    from vidscribe.highlight import parse_spec, render_highlight, resolve_video  # noqa: PLC0415
    from vidscribe.highlight import clip_engine  # noqa: PLC0415
    from vidscribe.video_io import is_complete_video  # noqa: PLC0415

    try:  # Windows 控制台默认 GBK，日志里的中文字幕会花屏
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


    if args.json:
        source = Path(args.json)
        if not source.is_absolute():
            source = cfg.root / source
        if not source.is_file():
            logger.error("找不到 JSON 文件: %s", source)
            return 2
        raw = source.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        logger.error("没有读到 JSON 内容（用 --json 指定文件，或从标准输入喂进来）")
        return 2

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON 解析失败: %s", exc)
        return 2
    try:
        spec = parse_spec(payload)          # 先解一遍：既校验 JSON，也用来定位源视频
    except ValueError as exc:
        # 多条写法（clips: [...]）走引擎时，用第一条来定位源视频；单条写法照旧报错
        alt = None if args.no_engine else clip_engine.first_clip_payload(payload)
        try:
            spec = parse_spec(alt) if alt else None
        except ValueError:
            spec = None
        if spec is None:
            logger.error("JSON 内容不合规: %s", exc)
            return 2

    fallback = None
    if args.video:
        candidate = Path(args.video)
        fallback = candidate if candidate.is_absolute() else cfg.root / candidate
    try:
        video = resolve_video(spec, cfg.path("output_dir"), cfg.path("input_dir"), fallback)
    except FileNotFoundError as exc:
        logger.error("%s（可用 --video 指定源视频）", exc)
        return 2

    manual = None
    if args.out:
        manual = Path(args.out)
        if not manual.is_absolute():
            manual = cfg.root / manual

    # ---- 剪辑引擎：用逐词时间戳把 AI 的粗区间修成语义边界（--no-engine 可关掉）----
    jobs: list[tuple[Any, Path]] = []
    if args.no_engine:
        try:
            one = spec.shifted(args.start_offset, args.end_offset)
        except ValueError as exc:
            logger.error("加减秒数不合规: %s", exc)
            return 2
        jobs.append((one, manual or _planned_target(cfg, video, one)))
    else:
        result = _clip_plans(cfg, video, payload)
        for line in clip_engine.describe_result(result):
            print(line, flush=True)
        if not result.plans:
            logger.error("剪辑引擎没给出任何可剪片段，不启动渲染")
            return 2
        if args.dry_run:
            logger.info("dry-run：只算不剪，已跳过渲染")
            return 0
        for index, plan in enumerate(result.plans, start=1):
            try:
                one = parse_spec(clip_engine.payload_for(plan))
                one = one.shifted(args.start_offset, args.end_offset)
            except ValueError as exc:
                logger.error("第 %d 段修正后的区间不能渲染: %s", index, exc)
                return 2
            jobs.append((one, _numbered_target(manual, index) if manual
                         else _planned_target(cfg, video, one, index)))

    for index, (job, out_path) in enumerate(jobs, start=1):
        if manual is None and out_path.exists():
            print(f"[剪辑引擎] 同名成品已经在盘上，跳过：{out_path.name}", flush=True)
            continue
        print(f"[剪辑引擎] 开始渲染第 {index}/{len(jobs)} 段 -> {out_path.name}", flush=True)
        try:
            result_info = render_highlight(video, job, out_path,
                                           on_log=lambda line: print(line, flush=True),
                                           freeze_seconds=float(
                                               cfg.highlight.get("freeze_tail_seconds", 2.0)))
        except Exception as exc:
            logger.error("剪辑失败: %s", exc)
            logger.debug(traceback.format_exc())
            return 1
        if not is_complete_video(out_path):     # 和登记闸门同一个判断（Batch 4）
            logger.error("成片封装不完整，不当成成品: %s", out_path)
            return 1
        print("[剪辑引擎] 成片验证通过", flush=True)
        write_json(out_path.with_suffix(".json"),
                   {"spec": job.raw,
                    "offsets": {"start": args.start_offset, "end": args.end_offset},
                    "result": result_info})
        logger.info("高光片段已生成: %s", out_path)
    return 0


def _numbered_target(target: Path, index: int) -> Path:
    """多段高光的输出名：第一段用原名，后面的加 _2 / _3，互不覆盖。"""
    if index <= 1:
        return target
    return target.with_name(f"{target.stem}_{index}{target.suffix}")


def _planned_target(cfg: Config, video: Path, job: Any, index: int = 1,
                    seconds: float | None = None) -> Path:
    """按**实际剪进去的时长**命名：`<视频名>_1095.mp4` = 10.95 秒（和 GUI 同一个口径）。

    区间 = 起剪 → 结束（含加减秒数；引擎开着就是修正后的区间）。冻帧不算，
    静音压缩剪掉的那部分也不算（`seconds` 给了就以它为准，那是保留片段的总长），
    「提取数据」导出的 duration 也是这个口径，两边永远对得上。
    同名不让位也不覆盖：调用方看到文件已存在就跳过这一段（已经剪过了）。
    """
    from vidscribe.highlight import default_target  # noqa: PLC0415

    raw = float(job.duration) if seconds is None else float(seconds)
    return default_target(_export_dir(cfg, video), video, round(max(0.0, raw), 2), index=index)


def _clip_plans(cfg: Config, video: Path, payload: Any) -> Any:
    """跑剪辑引擎：逐词时间戳从库里取，视频时长现探；取不到就退化成只做合法性校验。"""
    from vidscribe.highlight import clip_engine  # noqa: PLC0415
    from vidscribe.video_io import probe_video  # noqa: PLC0415

    segments: tuple[Any, ...] = ()
    duration: float | None = None
    use_engine = bool(cfg.highlight.get("clip_engine", False))
    try:
        from vidscribe.db import open_db, repo  # noqa: PLC0415

        db = open_db(cfg)
        try:
            row = repo.find_video(db, video)
            if row is not None:
                if use_engine:
                    segments = clip_engine.segments_for_video(db, int(row["id"]))
                # 库里没时长就现探一次写回（和 GUI 同一个入口，不另写一套）
                duration = repo.ensure_duration(db, int(row["id"]), video)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 - 没库也要能剪，只是没法修边界
        logger.warning("取不到逐词时间戳（%s），本次不修正边界", exc)
    if not use_engine:
        logger.info("剪辑引擎已关闭（highlight.clip_engine=false）："
                    "按 AI 的区间原样剪，只按视频时长收尾")
    elif not segments:
        logger.warning("库里没有这个视频的逐词时间戳，AI 区间将原样使用（只按视频时长收尾）")
    else:
        logger.info("逐词时间戳：%d 句", len(segments))
    if duration is None:
        try:
            duration = float(probe_video(video).duration) or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("探不到视频时长（%s），不按时长收尾", exc)
    return clip_engine.plan_clips(payload, segments, video_duration=duration,
                                  source_video=str(video))



def _export_dir(cfg: Config, video: Path) -> Path:
    """和 GUI 共用「导出目录」：读 gui_settings.json 的 export_dir，没设过就用该视频的结果目录。"""
    settings_file = cfg.root / "gui_settings.json"
    if settings_file.is_file():
        try:
            saved = json.loads(settings_file.read_text(encoding="utf-8")).get("export_dir")
        except Exception:  # noqa: BLE001 - 设置文件坏了不该让剪辑失败
            saved = None
        if saved and Path(saved).is_dir():
            return Path(saved)
    return cfg.path("output_dir") / video.stem


# ------------------------------------------------------------------ GUI


def _fmt_seconds(value: Any) -> str:
    """秒 -> 0:12 这种短写法；没有值就给个占位。"""
    try:
        total = float(value)
    except (TypeError, ValueError):
        return "-"
    if total <= 0:
        return "-"
    return f"{int(total) // 60}:{int(total) % 60:02d}"


def _pick_video(db: Any, target: str) -> Any:
    """`--video` 既收数字 id，也收文件名片段（够用就行，多条时取最早那个）。"""
    text = str(target).strip()
    if text.isdigit():
        return db.one("SELECT * FROM videos WHERE id = ?", (int(text),))
    row = db.one("SELECT * FROM videos WHERE file_name = ?", (text,))
    if row is not None:
        return row
    return db.one("SELECT * FROM videos WHERE file_name LIKE ? ORDER BY id LIMIT 1",
                  (f"%{text}%",))


def _print_asset_rows(rows: list[Any]) -> None:
    for row in rows:
        flags = []
        if int(row["is_current"] or 0):
            flags.append("当前")
        if row["deleted_at"]:
            flags.append("已删除")
        best = "-" if row["best_score"] is None else f"{float(row['best_score']):.2f}"
        print(f"  #{row['id']:<5} {str(row['name']):<14} {str(row['source_type']):<9}"
              f" {str(row['provider'] or '-'):<10} {str(row['model'] or '-'):<26}"
              f" 片段 {int(row['clip_count'] or 0):<3} 最高分 {best:<6}"
              f" {row['created_at']} {' '.join(flags)}")


def _print_lineage(info: dict[str, Any], spans: dict[str, Any] | None = None) -> None:
    asset = info.get("asset")
    prm = info.get("prm")
    print(f"  成品 #{info['artifact_id']}  {Path(str(info['path'])).name}"
          + ("" if info["exists_on_disk"] else "（文件已不在盘上）"))
    print(f"    视频      : {(info.get('video') or {}).get('file_name', '-')}")
    print(f"    分析批次  : {info.get('analysis_id') or '-'}")
    print(f"    高光方案  : "
          + ("-" if asset is None else
             f"#{asset['id']} {asset['name']}（{asset['source_type']}，"
             f"{asset['clip_count']} 个片段）"
             + ("（已删除）" if info["asset_deleted"] else "")))
    print(f"    AI / 模型 : {info.get('provider') or '-'} / {info.get('model') or '-'}")
    print(f"    AI 任务   : {info.get('task_id') or '-'}")
    print(f"    PRM       : "
          + ("-" if prm is None else
             f"#{prm['id']} {prm['name']}（{prm['filename']}）"
             + ("（已删除）" if info["prm_deleted"] else "")))
    if not spans:
        return
    for index, clip in enumerate(spans.get("ai") or (), start=1):
        score = clip.get("score")
        print(f"    AI 原始 {index} : {clip['start']} → {clip['end']}"
              f"（评分 {'-' if score is None else score}）")
    for index, plan in enumerate(spans.get("engine") or (), start=1):
        print(f"    Engine {index}  : {plan['start']} → {plan['end']}（{plan['duration']}s）")
        for note in plan["notes"]:
            print(f"      原因    : {note}")
        if not plan["notes"]:
            print("      原因    : 未调整（AI 区间本身就落在语义边界上）")
    actual = spans.get("actual") or ()
    for index, clip in enumerate(actual, start=1):
        print(f"    实际渲染{index} : {clip['start']} → {clip['end']}（{clip['duration']}s）")
    if not actual:
        print("    实际渲染  : 库里没有对应的 clips 记录（Batch 11 之前剪的老成品）")


def _read_json_arg(cfg: Config, path_text: str) -> Any:
    """读一个 JSON 文件（相对路径按项目根拼）。读不成就抛 ValueError。"""
    source = Path(path_text)
    if not source.is_absolute():
        source = cfg.root / source
    if not source.is_file():
        raise ValueError(f"找不到 JSON 文件：{source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc


def _assets_overview(cfg: Config, db: Any, assets: Any, limit: int) -> None:
    """视频表：时长 / 分析 / 高光方案 / 成品，全部来自 SQL。"""
    rows = db.all("SELECT id, file_name, duration FROM videos ORDER BY id DESC LIMIT ?",
                  (int(limit),))
    if not rows:
        print("库里还没有视频")
        return
    ids = [int(r["id"]) for r in rows]
    plans = assets.asset_counts(db, ids)
    products = assets.product_counts(db, ids)
    analysed = {int(r["video_id"]) for r in db.all(
        "SELECT DISTINCT video_id FROM analysis_runs WHERE status = ?", ("completed",))}
    print(f"{'ID':<5} {'时长':<7} {'分析':<5} {'方案':<5} {'成品':<5} 文件")
    for row in rows:
        vid = int(row["id"])
        print(f"{vid:<5} {_fmt_seconds(row['duration']):<7}"
              f" {'有' if vid in analysed else '无':<4} {plans.get(vid, 0):<5}"
              f" {products.get(vid, 0):<5} {row['file_name']}")
    print(f"（共列出 {len(rows)} 个视频，用 --video <id|文件名> 看详情）")


def _assets_detail(cfg: Config, db: Any, assets: Any, row: Any) -> None:
    """一个视频的全景：分析、方案清单、成品清单（每条带溯源）。"""
    view = assets.video_overview(db, int(row["id"]))
    analysis = view["analysis"]
    print(f"视频 #{row['id']}  {row['file_name']}")
    print(f"  路径      : {row['file_path']}")
    print(f"  时长      : {_fmt_seconds(row['duration'])}")
    print(f"  最近分析  : "
          + ("无" if analysis is None else
             f"#{analysis['id']} {analysis['status']} {analysis['started_at']}"
             f"（逐词 {view['word_count']} 个）"))
    every = assets.list_assets(db, int(row["id"]), include_deleted=True)
    print(f"  高光方案（{len(every)} 份，含已删除）：" if every else "  高光方案：无")
    _print_asset_rows(every)
    print(f"  成品（{len(view['products'])} 个）：" if view["products"] else "  成品：无")
    for info in view["products"]:
        _print_lineage(info)


def _assets_import_moments(cfg: Config, args: argparse.Namespace, db: Any, assets: Any,
                           video_row: Any) -> int:
    """结果清单（AI 只给两个时间点）→ 一行一份高光方案入库。

    区间不是 AI 给的：`plan_from_span` 拿库里的逐词时间戳把两个点吸到句边界上，
    所以这条路比 `--import-json` 多一个硬前提 —— 这个视频得先跑过分析。
    """
    from vidscribe import ai_protocol  # noqa: PLC0415
    from vidscribe.highlight import clip_engine, from_moments  # noqa: PLC0415

    source = Path(args.import_moments)
    if not source.is_absolute():
        source = cfg.root / source
    if not source.is_file():
        logger.error("找不到清单文件：%s", source)
        return 2
    rows = from_moments.rows_from_text(source.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        logger.error("清单里一行都没解出来（要的是一行一个 JSON 对象）：%s", source)
        return 2
    video_id = int(video_row["id"])
    name = str(video_row["file_name"])
    # 清单每一行都带着剧本第一行那个文件名。和 `--video` 指定的对不上就**直接拒**：
    # 那是「导错视频」的铁证，按指定视频硬算只会入库一批别人的区间，事后极难发现
    named = {one for one in (from_moments.video_of_row(row) for row in rows) if one}
    wrong = sorted(one for one in named if one != name)
    if wrong:
        logger.error("清单里写的是 %s，和指定的视频 %s 对不上，不导",
                     "、".join(wrong), name)
        return 2
    segments = clip_engine.segments_for_video(db, video_id)
    if not segments:
        logger.error("视频 #%d 库里没有语音，算不出区间——先跑一遍 run", video_id)
        return 2

    try:
        seconds = float(video_row["duration"]) if video_row["duration"] else None
    except (IndexError, KeyError, TypeError, ValueError):
        seconds = None
    made, logs = from_moments.payloads_from_rows(segments, rows, video_name=name,
                                                 duration=seconds)
    for line in logs:
        print(line, flush=True)
    if not made:
        logger.error("清单 %d 行一条都没算出合法区间，不登记", len(rows))
        return 2
    if args.dry_run:
        logger.info("dry-run：只算不入库（%d/%d 行可用）", len(made), len(rows))
        return 0

    # 翻译是可选的一步：填的是 `Speech text`（中文），失败也不拦入库 ——
    # 区间已经算准了，译文回头可以再补，没必要为一次模型加载失败丢掉整批方案
    if getattr(args, "translate", False):
        from vidscribe.translate import translate_items  # noqa: PLC0415

        wanted = [{"key": str(i), "text": speech}
                  for i, (_, speech, _) in enumerate(made) if speech]
        result = translate_items(cfg, wanted) if wanted else {"translations": {}}
        got = result.get("translations") or {}
        for i, (payload, _, _) in enumerate(made):
            text = got.get(str(i))
            if text:
                payload["clip"]["Trimclip"]["Speech text"] = text
        if wanted and not got:
            logger.warning("翻译一行都没成：%s，Speech text 先留空",
                           result.get("detail") or result.get("reason") or "原因不明")

    for index, (payload, speech, row) in enumerate(made, start=1):
        note = args.note or f"{source.name} 第 {index} 条"
        if speech and not payload["clip"]["Trimclip"]["Speech text"]:
            note += f"｜区间原文待译：{speech}"
        # raw_json 存 AI 原话（那两个时间点），current_json 存程序算出的区间，
        # 血缘上一眼能看出「AI 指哪儿」和「程序切到哪儿」差了多少。
        # 入库前先升级成新协议：和 AI 那条路存进 current_json 的形状保持一致
        asset_id = assets.create_asset(
            db, video_id, ai_protocol.payload_of(payload), source_type="imported",
            name=(f"{args.name} {index}" if args.name else None),
            note=note, raw_payload=row,
            make_current=(not args.no_current) and index == 1)
        clip = payload["clip"]
        logger.info("方案 #%d ← %.2f-%.2f（%.2fs）", asset_id,
                    clip["start"], clip["end"], clip["duration"])
    logger.info("清单 %d 行 → 入库 %d 份方案（Speech text 留空，等翻译）",
                len(rows), len(made))
    return 0


def _assets_extract(cfg: Config, args: argparse.Namespace, db: Any, assets: Any,
                    video_row: Any) -> int:
    """成品 → 第二轮混剪的输入行（和资产中心「提取数据」**同一份实现**）。

    只导**剪出过成品**的 JSON，一个成品一行：先看盘（文件真的还在），再去溯源它的 JSON。
    库里的 `exists_on_disk` 只是上次扫盘的旧状态，不拿它当准。

    加减秒数（startframe / freeze）优先用 JSON 里那份（剪辑时盖进去的，记的是这个成品
    当时的口径），没有才退回界面设置里的值 —— 和 GUI 那条路读的是同一个文件。

    成品渲染时剪过超时静音的，`gaps` 按**剪完之后**的片段算（`keeps_for` 现算，
    和渲染同一个 `trim_plan`）：不这么算，第二轮的旁白会压在人说话上。
    """
    from vidscribe import ai_protocol  # noqa: PLC0415
    from vidscribe.gui import settings as gui_settings  # noqa: PLC0415 - 纯 json，不拉 Qt
    from vidscribe.highlight import clip_engine  # noqa: PLC0415
    from vidscribe.highlight.extract import (  # noqa: PLC0415
        extract_line,
        keeps_for,
        unresolved_note,
        unresolved_seconds,
    )

    video_id = int(video_row["id"])
    name = str(video_row["file_name"])
    saved = gui_settings.load(cfg).get("highlight_offsets") or ()
    deltas = [ai_protocol.num(saved[i]) if i < len(saved) else None for i in (0, 1)]
    startframe = deltas[0] if deltas[0] is not None else 0.0
    freeze = deltas[1] if deltas[1] is not None else 0.0
    # 静音压缩：成品剪成哪几段不入库，这里按**成品的真实时长**反推（`keeps_for`），
    # 空隙才落在和盘上那个文件一致的坐标上。**不看 silence_keep** —— 配置改一次就得
    # 把旧成品全重剪才能导数据，那没必要。GUI 的「提取数据」走同一个函数
    speech = clip_engine.segments_for_video(db, video_id)
    if not speech:
        logger.warning("视频 #%d 没有逐词时间戳，gaps 全给空数组（第二轮就别插配音了）",
                       video_id)

    lines: list[str] = []
    empty = 0                 # 没有可用空隙的条数：第二轮插不了配音，报个数
    unknown = 0               # 时长既不等于区间长度、也反推不出剪法：clips 和文件对不上
    for info in assets.products_overview(db, video_id):
        path = Path(str(info["path"]))
        if not path.is_file():
            logger.info("跳过 %s（文件不在盘上）", path.name)
            continue
        if info["asset_id"] is None:
            logger.info("跳过 %s（成品没挂上高光 JSON，溯源不到）", path.name)
            continue
        spans = info.get("spans") or ()
        if len(spans) > 1:
            # 一个成品正常只对应一段，多出来的是历史脏数据（clips 张冠李戴）
            logger.warning("%s 挂着 %d 段实际区间，导出只用第一段", path.name, len(spans))
        region = ((spans[0].get("start"), spans[0].get("end")) if spans else None)
        recorded = (spans[0].get("duration") if spans else None)
        keeps = keeps_for(speech, region, made=recorded)
        # 时长解释不了：这条 clips 记录和盘上的文件本来就对不上，gaps 不可信
        if unresolved_seconds(recorded, keeps, region):
            unknown += 1

        one = extract_line(
            assets.asset_payload(db, int(info["asset_id"])), path.name,
            startframe=startframe, freeze=freeze, source=name,
            span=recorded,
            region=region,
            speech=speech,
            keeps=keeps)
        if one is None:
            logger.info("跳过 %s（溯源到的 JSON 里没有片段）", path.name)
            continue
        lines.append(one[0])
        if not one[1]:
            empty += 1

    if not lines:
        logger.error("视频 #%d 没有可提取的成品素材", video_id)
        return 2
    target = Path(args.out) if args.out else (
        cfg.path("output_dir") / f"高光提取_{video_id}_{len(lines)}条.txt")
    if not target.is_absolute():
        target = cfg.root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("提取 %d 条 → %s", len(lines), target)
    if empty:
        logger.warning("其中 %d 条没有可用原声空隙，第二轮只能当无旁白素材用", empty)
    if unknown:
        logger.warning("%s", unresolved_note(unknown, len(lines)))
    return 0



def _assets_render(cfg: Config, args: argparse.Namespace, db: Any, assets: Any) -> int:
    """只用库里的高光方案剪成片：**一次 AI 都不调**。"""
    from vidscribe.db import repo  # noqa: PLC0415
    from vidscribe.highlight import parse_spec, render_highlight  # noqa: PLC0415
    from vidscribe.highlight import clip_engine  # noqa: PLC0415
    from vidscribe.video_io import is_complete_video  # noqa: PLC0415

    asset = assets.get_asset(db, int(args.render))
    if asset is None:
        logger.error("没有 #%s 这个高光方案", args.render)
        return 2
    if asset["deleted_at"]:
        logger.error("方案 #%s 已删除，先 --restore 再剪", args.render)
        return 2
    payload = assets.loads(asset["current_json"])
    if payload is None:
        logger.error("方案 #%s 的 JSON 解不开", args.render)
        return 2
    video_row = db.one("SELECT * FROM videos WHERE id = ?", (int(asset["video_id"]),))
    if video_row is None:
        logger.error("方案 #%s 挂的视频在库里找不到", args.render)
        return 2
    video = Path(str(video_row["file_path"]))
    if not video.is_file():
        logger.error("源视频已不在盘上：%s", video)
        return 2

    manual = Path(args.out) if args.out else None
    if manual is not None and not manual.is_absolute():
        manual = cfg.root / manual

    result = _clip_plans(cfg, video, payload)
    for line in clip_engine.describe_result(result):
        print(line, flush=True)
    if not result.plans:
        logger.error("剪辑引擎没给出任何可剪片段，不启动渲染")
        return 2
    if args.dry_run:
        logger.info("dry-run：只算不剪，已跳过渲染")
        return 0

    made: list[Path] = []
    # 静音压缩：区间里超时的静音剪掉，只留 `silence_keep` 秒（0 = 不剪）。
    # 和 GUI 读同一个键、算同一个 `trim_plan`，成品口径只有一份
    silence_keep = max(0.0, float(cfg.highlight.get("silence_keep", 2.0)))
    silence_target = max(0.0, float(cfg.highlight.get("silence_target", 0.0)))
    words: tuple[Any, ...] = ()
    if silence_keep > 0:
        words = clip_engine.segments_for_video(db, int(video_row["id"]))
        if not words:
            logger.info("视频 #%d 没有逐词时间戳，这次不剪静音", int(video_row["id"]))
    for index, plan in enumerate(result.plans, start=1):
        try:
            job = parse_spec(clip_engine.payload_for(plan))
        except ValueError as exc:
            logger.error("第 %d 段不能渲染：%s", index, exc)
            return 2
        keeps = None
        if words:
            spans = clip_engine.trim_for(words, job.clip_start, job.clip_end,
                                         keep=silence_keep, target=silence_target)
            if len(spans) > 1:
                keeps = spans
        played = (round(sum(hi - lo for lo, hi in keeps), 3) if keeps
                  else max(0.0, float(job.duration)))
        if keeps:
            print(f"[静音压缩] 第 {index} 段 静音最多留 {silence_keep:.2f}s："
                  f"{job.duration:.2f}s → {played:.2f}s，剪成 {len(keeps)} 段 "
                  + " + ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in keeps), flush=True)
        # 和 GUI 的 render_asset 同名口径：<视频名>_<时长>.mp4，时长取实际剪进去的那个数
        out_path = (_numbered_target(manual, index) if manual is not None
                    else _planned_target(cfg, video, job, index, seconds=played))
        if manual is None and out_path.exists():
            print(f"[剪辑引擎] 同名成品已经在盘上，跳过：{out_path.name}", flush=True)
            continue
        print(f"[剪辑引擎] 开始渲染第 {index}/{len(result.plans)} 段 -> {out_path.name}", flush=True)
        try:
            info_render = render_highlight(
                video, job, out_path, on_log=lambda line: print(line, flush=True),
                freeze_seconds=float(cfg.highlight.get("freeze_tail_seconds", 2.0)),
                keep_spans=keeps)
        except Exception as exc:
            logger.error("剪辑失败：%s", exc)
            logger.debug(traceback.format_exc())
            return 1
        if not is_complete_video(out_path):
            logger.error("成片封装不完整，不当成成品：%s", out_path)
            return 1
        print("[剪辑引擎] 成片验证通过", flush=True)
        # 成品记账走数据层的同一个入口：写实际剪辑区间 → 登记成品 → 挂方案
        # PRM 不参与：那是问 AI 时用的提示词，渲染这一步压根不读它
        spec = assets.clip_spec_for(plan, job.clip_start, job.clip_end)
        if keeps:
            # 区间两头还是这一段在原视频里的范围，但成品里中间少了几截，
            # 时长记实际剪进去的那个数 —— 和文件名、提取数据同一个口径
            spec["duration"] = played
        info = assets.record_product(
            db, int(video_row["id"]), out_path,
            specs=[spec],
            asset_id=int(asset["id"]))
        made.append(out_path)
        logger.info("成品已生成并记账：%s（方案 #%s，实际区间 %.2f → %.2f）", out_path,
                    asset["id"], job.clip_start, job.clip_end)
        logger.debug("artifact #%s / clips %s", info["artifact_id"], info["clip_ids"])
    print(f"[高光方案] 本次共产出 {len(made)} 个成品，全部挂在方案 #{asset['id']} 名下")
    return 0


def cmd_assets(cfg: Config, args: argparse.Namespace) -> int:
    """高光方案管理：查、导入、复制、编辑、软删、设当前、只用 JSON 剪、成品溯源。

    不给任何开关就打印视频总表。所有写操作都只动库，不删任何文件。
    """
    from vidscribe.db import assets as db_assets  # noqa: PLC0415
    from vidscribe.db import open_db  # noqa: PLC0415

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    db = open_db(cfg)
    code = 0
    try:
        if args.extract:
            if not args.video:
                logger.error("--extract 要配 --video <id|文件名>")
                return 2
            row = _pick_video(db, args.video)
            if row is None:
                logger.error("库里找不到视频：%s", args.video)
                return 2
            return _assets_extract(cfg, args, db, db_assets, row)
        if args.import_moments:
            if not args.video:
                logger.error("--import-moments 要配 --video <id|文件名>，得知道挂在哪个视频下")
                return 2
            row = _pick_video(db, args.video)
            if row is None:
                logger.error("库里找不到视频：%s", args.video)
                return 2
            return _assets_import_moments(cfg, args, db, db_assets, row)
        if args.import_json:
            if not args.video:
                logger.error("--import-json 要配 --video <id|文件名>，得知道挂在哪个视频下")
                return 2
            row = _pick_video(db, args.video)
            if row is None:
                logger.error("库里找不到视频：%s", args.video)
                return 2
            try:
                payload = _read_json_arg(cfg, args.import_json)
            except ValueError as exc:
                logger.error("%s", exc)
                return 2
            count, best = db_assets.summarize(payload)
            if not count:
                logger.error("这份 JSON 里抠不出可用片段，不登记（免得队列以为有方案）")
                return 2
            asset_id = db_assets.create_asset(
                db, int(row["id"]), payload, source_type="imported",
                name=args.name, note=args.note or f"从 {args.import_json} 导入",
                make_current=not args.no_current)
            logger.info("已登记方案 #%d（%d 个片段，最高分 %s）", asset_id, count,
                        "-" if best is None else f"{best:.2f}")
        if args.copy is not None:
            new_id = db_assets.copy_asset(db, int(args.copy), name=args.name)
            if new_id is None:
                logger.error("没有 #%s 这个方案，复制不了", args.copy)
                code = 2
            else:
                logger.info("方案 #%s 已复制成 #%d（原件一个字没动）", args.copy, new_id)
        if args.edit is not None:
            if not args.json:
                logger.error("--edit 要配 --json <改好的 JSON 文件>")
                return 2
            try:
                payload = _read_json_arg(cfg, args.json)
            except ValueError as exc:
                logger.error("%s", exc)
                return 2
            new_id = db_assets.edit_asset(db, int(args.edit), payload,
                                          in_place=args.in_place, name=args.name)
            if new_id is None:
                logger.error("没有 #%s 这个方案（或已删除），改不了", args.edit)
                code = 2
            elif args.in_place:
                logger.info("方案 #%s 已就地更新（raw_json 仍是 AI 原话）", args.edit)
            else:
                logger.info("已在方案 #%s 上另开新方案 #%d（原方案不动）", args.edit, new_id)
        if args.delete is not None:
            if db_assets.delete_asset(db, int(args.delete)):
                kept = len(db_assets.products_for_asset(db, int(args.delete)))
                logger.info("方案 #%s 已软删（%d 个已有成品一个都没动）", args.delete, kept)
            else:
                logger.error("没有 #%s 这个方案，或它已经是删除状态", args.delete)
                code = 2
        if args.restore is not None:
            if db_assets.restore_asset(db, int(args.restore)):
                logger.info("方案 #%s 已恢复", args.restore)
            else:
                logger.error("没有 #%s 这个方案，或它本来就没删", args.restore)
                code = 2
        if args.set_current is not None:
            if db_assets.set_current_asset(db, int(args.set_current)):
                logger.info("方案 #%s 已设为当前方案（自动剪辑就用它）", args.set_current)
            else:
                logger.error("没有 #%s 这个方案，或它已删除", args.set_current)
                code = 2
        if args.by_ai:
            rows = db_assets.assets_by_ai(db, provider=args.by_ai, model=args.model)
            print(f"AI = {args.by_ai}"
                  + (f" / {args.model}" if args.model else "") + f"：{len(rows)} 份方案")
            _print_asset_rows(rows)
        if args.by_prm is not None:
            rows = db_assets.assets_by_prm(db, int(args.by_prm))
            print(f"PRM #{args.by_prm}：{len(rows)} 份方案")
            _print_asset_rows(rows)
            products = db_assets.products_for_prm(db, int(args.by_prm))
            print(f"这一版 PRM 剪出过 {len(products)} 个成品")
        if args.trace is not None:
            info = db_assets.artifact_lineage(db, int(args.trace))
            if info is None:
                logger.error("没有 #%s 这个成品记录", args.trace)
                code = 2
            else:
                _print_lineage(info, db_assets.lineage_spans(db, int(args.trace)))
        if args.render is not None:
            return _assets_render(cfg, args, db, db_assets) or code
        if args.video and not args.import_json:
            row = _pick_video(db, args.video)
            if row is None:
                logger.error("库里找不到视频：%s", args.video)
                return 2
            _assets_detail(cfg, db, db_assets, row)
            return code
        if not any((args.import_json, args.copy is not None, args.edit is not None,
                    args.delete is not None, args.restore is not None,
                    args.set_current is not None, args.by_ai, args.by_prm is not None,
                    args.trace is not None)):
            _assets_overview(cfg, db, db_assets, args.limit)
    finally:
        db.close()
    return code


def cmd_prm(cfg: Config, args: argparse.Namespace) -> int:
    """PRM 档案：列出 / 新增 / 改 / 软删 / 恢复 / 设默认。

    只登记名字、文件名、语言、版本这些元信息，**提示词内容始终只在文件里**。
    """
    from vidscribe.db import assets as db_assets  # noqa: PLC0415
    from vidscribe.db import open_db  # noqa: PLC0415

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    db = open_db(cfg)
    code = 0
    try:
        if args.add:
            if not args.file:
                logger.error("--add 要配 --file <提示词文件路径>")
                return 2
            path = Path(args.file)
            if not (path if path.is_absolute() else cfg.root / path).is_file():
                logger.warning("提示词文件现在不在盘上：%s（还是照登记，路径以后可以改）", path)
            prm_id = db_assets.create_prm(db, args.add, args.file,
                                          description=args.description,
                                          language=args.language, version=args.version,
                                          make_default=args.default)
            logger.info("已登记 PRM #%d %s（%s）%s", prm_id, args.add, args.file,
                        "，并设为默认" if args.default else "")
        if args.edit is not None:
            if db_assets.update_prm(db, int(args.edit), name=args.name, filename=args.file,
                                    description=args.description, language=args.language,
                                    version=args.version):
                logger.info("PRM #%s 已更新", args.edit)
            else:
                logger.error("PRM #%s 没改动（要么不存在，要么一个字段都没给）", args.edit)
                code = 2
        if args.delete is not None:
            if db_assets.delete_prm(db, int(args.delete)):
                kept = len(db_assets.products_for_prm(db, int(args.delete)))
                logger.info("PRM #%s 已软删（%d 个历史成品照旧查得到用的是它）",
                            args.delete, kept)
            else:
                logger.error("没有 #%s 这个 PRM，或它已经是删除状态", args.delete)
                code = 2
        if args.set_default is not None:
            if db_assets.set_default_prm(db, int(args.set_default)):
                logger.info("PRM #%s 已设为默认（不再硬编码 prm_en.txt）", args.set_default)
            else:
                logger.error("没有 #%s 这个 PRM，或它已删除", args.set_default)
                code = 2
        if args.list or not any((args.add, args.edit is not None, args.delete is not None,
                                 args.set_default is not None)):
            rows = db_assets.list_prms(db, include_deleted=args.all)
            if not rows:
                print("PRM 档案还是空的（发一次 AI 会自动把用到的提示词登记进来）")
            else:
                print(f"{'ID':<5} {'名字':<16} {'语言':<6} {'版本':<8} {'默认':<5} 文件")
                for row in rows:
                    marks = "是" if int(row["is_default"] or 0) else ""
                    if row["deleted_at"]:
                        marks = "已删除"
                    exists = "" if db_assets.prm_file(row, cfg.root).is_file() else "（文件不在）"
                    print(f"{int(row['id']):<5} {str(row['name']):<16}"
                          f" {str(row['language'] or '-'):<6} {str(row['version'] or '-'):<8}"
                          f" {marks:<5} {row['filename']}{exists}")
                used = db_assets.products_for_prm
                print("（成品数：" + "，".join(
                    f"#{int(r['id'])} {len(used(db, int(r['id'])))}" for r in rows) + "）")
    finally:
        db.close()
    return code


def cmd_gui(cfg: Config, args: argparse.Namespace) -> int:
    _apply_mirror(cfg)
    try:
        from vidscribe.gui.main_window import launch  # noqa: PLC0415
    except ImportError as exc:
        logger.error("GUI 依赖缺失（需要 PyQt5）：%s", exc)
        logger.error("安装命令: pip install PyQt5==5.15.11 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1

    video = args.video
    if video:
        path = Path(video)
        if not path.is_absolute():
            path = cfg.root / path
        video = str(path)
    return launch(cfg, video)


def cmd_ai(cfg: Config, args: argparse.Namespace) -> int:
    """只开 AI 面板（第二主界面）。主界面在后台备着但不显示，关掉面板就退出。"""
    _apply_mirror(cfg)
    try:
        from vidscribe.gui.main_window import launch  # noqa: PLC0415
    except ImportError as exc:
        logger.error("GUI 依赖缺失（需要 PyQt5）：%s", exc)
        logger.error("安装命令: pip install PyQt5==5.15.11 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1

    return launch(cfg, panel_only=True, auto=bool(getattr(args, "auto", False)))



# ------------------------------------------------------------------ 报告
def write_json_report(path: Path, cfg: Config, results: list[dict], total: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": total,
        "environment": bench.environment_snapshot(),
        "config": cfg.to_dict(),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _preview(output_dir: str, limit: int = 4) -> list[str]:
    lines: list[str] = []
    timeline_file = Path(output_dir) / "timeline.json"
    if not timeline_file.is_file():
        return lines
    try:
        with open(timeline_file, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception:
        return lines
    from vidscribe.language import labels_for  # noqa: PLC0415

    labels = labels_for(doc.get("output_language"))
    for entry in doc.get("timeline", [])[:limit]:
        lines.append(f"    [{fmt_time(entry['start'])} - {fmt_time(entry['end'])}]  ({entry['start']}s)")
        if entry.get("visual"):
            lines.append(f"      {labels['visual']}: {entry['visual']}")
        if entry.get("speech"):
            lines.append(f"      {labels['speech']}: {entry['speech']}")
    return lines


def write_final_report(path: Path, cfg: Config, results: list[dict], total: float) -> None:
    env = bench.environment_snapshot()
    gpu = env["gpu"]
    ok = [r for r in results if r.get("status") == "OK"]
    failed = [r for r in results if r.get("status") != "OK"]

    lines = [
        "=" * 72,
        "视频理解工具 - 最终报告 FINAL REPORT",
        "=" * 72,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时:   {total:.1f}s",
        f"结果:     成功 {len(ok)} / 失败 {len(failed)} / 共 {len(results)}",
        "",
        "-" * 72,
        "环境",
        "-" * 72,
        f"OS:          {env['os']}",
        f"Python:      {env['python']}",
        f"GPU:         {gpu.get('name', 'N/A')}  ({gpu.get('total_vram_mb', 'N/A')} MB, CC {gpu.get('capability', 'N/A')})",
        f"驱动/CUDA:   driver={gpu.get('driver', 'N/A')}  torch_cuda={gpu.get('torch_cuda', 'N/A')}  cudnn={gpu.get('cudnn', 'N/A')}",
        f"torch:       {env['packages'].get('torch')}",
        f"transformers:{env['packages'].get('transformers')}",
        f"qwen-vl-utils:{env['packages'].get('qwen-vl-utils')}",
        f"faster-whisper:{env['packages'].get('faster-whisper')} (ctranslate2 {env['packages'].get('ctranslate2')})",
        "",
    ]

    for r in results:
        lines += ["-" * 72, f"视频: {r['video']}   [{r.get('status')}]", "-" * 72]
        if r.get("status") != "OK":
            lines += [f"  错误: {r.get('error')}", ""]
            continue
        b = r.get("benchmark", {})
        video = b.get("video", {})
        timings = b.get("timings", {})
        peak = b.get("peak_vram") or {}
        vm = b.get("visual_model", {})
        sm = b.get("speech_model") or {}
        ld = r.get("language_decision") or {}
        lr = r.get("language_render") or {}
        audio_line = "NONE" if not video.get("has_audio") else (
            "OK" if ld.get("audio_available") else f"UNUSABLE ({ld.get('reason', '')})"
        )
        lang_line = "DEFAULT" if ld.get("default_used") else str(ld.get("output_language"))
        lines += [
            f"  输出目录:   {r['output_dir']}",
            f"  视频规格:   {video.get('duration')}s  {video.get('width')}x{video.get('height')}  {video.get('fps')} fps  音轨={video.get('has_audio')}",
            f"  Audio:      {audio_line}",
            f"  Language:   {lang_line}   detected={ld.get('detected_language')}({ld.get('language_confidence')})  "
            f"dominant={ld.get('dominant_language')}  secondary={ld.get('secondary_languages') or []}  "
            f"output_language={ld.get('output_language')}",
            f"  语言判定依据: {ld.get('reason')}",
            f"  最终语言渲染: 语种不符={lr.get('mismatched', 0)}  模型改写={lr.get('rewritten_by_model', 0)}  "
            f"模板/保留原文={lr.get('template_or_kept', 0)}",
            f"  视觉模型:   {vm.get('model_id')}  后端={vm.get('backend')}  帧来源={vm.get('frame_source')}  窗口={vm.get('windows')}  分析帧数={vm.get('analyzed_frames')}  降级次数={vm.get('degrade_attempts')}",
            f"  视觉参数:   {json.dumps(vm.get('params'), ensure_ascii=False)}",
            f"  语音模型:   {sm.get('size')} / {sm.get('device')} / {sm.get('compute_type')}   语言={r.get('language')}",
            f"  耗时(s):    探测={timings.get('probe_seconds', 0):.1f}  视觉={timings.get('visual_seconds', 0):.1f}  "
            f"语音={timings.get('speech_seconds', 0):.1f}  timeline={timings.get('timeline_seconds', 0):.1f}  "
            f"总计={timings.get('total_seconds', 0):.1f}",
            f"  Peak VRAM:  allocated={peak.get('allocated_mb', 'N/A')} MB  reserved={peak.get('reserved_mb', 'N/A')} MB",
            f"  产出条数:   视觉事件={r.get('visual_events')}  语音段={r.get('speech_segments')}  timeline={r.get('timeline_entries')}",
            f"  文件:       timeline.json / timeline.txt / timeline.srt({b.get('srt_kind')}) / visual_events.json / speech_events.json / benchmark.json / video_metadata.json",
            "",
            "  时间轴预览:",
        ]
        preview = _preview(r["output_dir"])
        lines += preview if preview else ["    (无)"]
        lines.append("")

    lines += ["=" * 72, "验收要点", "=" * 72]
    if ok:
        r = ok[0]
        lines += [
            f"1. 什么时候发生了什么 -> {r['output_dir']}\\timeline.txt / timeline.json 的 visual 字段",
            f"2. 什么时候说了什么   -> 同上的 speech 字段；词级时间戳在 speech_events.json",
            "3. 定位回原视频       -> timeline.json 的 start/end 是真实秒数（浮点），可直接 seek",
            "4. 最终语言           -> timeline.json 的 original_language / output_language；"
            "原始对白始终保存在 speech_events.json 的 original_text / original_language",
        ]
    else:
        lines.append("没有成功的视频，请查看 logs/ 下的日志定位问题。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_montage(cfg: Config, args: argparse.Namespace) -> int:
    """检查第二轮混剪的输出：能机器判的全判一遍，主观项另出一张评分表。

    为什么要有这个命令：混剪 PRM 里绝大多数规则都是可计算的（素材分数、同源、
    Rank 1 闭嘴、旁白说不说得完），而实测 AI 出错**全部**集中在这些地方。
    人盯着一页 JSON 一条条对时间戳既慢又漏，程序一秒就报完。

    返回码：0 = 全部合规；1 = 有不合规（好接到批处理里当闸门）；2 = 参数或文件不对。
    """
    from vidscribe.highlight import montage as check  # noqa: PLC0415

    try:  # Windows 控制台默认 GBK，报告里的中文会花屏
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    def _resolve(text: str) -> Path:
        one = Path(text)
        return one if one.is_absolute() else cfg.root / one

    pool_path = _resolve(args.pool)
    if not pool_path.is_file():
        logger.error("找不到提取数据：%s（就是「提取数据」导出的那份 txt）", pool_path)
        return 2
    pool = check.load_pool(pool_path.read_text(encoding="utf-8").splitlines())
    if not pool:
        logger.error("%s 里一行素材都没读出来", pool_path)
        return 2

    if args.json:
        source = _resolve(args.json)
        if not source.is_file():
            logger.error("找不到混剪 JSON：%s", source)
            return 2
        raw = source.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    checked = check.check_all(raw, pool)
    print(check.report(checked))

    if args.scorecard:
        target = _resolve(args.scorecard)
        target.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig：Excel 双击打开 CSV 时不带 BOM 会把中文认成乱码
        target.write_text(check.scorecard(checked, raw, pool), encoding="utf-8-sig")
        logger.info("评分表 → %s（机器判定已填好，故事/节奏/情绪那几项留空等你打分）", target)

    return 1 if any(bad for _i, bad in checked) else 0


# ------------------------------------------------------------------ 卡点舞混剪
def _dance_song(db: Any, cfg: Config, text: str):
    """`--song` 既能给库里的 id，也能给一个文件路径。返回 `TargetSong`。

    给路径时顺手注册 + 分析（命中指纹就直接复用，不重复分析同一首歌）。
    """
    from vidscribe.dance import material_ingest as ingest  # noqa: PLC0415
    from vidscribe.dance import material_repository as repo  # noqa: PLC0415

    if str(text).isdigit():
        row = repo.get_song(db, int(text))
        if row is None:
            raise ValueError(f"库里没有目标歌 #{text}")
        return ingest.TargetSong(song_id=int(row["id"]), path=Path(row["file_path"]),
                                 fingerprint=str(row["fingerprint"] or ""),
                                 duration=float(row["duration"] or 0.0),
                                 bpm=float(row["bpm"] or 0.0),
                                 sample_rate=int(row["sample_rate"] or 0))

    path = Path(text)
    if not path.is_absolute():
        for base in (cfg.root, Path(cfg.dance["song_dir"])):
            if (base / path).is_file():
                path = base / path
                break
    if not path.is_file():
        raise ValueError(f"目标歌文件不在盘上：{text}")
    return ingest.register_song(db, path)


def _dance_sources(cfg: Config, values: list[str]) -> list[Path]:
    """`--sources` 可以给目录、也可以给一串文件。目录就按视频后缀扫一遍。"""
    out: list[Path] = []
    for text in values or [cfg.dance["source_dir"]]:
        path = Path(text)
        if not path.is_absolute():
            path = cfg.root / path
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*")
                              if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".flv")))
        elif path.is_file():
            out.append(path)
        else:
            logger.warning("跳过不存在的源：%s", path)
    return out


def cmd_dance_montage(cfg: Config, args: argparse.Namespace) -> int:
    """卡点舞混剪：对齐 / 切片 / 素材 / 推荐 / 混剪 / 统计 / 历史 / 界面。

    这条命令**只操作 dance_* 那批表**，原有功能的表一个都不碰。返回码沿用全局约定：
    0 = 成功，1 = 业务失败（对齐失败、渲染失败），2 = 参数不对。
    """
    from vidscribe.db import open_db  # noqa: PLC0415

    try:  # Windows 控制台默认 GBK，中文报告会花屏
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.action == "gui":
        return cmd_dance_gui(cfg, args)

    cfg.ensure_dance_dirs()
    db = open_db(cfg)
    try:
        return _dance_dispatch(cfg, args, db)
    except ValueError as exc:          # 参数/输入问题：明确报 2，不打整段堆栈
        logger.error("%s", exc)
        return 2
    finally:
        db.close()


def _dance_dispatch(cfg: Config, args: argparse.Namespace, db: Any) -> int:
    from dataclasses import replace  # noqa: PLC0415

    from vidscribe.dance import history, material_ingest as ingest  # noqa: PLC0415
    from vidscribe.dance import material_repository as repo  # noqa: PLC0415
    from vidscribe.dance import material_selection as selection  # noqa: PLC0415
    from vidscribe.dance import (  # noqa: PLC0415
        media_backend, montage_render, music_structure, recommendation, statistics,
        strategy as strategy_mod,
    )
    from vidscribe.dance.types import FilterSpec  # noqa: PLC0415


    action = args.action
    if action == "songs":
        rows = repo.list_songs(db)
        if not rows:
            print("库里还没有目标歌。用 `dance-montage align --song <歌文件>` 登记第一首。")
            return 0
        print(f"{'ID':>4}  {'时长':>8}  {'BPM':>6}  {'素材':>5}  歌名")
        for row in rows:
            count = db.connect().execute(
                "SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ?",
                (int(row["id"]),)).fetchone()[0]
            print(f"{int(row['id']):>4}  {float(row['duration'] or 0):>7.2f}s  "
                  f"{float(row['bpm'] or 0):>6.1f}  {int(count):>5}  {row['title']}")
        return 0

    if not args.song:
        raise ValueError(f"{action} 需要 --song <歌id 或 歌文件路径>")
    song = _dance_song(db, cfg, args.song)
    slice_duration = float(args.slice or cfg.dance["slice_duration"])
    strategy_mod.ensure_presets(db)

    if action == "align":
        sources = _dance_sources(cfg, args.sources)
        if not sources:
            raise ValueError("一个源视频都没找到（--sources 给目录或文件）")
        workers = int(args.workers or cfg.dance["align_workers"])
        outcomes = ingest.align_batch(db, song, sources, workers=workers,
                                      force=bool(args.force),
                                      on_log=lambda line: logger.info("%s", line))
        bad = [o for o in outcomes if not o.ok]
        for outcome in outcomes:
            if outcome.ok:
                align = outcome.alignment
                print(f"[对齐] {outcome.source_path.name}｜偏移 {align.offset:+.3f}s"
                      f"｜置信 {align.confidence:.3f}｜{align.status}"
                      f"｜{'缓存' if outcome.cached else '新算'}")
            else:
                print(f"[对齐] {outcome.source_path.name}｜失败：{outcome.error}")
        print(f"共 {len(outcomes)} 个源，成功 {len(outcomes) - len(bad)}，失败 {len(bad)}")
        return 1 if bad else 0

    if action == "slice":
        sources = _dance_sources(cfg, args.sources)
        canvas = media_backend.Canvas(
            width=int(cfg.dance["canvas_width"]), height=int(cfg.dance["canvas_height"]),
            fps=float(cfg.dance["canvas_fps"]))
        backend = media_backend.resolve(str(args.backend or cfg.dance["media_backend"]))
        outcomes = ingest.align_batch(db, song, sources,
                                      workers=int(args.workers or cfg.dance["align_workers"]),
                                      on_log=lambda line: logger.info("%s", line))
        total = failed = 0
        for outcome in outcomes:
            if not outcome.ok:
                print(f"[切片] {outcome.source_path.name}｜跳过：对齐失败（{outcome.error}）")
                failed += 1
                continue
            result = ingest.slice_and_register(
                db, song, outcome, slice_duration=slice_duration,
                material_dir=cfg.dance_path("material_dir"), canvas=canvas,
                backend=backend, on_log=lambda line: logger.info("%s", line))
            total += len(result.material_ids)
            state = "成功" if result.ok else f"失败：{result.error}"
            print(f"[切片] {outcome.source_path.name}｜入库 {len(result.material_ids)} 条"
                  f"｜跳过 {result.skipped} 个位置｜{state}")
            failed += 0 if result.ok else 1
        print(f"素材库现有 {total} 条新素材（本次），失败 {failed} 个源")
        return 1 if failed else 0

    if action == "materials":
        spec = FilterSpec(target_song_id=song.song_id, limit=int(args.limit or 50))
        if args.preset:
            spec = selection.preset_spec(args.preset, spec)
        if args.position is not None:
            spec = replace(spec, segment_index=int(args.position))
        if args.person:
            spec = replace(spec, persons=tuple(args.person))

        found = selection.find_materials(db, spec)
        print(f"[素材] 目标歌《{song.path.stem}》命中 {len(found)} 条"

              f"{'（方案 ' + args.preset + '）' if args.preset else ''}")
        print(f"{'ID':>5}  {'位置':>4}  {'时间':>13}  {'人物':<10}  "
              f"{'对齐':>5}  {'候选':>4}  {'使用':>4}  {'出片':>4}  状态")
        for m in found:
            print(f"{m.id:>5}  {m.segment_index:>4}  "
                  f"{m.target_start:>6.2f}→{m.target_end:<6.2f}  "
                  f"{(m.person or '(未标注)'):<10}  {m.alignment_confidence:>5.2f}  "
                  f"{m.candidate_count:>4}  {m.use_count:>4}  {m.output_count:>4}  {m.status}")
        return 0

    if action == "recommend":
        if args.verify:
            original, again, same = recommendation.reproduce(db, int(args.verify))
            for line in recommendation.describe(again):
                print(line)
            print(f"[复现] 推荐 #{original.id} 与重跑结果"
                  f"{'一致 ✓' if same else '不一致（库状态可能变了）'}")
            return 0 if same else 1
        positions = ([int(args.position)] if args.position is not None
                     else [p.index for p in music_structure.target_positions(
                         song.duration, slice_duration)])

        plan = strategy_mod.resolve(db, int(args.strategy or 0))
        spec = selection.preset_spec(args.preset) if args.preset else None
        run = recommendation.recommend(db, song.song_id, positions, strategy=plan,
                                       seed=int(args.seed or 0), spec=spec)
        for line in recommendation.describe(run):
            print(line)
        return 0

    if action == "remix":
        canvas = media_backend.Canvas(
            width=int(cfg.dance["canvas_width"]), height=int(cfg.dance["canvas_height"]),
            fps=float(cfg.dance["canvas_fps"]))
        backend = media_backend.resolve(str(args.backend or cfg.dance["media_backend"]))
        manual = None
        if args.manual:
            manual = {int(k): int(v) for k, v in json.loads(
                Path(args.manual).read_text(encoding="utf-8")).items()}
        made = montage_render.remix(
            db, song.song_id, out_dir=(args.out or cfg.dance_path("output_dir")),
            slice_duration=slice_duration, versions=int(args.versions or 0),
            strategy_id=int(args.strategy or 0), seed=int(args.seed or 0),
            canvas=canvas, backend=backend, name=args.name or "",
            recommend_enabled=not args.no_recommend, manual=manual,
            render_video=not args.plan_only,
            spec=selection.preset_spec(args.preset) if args.preset else None,
            on_log=lambda line: print(line))
        bad = [v for v, r in made if not (r.ok or args.plan_only)]
        for version_id, result in made:
            for line in montage_render.describe(result):
                print(line)
            print(f"  版本 #{version_id}")
        return 1 if bad else 0

    if action == "stats":
        overview = statistics.song_overview(db, song.song_id)
        print(f"[统计] 目标歌《{song.path.stem}》{song.duration:.2f}s｜BPM {song.bpm:.1f}")

        print(f"  素材 {overview['materials']['total']} 条"
              f"（可用 {overview['materials']['ready']}，"
              f"从未使用 {overview['materials']['never_used']}，"
              f"从未出片 {overview['materials']['never_output']}）")
        print(f"  来源 {overview['materials']['sources']} 个｜"
              f"人物 {overview['materials']['persons']} 个｜"
              f"覆盖位置 {overview['materials']['positions']} 个｜"
              f"平均对齐置信 {overview['materials']['avg_confidence']:.3f}")
        print(f"  混剪版本：{overview['versions']}")
        print("  各类事件：" + "，".join(f"{k} {v}"
                                       for k, v in history.event_summary(db, song.song_id).items()))
        print(f"{'位置':>4}  {'素材':>4}  {'人物':>4}  {'使用':>4}  {'出片':>4}")
        for row in statistics.position_coverage(db, song.song_id):
            print(f"{row['segment_index']:>4}  {row['materials']:>4}  "
                  f"{row['persons']:>4}  {row['uses']:>4}  {row['outputs']:>4}")
        for row in statistics.person_breakdown(db, song.song_id):
            print(f"  {row['person']:<12} 素材 {row['materials']:>4}"
                  f"｜使用 {row['uses']:>4}｜出片 {row['outputs']:>4}"
                  f"｜覆盖 {row['positions']:>3} 个位置")
        return 0

    if action == "history":
        if args.recount:
            report = history.recount_song(db, song.song_id)
            print(f"[重算] 检查 {report.get('total', 0)} 条素材，"
                  f"修正 {report.get('changed', 0)} 条（计数一律以事件流水为准）")
            return 0
        rows = history.song_events(db, song.song_id, limit=int(args.limit or 50))
        print(f"[历史] 目标歌《{song.path.stem}》最近 {len(rows)} 条事件")

        for row in rows:
            print(f"  {row['created_at']}  {str(row['event']):<14} "
                  f"素材 #{row['material_id']:<5} 位置 {row['segment_index']}")
        for version in repo.versions_for_song(db, song.song_id, limit=int(args.limit or 50)):
            print(f"  版本 #{version['id']} 第 {version['version_index']} 版"
                  f"｜{version['clip_count']} 格｜{float(version['duration'] or 0):.2f}s"
                  f"｜{version['render_status']}｜综合重复率 "
                  f"{float(version['overall_repeat'] or 0):.3f}")
        return 0

    raise ValueError(f"未知动作 {action!r}")


def cmd_dance_gui(cfg: Config, args: argparse.Namespace) -> int:
    """启动 AI_卡点舞 独立界面。和主界面各开各的，互不影响。"""
    _apply_mirror(cfg)
    try:
        from vidscribe.gui.dance_montage import launch  # noqa: PLC0415
    except ImportError as exc:
        logger.error("GUI 依赖缺失（需要 PyQt5）：%s", exc)
        logger.error("安装命令: pip install PyQt5==5.15.11 "
                     "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1
    cfg.ensure_dance_dirs()
    return launch(cfg)


# ------------------------------------------------------------------ 参数解析

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vidscribe", description="本地 AI 视频理解：视觉事件 + 语音时间轴")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.json")
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="检查 Python / GPU / CUDA / 依赖")
    p_check.set_defaults(func=cmd_check)

    p_dl = sub.add_parser("download", help="预下载模型（优先国内镜像：ModelScope -> hf-mirror -> 官方）")
    p_dl.add_argument("--all", action="store_true", help="同时下载降级备用模型")
    p_dl.add_argument("--force", action="store_true", help="忽略本地缓存重新下载")
    p_dl.add_argument("--visual-model", default=None, help="改下载指定视觉模型，如 openbmb/MiniCPM-V-4_5-int4")
    p_dl.add_argument("--backend", default=None, choices=["auto", *BACKENDS], help="视觉后端")
    p_dl.set_defaults(func=cmd_download)

    p_run = sub.add_parser("run", help="处理视频（默认处理 input/ 下全部视频）")
    p_run.add_argument("videos", nargs="*", help="视频文件或目录")
    p_run.add_argument("--force", action="store_true", help="忽略断点缓存，全部重跑")
    p_run.add_argument("--skip-visual", action="store_true")
    p_run.add_argument("--skip-speech", action="store_true")
    p_run.add_argument("--force-speech", action="store_true",
                       help="只重跑语音识别（画面结果有缓存就复用），配合 --skip-visual 使用")
    p_run.add_argument("--limit", type=int, default=0, help="最多处理几个视频")
    p_run.add_argument("--translate", action="store_true",
                       help="分析完顺手翻译（模型还在显存里，省掉单独翻译时约 15s 的加载）")
    p_run.add_argument("--visual-model", default=None,
                       help="覆盖视觉模型，如 openbmb/MiniCPM-V-4_5-int4（可只写 MiniCPM-V-4_5-int4）")
    p_run.add_argument("--backend", default=None, choices=["auto", *BACKENDS],
                       help="视觉后端，默认按模型名自动判断")
    # 两路情绪各自可开可关；不给参数就按 config.json 里的设置走
    p_run.add_argument("--audio-emotion", dest="audio_emotion", action="store_true", default=None,
                       help="开启语音情绪识别（emotion2vec+，要额外加载模型）")
    p_run.add_argument("--no-audio-emotion", dest="audio_emotion", action="store_false",
                       help="关闭语音情绪识别")
    p_run.add_argument("--visual-emotion", dest="visual_emotion", action="store_true", default=None,
                       help="开启画面情绪识别（视觉模型同一次推理顺便判，不额外加载模型）")
    p_run.add_argument("--no-visual-emotion", dest="visual_emotion", action="store_false",
                       help="关闭画面情绪识别")
    # 声纹模型：GUI 的「声纹」下拉透传到这里；不给就按 config.json 走
    p_run.add_argument("--speaker-model", dest="speaker_model", default=None,
                       help="声纹模型：en（英文，默认）/ zh（中文）/ off（不分说话人）/ 完整模型 id")
    p_run.set_defaults(func=cmd_run)

    p_tr = sub.add_parser("translate", help="翻译已有结果（英->中 / 中->英），纯文本，不解码视频")
    p_tr.add_argument("target", nargs="?", default=None,
                      help="输出目录名、输出目录路径，或视频文件路径")
    p_tr.add_argument("--items", default=None,
                      help="只翻译这个 JSON 里给出的文本行（GUI 用：界面上还没译文的那些行）")
    p_tr.add_argument("--result", default=None, help="把翻译结果写到这个 JSON（配合 --items）")
    p_tr.add_argument("--retranslate", action="store_true",
                      help="连已有译文的条目也重新翻译（默认只补没译文的）")
    p_tr.add_argument("--visual-model", default=None, help="指定做翻译的模型，默认用配置里的视觉模型")
    p_tr.add_argument("--backend", default=None, choices=["auto", *BACKENDS], help="视觉后端")
    p_tr.set_defaults(func=cmd_translate)

    p_cache = sub.add_parser("cache", help="查看/清理缓存（work 断点与预览音频、logs 日志）")
    p_cache.add_argument("--clean", action="store_true", help="真的删除过期缓存")
    p_cache.add_argument("--dry-run", action="store_true", help="配合 --clean：只列出要删什么")
    p_cache.add_argument("--days", type=float, default=None,
                         help="多少天没动过算过期，默认取 runtime.cache_max_age_days（3）。"
                              "只影响 --clean 删哪些，不加 --clean 什么都不删")
    p_cache.set_defaults(func=cmd_cache)

    p_db = sub.add_parser("db", help="SQLite 库：建库 / 导入旧缓存 / 对账 / 体检 / 备份恢复 / 查孤儿")
    p_db.add_argument("--init", action="store_true",
                      help="建库/升级到当前结构版本（不给任何参数时也会做，这个只是把版本打出来）")
    p_db.add_argument("--import", dest="do_import", action="store_true",
                      help="扫 cache/、output/、视频库、AI 目录，把已有结果导进库（不删任何文件）")
    p_db.add_argument("--reconcile", action="store_true",
                      help="对账：库里记的视频/文件还在不在盘上，只改状态不删记录")
    p_db.add_argument("--recover", action="store_true",
                      help="恢复：卡住的 AI 任务退回等待，没跑完的分析标失败")
    p_db.add_argument("--fill-duration", action="store_true",
                      help="给老数据补时长：只处理 duration 为空且文件还在盘上的视频，"
                           "复用剪辑那条路的探测逻辑，探不到就保持空")
    p_db.add_argument("--check", action="store_true",
                      help="体检：integrity_check、foreign_key_check、版本、表与索引、能不能写。"
                           "有问题时退出码非 0")
    p_db.add_argument("--stats", action="store_true",
                      help="整库统计（视频/分析/任务/结果/片段/文件/逐词/模型分布），数字全部来自 SQL")
    p_db.add_argument("--backup", nargs="?", const="", default=None, metavar="路径",
                      help="用 SQLite backup API 备份（WAL 一起进去）。不给路径就落 database/backups/")
    p_db.add_argument("--restore", default=None, metavar="备份文件",
                      help="从备份恢复：先验备份、再给当前库留一份安全备份，最后写回并复检")
    p_db.add_argument("--vacuum", action="store_true",
                      help="整理库文件（手动才做）。有任务在跑就拒绝，除非加 --force")
    p_db.add_argument("--orphans", action="store_true",
                      help="只报告不删：表之间对不上的记录、登记了但文件没了的产物、盘上没登记的视频")
    p_db.add_argument("--force", action="store_true",
                      help="配合 --vacuum：明知有任务在跑也要做")
    p_db.set_defaults(func=cmd_db)


    p_hl = sub.add_parser("highlight", help="按 AI JSON 剪高光片段（起剪 / 冻帧 / 收尾三个时间严格照做）")
    p_hl.add_argument("--json", default=None, help="AI JSON 文件路径；不给则从标准输入读")
    p_hl.add_argument("--video", default=None, help="源视频路径，JSON 的 video 字段找不到时用它兜底")
    p_hl.add_argument("--out", default=None,
                      help="输出 MP4 路径，默认放导出目录（gui_settings.json 的 export_dir），"
                           "没设过就放 output/<视频名>/，文件名带成片时长（例 _689 = 6.89 秒）")
    p_hl.add_argument("--start-offset", type=float, default=0.0,
                      help="起剪点 = segments[0].sa + 本值，秒；负数提前起剪")
    p_hl.add_argument("--end-offset", type=float, default=0.0,
                      help="结束点 = segments[0].end + 本值，秒；正数多留一点")
    p_hl.add_argument("--dry-run", action="store_true",
                      help="只跑剪辑引擎算区间并打印中文报告，不渲染、不写文件")
    p_hl.add_argument("--no-engine", action="store_true",
                      help="不修正边界，segments[0].sa / .end 原样照剪（老行为）")


    p_hl.set_defaults(func=cmd_highlight)

    p_mg = sub.add_parser("montage",
                          help="检查第二轮混剪的输出：素材分数 / 同源 / Rank1 闭嘴 / 旁白说不说得完")
    p_mg.add_argument("--pool", required=True, metavar="提取TXT",
                      help="「提取数据」导出的那份 txt（一行一个成品素材），当对照的真值")
    p_mg.add_argument("--json", default=None, metavar="混剪JSON",
                      help="AI 交回来的混剪 JSON；不给则从标准输入读")
    p_mg.add_argument("--scorecard", default=None, metavar="CSV",
                      help="另存一张评分表：机器判定填好，故事/节奏/情绪留空等人打分")
    p_mg.set_defaults(func=cmd_montage)


    p_as = sub.add_parser("assets", help="高光方案：查/导入/复制/编辑/软删/设当前/只用 JSON 剪/成品溯源")
    p_as.add_argument("--video", default=None, help="视频 id 或文件名片段：看这个视频的详情")
    p_as.add_argument("--limit", type=int, default=30, help="总表列多少个视频，默认 30")
    p_as.add_argument("--import-json", dest="import_json", default=None, metavar="JSON",
                      help="把一份现成 JSON 登记成新方案（要配 --video），旧方案一个字不动")
    p_as.add_argument("--extract", action="store_true",
                      help="把这个视频剪出过的成品导成第二轮混剪的输入（要配 --video），"
                           "一个成品一行，带 gaps 和四维 scores；落点用 --out 指定")
    p_as.add_argument("--import-moments", dest="import_moments", default=None, metavar="清单",
                      help="结果清单（一行一个 setup_at/result_at）→ 一行一份方案；"
                           "区间由程序按逐词时间戳算（要配 --video，视频得先分析过）")
    p_as.add_argument("--translate", action="store_true",
                      help="配合 --import-moments：顺手把区间原文译成中文填进 Speech text")
    p_as.add_argument("--copy", default=None, metavar="方案ID", help="复制一份方案（原件不动）")
    p_as.add_argument("--edit", default=None, metavar="方案ID",
                      help="用 --json 的内容改方案；默认另开一条新方案")
    p_as.add_argument("--in-place", dest="in_place", action="store_true",
                      help="配合 --edit：就地改 current_json（raw_json 仍是 AI 原话）")
    p_as.add_argument("--json", default=None, help="配合 --edit：改好的 JSON 文件")
    p_as.add_argument("--delete", default=None, metavar="方案ID", help="软删方案（成品一个都不动）")
    p_as.add_argument("--restore", default=None, metavar="方案ID", help="把软删的方案捞回来")
    p_as.add_argument("--set-current", dest="set_current", default=None, metavar="方案ID",
                      help="设为当前方案（自动剪辑「已有 JSON」就用它）")
    p_as.add_argument("--name", default=None, help="配合 --import-json / --copy / --edit：方案名")
    p_as.add_argument("--note", default=None, help="配合 --import-json：备注")
    p_as.add_argument("--no-current", dest="no_current", action="store_true",
                      help="配合 --import-json：登记但不设为当前方案")
    p_as.add_argument("--by-ai", dest="by_ai", default=None, metavar="provider",
                      help="按 AI 来源查方案，可再加 --model")
    p_as.add_argument("--model", default=None, help="配合 --by-ai：再按模型名过滤")
    p_as.add_argument("--by-prm", dest="by_prm", default=None, metavar="PRM_ID",
                      help="按 PRM 查方案与成品")
    p_as.add_argument("--trace", default=None, metavar="成品ID",
                      help="成品反查：视频 / 分析 / 方案 / AI / 模型 / PRM")
    p_as.add_argument("--render", default=None, metavar="方案ID",
                      help="只用这份 JSON 剪成片，**不调用 AI**（不涉及 PRM）")
    p_as.add_argument("--out", default=None, help="配合 --render：输出 MP4 路径")
    p_as.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="配合 --render：只算区间不渲染")
    p_as.set_defaults(func=cmd_assets)

    p_prm = sub.add_parser("prm", help="PRM 档案：列出 / 新增 / 改 / 软删 / 设默认（内容仍在文件里）")
    p_prm.add_argument("--list", action="store_true", help="列出所有 PRM（默认行为）")
    p_prm.add_argument("--all", action="store_true", help="连软删掉的一起列")
    p_prm.add_argument("--add", default=None, metavar="名字", help="新增一份 PRM，要配 --file")
    p_prm.add_argument("--file", default=None, help="提示词文件路径（相对路径按项目根算）")
    p_prm.add_argument("--edit", default=None, metavar="PRM_ID", help="改某一份的元信息")
    p_prm.add_argument("--name", default=None, help="配合 --edit：改名字")
    p_prm.add_argument("--language", default=None, help="语言标记，比如 en / zh")
    p_prm.add_argument("--version", default=None, help="版本标记，比如 V1 / V2")
    p_prm.add_argument("--description", default=None, help="说明")
    p_prm.add_argument("--default", action="store_true", help="配合 --add：登记完就设为默认")
    p_prm.add_argument("--set-default", dest="set_default", default=None, metavar="PRM_ID",
                       help="把某一份设为默认（GUI 发 AI 时优先用它）")
    p_prm.add_argument("--delete", default=None, metavar="PRM_ID",
                       help="软删（历史成品照旧查得到用的是它）")
    p_prm.set_defaults(func=cmd_prm)

    p_gui = sub.add_parser("gui", help="启动 PyQt5 图形界面（左视频 / 右时间轴 / 底部语音）")
    p_gui.add_argument("video", nargs="?", default=None, help="启动时直接打开的视频")
    p_gui.set_defaults(func=cmd_gui)

    p_ai = sub.add_parser("ai", help="只开 AI 面板（第二主界面）：AI 设置 + 自动剪辑，不显示主界面")
    p_ai.add_argument("--auto", action="store_true",
                      help="开起来直接跑一遍自动剪辑，不用手点")
    p_ai.set_defaults(func=cmd_ai)

    p_dance = sub.add_parser(
        "dance-montage",
        help="AI_卡点舞：目标歌对齐 → 固定位置切片 → 素材库 → 多版本混剪")
    p_dance.add_argument("action", choices=("songs", "align", "slice", "materials",
                                           "recommend", "remix", "stats", "history", "gui"),
                         help="songs 列目标歌｜align 对齐｜slice 切片入库｜materials 查素材｜"
                              "recommend 推荐｜remix 出片｜stats 统计｜history 流水｜gui 开界面")
    p_dance.add_argument("--song", default=None, metavar="ID|文件",
                         help="目标歌：库里的 id，或一个音频/视频文件路径（会自动登记分析）")
    p_dance.add_argument("--sources", nargs="*", default=None, metavar="目录|文件",
                         help="源舞蹈视频，可给目录（递归扫）或一串文件。默认取 dance.source_dir")
    p_dance.add_argument("--slice", type=float, default=None, metavar="秒",
                         help="每格时长，常用 1.0/1.5/2.0/2.5/3.0，默认取 dance.slice_duration")
    p_dance.add_argument("--workers", type=int, default=None,
                         help="对齐并发数（4~6 够了，再多只会互相抢 IO）")
    p_dance.add_argument("--force", action="store_true", help="忽略对齐缓存，全部重算")
    p_dance.add_argument("--backend", default=None, help="媒体后端：auto/pyav/ffmpeg/nvenc")
    p_dance.add_argument("--out", default=None, help="成品目录，默认 dance.output_dir")
    p_dance.add_argument("--versions", type=int, default=None, help="一次出几个版本")
    p_dance.add_argument("--strategy", type=int, default=None, help="策略 id，默认用默认策略")
    p_dance.add_argument("--preset", default=None,
                         help="筛选方案：never_output/never_used/low_use/long_unused/"
                              "high_confidence/by_person/exclude_recent")
    p_dance.add_argument("--seed", type=int, default=None, help="随机种子（落库，可复现）")
    p_dance.add_argument("--position", type=int, default=None, help="只看某一个音乐位置")
    p_dance.add_argument("--person", nargs="*", default=None, help="只看某几个人物")
    p_dance.add_argument("--limit", type=int, default=None, help="列表最多显示几行")
    p_dance.add_argument("--name", default=None, help="这次混剪的名字")
    p_dance.add_argument("--manual", default=None, metavar="JSON",
                         help='纯手动选择：一个 {"位置": 素材id} 的 json 文件，配 --no-recommend')
    p_dance.add_argument("--no-recommend", dest="no_recommend", action="store_true",
                         help="关闭智能推荐，只用 --manual 给的选择")
    p_dance.add_argument("--plan-only", dest="plan_only", action="store_true",
                         help="只出编辑计划不渲染（想先看看这一版长什么样）")
    p_dance.add_argument("--verify", type=int, default=None, metavar="RUN_ID",
                         help="重跑一次历史推荐，验证结果可复现")
    p_dance.add_argument("--recount", action="store_true",
                         help="按事件流水把所有计数重算一遍（history 动作专用）")
    p_dance.set_defaults(func=cmd_dance_montage)
    return parser




def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    cfg = Config.load(_project_root(), args.config)
    cfg.ensure_dirs()
    setup_logging(cfg.path("log_dir"), name=f"{args.command}_{datetime.now():%Y%m%d_%H%M%S}")
    logger.info("命令: %s", args.command)
    try:
        return int(args.func(cfg, args))
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except Exception as exc:
        logger.error("未处理异常: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
