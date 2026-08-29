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


# ------------------------------------------------------------------ 高光剪辑
def cmd_highlight(cfg: Config, args: argparse.Namespace) -> int:
    """按 AI JSON 剪高光：clip.start 起剪，clip.end 冻帧，片尾由 --text-offset 决定。"""

    from vidscribe.highlight import default_target, parse_spec, render_highlight, resolve_video  # noqa: PLC0415

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
        spec = parse_spec(json.loads(raw))
        spec = spec.shifted(args.start_offset, args.end_offset, args.text_offset)
    except json.JSONDecodeError as exc:
        logger.error("JSON 解析失败: %s", exc)
        return 2
    except ValueError as exc:
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

    if args.out:
        target = Path(args.out)
        if not target.is_absolute():
            target = cfg.root / target
    else:
        target = default_target(_export_dir(cfg, video), video)

    try:
        result = render_highlight(video, spec, target, on_log=lambda line: print(line, flush=True))
    except Exception as exc:
        logger.error("剪辑失败: %s", exc)
        logger.debug(traceback.format_exc())
        return 1
    write_json(target.with_suffix(".json"),
               {"spec": spec.raw,
                "offsets": {"start": args.start_offset, "end": args.end_offset,
                            "text": args.text_offset},
                "result": result})

    logger.info("高光片段已生成: %s", target)
    return 0


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

    p_hl = sub.add_parser("highlight", help="按 AI JSON 剪高光片段（起剪 / 冻帧 / 收尾三个时间严格照做）")
    p_hl.add_argument("--json", default=None, help="AI JSON 文件路径；不给则从标准输入读")
    p_hl.add_argument("--video", default=None, help="源视频路径，JSON 的 video 字段找不到时用它兜底")
    p_hl.add_argument("--out", default=None,
                      help="输出 MP4 路径，默认放导出目录（gui_settings.json 的 export_dir），"
                           "没设过就放 output/<视频名>/，文件名带 _高光时刻")
    p_hl.add_argument("--start-offset", type=float, default=0.0,
                      help="起剪点 = clip.start + 本值，秒；负数提前起剪")
    p_hl.add_argument("--end-offset", type=float, default=0.0,
                      help="冻帧点 = clip.end + 本值，秒；正数晚一点冻结")
    p_hl.add_argument("--text-offset", type=float, default=0.0,
                      help="冻帧+字幕这段的时长，秒；0 只留一帧，不能为负")

    p_hl.set_defaults(func=cmd_highlight)

    p_gui = sub.add_parser("gui", help="启动 PyQt5 图形界面（左视频 / 右时间轴 / 底部语音）")
    p_gui.add_argument("video", nargs="?", default=None, help="启动时直接打开的视频")
    p_gui.set_defaults(func=cmd_gui)

    p_ai = sub.add_parser("ai", help="只开 AI 面板（第二主界面）：AI 设置 + 自动剪辑，不显示主界面")
    p_ai.add_argument("--auto", action="store_true",
                      help="开起来直接跑一遍自动剪辑，不用手点")
    p_ai.set_defaults(func=cmd_ai)
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
