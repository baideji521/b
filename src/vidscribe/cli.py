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
from vidscribe.timeline.exporters import fmt_time  # noqa: E402
from vidscribe.video_io import list_videos  # noqa: E402

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


# ------------------------------------------------------------------ 主流程
def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    _apply_mirror(cfg)
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
            f"  视觉模型:   {vm.get('model_id')}  帧来源={vm.get('frame_source')}  窗口={vm.get('windows')}  分析帧数={vm.get('analyzed_frames')}  降级次数={vm.get('degrade_attempts')}",
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
    p_dl.set_defaults(func=cmd_download)

    p_run = sub.add_parser("run", help="处理视频（默认处理 input/ 下全部视频）")
    p_run.add_argument("videos", nargs="*", help="视频文件或目录")
    p_run.add_argument("--force", action="store_true", help="忽略断点缓存，全部重跑")
    p_run.add_argument("--skip-visual", action="store_true")
    p_run.add_argument("--skip-speech", action="store_true")
    p_run.add_argument("--limit", type=int, default=0, help="最多处理几个视频")
    p_run.set_defaults(func=cmd_run)
    p_gui = sub.add_parser("gui", help="启动 PyQt5 图形界面（左视频 / 右时间轴 / 底部语音）")
    p_gui.add_argument("video", nargs="?", default=None, help="启动时直接打开的视频")
    p_gui.set_defaults(func=cmd_gui)
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
