"""进度上报：给 GUI 和命令行一个统一的、机器可读的进度来源。

设计取舍：
- 分析在子进程里跑（GUI 不 import torch/cv2），所以进度必须能穿过管道 ->
  往 stdout 打一行 `@@PROGRESS {json}`，GUI 解析这一行，日志里不重复刷屏。
- 各阶段耗时差距很大（视觉远大于语音），所以 overall 用固定权重折算，
  而不是"阶段数 / 总阶段数"那种会卡在某一格的假进度。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

MARKER = "@@PROGRESS "

# json：机器可读一行一条（GUI 用）；text：终端单行刷新进度条；off：不输出
MODE = (os.environ.get("VIDSCRIBE_PROGRESS") or "text").strip().lower()

# 实测量级：视觉 ~445s / 语音 ~37s / 探测 ~2s / timeline ~0s
STAGE_WEIGHTS: dict[str, float] = {
    "probe": 0.04,
    "speech": 0.12,
    "visual": 0.80,
    "timeline": 0.04,
}
STAGE_ORDER = ("probe", "speech", "visual", "timeline")

STAGE_LABELS = {
    "probe": "探测视频/镜头切点",
    "speech": "语音识别",
    "visual": "画面事件分析",
    "timeline": "时间轴合并导出",
}


def _overall(stage: str, fraction: float) -> float:
    done = sum(STAGE_WEIGHTS[s] for s in STAGE_ORDER if STAGE_ORDER.index(s) < STAGE_ORDER.index(stage)) \
        if stage in STAGE_ORDER else 0.0
    weight = STAGE_WEIGHTS.get(stage, 0.0)
    return round(min(1.0, max(0.0, done + weight * max(0.0, min(1.0, fraction)))), 4)


_last_text: dict[str, Any] = {"stage": None, "percent": -1}


def _emit_text(payload: dict[str, Any]) -> None:
    """终端进度：tty 用 \\r 单行刷新；被重定向时降级为按 5% 打点，避免刷屏。"""
    percent = payload["overall"] * 100.0
    stage = payload["stage"]
    interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
    if not interactive:
        step = int(percent // 5)
        if stage == _last_text["stage"] and step == _last_text["percent"]:
            return
        _last_text["stage"], _last_text["percent"] = stage, step
        sys.stderr.write(f"[进度 {percent:5.1f}%] {payload['stage_label']}｜{payload['detail']}\n")
        sys.stderr.flush()
        return
    filled = int(round(percent / 100.0 * 28))
    bar = "#" * filled + "-" * (28 - filled)
    line = f"\r[{bar}] {percent:5.1f}%  {payload['stage_label']}｜{payload['detail']}"
    sys.stderr.write(line[:160].ljust(120))
    if percent >= 99.999:
        sys.stderr.write("\n")
    sys.stderr.flush()


def report(stage: str, fraction: float, detail: str = "", *, video: str | None = None,
           done: float | None = None, total: float | None = None) -> None:
    """上报某个阶段的进度。fraction 是该阶段内部的 0~1 完成度。"""
    if MODE == "off":
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "fraction": round(max(0.0, min(1.0, float(fraction))), 4),
        "overall": _overall(stage, fraction),
        "detail": detail,
    }
    if video:
        payload["video"] = video
    if done is not None:
        payload["done"] = done
    if total is not None:
        payload["total"] = total
    if MODE == "json":
        sys.stdout.write(MARKER + json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    else:
        _emit_text(payload)


def parse(line: str) -> dict[str, Any] | None:
    """GUI 侧解析；不是进度行就返回 None。"""
    if not line.startswith(MARKER):
        return None
    try:
        return json.loads(line[len(MARKER):])
    except Exception:
        return None
