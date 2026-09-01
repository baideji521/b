"""按片尾红屏把合并视频切回一段段素材。

每个高光成品的最后都固定接了 1 秒纯红背景（见 highlight/clip.py 的 RED_TAIL_SECONDS）。
把这些成品拼成一条长视频之后，那几段红屏就是天然的分界线：

    素材 1 ── 🔴 ── 素材 2 ── 🔴 ── 素材 3 ── 🔴

这里只做一件事：找出红屏区间，反推出每段素材的 [起, 止]。红屏本身不参与任何识别，
也不算进任何一段素材里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..logging_setup import get_logger

logger = get_logger("timeline")

LogFn = Callable[[str], None]

#: 每隔多少秒验一帧。1 秒红屏在这个步长下有 10 个采样点，丢一两个也判得出来
SAMPLE_STEP_SECONDS = 0.1
#: 连续红屏至少这么长才算分界线。画面里偶然出现的红色物体撑不到这么久的"整屏纯红"
MIN_RED_RUN_SECONDS = 0.4
#: 一段素材至少这么长才算一段，滤掉红屏之间的零碎（比如连着两段红屏）
MIN_SEGMENT_SECONDS = 0.5
#: 判"纯红"的阈值：红通道够亮、绿蓝够暗，而且整屏颜色一致（标准差小）
RED_MIN_R = 200.0
RED_MAX_GB = 60.0
RED_MAX_STD = 24.0
#: 判色前先缩到这个尺寸，省得对全分辨率做统计
THUMB = 32


def is_red_frame(bgr: np.ndarray) -> bool:
    """这一帧是不是整屏纯红。bgr 是 cv2 读出来的原始帧。"""
    if bgr is None or bgr.size == 0:
        return False
    small = cv2.resize(bgr, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    blue, green, red = small[..., 0], small[..., 1], small[..., 2]
    if red.mean() < RED_MIN_R or green.mean() > RED_MAX_GB or blue.mean() > RED_MAX_GB:
        return False
    # 只看均值会把"红色物体占满大半屏"也算进来，再要求整屏颜色一致
    return bool(red.std() < RED_MAX_STD and green.std() < RED_MAX_STD
                and blue.std() < RED_MAX_STD)


def find_red_runs(video: Path, on_log: LogFn | None = None) -> list[tuple[float, float]]:
    """扫一遍视频，返回每段纯红画面的 [起, 止]（秒）。

    只解码采样帧：中间那些用 grab() 跳过，不做色彩转换，长视频也能扫得动。
    """
    log = on_log or (lambda line: logger.info("%s", line))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 打不开视频：{video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"读不到有效帧率：{video}")
        stride = max(1, int(round(fps * SAMPLE_STEP_SECONDS)))
        step = stride / fps                     # 实际采样间隔，比 0.1 更准
        runs: list[tuple[float, float]] = []
        run_start: float | None = None
        last_red: float = 0.0
        index = 0
        while True:
            if not cap.grab():
                break
            if index % stride == 0:
                ok, frame = cap.retrieve()
                moment = index / fps
                if ok and is_red_frame(frame):
                    if run_start is None:
                        run_start = moment
                    last_red = moment
                elif run_start is not None:
                    runs.append((run_start, last_red + step))
                    run_start = None
            index += 1
        if run_start is not None:
            runs.append((run_start, last_red + step))
    finally:
        cap.release()

    kept = [(round(a, 3), round(b, 3)) for a, b in runs if b - a >= MIN_RED_RUN_SECONDS]
    dropped = len(runs) - len(kept)
    log(f"[红屏] 采样步长 {step:.3f}s，找到 {len(kept)} 段红屏"
        + (f"（另有 {dropped} 段太短，按误判丢掉）" if dropped else ""))
    for order, (begin, finish) in enumerate(kept, start=1):
        log(f"[红屏] 第 {order} 段分界：{begin:.2f} → {finish:.2f}（{finish - begin:.2f}s）")
    return kept


def split_by_red(duration: float, runs: list[tuple[float, float]],
                 on_log: LogFn | None = None) -> list[tuple[float, float]]:
    """由红屏区间反推每段素材的 [起, 止]。红屏自己不算进任何一段。"""
    log = on_log or (lambda line: logger.info("%s", line))
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, finish in runs:
        if begin - cursor >= MIN_SEGMENT_SECONDS:
            spans.append((round(cursor, 3), round(begin, 3)))
        cursor = max(cursor, finish)
    if duration - cursor >= MIN_SEGMENT_SECONDS:
        # 最后一段没红屏收尾：可能是合并时漏掉了，也可能故意不加，照样当一段
        spans.append((round(cursor, 3), round(duration, 3)))
        if runs:
            log(f"[红屏] 末尾 {duration - cursor:.2f}s 后面没有红屏，"
                f"当成最后一段素材")
    log(f"[红屏] 切出 {len(spans)} 段素材")
    for order, (begin, finish) in enumerate(spans, start=1):
        log(f"[素材 {order}] {begin:.2f} → {finish:.2f}（{finish - begin:.2f}s）")
    return spans


def segments_of(video: Path, duration: float,
                on_log: LogFn | None = None) -> list[tuple[float, float]]:
    """一步到位：扫红屏 + 切段。给界面用的入口。"""
    return split_by_red(duration, find_red_runs(video, on_log), on_log)
