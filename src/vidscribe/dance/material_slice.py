"""固定音乐位置切片：把对齐好的源舞蹈视频按**目标歌的位置**切成素材。

这是本项目与普通 BeatSync 最大的区别（技术指导第七节）。普通做法是"分析这段动作
适合哪个 beat"，这里**根本不做那个判断** —— 源舞蹈视频本身已经和它的原音乐同步了，
一旦求出 source→target 的 offset，按目标歌位置切出来的片段就**天然**绑定到该音乐位置：

    source_time = target_time - offset

    例：target 10.0 → 12.0，offset 3.25  ⇒  source 6.75 → 8.75

所以这一层只有两件事：算区间、把区间渲成文件。**不推荐、不评分、不选择。**

一条铁律：源区间越界时**报错或明确跳过**，绝不静默 clamp。
clamp 会让素材内容和音乐位置错开半秒，而"素材已经绑定到该音乐位置"是整个系统的地基；
地基一旦被悄悄改了，后面所有推荐、组合、重复率都在一堆错位素材上计算，
而且事后完全查不出来。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Sequence

from ..logging_setup import get_logger
from . import MATERIAL_GENERATION_VERSION
from . import frame_effects
from .music_structure import target_positions
from .types import DanceAlignment, SliceSpec, SlicePlan

logger = get_logger("dance.slice")

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]

#: 渲染中的临时后缀。和 highlight/clip.py 同一个约定：故意不是视频后缀，
#: 崩溃留下的残片进不了成品扫描
PART_SUFFIX = ".part"
#: 允许的浮点误差（秒）。源区间刚好贴着片尾时不该因为 1e-9 被判越界
EPS = 1e-6
#: 首尾补边界帧最多能占一格的多大比例。补得比真画面还多的"素材"不是素材，
#: 是一张会动半下的照片 —— 余量填得再大也不许越过这条线
MAX_PAD_SHARE = 0.5


class SourceRangeError(ValueError):
    """源区间越界。**不 clamp、不静默** —— 见模块开头那条铁律。"""


def map_to_source(target_start: float, target_end: float, offset: float,
                  source_duration: float, *, segment_index: int = 0,
                  head_room: float = 0.0, tail_room: float = 0.0) -> SliceSpec:
    """单个音乐位置 → 源区间。越界直接抛 `SourceRangeError`。

    这是**严格版**：给单个位置用，越界就是错误。批量切片走 `plan_slices`，
    那边把越界记进 `skipped` 并附原因（也是显式的，只是不中断整批）。

    `head_room` / `tail_room`（秒，默认 0 = 老行为）是**首尾允许缺多少**：
    很多录屏素材开头少一点、结尾早结束一点，于是首段映射到源的负数、
    尾段超过源时长，整段就被判"没有素材"。给了余量之后，这两头改成
    **源区间取交集 + 缺的那一截记进 `head_pad` / `tail_pad`**，渲染时用边界帧补足：

        S1 目标 0→12，offset +1 → 源 −1→11
        head_room=2 ⇒ 源 0→11、head_pad=1、目标仍然是 0→12

    **目标区间不动、时长不变**：素材文件精确等于段落长度（开头那 1 秒是静帧），
    所以成片不会因为某一格短一截而整体前移，音乐一点都不漂。
    几何关系照旧精确：`source_start = target_start + head_pad − offset`。
    这**不是** clamp 源区间那种错位裁剪（那会让画面和音乐错开，是模块开头明令禁止的）。
    """
    start = round(float(target_start) - float(offset), 6)
    end = round(float(target_end) - float(offset), 6)
    head_pad = tail_pad = 0.0
    if end <= start:
        raise SourceRangeError(f"位置 #{segment_index} 的区间非法：{start} → {end}")
    room_head = max(0.0, float(head_room))
    room_tail = max(0.0, float(tail_room))
    if start < -EPS:
        missing = -start
        if missing > room_head + EPS:
            raise SourceRangeError(
                f"位置 #{segment_index}（目标 {target_start:.3f}→{target_end:.3f}）"
                f"映射到源 {start:.3f}s，早于源视频开头 {missing:.3f}s —— 不做 clamp，"
                f"这一段没有对应素材（首段余量 {room_head:.3f}s 不够）")
        head_pad = round(missing, 6)             # 这一截没画面，渲染时补边界帧
        start = 0.0
    duration = float(source_duration)
    if duration > 0 and end > duration + EPS:
        missing = end - duration
        if missing > room_tail + EPS:
            raise SourceRangeError(
                f"位置 #{segment_index}（目标 {target_start:.3f}→{target_end:.3f}）"
                f"映射到源 {start:.3f}→{end:.3f}s，超过源时长 {duration:.3f}s "
                f"{missing:.3f}s —— 不做 clamp，这一段没有对应素材"
                f"（尾段余量 {room_tail:.3f}s 不够）")
        tail_pad = round(missing, 6)
        end = round(duration, 6)
    if end - start <= EPS:
        raise SourceRangeError(
            f"位置 #{segment_index}：夹到源视频范围内之后一帧画面都不剩")
    span = round(float(target_end) - float(target_start), 6)
    if head_pad + tail_pad > span * MAX_PAD_SHARE + EPS:
        raise SourceRangeError(
            f"位置 #{segment_index}（目标 {target_start:.3f}→{target_end:.3f}）"
            f"只有 {end - start:.3f}s 真画面，要补 {head_pad + tail_pad:.3f}s 静帧，"
            f"超过这一格的一半 —— 这种素材没有意义，不给")
    return SliceSpec(segment_index=segment_index,
                     target_start=round(float(target_start), 6),
                     target_end=round(float(target_end), 6),
                     source_start=max(0.0, start), source_end=end,
                     head_pad=head_pad, tail_pad=tail_pad)


def plan_slices(alignment: DanceAlignment, *, song_duration: float, source_duration: float,
                slice_duration: float = 0.0, positions=None, source_video_id: int = 0,
                target_song_id: int = 0, head_room: float = 0.0, tail_room: float = 0.0,
                generation_version: str = MATERIAL_GENERATION_VERSION) -> SlicePlan:
    """一个源视频对一首目标歌的完整切片计划。

    位置从哪来，两条路，**只有这一处分叉**：

        positions 给了     → 按用户拍板的段落模板切（`segment_template.positions_of`），
                             每段长度可以各不相同
        positions 没给     → 按 `slice_duration` 等间隔切（老行为，一个字没改）

    每个位置逐个映射；映射不出来的**记进 `skipped` 并写明原因**，不静默丢、也不 clamp。
    GUI 会把 skipped 显示出来，用户一看就知道"这条素材只覆盖了歌的中段"。

    对齐结论是 `rejected` 时整份计划为空并给出原因 —— 一个不可信的 offset
    切出来的素材全是错位的，宁可一条都不出。
    """
    picks = tuple(positions) if positions else ()
    if not picks:
        if float(slice_duration) <= 0:
            raise ValueError(f"slice_duration 必须为正，收到 {slice_duration}")
        picks = target_positions(song_duration, slice_duration)
    plan_kwargs = {
        "source_video_id": source_video_id,
        "target_song_id": target_song_id,
        "alignment_offset": round(float(alignment.offset), 6),
        # 按模板切的时候各段长度不一，这里记 0 表示"格长不适用，看每段自己的区间"
        "slice_duration": round(float(slice_duration), 6),
        "generation_version": generation_version,
    }
    if alignment.status == "rejected":
        return SlicePlan(**plan_kwargs,
                         skipped=((-1, f"对齐被判 rejected，不切片：{'；'.join(alignment.notes)}"),))

    specs: list[SliceSpec] = []
    skipped: list[tuple[int, str]] = []
    for position in picks:
        try:
            specs.append(map_to_source(position.start, position.end, alignment.offset,
                                       source_duration, segment_index=position.index,
                                       head_room=head_room, tail_room=tail_room))
        except SourceRangeError as exc:
            skipped.append((position.index, str(exc)))
    logger.info("切片计划：offset %.3fs｜%d 个位置可切、%d 个跳过（源 %.2fs / 歌 %.2fs）",
                alignment.offset, len(specs), len(skipped), source_duration, song_duration)
    return SlicePlan(**plan_kwargs, specs=tuple(specs), skipped=tuple(skipped))


def coverage(plan: SlicePlan, song_duration: float) -> float:
    """这份计划覆盖了目标歌的多大比例 0~1。界面上用来一眼看出素材够不够铺满全曲。"""
    if float(song_duration) <= 0:
        return 0.0
    covered = sum(spec.duration for spec in plan.specs)
    return round(min(1.0, covered / float(song_duration)), 4)


# ================================================================== 渲染落地
def material_filename(source_stem: str, song_id: int, spec: SliceSpec,
                      generation_version: str) -> str:
    """素材文件名：`<源名>_s<歌id>_p<位置>_<切片版本>.mp4`。

    位置编号放进文件名是有意的：在资源管理器里按名字排序，同一个位置的所有候选就挨在一起，
    人工挑素材时肉眼一扫就能比。
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_stem)[:40]
    version = generation_version.replace(".", "").replace("/", "") or "v"
    return f"{safe}_s{int(song_id)}_p{spec.segment_index:03d}_{version}.mp4"


def render_material(source: str | Path, spec: SliceSpec, target: Path, *,
                    canvas=None, backend=None, stutters=None,
                    on_log: LogFn | None = None) -> dict:
    """把一条切片渲成**无声**素材文件。

    素材一律无声：目标歌是成片唯一正式音轨（技术指导第十六节），源舞蹈视频的原声
    在这一步就丢掉。留着它只会在混剪时诱惑人"混一点原声进去"，那是明确禁止的。

    `spec.head_pad` / `tail_pad` 不为 0 时（源视频那一头不够长），首尾用**边界帧**
    补足到段落的精确时长 —— 时长准了成片才不会整体前移，代价是那一截是静帧。

    `stutters` 是主音频上打的卡帧抖动点（**绝对秒数**）。这里裁到本段之内、再平移成
    素材内时间：一个点只影响它所在那一段，不许溢到隔壁段（和"不许跨段取素材"同一条铁律）。
    抖动只改帧序，`spec.duration` 一分一秒都不变。

    渲染走 `media_backend`，业务层不碰编码器细节，也不允许直接 subprocess ffmpeg。
    """
    from . import media_backend  # noqa: PLC0415 - 这里才会碰 cv2/av，GUI 主线程不该导入

    board = canvas if canvas is not None else media_backend.Canvas()
    engine = backend if backend is not None else media_backend.resolve("auto")
    local = [point.shifted(-float(spec.target_start)) for point
             in frame_effects.clip_to(stutters, spec.target_start, spec.target_end)]
    return engine.extract_clip(source, spec.source_start, spec.source_end, Path(target),
                               board, pad_head=spec.head_pad, pad_tail=spec.tail_pad,
                               stutters=local, on_log=on_log)



def render_plan(source: str | Path, plan: SlicePlan, out_dir: Path, *, canvas=None,
                backend=None, on_log: LogFn | None = None,
                on_progress: ProgressFn | None = None,
                skip_existing: bool = True) -> list[tuple[SliceSpec, Path]]:
    """把整份计划渲成一批素材文件，返回 `[(spec, 落地路径)]`。

    `skip_existing` 默认开：同名文件已经在盘上而且封装完整就跳过 ——
    素材是长期资产，重跑一遍不该把几百个文件重编一次。
    封装不完整的残留文件会被重渲（不完整的素材进了库比没有更糟）。

    单条渲染失败**不中断整批**：记一行日志继续下一条。几十个源视频批量切片时，
    一条坏文件不该让整晚的活白跑。
    """
    from . import media_backend  # noqa: PLC0415

    log = on_log or (lambda line: logger.info("%s", line))
    report = on_progress or (lambda done, total, stage: None)
    board = canvas if canvas is not None else media_backend.Canvas()
    engine = backend if backend is not None else media_backend.resolve("auto")
    stem = Path(source).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done: list[tuple[SliceSpec, Path]] = []
    total = len(plan.specs)
    for index, spec in enumerate(plan.specs, 1):
        name = material_filename(stem, plan.target_song_id, spec, plan.generation_version)
        target = out_dir / name
        if skip_existing and media_backend.is_complete_video(target):
            done.append((spec, target))
            report(index, total, "复用已有素材")
            continue
        try:
            render_material(source, spec, target, canvas=board, backend=engine, on_log=log)
        except Exception as exc:  # noqa: BLE001 - 一条坏素材不该毁掉整批
            log(f"[跳过] 位置 #{spec.segment_index} 渲染失败：{type(exc).__name__}: {exc}")
            report(index, total, "渲染失败")
            continue
        if not media_backend.is_complete_video(target):
            log(f"[跳过] 位置 #{spec.segment_index} 封装不完整，不登记")
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                # 删不掉残片不影响正确性（下一轮 is_complete_video 还会判它不完整并重渲），
                # 但必须留痕：静默 pass 会让"磁盘满/文件被占用"这类真问题查不出来
                logger.debug("删不掉不完整的素材残片 %s：%s", target, exc)
            continue

        done.append((spec, target))
        report(index, total, "渲染素材")
    log(f"[素材] {Path(source).name}：计划 {total} 条，落地 {len(done)} 条，"
        f"跳过 {len(plan.skipped)} 个位置")
    return done


__all__ = [
    "PART_SUFFIX", "SourceRangeError",
    "map_to_source", "plan_slices", "coverage",
    "material_filename", "render_material", "render_plan",
]


