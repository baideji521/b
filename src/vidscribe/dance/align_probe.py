"""对齐验收探针：把一次「一个视频 + 一首歌」的对齐结果摊开成能给人看的行。

这一层**没有任何自己的算法**。它存在的唯一理由是：界面要显示的东西
（每个音乐位置映射到源视频哪一段、能不能用、覆盖率多少、为什么不能用），
必须和真正切片时的判定**逐字一致**，所以全部现算于既有实现：

    位置从哪来   music_structure.target_positions   （只看目标歌，与源无关）
    区间怎么算   material_slice.map_to_source       （source = target - offset）
    能不能用     material_slice.plan_slices         （越界进 skipped，不 clamp）
    覆盖率       material_slice.coverage
    结论/理由    DanceAlignment.status / .notes（audio_align 算好的）

如果哪天切片规则改了，这里显示的东西会跟着一起改 —— 这正是要的。
界面上"看起来能用"而实际切片时被拒，是这个测试台最不能出现的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import material_slice
from .music_structure import target_positions
from .types import DanceAlignment


@dataclass(frozen=True)
class PositionRow:
    """一个固定音乐位置的验收结果。`ok=False` 时 `reason` 一定有话说。"""

    index: int
    target_start: float
    target_end: float
    source_start: float
    source_end: float
    ok: bool
    reason: str = ""

    @property
    def duration(self) -> float:
        return round(self.target_end - self.target_start, 6)

    @property
    def status_text(self) -> str:
        return "可用" if self.ok else "越界"

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "target_start": self.target_start,
                "target_end": self.target_end, "source_start": self.source_start,
                "source_end": self.source_end, "ok": self.ok, "reason": self.reason}


def map_moment(alignment: DanceAlignment, target_time: float) -> float:
    """目标歌时刻 → 源视频时刻。**只有这一个换算入口**，界面不许自己减。"""
    return alignment.source_time(float(target_time))


def probe_one(alignment: DanceAlignment, target_start: float, slice_duration: float,
              source_duration: float = 0.0, *, index: int = 0) -> PositionRow:
    """试一个卡点：目标区间 → 源区间 + 能不能用。

    越界不抛给调用方，翻译成 `ok=False` + 原因 —— 界面要的是"显示为什么不行"，
    而判定本身仍然是 `material_slice.map_to_source` 那一套，一个字都没改。
    """
    start = round(float(target_start), 6)
    end = round(start + float(slice_duration), 6)
    duration = float(source_duration if source_duration > 0 else alignment.source_duration)
    try:
        spec = material_slice.map_to_source(start, end, alignment.offset, duration,
                                            segment_index=int(index))
    except material_slice.SourceRangeError as exc:
        return PositionRow(index=int(index), target_start=start, target_end=end,
                           source_start=alignment.source_time(start),
                           source_end=alignment.source_time(end),
                           ok=False, reason=str(exc))
    return PositionRow(index=int(index), target_start=spec.target_start,
                       target_end=spec.target_end, source_start=spec.source_start,
                       source_end=spec.source_end, ok=True)


def probe_all(alignment: DanceAlignment, *, song_duration: float, source_duration: float,
              slice_duration: float) -> list[PositionRow]:
    """把整首目标歌按固定位置铺开，逐格标可用/越界。

    走的是 `plan_slices`，所以这里显示"可用"的位置，正式切片时一定也切得出来。
    对齐被判 `rejected` 时 `plan_slices` 整份计划为空（一个不可信的 offset 切出来
    全是错位素材），那就把**每一格**都标成不可用并写上同一个理由 —— 不是"没有位置"，
    是"这条源现在一格都不该用"。
    """
    positions = target_positions(song_duration, slice_duration)
    plan = material_slice.plan_slices(
        alignment, song_duration=song_duration, source_duration=source_duration,
        slice_duration=slice_duration)
    whole = [why for index, why in plan.skipped if index < 0]
    if whole:
        return [PositionRow(index=p.index, target_start=p.start, target_end=p.end,
                            source_start=alignment.source_time(p.start),
                            source_end=alignment.source_time(p.end),
                            ok=False, reason=whole[0]) for p in positions]

    reasons = {int(index): why for index, why in plan.skipped}
    good = {int(spec.segment_index): spec for spec in plan.specs}
    rows: list[PositionRow] = []
    for position in positions:
        spec = good.get(position.index)
        if spec is not None:
            rows.append(PositionRow(index=position.index, target_start=spec.target_start,
                                    target_end=spec.target_end,
                                    source_start=spec.source_start,
                                    source_end=spec.source_end, ok=True))
            continue
        rows.append(PositionRow(
            index=position.index, target_start=position.start, target_end=position.end,
            source_start=alignment.source_time(position.start),
            source_end=alignment.source_time(position.end),
            ok=False, reason=reasons.get(position.index, "这个位置没有对应的源区间")))
    return rows


def coverage_of(rows: list[PositionRow]) -> tuple[int, int, float]:
    """`(可用格数, 总格数, 覆盖率 0~1)`。覆盖率按格数算，和格长无关。"""
    total = len(rows)
    usable = sum(1 for row in rows if row.ok)
    ratio = round(usable / float(total), 4) if total else 0.0
    return usable, total, ratio


def diagnose(alignment: DanceAlignment, rows: list[PositionRow] | None = None) -> list[str]:
    """写成中文的诊断行。**只翻译后端已经算出来的东西，不新增判断。**

    行首符号：`✓` 这一项没问题、`⚠` 可用但要留神、`✗` 不能用。
    """
    from . import alignment_validation as validate  # noqa: PLC0415 - 只用几个阈值

    lines: list[str] = []
    support = alignment.chroma_offset is not None
    if support and abs(float(alignment.chroma_offset) - alignment.offset) <= validate.METHOD_TOLERANCE:
        lines.append(f"✓ 波形与 chroma 两路都指向 {alignment.offset:+.3f}s")
    elif support:
        lines.append(f"⚠ chroma 全局峰在 {float(alignment.chroma_offset):+.3f}s，"
                     f"和波形的 {alignment.offset:+.3f}s 不在一处"
                     "（和声周期性重复时会这样，看下面的支持度说明）")
    else:
        lines.append("⚠ chroma 这一路算不出来（音频太短或太安静），只有波形一个证人")

    if alignment.window_count > 1:
        mark = "✓" if alignment.max_deviation <= validate.OFFSET_TOLERANCE else "⚠"
        lines.append(f"{mark} {alignment.window_count} 个验证窗口，一致性 "
                     f"{alignment.agreement:.3f}，最大偏差 {alignment.max_deviation:.3f}s")
    else:
        lines.append("⚠ 只有 1 个验证窗口（源太短），拿不到「多窗口一致」这项证据，"
                     "置信度天然上不了顶")

    if rows:
        usable, total, ratio = coverage_of(rows)
        mark = "✓" if ratio >= 0.6 else "⚠"
        lines.append(f"{mark} 固定位置覆盖 {usable}/{total} 格（{ratio * 100:.2f}%）")

    if alignment.manual:
        lines.append(f"✓ 当前用的是**人工** offset {alignment.offset:+.3f}s"
                     f"（算法原值 {float(alignment.original_offset or 0.0):+.3f}s）："
                     f"{alignment.manual_reason}")

    if alignment.status == "ok":
        lines.append(f"✓ 结论 ok，置信度 {alignment.confidence:.3f}，可以进素材库")
    elif alignment.status == "low_confidence":
        lines.append(f"⚠ 置信度 {alignment.confidence:.3f} 低于 {validate.LOW_CONFIDENCE}，"
                     "建议先播几个卡点亲眼确认。常见原因：源视频里的原始音乐不完整、"
                     "不是同一首歌、或者视频被变速/剪辑过")
    elif alignment.status == "disagree":
        lines.append("✗ 波形与 chroma 两路结论互相矛盾，这条素材不该进库")
    else:
        lines.append(f"✗ 结论 {alignment.status}，不要用它切素材")
    lines.extend(f"　 {note}" for note in alignment.notes)
    return lines


__all__ = ["PositionRow", "map_moment", "probe_one", "probe_all", "coverage_of", "diagnose"]
