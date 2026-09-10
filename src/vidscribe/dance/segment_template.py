"""段落模板：用户拍板的 S1/S2/S3… 一串首尾相接的段落，所有源视频共用同一份。

和「固定音乐位置」的分工必须分清楚，这是整个功能的地基：

    人声停顿 / 拍点 = 系统算出来的**参考点**（vocal_activity / music_structure）
            ↓ 只在时间轴上标出来
    用户点击、拖动边界
            ↓
    SegmentSpan = **用户最终确定**的段落边界（这里）
            ↓ 所有源视频严格继承，不因某个视频的人声不同而重新分段
    切片 / 素材 / 混剪都按这份模板对齐

所以这个模块**只做编辑运算**（生成、切开、合并、拖边界、吸附、校验），
不分析音频、不碰数据库、不做推荐。每个操作都返回一份新的 `SegmentTemplate`
（`frozen=True`），撤销就是把上一份拿回来。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from ..logging_setup import get_logger
from .types import SegmentSpan, SegmentTemplate, TargetPosition, VocalPause

logger = get_logger("dance.segment")

#: 模板结构版本。改了字段含义就改它，落库时一起存
TEMPLATE_VERSION = "segment-v1"

#: 一段至少这么长（秒）。比这还短的段落切出来没法看，也没法卡点
MIN_SEGMENT_SECONDS = 0.5
#: 吸附半径（秒）：拖动边界时离参考点这么近就贴上去
SNAP_SECONDS = 0.12


class SegmentError(ValueError):
    """段落模板非法：有缝、重叠、越界、或某段短于 `MIN_SEGMENT_SECONDS`。

    刻意用异常而不是静默修正 —— 静默 clamp 会让用户以为自己拖到了那儿，
    实际上边界在别处，后面切出来的素材全是错的。
    """


# ==================================================================== 生成
def _build(duration: float, boundaries: Sequence[float], *, source: str,
           target_song_id: int = 0, labels: Sequence[str] = (),
           name: str = "", note: str = "") -> SegmentTemplate:
    """内部构造：`boundaries` 是内部分割点（不含 0 和结尾），排序去重后建段。"""
    total = round(float(duration), 6)
    if total <= 0:
        raise SegmentError("歌曲时长未知，没法分段")
    marks = [0.0]
    for value in sorted(round(float(v), 6) for v in boundaries):
        if value <= 0.0 or value >= total:
            raise SegmentError(f"分割点 {value:.3f}s 不在 0~{total:.3f}s 里面")
        if marks and abs(value - marks[-1]) < 1e-6:
            continue                      # 完全重合的点算同一个，不报错
        marks.append(value)
    marks.append(total)

    spans: list[SegmentSpan] = []
    for i in range(len(marks) - 1):
        label = labels[i] if i < len(labels) else ""
        spans.append(SegmentSpan(index=i, start=marks[i], end=marks[i + 1], label=label))
    template = SegmentTemplate(target_song_id=target_song_id, duration=total,
                               spans=tuple(spans), source=source, name=name, note=note,
                               version=TEMPLATE_VERSION)
    validate(template)
    return template


def uniform(duration: float, segment_seconds: float, *, target_song_id: int = 0,
            name: str = "") -> SegmentTemplate:
    """等间隔起步模板 —— 和 `music_structure.target_positions` 同一把尺子。

    末尾不足一格时**并进最后一段**（而不是像 `target_positions` 那样丢掉）：
    模板必须铺满整首歌，否则最后几秒没有任何段落负责，混剪那里就是空的。
    """
    total = round(float(duration), 6)
    step = round(float(segment_seconds), 6)
    if step < MIN_SEGMENT_SECONDS:
        raise SegmentError(f"段落长度至少 {MIN_SEGMENT_SECONDS}s，给的是 {step}s")
    count = max(1, int(total // step))
    boundaries = [round(i * step, 6) for i in range(1, count)]
    if total - count * step >= MIN_SEGMENT_SECONDS:   # 余下的够长就单独成段
        boundaries.append(round(count * step, 6))
    return _build(total, boundaries, source="uniform",
                  target_song_id=target_song_id, name=name)


def from_pauses(duration: float, pauses: Iterable[VocalPause], *,
                min_score: float = 0.0, target_song_id: int = 0,
                name: str = "") -> SegmentTemplate:
    """照人声停顿生成一份起步模板 —— **必须由用户点按钮触发**。

    系统绝不自动这么干（模块开头那条分工）：这只是把「按停顿分」做成一键起点，
    生成之后用户照样可以拖、可以切、可以合。
    停顿正中间作为分割点，因为那里离两侧人声最远，切过去最不容易切到字。
    """
    marks: list[float] = []
    for pause in pauses:
        if pause.score < float(min_score):
            continue
        moment = round(float(pause.middle), 6)
        if marks and moment - marks[-1] < MIN_SEGMENT_SECONDS:
            continue                      # 两个停顿挨太近，只留前一个
        if moment < MIN_SEGMENT_SECONDS or duration - moment < MIN_SEGMENT_SECONDS:
            continue
        marks.append(moment)
    if not marks:
        raise SegmentError("没有够用的人声停顿，先调低推荐度门槛或者手动分段")
    return _build(duration, marks, source="pause",
                  target_song_id=target_song_id, name=name)


def from_boundaries(duration: float, boundaries: Sequence[float], *,
                    target_song_id: int = 0, labels: Sequence[str] = (),
                    name: str = "", note: str = "") -> SegmentTemplate:
    """从一串分割点重建模板（读库、撤销、导入都走它），`source` 记为 manual。"""
    return _build(duration, boundaries, source="manual", target_song_id=target_song_id,
                  labels=labels, name=name, note=note)


# ==================================================================== 编辑
def _rebuilt(template: SegmentTemplate, boundaries: Sequence[float], *,
             labels: Sequence[str] = ()) -> SegmentTemplate:
    return _build(template.duration, boundaries, source="manual",
                  target_song_id=template.target_song_id,
                  labels=labels or [s.label for s in template.spans],
                  name=template.name, note=template.note)


def split_at(template: SegmentTemplate, moment: float) -> SegmentTemplate:
    """在 `moment` 处切一刀。切点离已有边界太近（撑不出一段）就报错。"""
    point = round(float(moment), 6)
    marks = list(template.boundaries)
    if any(abs(point - b) < 1e-6 for b in marks):
        raise SegmentError(f"{point:.3f}s 已经是分割点了")
    span = template.span_at(point)
    if span is None:
        raise SegmentError(f"{point:.3f}s 不在这首歌里面")
    if point - span.start < MIN_SEGMENT_SECONDS or span.end - point < MIN_SEGMENT_SECONDS:
        raise SegmentError(f"离 {span.name} 的边界太近，切出来会短于 {MIN_SEGMENT_SECONDS}s")
    marks.append(point)
    # 标签不再对得上（多了一段），干脆全清掉让它回落成 S1/S2…
    return _rebuilt(template, marks, labels=[])


def merge_at(template: SegmentTemplate, index: int) -> SegmentTemplate:
    """把第 `index` 段并进它前面那段（等于删掉它左边那个分割点）。"""
    if index <= 0 or index >= len(template.spans):
        raise SegmentError(f"第 {index} 段没法往前并（第一段前面没有东西）")
    marks = [b for i, b in enumerate(template.boundaries) if i != index - 1]
    return _rebuilt(template, marks, labels=[])


def move_boundary(template: SegmentTemplate, index: int, moment: float) -> SegmentTemplate:
    """拖动第 `index` 个内部分割点（0 起算）到 `moment`。

    左右都必须还留得下 `MIN_SEGMENT_SECONDS`，否则报错 —— 不 clamp，
    界面上该表现成"拖不过去"，而不是悄悄停在某个位置。
    """
    marks = list(template.boundaries)
    if not 0 <= index < len(marks):
        raise SegmentError(f"没有第 {index} 个分割点")
    point = round(float(moment), 6)
    low = marks[index - 1] if index > 0 else 0.0
    high = marks[index + 1] if index + 1 < len(marks) else template.duration
    if point - low < MIN_SEGMENT_SECONDS or high - point < MIN_SEGMENT_SECONDS:
        raise SegmentError(f"拖不过去：两边都得留 {MIN_SEGMENT_SECONDS}s")
    marks[index] = point
    return _rebuilt(template, marks)


def rename(template: SegmentTemplate, index: int, label: str) -> SegmentTemplate:
    """给某一段起名字（"副歌"、"换人这里"）。空字符串就回落成 S1/S2…"""
    if not 0 <= index < len(template.spans):
        raise SegmentError(f"没有第 {index} 段")
    spans = list(template.spans)
    spans[index] = SegmentSpan(index=index, start=spans[index].start, end=spans[index].end,
                               label=str(label).strip(), note=spans[index].note)
    return SegmentTemplate(target_song_id=template.target_song_id, duration=template.duration,
                           spans=tuple(spans), source="manual", name=template.name,
                           note=template.note, version=TEMPLATE_VERSION)


def snap(moment: float, candidates: Iterable[float],
         tolerance: float = SNAP_SECONDS) -> float:
    """把一个时刻吸附到最近的参考点（拍点 / 停顿中心）。够不上就原样返回。"""
    point = float(moment)
    picks = [float(c) for c in candidates]
    if not picks:
        return round(point, 6)
    nearest = min(picks, key=lambda c: abs(c - point))
    return round(nearest if abs(nearest - point) <= float(tolerance) else point, 6)


# ==================================================================== 校验
def validate(template: SegmentTemplate) -> None:
    """首尾相接、不重叠、不留缝、铺满整首歌、每段够长。不合就抛 `SegmentError`。"""
    spans = template.spans
    if not spans:
        raise SegmentError("模板里一段都没有")
    if abs(spans[0].start) > 1e-6:
        raise SegmentError(f"第一段不是从 0 开始，而是 {spans[0].start:.3f}s")
    if abs(spans[-1].end - template.duration) > 1e-6:
        raise SegmentError(f"最后一段到 {spans[-1].end:.3f}s，没铺满 {template.duration:.3f}s")
    for i, span in enumerate(spans):
        if span.index != i:
            raise SegmentError(f"第 {i} 段的 index 是 {span.index}，编号乱了")
        if span.duration < MIN_SEGMENT_SECONDS - 1e-6:
            raise SegmentError(f"{span.name} 只有 {span.duration:.3f}s，"
                               f"短于 {MIN_SEGMENT_SECONDS}s")
        if i and abs(span.start - spans[i - 1].end) > 1e-6:
            raise SegmentError(f"{spans[i - 1].name} 和 {span.name} 之间"
                               f"有缝或者重叠了")


# ==================================================================== 接口
def positions_of(template: SegmentTemplate) -> tuple[TargetPosition, ...]:
    """把模板摊成 `TargetPosition` —— 切片/探针那一层只认这个类型。

    这样 `material_slice.plan_slices` 之类的下游一行都不用改：
    等间隔位置和用户段落在它们眼里都只是「一串 (index, start, end)」。
    """
    return tuple(TargetPosition(index=s.index, start=s.start, end=s.end)
                 for s in template.spans)


def to_json(template: SegmentTemplate) -> str:
    """落库用的紧凑 JSON。只存边界和标签，其它都能重算出来。"""
    return json.dumps({
        "version": template.version or TEMPLATE_VERSION,
        "duration": round(template.duration, 6),
        "source": template.source,
        "name": template.name,
        "note": template.note,
        "boundaries": [round(b, 6) for b in template.boundaries],
        "labels": [s.label for s in template.spans],
    }, ensure_ascii=False)


def from_json(payload: str | dict[str, Any], *, target_song_id: int = 0) -> SegmentTemplate:
    """读库：JSON → 模板。存坏了照样抛 `SegmentError`，不返回半个模板。"""
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    try:
        duration = float(data["duration"])
        boundaries = [float(b) for b in data.get("boundaries", ())]
    except (KeyError, TypeError, ValueError) as exc:
        raise SegmentError(f"段落模板的 JSON 坏了：{exc}") from exc
    template = _build(duration, boundaries, source=str(data.get("source") or "manual"),
                      target_song_id=target_song_id,
                      labels=[str(x) for x in data.get("labels", ())],
                      name=str(data.get("name") or ""), note=str(data.get("note") or ""))
    return template


__all__ = ["TEMPLATE_VERSION", "MIN_SEGMENT_SECONDS", "SNAP_SECONDS", "SegmentError",
           "uniform", "from_pauses", "from_boundaries",
           "split_at", "merge_at", "move_boundary", "rename", "snap",
           "validate", "positions_of", "to_json", "from_json"]
