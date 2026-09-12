"""卡帧抖动（stutter）：**只改帧序，不改时长**。

这是「特效」那一层唯一的规矩：一段素材写出去多少帧是段落长度说了算，
抖动只能改「第 k 帧去取源视频的哪一帧」。时长一变，成片就会前移、音乐就漂
（和首尾补边界帧同一条约束，见 `SliceSpec` 的恒等式）。

做法就是把窗口内的帧**按档保持**：`hold=2` 时输出 0,0,2,2,4,4…，
窗口一过立刻回到 k 本身 —— 所以窗口里"卡"了一下，出窗口马上追回正确位置，
画面和鼓点不会因为抖过一次就一直迟半拍。这也是剪辑里 stutter 的标准手感。

时间一律用**输出时间**（这一条素材自己的 0 起点，含首尾补的静帧）。
主音频上打的点是绝对秒数，换算成输出时间只有一处：`at - target_start`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: 比这还短的窗口没有观感（30fps 下不足两帧），直接当没打
MIN_SECONDS = 0.05
#: 一档保持几帧。2 = 每两帧一跳，卡点视频里最常用的力度
DEFAULT_HOLD = 2
#: 一档最多保持几帧：再大就不是抖动而是定格了
MAX_HOLD = 8
#: 三种玩法：
#:   `stutter`  卡帧抖动 —— 每 `hold` 帧保持一次（哒、哒、哒）
#:   `pingpong` 来回放   —— 先正放到中点，再倒放回来（回旋镜）
#:   `rewind`   回放     —— **先倒放**回去，再正着放回来（"倒带重看一遍"那种感觉）
#: 后两种里 `hold` 表示"来回几趟"
KINDS = ("stutter", "pingpong", "rewind")
#: 来回最多几趟。30 趟已经是"高频来回"了，再多也只是被窗口帧数卡住
#: （一趟至少要 2 帧，所以真正的趟数会按窗口长度夹一次）
MAX_TRIPS = 30





@dataclass(frozen=True)
class Stutter:
    """一个效果点：从 `at` 起、持续 `duration` 秒。

    `kind` 决定窗口里怎么取帧：

        stutter   每 `hold` 帧保持一次（哒、哒、哒 —— 卡帧抖动）
        pingpong  正放到中点再倒放回来，`hold` 表示**来回几趟**（回旋镜）

    `at` 存的是**主音频绝对秒数**（主音频是唯一时间权威）。改了分段之后
    这个点仍然指着同一个鼓点，只是可能落到了隔壁段落。
    """

    at: float
    duration: float
    hold: int = DEFAULT_HOLD
    kind: str = "stutter"

    @property
    def end(self) -> float:
        return round(self.at + self.duration, 6)

    def valid(self) -> bool:
        floor = 1 if self.kind in ("pingpong", "rewind") else 2
        return (self.duration >= MIN_SECONDS and self.hold >= floor
                and self.kind in KINDS)


    def shifted(self, delta: float) -> "Stutter":
        """整点平移（主音频时间 → 素材内时间就靠它，delta = −target_start）。"""
        return Stutter(round(self.at + float(delta), 6), self.duration,
                       self.hold, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {"at": round(float(self.at), 3),
                "duration": round(float(self.duration), 3),
                "hold": int(self.hold), "kind": str(self.kind)}

    @classmethod
    def from_dict(cls, data: Any) -> "Stutter | None":
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind") or "stutter")
        if kind not in KINDS:
            kind = "stutter"
        ceiling = MAX_TRIPS if kind in ("pingpong", "rewind") else MAX_HOLD
        floor = 1 if kind in ("pingpong", "rewind") else 2

        try:
            point = cls(float(data.get("at") or 0.0),
                        float(data.get("duration") or 0.0),
                        max(floor, min(ceiling, int(data.get("hold") or DEFAULT_HOLD))),
                        kind)
        except (TypeError, ValueError):
            return None
        return point if point.valid() else None



def parse(rows: Any) -> list[Stutter]:
    """一堆字典 / Stutter 混着来也照收，按时间排好、丢掉不合法的。"""
    out: list[Stutter] = []
    for row in rows or ():
        point = row if isinstance(row, Stutter) else Stutter.from_dict(row)
        if point is not None and point.valid():
            out.append(point)
    out.sort(key=lambda item: item.at)
    return out


def clip_to(points: Sequence[Any], start: float, end: float) -> list[Stutter]:
    """把抖动点裁到 `[start, end)` 里。

    跨段落的一律裁短，完全不相交的丢掉 —— 和「不许跨段取素材」同一条铁律：
    一个点只能影响它所在的那一段，绝不许溢到隔壁段去。
    """
    left, right = float(start), float(end)
    out: list[Stutter] = []
    for point in parse(points):
        begin = max(left, float(point.at))
        finish = min(right, float(point.end))
        if finish - begin < MIN_SECONDS:
            continue
        out.append(Stutter(round(begin, 6), round(finish - begin, 6),
                           point.hold, point.kind))
    return out



def frame_plan(count: int, fps: float, points: Sequence[Any]) -> list[int]:
    """输出第 k 帧该取源序列里的第几帧。**长度恒等于 `count`**（时长守恒）。

    没有效果点时就是 `[0, 1, 2, …]`（恒等映射，调用方可以直接跳过）。
    窗口一结束一律回到 k 本身 —— 抖过/来回过一趟不会一直迟半拍。

    两种玩法：

        stutter   每 `hold` 帧保持一次：0,0,2,2,4,4…
        pingpong  正放到中点再倒放回来（`hold` = 来回几趟）：0,1,2,3,2,1…
    """
    total = max(0, int(count))
    plan = list(range(total))
    if total <= 0 or fps <= 0:
        return plan
    for point in parse(points):
        first = max(0, int(round(float(point.at) * fps)))
        last = min(total, int(round(float(point.end) * fps)))
        span = last - first
        if span < 2:
            continue
        if point.kind in ("pingpong", "rewind"):
            # 趟数按窗口长度夹一次：一趟至少 2 帧，30 趟塞不进 8 帧的窗口
            trips = max(1, min(MAX_TRIPS, int(point.hold), max(1, span // 2)))
            period = max(2, span // trips)

            half = period // 2
            back_first = point.kind == "rewind"
            for k in range(first, last):
                step = (k - first) % period
                if back_first:
                    # 先**倒放**回去（half → 0），再正着放回来（1 → …）
                    offset = half - step if step <= half else step - half
                else:
                    # 先正放到中点，再倒放回来
                    offset = step if step <= half else 2 * half - step
                plan[k] = first + max(0, min(offset, span - 1))
            continue

        hold = max(2, min(MAX_HOLD, int(point.hold)))
        for k in range(first, last):
            plan[k] = first + ((k - first) // hold) * hold
    return plan



def describe(points: Sequence[Any]) -> str:
    """给状态栏/提示用的一句话。"""
    parsed = parse(points)
    if not parsed:
        return "没有效果点"
    head = "、".join(f"{p.at:.2f}s×{p.duration:.2f}"
                     f"{'回放' if p.kind == 'rewind' else ('来回' if p.kind == 'pingpong' else '抖')}"
                     for p in parsed[:3])

    more = f" 等 {len(parsed)} 个" if len(parsed) > 3 else ""
    return f"效果点：{head}{more}"


__all__ = ["MIN_SECONDS", "DEFAULT_HOLD", "MAX_HOLD", "KINDS", "MAX_TRIPS", "Stutter",
           "parse", "clip_to", "frame_plan", "describe"]


