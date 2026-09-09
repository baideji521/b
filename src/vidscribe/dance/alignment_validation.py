"""对齐结果的校验层：多窗口一致性 → 最终置信度 → 结论，以及人工修正留痕。

**纯函数**，不读音频、不碰数据库、不认识 PyAV —— 只吃"每个窗口算出来的 offset"这些数字。
所有阈值都摆在下面，改哪个都能一眼看见影响面。

技术指导第五节第 6 条那句"不要只分析视频前 40 秒后就直接相信结果"是这一层存在的全部理由：
单窗口互相关在副歌重复、循环鼓点、翻录变速这些情况下会给出漂亮但错误的峰。
"""

from __future__ import annotations

from datetime import datetime

from .types import DanceAlignment, WindowResult

# ==================================================================== 阈值
#: 两个窗口的 offset 差多少算"一致"（秒）。视频帧长 1/30≈0.033s，
#: 0.05 意味着"肉眼看不出错位"，比它大就是真的对不上
OFFSET_TOLERANCE = 0.05
#: waveform 与 chroma 两路 offset 差多少还算互相印证（秒）。
#: chroma 的时间分辨率是一个 hop（512/22050≈0.023s），所以放宽到 0.12
METHOD_TOLERANCE = 0.12
#: 最终置信度低于此判 low_confidence（可用但建议人工确认）
LOW_CONFIDENCE = 0.55
#: 最终置信度低于此直接判 rejected（不可用）
REJECT_CONFIDENCE = 0.25
#: 多窗口最大偏差超过这么多秒直接判 rejected —— 这已经不是"精度不够"而是"对错了"
REJECT_DEVIATION = 0.5
#: offset 绝对值大于源时长的这个比例时判 rejected（边缘 offset 基本是伪峰）
EDGE_RATIO = 0.95
#: 默认验证窗口数与窗口长度（秒）
WINDOW_COUNT = 5
WINDOW_SECONDS = 20.0
#: 窗口至少要这么长才有意义；比它短就退化成单窗口
MIN_WINDOW_SECONDS = 4.0
#: 峰值闸门：峰值质量低于此就认为"互相关根本没挑出赢家"，见 `combine_confidence`。
#: 0.15 对应主峰比最强次峰高约 18%（映射是 1 - 1/ratio）
MIN_PEAK_QUALITY = 0.15



def window_plan(duration: float, window_seconds: float = WINDOW_SECONDS,
                count: int = WINDOW_COUNT) -> list[tuple[float, float]]:
    """规划验证窗口，返回 `[(起点秒, 窗口长度秒), ...]`。

    必须覆盖开头 / 中间 / 结尾，中间再按等距补齐到 `count` 个
    （技术指导第五节第 6 条）。素材短到放不下多个窗口时就老实退化成一个整段窗口，
    并让 `window_count == 1` 传下去 —— 上层据此不给"多窗口一致"的加分。
    """
    duration = max(0.0, float(duration))
    if duration <= 0:
        return []
    window = max(MIN_WINDOW_SECONDS, float(window_seconds))
    if duration <= window * 1.5:
        return [(0.0, duration)]
    count = max(1, int(count))
    if count == 1:
        return [(0.0, window)]
    last_start = max(0.0, duration - window)
    step = last_start / float(count - 1)
    plan: list[tuple[float, float]] = []
    for i in range(count):
        start = round(min(last_start, i * step), 3)
        if plan and abs(plan[-1][0] - start) < 1e-6:
            continue                    # 等距落到同一点（素材很短）时不重复算
        plan.append((start, window))
    return plan


def agreement_of(offsets: list[float]) -> tuple[float, float]:
    """多窗口一致性，返回 `(agreement 0~1, 最大偏差秒)`。

    一致性用"落在中位数 ±OFFSET_TOLERANCE 内的窗口占比"，不用标准差：
    标准差会被一个离群窗口拖垮，而实际情况往往是"4 个窗口一致、1 个撞上副歌重复"，
    那种情况仍然可信，只是要扣一点分。
    """
    values = [float(v) for v in offsets]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return 1.0, 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2.0)
    inside = sum(1 for v in values if abs(v - median) <= OFFSET_TOLERANCE)
    deviation = max(abs(v - median) for v in values)
    return round(inside / len(values), 4), round(deviation, 4)


def robust_offset(offsets: list[float], confidences: list[float] | None = None) -> float:
    """从多个窗口里定下最终 offset：取中位数，而不是"置信度最高那个窗口"。

    中位数天生抗离群：一个窗口整拍错位也拽不动结果。置信度只在偶数个窗口需要
    在中间两个之间二选一时用得上。
    """
    values = [float(v) for v in offsets]
    if not values:
        return 0.0
    order = sorted(range(len(values)), key=lambda i: values[i])
    middle = len(order) // 2
    if len(order) % 2:
        return round(values[order[middle]], 6)
    a, b = order[middle - 1], order[middle]
    if confidences and len(confidences) == len(values):
        return round(values[a if confidences[a] >= confidences[b] else b], 6)
    return round((values[a] + values[b]) / 2.0, 6)


def method_agreement(waveform_offset: float | None, chroma_offset: float | None) -> float:
    """波形与 chroma 两路的互相印证程度 0~1。

    有一路算不出来（None）时返回 0.5：既不奖励也不惩罚 —— "只有一个证人"
    和"两个证人吵架"不是一回事，后者才该扣分。
    """
    if waveform_offset is None or chroma_offset is None:
        return 0.5
    gap = abs(float(waveform_offset) - float(chroma_offset))
    if gap <= METHOD_TOLERANCE:
        return 1.0
    # 超出容差后线性衰减，到 10 倍容差归零
    span = METHOD_TOLERANCE * 9.0
    return round(max(0.0, 1.0 - (gap - METHOD_TOLERANCE) / span), 4)


def combine_confidence(waveform_confidence: float | None, chroma_confidence: float | None,
                       methods_agree: float, window_agreement: float,
                       window_count: int) -> float:
    """把四路证据合成最终置信度 0~1。

    加权部分：峰值质量 0.45 + 两法互证 0.25 + 多窗口一致 0.30。
    只有一个窗口时（素材太短）"多窗口一致"这一项按 0.5 计 —— 拿不到证据不等于证据为真，
    所以短素材的置信度天然上不了顶，这是有意的。

    最后一道是**峰值闸门**而不是简单上限：`gate = min(1, 峰值质量 / MIN_PEAK_QUALITY)`，
    整个加权分再乘上它。

    为什么必须有这道闸门：一段和目标歌毫不相干的噪声，靠"只有一个窗口所以按 0.5 算"
    加上一点偶然的和声支持，也能攒到 0.28 蒙混过关。而事实是 ——
    **互相关连一个能分辨的峰都没有，就根本不存在什么 offset 可谈**，后面几项全是无源之水。

    为什么是闸门而不是"直接拿峰值质量当上限"：节奏性音乐的互相关在 ±1 拍、±2 拍处
    天然有很高的旁瓣（同一首歌自己和自己比也一样），所以峰值比对真实音乐来说本就偏低
    （实测同一首 120BPM 曲子只有 0.17 左右）。而"整拍错位"这件事恰恰是靠**多窗口一致**
    排除的 —— 5 个窗口各自独立地都挑中同一个 offset，才是真正强的证据。
    拿峰值质量硬当上限会把这个最有力的判据整个丢掉。
    """
    peaks = [c for c in (waveform_confidence, chroma_confidence) if c is not None]
    peak_quality = float(sum(peaks) / len(peaks)) if peaks else 0.0
    windows = float(window_agreement) if int(window_count) > 1 else 0.5
    score = 0.45 * peak_quality + 0.25 * float(methods_agree) + 0.30 * windows
    gate = min(1.0, max(0.0, peak_quality) / MIN_PEAK_QUALITY)
    return round(float(min(1.0, max(0.0, score * gate))), 4)




def decide_status(confidence: float, max_deviation: float, offset: float,
                  source_duration: float, methods_agree: float,
                  target_duration: float = 0.0) -> tuple[str, list[str]]:
    """定结论，返回 `(status, 原因清单)`。原因是中文短句，直接显示给用户。

    判定顺序是从"最确定不能用"往下走，命中即止：
      1. 多窗口偏差过大 → rejected（这是"对错了"，不是"精度差")
      2. offset 顶到可搜索范围的边缘 → rejected（边缘伪峰）
      3. 置信度低于 REJECT_CONFIDENCE → rejected
      4. 两法互证为 0（明确吵架） → disagree
      5. 置信度低于 LOW_CONFIDENCE → low_confidence（可用，但建议人工确认）
      6. 其余 → ok

    第 2 条的上限**分方向**，这是有讲究的（口径：`source_time = target_time - offset`）：

      offset > 0  源视频的内容出现在目标歌里更靠后的位置 → 上限是**目标歌**时长
      offset < 0  源视频比目标歌先开始           → 上限是**源视频**时长

    用一个对称上限（比如两边都拿源时长比）会误杀真实结果：200 秒的歌配一个
    20 秒的源，源合法地对在歌的第 19 秒上，offset=+19 就会被"顶到源时长边缘"判死。
    """
    reasons: list[str] = []
    source = max(0.0, float(source_duration))
    target = max(0.0, float(target_duration))
    if float(max_deviation) > REJECT_DEVIATION:
        reasons.append(f"多窗口 offset 最大偏差 {max_deviation:.3f}s 超过 {REJECT_DEVIATION}s")
        return "rejected", reasons
    value = float(offset)
    limit = (target or source) if value >= 0 else source
    if limit > 0 and abs(value) >= limit * EDGE_RATIO:
        which = "目标歌" if value >= 0 and target else "源视频"
        reasons.append(f"offset {value:.3f}s 顶到{which}时长 {limit:.3f}s 的边缘，判为伪峰")
        return "rejected", reasons

    if float(confidence) < REJECT_CONFIDENCE:
        reasons.append(f"置信度 {confidence:.3f} 低于 {REJECT_CONFIDENCE}")
        return "rejected", reasons
    if float(methods_agree) <= 0.0:
        reasons.append("波形与 chroma 两路结论互相矛盾")
        return "disagree", reasons
    if float(confidence) < LOW_CONFIDENCE:
        reasons.append(f"置信度 {confidence:.3f} 低于 {LOW_CONFIDENCE}，建议人工确认")
        return "low_confidence", reasons
    return "ok", reasons


def manual_override(alignment: DanceAlignment, offset: float, reason: str,
                    operator: str = "", at: str = "") -> DanceAlignment:
    """人工改 offset，返回**新**对象；算法原值一律保留。

    技术指导第五节第 7 条：禁止静默覆盖。所以这里强制记六项 ——
    原 offset、人工 offset、原置信度、理由、操作时间、算法版本（算法版本本来就在对象里）。
    `reason` 留空直接抛 ValueError：三个月后回头看历史，没有理由的修正等于没有信息。

    重复修正时 `original_offset` 只在第一次写入，保留的始终是**算法**原值，
    不会被上一次人工值顶掉。
    """
    if not str(reason).strip():
        raise ValueError("人工修正必须写理由（禁止静默覆盖）")
    from dataclasses import replace  # noqa: PLC0415 - 只在这一处用

    stamp = at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    first_offset = (alignment.original_offset if alignment.original_offset is not None
                    else alignment.offset)
    first_confidence = (alignment.original_confidence if alignment.original_confidence is not None
                        else alignment.confidence)
    note = f"人工修正 {first_offset:.3f}s → {float(offset):.3f}s：{reason.strip()}"
    return replace(
        alignment,
        offset=round(float(offset), 6),
        status="manual",
        original_offset=round(float(first_offset), 6),
        original_confidence=first_confidence,
        manual_offset=round(float(offset), 6),
        manual_reason=reason.strip(),
        manual_at=stamp,
        manual_operator=str(operator or ""),
        notes=alignment.notes + (note,),
    )


def summarize(alignment: DanceAlignment) -> list[str]:
    """把对齐结论写成中文行，GUI 和 CLI 共用同一份措辞。"""
    lines = [
        f"[对齐] offset {alignment.offset:.3f}s｜置信度 {alignment.confidence:.3f}"
        f"｜结论 {alignment.status}｜算法 {alignment.algorithm_version}",
    ]
    if alignment.waveform_offset is not None:
        lines.append(f"[波形] offset {alignment.waveform_offset:.3f}s"
                     f"｜峰值置信度 {alignment.waveform_confidence or 0.0:.3f}")
    if alignment.chroma_offset is not None:
        lines.append(f"[Chroma] offset {alignment.chroma_offset:.3f}s"
                     f"｜峰值置信度 {alignment.chroma_confidence or 0.0:.3f}")
    lines.append(f"[窗口] {alignment.window_count} 个｜一致性 {alignment.agreement:.3f}"
                 f"｜最大偏差 {alignment.max_deviation:.3f}s")
    for window in alignment.windows:
        lines.append(f"  #{window.index} {window.window_start:.2f}s 起 {window.window_seconds:.1f}s"
                     f" → offset {window.offset:.3f}s（{window.confidence:.3f}）")
    if alignment.manual:
        lines.append(f"[人工] 原 {alignment.original_offset:.3f}s → {alignment.offset:.3f}s"
                     f"｜{alignment.manual_reason}｜{alignment.manual_at}"
                     f"｜{alignment.manual_operator or '未记名'}")
    lines.extend(f"[说明] {note}" for note in alignment.notes)
    return lines


__all__ = [
    "OFFSET_TOLERANCE", "METHOD_TOLERANCE", "LOW_CONFIDENCE", "REJECT_CONFIDENCE",
    "REJECT_DEVIATION", "EDGE_RATIO", "WINDOW_COUNT", "WINDOW_SECONDS", "MIN_WINDOW_SECONDS",
    "window_plan", "agreement_of", "robust_offset", "method_agreement",
    "combine_confidence", "decide_status", "manual_override", "summarize",
    "WindowResult",
]



