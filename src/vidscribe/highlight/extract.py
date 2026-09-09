"""「提取数据」：成品 + 它的高光 JSON → 第二轮混剪的输入行。

为什么单独一个模块：这一层算的是**业务口径**（成片时长怎么算、轨道怎么平移、
原声空隙从哪来），跟界面没有关系。原来它长在 `gui/assets_dialog.py` 里，
结果 CLI 想导同一份数据就得把 PyQt5 一起拉进来 —— 那是不该有的依赖。

搬出来之后两边共用同一份实现：资产中心的「提取数据」按钮和 `assets --extract`
导出的每一行**逐字一致**，不存在"界面导的和命令行导的不一样"这种事。

一行的形状（顺序固定，第二轮混剪按这个顺序读）：

    timeline → o / s / w / a / e → t → gaps → startframe → freeze
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .. import ai_protocol

#: 接缝合并的容差：只有「剪口两侧」这种差值本该是 0 的情况才合并。
#: 不敢放大 —— 一段真话（哪怕只有一个短词）夹在两处空隙中间时，那是两处，不能并成一处
_SEAM_EPS = 0.01


#: timeline 里只留这几个键：成片时长、老写法总分、AI 打的主观四维、类型。
#: `scores` 必须带出去 —— 第二轮混剪就是靠它排序取舍的。
#: 客观数字不在这里再算一遍：能插旁白的静音已经有 `gaps` 那一项了
EXTRACT_TIMELINE_KEYS = ("duration", "score", "scores", "type")
#: `t` 是纯描述块（Scene / Action / Speech text），没有时间，原样照抄
EXTRACT_EXTRA_KEYS = ("t",)


def span_gaps(speech: Any, spans: Any) -> list[list[float]]:
    """一个成品的原声空隙：直接用它的**实际渲染区间**（来自 clips）算，不碰 JSON。

    成品表要显示「这条能不能插配音」，只需要区间和语音，用不着去溯源高光 JSON。
    多段区间只认第一段——一个成品对应一段（多出来的是历史脏数据，调用方会告警）。
    """
    if not speech or not spans:
        return []
    from .clip_engine import silent_gaps  # noqa: PLC0415 - 重依赖，用到才导

    first = spans[0]
    left = ai_protocol.num(first.get("start"))
    right = ai_protocol.num(first.get("end"))
    if left is None or right is None or right <= left:
        return []
    return silent_gaps(speech, left, right)


#: 成品实际时长和「按当前配置现算」的时长差多少算对不上。
#: 渲染是按帧对齐的，差零点几帧属正常（实测 5.52 算出来渲成 5.533），0.2 秒足够宽
STALE_TOLERANCE = 0.2


def stale_seconds(recorded: Any, keeps: Sequence[tuple[float, float]] | None,
                  region: tuple[Any, Any] | None) -> float:
    """成品**实际时长**和按当前配置现算的时长差多少秒（0 = 对得上，或者判断不了）。

    只有「清待重剪」那个按钮用它：想让盘上的成品符合新配置（比如把时长上限收紧了、
    希望成片真的变短），就得知道哪些成品是老配置的产物。`keeps` 要传
    `keeps_by_config()` 的结果 —— 那是「按现在的配置该剪成什么样」。

    **提取数据不用它**：那边按成品的真实时长反推坐标（`keeps_for`），配置改了也导得对，
    不需要先重剪。提取侧要报的是另一回事，见 `unresolved_seconds`。

    只报数，不改行为 —— 悄悄换一套坐标只会把错误藏得更深。
    """

    have = ai_protocol.num(recorded)
    if have is None or have <= 0 or not region:
        return 0.0                      # 老数据没记时长，判断不了，不瞎报
    left = ai_protocol.num(region[0])
    right = ai_protocol.num(region[1])
    if left is None or right is None or right <= left:
        return 0.0
    want = (sum(hi - lo for lo, hi in keeps) if keeps else right - left)
    gap = abs(float(have) - float(want))
    return round(gap, 3) if gap > STALE_TOLERANCE else 0.0


def unresolved_seconds(recorded: Any, keeps: Sequence[tuple[float, float]] | None,
                       region: tuple[Any, Any] | None) -> float:
    """成品时长**反推不出来**时差多少秒（0 = 对得上，或者判断不了）。

    和 `stale_seconds` 不是一回事，别混：

      * `stale_seconds`   —— 成品和**当前配置**对不上（要不要重剪，人来定）
      * `unresolved_seconds` —— 成品的时长**解释不了**（数据本身有矛盾）

    提取数据已经不看配置了（`keeps_for` 按成品真实时长反推），所以「配置改过」
    不再影响导出。但还有一种真问题：库里记的时长既不等于区间长度，也凑不出任何
    一种静音剪法 —— 那说明 `clips` 那条记录和盘上的文件本来就对不上（比如
    区间被人改过、或者渲染中途换过参数）。这种情况下报出来的 `gaps` 不可信。

    只报数，不改行为：悄悄糊一个坐标只会把错误藏得更深。
    """
    have = ai_protocol.num(recorded)
    if have is None or have <= 0 or not region:
        return 0.0                      # 老数据没记时长，判断不了，不瞎报
    left = ai_protocol.num(region[0])
    right = ai_protocol.num(region[1])
    if left is None or right is None or right <= left:
        return 0.0
    if keeps:
        return 0.0                      # 反推出来了，就是它
    gap = abs(float(have) - (right - left))
    return round(gap, 3) if gap > STALE_TOLERANCE else 0.0


def unresolved_note(count: int, total: int) -> str:
    """「有几条时长解释不了」的人话提示；`count` 为 0 时返回空串。"""
    if count <= 0:
        return ""
    return (f"[提取数据] {count}/{total} 条的成品时长既不等于渲染区间的长度、"
            f"也凑不出任何一种静音剪法 —— 这些 clips 记录和盘上的文件对不上。"
            f"它们的 gaps 按「整段没剪」算，可能不准；"
            f"建议把这些成品删了重剪一遍。")


def stale_note(count: int, total: int) -> str:

    """「有几条对不上」的人话提示；`count` 为 0 时返回空串。"""
    if count <= 0:
        return ""
    return (f"[提取数据] {count}/{total} 条的成品时长和当前的静音配置对不上 —— "
            f"这些成品是用**别的配置**渲的。导出的 gaps 和挂字时间按现在的配置算，"
            f"和盘上的文件不是一回事，第二轮会插偏。"
            f"先删掉这些成品重剪一遍，再导。")


def gaps_note(gaps: Any) -> str:
    """空隙列的文字：`2 处 / 最长 2.45s`，没有就写「无」。"""
    items = list(gaps or ())
    if not items:
        return "无"
    longest = max(float(one[1]) - float(one[0]) for one in items)
    return f"{len(items)} 处 / 最长 {longest:.2f}s"


def _kept_gaps(speech: Any, keeps: Sequence[tuple[float, float]]) -> list[list[float]]:
    """按**成品实际保留的片段**算原声空隙，成片坐标（0 起）。

    静音被剪掉之后，成品里剩下的空隙不是原区间那些了：每一段保留片段各自内部还有
    没超时、没被动过的静音，它们在成片里的位置由前面几段的总长决定。
    所以逐段用 `silent_gaps` 找静音，边界再一律走 `shift_time` 换算 —— 坐标换算
    只许有这一处。自己拿"剪掉了多少秒"去减，早晚漏一路，第二轮的 TTS 就压在人说话上。

    接缝处不造假空隙：剪掉的那部分本来就不在成品里，接缝两侧如果各自都是静音，
    它们在成片里是**连着的一段**，合成一段报出去，不报两段（报两段等于告诉第二轮
    "这里有个说话的间隔"，而成品里根本没有）。
    """
    from .clip_engine import shift_time, silent_gaps  # noqa: PLC0415 - 重依赖，用到才导

    plan = [(float(one[0]), float(one[1])) for one in keeps
            if ai_protocol.num(one[0]) is not None and ai_protocol.num(one[1]) is not None
            and float(one[1]) > float(one[0])]
    if not plan:
        return []
    out: list[list[float]] = []
    for lo, hi in plan:
        for head, tail in silent_gaps(speech, lo, hi):
            # silent_gaps 给的是这一段内部的相对秒，先还原成原视频绝对秒再换算
            start = shift_time(plan, lo + head)
            end = shift_time(plan, lo + tail)
            if start is None or end is None or end <= start:
                continue           # 换算不出来（理论上不会）就丢掉，不猜
            if out and start - out[-1][1] <= _SEAM_EPS:
                out[-1][1] = round(end, 2)      # 剪口两侧的静音，成片里是同一段
                continue
            out.append([round(start, 2), round(end, 2)])
    return out


def keeps_by_config(speech: Any, region: tuple[Any, Any] | None, *,
                    keep: Any, target: Any = 0.0) -> list[tuple[float, float]] | None:
    """按**当前配置**这个区间该剪成哪几段；不用剪就返回 None。

    和 `keeps_for` 是两件事，别混：

      * `keeps_for(made=)`  —— 成品**实际**剪成了什么样（提取数据用这个）
      * `keeps_by_config()` —— 按现在的配置**应该**剪成什么样（判断要不要重剪用这个）

    只有「清待重剪」那个按钮需要它：拿它和成品的真实时长比一比，就知道这个成品
    是不是用别的配置渲的。提取数据**不许**用它 —— 那样改一次配置就得把旧成品
    全重剪一遍才能导出正确的 `gaps`。
    """
    from .clip_engine import trim_for  # noqa: PLC0415 - 重依赖，用到才导

    seconds = ai_protocol.num(keep)
    if not speech or not region or seconds is None or seconds <= 0:
        return None
    left = ai_protocol.num(region[0])
    right = ai_protocol.num(region[1])
    if left is None or right is None or right <= left:
        return None
    spans = trim_for(speech, left, right, keep=seconds,
                     target=ai_protocol.num(target) or 0.0)
    return spans if len(spans) > 1 else None


def keeps_for(speech: Any, region: tuple[Any, Any] | None, *,

              made: Any) -> list[tuple[float, float]] | None:
    """这个成品**实际**剪成了哪几段（原视频绝对秒）；没剪过就返回 None。

    口径由**成品自己**说了算，不由当前配置说了算：拿库里记的成品真实时长
    （`clips.duration`，渲染当时写的）去反推（`clip_engine.trim_as_made`）。

    为什么不能按配置现算：`silence_keep` 改一次，现算出来的片段描述的就是
    「一个还没渲出来的成品」—— 报给第二轮的 `gaps` 整体偏掉，而且从数据上完全
    看不出来。而且那样每改一次配置就得把所有旧成品重剪一遍才能导数据，没必要。

    `region` 是这个成品在原视频里的实际渲染区间（`clips.start_time` / `end_time`，
    渲染时静音压缩不改这两头，改的只是 `duration`）。

    返回 None 的几种情况，调用方一律按「没剪过静音」处理，行为一个字不变：
    时长和区间长度本来就一样（压根没剪）、没有逐词时间戳、区间不成立、
    时长缺失，以及**反推不出来**（`unresolved_seconds` 会把这种单独报出来）。
    """
    from .clip_engine import trim_as_made  # noqa: PLC0415 - 重依赖，用到才导

    seconds = ai_protocol.num(made)
    if not speech or not region or seconds is None or seconds <= 0:
        return None
    left = ai_protocol.num(region[0])
    right = ai_protocol.num(region[1])
    if left is None or right is None or right <= left:
        return None
    return trim_as_made(speech, left, right, made=float(seconds))



def _shift_tracks(tracks: dict[str, Any],
                  keeps: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """剪过静音的成品：轨道（o / s / w / a / e）的时间全部换算到**剪完之后**的成片坐标。

    进来的时间已经是「按实际起剪点平移过」的成片坐标（0 = 起剪点），但那是**剪之前**
    那条时间轴。区间中间少了几截，后面的字必须整体往前挪 —— 挪多少只许由 `shift_time`
    算（和 `gaps` 同一个函数）。自己拿"剪掉的总秒数"去减，第一处剪口之前的字就会被
    多减一次，成片里字和画面对不上。

    落在被剪掉那一段里的时间贴到**剪口**上，不丢：那一刻的画面在成片里就是剪口，
    字还在、只是位置换了；丢掉等于第二轮少一句台词。
    """
    from .clip_engine import shift_time  # noqa: PLC0415 - 重依赖，用到才导

    base = float(keeps[0][0])
    limit = round(sum(hi - lo for lo, hi in keeps), 3)

    def moved(value: float) -> float:
        moment = base + float(value)
        at = shift_time(keeps, moment)
        if at is None:
            # 剪掉的空档：贴到剪口（这一刻之前保留下来的总长）
            passed = 0.0
            for lo, hi in keeps:
                if moment < lo:
                    break
                passed += hi - lo
            at = passed
        return round(min(max(0.0, at), limit), 2)

    out: dict[str, Any] = {}
    for key, items in tracks.items():
        kept: list[Any] = []
        for item in items:
            if not isinstance(item, (list, tuple)) or not item:
                kept.append(item)
                continue
            first = ai_protocol.num(item[0])
            if first is None:
                kept.append(item)
                continue
            second = ai_protocol.num(item[1]) if len(item) > 1 else None
            one = list(item)
            if second is None:                      # 单点型 [time, kind, text]
                one[0] = moved(first)
                kept.append(one)
                continue
            one[0], one[1] = moved(first), moved(second)   # 区间型 [start, end, ...]
            if one[1] - one[0] <= 0:
                continue        # 整段都落在被剪掉的静音里，成片里没有它
            kept.append(one)
        out[key] = kept
    return out


def region_bounds(clip: dict[str, Any], region: tuple[Any, Any] | None, *,
                  startframe: float, span: float | None) -> tuple[float, float] | None:
    """这个成品在**原视频**里的区间 `(左, 右)`（绝对秒）；推不出来就返回 None。

    优先用 `region`（`clips.start_time` / `end_time`，引擎修正过的话就是修正后的真值）；
    没有才按 JSON 现推：起点 = `sa + startframe`，长度 = `span`。
    空隙和原声轨道都得按同一个区间算，所以这段推导只许有这一份 —— 两处各推一遍，
    早晚有一处漏了 `startframe`，报给第二轮的空隙和台词时间就各说一套。
    """
    left = ai_protocol.num(region[0]) if region else None
    right = ai_protocol.num(region[1]) if region else None
    if left is None:
        left = ai_protocol.num(clip.get("start"))
        if left is None:
            return None
        left += float(startframe)
    if right is None or right <= left:
        cut = ai_protocol.num(span)
        if cut is None:
            return None
        right = left + cut
    return float(left), float(right)


def voice_tracks(speech: Any, bounds: tuple[float, float] | None) -> dict[str, list[Any]]:
    """成品里的原声：`s` = 句区间 + 原文，`w` = 逐词区间 + 单词。

    时间是**剪之前**那条成片时间轴（0 = 起剪点）—— 剪过静音的由 `_shift_tracks`
    统一换算，和挂字、空隙走同一个 `shift_time`。

    为什么必须由程序给：召回那一轮的清单只写「结果在第几秒」和一个挂字文案，
    根本没有 `s` / `w`（PRM 明令不许它算时间）。于是第二轮拿到的台词只有
    `t.Speech text` 一整串没有时间的文本 —— 「避开关键台词」「别压住原声」这些规则
    就全都无从执行，只能瞎猜。逐词时间戳在库里、精确到毫秒，这是程序的本职。

    这不是替 AI 下判断：报出去的只有「哪一刻说了哪个词」这种**测量值**，
    哪句是关键台词、旁白往哪塞，仍然由第二轮自己判断。

    句子的文本走 `speech_between`，和 `t.Speech text`、清单侧逐字同一个口径；
    只蹭到边界一半的句子按它的规矩处理（有逐词就只收区间内的词，没有就整句丢掉）。
    """
    if not speech or not bounds:
        return {}
    from .from_moments import speech_between  # noqa: PLC0415 - 和清单侧共用同一个口径

    left, right = bounds
    if right <= left:
        return {}
    lines: list[Any] = []
    words: list[Any] = []
    for seg in speech:
        start = ai_protocol.num(getattr(seg, "start", None))
        end = ai_protocol.num(getattr(seg, "end", None))
        if start is None or end is None or end <= left or start >= right:
            continue
        lo, hi = max(left, start), min(right, end)
        text = speech_between([seg], left, right)
        if text and hi > lo:
            lines.append([round(lo - left, 2), round(hi - left, 2), text])
        for word in getattr(seg, "words", ()) or ():
            at = ai_protocol.num(getattr(word, "start", None))
            till = ai_protocol.num(getattr(word, "end", None))
            one = str(getattr(word, "text", "") or "").strip()
            # 只收整个词都在区间里的：半个词的时间戳报出去等于谎报「这里有话」
            if at is None or till is None or not one:
                continue
            if at < left - 0.001 or till > right + 0.001:
                continue
            words.append([round(at - left, 2), round(till - left, 2), one])
    lines.sort(key=lambda one: (one[0], one[1]))
    words.sort(key=lambda one: (one[0], one[1]))
    out: dict[str, list[Any]] = {}
    if lines:
        out["s"] = lines
    if words:
        out["w"] = words
    return out


def clip_gaps(clip: dict[str, Any], speech: Any, *, region: tuple[Any, Any] | None,
              startframe: float, span: float | None,
              keeps: Sequence[tuple[float, float]] | None = None) -> list[list[float]]:
    """这条素材内部的原声空隙，成片坐标（0 起）。数据不全一律给空数组。

    区间由 `region_bounds` 推（和原声轨道 `s` / `w` 同一份推导）。
    空隙本身交给 `clip_engine.silent_gaps` 算，口径只有那一处。

    `keeps`（`clip_engine.trim_plan` 给的保留片段，原视频绝对秒）：这个成品渲染时把超时
    静音剪掉了，成品里剩下的空隙得按剪完之后的实际片段算，否则第二轮按老坐标落 TTS，
    旁白就压在人说话上。不给 `keeps` 时行为一个字不变（没剪过的成品、老数据都走这条）。
    """
    if not speech:
        return []          # 这个视频没分析过语音：没有词级时间戳，不猜
    if keeps:
        return _kept_gaps(speech, keeps)   # 剪过的成品：区间已经不连续，按实际片段算
    from .clip_engine import silent_gaps  # noqa: PLC0415 - 重依赖，用到才导

    bounds = region_bounds(clip, region, startframe=startframe, span=span)
    if bounds is None:
        return []
    return silent_gaps(speech, bounds[0], bounds[1])


def spoken_text(speech: Any, keeps: Sequence[tuple[float, float]] | None,
                region: tuple[Any, Any] | None) -> str:
    """这个成品里**实际说了什么**（原文，按时间顺序拼好）。

    第二轮要写 TTS 旁白：不知道原声说了什么，就避不开撞词、也接不上话。
    只有 `Scene`/`Action` 那种画面描述替代不了台词。

    剪过静音的按**保留片段**逐段拼 —— 被剪掉的是静音，本来没有词，
    但按片段拼能保证「文本里的话都真的在成品里」，不会把剪掉那截的尾音算进来。
    """
    if not speech:
        return ""
    from .from_moments import speech_between  # noqa: PLC0415 - 和清单侧共用同一个口径

    spans: list[tuple[float, float]] = []
    if keeps:
        spans = [(float(a), float(b)) for a, b in keeps]
    elif region:
        left = ai_protocol.num(region[0])
        right = ai_protocol.num(region[1])
        if left is not None and right is not None and right > left:
            spans = [(left, right)]
    parts = [speech_between(speech, a, b) for a, b in spans]
    return " ".join(one for one in parts if one).strip()


def slim_clip(timeline: dict[str, Any], clip: dict[str, Any], *,
              duration: float | None, tracks: dict[str, Any],
              startframe: float, freeze: float,
              gaps: list[list[float]] | None = None,
              said: str = "") -> dict[str, Any]:
    """「提取数据」的一行：timeline + 重算过的时间轨道 + t 描述 + 空隙 + 两个加减秒数。

    `duration` 是**成片真实总时长**（播放段 + 末帧冻结），由调用方算好传进来；
    算不出来才退回 JSON 里 `timeline.duration` 的原值。`score` / `scores` / `type` 照抄原值。

    `tracks` 是已经按实际起剪点平移、并裁进成片范围的 o / s / w / a / e。
    `t` 没有时间，原样照抄。`gaps` 是程序算出的原声空隙（第二轮混剪的 TTS 落位靠它），
    **恒定出现**：没有可用空隙就是空数组，不省略、不给 null——第二轮据此判断"这条不插配音"。
    顺序固定：timeline → o → s → w → a → e → t → gaps → startframe → freeze。
    文字内容一个字都不改（中文、emoji 原样带出）。
    """
    slim: dict[str, Any] = {}
    for key in EXTRACT_TIMELINE_KEYS:
        value = duration if key == "duration" else timeline.get(key)
        if key == "duration" and value is None:
            value = timeline.get("duration")
        if value is None or value == "":
            continue
        slim[key] = value
    out: dict[str, Any] = {"timeline": slim}
    for key in ai_protocol.TRACK_KEYS:
        if key in tracks:
            out[key] = tracks[key]
    for key in EXTRACT_EXTRA_KEYS:
        if key not in clip:
            continue
        value = clip[key]
        if key == "t" and isinstance(value, dict):
            # `Speech text` 由程序填：第二轮写旁白得知道原声说了什么，
            # 光有 Scene / Action 那种画面描述避不开撞词。原 dict 不动，改副本
            value = {**value, "Speech text": said}
        out[key] = value
    out["gaps"] = list(gaps or ())
    out["startframe"] = round(float(startframe), 2)
    out["freeze"] = round(float(freeze), 2)
    return out


def extract_line(payload: Any, product: str, *, startframe: float,
                 freeze: float, source: str = "", span: float | None = None,
                 region: tuple[Any, Any] | None = None,
                 speech: Any = (),
                 keeps: Sequence[tuple[float, float]] | None = None,
                 ) -> tuple[str, list[list[float]]] | None:

    """一份 JSON → 「提取数据」的一行：`(JSON 文本, 空隙)`；抠不出片段就返回 None。

    `product` 是**成品文件名**（`xxx_1284.mp4`），写进 `video` —— 第二轮拿它当素材的
    唯一标识（每个 id 只能用一次、play_order 就是它的列表）。**不能写原视频名**：
    同一个原视频常常剪出好几条素材，都叫同一个名字的话第二轮分不清、程序也映射不回文件。
    `source` 是原视频名，单独一列 —— 同源去重靠它（同一条原片剪出来的几条素材
    不该进同一组混剪）。

    空隙一并返回，调用方据此过滤和报数——不用把导好的行再解析一遍。

    `t.Speech text` 由程序填：这个成品里**实际说了什么**（`spoken_text`）。
    第二轮写 TTS 旁白得知道原声说了什么 —— 才避得开撞词、接得上话。

    `s`（句）/ `w`（逐词）两条原声轨道也由程序填（`voice_tracks`）：召回那一轮的清单
    压根不给时间轴，光有 `t.Speech text` 那一整串没有时间的文本，第二轮就没法执行
    「避开关键台词」「别压住原声」—— 哪一句在第几秒都不知道。逐词时间戳在库里，
    以库为准；库里没有语音数据时这两条轨道不出现（不猜）。


    `timeline.duration` 用的是**实际剪进去的区间长度**，和成品文件名同一个口径：
    优先取这个成品的实际渲染区间（`span`，来自 clips，引擎修正过就是修正后的值），
    没有才按 JSON 现算 `(end + freeze) − (sa + startframe)`。
    冻帧不算在里面。
    轨道（o / s / w / a / e）照旧按加减秒数整体平移并裁进这个长度：加减秒数改的是
    画面对齐，字和声音得跟着走。两个加减秒数以 JSON 里那份为准（剪辑时盖进去的，
    记的是这个成品当时的口径），老数据没有才退回当前配置。

    `gaps`（原声空隙）由 `region` + `speech` 现算：`region` 是这个成品在原视频里的
    实际渲染区间（来自 clips），`speech` 是这个视频的句 + 逐词。缺哪一样就给空数组，
    绝不拿估算值糊上去——第二轮混剪要靠它落 TTS，宁可说"没有空隙"也不能报错的空隙。

    `keeps`（可选，`clip_engine.trim_plan` 给的保留片段）：这个成品渲染时剪掉了超时静音，
    空隙就得按剪完之后的成片坐标算，原样往下传给 `clip_gaps`。轨道（o / s / w / a / e）
    也跟着走 —— 先按剪之前那条时间轴裁进区间，再用 `shift_time` 整体换算（`_shift_tracks`），
    不然字和画面在成片里就错开了。给了 `keeps` 时 `timeline.duration` 应该是**保留片段的
    总长**（区间长度减去剪掉的部分）——但时长本身仍由调用方算好用 `span` 传进来，
    这里不重算时长，免得两处口径各说一套。
    """

    clips = ai_protocol.clips(payload)
    if not clips:
        return None
    timeline = ai_protocol.timeline_of(payload)
    clip = clips[0]        # 一个成品一份 JSON、一段高光，就是一行
    one_start = clip.get("startframe", startframe)
    one_freeze = clip.get("freeze", freeze)
    cut = ai_protocol.num(span)
    if cut is None:
        cut = ai_protocol.play_seconds(clip, startframe=one_start, freeze=one_freeze)
    if keeps:
        # 剪过静音：轨道先按**剪之前**那条时间轴裁（区间两头之间的长度），再整体换算到
        # 剪完之后的坐标。直接拿剪完的时长当上限会把尾巴那几句字冤枉地裁掉 ——
        # 它们在原区间里明明还在范围内
        envelope = round(float(keeps[-1][1]) - float(keeps[0][0]), 3)
        tracks = ai_protocol.retime_tracks(clip, startframe=one_start,
                                           freeze=one_freeze, span=envelope)
        tracks.update(voice_tracks(speech, (float(keeps[0][0]),
                                            float(keeps[0][0]) + envelope)))
        tracks = _shift_tracks(tracks, keeps)
    else:
        tracks = ai_protocol.retime_tracks(clip, startframe=one_start,
                                           freeze=one_freeze, span=cut)
        tracks.update(voice_tracks(speech, region_bounds(
            clip, region, startframe=one_start, span=cut)))
    gaps = clip_gaps(clip, speech, region=region,
                     startframe=one_start, span=cut, keeps=keeps)

    line = json.dumps(
        {"video": product,
         **({"source": source} if source else {}),
         **slim_clip(timeline, clip, duration=cut, tracks=tracks,
                     startframe=one_start, freeze=one_freeze, gaps=gaps,
                     said=spoken_text(speech, keeps, region))},
        ensure_ascii=False, separators=(",", ":"))
    return line, gaps
