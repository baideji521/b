"""AI 高光 JSON 协议：**只认这一种**写法，全项目共用这一个解析入口。

    {
      "video": "xxx.mp4",
      "timeline": {"duration": 11.83, "score": 90, "type": "搞笑反转", "reason": "..."},
      "segments": [{"sa": 48.55, "end": 60.38, "dst": [0.00, 11.83]}],
      "o": [[0.00, "word", "2087?"], ...],
      "s": [[0.00, 0.46, "2087?"], ...],
      "w": [[0.00, 0.46, "2087?"], ...],
      "a": [[0.00, 11.83, "查看滤镜结果并绝望吐槽", "室内房间"]],
      "e": [[0.00, 1.45, "neutral", 0.43]],
      "t": {"Scene": "...", "Action": "...", "Speech text": "..."}
    }

进剪辑环节时还会往顶层盖三个参数（有就替换、没有就新增，一律以「剪辑高光」窗里的
当前配置为准，JSON 里的旧值绝不参与这一次剪辑）：

    "startframe": -0.55   # 起始加减秒数，起剪点 = sa + 本值
    "freeze": 2.0         # 结束加减秒数，结束点 = end + 本值
    "freezeframe": 2.0    # 末帧冻结几秒（0 = 不冻）

这三个都不算在 `timeline.duration` 里——`duration` 只是 AI 报的剪辑区间长度，
成片真实时长按这三个参数现算。手动剪、AI 自动剪都一样盖。
老数据里可能还留着别的参数键（早先版本盖过的），读的时候一律当没看见，不校验也不报错。

时间域分两套，别混：

  - `segments[].sa` / `segments[].end`  → **原视频时间**，剪辑就用这两个数；
  - `segments[].dst` / `timeline.duration` / `o` / `s` / `w` / `a` / `e` → **成片时间**；
  - `t` 没有时间，纯描述。

`dst` 是 AI 报的成片落位（有定帧 N 秒就是 `[0, duration+N]`）。渲染**不吃 dst**：
成片尾巴上那几秒冻帧是渲染时追加的，所以 dst 只当参考和自检，绝不用它算剪辑区间。

AI 一次回复可以带**多份** JSON（顶层数组、JSONL 每行一份、或者几个 `{...}` 并排贴着），
由 `split_payloads` 统一拆成一份一份的 dict，每份都当成一个独立的「高光方案」各自入库。
单份 dict 走的还是原来那条路，老调用一个字都不用改。

老协议（`video` + `clip.start` / `clip.end` + `clip.overlays` + `Trimclip`）现在**认**：
`payload_of` 会把它自动升级成上面这个新形状——`clip.start/end` 落进 `segments[0].sa/.end`，
`clip.duration` 进 `timeline.duration`，`score` / `type` / `reason` 进 `timeline`，
`Trimclip`（在 `clip` 里，也容忍写在顶层）连同 `overlays.evaluation` 进 `t`，
`overlays` 的 word / emoji / comment 三条进 `o`。**要紧的一点**：老协议里 `overlays` 的
`time` 是**原视频绝对时间**，而新协议 `o` 是成片时间域（0 = `segments[0].sa`），
升级时会减掉 `sa` 再裁进 `[0, duration]`。老协议没有的轨道（`s` / `w` / `a` / `e`）不凭空造。

`split_payloads` 拆出来的每一份都已经升级过，所以库里 `current_json` 存的一律是新形状，
下游剪辑 / 渲染 / 提取不用各自再兼容一遍；AI 那份原话在 `raw_json` 里留档，追溯不受影响。

这一层只做「读协议 / 校验 / 回写」，不解码、不查库、不引重依赖（av、cv2 都不许进来），
所以 GUI 进程也能安全 import。
"""

from __future__ import annotations

import json
import math
from typing import Any

# 顶层的块名。EXTRA_KEYS 里的都是成片时间域的附加信息，剪辑不看，原样保管
TIMELINE_KEY = "timeline"
SEGMENTS_KEY = "segments"
EXTRA_KEYS = ("o", "s", "w", "a", "e", "t")
# 剪辑时盖进 JSON 的三个参数（全部来自「剪辑高光」窗，进剪辑环节就盖，有就替换、没有就新增）：
#   startframe  = 起始加减秒数（起剪点 = sa + 本值）
#   freeze      = 结束加减秒数（结束点 = end + 本值）
#   freezeframe = 末帧冻结秒数（config.json 的 highlight.freeze_tail_seconds）
# 这三个都**不算在 AI 报的 timeline.duration 里**——duration 只是剪辑区间的长度，
# 成片真实时长由这三个参数现算（见 play_seconds / final_duration）。
# 老数据里多出来的参数键不在这张表上，读的时候就当没有，绝不因此报错
OFFSET_KEYS = ("startframe", "freeze")
PARAM_KEYS = ("startframe", "freeze", "freezeframe")
# 成片时间域的时间轨道：o 挂字、s 语音段、w 逐词、a 动作、e 表情。
# 这些时间都是「相对 AI 那条成片时间轴」算的（0 = segments[0].sa），
# 实际剪辑用了加减秒数之后必须整条平移，否则字和声音就对不上画面了。t 没有时间，不在这里
TRACK_KEYS = ("o", "s", "w", "a", "e")
# timeline 里的文案字段（score 单独当数字处理）
TIMELINE_TEXT_KEYS = ("type", "reason")


def as_dict(payload: Any) -> dict[str, Any] | None:
    """把 JSON 文本或对象收成 dict；不是对象就返回 None。"""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def _scan_objects(text: str) -> list[str]:
    """括号配平扫描：从一段文本里抠出并排的顶层 `{...}`，返回每一段的原文。

    只认花括号的深度，且**必须跳过字符串字面量**——JSON 的文案里经常带 `{` `}`
    （比如 reason 里写了个表情或者模板占位符），不跳的话深度就乱了，两份 JSON 会被
    粘成一份。`\\` 在字符串里只管转义下一个字符，看到 `\\"` 别当成引号结束。
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for index, char in enumerate(text):
        if in_str:                       # 字符串里面：只关心怎么出去
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:index + 1])
                start = -1
    return out


def _split_raw(payload: Any) -> list[dict[str, Any]]:
    """只管拆、只留 dict，**不升级、不校验**（升级统一交给 split_payloads 收口）。"""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_split_raw(item))
        return out
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        # a. 整段就是一份 JSON（对象或数组），最常见，先试它
        try:
            return _split_raw(json.loads(text))
        except (TypeError, ValueError):
            pass
        # b. JSONL：一行一份，某行是垃圾就跳过，不带累别的行
        out = []
        for line in text.splitlines():
            data = as_dict(line.strip())
            if data is not None:
                out.append(data)
        if out:
            return out
        # c. 兜底：括号配平抠出并排的 `{...}`，逐个解
        for chunk in _scan_objects(text):
            data = as_dict(chunk)
            if data is not None:
                out.append(data)
        return out
    return []


def split_payloads(payload: Any) -> list[dict[str, Any]]:
    """一次回复拆成一份份 JSON 对象（**多份入库的唯一入口**）。

    AI 有时候一口气回好几份高光方案，写法还不统一。这里把所有见过的写法都收成
    「dict 列表」，每份各自入库成一个高光方案：

      - `dict`   → 单条 dict 原样包一层，老调用一个字都不用改（最要紧的一条）；
      - `list` / `tuple` → 递归摊平，只留里面的 dict，顺序照抄；
      - `str`    → 先整段 `json.loads`；不行就按 JSONL 一行一份试；还是一份都没有，
                   就用括号配平从文本里抠并排的顶层 `{...}`（AI 爱把几份 JSON 直接贴一块，
                   中间连逗号都没有，这一步专治这个）；
      - 其它（None、数字、纯文本…）→ 空列表。

    拆完在**收进结果的这一刻**统一走一遍 `payload_of`：老协议的那几份就地升级成新形状，
    所以入库存进 `current_json` 的一律是新协议，下游剪辑 / 渲染 / 提取照旧工作。
    已经是新形状的原样返回同一个 dict，不复制、不改写。

    只管拆、只管升级，**一个字段都不校验** —— 该不该剪交给 `validate` / `clips` 说。
    """
    out: list[dict[str, Any]] = []
    for data in _split_raw(payload):
        upgraded = payload_of(data)
        if upgraded is not None:
            out.append(upgraded)
    return out


def split_objects(payload: Any) -> list[dict[str, Any]]:
    """拆成一份份 dict，**不升级、不校验**（`split_payloads` 少了升级那一步）。

    给「AI 只给时间点的结果清单」用：那种行里没有 `clip`，走不了协议升级，
    但拆分逻辑（整段 JSON / JSONL / 并排 `{...}`）和高光方案一模一样，不该抄第二份。
    """
    return _split_raw(payload)


def num(value: Any) -> float | None:
    """能当秒数用就返回 float，否则 None。bool / NaN / inf 一律不算数字。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _from_legacy(data: dict[str, Any]) -> dict[str, Any] | None:
    """老协议 → 新形状；不是老形状就返回 None（调用方原样用回旧 dict）。

    老形状长这样（用户实际在用的「高光候选池」PRM 输出的就是它）：

        {"video": "x.mp4", "clip": {"start": 5.25, "end": 7.33, "duration": 2.08,
         "score": 92, "type": "悬念揭晓", "reason": "...",
         "overlays": {"word": {"time": 6.49, "text": "SECRET?", "kind": "word"},
                      "emoji": {...}, "comment": {...}, "evaluation": "..."},
         "Trimclip": {"Scene": "...", "Action": "...", "Speech text": "..."}}}

    三条同时满足才认：没有可用的 `segments`、`clip` 是 dict、`clip.start/end` 是有效区间。

    **最容易搞错的一点**：`overlays` 里的 `time` 是**原视频绝对时间**（样例里 6.49 落在
    5.25~7.33 中间），新协议的 `o` 是成片时间域（0 = `sa`），所以必须减掉 `start`
    再裁进 `[0, span]`。老协议没有的 `s` / `w` / `a` / `e` 一条都不造。
    """
    items = data.get(SEGMENTS_KEY)
    if isinstance(items, list) and any(isinstance(item, dict) for item in items):
        return None                      # 已经是新形状，不碰
    clip = data.get("clip")
    if not isinstance(clip, dict):
        return None
    start, end = num(clip.get("start")), num(clip.get("end"))
    if start is None or end is None or end <= start:
        return None
    # 成片时长：AI 报的 duration 优先，写得不对就按区间长度现算
    given = num(clip.get("duration"))
    span = round(given, 3) if given is not None and given > 0 else round(end - start, 3)
    timeline: dict[str, Any] = {"duration": span}
    score = num(clip.get("score"))
    if score is not None:
        timeline["score"] = score
    # `scores` 是 AI 打的主观四维，整块搬过来，**不合成成一个总分** ——
    # 第二轮混剪要按当次目标自己加权，合成分会把需求压平。
    # 程序算的客观数字不进这里：它们摆在剧本里给 AI 当依据看，
    # 同一个数在两处各算一遍早晚不同步。
    block = clip.get("scores")
    if isinstance(block, dict) and block:
        timeline["scores"] = dict(block)
    for key in TIMELINE_TEXT_KEYS:
        if clip.get(key):
            timeline[key] = str(clip[key])
    out: dict[str, Any] = {}
    if data.get("video"):
        out["video"] = data["video"]
    out[TIMELINE_KEY] = timeline
    out[SEGMENTS_KEY] = [{"sa": round(start, 3), "end": round(end, 3),
                          "dst": [0.0, span]}]
    overlays = clip.get("overlays")
    # o：挂字轨道，时间从原视频绝对时间换算到成片时间域
    marks: list[list[Any]] = []
    if isinstance(overlays, dict):
        for key in ("word", "emoji", "comment"):
            item = overlays.get(key)
            if not isinstance(item, dict):
                continue
            moment = num(item.get("time"))
            if moment is None:
                continue
            moment = round(min(max(0.0, moment - start), span), 2)
            marks.append([moment, str(item.get("kind") or key),
                          str(item.get("text") or "")])
    if marks:
        out["o"] = marks
    # t：纯描述块。Trimclip 正常挂在 clip 里，也容忍写在顶层（PRM 版本不一致），clip 优先
    trim = clip.get("Trimclip")
    if not isinstance(trim, dict):
        trim = data.get("Trimclip")
    note: dict[str, Any] = dict(trim) if isinstance(trim, dict) else {}
    if isinstance(overlays, dict):
        text = overlays.get("evaluation")
        if isinstance(text, str) and text.strip():
            note["evaluation"] = text
    if note:
        out["t"] = note
    # 三个剪辑参数：老数据可能盖在顶层，也可能盖在 clip 上，顶层优先。
    # 表上没有的键（老版本盖过的参数）一律不带过来，升级完的形状里就不该再有它
    for key in PARAM_KEYS:
        value = num(data.get(key))
        if value is None:
            value = num(clip.get(key))
        if value is not None:
            out[key] = value
    return out


def payload_of(payload: Any) -> dict[str, Any] | None:
    """**读一份 JSON 的唯一入口**：收成 dict，是老协议就顺手升级成新形状。

    已经是新形状的原样返回同一个 dict（不复制、不改写），所以新协议的行为一个字不变。
    """
    data = as_dict(payload)
    if data is None:
        return None
    upgraded = _from_legacy(data)
    return upgraded if upgraded is not None else data


def raw_segments(payload: Any) -> list[dict[str, Any]]:
    """`segments` 里的原始条目（只留 dict），一个都没有就是空列表。"""
    data = payload_of(payload)
    if data is None:
        return []
    items = data.get(SEGMENTS_KEY)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def timeline_of(payload: Any) -> dict[str, Any]:
    """`timeline` 块；没给或者写错类型就返回空 dict（调用方各自兜底）。"""
    data = payload_of(payload)
    block = data.get(TIMELINE_KEY) if data else None
    return block if isinstance(block, dict) else {}


def segment_times(segment: dict[str, Any]) -> tuple[float, float] | None:
    """一条 segment 的原视频区间 (sa, end)；缺字段、不是数字、end<=sa 都返回 None。"""
    start, end = num(segment.get("sa")), num(segment.get("end"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def dst_of(segment: dict[str, Any]) -> tuple[float, float] | None:
    """一条 segment 的成片落位 [dst0, dst1]；写得不对就 None（只用于参考/自检）。"""
    span = segment.get("dst")
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        return None
    first, second = num(span[0]), num(span[1])
    if first is None or second is None or second < first:
        return None
    return first, second


def clips(payload: Any, *, source_video: str = "") -> list[dict[str, Any]]:
    """协议 → 内部规范片段（**唯一的转换点**）。

    每条返回：`start` / `end`（原视频时间）、`duration`（成片时长，timeline 优先）、
    `score` / `type` / `reason`（来自 timeline）、`video`、`dst`，以及原样带着的
    `o` / `s` / `w` / `a` / `e` / `t`。下游算法（剪辑引擎、入库、展示）都吃这个形状，
    协议再变也只用改这一处。

    现阶段**只取第一条 segment**：多段拼接还没做，多给的段会被丢掉（`dropped_segments`
    记下丢了几条，调用方可以据此警告）。
    """
    data = payload_of(payload)
    if data is None:
        return []
    items = raw_segments(data)
    if not items:
        return []
    times = segment_times(items[0])
    if times is None:
        return []
    start, end = times
    timeline = timeline_of(data)
    duration = num(timeline.get("duration"))
    out: dict[str, Any] = {
        "start": start,
        "end": end,
        "duration": duration if duration is not None else round(end - start, 3),
        "score": num(timeline.get("score")),
        "type": str(timeline.get("type") or ""),
        "reason": str(timeline.get("reason") or ""),
        "video": str(data.get("video") or source_video or ""),
        "dst": list(dst_of(items[0]) or ()),
        "dropped_segments": max(0, len(items) - 1),
    }
    for key in EXTRA_KEYS:
        if key in data:
            out[key] = data[key]
    for key in OFFSET_KEYS:
        value = num(data.get(key))
        if value is not None:
            out[key] = value
    value = num(data.get("freezeframe"))
    if value is not None:
        out["freezeframe"] = value
    return [out]


def validate(payload: Any) -> str | None:
    """能不能剪：能就返回 None，不能就返回一句中文原因（给日志和弹窗用）。"""
    data = payload_of(payload)

    if data is None:
        return "JSON 根节点必须是对象"
    items = raw_segments(data)
    if not items:
        return f"缺少 {SEGMENTS_KEY}：新协议的剪辑区间写在 segments[].sa / segments[].end"
    if segment_times(items[0]) is None:
        return (f"{SEGMENTS_KEY}[0] 的 sa / end 不是有效区间"
                f"（sa={items[0].get('sa')!r} end={items[0].get('end')!r}）")
    return None


def build_payload(clip: dict[str, Any], *, start: float, end: float,
                  duration: float | None = None,
                  startframe: float | None = None,
                  freeze: float | None = None) -> dict[str, Any]:
    """把修正后的时间写回协议形状（渲染和 sidecar 都用这一份）。

    `start` / `end` 是原视频时间，落进 `segments[0].sa` / `.end`；
    `dst` 按成片重算成 `[0, duration]`——末帧冻结是渲染时追加的，不写进 dst。
    文案（type / reason / t / o / …）一个字都不改。

    `startframe` / `freeze` 给了就盖进去（有就替换），没给就沿用 clip 里原有的值——
    一进剪辑环节就该盖上，这样 JSON 自己记得住这一次是按什么加减秒数剪的。
    """
    span = round((duration if duration is not None else end - start), 3)
    timeline: dict[str, Any] = {"duration": span}
    if clip.get("score") is not None:
        timeline["score"] = clip["score"]
    for key in TIMELINE_TEXT_KEYS:
        if clip.get(key):
            timeline[key] = clip[key]
    out: dict[str, Any] = {}
    if clip.get("video"):
        out["video"] = clip["video"]
    out[TIMELINE_KEY] = timeline
    out[SEGMENTS_KEY] = [{"sa": round(start, 3), "end": round(end, 3),
                          "dst": [0.0, span]}]
    for key in EXTRA_KEYS:
        if key in clip:
            out[key] = clip[key]
    for key, given in zip(OFFSET_KEYS, (startframe, freeze)):
        value = num(given) if given is not None else num(clip.get(key))
        if value is not None:
            out[key] = round(value, 2)
    return out


def stamp_offsets(payload: Any, startframe: float, freeze: float, *,
                  freezeframe: float | None = None) -> dict[str, Any] | None:
    """把这一次剪辑用的参数盖进整份 JSON，返回改过的副本；不是对象就返回 None。

    三个参数（`startframe` / `freeze` / `freezeframe`）**有就替换、没有就新增**，
    一律以「剪辑高光」窗里的当前配置为准 —— JSON 里的旧值绝不参与这一次剪辑。
    `freezeframe` 没传就不动 JSON 里原有的（老数据兼容）。别的键一个字不改，
    老数据里多出来的参数键也原样留着、不解释也不报错；`raw_json` 那份 AI 原话在库里
    另存，永远不受影响。
    """
    data = as_dict(payload)
    if data is None:
        return None
    out = dict(data)
    out["startframe"] = round(float(startframe), 2)
    out["freeze"] = round(float(freeze), 2)
    if freezeframe is not None:
        out["freezeframe"] = round(max(0.0, float(freezeframe)), 2)
    return out


# ====================================================== 按实际剪辑区间重算时间轴
def play_seconds(clip: dict[str, Any], *, startframe: float, freeze: float) -> float | None:
    """实际播放段时长 = (end + freeze) − (sa + startframe)。算不出来就 None。

    冻帧不算在这里——它是渲染时接在播放段后面的，见 `final_duration`。
    """
    start, end = num(clip.get("start")), num(clip.get("end"))
    if start is None or end is None:
        return None
    span = (end + float(freeze)) - (start + float(startframe))
    return round(span, 2) if span > 0 else None


def final_duration(play: float, *, freeze_tail: float = 0.0) -> float:
    """成片真实总时长 = 播放段 + 末帧冻结。"""
    return round(float(play) + max(0.0, float(freeze_tail)), 2)


def duration_tag(seconds: float) -> str:
    """把时长写成文件名安全的样子：6.89 秒 → `689`（两位小数，小数点直接去掉）。

    成品名的时长口径就用它（`highlight/clip.py` 的 `default_target`），
    数据库那边批量改名也用同一个函数——只有一个真源，不许各写一份。
    """
    return f"{max(0.0, float(seconds)):.2f}".replace(".", "")


def _retime_item(item: Any, delta: float, span: float) -> Any:
    """一条轨道记录整体平移 delta 秒，并裁进 [0, span]；完全落在外面就返回 None。

    认两种写法（都是数组）：
      - 区间型 `[start, end, ...]`：前两个是数字，两端一起平移；
      - 单点型 `[time, kind, text]`：只有第一个是数字，平移它。
    其余字段（文字、标签、分值）原样带走，一个字不改。
    """
    if not isinstance(item, (list, tuple)) or not item:
        return item
    first = num(item[0])
    if first is None:
        return item
    second = num(item[1]) if len(item) > 1 else None
    out = list(item)
    if second is not None:          # 区间型
        start, end = first + delta, second + delta
        if end <= 0 or start >= span:
            return None             # 整段落在成片外面
        out[0] = round(max(0.0, start), 2)
        out[1] = round(min(span, end), 2)
        return out
    moment = first + delta          # 单点型
    if moment < -0.005 or moment > span + 0.005:
        return None
    out[0] = round(min(max(0.0, moment), span), 2)
    return out


def retime_tracks(clip: dict[str, Any], *, startframe: float, freeze: float,
                  span: float | None = None) -> dict[str, list[Any]]:
    """把 o / s / w / a / e 五条轨道按实际起剪点重算，返回 {键: 新数组}。

    AI 的时间是按它自己那条成片时间轴给的（0 = `sa`）。实际起剪点是 `sa + startframe`，
    所以每个时间都要减掉 `startframe`（起始填 -0.55 = 提前起剪，事件整体后移 0.55 秒）。
    `span` 是播放段长度，用来把越界的记录裁掉——`freeze` 收窄了尾巴时，落在外面的字
    不该还留在数据里。轨道里没有的键不出现在返回值里。
    """
    limit = span if span is not None else play_seconds(
        clip, startframe=startframe, freeze=freeze)
    if limit is None:
        limit = 0.0
    delta = -float(startframe)
    out: dict[str, list[Any]] = {}
    for key in TRACK_KEYS:
        items = clip.get(key)
        if not isinstance(items, list):
            continue
        moved = [_retime_item(item, delta, float(limit)) for item in items]
        out[key] = [item for item in moved if item is not None]
    return out
