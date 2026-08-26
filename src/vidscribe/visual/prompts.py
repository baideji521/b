"""Qwen3-VL 提示词：要求模型只回答"发生了什么"，时间由程序校准。"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "你是一个严谨的视频事件标注器。你只负责描述画面里客观发生的事情，"
    "不做主观推测，不编造画面中不存在的内容，不进行语音识别。"
    "你必须只输出 JSON，不要输出任何解释文字或 markdown 代码块标记。"
)

_SCHEMA = """{"events":[{"start":6.8,"end":9.1,"event":"事件短语(<=15字)","description":"客观描述(<=30字)","importance":"low|normal|high|critical","confidence":0.9,"ocr_text":null}]}"""


def build_user_prompt(window_start: float, window_end: float, timestamps: list[float],
                      previous_summary: str | None = None) -> str:
    """构造窗口级提示词。时间戳用绝对秒数，与最终 timeline 保持同一坐标系。

    输出要求刻意压缩：这台机器上解码速度受单步开销限制，输出 token 越少越快。
    """
    parts = [
        f"以上是同一段视频在 {window_start:.1f}s ~ {window_end:.1f}s 之间按时间顺序采样的 {len(timestamps)} 帧，"
        f"每帧前面的 <x.x seconds> 就是它的真实时间。",
    ]
    if previous_summary:
        parts.append(f"这段之前刚发生的内容（仅作上下文，不要重复输出）：{previous_summary}")
    parts += [
        "请把这段画面切分成连续事件，要求：",
        "1. 按连续动作理解，描述动作的变化过程，不要逐帧描述静态画面。",
        "2. 画面长时间没变化就只输出一个覆盖整段时间的事件，不要输出重复事件。",
        "3. 只在动作发生变化（移动、拿起/放下、进出画面、镜头切换、掉落等）时切分新事件。",
        f"4. start/end 用 {window_start:.1f} 到 {window_end:.1f} 之间的绝对秒数，升序且不重叠。",
        "5. importance：日常/静态 normal；背景轻微变化 low；摔倒/碰撞/掉落/突然进出 high；爆炸/事故/场景剧变 critical。",
        "6. 最多 5 条事件，description 不超过 30 字，只描述看得见的内容，不要猜测声音或说话内容。",
        "7. 只输出一行压缩 JSON，不要换行缩进，不要 markdown 代码块，结构如下：",
        _SCHEMA,
    ]
    return "\n".join(parts)


SUMMARY_MAX_CHARS = 220


def build_context_summary(events: list) -> str:
    """把上一个窗口的尾部事件压缩成上下文，保证跨窗口的场景连续性。"""
    if not events:
        return ""
    tail = events[-3:]
    items = [f"{ev.start:.1f}-{ev.end:.1f}s {ev.event}" for ev in tail]
    text = "；".join(items)
    return text[:SUMMARY_MAX_CHARS]
