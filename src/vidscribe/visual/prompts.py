"""Qwen3-VL 提示词：模型只回答"发生了什么"，时间由程序校准，语言由程序指定。

两条硬规则：
1. 最终自然语言（description/event）用程序传进来的 output_language，
   绝不写"用视频里的语言回答"——那样模型会在不同窗口之间反复摇摆。
2. 内部结构化事实（action/scene/subjects）固定英文小写标签，
   合并、去重、兜底渲染都依赖它们的稳定性。
"""

from __future__ import annotations

from ..language import language_name, normalize_code

SYSTEM_PROMPT_ZH = (
    "你是一个严谨的视频事件标注器。你只负责描述画面里客观发生的事情，"
    "不做主观推测，不编造画面中不存在的内容，不进行语音识别。"
    "你必须只输出 JSON，不要输出任何解释文字或 markdown 代码块标记。"
)

SYSTEM_PROMPT_EN = (
    "You are a precise video event annotator. Describe only what objectively happens on screen. "
    "Do not speculate, do not invent anything that is not visible, and never transcribe speech. "
    "Output JSON only - no explanations, no markdown code fences."
)

# 兼容旧引用
SYSTEM_PROMPT = SYSTEM_PROMPT_ZH

_SCHEMA_ZH = (
    '{"events":[{"start":6.8,"end":9.1,"event":"事件短语(<=15字)","description":"客观描述(<=30字)",'
    '"action":"english_snake_case","scene":"english_scene","subjects":["man","cup"],'
    '"importance":"low|normal|high|critical","confidence":0.9,"ocr_text":null}]}'
)

_SCHEMA_EN = (
    '{"events":[{"start":6.8,"end":9.1,"event":"short phrase(<=8 words)","description":"objective description(<=20 words)",'
    '"action":"english_snake_case","scene":"english_scene","subjects":["man","cup"],'
    '"importance":"low|normal|high|critical","confidence":0.9,"ocr_text":null}]}'
)


def system_prompt(output_language: str | None) -> str:
    return SYSTEM_PROMPT_ZH if normalize_code(output_language) == "zh" else SYSTEM_PROMPT_EN


def build_user_prompt(window_start: float, window_end: float, timestamps: list[float],
                      previous_summary: str | None = None,
                      output_language: str = "zh",
                      timestamp_mode: str = "markers") -> str:
    """构造窗口级提示词。时间戳用绝对秒数，与最终 timeline 保持同一坐标系。

    timestamp_mode:
      - "markers"：Qwen3-VL 会在每帧前注入 `<x.x seconds>`，提示词只需说明它是真实时间
      - "list"   ：MiniCPM-V 等没有原生时间戳注入的模型，把每帧真实秒数显式列出来

    输出要求刻意压缩：这台机器上解码速度受单步开销限制，输出 token 越少越快。
    """
    lang = normalize_code(output_language) or "zh"
    if lang == "zh":
        return _prompt_zh(window_start, window_end, timestamps, previous_summary, timestamp_mode)
    return _prompt_en(window_start, window_end, timestamps, previous_summary, lang, timestamp_mode)


def _ts_list(timestamps: list[float], limit: int = 24) -> str:
    items = [f"{t:.1f}" for t in timestamps[:limit]]
    tail = " ..." if len(timestamps) > limit else ""
    return ", ".join(items) + tail


def _prompt_zh(window_start: float, window_end: float, timestamps: list[float],
               previous_summary: str | None, timestamp_mode: str = "markers") -> str:
    if timestamp_mode == "list":
        head = (
            f"以上是同一段视频在 {window_start:.1f}s ~ {window_end:.1f}s 之间按时间顺序采样的 {len(timestamps)} 帧，"
            f"它们对应的真实时间（秒）依次是：{_ts_list(timestamps)}。"
        )
    else:
        head = (
            f"以上是同一段视频在 {window_start:.1f}s ~ {window_end:.1f}s 之间按时间顺序采样的 {len(timestamps)} 帧，"
            f"每帧前面的 <x.x seconds> 就是它的真实时间。"
        )
    parts = [head]
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
        "7. event 和 description 必须用中文；action / scene / subjects 必须用英文小写标签"
        "（如 action=\"picking_up\"、scene=\"indoor_room\"、subjects=[\"man\",\"cup\"]）。",
        "8. 只输出一行压缩 JSON，不要换行缩进，不要 markdown 代码块，结构如下：",
        _SCHEMA_ZH,
    ]
    return "\n".join(parts)


def _prompt_en(window_start: float, window_end: float, timestamps: list[float],
               previous_summary: str | None, lang: str, timestamp_mode: str = "markers") -> str:
    target = language_name(lang)
    if timestamp_mode == "list":
        head = (
            f"The frames above are {len(timestamps)} time-ordered samples of one video between "
            f"{window_start:.1f}s and {window_end:.1f}s. Their real timestamps in seconds are, in order: "
            f"{_ts_list(timestamps)}."
        )
    else:
        head = (
            f"The frames above are {len(timestamps)} time-ordered samples of one video between "
            f"{window_start:.1f}s and {window_end:.1f}s. The <x.x seconds> marker before each frame is its real time."
        )
    parts = [head]
    if previous_summary:
        parts.append(f"Context from just before this segment (do not repeat it): {previous_summary}")
    parts += [
        "Split this segment into continuous events. Requirements:",
        "1. Reason over continuous motion and describe how the action changes; do not describe frames one by one.",
        "2. If the scene stays unchanged for a long time, emit ONE event covering the whole span.",
        "3. Start a new event only when the action changes (movement, picking up/putting down, "
        "entering/leaving frame, shot cut, falling, etc.).",
        f"4. start/end must be absolute seconds between {window_start:.1f} and {window_end:.1f}, "
        "ascending and non-overlapping.",
        "5. importance: everyday/static -> normal; minor background change -> low; "
        "fall/collision/drop/sudden entry -> high; explosion/accident/drastic change -> critical.",
        "6. At most 5 events; description <= 20 words; describe only what is visible; never guess audio or speech.",
        f"7. Write event and description in {target}. Keep action / scene / subjects as lowercase English labels "
        '(e.g. action="picking_up", scene="indoor_room", subjects=["man","cup"]).',
        "8. Output a single line of compact JSON - no line breaks, no markdown fences:",
        _SCHEMA_EN,
    ]
    return "\n".join(parts)


SUMMARY_MAX_CHARS = 220


def build_context_summary(events: list) -> str:
    """把上一个窗口的尾部事件压缩成上下文，保证跨窗口的场景连续性。"""
    if not events:
        return ""
    tail = events[-3:]
    items = [f"{ev.start:.1f}-{ev.end:.1f}s {ev.event}" for ev in tail]
    text = "; ".join(items)
    return text[:SUMMARY_MAX_CHARS]


# ------------------------------------------------------- 语言改写（Renderer 用）
def build_rewrite_prompt(texts: list[str], output_language: str) -> tuple[str, str]:
    """把语种不符的描述批量改写成 output_language。返回 (system, user)。

    这是纯文本调用，复用已经在显存里的视觉模型，不额外加载翻译模型。
    """
    target = language_name(output_language)
    system = (
        f"You rewrite short video event descriptions into {target}. "
        "Keep the meaning identical, keep it short, do not add anything. "
        "Output only a JSON array of strings, same length and order as the input."
    )
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (
        f"Rewrite each line into {target}. Return only a JSON array of {len(texts)} strings.\n{numbered}"
    )
    return system, user
