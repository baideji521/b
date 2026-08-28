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

# 画面情绪词表：模型只准从这些英文标签里选。
# 和 action/scene/subjects 同一条规矩——内部事实固定英文；
# 显示名由 emotions.label_for() 按 output_language 决定，翻译层不碰它。
VISUAL_EMOTIONS: tuple[str, ...] = (
    "happy", "excited", "surprised", "angry", "sad", "fearful", "disgusted", "calm", "neutral",
)
EMOTION_VOCAB = "|".join(VISUAL_EMOTIONS)

_SCHEMA_ZH_HEAD = (
    '{"events":[{"start":6.8,"end":9.1,"event":"事件短语(<=15字)","description":"客观描述(<=30字)",'
    '"action":"english_snake_case","scene":"english_scene","subjects":["man","cup"],'
    '"importance":"low|normal|high|critical","confidence":0.9,"ocr_text":null'
)

_SCHEMA_EN_HEAD = (
    '{"events":[{"start":6.8,"end":9.1,"event":"short phrase(<=8 words)","description":"objective description(<=20 words)",'
    '"action":"english_snake_case","scene":"english_scene","subjects":["man","cup"],'
    '"importance":"low|normal|high|critical","confidence":0.9,"ocr_text":null'
)

# 情绪字段拼在 schema 末尾：关掉画面情绪时一个字都不多要，输出 token 不涨
_EMOTION_FIELD = f',"emotion":"{EMOTION_VOCAB}","emotion_intensity":0.8'


def _schema(lang: str, with_emotion: bool) -> str:
    head = _SCHEMA_ZH_HEAD if lang == "zh" else _SCHEMA_EN_HEAD
    return head + (_EMOTION_FIELD if with_emotion else "") + "}]}"


def system_prompt(output_language: str | None) -> str:
    return SYSTEM_PROMPT_ZH if normalize_code(output_language) == "zh" else SYSTEM_PROMPT_EN


def build_user_prompt(window_start: float, window_end: float, timestamps: list[float],
                      previous_summary: str | None = None,
                      output_language: str = "zh",
                      timestamp_mode: str = "markers",
                      with_emotion: bool = True) -> str:
    """构造窗口级提示词。时间戳用绝对秒数，与最终 timeline 保持同一坐标系。

    timestamp_mode:
      - "markers"：Qwen3-VL 会在每帧前注入 `<x.x seconds>`，提示词只需说明它是真实时间
      - "list"   ：MiniCPM-V 等没有原生时间戳注入的模型，把每帧真实秒数显式列出来

    with_emotion：让模型顺便判断画面里人物的情绪。走的是同一次推理，
    只多出两个字段的解码开销，不额外加载模型。

    输出要求刻意压缩：这台机器上解码速度受单步开销限制，输出 token 越少越快。
    """
    lang = normalize_code(output_language) or "zh"
    if lang == "zh":
        return _prompt_zh(window_start, window_end, timestamps, previous_summary,
                          timestamp_mode, with_emotion)
    return _prompt_en(window_start, window_end, timestamps, previous_summary, lang,
                      timestamp_mode, with_emotion)



def _ts_list(timestamps: list[float], limit: int = 24) -> str:
    items = [f"{t:.1f}" for t in timestamps[:limit]]
    tail = " ..." if len(timestamps) > limit else ""
    return ", ".join(items) + tail


def _prompt_zh(window_start: float, window_end: float, timestamps: list[float],
               previous_summary: str | None, timestamp_mode: str = "markers",
               with_emotion: bool = True) -> str:
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
        f"4. start/end 用 {window_start:.1f} 到 {window_end:.1f} 之间的绝对秒数，升序且不重叠；"
        "必须把整个窗口连续覆盖完（第一条从窗口开头起，最后一条到窗口结尾止，前一条的 end 就是后一条的 start），不要留空档。",
        "5. importance：日常/静态 normal；背景轻微变化 low；摔倒/碰撞/掉落/突然进出 high；爆炸/事故/场景剧变 critical。",
        "6. 最多 5 条事件，description 不超过 30 字，只描述看得见的内容，不要猜测声音或说话内容。",
        "7. event 和 description 必须用中文；action / scene / subjects 必须用英文小写标签"
        "（如 action=\"picking_up\"、scene=\"indoor_room\"、subjects=[\"man\",\"cup\"]）。",
    ]
    if with_emotion:
        parts.append(
            f"8. emotion 是画面里人物此刻表现出来的情绪，只能从 {EMOTION_VOCAB} 里选一个英文标签，"
            "依据是表情、姿态、动作幅度；emotion_intensity 是这个情绪的明显程度，0~1 的小数。"
            "画面里没有人、或者看不出情绪，就写 emotion=\"neutral\"、emotion_intensity=0。"
        )
    parts += [
        f"{9 if with_emotion else 8}. 只输出一行压缩 JSON，不要换行缩进，不要 markdown 代码块，结构如下：",
        _schema("zh", with_emotion),
    ]
    return "\n".join(parts)


def _prompt_en(window_start: float, window_end: float, timestamps: list[float],
               previous_summary: str | None, lang: str, timestamp_mode: str = "markers",
               with_emotion: bool = True) -> str:
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
        "ascending and non-overlapping; together they must cover the whole window with no gaps "
        "(first event starts at the window start, last one ends at the window end, "
        "each event's end is the next one's start).",
        "5. importance: everyday/static -> normal; minor background change -> low; "
        "fall/collision/drop/sudden entry -> high; explosion/accident/drastic change -> critical.",
        "6. At most 5 events; description <= 20 words; describe only what is visible; never guess audio or speech.",
        f"7. Write event and description in {target}. Keep action / scene / subjects as lowercase English labels "
        '(e.g. action="picking_up", scene="indoor_room", subjects=["man","cup"]).',
    ]
    if with_emotion:
        parts.append(
            f"8. emotion is the emotion the people on screen are showing; pick exactly one label from "
            f"{EMOTION_VOCAB}, judged from facial expression, posture and motion. "
            "emotion_intensity is how pronounced it is, a decimal in 0~1. "
            'If nobody is visible or no emotion is readable, use emotion="neutral" and emotion_intensity=0.'
        )
    parts += [
        f"{9 if with_emotion else 8}. Output a single line of compact JSON - no line breaks, no markdown fences:",
        _schema(lang, with_emotion),
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
        f"You translate short lines of video text (event descriptions or subtitles) into {target}. "
        "Keep the meaning identical, keep each line short, do not add or drop information. "
        "Keep numbers, names and brand words as they are. "
        "Output only a JSON array of strings, same length and order as the input."
    )
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (
        f"Translate each line into {target}. Return only a JSON array of {len(texts)} strings.\n{numbered}"
    )
    return system, user


def build_line_prompt(output_language: str) -> str:
    """一行一条序列的翻译提示词（batch 维度并行用），只出译文本身。

    和 build_rewrite_prompt 的区别：那个把 N 行塞进一个提示词、要求输出 JSON 数组，
    译文是串行解码出来的；这个每行独立成一条序列，N 行一起解码，
    实测 20 行 35.5s -> 3.5s（这台机器的逐 token 开销由 kernel 启动主导）。
    """
    target = language_name(output_language)
    return (
        f"Translate the line into {target}. "
        "Keep the meaning identical, keep it short, do not add or drop information. "
        "Keep numbers, names and brand words as they are. "
        "Output only the translation itself: no quotes, no explanation, no original text."
    )
