"""最终自然语言的选择与生成（Language Decider + Language Renderer）。

规则来自需求：最终用户可见的自然语言由**原始音频语言**决定，由程序判定，
不交给视觉模型自己发挥；内部结构化事实（action / scene / subjects）固定英文。

- Decider：faster-whisper 的检测结果 -> dominant_language -> output_language
- Renderer：把事件的最终描述统一到 output_language；语种不符时按顺序降级
  1) 模型改写（视觉模型还在显存里时，见 QwenVLAnalyzer.rewrite_texts）
  2) 内部英文事实 + 小词表模板
  3) 保留原文并标记 language_fallback=True（不伪造，交给报告如实说明）
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .logging_setup import get_logger

logger = get_logger(__name__)

# 第一阶段重点保证 zh / en，其余语言给出提示词名称即可自然支持
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
}

# whisper 可能返回的变体码 -> 统一码
_ALIASES = {
    "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "yue": "zh",  # 粤语音频最终输出中文
    "en-us": "en", "en-gb": "en",
    "pt-br": "pt",
    "nn": "no", "nb": "no",
}

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_THAI = re.compile(r"[\u0e00-\u0e7f]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def normalize_code(code: str | None) -> str | None:
    if not code:
        return None
    key = str(code).strip().lower().replace("_", "-")
    return _ALIASES.get(key, key.split("-")[0] or None)


def language_name(code: str | None) -> str:
    code = normalize_code(code)
    return LANGUAGE_NAMES.get(code or "", (code or "unknown").upper())


def scripts_in(text: str) -> set[str]:
    """文本里出现了哪些书写系统。用于交叉验证 whisper 的语言判定。"""
    found: set[str] = set()
    if not text:
        return found
    if _KANA.search(text):
        found.add("kana")
    if _HANGUL.search(text):
        found.add("hangul")
    if _CJK.search(text):
        found.add("cjk")
    if _CYRILLIC.search(text):
        found.add("cyrillic")
    if _ARABIC.search(text):
        found.add("arabic")
    if _THAI.search(text):
        found.add("thai")
    if _LATIN_WORD.search(text):
        found.add("latin")
    return found


def text_matches_language(text: str, code: str | None) -> bool:
    """粗粒度校验一段文本是否是目标语言的书写系统。

    只做"明显不符"的判断：中文输出里不该整段都是拉丁字母，英文输出里不该出现汉字。
    专有名词/型号（例如 "iPhone 15"）不会被判为不符，因为中文分支只要求出现汉字。
    """
    code = normalize_code(code)
    if not code or not text or not text.strip():
        return True
    found = scripts_in(text)
    if code == "zh":
        return "cjk" in found and "hangul" not in found
    if code == "ja":
        return bool(found & {"kana", "cjk"})
    if code == "ko":
        return "hangul" in found
    if code == "ru":
        return "cyrillic" in found
    if code == "ar":
        return "arabic" in found
    if code == "th":
        return "thai" in found
    # 拉丁语系：只要不含 CJK/韩文即可
    return not (found & {"cjk", "kana", "hangul"})


def _secondary_from_text(text: str, primary: str) -> set[str]:
    """在主语言之外还夹杂了什么（中文里夹英文单词等）。"""
    found = scripts_in(text)
    extra: set[str] = set()
    if primary == "zh" and "latin" in found:
        extra.add("en")
    if primary in ("en", "es", "fr", "de", "pt", "it") and "cjk" in found:
        extra.add("zh")
    if primary != "ja" and "kana" in found:
        extra.add("ja")
    if primary != "ko" and "hangul" in found:
        extra.add("ko")
    return extra


@dataclass
class LanguageDecision:
    audio_available: bool
    detected_language: str | None            # whisper 直接给出的语言
    language_confidence: float | None        # whisper 的 language_probability
    dominant_language: str | None            # 按时长加权投票后的主语言
    secondary_languages: list[str] = field(default_factory=list)
    output_language: str = "zh"              # 最终自然语言，由程序决定
    default_used: bool = False               # 是否用了配置里的兜底语言
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_output_language(speech_payload: dict[str, Any], *, default_language: str = "zh",
                           min_confidence: float = 0.4) -> LanguageDecision:
    """由程序决定最终输出语言。必须在视觉分析之前调用。"""
    default_language = normalize_code(default_language) or "zh"
    available = bool(speech_payload.get("available"))
    segments = [s for s in speech_payload.get("segments", []) if (s.get("text") or "").strip()]
    detected = normalize_code(speech_payload.get("language"))
    confidence = speech_payload.get("language_probability")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else None

    if not available or not segments:
        reason = speech_payload.get("reason") or ("no_speech_segments" if available else "no_audio")
        decision = LanguageDecision(
            audio_available=available, detected_language=detected, language_confidence=confidence,
            dominant_language=None, output_language=default_language, default_used=True,
            reason=f"{reason} -> 使用默认语言 {default_language}",
        )
        logger.info("没有可用语音（%s），最终语言使用默认值 %s", reason, decision.output_language)
        return decision

    # 按时长加权投票：多数视频只有一个语言，但混说时要看谁占主导
    votes: dict[str, float] = {}
    secondary: set[str] = set()
    for seg in segments:
        code = normalize_code(seg.get("language")) or detected or default_language
        span = max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.01)
        votes[code] = votes.get(code, 0.0) + span
        secondary |= _secondary_from_text(seg.get("text", ""), code)

    dominant = max(votes, key=lambda k: votes[k])
    secondary.discard(dominant)

    output = dominant
    default_used = False
    reason = f"dominant={dominant} (时长占比 {votes[dominant] / sum(votes.values()):.2f})"
    if dominant not in LANGUAGE_NAMES:
        output, default_used = default_language, True
        reason += f"；该语言暂未支持最终渲染，改用默认 {default_language}"
    elif confidence is not None and confidence < min_confidence:
        output, default_used = default_language, True
        reason += f"；语言置信度 {confidence:.2f} < {min_confidence:.2f}，改用默认 {default_language}"

    decision = LanguageDecision(
        audio_available=True, detected_language=detected, language_confidence=confidence,
        dominant_language=dominant, secondary_languages=sorted(secondary),
        output_language=output, default_used=default_used, reason=reason,
    )
    logger.info(
        "语言判定：detected=%s(%.2f) dominant=%s secondary=%s -> output_language=%s",
        detected, confidence or 0.0, dominant, sorted(secondary) or "-", output,
    )
    return decision


# ------------------------------------------------------------------ Renderer
# 内部固定英文动作标签 -> 中文短语。只覆盖高频动作，命中不了就走标记降级。
_ACTION_ZH = {
    "walking": "走动", "walk": "走动", "running": "跑动", "standing": "站着", "stand": "站着",
    "sitting": "坐着", "sit": "坐着", "talking": "说话", "speaking": "说话", "looking": "注视",
    "holding": "拿着", "hold": "拿着", "picking_up": "拿起", "pick_up": "拿起",
    "putting_down": "放下", "put_down": "放下", "eating": "吃东西", "drinking": "喝东西",
    "entering": "进入画面", "exiting": "离开画面", "smiling": "微笑", "pointing": "指向",
    "turning": "转身", "falling": "跌倒", "opening": "打开", "closing": "关闭",
    "showing": "展示", "handing": "递东西", "waving": "挥手", "dancing": "跳舞",
    "typing": "打字", "driving": "开车", "writing": "书写", "reading": "阅读",
}

_SUBJECT_ZH = {
    "man": "男子", "woman": "女子", "person": "人物", "people": "多人", "boy": "男孩",
    "girl": "女孩", "child": "小孩", "baby": "婴儿", "dog": "狗", "cat": "猫",
    "car": "汽车", "cup": "杯子", "phone": "手机", "table": "桌子", "chair": "椅子",
    "door": "门", "food": "食物", "hand": "手", "book": "书", "bag": "包",
    "bottle": "瓶子", "screen": "屏幕", "camera": "镜头",
}

_LABELS = {
    "zh": {"visual": "画面", "speech": "语音", "ocr": "画面文字", "no_speech": "无语音"},
    "en": {"visual": "Visual", "speech": "Speech", "ocr": "On-screen text", "no_speech": "no speech"},
    "ja": {"visual": "映像", "speech": "音声", "ocr": "画面テキスト", "no_speech": "音声なし"},
    "ko": {"visual": "화면", "speech": "음성", "ocr": "화면 텍스트", "no_speech": "음성 없음"},
}


def labels_for(output_language: str | None) -> dict[str, str]:
    return _LABELS.get(normalize_code(output_language) or "zh", _LABELS["en"])


def _humanize(token: str) -> str:
    return re.sub(r"[_\-]+", " ", str(token or "")).strip()


class LanguageRenderer:
    """最终自然语言层：内部事实 + 模型描述 -> output_language 的统一文本。"""

    def __init__(self, output_language: str):
        self.output_language = normalize_code(output_language) or "zh"
        self.mismatched = 0
        self.rewritten = 0
        self.fallback = 0

    # --- 判断 ---
    def matches(self, text: str | None) -> bool:
        return text_matches_language(text or "", self.output_language)

    def needs_rewrite(self, events: list[Any]) -> list[int]:
        """返回描述语种不符的事件下标。"""
        bad = [i for i, ev in enumerate(events)
               if (ev.description and not self.matches(ev.description))
               or (ev.event and not self.matches(ev.event))]
        self.mismatched = len(bad)
        if bad:
            logger.warning("%d/%d 个视觉事件的描述不是 %s，需要改写",
                           len(bad), len(events), language_name(self.output_language))
        return bad

    # --- 生成 ---
    def apply_rewrite(self, event: Any, text: str | None) -> bool:
        """把模型改写结果写回事件；仍不合格则返回 False。"""
        if not text or not self.matches(text):
            return False
        event.description = text.strip()
        if not self.matches(event.event):
            event.event = text.strip()[:20]
        event.description_language = self.output_language
        self.rewritten += 1
        return True

    def template_from_facts(self, event: Any) -> str | None:
        """用内部英文事实拼出目标语言描述（模型改写不可用时的兜底）。"""
        action = (event.action or "").strip().lower()
        subjects = [s for s in (event.subjects or []) if s]
        scene = (event.scene or "").strip()
        if not action and not subjects:
            return None
        if self.output_language == "zh":
            act_zh = _ACTION_ZH.get(action)
            if not act_zh:
                return None
            names = [_SUBJECT_ZH.get(s) for s in subjects]
            names = [n for n in names if n]
            who = "、".join(names) if names else "画面主体"
            text = f"{who}{act_zh}"
            return f"{text}（{_SUBJECT_ZH.get(scene, scene)}）" if scene and _SUBJECT_ZH.get(scene) else text
        # 英文及其它拉丁语系：内部标签本身就是英文，直接组句
        who = " and ".join(_humanize(s) for s in subjects) if subjects else "subject"
        text = f"{who} {_humanize(action)}".strip()
        return f"{text} ({_humanize(scene)})" if scene else text

    def finalize_event(self, event: Any) -> Any:
        """事件级最终渲染：合格直接用，不合格依次尝试模板、标记降级。"""
        if self.matches(event.description) and self.matches(event.event):
            event.description_language = self.output_language
            return event
        templated = self.template_from_facts(event)
        if templated and self.matches(templated):
            event.description = templated
            event.event = templated[:20]
            event.description_language = self.output_language
            event.language_fallback = True
            self.fallback += 1
            return event
        # 不伪造翻译：保留原文，明确标记语言不符，报告里如实反映
        event.description_language = "mixed"
        event.language_fallback = True
        self.fallback += 1
        return event

    def finalize_events(self, events: list[Any]) -> list[Any]:
        return [self.finalize_event(ev) for ev in events]

    def stats(self) -> dict[str, Any]:
        return {
            "output_language": self.output_language,
            "mismatched": self.mismatched,
            "rewritten_by_model": self.rewritten,
            "template_or_kept": self.fallback,
        }
