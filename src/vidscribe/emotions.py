"""情绪标签的中英对照。

规矩和 action/scene/subjects 一致：内部标签固定英文小写（`emotion_en`），
显示层的 `emotion` 跟随 output_language——视频是英文的就出英文，中文的就出中文，
免得英文视频的结果里混进中文情绪词。

标签集合覆盖两路来源：
- 画面情绪：视觉模型只准从 visual 词表里选（见 visual/prompts.py）
- 语音情绪：emotion2vec+ 的九类（angry/disgusted/fearful/happy/neutral/other/sad/surprised/unk）
"""

from __future__ import annotations

from .language import normalize_code

EMOTION_ZH: dict[str, str] = {
    "happy": "开心",
    "excited": "兴奋",
    "surprised": "吃惊",
    "angry": "生气",
    "sad": "难过",
    "fearful": "害怕",
    "disgusted": "厌恶",
    "calm": "平静",
    "neutral": "中立",
    "other": "其他",
    "unk": "未知",
    "<unk>": "未知",
    "unknown": "未知",
}


def label_for(emotion_en: str | None, output_language: str | None) -> str | None:
    """英文标签 -> 显示名。非中文输出直接用英文标签，表里没有的原样返回。"""
    if not emotion_en:
        return None
    if normalize_code(output_language) == "zh":
        return EMOTION_ZH.get(emotion_en, emotion_en)
    return emotion_en


_ZH_TO_EN = {zh: en for en, zh in EMOTION_ZH.items()}


def to_english(name: str | None) -> str | None:
    """任意显示名 -> 英文标签。已经是英文标签的原样返回，认不出的返回 None。

    老结果里只存了显示名（可能是英文也可能是中文），靠这个反查就能重新按当前
    语言渲染，不必为了换语言去重跑模型。
    """
    if not name:
        return None
    text = str(name).strip()
    if text.lower() in EMOTION_ZH:
        return text.lower()
    return _ZH_TO_EN.get(text)


def display_name(emotion_en: str | None, stored: str | None,
                 output_language: str | None) -> str | None:
    """显示名统一入口：优先用英文标签渲染，没有标签就从存下来的显示名反查。"""
    english = emotion_en or to_english(stored)
    return label_for(english, output_language) or stored or None

