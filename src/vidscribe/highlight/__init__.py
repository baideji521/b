"""高光剪辑：按 AI 给出的 JSON 把原视频剪成一段短片。

成品只有两段：原速播放段（带原声）+ 1 秒纯红背景（静音）。
没有冻帧、没有字幕、没有转场特效，也不混任何音效。
"""

from .clip import (
    HighlightSpec,
    default_target,
    parse_spec,
    render_highlight,
    resolve_video,
)

__all__ = ["HighlightSpec", "default_target", "parse_spec",
           "render_highlight", "resolve_video"]
