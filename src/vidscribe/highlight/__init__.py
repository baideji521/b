"""高光剪辑：按 AI 给出的 JSON 把原视频剪成「正常播放 + 冻帧特效 + 逐字字幕」的短片。"""

from .clip import (
    HighlightSpec,
    Overlay,
    default_target,
    parse_spec,
    render_highlight,
    resolve_video,
)

__all__ = ["HighlightSpec", "Overlay", "default_target", "parse_spec", "render_highlight",
           "resolve_video"]
