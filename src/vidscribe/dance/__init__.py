"""舞蹈素材资产 + 音乐对齐 + 自动切片 + 多版本混剪（独立子系统）。

和主分析流水线**完全分开**：这里一行都不碰 `pipeline.py`、`highlight/montage.py`、
`AnalyzeWorker`，也不新建第二个 SQLite —— 库、缓存、进度、日志全部复用 `b` 已有基础设施。

分层是硬约定，越层调用视为 bug：

    音频指纹/对齐   audio_fingerprint → audio_align → alignment_validation
            ↓
    目标音乐结构    music_structure（beat grid / features / rhythm bands / sections）
            ↓
    固定音乐位置切片 material_slice（source_time = target_time - offset）
            ↓
    素材资产库      material_repository（长期资产，不物理删除）+ history（事件账本）
            ↓
    评分/筛选/推荐  material_score → material_selection → recommendation → combination_search
            ↓
    编辑计划        montage_timeline（纯计划，禁止在这一层再做推荐）
            ↓
    渲染            montage_render（只执行已定稿的 Timeline；目标歌是唯一正式音轨）

为什么不引入 librosa：它拖着 numba/llvmlite 一整条编译链，而这里要的 STFT / mel /
MFCC / chroma / onset / beat_track 全都是几十行 numpy（见 `dsp.py`），
项目已有 `numpy>=1.26` 和 `av>=13.0`，不必为此加依赖。
"""

from __future__ import annotations

#: 对齐算法版本。改了算法就 +1，缓存与历史记录靠它区分「哪一版算出来的」
ALIGNMENT_ALGORITHM_VERSION = "align-v1"
#: 素材切片版本，进 dance_materials.generation_version
MATERIAL_GENERATION_VERSION = "slice-v1"
#: 推荐算法版本，进 dance_recommendation_runs.algorithm_version
RECOMMENDATION_ALGORITHM_VERSION = "recommend-v1"

__all__ = [
    "ALIGNMENT_ALGORITHM_VERSION",
    "MATERIAL_GENERATION_VERSION",
    "RECOMMENDATION_ALGORITHM_VERSION",
]
