"""AI_卡点舞 主窗口。**独立窗口**，和主界面各开各的，谁崩了都不影响谁。

**一页到底，没有 Tab**：编排台就是这个窗口的正面 ——

    顶栏（源视频 / 视频文件夹 / 主音频 / 开始对齐）
        ↓
    左半边：实时成片画面（上）+ 主音频波形/人声/段落（下）
    右半边：视频列表（文件夹里的视频 + 对齐结果，不入库）
        ↓
    片段仓库 + 预览 / 保存 / 导出

一期第二十四节那四个区域一个都没少，只是不再和编排台平分屏幕：

    音频对齐（offset / 置信度 / 重新对齐 / 手动修正）→ 右侧侧栏，Ctrl+1
    输入与操作（选歌、参数、开跑、进度、日志）      → 左侧侧栏，Ctrl+2  RemixPanel
    素材资产（筛选 + 素材库）                      → 弹窗  FilterPanel + MaterialLibraryPanel
    选择与推荐（候选池 + 智能推荐）                 → 弹窗  CandidatePanel + RecommendationPanel
    历史与统计                                    → 弹窗  HistoryPanel + StatisticsPanel

界面线程只做两件事：画界面、读库（都是聚合查询，毫秒级）。
所有重活在 `DanceMontageWorker` 里，它自己开自己的库连接。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtWidgets import (
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


from ...logging_setup import get_logger
from .. import settings as gui_settings
from .. import theme
from ..player import FramePlayer, FramePrefetcher
from . import dialogs
from .align_bench import AlignBenchPanel
from .alignment_panel import AlignmentPanel
from .candidate_panel import CandidatePanel
from .filter_panel import FilterPanel
from .history_panel import HistoryPanel
from .material_library import MaterialLibraryPanel
from .master_audio import MasterAudioPanel
from .matrix_panel import MatrixPanel
from .recommendation_panel import RecommendationPanel
from .remix_panel import RemixPanel
from .statistics_panel import StatisticsPanel
from .video_list import VideoListPanel
from .worker import DanceMontageWorker

logger = get_logger("dance.gui")

TITLE = "AI_卡点舞 · 舞蹈素材资产与多版本混剪"

#: 实时画面允许和主音频差多少秒（超过才纠一次）。**播放中不轻易纠**：
#: cv2 的 seek 要回退到关键帧重解一段、还会重置播放时钟，纠完又慢慢漂回来 →
#: 周期性顿一下，看着就是"卡"。两边都以真实时间为基准，长期不会跑飞
LIVE_RESYNC = 0.50
#: 两次纠偏之间至少隔这么久（秒）。防止在阈值附近来回抖导致连续 seek
LIVE_RESYNC_GAP = 1.0
#: 片段已经预解码进内存时用这个阈值：定位只是换数组下标，没有解码代价，
#: 所以差一帧多就拉回来（30fps 的一帧是 0.033s），画面和音就是逐帧对得上的
LIVE_CACHED_RESYNC = 0.05
#: 内存里最多留几段的帧。一段 2 秒的 3:4 片段（405×540 × 60 帧）约 39MB，
#: 留 3 段（当前 + 下一段 + 上一段）
FRAME_BUDGET = 3


class DanceMontageWindow(QMainWindow):
    """独立窗口。关掉它不影响主界面，主界面崩了也不影响它。"""

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.db: Any = None
        self.worker: DanceMontageWorker | None = None
        self._song_id = 0
        self._last_edit = "picks"      # Ctrl+Z 撤销哪一样：最后动过的那个
        self._live_material = 0        # 实时画面现在开着哪条素材（0 = 空）
        # 上次那份界面状态（窗口位置、分栏比例、各输入框、上次选的目录）。
        # 和主界面共用 gui_settings.json，但各占一个键，互不干扰
        self.settings = gui_settings.load(cfg)
        self.state: dict[str, Any] = self.settings.setdefault("dance_window", {})
        self._loading = True
        #: 最近一轮对齐的结果（`[{path,name,alignment,error}]`）。**只在内存里**：
        #: 切片入库之前，右侧那张实时矩阵每一格都靠它按 offset 现算
        self._align_rows: list[dict[str, Any]] = []
        #: 上一次给实时画面纠偏是什么时候（`time.monotonic()`）。播放中限流用
        self._live_synced_at = 0.0
        #: 后台预解码好的片段：`{(路径, 起, 止): (帧, 起始帧号, fps)}`。
        #: **解码绝不在 GUI 线程上做**（1080×1440 一帧 8ms，两秒片段就是 0.5 秒白屏），
        #: 所以这里存的是预取线程的成果，切段落时只是装上，一帧都不现场解
        self._frames: dict[tuple, tuple] = {}
        self.prefetch = FramePrefetcher(self)
        self.prefetch.ready.connect(self._frames_ready)
        dialogs.configure(bool(cfg.dance.get("native_dialogs", True)))
        dialogs.install_memory(self.state.setdefault("dirs", {}), self.save_settings)


        self.setWindowTitle(TITLE)
        self.resize(1500, 950)
        self.setMinimumSize(1000, 640)     # 小窗口也要能用（一期第十五节）

        self.remix = RemixPanel(cfg, self)
        self.filters = FilterPanel(self)
        self.library = MaterialLibraryPanel(self)
        self.alignment = AlignmentPanel(self)
        self.master = MasterAudioPanel(cfg, None, self)
        self.matrix = MatrixPanel(self)
        self.video_list = VideoListPanel(self)
        self.bench = AlignBenchPanel(cfg, self)
        self.candidates = CandidatePanel(self)
        self.recommend = RecommendationPanel(self)
        self.history = HistoryPanel(self)
        self.statistics = StatisticsPanel(self)

        self.setCentralWidget(self._build_body())
        self.setStatusBar(QStatusBar(self))
        self._wire()
        self._open_db()
        self.reload()
        self._apply_settings()
        self._loading = False

    # ------------------------------------------------------------ 记住上次的样子
    def _apply_settings(self) -> None:
        """把上次的窗口位置、分栏比例、各输入框套回来。坏值一律忽略，绝不因此打不开。"""
        state = self.state
        geo = state.get("window")
        if isinstance(geo, list) and len(geo) == 4 and all(isinstance(v, int) for v in geo):
            self.setGeometry(*geo)
        if state.get("maximized"):
            self.showMaximized()
        sizes = state.get("studio_split")
        if (isinstance(sizes, list) and len(sizes) == 2
                and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
            self.studio_split.setSizes(sizes)
        sizes = state.get("body_split")
        if (isinstance(sizes, list) and len(sizes) == 2
                and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
            self.body_split.setSizes(sizes)
        sizes = state.get("right_split")
        if (isinstance(sizes, list) and len(sizes) == 2
                and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
            self.right_split.setSizes(sizes)
        sizes = state.get("stage_split")
        if (isinstance(sizes, list) and len(sizes) == 2
                and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
            self.stage_split.setSizes(sizes)
        self.remix.restore(state.get("remix") or {})
        self.master.restore(state.get("master") or {})
        self.bench.restore(state.get("bench") or {})
        # 两个侧栏收着还是开着，也按上次那样恢复
        self.dock_align.setVisible(bool(state.get("dock_align")))
        self.dock_remix.setVisible(bool(state.get("dock_remix")))
        if str(self.remix.song.text()).strip().isdigit():
            self._song_typed(self.remix.song.text())

    def save_settings(self) -> None:
        """存设置。启动套用阶段不写回去，否则会把默认值盖掉上次那份。"""
        if getattr(self, "_loading", True):
            return
        # 最大化时 geometry() 是全屏尺寸，存 normalGeometry 才还原得回来
        rect = self.normalGeometry() if self.isMaximized() else self.geometry()
        if rect.width() > 0 and rect.height() > 0:
            self.state["window"] = [rect.x(), rect.y(), rect.width(), rect.height()]
        self.state["maximized"] = bool(self.isMaximized())
        self.state["studio_split"] = list(self.studio_split.sizes())
        self.state["body_split"] = list(self.body_split.sizes())
        self.state["right_split"] = list(self.right_split.sizes())
        self.state["stage_split"] = list(self.stage_split.sizes())
        self.state["dock_align"] = not self.dock_align.isHidden()
        self.state["dock_remix"] = not self.dock_remix.isHidden()
        self.state["remix"] = self.remix.state()
        self.state["master"] = self.master.state()
        self.state["bench"] = self.bench.state()
        gui_settings.save(self.cfg, self.settings)


    # ------------------------------------------------------------------ 界面
    def _build_body(self) -> QWidget:
        """**一页到底，没有 Tab**：顶栏 → 主音频 → 素材矩阵（实时成片行）→ 片段仓库。

        编排台是这个窗口唯一的正面：矩阵是横向铺开的（一列一个段落），
        必须占满整个宽度，所以不再和别的页平分空间。

        其余面板一个都没删，改成"要用才拉出来"：

            音频对齐   → 右侧侧栏（QDockWidget，顶栏按钮 / Ctrl+1 开关），随时能点到
            混剪控制台 → 左侧侧栏（选歌、参数、开跑、进度、日志）
            素材资产 / 选择与推荐 / 历史与统计 → 独立弹窗，开着也不挡编排台

        侧栏和弹窗里的面板还是 `__init__` 里那几个实例，信号连线一条没变 ——
        换的只是它们摆在哪儿。
        """
        page = QWidget(self)
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(self._build_entries())
        column.addWidget(self._build_studio(), 1)
        self._build_docks()
        return page

    def _build_entries(self) -> QWidget:
        """顶栏第一行：编排台之外那些面板的入口。**只是入口，功能都还在。**"""
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 4, 8, 0)
        row.setSpacing(6)
        title = QLabel("🎵 编排台", holder)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        row.addWidget(title)
        row.addWidget(QLabel("主音频 → 分段 → 素材矩阵 → 实时成片 → 片段仓库", holder), 1)

        self.btn_dock_align = QPushButton("音频对齐", holder)
        self.btn_dock_align.setCheckable(True)
        self.btn_dock_align.setToolTip("右侧拉出「音频对齐」：每个源视频的 offset / 置信度、"
                                       "重新对齐、手动修正（Ctrl+1）")
        self.btn_dock_align.clicked.connect(self._toggle_align_dock)

        self.btn_dock_remix = QPushButton("混剪控制台", holder)
        self.btn_dock_remix.setCheckable(True)
        self.btn_dock_remix.setToolTip("左侧拉出「输入与操作」：选目标歌、切片长度、开跑、"
                                       "进度和日志（Ctrl+2）")
        self.btn_dock_remix.clicked.connect(self._toggle_remix_dock)

        self.btn_open_matrix = QPushButton("素材矩阵", holder)
        self.btn_open_matrix.setToolTip("弹窗：列＝Segment，行＝视频，第一行＝实时成片。"
                                       "片段切好入库之后才有内容")
        self.btn_open_matrix.clicked.connect(lambda: self._open_panel("matrix"))

        self.btn_open_assets = QPushButton("素材资产", holder)
        self.btn_open_assets.setToolTip("弹窗：多维筛选 + 素材资产库")
        self.btn_open_assets.clicked.connect(lambda: self._open_panel("assets"))

        self.btn_open_choose = QPushButton("选择与推荐", holder)
        self.btn_open_choose.setToolTip("弹窗：候选池 + 智能推荐")
        self.btn_open_choose.clicked.connect(lambda: self._open_panel("choose"))

        self.btn_open_review = QPushButton("历史与统计", holder)
        self.btn_open_review.setToolTip("弹窗：混剪历史 + 素材使用统计")
        self.btn_open_review.clicked.connect(lambda: self._open_panel("review"))

        for button in (self.btn_dock_align, self.btn_dock_remix, self.btn_open_matrix,
                       self.btn_open_assets, self.btn_open_choose, self.btn_open_review):
            button.setMinimumHeight(28)
            row.addWidget(button)
        return holder

    def _build_docks(self) -> None:
        """两个侧栏：音频对齐（右）、混剪控制台（左）。默认都收着，不挡编排台。"""
        self.dock_align = QDockWidget("音频对齐", self)
        self.dock_align.setObjectName("dance_dock_align")
        self.dock_align.setWidget(self.alignment)
        self.dock_align.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_align)
        self.dock_align.hide()
        self.dock_align.visibilityChanged.connect(
            lambda shown: self.btn_dock_align.setChecked(bool(shown)))

        remix_holder = QWidget(self)
        remix_layout = QVBoxLayout(remix_holder)
        remix_layout.setContentsMargins(6, 6, 6, 6)
        hint = QLabel("固定音乐位置：所有源视频都贴在**目标歌**这同一把尺子上。"
                      "素材是长期资产，只会停用、不会删除。", remix_holder)
        hint.setWordWrap(True)
        remix_layout.addWidget(hint)
        remix_layout.addWidget(self.remix, 1)

        self.dock_remix = QDockWidget("输入与操作", self)
        self.dock_remix.setObjectName("dance_dock_remix")
        self.dock_remix.setWidget(remix_holder)
        self.dock_remix.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_remix)
        self.dock_remix.hide()
        self.dock_remix.visibilityChanged.connect(
            lambda shown: self.btn_dock_remix.setChecked(bool(shown)))

    def _toggle_align_dock(self) -> None:
        """开/收「音频对齐」侧栏。

        判定用 `isHidden()` 而不是 `isVisible()`：窗口自己还没 show 的时候，
        子控件的 `isVisible()` 恒为 False，用它会把"已经开着"误判成"收着"。
        """
        self.dock_align.setVisible(self.dock_align.isHidden())

    def _toggle_remix_dock(self) -> None:
        self.dock_remix.setVisible(self.dock_remix.isHidden())

    def _open_panel(self, key: str) -> QDialog:
        """把某组面板放进独立弹窗打开。**非模态**：开着也能继续排素材。

        弹窗只建一次，之后再点就是把它拉到前面 —— 面板实例始终是
        `__init__` 里那几个，`reload()` 照旧刷它们，开没开都一样。
        """
        dialogs_open = getattr(self, "_panel_windows", None)
        if dialogs_open is None:
            dialogs_open = {}
            self._panel_windows = dialogs_open
        existing = dialogs_open.get(key)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing

        specs = {
            "matrix": ("素材矩阵（列＝Segment，行＝视频，第一行＝实时成片）", Qt.Vertical,
                       ((self.matrix, 1),), (1280, 720)),
            "assets": ("素材资产（筛选 + 素材库）", Qt.Horizontal,
                       ((self.filters, 1), (self.library, 3)), (1180, 720)),
            "choose": ("选择与推荐（候选池 + 智能推荐）", Qt.Vertical,
                       ((self.candidates, 3), (self.recommend, 2)), (1040, 760)),
            "review": ("历史与统计", Qt.Horizontal,
                       ((self.history, 3), (self.statistics, 2)), (1120, 700)),
        }
        title, orientation, parts, size = specs[key]
        window = QDialog(self)
        window.setWindowTitle(title)
        window.setModal(False)
        window.resize(*size)
        layout = QVBoxLayout(window)
        layout.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(orientation, window)
        for index, (widget, stretch) in enumerate(parts):
            split.addWidget(widget)
            split.setStretchFactor(index, stretch)
        layout.addWidget(split)
        dialogs_open[key] = window
        window.show()
        return window

    def _build_studio(self) -> QWidget:
        """编排台：**主音频 → 分段 → 素材 → 成片**，一页走完：

            ① 一行工具栏：源视频 / 视频文件夹 / 主音频（＝目标歌）/ [开始音频对齐]
            ② 左半边：视频 + 音谱同一块 —— 上边实时播放画面，下边波形/人声/音谱 + 段落条
            ③ 右半边：素材矩阵（列＝Segment，行＝视频，第一行＝实时播放）
            ④ 页脚：入库三件事 + 片段仓库 + 预览/保存/导出

        ②③ 左右并排而不是上下堆：视频和音谱要竖着占满左半屏才看得清波形细节，
        矩阵横着铺在右半屏，挑素材时不用来回滚屏 —— 一眼能同时看到
        "这一刻的画面 / 这一刻的音 / 这一段有哪些候选"。

        主音频那条时间轴是**唯一的时间权威**：段落、素材、实时播放全按它算，
        每个 mp4 不自己维护一条时间轴。
        """
        holder = QWidget(self)
        self.studio = holder
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        stack = QSplitter(Qt.Vertical, holder)
        self.bench.use_compact_layout()   # 对齐台只留顶上那一行工具栏
        # 主音频就是目标歌：把主音频那一条并进顶栏，界面上一首歌只填一处
        self.bench.adopt_song_row(self.master.take_song_row())
        self.master.use_wide_layout()      # 主可视化区通栏，左边那列导航收起来
        self.master.path.textChanged.connect(self._song_picked)
        stack.addWidget(self.bench)      # 一行：源视频 / 批量选 / 主音频 / 开始对齐
        stack.addWidget(self._build_middle(holder))
        stack.setStretchFactor(0, 0)
        stack.setStretchFactor(1, 1)
        # 顶栏那一条按它自己要的高度给，别硬编一个数字把「开始音频对齐」压掉
        stack.setSizes([self.bench.minimumHeight() or 120, 900])
        self.studio_split = stack
        column.addWidget(stack, 1)

        # 入库那三个按钮（保存对齐结果 / 切片并加入素材库 / 导出当前测试片段）钉在整页页脚：
        # 它们是这一页的出口，不该夹在对齐区和主音频编辑区中间
        self.studio_ingest = self.bench.take_save_row()
        self.studio_ingest.setParent(holder)
        column.addWidget(self.studio_ingest)
        column.addWidget(self._build_repository())


        row = QHBoxLayout()

        row.setContentsMargins(8, 0, 8, 4)
        row.setSpacing(8)
        self.studio_hint = QLabel("先把源视频对齐入库，再去「素材矩阵/成片」挑每一段用谁。"
                                  "段落起止只由主音频那条时间轴决定。", holder)
        self.studio_hint.setWordWrap(True)
        self.btn_preview_final = QPushButton("▶ 预览", holder)
        self.btn_save_all = QPushButton("保存", holder)
        self.btn_export_final = QPushButton("导出", holder)
        for button in (self.btn_preview_final, self.btn_save_all, self.btn_export_final):
            button.setMinimumHeight(38)
            button.setMinimumWidth(110)
        font = self.btn_export_final.font()
        font.setBold(True)
        self.btn_export_final.setFont(font)

        row.addWidget(self.studio_hint, 1)
        row.addWidget(self.btn_preview_final)
        row.addWidget(self.btn_save_all)
        row.addWidget(self.btn_export_final)
        column.addLayout(row)

        self.btn_preview_final.clicked.connect(self._preview_final)
        self.btn_save_all.clicked.connect(self._save_everything)
        self.btn_export_final.clicked.connect(self._export_final)
        return holder

    def _build_middle(self, parent: QWidget) -> QWidget:
        """左右并排：**左半边＝视频 + 音谱，右半边＝视频列表 + 实时矩阵**。

            ┌───────────────────┬───────────────────┐
            │  ▶ 实时播放画面    │  📄 视频列表       │
            ├───────────────────┤  （对齐结果）      │
            │  波形/人声/音谱    ├───────────────────┤
            │  段落条            │  📦 实时矩阵       │
            │                   │  列＝主音频的分段  │
            └───────────────────┴───────────────────┘

        矩阵是**跟着主音频活的**：音谱上切一刀就多一列、取消一刀就少一列，
        入库之前就有内容（每一格＝这个视频在这一段对应的那截，按 offset 算）。
        主音频一播，实时播放行那一列跟着走，左边画面同步跳到对应位置。
        """
        split = QSplitter(Qt.Horizontal, parent)
        split.addWidget(self._build_stage())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([760, 760])
        self.body_split = split
        return split

    def _build_right(self) -> QWidget:
        """右半屏三块，从上到下：

            📄 视频列表      文件夹里的视频 + 对齐结果（双击＝播这个视频）
            ▶ 实时播放       成片的定稿行，**自己单独一行**
            📦 段落候选      列＝分段，行＝视频（点一下＝这一段用它，绿；双击＝试听这一格）

        实时播放行是从矩阵里搬出来的（`take_realtime_row()`），控件和信号都没变，
        只是不再和候选挤在同一张表里 —— 它是"最终用谁"，该单独占一行。
        """
        split = QSplitter(Qt.Vertical, self)
        split.addWidget(self.video_list)
        realtime = self.matrix.take_realtime_row()
        realtime.setParent(split)
        split.addWidget(realtime)
        split.addWidget(self.matrix)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 3)
        self.video_list.setMinimumWidth(320)
        self.video_list.setMinimumHeight(120)
        self.matrix.setMinimumHeight(160)
        split.setSizes([240, 110, 380])
        self.right_split = split
        return split





    def _build_stage(self) -> QWidget:
        """② 视频 + ③ 音谱区：**同一个布局里，上边视频、下边音谱**。

            ┌──────────────────────────────┐
            │  ▶ 实时播放（视频画面）        │  ← 上
            ├──────────────────────────────┤
            │  播放条 + 波形/人声/音谱 + 段落 │  ← 下
            └──────────────────────────────┘

        视频和音谱共用主音频那条唯一时间轴，所以必须上下贴在一起看 ——
        眼睛在同一列上下扫，就能对上"这一刻的画面 vs 这一刻的音"。
        """
        holder = QWidget(self)
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        stack = QSplitter(Qt.Vertical, holder)     # 视频/音谱各占多高由用户拖
        video = QWidget(stack)
        vbox = QVBoxLayout(video)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)
        self.live = FramePlayer(video)
        # 声音只由主音频出：素材自己的原声会和主音频打架
        self.live.set_audio_enabled(False)
        self.live.setMinimumHeight(180)
        self.live_note = QLabel("▶ 实时播放：还没开始播", video)
        self.live_note.setWordWrap(True)
        self.live_note.setStyleSheet(f"color:{theme.TEXT_DIM};")
        # 逐帧核卡点：预解码之后前后翻帧是零成本的，所以这两颗按钮就是"这一刀压在
        # 哪个鼓点上"的最终裁判。快捷键用 , / . （剪辑软件的老规矩），
        # ← → 已经被主音频那条时间轴占了，不抢
        self.btn_prev_frame = QPushButton("◀ 帧", video)
        self.btn_next_frame = QPushButton("帧 ▶", video)
        for button, delta in ((self.btn_prev_frame, -1), (self.btn_next_frame, 1)):
            button.setFixedWidth(52)
            button.setToolTip("停下来逐帧看刀口对不对（快捷键 , / .）")
            button.clicked.connect(lambda _=False, step=delta: self.live.step_frame(step))
        note_row = QHBoxLayout()
        note_row.setContentsMargins(0, 0, 0, 0)
        note_row.setSpacing(4)
        note_row.addWidget(self.live_note, 1)
        note_row.addWidget(self.btn_prev_frame)
        note_row.addWidget(self.btn_next_frame)
        vbox.addWidget(self.live, 1)
        vbox.addLayout(note_row)
        stack.addWidget(video)

        below = QWidget(stack)
        bbox = QVBoxLayout(below)
        bbox.setContentsMargins(0, 0, 0, 0)
        bbox.setSpacing(2)
        bbox.addWidget(self.master, 1)
        stack.addWidget(below)

        stack.setStretchFactor(0, 2)
        stack.setStretchFactor(1, 3)
        stack.setSizes([260, 420])
        self.stage_split = stack
        column.addWidget(stack, 1)
        return holder


    def _build_repository(self) -> QWidget:
        """💾 片段仓库：这首歌的候选片段一共有多少、按段落各多少，一键全存。"""
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(8)
        self.repo_note = QLabel("💾 片段仓库：还没选主音频", holder)
        self.repo_note.setWordWrap(True)
        self.btn_save_clips = QPushButton("💾 保存所有片段到仓库", holder)
        self.btn_save_clips.setMinimumHeight(38)
        self.btn_save_clips.setMinimumWidth(190)
        self.btn_save_clips.setToolTip(
            "把每个已对齐视频在**每一段**上的片段全部切出来入库（不是只存实时播放行选中那几条）。\n"
            "片段按「歌 + 音频 + 段落」分开存，以后自动编排只会在同一段的候选池里挑。")
        self.btn_save_clips.clicked.connect(self._save_all_clips)
        row.addWidget(self.repo_note, 1)
        row.addWidget(self.btn_save_clips)
        return holder

    def _refresh_repository(self) -> None:
        """把片段仓库那一栏的数字刷新一遍（音频名 / 歌名 / 每段几条）。"""
        if self.db is None or not self._song_id:
            self.repo_note.setText("💾 片段仓库：还没选主音频")
            return
        from ...dance import material_repository as repo

        info = repo.repository_summary(self.db, self._song_id)
        per = "、".join(f"S{index + 1}:{count}"
                       for index, count in sorted(info["per_segment"].items()))
        self.repo_note.setText(
            f"💾 片段仓库　音频名称：{info['audio_name'] or '—'}　·　"
            f"对应歌曲：{info['song_name'] or '—'}（#{info['song_id']}）　·　"
            f"已生成片段：{info['clip_count']}"
            + (f"　·　{per}" if per else "　·　还没切过片"))

    def _save_all_clips(self) -> None:
        """💾 保存所有片段到仓库：**所有候选**都切，不只是选中的那几条。

        走的还是正式那条流水线（`DanceMontageWorker` 的 slice 动作），
        它按当前段落模板给每个已对齐视频切出每一段 —— 片段天然带着
        `target_song_id + segment_index`，所以候选池自动按段落隔离。
        """
        if self.db is None or not self._song_id:
            QMessageBox.information(self, "还没选主音频", "先选主音频并分析，再存片段。")
            return
        if self.master.template is None:
            QMessageBox.information(self, "还没分段",
                                    "先在主音频上切好段落（✂ 切分），片段才知道按什么切。")
            return
        self.statusBar().showMessage("正在把所有候选片段切出来入库…", 4000)
        self.start()          # 和「切片并加入素材库」同一条正式流水线


    def _master_moved(self, moment: float) -> None:
        """主音频走到 `moment` 秒了 —— 这一条把整页串起来。

            主音频位置（唯一时间权威）
                ↓
            当前 Segment（SegmentTemplate 说的）
                ↓
            矩阵那一列描边 + 实时播放行那一格
                ↓
            实时画面 seek 到片段里对应的位置
        """
        index = self._segment_at(moment)
        self.matrix.set_current_segment(index)
        self._sync_live(self.matrix.current_payload(index) if index >= 0 else {}, moment)
        # 下一段提前解好：等播到边界再解就是当场卡一下（一帧 10ms × 60 帧）
        self._warm(index + 1)

    def _segment_at(self, moment: float) -> int:
        """这一刻落在第几段。**以矩阵当前那几列为准** —— 它就是用户看到的分段。

        以前这里只问 `master.template`：没切过分段时它是 None，于是播放头永远算出
        −1，而矩阵却是按等间隔位置铺好列的。结果就是"格子摆进实时播放行了，
        左边画面还是不播"。两套"当前是第几段"的算法必须合成一套。
        """
        for spec in self.matrix.segments:
            span = spec.get("span") or ()
            if len(span) == 2 and float(span[0]) <= float(moment) < float(span[1]):
                return int(spec.get("index", -1))
        template = self.master.template
        span = template.span_at(moment) if template is not None else None
        return int(span.index) if span is not None else -1

    def _sync_live(self, payload: dict, moment: float) -> None:
        """让实时画面跟上主音频。**这一段没素材就不播**。

        不找别的段、不用上一段顶替、不随机补一条 —— 画面就保持空，
        主音频照旧往下走。跨段落取素材会让画面和音乐错开，那是这套系统最不能出的错。

        `material_id` 只有 **0** 才算"没素材"：负数是编排台按 offset 现算的内存格子
        （放整条源视频），照样要播。

        播放时**不是每一帧都 seek**：主音频每 50ms 回调一次，每次都 seek 会和视频
        自己的时钟打架，画面看着像卡住。只在换素材、或者偏差超过 `LIVE_DRIFT` 时纠一次，
        中间让它自己顺着播。
        """
        material = int(payload.get("material_id") or 0)
        if material == 0:
            self._live_material = 0
            self.live.pause()
            self.live.close_video()
            self.live_note.setText("这一段没有素材 —— 画面保持空（不会自动补别的段）")
            return
        switched = material != self._live_material
        if switched:
            if not self.live.open(str(payload.get("path") or "")):
                self._live_material = 0
                self.live_note.setText(f"素材 #{material} 的文件打不开，画面保持空")
                return
            self._live_material = material
            self.live.set_audio_enabled(False)
            # 换素材时装上**后台已经解好的**这一段帧：GUI 线程只做一次列表赋值。
            # 还没解好就先流式播（老行为），等预取线程送到再无缝换成内存播放 ——
            # 唯一不许发生的是"在这儿现场解码"，那就是切段落卡半秒的元凶
            self._use_cached_segment(payload)
        # 片段内位置 = 主音频时间 − 基准。基准分两种：
        #   · 库里的片段：`target_start`（切片时已经按 offset 算过，文件从那一刻起）
        #   · 内存里的实时格子：`seek_base` = offset（放的是**整条源视频**）
        # 两种都只减一次，绝不再叠第二遍
        start = float(payload.get("seek_base", payload.get("target_start") or 0.0) or 0.0)
        want = max(0.0, float(moment) - start)
        # `FramePlayer.position` 是**方法**不是属性（和 `duration()` / `is_playing()` 一样）。
        # 写成 `float(getattr(self.live, "position", 0.0))` 的话 float() 拿到的是绑定方法
        # → TypeError，而这是在槽里，PyQt 直接把进程干掉。
        try:
            at = float(self.live.position())
        except (TypeError, ValueError, AttributeError, RuntimeError):
            at = -1.0
        playing = self.master.player.state() == QMediaPlayer.PlayingState
        # 纠偏只在**必要时**做，否则播放中每 50ms 一次 seek 会把画面顿成幻灯片：
        #   · 换素材 / 播放器还没开视频 → 必须定位
        #   · 主音频没在播（暂停、拖播放头、逐段检查）→ 立刻定位，这时要的就是准
        #   · 正在播 → 只有漂超过 LIVE_RESYNC 且距上次纠偏够久才动它，
        #     平时让视频自己那口绝对钟跑（它和主音频都按真实时间走）
        now = time.monotonic()
        # 缓存住的片段定位是零成本（换个数组下标），所以纠偏可以又快又勤 ——
        # 一帧的偏差就拉回来、也不用节流，这才是"能核卡点"的精度。
        # 没缓存上（片段太长/内存不够）才退回原来那套宽松阈值防抖
        cached = self.live.is_cached()
        limit = LIVE_CACHED_RESYNC if cached else LIVE_RESYNC
        far = at < 0.0 or abs(at - want) > limit
        due = cached or (now - self._live_synced_at) >= LIVE_RESYNC_GAP
        fixed = switched or not playing or (far and due)
        if fixed:
            self.live.seek(want)
            self._live_synced_at = now
        self.live.play() if playing else self.live.pause()
        self.live_note.setText(
            f"{payload.get('video_name') or ''}　素材 #{material}　"
            f"片段内 {want:.3f}s{'　⚡内存' if cached else ''}"
            f"{'（已对齐）' if fixed else ''}")

    def _live_window(self, payload: dict) -> tuple[float, float] | None:
        """这一格在**播放器时间轴上**占的区间 —— 预解码就缓存这一段。

        一条公式覆盖两种素材，不分叉：
        `[target_start − base, target_end − base]`，base 就是 `_sync_live` 里
        减的那个基准。库里的片段文件本身就是这一段（base = target_start → 0..时长），
        内存里的实时格子放的是整条源视频（base = offset → source_start..source_end）。
        """
        begin = payload.get("target_start")
        end = payload.get("target_end")
        if begin is None or end is None:
            return None
        base = float(payload.get("seek_base", payload.get("target_start") or 0.0) or 0.0)
        return (max(0.0, float(begin) - base), float(end) - base)

    # ------------------------------------------------------- 片段帧的后台预解码
    @staticmethod
    def _frame_key(path, window) -> tuple:
        return (os.path.normcase(os.path.normpath(str(path))),
                round(float(window[0]), 3), round(float(window[1]), 3))

    def _use_cached_segment(self, payload: dict) -> bool:
        """让实时画面用上这一格的内存帧：有就装，没有就排个预取的活。"""
        window = self._live_window(payload)
        path = str(payload.get("path") or "")
        if window is None or not path:
            return False
        bundle = self._frames.get(self._frame_key(path, window))
        if bundle is None:
            self.prefetch.request(path, *window)
            return False
        return self.live.adopt_cache(path, window[0], window[1], bundle)

    def _warm(self, index: int) -> None:
        """提前把某一段解好。**当前段 + 下一段**都暖着，播到边界就不会停一下。"""
        if index < 0:
            return
        payload = self.matrix.current_payload(index)
        if not payload:
            return
        window = self._live_window(payload)
        path = str(payload.get("path") or "")
        if window is None or not path:
            return
        if self._frame_key(path, window) in self._frames:
            return
        self.prefetch.request(path, *window)

    def _frames_ready(self, path: str, begin: float, end: float, bundle) -> None:
        """预取线程送来一段帧。存起来，正好是在播的那一段就立刻换上。

        只留 `FRAME_BUDGET` 段：一段 2 秒的 3:4 片段约 39MB，留太多等于慢慢吃光内存。
        """
        key = self._frame_key(path, (begin, end))
        self._frames[key] = bundle
        while len(self._frames) > FRAME_BUDGET:
            self._frames.pop(next(iter(self._frames)))     # dict 有序：最早那份先走
        if self.live.holds(path):
            self.live.adopt_cache(path, begin, end, bundle)

    def _preview_final(self) -> None:

        """▶ 预览：放主音频（成片的音轨就是它），画面预览走素材卡片上的 ▶。"""
        self.master.play()
        picks = self.matrix.picks()
        self.statusBar().showMessage(
            f"正在放主音频；已定 {len(picks)} 段。单条素材的画面点它卡片上的「▶ 预览」，"
            "整片效果要导出后看成品。", 8000)

    def _export_final(self) -> None:
        """导出：拿 FINAL TIMELINE 这份选择走原来的出片流水线，不另造一套。"""
        picks = self.matrix.picks()
        if not picks:
            QMessageBox.information(self, "还没有选择",
                                    "FINAL TIMELINE 上一段都还没定，先在素材池里挑。")
            return
        # 负号的 id 是"内存里的实时格子"（还没入库的整条源视频），出片流水线找不到它们。
        # 这里挡住并说清下一步，而不是让 worker 报一句看不懂的错
        if any(int(material) < 0 for material in picks.values()):
            QMessageBox.information(
                self, "先把片段入库",
                "现在选中的是实时矩阵里的临时格子（还没切成素材）。\n"
                "先点页脚「切片并加入素材库」把片段切出来，再回来导出。")
            return
        self.remix.set_manual(picks)
        job = self.remix.job()
        job["manual"] = dict(picks)
        job["recommend"] = False          # 用户已经拍板了，别让推荐再插手
        if not job.get("song"):
            QMessageBox.information(self, "还没选目标歌",
                                    "顶栏「混剪控制台」里先选一首目标歌（或填库里的 id）。")
            return
        self.start(job)

    def _wire(self) -> None:
        self.remix.start_requested.connect(self.start)
        self.remix.stop_requested.connect(self.stop)
        self.remix.song_changed.connect(self._song_typed)
        self.remix.slice_changed.connect(lambda _v: self.reload())
        self.remix.remove_song_requested.connect(self._remove_song)
        self.remix.show_retired.stateChanged.connect(lambda _s: self.reload())

        self.filters.btn_apply.clicked.connect(self._reload_materials)
        self.filters.changed.connect(lambda _spec: self._reload_materials())
        self.library.changed.connect(self.reload)
        self.alignment.realign_requested.connect(self._realign)
        self.alignment.changed.connect(self.reload)
        # 测试台确认后的入库请求才进正式流水线：它自己只算不写
        self.bench.ingest_requested.connect(self.start)
        self.bench.changed.connect(self.reload)
        # 点「开始音频对齐」→ 主音频要是还没解过，顺手先解一遍：
        # 对齐拿它当尺子，界面上的波形/人声/音谱也该同时出来（后台线程，不挡对齐）
        self.bench.btn_start.clicked.connect(self.master.ensure_analyzed)
        # 文件夹选完 → 右侧列表先把文件列出来；对齐算完 → 填 Offset / 置信度。
        # 两条都**只是显示**，落库仍然只发生在页脚那两个按钮上
        self.bench.folder_scanned.connect(self.video_list.set_files)
        self.bench.results_ready.connect(self.video_list.set_results)
        # 对齐结果留在内存里：切片入库之前，右侧那张实时矩阵就靠它算每一格
        self.bench.results_ready.connect(self._align_rows_ready)
        self.video_list.picked.connect(self._video_picked)
        # 右键删除/粘贴要同步到批量清单，否则"列表里没了，跑批还在算它"
        self.video_list.about_to_delete.connect(self._release_files)
        self.video_list.removed.connect(self._videos_removed)
        self.video_list.added.connect(self._videos_added)

        self.candidates.manual_changed.connect(self.remix.set_manual)
        # 矩阵里拖出来的选择就是"手动指定这一格用谁"，和候选面板同一个出口
        self.matrix.picks_changed.connect(self.remix.set_manual)
        self.matrix.picks_changed.connect(lambda _p: self._touched("picks"))
        self.matrix.refused.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.matrix.saved.connect(lambda text: self.statusBar().showMessage(text, 4000))
        self.matrix.preview_requested.connect(self._preview_material)
        # 双击某一格 = 在左边只播这一截（不改"这一段用谁"、不动主音频）
        self.matrix.segment_preview.connect(self._preview_segment)
        self.master.template_changed.connect(lambda _t: self._reload_matrix())
        self.master.template_changed.connect(lambda _t: self._touched("template"))
        # 主音频位置是唯一时间权威：它一动，当前段落 / 矩阵 / 实时画面全跟着动
        self.master.seek_requested.connect(self._master_moved)
        self.matrix.realtime_changed.connect(lambda _s, _m: self._master_moved(self.master.at))
        self.recommend.adopted.connect(self._adopt)
        self.recommend.changed.connect(self.reload)
        self.history.rerender_requested.connect(self._rerender)
        self._install_shortcuts()

    def _touched(self, what: str) -> None:
        """记住最后动的是哪一样，Ctrl+Z 才知道该撤销分段还是撤销编排。"""
        self._last_edit = what

    def _video_picked(self, path: str) -> None:
        """双击右侧视频列表某一行：设成对齐台当前那条源，**并且直接在左边播它**。

        播的是整条源视频（不是某一段），所以基准是 0：这一步只为"让我看看这条是什么"。
        主音频不动 —— 它是时间权威，不该被"随便看一眼"带跑。
        """
        value = str(path or "").strip()
        if not value:
            return
        self.bench.source.setText(value)
        self._live_material = 0            # 下次段落同步时会重新打开该段的素材
        if not self.live.open(value):
            self.statusBar().showMessage(f"打不开：{Path(value).name}", 6000)
            return
        self.live.set_audio_enabled(False)
        self.live.seek(0.0)
        self.live.play()
        self.live_note.setText(f"▶ 单独预览整条：{Path(value).name}"
                               "（主音频一走就切回该段落的素材）")
        self.statusBar().showMessage(f"正在单独预览 {Path(value).name}", 6000)

    def _song_picked(self, text: str) -> None:
        """主音频就是目标歌：这边填了文件，对齐要用的目标歌就跟着它走。

        对齐还认库里的歌 id，所以只在真是个路径时才盖 —— 用 `reload()` 填进去的
        那个数字 id 不能被清掉。
        """
        value = str(text or "").strip()
        if not value:
            return
        if self.bench.target.text().strip() != value:
            self.bench.target.setText(value)


    # ------------------------------------------------------------------ 库
    def _install_shortcuts(self) -> None:
        """全局快捷键。**只加不抢**：先看主界面有没有占用，这里挑的都是没人用的。

        Ctrl+S 保存当前编排、Ctrl+Z 撤销、Ctrl+Y 重做。
        播放/停顿导航那几个（Space / ← → / J K L / S / M）放在「主音频/分段」页里，
        因为它们只在那一页有意义（见 `MasterAudioPanel`）。
        """
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut

        def bind(keys: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(handler)

        bind("Ctrl+S", self._save_everything)
        bind("Ctrl+Z", self._undo)
        bind("Ctrl+Y", self._redo)
        bind("Ctrl+1", self._toggle_align_dock)
        bind("Ctrl+2", self._toggle_remix_dock)
        # 逐帧：剪辑软件的老规矩。← → 归主音频时间轴，这里不抢
        bind(",", lambda: self.live.step_frame(-1))
        bind(".", lambda: self.live.step_frame(1))

    def _save_everything(self) -> None:
        """Ctrl+S / 「保存」：分段和编排一起存（一页到底，两样都在同一页上）。"""
        self.master.save_template()
        self.matrix.save()
        self.save_settings()

    def _undo(self) -> None:
        """Ctrl+Z：撤销**最后动过的那一样**（分段还是素材编排）。"""
        if self._last_edit == "template":
            self.master.undo()
            return
        if not self.matrix.undo():
            self.master.undo()

    def _redo(self) -> None:
        if self._last_edit == "template":
            self.master.redo()
            return
        if not self.matrix.redo():
            self.master.redo()

    def _preview_segment(self, payload: dict) -> None:
        """双击某一格：在左边**只播这一截**（源 start→end），到点自动停。

        主音频一个字不动 —— 这只是"让我看看这一格长什么样"，不该把整页的
        时间轴带跑。播完把 `_live_material` 归零，主音频下次一走就切回该段落
        真正选中的那条素材。
        """
        path = str(payload.get("path") or "")
        if not path or not Path(path).is_file():
            self.statusBar().showMessage(f"文件不在了：{path}", 6000)
            return
        start = max(0.0, float(payload.get("source_start") or 0.0))
        end = float(payload.get("source_end") or 0.0)
        if not self.live.open(path):
            self.statusBar().showMessage(f"打不开：{Path(path).name}", 6000)
            return
        self.live.set_audio_enabled(False)
        # 单独看一格 = 最典型的"来回拖着核卡点"场景，先把这几秒全解进内存
        cached = self.live.preload(start, end) if end > start else False
        self.live.seek(start)
        self.live.play()
        self._live_material = 0
        span = (end - start) if end > start else 2.0
        QTimer.singleShot(int(max(0.2, span) * 1000), self.live.pause)
        self.live_note.setText(
            f"▶ 单独试听：{payload.get('video_name') or Path(path).name}　"
            f"源 {start:.2f}→{max(end, start):.2f}s"
            f"{'　⚡内存逐帧（, / . 逐帧核卡点）' if cached else ''}")

    def _preview_material(self, path: str) -> None:
        """预览一条素材：复用素材库那个播放器，不另造一个。"""
        target = Path(str(path))
        if not target.is_file():
            self.statusBar().showMessage(f"文件不在了：{target}", 6000)
            return
        opened = getattr(self.library, "preview", None)
        if callable(opened):
            opened(str(target))
            return
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_db(self) -> None:
        from ...db import open_db

        self.cfg.ensure_dance_dirs()
        self.db = open_db(self.cfg)
        self.master.db = self.db       # 段落模板要落库，这一页也得拿到同一个连接
        from ...dance import strategy as strategy_mod

        strategy_mod.ensure_presets(self.db)

    def _song_typed(self, text: str) -> None:
        """输入框里给的是库里的 id 时，立刻切过去；给路径就等跑完再说。"""
        if str(text).strip().isdigit():
            self._song_id = int(str(text).strip())
            self.reload()

    def _pool_spec(self):
        """给候选池用的筛选条件。

        和素材库列表共用同一套条件，但**换成配置里的候选池上限**：
        素材库那个「最多显示」是给眼睛看的（300 行够翻了），
        候选池那个是"算法能考虑多少条"，两者混用会让新素材在打分前就被丢掉。
        """
        from dataclasses import replace

        spec = self.filters.spec(self._song_id)
        return replace(spec, limit=int(self.cfg.dance["candidate_pool_size"]))

    def reload(self) -> None:
        """把四个区域全部按当前目标歌刷一遍。全是聚合查询，很快。"""
        if self.db is None:
            return
        from ...dance import material_repository as repo

        self.remix.set_songs(repo.list_songs(
            self.db, include_retired=self.remix.show_retired.isChecked()))

        slice_duration = self.remix.slice_duration()
        self.alignment.refresh(self.db, self._song_id)
        self.bench.refresh(self.db, self._song_id)
        self.candidates.refresh(self.db, self._song_id, slice_duration=slice_duration,
                                spec=self._pool_spec())

        self.recommend.refresh(self.db, self._song_id, slice_duration=slice_duration)
        self.history.refresh(self.db, self._song_id)
        self.statistics.refresh(self.db, self._song_id)
        self._reload_materials()
        self._reload_matrix()
        row = repo.get_song(self.db, self._song_id) if self._song_id else None
        if row is not None:
            self.statusBar().showMessage(
                f"目标歌 #{self._song_id}《{row['title']}》"
                f"{float(row['duration'] or 0):.2f}s｜BPM {float(row['bpm'] or 0):.1f}"
                f"｜每格 {slice_duration:g}s")
        else:
            self.statusBar().showMessage("还没选目标歌")

    def _reload_materials(self) -> None:
        if self.db is None or not self._song_id:
            self.library.show_materials(self.db, 0, [])
            return
        from ...dance import material_selection as selection

        spec = self.filters.spec(self._song_id)
        found = selection.find_materials(self.db, spec,
                                        exclude_recent=self.filters.exclude_recent_mode())
        self.library.show_materials(self.db, self._song_id, found)

    def showEvent(self, event) -> None:                  # noqa: N802 - Qt 的名字
        """窗口真正有宽高之后再铺一次分栏比例。**只做第一次。**

        `QSplitter.setSizes()` 在控件还没有几何尺寸的时候调用会被 Qt 按各面板的
        sizeHint 重新缩放：右边那张表按"五列全摊开"要宽度，于是左半屏被挤到两成宽
        （构造期设的 760/760 完全不作数）。等到 `showEvent` 里有了真实宽度，
        再按比例分一次，这才是用户看到的那一版。
        """
        super().showEvent(event)
        if getattr(self, "_split_applied", False):
            return
        self._split_applied = True
        self._apply_split_ratios()

    def _apply_split_ratios(self) -> None:
        """按"上次的比例（没有就用构造时那份）"重新分一次三个分栏。

        按**比例**而不是像素：窗口尺寸每次都可能不一样。面板数量也不写死 ——
        右栏是三块（视频列表 / 实时播放 / 候选矩阵），以后再加一块也不用改这儿。
        """
        for split, key in ((self.body_split, "body_split"),
                           (self.right_split, "right_split"),
                           (self.stage_split, "stage_split")):
            total = (split.width() if split.orientation() == Qt.Horizontal
                     else split.height())
            count = split.count()
            if total <= 0 or count <= 1:
                continue
            saved = self.state.get(key)
            weights = (saved if (isinstance(saved, list) and len(saved) == count
                                 and all(isinstance(v, int) and v >= 0 for v in saved)
                                 and sum(saved) > 0)
                       else list(split.sizes()))
            if sum(weights) <= 0:
                continue
            scale = total / float(sum(weights))
            sizes = [max(1, int(round(value * scale))) for value in weights]
            sizes[-1] = max(1, total - sum(sizes[:-1]))
            split.setSizes(sizes)

    def _align_rows_ready(self, rows) -> None:
        """一轮对齐算完 → 记在内存里，右侧矩阵立刻按它铺格子（**不落库**）。"""
        self._align_rows = [dict(row) for row in (rows or ()) if row.get("alignment")]
        self._reload_matrix()

    def _release_files(self, paths) -> None:
        """马上要删这些文件了 —— **先把占用松开**。

        Windows 上只要 cv2 还开着这个文件，`unlink` 就是 PermissionError。
        涉及三处：实时画面、对齐台自己那个播放器、内存里预解码的帧
        （帧本身不占文件句柄，但留着等于留一份已删文件的画面，一起清掉更干净）。
        """
        keys = {self._path_key(p) for p in (paths or ())}
        if not keys:
            return
        for player in (self.live, getattr(self.bench, "player", None)):
            holds = getattr(player, "holds", None)
            if not callable(holds):
                continue
            if any(holds(p) for p in paths):
                player.pause()
                player.close_video()
        if self.live.path() == "":
            self._live_material = 0
        for key in [k for k in self._frames if k[0] in keys]:
            self._frames.pop(key, None)

    def _videos_removed(self, paths) -> None:
        """右键删掉了几个视频（文件已经从磁盘上没了）—— 把它们从内存里也清干净。

        三处都要清，少一处就会出现"文件没了还在算它"：
        批量清单（下次跑批的输入）、内存里的对齐结果、按对齐结果铺的实时矩阵。
        正在播的就是被删的那条时，把画面收掉，别拿着不存在的文件继续 seek。
        """
        keys = {self._path_key(p) for p in (paths or ())}
        if not keys:
            return
        self.bench.drop_sources(paths)
        before = len(self._align_rows)
        self._align_rows = [row for row in self._align_rows
                            if self._path_key(row.get("path")) not in keys]
        if len(self._align_rows) != before:
            self._reload_matrix()
        self.live.pause()
        self.live.close_video()
        self._live_material = 0
        self.statusBar().showMessage(
            f"已删除 {len(keys)} 个视频文件，并从批量清单/对齐结果里清掉", 6000)

    def _videos_added(self, paths) -> None:
        """右键粘贴进来的视频：同步进批量清单，下一轮对齐就会算它们。"""
        added = self.bench.add_sources(paths)
        self.statusBar().showMessage(
            f"粘贴进 {added} 个视频，点「开始音频对齐」就会算它们", 6000)

    @staticmethod
    def _path_key(path) -> str:
        """比路径用的统一写法（Windows 上大小写和分隔符都可能不一样）。"""
        return os.path.normcase(os.path.normpath(str(path))) if path else ""


    def _live_spans(self) -> list[tuple[int, str, float, float]]:
        """当前的分段。**主音频上的模板说了算**，它一被切分/取消就跟着变。

        用户还没切过分段时退回等间隔位置，好让矩阵一开始就有列可看。
        """
        template = self.master.template
        if template is not None and template.spans:
            return [(int(s.index), s.name, float(s.start), float(s.end))
                    for s in template.spans]
        from ...dance.music_structure import target_positions

        duration = float(getattr(self.master, "duration", 0.0) or 0.0)
        if duration <= 0.0 and self.db is not None and self._song_id:
            from ...dance import material_repository as repo

            row = repo.get_song(self.db, self._song_id)
            duration = float(row["duration"] or 0.0) if row is not None else 0.0
        if duration <= 0.0:
            return []
        return [(p.index, f"S{p.index + 1}", p.start, p.end)
                for p in target_positions(duration, self.remix.slice_duration())]

    def _live_segments(self) -> list[dict[str, Any]]:
        """**入库之前**的矩阵内容：每个已对齐视频 × 每一段，按 offset 现算。

            源时间 = 目标时间 − offset      （对齐那层的唯一换算）

        一段整段落在源视频里面才给格子；越界的那格留空（不偷偷 clamp，
        那会生成画面和音乐错开的素材 —— 这套系统最不能出的错）。

        **首尾例外**：顶栏「首缺≤ / 尾缺≤」给了余量时，头尾两段改成源区间取交集，
        缺的那一截记进 `head_pad` / `tail_pad`，切片时用边界帧补足到整段时长。
        判定和切片是同一处（`material_slice.map_to_source`），所以矩阵里看到的格子，
        切片时一定切得出来。

        这些格子**不是素材**：没有 material_id、没进库，选中也只是"人工挑选的
        实验数据"，所以矩阵此时按 `attach(None, 0)` 走，一个字都不会写库。
        真要长期留着，走页脚「切片并加入素材库」。
        """
        from ...dance import material_slice

        spans = self._live_spans()
        if not spans or not self._align_rows:
            return []
        head_room, tail_room = self.bench.rooms()
        segments: list[dict[str, Any]] = []
        for index, title, start, end in spans:
            materials = []
            for number, row in enumerate(self._align_rows, start=1):
                align = row["alignment"]
                offset = float(align.offset)
                source_duration = float(align.source_duration or 0.0)
                try:
                    spec = material_slice.map_to_source(
                        float(start), float(end), offset, source_duration,
                        segment_index=int(index),
                        head_room=head_room, tail_room=tail_room)
                except material_slice.SourceRangeError:
                    continue                 # 这一段在这个视频里不存在，格子留空
                source_start, source_end = spec.source_start, spec.source_end
                # 首尾补过静帧的话标出来，别让人以为整段都有画面
                missing = round(spec.head_pad + spec.tail_pad, 3)
                short = f"　补 {missing:.2f}s" if missing > 0.001 else ""
                materials.append({
                    # 内存里的临时编号：负数，一眼看出"不是库里的素材"，
                    # 也保证不会撞上真实 material_id（自增主键都是正数）
                    "material_id": -(number * 1000 + int(index) + 1),
                    # **这一格属于哪一段**。少了它 `matrix_panel.accepts()` 会取 −1，
                    # 于是拖也拒绝、点也拒绝 —— 落格判定就认这个键
                    "segment_index": int(index),
                    "video_id": number,
                    "video_name": str(row.get("name") or Path(row["path"]).name),
                    "label": f"{number:03d}",
                    "detail": f"源 {source_start:.2f}→{source_end:.2f}s{short}",
                    "path": str(row["path"]),
                    # 播放时用：源视频里的位置 = 主音频时间 − offset
                    "seek_base": offset,
                    "source_start": source_start,
                    "source_end": source_end,
                    "target_start": float(spec.target_start),
                    "target_end": float(spec.target_end),
                    "note": f"{row.get('name') or ''}｜还没入库（内存里的实时格子）\n"
                            f"offset {offset:+.3f}s｜置信 {float(align.confidence):.3f}"
                            + (f"\n源视频这一头不够长，切片时补 {missing:.2f}s 边界帧"
                               if short else ""),
                })
            segments.append({"index": int(index), "title": title,
                             "span": (float(start), float(end)),
                             "current": 0, "materials": materials})
        return segments

    def _reload_matrix(self) -> None:
        """素材矩阵：一列一个段落，**列跟着主音频上的分段实时变**。

        两种内容，按"库里有没有切好的片段"自动选：

        1. 库里有素材 → 照库来（候选顺序 `dance_candidate_order`、这一段用谁
           `dance_final_selections`），重开工程能长回上次的样子；
        2. 库里还没有 → 用内存里的对齐结果现算（`_live_segments`），
           所以**入库之前矩阵就有内容**，音谱上切一刀就多一列。

        分段一律以**主音频上那份模板**为准（不是库里存的那份），
        这样 ✂ 切分 / × 取消切分 一按，矩阵立刻跟着分列，不用先保存。
        """
        if self.db is None or not self._song_id:
            # 还没选目标歌也要能看：只要对齐算过，就按内存里那份铺格子
            self.matrix.attach(None, 0)
            self.matrix.load(self._live_segments())
            return
        from ...dance import material_repository as repo

        spans = self._live_spans()
        chosen = repo.final_selections(self.db, self._song_id)
        # 每个源视频给一个短号（001 / 002 …）：矩阵格子里只写这个，
        # 全名放在行首和悬浮提示里 —— 一格 130 像素塞不下 tiktok 那种长文件名
        numbers: dict[int, int] = {}
        for index, _title, _start, _end in spans:
            for material in repo.get_candidates(self.db, self._song_id, index):
                numbers.setdefault(int(material.source_video_id), len(numbers) + 1)
        if not numbers:
            # 库里这首歌还没有任何片段 → 走内存那份实时矩阵，别摆一张全是「—」的空表
            self.matrix.attach(None, 0)
            self.matrix.load(self._live_segments())
            self._refresh_repository()
            return
        self.matrix.attach(self.db, self._song_id)
        segments = []
        for index, title, start, end in spans:
            # 候选池只从这一段拿：`get_candidates(song, segment)` 没有"全曲随便挑"的口子
            materials = repo.order_materials(
                repo.get_candidates(self.db, self._song_id, index),
                repo.candidate_order(self.db, self._song_id, index))
            segments.append({
                "index": index, "title": title, "span": (start, end),
                "current": int(chosen.get(index, 0)),
                "materials": [
                    {"material_id": m.id,
                     "video_id": int(m.source_video_id),
                     "video_name": (getattr(m, "source_name", "")
                                    or Path(m.file_path).stem or f"#{m.source_video_id}"),
                     # 格子里只写短号（视频在矩阵里的行号 + 素材号），全名在行首和悬浮里
                     "label": f"{numbers.get(int(m.source_video_id), 0):03d}",
                     "detail": f"源 {m.source_start:.2f}→{m.source_end:.2f}s",
                     "path": m.file_path,
                     "source_start": float(m.source_start),
                     "source_end": float(m.source_end),
                     "target_start": float(m.target_start),
                     "target_end": float(m.target_end),
                     "score": round(float(m.quality or 0.0) * 100.0, 0),
                     "note": f"素材 #{m.id}｜{Path(m.file_path).name}\n"
                             f"对齐置信 {m.alignment_confidence:.3f}｜"
                             f"质量 {float(m.quality or 0.0):.2f}｜用过 {m.use_count} 次"}
                    for m in materials]})
        self.matrix.load(segments)
        template = self.master.template
        if template is not None:
            self.matrix.hint.setText(
                f"{self.matrix.hint.text()}　·　段落来自模板「{template.name}」"
                f"（{template.source}）")
        self._refresh_repository()

    def _adopt(self, picks: dict) -> None:
        self.candidates.adopt(dict(picks))
        self.remix.set_manual(self.candidates.manual())

    # ------------------------------------------------------------ 下架 / 删除
    def _remove_song(self, song_id: int) -> None:
        """移除一首目标歌。**下架是默认答案，彻底删除要额外一次确认。**

        为什么不给一个干脆的"删除"了事：`dance_materials` / `dance_audio_alignments` /
        `dance_montages` 都是 `ON DELETE CASCADE`，真删下去会把素材、对齐、
        历史成片和它们的事件流水一起抹掉。而这个项目的立足点就是"素材是长期资产"。
        所以这里先把挂在它下面的数量摆出来，让用户自己选。
        """
        from ...dance import material_repository as repo

        if self.db is None:
            return
        row = repo.get_song(self.db, int(song_id))
        if row is None:
            QMessageBox.warning(self, "找不到", f"库里没有目标歌 #{song_id}")
            self.reload()
            return
        title = str(row["title"] or f"#{song_id}")

        # 已经下架的：这一步变成"恢复"
        if repo.song_retirement(self.db, int(song_id)) is not None:
            answer = QMessageBox.question(
                self, "恢复目标歌",
                f"《{title}》现在是已下架状态。要恢复它吗？\n"
                "（下架期间素材和历史一条都没动，恢复后立刻能继续用）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer == QMessageBox.Yes:
                repo.restore_song(self.db, int(song_id))
                self.reload()
            return

        usage = repo.song_usage(self.db, int(song_id))
        detail = repo.describe_usage(usage)


        box = QMessageBox(self)
        box.setWindowTitle("移除目标歌")
        box.setIcon(QMessageBox.Question)
        box.setText(f"《{title}》下面挂着：{detail}")
        box.setInformativeText(
            "下架（推荐）：只是从界面上隐藏，素材、对齐、历史成片一条都不动，随时能恢复。\n\n"
            "彻底删除：连带删掉上面列出的全部数据库记录，**不可撤销**。\n"
            "只有在这首歌是误加进来的（选错文件、重复导入）时才该用它。\n"
            "磁盘上已经切好的素材文件不会被删，需要的话自己去素材目录清理。")
        hide = box.addButton("下架（可恢复）", QMessageBox.AcceptRole)
        drop = box.addButton("彻底删除", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(hide)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is hide:
            reason, ok = QInputDialog.getText(
                self, "写个理由", "为什么下架这首歌？（会连同时间一起存档）")
            if not ok or not reason.strip():
                QMessageBox.information(self, "没有下架",
                                        "没写理由，这次不下架 —— 不允许静默隐藏。")
                return
            repo.retire_song(self.db, int(song_id), reason=reason.strip(), operator="gui")
            self.remix.append_log(f"[目标歌] 《{title}》已下架：{reason.strip()}")
        elif clicked is drop:
            again = QMessageBox.warning(
                self, "最后确认",
                f"真的要彻底删除《{title}》吗？\n\n"
                f"连带删除：{detail}\n"
                "这一步不可撤销。想留后路请改用「下架」。",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if again != QMessageBox.Yes:
                return
            removed = repo.delete_song(self.db, int(song_id), confirm=True)
            self.remix.append_log(
                f"[目标歌] 《{title}》已彻底删除，连带 {repo.describe_usage(removed)}")

        else:
            return

        if int(song_id) == self._song_id:          # 当前正看着它，得把界面切回空
            self._song_id = 0
            self.remix.song.clear()
        self.reload()



    # ------------------------------------------------------------------ 跑活
    def start(self, job: dict) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "还在跑", "上一轮还没结束，等它跑完或者先点停止。")
            return
        if not job.get("song"):
            QMessageBox.warning(self, "还没选目标歌", "先选一首目标歌（文件或库里的 id）。")
            return
        if not job.get("recommend") and not job.get("manual"):
            QMessageBox.warning(self, "关了推荐就得自己挑",
                                "智能推荐已关闭，但「候选池」里还没有任何手动选择。\n"
                                "去候选池里逐格钉选，或者点「按推荐填满所有格」。")
            return
        job["strategy_id"] = self.recommend.strategy_id()
        job["preset"] = str(self.filters.preset.currentData() or "")
        # 进度和日志都在「输入与操作」那一栏里，开跑就把它拉出来 ——
        # 侧栏收着的时候啥都看不见，用户只会以为程序卡死了
        self.dock_remix.setVisible(True)
        self.worker = DanceMontageWorker(self.cfg, job, self)
        self.worker.log.connect(self.remix.append_log)
        self.worker.stage.connect(self.remix.show_stage)
        self.worker.progress.connect(self.remix.show_progress)
        self.worker.done.connect(self._finished)
        self.remix.set_running(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def _finished(self, ok: bool, message: str, result: object) -> None:
        self.remix.set_running(False)
        self.remix.show_done(bool(ok), str(message))
        data = result if isinstance(result, dict) else {}
        if data.get("song_id"):
            self._song_id = int(data["song_id"])
        self.reload()
        outputs = data.get("outputs") or []
        if ok and outputs:
            QMessageBox.information(
                self, "出片了",
                message + "\n\n" + "\n".join(str(Path(p).name) for p in outputs[:6])
                + f"\n\n目录：{Path(str(outputs[0])).parent}")
        elif not ok:
            QMessageBox.warning(self, "没出来", message)

    def _realign(self) -> None:
        job = self.remix.job()
        job.update({"force": True, "do_slice": False, "do_remix": False})
        self.start(job)

    def _rerender(self, version_id: int) -> None:
        """重渲染一个历史版本：读回计划，原样再跑一遍渲染。"""
        if self.db is None:
            return
        from ...dance import media_backend, montage_render, montage_timeline

        context = montage_timeline.load(self.db, int(version_id))
        if context is None:
            QMessageBox.warning(self, "读不出来", f"版本 #{version_id} 的计划读不出来。")
            return
        problems = montage_timeline.validate(context)
        if problems:
            QMessageBox.warning(self, "这一版现在渲染不了", "\n".join(problems[:6]))
            return
        # 画布跟这一版用到的片段走：重渲染出来的尺寸要和当初那一版一致
        canvas = media_backend.resolve_canvas(
            self.cfg, [clip.file_path for clip in context.clips])
        self.remix.append_log(f"[重渲染] 版本 #{version_id}｜{len(context.clips)} 格")
        result = montage_render.render(
            self.db, context, out_dir=self.cfg.dance_path("output_dir"),
            version_id=int(version_id), canvas=canvas,
            on_log=self.remix.append_log)
        for line in montage_render.describe(result):
            self.remix.append_log(line)
        self.reload()
        if result.ok:
            QMessageBox.information(self, "重渲染完成", str(result.output))
        else:
            QMessageBox.warning(self, "重渲染失败", result.error or "未知原因")

    # ------------------------------------------------------------------ 收尾
    def closeEvent(self, event) -> None:                     # noqa: N802 - Qt 的名字
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self, "还在跑", "后台还在跑，真的关掉吗？（会等当前这一步做完）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait(15000)
        self.bench.shutdown()          # 测试台自己那两个线程也要收干净
        self.master.shutdown()         # 主音频分析线程同理
        self.prefetch.shutdown()       # 片段预解码线程
        self.save_settings()           # 窗口位置、分栏比例、各输入框都留到下次
        if self.db is not None:
            self.db.close()
            self.db = None
        super().closeEvent(event)


__all__ = ["TITLE", "DanceMontageWindow"]
