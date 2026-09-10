"""AI_卡点舞 主窗口。**独立窗口**，和主界面各开各的，谁崩了都不影响谁。

一期第二十四节的四个区域在这里落地：

    ① 左侧：输入与操作（选歌、选源、参数、开跑、进度、日志）  → RemixPanel
    ② 右上：素材资产（筛选 + 素材库）                        → FilterPanel + MaterialLibraryPanel
    ③ 右中：选择与推荐（候选池 + 智能推荐）                   → CandidatePanel + RecommendationPanel
    ④ 右下：历史与统计                                       → HistoryPanel + StatisticsPanel

界面线程只做两件事：画界面、读库（都是聚合查询，毫秒级）。
所有重活在 `DanceMontageWorker` 里，它自己开自己的库连接。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


from ...logging_setup import get_logger
from .. import settings as gui_settings
from .. import theme
from ..player import FramePlayer
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
from .video_coverage import VideoCoverageBar
from .worker import DanceMontageWorker

logger = get_logger("dance.gui")

TITLE = "AI_卡点舞 · 舞蹈素材资产与多版本混剪"


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
        sizes = state.get("split")
        if (isinstance(sizes, list) and len(sizes) == 2
                and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
            self.split.setSizes(sizes)
            self._left_width = sizes[0] or 560
        self.remix.restore(state.get("remix") or {})
        self.master.restore(state.get("master") or {})
        self.bench.restore(state.get("bench") or {})
        index = state.get("tab")
        if isinstance(index, int) and 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)
        # 开窗口时也得按当前这一页决定左栏收不收 —— setCurrentIndex 命中同一页不发信号
        self._tab_changed(self.tabs.currentIndex())
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
        # 测试台那一页会把左栏收成 0，别把 0 存成"用户想要的宽度"
        sizes = self.split.sizes()
        self.state["split"] = ([getattr(self, "_left_width", 560), sizes[1]]
                               if sizes and sizes[0] == 0 else list(sizes))
        self.state["tab"] = int(self.tabs.currentIndex())
        self.state["remix"] = self.remix.state()
        self.state["master"] = self.master.state()
        self.state["bench"] = self.bench.state()
        gui_settings.save(self.cfg, self.settings)


    # ------------------------------------------------------------------ 界面
    def _build_body(self) -> QWidget:
        assets = QWidget(self)
        assets_layout = QHBoxLayout(assets)
        assets_layout.setContentsMargins(6, 6, 6, 6)
        assets_layout.addWidget(self.filters, 1)
        assets_layout.addWidget(self.library, 3)

        choose = QWidget(self)
        choose_layout = QVBoxLayout(choose)
        choose_layout.setContentsMargins(6, 6, 6, 6)
        inner = QSplitter(Qt.Vertical, choose)
        inner.addWidget(self.candidates)
        inner.addWidget(self.recommend)
        inner.setStretchFactor(0, 3)
        inner.setStretchFactor(1, 2)
        choose_layout.addWidget(inner)

        review = QWidget(self)
        review_layout = QHBoxLayout(review)
        review_layout.setContentsMargins(6, 6, 6, 6)
        review_layout.addWidget(self.history, 3)
        review_layout.addWidget(self.statistics, 2)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_studio(), "🎵 编排台（主音频→分段→素材→成片）")
        self.tabs.addTab(assets, "素材资产")
        self.tabs.addTab(self.alignment, "音频对齐")
        self.tabs.addTab(choose, "选择与推荐")
        self.tabs.addTab(review, "历史与统计")

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        hint = QLabel("固定音乐位置：所有源视频都贴在**目标歌**这同一把尺子上。"
                      "素材是长期资产，只会停用、不会删除。", left)
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        left_layout.addWidget(self.remix, 1)

        split = QSplitter(Qt.Horizontal, self)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([560, 900])
        self.split = split
        self.tabs.currentChanged.connect(self._tab_changed)
        return split

    def _build_studio(self) -> QWidget:
        """编排台：**主音频 → 分段 → 素材 → 成片**，一页从上到下走完：

            ① 一行工具栏：源视频 / 视频文件夹 / 主音频（＝目标歌）/ [开始音频对齐]
            ② 视频 + 音谱同一块：**上边实时播放画面，下边视频位置 + 波形/人声/音谱 + 段落条**
            ③ 素材矩阵：列＝Segment，行＝视频，第一行＝实时播放（成片就读它）
            ④ 页脚：入库三件事 + 片段仓库 + 预览/保存/导出

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
        stack.addWidget(self._build_stage())
        stack.addWidget(self.matrix)
        stack.setStretchFactor(0, 0)
        stack.setStretchFactor(1, 3)
        stack.setStretchFactor(2, 2)
        # 顶栏那一条按它自己要的高度给，别硬编一个数字把「开始音频对齐」压掉；
        # 矩阵至少留 300 像素：实时播放行 + 几行视频要一眼看得全
        self.matrix.setMinimumHeight(300)
        stack.setSizes([self.bench.minimumHeight() or 120, 480, 420])
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

    def _build_stage(self) -> QWidget:
        """② 视频 + ③ 音谱区：**同一个布局里，上边视频、下边音谱**。

            ┌──────────────────────────────┐
            │  ▶ 实时播放（视频画面）        │  ← 上
            ├──────────────────────────────┤
            │  视频位置（一条总览带）        │
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
        vbox.addWidget(self.live, 1)
        vbox.addWidget(self.live_note)
        stack.addWidget(video)

        below = QWidget(stack)
        bbox = QVBoxLayout(below)
        bbox.setContentsMargins(0, 0, 0, 0)
        bbox.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(8, 0, 8, 0)
        head.addWidget(QLabel("视频位置", below))
        self.coverage_note = QLabel("", below)
        self.coverage_note.setStyleSheet(f"color:{theme.TEXT_DIM};")
        head.addWidget(self.coverage_note, 1)
        bbox.addLayout(head)
        self.coverage = VideoCoverageBar(below)
        self.coverage.seeked.connect(self.master.seek_to)
        bbox.addWidget(self.coverage)
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
        self.coverage.set_playhead(moment)
        template = self.master.template
        span = template.span_at(moment) if template is not None else None
        index = int(span.index) if span is not None else -1
        self.matrix.set_current_segment(index)
        self._sync_live(self.matrix.current_payload(index) if index >= 0 else {}, moment)

    def _sync_live(self, payload: dict, moment: float) -> None:
        """让实时画面跟上主音频。**这一段没素材就不播**。

        不找别的段、不用上一段顶替、不随机补一条 —— 画面就保持空，
        主音频照旧往下走。跨段落取素材会让画面和音乐错开，那是这套系统最不能出的错。

        片段文件是按对齐结果切好的（`target_start` 就是它在主音频上的起点），
        所以片段内位置 = `主音频时间 − 段落起点`，offset 已经在切片那一步算过，
        这里不再自己减第二遍。
        """
        material = int(payload.get("material_id") or 0)
        if material <= 0:
            self._live_material = 0
            self.live.pause()
            self.live.close_video()
            self.live_note.setText("这一段没有素材 —— 画面保持空（不会自动补别的段）")
            return
        if material != self._live_material:
            if not self.live.open(str(payload.get("path") or "")):
                self._live_material = 0
                self.live_note.setText(f"素材 #{material} 的文件打不开，画面保持空")
                return
            self._live_material = material
            self.live.set_audio_enabled(False)
        start = float(payload.get("target_start") or 0.0)
        self.live.seek(max(0.0, float(moment) - start))
        playing = self.master.player.state() == QMediaPlayer.PlayingState
        self.live.play() if playing else self.live.pause()
        self.live_note.setText(
            f"{payload.get('video_name') or ''}　素材 #{material}　"
            f"片段内 {max(0.0, float(moment) - start):.3f}s")

    def _preview_final(self) -> None:

        """▶ 预览：放主音频（成片的音轨就是它），画面预览走素材卡片上的 ▶。"""
        self.tabs.setCurrentIndex(0)
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
        self.remix.set_manual(picks)
        job = self.remix.job()
        job["manual"] = dict(picks)
        job["recommend"] = False          # 用户已经拍板了，别让推荐再插手
        if not job.get("song"):
            QMessageBox.information(self, "还没选目标歌",
                                    "左边「输入与操作」里先选一首目标歌（或填库里的 id）。")
            return
        self.start(job)

    def _tab_changed(self, index: int) -> None:
        """切到「编排台」时把左边那栏收起来，让它占满整个窗口。

        它是横向铺开的工作台（矩阵一列一个段落），挤在 900 像素里没法用；
        而它本来就不需要左边那套混剪参数。切回别的页时恢复原来的宽度。
        """
        wide = self.tabs.widget(int(index)) is self.studio
        sizes = self.split.sizes()
        if wide:
            if sizes[0] > 0:
                self._left_width = sizes[0]
            self.split.setSizes([0, sum(sizes) or 1])
        elif sizes[0] == 0:
            width = getattr(self, "_left_width", 560)
            self.split.setSizes([width, max(1, sum(sizes) - width)])


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

        self.candidates.manual_changed.connect(self.remix.set_manual)
        # 矩阵里拖出来的选择就是"手动指定这一格用谁"，和候选面板同一个出口
        self.matrix.picks_changed.connect(self.remix.set_manual)
        self.matrix.picks_changed.connect(lambda _p: self._touched("picks"))
        self.matrix.refused.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.matrix.saved.connect(lambda text: self.statusBar().showMessage(text, 4000))
        self.matrix.preview_requested.connect(self._preview_material)
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

    def _save_everything(self) -> None:
        """Ctrl+S / 「保存」：分段和编排一起存（编排台上两样都在同一页）。"""
        current = self.tabs.currentWidget()
        if current is self.studio:
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

    def _reload_matrix(self) -> None:
        """素材矩阵：一列一个段落。段落用用户拍板的模板，没模板就按等间隔位置。

        候选顺序和"这一段用谁"都从库里读（`dance_candidate_order` /
        `dance_final_selections`）—— 重新打开工程之后能长回上次的样子就靠这两张表。
        """
        if self.db is None or not self._song_id:
            self.matrix.load([])
            return
        from ...dance import material_repository as repo
        from ...dance.music_structure import target_positions

        self.matrix.attach(self.db, self._song_id)
        row = repo.get_song(self.db, self._song_id)
        duration = float(row["duration"] or 0.0) if row is not None else 0.0
        template = repo.active_segment_template(self.db, self._song_id)
        if template is not None:
            spans = [(s.index, s.name, s.start, s.end) for s in template.spans]
        else:
            spans = [(p.index, f"S{p.index + 1}", p.start, p.end) for p in
                     target_positions(duration, self.remix.slice_duration())]

        chosen = repo.final_selections(self.db, self._song_id)
        # 每个源视频给一个短号（001 / 002 …）：矩阵格子里只写这个，
        # 全名放在行首和悬浮提示里 —— 一格 130 像素塞不下 tiktok 那种长文件名
        numbers: dict[int, int] = {}
        for index, _title, _start, _end in spans:
            for material in repo.get_candidates(self.db, self._song_id, index):
                numbers.setdefault(int(material.source_video_id), len(numbers) + 1)
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
        if template is not None:
            self.matrix.hint.setText(
                f"{self.matrix.hint.text()}　·　段落来自模板「{template.name}」"
                f"（{template.source}）")
        self._refresh_repository()
        self._refresh_coverage(duration)

    def _refresh_coverage(self, duration: float) -> None:
        """视频位置那条带子：每个已对齐视频在**主音频**上盖住哪一段。

        换算只有一处：`target_time = source_time + offset`（对齐那层的约定），
        所以覆盖范围就是 `offset → offset + 源视频时长`，掐进 [0, 歌长] 里。
        每个 mp4 **不**单独画一条时间轴 —— 那是用户明确不要的界面。
        """
        if self.db is None or not self._song_id:
            self.coverage.set_coverage(0.0, [])
            self.coverage_note.setText("")
            return
        from ...dance import material_repository as repo

        items = []
        for row in repo.alignments_for_song(self.db, self._song_id,
                                            statuses=("ok", "manual")):
            offset = float(row["offset_seconds"] or 0.0)
            length = float(row["source_duration"] or 0.0)
            start = max(0.0, offset)
            end = min(float(duration) if duration > 0 else offset + length, offset + length)
            if end > start:
                items.append((start, end, str(row["source_name"] or f"#{row['id']}")))
        self.coverage.set_coverage(float(duration), items)
        self.coverage_note.setText(f"{len(items)} 个已对齐视频盖在这首歌上"
                                   if items else "还没有对齐好的源视频")



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
        canvas = media_backend.Canvas(width=int(self.cfg.dance["canvas_width"]),
                                      height=int(self.cfg.dance["canvas_height"]),
                                      fps=float(self.cfg.dance["canvas_fps"]))
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
        self.save_settings()           # 窗口位置、分栏比例、各输入框都留到下次
        if self.db is not None:
            self.db.close()
            self.db = None
        super().closeEvent(event)


__all__ = ["TITLE", "DanceMontageWindow"]
