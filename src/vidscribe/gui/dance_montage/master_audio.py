"""MASTER AUDIO 编辑区：主音频 + 人声导航 + 段落模板编辑。

界面分三块，对应你要的那张图：

    左：🎤 人声导航（每个停顿一行，点一行播放头就跳过去）
    中：时间轴（音谱图/波形/人声带/⏸/段落条/播放头，见 master_timeline.py）
    下：状态栏 —— 当前时间、当前段、在唱还是停、下一处停顿多远，以及那几个动作按钮

一条贯穿全局的分工，代码里到处都在守它：

    人声停顿 = 系统算出来的**参考点**（会犯错，只标出来）
    Segment 边界 = 用户**拍板**的结果（拖/切/合，存进 dance_segment_templates）

所以：分析永远不会自动改分段；「照停顿分」是一个得用户点的按钮；
拖边界拖不过去时报错（`segment_template.move_boundary` 抛 SegmentError），
界面把话说出来，绝不悄悄挪到别处。

重活（解码 + FFT + 人声判定）全在 `MasterAudioWorker` 里，主线程一次都不碰音频。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QShortcut,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...logging_setup import get_logger
from .. import theme
from . import dialogs
from .master_timeline import DISPLAY_LABELS, DISPLAY_MODES, MasterTimeline

logger = get_logger("dance.gui.master")

#: 波形缩略图桶数 / 音谱图列数：和控件宽度同量级就够，再多肉眼看不出
ENVELOPE_BUCKETS = 900
SPECTRUM_COLUMNS = 900
SPECTRUM_ROWS = 96

BUTTON_HEIGHT = 34
FIELD_HEIGHT = 30


def _big(button, height: int = BUTTON_HEIGHT, *, bold: bool = False):
    button.setMinimumHeight(height)
    if bold:
        font = button.font()
        font.setBold(True)
        button.setFont(font)
    return button


def _clock(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    return f"{minutes}:{total - minutes * 60:06.3f}"


class MasterAudioWorker(QThread):
    """解码主音频 → 波形包络 + 音谱图 + 拍网格 + 人声活动。**只算，不落库、不改分段。**

    重模块（av / numpy / dance.dsp）都在 `run()` 里 import，和别的 worker 一个规矩：
    GUI 进程刚起来的时候不该为了画一个面板去加载解码器。
    """

    log = pyqtSignal(str)
    done = pyqtSignal(bool, str, object)

    def __init__(self, cfg, path: str, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.path = str(path)

    def run(self) -> None:
        payload: dict[str, Any] = {}
        try:
            from ...dance import dsp  # noqa: PLC0415
            from ...dance import material_ingest as ingest  # noqa: PLC0415
            from ...dance import music_structure as structure  # noqa: PLC0415
            from ...dance import vocal_activity as vocal  # noqa: PLC0415
            from ...dance.audio_fingerprint import extract_analysis_audio  # noqa: PLC0415
            from ...db import open_db  # noqa: PLC0415


            self.log.emit(f"[主音频] 解码 {Path(self.path).name}")
            pcm = extract_analysis_audio(self.path)
            duration = round(pcm.size / float(dsp.DEFAULT_SR), 3)
            self.log.emit(f"[主音频] 时长 {duration:.3f}s，开始算波形和音谱图")

            payload["duration"] = duration
            payload["envelope"] = [float(v) for v in dsp.envelope(pcm, ENVELOPE_BUCKETS)]
            payload["spectrogram"] = dsp.spectrogram_image(
                pcm, dsp.DEFAULT_SR, columns=SPECTRUM_COLUMNS, rows=SPECTRUM_ROWS)

            grid = structure.detect_target_beat_grid(pcm)
            payload["beats"] = list(grid.beats)
            payload["bpm"] = float(grid.bpm)
            self.log.emit(f"[主音频] 拍网格：{grid.bpm:.1f} BPM，{len(grid.beats)} 拍"
                          f"（来源 {grid.source}）")

            activity = vocal.analyze_vocal_activity(pcm, dsp.DEFAULT_SR, tuple(grid.beats))
            payload["activity"] = activity
            self.log.emit(f"[主音频] 人声：{sum(1 for s in activity.spans if s.kind == 'vocal')}"
                          f" 段在唱，找到 {len(activity.pauses)} 处可用停顿")

            # 顺手把这首歌登记进库：位置尺子是目标歌的属性，没有歌就没有"第几段"可谈，
            # 段落模板也无处挂。按指纹幂等，同一首歌反复分析只有一行。
            db = open_db(self.cfg)
            try:
                song = ingest.register_song(db, self.path, on_log=self.log.emit)
                payload["song_id"] = int(song.song_id)
            finally:
                db.close()
            self.done.emit(True, "分析完成", payload)
        except Exception as exc:  # noqa: BLE001 - 后台线程里必须自己兜住
            logger.exception("主音频分析失败")
            self.done.emit(False, f"{type(exc).__name__}: {exc}", payload)


class MasterAudioPanel(QWidget):
    """主音频编辑区。段落定稿之后发 `template_changed(SegmentTemplate)` 通知别的页。"""

    template_changed = pyqtSignal(object)
    seek_requested = pyqtSignal(float)

    def __init__(self, cfg, db=None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.db = db
        self.worker: MasterAudioWorker | None = None

        self._activity = None          # VocalActivity
        self._template = None          # SegmentTemplate
        self._duration = 0.0
        self._beats: list[float] = []
        self._at = 0.0                 # 当前时间（秒）
        self._song_id = 0
        self._undo: list[Any] = []     # 每次改动前的模板，用来撤销
        self._redo: list[Any] = []     # 撤销掉的那些，用来重做
        self._stop_at: float | None = None   # 「播放当前段」到这里自动停
        self._analyzed_path = ""       # 上一次分析的是哪首，免得同一首分析两遍


        # 主音频播放器：QtMultimedia 自带位置回调，播放头能真的跟着走。
        # 素材预览那边用的是 FramePlayer（要出画面），这里只放音，两者不冲突。
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(50)
        self.player.positionChanged.connect(self._position_changed)
        self.player.stateChanged.connect(lambda _s: self._refresh_status())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._column = layout
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_transport())
        body = QSplitter(Qt.Horizontal, self)
        body.addWidget(self._build_navigator())
        body.addWidget(self._build_stage())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([260, 1100])
        self._body = body
        layout.addWidget(body, 1)
        layout.addWidget(self._build_status())
        self._install_shortcuts()

    # ---------------------------------------------------------------- 顶部
    def _build_header(self) -> QWidget:
        holder = QFrame(self)
        holder.setFrameShape(QFrame.NoFrame)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.path = QLineEdit(holder)
        self.path.setMinimumHeight(FIELD_HEIGHT)
        self.path.setPlaceholderText("主音频＝目标歌（这首歌决定所有段落）")
        btn_pick = _big(QPushButton("选主音频…", holder), FIELD_HEIGHT)
        # 「分析主音频」不再是一个要点的按钮：选好歌就自动开始分析，少一步。
        # 按钮本身留着（禁用状态还要用它挡住重复分析），只是不摆在界面上
        self.btn_analyze = _big(QPushButton("分析主音频", holder), 40, bold=True)
        self.btn_analyze.setVisible(False)
        self.bar = QProgressBar(holder)
        self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        self.bar.setMaximumWidth(160)
        self.info = QLabel("还没分析", holder)
        self.info.setStyleSheet(f"color:{theme.TEXT_DIM};")

        row.addWidget(QLabel("主音频", holder))
        row.addWidget(self.path, 1)
        row.addWidget(btn_pick)
        row.addWidget(self.bar)
        row.addWidget(self.info)

        btn_pick.clicked.connect(self._pick)
        self.btn_analyze.clicked.connect(self.analyze)
        self.path.editingFinished.connect(self._path_typed)
        self.header = holder
        return holder

    def take_song_row(self) -> QWidget:
        """把「主音频 + 选主音频…」那一条交出去，让编排台把它并进最上面那一行。

        主音频就是目标歌，界面上只该有一处填它。
        """
        self._column.removeWidget(self.header)
        self.header.setParent(None)
        return self.header


    # ------------------------------------------------------------ 播放控制条
    def _build_transport(self) -> QWidget:
        """🎵 MASTER AUDIO 的播放条：播放/暂停/停止、音量、时间、缩放、滚动。"""
        holder = QFrame(self)
        holder.setFrameShape(QFrame.StyledPanel)
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        self.btn_play = _big(QPushButton("▶ 播放", holder), bold=True)
        self.btn_pause = _big(QPushButton("⏸ 暂停", holder))
        self.btn_stop = _big(QPushButton("■ 停止", holder))
        self.clock = QLabel("0:00.000 / 0:00.000", holder)
        self.volume = QSlider(Qt.Horizontal, holder)
        self.volume.setRange(0, 100)
        self.volume.setValue(70)
        self.volume.setMaximumWidth(120)
        self.player.setVolume(self.volume.value())

        self.follow = QCheckBox("播放时自动跟随", holder)
        self.follow.setChecked(True)
        # 加减速：慢放是找准停顿最管用的一招 —— 0.5× 下人声起收听得清楚多了。
        # 只改播放速度，音频文件、offset、段落时间一个都不动，倍速下时钟也照常走真实秒
        self.speeds = QComboBox(holder)
        self.speeds.setMinimumHeight(FIELD_HEIGHT)
        self.speeds.setToolTip("只影响试听快慢；分段用的时间还是真实秒数")
        for text, rate in (("0.5×", 0.5), ("0.75×", 0.75), ("1.0×", 1.0),
                           ("1.25×", 1.25), ("1.5×", 1.5), ("2.0×", 2.0)):
            self.speeds.addItem(text, rate)
        self.speeds.setCurrentIndex(2)
        self.zooms = QComboBox(holder)

        self.zooms.setMinimumHeight(FIELD_HEIGHT)
        for text, span in (("全曲", 0.0), ("1 分钟", 60.0), ("30 秒", 30.0),
                           ("10 秒", 10.0)):
            self.zooms.addItem(text, span)
        btn_in = _big(QPushButton("放大 +", holder), FIELD_HEIGHT)
        btn_out = _big(QPushButton("缩小 −", holder), FIELD_HEIGHT)

        for widget in (self.btn_play, self.btn_pause, self.btn_stop, self.clock,
                       QLabel("音量", holder), self.volume,
                       QLabel("速度", holder), self.speeds):
            row.addWidget(widget)
        row.addStretch(1)
        for widget in (self.follow, QLabel("视图", holder), self.zooms, btn_in, btn_out):
            row.addWidget(widget)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.volume.valueChanged.connect(self.player.setVolume)
        self.speeds.currentIndexChanged.connect(self._speed_changed)
        self.zooms.currentIndexChanged.connect(self._zoom_preset)
        btn_in.clicked.connect(lambda: self.timeline.zoom(0.5))
        btn_out.clicked.connect(lambda: self.timeline.zoom(2.0))
        self._speed_changed()      # 空播放器的速率是 0，先按下拉框那一档钉成 1.0×
        return holder

    def _speed_changed(self, _index: int = 0) -> None:
        """播放速率换挡。停着的时候也先记上，下次一播就是这个速度。"""
        rate = float(self.speeds.currentData() or 1.0)
        self.player.setPlaybackRate(rate)



    # ------------------------------------------------------------ 左：导航
    def _build_navigator(self) -> QWidget:
        holder = QFrame(self)
        holder.setFrameShape(QFrame.StyledPanel)
        column = QVBoxLayout(holder)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(6)

        column.addWidget(QLabel("🎤 人声导航（点一行 → 播放头跳过去）", holder))
        self.nav = QListWidget(holder)
        self.nav.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nav.setAlternatingRowColors(True)
        self.nav.setMinimumWidth(220)
        column.addWidget(self.nav, 1)

        self.only_strong = QCheckBox("只看推荐度高的停顿（⭐）", holder)
        self.only_strong.setToolTip("推荐度 = 停得久 + 干净 + 贴着拍点。\n"
                                    "低分的停顿多半是换气，不适合换人。")
        column.addWidget(self.only_strong)

        self.nav.currentRowChanged.connect(self._nav_picked)
        self.only_strong.toggled.connect(self._fill_navigator)
        return holder

    # ------------------------------------------------------------ 中：时间轴
    def _build_stage(self) -> QWidget:
        holder = QWidget(self)
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self.timeline = MasterTimeline(holder)
        self.timeline.setToolTip("左键按住＝拖动播放位置；拖分段线＝改段落边界；\n"
                                 "中键拖（或 Alt+左键拖）＝抓着音轨左右挪；\n"
                                 "滚轮＝缩放，Shift+滚轮＝横向滚动；右键＝换显示 / 切分。")
        self.timeline.setContextMenuPolicy(Qt.CustomContextMenu)
        self.timeline.customContextMenuRequested.connect(self._timeline_menu)
        column.addWidget(self.timeline, 1)



        # 横向滚动条：窗口小于全曲时才有意义，所以跨度变了就跟着调
        self.scroll = QScrollBar(Qt.Horizontal, holder)
        self.scroll.setEnabled(False)
        self.scroll.valueChanged.connect(
            lambda value: self.timeline.scroll_to(value / 1000.0))
        self.timeline.view_changed.connect(self._view_changed)
        column.addWidget(self.scroll)

        tools = QHBoxLayout()
        tools.setSpacing(6)
        self.step = QDoubleSpinBox(holder)
        self.step.setRange(0.5, 30.0)
        self.step.setSingleStep(0.5)
        self.step.setValue(2.0)
        self.step.setSuffix(" 秒一段")
        self.step.setMinimumHeight(FIELD_HEIGHT)
        self.min_score = QDoubleSpinBox(holder)
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setValue(0.5)
        self.min_score.setPrefix("停顿推荐度 ≥ ")
        self.min_score.setMinimumHeight(FIELD_HEIGHT)
        btn_uniform = _big(QPushButton("等间隔起步", holder))
        btn_by_pause = _big(QPushButton("照人声停顿分", holder))
        self.btn_undo = _big(QPushButton("撤销上一步", holder))
        self.btn_undo.setEnabled(False)
        self.btn_save = _big(QPushButton("保存这份分段", holder), bold=True)

        for widget in (self.step, btn_uniform, self.min_score, btn_by_pause):
            tools.addWidget(widget)
        tools.addStretch(1)
        tools.addWidget(self.btn_undo)
        tools.addWidget(self.btn_save)
        column.addLayout(tools)

        self.timeline.seeked.connect(self._moved_to)
        self.timeline.boundary_dragged.connect(self._drag_preview)
        self.timeline.boundary_committed.connect(self._drag_commit)
        btn_uniform.clicked.connect(self._make_uniform)
        btn_by_pause.clicked.connect(self._make_by_pause)
        self.btn_undo.clicked.connect(self._undo_once)
        self.btn_save.clicked.connect(self.save_template)
        return holder

    # ------------------------------------------------------------ 下：状态栏
    def _build_status(self) -> QWidget:
        holder = QFrame(self)
        holder.setFrameShape(QFrame.StyledPanel)
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        self.status = QLabel("当前：—", holder)
        self.anchor = QComboBox(holder)
        self.anchor.setMinimumHeight(FIELD_HEIGHT)
        for text, key in (("跳到停顿开始", "start"), ("跳到停顿中心", "middle"),
                          ("跳到停顿结束", "end")):
            self.anchor.addItem(text, key)
        self.btn_prev = _big(QPushButton("◀ 上一个停顿", holder))
        self.btn_next = _big(QPushButton("▶ 下一个停顿", holder))
        self.btn_split = _big(QPushButton("✂ 切分", holder), bold=True)
        self.btn_unsplit = _big(QPushButton("× 取消切分", holder))
        self.btn_unsplit.setToolTip("删掉离播放位置最近的那条分段线（两段合成一段）。\n"
                                   "它删的是**边界**，一条素材都不会动。")
        self.btn_merge = _big(QPushButton("⇆ 合并前一段", holder))
        self.btn_play_span = _big(QPushButton("▶ 播放当前段", holder))

        row.addWidget(self.status, 1)
        for widget in (self.anchor, self.btn_prev, self.btn_next, self.btn_split,
                       self.btn_unsplit, self.btn_merge, self.btn_play_span):
            row.addWidget(widget)

        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(1))
        self.btn_split.clicked.connect(self._split_here)
        self.btn_unsplit.clicked.connect(self.unsplit_here)
        self.btn_merge.clicked.connect(self._merge_here)
        self.btn_play_span.clicked.connect(self.play_current_segment)
        return holder

    def _timeline_menu(self, point) -> None:
        """时间轴右键菜单。

        点在分段线上 → 只给「取消切分」（它删的是边界，不是素材）；
        点在段落里   → 播放当前段 / 在此处切分；
        另外永远有一层「显示」用来换波形 / 人声 / 音谱。
        菜单里**没有任何"标记"** —— 那个功能已经整块删掉了。
        """
        moment = self.timeline.time_at(point.x())
        boundary = self.timeline.boundary_near(point.x())
        menu = QMenu(self)
        show = menu.addMenu("显示")
        for key in DISPLAY_MODES:
            action = show.addAction(DISPLAY_LABELS[key])
            action.setCheckable(True)
            action.setChecked(self.timeline.display_mode == key)
            action.triggered.connect(lambda _c=False, k=key: self.timeline.set_display_mode(k))
        menu.addSeparator()
        if boundary >= 0:
            menu.addAction("× 取消切分").triggered.connect(
                lambda _c=False, m=moment: self.unsplit_here(m))
        else:
            menu.addAction("播放到此处").triggered.connect(
                lambda _c=False, m=moment: self._seek_and_play(m))
            if self.timeline.segment_at(moment) >= 0:
                menu.addAction("▶ 播放当前段").triggered.connect(
                    lambda _c=False, m=moment: self._play_segment_at(m))
            menu.addAction("✂ 在此处切分").triggered.connect(
                lambda _c=False, m=moment: self._split_at(m))
        menu.exec_(self.timeline.mapToGlobal(point))

    def _seek_and_play(self, moment: float) -> None:
        """「播放到此处」：先跳过去再放，播放位置永远是主音频说了算。"""
        self._moved_to(float(moment))
        self.player.setPosition(int(max(0.0, float(moment)) * 1000))
        self.play()

    def _play_segment_at(self, moment: float) -> None:
        self._moved_to(float(moment))
        self.play_current_segment()

    def _split_at(self, moment: float) -> None:
        """在右键点的那一刻切一刀（先把播放位置挪过去，再走同一条切分路径）。"""
        self._moved_to(float(moment))
        self._split_here()

    # ---------------------------------------------------------------- 快捷键

    def _install_shortcuts(self) -> None:
        """这一页自己的快捷键。作用域是**本控件及其子控件**，不抢主窗口那几个。

        Space / K 播放暂停、← J 上一个停顿、→ L 下一个停顿、
        S 在这里切分、Delete 取消最近的那条切分。
        """
        def bind(keys: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)

        bind("Space", self.toggle_play)
        bind("K", self.toggle_play)
        bind("Left", lambda: self._jump(-1))
        bind("J", lambda: self._jump(-1))
        bind("Right", lambda: self._jump(1))
        bind("L", lambda: self._jump(1))
        bind("S", self._split_here)
        bind("Delete", self.unsplit_here)


    # ---------------------------------------------------------------- 分析
    def _pick(self) -> None:
        # 签名是 (parent, title, folder, filters, key)：folder 给空串，
        # 起始目录交给 key 那份"上次去过哪儿"的记忆
        picked = dialogs.open_file(self, "选主音频", "", dialogs.AUDIO_FILTER,
                                   key="dance.song")
        if picked:
            self.path.setText(picked)
            self.analyze()          # 选完就直接分析，不用再点一下

    def _path_typed(self) -> None:
        """手打/粘贴完路径按回车也算"选好了"，同一首歌不重复分析。"""
        target = self.path.text().strip()
        if target and target != self._analyzed_path and Path(target).is_file():
            self.analyze()

    def analyze(self) -> None:
        """后台分析主音频。**不改分段** —— 分析只是把参考信息摆出来。"""
        target = self.path.text().strip()
        if not target:
            QMessageBox.information(self, "还没选主音频", "先选一首主音频再分析。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._analyzed_path = target
        self.btn_analyze.setEnabled(False)

        self.bar.setVisible(True)
        self.info.setText("正在解码和分析…")
        self.worker = MasterAudioWorker(self.cfg, target, self)
        self.worker.log.connect(logger.info)
        self.worker.done.connect(self._analyzed)
        self.worker.start()

    def _analyzed(self, ok: bool, message: str, payload: object) -> None:
        self.btn_analyze.setEnabled(True)
        self.bar.setVisible(False)
        data = payload if isinstance(payload, dict) else {}
        if not ok:
            self.info.setText(f"分析失败：{message}")
            QMessageBox.warning(self, "主音频分析失败", message)
            return

        self._duration = float(data.get("duration") or 0.0)
        self._song_id = int(data.get("song_id") or 0)
        self._beats = [float(b) for b in data.get("beats") or ()]
        self._activity = data.get("activity")
        self.timeline.set_song(self._duration, data.get("envelope") or [])
        self.timeline.set_spectrogram(data.get("spectrogram"))
        self.timeline.set_beats(self._beats)
        if self._activity is not None:
            self.timeline.set_vocal(self._activity.spans, self._activity.pauses)
            from ...dance import vocal_activity as vocal  # noqa: PLC0415

            self.timeline.set_zones(vocal.cut_zones(self._activity))
        self.timeline.set_view(0.0, 0.0)
        # 播放器指向这个文件：从这一刻起播放头是真的跟着音频走
        chosen = self.path.text().strip()
        if chosen:
            self.player.setMedia(
                QMediaContent(QUrl.fromLocalFile(str(Path(chosen).resolve()))))
        self._fill_navigator()
        self._load_saved_template()
        pauses = len(self._activity.pauses) if self._activity is not None else 0
        self.info.setText(f"{_clock(self._duration)}　·　{data.get('bpm', 0.0):.1f} BPM"
                          f"　·　{pauses} 处停顿")
        self._refresh_clock()
        self._refresh_status()

    # ------------------------------------------------------------ 导航列表
    def _fill_navigator(self) -> None:
        self.nav.clear()
        if self._activity is None:
            return
        threshold = 0.6 if self.only_strong.isChecked() else -1.0
        for pause in self._activity.pauses:
            if pause.score < threshold:
                continue
            star = " ⭐" if pause.score >= 0.6 else ""
            item = QListWidgetItem(f"⏸ {_clock(pause.start)}　停顿 {pause.duration:.2f}s{star}")
            item.setData(Qt.UserRole, pause.index)
            item.setToolTip(f"推荐度 {pause.score:.2f}"
                            + (f"，最近拍点 {pause.nearest_beat:.3f}s"
                               if pause.nearest_beat >= 0 else "，这首歌没有可用拍网格"))
            self.nav.addItem(item)
        if self.nav.count() == 0:
            self.nav.addItem(QListWidgetItem("（没有够格的停顿，把门槛调低看看）"))

    def _anchor_of(self, pause) -> float:
        """按当前「跳到哪里」的选择给出落点：停顿开始 / 中心 / 结束。"""
        key = str(self.anchor.currentData() or "start")
        if key == "middle":
            return pause.middle
        if key == "end":
            return pause.end
        return pause.start

    def _nav_picked(self, row: int) -> None:
        if row < 0 or self._activity is None:
            return
        item = self.nav.item(row)
        index = item.data(Qt.UserRole) if item is not None else None
        if index is None:
            return
        for pause in self._activity.pauses:
            if pause.index == int(index):
                self._seek(self._anchor_of(pause))
                return

    def _jump(self, direction: int) -> None:
        if self._activity is None:
            return
        pause = (self._activity.next_pause(self._at) if direction > 0
                 else self._activity.previous_pause(self._at))
        if pause is None:
            self.status.setText("没有下一处停顿了" if direction > 0 else "前面没有停顿了")
            return
        self._highlight(pause)
        self._seek(self._anchor_of(pause))

    def _highlight(self, pause) -> None:
        """当前停顿在左边列表里高亮 —— 用户得知道自己现在站在哪一个上。"""
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            if item is not None and item.data(Qt.UserRole) == pause.index:
                self.nav.blockSignals(True)
                self.nav.setCurrentRow(row)
                self.nav.blockSignals(False)
                return

    def _seek(self, moment: float) -> None:
        """跳到某一刻：播放器和界面一起动（播放中就接着播）。"""
        self._moved_to(moment)
        if self._duration > 0:
            self.player.setPosition(int(max(0.0, moment) * 1000))
        self.timeline.ensure_visible(moment)

    def _moved_to(self, moment: float) -> None:
        self._at = max(0.0, min(float(moment), self._duration or float(moment)))
        self.timeline.set_cursor_time(self._at)
        self.seek_requested.emit(round(self._at, 3))
        self._refresh_status()

    def seek_to(self, moment: float) -> None:
        """外面（视频位置那条带子、右键菜单）要求跳到某一秒：**播放器也真的跟着跳**。

        主音频是唯一时间权威，所以定位只能从这一个入口进，别处不许自己算时间。
        """
        self._moved_to(float(moment))
        if self._duration > 0:
            self.player.setPosition(int(max(0.0, float(moment)) * 1000))
        self.timeline.ensure_visible(float(moment))


    def _refresh_status(self) -> None:
        if self._duration <= 0:
            self.status.setText("当前：—")
            return
        parts = [f"当前 {_clock(self._at)}"]
        if self._template is not None:
            span = self._template.span_at(self._at)
            if span is not None:
                parts.append(f"{span.name}（{span.start:.3f}→{span.end:.3f}，"
                             f"{span.duration:.3f}s）")
        else:
            parts.append("还没分段")
        if self._activity is not None:
            parts.append("人声中" if self._activity.speaking_at(self._at) else "停顿中")
            nxt = self._activity.next_pause(self._at)
            parts.append(f"下一处停顿 {_clock(nxt.start)}（还有 {nxt.start - self._at:.3f}s）"
                         if nxt is not None else "后面没有停顿了")
        self.status.setText("　│　".join(parts))

    # ------------------------------------------------------------ 播放 / 视图
    def play(self) -> None:
        """从当前位置继续播。整曲播放不设自动停。"""
        if self._duration <= 0:
            return
        self._stop_at = None
        self.player.setPosition(int(self._at * 1000))
        self.player.play()

    def toggle_play(self) -> None:
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.play()

    def stop(self) -> None:
        self._stop_at = None
        self.player.stop()
        self.timeline.set_playhead(None)
        self._refresh_status()

    def play_current_segment(self) -> None:
        """▶ 播放当前段：从这一段的起点播到终点自动停。"""
        if self._template is None:
            self.status.setText("还没分段，先「等间隔起步」或者「照停顿分」")
            return
        span = self._template.span_at(self._at)
        if span is None:
            return
        self._stop_at = span.end
        self.player.setPosition(int(span.start * 1000))
        self.player.play()
        self.status.setText(f"正在播 {span.name}（{span.start:.3f}→{span.end:.3f}）")

    def _position_changed(self, milliseconds: int) -> None:
        """播放器每 50ms 回调一次：播放头、时钟、当前段、人声状态一起更新。"""
        moment = max(0.0, float(milliseconds) / 1000.0)
        if self._stop_at is not None and moment >= self._stop_at:
            self.player.pause()
            self._stop_at = None
        self._at = moment
        self.timeline.set_playhead(moment)
        self.timeline.set_cursor_time(moment)
        if self.follow.isChecked():
            self.timeline.ensure_visible(moment)
        self._refresh_clock()
        self._refresh_status()

    def _refresh_clock(self) -> None:
        self.clock.setText(f"{_clock(self._at)} / {_clock(self._duration)}")

    def _zoom_preset(self, index: int) -> None:
        span = float(self.zooms.itemData(index) or 0.0)
        self.timeline.set_view(max(0.0, self._at - span / 2.0), span)

    def _view_changed(self, start: float, span: float) -> None:
        """时间轴的可见窗口变了 → 把滚动条调成一样的范围（两边不许各说各话）。"""
        usable = max(0.0, self._duration - span) if span > 0 else 0.0
        self.scroll.blockSignals(True)
        self.scroll.setEnabled(span > 0 and usable > 0)
        self.scroll.setRange(0, int(usable * 1000))
        self.scroll.setPageStep(int(max(1.0, span) * 1000))
        self.scroll.setValue(int(start * 1000))
        self.scroll.blockSignals(False)

    # ------------------------------------------------------------ 段落编辑

    def _editor(self):
        from ...dance import segment_template as editor  # noqa: PLC0415

        return editor

    def _adopt(self, template, *, remember: bool = True) -> None:
        """换上一份新模板。`remember` 决定要不要把旧的压进撤销栈。"""
        if remember and self._template is not None:
            self._undo.append(self._template)
            self._redo.clear()          # 新动作一出，之前撤销掉的那条线就断了
            self.btn_undo.setEnabled(True)
        self._template = template
        self.timeline.set_template(template)
        self._refresh_status()
        self.template_changed.emit(template)

    def _complain(self, exc: Exception) -> None:
        """把"为什么不行"直接说出来。这个页面从不静默失败。"""
        self.status.setText(f"做不了：{exc}")
        logger.info("段落编辑被拒：%s", exc)

    def _make_uniform(self) -> None:
        if self._duration <= 0:
            QMessageBox.information(self, "先分析主音频", "还不知道这首歌多长，先点「分析主音频」。")
            return
        editor = self._editor()
        try:
            self._adopt(editor.uniform(self._duration, float(self.step.value()),
                                       target_song_id=self._song_id))
        except editor.SegmentError as exc:
            self._complain(exc)

    def _make_by_pause(self) -> None:
        """照人声停顿生成一份起步分段 —— 用户点了才做，系统自己绝不这么干。"""
        if self._activity is None:
            QMessageBox.information(self, "先分析主音频", "还没有人声停顿可用。")
            return
        editor = self._editor()
        try:
            self._adopt(editor.from_pauses(self._duration, self._activity.pauses,
                                           min_score=float(self.min_score.value()),
                                           target_song_id=self._song_id))
        except editor.SegmentError as exc:
            self._complain(exc)

    def _split_here(self) -> None:
        """✂ 在当前位置切一刀。先吸附到最近的参考点，免得切出 7.043 这种边界。"""
        if self._template is None:
            self._make_uniform()
            return
        editor = self._editor()
        moment = editor.snap(self._at, self.timeline.snap_points)
        try:
            self._adopt(editor.split_at(self._template, moment))
            self.status.setText(f"在 {moment:.3f}s 切了一刀"
                                f"（现在 {len(self._template.spans)} 段）")
        except editor.SegmentError as exc:
            self._complain(exc)

    def unsplit_here(self, moment: float | None = None) -> bool:
        """× 取消切分：删掉离 `moment`（默认当前播放位置）**最近的那条分段线**。

        它删的是边界，不是素材 —— `S2 | S3` 的线没了就变成一段 `S2+S3`。
        走的还是 `segment_template.merge_at`，所以"合不合法"仍然只有一处判定。
        整首歌只剩一段时没有内部边界可删，说清楚就好，不硬来。
        """
        if self._template is None:
            return False
        editor = self._editor()
        spans = list(self._template.spans)
        if len(spans) < 2:
            self.status.setText("只有一段，没有分段线可以取消")
            return False
        at = self._at if moment is None else float(moment)
        # 内部边界 = 第 1..n-1 段的起点；挑离目标最近的那条
        index = min(range(1, len(spans)), key=lambda i: abs(spans[i].start - at))
        line = spans[index].start
        try:
            self._adopt(editor.merge_at(self._template, index))
        except editor.SegmentError as exc:
            self._complain(exc)
            return False
        self.status.setText(f"取消了 {line:.3f}s 那条分段线"
                            f"（现在 {len(self._template.spans)} 段）")
        return True

    def _merge_here(self) -> None:

        if self._template is None:
            return
        editor = self._editor()
        span = self._template.span_at(self._at)
        if span is None:
            return
        try:
            self._adopt(editor.merge_at(self._template, span.index))
            self.status.setText(f"{span.name} 已并进前一段"
                                f"（现在 {len(self._template.spans)} 段）")
        except editor.SegmentError as exc:
            self._complain(exc)

    def _drag_preview(self, index: int, moment: float) -> None:
        """拖动过程中的实时预览：吸附到参考点，拖不过去就**不动**（画面上就是拖不过去）。"""
        if self._template is None:
            return
        editor = self._editor()
        target = editor.snap(moment, self.timeline.snap_points)
        try:
            moved = editor.move_boundary(self._template, index, target)
        except editor.SegmentError:
            return
        self._template = moved                    # 预览不进撤销栈，松手才算一步
        self.timeline.set_template(moved)

    def _drag_commit(self, index: int, moment: float) -> None:
        if self._template is None:
            return
        editor = self._editor()
        target = editor.snap(moment, self.timeline.snap_points)
        try:
            moved = editor.move_boundary(self._template, index, target)
        except editor.SegmentError as exc:
            self._complain(exc)
            self.timeline.set_template(self._template)
            return
        self._adopt(moved)
        self.status.setText(f"第 {index + 1} 个分割点定在 {target:.3f}s")

    def _undo_once(self) -> None:
        if not self._undo:
            self.btn_undo.setEnabled(False)
            return
        if self._template is not None:
            self._redo.append(self._template)
        self._adopt(self._undo.pop(), remember=False)
        self.btn_undo.setEnabled(bool(self._undo))
        self.status.setText("撤销了上一步")

    def undo(self) -> None:
        """Ctrl+Z（主窗口转过来的）。"""
        self._undo_once()

    def redo(self) -> None:
        """Ctrl+Y：把刚撤销掉的那一步再做回来。"""
        if not self._redo:
            self.status.setText("没有可以重做的分段改动")
            return
        if self._template is not None:
            self._undo.append(self._template)
            self.btn_undo.setEnabled(True)
        self._adopt(self._redo.pop(), remember=False)
        self.status.setText("重做了一步")

    # ---------------------------------------------------------------- 存/读
    def save_template(self) -> int:
        """把当前分段存进库。没有歌 id（还没分析）就说清楚，不偷偷不存。"""
        if self._template is None:
            QMessageBox.information(self, "还没有分段", "先「等间隔起步」或者「照停顿分」。")
            return 0
        if self.db is None or self._song_id <= 0:
            QMessageBox.warning(self, "存不了", "这首歌还没登记进库，先点「分析主音频」。")
            return 0
        from ...dance import material_repository as repo  # noqa: PLC0415

        template_id = repo.save_segment_template(self.db, self._template)
        self.status.setText(f"已保存：{len(self._template.spans)} 段（模板 #{template_id}）"
                            f"，所有源视频都按这份分段切")
        return template_id

    def _load_saved_template(self) -> None:
        """分析完先看看库里有没有存过的分段：有就用它，没有就等用户自己起步。"""
        if self.db is None or self._song_id <= 0:
            return
        from ...dance import material_repository as repo  # noqa: PLC0415

        saved = repo.active_segment_template(self.db, self._song_id)
        if saved is None:
            self.timeline.set_template(None)
            return
        self._undo.clear()
        self.btn_undo.setEnabled(False)
        self._adopt(saved, remember=False)
        self.status.setText(f"读到库里存过的分段：{len(saved.spans)} 段（来源 {saved.source}）")

    # ---------------------------------------------------------------- 杂务
    @property
    def template(self):
        """当前分段（可能是 None）。别的页面只读它，不许直接改。"""
        return self._template

    def set_playhead(self, moment: float | None) -> None:
        self.timeline.set_playhead(moment)

    @property
    def at(self) -> float:
        """当前播放位置（秒）。别的面板要跟着它走，就读这一个。"""
        return float(self._at)


    def state(self) -> dict[str, Any]:
        return {"path": self.path.text().strip(), "step": float(self.step.value()),
                "min_score": float(self.min_score.value()),
                "only_strong": bool(self.only_strong.isChecked()),
                "volume": int(self.volume.value()),
                "follow": bool(self.follow.isChecked()),
                "zoom": int(self.zooms.currentIndex()),
                "speed": int(self.speeds.currentIndex()),
                "anchor": int(self.anchor.currentIndex()),
                "body": self._body.sizes()}

    def restore(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self.path.setText(str(data.get("path") or ""))
        if data.get("step"):
            self.step.setValue(float(data["step"]))
        if data.get("min_score") is not None:
            self.min_score.setValue(float(data["min_score"]))
        self.only_strong.setChecked(bool(data.get("only_strong")))
        if data.get("volume") is not None:
            self.volume.setValue(int(data["volume"]))
        if data.get("follow") is not None:
            self.follow.setChecked(bool(data["follow"]))
        for key, widget in (("zoom", self.zooms), ("anchor", self.anchor)):
            index = data.get(key)
            if isinstance(index, int) and 0 <= index < widget.count():
                widget.blockSignals(True)
                widget.setCurrentIndex(index)
                widget.blockSignals(False)
        speed = data.get("speed")
        if isinstance(speed, int) and 0 <= speed < self.speeds.count():
            self.speeds.setCurrentIndex(speed)      # 这个要触发：播放器得真的换挡

        sizes = data.get("body")
        if isinstance(sizes, list) and len(sizes) == 2:
            self._body.setSizes([int(v) for v in sizes])

    def shutdown(self) -> None:
        self.player.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)


__all__ = ["ENVELOPE_BUCKETS", "SPECTRUM_COLUMNS", "SPECTRUM_ROWS",
           "MasterAudioWorker", "MasterAudioPanel"]







