"""「音频对齐 / 卡点测试」页 —— 这个系统的研发验收台。

它回答的是整条链路上最要紧、又最不容易凭数字确认的一件事：

    一个完整舞蹈视频（带它自己的原始音乐）+ 一首目标歌
        ↓  对齐
    offset 是多少、可不可信
        ↓  按固定音乐位置换算
    目标歌第 N 格 ↔ 源视频哪一段
        ↓  播放那一段
    **亲眼看这个动作，是不是真的卡在目标歌这一格上**

调用链（全部是现成的后端，这一层只显示）：

    AlignBenchPanel
      → DanceAlignWorker            后台线程，只算不写
        → material_ingest.align_batch(persist=False)
          → audio_align.find_alignment → alignment_validation.decide_status
      → align_probe.probe_all / probe_one    ← 复用 material_slice 的判定
        → material_slice.map_to_source / plan_slices
      → FramePlayer.seek/play        主界面那个播放器，不另造一个

两条铁律：
  1. **测试不入库。** 点「开始对齐」不会登记素材、不会切片、不会动任何计数。
     要入库必须再点「保存对齐结果」或「切片并加入素材库」。
  2. **不在这一层重算任何东西。** offset 只从后端来，`source = target − offset`
     只走 `DanceAlignment.source_time()`，越界与否只信 `material_slice`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...logging_setup import get_logger
from .. import theme
from ..player import FramePlayer
from . import dialogs
from .align_timeline import DualTimeline
from .align_worker import ClipJobWorker, DanceAlignWorker

logger = get_logger("dance.gui.bench")

POSITION_COLUMNS = ("#", "目标区间", "源区间", "长度", "状态", "原因")
BATCH_COLUMNS = ("源视频", "Offset", "置信度", "结论", "覆盖率", "可用格")

#: 结论的中文说法。和对齐面板共用同一份措辞，不许两处写得不一样
STATUS_TEXT = {"ok": "OK 可用", "low_confidence": "LOW 置信偏低",
               "disagree": "DISAGREE 两路矛盾", "rejected": "REJECTED 不可用",
               "manual": "MANUAL 人工修正"}
#: 播放到区间末尾的容差（秒）。播放器按帧推进，不给容差会永远差最后一帧
PLAY_EPS = 0.04
#: 按钮/输入框的最小高度。Qt 默认那个二十几像素在 1080p 上根本点不准 ——
#: 这一页是要一手改参数一手点播放来回试的，控件小一点就废了
BUTTON_HEIGHT = 34
FIELD_HEIGHT = 30


def _big(button, height: int = BUTTON_HEIGHT, *, bold: bool = False):
    """给按钮一个够点的高度；`bold` 留给每一栏里那个主按钮。"""
    button.setMinimumHeight(int(height))
    if bold:
        button.setStyleSheet("font-weight:600;")
    return button


def _tall(field, height: int = FIELD_HEIGHT):
    field.setMinimumHeight(int(height))
    return field


def _short_reason(reason: str) -> str:
    """表格里那一列只放一句人话，完整原因走 tooltip。

    后端给的理由是完整句子（"位置 #0（目标 0.000→2.000）映射到源 -3.274s，
    早于源视频开头 —— 不做 clamp，这一段没有对应素材"），
    直接塞进列宽一百来像素的表格里只会看到"位置 #0（…"，等于什么都没说。
    """
    text = str(reason or "")
    if not text:
        return ""
    if "早于源视频开头" in text:
        return "源里还没开始"
    if "超过源时长" in text:
        return "源已经结束"
    if "rejected" in text:
        return "对齐被拒，整条不可用"
    if "区间非法" in text:
        return "区间非法"
    return text.split("——")[0].strip()[:24]




class AlignBenchPanel(QWidget):
    """左边是操作与结论，右边是播放器与双时间轴。"""

    #: 用户确认过了，要把这条源正式切片入库 → 交给主窗口的 `DanceMontageWorker`
    ingest_requested = pyqtSignal(dict)
    #: 库里有东西变了（保存了对齐 / 入了素材），主窗口该刷一遍
    changed = pyqtSignal()

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.db: Any = None
        self._song_id = 0
        self._payload: dict[str, Any] = {}
        self._auto: Any = None           # 算法算出来的那份 DanceAlignment（原值）
        self._alignment: Any = None       # 当前在用的那份（可能是人工覆盖过的）
        self._rows: list[Any] = []
        self._batch: list[dict[str, Any]] = []
        self.worker: DanceAlignWorker | None = None
        self.side: ClipJobWorker | None = None
        self._loop = False
        self._play_from = 0.0
        self._play_to = 0.0
        self._audio_wav: str | None = None
        self._headline = "还没对齐"

        # 整页放进一个滚动区：窗口小到 1000×640 时，三栏工作台不可能全塞进去 ——
        # 那种情况下**宁可让用户滚**，也不许把卡点表和播放器一起压成两行高。
        # 但 ⑧ 入库那三个按钮**钉在滚动区外面**：它们是这一页的出口，
        # 永远得看得见、点得到，不能因为内容长就被滚到屏幕外。
        page = QWidget(self)
        inner = QVBoxLayout(page)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(8)
        inner.addWidget(self._build_header())
        stack = QSplitter(Qt.Vertical, page)      # 上下两块的高度也让用户自己拖
        stack.addWidget(self._build_body())
        stack.addWidget(self._build_bottom())
        stack.setStretchFactor(0, 4)
        stack.setStretchFactor(1, 1)
        stack.setSizes([520, 210])
        self._stack = stack
        inner.addWidget(stack, 1)
        page.setMinimumHeight(640)                # 低于这个高度就出滚动条

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(scroll, 1)
        outer.addWidget(self._build_save())

    # ================================================================ 建界面
    #
    # 布局的取舍，按"看不过来"这条抱怨定的：
    #
    #   顶部只留**一条工具带**（选素材 + 开跑 + 进度），不给它 groupbox 标题、
    #   不放任何"看一眼就好"的东西 —— 上边地方最贵，只配给每次都要动的控件。
    #   中间三栏并列：左「结果」中「卡点」右「确认」，视线从左到右就是干活顺序。
    #   偶尔才看的东西（批量结果、时间映射、手动 Offset、日志）全进底部标签页。
    #   入库那三个按钮钉在最底，永远看得见。
    #
    # 刻意**不给标题编号**：三栏是并列的，编号只会让人问"我该先看第几个"。
    # 标题用动词，位置本身说明顺序。
    # 按钮一律给足高度（`_big`），38/44 像素在 1080p 上才点得准。
    def _build_header(self) -> QWidget:
        holder = QFrame(self)
        holder.setFrameShape(QFrame.StyledPanel)
        grid = QGridLayout(holder)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(4, 3)

        self.source = _tall(QLineEdit(holder))
        self.source.setPlaceholderText("源舞蹈视频（完整视频 + 它自己的原始音乐）")
        btn_source = _big(QPushButton("选视频…", holder), 30)
        grid.addWidget(QLabel("源视频", holder), 0, 0)
        grid.addWidget(self.source, 0, 1)
        grid.addWidget(btn_source, 0, 2)

        self.target = _tall(QLineEdit(holder))
        self.target.setPlaceholderText("目标歌文件，或库里的歌 id")
        btn_target = _big(QPushButton("选音频…", holder), 30)
        grid.addWidget(QLabel("目标歌", holder), 0, 3)
        grid.addWidget(self.target, 0, 4)
        grid.addWidget(btn_target, 0, 5)

        # 批量清单不占版面：列表本体藏起来，只留一个计数和两个按钮。
        # 结果反正会横铺在下面「批量结果」那张表里，没必要在上边再摆一遍
        self.more = QListWidget(holder)
        self.more.setVisible(False)
        self.more_hint = QLabel("批量：0 个", holder)
        self.more_hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        btn_more = _big(QPushButton("批量选视频…", holder), 30)
        btn_clear = _big(QPushButton("清空批量", holder), 30)

        # 这一页只有两个"主"按钮（开始对齐、播放卡点），它们比别的高一档，
        # 眼睛一扫就知道该点哪个
        self.btn_start = _big(QPushButton("开始音频对齐", holder), 44, bold=True)
        self.btn_stop = _big(QPushButton("停止", holder), 40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMaximumWidth(90)
        self.btn_reset = _big(QPushButton("重置这一页", holder), 40)
        self.btn_reset.setMaximumWidth(120)
        self.btn_reset.setToolTip("只清这个页面上的东西（路径、结果、卡点、预览）。\n"
                                 "素材库里的正式素材一条都不会动。")
        self.force = QCheckBox("忽略缓存重算", holder)
        self.bar = QProgressBar(holder)
        self.bar.setRange(0, 100)
        self.bar.setMinimumHeight(24)
        self.bar.setFormat("%v / %m")
        self.bar.setToolTip("只算不写：对齐这一步不登记素材、不切片、不动任何计数")

        actions = QHBoxLayout()
        actions.setSpacing(6)
        for widget in (btn_more, btn_clear, self.more_hint):
            actions.addWidget(widget)
        actions.addSpacing(12)
        actions.addWidget(self.btn_start, 2)
        actions.addWidget(self.btn_stop)
        actions.addWidget(self.btn_reset)
        actions.addWidget(self.force)
        actions.addWidget(self.bar, 2)
        grid.addLayout(actions, 1, 0, 1, 6)

        btn_source.clicked.connect(self._pick_source)
        btn_target.clicked.connect(self._pick_target)
        btn_more.clicked.connect(self._pick_more)
        btn_clear.clicked.connect(self._clear_more)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset)
        return holder

    def _build_body(self) -> QWidget:
        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_result())          # 左：这条到底行不行
        slots = QSplitter(Qt.Vertical, body)          # 中：卡点工具 + 全曲卡点表
        slots.setChildrenCollapsible(False)
        slots.addWidget(self._build_tools())
        slots.addWidget(self._build_positions())
        slots.setStretchFactor(0, 1)
        slots.setStretchFactor(1, 4)
        slots.setSizes([175, 430])
        preview = QSplitter(Qt.Vertical, body)        # 右：看画面 + 看两条音轨
        preview.setChildrenCollapsible(False)
        preview.addWidget(self._build_player())
        preview.addWidget(self._build_timeline())
        preview.setStretchFactor(0, 3)
        preview.setStretchFactor(1, 2)

        body.addWidget(slots)
        body.addWidget(preview)
        body.setSizes([400, 540, 560])
        self._body, self._slots, self._preview = body, slots, preview
        return body


    def _build_player(self) -> QGroupBox:
        box = QGroupBox("确认：播这一格（只播算出来的那一段）", self)
        inner = QVBoxLayout(box)
        inner.setSpacing(8)
        self.player = FramePlayer(box)
        self.player.setMinimumHeight(180)
        # 主界面那句"把视频拖进来"在这一页是错的：这里的视频由上面的输入框决定
        self.player.view.setText("对齐之后在卡点表里点一格，这里就播源视频的那一段")
        inner.addWidget(self.player, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_play = _big(QPushButton("▶ 播放这个卡点", box), 44, bold=True)
        self.btn_halt = _big(QPushButton("■ 停", box), 44)
        self.btn_halt.setMaximumWidth(90)
        self.btn_loop = QCheckBox("⟳ 循环", box)
        self.btn_loop.setMinimumHeight(44)
        self.chk_sound = QCheckBox("带声音", box)
        self.chk_sound.setMinimumHeight(44)
        self.chk_sound.setToolTip(
            "放出来的是**源视频自己的原声**。\n"
            "这正是听对齐的办法：offset 求对了的话，源视频这一段的音乐\n"
            "就应该和目标歌这一格的音乐是同一段 —— 听着对上了就是对上了。\n"
            "（第一次勾选要先把音轨解成 wav，几秒钟，后台做。）")
        row.addWidget(self.btn_play, 1)
        row.addWidget(self.btn_halt)
        row.addWidget(self.btn_loop)
        row.addWidget(self.chk_sound)
        inner.addLayout(row)


        self.play_hint = QLabel("还没有可播的区间", box)
        self.play_hint.setWordWrap(True)
        self.play_hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        inner.addWidget(self.play_hint)

        self.player.positionChanged.connect(self._on_position)
        self.player.audioFailed.connect(self._audio_failed)
        self.btn_play.clicked.connect(self._play_span)
        self.btn_halt.clicked.connect(self._halt)
        self.btn_loop.stateChanged.connect(
            lambda state: setattr(self, "_loop", bool(state)))
        self.chk_sound.stateChanged.connect(self._toggle_sound)
        return box

    def _build_timeline(self) -> QGroupBox:
        box = QGroupBox("两条音轨（点一下跳到那个音乐位置）", self)
        layout = QVBoxLayout(box)
        self.timeline = DualTimeline(box)
        layout.addWidget(self.timeline, 1)
        self.timeline.moved.connect(self._timeline_moved)
        return box

    def _build_bottom(self) -> QWidget:
        """偶尔才看的东西全塞这儿：批量结果、时间映射与手动 Offset、运行日志。

        上边地方最贵，只配给"每次都要动"的控件；这三样都是"要看的时候点一下"。
        """
        tabs = QTabWidget(self)
        tabs.setMaximumHeight(240)

        self.batch = QTableWidget(0, len(BATCH_COLUMNS), tabs)
        self.batch.setMinimumHeight(70)
        self.batch.setHorizontalHeaderLabels(BATCH_COLUMNS)
        self.batch.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch.setSortingEnabled(True)          # 按置信度/覆盖率排序挑素材
        self.batch.setAlternatingRowColors(True)
        self.batch.verticalHeader().setVisible(False)
        self.batch.verticalHeader().setDefaultSectionSize(26)
        self.batch.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.batch.doubleClicked.connect(self._pick_batch_row)
        tabs.addTab(self.batch, "批量结果（点表头排序，双击某行切过去）")

        tabs.addTab(self._build_mapping(), "时间映射 / 手动 Offset")

        self.log = QTextEdit(tabs)
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        tabs.addTab(self.log, "运行日志")
        return tabs

    def _build_mapping(self) -> QWidget:
        """时间映射的说明 + 人工覆盖 offset。都是"偶尔用一次"，所以收在标签页里。"""
        holder = QWidget(self)
        layout = QVBoxLayout(holder)
        layout.setSpacing(6)

        self.mapping = QLabel("时间映射：还没有结果", holder)
        self.mapping.setWordWrap(True)
        self.mapping.setStyleSheet(
            f"color:{theme.ACCENT}; border:1px solid {theme.LINE};"
            f"border-radius:4px; padding:6px;")
        layout.addWidget(self.mapping)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.manual = _tall(QDoubleSpinBox(holder))
        self.manual.setRange(-3600.0, 3600.0)
        self.manual.setDecimals(3)
        self.manual.setSingleStep(0.01)
        self.btn_manual = _big(QPushButton("应用手动 Offset", holder))
        self.btn_auto = _big(QPushButton("回到算法值", holder))
        row.addWidget(QLabel("手动 Offset", holder))
        row.addWidget(self.manual, 1)
        row.addWidget(self.btn_manual)
        row.addWidget(self.btn_auto)
        layout.addLayout(row)

        self.origin = QLabel("来源：—", holder)
        self.origin.setWordWrap(True)
        layout.addWidget(self.origin)
        layout.addStretch(1)

        self.btn_manual.clicked.connect(self._apply_manual)
        self.btn_auto.clicked.connect(self._restore_auto)
        return holder



    def _build_result(self) -> QGroupBox:
        box = QGroupBox("结果：这条到底行不行", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # 批量时结果区显示的是**其中一条**，不写清是哪条会让人以为数字对不上
        self.current = QLabel("还没对齐", box)
        self.current.setWordWrap(True)
        self.current.setStyleSheet("font-size:14px; font-weight:600;")
        layout.addWidget(self.current)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        self.out: dict[str, QLabel] = {}
        fields = (("status", "结论"), ("offset", "Offset"),
                  ("final", "Final 置信度"), ("waveform", "Waveform 置信度"),
                  ("chroma", "Chroma 置信度"), ("windows", "验证窗口"),
                  ("source_duration", "源视频时长"), ("target_duration", "目标歌时长"))
        for index, (key, label) in enumerate(fields):
            name = QLabel(label, box)
            name.setStyleSheet(f"color:{theme.TEXT_DIM};")
            grid.addWidget(name, index, 0)
            value = QLabel("—", box)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True)
            if key in ("status", "offset"):          # 最要紧的两项，字大一号
                value.setStyleSheet("font-size:14px; font-weight:600;")
            grid.addWidget(value, index, 1)
            self.out[key] = value
        layout.addLayout(grid)

        self.diagnosis = QTextEdit(box)
        self.diagnosis.setReadOnly(True)
        self.diagnosis.setMinimumHeight(90)
        self.diagnosis.setPlaceholderText("诊断会写在这里")
        layout.addWidget(self.diagnosis, 1)
        return box


    def _build_tools(self) -> QGroupBox:
        box = QGroupBox("卡点：目标时间 → 源时间", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.at = _tall(QDoubleSpinBox(box))
        self.at.setRange(0.0, 36000.0)
        self.at.setDecimals(3)
        self.at.setValue(10.0)
        self.slice_seconds = _tall(QDoubleSpinBox(box))
        self.slice_seconds.setRange(0.2, 30.0)
        self.slice_seconds.setDecimals(3)
        self.slice_seconds.setSingleStep(0.5)
        self.slice_seconds.setValue(float(self.cfg.dance["slice_duration"]))
        grid.addWidget(QLabel("目标音乐时间(秒)", box), 0, 0)
        grid.addWidget(self.at, 0, 1)
        grid.addWidget(QLabel("切片长度(秒)", box), 0, 2)
        grid.addWidget(self.slice_seconds, 0, 3)
        layout.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(8)
        btn_map = _big(QPushButton("只看映射", box))
        btn_probe = _big(QPushButton("测试这个卡点", box), bold=True)
        row.addWidget(btn_map)
        row.addWidget(btn_probe, 1)
        layout.addLayout(row)

        self.probe_text = QLabel("目标区间 / 源区间 / 能不能用，会显示在这里", box)
        self.probe_text.setWordWrap(True)
        self.probe_text.setMinimumHeight(78)
        self.probe_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.probe_text.setStyleSheet(
            f"border:1px solid {theme.LINE}; border-radius:4px; padding:6px;")
        layout.addWidget(self.probe_text, 1)


        btn_map.clicked.connect(self._map_moment)
        btn_probe.clicked.connect(self._probe_span)
        return box

    def _build_positions(self) -> QGroupBox:
        box = QGroupBox("全曲固定卡点（点一行 → 播放器跳过去；双击 → 直接播）", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_all = _big(QPushButton("生成全部固定卡点", box), 40, bold=True)
        self.coverage = QLabel("覆盖率：—", box)
        self.coverage.setWordWrap(True)
        self.coverage.setStyleSheet("font-weight:600;")
        row.addWidget(btn_all, 2)
        row.addWidget(self.coverage, 3)
        layout.addLayout(row)

        self.positions = QTableWidget(0, len(POSITION_COLUMNS), box)
        self.positions.setHorizontalHeaderLabels(POSITION_COLUMNS)
        self.positions.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions.setSelectionMode(QAbstractItemView.SingleSelection)
        self.positions.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.positions.setAlternatingRowColors(True)
        self.positions.setToolTip("点某一行 → 播放器跳到对应的源区间；双击 → 直接播放。\n"
                                 "「原因」那一列鼠标停一下能看到完整那句话。")
        self.positions.verticalHeader().setVisible(False)
        self.positions.verticalHeader().setDefaultSectionSize(28)   # 行高够点
        self.positions.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        layout.addWidget(self.positions, 1)

        btn_all.clicked.connect(self.generate_positions)
        self.positions.clicked.connect(self._pick_position)
        self.positions.doubleClicked.connect(lambda _i: self._play_span())
        return box


    def _build_save(self) -> QGroupBox:
        box = QGroupBox("确认之后才入库（在这之前一条素材都不会产生）", self)
        row = QHBoxLayout(box)
        row.setSpacing(8)
        self.btn_save = _big(QPushButton("保存对齐结果", box), 40)
        self.btn_save.setToolTip("只把这条 offset 记进 dance_audio_alignments，不切片。")
        self.btn_ingest = _big(QPushButton("切片并加入素材库", box), 40, bold=True)
        self.btn_ingest.setToolTip("按当前切片长度把这条源切成素材、登记进素材库。\n"
                                  "走的是混剪那条正式流水线，和命令行 `dance-montage slice` 同一套。")
        self.btn_export = _big(QPushButton("导出当前测试片段…", box), 40)
        self.btn_export.setToolTip("把当前源区间单独渲一个文件出来，方便拿别的播放器细看。\n"
                                  "播放预览本身不会重新编码任何东西。")
        for button in (self.btn_save, self.btn_ingest, self.btn_export):
            row.addWidget(button, 1)
        self.btn_save.clicked.connect(self._save_alignment)
        self.btn_ingest.clicked.connect(self._ingest)
        self.btn_export.clicked.connect(self._export)
        return box


    # ================================================================ 数据入口
    def refresh(self, db: Any, song_id: int) -> None:
        """主窗口切歌时叫一下。目标歌输入框空着就替用户填上当前那首的 id。"""
        self.db = db
        self._song_id = int(song_id or 0)
        if self._song_id and not self.target.text().strip():
            self.target.setText(str(self._song_id))

    def say(self, line: str) -> None:
        self.log.append(str(line))

    # ------------------------------------------------------------ 记住上次的设置
    def state(self) -> dict[str, Any]:
        """值得记住的：两个路径、测试参数、三个勾选框、三条分割条的比例。

        **不记**对齐结果和卡点表 —— 那是算出来的东西，重开界面重算一遍才可信。
        """
        return {
            "source": self.source.text().strip(),
            "target": self.target.text().strip(),
            "at": float(self.at.value()),
            "slice": float(self.slice_seconds.value()),
            "loop": bool(self.btn_loop.isChecked()),
            "sound": bool(self.chk_sound.isChecked()),
            "force": bool(self.force.isChecked()),
            "body": list(self._body.sizes()),
            "slots": list(self._slots.sizes()),
            "preview": list(self._preview.sizes()),
            "stack": list(self._stack.sizes()),
        }

    def restore(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        for widget, key in ((self.source, "source"), (self.target, "target")):
            if isinstance(data.get(key), str) and data[key]:
                widget.setText(data[key])
        for widget, key in ((self.at, "at"), (self.slice_seconds, "slice")):
            if isinstance(data.get(key), (int, float)):
                widget.setValue(float(data[key]))
        for widget, key in ((self.btn_loop, "loop"), (self.force, "force")):
            if isinstance(data.get(key), bool):
                widget.setChecked(data[key])
        if isinstance(data.get("sound"), bool):
            # 勾选状态直接写进去、**不触发**槽：这会儿还没有源视频，
            # 触发只会立刻自己取消掉。真正解音轨是在 `_load_player` 里按这个勾选做的
            self.chk_sound.blockSignals(True)
            self.chk_sound.setChecked(data["sound"])
            self.chk_sound.blockSignals(False)
        for splitter, key in ((self._body, "body"), (self._slots, "slots"),
                              (self._preview, "preview"), (self._stack, "stack")):
            sizes = data.get(key)
            if (isinstance(sizes, list) and len(sizes) == splitter.count()
                    and all(isinstance(v, int) and v >= 0 for v in sizes)
                    and sum(sizes) > 0):
                splitter.setSizes(sizes)


    # ---------------------------------------------------------------- 选文件
    def _pick_source(self) -> None:
        path = dialogs.open_file(self, "选源舞蹈视频", self.cfg.dance_path("source_dir"),
                                 dialogs.VIDEO_FILTER, "dance.source")
        if path:
            self.source.setText(path)

    def _pick_target(self) -> None:
        path = dialogs.open_file(self, "选目标歌", self.cfg.dance_path("song_dir"),
                                 dialogs.AUDIO_FILTER, "dance.song")
        if path:
            self.target.setText(path)

    def _pick_more(self) -> None:
        for path in dialogs.open_files(self, "批量选源舞蹈视频",
                                       self.cfg.dance_path("source_dir"),
                                       dialogs.VIDEO_FILTER, "dance.source"):
            self.more.addItem(path)
        self._show_more_count()

    def _clear_more(self) -> None:
        self.more.clear()
        self._show_more_count()

    def _show_more_count(self) -> None:
        count = int(self.more.count())
        self.more_hint.setText(f"批量：{count} 个")
        self.more_hint.setToolTip("\n".join(self.more.item(i).text()
                                            for i in range(count))
                                  or "批量模式：目标歌固定，一次试多个舞蹈视频。\n"
                                     "留空就是单视频模式。")



    # ================================================================ 跑对齐
    def sources(self) -> list[str]:
        """要试的源视频。第一个永远是单视频框里那个（它才是主角）。"""
        out = [self.source.text().strip()] if self.source.text().strip() else []
        out.extend(self.more.item(i).text() for i in range(self.more.count()))
        seen: list[str] = []
        for path in out:
            if path not in seen:
                seen.append(path)
        return seen

    def job(self) -> dict[str, Any]:
        paths = self.sources()
        return {"song": self.target.text().strip(), "sources": paths,
                "workers": int(self.cfg.dance["align_workers"]),
                "force": self.force.isChecked(),
                "envelope": len(paths) == 1}

    def start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "还在跑", "上一次对齐还没结束。")
            return
        job = self.job()
        if not job["song"]:
            QMessageBox.warning(self, "还没选目标歌", "先选一首目标歌（文件或库里的 id）。")
            return
        if not job["sources"]:
            QMessageBox.warning(self, "还没选源视频", "先选一个源舞蹈视频。")
            return
        missing = [p for p in job["sources"] if not Path(p).is_file()]
        if missing:
            QMessageBox.warning(self, "文件不在盘上",
                                "这些路径找不到文件：\n" + "\n".join(missing[:6]))
            return

        self.log.clear()
        self.say(f"[开始] 目标歌 {job['song']}｜源 {len(job['sources'])} 个"
                 + ("｜忽略缓存" if job["force"] else ""))
        self.say("[提示] 这一步只算不写：不会登记素材、不会切片、不会改任何计数")
        self.worker = DanceAlignWorker(self.cfg, job, self)
        self.worker.log.connect(self.say)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._finished)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def shutdown(self) -> None:
        """主窗口关闭时叫一下：把后台线程收干净，别留着孤儿线程。"""
        for worker in (self.worker, self.side):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "stop"):
                    worker.stop()
                worker.wait(8000)
        self.player.close_video()

    def _on_progress(self, current: int, total: int, note: str) -> None:
        self.bar.setRange(0, max(1, int(total)))
        self.bar.setValue(int(current))
        if note:
            self.bar.setFormat(f"{note}  %v / %m")

    def _finished(self, ok: bool, message: str, payload: object) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        data = payload if isinstance(payload, dict) else {}
        self._payload = data
        self._batch = list(data.get("results") or [])
        self.say(f"[结束] {message}")
        self._fill_batch()

        best = self._best(self._batch)
        if best is None:
            self._show_failure(self._batch)
            return
        self._adopt(best)
        if not ok:
            QMessageBox.warning(self, "对齐没有全部成功", message)

    @staticmethod
    def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """批量之后先给用户看**最靠得住的那一条**，不是列表里碰巧排第一的那条。

        排序口径和批量表默认排序一致（结论好的在前、置信度高的在前），
        否则会出现"结果区显示一条 rejected、下面表里却全是 OK"这种自相矛盾的画面。
        """
        rank = {"ok": 0, "manual": 1, "low_confidence": 2, "disagree": 3, "rejected": 4}
        good = [r for r in rows if r.get("alignment") is not None]
        if not good:
            return None
        return min(good, key=lambda r: (rank.get(r["alignment"].status, 9),
                                        -float(r["alignment"].confidence)))


    def _show_failure(self, rows: list[dict[str, Any]]) -> None:
        """一条都没算出来。**必须说清为什么**，不能只在日志里留一行。"""
        self._auto = self._alignment = None
        for label in self.out.values():
            label.setText("—")
        why = "；".join(str(r.get("error") or "未知原因") for r in rows[:3]) or "未知原因"
        self._headline = "一条都没算出来"
        self.current.setText(self._headline)
        self.out["status"].setText("失败")
        self.mapping.setText("时间映射：算不出来")
        self.diagnosis.setPlainText(f"✗ 对齐失败\n\n原因：{why}\n\n"
                                    "常见情况：文件不是媒体文件、没有音轨、"
                                    "被别的程序占用、或者路径打错了。")
        self.timeline.clear()
        self.positions.setRowCount(0)
        self.coverage.setText("覆盖率：—")
        QMessageBox.warning(self, "对齐失败", why)

    def _adopt(self, row: dict[str, Any]) -> None:
        """把某一条结果设为"当前正在看的那条"：结果区、时间轴、播放器、卡点表全跟着换。"""
        alignment = row.get("alignment")
        if alignment is None:
            self._show_failure([row])
            return
        self.source.setText(str(row.get("path") or self.source.text()))
        name = str(row.get("name") or Path(str(row.get("path") or "")).name or "（未知）")
        if len(self._batch) > 1:
            place = next((i + 1 for i, r in enumerate(self._batch)
                          if r is row or r.get("path") == row.get("path")), 0)
            self._headline = (f"{name}（{len(self._batch)} 条里的第 {place} 条，"
                              "双击下面「批量结果」里任意一行可以换）")
        else:
            self._headline = name
        self._auto = alignment
        self._alignment = alignment
        self.manual.setValue(float(alignment.offset))
        self._render_result()
        self._load_player(Path(str(row.get("path") or "")))
        self.generate_positions()


    # ================================================================ 显示结果
    def _render_result(self) -> None:
        align = self._alignment
        if align is None:
            return
        source_duration = self._source_duration()
        status_text = STATUS_TEXT.get(align.status, align.status)
        # 抬头一行把最要紧的四件事说完：哪条源、能不能用、多可信、错开多少
        self.current.setText(f"{self._headline}　·　{status_text}"
                             f"　·　置信 {align.confidence:.3f}"
                             f"　·　offset {align.offset:+.3f}s")
        self.out["status"].setText(status_text)
        self.out["offset"].setText(f"{align.offset:+.3f} s")
        self.out["waveform"].setText(
            "—" if align.waveform_confidence is None
            else f"{float(align.waveform_confidence):.3f}")
        self.out["chroma"].setText(
            "—" if align.chroma_confidence is None
            else f"{float(align.chroma_confidence):.3f}"
                 f"（offset {float(align.chroma_offset or 0.0):+.3f}s）")
        self.out["final"].setText(f"{align.confidence:.3f}")
        self.out["windows"].setText(f"{align.window_count} 个"
                                    f"｜一致性 {align.agreement:.3f}"
                                    f"｜最大偏差 {align.max_deviation:.3f}s")
        self.out["source_duration"].setText(f"{source_duration:.2f} s")
        self.out["target_duration"].setText(f"{self._song_duration():.2f} s")
        self.origin.setText(f"来源：{'MANUAL（人工）' if align.manual else 'AUTO（算法）'}"
                            + (f"｜算法原值 {float(align.original_offset or 0.0):+.3f}s"
                               if align.manual else ""))
        self._show_mapping(float(self.at.value()))
        self.timeline.set_alignment(align.offset, self._song_duration(), source_duration)
        self.timeline.set_envelopes(self._payload.get("target_envelope") or [],
                                    self._payload.get("source_envelope") or [])

    def _show_mapping(self, moment: float) -> None:
        align = self._alignment
        if align is None:
            return
        source = align.source_time(float(moment))
        self.mapping.setText(f"时间映射　target {moment:.3f}s　↓　source {source:.3f}s"
                             f"　（source = target − offset，offset {align.offset:+.3f}s）")
        self.timeline.set_marker(float(moment))

    def _song_duration(self) -> float:
        align = self._alignment
        return float(self._payload.get("song_duration")
                     or (align.target_duration if align is not None else 0.0))

    def _source_duration(self) -> float:
        """源时长优先信**解码出来的音轨长度**，其次信对齐结果里带的那个。

        为什么不信容器时长：越界判定用的就是这个数，宽了会放过一段没有画面的区间。
        """
        align = self._alignment
        return float(self._payload.get("source_duration")
                     or (align.source_duration if align is not None else 0.0))

    # ================================================================ 固定卡点
    def _template_positions(self):
        """这首歌有用户拍板的段落模板就用它，没有才按切片长度等间隔。

        必须和 `material_ingest.slice_and_register` 用同一份位置 —— 否则测试台说
        "这一格能用"，正式切片却按另一把尺子来，那这个页面就白做了。
        """
        song_id = int(self._payload.get("song_id") or 0)
        if self.db is None or song_id <= 0:
            return None
        from ...dance import material_repository as repo  # noqa: PLC0415
        from ...dance import segment_template as editor  # noqa: PLC0415

        try:
            template = repo.active_segment_template(self.db, song_id)
        except editor.SegmentError:
            return None
        return editor.positions_of(template) if template is not None else None

    def generate_positions(self) -> None:
        """按当前分段（模板优先，否则等间隔）铺满整首歌，逐格标可用/越界，并算覆盖率。"""
        from ...dance import align_probe  # noqa: PLC0415

        align = self._alignment
        if align is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        song = self._song_duration()
        if song <= 0:
            QMessageBox.warning(self, "不知道目标歌多长",
                                "目标歌时长读不出来，没法算固定位置。")
            return
        self._rows = align_probe.probe_all(
            align, song_duration=song, source_duration=self._source_duration(),
            slice_duration=float(self.slice_seconds.value()),
            positions=self._template_positions())
        self._fill_positions()
        usable, total, ratio = align_probe.coverage_of(self._rows)
        text = (f"可用 {usable} / {total} 格｜不可用 {total - usable}"
                f"｜覆盖率 {ratio * 100:.2f}%")
        if total <= 1:
            # 切片长度和目标歌时长不搭时会出现这种"只有一格"的怪结果，
            # 用户看到 0/1 完全不知道发生了什么，得直说
            text += (f"　←　{float(self.slice_seconds.value()):g} 秒一格、"
                     f"歌只有 {song:.2f} 秒，整首歌只切出 {total} 格。"
                     "把切片长度调小（常用 2 秒）再生成一次。")
        self.coverage.setText(text)
        self.diagnosis.setPlainText("\n".join(align_probe.diagnose(align, self._rows)))

    def _fill_positions(self) -> None:
        self.positions.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            cells = (
                f"#{row.index}",
                f"{row.target_start:.3f} → {row.target_end:.3f}",
                f"{row.source_start:.3f} → {row.source_end:.3f}",
                f"{row.duration:.3f}",
                row.status_text,
                _short_reason(row.reason),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not row.ok and column == 4:
                    item.setForeground(Qt.red)
                if row.reason:
                    # 完整那句话放 tooltip：表格再宽也装不下"位置 #0（目标 …）映射到源 …"
                    item.setToolTip(row.reason)
                self.positions.setItem(index, column, item)
        self.positions.resizeColumnsToContents()
        self.positions.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)


    def _pick_position(self) -> None:
        """选中一行 → 把它设为当前测试区间，并把播放器停在源起点上。"""
        index = self.positions.currentRow()
        if index < 0 or index >= len(self._rows):
            return
        row = self._rows[index]
        self.at.setValue(float(row.target_start))
        self._show_mapping(float(row.target_start))
        self._set_span(row)

    def _set_span(self, row: Any) -> None:
        """把一格设成"待播区间"。越界的格子不给播 —— 它在源里根本不存在。"""
        self.timeline.set_span(row.target_start, row.target_end)
        self.probe_text.setText(
            f"目标区间 {row.target_start:.3f}s → {row.target_end:.3f}s\n"
            f"源区间　 {row.source_start:.3f}s → {row.source_end:.3f}s\n"
            f"状态　　 {row.status_text}"
            + (f"　（{row.reason}）" if row.reason else ""))
        if not row.ok:
            self._play_from = self._play_to = 0.0
            self.play_hint.setText("这一格越界，源视频里没有对应画面，播不了")
            return
        self._play_from = float(row.source_start)
        self._play_to = float(row.source_end)
        self.play_hint.setText(f"待播：源 {self._play_from:.3f}s → {self._play_to:.3f}s"
                               f"（对应目标歌 {row.target_start:.3f}s 那一格）")
        self.player.seek(self._play_from)
        self.timeline.set_playhead(self._play_from)

    # ================================================================ 映射工具
    def _map_moment(self) -> None:
        if self._alignment is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        self._show_mapping(float(self.at.value()))

    def _probe_span(self) -> None:
        """试一个卡点。判定完全交给 `align_probe.probe_one` → `material_slice`。"""
        from ...dance import align_probe  # noqa: PLC0415

        if self._alignment is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        row = align_probe.probe_one(self._alignment, float(self.at.value()),
                                    float(self.slice_seconds.value()),
                                    self._source_duration())
        self._show_mapping(float(self.at.value()))
        self._set_span(row)

    def _timeline_moved(self, moment: float) -> None:
        self.at.setValue(float(moment))
        self._probe_span()

    # ---------------------------------------------------------- 手动 offset
    def _apply_manual(self) -> None:
        """人工覆盖 offset。走后端的 `manual_override`，所以**理由必填**、原值必留。"""
        from ...dance import alignment_validation as validate  # noqa: PLC0415

        if self._alignment is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        value = float(self.manual.value())
        reason, ok = QInputDialog.getText(
            self, "必须写理由",
            f"把 offset 从 {self._alignment.offset:+.3f}s 改成 {value:+.3f}s，为什么？\n"
            "（会连同算法原值一起存档，禁止静默覆盖）")
        if not ok or not reason.strip():
            QMessageBox.warning(self, "没有改",
                                "没写理由，这次不生效 —— 不允许静默覆盖算法结果。")
            return
        self._alignment = validate.manual_override(self._alignment, value,
                                                   reason.strip(), operator="gui-bench")
        self.say(f"[人工] offset → {value:+.3f}s：{reason.strip()}")
        self._render_result()
        if self._rows:
            self.generate_positions()

    def _restore_auto(self) -> None:
        if self._auto is None:
            return
        self._alignment = self._auto
        self.manual.setValue(float(self._auto.offset))
        self.say(f"[人工] 已回到算法值 {self._auto.offset:+.3f}s")
        self._render_result()
        if self._rows:
            self.generate_positions()

    # ================================================================ 播放
    def _load_player(self, source: Path) -> None:
        if not source.is_file():
            return
        self._audio_wav = None              # 换了源视频，上一条音轨作废
        self.player.set_audio_file(None)
        if not self.player.open(str(source)):
            self.play_hint.setText(f"这个视频解不开：{source.name}")
            return
        self.play_hint.setText(f"已载入 {source.name}｜"
                               f"{self.player.duration():.2f}s，选一格就能播")
        if self.chk_sound.isChecked():
            self._prepare_audio()

    # ------------------------------------------------------------ 带声音预览
    def _toggle_sound(self, state: int) -> None:
        """勾上「带声音」= 放源视频自己的原声。

        这是**听对齐**的办法：offset 求对了的话，源视频这一段的音乐就该和目标歌
        这一格的音乐是同一段。所以不需要同时放两路声音（`winsound` 也只能放一路），
        听源视频的原声就等于在听"这个动作当时踩的是哪一句"。
        """
        if not bool(state):
            self.player.set_audio_enabled(False)
            return
        if self._audio_wav is not None and Path(self._audio_wav).is_file():
            self.player.set_audio_file(self._audio_wav)
            self.player.set_audio_enabled(True)
            return
        self._prepare_audio()

    def _prepare_audio(self) -> None:
        """后台把源视频的音轨解成 wav（`winsound` 只认 PCM wav）。

        放在后台是硬要求：解码走 PyAV，主线程干这个会把界面冻住几秒。
        """
        source = Path(self.source.text().strip())
        if not source.is_file():
            self.play_hint.setText("先选一个源视频，再勾「带声音」")
            self.chk_sound.setChecked(False)
            return
        if self.side is not None and self.side.isRunning():
            self.say("[声音] 上一个后台活儿还没完，等一下再勾")
            return
        self.say(f"[声音] 正在解 {source.name} 的音轨…（只做一次，之后走缓存）")
        self.side = ClipJobWorker(self.cfg, {
            "kind": "audio", "source": str(source),
            "cache_dir": str(self.cfg.path("cache_dir"))}, self)
        self.side.log.connect(self.say)
        self.side.done.connect(self._audio_done)
        self.side.start()

    def _audio_done(self, ok: bool, message: str) -> None:
        if not ok:
            self.chk_sound.setChecked(False)
            self.play_hint.setText(f"音轨解不出来，先看画面吧：{message}")
            return
        self._audio_wav = message
        if not self.player.set_audio_file(message):
            self.chk_sound.setChecked(False)
            self.play_hint.setText("这台机器放不出声音（没有 winsound），只能看画面")
            return
        self.player.set_audio_enabled(self.chk_sound.isChecked())
        self.say("[声音] 音轨就绪，现在播卡点会带源视频原声")

    def _audio_failed(self, reason: str) -> None:
        """播放器那边报声音不可用：取消勾选并说清楚，别让用户以为静音是对齐问题。"""
        self.chk_sound.setChecked(False)
        self.play_hint.setText(f"声音关了：{reason}")
        self.say(f"[声音] {reason}")


    def _play_span(self) -> None:
        """只播算出来的那一段：seek 到 source_start，播到 source_end 自动停。

        **不重新编码任何东西**（技术指导第三十二节）：播放器就是定位 + 逐帧解码。
        """
        if self._play_to <= self._play_from:
            QMessageBox.information(self, "没有可播的区间",
                                    "先在卡点表里选一格，或者点「测试这个卡点」。")
            return
        if self.player.duration() <= 0:
            QMessageBox.warning(self, "播放器里没有视频", "源视频没载入成功。")
            return
        self.player.seek(self._play_from)
        self.player.play()

    def _halt(self) -> None:
        self.player.pause()
        self.timeline.set_playhead(self._play_from if self._play_to > self._play_from
                                   else None)

    def _on_position(self, position: float) -> None:
        """播到区间末尾就停（或循环）。这是"只播这一段"的落实点。"""
        self.timeline.set_playhead(float(position))
        if self._play_to <= self._play_from or not self.player.is_playing():
            return
        if float(position) >= self._play_to - PLAY_EPS:
            if self._loop:
                self.player.seek(self._play_from)
            else:
                self.player.pause()

    # ================================================================ 批量结果
    def _fill_batch(self) -> None:
        """批量表。覆盖率现算 —— 用的还是 `probe_all`，和单视频那边同一套判定。"""
        from ...dance import align_probe  # noqa: PLC0415

        song = float(self._payload.get("song_duration") or 0.0)
        slice_seconds = float(self.slice_seconds.value())
        picks = self._template_positions()
        self.batch.setSortingEnabled(False)
        self.batch.setRowCount(len(self._batch))
        for index, row in enumerate(self._batch):
            align = row.get("alignment")
            if align is None:
                cells = (str(row.get("name") or ""), "—", "—",
                         f"失败：{row.get('error') or '未知'}", "—", "—")
            else:
                usable = total = 0
                ratio = 0.0
                if song > 0:
                    probes = align_probe.probe_all(
                        align, song_duration=song,
                        source_duration=float(align.source_duration or 0.0),
                        slice_duration=slice_seconds, positions=picks)
                    usable, total, ratio = align_probe.coverage_of(probes)
                cells = (str(row.get("name") or ""), f"{align.offset:+.3f}",
                         f"{align.confidence:.3f}",
                         STATUS_TEXT.get(align.status, align.status),
                         f"{ratio * 100:.2f}%", f"{usable}/{total}")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (1, 2, 4, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    # 文件名那列会被拉得很宽，居中的话字飘在中间反而难扫
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.batch.setItem(index, column, item)

        self.batch.setSortingEnabled(True)
        self.batch.resizeColumnsToContents()
        self.batch.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # 默认按置信度从高到低：批量的用处就是"先挑最靠得住的那几条进库"
        self.batch.sortByColumn(2, Qt.DescendingOrder)

    def _pick_batch_row(self) -> None:
        """双击批量表某一行 → 整个页面切到那条源上（结果、卡点、播放器一起换）。"""
        index = self.batch.currentRow()
        if index < 0:
            return
        name = self.batch.item(index, 0)
        if name is None:
            return
        row = next((r for r in self._batch if str(r.get("name")) == name.text()), None)
        if row is None:
            return
        # 波形缩略图是单视频模式才算的，换到别的源上就别拿旧图糊弄人
        if str(row.get("path")) != str(self._payload.get("envelope_of") or ""):
            self._payload.pop("source_envelope", None)
        self._adopt(row)

    # ================================================================ 入库
    def _target_song(self):
        """从这次测试的结果里还原一个 `TargetSong`，供落库用。"""
        from ...dance import material_ingest as ingest  # noqa: PLC0415

        song_id = int(self._payload.get("song_id") or 0)
        path = Path(str(self._payload.get("song_path") or ""))
        if not song_id or not path.is_file():
            return None
        return ingest.TargetSong(
            song_id=song_id, path=path,
            fingerprint=str(self._payload.get("fingerprint") or ""),
            duration=float(self._payload.get("song_duration") or 0.0),
            bpm=float(self._payload.get("bpm") or 0.0),
            sample_rate=int(self._payload.get("sample_rate") or 0)
            or ingest.DEFAULT_SR)

    def _save_alignment(self) -> None:
        """把当前这条 offset 正式记进库。**只记对齐，不切片、不加计数。**"""
        from ...dance import material_ingest as ingest  # noqa: PLC0415

        if self._alignment is None or self.db is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        song = self._target_song()
        source = Path(self.source.text().strip())
        if song is None:
            QMessageBox.warning(self, "存不了", "目标歌信息不全（重跑一次对齐再试）。")
            return
        outcome = ingest.persist_alignment(self.db, song, source, self._alignment,
                                          on_log=self.say)
        if outcome.error:
            QMessageBox.warning(self, "存不了", outcome.error)
            return
        QMessageBox.information(
            self, "已保存",
            f"对齐记录 #{outcome.alignment_id} 已入库。\n"
            "素材还没有切 —— 需要的话点「切片并加入素材库」。")
        self.changed.emit()

    def _ingest(self) -> None:
        """切片入库：交给**正式那条流水线**（`DanceMontageWorker` 的对齐+切片两步）。

        为什么不在这里自己切：切片会渲染文件、写 `dance_materials`、影响以后所有推荐，
        那是正式流程该干的事，测试台只负责"确认过了，请正式来一遍"。
        """
        if self._alignment is None:
            QMessageBox.information(self, "还没有对齐结果", "先点「开始音频对齐」。")
            return
        source = self.source.text().strip()
        if not source:
            QMessageBox.warning(self, "没有源视频", "源舞蹈视频那一栏是空的。")
            return
        usable = sum(1 for row in self._rows if row.ok)
        answer = QMessageBox.question(
            self, "切片并加入素材库",
            f"要把《{Path(source).name}》按 {float(self.slice_seconds.value()):g} 秒"
            f"切成素材入库吗？\n\n"
            f"预计能切出 {usable} 条（越界的 {len(self._rows) - usable} 格会跳过）。\n"
            "这一步会渲染文件、写素材库，走的是和命令行 `dance-montage slice` 同一条流水线。"
            + ("\n\n注意：当前 offset 是**人工**改过的，切出来的素材按人工值绑定位置。"
               if self._alignment.manual else ""),
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return
        if self._alignment.manual and self.db is not None:
            # 人工值必须先落库，正式流水线才会用上它（它按缓存键读对齐）
            self._save_alignment()
        self.ingest_requested.emit({
            "song": self.target.text().strip(),
            "sources": [source],
            "slice_duration": float(self.slice_seconds.value()),
            "do_slice": True, "do_remix": False, "render": False,
            "workers": 1, "recommend": True,
        })

    def _export(self) -> None:
        """把当前源区间单独渲一个文件出来。可选功能，渲染仍然走 `media_backend`。"""
        if self._play_to <= self._play_from:
            QMessageBox.information(self, "没有可导出的区间", "先选一格可用的卡点。")
            return
        if self.side is not None and self.side.isRunning():
            QMessageBox.information(self, "还在导", "上一个片段还没导完。")
            return
        source = Path(self.source.text().strip())
        default = (Path(self.cfg.dance_path("output_dir"))
                   / f"{source.stem}_test_{self._play_from:.3f}-{self._play_to:.3f}.mp4")
        target = dialogs.save_file(self, "导出测试片段", str(default.parent),
                                   "视频 (*.mp4)", "dance.export")
        if not target:
            return
        self.side = ClipJobWorker(self.cfg, {
            "source": str(source), "start": self._play_from, "end": self._play_to,
            "target": str(Path(target).with_suffix(".mp4"))}, self)
        self.side.log.connect(self.say)
        self.side.done.connect(self._exported)
        self.btn_export.setEnabled(False)
        self.side.start()

    def _exported(self, ok: bool, message: str) -> None:
        self.btn_export.setEnabled(True)
        if ok:
            QMessageBox.information(self, "导出完成", message)
        else:
            QMessageBox.warning(self, "导出失败", message)

    # ================================================================ 重置
    def reset(self) -> None:
        """只清页面，**不动库里任何正式数据**。"""
        self.player.pause()
        self.player.close_video()
        self._audio_wav = None
        self._headline = "还没对齐"
        self.source.clear()
        self.more.clear()
        self._show_more_count()
        self._payload = {}
        self._auto = self._alignment = None
        self._rows = []
        self._batch = []
        self._play_from = self._play_to = 0.0
        for label in self.out.values():
            label.setText("—")
        self.current.setText("还没对齐")
        self.mapping.setText("时间映射：还没有结果")
        self.origin.setText("来源：—")
        self.diagnosis.clear()
        self.probe_text.setText("目标区间 / 源区间 / 能不能用，会显示在这里")
        self.coverage.setText("覆盖率：—")
        self.play_hint.setText("还没有可播的区间")
        self.positions.setRowCount(0)
        self.batch.setRowCount(0)
        self.timeline.clear()
        self.bar.setValue(0)
        self.log.clear()
        self.say("[重置] 页面已清空（素材库里的正式素材一条都没动）")


__all__ = ["POSITION_COLUMNS", "BATCH_COLUMNS", "STATUS_TEXT", "AlignBenchPanel"]










