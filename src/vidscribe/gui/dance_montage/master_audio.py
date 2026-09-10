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

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...logging_setup import get_logger
from .. import theme
from . import dialogs
from .master_timeline import MasterTimeline

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_header())
        body = QSplitter(Qt.Horizontal, self)
        body.addWidget(self._build_navigator())
        body.addWidget(self._build_stage())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([260, 1100])
        self._body = body
        layout.addWidget(body, 1)
        layout.addWidget(self._build_status())

    # ---------------------------------------------------------------- 顶部
    def _build_header(self) -> QWidget:
        holder = QFrame(self)
        holder.setFrameShape(QFrame.StyledPanel)
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self.path = QLineEdit(holder)
        self.path.setMinimumHeight(FIELD_HEIGHT)
        self.path.setPlaceholderText("主音频（这首歌决定所有段落）")
        btn_pick = _big(QPushButton("选主音频…", holder), FIELD_HEIGHT)
        self.btn_analyze = _big(QPushButton("分析主音频", holder), 40, bold=True)
        self.bar = QProgressBar(holder)
        self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        self.bar.setMaximumWidth(160)
        self.info = QLabel("还没分析", holder)
        self.info.setStyleSheet(f"color:{theme.TEXT_DIM};")

        row.addWidget(QLabel("主音频", holder))
        row.addWidget(self.path, 1)
        row.addWidget(btn_pick)
        row.addWidget(self.btn_analyze)
        row.addWidget(self.bar)
        row.addWidget(self.info)

        btn_pick.clicked.connect(self._pick)
        self.btn_analyze.clicked.connect(self.analyze)
        return holder

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
        column.addWidget(self.timeline, 1)

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
        self.btn_prev = _big(QPushButton("◀ 上一个停顿", holder))
        self.btn_next = _big(QPushButton("▶ 下一个停顿", holder))
        self.btn_split = _big(QPushButton("✂ 在这里分段", holder), bold=True)
        self.btn_merge = _big(QPushButton("并进前一段", holder))

        row.addWidget(self.status, 1)
        for widget in (self.btn_prev, self.btn_next, self.btn_split, self.btn_merge):
            row.addWidget(widget)

        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(1))
        self.btn_split.clicked.connect(self._split_here)
        self.btn_merge.clicked.connect(self._merge_here)
        return holder

    # ---------------------------------------------------------------- 分析
    def _pick(self) -> None:
        picked = dialogs.open_file(self, "选主音频", dialogs.AUDIO_FILTER,
                                   key="dance.song")
        if picked:
            self.path.setText(picked)

    def analyze(self) -> None:
        """后台分析主音频。**不改分段** —— 分析只是把参考信息摆出来。"""
        target = self.path.text().strip()
        if not target:
            QMessageBox.information(self, "还没选主音频", "先选一首主音频再分析。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
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
        self._fill_navigator()
        self._load_saved_template()
        pauses = len(self._activity.pauses) if self._activity is not None else 0
        self.info.setText(f"{_clock(self._duration)}　·　{data.get('bpm', 0.0):.1f} BPM"
                          f"　·　{pauses} 处停顿")
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
            item.setData(Qt.UserRole, pause.middle)
            item.setToolTip(f"推荐度 {pause.score:.2f}"
                            + (f"，最近拍点 {pause.nearest_beat:.3f}s"
                               if pause.nearest_beat >= 0 else "，这首歌没有可用拍网格"))
            self.nav.addItem(item)
        if self.nav.count() == 0:
            self.nav.addItem(QListWidgetItem("（没有够格的停顿，把门槛调低看看）"))

    def _nav_picked(self, row: int) -> None:
        if row < 0:
            return
        item = self.nav.item(row)
        moment = item.data(Qt.UserRole) if item is not None else None
        if moment is None:
            return
        self._moved_to(float(moment))

    def _jump(self, direction: int) -> None:
        if self._activity is None:
            return
        pause = (self._activity.next_pause(self._at) if direction > 0
                 else self._activity.previous_pause(self._at))
        if pause is None:
            self.status.setText("没有下一处停顿了" if direction > 0 else "前面没有停顿了")
            return
        self._moved_to(pause.middle)

    def _moved_to(self, moment: float) -> None:
        self._at = max(0.0, min(float(moment), self._duration or float(moment)))
        self.timeline.set_cursor_time(self._at)
        self.seek_requested.emit(round(self._at, 3))
        self._refresh_status()

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

    # ------------------------------------------------------------ 段落编辑
    def _editor(self):
        from ...dance import segment_template as editor  # noqa: PLC0415

        return editor

    def _adopt(self, template, *, remember: bool = True) -> None:
        """换上一份新模板。`remember` 决定要不要把旧的压进撤销栈。"""
        if remember and self._template is not None:
            self._undo.append(self._template)
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
        self._adopt(self._undo.pop(), remember=False)
        self.btn_undo.setEnabled(bool(self._undo))
        self.status.setText("撤销了上一步")

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

    def state(self) -> dict[str, Any]:
        return {"path": self.path.text().strip(), "step": float(self.step.value()),
                "min_score": float(self.min_score.value()),
                "only_strong": bool(self.only_strong.isChecked()),
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
        sizes = data.get("body")
        if isinstance(sizes, list) and len(sizes) == 2:
            self._body.setSizes([int(v) for v in sizes])

    def shutdown(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)


__all__ = ["ENVELOPE_BUCKETS", "SPECTRUM_COLUMNS", "SPECTRUM_ROWS",
           "MasterAudioWorker", "MasterAudioPanel"]







