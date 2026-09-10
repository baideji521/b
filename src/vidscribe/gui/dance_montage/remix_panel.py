"""混剪面板：这个界面的主操作区。选歌、选源、定参数、开跑、看进度。

一期第二十一节要求"一个主行动按钮" —— 所以这里只有一个大按钮「开始」，
其余都是它的参数。停止是协作式的：跑到当前这一步做完才停，不会写坏文件。

切片时长做成预设（1.0/1.5/2.0/2.5/3.0/自定义）是一期第二十二节的原话。
"智能推荐"开关关掉之后，出片走候选池面板里那份手动选择。
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...logging_setup import get_logger
from . import dialogs
from .dialogs import AUDIO_FILTER, VIDEO_FILTER
from .worker import STAGES

logger = get_logger("dance.gui.remix")


#: 一期第二十二节钉的那几档。"自定义"落到 -1，由输入框接管
SLICE_PRESETS = ((1.0, "1.0 秒（快切）"), (1.5, "1.5 秒"), (2.0, "2.0 秒（常用）"),
                 (2.5, "2.5 秒"), (3.0, "3.0 秒（慢摆）"), (-1.0, "自定义…"))



class RemixPanel(QWidget):
    """参数 + 一个主行动按钮 + 九阶段进度 + 日志。"""

    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()
    song_changed = pyqtSignal(str)
    slice_changed = pyqtSignal(float)
    remove_song_requested = pyqtSignal(int)


    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self._manual: dict[int, int] = {}
        self._running = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_inputs())
        outer.addWidget(self._build_options())
        outer.addWidget(self._build_action())
        outer.addWidget(self._build_log(), 1)

    # ------------------------------------------------------------------ 建界面
    def _build_inputs(self) -> QGroupBox:
        box = QGroupBox("① 目标歌与源视频", self)
        grid = QGridLayout(box)

        self.song = QLineEdit(box)
        self.song.setPlaceholderText("目标歌文件（wav/mp3/mp4…），或库里的歌 id")
        btn_song = QPushButton("选歌…", box)
        self.songs = QComboBox(box)
        self.songs.addItem("（库里已有的目标歌）", "")
        self.btn_remove_song = QPushButton("移除这首…", box)
        self.btn_remove_song.setToolTip("把下拉框里选中的目标歌从界面上移除。\n"
                                       "默认是「下架」：素材、对齐、历史成片一条都不动，随时能恢复。\n"
                                       "确实是误加进来的，才走「彻底删除」那一档。")
        self.show_retired = QCheckBox("含已下架", box)
        self.show_retired.setToolTip("勾上之后下拉框里也列出已下架的歌，选中它再点「移除这首…」就能恢复")
        song_tools = QHBoxLayout()
        song_tools.setContentsMargins(0, 0, 0, 0)
        song_tools.addWidget(self.show_retired)
        song_tools.addWidget(self.btn_remove_song)
        grid.addWidget(QLabel("目标歌", box), 0, 0)
        grid.addWidget(self.song, 0, 1)
        grid.addWidget(btn_song, 0, 2)
        grid.addWidget(self.songs, 1, 1)
        grid.addLayout(song_tools, 1, 2)


        self.sources = QListWidget(box)
        self.sources.setMaximumHeight(110)
        btn_add = QPushButton("加源视频…", box)
        btn_dir = QPushButton("加整个目录…", box)
        btn_drop = QPushButton("清空列表", box)
        grid.addWidget(QLabel("源视频", box), 2, 0)
        grid.addWidget(self.sources, 2, 1, 3, 1)
        grid.addWidget(btn_add, 2, 2)
        grid.addWidget(btn_dir, 3, 2)
        grid.addWidget(btn_drop, 4, 2)

        btn_song.clicked.connect(self._pick_song)
        btn_add.clicked.connect(self._add_files)
        btn_dir.clicked.connect(self._add_dir)
        btn_drop.clicked.connect(self.sources.clear)
        self.btn_remove_song.clicked.connect(self._ask_remove_song)
        self.song.textChanged.connect(self.song_changed.emit)
        self.songs.currentIndexChanged.connect(self._pick_known_song)

        return box

    def _build_options(self) -> QGroupBox:
        box = QGroupBox("② 参数", self)
        grid = QGridLayout(box)

        self.slice_preset = QComboBox(box)
        for value, label in SLICE_PRESETS:
            self.slice_preset.addItem(label, value)
        self.slice_preset.setCurrentIndex(2)
        self.slice_custom = QDoubleSpinBox(box)
        self.slice_custom.setRange(0.2, 30.0)
        self.slice_custom.setSingleStep(0.1)
        self.slice_custom.setValue(float(self.cfg.dance["slice_duration"]))
        self.slice_custom.setEnabled(False)
        grid.addWidget(QLabel("每格时长", box), 0, 0)
        grid.addWidget(self.slice_preset, 0, 1)
        grid.addWidget(self.slice_custom, 0, 2)

        self.versions = QSpinBox(box)
        self.versions.setRange(1, 20)
        self.versions.setValue(int(self.cfg.dance["versions_per_run"]))
        grid.addWidget(QLabel("出几个版本", box), 0, 3)
        grid.addWidget(self.versions, 0, 4)

        self.workers = QSpinBox(box)
        self.workers.setRange(1, 6)
        self.workers.setValue(int(self.cfg.dance["align_workers"]))
        grid.addWidget(QLabel("对齐并发", box), 1, 0)
        grid.addWidget(self.workers, 1, 1)

        self.person = QLineEdit(box)
        self.person.setPlaceholderText("给这批素材标个人物名（可留空）")
        grid.addWidget(QLabel("人物标注", box), 1, 2)
        grid.addWidget(self.person, 1, 3, 1, 2)

        self.out_dir = QLineEdit(str(self.cfg.dance_path("output_dir")), box)
        btn_out = QPushButton("改输出目录…", box)
        grid.addWidget(QLabel("成品目录", box), 2, 0)
        grid.addWidget(self.out_dir, 2, 1, 1, 3)
        grid.addWidget(btn_out, 2, 4)

        self.recommend = QCheckBox("启用智能推荐（关掉就用「候选池」里那份手动选择）", box)
        self.recommend.setChecked(bool(self.cfg.dance["recommend_enabled"]))
        self.force = QCheckBox("忽略对齐缓存，全部重算", box)
        self.plan_only = QCheckBox("只出编辑计划，先不渲染", box)
        self.do_slice = QCheckBox("对齐后接着切片", box)
        self.do_slice.setChecked(True)
        self.do_remix = QCheckBox("切片后接着出片", box)
        self.do_remix.setChecked(True)
        grid.addWidget(self.recommend, 3, 0, 1, 5)
        grid.addWidget(self.do_slice, 4, 0, 1, 2)
        grid.addWidget(self.do_remix, 4, 2, 1, 3)
        grid.addWidget(self.force, 5, 0, 1, 2)
        grid.addWidget(self.plan_only, 5, 2, 1, 3)

        self.manual_hint = QLabel("手动选择：0 格", box)
        grid.addWidget(self.manual_hint, 6, 0, 1, 5)

        self.slice_preset.currentIndexChanged.connect(self._slice_mode)
        self.slice_custom.valueChanged.connect(
            lambda _v: self.slice_changed.emit(self.slice_duration()))
        btn_out.clicked.connect(self._pick_out)
        return box

    def _build_action(self) -> QGroupBox:
        box = QGroupBox("③ 开跑", self)
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.btn_start = QPushButton("开始", box)
        self.btn_start.setMinimumHeight(38)
        self.btn_stop = QPushButton("停止", box)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start, 3)
        row.addWidget(self.btn_stop, 1)
        layout.addLayout(row)

        self.stage_label = QLabel("待命", box)
        layout.addWidget(self.stage_label)
        self.stage_bar = QProgressBar(box)
        self.stage_bar.setRange(0, len(STAGES))
        layout.addWidget(self.stage_bar)
        self.step_bar = QProgressBar(box)
        self.step_bar.setRange(0, 100)
        self.step_bar.setFormat("%v / %m")
        layout.addWidget(self.step_bar)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        return box

    def _build_log(self) -> QGroupBox:
        box = QGroupBox("运行日志", self)
        layout = QVBoxLayout(box)
        self.log = QTextEdit(box)
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log)
        return box

    # ------------------------------------------------------------------ 取值
    def slice_duration(self) -> float:
        value = float(self.slice_preset.currentData() or 2.0)
        return float(self.slice_custom.value()) if value < 0 else value

    def job(self) -> dict[str, Any]:
        return {
            "song": self.song.text().strip(),
            "sources": [self.sources.item(i).text() for i in range(self.sources.count())],
            "slice_duration": self.slice_duration(),
            "workers": int(self.workers.value()),
            "versions": int(self.versions.value()),
            "person": self.person.text().strip(),
            "out_dir": self.out_dir.text().strip(),
            "force": self.force.isChecked(),
            "do_slice": self.do_slice.isChecked(),
            "do_remix": self.do_remix.isChecked(),
            "render": not self.plan_only.isChecked(),
            "recommend": self.recommend.isChecked(),
            "manual": dict(self._manual),
        }

    def set_manual(self, manual: dict[int, int]) -> None:
        self._manual = dict(manual)
        self.manual_hint.setText(
            f"手动选择：{len(self._manual)} 格"
            + ("（智能推荐已关，出片就用这份）" if not self.recommend.isChecked() else ""))

    def set_songs(self, rows) -> None:
        """把库里已有的目标歌铺进下拉框。已下架的加个前缀标出来。"""
        current = self.songs.currentData()
        self.songs.blockSignals(True)
        self.songs.clear()
        self.songs.addItem("（库里已有的目标歌）", "")
        for row in rows:
            retired = False
            try:                       # 老调用方可能传不带 retired_at 的行，不能因此崩
                retired = bool(row["retired_at"])
            except (IndexError, KeyError, TypeError):
                retired = False
            self.songs.addItem(
                f"{'【已下架】' if retired else ''}#{int(row['id'])} {row['title']}"
                f"（{float(row['duration'] or 0):.1f}s，BPM {float(row['bpm'] or 0):.0f}）",
                str(int(row["id"])))
        index = self.songs.findData(current)
        if index >= 0:
            self.songs.setCurrentIndex(index)
        self.songs.blockSignals(False)


    # ------------------------------------------------------------------ 交互
    def _slice_mode(self) -> None:
        custom = float(self.slice_preset.currentData() or 2.0) < 0
        self.slice_custom.setEnabled(custom)
        self.slice_changed.emit(self.slice_duration())

    def _pick_song(self) -> None:
        path = self._open_file("选目标歌", self.cfg.dance_path("song_dir"), AUDIO_FILTER)
        if path:
            self.song.setText(path)

    def _pick_known_song(self) -> None:
        data = str(self.songs.currentData() or "")
        if data:
            self.song.setText(data)

    def _ask_remove_song(self) -> None:
        """把下拉框里选中的那首歌交给主窗口处理（要查库、要摆数字，逻辑不放这一层）。"""
        from PyQt5.QtWidgets import QMessageBox

        data = str(self.songs.currentData() or "").strip()
        if not data.isdigit():
            QMessageBox.information(self, "先选一首",
                                    "请先在下拉框里选中一首库里已有的目标歌。\n"
                                    "（上面输入框里填的文件路径还没入库，谈不上移除）")
            return
        self.remove_song_requested.emit(int(data))


    def _add_files(self) -> None:
        for path in self._open_files("选源舞蹈视频", self.cfg.dance_path("source_dir"),
                                     VIDEO_FILTER):
            self.sources.addItem(path)

    def _add_dir(self) -> None:
        path = self._open_dir("选一个装满源视频的目录", self.cfg.dance_path("source_dir"))
        if path:
            self.sources.addItem(path)

    def _pick_out(self) -> None:
        path = self._open_dir("成品放哪儿", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    # ---------------------------------------------------- 选文件（统一走这三个）
    #
    # 规矩本身和为什么这么定，见 `dialogs.py`：一律用 Qt 自己画的对话框。
    # 这几个方法只是转发 —— 保留它们是因为面板内部和测试都按名字在调。
    def _dialog_options(self):
        return dialogs.options()

    def _start_dir(self, folder) -> str:
        return dialogs.start_dir(folder)

    def _open_file(self, title: str, folder, filters: str) -> str:
        return dialogs.open_file(self, title, folder, filters)

    def _open_files(self, title: str, folder, filters: str) -> list[str]:
        return dialogs.open_files(self, title, folder, filters)

    def _open_dir(self, title: str, folder) -> str:
        return dialogs.open_dir(self, title, folder)



    def _start(self) -> None:
        if self._running:
            return
        self.log.clear()
        self.start_requested.emit(self.job())

    def _stop(self) -> None:
        self.stop_requested.emit()

    # ------------------------------------------------------------------ 回显
    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if not running:
            self.step_bar.setRange(0, 100)
            self.step_bar.setValue(0)

    def append_log(self, line: str) -> None:
        self.log.append(line)

    def show_stage(self, name: str, index: int, total: int) -> None:
        self.stage_bar.setRange(0, total)
        self.stage_bar.setValue(index + 1)
        self.stage_label.setText(f"第 {index + 1}/{total} 步：{name}")

    def show_progress(self, current: int, total: int, note: str) -> None:
        self.step_bar.setRange(0, max(1, int(total)))
        self.step_bar.setValue(int(current))
        if note:
            self.step_bar.setFormat(f"{note}  %v / %m")

    def show_done(self, ok: bool, message: str) -> None:
        self.stage_label.setText(("完成：" if ok else "结束：") + message)
        self.stage_bar.setValue(self.stage_bar.maximum() if ok else self.stage_bar.value())


__all__ = ["SLICE_PRESETS", "RemixPanel"]
