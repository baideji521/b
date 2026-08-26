"""PyQt5 GUI：左侧视频播放器，右侧事件时间轴，底部语音文本。

点击时间轴条目或语音行 -> 播放器跳到对应真实秒数。
数据全部来自 output/<视频名>/ 下的 JSON，GUI 不做任何时间推算。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..constants import VIDEO_SUFFIXES
from ..timeline.exporters import fmt_time
from .player import FramePlayer

IMPORTANCE_COLOR = {
    "low": QColor("#8a8a8a"),
    "normal": QColor("#e6e6e6"),
    "high": QColor("#ffb454"),
    "critical": QColor("#ff5f5f"),
}


class AnalyzeWorker(QThread):
    """用子进程跑分析流水线。

    刻意不在 GUI 进程里 import torch/cv2：opencv 会改写 Qt 插件路径，
    torch 也会占住显存，分开进程更稳，而且日志可以实时回传。
    """

    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, root: Path, video: Path, force: bool = False):
        super().__init__()
        self.root = root
        self.video = video
        self.force = force
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        python = self.root / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            python = Path(sys.executable)
        cmd = [str(python), str(self.root / "run.py"), "run", str(self.video)]
        if self.force:
            cmd.append("--force")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        env.pop("QT_PLUGIN_PATH", None)

        self.log.emit("$ " + " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.log.emit(line.rstrip())
            code = self._proc.wait()
        except Exception as exc:
            self.log.emit(f"[错误] {type(exc).__name__}: {exc}")
            self.done.emit(False, str(exc))
            return
        self.done.emit(code == 0, f"退出码 {code}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, video: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.video_path: Path | None = None
        self.timeline: list[dict[str, Any]] = []
        self.speech: list[dict[str, Any]] = []
        self.worker: AnalyzeWorker | None = None

        self.setWindowTitle("视频事件 + 语音时间轴")
        self.resize(1500, 900)
        self._build_ui()

        if video:
            self.load_video(video)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        top = QHBoxLayout()
        self.btn_open = QPushButton("打开视频")
        self.btn_open.clicked.connect(self.on_open)
        self.btn_analyze = QPushButton("分析当前视频")
        self.btn_analyze.clicked.connect(lambda: self.on_analyze(False))
        self.btn_reanalyze = QPushButton("重新分析（忽略缓存）")
        self.btn_reanalyze.clicked.connect(lambda: self.on_analyze(True))
        self.btn_outdir = QPushButton("打开输出目录")
        self.btn_outdir.clicked.connect(self.on_open_outdir)

        self.cmb_importance = QComboBox()
        self.cmb_importance.addItems(["全部", "normal 以上", "high 以上", "仅 critical"])
        self.cmb_importance.currentIndexChanged.connect(self.refresh_timeline_table)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.0)
        self.spin_conf.setPrefix("置信度≥ ")
        self.spin_conf.valueChanged.connect(self.refresh_timeline_table)

        for w in (self.btn_open, self.btn_analyze, self.btn_reanalyze, self.btn_outdir):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(QLabel("重要性"))
        top.addWidget(self.cmb_importance)
        top.addWidget(self.spin_conf)

        # --- 左侧播放器（OpenCV 逐帧渲染，无声音）---
        self.player = FramePlayer()
        self.player.setMinimumSize(420, 420)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.stateChanged.connect(
            lambda playing: self.btn_play.setText("暂停" if playing else "播放")
        )

        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(lambda ms: self.player.seek(ms / 1000.0))
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setMinimumWidth(160)
        self.lbl_mute = QLabel("（预览无声音，语音见下方面板）")
        self.lbl_mute.setStyleSheet("color:#888;")

        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.lbl_time)
        controls.addWidget(self.lbl_mute)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.player, 1)
        left_layout.addLayout(controls)

        # --- 右侧时间轴 ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["时间", "重要性", "画面", "时间来源"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellClicked.connect(self.on_timeline_clicked)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("事件时间轴（点击跳转）"))
        right_layout.addWidget(self.table, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([760, 740])

        # --- 底部语音 + 日志 ---
        self.speech_list = QListWidget()
        self.speech_list.itemClicked.connect(self.on_speech_clicked)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))

        bottom = QSplitter(Qt.Horizontal)
        speech_box = QWidget()
        sb = QVBoxLayout(speech_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(QLabel("语音（点击跳转）"))
        sb.addWidget(self.speech_list)
        log_box = QWidget()
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.addWidget(QLabel("运行日志"))
        lb.addWidget(self.log_view)
        bottom.addWidget(speech_box)
        bottom.addWidget(log_box)
        bottom.setSizes([900, 600])

        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(split)
        vertical.addWidget(bottom)
        vertical.setSizes([600, 280])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top)
        layout.addWidget(vertical, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("请打开一个视频")

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(250)
        self._follow_timer.timeout.connect(self.highlight_current)
        self._follow_timer.start()

    # --------------------------------------------------------------- 数据加载
    def output_dir(self) -> Path | None:
        if self.video_path is None:
            return None
        return self.cfg.path("output_dir") / self.video_path.stem

    def load_video(self, video: Path) -> None:
        self.video_path = Path(video).resolve()
        if not self.player.open(self.video_path):
            self.append_log(f"[播放器] 无法解码 {self.video_path.name}")
        self.setWindowTitle(f"视频事件 + 语音时间轴 - {self.video_path.name}")
        self.load_results()

    def load_results(self) -> None:
        out = self.output_dir()
        self.timeline, self.speech = [], []
        if out is None:
            return
        timeline_file = out / "timeline.json"
        speech_file = out / "speech_events.json"
        if timeline_file.is_file():
            try:
                with open(timeline_file, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                self.timeline = doc.get("timeline", [])
                self.statusBar().showMessage(
                    f"{self.video_path.name}：{len(self.timeline)} 条时间轴，"
                    f"音频语言 {doc.get('original_language') or doc.get('language') or '无语音'}"
                    f" -> 输出语言 {doc.get('output_language') or '-'}"
                )
            except Exception as exc:
                self.append_log(f"[警告] 读取 timeline.json 失败: {exc}")
        else:
            self.statusBar().showMessage(f"{self.video_path.name}：还没有分析结果，点击“分析当前视频”")
        if speech_file.is_file():
            try:
                with open(speech_file, "r", encoding="utf-8") as fh:
                    self.speech = json.load(fh).get("segments", [])
            except Exception as exc:
                self.append_log(f"[警告] 读取 speech_events.json 失败: {exc}")
        self.refresh_timeline_table()
        self.refresh_speech_list()

    def refresh_timeline_table(self) -> None:
        min_rank = {0: -1, 1: 1, 2: 2, 3: 3}[self.cmb_importance.currentIndex()]
        rank = {"low": 0, "normal": 1, "high": 2, "critical": 3}
        min_conf = float(self.spin_conf.value())

        rows = []
        for entry in self.timeline:
            if entry.get("visual"):
                if rank.get(entry.get("importance", "normal"), 1) < min_rank:
                    continue
                if (entry.get("visual_confidence") or 0.0) < min_conf:
                    continue
            elif min_rank > 1 or min_conf > 0:
                continue  # 过滤模式下不显示纯语音条目
            rows.append(entry)

        self._rows = rows
        self.table.setRowCount(len(rows))
        for i, entry in enumerate(rows):
            time_text = f"{fmt_time(entry['start'])} - {fmt_time(entry['end'])}"
            visual = entry.get("visual") or f"（无画面事件）{entry.get('speech') or ''}"[:60]
            cells = [
                time_text,
                entry.get("importance", "-") if entry.get("visual") else "-",
                visual,
                entry.get("timestamp_source", "-"),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, float(entry["start"]))
                if col == 2 and entry.get("visual"):
                    color = IMPORTANCE_COLOR.get(entry.get("importance", "normal"))
                    if color:
                        item.setForeground(QBrush(color))
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

    def refresh_speech_list(self) -> None:
        self.speech_list.clear()
        for seg in self.speech:
            conf = seg.get("confidence")
            suffix = f"  (conf {conf:.2f})" if isinstance(conf, (int, float)) else ""
            item = QListWidgetItem(f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}] {seg['text']}{suffix}")
            item.setData(Qt.UserRole, float(seg["start"]))
            self.speech_list.addItem(item)

    # ------------------------------------------------------------------ 交互
    def on_open(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(VIDEO_SUFFIXES))
        start_dir = str(self.cfg.path("input_dir") if self.cfg.path("input_dir").is_dir() else self.cfg.root)
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", start_dir, f"视频文件 ({patterns})")
        if path:
            self.load_video(Path(path))

    def on_analyze(self, force: bool) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开一个视频")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "已有分析任务在运行")
            return
        self.append_log(f"开始分析 {self.video_path.name}（force={force}）…")
        self.btn_analyze.setEnabled(False)
        self.btn_reanalyze.setEnabled(False)
        self.worker = AnalyzeWorker(self.cfg.root, self.video_path, force)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_analyze_done)
        self.worker.start()

    def on_analyze_done(self, ok: bool, message: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_reanalyze.setEnabled(True)
        if ok:
            self.append_log(f"分析完成（{message}），重新加载结果")
            self.load_results()
        else:
            self.append_log(f"分析失败：{message}")
            QMessageBox.warning(self, "分析失败", f"{message}\n详细日志见 logs/ 目录")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.player.close_video()
        super().closeEvent(event)

    def on_open_outdir(self) -> None:
        out = self.output_dir()
        if out is None or not out.is_dir():
            QMessageBox.information(self, "提示", "还没有输出目录")
            return
        if os.name == "nt":
            os.startfile(str(out))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(out)])

    def on_timeline_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        self.seek(float(item.data(Qt.UserRole)))

    def on_speech_clicked(self, item: QListWidgetItem) -> None:
        self.seek(float(item.data(Qt.UserRole)))

    def seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        if not self.player.is_playing():
            self.player.play()

    def toggle_play(self) -> None:
        self.player.toggle()

    def on_position_changed(self, seconds: float) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(int(round(seconds * 1000)))
        self.lbl_time.setText(f"{fmt_time(seconds)} / {fmt_time(self.player.duration())}")

    def on_duration_changed(self, seconds: float) -> None:
        self.slider.setRange(0, max(int(round(seconds * 1000)), 0))

    def highlight_current(self) -> None:
        seconds = self.player.position()
        rows = getattr(self, "_rows", [])
        for i, entry in enumerate(rows):
            if entry["start"] - 0.01 <= seconds <= entry["end"] + 0.01:
                if self.table.currentRow() != i:
                    self.table.selectRow(i)
                break
        for i in range(self.speech_list.count()):
            item = self.speech_list.item(i)
            seg = self.speech[i] if i < len(self.speech) else None
            if seg and seg["start"] - 0.01 <= seconds <= seg["end"] + 0.01:
                font = item.font()
                if not font.bold():
                    font.setBold(True)
                    item.setFont(font)
                    self.speech_list.scrollToItem(item)
            else:
                font = item.font()
                if font.bold():
                    font.setBold(False)
                    item.setFont(font)

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)


def launch(cfg: Config, video: str | Path | None = None) -> int:
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = MainWindow(cfg, Path(video) if video else None)
    window.show()
    return app.exec_()
