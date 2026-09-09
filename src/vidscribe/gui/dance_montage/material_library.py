"""素材库面板：一首目标歌下的所有素材，一期第十八节那张表。

两条不能破的规矩写在这里：

  - **素材是长期资产**：界面上只有"停用"，没有"删除"。停用只改 status，行永远留着
  - 计数只读：`候选/使用/出片` 这三列不能在界面上手改，它们由事件流水推算
    （要修就点「按流水重算」，走 `history.recount_song`）
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("ID", "位置", "目标时间", "源时间", "人物", "来源视频",
           "对齐", "候选", "使用", "出片", "最近使用", "状态", "文件")

STATUS_TEXT = {
    "ready": "可用",
    "disabled": "已停用",
    "missing": "文件丢失",
    "invalid": "无效",
    "regenerated": "已被重切版本取代",
}


class MaterialLibraryPanel(QWidget):
    """素材总表 + 停用/启用 / 标注人物 / 打开文件 / 按流水重算。"""

    selected = pyqtSignal(int)            # material_id
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0
        self._rows: list[Any] = []

        box = QGroupBox("素材库", self)
        layout = QVBoxLayout(box)
        self.summary = QLabel("还没有素材", box)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(COLUMNS), box)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(
            len(COLUMNS) - 1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_disable = QPushButton("停用（不删文件）", box)
        self.btn_enable = QPushButton("恢复可用", box)
        self.btn_person = QPushButton("标注人物…", box)
        self.btn_open = QPushButton("在文件夹里打开", box)
        self.btn_recount = QPushButton("按流水重算计数", box)
        for button in (self.btn_disable, self.btn_enable, self.btn_person,
                       self.btn_open, self.btn_recount):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.btn_disable.clicked.connect(lambda: self._set_status("disabled"))
        self.btn_enable.clicked.connect(lambda: self._set_status("ready"))
        self.btn_person.clicked.connect(self._mark_person)
        self.btn_open.clicked.connect(self._open_file)
        self.btn_recount.clicked.connect(self._recount)
        self.table.itemSelectionChanged.connect(self._announce)

    # ------------------------------------------------------------------ 数据
    def show_materials(self, db: Any, song_id: int, materials: list[Any]) -> None:
        """铺表。素材由调用方（筛选面板 + selection.find_materials）给，这里只显示。"""
        self._db, self._song_id = db, int(song_id)
        self._rows = list(materials)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._rows))
        for index, m in enumerate(self._rows):
            cells = (
                str(m.id), str(m.segment_index),
                f"{m.target_start:.2f}→{m.target_end:.2f}",
                f"{m.source_start:.2f}→{m.source_end:.2f}",
                m.person or "(未标注)", m.source_name or f"#{m.source_video_id}",
                f"{m.alignment_confidence:.2f}",
                str(m.candidate_count), str(m.use_count), str(m.output_count),
                (m.last_used_at or "—")[:19], STATUS_TEXT.get(m.status, m.status),
                str(m.file_path or ""),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (0, 1, 6, 7, 8, 9):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 0:
                    item.setData(Qt.UserRole, int(m.id))
                self.table.setItem(index, column, item)
        self.table.setSortingEnabled(True)
        unused = sum(1 for m in self._rows if m.use_count <= 0)
        never = sum(1 for m in self._rows if m.output_count <= 0)
        self.summary.setText(
            f"命中 {len(self._rows)} 条素材：从未使用 {unused}｜从未出片 {never}。"
            "「使用」= 进过某一版编辑计划；「出片」= 那一版真的渲染成功了。"
            if self._rows else "这一组条件下没有素材。放宽筛选，或先去切片。")

    def _ids(self) -> list[int]:
        out: list[int] = []
        for index in {i.row() for i in self.table.selectedIndexes()}:
            item = self.table.item(index, 0)
            if item is not None:
                out.append(int(item.data(Qt.UserRole) or item.text()))
        return out

    def _announce(self) -> None:
        ids = self._ids()
        if len(ids) == 1:
            self.selected.emit(ids[0])

    # ------------------------------------------------------------------ 操作
    def _set_status(self, status: str) -> None:
        from ...dance import material_repository as repo

        ids = self._ids()
        if not ids or self._db is None:
            QMessageBox.information(self, "先选素材", "请先在表里选中至少一条素材。")
            return
        for material_id in ids:
            repo.set_material_status(self._db, material_id, status)
        QMessageBox.information(
            self, "已更新",
            f"{len(ids)} 条素材改为「{STATUS_TEXT.get(status, status)}」。\n"
            "文件还在盘上、历史记录也都在 —— 素材是长期资产，界面上不提供删除。")
        self.changed.emit()

    def _mark_person(self) -> None:
        ids = self._ids()
        if not ids or self._db is None:
            QMessageBox.information(self, "先选素材", "请先在表里选中至少一条素材。")
            return
        name, ok = QInputDialog.getText(self, "标注人物",
                                        f"给这 {len(ids)} 条素材标一个人物名：")
        if not ok:
            return
        marks = ",".join("?" for _ in ids)
        self._db.connect().execute(
            f"UPDATE dance_materials SET person = ?, updated_at = datetime('now') "
            f"WHERE id IN ({marks})", (name.strip(), *ids))
        self.changed.emit()

    def _open_file(self) -> None:
        ids = self._ids()
        if not ids or self._db is None:
            return
        from ...dance import material_repository as repo

        material = repo.get_material(self._db, ids[0])
        if material is None or not material.file_path:
            return
        path = material.file_path
        if not os.path.isfile(path):
            QMessageBox.warning(self, "文件不在盘上",
                                f"{path}\n素材记录还在库里，但文件找不到了。"
                                "可以把它标成「文件丢失」，或者重新切片。")
            return
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])

    def _recount(self) -> None:
        if self._db is None or not self._song_id:
            return
        from ...dance import history

        report = history.recount_song(self._db, self._song_id)
        QMessageBox.information(
            self, "重算完成",
            f"检查 {report.get('total', 0)} 条素材，修正 {report.get('changed', 0)} 条。\n"
            "所有计数一律从事件流水重算 —— 流水是唯一事实来源。")
        self.changed.emit()


__all__ = ["COLUMNS", "STATUS_TEXT", "MaterialLibraryPanel"]
