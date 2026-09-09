"""历史面板：每一版混剪 + 每一条使用事件。

技术指导第十四节：历史版本**永不覆盖**。所以这里能做的是"再渲染一遍某一版"
和"看某一版当时挑了哪些素材"，但没有任何"改掉历史"的入口。

事件流水是只读的：它是所有计数的唯一事实来源，界面上改它就等于伪造账本。
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

VERSION_COLUMNS = ("版本", "第几版", "格数", "时长", "渲染状态", "综合重复率", "成品")
EVENT_COLUMNS = ("时间", "事件", "素材", "位置", "人物")

#: 事件名的中文说明。四个计数各管一件事，界面上必须说清楚
EVENT_TEXT = {
    "candidate": "进过候选池",
    "selected": "被选中",
    "montage": "进了编辑计划（算一次使用）",
    "render_success": "渲染成功（算一次出片）",
    "render_failed": "渲染失败（不算出片）",
    "rejected": "被约束拒掉",
}


class HistoryPanel(QWidget):
    """上：版本列表（可重渲染）｜下：事件流水（只读）。"""

    rerender_requested = pyqtSignal(int)       # version_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0
        self._versions: list[Any] = []

        box = QGroupBox("历史版本与使用流水", self)
        layout = QVBoxLayout(box)
        self.summary = QLabel("还没有历史", box)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        split = QSplitter(Qt.Vertical, box)
        self.versions = QTableWidget(0, len(VERSION_COLUMNS), split)
        self.versions.setHorizontalHeaderLabels(VERSION_COLUMNS)
        self.versions.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.versions.setSelectionMode(QAbstractItemView.SingleSelection)
        self.versions.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.versions.verticalHeader().setVisible(False)
        self.versions.horizontalHeader().setSectionResizeMode(
            len(VERSION_COLUMNS) - 1, QHeaderView.Stretch)
        split.addWidget(self.versions)

        self.events = QTableWidget(0, len(EVENT_COLUMNS), split)
        self.events.setHorizontalHeaderLabels(EVENT_COLUMNS)
        self.events.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events.verticalHeader().setVisible(False)
        self.events.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        split.addWidget(self.events)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        buttons = QHBoxLayout()
        self.btn_detail = QPushButton("看这一版用了哪些素材", box)
        self.btn_render = QPushButton("再渲染一遍这一版", box)
        for button in (self.btn_detail, self.btn_render):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.btn_detail.clicked.connect(self._show_detail)
        self.btn_render.clicked.connect(self._rerender)

    # ------------------------------------------------------------------ 数据
    def refresh(self, db: Any, song_id: int) -> None:
        from ...dance import history
        from ...dance import material_repository as repo

        self._db, self._song_id = db, int(song_id)
        self._versions = list(repo.versions_for_song(db, int(song_id))) if song_id else []
        self.versions.setRowCount(len(self._versions))
        for index, row in enumerate(self._versions):
            cells = (
                f"#{int(row['id'])}", str(int(row["version_index"])),
                str(int(row["clip_count"] or 0)),
                f"{float(row['duration'] or 0):.2f}s",
                str(row["render_status"] or ""),
                f"{float(row['overall_repeat'] or 0):.3f}",
                str(row["output_path"] or "—"),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (1, 2, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 0:
                    item.setData(Qt.UserRole, int(row["id"]))
                self.versions.setItem(index, column, item)

        rows = history.song_events(db, int(song_id), limit=300) if song_id else []
        self.events.setRowCount(len(rows))
        for index, row in enumerate(rows):
            event = str(row["event"])
            cells = (str(row["created_at"] or "")[:19],
                     EVENT_TEXT.get(event, event),
                     f"#{int(row['material_id'])}",
                     str(row["segment_index"] if row["segment_index"] is not None else "—"),
                     str(row["person"] or "(未标注)"))
            for column, text in enumerate(cells):
                self.events.setItem(index, column, QTableWidgetItem(text))

        counts = history.event_summary(db, int(song_id)) if song_id else {}
        self.summary.setText(
            f"共 {len(self._versions)} 个历史版本，{len(rows)} 条事件（只显示最近 300 条）。"
            + ("　".join(f"{EVENT_TEXT.get(k, k)} {v}" for k, v in counts.items()))
            + "。历史版本永不覆盖，事件流水只增不改。")

    def _selected_version(self) -> int:
        row = self.versions.currentRow()
        if row < 0 or row >= len(self._versions):
            QMessageBox.information(self, "先选一版", "请先在上面的表里选一个版本。")
            return 0
        item = self.versions.item(row, 0)
        return int(item.data(Qt.UserRole)) if item is not None else 0

    # ------------------------------------------------------------------ 操作
    def _show_detail(self) -> None:
        version_id = self._selected_version()
        if not version_id or self._db is None:
            return
        from ...dance import montage_timeline

        context = montage_timeline.load(self._db, version_id)
        if context is None:
            QMessageBox.warning(self, "读不出来", f"版本 #{version_id} 的计划读不出来。")
            return
        QMessageBox.information(self, f"版本 #{version_id}",
                                "\n".join(montage_timeline.describe(context, limit=30)))

    def _rerender(self) -> None:
        version_id = self._selected_version()
        if version_id:
            self.rerender_requested.emit(version_id)


__all__ = ["VERSION_COLUMNS", "EVENT_COLUMNS", "EVENT_TEXT", "HistoryPanel"]
