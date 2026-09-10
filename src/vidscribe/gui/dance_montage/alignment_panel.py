"""对齐面板：一首目标歌下所有源视频的对齐结果，以及人工修正入口。

一期第十四节要求对齐**独立于任何混剪**存在 —— 所以这个面板不关心成片，
它只回答一件事："这个源视频和目标歌错开了多少，这个数字可信吗"。

人工修正必须写理由（`alignment_validation.manual_override` 强制），
界面上就把理由做成必填框，不给"直接盖掉算法结果"的口子。
"""

from __future__ import annotations

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

COLUMNS = ("源视频", "偏移(秒)", "置信度", "结论", "窗口数", "最大偏差", "方法", "人工修正")

#: 结论的中文说明。库里存的是英文枚举，界面上必须说人话
STATUS_TEXT = {
    "ok": "可用",
    "low_confidence": "置信偏低（建议人工确认）",
    "rejected": "已拒绝（不要用它切素材）",
    "manual": "人工修正",
}


class AlignmentPanel(QWidget):
    """对齐结果表 + 重新对齐 / 人工修正 / 拒绝。"""

    realign_requested = pyqtSignal()
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0
        self._rows: list[Any] = []

        box = QGroupBox("音频对齐（源视频 → 目标歌）", self)
        layout = QVBoxLayout(box)
        self.summary = QLabel("还没有对齐记录", box)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(COLUMNS), box)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_realign = QPushButton("重新对齐（忽略缓存）", box)
        self.btn_fix = QPushButton("人工修正偏移…", box)
        self.btn_reject = QPushButton("标记为不可用", box)
        self.btn_accept = QPushButton("采纳（标记可用）", box)
        for button in (self.btn_realign, self.btn_fix, self.btn_reject, self.btn_accept):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.btn_realign.clicked.connect(self.realign_requested.emit)
        self.btn_fix.clicked.connect(self._fix_offset)
        self.btn_reject.clicked.connect(lambda: self._set_status("rejected"))
        self.btn_accept.clicked.connect(lambda: self._set_status("ok"))

    # ------------------------------------------------------------------ 数据
    def refresh(self, db: Any, song_id: int) -> None:
        from ...dance import material_repository as repo

        self._db, self._song_id = db, int(song_id)
        self._rows = list(repo.alignments_for_song(db, int(song_id))) if song_id else []
        self.table.setRowCount(len(self._rows))
        good = low = bad = 0
        for index, row in enumerate(self._rows):
            status = str(row["status"] or "ok")
            good += status == "ok"
            low += status == "low_confidence"
            bad += status == "rejected"
            manual = row["manual_offset"]
            cells = (
                str(row["source_name"] or row["source_path"] or f"#{row['source_video_id']}"),
                # 列名是 offset_seconds 不是 offset —— OFFSET 是 SQL 关键字，
                # 建表时刻意避开了它。这里写错过一次，直接让界面抛 IndexError
                f"{float(row['offset_seconds'] or 0.0):+.3f}",

                f"{float(row['confidence'] or 0.0):.3f}",
                STATUS_TEXT.get(status, status),
                str(int(row["window_count"] or 0)),
                f"{float(row['max_deviation'] or 0.0):.3f}",
                str(row["method"] or ""),
                "—" if manual is None else f"{float(manual):+.3f}",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (1, 2, 4, 5, 7):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(index, column, item)
        self.summary.setText(
            f"共 {len(self._rows)} 条对齐：可用 {good}｜置信偏低 {low}｜已拒绝 {bad}。"
            "偏移的口径是 source_time = target_time - offset。"
            if self._rows else "还没有对齐记录。选好目标歌和源视频，点「开始对齐」。")

    def _selected(self):
        index = self.table.currentRow()
        if index < 0 or index >= len(self._rows):
            QMessageBox.information(self, "先选一行", "请先在表里选一条对齐记录。")
            return None
        return self._rows[index]

    # ------------------------------------------------------------------ 操作
    def _fix_offset(self) -> None:
        row = self._selected()
        if row is None or self._db is None:
            return
        from ...dance import material_repository as repo

        current = float(row["offset"] or 0.0)
        value, ok = QInputDialog.getDouble(
            self, "人工修正偏移",
            f"{row['source_name']}\n当前偏移 {current:+.3f} 秒\n"
            "口径：source_time = target_time - offset",
            current, -3600.0, 3600.0, 3)
        if not ok:
            return
        reason, ok = QInputDialog.getText(
            self, "必须写理由", "为什么要改这个偏移？（会连同原值一起存档）")
        if not ok or not reason.strip():
            QMessageBox.warning(self, "没有改", "没写理由，这次修正不生效 —— 不允许静默覆盖。")
            return
        repo.set_manual_offset(self._db, int(row["id"]), offset=float(value),
                              reason=reason.strip(), operator="gui")
        QMessageBox.information(
            self, "已修正",
            f"偏移改为 {value:+.3f} 秒。算法原值仍然存着，随时查得到。\n"
            "注意：已经切好的素材不会自动重切，需要重新切片才会用上新偏移。")
        self.refresh(self._db, self._song_id)
        self.changed.emit()

    def _set_status(self, status: str) -> None:
        row = self._selected()
        if row is None or self._db is None:
            return
        self._db.connect().execute(
            "UPDATE dance_audio_alignments SET status = ?, updated_at = datetime('now') "
            "WHERE id = ?", (status, int(row["id"])))
        self.refresh(self._db, self._song_id)
        self.changed.emit()


__all__ = ["COLUMNS", "STATUS_TEXT", "AlignmentPanel"]
