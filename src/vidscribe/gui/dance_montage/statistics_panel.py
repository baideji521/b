"""统计面板：素材够不够、位置有没有空、谁出镜太多、重复率高不高。

一期第三十九节要的"位置级使用统计"就在这儿：一条素材总共用了 8 次，
但如果全在第 3 格，那第 5 格用它仍然算新的 —— 所以按位置看才有意义。

所有数字都来自 SQL 聚合，不扫盘、不猜。
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

POSITION_COLUMNS = ("位置", "时间", "素材数", "人物数", "被用过", "出过片")
PERSON_COLUMNS = ("人物", "素材数", "使用次数", "出片次数", "覆盖位置", "平均对齐")


class StatisticsPanel(QWidget):
    """总览 + 位置覆盖 + 人物分布。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0

        box = QGroupBox("统计", self)
        layout = QVBoxLayout(box)
        self.overview = QLabel("先选一首目标歌", box)
        self.overview.setWordWrap(True)
        self.overview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.overview)

        self.gaps = QLabel("", box)
        self.gaps.setWordWrap(True)
        layout.addWidget(self.gaps)

        tables = QHBoxLayout()
        self.positions = self._table(POSITION_COLUMNS, box)
        self.persons = self._table(PERSON_COLUMNS, box)
        tables.addWidget(self.positions, 3)
        tables.addWidget(self.persons, 2)
        layout.addLayout(tables, 1)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新统计", box)
        row.addWidget(self.btn_refresh)
        row.addStretch(1)
        layout.addLayout(row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)
        self.btn_refresh.clicked.connect(lambda: self.refresh(self._db, self._song_id))

    def _table(self, columns, parent) -> QTableWidget:
        table = QTableWidget(0, len(columns), parent)
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        return table

    # ------------------------------------------------------------------ 数据
    def refresh(self, db: Any, song_id: int) -> None:
        from ...dance import history, statistics

        self._db, self._song_id = db, int(song_id)
        if db is None or not song_id:
            return
        data = statistics.song_overview(db, int(song_id))
        materials = data["materials"]
        versions = data["versions"]
        events = history.event_summary(db, int(song_id))
        self.overview.setText(
            f"素材 {materials['total']} 条（可用 {materials['ready']}，"
            f"从未使用 {materials['never_used']}，从未出片 {materials['never_output']}）"
            f"｜来源 {materials['sources']} 个｜人物 {materials['persons']} 个"
            f"｜覆盖 {materials['positions']} 个位置"
            f"｜平均对齐置信 {materials['avg_confidence']:.3f}\n"
            f"混剪版本 {versions['total']} 个（渲染成功 {versions['rendered']}，"
            f"失败 {versions['failed']}）｜平均综合重复率 {versions['avg_repeat']:.3f}"
            f"｜最好的一版 {versions['best_repeat']:.3f}\n"
            "事件：" + ("　".join(f"{k} {v}" for k, v in events.items()) or "还没有"))

        coverage = statistics.position_coverage(db, int(song_id))
        self.positions.setRowCount(len(coverage))
        thin = []
        for index, row in enumerate(coverage):
            if int(row["materials"]) <= 1:
                thin.append(int(row["segment_index"]))
            cells = (str(row["segment_index"]),
                     f"{float(row['start'] or 0):.2f}→{float(row['end'] or 0):.2f}",
                     str(row["materials"]), str(row["persons"]),
                     str(row["uses"]), str(row["outputs"]))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column != 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.positions.setItem(index, column, item)
        self.gaps.setText(
            f"素材偏少的位置（只有 1 条可选）：{thin}　—— 这些格子基本没得挑，"
            "多切几个源进来会明显好看。" if thin else
            "每个位置都有 2 条以上素材可选，组合搜索有得挑。")

        people = statistics.person_breakdown(db, int(song_id))
        self.persons.setRowCount(len(people))
        for index, row in enumerate(people):
            cells = (str(row["person"]), str(row["materials"]), str(row["uses"]),
                     str(row["outputs"]), str(row["positions"]),
                     f"{float(row['confidence'] or 0):.2f}")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.persons.setItem(index, column, item)


__all__ = ["POSITION_COLUMNS", "PERSON_COLUMNS", "StatisticsPanel"]
