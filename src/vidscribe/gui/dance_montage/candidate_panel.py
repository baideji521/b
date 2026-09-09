"""候选池面板：一个音乐位置有哪些素材可选，以及**人工点选**。

一期第十九节 + 技术指导第二十节：关掉智能推荐之后，用户必须仍然能一格一格
自己挑。所以这个面板同时是"看候选"和"定手动方案"的地方 ——
左边选位置，右边看该位置的候选（带评分明细），双击就把它钉在这一格上。

手动选择存在 `manual` 字典里（`{位置: 素材id}`），出片时整份传给
`montage_render.remix(recommend_enabled=False, manual=...)`。
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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("素材ID", "综合分", "人物", "来源", "对齐", "使用", "出片", "为什么是它")


class CandidatePanel(QWidget):
    """左：音乐位置列表（带素材数）｜右：该位置的候选池 + 手动钉选。"""

    manual_changed = pyqtSignal(dict)          # {位置: 素材id}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0
        self._slice = 2.0
        self._spec = None
        self._manual: dict[int, int] = {}
        self._pool: Any = None

        box = QGroupBox("候选池与人工选择", self)
        layout = QVBoxLayout(box)
        self.summary = QLabel("先选目标歌，再点左边的音乐位置", box)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        split = QSplitter(Qt.Horizontal, box)
        self.positions = QListWidget(split)
        self.positions.setMinimumWidth(180)
        split.addWidget(self.positions)

        right = QWidget(split)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(COLUMNS), right)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            len(COLUMNS) - 1, QHeaderView.Stretch)
        right_layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_pick = QPushButton("钉在这一格（手动选择）", right)
        self.btn_unpick = QPushButton("取消这一格的手动选择", right)
        self.btn_clear = QPushButton("清空全部手动选择", right)
        self.btn_auto = QPushButton("按推荐填满所有格", right)
        for button in (self.btn_pick, self.btn_unpick, self.btn_clear, self.btn_auto):
            buttons.addWidget(button)
        buttons.addStretch(1)
        right_layout.addLayout(buttons)
        split.addWidget(right)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.positions.currentRowChanged.connect(self._load_pool)
        self.table.doubleClicked.connect(lambda _i: self._pick())
        self.btn_pick.clicked.connect(self._pick)
        self.btn_unpick.clicked.connect(self._unpick)
        self.btn_clear.clicked.connect(self.clear_manual)
        self.btn_auto.clicked.connect(self._fill_from_top)

    # ------------------------------------------------------------------ 数据
    def refresh(self, db: Any, song_id: int, *, slice_duration: float = 2.0,
                spec=None) -> None:
        from ...dance import material_repository as repo
        from ...dance import music_structure, statistics

        self._db, self._song_id = db, int(song_id)
        self._slice = float(slice_duration)
        self._spec = spec
        self.positions.clear()
        if not song_id:
            return
        row = repo.get_song(db, int(song_id))
        duration = float(row["duration"] or 0.0) if row is not None else 0.0
        counts = {c["segment_index"]: c for c in statistics.position_coverage(db, int(song_id))}
        for pos in music_structure.target_positions(duration, self._slice):
            info = counts.get(pos.index, {})
            mark = "★" if pos.index in self._manual else " "
            item = QListWidgetItem(
                f"{mark} 位置 {pos.index:>3}  {pos.start:>6.2f}→{pos.end:<6.2f}  "
                f"{int(info.get('materials', 0))} 条")
            item.setData(Qt.UserRole, pos.index)
            self.positions.addItem(item)
        self.summary.setText(
            f"共 {self.positions.count()} 个固定音乐位置（每格 {self._slice:g} 秒）。"
            f"已手动钉选 {len(self._manual)} 格。位置是**目标歌的属性**，"
            "和用哪个源视频无关。")
        if self.positions.count():
            self.positions.setCurrentRow(0)

    def _load_pool(self, index: int) -> None:
        if self._db is None or index < 0:
            return
        item = self.positions.item(index)
        if item is None:
            return
        position = int(item.data(Qt.UserRole))
        from ...dance import material_selection as selection

        self._pool = selection.build_pool(self._db, self._song_id, position,
                                          spec=self._spec)
        materials = self._pool.materials
        self.table.setRowCount(len(self._pool.scored))
        from ...dance.material_score import explain

        chosen = self._manual.get(position)
        for row, score in enumerate(self._pool.scored):
            material = materials.get(score.material_id)
            reason = "；".join(explain(score, limit=3))
            cells = (
                f"{'★ ' if score.material_id == chosen else ''}{score.material_id}",
                f"{score.final_score:+.3f}",
                (material.person if material else "") or "(未标注)",
                (material.source_name if material else "") or "",
                f"{material.alignment_confidence:.2f}" if material else "",
                str(material.use_count if material else 0),
                str(material.output_count if material else 0),
                reason,
            )
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if column in (0, 1, 4, 5, 6):
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 0:
                    cell.setData(Qt.UserRole, int(score.material_id))
                self.table.setItem(row, column, cell)

    # ------------------------------------------------------------------ 手动
    def manual(self) -> dict[int, int]:
        return dict(self._manual)

    def adopt(self, picks: dict[int, int]) -> None:
        """把别处算好的一套选择（比如推荐头名）合并进手动选择，逐格还能再改。"""
        self._manual.update({int(k): int(v) for k, v in picks.items()})
        self.manual_changed.emit(dict(self._manual))
        self._refresh_marks()


    def clear_manual(self) -> None:
        self._manual.clear()
        self.manual_changed.emit({})
        self.refresh(self._db, self._song_id, slice_duration=self._slice, spec=self._spec)

    def _current_position(self) -> int | None:
        item = self.positions.currentItem()
        return None if item is None else int(item.data(Qt.UserRole))

    def _pick(self) -> None:
        position = self._current_position()
        row = self.table.currentRow()
        if position is None or row < 0:
            return
        cell = self.table.item(row, 0)
        if cell is None:
            return
        self._manual[position] = int(cell.data(Qt.UserRole))
        self.manual_changed.emit(dict(self._manual))
        self._refresh_marks()

    def _unpick(self) -> None:
        position = self._current_position()
        if position is not None and position in self._manual:
            del self._manual[position]
            self.manual_changed.emit(dict(self._manual))
            self._refresh_marks()

    def _fill_from_top(self) -> None:
        """每个位置都取候选池的头名。给"我懒得一格一格点"用。"""
        if self._db is None or not self._song_id:
            return
        from ...dance import material_selection as selection
        from ...dance import music_structure
        from ...dance import material_repository as repo

        row = repo.get_song(self._db, self._song_id)
        duration = float(row["duration"] or 0.0) if row is not None else 0.0
        positions = [p.index for p in music_structure.target_positions(duration, self._slice)]
        pools = selection.build_pools(self._db, self._song_id, positions, spec=self._spec)
        for pool in pools:
            if pool.scored:
                self._manual[int(pool.segment_index)] = int(pool.scored[0].material_id)
        self.manual_changed.emit(dict(self._manual))
        self._refresh_marks()

    def _refresh_marks(self) -> None:
        keep = self.positions.currentRow()
        self.refresh(self._db, self._song_id, slice_duration=self._slice, spec=self._spec)
        if 0 <= keep < self.positions.count():
            self.positions.setCurrentRow(keep)


__all__ = ["COLUMNS", "CandidatePanel"]
