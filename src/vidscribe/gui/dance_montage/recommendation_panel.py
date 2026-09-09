"""推荐面板：跑一次推荐、看排名和理由、给反馈、验证可复现。

技术指导第十七节要求推荐**可复现**，所以这个面板把种子摆在明面上：
种子看得见、改得了、跑完落库，还有一个「验证可复现」按钮当场重跑一遍对比。

一期第二十节要求"智能推荐可以关"，那个开关在混剪面板上（因为它影响的是出片路径），
这里只负责"推荐开着的时候，它在想什么"。
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("位置", "排名", "素材ID", "得分", "理由")


class RecommendationPanel(QWidget):
    """跑推荐 / 看排名 / 采纳 / 反馈 / 验证可复现。"""

    adopted = pyqtSignal(dict)             # {位置: 素材id}
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db: Any = None
        self._song_id = 0
        self._slice = 2.0
        self._run: Any = None

        box = QGroupBox("智能推荐", self)
        layout = QVBoxLayout(box)

        top = QHBoxLayout()
        top.addWidget(QLabel("策略", box))
        self.strategy = QComboBox(box)
        top.addWidget(self.strategy, 1)
        top.addWidget(QLabel("种子", box))
        self.seed = QSpinBox(box)
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setSpecialValueText("现取一个")
        top.addWidget(self.seed)
        self.keep_seed = QCheckBox("固定种子（同样的库 → 同样的推荐）", box)
        top.addWidget(self.keep_seed)
        self.btn_run = QPushButton("跑一次推荐", box)
        self.btn_verify = QPushButton("验证可复现", box)
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_verify)
        layout.addLayout(top)

        self.summary = QLabel("还没跑过推荐", box)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(COLUMNS), box)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            len(COLUMNS) - 1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_adopt = QPushButton("采纳头名（填进手动选择）", box)
        self.btn_good = QPushButton("这版推荐不错", box)
        self.btn_bad = QPushButton("这版推荐不行", box)
        for button in (self.btn_adopt, self.btn_good, self.btn_bad):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.btn_run.clicked.connect(self.run_once)
        self.btn_verify.clicked.connect(self._verify)
        self.btn_adopt.clicked.connect(self._adopt)
        self.btn_good.clicked.connect(lambda: self._feedback("good"))
        self.btn_bad.clicked.connect(lambda: self._feedback("bad"))

    # ------------------------------------------------------------------ 数据
    def refresh(self, db: Any, song_id: int, *, slice_duration: float = 2.0) -> None:
        from ...dance import material_repository as repo

        self._db, self._song_id = db, int(song_id)
        self._slice = float(slice_duration)
        current = self.strategy.currentData()
        self.strategy.clear()
        for item in repo.list_strategies(db):
            label = f"{item.name}（{item.kind}）" + ("｜默认" if item.is_default else "")
            self.strategy.addItem(label, int(item.id))

        index = self.strategy.findData(current)
        if index >= 0:
            self.strategy.setCurrentIndex(index)
        runs = repo.list_recommendation_runs(db, int(song_id), limit=1) if song_id else []
        if runs and self._run is None:
            self._show(repo.get_recommendation_run(db, int(runs[0]["id"])))

    def strategy_id(self) -> int:
        return int(self.strategy.currentData() or 0)

    def run_once(self) -> None:
        if self._db is None or not self._song_id:
            QMessageBox.information(self, "先选目标歌", "请先在上面选一首目标歌。")
            return
        from ...dance import music_structure, recommendation
        from ...dance import material_repository as repo
        from ...dance import strategy as strategy_mod

        row = repo.get_song(self._db, self._song_id)
        duration = float(row["duration"] or 0.0) if row is not None else 0.0
        positions = [p.index for p in music_structure.target_positions(duration, self._slice)]
        if not positions:
            QMessageBox.warning(self, "放不下", f"这首歌只有 {duration:.2f} 秒，"
                                              f"放不下一个 {self._slice:g} 秒的位置。")
            return
        plan = strategy_mod.resolve(self._db, self.strategy_id())
        seed = int(self.seed.value()) if (self.keep_seed.isChecked()
                                         and self.seed.value()) else 0
        run = recommendation.recommend(self._db, self._song_id, positions,
                                       strategy=plan, seed=seed)
        self.seed.setValue(int(run.random_seed))
        self._show(run)
        self.changed.emit()

    def _show(self, run: Any) -> None:
        self._run = run
        if run is None:
            return
        self.table.setRowCount(len(run.items))
        for index, item in enumerate(run.items):
            cells = (str(item.segment_index), str(item.rank), str(item.material_id),
                     f"{item.score:+.3f}", item.reason or "")
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if column != 4:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(index, column, cell)
        self.summary.setText(
            f"推荐 #{run.id}｜策略 {run.strategy_kind}｜种子 {run.random_seed}"
            f"｜算法 {run.algorithm_version}｜候选 {run.candidate_count} 条 → "
            f"推荐 {run.recommended_count} 条。种子和算法版本都已落库，"
            "换算法之后历史推荐仍然分得清是哪一版算的。")

    # ------------------------------------------------------------------ 操作
    def _verify(self) -> None:
        if self._db is None or self._run is None or not self._run.id:
            QMessageBox.information(self, "先跑一次", "得先有一条落库的推荐记录。")
            return
        from ...dance import recommendation

        _original, _again, same = recommendation.reproduce(self._db, int(self._run.id))
        if same:
            QMessageBox.information(self, "可复现 ✓",
                                    f"推荐 #{self._run.id} 用同一个种子重跑，"
                                    "排名一模一样。")
        else:
            QMessageBox.warning(self, "结果不同",
                                f"推荐 #{self._run.id} 重跑后排名变了。\n"
                                "如果这期间切了新素材或出过片，这是正常的 ——"
                                "历史会改变评分。库没动过却不一致才是 bug。")

    def _adopt(self) -> None:
        if self._run is None:
            return
        picks: dict[int, int] = {}
        for item in self._run.items:
            if int(item.rank) == 1:
                picks[int(item.segment_index)] = int(item.material_id)
        self.adopted.emit(picks)
        QMessageBox.information(self, "已采纳",
                                f"{len(picks)} 个位置的头名已填进「候选池」的手动选择里，"
                                "可以再逐格改。")

    def _feedback(self, verdict: str) -> None:
        if self._db is None or self._run is None or not self._run.id:
            return
        from ...dance import material_repository as repo

        repo.save_feedback(self._db, run_id=int(self._run.id), verdict=verdict)
        self.changed.emit()
        QMessageBox.information(self, "记下了",
                                "反馈会计入以后的评分（好评的素材更容易被推荐）。")


__all__ = ["COLUMNS", "RecommendationPanel"]
