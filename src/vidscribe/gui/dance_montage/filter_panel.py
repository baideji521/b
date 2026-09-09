"""筛选面板：一期第十七节那七套方案 A~G，加上几个手动条件。

这个面板**只产出一份 `FilterSpec`**，自己不查库、不打分。谁要用它自己去查 ——
这样"筛选条件"就能被素材库面板和候选池面板共用，不会出现两处各写一套筛法。
"""

from __future__ import annotations

from dataclasses import replace

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...dance.material_selection import PRESETS
from ...dance.types import SORT_KEYS, FilterSpec

#: 排序方式的中文名。库里存英文枚举，界面上说人话
SORT_TEXT = {
    "score_desc": "综合评分（高→低）",
    "use_count_asc": "使用次数（少→多）",
    "output_count_asc": "出片次数（少→多）",
    "last_used_at_asc": "最久没用的在前",
    "confidence_desc": "对齐置信（高→低）",
    "freshness_desc": "新鲜度（新→旧）",
}


class FilterPanel(QWidget):
    """七套预设方案 + 手动条件 → 一份 FilterSpec。"""

    changed = pyqtSignal(object)          # FilterSpec

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        box = QGroupBox("筛选素材", self)
        grid = QGridLayout(box)

        self.preset = QComboBox(box)
        self.preset.addItem("（不用预设方案）", "")
        for key, spec in PRESETS.items():
            self.preset.addItem(str(spec.get("label", key)), key)
        grid.addWidget(QLabel("方案", box), 0, 0)
        grid.addWidget(self.preset, 0, 1, 1, 3)

        self.sort = QComboBox(box)
        for key in SORT_KEYS:
            self.sort.addItem(SORT_TEXT.get(key, key), key)
        grid.addWidget(QLabel("排序", box), 1, 0)
        grid.addWidget(self.sort, 1, 1, 1, 3)

        self.never_used = QCheckBox("只看从未使用", box)
        self.never_output = QCheckBox("只看从未出片", box)
        self.exclude_recent = QCheckBox("排除最近用过的", box)
        grid.addWidget(self.never_used, 2, 0, 1, 2)
        grid.addWidget(self.never_output, 2, 2, 1, 2)
        grid.addWidget(self.exclude_recent, 3, 0, 1, 2)

        self.min_confidence = QDoubleSpinBox(box)
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.05)
        self.min_confidence.setDecimals(2)
        grid.addWidget(QLabel("对齐置信 ≥", box), 4, 0)
        grid.addWidget(self.min_confidence, 4, 1)

        self.max_use = QSpinBox(box)
        self.max_use.setRange(-1, 999)
        self.max_use.setValue(-1)
        self.max_use.setSpecialValueText("不限")
        grid.addWidget(QLabel("使用次数 ≤", box), 4, 2)
        grid.addWidget(self.max_use, 4, 3)

        self.days = QDoubleSpinBox(box)
        self.days.setRange(0.0, 3650.0)
        self.days.setDecimals(1)
        self.days.setSpecialValueText("不限")
        grid.addWidget(QLabel("多少天没用过", box), 5, 0)
        grid.addWidget(self.days, 5, 1)

        self.limit = QSpinBox(box)
        self.limit.setRange(10, 5000)
        self.limit.setValue(300)
        grid.addWidget(QLabel("最多显示", box), 5, 2)
        grid.addWidget(self.limit, 5, 3)

        self.person = QLineEdit(box)
        self.person.setPlaceholderText("人物，多个用逗号隔开")
        grid.addWidget(QLabel("人物", box), 6, 0)
        grid.addWidget(self.person, 6, 1, 1, 3)

        self.search = QLineEdit(box)
        self.search.setPlaceholderText("搜文件名 / 备注")
        grid.addWidget(QLabel("搜索", box), 7, 0)
        grid.addWidget(self.search, 7, 1, 1, 3)

        self.include_disabled = QCheckBox("连停用的一起显示（素材是长期资产，从不物理删除）", box)
        grid.addWidget(self.include_disabled, 8, 0, 1, 4)

        self.btn_apply = QPushButton("应用筛选", box)
        self.btn_reset = QPushButton("清空条件", box)
        grid.addWidget(self.btn_apply, 9, 0, 1, 2)
        grid.addWidget(self.btn_reset, 9, 2, 1, 2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self.btn_apply.clicked.connect(self._emit)
        self.btn_reset.clicked.connect(self.reset)
        self.preset.currentIndexChanged.connect(self._emit)
        self.sort.currentIndexChanged.connect(self._emit)

    # ------------------------------------------------------------------ 取值
    def spec(self, song_id: int = 0, segment_index: int | None = None) -> FilterSpec:
        """把界面上的选择拼成 FilterSpec。预设方案先铺底，手动条件再覆盖。"""
        base = FilterSpec(
            target_song_id=int(song_id) or None,
            segment_index=segment_index,
            statuses=(("ready", "disabled", "missing", "regenerated")
                      if self.include_disabled.isChecked() else ("ready",)),
            sort=str(self.sort.currentData() or "score_desc"),
            limit=int(self.limit.value()))
        key = str(self.preset.currentData() or "")
        if key:
            from ...dance.material_selection import preset_spec

            base = preset_spec(key, base)
        changes: dict = {}
        if self.never_used.isChecked():
            changes["never_used"] = True
        if self.never_output.isChecked():
            changes["never_output"] = True
        if self.min_confidence.value() > 0:
            changes["min_confidence"] = float(self.min_confidence.value())
        if self.max_use.value() >= 0:
            changes["max_use_count"] = int(self.max_use.value())
        if self.days.value() > 0:
            changes["long_unused_days"] = float(self.days.value())
        people = tuple(p.strip() for p in self.person.text().split(",") if p.strip())
        if people:
            changes["persons"] = people
        if self.search.text().strip():
            changes["search"] = self.search.text().strip()
        return replace(base, **changes) if changes else base

    def exclude_recent_mode(self) -> bool:
        """G 方案的"反过来筛"开关。语义翻转必须显式传给查询层，不能藏在 spec 里。"""
        return bool(self.exclude_recent.isChecked()
                    or str(self.preset.currentData() or "") == "exclude_recent")

    def reset(self) -> None:
        self.preset.setCurrentIndex(0)
        self.sort.setCurrentIndex(0)
        for check in (self.never_used, self.never_output, self.exclude_recent,
                      self.include_disabled):
            check.setChecked(False)
        self.min_confidence.setValue(0.0)
        self.max_use.setValue(-1)
        self.days.setValue(0.0)
        self.limit.setValue(300)
        self.person.clear()
        self.search.clear()
        self._emit()

    def _emit(self) -> None:
        self.changed.emit(self.spec())


__all__ = ["SORT_TEXT", "FilterPanel"]
