"""缓存管理对话框：看清楚每份缓存占多少、多久没动过，然后自己挑着删。

为什么要有它：以前是开软件自动删超过 3 天的缓存，删了才告诉你。现在改成一律手动——
这里列出 cache/videos 下每个视频一份、logs 下每个日志一个，勾选删除，或者按天数批量选。

删除只走 cache.remove()/cache.cleanup()，那两个函数有"路径必须在缓存目录内"的护栏，
output/ 的分析结果、models/ 的权重、input/ 的原片都碰不到。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .. import cache as cache_mod

KIND_TEXT = {"video": "视频缓存", "log": "日志"}


class CacheDialog(QDialog):
    """缓存清单 + 几个删除动作。关掉再开会重新扫盘。"""

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._items: list[dict[str, Any]] = []
        self.setWindowTitle("缓存管理")
        self.resize(860, 480)

        self.lbl_summary = QLabel("正在扫描…")
        self.lbl_summary.setProperty("role", "hint")

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["名称", "类型", "大小", "多久没动", "路径"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        # 勾选变了就刷新"已勾选 N 项 / 多大"；填表期间不响应，否则每写一格都要重算
        self._loading = False
        self.table.itemChanged.connect(lambda *_: None if self._loading else self.refresh_summary())

        self.spin_days = QDoubleSpinBox()
        self.spin_days.setRange(0.0, 365.0)
        self.spin_days.setDecimals(1)
        self.spin_days.setSingleStep(1.0)
        self.spin_days.setValue(float(cfg.runtime.get("cache_max_age_days", 3)))
        self.spin_days.setPrefix("超过 ")
        self.spin_days.setSuffix(" 天没动")
        self.spin_days.setToolTip("配合右边「按天数勾选」用；改这里不会自动删任何东西")

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        btn_pick_old = QPushButton("按天数勾选")
        btn_pick_old.clicked.connect(self.select_stale)
        btn_pick_logs = QPushButton("勾选日志")
        btn_pick_logs.clicked.connect(self.select_logs)
        btn_delete = QPushButton("删除勾选")
        btn_delete.setProperty("role", "primary")
        btn_delete.clicked.connect(self.delete_checked)
        btn_all = QPushButton("全部清空")
        btn_all.clicked.connect(self.delete_all)
        btn_open = QPushButton("打开缓存目录")
        btn_open.clicked.connect(self.open_dir)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)

        picks = QHBoxLayout()
        picks.addWidget(self.spin_days)
        picks.addWidget(btn_pick_old)
        picks.addWidget(btn_pick_logs)
        picks.addStretch(1)
        picks.addWidget(btn_refresh)

        actions = QHBoxLayout()
        actions.addWidget(btn_delete)
        actions.addWidget(btn_all)
        actions.addStretch(1)
        actions.addWidget(btn_open)
        actions.addWidget(btn_close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_summary)
        layout.addLayout(picks)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions)
        self.reload()

    # --------------------------------------------------------------- 读
    def reload(self) -> None:
        self._loading = True
        self._items = sorted(cache_mod.entries(self.cfg),
                             key=lambda it: it["bytes"], reverse=True)
        self.table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            name = QTableWidgetItem(str(item["name"]))
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(KIND_TEXT.get(item["kind"], item["kind"])))
            self.table.setItem(row, 2, QTableWidgetItem(cache_mod.human_size(item["bytes"])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{item['age_days']:.1f} 天"))
            self.table.setItem(row, 4, QTableWidgetItem(str(item["path"])))
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self._loading = False
        self.refresh_summary()

    def refresh_summary(self) -> None:
        total = sum(it["bytes"] for it in self._items)
        checked = self.checked_items()
        text = f"共 {len(self._items)} 项 / {cache_mod.human_size(total)}"
        if checked:
            picked = sum(it["bytes"] for it in checked)
            text += f"；已勾选 {len(checked)} 项 / {cache_mod.human_size(picked)}"
        info = cache_mod.read_state(self.cfg)
        if info.get("last_cleanup"):
            text += f"；上次清理 {info['last_cleanup']}"
        text += f"｜{cache_mod.cache_dir(self.cfg)}"
        self.lbl_summary.setText(text)

    def checked_items(self) -> list[dict[str, Any]]:
        picked = []
        for row, item in enumerate(self._items):
            cell = self.table.item(row, 0)
            if cell is not None and cell.checkState() == Qt.Checked:
                picked.append(item)
        return picked

    # --------------------------------------------------------------- 勾选
    def _set_checked(self, rows: list[int]) -> None:
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            if cell is not None:
                cell.setCheckState(Qt.Checked if row in rows else Qt.Unchecked)
        self.refresh_summary()

    def select_stale(self) -> None:
        days = float(self.spin_days.value())
        rows = [i for i, it in enumerate(self._items) if it["age_days"] >= days]
        self._set_checked(rows)
        if not rows:
            self.lbl_summary.setText(f"没有超过 {days:g} 天没动过的缓存")

    def select_logs(self) -> None:
        self._set_checked([i for i, it in enumerate(self._items) if it["kind"] == "log"])

    # --------------------------------------------------------------- 删
    def _do_remove(self, items: list[dict[str, Any]], what: str) -> None:
        size = cache_mod.human_size(sum(it["bytes"] for it in items))
        ok = QMessageBox.question(
            self, "缓存管理",
            f"要删掉{what}吗？\n{len(items)} 项 / {size}\n\n"
            f"删掉只是下次分析这些视频要重跑，output/ 里的结果不受影响。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        result = cache_mod.remove(self.cfg, [it["path"] for it in items])
        # 记一笔清理时间，界面和 cache 命令都会显示
        state = cache_mod.read_state(self.cfg)
        state["last_cleanup_ts"] = time.time()
        state["last_cleanup"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["last_removed"] = result["removed"]
        state["last_freed_bytes"] = result["freed_bytes"]
        cache_mod.write_state(self.cfg, state)

        line = (f"[缓存] 删掉 {len(result['removed'])} 项，腾出 "
                f"{cache_mod.human_size(result['freed_bytes'])}")
        if result["failed"]:
            line += f"；{len(result['failed'])} 项失败（可能正被占用）：{result['failed'][0]}"
        if self._log:
            self._log(line)
        self.reload()
        if result["failed"]:
            QMessageBox.warning(self, "缓存管理",
                                "有几项没删掉，可能正被占用（比如预览音轨还在播）：\n"
                                + "\n".join(result["failed"][:8]))

    def delete_checked(self) -> None:
        items = self.checked_items()
        if not items:
            QMessageBox.information(self, "缓存管理", "先勾几项")
            return
        self._do_remove(items, "勾选的这些缓存")

    def delete_all(self) -> None:
        if not self._items:
            QMessageBox.information(self, "缓存管理", "缓存是空的")
            return
        self._do_remove(list(self._items), "全部缓存（含日志）")

    def open_dir(self) -> None:
        path = Path(cache_mod.cache_dir(self.cfg))
        path.mkdir(parents=True, exist_ok=True)
        from PyQt5.QtCore import QUrl  # noqa: PLC0415
        from PyQt5.QtGui import QDesktopServices  # noqa: PLC0415

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
