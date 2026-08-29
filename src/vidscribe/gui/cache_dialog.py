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
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .. import cache as cache_mod
from ..db import open_db
from ..db import repo as db_repo
from ..db.importer import refresh_from_disk

KIND_TEXT = {"video": "视频缓存", "log": "日志"}


class CacheDialog(QDialog):
    """缓存清单 + 几个删除动作。关掉再开会重新扫盘。"""

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._items: list[dict[str, Any]] = []
        self._db_handle = None
        self._db_failed = False
        self._db_videos: dict[str, Any] = {}
        self.setWindowTitle("缓存管理")
        self.resize(860, 480)

        self.lbl_summary = QLabel("正在扫描…")
        self.lbl_summary.setProperty("role", "hint")

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["名称", "类型", "视频", "大小", "多久没动", "路径"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(5, QHeaderView.Stretch)
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

        self.chk_drop_wav = QCheckBox("分析完就删预览音轨")
        self.chk_drop_wav.setChecked(bool(cfg.runtime.get("drop_preview_audio", False)))
        self.chk_drop_wav.setToolTip("开了以后 cache 里只留 json：分析一跑完就把这个视频的 "
                                    "preview_audio.wav 删掉。代价是下次看波形/听预览要重新解音轨")
        self.chk_drop_wav.toggled.connect(self.on_drop_wav_toggled)

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        btn_pick_old = QPushButton("按天数勾选")
        btn_pick_old.clicked.connect(self.select_stale)
        btn_pick_logs = QPushButton("勾选日志")
        btn_pick_logs.clicked.connect(self.select_logs)
        btn_pick_orphan = QPushButton("勾选视频没了的")
        btn_pick_orphan.setToolTip("对应视频已经不在盘上的缓存（视频删了）。视频只是放在库外面的不会被勾")
        btn_pick_orphan.clicked.connect(self.select_orphans)
        btn_drop_wav = QPushButton("只删音轨")
        btn_drop_wav.setToolTip("把所有视频的 preview_audio.wav 删掉，分析结果（json）全留着")
        btn_drop_wav.clicked.connect(self.drop_audio)
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
        picks.addWidget(btn_pick_orphan)
        picks.addStretch(1)
        picks.addWidget(self.chk_drop_wav)
        picks.addWidget(btn_refresh)

        actions = QHBoxLayout()
        actions.addWidget(btn_delete)
        actions.addWidget(btn_drop_wav)
        actions.addWidget(btn_all)
        actions.addStretch(1)
        actions.addWidget(btn_open)
        actions.addWidget(btn_close)


        self.edit_library = QLineEdit(str(cfg.data["paths"].get("video_dir", "") or ""))
        self.edit_library.setPlaceholderText("留空＝不判断；填了就按这个目录（含子目录）认视频")
        self.edit_library.setToolTip("集中管理用的视频库：缓存管理据此判断每份缓存的视频还在不在")
        self.edit_library.editingFinished.connect(self.save_library)
        btn_library = QPushButton("选目录…")
        btn_library.clicked.connect(self.browse_library)

        library = QHBoxLayout()
        library.addWidget(QLabel("视频库"))
        library.addWidget(self.edit_library, 1)
        library.addWidget(btn_library)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_summary)
        layout.addLayout(library)
        layout.addLayout(picks)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions)
        self.reload()

    # --------------------------------------------------------------- 读
    def _db(self):
        """数据库句柄。打不开就记一句，「视频」列显示「不清楚」，别让对话框崩。"""
        if self._db_handle is None and not self._db_failed:
            try:
                self._db_handle = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                self._db_failed = True
                if self._log:
                    self._log(f"[缓存] 数据库打不开，视频在不在只能显示不清楚：{exc}")
        return self._db_handle

    def _load_db_videos(self) -> None:
        """打开/刷新时跟磁盘对一次账，然后把 videos 表整张读进内存备查。

        「占多少空间」「多久没动」这两列必须来自磁盘（那是文件本身的属性），
        「视频」这一列的在不在 / 在不在库则一律来自数据库。
        """
        self._db_videos = {}
        db = self._db()
        if db is None:
            return
        try:
            refresh_from_disk(self.cfg, db)
            for row in db_repo.list_videos(db):
                self._db_videos[str(row["file_path"])] = row
        except Exception as exc:  # noqa: BLE001
            if self._log:
                self._log(f"[缓存] 对账失败，视频状态可能不准：{exc}")

    def _db_row(self, video: Any):
        """按 state.json 里记的视频路径在库里找那一行。找不到返回 None（显示「不清楚」）。"""
        if not video:
            return None
        row = self._db_videos.get(str(video))
        if row is not None:
            return row
        try:
            return self._db_videos.get(str(Path(video).resolve()))
        except OSError:
            return None

    def reload(self) -> None:
        self._loading = True
        self._load_db_videos()
        self._items = sorted(cache_mod.entries(self.cfg),
                             key=lambda it: it["bytes"], reverse=True)
        for item in self._items:
            if item["kind"] != "video":
                continue
            row = self._db_row(item.get("video"))
            if row is not None:
                item["video_exists"] = bool(row["exists_on_disk"])
                item["in_library"] = None if row["in_library"] is None else bool(row["in_library"])
        self.table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            name = QTableWidgetItem(str(item["name"]))
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(KIND_TEXT.get(item["kind"], item["kind"])))
            self.table.setItem(row, 2, QTableWidgetItem(self._library_text(item)))
            self.table.setItem(row, 3, QTableWidgetItem(cache_mod.human_size(item["bytes"])))
            self.table.setItem(row, 4, QTableWidgetItem(f"{item['age_days']:.1f} 天"))
            self.table.setItem(row, 5, QTableWidgetItem(str(item["path"])))
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self._loading = False
        self.refresh_summary()

    @staticmethod
    def _library_text(item: dict[str, Any]) -> str:
        """「视频」列：这份缓存的视频还在不在、在不在视频库里。"""
        if item["kind"] != "video":
            return "—"
        exists = item.get("video_exists")
        if exists is None:
            return "不清楚"          # state.json 没记视频路径，不好判断
        if not exists:
            return "视频没了"
        return "在库" if item.get("in_library") else "在库外"

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
        gone = [it for it in self._items if it.get("video_exists") is False]
        if gone:
            size = sum(it["bytes"] for it in gone)
            text += f"；{len(gone)} 份视频已删 / {cache_mod.human_size(size)}"
        outside = [it for it in self._items if it.get("video_exists") and it.get("in_library") is False]
        if outside:
            text += f"；{len(outside)} 份视频在库外"
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

    def select_orphans(self) -> None:
        """勾选视频已经不在盘上的那些缓存（视频删了）。视频只是放在库外面的不算。"""
        rows = [i for i, it in enumerate(self._items) if it.get("video_exists") is False]
        self._set_checked(rows)
        if not rows:
            self.lbl_summary.setText("每份缓存的视频都还在，没有要清的")

    # --------------------------------------------------------------- 视频库
    def save_library(self) -> None:
        """视频库目录写回 config.json 的 paths.video_dir，然后重扫一遍。"""
        value = self.edit_library.text().strip()
        if value == str(self.cfg.data["paths"].get("video_dir", "") or ""):
            return
        self.cfg.data["paths"]["video_dir"] = value
        self.cfg.save_patch({"paths": {"video_dir": value}})
        if self._log:
            self._log(f"[缓存] 视频库设为 {value or '（空，不判断）'}")
        self.reload()

    def browse_library(self) -> None:
        start = self.edit_library.text().strip() or str(self.cfg.path("input_dir"))
        picked = QFileDialog.getExistingDirectory(self, "选视频库目录", start)
        if not picked:
            return
        self.edit_library.setText(picked)
        self.save_library()


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

    def on_drop_wav_toggled(self, checked: bool) -> None:
        """开关写回 config.json，下次分析完就自动删这个视频的预览音轨。"""
        self.cfg.runtime["drop_preview_audio"] = bool(checked)
        self.cfg.save_patch({"runtime": {"drop_preview_audio": bool(checked)}})
        if self._log:
            self._log("[缓存] 分析完自动删预览音轨：" + ("开" if checked else "关"))

    def drop_audio(self) -> None:
        """把所有视频的 preview_audio.wav 删掉，json 全留着。"""
        wavs = sorted(cache_mod.videos_root(cache_mod.cache_dir(self.cfg))
                      .glob(f"*/{cache_mod.PREVIEW_AUDIO}"))
        if not wavs:
            QMessageBox.information(self, "缓存管理", "缓存里没有预览音轨")
            return
        size = cache_mod.human_size(sum(p.stat().st_size for p in wavs))
        ok = QMessageBox.question(
            self, "缓存管理",
            f"要删掉 {len(wavs)} 个预览音轨吗？\n共 {size}\n\n"
            f"分析结果（json）一个不动，只是下次看波形/听预览要重新解一遍音轨。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        result = cache_mod.drop_preview_audio(self.cfg)
        if self._log:
            self._log(f"[缓存] 删掉 {result['removed']} 个预览音轨，腾出 {result['freed_text']}")
        self.reload()

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
