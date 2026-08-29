"""数据管理：高光方案（highlight_assets）和 PRM 档案（prm_profiles）两个窗口。

- `AssetDialog`：一个视频有哪些高光 JSON 方案、每份是哪家 AI 给的、能剪几段、
  剪出过哪些成品；可以设当前 / 复制 / 导入 / 软删 / 恢复，也能「按此方案剪」（不调 AI）。
- `PrmDialog`：PRM 档案的增删改和设默认。**提示词内容始终只在文件里**，
  这儿只记名字、文件名、语言、版本。

两个窗口都只读写数据库，不删任何文件；删除一律软删，历史成品照旧查得到来源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..db import assets as db_assets
from ..db import open_db


def _cell(text: Any) -> QTableWidgetItem:
    item = QTableWidgetItem("" if text is None else str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


class AssetDialog(QDialog):
    """高光方案管理。左上挑视频，中间是方案表，下面是成品与来源。"""

    HEADERS = ("ID", "方案", "来源", "AI", "模型", "片段", "最高分", "当前", "状态")

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = parent
        self._db: Any = None
        self.setWindowTitle("高光方案管理")
        self.setMinimumSize(900, 620)
        self.setSizeGripEnabled(True)

        self.cmb_video = QComboBox()
        self.cmb_video.setMinimumWidth(420)
        self.cmb_video.currentIndexChanged.connect(lambda _=0: self.refresh_assets())
        self.chk_deleted = QPushButton("含已删除")
        self.chk_deleted.setCheckable(True)
        self.chk_deleted.clicked.connect(lambda _=False: self.refresh_assets())

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._show_detail)

        self.view_detail = QPlainTextEdit()
        self.view_detail.setReadOnly(True)
        self.view_detail.setFixedHeight(150)
        self.view_detail.setPlaceholderText("选中一份方案，这里显示它的成品和溯源信息")

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel("视频"))
        head.addWidget(self.cmb_video, 1)
        head.addWidget(self.chk_deleted)
        outer.addLayout(head)
        outer.addWidget(self.table, 1)
        outer.addWidget(self.view_detail)
        outer.addLayout(self._build_buttons())
        self.reload()

    # ------------------------------------------------------------ 界面
    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for title, tip, slot in (
                ("设为当前", "自动剪辑选「已有 JSON」时就用这份", self.on_set_current),
                ("复制", "复制一份，副本随便改，原件不动", self.on_copy),
                ("导入 JSON…", "把一份现成 JSON 登记成新方案（旧方案不动）", self.on_import),
                ("按此方案剪", "只用这份 JSON 剪成片，不调用 AI", self.on_render),
                ("软删", "只打删除标记，已有成品一个都不动", self.on_delete),
                ("恢复", "把软删的方案捞回来", self.on_restore),
                ("PRM 管理…", "管提示词档案（内容仍在文件里）", self.on_prm),
                ("刷新", "重新查库", self.reload)):
            btn = QPushButton(title)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    # ------------------------------------------------------------ 数据
    def _handle(self):
        if self._db is None:
            try:
                self._db = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "高光方案", f"数据库打不开：{exc}")
                return None
        return self._db

    def reload(self) -> None:
        """重填视频下拉（有方案的排前面），再刷方案表。"""
        db = self._handle()
        if db is None:
            return
        rows = db.all("SELECT id, file_name FROM videos ORDER BY id DESC LIMIT 200", ())
        ids = [int(r["id"]) for r in rows]
        counts = db_assets.asset_counts(db, ids)
        products = db_assets.product_counts(db, ids)
        keep = self.cmb_video.currentData()
        self.cmb_video.blockSignals(True)
        self.cmb_video.clear()
        ordered = sorted(rows, key=lambda r: (-counts.get(int(r["id"]), 0), -int(r["id"])))
        for row in ordered:
            vid = int(row["id"])
            self.cmb_video.addItem(
                f"#{vid} {row['file_name']}（方案 {counts.get(vid, 0)} / 成品 {products.get(vid, 0)}）",
                vid)
        if keep is not None:
            self.cmb_video.setCurrentIndex(max(0, self.cmb_video.findData(keep)))
        self.cmb_video.blockSignals(False)
        self.refresh_assets()

    def _video_id(self) -> int | None:
        data = self.cmb_video.currentData()
        return int(data) if data is not None else None

    def refresh_assets(self) -> None:
        db = self._handle()
        vid = self._video_id()
        self.table.setRowCount(0)
        if db is None or vid is None:
            return
        rows = db_assets.list_assets(db, vid, include_deleted=self.chk_deleted.isChecked())
        current = db_assets.current_asset(db, vid)
        current_id = int(current["id"]) if current is not None else None
        for row in rows:
            line = self.table.rowCount()
            self.table.insertRow(line)
            best = "" if row["best_score"] is None else f"{float(row['best_score']):.2f}"
            cells = (int(row["id"]), row["name"], row["source_type"], row["provider"] or "-",
                     row["model"] or "-", int(row["clip_count"] or 0), best,
                     "★" if int(row["id"]) == current_id else "",
                     "已删除" if row["deleted_at"] else "在用")
            for col, value in enumerate(cells):
                self.table.setItem(line, col, _cell(value))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._show_detail()

    def _selected_asset(self) -> int | None:
        line = self.table.currentRow()
        if line < 0:
            return None
        item = self.table.item(line, 0)
        return int(item.text()) if item is not None else None

    def _show_detail(self) -> None:
        db = self._handle()
        asset_id = self._selected_asset()
        if db is None or asset_id is None:
            self.view_detail.setPlainText("")
            return
        row = db_assets.get_asset(db, asset_id)
        if row is None:
            return
        lines = [f"方案 #{asset_id} {row['name']}（{row['source_type']}，"
                 f"版本 {row['version']}，登记于 {row['created_at']}）"]
        if row["parent_id"]:
            lines.append(f"  由方案 #{row['parent_id']} 派生（raw_json 仍是最初那份 AI 原话）")
        if row["note"]:
            lines.append(f"  备注：{row['note']}")
        products = db_assets.products_for_asset(db, asset_id)
        lines.append(f"  成品 {len(products)} 个：")
        for art in products:
            info = db_assets.artifact_lineage(db, int(art["id"])) or {}
            prm = info.get("prm")
            lines.append(f"    #{art['id']} {Path(str(art['path'])).name}"
                         + ("（文件已不在盘上）" if not art["exists_on_disk"] else "")
                         + (f"  PRM：{prm['name']}" if prm else "  PRM：-"))
        self.view_detail.setPlainText("\n".join(lines))

    # ------------------------------------------------------------ 动作
    def _need(self) -> tuple[Any, int] | None:
        db = self._handle()
        asset_id = self._selected_asset()
        if db is None:
            return None
        if asset_id is None:
            QMessageBox.information(self, "高光方案", "先在表里选一份方案")
            return None
        return db, asset_id

    def on_set_current(self) -> None:
        got = self._need()
        if got is None:
            return
        db, asset_id = got
        if db_assets.set_current_asset(db, asset_id):
            self._note(f"[高光方案] 方案 #{asset_id} 已设为当前")
        else:
            QMessageBox.information(self, "高光方案", "已删除的方案不能设为当前")
        self.refresh_assets()

    def on_copy(self) -> None:
        got = self._need()
        if got is None:
            return
        db, asset_id = got
        new_id = db_assets.copy_asset(db, asset_id)
        self._note(f"[高光方案] 方案 #{asset_id} 已复制成 #{new_id}（原件没动）")
        self.refresh_assets()

    def on_import(self) -> None:
        db = self._handle()
        vid = self._video_id()
        if db is None or vid is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选一份高光 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            import json  # noqa: PLC0415

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "高光方案", f"这份 JSON 读不进来：{exc}")
            return
        count, _best = db_assets.summarize(payload)
        if not count:
            QMessageBox.information(self, "高光方案", "这份 JSON 里抠不出可用片段，不登记")
            return
        asset_id = db_assets.create_asset(db, vid, payload, source_type="imported",
                                         note=f"从 {Path(path).name} 导入")
        self._note(f"[高光方案] 已登记方案 #{asset_id}（{count} 个高光，来自 {Path(path).name}）")
        self.reload()

    def on_render(self) -> None:
        got = self._need()
        if got is None:
            return
        _db, asset_id = got
        runner = getattr(self._window, "render_asset", None)
        if not callable(runner):
            QMessageBox.information(self, "高光方案", "这个窗口没连上主界面，剪不了")
            return
        if runner(asset_id):
            self.close()

    def on_delete(self) -> None:
        got = self._need()
        if got is None:
            return
        db, asset_id = got
        kept = len(db_assets.products_for_asset(db, asset_id))
        if QMessageBox.question(
                self, "软删方案",
                f"把方案 #{asset_id} 标成已删除？\n已经剪出来的 {kept} 个成品一个都不会动。",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        if db_assets.delete_asset(db, asset_id):
            self._note(f"[高光方案] 方案 #{asset_id} 已软删（成品未动）")
        self.refresh_assets()

    def on_restore(self) -> None:
        got = self._need()
        if got is None:
            return
        db, asset_id = got
        if db_assets.restore_asset(db, asset_id):
            self._note(f"[高光方案] 方案 #{asset_id} 已恢复")
        self.refresh_assets()

    def on_prm(self) -> None:
        dialog = PrmDialog(self.cfg, self, log=self._log)
        dialog.exec_()

    def _note(self, text: str) -> None:
        if self._log:
            self._log(text)


class PrmDialog(QDialog):
    """PRM 档案管理。只记元信息，提示词内容始终只在文件里。"""

    HEADERS = ("ID", "名字", "语言", "版本", "默认", "文件", "成品", "状态")

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._db: Any = None
        self.setWindowTitle("PRM 管理")
        self.setMinimumSize(760, 460)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._fill_form)

        self.edit_name = QLineEdit()
        self.edit_file = QLineEdit()
        self.edit_lang = QLineEdit()
        self.edit_version = QLineEdit()
        self.edit_file.setPlaceholderText("提示词文件路径，相对路径按项目根算，比如 prm/prm_en.txt")
        pick = QPushButton("选文件…")
        pick.clicked.connect(self.on_pick)

        form = QFormLayout()
        form.addRow("名字", self.edit_name)
        file_row = QHBoxLayout()
        file_row.addWidget(self.edit_file, 1)
        file_row.addWidget(pick)
        holder = QWidget()
        holder.setLayout(file_row)
        form.addRow("文件", holder)
        form.addRow("语言", self.edit_lang)
        form.addRow("版本", self.edit_version)

        outer = QVBoxLayout(self)
        outer.addWidget(self.table, 1)
        outer.addLayout(form)
        outer.addLayout(self._build_buttons())
        self.reload()

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for title, tip, slot in (
                ("新增", "按上面的名字/文件登记一份新的 PRM", self.on_add),
                ("保存修改", "把上面的内容写回选中的那一份", self.on_edit),
                ("设为默认", "GUI 发 AI 时优先用它（不再硬编码 prm_en.txt）", self.on_default),
                ("软删", "打删除标记；历史成品照旧查得到用的是它", self.on_delete),
                ("刷新", "重新查库", self.reload)):
            btn = QPushButton(title)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    def _handle(self):
        if self._db is None:
            try:
                self._db = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "PRM 管理", f"数据库打不开：{exc}")
                return None
        return self._db

    def reload(self) -> None:
        db = self._handle()
        self.table.setRowCount(0)
        if db is None:
            return
        for row in db_assets.list_prms(db, include_deleted=True):
            line = self.table.rowCount()
            self.table.insertRow(line)
            path = db_assets.prm_file(row, self.cfg.root)
            cells = (int(row["id"]), row["name"], row["language"] or "-",
                     row["version"] or "-", "★" if int(row["is_default"] or 0) else "",
                     str(row["filename"]) + ("" if path and path.is_file() else "（文件不在）"),
                     len(db_assets.products_for_prm(db, int(row["id"]))),
                     "已删除" if row["deleted_at"] else "在用")
            for col, value in enumerate(cells):
                self.table.setItem(line, col, _cell(value))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

    def _selected(self) -> int | None:
        line = self.table.currentRow()
        if line < 0:
            return None
        item = self.table.item(line, 0)
        return int(item.text()) if item is not None else None

    def _fill_form(self) -> None:
        db = self._handle()
        prm_id = self._selected()
        if db is None or prm_id is None:
            return
        row = db_assets.get_prm(db, prm_id)
        if row is None:
            return
        self.edit_name.setText(str(row["name"]))
        self.edit_file.setText(str(row["filename"]))
        self.edit_lang.setText(str(row["language"] or ""))
        self.edit_version.setText(str(row["version"] or ""))

    def on_pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选提示词文件", str(self.cfg.root),
                                              "文本 (*.txt *.md);;所有文件 (*)")
        if path:
            self.edit_file.setText(path)

    def on_add(self) -> None:
        db = self._handle()
        if db is None:
            return
        name = self.edit_name.text().strip()
        filename = self.edit_file.text().strip()
        if not name or not filename:
            QMessageBox.information(self, "PRM 管理", "名字和文件都得填")
            return
        prm_id = db_assets.create_prm(db, name, filename,
                                     language=self.edit_lang.text().strip() or None,
                                     version=self.edit_version.text().strip() or None)
        self._note(f"[PRM] 已登记 #{prm_id} {name}（{filename}）")
        self.reload()

    def on_edit(self) -> None:
        db = self._handle()
        prm_id = self._selected()
        if db is None or prm_id is None:
            QMessageBox.information(self, "PRM 管理", "先在表里选一份")
            return
        ok = db_assets.update_prm(db, prm_id,
                                  name=self.edit_name.text().strip() or None,
                                  filename=self.edit_file.text().strip() or None,
                                  language=self.edit_lang.text().strip() or None,
                                  version=self.edit_version.text().strip() or None)
        self._note(f"[PRM] #{prm_id} {'已更新' if ok else '没改动'}")
        self.reload()

    def on_default(self) -> None:
        db = self._handle()
        prm_id = self._selected()
        if db is None or prm_id is None:
            QMessageBox.information(self, "PRM 管理", "先在表里选一份")
            return
        if db_assets.set_default_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已设为默认")
        else:
            QMessageBox.information(self, "PRM 管理", "已删除的 PRM 不能设为默认")
        self.reload()

    def on_delete(self) -> None:
        db = self._handle()
        prm_id = self._selected()
        if db is None or prm_id is None:
            QMessageBox.information(self, "PRM 管理", "先在表里选一份")
            return
        kept = len(db_assets.products_for_prm(db, prm_id))
        if QMessageBox.question(
                self, "软删 PRM",
                f"把 PRM #{prm_id} 标成已删除？\n{kept} 个历史成品照旧查得到用的是它。",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        if db_assets.delete_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已软删")
        self.reload()

    def _note(self, text: str) -> None:
        if self._log:
            self._log(text)
