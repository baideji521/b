"""视频资产中心：我的视频生产资产库。

一个非模态窗口，两页：

    [视频资产]  视频（第一层）→ 高光 JSON（第二层）→ 成品与血缘（第三层）
    [PRM 管理]  PRM 是剪辑规则模板：新建 / 编辑 / 复制 / 恢复 / 删除 / 设默认 / 改内容

界面上的四个正式名词就是：**视频 / 高光 JSON / 成品 / PRM**。JSON 一律叫
「高光 JSON #3」，方案名只当辅助信息；主键、artifact、analysis_id 这些技术字段
只在血缘树里出现，平时不给用户看。

设计上要回答的七个问题（都不用再开第二个窗口）：

  ① 我有哪些视频      → 左侧视频列表（搜索 / 状态 / AI / 排序 / 表头排序 / 条数）
  ② 有没有分析        → 视频行的「分析」列 + 右侧当前视频工作区
  ③ 有几个高光 JSON    → 视频行的「JSON」列 + 右侧 JSON 表
  ④ 每个 JSON 谁生成的 → JSON 表的「AI / 模型」列（库里没记就是「—」，绝不猜）
  ⑤ JSON 选了哪些区间  → 选中 JSON 后右边**直接**列出每一段（原文默认收起）
  ⑥ 用哪个 PRM 剪过什么 → 成品表的「来源 JSON / PRM / 实际区间」
  ⑦ 成品怎么来的      → 血缘树：视频 → 分析 → JSON → AI → Engine → PRM → 成品

区间永远三层摆开：AI 原始 → Clip Engine 修正 → 实际渲染，一致标 ✓、不一致标 ⚠。
数量、区间、血缘全部来自数据库（`assets.center_rows` 一次聚合），GUI 不扫磁盘、
不逐个视频发 SQL。删除一律软删，任何文件都不动。
"""


from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,

    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..db import assets as db_assets
from ..db import open_db
from ..db import repo as db_repo

STATUS_CHOICES = (("全部", "all"), ("已分析", "analysed"), ("未分析", "not_analysed"))
# 「有没有 JSON」「有没有成品」各自一个下拉：三个条件能一起生效（场景 A 一步到位）
JSON_CHOICES = (("全部", "any"), ("有 JSON", "has"), ("无 JSON", "none"))
PRODUCT_CHOICES = (("全部", "any"), ("有成品", "has"), ("无成品", "none"))
ORDER_CHOICES = (("最近更新", "recent"), ("最近处理", "processed"), ("视频名称", "name"),
                 ("视频时长", "duration"), ("JSON 数量", "json"), ("高光数量", "highlight"),
                 ("成品数量", "product"))
# 点表头 = 换「排序」下拉，排序永远只有 center_rows 这一处真源（列号 → 排序键）
ORDER_BY_COLUMN = {1: "name", 2: "duration", 4: "json", 5: "highlight",
                   6: "product", 8: "recent"}
COLUMN_BY_ORDER = {key: col for col, key in ORDER_BY_COLUMN.items()}
# 主列表条数：不写死 200，几千个视频也能一次看完（排序筛选都在 SQL 之后做）
PAGE_CHOICES = (("200 条", 200), ("500 条", 500), ("1000 条", 1000),
                ("5000 条", 5000), ("全部", 1_000_000))



# ------------------------------------------------------------------ 小工具
def _cell(text: Any, center: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem("" if text is None else str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if center:
        item.setTextAlignment(Qt.AlignCenter)
    return item


def _num(value: Any) -> QTableWidgetItem:
    """数字格：按数值排序，不按字符串（表头点一下就得排对）。"""
    item = QTableWidgetItem()
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    item.setTextAlignment(Qt.AlignCenter)
    try:
        number = float(value)
    except (TypeError, ValueError):
        item.setData(Qt.DisplayRole, 0)
        item.setText("—")
        return item
    item.setData(Qt.DisplayRole, int(number) if number == int(number) else number)
    return item


def _seconds(value: Any) -> str:
    try:
        total = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{total:.0f}s" if total > 0 else "—"


class _SortItem(QTableWidgetItem):
    """显示一句人话、排序按数字。

    「时长」这一列显示的是 `285s`，但点表头必须按秒数排，不能按字符串排
    （不然 `9s` 会排到 `285s` 后面，空值也会乱窜）。
    """

    def __init__(self, text: str, key: float):
        super().__init__(str(text))
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)
        self.setTextAlignment(Qt.AlignCenter)
        self._key = float(key)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _SortItem):
            return self._key < other._key
        return super().__lt__(other)


def _duration_cell(value: Any) -> QTableWidgetItem:
    """时长格：显示 `285s`，排序按秒；库里没有时长就是「—」，排最前面。"""
    try:
        total = float(value)
    except (TypeError, ValueError):
        total = -1.0
    return _SortItem(_seconds(value), total)



def _score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _ai_label(row: Any) -> str:
    """把 provider / model 拼成一句人话；两个都空就是「—」（绝不猜 AI 名字）。"""
    provider = str((row["provider"] if row is not None else "") or "").strip()
    model = str((row["model"] if row is not None else "") or "").strip()
    if provider and model:
        return f"{provider} / {model}"
    return provider or model or "—"


def _json_title(row: Any) -> str:
    """界面上 JSON 的正式名字：`高光 JSON #3`。方案名只作为辅助信息另外显示。"""
    return f"高光 JSON #{int(row['id'])}"


def _id_item(text: str, ident: int, *, key: float | None = None,
             bold: bool = False) -> QTableWidgetItem:
    """带隐藏 id 的格子：界面上只显示人话，id 藏在 UserRole 里（用户不该看见主键）。"""
    item = _SortItem(text, key if key is not None else float(ident))
    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    item.setData(Qt.UserRole, int(ident))
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    return item


def _row_id(table: QTableWidget, line: int, col: int = 0) -> int | None:
    """读某一行藏着的 id。"""
    if line < 0:
        return None
    item = table.item(line, col)
    if item is None:
        return None
    value = item.data(Qt.UserRole)
    return None if value is None else int(value)



def _short_time(value: Any) -> str:
    text = str(value or "")
    return text.replace("T", " ")[:16] if text else "—"


def _span(start: Any, end: Any) -> str:
    return f"{_score(start)} → {_score(end)}"


def _reveal(widget: QWidget, path: Path) -> None:
    """在文件管理器里定位这个文件；不在盘上就说清楚。"""
    if not path.exists():
        QMessageBox.information(widget, "资产中心", f"文件已经不在盘上：\n{path}")
        return
    if os.name == "nt":
        os.startfile(str(path.parent))  # noqa: S606 - 打开用户自己的成品目录
        return
    QMessageBox.information(widget, "资产中心", str(path))


def _highlight_rows(payload: Any) -> list[dict[str, Any]]:
    """从 JSON 里抠出高光区间清单（复用 repo 的解析，不另写一套）。"""
    try:
        return db_repo.clips_from_payload(payload) or []
    except Exception:  # noqa: BLE001
        return []


def _title(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


def _primary(button: QPushButton) -> QPushButton:
    """主动作：加粗 + 稍高，和「删除/复制」这些次要动作分开视觉权重。"""
    font = button.font()
    font.setBold(True)
    button.setFont(font)
    button.setMinimumHeight(30)
    return button


def _plain_table(headers: tuple[str, ...], *, stretch: int | None = None) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setAlternatingRowColors(True)
    if stretch is not None:
        table.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.Stretch)
    return table


# ================================================================ JSON 详情
class JsonPanel(QWidget):
    """选中的高光 JSON：来源、每一段区间、Engine 预演、原文。**内联，不弹窗。**

    编辑绝不覆盖原件：保存走 `assets.edit_asset(..., in_place=False)`，生成一条新方案，
    `raw_json` 永远是最初那份 AI 原话。
    """

    HEADERS = ("#", "起点", "终点", "时长", "评分", "类型", "评价")
    LAYER_HEADERS = ("层级", "起点", "终点", "时长", "结论")

    saved = pyqtSignal(int)          # 另存成新方案（带出新 id）

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._db: Any = None
        self.asset_id: int | None = None

        self.lbl_head = _title("还没选 JSON")
        self.lbl_meta = QLabel("左边选一份 JSON，这里显示它的来源和每一段高光区间")
        self.lbl_meta.setWordWrap(True)

        self.table = _plain_table(self.HEADERS, stretch=6)
        self.table.setMinimumHeight(140)

        # 三层区间：固定三行网格（层级名定宽、数字等宽右对齐、最右一格结论徽标）
        self.tbl_layers = _plain_table(self.LAYER_HEADERS, stretch=4)
        self.tbl_layers.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_layers.setFixedHeight(3 * 24 + 26)
        self.tbl_layers.verticalHeader().setDefaultSectionSize(24)
        self.tbl_layers.setToolTip("AI 原始区间 → Clip Engine 修正 → 实际渲染区间"
                                   "（规则来源就是真剪时用的那一套）")
        self.lbl_engine = QLabel("—")          # 结论 + 原因（Engine 为什么改了）
        self.lbl_engine.setWordWrap(True)
        self.lbl_engine.setFrameShape(QFrame.StyledPanel)


        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setMinimumHeight(120)

        # 平时一个动作按钮都不显示：编辑从「更多 ▾ → 编辑」进来，进了编辑态才出现这两个
        self.btn_save = QPushButton("保存为新 JSON")
        self.btn_save.setToolTip("存成一份新的高光 JSON，原件一个字不动")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_cancel = QPushButton("取消编辑")
        self.btn_cancel.clicked.connect(self.on_cancel_edit)
        self.lbl_editing = QLabel("正在编辑：保存会新建一份高光 JSON，原件不动")

        self.lbl_spans = QLabel("区间：AI 原始 → Clip Engine 修正 → 实际渲染")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 4, 4)
        outer.addWidget(self.lbl_head)
        outer.addWidget(self.lbl_meta)
        outer.addWidget(_title("高光区间（AI 原始）"))
        outer.addWidget(self.table, 2)
        outer.addWidget(self.lbl_spans)
        outer.addWidget(self.tbl_layers)
        outer.addWidget(self.lbl_engine)
        outer.addWidget(self.view, 3)
        row = QHBoxLayout()
        row.addWidget(self.lbl_editing, 1)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_cancel)
        outer.addLayout(row)
        self.view.setVisible(False)          # 原文默认收起，不占主界面
        self._set_editing(False)

    # ------------------------------------------------------------ 显示状态
    def _set_editing(self, editing: bool) -> None:
        """编辑态才显示 [保存为新 JSON][取消编辑]，平时这一行整条隐藏。"""
        for widget in (self.btn_save, self.btn_cancel, self.lbl_editing):
            widget.setVisible(editing)
        self.view.setReadOnly(not editing)

    def raw_visible(self) -> bool:
        return self.view.isVisibleTo(self)

    def set_raw_visible(self, shown: bool) -> None:
        """原文显示 / 收起（入口在「更多 ▾ → 显示 JSON 原文」）。"""
        self.view.setVisible(bool(shown))



    # ------------------------------------------------------------ 数据
    def _handle(self):
        if self._db is None:
            try:
                self._db = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "高光 JSON", f"数据库打不开：{exc}")
                return None
        return self._db

    def clear(self, message: str | None = None) -> None:
        self.asset_id = None
        self.lbl_head.setText(message or "还没选高光 JSON")
        self.lbl_meta.setText("左边选一份高光 JSON，这里显示它是谁生成的、选了哪些区间")
        self.table.setRowCount(0)
        self.tbl_layers.setRowCount(0)
        self.lbl_engine.setText("—")
        self.view.setPlainText("")
        self._set_editing(False)

    def show_asset(self, asset_id: int | None, *, row: Any = None,
                   products: Any = None) -> None:
        """显示一份高光 JSON。`None` 就清空。

        `row` / `products` 是给「列表刷新」用的：那边手上已经有这一行和成品清单了，
        传进来就少两次查询（不传就自己查，单点调用照样能用）。
        """
        if asset_id is None:
            self.clear()
            return
        db = self._handle()
        if db is None:
            return
        if row is None or int(row["id"]) != int(asset_id):
            row = db_assets.get_asset(db, int(asset_id))
        if row is None:
            self.clear(f"高光 JSON #{asset_id} 已经不在库里")
            return
        self.asset_id = int(asset_id)
        self._set_editing(False)

        self.lbl_head.setText(_json_title(row)
                              + ("　（已删除）" if row["deleted_at"] else ""))
        if products is None:
            products = db_assets.products_for_asset(db, int(row["id"]))
        origin = ("导入" if str(row["source_type"]) == "imported"
                  else ("复制" if row["parent_id"] else "AI"))
        meta = [f"AI：{_ai_label(row)}",
                f"生成时间：{_short_time(row['created_at'])}",
                f"高光数量：{int(row['clip_count'] or 0)}",
                f"最高评分：{_score(row['best_score'])}",
                f"{'★ 当前 JSON' if int(row['is_current'] or 0) else '○ 历史 JSON'}",
                f"名称：{row['name']}",
                f"来源：{origin}"
                + (f"（复制自 高光 JSON #{row['parent_id']}）" if row["parent_id"] else ""),
                f"已剪出成品：{len(products)} 个"]
        if row["note"]:
            meta.append(f"备注：{row['note']}")
        self.lbl_meta.setText("　｜　".join(meta[:5]) + "\n" + "　｜　".join(meta[5:]))


        payload = db_assets.loads(row["current_json"])
        clips = _highlight_rows(payload)
        self.table.setRowCount(0)
        for index, clip in enumerate(clips, start=1):
            line = self.table.rowCount()
            self.table.insertRow(line)
            start, end = clip.get("start"), clip.get("end")
            span = "—"
            try:
                span = f"{float(end) - float(start):.2f}s"
            except (TypeError, ValueError):
                pass
            self.table.setItem(line, 0, _num(index))
            self.table.setItem(line, 1, _cell(_score(start), center=True))
            self.table.setItem(line, 2, _cell(_score(end), center=True))
            self.table.setItem(line, 3, _cell(span, center=True))
            self.table.setItem(line, 4, _cell(_score(clip.get("score")), center=True))
            self.table.setItem(line, 5, _cell(clip.get("type") or "—", center=True))
            self.table.setItem(line, 6, _cell(str(clip.get("evaluation") or "")[:80]))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)

        self.lbl_engine.setText(self._layers_text(db, int(row["id"]), products, row))
        text = row["current_json"]
        if payload is not None:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.view.setPlainText(str(text))

    def _layer_rows(self, db: Any, asset_id: int, products: Any = (),
                    row: Any = None) -> tuple[list[tuple[str, ...]], list[str]]:
        """三层区间的网格数据 + 结论行。**计算全在 `assets` 里，这里只排版。**

        实际渲染那一层只有剪过才有：取这份 JSON 最新的成品，读它的 `clips`。
        差异不藏：一致 ✓、不一致 ⚠、Engine 改过就写原因。
        """
        product = list(products)[-1] if products else None
        spans = db_assets.asset_layers(       # 三层一次算完，不再 spans + lineage 各查一遍
            db, asset_id, row=row,
            artifact_id=None if product is None else int(product["id"]))
        if not spans["engine"]:
            return [], ["算不出 Engine 区间（这个视频还没有逐词时间戳，"
                        "或 JSON 里没有可用片段）"]
        actual: list[dict[str, Any]] = spans["actual"]
        grid: list[tuple[str, ...]] = []
        notes: list[str] = []
        total = len(spans["engine"])
        for index, plan in enumerate(spans["engine"], start=1):
            tag = "" if total == 1 else f"第 {index} 段 "
            same = (abs(float(plan["ai_start"]) - float(plan["start"])) < 0.01
                    and abs(float(plan["ai_end"]) - float(plan["end"])) < 0.01)
            grid.append((f"{tag}AI 原始", _score(plan["ai_start"]), _score(plan["ai_end"]),
                         f"{_score(float(plan['ai_end']) - float(plan['ai_start']))}s", "—"))
            grid.append((f"{tag}Clip Engine", _score(plan["start"]), _score(plan["end"]),
                         f"{_score(plan['duration'])}s",
                         "⚠ ≤15s" if plan["capped"] else ("✓ 未调整" if same else "已修正")))
            for note in plan["notes"]:
                notes.append(f"第 {index} 段 Engine 原因：{note}")
            if index <= len(actual):
                clip = actual[index - 1]
                hit = (abs(float(plan["start"]) - float(clip["start"] or 0)) < 0.01
                       and abs(float(plan["end"]) - float(clip["end"] or 0)) < 0.01)
                grid.append((f"{tag}实际渲染", _score(clip["start"]), _score(clip["end"]),
                             f"{_score(clip['duration'])}s", "✓ 一致" if hit else "⚠ 不一致"))
            else:
                grid.append((f"{tag}实际渲染", "—", "—", "—", "还没剪"))
        if not actual:
            notes.append("实际渲染：还没剪过，点「直接剪辑」就按上面的 Engine 区间出成品")
            return grid, notes
        same_all = (len(actual) == len(spans["engine"])
                    and all(abs(float(p["start"]) - float(c["start"] or 0)) < 0.01
                            and abs(float(p["end"]) - float(c["end"] or 0)) < 0.01
                            for p, c in zip(spans["engine"], actual)))
        notes.append("✓ Engine 与实际渲染一致" if same_all
                     else "⚠ Engine 与实际渲染不一致（加减秒之后剪的，或分析数据后来变过）")
        return grid, notes

    def _layers_text(self, db: Any, asset_id: int, products: Any = (),
                     row: Any = None) -> str:
        """填三层区间网格，返回结论 / 原因那几行文字（给下面的结论条用）。"""
        grid, notes = self._layer_rows(db, asset_id, products, row)
        self.tbl_layers.setRowCount(0)
        for cells in grid:
            line = self.tbl_layers.rowCount()
            self.tbl_layers.insertRow(line)
            self.tbl_layers.setItem(line, 0, _cell(cells[0]))
            for column in (1, 2, 3):
                item = _cell(cells[column])
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_layers.setItem(line, column, item)
            self.tbl_layers.setItem(line, 4, _cell(cells[4], center=True))
        self.tbl_layers.resizeColumnsToContents()
        self.tbl_layers.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        return "\n".join(notes)



    # ------------------------------------------------------------ 动作
    def on_edit(self) -> None:
        """进入编辑态（入口只有「更多 ▾ → 编辑」）：原文展开、出现保存 / 取消。"""
        if self.asset_id is None:
            return
        self.set_raw_visible(True)
        self._set_editing(True)
        self.view.setFocus()

    def on_cancel_edit(self) -> None:
        """放弃这次编辑：原文回到库里那份，什么都没改。"""
        self._set_editing(False)
        if self.asset_id is not None:
            self.show_asset(self.asset_id)

    def on_save(self) -> None:
        if self.asset_id is None:
            return
        try:
            payload = json.loads(self.view.toPlainText())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "高光 JSON", f"这不是合法 JSON，没保存：{exc}")
            return
        count, _best = db_assets.summarize(payload)
        if not count and QMessageBox.question(
                self, "高光 JSON", "改完以后抠不出可用片段，这份 JSON 剪不了。还要存吗？",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        db = self._handle()
        if db is None:
            return
        new_id = db_assets.edit_asset(db, self.asset_id, payload, in_place=False)
        if new_id is None:
            QMessageBox.warning(self, "高光 JSON", "存不进去（这份 JSON 可能已删除）")
            return
        if self._log:
            self._log(f"[高光 JSON] 编辑 #{self.asset_id} 后另存为 高光 JSON #{new_id}"
                      "（原件没动）")
        QMessageBox.information(self, "高光 JSON",
                                f"已另存为 高光 JSON #{new_id}，"
                                f"原来的 高光 JSON #{self.asset_id} 一个字没改。")

        self._set_editing(False)
        self.saved.emit(int(new_id))

    def on_copy_text(self) -> None:
        QApplication.clipboard().setText(self.view.toPlainText())
        if self._log and self.asset_id is not None:
            self._log(f"[高光 JSON] JSON #{self.asset_id} 原文已复制到剪贴板")


# ================================================================ 按高光 JSON 剪辑
class RenderDialog(QDialog):
    """选 PRM，然后**只用这份 JSON 出成品**——这条路一次 AI 都不调。

    资产中心本身是非模态窗口，这里是唯一一层确认对话框，不再有窗口套娃。
    """

    def __init__(self, cfg: Any, asset_row: Any, window: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = window
        self.asset_row = asset_row
        self.setWindowTitle("按这份 JSON 剪辑")
        self.setMinimumWidth(520)

        self.cmb_prm = QComboBox()
        self._fill_prms()
        flow = QLabel("高光 JSON\n  ↓\nClip Engine（修正区间）\n  ↓\n渲染\n  ↓\n成品 MP4")
        flow.setFrameShape(QFrame.StyledPanel)
        note = QLabel("此操作不会调用 AI。文件名带 JSON 名和 PRM 名，"
                      "所以同一份 JSON 换 PRM 再剪不会互相覆盖。")
        note.setWordWrap(True)

        info = QFormLayout()
        info.addRow("高光 JSON", QLabel(f"{_json_title(asset_row)}　{asset_row['name']}"
                                        f"（{int(asset_row['clip_count'] or 0)} 个高光，"
                                        f"最高分 {_score(asset_row['best_score'])}）"))

        info.addRow("来源 AI", QLabel(_ai_label(asset_row)))
        info.addRow("PRM", self.cmb_prm)
        info.addRow("流程", flow)
        info.addRow("说明", note)

        start = _primary(QPushButton("开始剪辑"))
        start.setToolTip("直接把这份 JSON 交给剪辑引擎和渲染，不会再问 AI")
        start.clicked.connect(self.on_start)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(start)
        row.addWidget(cancel)

        outer = QVBoxLayout(self)
        outer.addLayout(info)
        outer.addLayout(row)

    def _fill_prms(self) -> None:
        db = open_db(self.cfg)
        try:
            rows = db_assets.list_prms(db)
            default = db_assets.default_prm(db)
        finally:
            db.close()
        if not rows:
            self.cmb_prm.addItem("（库里还没有 PRM 档案，用配置里的提示词）", 0)
            return
        for row in rows:
            mark = "（默认）" if int(row["is_default"] or 0) else ""
            self.cmb_prm.addItem(f"{row['name']}{mark}", int(row["id"]))
        if default is not None:
            self.cmb_prm.setCurrentIndex(max(0, self.cmb_prm.findData(int(default["id"]))))

    def on_start(self) -> None:
        runner = getattr(self._window, "render_asset", None)
        if not callable(runner):
            QMessageBox.information(self, "直接剪辑", "没连上主界面，剪不了")
            return
        prm_id = int(self.cmb_prm.currentData() or 0)
        if runner(int(self.asset_row["id"]), prm_id or None):
            self.accept()


# ================================================================ PRM 管理页
class PrmPanel(QWidget):
    """PRM 是独立资产：新建 / 编辑 / 复制 / 恢复 / 删除 / 设默认，内容也能在这里改。

    库里只存档案（名字 / 文件 / 语言 / 版本），提示词正文始终在文件里；
    改内容是显式动作，点「保存内容」才会写文件。
    """

    HEADERS = ("ID", "名称", "文件", "语言", "版本", "默认", "剪出成品", "更新时间", "状态")

    changed = pyqtSignal()

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._db: Any = None

        self.table = _plain_table(self.HEADERS, stretch=2)
        self.table.setColumnHidden(0, True)          # id 留着取值，界面上不露主键
        self.table.itemSelectionChanged.connect(self.on_selected)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔎 搜索 PRM 名称 / 文件")
        self.edit_search.setClearButtonEnabled(True)
        self.edit_search.textChanged.connect(lambda _="": self.reload())


        self.edit_name = QLineEdit()
        self.edit_file = QLineEdit()
        self.edit_file.setPlaceholderText("提示词文件路径，相对路径按项目根算，例如 prm/prm_en.txt")
        self.edit_lang = QLineEdit()
        self.edit_version = QLineEdit()
        pick = QPushButton("选文件…")
        pick.clicked.connect(self.on_pick)
        file_row = QHBoxLayout()
        file_row.addWidget(self.edit_file, 1)
        file_row.addWidget(pick)
        holder = QWidget()
        holder.setLayout(file_row)

        self.lbl_times = QLabel("—")
        form = QFormLayout()
        form.addRow("名称", self.edit_name)
        form.addRow("文件", holder)
        form.addRow("语言", self.edit_lang)
        form.addRow("版本", self.edit_version)
        form.addRow("时间", self.lbl_times)

        self.view_text = QPlainTextEdit()
        self.view_text.setPlaceholderText("选一份 PRM，这里显示提示词正文（可以改，点「保存内容」才写文件）")
        self.btn_load = QPushButton("重新载入内容")
        self.btn_load.clicked.connect(self.on_load_text)
        self.btn_write = QPushButton("保存内容")
        self.btn_write.setToolTip("把上面的正文写回提示词文件（会先确认）")
        self.btn_write.clicked.connect(self.on_write_text)

        self.chk_all = QCheckBox("含已删除")
        self.chk_all.stateChanged.connect(lambda _=0: self.reload())

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(_title("PRM 管理"))
        head.addWidget(QLabel("PRM 是剪辑规则（发给 AI 的提示词），和高光 JSON 是两回事"), 1)
        head.addWidget(self.chk_all)
        outer.addLayout(head)
        outer.addWidget(self.edit_search)


        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(self.table, 1)
        left_lay.addLayout(self._buttons())
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addLayout(form)
        right_lay.addWidget(QLabel("提示词正文"))
        right_lay.addWidget(self.view_text, 1)
        text_row = QHBoxLayout()
        text_row.addWidget(self.btn_load)
        text_row.addWidget(self.btn_write)
        text_row.addStretch(1)
        right_lay.addLayout(text_row)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)
        self.reload()

    def _buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(_primary(self._button("新建", "按右边填的名称 / 文件登记一份新 PRM",
                                           self.on_add)))
        for title, tip, slot in (
                ("保存档案", "把右边的名称 / 文件 / 语言 / 版本写回选中的那一份", self.on_edit),
                ("复制", "复制一份档案（默认指同一个文件），原件不动", self.on_copy),
                ("设为默认", "没指定 PRM 时就用它", self.on_default),
                ("恢复", "把软删的捞回来", self.on_restore),
                ("删除", "软删；历史成品照旧查得到用的是它", self.on_delete),
                ("刷新", "重新查库", self.reload)):
            row.addWidget(self._button(title, tip, slot))
        row.addStretch(1)
        return row

    @staticmethod
    def _button(title: str, tip: str, slot) -> QPushButton:
        btn = QPushButton(title)
        btn.setToolTip(tip)
        btn.clicked.connect(slot)
        return btn

    def _handle(self):
        if self._db is None:
            try:
                self._db = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "PRM", f"数据库打不开：{exc}")
                return None
        return self._db

    def reload(self) -> None:
        db = self._handle()
        keep = self.selected()
        self.table.setRowCount(0)
        if db is None:
            return
        for row in db_assets.list_prms(db, include_deleted=self.chk_all.isChecked()):
            key = self.edit_search.text().strip().lower()
            if key and key not in f"{row['name']} {row['filename']}".lower():
                continue
            path = db_assets.prm_file(row, self.cfg.root)
            missing = "" if path is not None and path.is_file() else "（文件不在）"
            line = self.table.rowCount()
            self.table.insertRow(line)
            self.table.setItem(line, 0, _num(int(row["id"])))

            self.table.setItem(line, 1, _cell(row["name"]))
            self.table.setItem(line, 2, _cell(f"{row['filename']}{missing}"))
            self.table.setItem(line, 3, _cell(row["language"] or "—", center=True))
            self.table.setItem(line, 4, _cell(row["version"] or "—", center=True))
            self.table.setItem(line, 5, _cell("★" if int(row["is_default"] or 0) else "",
                                              center=True))
            self.table.setItem(line, 6, _num(len(db_assets.products_for_prm(db,
                                                                           int(row["id"])))))
            self.table.setItem(line, 7, _cell(_short_time(row["updated_at"]), center=True))
            self.table.setItem(line, 8, _cell("已删除" if row["deleted_at"] else "在用",
                                              center=True))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.select(keep)

    def select(self, prm_id: int | None) -> None:
        if not self.table.rowCount():
            return
        target = 0
        if prm_id is not None:
            for line in range(self.table.rowCount()):
                item = self.table.item(line, 0)
                if item is not None and int(item.text() or 0) == int(prm_id):
                    target = line
                    break
        self.table.selectRow(target)

    def selected(self) -> int | None:
        line = self.table.currentRow()
        if line < 0:
            return None
        item = self.table.item(line, 0)
        return int(item.text()) if item is not None and item.text() else None

    def on_selected(self) -> None:
        db = self._handle()
        prm_id = self.selected()
        if db is None or prm_id is None:
            return
        row = db_assets.get_prm(db, prm_id)
        if row is None:
            return
        self.edit_name.setText(str(row["name"]))
        self.edit_file.setText(str(row["filename"]))
        self.edit_lang.setText(str(row["language"] or ""))
        self.edit_version.setText(str(row["version"] or ""))
        self.lbl_times.setText(f"创建 {_short_time(row['created_at'])}"
                               f"　修改 {_short_time(row['updated_at'])}")
        self.on_load_text()

    def _note(self, text: str) -> None:
        if self._log:
            self._log(text)
        self.changed.emit()

    def _need(self) -> tuple[Any, int] | None:
        db = self._handle()
        prm_id = self.selected()
        if db is None:
            return None
        if prm_id is None:
            QMessageBox.information(self, "PRM", "先在表里选一份")
            return None
        return db, prm_id

    def _path(self) -> Path | None:
        db = self._handle()
        prm_id = self.selected()
        if db is None or prm_id is None:
            return None
        row = db_assets.get_prm(db, prm_id)
        return None if row is None else db_assets.prm_file(row, self.cfg.root)

    def on_pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选提示词文件", str(self.cfg.root),
                                              "文本 (*.txt *.md);;所有文件 (*)")
        if path:
            self.edit_file.setText(path)

    def on_load_text(self) -> None:
        path = self._path()
        if path is None or not path.is_file():
            self.view_text.setPlainText("")
            self.view_text.setPlaceholderText("这份 PRM 的文件现在不在盘上")
            return
        try:
            self.view_text.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            self.view_text.setPlainText(f"读不出来：{exc}")

    def on_write_text(self) -> None:
        path = self._path()
        if path is None:
            QMessageBox.information(self, "PRM", "先选一份 PRM")
            return
        if QMessageBox.question(self, "保存 PRM 内容",
                                f"把正文写回这个文件？\n{path}",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.view_text.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "PRM", f"写不进去：{exc}")
            return
        self._note(f"[PRM] 已把正文写回 {path}")

    def on_add(self) -> None:
        db = self._handle()
        if db is None:
            return
        name = self.edit_name.text().strip()
        filename = self.edit_file.text().strip()
        if not name or not filename:
            QMessageBox.information(self, "PRM", "名称和文件都得填")
            return
        prm_id = db_assets.create_prm(db, name, filename,
                                      language=self.edit_lang.text().strip() or None,
                                      version=self.edit_version.text().strip() or None)
        self._note(f"[PRM] 已登记 #{prm_id} {name}（{filename}）")
        self.reload()
        self.select(prm_id)

    def on_edit(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        ok = db_assets.update_prm(db, prm_id, name=self.edit_name.text().strip() or None,
                                  filename=self.edit_file.text().strip() or None,
                                  language=self.edit_lang.text().strip() or None,
                                  version=self.edit_version.text().strip() or None)
        self._note(f"[PRM] #{prm_id} {'已更新' if ok else '没改动'}")
        self.reload()

    def on_copy(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        new_id = db_assets.copy_prm(db, prm_id)
        self._note(f"[PRM] #{prm_id} 已复制成 #{new_id}（原件没动）")
        self.reload()
        self.select(new_id)

    def on_default(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        if db_assets.set_default_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已设为默认")
        else:
            QMessageBox.information(self, "PRM", "已删除的 PRM 不能设为默认")
        self.reload()

    def on_delete(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        kept = len(db_assets.products_for_prm(db, prm_id))
        if QMessageBox.question(
                self, "删除 PRM",
                f"把 PRM #{prm_id} 标成已删除？\n{kept} 个历史成品照旧查得到用的是它。",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        if db_assets.delete_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已软删")
        self.reload()

    def on_restore(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        if db_assets.restore_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已恢复")
        else:
            QMessageBox.information(self, "PRM", "这一份本来就没删")
        self.reload()


# ================================================================ 视频资产页
class VideoAssetsPage(QWidget):
    """左边视频列表，右边这个视频的高光 JSON 和成品血缘。"""

    VIDEO_HEADERS = ("ID", "视频", "时长", "分析", "JSON", "高光", "成品", "AI / 模型", "更新时间")
    ASSET_HEADERS = ("当前", "高光 JSON", "AI / 模型", "高光数", "最高分", "成品",
                     "生成时间", "状态")
    PRODUCT_HEADERS = ("ID", "成品", "来源 JSON", "PRM", "时长", "实际区间", "生成时间", "状态")


    changed = pyqtSignal()

    def __init__(self, cfg: Any, window: Any = None, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = window
        self._db: Any = None
        self._rows: list[dict[str, Any]] = []
        self._asset_rows: list[Any] = []          # 当前视频的 JSON 行（刷新时手上就有）
        self._product_rows: list[dict[str, Any]] | None = None   # 当前视频的成品全景
        self._lineage_for: int | None = None      # 血缘树现在画的是哪个成品

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(split, 1)
        self.reload()

    # ------------------------------------------------------------ 左：视频库
    def _build_left(self) -> QWidget:
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔎 搜索视频 / 文件名 / ID")
        self.edit_search.setClearButtonEnabled(True)
        self.edit_search.returnPressed.connect(self.reload)
        self.edit_search.textChanged.connect(self._search_changed)

        self.cmb_status = QComboBox()
        self.cmb_status.setToolTip("有没有做过视觉 / 语音分析")
        for title, key in STATUS_CHOICES:
            self.cmb_status.addItem(title, key)
        self.cmb_json = QComboBox()
        self.cmb_json.setToolTip("有没有高光 JSON（和「成品」是两个独立条件，可以一起用）")
        for title, key in JSON_CHOICES:
            self.cmb_json.addItem(title, key)
        self.cmb_product = QComboBox()
        self.cmb_product.setToolTip("有没有剪出成品；「有 JSON + 无成品」= 还没剪的那些")
        for title, key in PRODUCT_CHOICES:
            self.cmb_product.addItem(title, key)
        self.cmb_ai = QComboBox()
        self.cmb_ai.setToolTip("按生成 JSON 的 AI 来源筛选")
        self.cmb_order = QComboBox()
        for title, key in ORDER_CHOICES:
            self.cmb_order.addItem(title, key)
        self.cmb_page = QComboBox()
        self.cmb_page.setToolTip("一次显示多少条（不写死 200）")
        for title, key in PAGE_CHOICES:
            self.cmb_page.addItem(title, key)
        self.cmb_page.setCurrentIndex(1)
        for widget in (self.cmb_status, self.cmb_json, self.cmb_product,
                       self.cmb_ai, self.cmb_order, self.cmb_page):
            widget.currentIndexChanged.connect(lambda _=0: self.reload())

        filters = QHBoxLayout()
        filters.addWidget(QLabel("状态"))
        filters.addWidget(self.cmb_status)
        filters.addWidget(QLabel("JSON"))
        filters.addWidget(self.cmb_json)
        filters.addWidget(QLabel("成品"))
        filters.addWidget(self.cmb_product)
        filters.addWidget(QLabel("AI"))
        filters.addWidget(self.cmb_ai)
        filters.addWidget(QLabel("排序"))
        filters.addWidget(self.cmb_order)
        filters.addWidget(self.cmb_page)
        filters.addStretch(1)


        self.tbl_videos = _plain_table(self.VIDEO_HEADERS, stretch=1)
        self.tbl_videos.setColumnHidden(0, True)          # id 留着取值，不占版面
        head = self.tbl_videos.horizontalHeader()
        head.setSortIndicatorShown(True)
        head.setSectionsClickable(True)
        head.sectionClicked.connect(self.on_header_sort)
        self.tbl_videos.itemSelectionChanged.connect(self.on_video_changed)
        self.tbl_videos.doubleClicked.connect(lambda _=None: self.on_open_video())

        self.lbl_count = QLabel("—")

        box = QGroupBox("视频库")
        lay = QVBoxLayout(box)
        lay.addWidget(self.edit_search)
        lay.addLayout(filters)
        lay.addWidget(self.tbl_videos, 1)
        lay.addWidget(self.lbl_count)
        return box

    # ------------------------------------------------------------ 右：详情
    def _build_right(self) -> QWidget:
        self.lbl_video = _title("还没选视频")
        font = self.lbl_video.font()
        font.setPointSize(font.pointSize() + 6)      # 当前视频 = 页面唯一最大标题
        self.lbl_video.setFont(font)
        self.lbl_state = QLabel("左边选一个视频，这里就是它的工作区")
        self.lbl_meta = QLabel("")                   # 统计行：JSON ｜ 高光 ｜ 成品
        self.lbl_meta.setWordWrap(True)
        stats = self.lbl_meta.font()
        stats.setPointSize(stats.pointSize() + 1)
        stats.setBold(True)
        self.lbl_meta.setFont(stats)

        self.btn_source = QPushButton("查看原视频")
        self.btn_source.setToolTip("在文件管理器里定位这个视频文件")
        self.btn_source.clicked.connect(self.on_reveal_video)
        self.btn_render = _primary(QPushButton("直接剪辑"))   # 唯一主动作，唯一加粗
        self.btn_render.setToolTip("用选中的这份高光 JSON 出成品：选个 PRM 就开剪，一次 AI 都不调")
        self.btn_render.clicked.connect(self.on_render)
        self.btn_open = QPushButton("打开成品")
        self.btn_open.setToolTip("在文件管理器里定位选中的成品")
        self.btn_open.clicked.connect(self.on_reveal)
        self.btn_view = QPushButton("查看")
        self.btn_view.setToolTip("看选中 JSON 的每一段区间（右边直接显示，不弹窗）")
        self.btn_view.clicked.connect(self.on_view)

        quick = QHBoxLayout()
        quick.addWidget(self.btn_source)
        quick.addWidget(self.btn_render)
        quick.addWidget(self.btn_open)
        quick.addStretch(1)


        split = QSplitter(Qt.Vertical)
        split.addWidget(self._build_assets())
        split.addWidget(self._build_products())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.lbl_video)
        lay.addWidget(self.lbl_state)
        lay.addWidget(self.lbl_meta)
        lay.addLayout(quick)
        lay.addWidget(split, 1)
        return page

    def _build_assets(self) -> QWidget:
        self.lbl_current = QLabel("当前 JSON：—")
        self.chk_current_only = QCheckBox("只看当前")
        self.chk_current_only.setToolTip("默认展开全部高光 JSON；勾上只看当前那一份")
        self.chk_current_only.stateChanged.connect(lambda _=0: self.refresh_assets())
        # 「含已删除」属于低频，收进「更多 ▾」，这里只留状态（不进版面）
        self.chk_deleted = QCheckBox("含已删除")
        self.chk_deleted.setVisible(False)
        self.chk_deleted.stateChanged.connect(lambda _=0: self.refresh_assets())

        self.tbl_assets = _plain_table(self.ASSET_HEADERS, stretch=1)
        self.tbl_assets.itemSelectionChanged.connect(self.on_asset_changed)

        self.json_panel = JsonPanel(self.cfg, self, log=self._log)
        self.json_panel.saved.connect(self.on_json_saved)

        more = QHBoxLayout()
        more.addWidget(self.btn_view)
        self.btn_more = QPushButton("更多 ▾")
        self.btn_more.setToolTip("低频动作都在这里：编辑 / 复制 / 设为当前 / 导入 / 删除 / 恢复 / 原文")
        menu = QMenu(self.btn_more)
        menu.addAction("编辑（保存会新建一份）", self.on_edit_json)
        menu.addAction("复制这份 JSON", self.on_copy)
        menu.addAction("设为当前 JSON", self.on_set_current)
        menu.addSeparator()
        menu.addAction("导入现成 JSON…", self.on_import)
        menu.addAction("删除（软删）", self.on_delete)
        menu.addAction("恢复已删除", self.on_restore)
        menu.addSeparator()
        self.act_raw = menu.addAction("显示 JSON 原文", self.on_toggle_raw)
        self.act_raw.setCheckable(True)
        menu.addAction("复制 JSON 原文", self.on_copy_text)
        self.act_deleted = menu.addAction("列出已删除的 JSON", self.on_toggle_deleted)
        self.act_deleted.setCheckable(True)
        menu.addAction("复制血缘", self.on_copy_lineage)
        self.btn_more.setMenu(menu)
        more.addWidget(self.btn_more)
        more.addStretch(1)



        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(self.lbl_current, 1)
        head.addWidget(self.chk_current_only)
        head.addWidget(self.chk_deleted)
        left_lay.addLayout(head)
        left_lay.addWidget(self.tbl_assets, 1)
        left_lay.addLayout(more)

        inner = QSplitter(Qt.Horizontal)
        inner.addWidget(left)
        inner.addWidget(self.json_panel)
        inner.setStretchFactor(0, 3)
        inner.setStretchFactor(1, 2)

        box = QGroupBox("高光 JSON（选中一份，上面的「直接剪辑」就用它）")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(inner, 1)
        return box

    def _build_products(self) -> QWidget:
        self.tbl_products = _plain_table(self.PRODUCT_HEADERS, stretch=1)
        self.tbl_products.setColumnHidden(0, True)
        self.tbl_products.itemSelectionChanged.connect(self.refresh_lineage)

        self.lbl_product_head = QLabel("—")
        self.lbl_product_head.setWordWrap(True)
        self.lbl_product = QLabel("选一个成品")
        self.lbl_product.setWordWrap(True)
        self.tree_lineage = QTreeWidget()
        self.tree_lineage.setHeaderLabels(["血缘", "内容"])
        self.tree_lineage.setColumnWidth(0, 210)
        self.tree_lineage.setAlternatingRowColors(True)
        self.tree_lineage.setToolTip("点「高光 JSON」或「实际成品」节点，上面的表格会跳到对应那一行")
        self.tree_lineage.itemClicked.connect(self.on_lineage_clicked)


        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(self.lbl_product)
        right_lay.addWidget(self.tree_lineage, 1)


        inner = QSplitter(Qt.Horizontal)
        inner.addWidget(self.tbl_products)
        inner.addWidget(right)
        inner.setStretchFactor(0, 3)
        inner.setStretchFactor(1, 2)

        box = QGroupBox("成品")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.lbl_product_head)
        lay.addWidget(inner, 1)
        return box


    # ------------------------------------------------------------ 数据
    def _handle(self):
        if self._db is None:
            try:
                self._db = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "资产中心", f"数据库打不开：{exc}")
                return None
        return self._db

    def _search_changed(self, text: str) -> None:
        if not text.strip():        # 清空搜索框立刻回到全量，不用再按回车
            self.reload()

    def reload(self) -> None:
        """重查视频列表：一次 SQL 聚合，不扫磁盘，刷新后保持原来的选中行。"""
        db = self._handle()
        if db is None:
            return
        keep = self.current_video_id()
        self._fill_ai_choices(db)
        try:
            self._rows = db_assets.center_rows(
                db, search=self.edit_search.text().strip() or None,
                provider=self.cmb_ai.currentData() or None,
                status=str(self.cmb_status.currentData() or "all"),
                json=str(self.cmb_json.currentData() or "any"),
                product=str(self.cmb_product.currentData() or "any"),
                order=str(self.cmb_order.currentData() or "recent"),
                limit=int(self.cmb_page.currentData() or 500))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "资产中心", f"列表查不出来：{exc}")
            return


        self.tbl_videos.blockSignals(True)
        self.tbl_videos.setRowCount(0)
        for item in self._rows:
            line = self.tbl_videos.rowCount()
            self.tbl_videos.insertRow(line)
            ai = f"{item['provider']} / {item['model']}".strip(" /") or "—"
            self.tbl_videos.setItem(line, 0, _num(item["id"]))
            self.tbl_videos.setItem(line, 1, _cell(item["file_name"]))
            self.tbl_videos.setItem(line, 2, _duration_cell(item["duration"]))
            self.tbl_videos.setItem(line, 3, _cell("✓" if item["analysed"] else "✗",
                                                   center=True))
            self.tbl_videos.setItem(line, 4, _num(item["json_count"]))
            self.tbl_videos.setItem(line, 5, _num(item["highlight_count"]))
            self.tbl_videos.setItem(line, 6, _num(item["product_count"]))
            self.tbl_videos.setItem(line, 7, _cell(ai))
            self.tbl_videos.setItem(line, 8, _cell(_short_time(item["updated_at"]),
                                                   center=True))
        self.tbl_videos.resizeColumnsToContents()
        self.tbl_videos.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._sync_sort_indicator()
        self.tbl_videos.blockSignals(False)
        self.lbl_count.setText(f"共 {len(self._rows)} 个视频"
                               + ("（已按筛选条件过滤）" if self._filtering() else "")
                               + "　双击一行看它的高光 JSON")
        self.select_video(keep)

    def _sync_sort_indicator(self) -> None:
        """表头的排序箭头跟着「排序」下拉走（表格自己不排，避免和 SQL 排序打架）。"""
        key = str(self.cmb_order.currentData() or "recent")
        head = self.tbl_videos.horizontalHeader()
        head.setSortIndicator(COLUMN_BY_ORDER.get(key, -1),
                              Qt.AscendingOrder if key == "name" else Qt.DescendingOrder)

    def on_header_sort(self, section: int) -> None:
        """点表头 = 改「排序」下拉，排序仍然只在 center_rows 里算一次。"""
        key = ORDER_BY_COLUMN.get(int(section))
        if key is None:
            return
        index = self.cmb_order.findData(key)
        if index < 0:
            return
        if index == self.cmb_order.currentIndex():
            self.reload()               # 已经是这一列：重查一次就好
            return
        self.cmb_order.setCurrentIndex(index)   # currentIndexChanged 会带上 reload()

    def _filtering(self) -> bool:
        return bool(self.edit_search.text().strip() or self.cmb_ai.currentData()
                    or str(self.cmb_json.currentData() or "any") != "any"
                    or str(self.cmb_product.currentData() or "any") != "any"
                    or str(self.cmb_status.currentData() or "all") != "all")

    def _fill_ai_choices(self, db: Any) -> None:
        """AI 下拉只放库里真的出现过的来源，另外加一档「未知」。绝不编 AI 名字。"""
        keep = self.cmb_ai.currentData()
        self.cmb_ai.blockSignals(True)
        self.cmb_ai.clear()
        self.cmb_ai.addItem("全部", "")
        try:
            for name in db_assets.known_providers(db):
                self.cmb_ai.addItem(name, name)
        except Exception:  # noqa: BLE001
            pass
        self.cmb_ai.addItem("未知（没记下 AI）", db_assets.NO_PROVIDER)
        if keep:
            self.cmb_ai.setCurrentIndex(max(0, self.cmb_ai.findData(keep)))
        self.cmb_ai.blockSignals(False)


    def current_video_id(self) -> int | None:
        line = self.tbl_videos.currentRow()
        if line < 0:
            return None
        item = self.tbl_videos.item(line, 0)
        return int(item.text()) if item is not None and item.text() else None

    def select_video(self, video_id: int | None) -> None:
        """选中指定视频（列表刷新 / 重排后尽量停在原来那个视频上）。"""
        target = 0
        if video_id is not None:
            for line in range(self.tbl_videos.rowCount()):
                item = self.tbl_videos.item(line, 0)
                if item is not None and item.text() and int(item.text()) == int(video_id):
                    target = line
                    break
        if self.tbl_videos.rowCount():
            self.tbl_videos.selectRow(target)
        else:
            self.on_video_changed()

    def on_video_changed(self) -> None:
        vid = self.current_video_id()
        db = self._handle()
        self._product_rows = None        # 换视频 = 成品缓存作废，下一次刷新重新查
        if vid is None or db is None:
            self.lbl_video.setText("还没选视频")
            self.lbl_state.setText("左边选一个视频，这里就是它的工作区")
            self.lbl_meta.setText("")
            self.tbl_assets.setRowCount(0)
            self.tbl_products.setRowCount(0)
            self.json_panel.clear()
            self.tree_lineage.clear()
            return
        row = next((r for r in self._rows if r["id"] == vid), None)
        self.lbl_video.setText(f"{row['file_name'] if row else f'视频 #{vid}'}")
        analysis = db_repo.latest_analysis(db, vid)
        state = f"{_seconds(row['duration']) if row else '—'}　　" \
                f"{'已分析' if analysis is not None else '未分析'}"
        if row and row["file_path"]:
            state += f"　　{row['file_path']}"
        self.lbl_state.setText(state)
        self.lbl_meta.setText("　｜　".join((
            f"JSON　{row['json_count'] if row else 0}",
            f"高光　{row['highlight_count'] if row else 0}",
            f"成品　{row['product_count'] if row else 0}")))
        self.refresh_assets()
        self.refresh_products(reload=False)   # 上面那一步已经把成品缓存填好了



    def on_open_video(self) -> None:
        """双击视频：进入详情——把焦点放到它的 JSON 表上。"""
        if self.tbl_assets.rowCount():
            self.tbl_assets.selectRow(0)
            self.tbl_assets.setFocus()

    # ------------------------------------------------------------ JSON
    def refresh_assets(self) -> None:
        db = self._handle()
        vid = self.current_video_id()
        keep = self.selected_asset()
        self.tbl_assets.blockSignals(True)
        self.tbl_assets.setRowCount(0)
        if db is None or vid is None:
            self.tbl_assets.blockSignals(False)
            self.lbl_current.setText("当前 JSON：—")
            self.json_panel.clear()
            return
        current = db_assets.current_asset(db, vid)
        rows = db_assets.list_assets(db, vid, include_deleted=self.chk_deleted.isChecked())
        counts = db_assets.product_counts_for_assets(db, vid)   # 一条 SQL，不逐行查
        if self.chk_current_only.isChecked() and current is not None:
            rows = [r for r in rows if int(r["id"]) == int(current["id"])]
        self._asset_rows = list(rows)        # 详情面板直接用这些行，不再逐行回查
        if not rows:
            self.lbl_current.setText("这个视频还没有高光 JSON —— 让 AI 分析一次，"
                                     "或者用「更多 → 导入现成 JSON」")
        else:
            self.lbl_current.setText(
                f"已有 {len(rows)} 个高光 JSON　｜　当前："
                + (f"★ {_json_title(current)}（{_ai_label(current)}）"
                   if current is not None else "还没设"))
        for row in rows:
            line = self.tbl_assets.rowCount()
            self.tbl_assets.insertRow(line)
            is_current = current is not None and int(row["id"]) == int(current["id"])
            self.tbl_assets.setItem(line, 0, _cell("★ 当前" if is_current else "○ 历史",
                                                   center=True))
            self.tbl_assets.setItem(line, 1, _id_item(_json_title(row), int(row["id"]),
                                                      bold=is_current))
            self.tbl_assets.setItem(line, 2, _cell(_ai_label(row), center=True))
            self.tbl_assets.setItem(line, 3, _num(int(row["clip_count"] or 0)))
            self.tbl_assets.setItem(line, 4, _cell(_score(row["best_score"]), center=True))
            self.tbl_assets.setItem(line, 5, _num(counts.get(int(row["id"]), 0)))
            self.tbl_assets.setItem(line, 6, _cell(_short_time(row["created_at"]), center=True))
            self.tbl_assets.setItem(line, 7, _cell("已删除" if row["deleted_at"] else "在用",
                                                   center=True))
        self.tbl_assets.resizeColumnsToContents()
        self.tbl_assets.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_assets.blockSignals(False)
        if not rows:
            self.json_panel.clear("这个视频还没有高光 JSON")
            return
        self.select_asset(keep if keep is not None
                          else (int(current["id"]) if current is not None else None))

    def select_asset(self, asset_id: int | None) -> None:
        if not self.tbl_assets.rowCount():
            self.json_panel.clear("这个视频还没有高光 JSON")
            return
        target = 0
        if asset_id is not None:
            for line in range(self.tbl_assets.rowCount()):
                if _row_id(self.tbl_assets, line, 1) == int(asset_id):
                    target = line
                    break
        self.tbl_assets.selectRow(target)

    def selected_asset(self) -> int | None:
        return _row_id(self.tbl_assets, self.tbl_assets.currentRow(), 1)


    def on_asset_changed(self) -> None:
        asset_id = self.selected_asset()
        row = next((r for r in self._asset_rows
                    if asset_id is not None and int(r["id"]) == int(asset_id)), None)
        self.json_panel.show_asset(asset_id, row=row,
                                   products=self._products_of(asset_id))
        self.refresh_products(reload=False)   # 成品区跟着标出「这份 JSON 剪出来的」

    def _products_of(self, asset_id: int | None) -> list[dict[str, Any]] | None:
        """从成品全景缓存里挑出这份 JSON 剪出的成品（旧的在前，和以前的顺序一致）。

        缓存还没建就返回 None —— 详情面板会自己查一次，不会显示错。
        """
        if asset_id is None or self._product_rows is None:
            return None
        mine = [{"id": info["artifact_id"]} for info in self._product_rows
                if info["asset_id"] is not None and int(info["asset_id"]) == int(asset_id)]
        mine.reverse()          # products_overview 是新的在前
        return mine


    def on_json_saved(self, new_id: int) -> None:
        self.chk_current_only.setChecked(False)     # 新方案在历史里，展开才看得见
        self.refresh_assets()
        self.select_asset(int(new_id))
        self.changed.emit()

    def _need_asset(self) -> tuple[Any, int] | None:
        db = self._handle()
        asset_id = self.selected_asset()
        if db is None:
            return None
        if asset_id is None:
            QMessageBox.information(self, "资产中心", "先在「高光 JSON」里选一份")
            return None
        return db, asset_id

    def _note(self, text: str) -> None:
        if self._log:
            self._log(text)
        self.changed.emit()

    def on_view(self) -> None:
        """查看：右边面板直接显示这份 JSON 的区间（不弹窗）。"""
        got = self._need_asset()
        if got is None:
            return
        _db, asset_id = got
        self.json_panel.show_asset(asset_id)
        self.json_panel.table.setFocus()

    def on_edit_json(self) -> None:
        """编辑：进编辑态；保存时走 `edit_asset(in_place=False)`，原 JSON 不动。"""
        got = self._need_asset()
        if got is None:
            return
        _db, asset_id = got
        self.json_panel.show_asset(asset_id)
        self.act_raw.setChecked(True)                 # 要改就得看得见原文
        self.json_panel.set_raw_visible(True)
        self.json_panel.on_edit()

    def on_reveal_video(self) -> None:
        """查看原视频：在文件管理器里定位这个视频文件。"""
        vid = self.current_video_id()
        row = next((r for r in self._rows if r["id"] == vid), None)
        if row is None or not row["file_path"]:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        _reveal(self, Path(str(row["file_path"])))


    def on_copy(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        new_id = db_assets.copy_asset(db, asset_id)
        if new_id is None:
            QMessageBox.information(self, "资产中心", "复制不了（这份 JSON 不存在）")
            return
        self._note(f"[高光 JSON] #{asset_id} 已复制成 #{new_id}（原件没动）")
        self.chk_current_only.setChecked(False)
        self.refresh_assets()
        self.select_asset(int(new_id))

    def on_import(self) -> None:
        db = self._handle()
        vid = self.current_video_id()
        if db is None or vid is None:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选一份高光 JSON", str(self.cfg.root),
                                              "JSON (*.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "资产中心", f"这份 JSON 读不进来：{exc}")
            return
        count, _best = db_assets.summarize(payload)
        if not count:
            QMessageBox.information(self, "资产中心", "这份 JSON 里抠不出可用片段，不登记")
            return
        asset_id = db_assets.create_asset(db, vid, payload, source_type="imported",
                                          note=f"从 {Path(path).name} 导入")
        self._note(f"[高光 JSON] 已登记 #{asset_id}（{count} 个高光，来自 {Path(path).name}）")
        self.reload()
        self.select_asset(int(asset_id))

    def on_set_current(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        if db_assets.set_current_asset(db, asset_id):
            self._note(f"[高光 JSON] #{asset_id} 已设为当前 JSON（其他 JSON 一份都没删）")
        else:
            QMessageBox.information(self, "资产中心", "已删除的 JSON 不能设为当前")

        self.refresh_assets()
        self.select_asset(asset_id)

    def on_delete(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        kept = len(db_assets.products_for_asset(db, asset_id))
        if QMessageBox.question(
                self, "删除高光 JSON",
                f"把 高光 JSON #{asset_id} 标成已删除？\n"
                f"已经剪出来的 {kept} 个成品一个都不会动。",

                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        if db_assets.delete_asset(db, asset_id):
            self._note(f"[高光 JSON] #{asset_id} 已软删（成品未动）")
        self.refresh_assets()

    def on_restore(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        if db_assets.restore_asset(db, asset_id):
            self._note(f"[高光 JSON] #{asset_id} 已恢复")
        else:
            QMessageBox.information(self, "资产中心", "这一份本来就没删")
        self.refresh_assets()
        self.select_asset(asset_id)

    def on_render(self) -> None:
        """按选中的 JSON 出成品：选 PRM → 交给 `MainWindow.render_asset`（不调 AI）。"""
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        row = db_assets.get_asset(db, asset_id)
        if row is None or row["deleted_at"]:
            QMessageBox.information(self, "资产中心", "这份 JSON 已删除，先恢复再剪")
            return
        if int(row["clip_count"] or 0) <= 0:
            QMessageBox.information(self, "资产中心", "这份 JSON 抠不出可用片段，剪不了")
            return
        window = self._window if hasattr(self._window, "render_asset") else None
        if window is None:
            QMessageBox.information(self, "资产中心",
                                    "这个窗口没连上主界面，剪不了。"
                                    "命令行可以用：run.py assets --render <JSON ID> --prm <PRM>")
            return
        dialog = RenderDialog(self.cfg, row, window, self, log=self._log)
        if dialog.exec_() == QDialog.Accepted:
            self._note(f"[高光 JSON] #{asset_id} 已交给主界面渲染（进度和日志在主界面）")

    # ------------------------------------------------------------ 成品
    def refresh_products(self, *, reload: bool = True) -> None:
        """成品表。`reload=False` 只是重画（换选中的 JSON 时用），一条 SQL 都不发。"""
        db = self._handle()
        vid = self.current_video_id()
        picked = self.selected_asset()
        keep = self.selected_product()
        self.tbl_products.blockSignals(True)
        self.tbl_products.setRowCount(0)
        if db is None or vid is None:
            self._product_rows = None
            self._lineage_for = None
            self.tree_lineage.clear()
            self.tbl_products.blockSignals(False)
            self.lbl_product_head.setText("—")
            return
        if reload or self._product_rows is None:
            self._product_rows = db_assets.products_overview(db, vid)  # 两条 SQL 带出全部成品 + 区间
        rows = self._product_rows
        mine: list[str] = []
        for info in rows:
            spans = info["spans"]
            span = "—"
            length = "—"
            if spans:
                span = "；".join(_span(s["start"], s["end"]) for s in spans)
                length = f"{sum(float(s['duration'] or 0.0) for s in spans):.2f}s"
            same_json = (info["asset_id"] is not None and picked is not None
                         and int(info["asset_id"]) == int(picked))
            if same_json:
                mine.append(str(info["prm_name"] or "—"))
            line = self.tbl_products.rowCount()
            self.tbl_products.insertRow(line)
            self.tbl_products.setItem(line, 0, _num(info["artifact_id"]))
            self.tbl_products.setItem(line, 1, _id_item(Path(info["path"]).name,
                                                        info["artifact_id"], bold=same_json))
            self.tbl_products.setItem(line, 2, _cell(
                "—" if info["asset_id"] is None
                else f"高光 JSON #{info['asset_id']}"
                + ("（已删除）" if info["asset_deleted"] else ""), center=True))
            self.tbl_products.setItem(line, 3, _cell(
                "—" if info["prm_name"] is None else str(info["prm_name"])
                + ("（已删除）" if info["prm_deleted"] else ""), center=True))
            self.tbl_products.setItem(line, 4, _cell(length, center=True))
            self.tbl_products.setItem(line, 5, _cell(span))
            self.tbl_products.setItem(line, 6, _cell(_short_time(info["created_at"]),
                                                     center=True))
            self.tbl_products.setItem(line, 7, _cell(
                "✓ 完成" if info["exists_on_disk"] else "⚠ 文件不在盘上", center=True))
        self.tbl_products.resizeColumnsToContents()
        self.tbl_products.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        if not rows:
            self.tbl_products.blockSignals(False)
            self.tree_lineage.clear()
            self._lineage_for = None
            self.lbl_product_head.setText("这个视频还没有成品 —— 上面选一份高光 JSON，"
                                          "点「直接剪辑」就能出成品")
            self.lbl_product.setText("选一个成品，这里显示它的血缘")
            return
        head = f"共 {len(rows)} 个成品"
        if picked is not None:
            head += (f"　｜　选中的高光 JSON #{picked} 剪出了 {len(mine)} 个"
                     + (f"（PRM：{' / '.join(mine)}）" if mine else "：还没剪过，点「直接剪辑」"))
        self.lbl_product_head.setText(head)
        target = 0                       # 选回原来那一行，别让血缘树白重画一遍
        for line in range(self.tbl_products.rowCount()):
            if _row_id(self.tbl_products, line, 1) == keep:
                target = line
                break
        self.tbl_products.selectRow(target)
        self.tbl_products.blockSignals(False)
        if self.selected_product() != self._lineage_for:
            self.refresh_lineage()



    def selected_product(self) -> int | None:
        line = self.tbl_products.currentRow()
        if line < 0:
            return None
        item = self.tbl_products.item(line, 0)
        return int(item.text()) if item is not None and item.text() else None

    def refresh_lineage(self) -> None:
        """成品 → PRM → JSON → AI → 视频，摆成一棵能读的树，三层区间一并显示。"""
        db = self._handle()
        artifact_id = self.selected_product()
        self.tree_lineage.clear()
        self._lineage_for = artifact_id
        if db is None or artifact_id is None:
            self.lbl_product.setText("选一个成品")
            return
        info = db_assets.artifact_lineage(db, artifact_id)
        if info is None:
            self.lbl_product.setText("查不到这个成品的记录")
            return
        video = info.get("video") or {}
        asset = info.get("asset")
        prm = info.get("prm")
        spans = db_assets.lineage_spans(db, artifact_id)
        path = Path(str(info["path"]))
        source = "—" if asset is None else _json_title(asset)
        self.lbl_product.setText(
            f"{path.name}\n"
            f"来源 JSON：{source}"
            f"　｜　PRM：{'—' if prm is None else prm['name']}"
            f"　｜　生成时间：{_short_time(info.get('created_at'))}"
            f"　｜　状态：{'✓ 完成' if info['exists_on_disk'] else '⚠ 文件不在盘上'}")


        def node(parent, title, value="", *, asset_id=None, artifact_id=None):
            item = QTreeWidgetItem([str(title), str(value)])
            if asset_id is not None:
                item.setData(0, Qt.UserRole, ("asset", int(asset_id)))
            if artifact_id is not None:
                item.setData(0, Qt.UserRole, ("product", int(artifact_id)))
            (parent.addChild(item) if isinstance(parent, QTreeWidgetItem)
             else self.tree_lineage.addTopLevelItem(item))
            return item

        root = node(None, "视频", video.get("file_name", "—"))
        node(root, "路径", video.get("file_path", "—"))
        node(root, "分析批次", info.get("analysis_id") or "—")
        json_node = node(root, "高光 JSON",
                         "—" if asset is None else f"{_json_title(asset)}（{asset['name']}）"
                         + ("（已删除）" if info.get("asset_deleted") else ""),
                         asset_id=None if asset is None else int(asset["id"]))

        node(json_node, "AI", info.get("provider") or "—")
        node(json_node, "模型", info.get("model") or "—")
        node(json_node, "AI 任务", info.get("task_id") or "—")
        for index, clip in enumerate(spans["ai"], start=1):
            node(json_node, f"原始区间 {index}",
                 f"{_span(clip.get('start'), clip.get('end'))}"
                 f"　评分 {_score(clip.get('score'))}")
        engine_node = node(root, "Clip Engine", "确定性修正（规则来源只有 plan_clips）")
        for index, plan in enumerate(spans["engine"], start=1):
            one = node(engine_node, f"修正区间 {index}",
                       f"{_span(plan['start'], plan['end'])}（{_score(plan['duration'])}s）")
            node(one, "15 秒上限", "触发" if plan["capped"] else "未触发")
            for note in plan["notes"]:
                node(one, "原因", note)
            if not plan["notes"]:
                node(one, "原因", "未调整（AI 区间本身就落在语义边界上）")
        node(root, "PRM", "—" if prm is None else f"{prm['name']}（{prm['filename']}）"
             + ("（已删除）" if info.get("prm_deleted") else ""))
        final = node(root, "实际成品", path.name, artifact_id=artifact_id)
        for index, clip in enumerate(spans["actual"], start=1):
            node(final, f"实际区间 {index}",
                 f"{_span(clip['start'], clip['end'])}（{_score(clip['duration'])}s）")
        if not spans["actual"]:
            node(final, "实际区间", "库里没有 clips 记录（Batch 11 之前剪的老成品）")
        elif spans["engine"] and len(spans["engine"]) == len(spans["actual"]):
            same = all(abs(float(p["start"]) - float(c["start"] or 0)) < 0.01
                       and abs(float(p["end"]) - float(c["end"] or 0)) < 0.01
                       for p, c in zip(spans["engine"], spans["actual"]))
            node(final, "结论", "✓ Engine 与实际渲染一致" if same
                 else "⚠ Engine 与实际渲染不一致（加减秒之后剪的，或分析数据后来变过）")
        self.tree_lineage.expandAll()

    def on_lineage_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        """点血缘节点就定位到对应那一行（不弹窗、不改数据）。"""
        marker = item.data(0, Qt.UserRole)
        if not marker:
            return
        kind, ident = marker
        if kind == "asset":
            self.select_asset(int(ident))
            self.tbl_assets.setFocus()
            return
        if kind == "product":
            for line in range(self.tbl_products.rowCount()):
                if _row_id(self.tbl_products, line, 1) == int(ident):
                    self.tbl_products.selectRow(line)
                    self.tbl_products.setFocus()
                    return

    def on_toggle_raw(self) -> None:
        """更多 ▾ → 显示 / 收起 JSON 原文。"""
        shown = bool(self.act_raw.isChecked())
        self.json_panel.set_raw_visible(shown)

    def on_toggle_deleted(self) -> None:
        """更多 ▾ → 列不列已删除的 JSON（状态还是那个隐藏的勾选框）。"""
        self.chk_deleted.setChecked(bool(self.act_deleted.isChecked()))

    def on_copy_text(self) -> None:
        """更多 ▾ → 复制 JSON 原文。"""
        self.json_panel.on_copy_text()

    def on_reveal(self) -> None:
        db = self._handle()
        artifact_id = self.selected_product()
        if db is None or artifact_id is None:
            QMessageBox.information(self, "资产中心", "先在「成品」里选一个")
            return
        path = db_assets.product_path(db, artifact_id)
        if path is not None:
            _reveal(self, path)

    def on_copy_lineage(self) -> None:
        lines: list[str] = []

        def walk(item: QTreeWidgetItem, depth: int) -> None:
            lines.append("  " * depth + f"{item.text(0)}：{item.text(1)}".rstrip("："))
            for index in range(item.childCount()):
                walk(item.child(index), depth + 1)

        for index in range(self.tree_lineage.topLevelItemCount()):
            walk(self.tree_lineage.topLevelItem(index), 0)
        if lines:
            QApplication.clipboard().setText("\n".join(lines))
            self._note("[资产中心] 血缘已复制到剪贴板")


# ================================================================ 资产中心
class AssetCenter(QWidget):
    """视频资产中心：非模态窗口，两页 —— [视频资产] 和 [PRM 管理]。"""

    changed = pyqtSignal()

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = parent
        self.setWindowFlags(Qt.Window)            # 有 parent 也要当独立窗口
        self.setWindowTitle("视频资产中心")
        self.setMinimumSize(1180, 760)

        self.videos = VideoAssetsPage(cfg, window=parent, parent=self, log=log)
        self.prm_panel = PrmPanel(cfg, self, log=log)
        self.videos.changed.connect(self._on_changed)
        self.prm_panel.changed.connect(self._on_changed)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.videos, "视频资产")
        self.tabs.addTab(self.prm_panel, "PRM 管理")

        self.btn_reload = QPushButton("🔄 刷新")
        self.btn_reload.setToolTip("重新查一次库（视频 / 高光 JSON / 成品 / PRM 一起刷）")
        self.btn_reload.clicked.connect(self.reload)
        head = QHBoxLayout()
        head.addWidget(_title("视频资产中心"))
        head.addWidget(QLabel("视频 → 高光 JSON → Clip Engine → PRM → 成品"), 1)
        head.addWidget(self.btn_reload)

        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        foot = QHBoxLayout()
        foot.addWidget(QLabel("全部可搜、可筛、可追溯：选视频看它的高光 JSON，选 JSON 看区间和成品"))
        foot.addStretch(1)
        foot.addWidget(close)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addLayout(head)
        outer.addWidget(self.tabs, 1)
        outer.addLayout(foot)


    # ------------------------------------------------------------ 转发
    def _on_changed(self) -> None:
        self.changed.emit()

    def reload(self) -> None:
        self.videos.reload()
        self.prm_panel.reload()

    def show_prm_page(self) -> None:
        self.tabs.setCurrentIndex(1)

    # 老代码（AI 面板、验证脚本）习惯直接摸这些名字，保持能用
    @property
    def tbl_videos(self):
        return self.videos.tbl_videos

    @property
    def tbl_assets(self):
        return self.videos.tbl_assets

    @property
    def tbl_products(self):
        return self.videos.tbl_products

    @property
    def chk_current_only(self):
        return self.videos.chk_current_only

    @property
    def json_panel(self):
        return self.videos.json_panel

    @property
    def tree_lineage(self):
        return self.videos.tree_lineage

    @property
    def lbl_meta(self):
        return self.videos.lbl_meta

    @property
    def lbl_current(self):
        return self.videos.lbl_current

    def current_video_id(self) -> int | None:
        return self.videos.current_video_id()

    def select_video(self, video_id: int | None) -> None:
        self.videos.select_video(video_id)

    def selected_asset(self) -> int | None:
        return self.videos.selected_asset()

    def refresh_assets(self) -> None:
        self.videos.refresh_assets()

    def refresh_products(self, *, reload: bool = True) -> None:
        self.videos.refresh_products(reload=reload)

    def refresh_lineage(self) -> None:
        self.videos.refresh_lineage()

    def on_view(self) -> None:
        self.videos.on_view()

    def on_render(self) -> None:
        self.videos.on_render()

    def on_copy(self) -> None:
        self.videos.on_copy()

    def on_import(self) -> None:
        self.videos.on_import()

    def on_set_current(self) -> None:
        self.videos.on_set_current()

    def on_delete(self) -> None:
        self.videos.on_delete()

    def on_restore(self) -> None:
        self.videos.on_restore()

    def on_reveal(self) -> None:
        self.videos.on_reveal()

    def on_copy_lineage(self) -> None:
        self.videos.on_copy_lineage()

    def on_default(self) -> None:
        self.prm_panel.on_default()

    def exec_(self) -> int:
        """兼容老调用：以前是模态弹窗，现在只是把窗口拎到前面（不再阻塞）。"""
        self.show()
        self.raise_()
        self.activateWindow()
        return 0


# 老入口还叫 AssetDialog（别处这么引），指向同一个资产中心
class AssetDialog(AssetCenter):
    """兼容名字：`AssetDialog` 就是视频资产中心。"""
