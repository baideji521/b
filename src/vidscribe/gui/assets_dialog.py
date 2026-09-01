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
import shutil
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QPushButton,
    QSizePolicy,
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
from ..db.importer import refresh_from_disk
from .ai_options import DropDirEdit, dir_row

STATUS_CHOICES = (("全部", "all"), ("已分析", "analysed"), ("未分析", "not_analysed"))
# 「有没有 JSON」「有没有成品」各自一个下拉：三个条件能一起生效（场景 A 一步到位）
JSON_CHOICES = (("全部", "any"), ("有 JSON", "has"), ("无 JSON", "none"))
PRODUCT_CHOICES = (("全部", "any"), ("有成品", "has"), ("无成品", "none"))
ORDER_CHOICES = (("最近更新", "recent"), ("最近处理", "processed"), ("视频名称", "name"),
                 ("视频时长", "duration"), ("JSON 数量", "json"), ("高光数量", "highlight"),
                 ("成品数量", "product"))
# 点表头 = 换「排序」下拉，排序永远只有 center_rows 这一处真源（列号 → 排序键）
# 列号：0 隐藏 ID，1 勾选框，2 视频，3 目录，4 时长，5 分析，6 JSON，7 高光，8 成品，
# 9 AI/模型，10 更新时间
ORDER_BY_COLUMN = {2: "name", 4: "duration", 6: "json", 7: "highlight",
                   8: "product", 10: "recent"}
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
        self.table.setMinimumHeight(40)

        # 三层区间：这一块占面板纵向的一半（和上面的段列表各一半），
        # 段数多的时候直接在里面滚，不再被写死的最大高度截掉
        self.tbl_layers = _plain_table(self.LAYER_HEADERS, stretch=4)
        self.tbl_layers.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_layers.verticalHeader().setDefaultSectionSize(20)
        self.tbl_layers.setMinimumHeight(3 * 20 + 22)
        self.tbl_layers.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tbl_layers.setToolTip("AI 原始区间 → Clip Engine 修正 → 实际渲染区间"
                                   "（规则来源就是真剪时用的那一套）")
        self.lbl_engine = QLabel("—")          # 结论 + 原因（Engine 为什么改了）
        self.lbl_engine.setWordWrap(True)
        self.lbl_engine.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # 原因可能好几行，但**不许**因此把整个面板的最小高度顶上去：
        # 放进一个定高的滚动区，写多少都只占这一条，读不完就在里面滚。
        self.box_engine = QScrollArea()
        self.box_engine.setWidget(self.lbl_engine)
        self.box_engine.setWidgetResizable(True)
        self.box_engine.setFixedHeight(40)
        self.box_engine.setFrameShape(QFrame.StyledPanel)


        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setMinimumHeight(80)

        # 平时一个动作按钮都不显示：编辑从「更多 ▾ → 编辑」进来，进了编辑态才出现这两个
        self.btn_save = QPushButton("保存为新 JSON")
        self.btn_save.setToolTip("存成一份新的高光 JSON，原件一个字不动")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_cancel = QPushButton("取消编辑")
        self.btn_cancel.clicked.connect(self.on_cancel_edit)
        self.lbl_editing = QLabel("正在编辑：保存会新建一份高光 JSON，原件不动")
        self.lbl_editing.setWordWrap(True)      # 同理：不换行会顶宽整个面板

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(self.lbl_head)
        outer.addWidget(self.lbl_meta)
        outer.addWidget(self.table, 1)
        outer.addWidget(self.tbl_layers, 1)      # 三层区间：纵向占一半
        outer.addWidget(self.box_engine)
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
        self._fill_layers([("AI 原始", "—", "—", "—", "—"),
                           ("Clip Engine", "—", "—", "—", "—"),
                           ("实际渲染", "—", "—", "—", "—")])
        self.lbl_engine.setText("选一份高光 JSON，这里给出 AI 原始 / Clip Engine / 实际渲染"
                                "三层区间的结论和原因")
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
            # 三层永远都在（就算算不出来也摆着三行），只是值是「—」并写清为什么
            return ([("AI 原始", "—", "—", "—", "—"),
                     ("Clip Engine", "—", "—", "—", "算不出"),
                     ("实际渲染", "—", "—", "—", "还没剪")],
                    ["⚠ 算不出 Clip Engine 区间（这个视频还没有逐词时间戳，"
                     "或 JSON 里没有可用片段）"])
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
                         "✓ 未调整" if same else "已修正"))
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
        # 结论写在最前面，原因紧跟其后 —— 用户先看到「一致 / 不一致」，再看为什么
        if not actual:
            return grid, ["○ 还没剪过：点「直接剪辑」就按上面的 Clip Engine 区间出成品", *notes]
        same_all = (len(actual) == len(spans["engine"])
                    and all(abs(float(p["start"]) - float(c["start"] or 0)) < 0.01
                            and abs(float(p["end"]) - float(c["end"] or 0)) < 0.01
                            for p, c in zip(spans["engine"], actual)))
        verdict = ("✓ Engine 与实际渲染一致" if same_all else
                   "⚠ 实际渲染与 Engine 不一致（加减秒之后剪的，或分析数据后来变过）")
        return grid, [verdict, *notes]

    def _fill_layers(self, grid: list[tuple[str, ...]]) -> None:
        """把三层区间填进网格：高度按行数走，最少三行（层级永远看得见）。"""
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
        # 高度交给布局（占面板一半），这里只保证至少三层看得见
        self.tbl_layers.setMinimumHeight(3 * 20 + 22)

    def _layers_text(self, db: Any, asset_id: int, products: Any = (),
                     row: Any = None) -> str:
        """填三层区间网格，返回结论 / 原因那几行文字（给下面的结论条用）。

        网格高度按实际行数算：一段就是三行，两段就是六行，绝不把后面的层级藏起来。
        """
        grid, notes = self._layer_rows(db, asset_id, products, row)
        self._fill_layers(grid)
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
class PrmEditDialog(QDialog):
    """新增 / 修改一份 PRM：只有名称、来源文件、提示词正文三样。

    **正文存数据库**，「来源文件」只是「当初从哪个文件导进来的」这条记录。
    「导入文件…」当场把文件正文读进来，名称留空就按文件名填（`prm_高潮_不回落.txt`
    → `prm_高潮_不回落`），之后随时可以自己改。点「保存」才写库。
    """

    def __init__(self, cfg: Any, parent=None, row: Any = None):
        super().__init__(parent)
        self.cfg = cfg
        self._row = row
        self.setWindowTitle("新增 PRM" if row is None else f"修改 PRM：{row['name']}")
        self.setMinimumSize(560, 520)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("给这套规则起个名字，比如 prm_高潮_不回落")
        self.edit_file = QLineEdit()
        self.edit_file.setPlaceholderText("来源文件（可留空）：只作记录，发 AI 用的是库里的正文")
        btn_pick = QPushButton("导入文件…")
        btn_pick.setToolTip("选一个 txt：正文当场读进下面，名称留空就按文件名填")
        btn_pick.clicked.connect(self.on_pick)
        file_row = QHBoxLayout()
        file_row.addWidget(self.edit_file, 1)
        file_row.addWidget(btn_pick)
        holder = QWidget()
        holder.setLayout(file_row)

        form = QFormLayout()
        form.addRow("名称", self.edit_name)
        form.addRow("来源文件", holder)

        self.view_text = QPlainTextEdit()
        self.view_text.setPlaceholderText("提示词正文：发给 AI 的剪辑规则（上传时文件名统一成 prompt.txt）")

        self.btn_ok = _primary(QPushButton("保存"))
        self.btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(btn_cancel)
        foot.addWidget(self.btn_ok)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(QLabel("提示词正文"))
        outer.addWidget(self.view_text, 1)
        outer.addLayout(foot)

        if row is not None:
            self.edit_name.setText(str(row["name"]))
            self.edit_file.setText(str(row["filename"]))
            self.view_text.setPlainText(str(row["content"] or ""))
        self.edit_name.setFocus()

    def on_pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选提示词文件", str(self.cfg.root),
                                              "文本 (*.txt *.md);;所有文件 (*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            QMessageBox.warning(self, "PRM", f"文件读不出来：{exc}")
            return
        if not text.strip():
            QMessageBox.information(self, "PRM", "这个文件是空的，没有正文可导入")
            return
        self.edit_file.setText(path)
        self.view_text.setPlainText(text)
        if not self.edit_name.text().strip():
            self.edit_name.setText(Path(path).stem)

    def payload(self) -> tuple[str, str, str]:
        """(名称, 来源文件, 正文)。来源文件留空时按名称兜一个 `名称.txt`。"""
        name = self.edit_name.text().strip()
        source = self.edit_file.text().strip() or f"{name}.txt"
        return name, source, self.view_text.toPlainText()

    def accept(self) -> None:
        name, _source, text = self.payload()
        if not name:
            QMessageBox.information(self, "PRM", "名称得填")
            return
        if not text.strip():
            QMessageBox.information(self, "PRM", "正文是空的：自己写一段，或点「导入文件…」读一个")
            return
        super().accept()


class PrmPanel(QWidget):
    """PRM 清单页：一张表 + 一排动作，改内容走「修改」弹窗（双击 / 右键也行）。

    **提示词正文存在数据库里**（prm_profiles.content），库就是唯一权威；
    「来源文件」只记当初从哪个文件导进来的，发 AI 时不读它，上传时文件名统一成
    prompt.txt。语言 / 版本这两个字段库里还留着，界面上不再露。
    """


    HEADERS = ("ID", "名称", "剪出成品", "更新时间", "状态")


    changed = pyqtSignal()
    notice = pyqtSignal(str)         # 一句人话的操作结果，显示在窗口顶部

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._db: Any = None

        self.table = _plain_table(self.HEADERS, stretch=1)
        self.table.setColumnHidden(0, True)          # id 留着取值，界面上不露主键
        #: id → (使用中, 已删除)：按钮状态不去解析界面上的字
        self._states: dict[int, tuple[bool, bool]] = {}
        self.table.itemSelectionChanged.connect(self.on_selected)
        self.table.doubleClicked.connect(lambda _=None: self.on_double_click())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_menu)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔎 搜索 PRM 名称 / 文件")
        self.edit_search.setClearButtonEnabled(True)
        self.edit_search.textChanged.connect(lambda _="": self.reload())


        self.chk_all = QCheckBox("含已删除")
        self.chk_all.stateChanged.connect(lambda _=0: self.reload())

        # 一份 PRM 都没有时给出下一步，而不是留一张空表（Phase 16 空状态）
        self.lbl_empty = QLabel("暂无 PRM，请先创建或导入一套 PRM"
                                "（点「新增」，在弹出的窗口里填名称、导入正文）")
        self.lbl_empty.setWordWrap(True)

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(_title("PRM 管理"))
        head.addWidget(QLabel("PRM 是剪辑规则（发给 AI 的提示词），和高光 JSON 是两回事"), 1)
        head.addWidget(self.chk_all)
        outer.addLayout(head)
        outer.addWidget(self.edit_search)
        # 页面就是一张清单：改内容走「修改」弹窗（双击 / 右键也行），页里不再挂编辑区
        outer.addWidget(self.table, 1)
        outer.addWidget(self.lbl_empty)
        outer.addLayout(self._buttons())
        self.reload()


    def _buttons(self) -> QHBoxLayout:
        """一级只留「新增」「修改」「停用 / 启用」，其余收进「更多 ▾」（右键菜单里也有）。"""
        row = QHBoxLayout()
        self.btn_add = _primary(self._button(
            "新增", "弹出窗口填名称 + 导入正文，登记一份新 PRM", self.on_new))
        row.addWidget(self.btn_add)
        self.btn_modify = self._button(
            "修改", "改选中那一份的名称 / 来源文件 / 正文（双击这一行也一样）", self.on_modify)
        row.addWidget(self.btn_modify)
        # 「停用 / 启用」按钮：文字跟着选中那一份的使用状况变，一眼看清点下去会发生什么
        self.btn_toggle = self._button(
            "停用", "发 AI 时「使用中」的每一份都会带上，停用的一份都不发；"
                    "一份都不启用就这一轮不发 AI", self.on_toggle_enabled)
        row.addWidget(self.btn_toggle)
        self.btn_more = QPushButton("更多 ▾")
        self.btn_more.setToolTip("复制 / 设为默认 / 恢复 / 删除 / 刷新 —— 不常用的都收在这儿")
        self._more_menu = QMenu(self.btn_more)
        for title, tip, slot in (
                ("复制", "复制一份档案（默认指同一个文件），原件不动", self.on_copy),
                ("设为默认", "没指定 PRM 时就用它", self.on_default),
                ("恢复", "把软删的捞回来", self.on_restore),
                ("删除", "软删；历史成品照旧查得到用的是它", self.on_delete),

                ("刷新", "重新查库", self.reload)):
            action = self._more_menu.addAction(title, slot)
            action.setToolTip(tip)
        self.btn_more.setMenu(self._more_menu)
        row.addWidget(self.btn_more)
        row.addStretch(1)
        return row


    def _sync_toggle_button(self) -> None:
        """按选中那一份的使用状况给按钮换字：在用的给「停用」，停用的给「启用」。"""
        button = getattr(self, "btn_toggle", None)
        if button is None:
            return
        state = self._states.get(self.selected() or -1)
        if state is None:
            button.setEnabled(False)
            return
        enabled, deleted = state
        button.setText("启用" if not enabled else "停用")
        button.setEnabled(not deleted)      # 软删的先恢复再谈发不发

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
        self._states = {}
        if db is None:
            return
        made = db_assets.product_counts_for_prms(db)    # 一条 SQL，不逐行查成品
        for row in db_assets.list_prms(db, include_deleted=self.chk_all.isChecked()):
            key = self.edit_search.text().strip().lower()
            if key and key not in f"{row['name']} {row['filename']}".lower():
                continue
            prm_id = int(row["id"])
            deleted = bool(row["deleted_at"])
            enabled = bool(int(row["enabled"] or 0))
            line = self.table.rowCount()
            self.table.insertRow(line)
            self.table.setItem(line, 0, _num(prm_id))
            name = _cell(row["name"])
            # 来源文件 / 语言 / 版本都在右边详情里，表里只挂个 tooltip，不占一列
            name.setToolTip(self._row_tip(row))
            self.table.setItem(line, 1, name)
            self.table.setItem(line, 2, _num(made.get(prm_id, 0)))
            self.table.setItem(line, 3, _cell(_short_time(row["updated_at"]), center=True))
            self.table.setItem(line, 4, _cell(self._row_state(row), center=True))
            self._states[prm_id] = (enabled, deleted)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # 空状态说人话：不留一张白表让用户猜（Phase 16）
        self.lbl_empty.setVisible(not self.table.rowCount())
        self.select(keep)
        self._sync_toggle_button()

    def _row_state(self, row) -> str:
        """一格说清这一份的处境：删没删、发不发、是不是默认、正文有没有。"""
        if row["deleted_at"]:
            return "已删除"
        parts = []
        if int(row["is_default"] or 0):
            parts.append("★ 默认")
        # 「使用中」= 发 AI 时会带上这一份；停用的一份都不发
        parts.append("✓ 使用中" if int(row["enabled"] or 0) else "停用")
        if not row["content"]:
            path = db_assets.prm_file(row, self.cfg.root)
            parts.append("正文待导入" if path is not None and path.is_file() else "⚠ 没有正文")
        return " · ".join(parts)

    def _row_tip(self, row) -> str:
        """鼠标停在名称上时给出来源文件、语言、版本这些次要信息。"""
        text = str(row["content"] or "")
        lines = [f"来源文件：{row['filename']}",
                 f"语言：{row['language'] or '—'}　版本：{row['version'] or '—'}",
                 f"正文：{len(text)} 字（存在数据库里）" if text else "正文：库里还没有"]
        return "\n".join(lines)

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
        """选中一行：只同步按钮状态。老库里正文还在文件里的，顺手导进库一次。"""
        db = self._handle()
        prm_id = self.selected()
        self._sync_toggle_button()
        if db is None or prm_id is None:
            return
        row = db_assets.get_prm(db, prm_id)
        if row is None or row["content"]:
            return
        if db_assets.prm_text(db, prm_id, self.cfg.root):
            self.reload()          # 正文刚自愈进库，状态列那句提示得跟着消失

    def _note(self, text: str, flash: str | None = None) -> None:

        if self._log:
            self._log(text)
        if flash:
            self.notice.emit(flash)      # 顶部直接给一句反馈，不用去翻日志
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

    def _dialog(self, row: Any = None) -> PrmEditDialog:
        """建一个编辑弹窗（测试里把 exec_ 换掉就能不弹窗跑完整条路）。"""
        return PrmEditDialog(self.cfg, self, row=row)

    def on_new(self) -> None:
        """新增：弹窗里填名称 + 导入/手写正文，点保存才写库。"""
        db = self._handle()
        if db is None:
            return
        dlg = self._dialog()
        if dlg.exec_() != QDialog.Accepted:
            return
        name, source, text = dlg.payload()
        if db_assets.prm_name_taken(db, name):
            QMessageBox.information(self, "PRM", f"已经有一份叫「{name}」了，换个名字")
            return
        prm_id = db_assets.create_prm(db, name, source, content=text)
        self._note(f"[PRM] 已登记 #{prm_id} {name}（正文 {len(text)} 字存在库里）",
                   f"✓ 已新增 PRM：{name}")
        self.reload()
        self.select(prm_id)

    def on_modify(self) -> None:
        """修改：弹窗里改名称 / 来源 / 正文，点保存一次写回库（只动库，不碰文件）。"""
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        row = db_assets.get_prm(db, prm_id)
        if row is None:
            return
        if not row["content"]:
            # 老库那份正文还在文件里：先导进库，弹窗里才有东西给你改
            db_assets.prm_text(db, prm_id, self.cfg.root)
            row = db_assets.get_prm(db, prm_id)
        dlg = self._dialog(row)
        if dlg.exec_() != QDialog.Accepted:
            return
        name, source, text = dlg.payload()
        if db_assets.prm_name_taken(db, name, except_id=prm_id):
            # 库里名字唯一（idx_prm_name_live），撞名直接说清楚，别让改名静悄悄失败
            QMessageBox.information(self, "PRM", f"已经有一份叫「{name}」了，换个名字")
            return
        ok = db_assets.update_prm(db, prm_id, name=name, filename=source, content=text)
        self._note(f"[PRM] #{prm_id} {'已更新' if ok else '没改动'}"
                   f"（正文 {len(text)} 字）",
                   f"✓ 已保存：{name}" if ok else "没改动")
        self.reload()
        self.select(prm_id)

    def on_copy(self) -> None:

        got = self._need()
        if got is None:
            return
        db, prm_id = got
        new_id = db_assets.copy_prm(db, prm_id)
        self._note(f"[PRM] #{prm_id} 已复制成 #{new_id}（原件没动）", "✓ 已复制 PRM")
        self.reload()
        self.select(new_id)

    def on_default(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        if db_assets.set_default_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已设为默认", "✓ 已设为默认 PRM")
        else:
            QMessageBox.information(self, "PRM", "已删除的 PRM 不能设为默认")
        self.reload()

    def on_toggle_enabled(self) -> None:
        """启用 / 停用这一份 PRM：**发 AI 时启用的每一份都会带上，停用的一份都不发。**

        一份都不启用就等于「这一轮不发 AI」——那是用户主动选的，界面只提醒一句。
        """
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        row = db_assets.get_prm(db, prm_id)
        if row is None:
            return
        if row["deleted_at"]:
            QMessageBox.information(self, "PRM", "已删除的 PRM 不参与发送，先「恢复」再启用")
            return
        want = not int(row["enabled"] or 0)
        if db_assets.set_prm_enabled(db, prm_id, want):
            live = len(db_assets.enabled_prms(db))
            self._note(f"[PRM] #{prm_id}「{row['name']}」已{'启用' if want else '停用'}"
                       f"（现在使用中 {live} 份，发 AI 会带上这几份）",
                       f"✓ 已{'启用' if want else '停用'}，使用中 {live} 份"
                       + ("；一份都没启用，这一轮不会发 AI" if not live else ""))
        self.reload()
        self.select(prm_id)

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
            self._note(f"[PRM] #{prm_id} 已软删", "✓ 已移入回收状态")
        self.reload()

    def on_restore(self) -> None:
        got = self._need()
        if got is None:
            return
        db, prm_id = got
        if db_assets.restore_prm(db, prm_id):
            self._note(f"[PRM] #{prm_id} 已恢复", "✓ 已恢复")
        else:
            QMessageBox.information(self, "PRM", "这一份本来就没删")
        self.reload()
        self.select(prm_id)

    # ------------------------------------------------------------ 单击 / 双击 / 右键
    def on_double_click(self) -> None:
        """双击一行 = 修改这一份（弹出编辑窗口）。"""
        if self.selected() is None:
            return
        self.on_modify()

    def on_menu(self, pos) -> None:
        """右键 PRM：修改 / 新增、复制、设为默认、恢复/删除、复制正文、打开来源文件。

        已经是默认的那一份不再显示「设为默认」，而是显示一条灰的「★ 默认 PRM」；
        已删除的显示「恢复」，在用的显示「删除」——菜单永远只给现在能做的动作。
        """
        line = self.table.rowAt(pos.y())
        if line >= 0:
            self.table.selectRow(line)
        db = self._handle()
        prm_id = self.selected()
        if db is None or prm_id is None:
            return
        row = db_assets.get_prm(db, prm_id)
        if row is None:
            return
        menu = QMenu(self)
        menu.addAction("修改", self.on_modify)
        menu.addAction("新增", self.on_new)
        menu.addAction("复制", self.on_copy)
        menu.addSeparator()

        # 使用状况：发 AI 时带哪几份就看这里（停用的一份都不发）
        menu.addAction("停用（不发给 AI）" if int(row["enabled"] or 0) else "启用（发给 AI）",
                       self.on_toggle_enabled)
        if int(row["is_default"] or 0):
            star = menu.addAction("★ 默认 PRM")
            star.setEnabled(False)
        else:
            menu.addAction("设为默认", self.on_default)
        if row["deleted_at"]:
            menu.addAction("恢复", self.on_restore)
        else:
            menu.addAction("删除", self.on_delete)
        menu.addSeparator()
        menu.addAction("复制提示词正文", self.on_copy_text)
        menu.addAction("打开提示词文件", self.on_open_file)
        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def on_copy_text(self) -> None:
        """把库里的提示词正文复制到剪贴板（不改文件）。"""
        db = self._handle()
        prm_id = self.selected()
        if db is None or prm_id is None:
            QMessageBox.information(self, "PRM", "先选一份 PRM")
            return
        text = db_assets.prm_text(db, prm_id, self.cfg.root) or ""
        QApplication.clipboard().setText(text)
        if self._log:
            self._log(f"[PRM] 提示词正文已复制到剪贴板（{len(text)} 字）")
        self.notice.emit("✓ 已复制提示词正文")

    def on_open_file(self) -> None:
        """打开这份 PRM 的来源文件（不在盘上就直说，绝不悄悄新建）。"""
        path = self._path()
        if path is None:
            QMessageBox.information(self, "PRM", "先选一份 PRM")
            return
        if not path.is_file():
            QMessageBox.information(self, "PRM", f"文件已经不在盘上：\n{path}")
            return
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - 打开用户自己的提示词文件
            return
        QMessageBox.information(self, "PRM", str(path))


# ================================================================ 视频资产页
class VideoAssetsPage(QWidget):
    """左边视频列表，右边这个视频的高光 JSON 和成品血缘。"""

    VIDEO_HEADERS = ("ID", "✓", "视频", "目录", "时长", "分析", "JSON", "高光", "成品",
                     "AI / 模型", "更新时间")
    CHECK_COLUMN = 1        # 勾选框那一列（缩略图左边）
    NAME_COLUMN = 2         # 视频名那一列：缩略图挂在它上面
    DIR_COLUMN = 3          # 这个文件所在的目录：扫描目录下有几十个子目录时一眼看出来源
    JSON_COLUMN = 6         # 点这一格 = 开「③ 高光 JSON」弹窗
    PRODUCT_COLUMN = 8      # 点这一格 = 开「④ 成品与血缘」弹窗
    # JSON 表：状态、名字、自己的方案名、区间、评分、AI、模型、成品、创建时间都摆出来，
    # 主键只藏在 UserRole 里（界面上不出现裸 ID 列）。
    ASSET_HEADERS = ("当前", "高光 JSON", "名称", "区间", "时长", "高光数", "评分",
                     "AI", "模型", "成品", "创建时间", "状态")
    PRODUCT_HEADERS = ("ID", "成品", "来源 JSON", "PRM", "时长", "实际区间", "生成时间", "状态")
    THUMB_SIZE = QSize(96, 54)     # 视频列表 / 成品表每行左边那张缩略图（16:9）
    THUMB_BATCH = 8                # 一轮最多解几帧：滚动时不许把界面按住


    changed = pyqtSignal()
    notice = pyqtSignal(str)          # 操作结果一句人话（顶部反馈，不用翻日志）
    focus_changed = pyqtSignal(str)   # 「当前：视频 X · 高光 JSON #Y · N 个成品」
    checks_changed = pyqtSignal(int)  # 勾选了几个（底部批量按钮跟着灰/亮）

    def __init__(self, cfg: Any, window: Any = None, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = window
        self._db: Any = None
        self._rows: list[dict[str, Any]] = []
        self._asset_rows: list[Any] = []          # 当前视频的 JSON 行（刷新时手上就有）
        self._product_rows: list[dict[str, Any]] | None = None   # 当前视频的成品全景
        # 详情区（当前视频 / 高光 JSON / 成品 / 血缘）现在画的是哪个视频。
        # 列表重画后行号会留在原地但那一行已经换成别的视频了，Qt 这种情况不发
        # itemSelectionChanged，光靠信号刷详情会拿到上一个视频的成品 —— 靠这个字段兜住。
        self._shown_video: int | None = None
        self._lineage_for: int | None = None      # 血缘树现在画的是哪个成品
        self._thumbs: dict[str, QIcon | None] = {}   # 视频路径 → 缩略图（一个视频只解一次帧）
        self._checked: set[int] = set()            # 勾上的视频 id（换筛选也不丢）

        # 页面只剩「① 视频库」一栏：以前右边那半（当前视频 / 高光 JSON / 成品血缘）
        # 要挤在 400~600px 里，四块表格叠在一起怎么排都难看。现在整块搬进两个弹窗，
        # 入口全在视频列表的右键菜单里。**控件一个没换**，刷新逻辑照旧写这些控件。
        self._build_detail_windows()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self._build_left(), 1)
        self.reload()

    # ------------------------------------------------------------ 左：视频库
    def _build_left(self) -> QWidget:
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔎 搜索文件名 / 视频 ID")
        self.edit_search.setClearButtonEnabled(True)
        self.edit_search.returnPressed.connect(self.reload)
        self.edit_search.textChanged.connect(self._search_changed)

        self.cmb_status = QComboBox()
        self.cmb_status.setToolTip("状态：全部 / 已分析 / 未分析（有没有做过视觉 / 语音分析）")
        for title, key in STATUS_CHOICES:
            self.cmb_status.addItem(title, key)
        self.cmb_json = QComboBox()
        self.cmb_json.setToolTip("JSON：全部 / 有 JSON / 无 JSON"
                                 "（和「成品」是两个独立条件，可以一起用）")
        for title, key in JSON_CHOICES:
            self.cmb_json.addItem(title, key)
        self.cmb_product = QComboBox()
        self.cmb_product.setToolTip("成品：全部 / 有成品 / 无成品；"
                                    "「JSON = 有」+「成品 = 无」就是还没剪的那些")
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
            # 六个下拉按「4 个字」算最小宽度：以前每个都按最长选项撑，左栏最小宽被顶到 620，
            # 加上右侧工作区，整页最小宽超过 1500，1240 甚至 1000 的窗口里两边都会被裁
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(3)
            widget.currentIndexChanged.connect(lambda _=0: self.reload())

        # 两个目录筛选：扫描目录下常常有几十个子目录，得能只看其中一个。
        # 选完即存（写 assets.filter_video_dir / filter_product_dir），
        # 而且「看全部视频（清掉筛选）」**不会**把它们清掉——那是手动挑的作用域。
        self.cmb_video_dir = QComboBox()
        self.cmb_video_dir.setToolTip("原视频目录：只看原视频落在这个目录（含子目录）里的。"
                                      "选完即存，清筛选也不会动它")
        self.cmb_product_dir = QComboBox()
        self.cmb_product_dir.setToolTip("成品目录：只看**成品**落在这个目录（含子目录）里的。"
                                        "选完即存，清筛选也不会动它")
        for widget in (self.cmb_video_dir, self.cmb_product_dir):
            # 目录字符串很长，这里也按「3 个字」算最小宽（整页最小宽必须守住 960）
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(3)
            widget.currentIndexChanged.connect(lambda _=0: self._pick_dir_filter())

        # 六个筛选控件摆成两行三列：一排横着摆在窄窗口下会把下拉挤出可视区
        filters = QGridLayout()
        filters.setHorizontalSpacing(6)
        filters.setVerticalSpacing(4)
        for column, (title, widget) in enumerate((
                ("状态", self.cmb_status), ("JSON", self.cmb_json),
                ("成品", self.cmb_product))):
            filters.addWidget(QLabel(title), 0, column * 2)
            filters.addWidget(widget, 0, column * 2 + 1)
        for column, (title, widget) in enumerate((
                ("AI", self.cmb_ai), ("排序", self.cmb_order), ("条数", self.cmb_page))):
            filters.addWidget(QLabel(title), 1, column * 2)
            filters.addWidget(widget, 1, column * 2 + 1)
        filters.addWidget(QLabel("原视频目录"), 2, 0)
        filters.addWidget(self.cmb_video_dir, 2, 1, 1, 5)
        filters.addWidget(QLabel("成品目录"), 3, 0)
        filters.addWidget(self.cmb_product_dir, 3, 1, 1, 5)
        for column in (1, 3, 5):
            filters.setColumnStretch(column, 1)


        self.tbl_videos = _plain_table(self.VIDEO_HEADERS, stretch=self.NAME_COLUMN)
        self.tbl_videos.setColumnHidden(0, True)          # id 留着取值，不占版面
        self.tbl_videos.setMinimumWidth(240)
        self.tbl_videos.setToolTip("双击 = 播放视频；点「JSON」那一格 = 开高光 JSON 弹窗，"
                                   "点「成品」那一格 = 开成品与血缘弹窗；"
                                   "右键 = 这个视频的其余操作（筛选 / 删除 / 打开目录）；"
                                   "左边的勾选框配底部那排批量按钮用")
        # 勾选列窄窄一条，宽度不跟内容变（勾选框就在缩略图左边）
        head = self.tbl_videos.horizontalHeader()
        head.setSectionResizeMode(self.CHECK_COLUMN, QHeaderView.Fixed)
        self.tbl_videos.setColumnWidth(self.CHECK_COLUMN, 28)
        # 勾选框不给 ItemIsUserCheckable：自己接 itemClicked 翻转，免得和编辑触发器打架。
        # 同一个入口还管「点 JSON / 成品那一格就开对应弹窗」
        self.tbl_videos.itemClicked.connect(self._on_cell_clicked)
        # 每行左边挂视频缩略图：图标尺寸和行高得配套，不然图会被压扁
        self.tbl_videos.setIconSize(self.THUMB_SIZE)
        self.tbl_videos.verticalHeader().setDefaultSectionSize(self.THUMB_SIZE.height() + 8)
        # 缩略图是滚到哪解到哪：滚动条一动就补当前可见的那几行
        self.tbl_videos.verticalScrollBar().valueChanged.connect(
            lambda _=0: self.paint_visible_thumbs())
        head.setSortIndicatorShown(True)
        head.setSectionsClickable(True)
        head.sectionClicked.connect(self.on_header_sort)
        self.tbl_videos.itemSelectionChanged.connect(self.on_video_changed)
        self.tbl_videos.doubleClicked.connect(lambda _=None: self.on_play_video())
        self.tbl_videos.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_videos.customContextMenuRequested.connect(self.on_video_menu)

        self.lbl_count = QLabel("—")
        # 这一行会写很长的操作提示：**必须换行**，不然它的最小宽度会把整页顶宽，
        # 窄窗口下列表被裁（test_center_fits_its_own_minimum_window 盯着这条）
        self.lbl_count.setWordWrap(True)

        box = QGroupBox("① 视频库")
        lay = QVBoxLayout(box)
        lay.addWidget(self.edit_search)
        lay.addLayout(filters)
        lay.addWidget(self.tbl_videos, 1)
        lay.addWidget(self.lbl_count)
        return box

    # ---------------------------------------------------- 详情：两个独立弹窗
    def _build_detail_windows(self) -> None:
        """把原来右半边整块搬进两个弹窗：③ 高光 JSON（含②当前视频卡片）、④ 成品与血缘。

        只换了容器，控件本身一个都没重建：`tbl_assets` / `json_panel` /
        `tbl_products` / `tree_lineage` 还是同一批对象，`refresh_*` 一行都不用改。
        弹窗是非模态的，开着也能继续在视频列表里换视频，内容跟着换。
        """
        json_body = QWidget()
        lay = QVBoxLayout(json_body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._build_hero())
        lay.addWidget(self._build_assets(), 1)
        self.dlg_json = self._detail_window("③ 高光 JSON", json_body, 1000, 620)
        self.dlg_products = self._detail_window("④ 成品与血缘", self._build_products(),
                                                900, 560)

    def _detail_window(self, title: str, body: QWidget, width: int, height: int) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{title} — 视频资产中心")
        dlg.setWindowFlags(Qt.Window)      # 独立窗口：能挪、能最大化、不挡主列表
        dlg.setSizeGripEnabled(True)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(body, 1)
        dlg.resize(width, height)
        return dlg

    def _popup(self, dlg: QDialog, title: str) -> None:
        row = self._row_of(self.current_video_id())
        name = str(row["file_name"]) if row else "未选视频"
        dlg.setWindowTitle(f"{title} — {name}")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _build_hero(self) -> QWidget:
        self.lbl_step = QLabel("② 当前视频")      # 层级标号：列表是索引，弹窗是工作区
        self.lbl_video = _title("请选择一个视频")
        font = self.lbl_video.font()
        font.setPointSize(max(9, font.pointSize()) + 6)   # 没有 QSS 时也要大一号
        self.lbl_video.setFont(font)
        # 程序的主题 QSS 里有一条 `QWidget { font-size: 12px }`，它会盖掉上面的 pointSize；
        # 直接给这个 label 自己写一条更具体的样式，保证「当前视频」在真程序里也是最大的字。
        self.lbl_video.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.lbl_video.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.lbl_state = QLabel("在「① 视频库」里选一个视频，这个弹窗就是它的工作区")
        self.lbl_state.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.lbl_meta = QLabel("")                   # 统计行：JSON ｜ 高光 ｜ 成品
        self.lbl_meta.setWordWrap(True)
        stats = self.lbl_meta.font()
        stats.setPointSize(stats.pointSize() + 1)
        stats.setBold(True)
        self.lbl_meta.setFont(stats)
        self.lbl_meta.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # 当前视频 = 这个弹窗唯一的视觉焦点：单独一块卡片，标题字号最大
        hero = QFrame()
        hero.setFrameShape(QFrame.StyledPanel)
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(8, 2, 8, 2)
        hero_lay.setSpacing(2)
        hero_lay.addWidget(self.lbl_step)
        hero_lay.addWidget(self.lbl_video)
        hero_lay.addWidget(self.lbl_state)
        hero_lay.addWidget(self.lbl_meta)
        self.box_current = hero
        return hero

    def _build_assets(self) -> QWidget:
        # 高频动作只留两个：主按钮「直接剪辑」+ 普通按钮「查看」，
        # 两个都摆在「③ 高光 JSON」那一区里 —— 它们作用的对象就是选中的那份 JSON。
        # 「查看原视频 / 打开成品」这类定位动作一律进「更多 ▾」和右键菜单，不抢版面。
        self.btn_render = _primary(QPushButton("直接剪辑"))   # 唯一主动作，唯一加粗
        self.btn_render.setToolTip("用选中的这份高光 JSON 出成品：选个 PRM 就开剪，一次 AI 都不调")
        self.btn_render.clicked.connect(self.on_render)
        self.btn_view = QPushButton("查看")
        self.btn_view.setToolTip("看选中 JSON 的每一段区间（右边直接显示，不弹窗）")
        self.btn_view.clicked.connect(self.on_view)

        self.lbl_current = QLabel("当前 JSON：—")
        # 这一行的内容会随视频名 / 备注变长；不换行的话它的最小宽度会把整个右侧工作区顶宽，
        # 窄窗口下就变成"两栏都被裁、左边视频列表看不全"
        self.lbl_current.setWordWrap(True)
        self.chk_current_only = QCheckBox("只看当前")
        self.chk_current_only.setToolTip("默认展开全部高光 JSON；勾上只看当前那一份")
        self.chk_current_only.stateChanged.connect(lambda _=0: self.refresh_assets())
        # 「含已删除」属于低频，收进「更多 ▾」，这里只留状态（不进版面）
        self.chk_deleted = QCheckBox("含已删除")
        self.chk_deleted.setVisible(False)
        self.chk_deleted.stateChanged.connect(lambda _=0: self.refresh_assets())

        self.tbl_assets = _plain_table(self.ASSET_HEADERS, stretch=1)
        self.tbl_assets.setMinimumHeight(66)
        self.tbl_assets.itemSelectionChanged.connect(self.on_asset_changed)
        self.tbl_assets.doubleClicked.connect(lambda _=None: self.on_view())
        self.tbl_assets.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_assets.customContextMenuRequested.connect(self.on_asset_menu)

        self.json_panel = JsonPanel(self.cfg, self, log=self._log)
        self.json_panel.saved.connect(self.on_json_saved)

        more = QHBoxLayout()
        more.addWidget(self.btn_render)        # 唯一 Primary，就在 JSON 列表旁边
        more.addWidget(self.btn_view)
        self.btn_more = QPushButton("更多 ▾")
        self.btn_more.setToolTip("低频动作都在这里：编辑 / 复制 / 设为当前 / 导入 / 删除 / 恢复 / "
                                 "原文 / 定位文件")
        menu = QMenu(self.btn_more)
        menu.addAction("编辑（保存会新建一份）", self.on_edit_json)
        menu.addAction("复制这份 JSON", self.on_copy)
        menu.addAction("设为当前 JSON", self.on_set_current)
        menu.addSeparator()
        menu.addAction("导入现成 JSON…", self.on_import)
        menu.addAction("删除（软删）", self.on_delete)
        menu.addAction("恢复已删除", self.on_restore)
        menu.addSeparator()
        menu.addAction("查看原视频", self.on_reveal_video)
        menu.addAction("打开成品", self.on_reveal)
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
        inner.setChildrenCollapsible(False)
        inner.setSizes([420, 320])
        self.split_assets = inner

        box = QGroupBox("③ 高光 JSON")
        # 组框标题不会被省略，写长了会把整块的最小宽度顶上去（窄窗口下右侧内容被裁），
        # 所以说明只留在 tooltip 里
        box.setToolTip("选中一份，上面的「直接剪辑」就用它；双击 = 查看，右键 = 更多")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(inner, 1)
        return box

    def _build_products(self) -> QWidget:
        self.tbl_products = _plain_table(self.PRODUCT_HEADERS, stretch=1)
        self.tbl_products.setColumnHidden(0, True)
        self.tbl_products.setMinimumHeight(60)
        # 每行左边挂当前视频的缩略图：行高和图标尺寸得配套，不然图会被压扁
        self.tbl_products.setIconSize(self.THUMB_SIZE)
        self.tbl_products.verticalHeader().setDefaultSectionSize(self.THUMB_SIZE.height() + 8)
        self.tbl_products.itemSelectionChanged.connect(self.refresh_lineage)
        self.tbl_products.doubleClicked.connect(lambda _=None: self.on_open_product())
        self.tbl_products.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_products.customContextMenuRequested.connect(self.on_product_menu)

        self.lbl_product_head = QLabel("—")
        self.lbl_product_head.setWordWrap(True)
        self.lbl_product = QLabel("选一个成品")
        self.lbl_product.setWordWrap(True)
        # 技术细节（分析批次 / AI 任务 / Engine 原因 / 路径）默认收起，勾上才进血缘树
        self.chk_details = QCheckBox("详细信息")
        self.chk_details.setToolTip("展开分析批次、AI 任务 ID、Engine 修正原因这些技术细节")
        self.chk_details.stateChanged.connect(lambda _=0: self.refresh_lineage())
        self.tree_lineage = QTreeWidget()
        self.tree_lineage.setHeaderLabels(["血缘", "内容"])
        self.tree_lineage.setColumnWidth(0, 210)
        self.tree_lineage.setAlternatingRowColors(True)
        self.tree_lineage.setMinimumHeight(60)
        self.tree_lineage.setToolTip("点「高光 JSON」「PRM」或「实际成品」节点，"
                                     "对应的表格 / 页签会自动跳过去")
        self.tree_lineage.itemClicked.connect(self.on_lineage_clicked)


        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(self.lbl_product, 1)
        head.addWidget(self.chk_details)
        right_lay.addLayout(head)
        right_lay.addWidget(self.tree_lineage, 1)


        inner = QSplitter(Qt.Horizontal)
        inner.addWidget(self.tbl_products)
        inner.addWidget(right)
        inner.setStretchFactor(0, 3)
        inner.setStretchFactor(1, 2)
        inner.setChildrenCollapsible(False)
        inner.setSizes([420, 320])
        self.split_products = inner

        box = QGroupBox("④ 成品")
        box.setToolTip("双击 = 打开成品，右键 = 追溯来源 JSON / PRM / 血缘")
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
        self._fill_dir_choices(db)
        try:
            self._rows = db_assets.center_rows(
                db, search=self.edit_search.text().strip() or None,
                provider=self.cmb_ai.currentData() or None,
                status=str(self.cmb_status.currentData() or "all"),
                json=str(self.cmb_json.currentData() or "any"),
                product=str(self.cmb_product.currentData() or "any"),
                video_dir=str(self.cmb_video_dir.currentData() or "") or None,
                product_dir=str(self.cmb_product_dir.currentData() or "") or None,
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
            box = _cell("")
            box.setCheckState(Qt.Checked if int(item["id"]) in self._checked
                              else Qt.Unchecked)
            folder = _cell(str(Path(str(item["file_path"])).parent))
            folder.setToolTip(str(item["file_path"]))     # 悬停看完整路径
            self.tbl_videos.setItem(line, 0, _num(item["id"]))
            self.tbl_videos.setItem(line, self.CHECK_COLUMN, box)
            self.tbl_videos.setItem(line, self.NAME_COLUMN, _cell(item["file_name"]))
            self.tbl_videos.setItem(line, self.DIR_COLUMN, folder)
            self.tbl_videos.setItem(line, 4, _duration_cell(item["duration"]))
            self.tbl_videos.setItem(line, 5, _cell("✓" if item["analysed"] else "✗",
                                                   center=True))
            self.tbl_videos.setItem(line, self.JSON_COLUMN, _num(item["json_count"]))
            self.tbl_videos.setItem(line, 7, _num(item["highlight_count"]))
            self.tbl_videos.setItem(line, self.PRODUCT_COLUMN, _num(item["product_count"]))
            self.tbl_videos.setItem(line, 9, _cell(ai))
            self.tbl_videos.setItem(line, 10, _cell(_short_time(item["updated_at"]),
                                                    center=True))
        self.tbl_videos.resizeColumnsToContents()
        self.tbl_videos.setColumnWidth(self.CHECK_COLUMN, 28)
        # 目录列按内容能撑到几百像素，压到 220：视频名那一列才是主角
        self.tbl_videos.setColumnWidth(
            self.DIR_COLUMN, min(220, self.tbl_videos.columnWidth(self.DIR_COLUMN)))
        self.tbl_videos.horizontalHeader().setSectionResizeMode(
            self.NAME_COLUMN, QHeaderView.Stretch)
        self._sync_sort_indicator()
        self.tbl_videos.blockSignals(False)
        self.lbl_count.setText(f"共 {len(self._rows)} 个视频"
                               + ("（已按筛选条件过滤）" if self._filtering() else "")
                               + "　双击 = 播放视频，点 JSON / 成品那一格开弹窗，"
                                 "右键 = 筛选 / 删除；勾选框配底部批量按钮")
        self.select_video(keep)
        self.paint_visible_thumbs()      # 只解看得见的那几行，列表再长也不卡
        self.checks_changed.emit(len(self.checked_ids()))

    def _fill_dir_choices(self, db: Any) -> None:
        """把库里出现过的目录塞进两个目录下拉，并停在配置里存着的那一个。

        目录列表跟着库走（扫过什么就有什么），选中的那一项来自
        `assets.filter_video_dir` / `filter_product_dir`——配置里的目录哪怕这一次
        库里没有，也照样保留成一项，不然一刷新就把用户手动挑的作用域弄丢了。
        """
        try:
            video_dirs, product_dirs = db_assets.known_dirs(db)
        except Exception as exc:  # noqa: BLE001
            self._note(f"[资产中心] 目录列表查不出来：{exc}")
            return
        section = self.cfg.assets
        for combo, folders, key in ((self.cmb_video_dir, video_dirs, "filter_video_dir"),
                                    (self.cmb_product_dir, product_dirs,
                                     "filter_product_dir")):
            picked = str(combo.currentData() or "") or str(section.get(key) or "")
            items = list(folders)
            if picked and picked not in items:
                items.append(picked)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("全部目录", "")
            for folder in items:
                combo.addItem(folder, folder)
            index = combo.findData(picked) if picked else 0
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _pick_dir_filter(self) -> None:
        """挑了目录：立刻存进全局配置（`assets`），再按新作用域刷列表。"""
        patch = {"filter_video_dir": str(self.cmb_video_dir.currentData() or ""),
                 "filter_product_dir": str(self.cmb_product_dir.currentData() or "")}
        try:
            self.cfg.save_patch({"assets": patch})
        except Exception as exc:  # noqa: BLE001 - 存不进去不该打断正在看的列表
            self._note(f"[资产中心] 目录筛选存不进 config.json：{exc}")
        self.reload()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.paint_visible_thumbs()      # 窗口拉高会露出更多行，补上它们的缩略图

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
                    or str(self.cmb_status.currentData() or "all") != "all"
                    or self.cmb_video_dir.currentData()
                    or self.cmb_product_dir.currentData())

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
            self._ensure_details()
        else:
            self.on_video_changed()

    def _ensure_details(self) -> None:
        """详情区跟当前选中的视频对上号；对不上就当场重画。

        列表重画（换筛选 / 排序 / 刷新）后选中的行号常常不变，但那一行已经是**另一个
        视频**了；Qt 这时不发 itemSelectionChanged，详情区（尤其是成品表）会留在上一个
        视频上 —— 那样点「打开成品」打开的就是别人的成品文件。
        """
        if self._shown_video != self.current_video_id():
            self.on_video_changed()

    def on_video_changed(self) -> None:
        vid = self.current_video_id()
        db = self._handle()
        self._shown_video = vid          # 详情区从这一刻起画的是这个视频
        self._product_rows = None        # 换视频 = 成品缓存作废，下一次刷新重新查
        if vid is None or db is None:
            self.lbl_video.setText("请选择一个视频")
            self.lbl_state.setText("回到「① 视频库」点一个视频，这个弹窗就是它的工作区")
            self.lbl_meta.setText("JSON　—　｜　高光　—　｜　成品　—")
            self.tbl_assets.setRowCount(0)
            self.tbl_products.setRowCount(0)
            self.json_panel.clear("请先选择一个视频")
            self.tree_lineage.clear()
            self.tree_lineage.addTopLevelItem(QTreeWidgetItem(
                ["血缘", "选一个视频，再选它的成品，这里显示完整来源"]))
            self._emit_focus()
            return
        row = next((r for r in self._rows if r["id"] == vid), None)
        self.lbl_video.setText(f"{row['file_name'] if row else f'视频 #{vid}'}")
        analysis = db_repo.latest_analysis(db, vid)
        state = f"{_seconds(row['duration']) if row else '—'}　　" \
                f"{'已分析' if analysis is not None else '未分析'}"
        # 完整路径不写进标题区（一条长路径会把整块最小宽度顶到 1300+），
        # 放进 tooltip，右键「复制视频路径 / 打开所在文件夹」照旧能用
        self.lbl_state.setText(state)
        self.lbl_state.setToolTip(str(row["file_path"]) if row and row["file_path"] else "")
        self.lbl_meta.setText("　｜　".join((
            f"JSON　{row['json_count'] if row else 0}",
            f"高光　{row['highlight_count'] if row else 0}",
            f"成品　{row['product_count'] if row else 0}")))
        self.refresh_assets()
        self.refresh_products(reload=False)   # 上面那一步已经把成品缓存填好了
        self._emit_focus()



    def on_open_video(self) -> None:
        """打开「③ 高光 JSON」弹窗（右键菜单里的「查看高光 JSON」）。"""
        if self.current_video_id() is None:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        self._ensure_details()      # JSON 表也一样，得是当前这个视频的
        self._popup(self.dlg_json, "③ 高光 JSON")
        if self.tbl_assets.rowCount() and self.selected_asset() is None:
            self.tbl_assets.selectRow(0)
        self.tbl_assets.setFocus()

    # ------------------------------------------------------------ 右键：视频
    def _row_of(self, video_id: int | None) -> dict[str, Any] | None:
        return next((r for r in self._rows if int(r["id"]) == int(video_id)), None) \
            if video_id is not None else None

    def _center(self):
        """往上找资产中心本体（切 PRM 页要用它）；找不到就返回 None。"""
        node = self.parent()
        while node is not None:
            if hasattr(node, "show_prm"):
                return node
            node = node.parent()
        return None

    def on_video_menu(self, pos) -> None:
        """右键视频：这里就是视频页的全部操作入口——详情弹窗、筛选、删除。"""
        line = self.tbl_videos.rowAt(pos.y())
        if line >= 0:
            self.tbl_videos.selectRow(line)
        row = self._row_of(self.current_video_id())
        if row is None:
            return
        menu = QMenu(self)
        menu.addAction("播放视频（双击同效）", self.on_play_video)
        menu.addSeparator()
        # 「查看高光 JSON」「查看成品与血缘」不在菜单里了：直接点那一行的
        # 「JSON」/「成品」格子就开对应弹窗，少一层菜单
        menu.addAction("只看这个视频的高光", self.on_only_json)
        menu.addAction("只看这个视频的成品", self.on_only_products)
        menu.addAction("看全部视频（清掉筛选）", self.on_clear_filters)
        menu.addSeparator()
        menu.addAction("打开所在文件夹", self.on_reveal_video)
        menu.addAction("复制视频路径", self.on_copy_video_path)
        menu.addSeparator()
        menu.addAction("从库里删除这个视频（不删文件）", self.on_forget_video)
        menu.addAction("删除该视频（包含本地文件）", self.on_delete_video_file)
        menu.exec_(self.tbl_videos.viewport().mapToGlobal(pos))

    def on_clear_filters(self) -> None:
        """把状态 / JSON / 成品 / AI 四个筛选和搜索框复位。

        **两个目录筛选故意不动**：那是手动挑的作用域（存在全局配置里），
        清筛选只是"这一屏别再挑挑拣拣"，不该把作用域也一起丢掉。
        """
        for combo in (self.cmb_status, self.cmb_json, self.cmb_product, self.cmb_ai):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.edit_search.blockSignals(True)
        self.edit_search.clear()
        self.edit_search.blockSignals(False)
        self.reload()
        kept = self.cmb_video_dir.currentData() or self.cmb_product_dir.currentData()
        self.notice.emit("✓ 已清掉筛选" + ("（目录筛选保留）" if kept else "，显示全部视频"))

    def on_forget_video(self) -> None:
        """把这个视频从库里删掉：磁盘文件一个不动，库里的记录全没。**不可恢复。**"""
        db = self._handle()
        vid = self.current_video_id()
        row = self._row_of(vid)
        if db is None or vid is None or row is None:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        gone = db_repo.video_footprint(db, int(vid))
        detail = (f"分析 {gone['analyses']} 批（画面事件 {gone['events']} 条 / "
                  f"语音 {gone['segments']} 段）\n"
                  f"AI 任务 {gone['tasks']} 条 · AI 回复 {gone['results']} 条 · "
                  f"高光 JSON {gone['assets']} 份 · "
                  f"片段 {gone['clips']} 个 · 文件登记 {gone['artifacts']} 条")
        ask = QMessageBox(QMessageBox.Warning, "从库里删除视频",
                          f"把「{row['file_name']}」从数据库里删掉？\n\n"
                          f"会一起没掉：\n{detail}\n\n"
                          "磁盘上的 mp4、剧本 TXT、成品文件都不会动，\n"
                          "但库里的这些记录**删了就找不回来**。",
                          QMessageBox.Yes | QMessageBox.No, self)
        ask.setDefaultButton(QMessageBox.No)
        if ask.exec_() != QMessageBox.Yes:
            return
        result = db_repo.forget_video(db, int(vid))
        if result is None:
            QMessageBox.information(self, "资产中心", "这个视频已经不在库里了")
            self.reload()
            return
        self._note(f"[资产中心] 已从库里删除 {row['file_name']}（文件没动）："
                   f"分析 {result['analyses']} 批 / 高光 JSON {result['assets']} 份 / "
                   f"片段 {result['clips']} 个 / 文件登记 {result['artifacts']} 条",
                   f"✓ 已从库里删除 {row['file_name']}（文件没动）")
        self.reload()


    def on_play_video(self) -> None:
        """打开视频本体（交给系统默认播放器）。"""
        row = self._row_of(self.current_video_id())
        if row is None or not row["file_path"]:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        path = Path(str(row["file_path"]))
        if not path.is_file():
            QMessageBox.information(self, "资产中心", f"文件已经不在盘上：\n{path}")
            return
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - 打开用户自己的视频
            return
        QMessageBox.information(self, "资产中心", str(path))

    def on_focus_products(self) -> None:
        """打开「④ 成品与血缘」弹窗（右键菜单里的「查看成品」）。"""
        if self.current_video_id() is None:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        self._ensure_details()      # 成品表必须是当前这个视频的，不然打开的是别人的成品
        if not self.tbl_products.rowCount():
            QMessageBox.information(self, "资产中心", "这个视频还没有成品")
            return
        self._popup(self.dlg_products, "④ 成品与血缘")
        if self.selected_product() is None:
            self.tbl_products.selectRow(0)
        self.tbl_products.setFocus()

    def on_only_json(self) -> None:
        """只看这个视频的高光：搜索框锁到它，JSON 筛选切「有 JSON」。"""
        row = self._row_of(self.current_video_id())
        if row is None:
            return
        self.cmb_json.setCurrentIndex(max(0, self.cmb_json.findData("has")))
        self.cmb_product.setCurrentIndex(max(0, self.cmb_product.findData("any")))
        self.edit_search.setText(str(row["file_name"]))
        self.reload()

    def on_only_products(self) -> None:
        """只看这个视频的成品：搜索框锁到它，成品筛选切「有成品」。"""
        row = self._row_of(self.current_video_id())
        if row is None:
            return
        self.cmb_product.setCurrentIndex(max(0, self.cmb_product.findData("has")))
        self.cmb_json.setCurrentIndex(max(0, self.cmb_json.findData("any")))
        self.edit_search.setText(str(row["file_name"]))
        self.reload()

    def on_copy_video_path(self) -> None:
        row = self._row_of(self.current_video_id())
        if row is None or not row["file_path"]:
            return
        QApplication.clipboard().setText(str(row["file_path"]))
        self._note(f"[资产中心] 视频路径已复制：{row['file_path']}", "✓ 已复制视频路径")

    def on_delete_video_file(self) -> None:
        """**连磁盘上的视频文件一起删**，再删库里的全部登记。不可恢复。

        只删视频本体：已经剪出来的成品 mp4、导出的剧本 TXT 都留在盘上
        （它们不是"这个视频"，删掉会连带毁掉别的成果）。
        """
        db = self._handle()
        vid = self.current_video_id()
        row = self._row_of(vid)
        if db is None or vid is None or row is None:
            QMessageBox.information(self, "资产中心", "先选一个视频")
            return
        path = Path(str(row["file_path"])) if row["file_path"] else None
        gone = db_repo.video_footprint(db, int(vid))
        ask = QMessageBox(
            QMessageBox.Critical, "删除视频（包含本地文件）",
            f"要删掉「{row['file_name']}」**本身这个文件**吗？\n\n"
            f"磁盘文件：{path if path else '—'}\n"
            f"库里同时没掉：分析 {gone['analyses']} 批 · AI 任务 {gone['tasks']} 条 · "
            f"高光 JSON {gone['assets']} 份 · 片段 {gone['clips']} 个 · "
            f"文件登记 {gone['artifacts']} 条\n\n"
            "已经剪出来的成品 mp4 和剧本 TXT 会保留。\n"
            "**文件删了进不了回收站，找不回来。**",
            QMessageBox.Yes | QMessageBox.No, self)
        ask.setDefaultButton(QMessageBox.No)
        if ask.exec_() != QMessageBox.Yes:
            return
        if path is not None and path.is_file():
            try:
                os.remove(path)
            except OSError as exc:
                QMessageBox.warning(self, "资产中心",
                                    f"文件删不掉（库里的登记也没动）：\n{exc}")
                return
        db_repo.forget_video(db, int(vid))
        self._checked.discard(int(vid))
        self._note(f"[资产中心] 已删除视频文件并清掉登记：{path}",
                   f"✓ 已删除 {row['file_name']}（文件和登记都没了）")
        self.reload()

    # ------------------------------------------------------- 勾选 + 批量操作
    def _on_cell_clicked(self, item: QTableWidgetItem) -> None:
        """点格子的总入口：勾选列翻转勾，JSON / 成品列直接开对应弹窗，其余不管。

        这两列本来就是「有几份 / 有几个」的数字，点它去看详情最直觉，
        所以右键菜单里那两条「查看…」就撤了。
        """
        column = item.column()
        if column == self.CHECK_COLUMN:
            self._toggle_check(item)
            return
        if column not in (self.JSON_COLUMN, self.PRODUCT_COLUMN):
            return
        self.tbl_videos.selectRow(item.row())      # 弹窗看的是"当前视频"，先选中它
        self._ensure_details()                     # 这一行可能刚被重画换了视频
        if column == self.JSON_COLUMN:
            self.on_open_video()
        else:
            self.on_focus_products()

    def _toggle_check(self, item: QTableWidgetItem) -> None:
        """点勾选列 = 翻转这一行的勾。

        勾选状态记在 `self._checked`（视频 id），不记在表格里：换筛选、翻页、
        刷新之后勾还在，重画时按 id 还原。
        """
        if item.column() != self.CHECK_COLUMN:
            return
        marker = self.tbl_videos.item(item.row(), 0)
        if marker is None or not marker.text():
            return
        vid = int(marker.text())
        if vid in self._checked:
            self._checked.discard(vid)
            item.setCheckState(Qt.Unchecked)
        else:
            self._checked.add(vid)
            item.setCheckState(Qt.Checked)
        self.checks_changed.emit(len(self._checked))

    def checked_ids(self) -> list[int]:
        """当前列表里被勾上的视频 id（按列表顺序，只算这一次筛选看得到的行）。"""
        out: list[int] = []
        for line in range(self.tbl_videos.rowCount()):
            marker = self.tbl_videos.item(line, 0)
            if marker is None or not marker.text():
                continue
            if int(marker.text()) in self._checked:
                out.append(int(marker.text()))
        return out

    def _repaint_checks(self) -> None:
        for line in range(self.tbl_videos.rowCount()):
            box = self.tbl_videos.item(line, self.CHECK_COLUMN)
            marker = self.tbl_videos.item(line, 0)
            if box is None or marker is None or not marker.text():
                continue
            box.setCheckState(Qt.Checked if int(marker.text()) in self._checked
                              else Qt.Unchecked)
        self.checks_changed.emit(len(self._checked))

    def on_check_all(self) -> None:
        """全选：把**当前列表里**的每一行都勾上（筛掉的那些不动）。"""
        for line in range(self.tbl_videos.rowCount()):
            marker = self.tbl_videos.item(line, 0)
            if marker is not None and marker.text():
                self._checked.add(int(marker.text()))
        self._repaint_checks()
        self.notice.emit(f"✓ 已勾选 {len(self.checked_ids())} 个视频")

    def on_invert_checks(self) -> None:
        """反选：当前列表里勾上的取消、没勾的勾上（筛掉的那些不动，文件一个不碰）。

        一个都没勾时按下去就等于全选；全勾着按下去就等于清空，所以「清空」那个键
        没必要单独留。
        """
        for line in range(self.tbl_videos.rowCount()):
            marker = self.tbl_videos.item(line, 0)
            if marker is None or not marker.text():
                continue
            vid = int(marker.text())
            if vid in self._checked:
                self._checked.discard(vid)
            else:
                self._checked.add(vid)
        self._repaint_checks()
        self.notice.emit(f"✓ 反选完成，现在勾着 {len(self.checked_ids())} 个视频")

    def _checked_rows(self) -> list[dict[str, Any]]:
        return [row for row in (self._row_of(vid) for vid in self.checked_ids())
                if row is not None]

    def on_rename_checked(self) -> None:
        """编辑 = 重命名：改磁盘上的文件名，库里的路径跟着改。只勾 1 个时可用。"""
        rows = self._checked_rows()
        if len(rows) != 1:
            QMessageBox.information(self, "资产中心",
                                    f"重命名一次只能改一个（现在勾了 {len(rows)} 个）")
            return
        db = self._handle()
        row = rows[0]
        old = Path(str(row["file_path"]))
        if db is None:
            return
        if not old.is_file():
            QMessageBox.information(self, "资产中心", f"文件已经不在盘上：\n{old}")
            return
        name, ok = QInputDialog.getText(self, "重命名视频", "新的文件名：", text=old.name)
        name = (name or "").strip()
        if not ok or not name or name == old.name:
            return
        if Path(name).name != name:
            QMessageBox.warning(self, "资产中心", "只能改文件名，不要带路径分隔符")
            return
        target = old.with_name(name)
        if target.exists():
            QMessageBox.warning(self, "资产中心", f"这个名字已经被占了：\n{target}")
            return
        try:
            os.replace(old, target)
        except OSError as exc:
            QMessageBox.warning(self, "资产中心", f"改名失败（库里没动）：\n{exc}")
            return
        db_repo.rename_video(db, int(row["id"]), target)
        self._note(f"[资产中心] 已重命名：{old.name} → {target.name}",
                   f"✓ 已重命名为 {target.name}")
        self.reload()

    def on_copy_checked(self) -> None:
        """复制 = 把勾上的视频文件拷到指定目录（原文件不动，库里也不加新登记）。"""
        rows = self._checked_rows()
        if not rows:
            QMessageBox.information(self, "资产中心", "先勾几个视频")
            return
        where = QFileDialog.getExistingDirectory(self, "拷到哪个目录")
        if not where:
            return
        folder = Path(where)
        copied: list[str] = []
        failed: list[str] = []
        for row in rows:
            source = Path(str(row["file_path"]))
            target = folder / source.name
            if not source.is_file():
                failed.append(f"{source.name}（文件不在盘上）")
                continue
            if target.exists():
                failed.append(f"{source.name}（目标目录已经有同名文件）")
                continue
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                failed.append(f"{source.name}（{exc}）")
                continue
            copied.append(source.name)
        text = f"[资产中心] 已拷 {len(copied)} 个视频到 {folder}"
        if failed:
            text += "；没拷成的：" + "、".join(failed)
        self._note(text, f"✓ 已拷 {len(copied)} 个视频到 {folder.name}"
                         + (f"（{len(failed)} 个没拷成，详见日志）" if failed else ""))
        if failed:
            QMessageBox.information(self, "资产中心",
                                    "这些没拷成：\n" + "\n".join(failed))

    def on_forget_checked(self) -> None:
        """删除 = 只删库里的登记，**磁盘文件一个都不动**（要连文件删走右键那一条）。"""
        db = self._handle()
        rows = self._checked_rows()
        if db is None or not rows:
            QMessageBox.information(self, "资产中心", "先勾几个视频")
            return
        names = "、".join(str(row["file_name"]) for row in rows[:6])
        if len(rows) > 6:
            names += f" 等 {len(rows)} 个"
        ask = QMessageBox(
            QMessageBox.Warning, "从库里删除勾选的视频",
            f"把这些视频从数据库里删掉？\n\n{names}\n\n"
            "它们的分析 / 高光 JSON / 成品登记会一起没掉，\n"
            "磁盘上的文件一个都不动，但库里的记录**删了找不回来**。",
            QMessageBox.Yes | QMessageBox.No, self)
        ask.setDefaultButton(QMessageBox.No)
        if ask.exec_() != QMessageBox.Yes:
            return
        done = 0
        for row in rows:
            if db_repo.forget_video(db, int(row["id"])) is not None:
                done += 1
            self._checked.discard(int(row["id"]))
        self._note(f"[资产中心] 已从库里删除 {done} 个视频的登记（文件没动）",
                   f"✓ 已删掉 {done} 个视频的登记（文件没动）")
        self.reload()



    # ------------------------------------------------------------ 右键：高光 JSON
    def on_asset_menu(self, pos) -> None:
        """右键高光 JSON：菜单按这一行的状态给动作（当前 / 已删除 各不一样）。"""
        line = self.tbl_assets.rowAt(pos.y())
        if line >= 0:
            self.tbl_assets.selectRow(line)
        asset_id = self.selected_asset()
        if asset_id is None:
            return
        row = next((r for r in self._asset_rows if int(r["id"]) == int(asset_id)), None)
        deleted = bool(row is not None and row["deleted_at"])
        current = db_assets.current_asset(self._handle(), self.current_video_id() or 0) \
            if self._handle() is not None else None
        is_current = current is not None and int(current["id"]) == int(asset_id)

        menu = QMenu(self)
        menu.addAction("查看", self.on_view)
        menu.addAction("编辑（保存会新建一份）", self.on_edit_json)
        menu.addAction("直接剪辑", self.on_render)
        menu.addSeparator()
        if is_current:
            star = menu.addAction("★ 当前 JSON")
            star.setEnabled(False)
        else:
            menu.addAction("设为当前", self.on_set_current)
        menu.addAction("复制 JSON", self.on_copy)
        menu.addSeparator()
        menu.addAction("导入 JSON…", self.on_import)
        raw = menu.addAction("显示原文", self.on_menu_raw)
        raw.setCheckable(True)
        raw.setChecked(self.json_panel.raw_visible())
        menu.addAction("复制原文", self.on_copy_text)
        menu.addSeparator()
        if deleted:
            menu.addAction("恢复", self.on_restore)
        else:
            menu.addAction("删除", self.on_delete)
        menu.exec_(self.tbl_assets.viewport().mapToGlobal(pos))

    def on_menu_raw(self) -> None:
        """右键里的「显示原文」：和「更多 ▾」里的那个勾选保持一致。"""
        shown = not self.json_panel.raw_visible()
        self.act_raw.setChecked(shown)
        self.json_panel.set_raw_visible(shown)

    # ------------------------------------------------------------ 右键：成品
    def _product_info(self, artifact_id: int | None) -> dict[str, Any] | None:
        if artifact_id is None or not self._product_rows:
            return None
        return next((i for i in self._product_rows
                     if int(i["artifact_id"]) == int(artifact_id)), None)

    def on_product_menu(self, pos) -> None:
        """右键成品：打开 / 定位文件 / 追溯来源 JSON、PRM、完整血缘 / 复制路径。"""
        line = self.tbl_products.rowAt(pos.y())
        if line >= 0:
            self.tbl_products.selectRow(line)
        info = self._product_info(self.selected_product())
        if info is None:
            return
        menu = QMenu(self)
        menu.addAction("打开成品", self.on_open_product)
        menu.addAction("打开所在文件夹", self.on_reveal)
        menu.addSeparator()
        source = menu.addAction("查看来源 JSON", self.on_show_source_json)
        source.setEnabled(info["asset_id"] is not None)
        prm = menu.addAction("查看 PRM", self.on_show_prm)
        prm.setEnabled(info["prm_id"] is not None)
        menu.addAction("查看完整血缘", self.on_show_lineage)
        menu.addAction("复制血缘", self.on_copy_lineage)
        menu.addSeparator()
        menu.addAction("复制文件路径", self.on_copy_product_path)
        menu.exec_(self.tbl_products.viewport().mapToGlobal(pos))

    def on_open_product(self) -> None:
        """打开成品文件本身（双击成品走的也是这条）。"""
        info = self._product_info(self.selected_product())
        if info is None:
            QMessageBox.information(self, "资产中心", "先在「成品」里选一个")
            return
        path = Path(str(info["path"]))
        if not path.is_file():
            QMessageBox.information(self, "资产中心", f"文件已经不在盘上：\n{path}")
            return
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - 打开用户自己的成品
            return
        QMessageBox.information(self, "资产中心", str(path))

    def on_show_source_json(self) -> None:
        """成品 → 来源 JSON：选中那一行、滚过去、面板跟着刷新。"""
        info = self._product_info(self.selected_product())
        if info is None or info["asset_id"] is None:
            QMessageBox.information(self, "资产中心", "这个成品没记来源 JSON")
            return
        self.focus_asset(int(info["asset_id"]))

    def focus_asset(self, asset_id: int) -> None:
        """把 JSON 表停在这一份上（找不到就展开已删除再找一次）。"""
        self.select_asset(int(asset_id))
        if self.selected_asset() != int(asset_id) and not self.chk_deleted.isChecked():
            self.act_deleted.setChecked(True)
            self.chk_deleted.setChecked(True)     # 来源 JSON 被软删了也要能跳过去
            self.select_asset(int(asset_id))
        line = self.tbl_assets.currentRow()
        if line >= 0 and self.tbl_assets.item(line, 1) is not None:
            self.tbl_assets.scrollToItem(self.tbl_assets.item(line, 1))
        self.tbl_assets.setFocus()

    def on_show_prm(self) -> None:
        """成品 → PRM：切到 PRM 页并选中那一份。"""
        info = self._product_info(self.selected_product())
        if info is None or info["prm_id"] is None:
            QMessageBox.information(self, "资产中心", "这个成品没记 PRM")
            return
        self.show_prm(int(info["prm_id"]))

    def show_prm(self, prm_id: int) -> None:
        center = self._center()
        if center is None:
            QMessageBox.information(self, "资产中心", "这个窗口没挂在资产中心里，切不过去")
            return
        center.show_prm(int(prm_id))

    def on_show_lineage(self) -> None:
        """查看完整血缘：展开技术细节并把血缘树重画一遍。"""
        self.chk_details.setChecked(True)
        self.refresh_lineage()
        self.tree_lineage.setFocus()

    def on_copy_product_path(self) -> None:
        info = self._product_info(self.selected_product())
        if info is None:
            return
        QApplication.clipboard().setText(str(info["path"]))
        self._note(f"[资产中心] 成品路径已复制：{info['path']}", "✓ 已复制成品路径")


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
            self.lbl_current.setText("当前视频暂无高光 JSON —— 让 AI 分析一次，"
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
            made = counts.get(int(row["id"]), 0)
            clips = _highlight_rows(db_assets.loads(row["current_json"]))
            span, length = "—", None
            if clips:
                try:
                    first, last = clips[0], clips[-1]
                    span = _span(first.get("start"), last.get("end"))
                    length = sum(float(c.get("end") or 0.0) - float(c.get("start") or 0.0)
                                 for c in clips)
                except (TypeError, ValueError):
                    span, length = "—", None
            self.tbl_assets.setItem(line, 0, _cell("★ 当前" if is_current else "○ 历史",
                                                   center=True))
            self.tbl_assets.setItem(line, 1, _id_item(_json_title(row), int(row["id"]),
                                                      bold=is_current))
            self.tbl_assets.setItem(line, 2, _cell(str(row["name"] or "—")))
            self.tbl_assets.setItem(line, 3, _cell(span, center=True))
            self.tbl_assets.setItem(line, 4, _duration_cell(length))
            self.tbl_assets.setItem(line, 5, _num(int(row["clip_count"] or 0)))
            self.tbl_assets.setItem(line, 6, _cell(_score(row["best_score"]), center=True))
            self.tbl_assets.setItem(line, 7, _cell(str(row["provider"] or "—"), center=True))
            self.tbl_assets.setItem(line, 8, _cell(str(row["model"] or "—"), center=True))
            self.tbl_assets.setItem(line, 9, _SortItem(f"✓ 已生成 {made} 个成品" if made
                                                       else "○ 未剪辑", made))
            self.tbl_assets.setItem(line, 10, _cell(_short_time(row["created_at"]),
                                                    center=True))
            self.tbl_assets.setItem(line, 11, _cell("已删除" if row["deleted_at"] else "在用",
                                                    center=True))
        self.tbl_assets.resizeColumnsToContents()
        self.tbl_assets.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_assets.blockSignals(False)
        if not rows:
            self.json_panel.clear("当前视频暂无高光 JSON")
            return
        self.select_asset(keep if keep is not None
                          else (int(current["id"]) if current is not None else None))

    def select_asset(self, asset_id: int | None) -> None:
        if not self.tbl_assets.rowCount():
            self.json_panel.clear("当前视频暂无高光 JSON")
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
        self._emit_focus()

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
        self.notice.emit(f"✓ 已保存为 高光 JSON #{new_id}（原件没动）")
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

    def _note(self, text: str, flash: str | None = None) -> None:
        if self._log:
            self._log(text)
        if flash:
            self.notice.emit(flash)      # 顶部直接给一句反馈，不用去翻日志
        self.changed.emit()

    def _emit_focus(self) -> None:
        """把「当前：视频 X · 高光 JSON #Y · N 个成品」推给窗口顶部（纯内存，不查库）。"""
        self.focus_changed.emit(self.focus_text())

    def focus_text(self) -> str:
        """一行话说清现在选中的是什么，没选就直说该干什么。"""
        vid = self.current_video_id()
        if vid is None:
            return "当前：请选择一个视频"
        row = self._row_of(vid)
        name = str(row["file_name"]) if row else f"视频 #{vid}"
        asset_id = self.selected_asset()
        if asset_id is None:
            return f"当前：{name} · 暂无高光 JSON"
        made = 0
        if self._product_rows is not None:
            made = sum(1 for info in self._product_rows
                       if info["asset_id"] is not None
                       and int(info["asset_id"]) == int(asset_id))
        return f"当前：{name} · 高光 JSON #{asset_id} · {made} 个成品"

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
        self._note(f"[高光 JSON] #{asset_id} 已复制成 #{new_id}（原件没动）", "✓ 已复制 JSON")
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
        self._note(f"[高光 JSON] 已登记 #{asset_id}（{count} 个高光，来自 {Path(path).name}）",
                   f"✓ 已导入 {Path(path).name}")
        self.reload()
        self.select_asset(int(asset_id))

    def on_set_current(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        if db_assets.set_current_asset(db, asset_id):
            self._note(f"[高光 JSON] #{asset_id} 已设为当前 JSON（其他 JSON 一份都没删）",
                       "✓ 已设为当前 JSON")
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
            self._note(f"[高光 JSON] #{asset_id} 已软删（成品未动）", "✓ 已移入回收状态")
        self.refresh_assets()

    def on_restore(self) -> None:
        got = self._need_asset()
        if got is None:
            return
        db, asset_id = got
        if db_assets.restore_asset(db, asset_id):
            self._note(f"[高光 JSON] #{asset_id} 已恢复", "✓ 已恢复")
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
        self.notice.emit("开始剪辑…（选好 PRM 点「开始剪辑」，进度在主界面）")
        if dialog.exec_() == QDialog.Accepted:
            self._note(f"[高光 JSON] #{asset_id} 已交给主界面渲染（进度和日志在主界面）",
                       "✓ 成品已生成（详见主界面日志）")

    # ------------------------------------------------------------ 成品
    def _thumb_for(self, path: Path | None) -> QIcon | None:
        """一个视频文件 → 一帧缩略图。取不到就返回 None，绝不报错。

        `cv2` 只在这里按需 import：它会改写 `QT_QPA_PLATFORM_PLUGIN_PATH`，
        必须等 QApplication 起来之后再导（和 `gui/player.py` 同一条规矩）。
        取 10% 处那一帧（开头常是黑场），一个视频只解一次，结果按路径缓存
        （解不出来也缓存 None，不会每次刷新都重试同一个坏文件）。
        """
        if path is None or not path.is_file():
            return None
        key = str(path)
        if key in self._thumbs:
            return self._thumbs[key]
        icon: QIcon | None = None
        try:
            import cv2  # noqa: PLC0415 - 见上面注释，必须延后导入

            cap = cv2.VideoCapture(key)
            try:
                if cap.isOpened():
                    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                    if frames > 10:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * 0.1))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        high, wide, _ = rgb.shape
                        image = QImage(rgb.data, wide, high, 3 * wide,
                                       QImage.Format_RGB888).copy()
                        icon = QIcon(QPixmap.fromImage(image).scaled(
                            self.THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            finally:
                cap.release()
        except Exception as exc:  # noqa: BLE001 - 解不出来就没有图，界面照旧能用
            self._note(f"[资产中心] 取不到 {path.name} 的缩略图：{exc}")
        self._thumbs[key] = icon
        return icon

    def _video_thumb(self) -> QIcon | None:
        """当前视频的缩略图（成品表整表共用这一张）。"""
        row = self._row_of(self.current_video_id())
        return self._thumb_for(Path(str(row["file_path"]))
                               if row and row["file_path"] else None)

    def paint_visible_thumbs(self) -> None:
        """只给**现在看得见的那几行**解缩略图：几百个视频也不会卡在打开的那一下。

        滚动 / 换筛选 / 改窗口大小都会再叫一次；解过的走缓存不重复解码。
        一次最多解 `THUMB_BATCH` 行，还有剩的就下一轮事件循环接着解，
        免得一次滚动把界面按住好几秒。
        """
        table = self.tbl_videos
        if not table.rowCount():
            return
        height = max(table.viewport().height(), 0)
        step = self.THUMB_SIZE.height() + 8
        top = table.rowAt(0)
        bottom = table.rowAt(max(height - 1, 0))
        first = max(top, 0)
        # 表格还没排版时（窗口刚建、viewport 高度是 0）rowAt 会返回 -1：
        # 这时按一屏能放几行估个上限，绝不能顺着往下把全部几百个视频都解一遍
        guess = first + height // step
        last = min(bottom if bottom >= 0 else guess, table.rowCount() - 1)
        done = pending = 0
        for line in range(first, last + 1):
            item = table.item(line, self.NAME_COLUMN)
            marker = table.item(line, 0)
            if item is None or marker is None or not item.icon().isNull():
                continue
            if done >= self.THUMB_BATCH:
                pending += 1
                continue
            row = self._row_of(int(marker.text()) if marker.text() else None)
            icon = self._thumb_for(Path(str(row["file_path"]))
                                   if row and row["file_path"] else None)
            done += 1
            if icon is not None:
                item.setIcon(icon)
        if pending:
            QTimer.singleShot(0, self.paint_visible_thumbs)



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
        thumb = self._video_thumb()      # 整表共用一张：成品都是同一个视频剪出来的
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
            name_item = _id_item(Path(info["path"]).name, info["artifact_id"], bold=same_json)
            if thumb is not None:
                name_item.setIcon(thumb)     # 行首的视频缩略图，纯显示，不改任何数据
            self.tbl_products.setItem(line, 1, name_item)
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
            self.lbl_product_head.setText("当前视频还没有成品 —— 上面选一份高光 JSON，"
                                          "点「直接剪辑」就能出成品")
            self.lbl_product.setText("当前视频还没有成品，选一份 JSON 剪一次就有了")
            self.tree_lineage.clear()
            self.tree_lineage.addTopLevelItem(QTreeWidgetItem(
                ["血缘", "剪出成品之后，这里显示 视频 → JSON → Engine → PRM → 成品"]))
            self._emit_focus()
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
            self.lbl_product.setText("选一个成品，这里显示它是怎么来的")
            self.tree_lineage.addTopLevelItem(
                QTreeWidgetItem(["血缘", "上面选一个成品，这里显示 视频 → JSON → Engine → PRM → 成品"]))
            return
        info = db_assets.artifact_lineage(db, artifact_id)
        if info is None:
            self.lbl_product.setText("该成品暂无完整血缘记录")
            self.tree_lineage.addTopLevelItem(
                QTreeWidgetItem(["血缘", "该成品暂无完整血缘记录（库里查不到这条 artifact）"]))
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


        def node(parent, title, value="", *, asset_id=None, artifact_id=None, prm_id=None):
            item = QTreeWidgetItem([str(title), str(value)])
            if asset_id is not None:
                item.setData(0, Qt.UserRole, ("asset", int(asset_id)))
            if artifact_id is not None:
                item.setData(0, Qt.UserRole, ("product", int(artifact_id)))
            if prm_id is not None:
                item.setData(0, Qt.UserRole, ("prm", int(prm_id)))
            (parent.addChild(item) if isinstance(parent, QTreeWidgetItem)
             else self.tree_lineage.addTopLevelItem(item))
            return item

        deep = self.chk_details.isChecked()      # 技术细节默认收起，勾「详细信息」才展开
        root = node(None, "视频", video.get("file_name", "—"))
        if deep:
            node(root, "路径", video.get("file_path", "—"))
            node(root, "分析批次", info.get("analysis_id") or "—")
        json_node = node(root, "高光 JSON",
                         "—" if asset is None else f"{_json_title(asset)}（{asset['name']}）"
                         + ("（已删除）" if info.get("asset_deleted") else ""),
                         asset_id=None if asset is None else int(asset["id"]))

        node(json_node, "AI", info.get("provider") or "—")
        node(json_node, "模型", info.get("model") or "—")
        if deep:
            node(json_node, "AI 任务", info.get("task_id") or "—")
        for index, clip in enumerate(spans["ai"], start=1):
            node(json_node, f"原始区间 {index}",
                 f"{_span(clip.get('start'), clip.get('end'))}"
                 f"　评分 {_score(clip.get('score'))}")
        engine_node = node(root, "Clip Engine", "确定性修正（规则来源只有 plan_clips）")
        for index, plan in enumerate(spans["engine"], start=1):
            one = node(engine_node, f"修正区间 {index}",
                       f"{_span(plan['start'], plan['end'])}（{_score(plan['duration'])}s）")
            # 「为什么最后是这个区间」属于用户必须能直接看到的答案，不藏进「详细信息」
            for note in plan["notes"]:
                node(one, "原因", note)
            if not plan["notes"]:
                node(one, "原因", "未调整（AI 区间本身就落在语义边界上）")
        node(root, "PRM", "—" if prm is None else f"{prm['name']}（{prm['filename']}）"
             + ("（已删除）" if info.get("prm_deleted") else ""),
             prm_id=None if prm is None else int(prm["id"]))
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
        """点血缘节点就定位到对应那一行 / 那一页（不弹窗、不改数据）。"""
        marker = item.data(0, Qt.UserRole)
        if not marker:
            return
        kind, ident = marker
        if kind == "asset":
            self.focus_asset(int(ident))
            return
        if kind == "prm":
            self.show_prm(int(ident))
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
        self.notice.emit("✓ 已复制 JSON 原文")

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
            self._note("[资产中心] 血缘已复制到剪贴板", "✓ 已复制血缘")


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
        # 最小尺寸压到 960×600：1000×620、1024×680、1152×720 这些小屏也能完整显示，
        # 不靠「把 minimumHeight 调大」来掩盖布局问题（Phase 16 §14）
        self.setMinimumSize(960, 600)
        self.resize(1240, 800)

        self.videos = VideoAssetsPage(cfg, window=parent, parent=self, log=log)
        self.prm_panel = PrmPanel(cfg, self, log=log)
        self.videos.changed.connect(self._on_changed)
        self.prm_panel.changed.connect(self._on_changed)
        self.videos.notice.connect(self.flash)
        self.prm_panel.notice.connect(self.flash)
        self.videos.focus_changed.connect(self.set_focus_text)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.videos, "视频资产")
        self.tabs.addTab(self.prm_panel, "PRM 管理")

        self.btn_reload = QPushButton("🔄 刷新")
        self.btn_reload.setToolTip("重新查一次库（视频 / 高光 JSON / 成品 / PRM 一起刷）")
        self.btn_reload.clicked.connect(self.reload)
        self.btn_dirs = QPushButton("目录…")
        self.btn_dirs.setToolTip("资产中心自己的输入 / 输出目录（扫原始视频 / 扫高光成品），"
                                 "跟 AI 面板那两个互不相干")
        self.btn_dirs.clicked.connect(self.on_dirs)
        subtitle = QLabel("管理视频 → 高光 JSON → 成品的完整资产关系")
        subtitle.setWordWrap(True)
        head = QHBoxLayout()
        head.addWidget(_title("视频资产中心"))
        head.addWidget(subtitle, 1)
        head.addWidget(self.btn_dirs)
        head.addWidget(self.btn_reload)

        # 顶部一行工作流：第一次用的人照着 ①②③④ 点就能出成品（不做教程弹窗）
        self.lbl_steps = QLabel("① 选视频　→　点这一行的「JSON」格子　→　"
                                "② 选 JSON　→　③ 直接剪辑　→　④ 点「成品」格子看成品与血缘")
        steps = self.lbl_steps.font()
        steps.setBold(True)
        self.lbl_steps.setFont(steps)
        self.lbl_steps.setWordWrap(True)
        self.lbl_tip = QLabel("双击 = 播放视频 · 点 JSON / 成品格子 = 开弹窗 · 右键 = 其余操作")
        self.lbl_tip.setWordWrap(True)
        self.lbl_tip.setToolTip("视频页的高光 JSON、成品血缘点那两列的格子就开弹窗；"
                                "筛选 / 删除 / 打开目录在右键菜单里；PRM 页同样有右键菜单")
        hint = QHBoxLayout()
        hint.addWidget(self.lbl_steps)
        hint.addStretch(1)
        hint.addWidget(self.lbl_tip)

        # 现在选中的是什么 + 上一步操作的结果，都在同一行，谁都不用去翻日志
        self.lbl_now = QLabel("当前：请选择一个视频")
        self.lbl_now.setWordWrap(True)
        self.lbl_flash = QLabel("")
        self.lbl_flash.setWordWrap(True)
        state = QHBoxLayout()
        state.addWidget(self.lbl_now, 1)
        state.addWidget(self.lbl_flash)

        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        note = QLabel("全部可搜、可筛、可追溯：选视频看它的高光 JSON，选 JSON 看区间和成品")
        note.setWordWrap(True)
        foot = QHBoxLayout()
        foot.addWidget(note)
        foot.addStretch(1)
        foot.addWidget(close)

        # 批量操作条：作用对象永远是「① 视频库」里**勾上**的那些视频
        self.lbl_checked = QLabel("勾选 0 个")
        self.btn_check_all = QPushButton("全选")
        self.btn_check_all.setToolTip("把当前列表（含筛选结果）里的每一行都勾上")
        self.btn_check_all.clicked.connect(self.videos.on_check_all)
        self.btn_rename = QPushButton("编辑")
        self.btn_rename.setToolTip("重命名：改磁盘上的文件名，库里的路径跟着改（只勾 1 个时可用）")
        self.btn_rename.clicked.connect(self.videos.on_rename_checked)
        self.btn_copy_files = QPushButton("复制")
        self.btn_copy_files.setToolTip("把勾上的视频文件拷到指定目录，原文件不动")
        self.btn_copy_files.clicked.connect(self.videos.on_copy_checked)
        self.btn_forget = QPushButton("删除")
        self.btn_forget.setToolTip("只删库里的登记，磁盘文件一个不动"
                                   "（要连文件一起删，走视频右键菜单）")
        self.btn_forget.clicked.connect(self.videos.on_forget_checked)
        self.btn_invert_checks = QPushButton("反选")
        self.btn_invert_checks.setToolTip("当前列表里勾着的取消、没勾的勾上（不删任何东西）")
        self.btn_invert_checks.clicked.connect(self.videos.on_invert_checks)
        batch = QHBoxLayout()
        batch.addWidget(self.lbl_checked)
        for button in (self.btn_check_all, self.btn_rename, self.btn_copy_files,
                       self.btn_forget, self.btn_invert_checks):
            batch.addWidget(button)
        batch.addStretch(1)
        self.videos.checks_changed.connect(self.on_checks_changed)
        self.on_checks_changed(0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)
        outer.addLayout(head)
        outer.addLayout(hint)
        outer.addLayout(state)
        outer.addWidget(self.tabs, 1)
        outer.addLayout(batch)
        outer.addLayout(foot)
        self.set_focus_text(self.videos.focus_text())
        self._build_dirs_dialog()

    # -------------------------------------------------- 自己的输入 / 输出目录
    def _build_dirs_dialog(self) -> None:
        """资产中心**自己的**两个目录，跟 AI 面板的 AI_输入 / AI_输出 完全分开。

        - 原始视频目录：扫出来的视频登记进库，之后就出现在「① 视频库」里；
        - 高光成品目录：扫出来的成品认回对应视频（老成品、手动挪过来的都能补登记）。

        改完即存（写 `assets.input_dir` / `assets.output_dir`），扫盘只在点
        「扫描目录」时做——每次开窗就扫大目录会把界面按住好几秒。
        视频页上不摆按钮，所以这一摊单独一个非模态小窗口，入口是顶部的「目录…」。
        """
        self._dirs_ready = False
        self._saved_dirs: dict[str, str] | None = None
        self._dir_timer = QTimer(self)
        self._dir_timer.setSingleShot(True)
        self._dir_timer.setInterval(350)
        self._dir_timer.timeout.connect(self._save_dirs)

        section = self.cfg.assets
        self.edit_assets_in = DropDirEdit(str(section.get("input_dir") or ""))
        self.edit_assets_out = DropDirEdit(str(section.get("output_dir") or ""))
        self.edit_assets_in.setToolTip("原始视频放哪儿。点「扫描目录」把这里的视频登记进库，"
                                       "登记完就出现在视频列表里。改完即存")
        self.edit_assets_out.setToolTip("高光成品放哪儿。点「扫描目录」把这里已有的成品"
                                        "认回对应视频（含 <视频名>.json 这类老方案）。改完即存")
        for edit in (self.edit_assets_in, self.edit_assets_out):
            edit.textChanged.connect(lambda _="": self._dirs_touched())
            edit.editingFinished.connect(self._save_dirs)
        self.btn_scan_dirs = QPushButton("扫描目录")
        self.btn_scan_dirs.setToolTip("按这两个目录扫一次盘：新视频登记进库、"
                                      "已有成品认回对应视频，然后刷新列表")
        self.btn_scan_dirs.clicked.connect(self.on_scan_dirs)

        body = QWidget()
        grid = QGridLayout(body)
        grid.addWidget(QLabel("原始视频"), 0, 0)
        grid.addWidget(dir_row(self, self.edit_assets_in, "选择原始视频目录"), 0, 1)
        grid.addWidget(QLabel("高光成品"), 1, 0)
        grid.addWidget(dir_row(self, self.edit_assets_out, "选择高光成品目录"), 1, 1)
        note = QLabel("这两个目录只归资产中心用：跟 AI 面板的 AI_输入 / AI_输出 谁也不覆盖谁。"
                      "改完即存，扫盘点右边那个按钮。")
        note.setWordWrap(True)
        grid.addWidget(note, 2, 0, 1, 2)
        grid.addWidget(self.btn_scan_dirs, 0, 2, 2, 1)
        grid.setColumnStretch(1, 1)

        self.dlg_dirs = QDialog(self)
        self.dlg_dirs.setWindowTitle("资产中心目录 — 视频资产中心")
        self.dlg_dirs.setWindowFlags(Qt.Window)
        lay = QVBoxLayout(self.dlg_dirs)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(body)
        self.dlg_dirs.resize(720, 160)
        self._dirs_ready = True

    def on_dirs(self) -> None:
        self.dlg_dirs.show()
        self.dlg_dirs.raise_()
        self.dlg_dirs.activateWindow()

    def _dirs_touched(self) -> None:
        if self._dirs_ready:
            self._dir_timer.start()

    def _save_dirs(self) -> None:
        """两个目录落进 config.json 的 `assets` 一节（只写这两个键，别的一个不碰）。"""
        self._dir_timer.stop()
        if not self._dirs_ready:
            return
        patch = {"input_dir": self.edit_assets_in.text().strip(),
                 "output_dir": self.edit_assets_out.text().strip()}
        if patch == self._saved_dirs:
            return
        self._saved_dirs = dict(patch)
        try:
            self.cfg.save_patch({"assets": patch})
        except Exception as exc:  # noqa: BLE001 - 存不进去不该打断正在看的列表
            self.flash(f"目录存不进 config.json：{exc}")

    def assets_dir(self, key: str) -> Path | None:
        """`assets.input_dir` / `assets.output_dir`，留空或者不在盘上就返回 None。"""
        raw = str(self.cfg.assets.get(key) or "").strip()
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_dir() else None

    def on_scan_dirs(self) -> None:
        """按自己这两个目录扫一次盘：登记新视频 + 认回已有成品，然后刷新列表。"""
        self._save_dirs()          # 防抖里可能还压着一次改动，扫的就是你看到的目录
        source = self.assets_dir("input_dir")
        product = self.assets_dir("output_dir")
        if source is None and product is None:
            QMessageBox.information(self.dlg_dirs, "资产中心",
                                    "先填这两个目录（至少一个），再点「扫描目录」。\n"
                                    "填了还提示这句，就是路径不在盘上。")
            return
        try:
            stats = refresh_from_disk(self.cfg, None,
                                      folders=[source] if source is not None else [],
                                      ai_out=product)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self.dlg_dirs, "资产中心", f"扫不动：{exc}")
            return
        if self._log:
            self._log(f"[资产中心] 扫完自己的目录：看到 {stats.get('videos_seen', 0)} 个视频"
                      f"（新登记 {stats.get('videos_new', 0)}），"
                      f"成品/文件 {stats.get('artifacts', 0)} 条，"
                      f"历史高光 JSON {stats.get('ai_results', 0)} 份")
        self.flash(f"✓ 新登记 {stats.get('videos_new', 0)} 个视频，"
                   f"认回 {stats.get('artifacts', 0)} 个文件")
        self.reload()

    def on_checks_changed(self, count: int) -> None:
        """勾选数一变：按钮该灰的灰（没勾就只有「全选」「反选」能点）。"""
        self.lbl_checked.setText(f"勾选 {count} 个")
        self.btn_rename.setEnabled(count == 1)
        for button in (self.btn_copy_files, self.btn_forget):
            button.setEnabled(count > 0)

    # ------------------------------------------------------------ 顶部状态
    def set_focus_text(self, text: str) -> None:
        """「当前：视频 X · 高光 JSON #Y · N 个成品」。"""
        self.lbl_now.setText(text)

    def flash(self, text: str) -> None:
        """操作结果就写在顶部，用户不用去日志窗口找（下一次操作会覆盖它）。"""
        self.lbl_flash.setText(text)


    # ------------------------------------------------------------ 转发
    def _on_changed(self) -> None:
        self.changed.emit()

    def reload(self) -> None:
        self.videos.reload()
        self.prm_panel.reload()

    def show_prm_page(self) -> None:
        self.tabs.setCurrentIndex(1)

    def show_prm(self, prm_id: int | None = None) -> None:
        """切到 PRM 页，并且尽量停在指定那一份上（成品 / 血缘里点 PRM 走这里）。"""
        self.show_prm_page()
        if prm_id is None:
            return
        self.prm_panel.chk_all.setChecked(True)   # 软删的 PRM 也要能被追溯到
        self.prm_panel.select(int(prm_id))
        self.prm_panel.table.setFocus()

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
