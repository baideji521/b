"""素材矩阵（卡片式）+ FINAL TIMELINE：列内自由拖，跨列一律拒绝。

界面就是你要的那三层里的后两层：

    ② 素材池：一列 = 一个段落（S1/S2/S3…），列里每张卡片是一条候选素材，
              最上面那张带 ⭐ = 这一格当前用的
    ③ FINAL TIMELINE：一格一个槽，从上面的列里拖一张卡片进来就替换

**硬限制**：拖动只能在同一个段落里进行。`S4/5.mp4` 拖到 S1 的列或 S1 的槽上，
直接拒绝（光标显示禁止、松手什么也不发生）——
素材是"已经绑定到某个音乐位置"的资产，跨段落挪等于让画面和音乐错开，
那正是这个系统最不能出的错。判定只有一处：`accepts()`。

这一层不选素材、不打分、不落库：拖完只发 `picks_changed({段落: 素材id})`，
交给上层（推荐/混剪那套）去用。
"""

from __future__ import annotations

import json
from typing import Any

from PyQt5.QtCore import QMimeData, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...logging_setup import get_logger
from .. import theme

logger = get_logger("dance.gui.matrix")

#: 拖拽用的自定义 MIME 类型。用自己的类型而不是 text/plain：
#: 免得从别处拖来一段文字也被当成素材
MIME = "application/x-dance-material"

CARD_HEIGHT = 44
COLUMN_WIDTH = 210
SLOT_WIDTH = 150
SLOT_HEIGHT = 74


def pack(payload: dict[str, Any]) -> QMimeData:
    data = QMimeData()
    data.setData(MIME, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return data


def unpack(data: QMimeData) -> dict[str, Any]:
    """从 MIME 里取回 payload。不是我们的东西、或者坏的，一律返回空字典。"""
    if data is None or not data.hasFormat(MIME):
        return {}
    try:
        payload = json.loads(bytes(data.data(MIME)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def accepts(segment_index: int, payload: dict[str, Any]) -> bool:
    """这份 payload 能落在第 `segment_index` 段上吗？**唯一的判定处。**"""
    if not payload or "material_id" not in payload:
        return False
    return int(payload.get("segment_index", -1)) == int(segment_index)


class MaterialColumn(QListWidget):
    """一个段落的素材列。列内可拖动排序；最上面那张就是"这一格当前用的"。"""

    reordered = pyqtSignal(int, int)          # 段落, 现在排第一的素材 id
    refused = pyqtSignal(int, int)            # 段落, 被拒绝的那条素材属于哪个段落

    def __init__(self, segment_index: int, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.segment_index = int(segment_index)
        self.title = title or f"S{self.segment_index + 1}"
        self.setObjectName(f"column_{self.segment_index}")
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setMinimumWidth(COLUMN_WIDTH)
        self.setUniformItemSizes(True)

    # ------------------------------------------------------------------ 数据
    def load(self, rows) -> None:
        """`rows` 是 `[{material_id, label, note}]`，第一条即当前使用。"""
        self.clear()
        for index, row in enumerate(rows or ()):
            self.addItem(self._item(row, index == 0))

    def _item(self, row: dict[str, Any], current: bool) -> QListWidgetItem:
        star = "⭐ " if current else ""
        item = QListWidgetItem(f"{star}{row.get('label') or row.get('material_id')}"
                               f"　/　{self.title}")
        item.setData(Qt.UserRole, {"material_id": int(row.get("material_id") or 0),
                                   "segment_index": self.segment_index,
                                   "label": str(row.get("label") or "")})
        item.setToolTip(str(row.get("note") or "拖到最上面 = 这一格改用它"))
        item.setSizeHint(QSize(COLUMN_WIDTH - 24, CARD_HEIGHT))
        return item

    def payloads(self) -> list[dict[str, Any]]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def current_pick(self) -> int:
        """排第一的那条素材 id；列是空的就 0。"""
        return int(self.payloads()[0]["material_id"]) if self.count() else 0

    # ------------------------------------------------------------------ 拖拽
    def mimeTypes(self) -> list[str]:            # noqa: N802 - Qt 的名字
        return [MIME]

    def mimeData(self, items) -> QMimeData:      # noqa: N802
        payload = items[0].data(Qt.UserRole) if items else {}
        return pack(payload or {})

    def dragEnterEvent(self, event) -> None:     # noqa: N802
        self._filter(event)

    def dragMoveEvent(self, event) -> None:      # noqa: N802
        self._filter(event)

    def _filter(self, event) -> None:
        payload = unpack(event.mimeData())
        if accepts(self.segment_index, payload):
            event.acceptProposedAction()
            return
        # 跨段落：明确拒绝，光标就是禁止符号，松手什么也不会发生
        if payload:
            self.refused.emit(self.segment_index, int(payload.get("segment_index", -1)))
        event.ignore()

    def dropEvent(self, event) -> None:          # noqa: N802
        payload = unpack(event.mimeData())
        if not accepts(self.segment_index, payload):
            event.ignore()
            return
        row = self.indexAt(event.pos()).row()
        self.drop_payload(payload, row if row >= 0 else self.count())
        event.acceptProposedAction()

    def drop_payload(self, payload: dict[str, Any], row: int) -> bool:
        """把一条素材插到第 `row` 位（同一条已经在列里就是移动）。跨段落一律 False。

        单独抽出来是为了能不靠真实鼠标拖拽来测这件事 —— QDrag 在离屏环境下跑不起来，
        但"能不能落、落到哪"这条规则必须有测试盯着。
        """
        if not accepts(self.segment_index, payload):
            self.refused.emit(self.segment_index, int(payload.get("segment_index", -1)))
            return False
        material_id = int(payload["material_id"])
        existing = [p["material_id"] for p in self.payloads()]
        if material_id in existing:
            source = existing.index(material_id)
            item = self.takeItem(source)
            if row > source:
                row -= 1
        else:
            item = self._item(payload, False)
        self.insertItem(max(0, min(row, self.count())), item)
        self._restar()
        self.reordered.emit(self.segment_index, self.current_pick())
        return True

    def _restar(self) -> None:
        """⭐只跟着"排第一"这件事走，别的都不带。"""
        for index in range(self.count()):
            item = self.item(index)
            payload = item.data(Qt.UserRole) or {}
            label = payload.get("label") or payload.get("material_id")
            star = "⭐ " if index == 0 else ""
            item.setText(f"{star}{label}　/　{self.title}")


class TimelineSlot(QFrame):
    """FINAL TIMELINE 上的一格。只接同一个段落的卡片，别的一律拒绝。"""

    replaced = pyqtSignal(int, int)            # 段落, 新的素材 id
    refused = pyqtSignal(int, int)

    def __init__(self, segment_index: int, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.segment_index = int(segment_index)
        self.title = title or f"S{self.segment_index + 1}"
        self.material_id = 0
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(SLOT_WIDTH, SLOT_HEIGHT)

        column = QVBoxLayout(self)
        column.setContentsMargins(6, 4, 6, 4)
        column.setSpacing(2)
        self.head = QLabel(self.title, self)
        self.head.setStyleSheet(f"color:{theme.ACCENT};")
        self.body = QLabel("拖一张卡片进来", self)
        self.body.setWordWrap(True)
        self.body.setStyleSheet(f"color:{theme.TEXT_DIM};")
        column.addWidget(self.head)
        column.addWidget(self.body, 1)

    def set_material(self, material_id: int, label: str = "") -> None:
        self.material_id = int(material_id or 0)
        self.body.setText(str(label or material_id or "拖一张卡片进来"))
        # 有素材就用默认前景色（跟着主题走），空的才画淡 —— 别在这儿写死颜色，
        # 否则换主题时这一行会变成看不见的浅灰
        self.body.setStyleSheet("" if self.material_id else f"color:{theme.TEXT_DIM};")

    def dragEnterEvent(self, event) -> None:     # noqa: N802
        self._filter(event)

    def dragMoveEvent(self, event) -> None:      # noqa: N802
        self._filter(event)

    def _filter(self, event) -> None:
        payload = unpack(event.mimeData())
        if accepts(self.segment_index, payload):
            event.acceptProposedAction()
            return
        if payload:
            self.refused.emit(self.segment_index, int(payload.get("segment_index", -1)))
        event.ignore()

    def dropEvent(self, event) -> None:          # noqa: N802
        if self.drop_payload(unpack(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def drop_payload(self, payload: dict[str, Any]) -> bool:
        """替换这一格。跨段落返回 False 并且**什么都不改**。"""
        if not accepts(self.segment_index, payload):
            self.refused.emit(self.segment_index, int(payload.get("segment_index", -1)))
            return False
        self.set_material(int(payload["material_id"]), str(payload.get("label") or ""))
        self.replaced.emit(self.segment_index, self.material_id)
        return True


class MatrixPanel(QWidget):
    """② 素材池（一列一个段落）+ ③ FINAL TIMELINE。拖完只发 `picks_changed`。"""

    picks_changed = pyqtSignal(dict)           # {段落: 素材id}
    refused = pyqtSignal(str)                  # 给状态栏的一句人话

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.columns: list[MaterialColumn] = []
        self.slots: list[TimelineSlot] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("📦 素材池：同一列里随便拖，拖到最上面就是这一格改用它"
                                "（跨段落拖不进去）", self))
        self.pool_area = QScrollArea(self)
        self.pool_area.setWidgetResizable(True)
        self.pool = QWidget(self.pool_area)
        self.pool_row = QHBoxLayout(self.pool)
        self.pool_row.setContentsMargins(0, 0, 0, 0)
        self.pool_row.setSpacing(6)
        self.pool_area.setWidget(self.pool)
        layout.addWidget(self.pool_area, 3)

        layout.addWidget(QLabel("🎬 FINAL TIMELINE：从上面拖一张卡片下来替换这一格", self))
        self.line_area = QScrollArea(self)
        self.line_area.setWidgetResizable(True)
        self.line_area.setMaximumHeight(SLOT_HEIGHT + 28)
        self.line = QWidget(self.line_area)
        self.line_row = QHBoxLayout(self.line)
        self.line_row.setContentsMargins(0, 0, 0, 0)
        self.line_row.setSpacing(4)
        self.line_area.setWidget(self.line)
        layout.addWidget(self.line_area)

        self.hint = QLabel("还没有素材。先在「主音频/分段」定好分段，再切片入库。", self)
        self.hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        layout.addWidget(self.hint)

    # ------------------------------------------------------------------ 数据
    def load(self, segments) -> None:
        """`segments` 是 `[{"index":0,"title":"S1","materials":[{material_id,label,note}]}]`。"""
        self._clear()
        for spec in segments or ():
            index = int(spec.get("index", 0))
            title = str(spec.get("title") or f"S{index + 1}")
            rows = list(spec.get("materials") or ())

            box = QWidget(self.pool)
            column_layout = QVBoxLayout(box)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(2)
            column_layout.addWidget(QLabel(f"{title}　（{len(rows)} 条候选）", box))
            column = MaterialColumn(index, title, box)
            column.load(rows)
            column.reordered.connect(self._pick_changed)
            column.refused.connect(self._say_refused)
            column_layout.addWidget(column, 1)
            self.pool_row.addWidget(box)
            self.columns.append(column)

            slot = TimelineSlot(index, title, self.line)
            if rows:
                slot.set_material(int(rows[0].get("material_id") or 0),
                                  str(rows[0].get("label") or ""))
            slot.replaced.connect(self._slot_replaced)
            slot.refused.connect(self._say_refused)
            self.line_row.addWidget(slot)
            self.slots.append(slot)

        self.pool_row.addStretch(1)
        self.line_row.addStretch(1)
        total = sum(c.count() for c in self.columns)
        self.hint.setText(f"{len(self.columns)} 段　·　共 {total} 条候选素材"
                          if self.columns else
                          "还没有素材。先在「主音频/分段」定好分段，再切片入库。")

    def _clear(self) -> None:
        for row in (self.pool_row, self.line_row):
            while row.count():
                item = row.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        self.columns.clear()
        self.slots.clear()

    # ------------------------------------------------------------------ 交互
    def picks(self) -> dict[int, int]:
        """当前每一格用哪条素材。FINAL TIMELINE 上摆着的那份才算数。"""
        return {slot.segment_index: slot.material_id
                for slot in self.slots if slot.material_id}

    def _pick_changed(self, segment_index: int, material_id: int) -> None:
        """列里的顺序变了 → 那一格的 FINAL TIMELINE 也跟着换成新的第一名。"""
        for slot in self.slots:
            if slot.segment_index == int(segment_index):
                label = ""
                for column in self.columns:
                    if column.segment_index == int(segment_index) and column.count():
                        label = str((column.payloads()[0] or {}).get("label") or "")
                slot.set_material(int(material_id), label)
        self.picks_changed.emit(self.picks())

    def _slot_replaced(self, _segment_index: int, _material_id: int) -> None:
        self.picks_changed.emit(self.picks())

    def _say_refused(self, target: int, came_from: int) -> None:
        self.refused.emit(f"拖不过去：那条素材绑在 S{int(came_from) + 1}，"
                          f"不能放到 S{int(target) + 1} —— 素材和音乐位置是绑死的")


__all__ = ["MIME", "COLUMN_WIDTH", "SLOT_WIDTH", "SLOT_HEIGHT",
           "pack", "unpack", "accepts",
           "MaterialColumn", "TimelineSlot", "MatrixPanel"]
