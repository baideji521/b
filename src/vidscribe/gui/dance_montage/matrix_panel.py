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
    QPushButton,
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
        self.current_id = 0
        self.setObjectName(f"column_{self.segment_index}")
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setMinimumWidth(COLUMN_WIDTH)
        self.setUniformItemSizes(True)

    # ------------------------------------------------------------------ 数据
    def load(self, rows, current_id: int = 0) -> None:
        """`rows` 是 `[{material_id, label, score, path, note}]`。

        `current_id` 是**库里存着的最终选择**；给了就按它标 ⭐ CURRENT，
        没给（0）就按第一条算 —— 前者才是"重新打开工程后恢复"的依据。
        """
        self.clear()
        self.current_id = int(current_id or 0)
        for index, row in enumerate(rows or ()):
            material_id = int(row.get("material_id") or 0)
            chosen = (material_id == self.current_id) if self.current_id else index == 0
            self.addItem(self._item(row, chosen))
        if not self.current_id and self.count():
            self.current_id = int((self.payloads()[0] or {}).get("material_id") or 0)

    def _item(self, row: dict[str, Any], current: bool) -> QListWidgetItem:
        payload = {"material_id": int(row.get("material_id") or 0),
                   "segment_index": self.segment_index,
                   "label": str(row.get("label") or ""),
                   "path": str(row.get("path") or ""),
                   "score": float(row.get("score") or 0.0)}
        item = QListWidgetItem(self._caption(payload, current))
        item.setData(Qt.UserRole, payload)
        item.setToolTip(str(row.get("note") or "")
                        + "\n拖到最上面 = 这一格改用它（只能在本列内拖）")
        item.setSizeHint(QSize(COLUMN_WIDTH - 24, CARD_HEIGHT))
        return item

    def _caption(self, payload: dict[str, Any], current: bool) -> str:
        """卡片一行说清四件事：是否当前使用、素材名、评分、绑在哪一段。"""
        label = payload.get("label") or payload.get("material_id")
        score = float(payload.get("score") or 0.0)
        mark = "⭐ CURRENT　" if current else ""
        rating = f"　⭐{score:.0f}" if score > 0 else ""
        return f"{mark}{label}{rating}　/　{self.title}"

    def payloads(self) -> list[dict[str, Any]]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def current_pick(self) -> int:
        """这一格当前用哪条素材。**排第一的就是它**（拖到最上面 = 改用它）。"""
        return int(self.payloads()[0]["material_id"]) if self.count() else 0

    def set_current(self, material_id: int) -> bool:
        """「设为当前」按钮：把某条素材提到最上面。不在本列里就返回 False。"""
        for payload in self.payloads():
            if int(payload.get("material_id") or 0) == int(material_id):
                return self.drop_payload(payload, 0)
        return False

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
        """⭐ CURRENT 只跟着"排第一"这件事走，别的都不带。"""
        self.current_id = self.current_pick()
        for index in range(self.count()):
            item = self.item(index)
            payload = item.data(Qt.UserRole) or {}
            item.setText(self._caption(payload, index == 0))


class TimelineSlot(QFrame):
    """FINAL TIMELINE 上的一格。只接同一个段落的卡片，别的一律拒绝。"""

    replaced = pyqtSignal(int, int)            # 段落, 新的素材 id
    refused = pyqtSignal(int, int)

    def __init__(self, segment_index: int, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.segment_index = int(segment_index)
        self.title = title or f"S{self.segment_index + 1}"
        self.material_id = 0
        self.span: tuple[float, float] | None = None
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
        self.time = QLabel("", self)
        self.time.setStyleSheet(f"color:{theme.TEXT_DIM};")
        column.addWidget(self.head)
        column.addWidget(self.body, 1)
        column.addWidget(self.time)

    def set_span(self, start: float, end: float) -> None:
        """这一格在**主音频**上的起止。它来自 Segment Template，这里只显示，永远不改。"""
        self.span = (round(float(start), 3), round(float(end), 3))
        self.time.setText(f"{self.span[0]:.3f}→{self.span[1]:.3f}"
                          f"（{self.span[1] - self.span[0]:.3f}s）")
        self.head.setText(f"{self.title}")

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
    """② 素材池（一列一个段落）+ ③ FINAL TIMELINE。

    每一次改动（拖排序、替换、设为当前）都**立刻落库**：
    `dance_final_selections`（这一段用谁）+ `dance_candidate_order`（候选顺序）。
    重新打开工程时按这两张表恢复 —— 界面状态不是唯一真相。

    库里那一步还会**再校验一次**素材的 segment_index（`repo.set_final_selection`），
    所以跨段落的选择拦两道：界面不给落，业务层也不给写。
    """

    picks_changed = pyqtSignal(dict)           # {段落: 素材id}
    refused = pyqtSignal(str)                  # 给状态栏的一句人话
    preview_requested = pyqtSignal(str)        # 素材文件路径
    saved = pyqtSignal(str)                    # 落库之后的一句人话

    def __init__(self, parent=None, db=None, song_id: int = 0) -> None:
        super().__init__(parent)
        self.db = db
        self.song_id = int(song_id or 0)
        self.columns: list[MaterialColumn] = []
        self.slots: list[TimelineSlot] = []
        self._history: list[dict[int, int]] = []      # 撤销栈：每步存一份 picks
        self._future: list[dict[int, int]] = []       # 重做栈
        self._quiet = False                           # 恢复/撤销时不再往栈里压

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("📦 素材池：同一列里随便拖，拖到最上面就是这一格改用它"
                                "（跨段落拖不进去；每次改动立刻存库）", self))
        self.pool_area = QScrollArea(self)
        self.pool_area.setWidgetResizable(True)
        self.pool = QWidget(self.pool_area)
        self.pool_row = QHBoxLayout(self.pool)
        self.pool_row.setContentsMargins(0, 0, 0, 0)
        self.pool_row.setSpacing(6)
        self.pool_area.setWidget(self.pool)
        layout.addWidget(self.pool_area, 3)

        layout.addWidget(QLabel("🎬 FINAL TIMELINE：从上面拖一张卡片下来替换这一格"
                                "（只换「谁负责这一段」，段落起止一个字都不动）", self))
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

    def attach(self, db, song_id: int) -> None:
        """换库 / 换歌。落库和恢复都认这两个。"""
        self.db = db
        self.song_id = int(song_id or 0)
        self._history.clear()
        self._future.clear()

    # ------------------------------------------------------------------ 数据
    def load(self, segments) -> None:
        """`segments` = `[{"index":0,"title":"S1","materials":[{material_id,label,score,path,note}],
        "current":素材id}]`。`current` 一般来自库里的最终选择。"""
        self._clear()
        self._quiet = True
        for spec in segments or ():
            index = int(spec.get("index", 0))
            title = str(spec.get("title") or f"S{index + 1}")
            rows = list(spec.get("materials") or ())
            current = int(spec.get("current") or 0)

            box = QWidget(self.pool)
            column_layout = QVBoxLayout(box)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(2)
            column_layout.addWidget(QLabel(f"{title}　（{len(rows)} 条候选）", box))
            column = MaterialColumn(index, title, box)
            column.load(rows, current)
            column.reordered.connect(self._pick_changed)
            column.refused.connect(self._say_refused)
            column_layout.addWidget(column, 1)

            buttons = QHBoxLayout()
            buttons.setSpacing(4)
            pick = QPushButton("设为当前", box)
            pick.setMinimumHeight(28)
            pick.clicked.connect(lambda _c=False, c=column: self._set_current(c))
            play = QPushButton("▶ 预览", box)
            play.setMinimumHeight(28)
            play.clicked.connect(lambda _c=False, c=column: self._preview(c))
            buttons.addWidget(pick)
            buttons.addWidget(play)
            column_layout.addLayout(buttons)

            self.pool_row.addWidget(box)
            self.columns.append(column)

            slot = TimelineSlot(index, title, self.line)
            chosen = current or (int(rows[0].get("material_id") or 0) if rows else 0)
            label = ""
            for row in rows:
                if int(row.get("material_id") or 0) == chosen:
                    label = str(row.get("label") or "")
            span = spec.get("span") or ()
            if len(span) == 2:
                slot.set_span(float(span[0]), float(span[1]))
            slot.set_material(chosen, label)
            slot.replaced.connect(self._slot_replaced)
            slot.refused.connect(self._say_refused)
            self.line_row.addWidget(slot)
            self.slots.append(slot)

        self.pool_row.addStretch(1)
        self.line_row.addStretch(1)
        self._quiet = False
        total = sum(c.count() for c in self.columns)
        chosen = len(self.picks())
        self.hint.setText(f"{len(self.columns)} 段　·　共 {total} 条候选素材"
                          f"　·　已定 {chosen} 段"
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

    def _column(self, segment_index: int):
        for column in self.columns:
            if column.segment_index == int(segment_index):
                return column
        return None

    def _slot(self, segment_index: int):
        for slot in self.slots:
            if slot.segment_index == int(segment_index):
                return slot
        return None

    def _label_of(self, segment_index: int, material_id: int) -> str:
        column = self._column(segment_index)
        for payload in (column.payloads() if column is not None else ()):
            if int(payload.get("material_id") or 0) == int(material_id):
                return str(payload.get("label") or "")
        return ""

    def _pick_changed(self, segment_index: int, material_id: int) -> None:
        """列里的顺序变了 → FINAL TIMELINE 那一格跟着换成新的第一名，并立刻落库。"""
        before = self.picks()
        slot = self._slot(segment_index)
        if slot is not None:
            slot.set_material(int(material_id), self._label_of(segment_index, material_id))
        if not self._quiet:
            self._history.append(before)
            self._future.clear()
        self._persist(segment_index, int(material_id))
        self.picks_changed.emit(self.picks())

    def _slot_replaced(self, segment_index: int, material_id: int) -> None:
        """往槽里拖 = 这一段改用它，池子里那一列的 ⭐ 也要跟着走（同一条路落库）。"""
        column = self._column(segment_index)
        if column is not None and column.current_pick() != int(material_id):
            column.set_current(int(material_id))       # 会走 _pick_changed，落库在那儿
            return
        self._persist(segment_index, int(material_id))
        self.picks_changed.emit(self.picks())

    def _persist(self, segment_index: int, material_id: int) -> None:
        """落库：最终选择 + 这一段的候选顺序。库里再校验一次跨段落。"""
        if self.db is None or self.song_id <= 0 or material_id <= 0:
            return
        from ...dance import material_repository as repo  # noqa: PLC0415

        column = self._column(segment_index)
        try:
            repo.set_final_selection(self.db, self.song_id, int(segment_index),
                                     int(material_id))
        except repo.SelectionError as exc:
            # 业务层拒绝了（界面理论上拦得住，拦漏了就在这儿兜住）
            self.refused.emit(f"没存：{exc}")
            return
        if column is not None:
            repo.save_candidate_order(
                self.db, self.song_id, int(segment_index),
                [int(p.get("material_id") or 0) for p in column.payloads()])
        self.saved.emit(f"已存：S{int(segment_index) + 1} 用素材 #{material_id}")

    def _set_current(self, column) -> None:
        """「设为当前」：把这一列里选中的那条提到最上面。"""
        item = column.currentItem() or (column.item(0) if column.count() else None)
        if item is None:
            return
        payload = item.data(Qt.UserRole) or {}
        column.set_current(int(payload.get("material_id") or 0))

    def _preview(self, column) -> None:
        """「▶ 预览」：把选中素材的路径发出去，由外面那个播放器播。"""
        item = column.currentItem() or (column.item(0) if column.count() else None)
        payload = (item.data(Qt.UserRole) if item is not None else None) or {}
        path = str(payload.get("path") or "")
        if not path:
            self.refused.emit("这条素材没有文件路径，播不了")
            return
        self.preview_requested.emit(path)

    # -------------------------------------------------------------- 撤销/重做
    def undo(self) -> bool:
        if not self._history:
            return False
        self._future.append(self.picks())
        return self._apply_snapshot(self._history.pop())

    def redo(self) -> bool:
        if not self._future:
            return False
        self._history.append(self.picks())
        return self._apply_snapshot(self._future.pop())

    def _apply_snapshot(self, picks: dict[int, int]) -> bool:
        """把某一份 `{段落: 素材}` 套回界面并落库。撤销/重做共用它。"""
        self._quiet = True
        try:
            for segment_index, material_id in picks.items():
                column = self._column(segment_index)
                if column is not None:
                    column.set_current(int(material_id))
                slot = self._slot(segment_index)
                if slot is not None:
                    slot.set_material(int(material_id),
                                      self._label_of(segment_index, material_id))
                self._persist(int(segment_index), int(material_id))
        finally:
            self._quiet = False
        self.picks_changed.emit(self.picks())
        return True

    def save(self) -> int:
        """Ctrl+S：把当前这份编排整份写一遍（平时每步都自动存，这里是"再确认一次"）。"""
        picks = self.picks()
        for segment_index, material_id in picks.items():
            self._persist(int(segment_index), int(material_id))
        self.saved.emit(f"已保存整份编排：{len(picks)} 段")
        return len(picks)

    def _say_refused(self, target: int, came_from: int) -> None:
        self.refused.emit(f"拖不过去：那条素材绑在 S{int(came_from) + 1}，"
                          f"不能放到 S{int(target) + 1} —— 素材和音乐位置是绑死的")


__all__ = ["MIME", "COLUMN_WIDTH", "SLOT_WIDTH", "SLOT_HEIGHT",
           "pack", "unpack", "accepts",
           "MaterialColumn", "TimelineSlot", "MatrixPanel"]
