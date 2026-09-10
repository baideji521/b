"""素材矩阵：**列 = Segment，行 = 视频，第一行 = 实时播放行**。

    　　　　│    S1    │    S2    │    S3    │    S4    │
    　　　　│  0-3s    │  3-5s    │  5-8s    │  8-12s   │
    ────────┼──────────┼──────────┼──────────┼──────────┤
    ▶ 实时  │   001    │   003    │    —     │   002    │   ← 最终成片就读这一行
    001.mp4 │  [S1]    │  [S2]    │  [S3]    │  [S4]    │
    002.mp4 │  [S1]    │  [S2]    │  [S3]    │  [S4]    │
    003.mp4 │  [S1]    │  [S2]    │  [S3]    │  [S4]    │

用法只有一个动作：**在同一列里把某个视频的格子往上拖到实时播放行**，松手即生效。
不需要"选择 + 应用 + 确认"三步。

**硬限制**：拖动只能在同一列（同一个 Segment）里。`003/S3` 拖到实时播放行的 S1
一律拒绝 —— 素材是"已经绑定到某个音乐位置"的资产，跨段落挪等于让画面和音乐错开。
判定只有一处：`accepts()`；库里 `repo.set_final_selection` 再拦一道。

列的数量完全跟着 SegmentTemplate 走：切一刀就多一列，取消切分就少一列，
调用方重新 `load()` 一次即可，不需要重新导入工程。

性能：格子只是文本卡片，**不给每格建播放器**（100 视频 × 50 段会直接把机器拖死）。
真正的画面预览由外面那一个播放器负责，只跟着实时播放行走。
"""

from __future__ import annotations

import json
from typing import Any

from PyQt5.QtCore import QMimeData, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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

CELL_WIDTH = 132
CELL_HEIGHT = 46
HEAD_WIDTH = 150
PLACEHOLDER = "—"


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


class CandidateCell(QFrame):
    """矩阵里的一格：某个视频在某个 Segment 上的那段素材。按住往上拖就能用它。"""

    picked = pyqtSignal(dict)          # 双击 = 想预览这一格

    def __init__(self, payload: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.payload = dict(payload or {})
        self.segment_index = int(self.payload.get("segment_index", -1))
        self.material_id = int(self.payload.get("material_id") or 0)
        self._press: QPoint | None = None
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(CELL_WIDTH, CELL_HEIGHT)
        self.setToolTip(f"{self.payload.get('video_name') or ''}　"
                        f"S{self.segment_index + 1}\n"
                        f"{self.payload.get('note') or ''}\n"
                        "按住往上拖到「实时播放」那一行 = 这一段用它（只能在本列内拖）")

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 3, 6, 3)
        box.setSpacing(0)
        self.title = QLabel(str(self.payload.get("label") or self.material_id), self)
        self.title.setToolTip(self.title.text())
        self.detail = QLabel(str(self.payload.get("detail") or ""), self)
        self.detail.setStyleSheet(f"color:{theme.TEXT_DIM};")
        box.addWidget(self.title)
        box.addWidget(self.detail)

    def mousePressEvent(self, event) -> None:            # noqa: N802 - Qt 的名字
        if event.button() == Qt.LeftButton:
            self._press = event.pos()

    def mouseMoveEvent(self, event) -> None:             # noqa: N802
        """走了几个像素才算拖：不然轻轻一点也会启动拖拽，手一抖就换素材。"""
        if self._press is None or not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self._press).manhattanLength() < 12:
            return
        drag = QDrag(self)
        drag.setMimeData(pack(self.payload))
        drag.exec_(Qt.CopyAction)
        self._press = None

    def mouseDoubleClickEvent(self, event) -> None:      # noqa: N802
        self.picked.emit(dict(self.payload))


class RealtimeCell(QFrame):
    """实时播放行的一格：这一段**最终用谁**。只接同一列拖过来的素材。"""

    replaced = pyqtSignal(int, int)            # 段落, 新的素材 id
    refused = pyqtSignal(int, int)            # 段落, 被拒的那条属于哪一段

    def __init__(self, segment_index: int, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.segment_index = int(segment_index)
        self.title = title or f"S{self.segment_index + 1}"
        self.material_id = 0
        self.payload: dict[str, Any] = {}
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(CELL_WIDTH, CELL_HEIGHT)

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 3, 6, 3)
        box.setSpacing(0)
        self.body = QLabel(PLACEHOLDER, self)
        self.hint = QLabel("拖一格上来", self)
        self.hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        box.addWidget(self.body)
        box.addWidget(self.hint)
        self._paint_state(False)

    def _paint_state(self, playing: bool) -> None:
        """当前正在播的那一列描个边；空的那格淡着显示（一眼看出"这段没素材"）。"""
        edge = theme.PLAYING if playing else theme.LINE
        self.setStyleSheet(f"QFrame{{border:{2 if playing else 1}px solid {edge};}}")
        self.body.setStyleSheet("" if self.material_id else f"color:{theme.TEXT_DIM};")

    def set_playing(self, playing: bool) -> None:
        self._paint_state(bool(playing))

    def set_material(self, payload: dict[str, Any] | None) -> None:
        """摆上（或清掉）这一格的素材。空 = 这一段没人负责，播放时就是黑的。"""
        self.payload = dict(payload or {})
        self.material_id = int(self.payload.get("material_id") or 0)
        self.body.setText(str(self.payload.get("label") or PLACEHOLDER))
        self.hint.setText(str(self.payload.get("video_name") or "")
                          if self.material_id else "拖一格上来")
        self._paint_state(False)

    def dragEnterEvent(self, event) -> None:             # noqa: N802
        self._filter(event)

    def dragMoveEvent(self, event) -> None:              # noqa: N802
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

    def dropEvent(self, event) -> None:                  # noqa: N802
        if self.drop_payload(unpack(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def drop_payload(self, payload: dict[str, Any]) -> bool:
        """换掉这一格。跨段落返回 False 并且**什么都不改**。

        单独抽出来是为了能不靠真实鼠标拖拽来测这条规则 —— QDrag 在离屏环境下
        跑不起来，但"能不能落"必须有测试盯着。
        """
        if not accepts(self.segment_index, payload):
            self.refused.emit(self.segment_index, int(payload.get("segment_index", -1)))
            return False
        self.set_material(payload)
        self.replaced.emit(self.segment_index, self.material_id)
        return True


class MatrixPanel(QWidget):
    """素材矩阵 + 实时播放行。

    每一次改动都**立刻落库**：`dance_final_selections`（这一段最终用谁）、
    `dance_candidate_order`（这一段候选的排列）、`dance_manual_selections`
    （人工挑选的流水账，只追加不覆盖，以后自动编排要拿它当"人给的答案"）。
    重新打开工程时按这些表恢复 —— 界面状态不是唯一真相。

    库里那一步还会**再校验一次**素材的 segment_index（`repo.set_final_selection`），
    所以跨段落的选择拦两道：界面不给落，业务层也不给写。
    """

    picks_changed = pyqtSignal(dict)           # {段落: 素材id}
    refused = pyqtSignal(str)                  # 给状态栏的一句人话
    preview_requested = pyqtSignal(str)        # 素材文件路径
    saved = pyqtSignal(str)                    # 落库之后的一句人话
    realtime_changed = pyqtSignal(int, int)    # 段落, 现在这一段用哪条素材

    def __init__(self, parent=None, db=None, song_id: int = 0) -> None:
        super().__init__(parent)
        self.db = db
        self.song_id = int(song_id or 0)
        self.cells: dict[tuple[int, int], CandidateCell] = {}   # (视频id, 段落) → 格子
        self.realtime: list[RealtimeCell] = []
        self.videos: list[dict[str, Any]] = []
        self.segments: list[dict[str, Any]] = []
        self._current = -1                            # 正在播第几段
        self._last_picks: dict[int, int] = {}         # 改之前那份选择（撤销栈压它）
        self._history: list[dict[int, int]] = []      # 撤销栈：每步存一份 picks
        self._future: list[dict[int, int]] = []       # 重做栈
        self._quiet = False                           # 恢复/撤销时不再往栈里压

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("📦 素材矩阵：列＝Segment，行＝视频，第一行＝实时播放（成片就读它）。"
                                "把某个视频的格子往上拖到实时播放行 = 这一段用它；"
                                "跨列拖不过去。", self))

        self.area = QScrollArea(self)
        self.area.setWidgetResizable(True)
        self.board = QWidget(self.area)
        self.grid = QGridLayout(self.board)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(4)
        self.area.setWidget(self.board)
        layout.addWidget(self.area, 1)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.hint = QLabel("还没有素材。先定好分段，再把源视频对齐入库。", self)
        self.hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        self.btn_clear = QPushButton("清掉这一段的选择", self)
        self.btn_clear.setMinimumHeight(30)
        self.btn_clear.setToolTip("把「实时播放」行里当前那一段清空（这一段就变成没素材）")
        self.btn_clear.clicked.connect(self._clear_current)
        bar.addWidget(self.hint, 1)
        bar.addWidget(self.btn_clear)
        layout.addLayout(bar)

    def attach(self, db, song_id: int) -> None:
        """换库 / 换歌。落库和恢复都认这两个。"""
        self.db = db
        self.song_id = int(song_id or 0)
        self._history.clear()
        self._future.clear()

    # ------------------------------------------------------------------ 数据
    def load(self, segments) -> None:
        """按 SegmentTemplate 重建整张矩阵。

        `segments` = `[{"index":0, "title":"S1", "span":(0.0,3.0),
                        "materials":[{material_id, label, path, video_id, video_name,
                                      detail, note}],
                        "current": 素材id}]`

        列数 = `len(segments)`，所以切分/取消切分之后调用方重新 `load()`
        矩阵就跟着增列减列；行 = 出现过的所有视频（按第一次出现的顺序）。
        """
        self._clear()
        self._quiet = True
        self.segments = [dict(spec) for spec in segments or ()]

        # 行的顺序：先按候选顺序里出现的先后，保证"同一个视频永远在同一行"
        order: list[dict[str, Any]] = []
        seen: set[int] = set()
        for spec in self.segments:
            for row in spec.get("materials") or ():
                video_id = int(row.get("video_id") or 0)
                if video_id in seen:
                    continue
                seen.add(video_id)
                order.append({"video_id": video_id,
                              "video_name": str(row.get("video_name") or f"#{video_id}")})
        self.videos = order

        self._build_header()
        self._build_realtime()
        self._build_rows()
        self.grid.setRowStretch(len(self.videos) + 2, 1)
        self.grid.setColumnStretch(len(self.segments) + 1, 1)
        self._quiet = False
        self._last_picks = self.picks()
        self._say_summary()

    def _build_header(self) -> None:
        head = QLabel("视频 ＼ 段落", self.board)
        head.setFixedWidth(HEAD_WIDTH)
        self.grid.addWidget(head, 0, 0)
        for column, spec in enumerate(self.segments, start=1):
            span = spec.get("span") or ()
            title = str(spec.get("title") or f"S{int(spec.get('index', 0)) + 1}")
            text = title if len(span) != 2 else \
                f"{title}\n{float(span[0]):.3f}→{float(span[1]):.3f}s"
            label = QLabel(text, self.board)
            label.setFixedWidth(CELL_WIDTH)
            label.setStyleSheet(f"color:{theme.ACCENT};")
            self.grid.addWidget(label, 0, column)

    def _build_realtime(self) -> None:
        head = QLabel("▶ 实时播放", self.board)
        head.setFixedWidth(HEAD_WIDTH)
        head.setToolTip("成片就读这一行；某一段空着 = 那一段没有素材，播放时画面就是空的")
        self.grid.addWidget(head, 1, 0)
        for column, spec in enumerate(self.segments, start=1):
            index = int(spec.get("index", column - 1))
            cell = RealtimeCell(index, str(spec.get("title") or ""), self.board)
            chosen = int(spec.get("current") or 0)
            for row in spec.get("materials") or ():
                if int(row.get("material_id") or 0) == chosen:
                    cell.set_material(row)
            cell.replaced.connect(self._realtime_replaced)
            cell.refused.connect(self._say_refused)
            self.grid.addWidget(cell, 1, column)
            self.realtime.append(cell)

    def _build_rows(self) -> None:
        by_segment = {int(spec.get("index", i)): (spec.get("materials") or ())
                      for i, spec in enumerate(self.segments)}
        for row_index, video in enumerate(self.videos, start=2):
            head = QLabel(str(video["video_name"]), self.board)
            head.setFixedWidth(HEAD_WIDTH)
            head.setToolTip(str(video["video_name"]))
            self.grid.addWidget(head, row_index, 0)
            for column, spec in enumerate(self.segments, start=1):
                index = int(spec.get("index", column - 1))
                payload = self._payload_of(by_segment.get(index, ()), video["video_id"])
                if payload is None:
                    empty = QLabel(PLACEHOLDER, self.board)
                    empty.setFixedSize(CELL_WIDTH, CELL_HEIGHT)
                    empty.setAlignment(Qt.AlignCenter)
                    empty.setStyleSheet(f"color:{theme.TEXT_DIM};")
                    empty.setToolTip(f"这个视频在 S{index + 1} 没有可用素材")
                    self.grid.addWidget(empty, row_index, column)
                    continue
                cell = CandidateCell(payload, self.board)
                cell.picked.connect(lambda p: self.preview_requested.emit(str(p.get("path") or "")))
                self.grid.addWidget(cell, row_index, column)
                self.cells[(int(video["video_id"]), index)] = cell

    @staticmethod
    def _payload_of(rows, video_id: int) -> dict[str, Any] | None:
        for row in rows:
            if int(row.get("video_id") or 0) == int(video_id):
                return dict(row)
        return None

    def _clear(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.cells.clear()
        self.realtime.clear()
        self.videos.clear()
        self.segments.clear()
        self._current = -1

    # ------------------------------------------------------------------ 交互
    def picks(self) -> dict[int, int]:
        """当前每一段用哪条素材。**实时播放行摆着的那份才算数**（成片也读它）。"""
        return {cell.segment_index: cell.material_id
                for cell in self.realtime if cell.material_id}

    def current_payload(self, segment_index: int) -> dict[str, Any]:
        """这一段实时播放行上那条素材的全部信息（空段返回空字典）。"""
        cell = self._cell(int(segment_index))
        return dict(cell.payload) if cell is not None else {}

    def set_current_segment(self, segment_index: int) -> None:
        """主音频播到哪一段了 → 那一列描边。**只是显示**，不改任何选择。"""
        index = int(segment_index)
        if index == self._current:
            return
        self._current = index
        for cell in self.realtime:
            cell.set_playing(cell.segment_index == index)

    @property
    def current_segment(self) -> int:
        return self._current

    def _cell(self, segment_index: int):
        for cell in self.realtime:
            if cell.segment_index == int(segment_index):
                return cell
        return None

    def choose(self, segment_index: int, material_id: int) -> bool:
        """用代码指定"这一段用谁"（等于把那一格拖上去）。跨段落一律 False。"""
        cell = self._cell(segment_index)
        if cell is None:
            return False
        for (video_id, index), candidate in self.cells.items():
            if index == int(segment_index) and candidate.material_id == int(material_id):
                del video_id      # 只是用来找格子，本身不参与判断
                return cell.drop_payload(candidate.payload)
        return False

    def _realtime_replaced(self, segment_index: int, material_id: int) -> None:
        """实时播放行换人 = 一次人工决策：立刻落库 + 记流水账 + 通知外面换预览。

        撤销栈压的是 `self._last_picks`（**改之前**那份），不是现在这份 ——
        这个信号是格子改完才发的，此时 `picks()` 已经是新状态了。
        """
        if not self._quiet:
            self._history.append(dict(self._last_picks))
            self._future.clear()
        self._persist(int(segment_index), int(material_id))
        self._last_picks = self.picks()
        self.picks_changed.emit(self.picks())
        self.realtime_changed.emit(int(segment_index), int(material_id))
        self._say_summary()

    def _clear_current(self) -> None:
        """把当前那一段清空：这一段就变成"没素材"，播放时画面保持空。"""
        index = self._current if self._current >= 0 else (
            self.realtime[0].segment_index if self.realtime else -1)
        cell = self._cell(index)
        if cell is None:
            return
        if not self._quiet:
            self._history.append(dict(self._last_picks))
            self._future.clear()
        cell.set_material(None)
        if self.db is not None and self.song_id > 0:
            from ...dance import material_repository as repo  # noqa: PLC0415

            repo.clear_final_selection(self.db, self.song_id, int(index))
        self._last_picks = self.picks()
        self.picks_changed.emit(self.picks())
        self.realtime_changed.emit(int(index), 0)
        self._say_summary()


    def _persist(self, segment_index: int, material_id: int) -> None:
        """落库：最终选择 + 这一段的候选顺序 + 人工选择流水账。

        库里会**再校验一次**素材的 segment_index，跨段落在这一层也写不进去。
        """
        if self.db is None or self.song_id <= 0 or material_id <= 0:
            return
        from ...dance import material_repository as repo  # noqa: PLC0415

        try:
            repo.set_final_selection(self.db, self.song_id, int(segment_index),
                                     int(material_id))
        except repo.SelectionError as exc:
            # 业务层拒绝了（界面理论上拦得住，拦漏了就在这儿兜住）
            self.refused.emit(f"没存：{exc}")
            return
        span = self._span_of(int(segment_index))
        try:
            repo.log_manual_selection(self.db, target_song_id=self.song_id,
                                      segment_index=int(segment_index),
                                      material_id=int(material_id),
                                      segment_start=span[0], segment_end=span[1])
        except repo.SelectionError as exc:      # 理论到不了这儿，到了也别把界面搞崩
            logger.warning("人工选择没记上：%s", exc)
        order = [int(self.cells[(int(v["video_id"]), int(segment_index))].material_id)
                 for v in self.videos
                 if (int(v["video_id"]), int(segment_index)) in self.cells]
        repo.save_candidate_order(self.db, self.song_id, int(segment_index), order)
        self.saved.emit(f"已存：S{int(segment_index) + 1} 用素材 #{material_id}")

    def _span_of(self, segment_index: int) -> tuple[float, float]:
        for spec in self.segments:
            if int(spec.get("index", -1)) == int(segment_index):
                span = spec.get("span") or ()
                if len(span) == 2:
                    return (float(span[0]), float(span[1]))
        return (0.0, 0.0)

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
            for cell in self.realtime:
                wanted = int(picks.get(cell.segment_index, 0) or 0)
                if wanted <= 0:
                    cell.set_material(None)
                    continue
                self.choose(cell.segment_index, wanted)
        finally:
            self._quiet = False
        self._last_picks = self.picks()
        self.picks_changed.emit(self.picks())
        self._say_summary()
        return True

    def save(self) -> int:
        """Ctrl+S：把当前这份编排整份写一遍（平时每步都自动存，这里是"再确认一次"）。"""
        picks = self.picks()
        for segment_index, material_id in picks.items():
            self._persist(int(segment_index), int(material_id))
        self.saved.emit(f"已保存整份编排：{len(picks)} 段")
        return len(picks)

    # ------------------------------------------------------------------ 话术
    def _say_refused(self, target: int, came_from: int) -> None:
        self.refused.emit(f"拖不过去：那条素材绑在 S{int(came_from) + 1}，"
                          f"不能放到 S{int(target) + 1} —— 素材和音乐位置是绑死的")

    def _say_summary(self) -> None:
        if not self.segments:
            self.hint.setText("还没有素材。先定好分段，再把源视频对齐入库。")
            return
        chosen = len(self.picks())
        empty = [f"S{c.segment_index + 1}" for c in self.realtime if not c.material_id]
        tail = f"　·　还空着：{'、'.join(empty)}（这些段播放时画面就是空的）" if empty else ""
        self.hint.setText(f"{len(self.segments)} 段　·　{len(self.videos)} 个视频"
                          f"　·　共 {len(self.cells)} 格候选　·　已定 {chosen} 段{tail}")


__all__ = ["MIME", "CELL_WIDTH", "CELL_HEIGHT", "HEAD_WIDTH",
           "pack", "unpack", "accepts",
           "CandidateCell", "RealtimeCell", "MatrixPanel"]






