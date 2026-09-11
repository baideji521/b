"""右侧「视频列表」：视频文件夹里的每个视频 + 它跟主音频的对齐结果。

    #   视频                     时长      状态        Offset    置信度
    ─────────────────────────────────────────────────────────────────
    1   202609100047_tiktok.mp4  62.40s   ✅ 可用     +3.201    0.981
    2   girl02.mp4               61.80s   ⚠ 低置信    +8.700    0.431

第一列「#」是**屏幕上的序号**：从 1 开始连号，点表头换排序、删行、粘进新视频
之后都会重新数一遍，所以它永远等于"这是从上往下第几个"。它**不是身份**——
每一行的真身还是存在「视频」那一格里的全路径（`Qt.UserRole`）。


**只列文件夹里的视频，不读库。**对齐算完就填进来，一条素材都不会因此产生 ——
入库是页脚「切片并加入素材库」那个按钮的事，两件事分开。

选了文件夹立刻就有行（状态「未对齐」），所以点「开始音频对齐」之前
就能看清这次要处理哪些文件、有没有选错目录。**但对齐跑完之后，
还挂着「未对齐」的行会被清掉** —— 那种行只有两种来源：算之前的占位、
或者路径写法不一样导致结果落到了新行上，留着只会让人以为"这个没算"。

右键菜单：全选 / 复制（文件本身，能直接粘到资源管理器）/ 粘贴（把复制来的
视频加进列表）/ 删除。**删除是真删本地文件**，所以先弹一次确认，且不可撤销。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QMimeData, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QShortcut,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme

COLUMNS = ("#", "视频", "时长", "状态", "Offset", "置信度")

#: 列号一律走这几个名字，别再写字面量 —— 加了「#」之后所有列都往右挪了一格
COL_INDEX = 0
COL_NAME = 1
COL_STATUS = 3


#: 比全局 12px 小一号：这一栏是"扫一眼几十行"的清单，字小才装得下
FONT_SIZE = 11
ROW_HEIGHT = 22
PLACEHOLDER = "—"
#: 还没算过的行长这样。对齐跑完之后凡是还写着这个的行都要清掉
UNALIGNED = "未对齐"

#: 粘贴/拖进来时认这些后缀（和 align_bench.VIDEO_SUFFIXES 一致）
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".m4v", ".webm")

#: 对齐状态 → 人话。和 align_bench 的 STATUS_TEXT 同一套说法
STATUS_TEXT = {
    "ok": "✅ 可用",
    "manual": "✍ 手动改过",
    "low_confidence": "⚠ 低置信",
    "rejected": "✖ 不一致",
    "failed": "✖ 失败",
}


class VideoListPanel(QWidget):
    """视频文件夹里的视频清单 + 对齐结果。**只显示，不落库、不改任何东西。**"""

    picked = pyqtSignal(str)          # 双击某一行 = 想看这个视频（发文件全路径）
    about_to_delete = pyqtSignal(list)  # 马上要删这些文件 → 外面赶紧松开占用
    removed = pyqtSignal(list)        # 右键删除 = 这些文件已经从磁盘上没了
    added = pyqtSignal(list)          # 右键粘贴 = 这些文件进了列表

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # 表格的 sizeHint 会按"五列都摊开"来要宽度，摆在分栏里会把左半边挤没。
        # 横向声明 Ignored：宽度由分栏和 `setMinimumWidth` 说话，不由表格自己要
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        self.hint = QLabel("📄 视频列表：先在顶栏选「视频文件夹」，"
                           "这里就列出里面的视频；对齐算完填 Offset 和置信度"
                           "（**不会入库**）。", self)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:{FONT_SIZE}px;")
        layout.addWidget(self.hint)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.setSortingEnabled(True)
        #: 正在重新数序号。`sortIndicatorChanged` 会在排序时打回来，不挡一下会套娃
        self._numbering = False
        # 字号比全局小一号：几十行的清单，字小一点一屏能多看十几行
        self.table.setStyleSheet(f"font-size:{FONT_SIZE}px;")
        self.table.horizontalHeader().setStyleSheet(f"font-size:{FONT_SIZE}px;")
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            COL_INDEX, QHeaderView.ResizeToContents)
        # 换了排序方式，序号要跟着重新数：它表示"从上往下第几个"，不是身份
        self.table.horizontalHeader().sortIndicatorChanged.connect(
            lambda *_args: self._renumber())

        self.table.doubleClicked.connect(self._row_picked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        # 键盘走一遍同样的动作。Ctrl+A 是 QTableWidget 自带的，其余三个自己绑；
        # 作用域限定在这张表上，别去抢窗口级快捷键
        for keys, handler in (("Ctrl+C", self.copy_selected),
                              ("Ctrl+V", self.paste_files),
                              ("Del", self.delete_selected)):
            short = QShortcut(QKeySequence(keys), self.table)
            short.setContext(Qt.WidgetWithChildrenShortcut)
            short.activated.connect(handler)
        layout.addWidget(self.table, 1)

    # ------------------------------------------------------------------ 右键
    def _menu(self, where) -> None:
        """右键：全选 / 复制 / 粘贴 / 删除。**删除是真删本地文件。**"""
        menu = QMenu(self)
        menu.addAction("全选（Ctrl+A）", self.table.selectAll)
        act_copy = menu.addAction("复制（Ctrl+C）", self.copy_selected)
        act_paste = menu.addAction("粘贴（Ctrl+V）", self.paste_files)
        menu.addSeparator()
        act_delete = menu.addAction("删除（连本地文件一起删）", self.delete_selected)
        picked = self.selected_paths()
        act_copy.setEnabled(bool(picked))
        act_delete.setEnabled(bool(picked))
        act_paste.setEnabled(bool(self._clipboard_files()))
        menu.exec_(self.table.viewport().mapToGlobal(where))

    def selected_paths(self) -> list[str]:
        """选中行对应的文件（按屏幕顺序，去重）。"""
        out: list[str] = []
        for index in sorted({i.row() for i in self.table.selectedIndexes()}):
            path = self._path_at(index)
            if path and path not in out:
                out.append(path)
        return out

    def copy_selected(self) -> list[str]:
        """复制：既放 URL（资源管理器里能直接粘贴出文件），也放纯文本路径。"""
        picked = self.selected_paths()
        if not picked:
            return []
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(p) for p in picked])
        data.setText("\n".join(picked))
        QApplication.clipboard().setMimeData(data)      # 剪贴板接管所有权
        self._say_line(f"已复制 {len(picked)} 个视频到剪贴板")
        return picked

    def _clipboard_files(self) -> list[str]:
        """剪贴板里的视频文件（URL 优先，其次一行一个路径）。"""
        data = QApplication.clipboard().mimeData()
        found: list[str] = []
        if data is None:
            return found
        candidates = [url.toLocalFile() for url in data.urls()] if data.hasUrls() else []
        if not candidates and data.hasText():
            candidates = [line.strip().strip('"') for line in data.text().splitlines()]
        for raw in candidates:
            path = Path(raw) if raw else None
            if path is None or not path.is_file():
                continue
            if path.suffix.lower() in VIDEO_SUFFIXES:
                found.append(str(path))
        return found

    def paste_files(self) -> list[str]:
        """粘贴：把剪贴板里的视频**加进列表**（不复制文件、不动磁盘）。"""
        fresh = [p for p in self._clipboard_files() if self._row_of(p) is None]
        if not fresh:
            self._say_line("剪贴板里没有能加进来的视频")
            return []
        self.add_files(fresh)
        self.added.emit(list(fresh))
        return fresh

    def delete_selected(self) -> list[str]:
        """删除：**连本地文件一起删**，不可撤销，所以先确认一次。

        删之前先发 `about_to_delete`，让外面**松开文件**（播放器还开着的话
        Windows 会锁住这个文件，unlink 直接 PermissionError）。真被占住的再等一下重试
        一次 —— 系统释放句柄有时慢半拍。
        """
        picked = self.selected_paths()
        if not picked:
            return []
        names = "\n".join(Path(p).name for p in picked[:12])
        more = f"\n…… 还有 {len(picked) - 12} 个" if len(picked) > 12 else ""
        answer = QMessageBox.warning(
            self, "删除视频文件",
            f"要把这 {len(picked)} 个视频**从本地文件夹里删掉**吗？\n"
            f"删了就找不回来了（不进回收站）。\n\n{names}{more}",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return []
        self.about_to_delete.emit(list(picked))
        QApplication.processEvents()          # 让上面那些"松手"的活真的做完
        gone, failed = [], []
        for path in picked:
            error = self._unlink(path)
            if error:
                failed.append(f"{Path(path).name}：{error}")
                continue
            gone.append(path)
        self.drop_files(gone)
        if gone:
            self.removed.emit(list(gone))
        if failed:
            QMessageBox.warning(self, "有几个删不掉",
                                "可能正被别的程序占用：\n" + "\n".join(failed[:10]))
        self._say_line(f"已删除 {len(gone)} 个视频文件"
                       + (f"，{len(failed)} 个删不掉" if failed else ""))
        return gone

    @staticmethod
    def _unlink(path: str) -> str:
        """删一个文件，返回空串表示删掉了，否则返回原因。占用类错误重试一次。"""
        for attempt in (0, 1):
            try:
                Path(path).unlink(missing_ok=True)   # 文件早就没了也算删成功
                return ""
            except PermissionError as exc:
                if attempt:
                    return str(exc.strerror or exc)
                time.sleep(0.2)                     # 句柄刚松开，系统还没反应过来
            except OSError as exc:
                return str(exc.strerror or exc)
        return "一直被占用"

    def add_files(self, paths) -> int:
        """往列表里补行（不清空已有的），返回真加了几行。"""
        self.table.setSortingEnabled(False)
        added = 0
        for raw in paths or ():
            path = str(raw)
            if not path or self._row_of(path) is not None:
                continue
            row = self.table.rowCount()
            self.table.setRowCount(row + 1)
            self._write(row, path, (Path(path).name, PLACEHOLDER, UNALIGNED,
                                    PLACEHOLDER, PLACEHOLDER))
            added += 1
        self.table.setSortingEnabled(True)
        self.table.sortItems(COL_NAME, Qt.AscendingOrder)
        self._renumber()
        self._say(self._aligned_count())
        return added


    def drop_files(self, paths) -> int:
        """把这些文件的行从表里去掉（只动表格，磁盘的事由调用方负责）。"""
        gone = 0
        for raw in paths or ():
            row = self._row_of(str(raw))
            if row is None:
                continue
            self.table.removeRow(row)
            gone += 1
        if gone:
            self._renumber()                      # 删掉几行，序号重新连上
            self._say(self._aligned_count())
        return gone


    def _aligned_count(self) -> int:
        return sum(1 for row in range(self.table.rowCount())
                   if (self.table.item(row, COL_STATUS).text()
                       if self.table.item(row, COL_STATUS) else "")
                   in STATUS_TEXT.values())


    def _say_line(self, line: str) -> None:
        """在提示行尾巴上补一句刚做了什么（不覆盖清单统计）。"""
        self.hint.setText(f"{self.hint.text()}　·　{line}")

    def _row_picked(self) -> None:
        path = self._path_at(self.table.currentRow())
        if path:
            self.picked.emit(path)

    def _path_at(self, row: int) -> str:
        """这一行是哪个文件。**路径存在单元格里**，不靠行号 —— 表头一点就重排了。"""
        item = self.table.item(int(row), COL_NAME)
        return str(item.data(Qt.UserRole) or "") if item is not None else ""


    def _row_of(self, path: str) -> int | None:
        """按**规范化后的路径**找行。

        Windows 上同一个文件能写成好几个样子（`D:/a\\b.mp4` / `D:\\A\\B.MP4`），
        直接比字符串会认不出来 —— 那样对齐结果会另开一行，原来那行就永远挂着
        「未对齐」，看着像"这个没算"。这就是之前那些碍眼的假行。
        """
        want = self._key(path)
        for row in range(self.table.rowCount()):
            if self._key(self._path_at(row)) == want:
                return row
        return None

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.normpath(str(path))) if path else ""

    def set_files(self, paths) -> None:
        """把文件夹里的视频铺成行。**这一步不解码、不读库**，所以几十个文件也是瞬间。

        时长要解码才知道，所以对齐之前留「—」；对齐算完 `set_results` 会补上。
        """
        files = [str(p) for p in (paths or ())]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(files))
        for row, path in enumerate(files):
            self._write(row, path, (Path(path).name, PLACEHOLDER, UNALIGNED,
                                    PLACEHOLDER, PLACEHOLDER))
        self.table.setSortingEnabled(True)
        # 明确按文件名升序：不指定的话 Qt 会沿用上一次的排序指示器，
        # 铺完行顺序就变了（曾经因此把对齐结果写到了别人那一行）
        self.table.sortItems(COL_NAME, Qt.AscendingOrder)
        self._renumber()
        self._say(0)


    def set_results(self, rows) -> None:
        """填对齐结果。`rows` 就是对齐台那份 `[{path,name,alignment,error}]`。

        按**路径**找行，不按行号：表格可以点表头排序，行号随时会变。
        不在列表里的（单选的那条）补一行。

        填完把**还挂着「未对齐」的行清掉**：这一批算过的都已经有状态了，
        剩下那些只是算之前的占位，留着纯碍眼（"明明对齐了还写着未对齐"）。
        """
        self.table.setSortingEnabled(False)
        done = 0
        for item in rows or ():
            path = str(item.get("path") or "")
            if not path:
                continue
            row = self._row_of(path)
            if row is None:
                row = self.table.rowCount()
                self.table.setRowCount(row + 1)
            align = item.get("alignment")
            if align is None:
                error = str(item.get("error") or "未知")
                self._write(row, path, (Path(path).name, PLACEHOLDER,
                                        f"✖ 失败：{error}", PLACEHOLDER, PLACEHOLDER))
                continue
            done += 1
            self._write(row, path, (Path(path).name,
                                    f"{float(align.source_duration or 0.0):.2f}s",
                                    STATUS_TEXT.get(align.status, str(align.status)),
                                    f"{float(align.offset):+.3f}",
                                    f"{float(align.confidence):.3f}"))
        self._drop_unaligned()
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        self._renumber()
        self._say(done)


    def _drop_unaligned(self) -> int:
        """清掉状态还是「未对齐」的行。倒着删，行号才不会边删边错位。"""
        gone = 0
        for row in range(self.table.rowCount() - 1, -1, -1):
            cell = self.table.item(row, COL_STATUS)

            if cell is not None and cell.text() == UNALIGNED:
                self.table.removeRow(row)
                gone += 1
        return gone

    def _write(self, row: int, path: str, cells) -> None:
        """写一行的正文（序号那一格由 `_renumber()` 统一填）。"""
        for offset, text in enumerate(cells):
            column = COL_NAME + offset
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if column == COL_NAME
                                  else Qt.AlignRight | Qt.AlignVCenter)
            if column == COL_NAME:
                item.setData(Qt.UserRole, path)   # 行的真身：全路径
                item.setToolTip(path)
            self.table.setItem(row, column, item)

    def _renumber(self) -> None:
        """把「#」那一列重新数成 1..N（按**屏幕上现在的顺序**）。

        序号写成整数（`setData(DisplayRole, int)`）：按这一列排序时才是 1,2,…,10,11，
        写成字符串会变成 1,10,11,2 那种字典序。
        重数期间关掉排序 —— 一边排一边写会把行顺序搅乱，而且 `sortIndicatorChanged`
        还会打回这里来（所以另外用 `_numbering` 挡一层）。
        """
        if self._numbering:
            return
        self._numbering = True
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        try:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, COL_INDEX)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(row, COL_INDEX, item)
                item.setData(Qt.DisplayRole, row + 1)
        finally:
            self.table.setSortingEnabled(sorting)
            self._numbering = False


    def _say(self, aligned: int) -> None:
        total = self.table.rowCount()
        if not total:
            self.hint.setText("📄 视频列表：先在顶栏选「视频文件夹」，"
                              "这里就列出里面的视频；对齐算完填 Offset 和置信度"
                              "（不会入库）。")
            return
        tail = f"　·　已对齐 {aligned} 个" if aligned else "　·　还没对齐"
        self.hint.setText(f"📄 视频列表：文件夹里 {total} 个视频{tail}"
                          "　·　右键可全选/复制/粘贴/删除（删除连本地文件一起删）")

    def rows(self) -> list[dict[str, Any]]:
        """当前表里每一行的文字（按屏幕上的顺序，给测试和日志用）。"""
        out = []
        for row in range(self.table.rowCount()):
            out.append({COLUMNS[column]: (self.table.item(row, column).text()
                                          if self.table.item(row, column) else "")
                        for column in range(len(COLUMNS))})
        return out


__all__ = ["VideoListPanel", "COLUMNS", "COL_INDEX", "COL_NAME", "COL_STATUS",
           "STATUS_TEXT", "FONT_SIZE", "UNALIGNED"]



