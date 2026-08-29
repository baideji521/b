"""AI 面板（第二主界面）+ AI 接口设置。

- `AiPanel`：自动剪辑那一摊——三种模式、AI 专属目录、任务统计、每个 mp4 卡在哪一步的
  任务表、开跑/停止和日志。非模态，开着它照样能用主界面；也能单独跑（run.py ai）。
- `AiApiDialog`：AI 接口本身的设置——找哪家 AI、走接口还是网页版扩展、key、模型、超时、
  Bridge 端口、扩展上传方式。主界面上的「AI接口」按钮开这个。

两个窗口各写各的键（都是 config.json 的 bridge 一节，只写自己那几个），互不覆盖。
AI_输入目录 / AI_输出目录 只归 AI 用，跟界面第一行的「导入文件」「导出目录…」互不相干。
"""


from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..bridge import providers
from ..db import open_db
from ..db import repo as db_repo
from ..db.importer import refresh_from_disk

# 任务表里每一步的记号：干完了 / 正在干 / 还没轮到 / 砸了
DONE, RUNNING, WAITING, FAILED = "✓", "●", "—", "✕"

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".m4v",
                  ".ts", ".mpg", ".mpeg", ".wmv")

JOBS = (
    ("full", "剪辑成片", "MP4 + TXT\n↓\nJSON\n↓\n自动剪辑"),
    ("collect", "收取脚本", "MP4 + TXT\n↓\nJSON\n↓\n保存 JSON"),
    ("script", "脚本剪辑", "MP4 + JSON\n↓\n自动剪辑\n↓\n成品"),
)


class DropDirEdit(QLineEdit):
    """能直接把文件夹拖进来的路径输入框。

    拖文件进来也认——取它所在的文件夹，省得你先打开一层再拖。
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("把文件夹拖进来，或点右边「浏览…」")

    @staticmethod
    def _folder_from(event) -> str | None:
        data = event.mimeData()
        if not data.hasUrls():
            return None
        for url in data.urls():
            local = url.toLocalFile()
            if not local:
                continue
            path = Path(local)
            if path.is_dir():
                return str(path)
            if path.exists():  # 拖进来的是文件：用它所在的目录
                return str(path.parent)
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 的命名
        if self._folder_from(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._folder_from(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        folder = self._folder_from(event)
        if folder:
            self.setText(folder)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


def _open_dir(widget, raw: str) -> None:
    """在文件管理器里打开这个目录；不存在就问一句要不要建。"""
    text = (raw or "").strip()
    if not text:
        QMessageBox.information(widget, "AI 面板", "先填个目录")
        return
    path = Path(text)
    if not path.is_dir():
        ok = QMessageBox.question(widget, "AI 面板", f"目录还不存在：\n{path}\n\n要现在建吗？",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ok != QMessageBox.Yes:
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(widget, "AI 面板", f"建不了：{exc}")
            return
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - 打开自己选的目录
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _dir_row(owner, edit: DropDirEdit, title: str) -> QWidget:
    """一行：路径框 + 选择目录 + 打开。返回装好的容器控件。"""
    browse = QPushButton("选择目录")
    browse.clicked.connect(lambda: _browse_into(owner, edit, title))
    opener = QPushButton("打开")
    opener.clicked.connect(lambda: _open_dir(owner, edit.text()))
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    row.addWidget(browse)
    row.addWidget(opener)
    return box


def _browse_into(owner, edit: DropDirEdit, title: str) -> None:
    start = edit.text().strip() or str(owner.cfg.root)
    chosen = QFileDialog.getExistingDirectory(owner, title, start)
    if chosen:
        edit.setText(chosen)


# ==================================================================== AI 面板
class AiPanel(QDialog):
    """自动剪辑的操作台：选模式、指目录、看每个 mp4 卡在哪一步、开跑。

    表里的四个状态是各自独立查出来的，不靠猜，全部来自数据库：
    - 分析：analysis_runs 里有跑成功的记录
    - TXT：artifacts 里有 merged_txt
    - JSON：有 ai_results，或者 artifacts 里有 ai_script
    - 成品：artifacts 里有 final_video，或者 clips 里有 rendered
    新出现的文件是在打开面板 / 点刷新时登记进库的，不在这儿翻目录。
    """

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = parent
        self._job = str(cfg.bridge.get("ai_job") or "full")
        self._active_stem = ""
        self._active_step = ""
        # 状态全部来自数据库；句柄懒加载，开不起来就只记一句日志
        self._db_handle: Any = None
        self._db_failed = False

        self.setWindowTitle("AI 自动剪辑")
        self.setModal(False)
        self.setMinimumSize(820, 640)
        self.setSizeGripEnabled(True)
        # QDialog 默认只给个关闭按钮，这儿当第二主界面用，最小化/最大化都得有
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)

        outer = QVBoxLayout(self)
        outer.addLayout(self._build_header())
        outer.addWidget(self._build_modes())
        outer.addWidget(self._build_sources())


        outer.addWidget(self._build_dirs())
        outer.addWidget(self._build_stats())
        outer.addWidget(self._build_current())
        outer.addWidget(self._build_table(), 1)
        outer.addWidget(self._build_log())
        outer.addLayout(self._build_buttons())
        # 打开面板先登记一次（你手动丢进目录的 mp4/txt 也认），之后只查库
        self.refresh_tasks(sync=True)

    # ------------------------------------------------------------ 各块界面
    def _build_header(self) -> QHBoxLayout:
        """标题行：左边写清这是哪儿，右上角是跟主界面同步的连接状态药丸。"""
        title = QLabel("AI 自动剪辑")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        self.lbl_conn = QLabel("未启动")
        self.lbl_conn.setProperty("role", "pill")
        self.lbl_conn.setProperty("state", "off")
        self.lbl_conn.setAlignment(Qt.AlignCenter)
        self.lbl_conn.setMinimumWidth(
            self.lbl_conn.fontMetrics().horizontalAdvance(":65535 配对窗口 120s") + 32)
        self.lbl_conn.setToolTip("Bridge / AI 的连接状态，跟主界面那个药丸是同一个来源")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(title, 0, Qt.AlignVCenter)
        row.addStretch(1)
        row.addWidget(self.lbl_conn, 0, Qt.AlignVCenter)
        return row

    def set_connection(self, text: str, mood: str) -> None:
        """主界面刷状态时顺手推过来。改了 state 属性要 unpolish/polish 才换色。"""
        self.lbl_conn.setText(text)
        if self.lbl_conn.property("state") != mood:
            self.lbl_conn.setProperty("state", mood)
            self.lbl_conn.style().unpolish(self.lbl_conn)
            self.lbl_conn.style().polish(self.lbl_conn)

    def _build_modes(self) -> QWidget:

        box = QGroupBox("干哪一串")
        row = QHBoxLayout(box)
        self._job_group = QButtonGroup(self)
        self._job_group.setExclusive(True)
        for name, title, flow in JOBS:
            card = QPushButton(f"{title}\n\n{flow}")
            card.setCheckable(True)
            card.setMinimumHeight(120)
            card.setChecked(name == self._job)
            card.setToolTip({"full": "缺 <视频名>.txt 就先按主界面配置分析，再发 AI，"
                                     "回的 JSON 按主界面高光配置剪，成品落 AI_输出目录",
                             "collect": "同「剪辑成片」，但拿到 JSON 只存成 <视频名>_脚本.json，不渲染",
                             "script": "跳过 AI：直接用现成的 <视频名>_脚本.json 开剪"}[name])
            card.clicked.connect(lambda _=False, key=name: self._pick_job(key))
            self._job_group.addButton(card)
            row.addWidget(card, 1)
        return box

    def _build_sources(self) -> QWidget:
        """处理范围 + PRM：这一轮拿哪些视频、发 AI 时用哪一版提示词。

        这里只留跑批真正需要的三个选择，**JSON 的管理全部搬到「AI 剪辑资产中心」**：
        - 处理范围「只有已有 JSON 的视频」这一档一次 AI 都不调，直接拿库里的当前方案开剪；
        - 「只有没有 JSON 的视频」相反，只跑还没有高光资产的视频（判断用的是
          `assets.videos_with_assets`，和资产中心看到的一模一样，不是看 ai_results 有没有行）；
        - 高光 JSON 用哪一份 = 每个视频的**当前方案**，想换就去资产中心点「设为当前」。
        """
        box = QGroupBox("处理范围 / 提示词")
        row = QHBoxLayout(box)
        self.cmb_source = QComboBox()
        for label, key in (("全部视频（有 JSON 的直接剪，没有的问 AI）", "all"),
                           ("只有已有 JSON 的视频（不调 AI）", "existing"),
                           ("只有没有 JSON 的视频（问 AI）", "missing")):
            self.cmb_source.addItem(label, key)
        current = str(self.cfg.bridge.get("highlight_source") or "all")
        self.cmb_source.setCurrentIndex(max(0, self.cmb_source.findData(current)))
        self.cmb_source.setToolTip("「已有 JSON」这一档一次 AI 都不调，纯用库里的当前方案开剪")

        self.cmb_prm = QComboBox()
        self.cmb_prm.setMinimumWidth(200)
        self.cmb_prm.setToolTip("发给 AI 的提示词用哪一版；内容仍旧只在文件里，库里只记档案。"
                                "成品会记住用的是这一版")
        self._reload_prms()

        self.btn_assets = QPushButton("视频资产中心")
        self.btn_assets.setToolTip("视频 → 高光 JSON → PRM → 成品：搜索、看区间、看 AI 来源、"
                                   "编辑复制、按 JSON 直接出成品、成品反向追溯")
        self.btn_assets.clicked.connect(self.on_assets)

        row.addWidget(QLabel("处理范围"))
        row.addWidget(self.cmb_source, 1)
        row.addWidget(QLabel("PRM"))
        row.addWidget(self.cmb_prm, 1)
        row.addWidget(self.btn_assets)
        return box

    def _reload_prms(self) -> None:
        """把 PRM 档案填进下拉。库里还没有档案就只放一个「按配置」占位。"""
        keep = self.cmb_prm.currentData()
        self.cmb_prm.blockSignals(True)
        self.cmb_prm.clear()
        self.cmb_prm.addItem("按配置（默认 PRM）", 0)
        db = self._db()
        if db is not None:
            try:
                from ..db import assets as db_assets  # noqa: PLC0415

                for row in db_assets.list_prms(db):
                    mark = "（默认）" if int(row["is_default"] or 0) else ""
                    self.cmb_prm.addItem(f"{row['name']}{mark}", int(row["id"]))
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[PRM] 档案列不出来：{exc}")
        want = keep if keep else int(self.cfg.bridge.get("prm_id") or 0)
        self.cmb_prm.setCurrentIndex(max(0, self.cmb_prm.findData(want)))
        self.cmb_prm.blockSignals(False)

    def on_assets(self) -> None:
        """打开视频资产中心（视频 → 高光 JSON → PRM → 成品）。

        资产中心是**非模态**的独立窗口，而且全程只有一份：主窗口那个
        `MainWindow.on_asset_center()` 才是真正开窗的地方，这里只是把它叫起来，
        免得面板和主窗口各开一个、互相看不到对方的改动。
        """
        window = self._window
        opener = getattr(window, "on_asset_center", None)
        if callable(opener):
            opener()
            return
        # 没有主窗口（单独跑面板做测试）时，自己维持一份非模态窗口
        from .assets_dialog import AssetCenter  # noqa: PLC0415 - 只在点开时才建窗口

        center = getattr(self, "_asset_center", None)
        if center is None:
            center = AssetCenter(self.cfg, window or self, log=self._log)
            center.changed.connect(self.on_assets_changed)
            self._asset_center = center
        center.reload()
        center.show()
        center.raise_()
        center.activateWindow()

    def on_assets_changed(self) -> None:
        """资产中心里动过东西（PRM / JSON / 成品）就顺手把面板上的数字跟上。"""
        self._reload_prms()
        self.refresh_tasks()


    def _build_dirs(self) -> QWidget:
        bridge = self.cfg.bridge
        self.edit_input = DropDirEdit(str(bridge.get("ai_input_dir") or ""))
        self.edit_output = DropDirEdit(str(bridge.get("ai_output_dir") or ""))
        self.edit_input.setToolTip("要处理的视频、以及发给 AI 的 <视频名>.txt 都在这儿")
        self.edit_output.setToolTip("脚本 JSON 和高光成品落在这儿。留空＝用界面上选的导出目录")
        self.edit_input.editingFinished.connect(self.refresh_tasks)
        self.edit_output.editingFinished.connect(self.refresh_tasks)
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("AI_输入目录", _dir_row(self, self.edit_input, "选择 AI_输入目录"))
        form.addRow("AI_输出目录", _dir_row(self, self.edit_output, "选择 AI_输出目录"))
        return box

    def _build_stats(self) -> QWidget:
        """自动剪辑总览。九个数字互不重叠，每个视频只落进一个桶（见 repo.video_queue_statistics）。"""
        box = QGroupBox("自动剪辑任务总览")
        grid = QGridLayout(box)
        self._stat_labels: dict[str, QLabel] = {}
        boxes = (("total", "总视频"), ("json", "已获取 JSON"), ("no_json", "未获取 JSON"),
                 ("pending_render", "待剪辑"), ("waiting_ai", "等待 AI"), ("rendering", "剪辑中"),
                 ("done", "已完成"), ("failed", "失败"), ("cancelled", "已取消"))
        tips = {"total": "AI_输入目录里、文件还在盘上的视频",
                "json": "手上有一份能直接开剪的 AI JSON（解得开、抠得出片段）",
                "no_json": "还没有可用 AI JSON，也没在跑、也没成品",
                "pending_render": "AI JSON 已就位、没成品、当前没有在跑的任务",
                "waiting_ai": "任务在 uploading / waiting：正在提交给 AI，或者已提交在等回话",
                "rendering": "任务在 processing：JSON 已确认可用，正在渲染成品",
                "done": "有还在盘上的有效成品",
                "failed": "有 JSON 但任务记成 failed，且没有成品",
                "cancelled": "有 JSON 但任务被取消，且没有成品"}
        for index, (key, title) in enumerate(boxes):
            line, col = divmod(index, 3)   # 九格排成 3×3：总量 / 在跑 / 结局各占一行
            head = QLabel(title)
            head.setAlignment(Qt.AlignCenter)
            head.setToolTip(tips[key])
            value = QLabel("0")
            value.setAlignment(Qt.AlignCenter)
            value.setToolTip(tips[key])
            font = value.font()
            font.setPointSize(font.pointSize() + 4)
            font.setBold(True)
            value.setFont(font)
            grid.addWidget(head, line * 2, col)
            grid.addWidget(value, line * 2 + 1, col)
            self._stat_labels[key] = value
        return box

    def _build_current(self) -> QWidget:
        box = QWidget()
        row = QVBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self.lbl_current = QLabel("当前任务：闲着")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("%p%（0 / 0）")
        row.addWidget(self.lbl_current)
        row.addWidget(self.bar)
        return box

    def _build_table(self) -> QWidget:
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["文件", "分析", "TXT", "JSON", "剪辑", "状态"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        return self.table

    def _build_log(self) -> QWidget:
        self.view_log = QPlainTextEdit()
        self.view_log.setReadOnly(True)
        self.view_log.setMaximumBlockCount(500)
        self.view_log.setFixedHeight(120)
        self.view_log.setPlaceholderText("自动剪辑和 AI 对接的日志会出现在这儿")
        return self.view_log

    def _build_buttons(self) -> QHBoxLayout:
        self.btn_auto = QPushButton("▶ 自动剪辑")
        self.btn_auto.setToolTip("按选中的那一串，把 AI_输入目录里的视频挨个跑完；"
                                 "AI_输出目录里已经有同名成品的会跳过")
        self.btn_auto.clicked.connect(self.on_auto)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setToolTip("中止排着的队，并取消正在跑的 AI 任务")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setToolTip("登记目录里新出现的文件、跟磁盘对一次账，然后按数据库刷新")
        self.btn_refresh.clicked.connect(lambda: self.refresh_tasks(sync=True))
        self.btn_api = QPushButton("AI接口…")
        self.btn_api.setToolTip("找哪家 AI、走接口还是网页版扩展、key、模型、端口")
        self.btn_api.clicked.connect(self.on_api_options)
        self.btn_save = QPushButton("保存设置")
        self.btn_save.clicked.connect(lambda: self.save(close=False))
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        self.lbl_state = QLabel("闲着")
        self.lbl_state.setProperty("role", "pill")
        row = QHBoxLayout()
        row.addWidget(self.btn_auto)
        row.addWidget(self.btn_stop)
        row.addWidget(self.lbl_state, 1)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_api)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)
        return row

    # ------------------------------------------------------------ 状态查询
    def _in_dir(self) -> Path | None:
        text = self.edit_input.text().strip()
        return Path(text) if text and Path(text).is_dir() else None

    def _out_dir(self) -> Path | None:
        text = self.edit_output.text().strip()
        if text and Path(text).is_dir():
            return Path(text)
        getter = getattr(self._window, "export_root", None)
        if callable(getter):
            root = Path(getter())
            return root if root.is_dir() else None
        return None

    def _states(self, video: Path) -> dict[str, bool]:
        """一个 mp4 的四个状态。全部来自数据库，这里不碰磁盘。

        分析看 analysis_runs、TXT 看 artifacts.merged_txt、JSON 看 ai_results / artifacts.ai_script、
        成品看 artifacts.final_video 或 clips.rendered —— 四个各自独立，不互相推断。
        新文件是在刷新时的 refresh_from_disk() 里登记的，不是在这儿扫出来的。
        """
        empty = {"analysed": False, "txt": False, "json": False, "clipped": False}
        db = self._db()
        if db is None:
            return empty
        row = db_repo.find_video(db, video)
        if row is None:
            return empty
        return db_repo.video_state(db, int(row["id"]))

    def _row_marks(self, video: Path, states: dict[str, bool]) -> tuple[list[str], str]:
        """把四个状态翻成表里的记号和一句状态。正在跑的那一步用 ●。"""
        running = video.stem == self._active_stem
        marks = [DONE if states["analysed"] else WAITING,
                 DONE if states["txt"] else WAITING,
                 DONE if states["json"] else WAITING,
                 DONE if states["clipped"] else WAITING]
        if states["clipped"]:
            return marks, "完成"
        if running:
            step = self._active_step or "跑着"
            index = {"分析": 0, "导出": 1, "发送": 2, "剪辑": 3}.get(step)
            if index is not None:
                marks[index] = RUNNING
            return marks, {"分析": "分析中", "导出": "导出 TXT", "发送": "等 AI 回",
                           "剪辑": "剪辑中"}.get(step, "跑着")
        if states["json"]:
            return marks, "等剪辑"
        if states["txt"]:
            return marks, "等 AI"
        if states["analysed"]:
            return marks, "等导出 TXT"
        return marks, "未处理"

    # ------------------------------------------------------------ 对外接口
    def _db(self):
        """数据库句柄。开不起来就记一句，界面不崩（状态会显示为全空）。"""
        if self._db_handle is None and not self._db_failed:
            try:
                self._db_handle = open_db(self.cfg)
            except Exception as exc:
                self._db_failed = True
                self.append_log(f"[AI 面板] 数据库打不开，状态无法显示：{exc}")
        return self._db_handle

    def _sync_disk(self) -> None:
        """磁盘扫描只在这里发生：登记新出现的视频/TXT/JSON/成品，再跟磁盘对账。

        打开面板和点「刷新」时各来一次。之后所有状态判断都只查库。
        """
        db = self._db()
        if db is None:
            return
        folders = [p for p in (self._in_dir(),) if p is not None]
        try:
            refresh_from_disk(self.cfg, db, folders=folders or None, ai_out=self._out_dir())
        except Exception as exc:
            self.append_log(f"[AI 面板] 登记/对账失败：{exc}")

    def refresh_tasks(self, sync: bool = False) -> None:
        """刷新统计和任务表。状态一律查库。

        sync=True（打开面板、点刷新）时先登记新文件并对账；跑批过程中每一步只查库，
        40 个视频也就是几条 SQL，不会每个视频再去翻目录。
        """
        db = self._db()
        if db is None:
            self.table.setRowCount(0)
            return
        if sync:
            self._sync_disk()
        rows = db_repo.videos_under(db, self._in_dir())
        videos = [Path(r["file_path"]) for r in rows]
        ids = [int(r["id"]) for r in rows]
        states_by_id = db_repo.states_for_videos(db, ids)
        # 收取脚本这一串拿到 JSON 就算完事，其余两串要出成品才算
        done_key = "json" if self._job == "collect" else "clipped"
        stats = db_repo.video_queue_statistics(db, ids, mode=self._job, done_key=done_key)
        self.table.setRowCount(len(videos))
        for row, (video, vid) in enumerate(zip(videos, ids)):
            states = states_by_id.get(vid, {"analysed": False, "txt": False,
                                            "json": False, "clipped": False})
            marks, status = self._row_marks(video, states)
            cells = [video.name, *marks, status]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        for key, label in self._stat_labels.items():
            label.setText(str(stats[key]))

    def set_active(self, stem: str = "", step: str = "") -> None:
        """当前在处理哪个视频、走到哪一步（分析 / 导出 / 发送 / 剪辑）。"""
        self._active_stem = stem
        self._active_step = step
        if stem:
            self.lbl_current.setText(f"当前任务：{stem}　{step or ''}".rstrip())
        else:
            self.lbl_current.setText("当前任务：闲着")
        self.refresh_tasks()

    def set_queue_progress(self, done: int, total: int) -> None:
        pct = int(round(done / total * 100)) if total else 0
        self.bar.setValue(pct)
        self.bar.setFormat(f"%p%（{done} / {total}）")

    def append_log(self, line: str) -> None:
        """主界面把 AI 相关的日志转播过来，跑的时候不用切回去看。

        日志框可能还没建好（建界面时就有查库的活儿），那种情况转给主界面的日志。
        """
        view = getattr(self, "view_log", None)
        if view is None:
            if self._log:
                self._log(line)
            return
        view.appendPlainText(line)

    def set_running(self, running: bool, state: str = "") -> None:
        """自动剪辑开跑 / 收工时由主界面调，用来锁按钮和改状态字。"""
        self.btn_auto.setEnabled(not running)
        self.lbl_state.setText(state or ("跑着" if running else "闲着"))

    def set_standalone(self) -> None:
        """单独运行（run.py ai）时当正经主窗口用：任务栏有它，最小化/最大化/拉伸都全。"""
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint
                            | Qt.WindowCloseButtonHint)
        self.resize(960, 820)

    # ------------------------------------------------------------ 交互
    def _pick_job(self, key: str) -> None:
        self._job = key
        self.refresh_tasks()

    def on_auto(self) -> None:
        self.save(close=False)  # 先把眼前这套落盘，跑的就是你看到的
        run = getattr(self._window, "on_auto_clip", None)
        if callable(run):
            run()

    def on_stop(self) -> None:
        stop = getattr(self._window, "on_bridge_stop", None)
        if callable(stop):
            stop()

    def on_api_options(self) -> None:
        opener = getattr(self._window, "on_ai_api", None)
        if callable(opener):
            opener()

    # ------------------------------------------------------------ 保存
    def save(self, close: bool = True) -> None:
        """只写自己这几个键：干哪一串 + 两个 AI 专属目录 + 高光来源 + PRM。接口那些在「AI接口」里存。"""
        prm_id = int(self.cmb_prm.currentData() or 0)
        patch = {"bridge": {"ai_job": self._job,
                            "ai_input_dir": self.edit_input.text().strip(),
                            "ai_output_dir": self.edit_output.text().strip(),
                            "highlight_source": str(self.cmb_source.currentData() or "all"),
                            "prm_id": prm_id}}
        try:
            path = self.cfg.save_patch(patch)
        except OSError as exc:
            QMessageBox.warning(self, "AI 面板", f"写 config.json 失败：{exc}")
            return
        if self._log:
            titles = {name: title for name, title, _ in JOBS}
            sources = {"all": "全部", "existing": "只挑已有 JSON", "missing": "只挑没有 JSON"}
            self._log(f"[AI 面板] 已保存到 {path}：{titles.get(self._job, self._job)}；"
                      f"AI_输入目录 {patch['bridge']['ai_input_dir'] or '（留空）'}；"
                      f"AI_输出目录 {patch['bridge']['ai_output_dir'] or '（留空，用导出目录）'}；"
                      f"高光来源 {sources.get(patch['bridge']['highlight_source'], '全部')}；"
                      f"PRM {self.cmb_prm.currentText() if prm_id else '按配置'}")
        self.refresh_tasks()
        if close:
            self.accept()


# ================================================================ AI 接口设置
class AiApiDialog(QDialog):
    """AI 接口设置：找哪家 AI、走接口还是网页版扩展、key、模型、超时、端口、上传方式。

    点「保存」才落盘，只写 bridge 里跟接口有关的那些键，不碰 AI 面板的目录和模式。
    只有端口要重开 GUI 才换得过去，因为 Bridge 服务在启动时就绑好了。
    """

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self.setWindowTitle("AI 接口")
        self.setMinimumWidth(560)
        bridge = cfg.bridge

        self.cmb_provider = QComboBox()
        for name, spec in providers.PROVIDERS.items():
            self.cmb_provider.addItem(spec["label"], name)
        self._provider = providers.normalize(bridge.get("provider"))
        self.cmb_provider.setCurrentIndex(max(0, self.cmb_provider.findData(self._provider)))
        # 每家的 key / 模型各存一份，切来切去不会互相覆盖，保存时一起落盘
        self._draft = {name: {"api_key": str(providers.node(bridge, name).get("api_key") or ""),
                              "api_model": str(providers.settings(bridge, name)["api_model"])}
                       for name in providers.PROVIDERS}

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("接口直连（不开浏览器，要 API key）", "api")
        self.cmb_mode.addItem("网页版扩展（用浏览器里的对话页）", "extension")
        self.cmb_mode.setCurrentIndex(max(0, self.cmb_mode.findData(str(bridge.get("mode") or "api"))))

        self.edit_key = QLineEdit()
        self.edit_key.setEchoMode(QLineEdit.Password)

        self.cmb_model = QComboBox()
        self.cmb_model.setEditable(True)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(30, 3600)
        self.spin_timeout.setSuffix(" 秒")
        self.spin_timeout.setValue(int(float(bridge.get("api_timeout") or 300)))

        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(int(bridge.get("port") or 5998))
        self.spin_port.setToolTip("Bridge 监听端口，扩展选项页要填同一个；改完要重开 GUI")

        self.cmb_upload = QComboBox()
        self.cmb_upload.addItem("自动拖文件", "auto")
        self.cmb_upload.addItem("我自己选文件", "manual")
        self.cmb_upload.addItem("只看我操作（扩展不动手）", "observe")
        self.cmb_upload.setToolTip("只看我操作：页面打开后扩展一个键都不点，只把你碰过的"
                                   "元素记进日志，用来查它该点哪里")
        self.cmb_upload.setCurrentIndex(
            max(0, self.cmb_upload.findData(str(bridge.get("upload_mode") or "auto"))))

        self.chk_side = QCheckBox("对话页放到不抢焦点的小窗口")
        self.chk_side.setChecked(bool(bridge.get("side_window", True)))
        self.chk_side.setToolTip("后台标签页会被浏览器冻结，什么都干不了；独立窗口照常渲染")
        self.chk_focus = QCheckBox("允许浏览器跳到前台")
        self.chk_focus.setChecked(bool(bridge.get("focus_browser", False)))
        self.chk_clip = QCheckBox("拿到 JSON 就自动开剪")
        self.chk_clip.setChecked(bool(bridge.get("auto_clip", True)))

        self.hint = QLabel()
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("找哪家 AI", self.cmb_provider)
        form.addRow("走哪条路", self.cmb_mode)
        form.addRow("API key", self.edit_key)
        form.addRow("模型", self.cmb_model)
        form.addRow("超时", self.spin_timeout)
        form.addRow("Bridge 端口", self.spin_port)
        form.addRow("扩展上传方式", self.cmb_upload)
        form.addRow(self.chk_side)
        form.addRow(self.chk_focus)
        form.addRow(self.chk_clip)
        form.addRow(self.hint)
        form.addRow(buttons)
        self.cmb_mode.currentIndexChanged.connect(self.sync_enabled)
        self.cmb_provider.currentIndexChanged.connect(self.on_provider_changed)
        self.load_provider(self._provider)
        self.sync_enabled()

    # --------------------------------------------------------- 提供方切换
    def on_provider_changed(self) -> None:
        """换提供方：先把当前这家的 key / 模型收进草稿，再摊开新那家的。"""
        self.stash_provider()
        self._provider = providers.normalize(self.cmb_provider.currentData())
        self.load_provider(self._provider)
        self.sync_enabled()

    def stash_provider(self) -> None:
        draft = self._draft.setdefault(self._provider, {})
        draft["api_key"] = self.edit_key.text().strip()
        draft["api_model"] = self.cmb_model.currentText().strip()

    def load_provider(self, name: str) -> None:
        spec = providers.PROVIDERS[name]
        draft = self._draft.get(name, {})
        self.edit_key.setText(str(draft.get("api_key") or ""))
        self.edit_key.setPlaceholderText(f"留空则读环境变量 {spec['key_env']}")
        self.edit_key.setToolTip(f"去 {spec['key_page']} 领；不想写进仓库就设环境变量")
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.addItems(spec["models"])
        self.cmb_model.setCurrentText(str(draft.get("api_model") or spec["models"][0]))
        self.cmb_model.blockSignals(False)
        self.hint.setText(
            f"接口直连纯后台跑，失败原因明确（{spec['label']} key 去 {spec['key_page']} 领）；"
            f"网页版扩展要开着浏览器上 {spec['ai_url']}，而且窗口被完全盖住时页面会被冻结。")

    def sync_enabled(self) -> None:
        """按选的路子灰掉用不上的项，免得改了半天不生效还以为坏了。"""
        api = self.cmb_mode.currentData() == "api"
        for w in (self.edit_key, self.cmb_model, self.spin_timeout):
            w.setEnabled(api)
        for w in (self.cmb_upload, self.chk_side, self.chk_focus):
            w.setEnabled(not api)

    # ------------------------------------------------------------- 保存
    def save(self) -> None:
        old_port = int(self.cfg.bridge.get("port") or 5998)
        self.stash_provider()
        bridge: dict[str, Any] = {
            "mode": self.cmb_mode.currentData(),
            "provider": self._provider,
            "api_timeout": int(self.spin_timeout.value()),
            "port": int(self.spin_port.value()),
            "upload_mode": self.cmb_upload.currentData(),
            "side_window": self.chk_side.isChecked(),
            "focus_browser": self.chk_focus.isChecked(),
            "auto_clip": self.chk_clip.isChecked(),
        }
        # 每家的 key / 模型写回各自那一节（Gemini 是 bridge 下的老键，DeepSeek 在 bridge.deepseek）
        for name, draft in self._draft.items():
            section = providers.section_for(name)
            if section:
                bridge.setdefault(section, {}).update(draft)
            else:
                bridge.update(draft)
        try:
            path = self.cfg.save_patch({"bridge": bridge})
        except OSError as exc:
            QMessageBox.warning(self, "AI 接口", f"写 config.json 失败：{exc}")
            return
        if self._log:
            mode = "接口直连" if bridge["mode"] == "api" else "网页版扩展"
            spec = providers.PROVIDERS[self._provider]
            self._log(f"[AI 接口] 已保存到 {path}：{mode}，{spec['label']} "
                      f"{self._draft[self._provider]['api_model']}，端口 {bridge['port']}")
        if int(self.spin_port.value()) != old_port:
            QMessageBox.information(self, "AI 接口",
                                    "端口改了，要重开 GUI 才会换过去；"
                                    "扩展选项页里的地址也要跟着改。")
        self.accept()
