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
import time
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
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

# 三种串各自的开关键。`ai_job` 仍旧是状态机唯一认的那个值，这三个是它的布尔映射：
# 配置文件里一眼能看出干的是哪一串，老配置只有 ai_job 也照旧能读（见 _job_from_config）
JOB_FLAGS = {"full": "ai_clip_video", "collect": "ai_collect_script", "script": "ai_script_clip"}

# 「处理范围」一个下拉说完全部：**干哪一串 + 跑哪些视频**。
# 以前是「三张模式卡片 × 三档范围」，九种组合里一半是矛盾的（比如「脚本剪辑 + 只跑没有
# JSON 的视频」＝每一条都注定失败）。这里只留下有意义的四档，每一档直接对应
# (ai_job, highlight_source) 一对值——底下的状态机一个字都没改。
SCOPES = (
    ("clip_all", "全部视频：有 JSON 的直接剪，没有的问 AI 再剪", "full", "all",
     "库里没有分析结果就先按主界面配置分析，再把 PRM + 完整剧本（只有两份 txt，"
     "绝不上传视频）发给 AI；回的高光 JSON 入库后按主界面高光配置剪，"
     "高光片段落 AI_输出目录。已经有 JSON 的视频直接拿库里的当前方案开剪，不问 AI"),
    ("clip_missing", "只跑还没有 JSON 的视频：问 AI 拿 JSON 再剪", "full", "missing",
     "只挑还没有高光 JSON 的视频（判断用 assets.videos_with_assets，和资产中心看到的"
     "一模一样）：分析 → 发 AI → JSON 入库 → 剪。已经有 JSON 的这一轮不动"),
    ("clip_existing", "只跑已有 JSON 的视频：直接剪，一次 AI 都不调", "script", "existing",
     "一次 AI 都不调：直接用每个视频库里的**当前方案**开剪。想换用哪一份 JSON，"
     "去资产中心点「设为当前」"),
    ("collect_missing", "只收 JSON 不剪辑：问 AI 拿 JSON 入库就算完", "collect", "missing",
     "同「问 AI 拿 JSON 再剪」，但高光 JSON 入库就算这一条干完，不剪辑。"
     "只跑还没有 JSON 的视频——已经有 JSON 的再问一次 AI 是白花钱"),
)

SCOPE_JOB = {key: job for key, _label, job, _source, _tip in SCOPES}
SCOPE_SOURCE = {key: source for key, _label, _job, source, _tip in SCOPES}


def _job_from_config(bridge: dict[str, Any]) -> str:
    """配置里存的是哪一种模式。ai_job 优先，没有就看三个布尔开关，都没有＝剪辑成片。"""
    job = str(bridge.get("ai_job") or "").strip()
    if job in JOB_FLAGS:
        return job
    for name, key in JOB_FLAGS.items():
        if bridge.get(key):
            return name
    return "full"


def _scope_from_config(bridge: dict[str, Any]) -> str:
    """老配置（ai_job + highlight_source 两个键）反推成「处理范围」这一档。

    对不上任何一档就按 job 归到最接近的那一档，绝不因为配置怪就把界面搞空。
    """
    job = _job_from_config(bridge)
    source = str(bridge.get("highlight_source") or "all").strip()
    for key, _label, want_job, want_source, _tip in SCOPES:
        if job == want_job and source == want_source:
            return key
    fallback = {"full": "clip_all", "script": "clip_existing", "collect": "collect_missing"}
    return fallback.get(job, "clip_all")



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


def dir_row(owner, edit: DropDirEdit, title: str) -> QWidget:
    """一行：路径框 + 选择目录 + 打开。返回装好的容器控件。

    AI 面板和视频资产中心都用它摆自己的目录行（两边的键各写各的，互不相干）。
    """
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

    表里那六步各自独立查出来，不靠猜，**全部来自数据库**：
    - 分析：analysis_runs 里有跑成功的记录
    - 剧本：库里的分析结果够生成完整剧本（script_ready_videos）
    - 高光分析：这个视频问过 AI（highlight_attempted_videos）
    - 高光 JSON：库里那份解得开、抠得出片段（reusable_json_videos）
    - 剪辑：clips 里有剪出来过的片段
    - 成品：artifacts 里有还在盘上的高光片段 mp4
    盘上有没有 TXT / JSON 文件**不是业务状态**，只在刷新时登记进库供显示与兼容。
    """

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = parent
        self._scope = _scope_from_config(cfg.bridge)
        self._job = SCOPE_JOB[self._scope]
        self._active_stem = ""
        self._active_step = ""
        # 改哪儿存哪儿：没有「保存配置」这一步。落盘做了防抖（拖目录、连点模式都只写一次），
        # _ready 在界面搭完之前拦住保存，免得建控件的过程中就往 config.json 写半份
        self._ready = False
        self._saved_patch: dict[str, Any] | None = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(350)
        self._save_timer.timeout.connect(self._flush_settings)
        # 状态全部来自数据库；句柄懒加载，开不起来就只记一句日志
        self._db_handle: Any = None
        self._db_failed = False

        self.setWindowTitle("AI 自动剪辑")
        self.setModal(False)
        # 字号整体压一档：这一屏塞了统计 + 任务表 + 日志，字大了小屏幕根本放不下
        font = self.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        self.setFont(font)
        self.setMinimumSize(760, 560)
        self.setSizeGripEnabled(True)
        # QDialog 默认只给个关闭按钮，这儿当第二主界面用，最小化/最大化都得有
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)

        outer = QVBoxLayout(self)
        outer.addLayout(self._build_header())
        outer.addWidget(self._build_sources())


        outer.addWidget(self._build_dirs())
        outer.addWidget(self._build_stats())
        outer.addWidget(self._build_current())
        outer.addWidget(self._build_table(), 1)
        outer.addWidget(self._build_log())
        outer.addLayout(self._build_buttons())
        # 打开面板先登记一次（你手动丢进目录的 mp4/txt 也认），之后只查库
        self._ready = True
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

    def _build_sources(self) -> QWidget:
        """处理范围 + PRM：这一轮**干哪一串、跑哪些视频**，发 AI 时带哪几份提示词。

        「干哪一串」那三张卡片（剪辑成片 / 收取脚本 / 脚本剪辑）已经并进这个下拉：
        每一档就是原来一张卡片 + 一个范围的组合，矛盾组合（比如「脚本剪辑 + 只跑没有
        JSON 的视频」）直接不给选。底下的状态机照旧只认 `ai_job` + `highlight_source`。

        其余口径不变，**JSON 的管理全部在「AI 剪辑资产中心」**：
        - 「只跑还没有 JSON 的视频」判断用 `assets.videos_with_assets`，和资产中心一致，
          不是看 ai_results 有没有行；
        - 高光 JSON 用哪一份 = 每个视频的**当前方案**，想换就去资产中心点「设为当前」；
        - PRM **不在这里选**：按 PRM 的使用状况发——「使用中」的每一份都发，
          停用的一份都不发，一份都没启用就这一轮不发 AI。启用/停用在资产中心的 PRM 管理页。
        """
        box = QGroupBox("处理范围 / 提示词")
        row = QHBoxLayout(box)
        self.cmb_source = QComboBox()
        self.cmb_source.setMinimumWidth(320)
        for key, label, _job, _source, tip in SCOPES:
            self.cmb_source.addItem(label, key)
            self.cmb_source.setItemData(self.cmb_source.count() - 1, tip, Qt.ToolTipRole)
        self.cmb_source.setCurrentIndex(max(0, self.cmb_source.findData(self._scope)))
        self.cmb_source.setToolTip("这一个下拉说完「干哪一串 + 跑哪些视频」："
                                   "「直接剪」那两档一次 AI 都不调，"
                                   "「收 JSON」那档拿到 JSON 就算完、不剪辑。选完即存")
        self.cmb_source.currentIndexChanged.connect(lambda _=0: self._pick_scope())

        # 「不跑成品」：成品库里已经有这个视频的有效成品就整条跳过（默认勾上）。
        # 取消勾选＝已有成品也重跑一遍，会再出一份新成品。勾完即存
        self.chk_skip_done = QCheckBox("不跑成品")
        self.chk_skip_done.setChecked(bool(self.cfg.bridge.get("skip_done_products", True)))
        self.chk_skip_done.setToolTip("勾上（默认）：成品库里已经有这个视频的成品就直接跳过，"
                                      "不重新分析、不重新问 AI、不重新剪。"
                                      "取消勾选＝已有成品也照样重跑一遍。勾完即存")
        self.chk_skip_done.stateChanged.connect(lambda _=0: self._pick_skip_done())

        # PRM 不再在这里挑一份：发哪几份完全看使用状况，这里只显示现在会发什么
        self.lbl_prm = QLabel("—")
        self.lbl_prm.setWordWrap(True)
        self.lbl_prm.setToolTip("发给 AI 的提示词按 PRM 的使用状况来：「使用中」的每一份都会"
                                "当附件带上，停用的一份都不发；一份都没启用就这一轮不发 AI。"
                                "启用 / 停用在资产中心的 PRM 管理页")
        self._reload_prms()

        self.btn_assets = QPushButton("视频资产中心")
        self.btn_assets.setToolTip("视频 → 高光 JSON → PRM → 成品：搜索、看区间、看 AI 来源、"
                                   "编辑复制、按 JSON 直接出成品、成品反向追溯")
        self.btn_assets.clicked.connect(self.on_assets)

        row.addWidget(QLabel("处理范围"))
        row.addWidget(self.cmb_source, 1)
        row.addWidget(self.chk_skip_done)
        row.addWidget(QLabel("PRM"))
        row.addWidget(self.lbl_prm, 1)
        row.addWidget(self.btn_assets)
        return box

    def _reload_prms(self) -> None:
        """刷新「现在会发哪几份 PRM」这一行（只读，选哪几份靠 PRM 管理页的启用状态）。"""
        db = self._db()
        rows: list = []
        if db is not None:
            try:
                from ..db import assets as db_assets  # noqa: PLC0415

                rows = list(db_assets.enabled_prms(db))
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[PRM] 档案列不出来：{exc}")
        if rows:
            names = "、".join(str(row["name"]) for row in rows)
            self.lbl_prm.setText(f"使用中 {len(rows)} 份：{names}")
        else:
            self.lbl_prm.setText("一份都没启用 → 不发 AI（去 PRM 管理页启用）")

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
        self.edit_input.setToolTip("要处理的视频、以及发给 AI 的完整剧本 <视频名>.txt 都在这儿。"
                                   "改完即存，不用再点保存")
        self.edit_output.setToolTip("高光 JSON 和高光成品落在这儿。留空＝用界面上选的导出目录。"
                                    "改完即存，不用再点保存")
        # 敲字、拖文件夹进来、点「浏览…」都算改动：防抖之后落盘 + 按新目录刷新
        self.edit_input.textChanged.connect(self._settings_touched)
        self.edit_output.textChanged.connect(self._settings_touched)
        self.edit_input.editingFinished.connect(self._flush_settings)
        self.edit_output.editingFinished.connect(self._flush_settings)
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("AI_输入目录", dir_row(self, self.edit_input, "选择 AI_输入目录"))
        form.addRow("AI_输出目录", dir_row(self, self.edit_output, "选择 AI_输出目录"))
        return box

    def _build_stats(self) -> QWidget:
        """任务统计：只留七个头号数字（业务链上每一步做到哪儿了）。

        原来底下还有一排九格（总视频 / 待剪辑 / 已完成 / 已获取 JSON / 等待 AI / 失败 /
        未获取 JSON / 剪辑中 / 已取消）——和头号数字讲的是同一件事，还多占半屏，撤了。
        `repo.video_queue_statistics` 照旧算，只是界面不再摆那一排。
        """
        box = QGroupBox("自动剪辑任务总览")
        outer = QVBoxLayout(box)
        outer.addLayout(self._build_headline())
        return box

    def _build_headline(self) -> QGridLayout:
        """七个头号数字：业务链上每一步各一个，全部查库，跟着任务进度自己刷。"""
        grid = QGridLayout()
        self._head_labels: dict[str, QLabel] = {}
        heads = (("total", "总任务", "AI_输入目录里、文件还在盘上的视频总数"),
                 ("analysed", "已分析", "analysis_runs 里有跑成功的本地分析"),
                 ("script", "已有剧本", "库里的分析结果够生成完整剧本（不看盘上有没有 TXT）"),
                 ("attempted", "已分析高光", "已经把 PRM + 完整剧本交给 AI 做过高光分析"),
                 ("json", "已获取 JSON", "库里有一份能直接开剪的高光 JSON"),
                 ("rendered", "已剪辑", "clips 里有剪出来过的高光片段"),
                 ("made", "成品", "artifacts 里有还在盘上的高光片段 mp4"))
        for col, (key, title, tip) in enumerate(heads):
            head = QLabel(title)
            head.setAlignment(Qt.AlignCenter)
            head.setToolTip(tip)
            value = QLabel("0")
            value.setAlignment(Qt.AlignCenter)
            value.setToolTip(tip)
            value.setStyleSheet("font-size: 14px; font-weight: 600;")
            grid.addWidget(head, 0, col)
            grid.addWidget(value, 1, col)
            self._head_labels[key] = value
        return grid

    def _build_current(self) -> QWidget:
        """当前任务那一块：一行说明 + 两条进度条。

        上面那条是**这一个视频**自己的进度（分析到哪一步、渲染到第几帧），
        下面那条才是整个队列跑到第几个。两条分开看，才知道是"卡住了"还是"这条本来就慢"。
        """
        box = QWidget()
        row = QVBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.lbl_current = QLabel("当前任务：闲着")
        self.bar_video = QProgressBar()
        self.bar_video.setRange(0, 100)
        self.bar_video.setValue(0)
        self.bar_video.setFormat("单条视频 %p%")
        self.bar_video.setToolTip("这一个视频自己的进度：本地分析的各阶段、以及渲染的帧数")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("队列 %p%（0 / 0）")
        self.bar.setToolTip("整个队列跑到第几个视频")
        row.addWidget(self.lbl_current)
        row.addWidget(self.bar_video)
        row.addWidget(self.bar)
        return box

    def _build_table(self) -> QWidget:
        """任务表：文件 + 业务链上的六步 + 一句人话。每一列都只来自数据库。

        右边挂整批的操作按钮：现在只有「清空非中英视频」。
        """
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["文件", "分析", "剧本", "高光分析", "高光 JSON", "剪辑", "成品", "状态"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 8):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        self.btn_clear_video = QPushButton("清空非中英视频")
        self.btn_clear_video.setToolTip(
            "只清语言不是英文 / 中文的视频（语言预检拦下来的那些，"
            "`videos.blocked_language` 有值）——「跳过」的原因很多，这里一概不碰。"
            "清的是：原视频文件 + 它自己产出的附带文件（合并 TXT / 剧本 / 高光 JSON / "
            "片段 / 成品）+ 库里的全部记录。没有非中英视频时这个按钮是灰的。**不可恢复**")
        self.btn_clear_video.clicked.connect(self.on_clear_video)
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.btn_clear_video)
        side.addStretch(1)

        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.table, 1)
        row.addLayout(side)
        self._sync_row_buttons()
        return box

    def _foreign_videos(self) -> list[tuple[int, Path, str]]:
        """AI_输入目录里语言不是英文 / 中文的视频：(id, 路径, 语言码)。查库，不翻目录。

        判据只有一个：语言预检把它拦下来时写进 `videos.blocked_language` 的语言码。
        状态显示「跳过」的原因有很多（已有成品、AI 取消……），那些不算。
        """
        db = self._db()
        if db is None:
            return []
        try:
            rows = db_repo.blocked_language_videos(db, self._in_dir())
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[AI 面板] 查不出非中英视频：{exc}")
            return []
        return [(int(r["id"]), Path(r["file_path"]), str(r["blocked_language"] or ""))
                for r in rows]

    def _sync_row_buttons(self) -> None:
        """「清空非中英视频」只在真有非中英视频时才亮（没有就没什么可清的）。"""
        button = getattr(self, "btn_clear_video", None)
        if button is not None:
            button.setEnabled(bool(self._foreign_videos()))

    def _attached_files(self, db, video_id: int) -> list[Path]:
        """这个视频自己产出的附带文件：登记在 artifacts 里的合并 TXT / 剧本 / JSON /
        片段 / 成品。查库拿路径，不去目录里猜。"""
        try:
            rows = db_repo.get_artifacts(db, video_id)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[AI 面板] 查不出附带文件（这次只删原视频）：{exc}")
            return []
        return [Path(r["path"]) for r in rows if r["path"]]

    def on_clear_video(self) -> None:
        """整批清掉非中英视频：原视频 + 它的附带文件 + 库里的全部记录。不可恢复。

        语言不是英文 / 中文的视频对这条产线没用，留着每次都要重新扫、重新判一遍。
        清完连登记都没了，自动剪辑扫不到，也就谈不上再跑。
        """
        foreign = self._foreign_videos()
        if not foreign:
            QMessageBox.information(self, "AI 面板",
                                    "AI_输入目录里没有非中英视频（语言预检没拦下谁），"
                                    "没什么要清的")
            return
        running = getattr(self._window, "auto_running", None)
        if callable(running) and running():
            QMessageBox.information(self, "AI 面板",
                                    "自动剪辑正在跑，先点「停止」再清非中英视频"
                                    "（正在读的文件删不掉，也容易把手上这条搞乱）")
            return
        db = self._db()
        if db is None:
            QMessageBox.warning(self, "AI 面板", "数据库打不开，什么都没动")
            return
        listed = "\n".join(f"· {video.name}（语言 {code or '未知'}）"
                           for _vid, video, code in foreign[:12])
        if len(foreign) > 12:
            listed += f"\n· …另外还有 {len(foreign) - 12} 个"
        ask = QMessageBox.question(
            self, "清空非中英视频",
            f"要清掉这 {len(foreign)} 个非中英视频吗？\n\n"
            f"{listed}\n\n"
            "一起没掉的还有：它们自己产出的附带文件（合并 TXT / 剧本 / 高光 JSON / "
            "片段 / 成品），以及库里的分析 / AI 任务 / 高光 JSON / 片段 / 文件登记。\n"
            "别的视频（语言是英文 / 中文的）一个都不动。\n"
            "文件删了不进回收站，找不回来。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ask != QMessageBox.Yes:
            return
        removed = kept = files = 0
        for vid, video, code in foreign:
            attached = self._attached_files(db, vid)
            if video.is_file():
                try:
                    os.remove(video)
                except OSError as exc:
                    kept += 1
                    self.append_log(f"[AI 面板] {video.name} 删不掉，"
                                    f"它的附带文件和库里数据也一并留着：{exc}")
                    continue
            for path in attached:
                if not path.is_file():
                    continue
                try:
                    os.remove(path)
                    files += 1
                except OSError as exc:
                    self.append_log(f"[AI 面板] 附带文件删不掉：{path}（{exc}）")
            try:
                db_repo.forget_video(db, vid)
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[AI 面板] {video.name} 库里的数据没清干净：{exc}")
            removed += 1
            self.append_log(f"[AI 面板] 已清空非中英视频 {video.name}（语言 {code or '未知'}）")
        self.append_log(f"[AI 面板] 清空非中英视频完成：清掉 {removed} 个视频、"
                        f"{files} 个附带文件"
                        + (f"，{kept} 个删不掉（原样留着）" if kept else ""))
        self.refresh_tasks()

    def _build_log(self) -> QWidget:
        """日志：子进程、渲染、AI 对接的每一行都往这儿贴，出问题不用切回主界面翻。"""
        self.view_log = QPlainTextEdit()
        self.view_log.setReadOnly(True)
        self.view_log.setMaximumBlockCount(4000)
        self.view_log.setMinimumHeight(150)
        self.view_log.setPlaceholderText("自动剪辑、本地分析子进程、剪辑渲染和 AI 对接的日志都在这儿")
        return self.view_log

    def _build_buttons(self) -> QHBoxLayout:
        self.btn_auto = QPushButton("▶ 自动剪辑")
        self.btn_auto.setToolTip("按选中的那一串，把 AI_输入目录里的视频挨个跑完；"
                                 "库里已经登记了有效成品的会跳过")
        self.btn_auto.clicked.connect(self.on_auto)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setToolTip("不再领新任务，手上这条走完当前这一步就退回等待，"
                                 "下次点「自动剪辑」接着跑；同时取消正在等的 AI 请求")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setToolTip("登记目录里新出现的文件、跟磁盘对一次账，然后按数据库刷新")
        self.btn_refresh.clicked.connect(lambda: self.refresh_tasks(sync=True))
        self.btn_api = QPushButton("AI接口…")
        self.btn_api.setToolTip("找哪家 AI、走接口还是网页版扩展、key、模型、端口")
        self.btn_api.clicked.connect(self.on_api_options)
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

    def _task_states(self, db, ids: list[int]) -> dict[int, str]:
        """每个视频当前那条任务走到哪儿了。五条聚合 SQL，不按视频逐条查。

        一个视频可能有好几条历史任务，取"走得最远"的那一个：
        渲染中 > 上传中 > 等 AI 回 > 失败 > 已取消。返回空串＝这一档没有任务记录。
        """
        if not ids:
            return {}
        buckets = (("processing", ("processing",)), ("uploading", ("uploading",)),
                   ("waiting", ("waiting",)), ("failed", ("failed",)),
                   ("cancelled", ("cancelled",)))
        out: dict[int, str] = {}
        for name, states in buckets:
            for vid in db_repo.task_videos(db, ids, states, mode=self._job):
                out.setdefault(int(vid), name)
        return out

    def _row_marks(self, video: Path, states: dict[str, bool],
                   task: str = "") -> tuple[list[str], str]:
        """把库里的状态翻成表里的六个记号和一句人话。跑着的那一步 ●，砸了的那一步 ✕。

        六步就是业务链本身，每一步各自独立判定，**全部来自数据库**：
        分析（analysis_runs）、剧本（库里能不能生成完整剧本）、高光分析（问过 AI 没有）、
        高光 JSON（库里那份能不能直接开剪）、剪辑（clips 里剪过）、成品（final_video 还在盘上）。
        `task` 是这条视频当前任务的状态，只用来把"在等谁"说清楚，不会反过来推翻产物。
        文件在不在盘上（states 里的 txt / json）绝不参与这里的判断。
        """
        running = video.stem == self._active_stem
        marks = [DONE if states.get("analysed") else WAITING,
                 DONE if states.get("script") else WAITING,
                 DONE if states.get("attempted") else WAITING,
                 DONE if states.get("json_ok") else WAITING,
                 DONE if states.get("rendered") else WAITING,
                 DONE if states.get("clipped") else WAITING]
        if states.get("clipped"):
            return marks, "成品完成"
        if running:
            step = self._active_step or "跑着"
            index = {"分析": 0, "导出": 1, "发送": 2, "剪辑": 4}.get(step)
            if index is not None:
                marks[index] = RUNNING
            return marks, {"分析": "分析中", "导出": "生成剧本", "发送": "上传中",
                           "剪辑": "剪辑中"}.get(step, "跑着")
        if task == "processing":
            marks[4] = RUNNING
            return marks, "剪辑中"
        if task == "uploading":
            marks[2] = RUNNING
            return marks, "上传中"
        if task == "waiting":
            marks[2] = RUNNING
            return marks, "等待高光 JSON"
        if task == "failed":
            if not states.get("analysed"):
                marks[0] = FAILED
                return marks, "分析失败"
            if not states.get("json_ok"):
                marks[3] = FAILED
                return marks, "高光 JSON 失败"
            marks[5] = FAILED
            return marks, "剪辑失败"
        if task == "cancelled" and not states.get("json_ok"):
            return marks, "跳过"
        if states.get("json_ok"):
            return marks, "高光 JSON 就绪"
        if states.get("attempted"):
            return marks, "等待高光 JSON"
        if states.get("script"):
            return marks, "等待高光分析"
        if states.get("analysed"):
            return marks, "分析没出内容"
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
        40 个视频也就是几条 SQL，不会每个视频再去翻目录。每一步的进度回调都会叫到这儿，
        所以四个头号数字是跟着任务自己动的，不用手点刷新。
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
        made = len(db_repo.artifact_videos(db, ids, "final_video"))
        tasks = self._task_states(db, ids)
        self.table.setRowCount(len(videos))
        for row, (video, vid) in enumerate(zip(videos, ids)):
            states = states_by_id.get(vid, {})
            marks, status = self._row_marks(video, states, tasks.get(vid, ""))
            cells = [video.name, *marks, status]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setToolTip(str(video))
                self.table.setItem(row, col, item)
        # 头号七格＝业务链每一步做到哪儿了，全部查库；成品一律看 final_video（跟模式无关）
        head = {"total": stats["total"], "analysed": stats["analysed"],
                "script": stats["script"], "attempted": stats["attempted"],
                "json": stats["json"], "rendered": stats["rendered"], "made": made}
        for key, label in self._head_labels.items():
            label.setText(str(head[key]))
        self._sync_row_buttons()

    def set_active(self, stem: str = "", step: str = "") -> None:
        """当前在处理哪个视频、走到哪一步（分析 / 导出 / 发送 / 剪辑）。"""
        switched = stem != self._active_stem or step != self._active_step
        self._active_stem = stem
        self._active_step = step
        if stem:
            self.lbl_current.setText(f"当前任务：{stem}　{step or ''}".rstrip())
        else:
            self.lbl_current.setText("当前任务：闲着")
        if switched:
            # 换视频、或者同一个视频进了下一步（分析 → 导出 → 发送 → 剪辑）：这条从头走
            self.set_video_progress(0.0, step or "")
        self.refresh_tasks()

    def set_video_progress(self, ratio: float, text: str = "") -> None:
        """单条视频自己的进度（0~1）。text 是这一刻在干什么，直接写在条上。"""
        pct = int(round(min(1.0, max(0.0, float(ratio))) * 100))
        self.bar_video.setValue(pct)
        self.bar_video.setFormat(f"单条视频 %p%　{text}" if text else "单条视频 %p%")

    def set_queue_progress(self, done: int, total: int) -> None:
        pct = int(round(done / total * 100)) if total else 0
        self.bar.setValue(pct)
        self.bar.setFormat(f"队列 %p%（{done} / {total}）")
        self.refresh_tasks()   # 一条任务落定就把统计跟上，不用等人点刷新

    def append_log(self, line: str) -> None:
        """主界面把日志转播过来，跑的时候不用切回去看。

        每行带上时分秒：批量跑几十条的时候，"卡在哪一步、卡了多久"全靠这个时间戳。
        日志框可能还没建好（建界面时就有查库的活儿），那种情况转给主界面的日志。
        """
        view = getattr(self, "view_log", None)
        if view is None:
            if self._log:
                self._log(line)
            return
        view.appendPlainText(f"{time.strftime('%H:%M:%S')} {line}")

    def set_running(self, running: bool, state: str = "") -> None:
        """自动剪辑开跑 / 收工时由主界面调，用来锁按钮和改状态字。

        收工那一下顺手跟磁盘对一次账：最后一个成品可能刚落地，数字要立刻对得上。
        """
        self.btn_auto.setEnabled(not running)
        self.lbl_state.setText(state or ("跑着" if running else "闲着"))
        self.refresh_tasks(sync=not running)

    def set_standalone(self) -> None:
        """单独运行（run.py ai）时当正经主窗口用：任务栏有它，最小化/最大化/拉伸都全。"""
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint
                            | Qt.WindowCloseButtonHint)
        self.resize(900, 700)

    # ------------------------------------------------------------ 交互
    def _pick_scope(self) -> None:
        """处理范围换了：这一档等于「干哪一串 + 跑哪些视频」，立刻落盘并重算任务表。"""
        self._scope = str(self.cmb_source.currentData() or "clip_all")
        self._job = SCOPE_JOB.get(self._scope, "full")
        self._settings_touched()      # 选完即存，下次开程序还是这一档
        self.refresh_tasks()

    def set_scope(self, key: str) -> None:
        """按 key 选中某一档处理范围（`SCOPES` 里的 key），走和用户点选完全一样的路。"""
        index = self.cmb_source.findData(key)
        if index < 0:
            return
        self.cmb_source.setCurrentIndex(index)
        self._pick_scope()

    def _pick_skip_done(self) -> None:
        """「不跑成品」勾选状态变了：立刻落盘，并按新口径重算任务表里的标记。"""
        self._settings_touched()
        self.refresh_tasks()



    def on_auto(self) -> None:
        self._flush_settings()  # 防抖里可能还压着一次改动，跑的就是你看到的
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
    def _settings_touched(self, *_args) -> None:
        """界面上动了一下（目录 / 模式 / 范围 / PRM）：防抖之后自动落盘。

        没有「保存配置」这一步——敲一半的路径不会每个字符写一次盘，
        停手 350ms 或者焦点离开就写。界面还没搭完（`_ready` 为假）时一律不写。
        """
        if not self._ready:
            return
        self._save_timer.start()

    def _flush_settings(self) -> None:
        """把防抖里压着的那次改动立刻写掉（焦点离开、关窗、点「自动剪辑」都走这儿）。"""
        self.save(close=False)

    def save(self, close: bool = True) -> None:
        """落盘：处理范围（= 干哪一串 + 跑哪些视频）+ 两个 AI 专属目录。接口那些在「AI接口」里存。

        改完即存的唯一入口——目录、范围一改就（防抖后）自动调到这儿，
        不需要点任何「保存配置」。一档处理范围会写成 `ai_job` + `highlight_source` 两个键，
        再加三个布尔映射（配置文件里一眼看出干的是哪一串），互斥由这里保证。
        AI_输入目录 / AI_输出目录 只写 `ai_input_dir` / `ai_output_dir`，
        绝不碰 `paths.input_dir` / `paths.output_dir`（那是主界面的导入/导出目录）。
        写盘失败只记一句日志：正在跑的活不该被它打断。
        """
        self._save_timer.stop()
        if not self._ready:
            return
        scope = str(self.cmb_source.currentData() or "clip_all")
        bridge = {"ai_job": SCOPE_JOB.get(scope, "full"),
                  "ai_input_dir": self.edit_input.text().strip(),
                  "ai_output_dir": self.edit_output.text().strip(),
                  "highlight_source": SCOPE_SOURCE.get(scope, "all"),
                  "skip_done_products": bool(self.chk_skip_done.isChecked())}
        for name, key in JOB_FLAGS.items():
            bridge[key] = (name == bridge["ai_job"])
        patch = {"bridge": bridge}
        if patch != self._saved_patch:      # 没变就不写，免得刷新一次动一次文件
            try:
                self.cfg.save_patch(patch)
            except OSError as exc:
                self.append_log(f"[AI 面板] 设置写不进 config.json（这次只在内存里生效）：{exc}")
                return
            self._saved_patch = patch
            self.append_log(
                f"[AI 面板] 设置已存：处理范围「{self.cmb_source.currentText()}」；"
                + ("已有成品的跳过（不跑成品）" if bridge["skip_done_products"]
                   else "已有成品也重跑一遍") + "；"
                f"AI_输入目录 {bridge['ai_input_dir'] or '（留空）'}；"
                f"AI_输出目录 {bridge['ai_output_dir'] or '（留空，用导出目录）'}；"
                f"PRM 按使用状况（{self.lbl_prm.text()}）")
            self.refresh_tasks()
        if close:
            self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 的命名
        """关窗前把防抖里压着的那次改动写掉，别让最后一下白改。"""
        self._flush_settings()
        super().closeEvent(event)




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
