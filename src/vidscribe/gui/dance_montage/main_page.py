"""AI_卡点舞 主窗口。**独立窗口**，和主界面各开各的，谁崩了都不影响谁。

一期第二十四节的四个区域在这里落地：

    ① 左侧：输入与操作（选歌、选源、参数、开跑、进度、日志）  → RemixPanel
    ② 右上：素材资产（筛选 + 素材库）                        → FilterPanel + MaterialLibraryPanel
    ③ 右中：选择与推荐（候选池 + 智能推荐）                   → CandidatePanel + RecommendationPanel
    ④ 右下：历史与统计                                       → HistoryPanel + StatisticsPanel

界面线程只做两件事：画界面、读库（都是聚合查询，毫秒级）。
所有重活在 `DanceMontageWorker` 里，它自己开自己的库连接。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


from ...logging_setup import get_logger
from .alignment_panel import AlignmentPanel
from .candidate_panel import CandidatePanel
from .filter_panel import FilterPanel
from .history_panel import HistoryPanel
from .material_library import MaterialLibraryPanel
from .recommendation_panel import RecommendationPanel
from .remix_panel import RemixPanel
from .statistics_panel import StatisticsPanel
from .worker import DanceMontageWorker

logger = get_logger("dance.gui")

TITLE = "AI_卡点舞 · 舞蹈素材资产与多版本混剪"


class DanceMontageWindow(QMainWindow):
    """独立窗口。关掉它不影响主界面，主界面崩了也不影响它。"""

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.db: Any = None
        self.worker: DanceMontageWorker | None = None
        self._song_id = 0

        self.setWindowTitle(TITLE)
        self.resize(1500, 950)
        self.setMinimumSize(1000, 640)     # 小窗口也要能用（一期第十五节）

        self.remix = RemixPanel(cfg, self)
        self.filters = FilterPanel(self)
        self.library = MaterialLibraryPanel(self)
        self.alignment = AlignmentPanel(self)
        self.candidates = CandidatePanel(self)
        self.recommend = RecommendationPanel(self)
        self.history = HistoryPanel(self)
        self.statistics = StatisticsPanel(self)

        self.setCentralWidget(self._build_body())
        self.setStatusBar(QStatusBar(self))
        self._wire()
        self._open_db()
        self.reload()

    # ------------------------------------------------------------------ 界面
    def _build_body(self) -> QWidget:
        assets = QWidget(self)
        assets_layout = QHBoxLayout(assets)
        assets_layout.setContentsMargins(6, 6, 6, 6)
        assets_layout.addWidget(self.filters, 1)
        assets_layout.addWidget(self.library, 3)

        choose = QWidget(self)
        choose_layout = QVBoxLayout(choose)
        choose_layout.setContentsMargins(6, 6, 6, 6)
        inner = QSplitter(Qt.Vertical, choose)
        inner.addWidget(self.candidates)
        inner.addWidget(self.recommend)
        inner.setStretchFactor(0, 3)
        inner.setStretchFactor(1, 2)
        choose_layout.addWidget(inner)

        review = QWidget(self)
        review_layout = QHBoxLayout(review)
        review_layout.setContentsMargins(6, 6, 6, 6)
        review_layout.addWidget(self.history, 3)
        review_layout.addWidget(self.statistics, 2)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(assets, "素材资产")
        self.tabs.addTab(self.alignment, "音频对齐")
        self.tabs.addTab(choose, "选择与推荐")
        self.tabs.addTab(review, "历史与统计")

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        hint = QLabel("固定音乐位置：所有源视频都贴在**目标歌**这同一把尺子上。"
                      "素材是长期资产，只会停用、不会删除。", left)
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        left_layout.addWidget(self.remix, 1)

        split = QSplitter(Qt.Horizontal, self)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([560, 900])
        return split

    def _wire(self) -> None:
        self.remix.start_requested.connect(self.start)
        self.remix.stop_requested.connect(self.stop)
        self.remix.song_changed.connect(self._song_typed)
        self.remix.slice_changed.connect(lambda _v: self.reload())
        self.remix.remove_song_requested.connect(self._remove_song)
        self.remix.show_retired.stateChanged.connect(lambda _s: self.reload())

        self.filters.btn_apply.clicked.connect(self._reload_materials)
        self.filters.changed.connect(lambda _spec: self._reload_materials())
        self.library.changed.connect(self.reload)
        self.alignment.realign_requested.connect(self._realign)
        self.alignment.changed.connect(self.reload)
        self.candidates.manual_changed.connect(self.remix.set_manual)
        self.recommend.adopted.connect(self._adopt)
        self.recommend.changed.connect(self.reload)
        self.history.rerender_requested.connect(self._rerender)

    # ------------------------------------------------------------------ 库
    def _open_db(self) -> None:
        from ...db import open_db

        self.cfg.ensure_dance_dirs()
        self.db = open_db(self.cfg)
        from ...dance import strategy as strategy_mod

        strategy_mod.ensure_presets(self.db)

    def _song_typed(self, text: str) -> None:
        """输入框里给的是库里的 id 时，立刻切过去；给路径就等跑完再说。"""
        if str(text).strip().isdigit():
            self._song_id = int(str(text).strip())
            self.reload()

    def _pool_spec(self):
        """给候选池用的筛选条件。

        和素材库列表共用同一套条件，但**换成配置里的候选池上限**：
        素材库那个「最多显示」是给眼睛看的（300 行够翻了），
        候选池那个是"算法能考虑多少条"，两者混用会让新素材在打分前就被丢掉。
        """
        from dataclasses import replace

        spec = self.filters.spec(self._song_id)
        return replace(spec, limit=int(self.cfg.dance["candidate_pool_size"]))

    def reload(self) -> None:
        """把四个区域全部按当前目标歌刷一遍。全是聚合查询，很快。"""
        if self.db is None:
            return
        from ...dance import material_repository as repo

        self.remix.set_songs(repo.list_songs(
            self.db, include_retired=self.remix.show_retired.isChecked()))

        slice_duration = self.remix.slice_duration()
        self.alignment.refresh(self.db, self._song_id)
        self.candidates.refresh(self.db, self._song_id, slice_duration=slice_duration,
                                spec=self._pool_spec())

        self.recommend.refresh(self.db, self._song_id, slice_duration=slice_duration)
        self.history.refresh(self.db, self._song_id)
        self.statistics.refresh(self.db, self._song_id)
        self._reload_materials()
        row = repo.get_song(self.db, self._song_id) if self._song_id else None
        if row is not None:
            self.statusBar().showMessage(
                f"目标歌 #{self._song_id}《{row['title']}》"
                f"{float(row['duration'] or 0):.2f}s｜BPM {float(row['bpm'] or 0):.1f}"
                f"｜每格 {slice_duration:g}s")
        else:
            self.statusBar().showMessage("还没选目标歌")

    def _reload_materials(self) -> None:
        if self.db is None or not self._song_id:
            self.library.show_materials(self.db, 0, [])
            return
        from ...dance import material_selection as selection

        spec = self.filters.spec(self._song_id)
        found = selection.find_materials(self.db, spec,
                                        exclude_recent=self.filters.exclude_recent_mode())
        self.library.show_materials(self.db, self._song_id, found)

    def _adopt(self, picks: dict) -> None:
        self.candidates.adopt(dict(picks))
        self.remix.set_manual(self.candidates.manual())

    # ------------------------------------------------------------ 下架 / 删除
    def _remove_song(self, song_id: int) -> None:
        """移除一首目标歌。**下架是默认答案，彻底删除要额外一次确认。**

        为什么不给一个干脆的"删除"了事：`dance_materials` / `dance_audio_alignments` /
        `dance_montages` 都是 `ON DELETE CASCADE`，真删下去会把素材、对齐、
        历史成片和它们的事件流水一起抹掉。而这个项目的立足点就是"素材是长期资产"。
        所以这里先把挂在它下面的数量摆出来，让用户自己选。
        """
        from ...dance import material_repository as repo

        if self.db is None:
            return
        row = repo.get_song(self.db, int(song_id))
        if row is None:
            QMessageBox.warning(self, "找不到", f"库里没有目标歌 #{song_id}")
            self.reload()
            return
        title = str(row["title"] or f"#{song_id}")

        # 已经下架的：这一步变成"恢复"
        if repo.song_retirement(self.db, int(song_id)) is not None:
            answer = QMessageBox.question(
                self, "恢复目标歌",
                f"《{title}》现在是已下架状态。要恢复它吗？\n"
                "（下架期间素材和历史一条都没动，恢复后立刻能继续用）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer == QMessageBox.Yes:
                repo.restore_song(self.db, int(song_id))
                self.reload()
            return

        usage = repo.song_usage(self.db, int(song_id))
        detail = repo.describe_usage(usage)


        box = QMessageBox(self)
        box.setWindowTitle("移除目标歌")
        box.setIcon(QMessageBox.Question)
        box.setText(f"《{title}》下面挂着：{detail}")
        box.setInformativeText(
            "下架（推荐）：只是从界面上隐藏，素材、对齐、历史成片一条都不动，随时能恢复。\n\n"
            "彻底删除：连带删掉上面列出的全部数据库记录，**不可撤销**。\n"
            "只有在这首歌是误加进来的（选错文件、重复导入）时才该用它。\n"
            "磁盘上已经切好的素材文件不会被删，需要的话自己去素材目录清理。")
        hide = box.addButton("下架（可恢复）", QMessageBox.AcceptRole)
        drop = box.addButton("彻底删除", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(hide)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is hide:
            reason, ok = QInputDialog.getText(
                self, "写个理由", "为什么下架这首歌？（会连同时间一起存档）")
            if not ok or not reason.strip():
                QMessageBox.information(self, "没有下架",
                                        "没写理由，这次不下架 —— 不允许静默隐藏。")
                return
            repo.retire_song(self.db, int(song_id), reason=reason.strip(), operator="gui")
            self.remix.append_log(f"[目标歌] 《{title}》已下架：{reason.strip()}")
        elif clicked is drop:
            again = QMessageBox.warning(
                self, "最后确认",
                f"真的要彻底删除《{title}》吗？\n\n"
                f"连带删除：{detail}\n"
                "这一步不可撤销。想留后路请改用「下架」。",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if again != QMessageBox.Yes:
                return
            removed = repo.delete_song(self.db, int(song_id), confirm=True)
            self.remix.append_log(
                f"[目标歌] 《{title}》已彻底删除，连带 {repo.describe_usage(removed)}")

        else:
            return

        if int(song_id) == self._song_id:          # 当前正看着它，得把界面切回空
            self._song_id = 0
            self.remix.song.clear()
        self.reload()



    # ------------------------------------------------------------------ 跑活
    def start(self, job: dict) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "还在跑", "上一轮还没结束，等它跑完或者先点停止。")
            return
        if not job.get("song"):
            QMessageBox.warning(self, "还没选目标歌", "先选一首目标歌（文件或库里的 id）。")
            return
        if not job.get("recommend") and not job.get("manual"):
            QMessageBox.warning(self, "关了推荐就得自己挑",
                                "智能推荐已关闭，但「候选池」里还没有任何手动选择。\n"
                                "去候选池里逐格钉选，或者点「按推荐填满所有格」。")
            return
        job["strategy_id"] = self.recommend.strategy_id()
        job["preset"] = str(self.filters.preset.currentData() or "")
        self.worker = DanceMontageWorker(self.cfg, job, self)
        self.worker.log.connect(self.remix.append_log)
        self.worker.stage.connect(self.remix.show_stage)
        self.worker.progress.connect(self.remix.show_progress)
        self.worker.done.connect(self._finished)
        self.remix.set_running(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def _finished(self, ok: bool, message: str, result: object) -> None:
        self.remix.set_running(False)
        self.remix.show_done(bool(ok), str(message))
        data = result if isinstance(result, dict) else {}
        if data.get("song_id"):
            self._song_id = int(data["song_id"])
        self.reload()
        outputs = data.get("outputs") or []
        if ok and outputs:
            QMessageBox.information(
                self, "出片了",
                message + "\n\n" + "\n".join(str(Path(p).name) for p in outputs[:6])
                + f"\n\n目录：{Path(str(outputs[0])).parent}")
        elif not ok:
            QMessageBox.warning(self, "没出来", message)

    def _realign(self) -> None:
        job = self.remix.job()
        job.update({"force": True, "do_slice": False, "do_remix": False})
        self.start(job)

    def _rerender(self, version_id: int) -> None:
        """重渲染一个历史版本：读回计划，原样再跑一遍渲染。"""
        if self.db is None:
            return
        from ...dance import media_backend, montage_render, montage_timeline

        context = montage_timeline.load(self.db, int(version_id))
        if context is None:
            QMessageBox.warning(self, "读不出来", f"版本 #{version_id} 的计划读不出来。")
            return
        problems = montage_timeline.validate(context)
        if problems:
            QMessageBox.warning(self, "这一版现在渲染不了", "\n".join(problems[:6]))
            return
        canvas = media_backend.Canvas(width=int(self.cfg.dance["canvas_width"]),
                                      height=int(self.cfg.dance["canvas_height"]),
                                      fps=float(self.cfg.dance["canvas_fps"]))
        self.remix.append_log(f"[重渲染] 版本 #{version_id}｜{len(context.clips)} 格")
        result = montage_render.render(
            self.db, context, out_dir=self.cfg.dance_path("output_dir"),
            version_id=int(version_id), canvas=canvas,
            on_log=self.remix.append_log)
        for line in montage_render.describe(result):
            self.remix.append_log(line)
        self.reload()
        if result.ok:
            QMessageBox.information(self, "重渲染完成", str(result.output))
        else:
            QMessageBox.warning(self, "重渲染失败", result.error or "未知原因")

    # ------------------------------------------------------------------ 收尾
    def closeEvent(self, event) -> None:                     # noqa: N802 - Qt 的名字
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self, "还在跑", "后台还在跑，真的关掉吗？（会等当前这一步做完）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait(15000)
        if self.db is not None:
            self.db.close()
            self.db = None
        super().closeEvent(event)


__all__ = ["TITLE", "DanceMontageWindow"]
