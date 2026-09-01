"""AI GUI 自动剪辑模块：配置实时保存 / 扫描 / 三个状态 / 三条流程 / 统计 / 停止续跑。

盯的是这一批的红线：
  T1  AI_输入目录 / AI_输出目录 用的是自己的键，绝不碰 paths.input_dir / output_dir
  T2  改目录不用点保存：改完（防抖 flush）就在 config.json 里
  T3  选模式不用点保存，而且三个开关互斥
  T4  重开面板读回上次的模式和两个目录
  T5  老配置只有 ai_job 也照旧认；只有三个布尔开关时也认
  T6  面板上没有「保存设置」这种必须点的按钮
  T7  扫描：库里有分析结果的 / 只有 MP4 的 / 多个 MP4 都进任务表，没分析过的不被丢掉
  T8  六步各自独立（分析 / 剧本 / 高光分析 / 高光 JSON / 剪辑 / 成品），不互相推断
  T9  状态词表：未处理 / 等待高光分析 / 上传中 / 等待高光 JSON / 高光 JSON 就绪 /
      高光 JSON 失败 / 剪辑中 / 剪辑失败 / 成品完成 / 跳过 / 分析失败
  T10 七个头号数字：总任务 / 已分析 / 已有剧本 / 已分析高光 / 已获取 JSON / 已剪辑 / 成品
  T11 统计跟着事件自己动：成品 +1 立刻反映，不用点刷新
  T12 有 TXT 就不重跑分析，直接发 AI
  T13 没 TXT 先按主界面配置分析（不是拿 AI 那份配置）
  T14 分析失败＝这条记 failed，而且一个字节都没发出去
  T15 已经有成品的 MP4 在分析 / 上传 / 取 JSON **之前**就被跳过
  T16 干完没干完不看文件在不在，看库里的任务生命周期 + artifacts
  T17 发给扩展的只有 PRM + 这个视频的 TXT（视频本体不上传），任务里标明是哪个 MP4；
      PRM 正文来自数据库，上传时文件名统一成 prompt.txt / prompt_2.txt…
  T18 MP4 不在盘上不发（对象没了），TXT 不在盘上也不发（先分析生成）
  T19 收取脚本这一串不渲染：库里有可复用高光 JSON 才算干完（文件在不在不算）
  T20 脚本剪辑这一串一次 AI 都不调
  T21 停止：不再领新任务，手上这条退回 pending（不是 cancelled），下次接着跑
  T22 AI 自动化只是调度器：面板里没有第二套剪辑引擎，也不阻塞 GUI 线程
  T23 面板不产生 AI 调用（ai_tasks / ai_results 一行都不加）
  T24 面板里没有裸 SQL
  T25 语言被拦下的视频（videos.blocked_language）不排队；已排队的领到手就判 cancelled
  T26 两条进度条：单条视频一条、队列一条，单条那条在上面；字号比默认小一档
  T27 主界面的**每一行**日志都转播进面板（不再只挑 [自动剪辑] / [AI 开头那几行）
  T28 任务总览只剩七个头号数字，底下那排九格（总视频 / 待剪辑 / 已完成…）已经撤了
  T29 「清空非中英视频」＝只清语言预检拦下的那些（`videos.blocked_language` 有值）：
      原视频 + 附带文件 + 库里记录一起清；没有时按钮是灰的，自动剪辑在跑时拒绝动手

全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_ai_panel_auto.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (QApplication, QGroupBox, QLabel,   # noqa: E402
                             QMessageBox, QPlainTextEdit, QPushButton)

from vidscribe.config import Config                       # noqa: E402
from vidscribe.db import assets as db_assets              # noqa: E402
from vidscribe.db import open_db                          # noqa: E402
from vidscribe.db import repo as db_repo                  # noqa: E402
from vidscribe.gui import ai_options as ao                # noqa: E402
from vidscribe.gui import main_window as mw               # noqa: E402

PANEL_SRC = (ROOT / "src" / "vidscribe" / "gui" / "ai_options.py").read_text(encoding="utf-8")

_APP = None


def app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
    """临时工程：AI 的两个目录跟主界面的导入/导出目录**故意指到不同地方**。"""
    for sub in ("database", "input", "output", "logs", "cache", "prm",
                "ai_in", "ai_out"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data.setdefault("paths", {}).update({
        "db_dir": str(tmp_path / "database"),
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "input_dir": str(tmp_path / "input"),
        "video_dir": "",
        "log_dir": str(tmp_path / "logs"),
    })
    bridge = data.setdefault("bridge", {})
    bridge["ai_input_dir"] = str(tmp_path / "ai_in")
    bridge["ai_output_dir"] = str(tmp_path / "ai_out")
    bridge["ai_job"] = "full"
    bridge["highlight_source"] = "all"
    bridge["prm_id"] = 0
    bridge["mode"] = "extension"
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str, folder: str = "ai_in") -> Path:
    path = cfg.root / folder / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def video_row(cfg, db, name: str) -> tuple[Path, int]:
    video = fake_video(cfg, name)
    return video, db_repo.upsert_video(db, video)


def panel(cfg, log=None) -> ao.AiPanel:
    app()
    return ao.AiPanel(cfg, None, log=log or (lambda _line: None))


def on_disk(cfg) -> dict:
    return json.loads((cfg.root / "config.json").read_text(encoding="utf-8"))


def row_of(view, name: str) -> list[str]:
    for row in range(view.table.rowCount()):
        if view.table.item(row, 0).text() == name:
            return [view.table.item(row, col).text()
                    for col in range(view.table.columnCount())]
    raise AssertionError(f"任务表里没有 {name}")


def head(view) -> dict[str, int]:
    return {key: int(label.text()) for key, label in view._head_labels.items()}


def touched(view) -> None:
    """模拟「用户改完了」：把防抖压着的那次改动立刻写掉。"""
    view._flush_settings()


def same(a: Path | None, b: Path | None) -> bool:
    """同一个文件吗。Windows 上 8.3 短名（ADMINI~1）和长名会混着出现，按真实路径比。"""
    if a is None or b is None:
        return a is b
    return Path(a).resolve() == Path(b).resolve()


def artifact(db, vid: int, kind: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x" * 64, encoding="utf-8")
    return db_repo.register_artifact(db, vid, kind, path)


def ai_json(db, vid: int, *, clips: bool = True, task_id: int | None = None) -> None:
    payload = {"video": "v.mp4",
               "clip": {"start": 1.0, "end": 9.0, "score": 0.9,
                        "type": "hook", "reason": "r"}} if clips else {"clips": []}
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=payload,
                           raw_response=json.dumps(payload))


def analysis(db, vid: int, *, events: bool = True) -> int:
    """往库里放一条跑成功的分析。有事件＝库里能出完整剧本（面板的「剧本」那一列）。"""
    sig = {"vision_model": "m", "vision_config": None, "vision_config_hash": "h",
           "asr_model": "a", "asr_config": None, "asr_config_hash": "h"}
    aid = db_repo.create_analysis(db, vid, sig)
    if events:
        db_repo.save_visual_events(db, aid, [{"id": 1, "start": 0.0, "end": 4.0,
                                              "description": "有人走进来",
                                              "confidence": 0.8}])
    db_repo.finish_analysis(db, aid, scene_count=1 if events else 0, speech_count=0)
    return aid


def rendered_clip(db, vid: int, output: Path) -> None:
    """库里记一笔"这个视频剪过"（clips.rendered），面板的「剪辑」那一列看这个。"""
    result = db.one("SELECT id FROM ai_results WHERE video_id = ? ORDER BY id DESC", (vid,))
    spec = {"start": 1.0, "end": 9.0, "score": 0.9, "type": "hook", "reason": "r"}
    db_repo.create_clip(db, vid, spec,
                        ai_result_id=int(result["id"]) if result is not None else None,
                        status="rendered", output_path=output)


def counts(db) -> tuple[int, int]:
    return (int(db.value("SELECT COUNT(*) FROM ai_tasks") or 0),
            int(db.value("SELECT COUNT(*) FROM ai_results") or 0))


# ------------------------------------------------------------------ 替身
class Win:
    """够跑「领任务 → 分析/发送/剪辑 → 落状态」这一串的主窗口替身。"""

    on_auto_clip = mw.MainWindow.on_auto_clip
    _auto_step = mw.MainWindow._auto_step
    _enqueue_auto_tasks = mw.MainWindow._enqueue_auto_tasks
    _source_allows = mw.MainWindow._source_allows
    _resume_existing_ai_json = mw.MainWindow._resume_existing_ai_json
    _asset_json_for_render = mw.MainWindow._asset_json_for_render
    _reusable_highlight_json = mw.MainWindow._reusable_highlight_json
    script_payload = mw.MainWindow.script_payload
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _auto_chain_done = mw.MainWindow._auto_chain_done
    _skip_because_done = mw.MainWindow._skip_because_done
    skip_done_products = mw.MainWindow.skip_done_products
    _language_blocked = mw.MainWindow._language_blocked
    _set_video_progress = mw.MainWindow._set_video_progress
    _auto_product_ready = mw.MainWindow._auto_product_ready
    _ai_files_ok = mw.MainWindow._ai_files_ok
    _auto_advance = mw.MainWindow._auto_advance
    _auto_finish = mw.MainWindow._auto_finish
    _settle_auto_task = mw.MainWindow._settle_auto_task
    _mark_auto_waiting = mw.MainWindow._mark_auto_waiting
    _mark_auto_rendering = mw.MainWindow._mark_auto_rendering
    _db_video_id = mw.MainWindow._db_video_id
    _register_artifact = mw.MainWindow._register_artifact
    _auto_save_script = mw.MainWindow._auto_save_script
    highlight_source = mw.MainWindow.highlight_source
    selected_prm = mw.MainWindow.selected_prm
    enabled_prms = mw.MainWindow.enabled_prms
    has_prm_profiles = mw.MainWindow.has_prm_profiles
    resolve_prompt_files = mw.MainWindow.resolve_prompt_files
    resolve_prompt_file = mw.MainWindow.resolve_prompt_file
    _write_prompt_files = mw.MainWindow._write_prompt_files
    send_file_to_ai = mw.MainWindow.send_file_to_ai
    dispatch_ai = mw.MainWindow.dispatch_ai
    auto_running = mw.MainWindow.auto_running
    auto_busy = mw.MainWindow.auto_busy
    ai_dir = mw.MainWindow.ai_dir
    export_root = mw.MainWindow.export_root
    _sync_disk = mw.MainWindow._sync_disk
    _worker_id = mw.MainWindow._worker_id
    on_bridge_stop = mw.MainWindow.on_bridge_stop
    clean_bridge_temp = mw.MainWindow.clean_bridge_temp
    VIDEO_SUFFIXES = mw.MainWindow.VIDEO_SUFFIXES

    def __init__(self, cfg, db):
        self.cfg = cfg
        self._db_handle = db
        self._db_failed = False
        self._auto_job = "full"
        self._auto_task_id = None
        self._auto_video = None
        self._auto_active = False
        self._auto_stop = False
        self._auto_done = 0
        self._auto_total = 0
        self._last_highlight_json = ""
        self._last_prompt = {}
        self._last_prm_id = None
        self._last_asset_id = None
        self._bridge_temp_files = []
        self.video_path = None
        self.export_dir = None
        self.worker = None
        self.clip_worker = None
        self.ai_worker = None
        self.bridge = FakeBridge()
        self.ai_panel = None
        self.speech = []
        self.timeline = []
        self.logs: list[str] = []
        self.calls = {k: 0 for k in ("on_analyze", "run_highlight", "load_video",
                                     "write_ai_text")}

    # --- 叶子调用换成计数替身：这一批测的是调度，不是渲染/分析本身
    def _db(self):
        return self._db_handle

    def append_log(self, line: str) -> None:
        self.logs.append(line)

    def load_video(self, video):
        self.calls["load_video"] += 1
        self.video_path = Path(video)

    def on_analyze(self, force: bool = False) -> None:
        self.calls["on_analyze"] += 1

    def run_highlight(self, text: str, ai: bool = False, name_suffix: str = "") -> None:
        self.calls["run_highlight"] += 1
        self._last_highlight_json = text

    def write_ai_text(self):
        self.calls["write_ai_text"] += 1
        target = self.ai_dir("ai_input_dir") / f"{self._auto_video.stem}.txt"
        target.write_text("merged", encoding="utf-8")
        return target, 1

    def output_dir(self):
        return None

    def refresh_bridge_label(self) -> None:
        pass

    def _set_auto_state(self, idle: bool, state: str = "") -> None:
        if self.ai_panel is not None:
            self.ai_panel.set_running(not idle, state)

    def _set_auto_step(self, stem: str, step: str) -> None:
        if self.ai_panel is not None:
            self.ai_panel.set_active(stem, step)

    def _set_auto_progress(self, done: int) -> None:
        if self.ai_panel is not None:
            self.ai_panel.set_queue_progress(done, max(self._auto_total, done))

    def _note_prompt_use(self, prompt_path) -> None:
        self._last_prompt = {"path": str(prompt_path)}


class FakeBridge:
    """只记下"任务长什么样"的 Bridge 替身，不起 HTTP 服务。"""

    def __init__(self):
        self.tasks: list[tuple[str, dict, list[Path]]] = []
        self.cancelled = 0

    def submit(self, task_type, payload, files=None):
        self.tasks.append((task_type, payload, [Path(p) for p in (files or [])]))
        return f"task-{len(self.tasks)}"

    def state(self):
        return {"extension_online": True}

    def cancel(self):
        self.cancelled += 1


def quiet():
    """别在离屏环境里真弹窗。"""
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)


def prm(cfg) -> Path:
    path = cfg.root / "prm" / "prm_en.txt"
    path.write_text("rules", encoding="utf-8")
    cfg.bridge["prompt_file"] = str(path)
    return path


def wired(cfg, db) -> tuple[Win, ao.AiPanel]:
    """主窗口替身 + 面板，两边接上（面板收进度回调）。"""
    quiet()
    window = Win(cfg, db)
    view = panel(cfg, log=window.append_log)
    window.ai_panel = view
    return window, view


# ------------------------------------------------------------------ T1-T6 配置
def test_ai_dirs_have_their_own_keys(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    view.edit_input.setText(str(tmp_path / "ai_in2"))
    (tmp_path / "ai_in2").mkdir()
    touched(view)
    data = on_disk(cfg)
    assert data["bridge"]["ai_input_dir"] == str(tmp_path / "ai_in2")
    assert data["paths"]["input_dir"] == str(tmp_path / "input"), "不许动主界面的导入目录"
    assert data["paths"]["output_dir"] == str(tmp_path / "output"), "不许动主界面的导出目录"
    assert "ai_input_dir" not in data["paths"] and "ai_output_dir" not in data["paths"]


def test_dirs_save_without_a_button(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    (tmp_path / "out2").mkdir()
    view.edit_output.setText(str(tmp_path / "out2"))
    touched(view)   # 只是"焦点离开"，没有点任何保存按钮
    assert on_disk(cfg)["bridge"]["ai_output_dir"] == str(tmp_path / "out2")


def test_scope_saves_job_and_source_together(tmp_path: Path) -> None:
    """一个「处理范围」下拉写两个键：干哪一串（ai_job + 布尔映射）+ 跑哪些视频。"""
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    view.set_scope("collect_missing")
    touched(view)
    saved = on_disk(cfg)["bridge"]
    assert saved["ai_job"] == "collect" and saved["highlight_source"] == "missing"
    assert saved["ai_collect_script"] is True
    assert saved["ai_clip_video"] is False and saved["ai_script_clip"] is False

    view.set_scope("clip_existing")
    touched(view)
    saved = on_disk(cfg)["bridge"]
    assert saved["ai_job"] == "script" and saved["highlight_source"] == "existing", \
        "「只跑已有 JSON 的：直接剪」= script + existing"
    assert saved["ai_script_clip"] is True and saved["ai_collect_script"] is False
    # 三张模式卡片已经并进这一个下拉，界面上不许再有它们
    assert not hasattr(view, "_job_group"), "「干哪一串」那三张卡片应该已经删掉"
    assert view.cmb_source.count() == len(ao.SCOPES), "四档处理范围要全在下拉里"


def test_settings_come_back_after_restart(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    view.set_scope("clip_existing")
    (tmp_path / "again").mkdir()
    view.edit_output.setText(str(tmp_path / "again"))
    touched(view)
    fresh = Config.load(tmp_path, tmp_path / "config.json")
    again = panel(fresh)
    assert again._job == "script"
    assert again._scope == "clip_existing", "重开还是同一档处理范围"
    assert again.cmb_source.currentData() == "clip_existing"
    assert again.edit_output.text() == str(tmp_path / "again")
    assert again.edit_input.text() == str(tmp_path / "ai_in")


def test_old_and_new_config_shapes_both_read(tmp_path: Path) -> None:
    assert ao._job_from_config({"ai_job": "collect"}) == "collect"
    assert ao._job_from_config({"ai_script_clip": True}) == "script"
    assert ao._job_from_config({}) == "full"
    assert ao._job_from_config({"ai_job": "什么鬼"}) == "full"
    # 老配置的 (ai_job, highlight_source) 组合要能反推成一档处理范围
    assert ao._scope_from_config({"ai_job": "full", "highlight_source": "all"}) == "clip_all"
    assert ao._scope_from_config({"ai_job": "full",
                                  "highlight_source": "missing"}) == "clip_missing"
    assert ao._scope_from_config({"ai_job": "script",
                                  "highlight_source": "existing"}) == "clip_existing"
    assert ao._scope_from_config({}) == "clip_all"
    # 矛盾组合（脚本剪辑 + 只跑没有 JSON 的）不许把界面搞空，按串归到最近那一档
    assert ao._scope_from_config({"ai_job": "script",
                                  "highlight_source": "missing"}) == "clip_existing"


def test_skip_done_products_saves_live(tmp_path: Path) -> None:
    """「不跑成品」：默认勾上，勾完即存，下次开程序还是这个状态。"""
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    assert view.chk_skip_done.isChecked() is True, "「不跑成品」默认要勾上"
    view.chk_skip_done.setChecked(False)
    touched(view)
    assert on_disk(cfg)["bridge"]["skip_done_products"] is False, "取消勾选要落盘"
    fresh = Config.load(tmp_path, tmp_path / "config.json")
    again = panel(fresh)
    assert again.chk_skip_done.isChecked() is False, "重启后要记得取消过勾选"
    again.chk_skip_done.setChecked(True)
    touched(again)
    assert on_disk(fresh)["bridge"]["skip_done_products"] is True, "重新勾上也要落盘"


def test_unchecking_skip_reruns_finished_videos(tmp_path: Path) -> None:
    """取消「不跑成品」：成品库里有成品的视频照样重跑一遍（默认那档见上一条）。"""
    cfg, db = make_project(tmp_path)
    prm(cfg)
    _video, vid = video_row(cfg, db, "done.mp4")
    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "done.txt")
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "done_成品.mp4")
    cfg.bridge["skip_done_products"] = False
    window, _view = wired(cfg, db)
    window.on_auto_clip()
    assert int(db.value("SELECT COUNT(*) FROM ai_tasks") or 0) == 1, "取消勾选就要给它排队"
    assert len(window.bridge.tasks) == 1, "已有成品也要重新发一次 AI"
    # 任务完成的判据不许跟着这个勾选框走，否则剪完永远落不了 completed
    assert window._auto_chain_done(Path(_video)) is True, "库里有成品就是干完了"
    assert window._skip_because_done(Path(_video)) is False, "取消勾选时不许跳过"


def test_no_mandatory_save_button(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    labels = [b.text() for b in view.findChildren(QPushButton)]
    for bad in ("保存设置", "保存配置"):
        assert bad not in labels, f"不许有必须点的「{bad}」"
    assert "▶ 自动剪辑" in labels and "■ 停止" in labels


# ------------------------------------------------------------------ T7-T11 扫描 / 状态 / 统计
def test_scan_keeps_every_mp4(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    both, vid = video_row(cfg, db, "both.mp4")
    analysis(db, vid)                         # 库里有分析结果 -> 能出完整剧本
    video_row(cfg, db, "lonely.mp4")          # 只有 MP4，库里什么都没有
    video_row(cfg, db, "third.mp4")
    view = panel(cfg)
    names = {view.table.item(r, 0).text() for r in range(view.table.rowCount())}
    assert names == {"both.mp4", "lonely.mp4", "third.mp4"}, "没分析过的 MP4 不许被丢掉"
    assert row_of(view, "both.mp4")[2] == ao.DONE, "库里能出剧本"
    assert row_of(view, "lonely.mp4")[2] == ao.WAITING


def test_three_statuses_are_independent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "a.mp4")
    view = panel(cfg)
    assert row_of(view, "a.mp4")[1:7] == [ao.WAITING] * 6
    ai_json(db, vid)                      # 只有高光 JSON：分析、剪辑、成品都还没有
    view.refresh_tasks()
    marks = row_of(view, "a.mp4")
    assert marks[4] == ao.DONE and marks[3] == ao.DONE, "有 AI 结果＝问过 AI 且 JSON 可用"
    assert marks[1] == ao.WAITING and marks[5] == ao.WAITING and marks[6] == ao.WAITING
    analysis(db, vid)
    view.refresh_tasks()
    marks = row_of(view, "a.mp4")
    assert marks[1] == ao.DONE and marks[2] == ao.DONE, "分析和剧本是分开判的"
    product = cfg.root / "ai_out" / "a_成品.mp4"
    rendered_clip(db, vid, product)
    view.refresh_tasks()
    marks = row_of(view, "a.mp4")
    assert marks[5] == ao.DONE and marks[6] == ao.WAITING, "剪过 ≠ 成品还在盘上"
    artifact(db, vid, "final_video", product)
    view.refresh_tasks()
    assert row_of(view, "a.mp4")[6] == ao.DONE


def test_status_words_cover_every_stage(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "s.mp4")
    view = panel(cfg)
    assert row_of(view, "s.mp4")[7] == "未处理"

    analysis(db, vid)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "等待高光分析"

    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    db_repo.claim_ai_task(db, task_id)                 # uploading
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "上传中"

    db_repo.mark_ai_task_waiting(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "等待高光 JSON"

    ai_json(db, vid, task_id=task_id)
    db_repo.mark_ai_task_rendering(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "剪辑中"

    db_repo.complete_ai_task(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "高光 JSON 就绪"

    artifact(db, vid, "final_video", cfg.root / "ai_out" / "s_成品.mp4")
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[7] == "成品完成"

    # 正在跑的那一步：分析中
    view.set_active("s.mp4".replace(".mp4", ""), "分析")
    assert row_of(view, "s.mp4")[7] == "成品完成", "有成品就是完成，不被「在跑」盖掉"
    view.set_active("", "")

    # 分析失败 / 高光 JSON 失败 / 剪辑失败 / 跳过：全靠 ai_tasks 的状态 + 库里有什么
    bad, bad_id = video_row(cfg, db, "bad.mp4")
    bad_task, _ = db_repo.enqueue_ai_task(db, bad_id, mode="full")
    db_repo.fail_ai_task(db, bad_task, "分析炸了")
    view.refresh_tasks(sync=True)
    assert row_of(view, "bad.mp4")[7] == "分析失败"
    analysis(db, bad_id)
    view.refresh_tasks()
    assert row_of(view, "bad.mp4")[7] == "高光 JSON 失败"
    ai_json(db, bad_id)
    view.refresh_tasks()
    assert row_of(view, "bad.mp4")[7] == "剪辑失败"

    skipped, skip_id = video_row(cfg, db, "skip.mp4")
    skip_task, _ = db_repo.enqueue_ai_task(db, skip_id, mode="full")
    db_repo.cancel_ai_task(db, skip_task, "停了")
    view.refresh_tasks(sync=True)
    assert row_of(view, "skip.mp4")[7] == "跳过"

    words = {row_of(view, n)[7] for n in ("s.mp4", "bad.mp4", "skip.mp4")}
    assert words <= {"未处理", "分析中", "分析失败", "分析没出内容", "等待高光分析",
                     "上传中", "等待高光 JSON", "高光 JSON 就绪", "高光 JSON 失败",
                     "剪辑中", "剪辑失败", "成品完成", "跳过", "生成剧本", "跑着"}


def test_headline_numbers(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _a, a_id = video_row(cfg, db, "a.mp4")
    _b, b_id = video_row(cfg, db, "b.mp4")
    view = panel(cfg)
    assert head(view) == {"total": 2, "analysed": 0, "script": 0, "attempted": 0,
                          "json": 0, "rendered": 0, "made": 0}
    analysis(db, a_id)
    view.refresh_tasks()
    assert head(view)["analysed"] == 1 and head(view)["script"] == 1
    ai_json(db, a_id)
    view.refresh_tasks()
    assert head(view)["json"] == 1 and head(view)["attempted"] == 1
    product = cfg.root / "ai_out" / "a_成品.mp4"
    rendered_clip(db, a_id, product)
    artifact(db, a_id, "final_video", product)
    view.refresh_tasks()
    assert head(view) == {"total": 2, "analysed": 1, "script": 1, "attempted": 1,
                          "json": 1, "rendered": 1, "made": 1}


def test_stats_follow_the_run_without_a_refresh_click(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _v, vid = video_row(cfg, db, "a.mp4")
    window, view = wired(cfg, db)
    before = head(view)
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "a_成品.mp4")
    window._auto_total = 1
    window._set_auto_progress(1)          # 主界面推进度 -> 面板自己刷
    after = head(view)
    assert before["made"] == 0 and after["made"] == 1, "不点刷新也要跟上"


# ------------------------------------------------------------------ T12-T18 流程
def test_existing_txt_skips_analysis(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video, vid = video_row(cfg, db, "a.mp4")
    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "a.txt")
    window, view = wired(cfg, db)
    window.on_auto_clip()
    assert window.calls["on_analyze"] == 0, "有 TXT 就不许重跑分析"
    assert len(window.bridge.tasks) == 1


def test_missing_txt_analyses_first(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video_row(cfg, db, "a.mp4")
    window, view = wired(cfg, db)
    window.on_auto_clip()
    assert window.calls["on_analyze"] == 1, "缺 TXT 得先分析"
    assert window.bridge.tasks == [], "分析还没完，一个字节都不许发"


def test_analysis_failure_stops_this_one(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video, vid = video_row(cfg, db, "a.mp4")
    window, view = wired(cfg, db)
    window.on_auto_clip()
    task_id = window._auto_task_id
    window._auto_advance("failed", "分析失败：炸了")   # 分析线程回来说失败
    row = db.one("SELECT status, error FROM ai_tasks WHERE id = ?", (task_id,))
    assert row["status"] in ("failed", "pending")
    assert window.bridge.tasks == [], "分析失败不许上传"
    view.refresh_tasks()
    assert row_of(view, "a.mp4")[7] in ("分析失败", "未处理")


def test_finished_videos_are_skipped_before_anything(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video, vid = video_row(cfg, db, "done.mp4")
    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "done.txt")
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "done_成品.mp4")
    window, view = wired(cfg, db)
    window.on_auto_clip()
    assert window.calls["load_video"] == 0, "已有成品的不许再打开"
    assert window.calls["on_analyze"] == 0, "已有成品的不许再分析"
    assert window.bridge.tasks == [], "已有成品的不许再发 AI"
    assert int(db.value("SELECT COUNT(*) FROM ai_tasks") or 0) == 0, "根本不该排队"


def test_blocked_language_video_never_enters_the_queue(tmp_path: Path) -> None:
    """上次被语言拦下的视频（videos.blocked_language 非空）连队都不排。"""
    cfg, db = make_project(tmp_path)
    prm(cfg)
    fake_video(cfg, "id.mp4")
    fake_video(cfg, "en.mp4")
    bad_id = db_repo.upsert_video(db, cfg.root / "ai_in" / "id.mp4")
    db_repo.set_blocked_language(db, bad_id, "id")
    window, _view = wired(cfg, db)
    window.on_auto_clip()
    queued = sorted(str(row["file_name"]) for row in db.all(
        "SELECT v.file_name FROM ai_tasks t JOIN videos v ON v.id = t.video_id"))
    assert queued == ["en.mp4"], f"被语言拦下的视频不许排队：{queued}"
    assert any("不再排队" in line for line in window.logs), "得说清楚为什么少了一条"


def test_blocked_language_task_is_cancelled_not_retried(tmp_path: Path) -> None:
    """排队之后才被拦下的：领到手也不跑，直接判 cancelled（不重试、不发 AI）。"""
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video, vid = video_row(cfg, db, "id.mp4")
    window, _view = wired(cfg, db)
    created, _reused, _already, _off = window._enqueue_auto_tasks([video], "full")
    assert created == 1, "先得有一条在队里"
    task_id = int(db.value("SELECT id FROM ai_tasks") or 0)
    db_repo.set_blocked_language(db, vid, "id")   # 上一轮排完队之后才判出来
    window._auto_job = "full"
    window._auto_total = 1
    window._auto_step()
    row = db.one("SELECT status FROM ai_tasks WHERE id = ?", (task_id,))
    assert row["status"] == "cancelled", f"语言不符要落 cancelled（不重试），现在是 {row['status']}"
    assert window.calls["on_analyze"] == 0, "不许再分析"
    assert window.calls["load_video"] == 0, "连打开都不必"
    assert window.bridge.tasks == [], "一个字节都不许发 AI"


# ------------------------------------------------------------------ T26-T27 界面/日志
def test_panel_has_two_progress_bars(tmp_path: Path) -> None:
    """单条视频一条、队列一条，而且单条那条排在队列那条**上边**。"""
    cfg, _db = make_project(tmp_path)
    view = panel(cfg)
    layout = view.bar.parentWidget().layout()
    assert layout.indexOf(view.bar_video) < layout.indexOf(view.bar), \
        "单条视频进度条必须在队列进度条上边"
    view.set_video_progress(0.42, "语音识别")
    assert view.bar_video.value() == 42
    assert "语音识别" in view.bar_video.format()
    view.set_queue_progress(1, 4)
    assert view.bar.value() == 25 and "1 / 4" in view.bar.format()
    view.set_active("a", "分析")
    assert view.bar_video.value() == 0, "换视频 / 换步骤时单条那条要从头开始"


def test_panel_font_is_one_point_smaller(tmp_path: Path) -> None:
    """面板整体字号比默认小一档（这一屏塞了统计 + 任务表 + 日志）。"""
    cfg, _db = make_project(tmp_path)
    view = panel(cfg)
    default = app().font().pointSize()
    assert view.font().pointSize() == max(7, default - 1), \
        f"面板字号应该是 {max(7, default - 1)}，现在是 {view.font().pointSize()}"
    assert "font-size: 20px" not in PANEL_SRC, "头号数字别再用 20px 那么大"


def test_single_video_progress_reaches_the_panel(tmp_path: Path) -> None:
    """本地分析 / 渲染的进度推给面板上面那条（队列那条不动）。"""
    cfg, db = make_project(tmp_path)
    window, view = wired(cfg, db)
    view.set_queue_progress(2, 5)
    window._set_video_progress(0.5, "语音识别 3/6")
    assert view.bar_video.value() == 50 and "语音识别" in view.bar_video.format()
    assert view.bar.value() == 40, "单条那条动，队列那条不许被带着动"


def test_every_log_line_reaches_the_panel(tmp_path: Path) -> None:
    """主界面每一行日志都转播进面板：子进程输出、渲染、AI 对接全都要能看到。"""
    cfg, db = make_project(tmp_path)
    _window, view = wired(cfg, db)

    class Host:
        """借主窗口那一个 append_log 来测转发规则，不建整个主界面。"""

        append_log = mw.MainWindow.append_log

        def __init__(self, target):
            self.log_view = QPlainTextEdit()
            self.ai_panel = target

    host = Host(view)
    host.append_log("[分析] 语音识别｜第 3/6 段（62%）")
    host.append_log("$ python run.py highlight")
    host.append_log("[自动剪辑] 任务 #1 a.mp4")
    text = view.view_log.toPlainText()
    for needed in ("语音识别", "run.py highlight", "任务 #1"):
        assert needed in text, f"面板日志漏了「{needed}」：{text}"


def test_overview_keeps_only_the_seven_headline_numbers(tmp_path: Path) -> None:
    """任务总览只剩七个头号数字：底下那排九格（总视频 / 待剪辑 / 已完成…）已经撤了。"""
    cfg, _db = make_project(tmp_path)
    view = panel(cfg)
    assert not hasattr(view, "_stat_labels"), "九格的标签还留着，说明没真删"
    assert set(view._head_labels) == {"total", "analysed", "script", "attempted",
                                      "json", "rendered", "made"}
    box = next((child for child in view.findChildren(QGroupBox)
                if child.title() == "自动剪辑任务总览"), None)
    assert box is not None, "找不到「自动剪辑任务总览」"
    titles = {label.text() for label in box.findChildren(QLabel)}
    for gone in ("总视频", "待剪辑", "已完成", "等待 AI", "剪辑中", "已取消",
                 "未获取 JSON", "失败"):
        assert gone not in titles, f"总览里不该再有「{gone}」这一格"
    for kept in ("总任务", "已分析", "已有剧本", "已分析高光", "已获取 JSON",
                 "已剪辑", "成品"):
        assert kept in titles, f"总览里少了「{kept}」"


def test_clear_foreign_videos_removes_files_and_rows(tmp_path: Path) -> None:
    """「清空非中英视频」＝只清语言预检拦下的那些：原视频 + 附带文件 + 库里记录。

    「跳过」的原因很多，这里只认 `videos.blocked_language`；
    没有非中英视频时按钮是灰的；自动剪辑在跑时拒绝动手；中英视频一律不碰。
    """
    quiet()
    cfg, db = make_project(tmp_path)
    foreign, vid = video_row(cfg, db, "id.mp4")
    other, other_id = video_row(cfg, db, "ko.mp4")
    keep, keep_id = video_row(cfg, db, "en.mp4")
    merged = cfg.root / "ai_in" / "id_merged.txt"
    merged.write_text("x", encoding="utf-8")
    artifact(db, vid, "merged_txt", merged)
    keep_txt = cfg.root / "ai_in" / "en_merged.txt"
    keep_txt.write_text("x", encoding="utf-8")
    artifact(db, keep_id, "merged_txt", keep_txt)
    db_assets.create_asset(db, vid, {"video": "id.mp4",
                                     "clip": {"start": 1.0, "end": 4.0, "score": 0.9}})
    view = panel(cfg)
    view.refresh_tasks(sync=True)
    assert view.btn_clear_video.text() == "清空非中英视频"
    assert view.btn_clear_video.isEnabled() is False, "没有非中英视频时按钮该是灰的"

    db_repo.set_blocked_language(db, vid, "id")
    db_repo.set_blocked_language(db, other_id, "ko")
    view.refresh_tasks()
    assert view.btn_clear_video.isEnabled() is True, "有非中英视频就该能点"
    assert {path.name for _vid, path, _code in view._foreign_videos()} == {"id.mp4", "ko.mp4"}

    # 自动剪辑在跑的时候拒绝动手
    view._window = type("Running", (), {"auto_running": staticmethod(lambda: True)})()
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    view.on_clear_video()
    assert foreign.is_file(), "自动剪辑正在跑时不许删文件"

    view._window = None
    view.on_clear_video()
    assert not foreign.exists() and not other.exists(), "非中英的原视频都该删掉"
    assert not merged.exists(), "它自己产出的附带文件也该跟着删"
    assert keep.is_file() and keep_txt.is_file(), "中英视频和它的文件一个都不许动"
    assert db_repo.get_video_by_path(db, foreign) is None, "库里这条视频登记应该没了"
    assert db_assets.list_assets(db, vid) == [], "它的高光 JSON 也该跟着清掉"
    kept = db_repo.get_video_by_path(db, keep)
    assert kept is not None and int(kept["id"]) == keep_id, "别人的登记不许动"
    names = {view.table.item(line, 0).text() for line in range(view.table.rowCount())}
    assert names == {"en.mp4"}, f"列表没刷新对：{names}"
    assert view.btn_clear_video.isEnabled() is False, "清完就没什么可清的了，按钮该灰回去"


def test_done_is_a_database_fact_not_a_file_guess(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "a.mp4")
    window = Win(cfg, db)
    window._auto_video = video
    stray = cfg.root / "ai_out" / "a_成品.mp4"
    stray.write_text("x" * 64, encoding="utf-8")
    assert window._auto_done_file(video) is None, "光有文件不算干完"
    assert window._auto_chain_done(video) is False, "光有文件不算干完"
    assert window._auto_product_ready() is False
    db_repo.register_artifact(db, vid, "final_video", stray)
    assert same(window._auto_done_file(video), stray), "登记进库才算"
    assert window._auto_chain_done(video) is True
    assert window._auto_product_ready() is True


def test_task_carries_mp4_and_txt(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prompt = prm(cfg)
    video, vid = video_row(cfg, db, "a.mp4")
    txt = cfg.root / "ai_in" / "a.txt"
    artifact(db, vid, "merged_txt", txt)
    window, view = wired(cfg, db)
    window.on_auto_clip()
    assert len(window.bridge.tasks) == 1
    _kind, payload, files = window.bridge.tasks[0]
    assert payload["video"] == "a.mp4" and payload["text"] == "a.txt", "任务得说清是哪个 MP4 的 TXT"
    names = {p.name for p in files}
    assert names == {"prompt.txt", "a.txt"}, \
        f"只上传 PRM（统一改名 prompt.txt）+ 这个视频的 TXT：{sorted(names)}"
    assert "a.mp4" not in names, "视频本体不上传给 AI"
    assert any(same(p, txt) for p in files)
    sent = next(p for p in files if p.name == "prompt.txt")
    assert sent.read_text(encoding="utf-8") == prompt.read_text(encoding="utf-8"), \
        "prompt.txt 里装的就是这份 PRM 的正文"


def test_every_enabled_prm_goes_out(tmp_path: Path) -> None:
    """两份 PRM 都是「使用中」→ 附件里两份都在，名字统一成 prompt.txt / prompt_2.txt。"""
    cfg, db = make_project(tmp_path)
    en = prm(cfg)
    zh = cfg.root / "prm" / "prm_zh.txt"
    zh.write_text("中文规则", encoding="utf-8")
    db_assets.create_prm(db, "PRM 英文", str(en))
    db_assets.create_prm(db, "PRM 中文", str(zh))
    _video, vid = video_row(cfg, db, "a.mp4")
    txt = cfg.root / "ai_in" / "a.txt"
    artifact(db, vid, "merged_txt", txt)
    window, _view = wired(cfg, db)
    window.on_auto_clip()
    assert len(window.bridge.tasks) == 1, "两份 PRM 仍旧只发一个任务"
    _kind, _payload, files = window.bridge.tasks[0]
    names = {p.name for p in files}
    assert names == {"prompt.txt", "prompt_2.txt", "a.txt"}, \
        f"使用中的每一份 PRM 都要带上（统一命名）：{sorted(names)}"
    texts = {p.read_text(encoding="utf-8") for p in files if p.name.startswith("prompt")}
    assert texts == {en.read_text(encoding="utf-8"), "中文规则"}, \
        f"两份正文都要发出去：{texts}"


def test_disabled_prm_is_never_sent(tmp_path: Path) -> None:
    """停用的一份都不发；全停用就这一条不发 AI（也不算失败，视频跳过）。"""
    cfg, db = make_project(tmp_path)
    en = prm(cfg)
    zh = cfg.root / "prm" / "prm_zh.txt"
    zh.write_text("中文规则", encoding="utf-8")
    en_id = db_assets.create_prm(db, "PRM 英文", str(en))
    zh_id = db_assets.create_prm(db, "PRM 中文", str(zh))
    db_assets.set_prm_enabled(db, zh_id, False)
    _video, vid = video_row(cfg, db, "a.mp4")
    txt = cfg.root / "ai_in" / "a.txt"
    artifact(db, vid, "merged_txt", txt)
    window, _view = wired(cfg, db)
    window.on_auto_clip()
    _kind, _payload, files = window.bridge.tasks[0]
    assert {p.name for p in files} == {"prompt.txt", "a.txt"}, "停用的那一份不许发出去"
    sent = next(p for p in files if p.name == "prompt.txt")
    assert sent.read_text(encoding="utf-8") == en.read_text(encoding="utf-8"), \
        "发出去的正文得是启用那一份的"

    # 再把最后一份也停掉：这一条连附件都凑不出来，dispatch 直接拒发并说清原因
    db_assets.set_prm_enabled(db, en_id, False)
    assert window.resolve_prompt_files() == [], "全停用就该一份提示词都不给"
    window.bridge.tasks.clear()
    window.logs.clear()
    assert window.dispatch_ai([], txt, 0, video=_video) is False, "一份都没启用就不许发 AI"
    assert window.bridge.tasks == [], "拒发就不许往 bridge 里塞任务"
    assert any("没有使用中的 PRM" in line for line in window.logs), \
        f"不发也要在日志里说清为什么：{window.logs[-3:]}"


def test_never_sends_half_a_pair(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prompt = prm(cfg)
    video, vid = video_row(cfg, db, "a.mp4")
    txt = cfg.root / "ai_in" / "a.txt"
    txt.write_text("merged", encoding="utf-8")
    window = Win(cfg, db)
    window._auto_video = video
    video.unlink()                                    # MP4 没了：对象都不在了，别发
    assert window.dispatch_ai(prompt, txt, 0, video=video) is False
    assert window.bridge.tasks == [], "MP4 不在盘上就不发"
    video.write_bytes(b"again" + bytes(range(256)))
    txt.unlink()                                      # 换成 TXT 没了
    assert window.dispatch_ai(prompt, txt, 0, video=video) is False
    assert window.bridge.tasks == [], "TXT 不在盘上也不发（先分析生成）"


# ------------------------------------------------------------------ T19-T21 模式 / 停止
def test_collect_only_saves_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    cfg.bridge["ai_job"] = "collect"
    video, vid = video_row(cfg, db, "a.mp4")
    window = Win(cfg, db)
    window._auto_job = "collect"
    window._auto_video = video
    window._last_highlight_json = json.dumps({"clip": {"start": 1, "end": 5}})
    assert window._auto_chain_done(video) is False, "还没入库就不算干完"
    assert window._auto_save_script() is True
    target = cfg.root / "ai_out" / "a_脚本.json"
    assert target.is_file(), "导出的高光 JSON 落在 AI_输出目录"
    assert window._auto_chain_done(video) is False, "**光有文件不算干完**，得库里有可复用 JSON"
    ai_json(db, vid)
    assert window._auto_chain_done(video) is True, "库里有可复用高光 JSON 才算干完"
    assert window.calls["run_highlight"] == 0, "收取高光 JSON 不许渲染"


def test_script_mode_never_calls_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    cfg.bridge["ai_job"] = "script"
    video, vid = video_row(cfg, db, "a.mp4")
    script = cfg.root / "ai_in" / "a_脚本.json"
    script.write_text(json.dumps({"clip": {"start": 1.0, "end": 6.0}}), encoding="utf-8")
    db_repo.register_artifact(db, vid, "ai_script", script)
    window, view = wired(cfg, db)
    before = counts(db)
    window.on_auto_clip()
    assert window.calls["run_highlight"] == 1, "脚本剪辑该直接开剪"
    assert window.bridge.tasks == [], "脚本剪辑一次 AI 都不许调"
    assert counts(db)[1] == before[1], "ai_results 不许多一行"


def test_stop_keeps_the_queue_for_next_time(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm(cfg)
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        video_row(cfg, db, name)
    window, view = wired(cfg, db)
    window.on_auto_clip()
    open_before = db_repo.queue_counts(db, mode="full")["open"]
    assert open_before == 3
    window.on_bridge_stop()
    assert window._auto_stop is True
    after = db_repo.queue_counts(db, mode="full")
    assert after["cancelled"] == 0, "「停止」不许把排着的任务作废"
    assert after["pending"] == 3, "手上那条要退回 pending，下次接着跑"
    calls = dict(window.calls)
    window._auto_step()                       # 停了之后不许再领
    assert window.calls == calls
    window.on_auto_clip()                     # 下一轮照旧接得上
    assert window._auto_stop is False
    assert window._auto_task_id is not None


# ------------------------------------------------------------------ T22-T24 边界
def test_panel_is_only_a_scheduler(tmp_path: Path) -> None:
    for banned in ("plan_clips(", "HighlightWorker", "RenderDialog", "ffmpeg",
                   "subprocess", "AnalyzeWorker"):
        assert banned not in PANEL_SRC, f"面板里不许出现 {banned}（它只是调度器）"
    assert "exec_()" not in PANEL_SRC, "面板不许阻塞 GUI 线程"


def test_panel_makes_no_ai_calls(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _v, vid = video_row(cfg, db, "a.mp4")
    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "a.txt")
    before = counts(db)
    view = panel(cfg)
    view.refresh_tasks(sync=True)
    view.set_scope("collect_missing")
    view.set_scope("clip_existing")
    touched(view)
    assert counts(db) == before, "光看面板不该产生任何 AI 任务/结果"


def test_panel_has_no_raw_sql(tmp_path: Path) -> None:
    for banned in ("SELECT ", "INSERT ", "UPDATE ", "DELETE "):
        assert banned not in PANEL_SRC, f"面板里不许写裸 SQL（{banned.strip()}）"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_ai_dirs_have_their_own_keys,
    test_dirs_save_without_a_button,
    test_scope_saves_job_and_source_together,
    test_skip_done_products_saves_live,
    test_unchecking_skip_reruns_finished_videos,
    test_settings_come_back_after_restart,
    test_old_and_new_config_shapes_both_read,
    test_no_mandatory_save_button,
    test_scan_keeps_every_mp4,
    test_three_statuses_are_independent,
    test_status_words_cover_every_stage,
    test_headline_numbers,
    test_stats_follow_the_run_without_a_refresh_click,
    test_existing_txt_skips_analysis,
    test_missing_txt_analyses_first,
    test_analysis_failure_stops_this_one,
    test_finished_videos_are_skipped_before_anything,
    test_blocked_language_video_never_enters_the_queue,
    test_blocked_language_task_is_cancelled_not_retried,
    test_panel_has_two_progress_bars,
    test_panel_font_is_one_point_smaller,
    test_single_video_progress_reaches_the_panel,
    test_every_log_line_reaches_the_panel,
    test_overview_keeps_only_the_seven_headline_numbers,
    test_clear_foreign_videos_removes_files_and_rows,
    test_done_is_a_database_fact_not_a_file_guess,
    test_task_carries_mp4_and_txt,
    test_every_enabled_prm_goes_out,
    test_disabled_prm_is_never_sent,
    test_never_sends_half_a_pair,
    test_collect_only_saves_json,
    test_script_mode_never_calls_ai,
    test_stop_keeps_the_queue_for_next_time,
    test_panel_is_only_a_scheduler,
    test_panel_makes_no_ai_calls,
    test_panel_has_no_raw_sql,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="aipanel_"))
        try:
            fn(work)
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
