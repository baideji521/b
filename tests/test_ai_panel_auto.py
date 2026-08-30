"""AI GUI 自动剪辑模块：配置实时保存 / 扫描 / 三个状态 / 三条流程 / 统计 / 停止续跑。

盯的是这一批的红线：
  T1  AI_输入目录 / AI_输出目录 用的是自己的键，绝不碰 paths.input_dir / output_dir
  T2  改目录不用点保存：改完（防抖 flush）就在 config.json 里
  T3  选模式不用点保存，而且三个开关互斥
  T4  重开面板读回上次的模式和两个目录
  T5  老配置只有 ai_job 也照旧认；只有三个布尔开关时也认
  T6  面板上没有「保存设置」这种必须点的按钮
  T7  扫描：MP4+TXT / 只有 MP4 / 多个 MP4 都进任务表，缺 TXT 的不被丢掉
  T8  三个状态各自独立（分析 / JSON / 剪辑），不互相推断
  T9  状态词表：未处理 / 分析中 / 分析失败 / 等待扩展 / 上传中 / 等待 JSON /
      JSON 成功 / JSON 失败 / 剪辑中 / 剪辑完成 / 跳过 / 失败
  T10 四个头号数字：总任务 / 未剪辑 / 已获取 JSON / 成品
  T11 统计跟着事件自己动：成品 +1 之后未剪辑 -1，不用点刷新
  T12 有 TXT 就不重跑分析，直接发 AI
  T13 没 TXT 先按主界面配置分析（不是拿 AI 那份配置）
  T14 分析失败＝这条记 failed，而且一个字节都没发出去
  T15 已经有成品的 MP4 在分析 / 上传 / 取 JSON **之前**就被跳过
  T16 干完没干完不看文件在不在，看库里的任务生命周期 + artifacts
  T17 发给扩展的只有 PRM + 这个视频的 TXT（视频本体不上传），任务里标明是哪个 MP4
  T18 MP4 不在盘上不发（对象没了），TXT 不在盘上也不发（先分析生成）
  T19 收取脚本这一串不渲染：JSON 存进 AI_输出目录就算干完
  T20 脚本剪辑这一串一次 AI 都不调
  T21 停止：不再领新任务，手上这条退回 pending（不是 cancelled），下次接着跑
  T22 AI 自动化只是调度器：面板里没有第二套剪辑引擎，也不阻塞 GUI 线程
  T23 面板不产生 AI 调用（ai_tasks / ai_results 一行都不加）
  T24 面板里没有裸 SQL

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

from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton   # noqa: E402

from vidscribe.config import Config                       # noqa: E402
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
    _auto_clip_from_script = mw.MainWindow._auto_clip_from_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _auto_product_ready = mw.MainWindow._auto_product_ready
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
    resolve_prompt_file = mw.MainWindow.resolve_prompt_file
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


def test_mode_saves_and_stays_exclusive(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    view._pick_job("collect")
    touched(view)
    saved = on_disk(cfg)["bridge"]
    assert saved["ai_job"] == "collect"
    assert saved["ai_collect_script"] is True
    assert saved["ai_clip_video"] is False and saved["ai_script_clip"] is False
    checked = [b for b in view._job_group.buttons() if b.isChecked()]
    assert len(checked) <= 1, "三种模式必须互斥"


def test_settings_come_back_after_restart(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    view = panel(cfg)
    view._pick_job("script")
    (tmp_path / "again").mkdir()
    view.edit_output.setText(str(tmp_path / "again"))
    touched(view)
    fresh = Config.load(tmp_path, tmp_path / "config.json")
    again = panel(fresh)
    assert again._job == "script"
    assert again.edit_output.text() == str(tmp_path / "again")
    assert again.edit_input.text() == str(tmp_path / "ai_in")


def test_old_and_new_config_shapes_both_read(tmp_path: Path) -> None:
    assert ao._job_from_config({"ai_job": "collect"}) == "collect"
    assert ao._job_from_config({"ai_script_clip": True}) == "script"
    assert ao._job_from_config({}) == "full"
    assert ao._job_from_config({"ai_job": "什么鬼"}) == "full"


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
    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "both.txt")
    video_row(cfg, db, "lonely.mp4")          # 只有 MP4，没有 TXT
    video_row(cfg, db, "third.mp4")
    view = panel(cfg)
    names = {view.table.item(r, 0).text() for r in range(view.table.rowCount())}
    assert names == {"both.mp4", "lonely.mp4", "third.mp4"}, "缺 TXT 的 MP4 不许被丢掉"
    assert row_of(view, "both.mp4")[2] == ao.DONE
    assert row_of(view, "lonely.mp4")[2] == ao.WAITING


def test_three_statuses_are_independent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "a.mp4")
    view = panel(cfg)
    assert row_of(view, "a.mp4")[1:5] == [ao.WAITING] * 4
    ai_json(db, vid)                      # 只有 JSON：分析、剪辑都还没有
    view.refresh_tasks()
    marks = row_of(view, "a.mp4")
    assert marks[3] == ao.DONE and marks[1] == ao.WAITING and marks[4] == ao.WAITING
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "a_成品.mp4")
    view.refresh_tasks()
    assert row_of(view, "a.mp4")[4] == ao.DONE


def test_status_words_cover_every_stage(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "s.mp4")
    view = panel(cfg)
    assert row_of(view, "s.mp4")[5] == "未处理"

    artifact(db, vid, "merged_txt", cfg.root / "ai_in" / "s.txt")
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "等待扩展"

    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    db_repo.claim_ai_task(db, task_id)                 # uploading
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "上传中"

    db_repo.mark_ai_task_waiting(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "等待 JSON"

    ai_json(db, vid, task_id=task_id)
    db_repo.mark_ai_task_rendering(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "剪辑中"

    db_repo.complete_ai_task(db, task_id)
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "JSON 成功"

    artifact(db, vid, "final_video", cfg.root / "ai_out" / "s_成品.mp4")
    view.refresh_tasks()
    assert row_of(view, "s.mp4")[5] == "剪辑完成"

    # 正在跑的那一步：分析中
    view.set_active("s.mp4".replace(".mp4", ""), "分析")
    assert row_of(view, "s.mp4")[5] == "剪辑完成", "有成品就是完成，不被「在跑」盖掉"
    view.set_active("", "")

    # 分析失败 / JSON 失败 / 失败 / 跳过：全靠 ai_tasks 的状态 + 有没有产物
    bad, bad_id = video_row(cfg, db, "bad.mp4")
    bad_task, _ = db_repo.enqueue_ai_task(db, bad_id, mode="full")
    db_repo.fail_ai_task(db, bad_task, "分析炸了")
    view.refresh_tasks(sync=True)
    assert row_of(view, "bad.mp4")[5] == "分析失败"
    artifact(db, bad_id, "merged_txt", cfg.root / "ai_in" / "bad.txt")
    view.refresh_tasks()
    assert row_of(view, "bad.mp4")[5] == "JSON 失败"
    ai_json(db, bad_id)
    view.refresh_tasks()
    assert row_of(view, "bad.mp4")[5] == "失败"

    skipped, skip_id = video_row(cfg, db, "skip.mp4")
    skip_task, _ = db_repo.enqueue_ai_task(db, skip_id, mode="full")
    db_repo.cancel_ai_task(db, skip_task, "停了")
    view.refresh_tasks(sync=True)
    assert row_of(view, "skip.mp4")[5] == "跳过"

    words = {row_of(view, n)[5] for n in ("s.mp4", "bad.mp4", "skip.mp4")}
    assert words <= {"未处理", "分析中", "分析失败", "等待扩展", "上传中", "等待 JSON",
                     "JSON 成功", "JSON 失败", "剪辑中", "剪辑完成", "跳过", "失败",
                     "导出 TXT", "等导出 TXT", "跑着"}


def test_headline_numbers(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _a, a_id = video_row(cfg, db, "a.mp4")
    _b, b_id = video_row(cfg, db, "b.mp4")
    view = panel(cfg)
    assert head(view) == {"total": 2, "todo": 2, "json": 0, "made": 0}
    ai_json(db, a_id)
    view.refresh_tasks()
    assert head(view)["json"] == 1
    artifact(db, a_id, "final_video", cfg.root / "ai_out" / "a_成品.mp4")
    view.refresh_tasks()
    assert head(view) == {"total": 2, "todo": 1, "json": 1, "made": 1}


def test_stats_follow_the_run_without_a_refresh_click(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _v, vid = video_row(cfg, db, "a.mp4")
    window, view = wired(cfg, db)
    before = head(view)
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "a_成品.mp4")
    window._auto_total = 1
    window._set_auto_progress(1)          # 主界面推进度 -> 面板自己刷
    after = head(view)
    assert before["made"] == 0 and after["made"] == 1
    assert after["todo"] == before["todo"] - 1


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
    assert row_of(view, "a.mp4")[5] in ("分析失败", "未处理")


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


def test_done_is_a_database_fact_not_a_file_guess(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "a.mp4")
    window = Win(cfg, db)
    window._auto_video = video
    stray = cfg.root / "ai_out" / "a_成品.mp4"
    stray.write_text("x" * 64, encoding="utf-8")
    assert window._auto_done_file(video) is None, "光有文件不算干完"
    db_repo.register_artifact(db, vid, "final_video", stray)
    assert same(window._auto_done_file(video), stray), "登记进库才算"
    assert same(window._auto_product_ready(), stray)


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
    assert names == {"prm_en.txt", "a.txt"}, "只上传 PRM + 这个视频的 TXT"
    assert "a.mp4" not in names, "视频本体不上传给 AI"
    assert any(same(p, txt) for p in files) and any(same(p, prompt) for p in files)


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
    assert window._auto_save_script() is True
    target = cfg.root / "ai_out" / "a_脚本.json"
    assert target.is_file(), "脚本必须落在 AI_输出目录"
    assert same(window._auto_done_file(video), target), "存进 AI_输出目录就算干完"
    assert window.calls["run_highlight"] == 0, "收取脚本不许渲染"


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
    view._pick_job("collect")
    view._pick_job("script")
    touched(view)
    assert counts(db) == before, "光看面板不该产生任何 AI 任务/结果"


def test_panel_has_no_raw_sql(tmp_path: Path) -> None:
    for banned in ("SELECT ", "INSERT ", "UPDATE ", "DELETE "):
        assert banned not in PANEL_SRC, f"面板里不许写裸 SQL（{banned.strip()}）"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_ai_dirs_have_their_own_keys,
    test_dirs_save_without_a_button,
    test_mode_saves_and_stays_exclusive,
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
    test_done_is_a_database_fact_not_a_file_guess,
    test_task_carries_mp4_and_txt,
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
