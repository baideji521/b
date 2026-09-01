"""业务链收口：MP4 → 本地分析 → 完整剧本 → PRM + 完整剧本 → 扩展 AI → 高光 JSON → 剪辑 → 高光片段。

盯的是这条链上不许再含糊的九条红线：
  B1 绝不把视频发给 AI：附件里出现任何视频文件立刻拒发，bridge.submit 一次都不许调
  B2 库里的高光 JSON 是权威：把盘上的 _脚本.json 删干净，三种模式照样认得出"已有 JSON"
  B3 库里的分析结果是权威：把 output/ 和 cache/ 删干净，照样不重跑分析、直接出剧本
  B4 库里什么都没有 -> 先本地分析（on_analyze），一个字节都不发
  B5 库里已有可复用高光 JSON -> 0 次 AI 调用 + 1 次渲染
  B6 已经有成品 -> 0 次 AI + 0 次渲染，连队列都不排
  B7 三种模式行为互不相同：full 发 AI / collect 入库即完成 / script 无 JSON 直接失败且不问 AI
  B8 用户自己的 TXT 不许被覆盖：默认只写 cache 临时件，任务结束删掉
  B9 面板的列和头号数字全部来自数据库：盘上多几个文件不改变任何一列

全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_business_chain.py`。
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

from PyQt5.QtWidgets import QApplication, QMessageBox   # noqa: E402

from vidscribe.config import Config                     # noqa: E402
from vidscribe.db import open_db                        # noqa: E402
from vidscribe.db import repo as db_repo                # noqa: E402
from vidscribe.gui import ai_options as ao              # noqa: E402
from vidscribe.gui import main_window as mw             # noqa: E402

_APP = None


def app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
    for sub in ("database", "input", "output", "logs", "cache", "prm", "ai_in", "ai_out"):
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
    bridge["keep_merged_file"] = False
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str = "a.mp4") -> Path:
    path = cfg.root / "ai_in" / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def prm(cfg) -> Path:
    path = cfg.root / "prm" / "prm_en.txt"
    path.write_text("rules", encoding="utf-8")
    cfg.bridge["prompt_file"] = str(path)
    return path


def video_row(cfg, db, name: str = "a.mp4") -> tuple[Path, int]:
    video = fake_video(cfg, name)
    return video, db_repo.upsert_video(db, video, info={"duration": 30.0})


def analysis(db, vid: int) -> int:
    """一条跑成功的分析：有画面事件也有语音段，够生成完整剧本。"""
    sig = {"vision_model": "m", "vision_config": None, "vision_config_hash": "h",
           "asr_model": "a", "asr_config": None, "asr_config_hash": "h"}
    aid = db_repo.create_analysis(db, vid, sig)
    db_repo.save_visual_events(db, aid, [
        {"id": 1, "start": 0.0, "end": 6.0, "event": "", "description": "有人走进厨房",
         "confidence": 0.8, "importance": "normal", "timestamp_source": "frame_based"},
        {"id": 2, "start": 6.0, "end": 14.0, "event": "", "description": "他打翻了杯子",
         "confidence": 0.9, "importance": "high", "timestamp_source": "frame_based"}])
    db_repo.save_speech_segments(db, aid, [
        {"id": 1, "start": 1.0, "end": 4.4, "text": "我先去拿个杯子",
         "confidence": 0.9, "language": "zh",
         "words": [{"word": "我先", "start": 1.0, "end": 2.0, "probability": 0.9}]}])
    db_repo.note_render(db, aid, output_language="zh",
                        render_config={"min_overlap_seconds": 0.2,
                                       "importance_filter": "low",
                                       "confidence_filter": 0.0},
                        face_available=False)
    db_repo.finish_analysis(db, aid, scene_count=2, speech_count=1)
    return aid


HIGHLIGHT_JSON = {"video": "a.mp4",
                  "clip": {"start": 2.0, "end": 9.0, "score": 0.9,
                           "type": "hook", "reason": "打翻杯子"}}


def ai_json(db, vid: int, *, task_id: int | None = None) -> None:
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=HIGHLIGHT_JSON,
                           raw_response=json.dumps(HIGHLIGHT_JSON))


def artifact(db, vid: int, kind: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x" * 64, encoding="utf-8")
    return db_repo.register_artifact(db, vid, kind, path)


def counts(db) -> tuple[int, int]:
    return (int(db.value("SELECT COUNT(*) FROM ai_tasks") or 0),
            int(db.value("SELECT COUNT(*) FROM ai_results") or 0))


class FakeBridge:
    """只记账的 Bridge 替身。submit 被调到就说明"真的发出去了"。"""

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


class Win:
    """跑完整业务链需要的主窗口替身：判定/发送/导出全用真方法，只有叶子活儿计数。"""

    on_auto_clip = mw.MainWindow.on_auto_clip
    _auto_step = mw.MainWindow._auto_step
    _enqueue_auto_tasks = mw.MainWindow._enqueue_auto_tasks
    _source_allows = mw.MainWindow._source_allows
    _resume_existing_ai_json = mw.MainWindow._resume_existing_ai_json
    _asset_json_for_render = mw.MainWindow._asset_json_for_render
    _reusable_highlight_json = mw.MainWindow._reusable_highlight_json
    _auto_chain_done = mw.MainWindow._auto_chain_done
    _skip_because_done = mw.MainWindow._skip_because_done
    skip_done_products = mw.MainWindow.skip_done_products
    _language_blocked = mw.MainWindow._language_blocked
    _auto_done_file = mw.MainWindow._auto_done_file
    _auto_product_ready = mw.MainWindow._auto_product_ready
    _auto_after_analyze = mw.MainWindow._auto_after_analyze
    _auto_save_script = mw.MainWindow._auto_save_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_advance = mw.MainWindow._auto_advance
    _auto_finish = mw.MainWindow._auto_finish
    _settle_auto_task = mw.MainWindow._settle_auto_task
    _mark_auto_waiting = mw.MainWindow._mark_auto_waiting
    _mark_auto_rendering = mw.MainWindow._mark_auto_rendering
    _db_video_id = mw.MainWindow._db_video_id
    _register_artifact = mw.MainWindow._register_artifact
    script_payload = mw.MainWindow.script_payload
    write_script_text = mw.MainWindow.write_script_text
    # 剧本会问一句"这是不是合并视频"：自动链里的视频不是拼接的，直接给空表
    pieces_spans = lambda self, force=False: []  # noqa: E731
    write_ai_text = mw.MainWindow.write_ai_text
    _archive_script_txt = mw.MainWindow._archive_script_txt
    _ai_files_ok = mw.MainWindow._ai_files_ok
    dispatch_ai = mw.MainWindow.dispatch_ai
    send_file_to_ai = mw.MainWindow.send_file_to_ai
    clean_bridge_temp = mw.MainWindow.clean_bridge_temp
    highlight_source = mw.MainWindow.highlight_source
    selected_prm = mw.MainWindow.selected_prm
    enabled_prms = mw.MainWindow.enabled_prms
    has_prm_profiles = mw.MainWindow.has_prm_profiles
    resolve_prompt_files = mw.MainWindow.resolve_prompt_files
    resolve_prompt_file = mw.MainWindow.resolve_prompt_file
    _write_prompt_files = mw.MainWindow._write_prompt_files
    auto_running = mw.MainWindow.auto_running
    auto_busy = mw.MainWindow.auto_busy
    ai_dir = mw.MainWindow.ai_dir
    export_root = mw.MainWindow.export_root
    _sync_disk = mw.MainWindow._sync_disk
    _worker_id = mw.MainWindow._worker_id
    on_bridge_stop = mw.MainWindow.on_bridge_stop
    VIDEO_SUFFIXES = mw.MainWindow.VIDEO_SUFFIXES

    def __init__(self, cfg, db, job: str = "full"):
        self.cfg = cfg
        self._db_handle = db
        self._db_failed = False
        self._auto_job = job
        self.cfg.bridge["ai_job"] = job
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
        self._bridge_temp_files: list[Path] = []
        self.video_path = None
        self.export_dir = None
        self.worker = None
        self.clip_worker = None
        self.ai_worker = None
        self.bridge = FakeBridge()
        self.ai_panel = None
        self.speech = []
        self.timeline = []
        self.timeline_doc = {}
        self.show_translated = False
        self.logs: list[str] = []
        self.calls = {k: 0 for k in ("on_analyze", "run_highlight", "load_video")}

    # --- 叶子替身
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

    def export_language(self) -> str:
        return "zh"

    def output_dir(self):
        return None

    def refresh_bridge_label(self) -> None:
        pass

    def _set_auto_state(self, idle: bool, state: str = "") -> None:
        pass

    def _set_auto_step(self, stem: str, step: str) -> None:
        pass

    def _set_auto_progress(self, done: int) -> None:
        pass

    def _note_prompt_use(self, prompt_path) -> None:
        self._last_prompt = {"path": str(prompt_path)}


def quiet() -> None:
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)


def logged(window: Win, needle: str) -> bool:
    return any(needle in line for line in window.logs)


# ------------------------------------------------------------------ B1
def test_never_sends_a_video_to_ai(tmp_path: Path) -> None:
    """附件里出现视频就立刻拒发，而且 bridge.submit 一次都不许调。"""
    cfg, db = make_project(tmp_path)
    quiet()
    prompt = prm(cfg)
    video, vid = video_row(cfg, db)
    window = Win(cfg, db)
    window._auto_video = video
    window.video_path = video

    # 直接把 MP4 当"剧本"塞进去：硬闸门必须挡住
    assert window.dispatch_ai(prompt, video, 3, video=video) is False, "视频不许当附件发出去"
    assert window.bridge.tasks == [], "拒发就是一个字节都不发"
    assert logged(window, "拒绝发送 AI 任务：files 中检测到视频文件"), window.logs

    # 走 send_file_to_ai 这条老路也得挡住
    window.logs.clear()
    assert window.send_file_to_ai(video) is False
    assert window.bridge.tasks == [], "另一条入口也不许漏"
    assert logged(window, "拒绝发送 AI 任务：files 中检测到视频文件"), window.logs

    # 正常的两份 txt 才放行，而且附件里绝不含视频
    window.logs.clear()
    script = cfg.root / "cache" / "a_merged.txt"
    script.write_text("剧本", encoding="utf-8")
    assert window.dispatch_ai(prompt, script, 3, video=video) is True
    _kind, payload, files = window.bridge.tasks[0]
    assert {p.name for p in files} == {prompt.name, script.name}
    assert all(p.suffix == ".txt" for p in files), "附件只能是 .txt"
    assert payload["video"] == video.name, "任务里仍要标明是哪个 MP4 的"
    db.close()


# ------------------------------------------------------------------ B2
def test_db_json_survives_deleting_the_files(tmp_path: Path) -> None:
    """把盘上的 _脚本.json 删干净，三种模式照样认得出"库里已有高光 JSON"。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db)
    ai_json(db, vid)
    stray = cfg.root / "ai_out" / "a_脚本.json"
    stray.write_text(json.dumps(HIGHLIGHT_JSON), encoding="utf-8")
    stray.unlink()                                  # 文件没了，库里还在
    assert not stray.exists()

    for job in ("full", "collect", "script"):
        window = Win(cfg, db, job=job)
        window._auto_video = video
        window.video_path = video
        reusable = window._reusable_highlight_json(video)
        assert reusable is not None, f"{job}：库里那份高光 JSON 必须还认得出"
        assert db_repo.clips_from_payload(json.loads(reusable)), f"{job}：JSON 得能抠出片段"
    assert vid in db_repo.reusable_json_videos(db, [vid]), "队列判据也只查库"
    db.close()


# ------------------------------------------------------------------ B3
def test_db_analysis_survives_deleting_output_and_cache(tmp_path: Path) -> None:
    """删掉 output/ 和 cache/：照样能出完整剧本，也不许重跑分析。"""
    cfg, db = make_project(tmp_path)
    prm(cfg)
    video, vid = video_row(cfg, db)
    analysis(db, vid)
    shutil.rmtree(cfg.path("output_dir"), ignore_errors=True)
    shutil.rmtree(cfg.path("cache_dir"), ignore_errors=True)

    window = Win(cfg, db)
    window._auto_video = video
    window.video_path = video
    payload = window.script_payload(video)
    assert payload is not None and (payload["segments"] or payload["events"]), \
        "分析结果的权威来源是库，不是 output/cache"

    window._auto_after_analyze()
    assert window.calls["on_analyze"] == 0, "库里有结果就不许重跑分析"
    assert len(window.bridge.tasks) == 1, "剧本该由库现生成然后发出去"
    _kind, _payload, files = window.bridge.tasks[0]
    assert all(p.suffix == ".txt" for p in files) and len(files) == 2
    db.close()


# ------------------------------------------------------------------ B4
def test_no_analysis_means_analyse_first(tmp_path: Path) -> None:
    """库里什么都没有：先本地分析，一个字节都不发。"""
    cfg, db = make_project(tmp_path)
    quiet()
    prm(cfg)
    video_row(cfg, db)
    window = Win(cfg, db)
    window.on_auto_clip()
    assert window.calls["on_analyze"] == 1, "没分析结果就得先分析"
    assert window.calls["run_highlight"] == 0, "没素材不许开剪"
    assert window.bridge.tasks == [], "分析还没完，什么都不许发"
    db.close()


# ------------------------------------------------------------------ B5
def test_existing_json_renders_without_asking_ai(tmp_path: Path) -> None:
    """库里已有可复用高光 JSON：0 次 AI 调用 + 正好 1 次渲染。"""
    cfg, db = make_project(tmp_path)
    quiet()
    prm(cfg)
    video, vid = video_row(cfg, db)
    analysis(db, vid)
    ai_json(db, vid)
    before = counts(db)
    window = Win(cfg, db)
    window.on_auto_clip()
    assert window.bridge.tasks == [], "库里有 JSON 就一次 AI 都不许调"
    assert window.calls["run_highlight"] == 1, "该直接开剪"
    assert window.calls["on_analyze"] == 0, "更不许重跑分析"
    assert counts(db)[1] == before[1], "ai_results 不许多一行"
    db.close()


# ------------------------------------------------------------------ B6
def test_finished_video_costs_nothing(tmp_path: Path) -> None:
    """已经有成品：0 次 AI、0 次渲染，连队列都不排。"""
    cfg, db = make_project(tmp_path)
    quiet()
    prm(cfg)
    video, vid = video_row(cfg, db, "done.mp4")
    analysis(db, vid)
    ai_json(db, vid)
    artifact(db, vid, "final_video", cfg.root / "ai_out" / "done_高光时刻.mp4")
    window = Win(cfg, db)
    window.on_auto_clip()
    assert window.calls["load_video"] == 0, "有成品的不许再打开"
    assert window.calls["on_analyze"] == 0 and window.calls["run_highlight"] == 0
    assert window.bridge.tasks == [], "有成品的不许再发 AI"
    assert counts(db)[0] == 0, "根本不该排队"
    db.close()


# ------------------------------------------------------------------ B7
def test_three_modes_do_different_things(tmp_path: Path) -> None:
    """三种模式在同一份库状态下的行为必须互不相同。"""
    # full：库里有分析没 JSON -> 生成剧本发 AI，不渲染
    cfg, db = make_project(tmp_path)
    quiet()
    prm(cfg)
    video, vid = video_row(cfg, db)
    analysis(db, vid)
    full = Win(cfg, db, job="full")
    full.on_auto_clip()
    assert len(full.bridge.tasks) == 1 and full.calls["run_highlight"] == 0, "full 该去问 AI"

    # collect：库里已有可复用 JSON -> 直接算完成，不渲染、不问 AI
    ai_json(db, vid)
    collect = Win(cfg, db, job="collect")
    collect.on_auto_clip()
    assert collect.bridge.tasks == [], "collect 有 JSON 就不问 AI"
    assert collect.calls["run_highlight"] == 0, "collect 永远不渲染"
    assert collect._auto_chain_done(video) is True, "库里有可复用 JSON 就是干完了"

    # script：库里没有 JSON -> 记 failed，绝不问 AI
    cfg2, db2 = make_project(tmp_path / "second")
    prm(cfg2)
    video2, vid2 = video_row(cfg2, db2, "b.mp4")
    analysis(db2, vid2)
    script = Win(cfg2, db2, job="script")
    script.on_auto_clip()
    assert script.bridge.tasks == [], "script 一次 AI 都不许调"
    assert script.calls["run_highlight"] == 0, "没 JSON 就没得剪"
    assert logged(script, "库里没有高光 JSON"), script.logs
    row = db2.one("SELECT status FROM ai_tasks WHERE video_id = ?", (vid2,))
    assert row is not None and row["status"] in ("failed", "pending")
    db.close()
    db2.close()


# ------------------------------------------------------------------ B8
def test_user_txt_is_never_overwritten(tmp_path: Path) -> None:
    """剧本 TXT 只是传输载体：默认落 cache 并在收尾时删掉，用户自己的文件一个字节不动。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db)
    analysis(db, vid)
    mine = cfg.root / "ai_in" / "a.txt"
    mine.write_text("这是我自己写的，别动", encoding="utf-8")

    window = Win(cfg, db)
    window._auto_video = video
    window.video_path = video
    merged, count = window.write_ai_text()
    assert count > 0
    assert merged.parent.resolve() == cfg.path("cache_dir").resolve(), "默认只写 cache"
    assert mine.read_text(encoding="utf-8") == "这是我自己写的，别动", "用户的 TXT 不许被覆盖"
    assert db_repo.artifact_path(db, vid, "merged_txt") is None, "临时件不许登记成产物"
    assert merged in window._bridge_temp_files, "临时件要进待清理清单"
    window.clean_bridge_temp()
    assert not merged.exists(), "任务收尾要把临时剧本删掉"

    # 打开归档开关才另存一份，而且不覆盖同名文件
    cfg.bridge["keep_merged_file"] = True
    archived_window = Win(cfg, db)
    archived_window._auto_video = video
    archived_window.video_path = video
    merged2, _count = archived_window.write_ai_text()
    kept = db_repo.artifact_path(db, vid, "merged_txt")
    assert kept is not None and kept.is_file(), "归档那份才登记"
    assert mine.read_text(encoding="utf-8") == "这是我自己写的，别动", "归档也不许踩用户的文件"
    db.close()


# ------------------------------------------------------------------ B9
def test_panel_columns_come_from_the_database(tmp_path: Path) -> None:
    """面板的八列和七个头号数字全部来自库：盘上多几个文件不改变任何一列。"""
    cfg, db = make_project(tmp_path)
    app()
    video, vid = video_row(cfg, db)
    view = ao.AiPanel(cfg, None, log=lambda _line: None)
    assert [view.table.horizontalHeaderItem(c).text() for c in range(view.table.columnCount())] == \
        ["文件", "分析", "剧本", "高光分析", "高光 JSON", "剪辑", "成品", "状态"]
    assert set(view._head_labels) == {"total", "analysed", "script", "attempted",
                                      "json", "rendered", "made"}

    def marks() -> list[str]:
        for row in range(view.table.rowCount()):
            if view.table.item(row, 0).text() == video.name:
                return [view.table.item(row, c).text()
                        for c in range(view.table.columnCount())]
        raise AssertionError("任务表里没有这个视频")

    before = marks()
    # 盘上凭空多出剧本 TXT 和高光 JSON：一列都不许变
    (cfg.root / "ai_in" / "a.txt").write_text("看着像剧本", encoding="utf-8")
    (cfg.root / "ai_out" / "a_脚本.json").write_text("{}", encoding="utf-8")
    view.refresh_tasks(sync=True)
    assert marks() == before, "文件存在不是业务状态"

    # 库里真的有东西了，才该变
    analysis(db, vid)
    view.refresh_tasks()
    assert marks()[1] == ao.DONE and marks()[2] == ao.DONE
    ai_json(db, vid)
    view.refresh_tasks()
    assert marks()[3] == ao.DONE and marks()[4] == ao.DONE
    heads = {key: int(label.text()) for key, label in view._head_labels.items()}
    assert heads == {"total": 1, "analysed": 1, "script": 1, "attempted": 1,
                     "json": 1, "rendered": 0, "made": 0}
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_never_sends_a_video_to_ai,
    test_db_json_survives_deleting_the_files,
    test_db_analysis_survives_deleting_output_and_cache,
    test_no_analysis_means_analyse_first,
    test_existing_json_renders_without_asking_ai,
    test_finished_video_costs_nothing,
    test_three_modes_do_different_things,
    test_user_txt_is_never_overwritten,
    test_panel_columns_come_from_the_database,
)


def main() -> int:
    ok = 0
    for test in TESTS:
        tmp = Path(tempfile.mkdtemp(prefix="chain_"))
        try:
            test(tmp)
        except AssertionError as exc:
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
            ok += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{ok}/{len(TESTS)} 通过")
    return 0 if ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
