"""高光方案资产 + PRM 档案（Phase 7 Batch 9）。

盯的是 Batch 8 之前的缺口：**AI 回的高光 JSON 只是一次性结果**——新的一份进来
就把旧的盖掉，事后既查不到、也没法拿旧 JSON 重剪，更说不清某个成品是哪份 JSON、
哪家 AI、哪一版 PRM 出来的。这一批把 JSON 变成可查、可留痕、可复用的资产。

覆盖：
  T1  同一个视频可以有多份方案，登记新的绝不覆盖旧的
  T2  raw_json 永远是 AI 原话（current_json 变了也不动它）
  T3  编辑默认另开一条：parent_id 指回、版本 +1、原方案一字不动
  T4  in_place 编辑只改 current_json，raw_json 照旧
  T5  复制出来的是独立副本，改副本不影响原件
  T6  多个 AI 来源（provider / model）各自记账，按 AI 查得到
  T7  按视频查方案；软删的默认不出现，include_deleted 才出
  T8  按 PRM 查方案
  T9  软删方案：成品一个不动，且成品仍能溯源到「方案（已删除）」
  T10 恢复软删的方案
  T11 当前方案：is_current 唯一；软删之后退回最近一份
  T12 videos_with_assets 只认抠得出片段的方案（clip_count > 0）
  T13 坏 JSON 不当好的用（loads / summarize / asset_payload）
  T14 成品全链路可追溯：视频 / 分析 / 方案 / AI / 模型 / 任务 / PRM
  T15 没有方案的视频照旧走 AI（自动剪辑发 AI 一次）
  T16 已有方案的视频**一次 AI 都不调**，直接按库里的 JSON 开剪
  T17 高光来源 existing / missing 会筛掉不合口味的视频，不给它们排队
  T18 一份 JSON + 两版 PRM = 两个成品并存，各自记得用的是哪一版
  T19 PRM 增删改 + 设默认 + ensure_prm 幂等
  T20 软删 PRM 之后，历史成品照旧查得到用的是它（prm_deleted 标记）
  T21 发 AI 时优先用选中的 PRM 档案（不再硬编码 prm_en.txt）
  T22 选中的 PRM 被软删 -> 退回默认 PRM，不炸
  T23 schema v4：老库升上来和新建库的表/索引完全一致
  T24 升级只加不改：老数据一行不动，artifacts 的新列是 NULL

功能测试直接调 `MainWindow` 上的真方法（绑到轻量替身上，不建窗口），
渲染 / 发 AI 这些叶子调用换成计数替身。全部用临时目录里的临时库，
**绝不碰项目真实数据库**。
可以直接 `python tests/test_highlight_assets.py`，也可以 `pytest tests/test_highlight_assets.py`。
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 只导入模块，不建窗口

from vidscribe.config import Config                      # noqa: E402
from vidscribe.db import assets as db_assets             # noqa: E402
from vidscribe.db import migrations, open_db             # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db import schema                          # noqa: E402
from vidscribe.gui import main_window as mw              # noqa: E402


def payload(start: float = 4.0, end: float = 13.0, score: float = 0.87,
            video: str = "v.mp4") -> dict:
    return {"video": video,
            "clip": {"start": start, "end": end, "score": score,
                     "type": "hook", "reason": "r"}}


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
    for sub in ("database", "input", "output", "logs", "ai_out", "cache", "prm"):
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
    data.setdefault("bridge", {})
    data["bridge"]["ai_input_dir"] = str(tmp_path / "input")
    data["bridge"]["ai_output_dir"] = str(tmp_path / "ai_out")
    data["bridge"]["ai_job"] = "full"
    data["bridge"]["highlight_source"] = "all"
    data["bridge"]["prm_id"] = 0
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str) -> Path:
    path = cfg.path("input_dir") / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def video_row(cfg, db, name: str) -> tuple[Path, int]:
    video = fake_video(cfg, name)
    return video, db_repo.upsert_video(db, video)


def prm_file(cfg, name: str, text: str = "rules") -> Path:
    path = cfg.root / "prm" / name
    path.write_text(text, encoding="utf-8")
    return path


class Win:
    """够跑「高光来源 / 方案复用 / PRM 选择」这几条链路的替身。"""

    _auto_step = mw.MainWindow._auto_step
    _enqueue_auto_tasks = mw.MainWindow._enqueue_auto_tasks
    _resume_existing_ai_json = mw.MainWindow._resume_existing_ai_json
    _asset_json_for_render = mw.MainWindow._asset_json_for_render
    _source_allows = mw.MainWindow._source_allows
    highlight_source = mw.MainWindow.highlight_source
    selected_prm = mw.MainWindow.selected_prm
    resolve_prompt_file = mw.MainWindow.resolve_prompt_file
    _save_ai_result = mw.MainWindow._save_ai_result
    _register_highlight_asset = mw.MainWindow._register_highlight_asset
    _register_artifact = mw.MainWindow._register_artifact
    _register_final_video = mw.MainWindow._register_final_video
    _link_final_video = mw.MainWindow._link_final_video
    _auto_save_script = mw.MainWindow._auto_save_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _db_video_id = mw.MainWindow._db_video_id
    _settle_auto_task = mw.MainWindow._settle_auto_task
    _mark_auto_rendering = mw.MainWindow._mark_auto_rendering

    def __init__(self, cfg, db):
        self.cfg = cfg
        self._db_handle = db
        self._db_failed = False
        self._process_started_at = db_repo.now()
        self._queue_lock = None
        self._auto_job = "full"
        self._auto_task_id = None
        self._auto_video = None
        self.video_path = None
        self._auto_active = False
        self._auto_done = 0
        self._auto_total = 0
        self._last_highlight_json = ""
        self._last_prompt = {}
        self._last_prm_id = None
        self._last_asset_id = None
        self.clip_worker = None
        self.speech = []
        self.timeline = []
        self.rendered = ""
        self.calls = {k: 0 for k in ("send_file_to_ai", "dispatch_ai", "run_highlight",
                                     "on_analyze", "_auto_after_analyze", "load_video",
                                     "finish")}
        self.logs: list[str] = []

    def _db(self):
        return self._db_handle

    def ai_dir(self, key):
        return Path(str(self.cfg.bridge.get(key)))

    def export_root(self):
        return self.cfg.path("output_dir")

    def _worker_id(self):
        return "gui-test"

    def append_log(self, message):
        self.logs.append(str(message))

    def load_video(self, path):
        self.calls["load_video"] += 1
        self.video_path = Path(path)

    def send_file_to_ai(self, path):
        self.calls["send_file_to_ai"] += 1
        return True

    def dispatch_ai(self, *a, **k):
        self.calls["dispatch_ai"] += 1

    def run_highlight(self, text, ai=False, name_suffix=""):
        self.calls["run_highlight"] += 1
        self.rendered = text

    def on_analyze(self, *a, **k):
        self.calls["on_analyze"] += 1

    def _auto_after_analyze(self):
        self.calls["_auto_after_analyze"] += 1

    def _auto_finish(self, *a, **k):
        self.calls["finish"] += 1

    def _auto_advance(self, *a, **k):
        pass

    def _set_auto_state(self, *a, **k):
        pass

    def _set_auto_step(self, *a, **k):
        pass

    def _set_auto_progress(self, *a, **k):
        pass

    def auto_running(self):
        return False

    def auto_busy(self):
        return ""


class FakeWorker:
    """只提供 cut_ranges，让 `_register_final_video` 走完整回写那一段。"""

    def __init__(self, ranges):
        self.cut_ranges = list(ranges)


# ------------------------------------------------------------------ T1
def test_multiple_assets_never_overwrite(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t1.mp4")
    first = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    second = db_assets.create_asset(db, vid, payload(20.0, 28.0))
    third = db_assets.create_asset(db, vid, payload(40.0, 47.0))

    rows = db_assets.list_assets(db, vid)
    assert [int(r["id"]) for r in rows] == [first, second, third], "三份方案都得在"
    assert [r["name"] for r in rows] == ["方案 A", "方案 B", "方案 C"]
    kept = db_assets.asset_payload(db, first)
    assert kept["clip"]["start"] == 4.0, "登记新方案不许改旧方案的 JSON"
    assert db_assets.asset_counts(db, [vid])[vid] == 3
    db.close()


# ------------------------------------------------------------------ T2
def test_raw_json_is_always_the_ai_original(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t2.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    edited = db_assets.edit_asset(db, origin, payload(5.5, 12.0))

    row = db_assets.get_asset(db, edited)
    assert json.loads(row["raw_json"])["clip"]["start"] == 4.0, "raw_json 得是 AI 原话"
    assert json.loads(row["current_json"])["clip"]["start"] == 5.5
    db.close()


# ------------------------------------------------------------------ T3
def test_edit_opens_a_new_asset(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t3.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    edited = db_assets.edit_asset(db, origin, payload(6.0, 14.0))

    assert edited != origin, "默认必须另开一条，不是就地改"
    new_row = db_assets.get_asset(db, edited)
    old_row = db_assets.get_asset(db, origin)
    assert int(new_row["parent_id"]) == origin and int(new_row["version"]) == 2
    assert new_row["source_type"] == "edited"
    assert json.loads(old_row["current_json"])["clip"]["start"] == 4.0, "原方案一字不动"
    assert int(new_row["is_current"]) == 1 and int(old_row["is_current"]) == 0
    db.close()


# ------------------------------------------------------------------ T4
def test_in_place_edit_keeps_raw(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t4.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    same = db_assets.edit_asset(db, origin, payload(7.0, 15.0), in_place=True)

    assert same == origin, "就地改就该还是那一条"
    row = db_assets.get_asset(db, origin)
    assert json.loads(row["raw_json"])["clip"]["start"] == 4.0
    assert json.loads(row["current_json"])["clip"]["start"] == 7.0
    assert len(db_assets.list_assets(db, vid)) == 1
    db.close()


# ------------------------------------------------------------------ T5
def test_copy_is_independent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t5.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    copy = db_assets.copy_asset(db, origin)
    db_assets.edit_asset(db, copy, payload(9.0, 17.0), in_place=True)

    assert json.loads(db_assets.get_asset(db, origin)["current_json"])["clip"]["start"] == 4.0
    assert json.loads(db_assets.get_asset(db, copy)["current_json"])["clip"]["start"] == 9.0
    assert db_assets.get_asset(db, copy)["source_type"] == "copied"
    db.close()


# ------------------------------------------------------------------ T6
def test_multiple_ai_sources_are_queryable(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t6.mp4")
    db_assets.create_asset(db, vid, payload(), provider="gemini", model="gemini-2.5-flash")
    db_assets.create_asset(db, vid, payload(20.0, 27.0), provider="gemini",
                           model="gemini-2.5-pro")
    db_assets.create_asset(db, vid, payload(30.0, 37.0), provider="deepseek",
                           model="deepseek-chat")

    assert len(db_assets.assets_by_ai(db, provider="gemini")) == 2
    assert len(db_assets.assets_by_ai(db, provider="gemini", model="gemini-2.5-pro")) == 1
    assert len(db_assets.assets_by_ai(db, provider="deepseek")) == 1
    db.close()


# ------------------------------------------------------------------ T7
def test_deleted_assets_hide_by_default(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t7.mp4")
    keep = db_assets.create_asset(db, vid, payload())
    gone = db_assets.create_asset(db, vid, payload(20.0, 27.0))
    assert db_assets.delete_asset(db, gone) is True

    live = [int(r["id"]) for r in db_assets.list_assets(db, vid)]
    every = [int(r["id"]) for r in db_assets.list_assets(db, vid, include_deleted=True)]
    assert live == [keep] and every == [keep, gone]
    assert db_assets.get_asset(db, gone) is not None, "软删的还得查得到"
    db.close()


# ------------------------------------------------------------------ T8
def test_assets_by_prm(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t8.mp4")
    one = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt")
    two = db_assets.create_prm(db, "PRM V2", "prm/prm_zh.txt")
    db_assets.create_asset(db, vid, payload(), prm_id=one)
    db_assets.create_asset(db, vid, payload(20.0, 27.0), prm_id=one)
    db_assets.create_asset(db, vid, payload(30.0, 37.0), prm_id=two)

    assert len(db_assets.assets_by_prm(db, one)) == 2
    assert len(db_assets.assets_by_prm(db, two)) == 1
    db.close()


# ------------------------------------------------------------------ T9
def test_soft_delete_keeps_products(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t9.mp4")
    asset = db_assets.create_asset(db, vid, payload())
    product = cfg.path("output_dir") / "t9_高光时刻.mp4"
    product.write_bytes(b"x" * 4096)
    artifact = db_repo.register_artifact(db, vid, "final_video", product)
    db_assets.link_artifact(db, artifact, asset_id=asset)

    db_assets.delete_asset(db, asset)
    assert product.is_file(), "软删方案绝不许删成品文件"
    assert len(db_assets.products_for_asset(db, asset)) == 1, "成品照旧挂在它名下"
    trace = db_assets.artifact_lineage(db, artifact)
    assert trace["asset"]["id"] == asset and trace["asset_deleted"] is True
    db.close()


# ------------------------------------------------------------------ T10
def test_restore_asset(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t10.mp4")
    asset = db_assets.create_asset(db, vid, payload())
    db_assets.delete_asset(db, asset)
    assert db_assets.restore_asset(db, asset) is True
    assert db_assets.restore_asset(db, asset) is False, "没删的不用恢复"
    assert [int(r["id"]) for r in db_assets.list_assets(db, vid)] == [asset]
    db.close()


# ------------------------------------------------------------------ T11
def test_current_asset_is_unique_and_falls_back(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t11.mp4")
    first = db_assets.create_asset(db, vid, payload())
    second = db_assets.create_asset(db, vid, payload(20.0, 27.0))

    assert int(db_assets.current_asset(db, vid)["id"]) == second, "最后登记的就是当前"
    live = db.all("SELECT id FROM highlight_assets WHERE video_id = ? AND is_current = 1 "
                  "AND deleted_at IS NULL", (vid,))
    assert len(live) == 1, "当前方案只能有一个"
    assert db_assets.set_current_asset(db, first) is True
    assert int(db_assets.current_asset(db, vid)["id"]) == first
    db_assets.delete_asset(db, first)
    assert int(db_assets.current_asset(db, vid)["id"]) == second, "删了就退回最近一份"
    db.close()


# ------------------------------------------------------------------ T12
def test_videos_with_assets_needs_real_clips(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _v1, good = video_row(cfg, db, "t12a.mp4")
    _v2, empty = video_row(cfg, db, "t12b.mp4")
    _v3, plain = video_row(cfg, db, "t12c.mp4")
    db_assets.create_asset(db, good, payload())
    db_assets.create_asset(db, empty, {"error": "no highlight found"})

    found = db_assets.videos_with_assets(db, [good, empty, plain])
    assert found == {good}, "抠不出片段的 JSON 不算有方案"
    db.close()


# ------------------------------------------------------------------ T13
def test_broken_json_is_never_used(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t13.mp4")
    asset = db_assets.create_asset(db, vid, "{ not json at all")

    assert db_assets.loads("{ nope") is None
    assert db_assets.summarize("{ nope") == (0, None)
    assert db_assets.asset_payload(db, asset) is None
    assert db_assets.videos_with_assets(db, [vid]) == set()
    db.close()


# ------------------------------------------------------------------ T14
def test_product_traces_back_to_everything(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t14.mp4")
    run_id = db_repo.create_analysis(db, vid, {})
    db_repo.finish_analysis(db, run_id)
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", language="en", version="V1")
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    asset = db_assets.create_asset(db, vid, payload(), provider="gemini",
                                   model="gemini-2.5-flash", analysis_id=run_id,
                                   source_task_id=task_id, prm_id=prm)
    product = cfg.path("output_dir") / "t14_高光时刻.mp4"
    product.write_bytes(b"y" * 4096)
    artifact = db_repo.register_artifact(db, vid, "final_video", product)
    db_assets.link_artifact(db, artifact, asset_id=asset, prm_id=prm)

    trace = db_assets.artifact_lineage(db, artifact)
    assert trace["video"]["file_name"] == video.name
    assert trace["analysis_id"] == run_id
    assert trace["asset"]["id"] == asset
    assert (trace["provider"], trace["model"]) == ("gemini", "gemini-2.5-flash")
    assert trace["task_id"] == task_id
    assert trace["prm"]["name"] == "PRM V1" and trace["prm_deleted"] is False
    overview = db_assets.video_overview(db, vid)
    assert len(overview["assets"]) == 1 and len(overview["products"]) == 1
    db.close()


# ------------------------------------------------------------------ T15
def test_video_without_asset_still_asks_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t15.mp4")
    txt = video.with_suffix(".txt")
    txt.write_text("merged text", encoding="utf-8")
    db_repo.register_artifact(db, vid, "merged_txt", txt)
    db_repo.enqueue_ai_task(db, vid, mode="full")

    win = Win(cfg, db)
    win._auto_step()
    assert win.calls["send_file_to_ai"] == 1, "没方案就该问 AI"
    assert win.calls["run_highlight"] == 0
    db.close()


# ------------------------------------------------------------------ T16
def test_video_with_asset_renders_without_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t16.mp4")
    txt = video.with_suffix(".txt")
    txt.write_text("merged text", encoding="utf-8")
    db_repo.register_artifact(db, vid, "merged_txt", txt)
    asset = db_assets.create_asset(db, vid, payload(video=video.name))
    db_repo.enqueue_ai_task(db, vid, mode="full")

    win = Win(cfg, db)
    win._auto_step()
    assert win.calls["run_highlight"] == 1, "库里有方案就直接开剪"
    assert win.calls["send_file_to_ai"] == 0 and win.calls["dispatch_ai"] == 0, "一次 AI 都不许调"
    assert json.loads(win.rendered)["clip"]["start"] == 4.0
    assert win._last_asset_id == asset, "记住按哪份方案剪的，成品要靠它溯源"
    task = db_repo.get_ai_task(db, win._auto_task_id)
    assert task["status"] == "processing", "状态机照旧：素材齐了就是在剪"
    db.close()


# ------------------------------------------------------------------ T17
def test_highlight_source_filters_the_queue(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    with_asset, vid_a = video_row(cfg, db, "t17a.mp4")
    without, vid_b = video_row(cfg, db, "t17b.mp4")
    db_assets.create_asset(db, vid_a, payload(video=with_asset.name))
    videos = [with_asset, without]

    win = Win(cfg, db)
    cfg.bridge["highlight_source"] = "existing"
    created, _reused, _already, skipped = win._enqueue_auto_tasks(videos, "full")
    assert (created, skipped) == (1, 1), "只挑已有 JSON 的，另一个不该排队"
    assert db_repo.get_ai_task(db, 1)["video_id"] == vid_a

    cfg.bridge["highlight_source"] = "missing"
    win2 = Win(cfg, db)
    created2, _r2, _a2, skipped2 = win2._enqueue_auto_tasks(videos, "collect")
    assert (created2, skipped2) == (1, 1), "只挑没 JSON 的，有方案的那个不排队"
    row = db.one("SELECT video_id FROM ai_tasks WHERE mode = ?", ("collect",))
    assert int(row["video_id"]) == vid_b

    cfg.bridge["highlight_source"] = "all"
    win3 = Win(cfg, db)
    _c3, _r3, _a3, skipped3 = win3._enqueue_auto_tasks(videos, "script")
    assert skipped3 == 0, "全部这一档不筛"
    db.close()


# ------------------------------------------------------------------ T18
def test_one_json_two_prms_two_products(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t18.mp4")
    asset = db_assets.create_asset(db, vid, payload(video=video.name))
    prm_one = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", version="V1")
    prm_two = db_assets.create_prm(db, "PRM V2", "prm/prm_zh.txt", version="V2")

    win = Win(cfg, db)
    made = []
    for prm_id, tail in ((prm_one, "V1"), (prm_two, "V2")):
        product = cfg.path("output_dir") / f"t18_高光时刻_{tail}.mp4"
        product.write_bytes(b"p" * 4096)
        win._auto_video = video
        win._last_asset_id = asset
        win._last_prm_id = prm_id
        win.clip_worker = FakeWorker([(4.0, 12.0)])
        win._register_final_video(str(product))
        made.append(product)

    assert all(p.is_file() for p in made), "两个成品必须同时存在"
    products = db_assets.products_for_asset(db, asset)
    assert len(products) == 2, "一份 JSON 剪出两个成品，两条都要挂在它名下"
    assert {int(r["prm_id"]) for r in products} == {prm_one, prm_two}, "各自记得用的哪版 PRM"
    assert len(db_assets.products_for_prm(db, prm_one)) == 1
    db.close()


# ------------------------------------------------------------------ T19
def test_prm_crud_and_default(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    one = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", language="en")
    two = db_assets.create_prm(db, "PRM V2", "prm/prm_zh.txt", language="zh",
                               make_default=True)

    assert int(db_assets.default_prm(db)["id"]) == two
    assert db_assets.set_default_prm(db, one) is True
    assert int(db_assets.default_prm(db)["id"]) == one
    live = db.all("SELECT id FROM prm_profiles WHERE is_default = 1 AND deleted_at IS NULL", ())
    assert len(live) == 1, "默认 PRM 只能有一个"

    assert db_assets.update_prm(db, two, version="V2.1") is True
    assert db_assets.get_prm(db, two)["version"] == "V2.1"
    assert db_assets.ensure_prm(db, "prm/prm_en.txt") == one, "同一份文件不许重复登记"
    fresh = db_assets.ensure_prm(db, str(cfg.root / "prm" / "prm_new.txt"))
    assert fresh not in (one, two) and db_assets.get_prm(db, fresh)["name"] == "prm_new"
    db.close()


# ------------------------------------------------------------------ T20
def test_deleted_prm_is_still_traceable(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t20.mp4")
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt")
    asset = db_assets.create_asset(db, vid, payload(), prm_id=prm)
    product = cfg.path("output_dir") / "t20_高光时刻.mp4"
    product.write_bytes(b"q" * 4096)
    artifact = db_repo.register_artifact(db, vid, "final_video", product)
    db_assets.link_artifact(db, artifact, asset_id=asset, prm_id=prm)

    assert db_assets.delete_prm(db, prm) is True
    assert [int(r["id"]) for r in db_assets.list_prms(db)] == [], "列表里不该再出现"
    trace = db_assets.artifact_lineage(db, artifact)
    assert trace["prm"]["name"] == "PRM V1" and trace["prm_deleted"] is True
    db.close()


# ------------------------------------------------------------------ T21
def test_prompt_comes_from_the_selected_prm(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    chosen = prm_file(cfg, "prm_zh.txt", "中文规则")
    prm_file(cfg, "prm_en.txt", "english rules")
    prm_id = db_assets.create_prm(db, "PRM V2", "prm/prm_zh.txt", language="zh")
    cfg.bridge["prm_id"] = prm_id
    cfg.bridge["prompt_file"] = "prm/prm_en.txt"

    win = Win(cfg, db)
    assert win.resolve_prompt_file() == chosen, "选了 PRM 就得用它，不许写死 prm_en.txt"
    db.close()


# ------------------------------------------------------------------ T22
def test_deleted_selection_falls_back_to_default(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    prm_file(cfg, "prm_en.txt", "english rules")
    fallback = prm_file(cfg, "prm_ja.txt", "日本語")
    gone = db_assets.create_prm(db, "PRM 旧", "prm/prm_zh.txt")
    keep = db_assets.create_prm(db, "PRM 新", "prm/prm_ja.txt", make_default=True)
    db_assets.delete_prm(db, gone)
    cfg.bridge["prm_id"] = gone

    win = Win(cfg, db)
    assert int(win.selected_prm()["id"]) == keep, "选中的被软删就退回默认"
    assert win.resolve_prompt_file() == fallback
    db.close()


# ------------------------------------------------------------------ T23
def test_migration_matches_a_fresh_v4_database(tmp_path: Path) -> None:
    def objects(conn):
        rows = conn.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE ?",
                            ("sqlite_%",)).fetchall()
        return {(t, n) for t, n in rows}

    def v3_statements() -> list[str]:
        """把 schema.TABLES 退回 v4 之前的样子：去掉两张新表、成品的两个新列和相关索引。"""
        out: list[str] = []
        for statement in schema.TABLES:
            if "CREATE TABLE" in statement and " artifacts" in statement:
                keep = [line for line in statement.splitlines()
                        if "highlight_asset_id" not in line and "prm_id" not in line
                        and not line.strip().startswith("--")]
                out.append("\n".join(keep))
                continue
            if "prm_profiles" in statement or "highlight_asset" in statement:
                continue
            if "artifacts(prm_id)" in statement:
                continue
            out.append(statement)
        return out

    fresh = sqlite3.connect(":memory:")
    assert migrations.apply(fresh) == 4, "新建库就是 v4"

    old = sqlite3.connect(":memory:")
    old.execute("BEGIN")
    for statement in v3_statements():
        old.execute(statement)
    old.execute("PRAGMA user_version=3")
    old.commit()
    assert ("table", "highlight_assets") not in objects(old), "造出来的老库不该有新表"
    assert migrations.apply(old) == 4, "老库能升到 v4"

    missing = objects(fresh) - objects(old)
    assert not missing, f"升级漏了这些对象：{sorted(missing)}"
    columns = lambda conn: [r[1] for r in conn.execute("PRAGMA table_info(artifacts)")]
    assert columns(fresh) == columns(old), "artifacts 的列必须一致"
    fresh.close()
    old.close()


# ------------------------------------------------------------------ T24
def test_upgrade_only_adds(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t24.mp4")
    product = cfg.path("output_dir") / "t24_高光时刻.mp4"
    product.write_bytes(b"r" * 4096)
    artifact = db_repo.register_artifact(db, vid, "final_video", product)

    row = db.one("SELECT * FROM artifacts WHERE id = ?", (artifact,))
    assert row["highlight_asset_id"] is None and row["prm_id"] is None, \
        "老成品的新列就该是 NULL，不许瞎猜来源"
    assert int(db.value("PRAGMA user_version")) == 4
    assert db_assets.artifact_lineage(db, artifact)["asset"] is None, "查不到来源就老实说没有"
    db.close()


# ------------------------------------------------------------------ T25
def test_center_rows_aggregates_and_filters(tmp_path: Path) -> None:
    """资产中心主列表：一次查询就给出方案数 / 高光数 / 成品数 / 最近 AI，还能筛能排。"""
    cfg, db = make_project(tmp_path)
    rich, vid_a = video_row(cfg, db, "c1.mp4")
    lean, vid_b = video_row(cfg, db, "c2.mp4")
    _bare, vid_c = video_row(cfg, db, "zz_bare.mp4")
    db_assets.create_asset(db, vid_a, payload(video=rich.name), provider="gemini",
                           model="gemini-2.5-flash")
    db_assets.create_asset(db, vid_a, {"clips": [payload()["clip"], payload(30.0, 38.0)["clip"]]},
                           provider="qwen", model="qwen3-vl")
    db_assets.create_asset(db, vid_b, payload(video=lean.name), provider="gemini",
                           model="gemini-2.5-pro")
    product = cfg.path("output_dir") / "c1_高光时刻.mp4"
    product.write_bytes(b"z" * 4096)
    db_repo.register_artifact(db, vid_a, "final_video", product)

    rows = {r["id"]: r for r in db_assets.center_rows(db)}
    assert rows[vid_a]["json_count"] == 2 and rows[vid_a]["highlight_count"] == 3
    assert rows[vid_a]["product_count"] == 1 and rows[vid_b]["product_count"] == 0
    assert rows[vid_a]["provider"] == "qwen", "最近一份 JSON 的 AI 就是列表里显示的那个"
    assert rows[vid_c]["json_count"] == 0

    assert [r["id"] for r in db_assets.center_rows(db, search="c2")] == [vid_b]
    assert {r["id"] for r in db_assets.center_rows(db, provider="gemini")} == {vid_a, vid_b}
    assert {r["id"] for r in db_assets.center_rows(db, status="no_json")} == {vid_c}
    assert {r["id"] for r in db_assets.center_rows(db, status="has_product")} == {vid_a}
    assert [r["id"] for r in db_assets.center_rows(db, order="json")][0] == vid_a
    assert db_assets.center_rows(db, order="name")[0]["file_name"] == "c1.mp4"
    assert set(db_assets.known_providers(db)) == {"gemini", "qwen"}
    db.close()


# ------------------------------------------------------------------ T26
def test_center_rows_ignores_deleted_assets(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "c3.mp4")
    keep = db_assets.create_asset(db, vid, payload(), provider="gemini", model="m")
    gone = db_assets.create_asset(db, vid, payload(20.0, 27.0), provider="gemini", model="m")
    db_assets.delete_asset(db, gone)

    row = db_assets.center_rows(db, search="c3")[0]
    assert row["json_count"] == 1 and row["highlight_count"] == 1, "软删的不算在数量里"
    assert db_assets.get_asset(db, gone) is not None, "但它本身还在库里"
    assert int(db_assets.current_asset(db, vid)["id"]) == keep
    db.close()


# ------------------------------------------------------------------ T27
def test_prm_copy_and_restore(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    origin = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", language="en", version="V1")
    copy = db_assets.copy_prm(db, origin)
    assert copy is not None and copy != origin
    copied = db_assets.get_prm(db, copy)
    assert copied["name"] != "PRM V1" and copied["filename"] == "prm/prm_en.txt"
    assert copied["version"] == "V1", "复制连语言/版本一起带过去"
    assert db_assets.get_prm(db, origin)["name"] == "PRM V1", "原档案一个字不动"

    assert db_assets.delete_prm(db, copy) is True
    assert db_assets.restore_prm(db, copy) is True
    assert db_assets.restore_prm(db, copy) is False, "没删的不用恢复"
    assert {int(r["id"]) for r in db_assets.list_prms(db)} == {origin, copy}
    db.close()


# ------------------------------------------------------------------ T28
def test_center_rows_combines_json_and_product(tmp_path: Path) -> None:
    """三维筛选：JSON 有/无 × 成品 有/无 × 分析状态，可以同时生效；旧 status 值还认。"""
    cfg, db = make_project(tmp_path)
    _a, both = video_row(cfg, db, "d1.mp4")          # 有 JSON + 有成品
    _b, only_json = video_row(cfg, db, "d2.mp4")     # 有 JSON + 无成品
    _c, bare = video_row(cfg, db, "d3.mp4")          # 都没有
    db_assets.create_asset(db, both, payload(), provider="gemini", model="m")
    db_assets.create_asset(db, only_json, payload(), provider="gemini", model="m")
    product = cfg.path("output_dir") / "d1_高光时刻.mp4"
    product.write_bytes(b"z" * 2048)
    db_repo.register_artifact(db, both, "final_video", product)

    def ids(**kw):
        return {int(r["id"]) for r in db_assets.center_rows(db, **kw)}

    assert ids(json="has", product="none") == {only_json}, "有 JSON + 无成品要一步筛出来"
    assert ids(json="has", product="has") == {both}
    assert ids(json="none") == {bare}
    assert ids(product="has") == {both}
    assert ids() == {both, only_json, bare}, "默认什么都不筛"
    # 旧参数还认（CLI / 老代码传的是 status）
    assert ids(status="no_json") == {bare}
    assert ids(status="has_product") == {both}
    assert ids(status="has_json", product="none") == {only_json}, "新参数优先，旧的当兜底"
    db.close()


# ------------------------------------------------------------------ T29
def test_batch_apis_replace_per_row_queries(tmp_path: Path) -> None:
    """成品/计数走批量接口：一个视频一次查完，界面不用逐行 products_for_asset。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "d4.mp4")
    prm = db_assets.create_prm(db, "PRM V1", str(prm_file(cfg, "prm_zh.txt")))
    first = db_assets.create_asset(db, vid, payload(), provider="gemini", model="m")
    second = db_assets.create_asset(db, vid, payload(20.0, 27.0), provider="qwen", model="q")
    made = []
    for index in range(3):
        target = cfg.path("output_dir") / f"d4_{index}.mp4"
        target.write_bytes(b"z" * 1024)
        made.append(int(db_assets.record_product(
            db, vid, target, specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
            asset_id=first if index < 2 else second, prm_id=prm)["artifact_id"]))

    assert db_assets.product_counts_for_assets(db, vid) == {first: 2, second: 1}
    assert [int(r["id"]) for r in db_assets.list_products(db, vid)] == sorted(made, reverse=True)
    assert db_assets.product_path(db, made[0]).name == "d4_0.mp4"
    assert db_assets.product_path(db, 10**6) is None, "查不到就老实返回 None"

    overview = db_assets.products_overview(db, vid)
    assert len(overview) == 3 and overview[0]["artifact_id"] == max(made), "新的排前面"
    head = overview[0]
    assert head["asset_id"] == second and head["prm_name"] == "PRM V1"
    assert head["asset_deleted"] is False and head["exists_on_disk"] is True
    assert [(s["start"], s["end"]) for s in head["spans"]] == [(4.0, 13.0)], "实际区间来自 clips"

    # 软删来源之后，成品照样列得出来，只是多个「已删除」标记
    db_assets.delete_asset(db, second)
    again = db_assets.products_overview(db, vid)[0]
    assert again["asset_id"] == second and again["asset_deleted"] is True

    # 三层区间：一次算完，和分开算的结果一致
    layers = db_assets.asset_layers(db, first, artifact_id=made[0])
    spans = db_assets.asset_spans(db, first)
    trace = db_assets.lineage_spans(db, made[0])
    assert layers["ai"] == spans["ai"] and layers["engine"] == spans["engine"]
    assert layers["actual"] == trace["actual"], "实际渲染那一层还是 clips 说的算"
    assert db_assets.asset_layers(db, first)["actual"] == [], "不给成品就不算实际渲染"
    db.close()


# ------------------------------------------------------------------ S1

def _function(source: str, name: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"源码里找不到 {name}()")


def _call_names(func) -> list[str]:
    """按出现顺序列出被调用的名字（ast.walk 是广度优先，得自己按行号排）。"""
    found = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if name:
                found.append((node.lineno, node.col_offset, name))
    return [name for _line, _col, name in sorted(found)]


def test_auto_step_prefers_the_library_over_ai(tmp_path: Path) -> None:
    """结构守卫：自动剪辑必须先问库里有没有方案，再考虑发 AI。

    GUI 在这台机器上建不起窗口（Qt 无头会崩），所以界面那一层用源码结构守。
    """
    source = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")
    names = _call_names(_function(source, "_auto_step"))
    assert "_asset_json_for_render" in names, "少了「库里有方案就直接剪」这一步"
    assert names.index("_asset_json_for_render") < names.index("send_file_to_ai"), \
        "得先看库里有没有方案，再决定要不要发 AI"
    assert names.index("_resume_existing_ai_json") < names.index("_asset_json_for_render"), \
        "本任务自己的 AI 结果优先级最高"
    assert "_mark_auto_rendering" in names, "状态机不能丢：素材齐了要落 processing"


# ------------------------------------------------------------------ S2
def test_product_registration_records_its_source(tmp_path: Path) -> None:
    source = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")
    names = _call_names(_function(source, "_register_final_video"))
    assert "_register_artifact" in names and "_link_final_video" in names
    assert names.index("_register_artifact") < names.index("_link_final_video"), \
        "先登记成品拿到 id，才能挂方案 / PRM"
    linker = _call_names(_function(source, "_link_final_video"))
    assert "link_artifact" in linker

    prompt = _call_names(_function(source, "resolve_prompt_file"))
    assert prompt.index("prm_file") < len(prompt), "提示词优先取 PRM 档案"
    assert "selected_prm" in prompt, "得先问「选的是哪一版 PRM」"


# ------------------------------------------------------------------ S3
def test_panel_saves_the_new_switches(tmp_path: Path) -> None:
    source = (ROOT / "src" / "vidscribe" / "gui" / "ai_options.py").read_text(encoding="utf-8")
    assert "highlight_source" in source and "prm_id" in source, "面板得能存这两个开关"
    saver = _function(source, "save")
    dumped = ast.dump(saver)
    assert "highlight_source" in dumped and "prm_id" in dumped, "save() 必须把两个开关写进 config"
    panel = (ROOT / "src" / "vidscribe" / "gui" / "assets_dialog.py").read_text(encoding="utf-8")
    for needed in ("class AssetCenter", "class AssetDialog", "class VideoAssetsPage",
                   "class PrmPanel", "class JsonPanel", "class RenderDialog",
                   "def on_render", "def on_delete",
                   "def on_set_current", "def on_default", "def refresh_lineage",
                   "center_rows"):
        assert needed in panel, f"资产中心少了 {needed}"
    # 弹窗套娃已经拆掉：JSON 详情和 PRM 都是页内面板，不再是子对话框
    for gone in ("class JsonDialog", "class PrmDialog("):
        assert gone not in panel, f"{gone} 应该已经被页内面板取代"



# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_multiple_assets_never_overwrite,
    test_raw_json_is_always_the_ai_original,
    test_edit_opens_a_new_asset,
    test_in_place_edit_keeps_raw,
    test_copy_is_independent,
    test_multiple_ai_sources_are_queryable,
    test_deleted_assets_hide_by_default,
    test_assets_by_prm,
    test_soft_delete_keeps_products,
    test_restore_asset,
    test_current_asset_is_unique_and_falls_back,
    test_videos_with_assets_needs_real_clips,
    test_broken_json_is_never_used,
    test_product_traces_back_to_everything,
    test_video_without_asset_still_asks_ai,
    test_video_with_asset_renders_without_ai,
    test_highlight_source_filters_the_queue,
    test_one_json_two_prms_two_products,
    test_prm_crud_and_default,
    test_deleted_prm_is_still_traceable,
    test_prompt_comes_from_the_selected_prm,
    test_deleted_selection_falls_back_to_default,
    test_migration_matches_a_fresh_v4_database,
    test_upgrade_only_adds,
    test_center_rows_aggregates_and_filters,
    test_center_rows_ignores_deleted_assets,
    test_prm_copy_and_restore,
    test_center_rows_combines_json_and_product,
    test_batch_apis_replace_per_row_queries,
    test_auto_step_prefers_the_library_over_ai,
    test_product_registration_records_its_source,
    test_panel_saves_the_new_switches,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="assets_"))
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
