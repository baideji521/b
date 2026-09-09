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
  T21 发 AI 按 PRM 使用状况：启用几份就发几份，停用的一份都不发
  T22 一份档案都没登记过时才退回老的 prm_en.txt 候选（和「全停用」不是一回事）
  T23 schema v4：老库升上来和新建库的表/索引完全一致
  T24 升级只加不改：老数据一行不动，artifacts 的新列是 NULL
  T25 PRM 正文存在库里：新建带正文、老库按 filename 自愈导入一次、改名改正文、副本独立

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

from vidscribe import ai_protocol                       # noqa: E402
from vidscribe.config import Config                      # noqa: E402
from vidscribe.db import assets as db_assets             # noqa: E402
from vidscribe.db import migrations, open_db             # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db import schema                          # noqa: E402
from vidscribe.gui import main_window as mw              # noqa: E402
from vidscribe.highlight import clip as clip_mod         # noqa: E402


def payload(start: float = 4.0, end: float = 13.0, score: float = 0.87,
            video: str = "v.mp4", *, more: tuple[tuple[float, float], ...] = ()) -> dict:
    """一份**新协议**的高光 JSON（唯一认的写法，见 src/vidscribe/ai_protocol.py）。

    `start` / `end` 是原视频时间，落在 `segments[0].sa` / `.end`；成片时长和文案
    在 `timeline` 里。`more` 多给的段用来验证"一份 JSON 只算一个高光"。
    """
    spans = ((start, end),) + more
    return {"video": video,
            "timeline": {"duration": round(end - start, 3), "score": score,
                         "type": "hook", "reason": "r"},
            "segments": [{"sa": sa, "end": stop, "dst": [0.0, round(stop - sa, 3)]}
                         for sa, stop in spans],
            "t": {"Scene": "室内", "Action": "打翻杯子", "Speech text": "r"}}



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
    # assets.* 里放的是用户真实目录，测试必须整段换成临时目录、
    # 并且把两个目录筛选清空，否则资产中心会把临时库里的视频全滤掉。
    data["assets"] = {
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "ai_out"),
        "filter_video_dir": "",
        "filter_product_dir": "",
    }
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
    enabled_prms = mw.MainWindow.enabled_prms
    has_prm_profiles = mw.MainWindow.has_prm_profiles
    resolve_prompt_files = mw.MainWindow.resolve_prompt_files
    resolve_prompt_file = mw.MainWindow.resolve_prompt_file
    _write_prompt_files = mw.MainWindow._write_prompt_files
    _save_ai_result = mw.MainWindow._save_ai_result
    _register_highlight_asset = mw.MainWindow._register_highlight_asset
    _register_artifact = mw.MainWindow._register_artifact
    _register_final_video = mw.MainWindow._register_final_video
    _asset_result_id = mw.MainWindow._asset_result_id         # 这次那份 JSON 是哪次 AI 结果
    _link_final_video = mw.MainWindow._link_final_video
    _auto_save_script = mw.MainWindow._auto_save_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _auto_chain_done = mw.MainWindow._auto_chain_done
    _skip_because_done = mw.MainWindow._skip_because_done
    skip_done_products = mw.MainWindow.skip_done_products
    _language_blocked = mw.MainWindow._language_blocked
    # 分析完一句语音都没有的视频不发 AI（没人说话 = 没互动，也算不出区间）
    _silent_video = mw.MainWindow._silent_video
    # 文件里根本没音轨的视频连队都不排（没声音 = 没剧本）
    _mute_video = mw.MainWindow._mute_video
    _reusable_highlight_json = mw.MainWindow._reusable_highlight_json
    script_payload = mw.MainWindow.script_payload
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
        self._auto_stop = False   # 点过「停止」没有：跟 MainWindow 的字段对齐
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
    assert kept["segments"][0]["sa"] == 4.0, "登记新方案不许改旧方案的 JSON"
    assert db_assets.asset_counts(db, [vid])[vid] == 3
    db.close()


# ------------------------------------------------------------------ T2
def test_raw_json_is_always_the_ai_original(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t2.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    edited = db_assets.edit_asset(db, origin, payload(5.5, 12.0))

    row = db_assets.get_asset(db, edited)
    assert json.loads(row["raw_json"])["segments"][0]["sa"] == 4.0, "raw_json 得是 AI 原话"
    assert json.loads(row["current_json"])["segments"][0]["sa"] == 5.5
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
    assert json.loads(old_row["current_json"])["segments"][0]["sa"] == 4.0, "原方案一字不动"
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
    assert json.loads(row["raw_json"])["segments"][0]["sa"] == 4.0
    assert json.loads(row["current_json"])["segments"][0]["sa"] == 7.0
    assert len(db_assets.list_assets(db, vid)) == 1
    db.close()


# ------------------------------------------------------ T4b（血缘：偏移与成品 1:1）
def test_offsets_fork_a_new_version(tmp_path: Path) -> None:
    """盖 startframe / freeze：没出过成品的方案原地盖（一个成品一份 JSON），
    已经有成品挂着的才另存一版（老成品的溯源不许被改）。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t4b.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0), make_current=True)

    # ① 还没出成品：原地盖，不许凭空多出一版没有成品的 JSON
    used, forked = db_assets.fork_with_offsets(db, origin, -0.55, 2.0)
    assert (used, forked) == (origin, False), "没成品挂着就该原地盖，不另存"
    assert len(db_assets.list_assets(db, vid)) == 1, "方案数不许变"
    current = json.loads(db_assets.get_asset(db, origin)["current_json"])
    assert current["startframe"] == -0.55 and current["freeze"] == 2.0, current
    # AI 原话永远不动
    assert "startframe" not in json.loads(db_assets.get_asset(db, origin)["raw_json"])

    # ② 值一模一样：什么都不做
    assert db_assets.fork_with_offsets(db, origin, -0.55, 2.0) == (origin, False)
    assert len(db_assets.list_assets(db, vid)) == 1

    # ③ 已经有成品挂着，又换了加减秒数：另存一版，老成品那份内容一个字不许变
    product = cfg.path("output_dir") / "t4b_高光时刻.mp4"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"x" * 4096)
    artifact = db_repo.register_artifact(db, vid, "final_video", product)
    db_assets.link_artifact(db, artifact, asset_id=origin)

    new_id, forked_again = db_assets.fork_with_offsets(db, origin, 1.5, -2.0)
    assert forked_again is True and new_id != origin, "有成品挂着就必须另存一版"
    row = db_assets.get_asset(db, int(new_id))
    assert int(row["parent_id"]) == origin, "新版本得挂在原方案下面（血缘要连得上）"
    assert int(row["is_current"]) == 1, "剪的是这一版，它该成为当前方案"
    fresh = json.loads(row["current_json"])
    assert fresh["startframe"] == 1.5 and fresh["freeze"] == -2.0, fresh
    old = json.loads(db_assets.get_asset(db, origin)["current_json"])
    assert old["startframe"] == -0.55 and old["freeze"] == 2.0, "老成品那份不许被改"
    assert db_assets.artifact_lineage(db, artifact)["asset"]["id"] == origin
    assert len(db_assets.list_assets(db, vid)) == 2
    db.close()


# ------------------------------------------------------------------ T5
def test_copy_is_independent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "t5.mp4")
    origin = db_assets.create_asset(db, vid, payload(4.0, 13.0))
    copy = db_assets.copy_asset(db, origin)
    db_assets.edit_asset(db, copy, payload(9.0, 17.0), in_place=True)

    assert json.loads(db_assets.get_asset(db, origin)["current_json"])["segments"][0]["sa"] == 4.0
    assert json.loads(db_assets.get_asset(db, copy)["current_json"])["segments"][0]["sa"] == 9.0
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


def test_moment_list_becomes_one_asset_per_line(tmp_path: Path) -> None:
    """结果清单入库：一行一份方案，raw_json 是 AI 原话、current_json 是程序算的区间。"""
    from argparse import Namespace

    from vidscribe import cli

    cfg, db = make_project(tmp_path)
    _, vid = video_row(cfg, db, "moments.mp4")
    run_id = db_repo.create_analysis(db, vid, {})
    db_repo.finish_analysis(db, run_id)
    db_repo.save_speech_segments(db, run_id, [
        {"start": 10.0, "end": 12.0, "text": "铺垫",
         "words": [{"word": "铺", "start": 10.0, "end": 11.0},
                   {"word": "垫", "start": 11.0, "end": 12.0}]},
        {"start": 13.0, "end": 15.0, "text": "结果",
         "words": [{"word": "结", "start": 13.0, "end": 14.0},
                   {"word": "果", "start": 14.0, "end": 15.0}]},
        {"start": 40.0, "end": 43.0, "text": "另一件事",
         "words": [{"word": "另", "start": 40.0, "end": 41.5},
                   {"word": "事", "start": 41.5, "end": 43.0}]},
    ])
    listing = tmp_path / "moments.txt"
    listing.write_text(
        '{"setup_at": 10.5, "result_at": 14.2, "score": 91, "word": "冲击"}\n'
        '{"setup_at": 40.4, "result_at": 42.0, "score": 77, "word": "第二条"}\n',
        encoding="utf-8")

    args = Namespace(import_moments=str(listing), name=None, note=None,
                     no_current=False, dry_run=False)
    row = db.one("SELECT * FROM videos WHERE id = ?", (vid,))
    assert cli._assets_import_moments(cfg, args, db, db_assets, row) == 0

    rows = db_assets.list_assets(db, vid)
    assert len(rows) == 2, "两行清单必须变成两份方案"
    # 每份方案都只有一个片段：一条素材一段，多段拼接是混剪那一关的事
    assert [int(r["clip_count"]) for r in rows] == [1, 1]
    # 当前方案只认第一条，不是最后入库的那一条
    current = db_assets.current_asset(db, vid)
    first = min(rows, key=lambda r: int(r["id"]))
    assert int(current["id"]) == int(first["id"])
    # raw_json 是清单原行（AI 原话），current_json 是程序算出的区间
    raw = db_assets.loads(first["raw_json"])
    assert raw["setup_at"] == 10.5 and raw["result_at"] == 14.2
    span = ai_protocol.clips(db_assets.loads(first["current_json"]))[0]
    assert (span["start"], span["end"]) == (10.0, 15.0), span
    assert "区间原文待译" in str(first["note"])

    # 清单解不出东西 → 不许登记空方案
    empty = tmp_path / "empty.txt"
    empty.write_text("这里没有 JSON", encoding="utf-8")
    args.import_moments = str(empty)
    assert cli._assets_import_moments(cfg, args, db, db_assets, row) == 2
    assert len(db_assets.list_assets(db, vid)) == 2

    # 视频没分析过 → 明确报错，不许拿 AI 给的两个点当区间硬剪
    _, blank = video_row(cfg, db, "no_speech.mp4")
    args.import_moments = str(listing)
    blank_row = db.one("SELECT * FROM videos WHERE id = ?", (blank,))
    assert cli._assets_import_moments(cfg, args, db, db_assets, blank_row) == 2
    assert db_assets.list_assets(db, blank) == []

    # --translate：译文填进 Speech text，备注里就不再挂"待译"
    from vidscribe import translate as translate_mod
    calls: list[list[dict]] = []

    def fake_translate(_cfg, items, **_kw):
        calls.append(list(items))
        return {"ok": True, "translations": {it["key"]: "译文" for it in items}}

    real = translate_mod.translate_items
    translate_mod.translate_items = fake_translate
    try:
        args.translate = True
        assert cli._assets_import_moments(cfg, args, db, db_assets, row) == 0
    finally:
        translate_mod.translate_items = real
    assert calls and calls[0][0]["text"], "该把区间原文交给翻译，而不是空字符串"
    fresh = max(db_assets.list_assets(db, vid), key=lambda r: int(r["id"]))
    latest = db_assets.loads(fresh["current_json"])
    assert ai_protocol.payload_of(latest)["t"]["Speech text"] == "译文"
    assert "待译" not in str(fresh["note"])
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
    assert json.loads(win.rendered)["segments"][0]["sa"] == 4.0
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
def test_prm_text_lives_in_the_database(tmp_path: Path) -> None:
    """PRM 正文存在库里：能改名、能改正文、能复制；老库的文件正文会自愈导入。"""
    cfg, db = make_project(tmp_path)
    # 新建时直接给正文：一个字节都不用落盘
    only_db = db_assets.create_prm(db, "只在库里", "手写.txt", content="库里的规则")
    assert db_assets.prm_text(db, only_db, cfg.root) == "库里的规则"

    # 老口径：只登记了文件，第一次取正文时导进库，之后文件删了也照样取得到
    source = prm_file(cfg, "prm_old.txt", "文件里的老规则")
    legacy = db_assets.create_prm(db, "老档案", "prm/prm_old.txt")
    assert db_assets.get_prm(db, legacy)["content"] is None, "登记时还没导入"
    assert db_assets.prm_text(db, legacy, cfg.root) == "文件里的老规则"
    assert db_assets.get_prm(db, legacy)["content"] == "文件里的老规则", "导入要落库"
    source.unlink()
    assert db_assets.prm_text(db, legacy, cfg.root) == "文件里的老规则", "文件没了也不影响"

    # 改名 + 改正文都在库里
    assert db_assets.update_prm(db, only_db, name="改过名字", content="改过的规则") is True
    assert db_assets.get_prm(db, only_db)["name"] == "改过名字"
    assert db_assets.prm_text(db, only_db, cfg.root) == "改过的规则"

    # 复制出来的是独立一份：正文跟着走，改副本不动原件
    copy = db_assets.copy_prm(db, only_db)
    assert db_assets.prm_text(db, copy, cfg.root) == "改过的规则"
    db_assets.update_prm(db, copy, content="副本自己的规则")
    assert db_assets.prm_text(db, only_db, cfg.root) == "改过的规则", "原件不许被带着改"
    db.close()


# ------------------------------------------------------------------ T21
def test_prompts_follow_the_prm_usage(tmp_path: Path) -> None:
    """按 PRM 使用状况发：都启用就发两份，停用的不发，全停用就一份都不发。

    发出去的是**库里的正文**写成的统一命名文件（prompt.txt / prompt_2.txt…），
    所以这里对的是「文件名 + 正文」，不再是原来那份 PRM 文件的路径。
    """
    cfg, db = make_project(tmp_path)
    prm_file(cfg, "prm_zh.txt", "中文规则")
    prm_file(cfg, "prm_en.txt", "english rules")
    zh_id = db_assets.create_prm(db, "PRM 中文", "prm/prm_zh.txt", language="zh")
    en_id = db_assets.create_prm(db, "PRM 英文", "prm/prm_en.txt", language="en")
    cfg.bridge["prompt_file"] = "prm/prm_en.txt"

    win = Win(cfg, db)
    sent = lambda: [(p.name, p.read_text(encoding="utf-8"))
                    for p in win.resolve_prompt_files()]
    assert sent() == [("prompt.txt", "中文规则"), ("prompt_2.txt", "english rules")], \
        "两份都在用就两份都发，名字按顺序统一"
    db_assets.set_prm_enabled(db, en_id, False)
    assert sent() == [("prompt.txt", "中文规则")], "停用的那一份一次都不许发"
    db_assets.set_prm_enabled(db, zh_id, False)
    assert win.resolve_prompt_files() == [], "全停用就一份都不发（这一条不发 AI）"
    db_assets.set_prm_enabled(db, en_id, True)
    assert sent() == [("prompt.txt", "english rules")], "重新启用就该回来"
    db.close()


# ------------------------------------------------------------------ T22
def test_no_profiles_falls_back_to_the_old_path(tmp_path: Path) -> None:
    """一份档案都没登记过 ≠ 全停用：还没建档时照旧走 prm_en.txt 那串候选。"""
    cfg, db = make_project(tmp_path)
    prm_file(cfg, "prm_en.txt", "english rules")
    cfg.bridge["prompt_file"] = "prm/prm_en.txt"

    win = Win(cfg, db)
    got = win.resolve_prompt_files()
    assert [(p.name, p.read_text(encoding="utf-8")) for p in got] \
        == [("prompt.txt", "english rules")], "没建过档就得有兜底，不然一条都发不出去"
    # 建了档并且启用：就只认档案里的正文，不再回头用配置里的路径
    prm_file(cfg, "prm_zh.txt", "中文规则")
    db_assets.create_prm(db, "PRM 中文", "prm/prm_zh.txt")
    got = win.resolve_prompt_files()
    assert [(p.name, p.read_text(encoding="utf-8")) for p in got] \
        == [("prompt.txt", "中文规则")]
    assert int(win.selected_prm()["id"]) == 1, "溯源用的主 PRM = 启用中的第一份"
    db.close()


# ------------------------------------------------------------------ T23
def test_migration_matches_a_fresh_v4_database(tmp_path: Path) -> None:
    def objects(conn):
        rows = conn.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE ?",
                            ("sqlite_%",)).fetchall()
        return {(t, n) for t, n in rows}

    def v3_statements() -> list[str]:
        """把 schema.TABLES 退回 v4 之前的样子：去掉后来加的表、列和相关索引。

        v5 的东西（expression_spans、analysis_runs 的三个渲染列）和 v7 的
        videos.blocked_language、v10 的 videos.no_audio 也一并去掉，否则造出来的
        "老库"里已经有了，升级脚本的 ADD COLUMN 会撞上重名。
        """
        out: list[str] = []
        for statement in schema.TABLES:
            if "CREATE TABLE IF NOT EXISTS videos" in statement:
                keep = [line for line in statement.splitlines()
                        if "blocked_language" not in line and "no_audio" not in line
                        and not line.strip().startswith("--")]
                out.append("\n".join(keep))
                continue
            if "CREATE TABLE IF NOT EXISTS artifacts" in statement:
                keep = [line for line in statement.splitlines()
                        if "highlight_asset_id" not in line and "prm_id" not in line
                        and not line.strip().startswith("--")]
                out.append("\n".join(keep))
                continue
            if "CREATE TABLE IF NOT EXISTS analysis_runs" in statement:
                keep = [line for line in statement.splitlines()
                        if "output_language" not in line and "render_config" not in line
                        and "face_available" not in line
                        and not line.strip().startswith("--")]
                # 去掉三列后 created_at 那行末尾还挂着逗号，补回来
                text = "\n".join(keep).replace("created_at         TEXT    NOT NULL,",
                                               "created_at         TEXT    NOT NULL")
                out.append(text)
                continue
            if "expression_spans" in statement:
                continue
            if "prm_profiles" in statement or "highlight_asset" in statement:
                continue
            if "artifacts(prm_id)" in statement:
                continue
            out.append(statement)
        return out

    fresh = sqlite3.connect(":memory:")
    assert migrations.apply(fresh) == 10, "新建库就是 v10"

    old = sqlite3.connect(":memory:")
    old.execute("BEGIN")
    for statement in v3_statements():
        old.execute(statement)
    old.execute("PRAGMA user_version=3")
    old.commit()
    assert ("table", "highlight_assets") not in objects(old), "造出来的老库不该有新表"
    assert ("table", "expression_spans") not in objects(old), "造出来的老库不该有表情表"
    assert migrations.apply(old) == 10, "老库能一路升到 v10"

    missing = objects(fresh) - objects(old)
    assert not missing, f"升级漏了这些对象：{sorted(missing)}"
    columns = lambda conn: [r[1] for r in conn.execute("PRAGMA table_info(artifacts)")]
    assert columns(fresh) == columns(old), "artifacts 的列必须一致"
    runs = lambda conn: [r[1] for r in conn.execute("PRAGMA table_info(analysis_runs)")]
    assert runs(fresh) == runs(old), "analysis_runs 的列必须一致"
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
    assert int(db.value("PRAGMA user_version")) == 10
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
    # 一份 JSON 里给两段 segments：新协议只剪第一段，所以高光数照旧只算 1
    db_assets.create_asset(db, vid_a, payload(more=((30.0, 38.0),)),
                           provider="qwen", model="qwen3-vl")
    db_assets.create_asset(db, vid_b, payload(video=lean.name), provider="gemini",
                           model="gemini-2.5-pro")
    product = cfg.path("output_dir") / "c1_高光时刻.mp4"
    product.write_bytes(b"z" * 4096)
    db_repo.register_artifact(db, vid_a, "final_video", product)

    rows = {r["id"]: r for r in db_assets.center_rows(db)}
    assert rows[vid_a]["json_count"] == 2 and rows[vid_a]["highlight_count"] == 2, \
        "两份方案、每份只算一个高光（多余的 segments 不计数）"
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
def test_deleted_product_file_drops_out_of_the_counts(tmp_path: Path) -> None:
    """成品文件被手动删掉后：对账一次，成品数归零、「无成品」筛得出来。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "d5.mp4")
    prm = db_assets.create_prm(db, "PRM V1", str(prm_file(cfg, "prm_zh.txt")))
    asset = db_assets.create_asset(db, vid, payload(), provider="gemini", model="m")
    product = cfg.path("output_dir") / "d5_高光时刻.mp4"
    product.write_bytes(b"z" * 2048)
    db_assets.record_product(db, vid, product,
                             specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
                             asset_id=asset, prm_id=prm)
    assert db_assets.center_rows(db, search="d5")[0]["product_count"] == 1

    product.unlink()                       # 用户在资源管理器里删掉了成品
    assert db_assets.sync_product_presence(db) == 1, "对账要发现这一条不在盘上了"
    row = db_assets.center_rows(db, search="d5")[0]
    assert row["product_count"] == 0, "文件没了就不该再显示有成品"
    assert {int(r["id"]) for r in db_assets.center_rows(db, product="none")} == {vid}
    assert db_assets.center_rows(db, product="has") == []
    assert db_assets.product_counts(db, [vid]) == {vid: 0}
    assert db_assets.product_counts_for_assets(db, vid) == {}
    assert db_assets.product_counts_for_prms(db) == {}
    assert db_assets.sync_product_presence(db) == 0, "再对一次没有变化"
    assert len(db_assets.products_overview(db, vid)) == 1, "血缘照旧留着，只是标记不在盘上"
    assert db_assets.products_overview(db, vid)[0]["exists_on_disk"] is False
    db.close()


# ------------------------------------------------------------------ T30
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
    """结构守卫：自动剪辑必须先问库里有没有可复用高光 JSON，再考虑发 AI。

    GUI 在这台机器上建不起窗口（Qt 无头会崩），所以界面那一层用源码结构守。
    库里那三处来源（本任务的 AI 结果 / 资产中心的当前方案 / ai_results）统一收在
    `_reusable_highlight_json()` 里，`_auto_step` 只认这一个入口。
    """
    source = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")
    names = _call_names(_function(source, "_auto_step"))
    assert "_reusable_highlight_json" in names, "少了「库里有高光 JSON 就直接剪」这一步"
    assert names.index("_reusable_highlight_json") < names.index("send_file_to_ai"), \
        "得先看库里有没有高光 JSON，再决定要不要发 AI"
    assert "_mark_auto_rendering" in names, "状态机不能丢：素材齐了要落 processing"

    inner = _call_names(_function(source, "_reusable_highlight_json"))
    assert "_resume_existing_ai_json" in inner and "_asset_json_for_render" in inner, \
        "库里的三处来源都得查：本任务结果、资产中心当前方案、ai_results"
    assert inner.index("_resume_existing_ai_json") < inner.index("_asset_json_for_render"), \
        "本任务自己的 AI 结果优先级最高"


# ------------------------------------------------------------------ S2
def test_product_registration_records_its_source(tmp_path: Path) -> None:
    source = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")
    names = _call_names(_function(source, "_register_final_video"))
    assert "_register_artifact" in names and "_link_final_video" in names
    assert names.index("_register_artifact") < names.index("_link_final_video"), \
        "先登记成品拿到 id，才能挂方案 / PRM"
    linker = _call_names(_function(source, "_link_final_video"))
    assert "link_artifact" in linker

    prompt = _call_names(_function(source, "resolve_prompt_files"))
    assert "prm_text" in prompt, "提示词正文只认数据库里的 content"
    assert "_write_prompt_files" in prompt, "上传前要写成统一命名的 prompt.txt"
    assert "enabled_prms" in prompt, "发哪几份得先问「哪几份在用」"


def test_bridge_hands_every_txt_to_the_extension(tmp_path: Path) -> None:
    """启用几份 PRM 就发几份：任务清单里每份都有自己的下标和取文件地址。

    扩展只能按下标去 /v1/ai/file 取内容，所以清单顺序、下标、越界保护都得对；
    少一份或串一份，AI 拿到的提示词就是残缺的。
    """
    from vidscribe.bridge.server import BridgeServer   # noqa: PLC0415

    files = []
    for name in ("prm_en.txt", "prm_zh.txt", "merged.txt"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files.append(path)

    bridge = BridgeServer(token="t")
    task_id = bridge.submit("gemini_json", {"url": "x"}, files=files)
    listed = bridge._find(task_id).payload["files"]      # noqa: SLF001
    assert [item["index"] for item in listed] == [0, 1, 2], "三份文件得各有下标"
    assert [item["name"] for item in listed] == [p.name for p in files], "顺序不能串"
    for item in listed:
        assert f"task_id={task_id}" in item["url"] and f"index={item['index']}" in item["url"], \
            "取文件地址要带真 task_id 和自己的下标"
    for index, path in enumerate(files):
        assert bridge._task_file(task_id, index) == path, f"下标 {index} 取错文件"  # noqa: SLF001
    assert bridge._task_file(task_id, len(files)) is None, "越界得取不到"  # noqa: SLF001


def test_extension_verifies_every_attachment(tmp_path: Path) -> None:
    """扩展侧：几份 txt 都得单独数，不许只认第一份就当全挂上了。

    以前是 `probe = names.slice(0, 1)`（按「就两个文件，一起拖进去」写的），
    PRM 一多就会漏：只挂上一份也判成功，AI 按残缺提示词答，回来的 JSON 是错的。

    数卡片后来改成**一次调用判完全部**（`pageScanAttachments`；老写法是每个文件名各发
    一次 executeScript，N 个文件 × 多轮轮询就是几十次往返，容易读到半成品）。所以这里
    盯的是新形状下的同三件事：拿到的是全量文件名、同前缀不互相误命中、每一处判定都按
    文件总数来。
    """
    js = (ROOT / "AI_剪辑师_好帮手" / "src" / "ai-task.js").read_text(encoding="utf-8")
    assert "names.slice(0, 1)" not in js and "const probe" not in js, \
        "不许再只拿第一个文件当门槛"
    assert "function pageScanAttachments(cardSelectors, names)" in js, \
        "数卡片要知道同批还有哪些名字，免得同前缀互相误命中"
    assert "[site.cards || [], names]" in js, "扫卡片必须把全量文件名带进页面"
    assert "if (ambiguous) continue;" in js, "同批别的文件也认这张卡时不能拿来数"
    assert js.count("await scanCards()") >= 3, \
        "塞之前的底数、等卡片、发送前复查、手动等待都要重新扫一遍"
    assert "await waitCards(names," in js and "waitCards(probe" not in js, \
        "等卡片要等全部文件"
    assert js.count("- cardsBase >= names.length") == 2, \
        "自动和手动两处的卡片张数门槛都得是文件总数"



# ------------------------------------------------------------------ S3
def test_panel_saves_the_new_switches(tmp_path: Path) -> None:
    source = (ROOT / "src" / "vidscribe" / "gui" / "ai_options.py").read_text(encoding="utf-8")
    assert "highlight_source" in source, "面板得能存处理范围这个开关"
    assert "prm_id" not in source, "PRM 不再由面板选一份（改成按使用状况发），别再存 prm_id"
    saver = _function(source, "save")
    dumped = ast.dump(saver)
    assert "highlight_source" in dumped, "save() 必须把处理范围写进 config"
    assert "prm_id" not in dumped, "save() 不许再写 prm_id"
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



# ------------------------------------------------------------------ S4 资产中心界面
def _center(cfg):
    """真的把资产中心建出来（离屏）。返回 (app, center)。"""
    from PyQt5.QtWidgets import QApplication          # noqa: PLC0415

    from vidscribe.gui.assets_dialog import AssetCenter  # noqa: PLC0415
    app = QApplication.instance() or QApplication([])
    center = AssetCenter(cfg, None, log=lambda _t: None)
    center.show()
    center.resize(1240, 800)
    app.processEvents()
    return app, center


def test_center_fits_its_own_minimum_window(tmp_path: Path) -> None:
    """整页需要的最小宽度不能超过窗口允许的最小宽度，否则列表会被裁、看不全。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "a.mp4")
    db_assets.create_asset(db, vid, payload(video="a.mp4"))
    app, center = _center(cfg)
    page = center.videos
    page.tbl_videos.selectRow(0)
    app.processEvents()
    limit = center.minimumWidth()
    need = page.minimumSizeHint().width()
    assert need <= limit, f"整页最小宽 {need} > 窗口最小宽 {limit}：窄窗口下会裁掉内容"
    for name in ("split_assets", "split_products"):
        split = getattr(page, name)
        assert split.minimumSizeHint().width() <= limit, f"{name} 自己就超了"
    # 详情弹窗（右键「查看高光 JSON」）打开之后，视频列表还是完整的一栏
    before = page.tbl_videos.width()
    page.on_open_video()
    app.processEvents()
    assert page.dlg_json.isVisible(), "右键「查看高光 JSON」要真的把弹窗打开"
    assert page.tbl_videos.isVisible() and page.tbl_videos.width() >= min(before, 200), \
        "开了详情弹窗，视频列表必须还在"
    page.dlg_json.close()
    center.close()
    db.close()


def test_clear_filters_brings_the_whole_list_back(tmp_path: Path) -> None:
    """「只看这个视频的高光」之后，得有一个入口把筛选清掉、回到全部视频。"""
    cfg, db = make_project(tmp_path)
    _a, a_id = video_row(cfg, db, "a.mp4")
    db_assets.create_asset(db, a_id, payload(video="a.mp4"))
    video_row(cfg, db, "b.mp4")
    app, center = _center(cfg)
    page = center.videos
    assert page.tbl_videos.rowCount() == 2
    page.tbl_videos.selectRow(0)
    app.processEvents()
    page.on_only_json()                    # 右键「只看这个视频的高光」
    app.processEvents()
    assert page.tbl_videos.rowCount() == 1, "筛选之后只剩这一个视频"
    page.on_clear_filters()
    app.processEvents()
    assert page.tbl_videos.rowCount() == 2, "清掉筛选要回到全部视频"
    assert page.edit_search.text() == "" and page.cmb_json.currentIndex() == 0
    center.close()
    db.close()


def test_forget_video_removes_the_records_but_keeps_the_file(tmp_path: Path) -> None:
    """从库里删除视频：库里的记录全没（不可恢复），磁盘上的文件一个都不动。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "a.mp4")
    keep, keep_id = video_row(cfg, db, "b.mp4")
    asset_id = db_assets.create_asset(db, vid, payload(video="a.mp4"))
    product = cfg.root / "ai_out" / "a_高光时刻.mp4"
    product.write_text("x" * 64, encoding="utf-8")
    art_id = db_repo.register_artifact(db, vid, "final_video", product)
    db_assets.link_artifact(db, art_id, asset_id=asset_id)
    before = db_repo.video_footprint(db, vid)
    assert before["assets"] >= 1 and before["artifacts"] >= 1

    gone = db_repo.forget_video(db, vid)
    assert gone == before, f"返回的账目要和删掉的一致：{gone} != {before}"
    assert db_repo.find_video(db, video) is None, "库里不该还有这个视频"
    assert db_repo.video_footprint(db, vid) == {k: 0 for k in before}, "关联记录必须跟着走"
    assert video.is_file() and product.is_file(), "磁盘上的文件一个都不许动"
    assert db_repo.find_video(db, keep) is not None, "别的视频不许受影响"
    assert db_repo.forget_video(db, vid) is None, "删过的再删只返回 None，不炸"

    # 界面上的入口：确认框点「是」之后列表少一行
    from PyQt5.QtWidgets import QMessageBox            # noqa: PLC0415
    app, center = _center(cfg)
    page = center.videos
    assert page.tbl_videos.rowCount() == 1, "刚才删掉的那个不该还在列表里"
    page.tbl_videos.selectRow(0)
    app.processEvents()
    original = QMessageBox.exec_
    QMessageBox.exec_ = lambda _self: QMessageBox.Yes      # 点「是」
    try:
        page.on_forget_video()
    finally:
        QMessageBox.exec_ = original
    app.processEvents()
    assert page.tbl_videos.rowCount() == 0, "界面上的删除也要真的删掉"
    assert keep.is_file(), "界面删除同样不碰磁盘文件"
    center.close()
    db.close()


def test_product_rows_show_the_video_thumbnail(tmp_path: Path) -> None:
    """成品表每一行左边挂当前视频的缩略图；解不出画面的视频只是没图，绝不报错。"""
    import cv2                                        # noqa: PLC0415
    import numpy as np                                # noqa: PLC0415

    cfg, db = make_project(tmp_path)
    # 写一个真的能解码的小 mp4（假字节的视频取不出帧，测不出缩略图这条）
    real = cfg.path("input_dir") / "real.mp4"
    writer = cv2.VideoWriter(str(real), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    assert writer.isOpened(), "这台机器写不出 mp4，测不了缩略图"
    for step in range(20):
        writer.write(np.full((90, 160, 3), 10 * step + 20, dtype=np.uint8))
    writer.release()
    vid = db_repo.upsert_video(db, real)
    asset_id = db_assets.create_asset(db, vid, payload(video="real.mp4"))
    product = cfg.root / "ai_out" / "real_高光时刻.mp4"
    product.write_text("x" * 64, encoding="utf-8")
    db_assets.link_artifact(db, db_repo.register_artifact(db, vid, "final_video", product),
                            asset_id=asset_id)
    broken, broken_id = video_row(cfg, db, "broken.mp4")     # 假字节：解不出帧
    other = cfg.root / "ai_out" / "broken_高光时刻.mp4"
    other.write_text("y" * 64, encoding="utf-8")
    db_repo.register_artifact(db, broken_id, "final_video", other)

    app, center = _center(cfg)
    page = center.videos
    page.select_video(vid)
    app.processEvents()
    assert page.tbl_products.rowCount() == 1, "成品要列出来"
    icon = page.tbl_products.item(0, 1).icon()
    assert not icon.isNull(), "成品行左边要有当前视频的缩略图"
    assert page.tbl_products.iconSize().width() == page.THUMB_SIZE.width(), \
        "图标尺寸得和 THUMB_SIZE 一致，否则缩略图会被压扁"
    assert page.tbl_products.verticalHeader().defaultSectionSize() \
        > page.THUMB_SIZE.height(), "行高要放得下缩略图"
    assert page.tbl_products.item(0, 1).text() == product.name, "文件名照旧要在"

    page.select_video(broken_id)                  # 解不出帧：没图，但不许炸
    app.processEvents()
    assert page.tbl_products.rowCount() == 1, "解不出缩略图也要照常列成品"
    assert page.tbl_products.item(0, 1).icon().isNull(), "解不出画面就不放图"
    assert broken.is_file() and real.is_file(), "取缩略图不许动磁盘上的文件"
    center.close()
    db.close()


def test_video_list_shows_thumbnails_lazily(tmp_path: Path) -> None:
    """视频库列表每行左边也要有缩略图，而且一轮只解看得见的那几行（不许开着就全解）。"""
    import cv2                                        # noqa: PLC0415
    import numpy as np                                # noqa: PLC0415

    cfg, db = make_project(tmp_path)
    real = cfg.path("input_dir") / "list.mp4"
    writer = cv2.VideoWriter(str(real), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    assert writer.isOpened(), "这台机器写不出 mp4，测不了缩略图"
    for step in range(20):
        writer.write(np.full((90, 160, 3), 10 * step + 20, dtype=np.uint8))
    writer.release()
    real_id = db_repo.upsert_video(db, real)
    for index in range(12):                     # 凑够行数，才看得出「一轮只解一批」
        video_row(cfg, db, f"bulk{index}.mp4")

    app, center = _center(cfg)
    page = center.videos
    page.tbl_videos.resize(700, 600)            # 让 viewport 真的能放下十几行
    app.processEvents()
    page._thumbs.clear()
    page.paint_visible_thumbs()                 # 一轮
    assert len(page._thumbs) <= page.THUMB_BATCH, \
        f"一轮最多解 {page.THUMB_BATCH} 帧，实际解了 {len(page._thumbs)}"

    for _ in range(6):                          # 后续几轮把剩下的补齐
        page.paint_visible_thumbs()
    # 能解码的那个视频：单独筛出来（它一定在可见范围里），列表左边必须有图
    page.edit_search.setText("list.mp4")
    page.reload()
    app.processEvents()
    assert page.tbl_videos.rowCount() == 1, "搜索只该剩这一个视频"
    assert page.tbl_videos.item(0, 0).text() == str(real_id), "筛出来的得是那个真视频"
    page.paint_visible_thumbs()
    line = 0
    assert not page.tbl_videos.item(line, page.NAME_COLUMN).icon().isNull(), \
        "能解码的视频，列表里必须看到缩略图"
    assert page.tbl_videos.iconSize().width() == page.THUMB_SIZE.width(), \
        "视频列表的图标尺寸也得和 THUMB_SIZE 一致"
    assert page.tbl_videos.verticalHeader().defaultSectionSize() > page.THUMB_SIZE.height(), \
        "视频列表的行高要放得下缩略图"
    assert page.tbl_videos.item(line, page.NAME_COLUMN).text() == real.name, "文件名照旧要在"
    center.close()
    db.close()


# ------------------------------------------------------------------ 直接跑
# ------------------------------------------------------------------ T30
def test_every_json_waits_for_its_own_product(tmp_path: Path) -> None:
    """一份 JSON 一个成品：`assets_without_product` 就是自动剪辑的待剪清单。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "d6.mp4")
    first = db_assets.create_asset(db, vid, payload(), provider="gemini", model="m")
    second = db_assets.create_asset(db, vid, payload(20.0, 27.0), provider="gemini", model="m")
    third = db_assets.create_asset(db, vid, payload(30.0, 36.0), provider="gemini", model="m")

    def pending():
        return [int(r["id"]) for r in db_assets.assets_without_product(db, vid)]

    assert pending() == [first, second, third], "三份都还没出成品，按登记顺序排"

    made = cfg.path("output_dir") / "d6_689.mp4"
    made.write_bytes(b"z" * 2048)
    db_assets.record_product(db, vid, made,
                             specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
                             asset_id=first)
    assert pending() == [second, third], "出过成品的那份不再排队"
    assert db_assets.product_progress(db, [vid]) == {vid: (1, 3)}, \
        "三份 JSON 只出了一个成品，面板要显示 1/3，不能算完成"

    db_assets.delete_asset(db, second)
    assert pending() == [third], "软删的不算"

    made.unlink()                       # 成品被手工删掉
    db_assets.sync_product_presence(db)
    assert pending() == [first, third], "成品没了，这份 JSON 重新回到待剪"

    # 名字撞了不让位也不覆盖：自动剪辑那边看到文件已存在就跳过这一段
    taken = cfg.path("output_dir") / "d6_555.mp4"
    taken.write_bytes(b"z" * 1024)
    assert taken.exists() and clip_mod.default_target(
        cfg.path("output_dir"), video, 5.55).name == "d6_555.mp4"

    # 丢失的成品记录可以清掉：只删记录，clips 退回 planned
    alive = cfg.path("output_dir") / "d6_900.mp4"
    alive.write_bytes(b"z" * 1024)
    db_assets.record_product(db, vid, alive,
                             specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
                             asset_id=third)
    db_assets.sync_product_presence(db)
    assert len(db_assets.products_overview(db, vid)) == 2, "一条丢失 + 一条在盘上"
    assert db_assets.forget_missing_products(db, vid) == 1
    left = db_assets.products_overview(db, vid)
    assert [Path(i["path"]).name for i in left] == ["d6_900.mp4"], "只剩还在盘上的那条"
    assert db_assets.forget_missing_products(db, vid) == 0, "没有丢失的就什么都不删"
    # 全库版一次清完；目录整个不在（外接盘掉线）的一条都不许动
    ghost = cfg.path("output_dir") / "d6_111.mp4"
    ghost.write_bytes(b"z" * 512)
    db_assets.record_product(db, vid, ghost,
                             specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
                             asset_id=third)
    ghost.unlink()
    offline = tmp_path / "没挂上的盘" / "d6_222.mp4"
    db_repo.register_artifact(db, vid, "final_video", offline)
    db_assets.sync_product_presence(db)
    assert db_assets.purge_missing_products(db) == 1, "只清目录还在、文件没了的那条"
    left_paths = {Path(i["path"]).name for i in db_assets.products_overview(db, vid)}
    assert left_paths == {"d6_900.mp4", "d6_222.mp4"}, \
        "盘没挂上的那条记录留着，血缘不许白丢"
    db.close()


# ------------------------------------------------------------------ T31
def test_product_clips_stop_claiming_other_json(tmp_path: Path) -> None:
    """实际渲染区间只能是自己剪出来的：别的 JSON 的片段被认领了要能修回去。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "d7.mp4")
    first_result = db_repo.save_ai_result(db, vid, json_data=payload(), validated=True)
    second_result = db_repo.save_ai_result(db, vid, json_data=payload(20.0, 27.0),
                                          validated=True)
    first = db_assets.create_asset(db, vid, payload(), ai_result_id=first_result)
    second = db_assets.create_asset(db, vid, payload(20.0, 27.0),
                                    ai_result_id=second_result)

    made = cfg.path("output_dir") / "d7_689.mp4"
    made.write_bytes(b"z" * 2048)
    info = db_assets.record_product(
        db, vid, made, specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}],
        asset_id=first)
    # 模拟老版本的张冠李戴：第二份 JSON 的片段也被写上了第一个成品的路径
    stray = db_repo.create_clip(db, vid, {"start": 20.0, "end": 27.0, "duration": 7.0},
                                ai_result_id=second_result, status="rendered",
                                output_path=made)
    assert len(db_assets.clips_for_product(db, vid, made)) == 2, "两条都指着同一个成品"

    assert db_assets.repair_product_clips(db, vid) == 1
    left = db_assets.clips_for_product(db, vid, made)
    assert len(left) == 1 and float(left[0]["end_time"]) == 13.0, \
        "只剩自己剪出来的那条"
    back = db.one("SELECT status, output_path FROM clips WHERE id = ?", (stray,))
    assert back["status"] == "planned" and back["output_path"] is None, \
        "被退回去的片段回到待剪，不再指着别人的成品"
    assert db_assets.repair_product_clips(db, vid) == 0, "修过一次就没得修了"
    # 同名重剪（删掉成品再剪、时长又一样）：登记前先退回旧记录，不许假装成这次的区间
    again = db_repo.create_clip(db, vid, {"start": 30.0, "end": 41.0, "duration": 11.0},
                                ai_result_id=second_result, status="rendered",
                                output_path=made)
    assert db_assets.detach_product_clips(db, vid, made) == 2, \
        "指着这个路径的旧记录（含自己那条）全退回，交给这次渲染重新写"
    assert db_assets.clips_for_product(db, vid, made) == []
    back_again = db.one("SELECT status FROM clips WHERE id = ?", (again,))
    assert back_again["status"] == "planned"
    # 血缘里的实际渲染跟着变干净
    spans = db_assets.lineage_spans(db, int(info["artifact_id"]))
    assert spans["actual"] == [], "退回之后由这次渲染补建，不留旧值"
    assert int(second) > 0                    # 第二份 JSON 本身一个字没动
    db.close()


# ------------------------------------------------------------------ T32
def test_products_rename_to_the_duration_rule(tmp_path: Path) -> None:
    """批量改名：老名字按实际区间长度改成规范名，文件和库（血缘）一起走。"""
    cfg, db = make_project(tmp_path)
    _video, vid = video_row(cfg, db, "d8.mp4")
    asset = db_assets.create_asset(db, vid, payload(4.0, 13.0), provider="gemini", model="m")
    old = cfg.path("output_dir") / "d8_高光时刻.mp4"
    old.write_bytes(b"z" * 2048)
    made = db_assets.record_product(
        db, vid, old, specs=[{"start": 4.0, "end": 13.0, "duration": 9.0}], asset_id=asset)

    plan = db_assets.product_rename_plan(db, [vid])
    assert len(plan) == 1 and plan[0]["skip"] is None
    assert Path(plan[0]["new"]).name == "d8_900.mp4", "9.0 秒 → _900"
    assert Path(plan[0]["old"]) == old, "计划只算不动手"
    assert old.is_file()

    done, failed = db_assets.apply_product_rename(db, plan)
    assert (done, failed) == (1, [])
    new = cfg.path("output_dir") / "d8_900.mp4"
    assert new.is_file() and not old.exists(), "文件真的改名了"
    # 库跟着走：artifacts.path 和 clips.output_path 都指新名字，血缘不断
    assert db_repo.artifact_path(db, vid, "final_video") == new
    assert [Path(i["path"]).name for i in db_assets.products_overview(db, vid)] == ["d8_900.mp4"]
    spans = db_assets.lineage_spans(db, int(made["artifact_id"]))
    assert [s["end"] for s in spans["actual"]] == [13.0], "实际渲染区间还查得到"
    trace = db_assets.artifact_lineage(db, int(made["artifact_id"]))
    assert trace is not None and int(trace["asset"]["id"]) == asset, "来源 JSON 还挂着"
    # 名字已经对了就不再动
    again = db_assets.product_rename_plan(db, [vid])
    assert again[0]["skip"] == "名字已经对了"
    assert db_assets.apply_product_rename(db, again) == (0, [])

    # 同名撞车：两个成品剪出同样时长时后者覆盖前者，被顶掉的记录一并清掉
    twin = db_assets.create_asset(db, vid, payload(20.0, 29.0), provider="gemini", model="m")
    other = cfg.path("output_dir") / "d8_高光时刻_方案 B.mp4"
    other.write_bytes(b"z" * 1024)
    db_assets.record_product(
        db, vid, other, specs=[{"start": 20.0, "end": 29.0, "duration": 9.0}],
        asset_id=twin)
    plan2 = [i for i in db_assets.product_rename_plan(db, [vid]) if not i["skip"]]
    assert len(plan2) == 1 and plan2[0]["overwrite"] is True, "撞名标成覆盖，不再跳过"
    assert db_assets.apply_product_rename(db, plan2) == (1, [])
    left = db_assets.products_overview(db, vid)
    assert [Path(i["path"]).name for i in left] == ["d8_900.mp4"], \
        "盘上只剩一个文件，库里就只剩一条记录"
    assert not other.exists()
    db.close()


def test_stale_products_are_reported_not_silently_reshaped(work: Path) -> None:
    """成品是用**别的**静音配置渲的，提取时要报出来 —— 不能悄悄换一套坐标。

    提取侧的 `keeps_for` 是现算的：它假设「渲染那会儿用的就是现在这套配置」。
    改了配置又没重剪，导出的 gaps / 挂字时间描述的就是一个还没渲出来的成品。
    """
    from vidscribe.highlight.extract import stale_note, stale_seconds

    region = (10.0, 20.0)          # 区间 10 秒
    # 一、库里记着 10.0（渲的时候没剪静音），现在也算「不剪」-> 对得上
    assert stale_seconds(10.0, None, region) == 0.0
    # 帧对齐的零点几帧误差不算对不上
    assert stale_seconds(10.02, None, region) == 0.0
    # 二、库里记着 10.0，但现在算出来要剪成 7 秒 -> 对不上，差 3 秒
    assert stale_seconds(10.0, [(10.0, 13.0), (16.0, 20.0)], region) == 3.0
    # 三、库里记着 7.0（渲的时候剪过），现在算「不剪」-> 也对不上
    assert stale_seconds(7.0, None, region) == 3.0
    # 四、老数据没记时长 / 区间不成立 -> 判断不了，不瞎报
    assert stale_seconds(None, None, region) == 0.0
    assert stale_seconds(0, None, region) == 0.0
    assert stale_seconds(10.0, None, None) == 0.0
    assert stale_seconds(10.0, None, (20.0, 10.0)) == 0.0

    assert stale_note(0, 5) == "", "一条都没问题时不许弹废话"
    note = stale_note(2, 5)
    assert "2/5" in note and "重剪" in note, note


TESTS = (
    test_stale_products_are_reported_not_silently_reshaped,
    test_multiple_assets_never_overwrite,
    test_raw_json_is_always_the_ai_original,
    test_edit_opens_a_new_asset,
    test_in_place_edit_keeps_raw,
    test_offsets_fork_a_new_version,
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
    test_prm_text_lives_in_the_database,
    test_prompts_follow_the_prm_usage,
    test_no_profiles_falls_back_to_the_old_path,
    test_migration_matches_a_fresh_v4_database,
    test_upgrade_only_adds,
    test_center_rows_aggregates_and_filters,
    test_center_rows_ignores_deleted_assets,
    test_prm_copy_and_restore,
    test_center_rows_combines_json_and_product,
    test_deleted_product_file_drops_out_of_the_counts,
    test_every_json_waits_for_its_own_product,
    test_product_clips_stop_claiming_other_json,
    test_products_rename_to_the_duration_rule,
    test_batch_apis_replace_per_row_queries,
    test_auto_step_prefers_the_library_over_ai,
    test_product_registration_records_its_source,
    test_bridge_hands_every_txt_to_the_extension,
    test_extension_verifies_every_attachment,
    test_panel_saves_the_new_switches,
    test_center_fits_its_own_minimum_window,
    test_clear_filters_brings_the_whole_list_back,
    test_forget_video_removes_the_records_but_keeps_the_file,
    test_product_rows_show_the_video_thumbnail,
    test_video_list_shows_thumbnails_lazily,
    test_moment_list_becomes_one_asset_per_line,
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
