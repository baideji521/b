"""生产链路闭环（Phase 7 Batch 11）。

盯的是 Batch 10 收尾时暴露的两个断点：

  1. `videos.duration` 一直是 NULL，资产中心只能显示「时长 -」；
  2. 只用 JSON 剪的那条路**不写 clips**，所以成品查不到「实际剪辑区间」，
     血缘断在「AI 原始区间」那一层。

覆盖：
  T1  视频时长：探到就写库、已有值不覆盖、探不到保持 NULL、资产中心读得到
  T2  JSON → 成品：一份现成 JSON 能直接剪出成品并记账
  T3  这条路 0 次 AI：ai_tasks / ai_results 一行都不长，代码里也没有 AI 调用
  T4  Clip Engine 必经：GUI 与 CLI 都过 plan_clips，没有旁路
  T5  实际 clip 登记：clips.start/end 就是实际渲染区间，不是 AI 原值
  T6  渲染没通过完整性检查就不登记（成品闸门在记账之前）
  T7  一份 JSON 配两版 PRM = 两个成品，各自的 clips 与 PRM 都对得上
  T8  一个视频两份 JSON，各自都能剪出自己的成品
  T9  成品反查：视频 / JSON / AI / 模型 / PRM / AI 原始区间 / 实际渲染区间
  T10 生成成品前后 `raw_json` 一个字节都没变

功能测试用临时目录里的临时库，**绝不碰项目真实数据库**；渲染这类叶子调用不真跑，
只验证记账链路。结构性断言（T4/T6）直接读源码 AST，写清楚是源码级检查。
可以直接 `python tests/test_production_chain.py`。
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_highlight_assets import Win, make_project, video_row   # noqa: E402

from vidscribe import video_io                                   # noqa: E402
from vidscribe.db import assets as db_assets                     # noqa: E402
from vidscribe.db import repo as db_repo                         # noqa: E402
from vidscribe.highlight import clip_engine                      # noqa: E402


# ------------------------------------------------------------------ 小道具
def ai_payload(start: float = 8.23, end: float = 23.49, score: float = 0.91,
               video: str = "v.mp4") -> dict:
    """一份典型的 AI 高光 JSON（带 overlays，能过 parse_spec 那一关）。"""
    return {"video": video,
            "clip": {"start": start, "end": end, "duration": round(end - start, 3),
                     "score": score, "type": "hook", "reason": "赌注揭晓",
                     "overlays": {"comment": {"time": end, "text": "no way",
                                              "kind": "comment"},
                                  "evaluation": "节奏明快"}}}


class FakeInfo:
    """probe_video 的返回值替身。"""

    def __init__(self, duration: float):
        self.duration = duration
        self.fps = 30.0
        self.width = 1080
        self.height = 1920


class FakeProbe:
    """记数版 probe_video：可以设成探不到（抛异常）。"""

    def __init__(self, duration: float | None = 42.5):
        self.duration = duration
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        if self.duration is None:
            raise RuntimeError("这不是视频")
        return FakeInfo(self.duration)


class FakeWorker:
    """渲染线程替身：交出实际剪的区间 + 每段的记账行。"""

    def __init__(self, plans_and_ranges):
        self.cut_ranges = [(start, end) for _plan, start, end in plans_and_ranges]
        self.cut_specs = [db_assets.clip_spec_for(plan, start, end)
                          for plan, start, end in plans_and_ranges]


def one_plan(payload: dict, start: float, end: float):
    """拿真引擎算一条计划（没有逐词数据时就是 AI 原区间 + 15 秒上限）。"""
    result = clip_engine.plan_clips(payload, ())
    assert result.plans, "引擎至少要给出一条计划"
    return result.plans[0], start, end


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.AST:
    tree = ast.parse(source)
    found = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    assert found, f"源码里找不到 {name}"
    return found[0]


def _method(source: str, cls: str, name: str) -> ast.AST:
    tree = ast.parse(source)
    holder = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls]
    assert holder, f"源码里找不到 class {cls}"
    found = [n for n in holder[0].body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert found, f"{cls} 里找不到 {name}"
    return found[0]


def _call_names(node: ast.AST) -> list[str]:
    """函数体内的调用名，按出现顺序（ast.walk 是广度优先，得自己排）。"""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    names = []
    for call in calls:
        target = call.func
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)
    return names


# ------------------------------------------------------------------ T1
def test_duration_is_filled_once_and_never_overwritten(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "dur.mp4")
    assert db.value("SELECT duration FROM videos WHERE id = ?", (vid,)) is None, \
        "刚登记的视频还没探过时长"

    probe = FakeProbe(42.5)
    original = video_io.probe_video
    video_io.probe_video = probe
    try:
        assert db_repo.ensure_duration(db, vid, video) == 42.5
        assert float(db.value("SELECT duration FROM videos WHERE id = ?", (vid,))) == 42.5
        assert probe.calls == 1

        # 已有值：不再探、也不覆盖
        assert db_repo.ensure_duration(db, vid, video) == 42.5
        assert probe.calls == 1, "库里已经有时长就不该再探一次"

        rows = db_assets.center_rows(db)
        mine = next(r for r in rows if r["id"] == vid)
        assert float(mine["duration"]) == 42.5, "资产中心的时长只来自库里这一列"

        # 探不到：保持 NULL，安全失败
        _video2, vid2 = video_row(cfg, db, "bad.mp4")
        video_io.probe_video = FakeProbe(None)
        assert db_repo.ensure_duration(db, vid2, cfg.path("input_dir") / "bad.mp4") is None
        assert db.value("SELECT duration FROM videos WHERE id = ?", (vid2,)) is None

        # 补齐入口只碰空的那些
        video_io.probe_video = FakeProbe(7.5)
        stats = db_repo.fill_missing_durations(db)
        assert stats["filled"] == 1 and stats["checked"] == 1, stats
        assert float(db.value("SELECT duration FROM videos WHERE id = ?", (vid,))) == 42.5, \
            "补齐不许改已有的值"
    finally:
        video_io.probe_video = original
    db.close()


# ------------------------------------------------------------------ T2 / T5
def test_json_render_registers_product_and_real_clip(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t2.mp4")
    payload = ai_payload(video=video.name)
    asset = db_assets.create_asset(db, vid, payload, source_type="imported")
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", make_default=True)

    product = cfg.path("output_dir") / "t2_方案 A_PRM V1.mp4"
    product.write_bytes(b"z" * 4096)
    plan, start, end = one_plan(payload, 8.23, 19.39)
    info = db_assets.record_product(
        db, vid, product, specs=[db_assets.clip_spec_for(plan, start, end)],
        asset_id=asset, prm_id=prm)

    assert info["artifact_id"] and len(info["clip_ids"]) == 1
    clips = db_assets.clips_for_product(db, vid, product)
    assert len(clips) == 1
    clip = clips[0]
    assert float(clip["start_time"]) == 8.23 and float(clip["end_time"]) == 19.39, \
        "clips 里必须是实际渲染区间"
    assert float(clip["duration"]) == 11.16
    assert clip["status"] == "rendered" and clip["output_path"] == str(product)
    assert clip["reason"] == "赌注揭晓" and clip["evaluation"] == "节奏明快", "文案不许被改写"
    assert float(db_assets.asset_payload(db, asset)["clip"]["end"]) == 23.49, \
        "JSON 里的 AI 原始区间一个字不动"
    db.close()


# ------------------------------------------------------------------ T3
def test_json_render_never_touches_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t3.mp4")
    payload = ai_payload(video=video.name)
    asset = db_assets.create_asset(db, vid, payload, source_type="imported")
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", make_default=True)

    product = cfg.path("output_dir") / "t3_成品.mp4"
    product.write_bytes(b"z" * 2048)
    plan, start, end = one_plan(payload, 8.23, 19.39)
    db_assets.record_product(db, vid, product,
                             specs=[db_assets.clip_spec_for(plan, start, end)],
                             asset_id=asset, prm_id=prm)

    assert db.value("SELECT COUNT(*) FROM ai_tasks", ()) == 0, "这条路不许建 AI 任务"
    assert db.value("SELECT COUNT(*) FROM ai_results", ()) == 0, "这条路不许写 AI 结果"

    banned = ("dispatch_ai", "send_file_to_ai", "enqueue_ai_task", "create_ai_task",
              "save_ai_result", "_save_ai_result")
    for name in ("record_product", "record_clips", "clip_spec_for"):
        body = ast.dump(_function(_source("src/vidscribe/db/assets.py"), name))
        for bad in banned:
            assert bad not in body, f"{name} 不该碰 {bad}"
    render = _call_names(_function(_source("src/vidscribe/cli.py"), "_assets_render"))
    for bad in banned:
        assert bad not in render, f"assets --render 不该调 {bad}"
    db.close()


# ------------------------------------------------------------------ T4
def test_clip_engine_is_on_every_path(tmp_path: Path) -> None:
    """GUI 和 CLI 都必须过 plan_clips，谁都不许拿 JSON 的 start/end 直接渲染。"""
    cli = _source("src/vidscribe/cli.py")
    plans = _call_names(_function(cli, "_clip_plans"))
    assert "plan_clips" in plans, "CLI 的计划只能由引擎给"
    render = _call_names(_function(cli, "_assets_render"))
    assert "_clip_plans" in render and "payload_for" in render, \
        "CLI 渲染必须用引擎修正后的 payload"
    assert render.index("_clip_plans") < render.index("render_highlight")

    gui = _source("src/vidscribe/gui/main_window.py")
    run = _call_names(_method(gui, "HighlightWorker", "run"))
    assert "_plan_result" in run and "payload_for" in run, "GUI 渲染同样只走引擎"
    assert run.index("_plan_result") < run.index("render_highlight")
    planner = _call_names(_method(gui, "HighlightWorker", "_plan_result"))
    assert "plan_clips" in planner


# ------------------------------------------------------------------ T6
def test_failed_render_registers_nothing(tmp_path: Path) -> None:
    """完整性检查没过就不该有任何记账；记账一定排在闸门后面（源码级）。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t6.mp4")
    payload = ai_payload(video=video.name)
    asset = db_assets.create_asset(db, vid, payload, source_type="imported")

    # 渲染没成功 -> 谁都不会调 record_product，库里应当干净
    assert db.value("SELECT COUNT(*) FROM artifacts", ()) == 0
    assert db.value("SELECT COUNT(*) FROM clips", ()) == 0
    assert db_assets.products_for_asset(db, asset) == []

    cli = _call_names(_function(_source("src/vidscribe/cli.py"), "_assets_render"))
    assert cli.index("is_complete_video") < cli.index("record_product"), \
        "先验成片，再记账"
    gui = _call_names(_method(_source("src/vidscribe/gui/main_window.py"),
                              "HighlightWorker", "run"))
    assert gui.index("is_complete_video") < gui.index("clip_spec_for"), \
        "GUI 也是先验成片，再记实际区间"
    db.close()


# ------------------------------------------------------------------ T7
def test_one_json_two_prms_two_products(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t7.mp4")
    payload = ai_payload(video=video.name)
    asset = db_assets.create_asset(db, vid, payload, source_type="imported")
    first = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", make_default=True)
    second = db_assets.create_prm(db, "PRM V2", "prm/prm_zh.txt")

    made = []
    for prm_id, name, span in ((first, "PRM V1", (8.23, 19.39)),
                               (second, "PRM V2", (8.23, 17.90))):
        product = cfg.path("output_dir") / f"t7_方案 A_{name}.mp4"
        product.write_bytes(b"z" * 1024)
        plan, start, end = one_plan(payload, *span)
        db_assets.record_product(db, vid, product,
                                 specs=[db_assets.clip_spec_for(plan, start, end)],
                                 asset_id=asset, prm_id=prm_id)
        made.append(product)

    products = db_assets.products_for_asset(db, asset)
    assert len(products) == 2, "一份 JSON 配两版 PRM 就该有两个成品"
    assert {int(r["prm_id"]) for r in products} == {first, second}
    assert all(p.is_file() for p in made), "两个成品文件都要在，不能互相覆盖"
    assert float(db_assets.clips_for_product(db, vid, made[0])[0]["end_time"]) == 19.39
    assert float(db_assets.clips_for_product(db, vid, made[1])[0]["end_time"]) == 17.90
    db.close()


# ------------------------------------------------------------------ T8
def test_one_video_many_jsons_each_renders(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t8.mp4")
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", make_default=True)
    first_payload = ai_payload(8.23, 23.49, video=video.name)
    second_payload = ai_payload(40.0, 52.0, score=0.84, video=video.name)
    first = db_assets.create_asset(db, vid, first_payload, source_type="imported")
    second = db_assets.create_asset(db, vid, second_payload, source_type="imported")

    for asset, payload, span in ((first, first_payload, (8.23, 19.39)),
                                 (second, second_payload, (40.0, 51.2))):
        product = cfg.path("output_dir") / f"t8_#{asset}.mp4"
        product.write_bytes(b"z" * 1024)
        plan, start, end = one_plan(payload, *span)
        db_assets.record_product(db, vid, product,
                                 specs=[db_assets.clip_spec_for(plan, start, end)],
                                 asset_id=asset, prm_id=prm)

    assert len(db_assets.products_for_asset(db, first)) == 1
    assert len(db_assets.products_for_asset(db, second)) == 1
    assert len(db_repo.get_clips(db, vid)) == 2, "两份 JSON 各自登记自己的实际区间"
    assert db_assets.asset_counts(db, [vid])[vid] == 2
    db.close()


# ------------------------------------------------------------------ T9 / T10
def test_full_lineage_and_untouched_raw_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "t9.mp4")
    payload = ai_payload(video=video.name)
    task_id, _created = db_repo.enqueue_ai_task(db, vid, mode="full", provider="gemini")
    result_id = db_repo.save_ai_result(db, vid, task_id=task_id, json_data=payload,
                                      candidate_count=1, winner_score=0.91, validated=True)
    asset = db_assets.create_asset(db, vid, payload, source_type="ai",
                                   provider="gemini", model="gemini-2.5-flash",
                                   ai_result_id=result_id, source_task_id=task_id)
    prm = db_assets.create_prm(db, "PRM V2", "prm/prm_en.txt", version="V2",
                               make_default=True)
    before = db.value("SELECT raw_json FROM highlight_assets WHERE id = ?", (asset,))

    product = cfg.path("output_dir") / "t9_方案 A_PRM V2.mp4"
    product.write_bytes(b"z" * 4096)
    plan, start, end = one_plan(payload, 8.23, 19.39)
    made = db_assets.record_product(
        db, vid, product, specs=[db_assets.clip_spec_for(plan, start, end)],
        asset_id=asset, prm_id=prm)

    trace = db_assets.artifact_lineage(db, made["artifact_id"])
    assert trace is not None
    assert trace["video"]["file_name"] == video.name
    assert int(trace["asset"]["id"]) == asset
    assert trace["provider"] == "gemini" and trace["model"] == "gemini-2.5-flash"
    assert int(trace["prm"]["id"]) == prm and trace["prm"]["version"] == "V2"

    ai_clip = db_repo.clips_from_payload(db_assets.asset_payload(db, asset))[0]
    assert (ai_clip["start"], ai_clip["end"]) == (8.23, 23.49), "AI 原始区间"
    engine = clip_engine.plan_clips(db_assets.asset_payload(db, asset), ()).plans[0]
    assert (engine.ai_start, engine.ai_end) == (8.23, 23.49)
    assert engine.end <= 23.49, "引擎只会往回收，不会凭空拉长"
    actual = db_assets.clips_for_product(db, vid, product)[0]
    assert (float(actual["start_time"]), float(actual["end_time"])) == (8.23, 19.39), \
        "实际渲染区间"

    after = db.value("SELECT raw_json FROM highlight_assets WHERE id = ?", (asset,))
    assert after == before, "生成成品前后 raw_json 必须一模一样"
    db.close()


# ------------------------------------------------------------------ GUI 那条路
def test_gui_register_final_video_creates_clips(tmp_path: Path) -> None:
    """只用 JSON 剪的 GUI 路径：库里本来没有 planned 行，成品之后要补出 rendered 行。"""
    cfg, db = make_project(tmp_path)
    video, vid = video_row(cfg, db, "gui.mp4")
    payload = ai_payload(video=video.name)
    asset = db_assets.create_asset(db, vid, payload, source_type="imported")
    prm = db_assets.create_prm(db, "PRM V1", "prm/prm_en.txt", make_default=True)

    product = cfg.path("output_dir") / "gui_方案 A_PRM V1.mp4"
    product.write_bytes(b"z" * 4096)
    win = Win(cfg, db)
    win._auto_video = video
    win._last_asset_id = asset
    win._last_prm_id = prm
    win.clip_worker = FakeWorker([one_plan(payload, 8.23, 19.39)])
    win._register_final_video(str(product))

    clips = db_assets.clips_for_product(db, vid, product)
    assert len(clips) == 1, "JSON 直接剪也要留下 clips"
    assert (float(clips[0]["start_time"]), float(clips[0]["end_time"])) == (8.23, 19.39)
    assert clips[0]["status"] == "rendered"
    products = db_assets.products_for_asset(db, asset)
    assert len(products) == 1 and int(products[0]["prm_id"]) == prm
    assert db.value("SELECT COUNT(*) FROM ai_results", ()) == 0, "全程 0 次 AI"
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_duration_is_filled_once_and_never_overwritten,
    test_json_render_registers_product_and_real_clip,
    test_json_render_never_touches_ai,
    test_clip_engine_is_on_every_path,
    test_failed_render_registers_nothing,
    test_one_json_two_prms_two_products,
    test_one_video_many_jsons_each_renders,
    test_full_lineage_and_untouched_raw_json,
    test_gui_register_final_video_creates_clips,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="chain_"))
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
