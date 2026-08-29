"""自动剪辑任务与产物生命周期回归测试（Phase 7）。

盯的是"数据库状态、AI 结果、片段、产物、磁盘"这五样最终必须对得上，具体覆盖：

1. enqueue 幂等：同一个视频 + 同一种任务 + 同一种模式，同时只允许一条没跑完的任务
2. claim 原子性：多连接同时抢，一条任务只会被一个 worker 领走
3. heartbeat：心跳新鲜的 processing 不被误捞，心跳超时的退回 pending 且不吃重试额度
4. completed 之后心跳刷不动（别让已完成的任务被"弄活"）
5. max_attempts：1 次就失败定格，2 次时先退回再定格
6. AI 结果与片段的外键指向正确，任务失败也不删 AI 结果
7. 产物跟着磁盘走：文件删了 reconcile 标 0，文件回来标 1
8. 成品存在性：文件不存在、0 字节都不算成品，只有真文件算
9. reconcile 反复跑不会多出产物记录，也不动历史
10. 提示词指纹：dispatch 时才记，内容变了 hash 就变，任务与结果两侧能关联
11. P1-2：开机自动恢复必须先跟磁盘对账，避免把"已经剪好但没登记"的任务白剪一遍

Batch 7 追加（状态机拆分：`waiting` = 等 AI，`processing` = 正在渲染）：
  T29 claim：pending -> uploading
  T30 uploading -> waiting（completed 不许被盖回去）
  T31 waiting/uploading -> processing；pending 与已结束状态一律不许倒退进来
  T32 processing -> completed
  T33 AI JSON 解不开 -> failed/pending，绝不进 processing（含生产代码闸门顺序）
  T34 AI JSON 抠不出片段 -> failed/pending，绝不进 processing
  T35 waiting 能刷心跳
  T36 processing 能刷心跳
  T37 waiting 孤儿 -> pending
  T38 processing 孤儿 -> pending
  T39 uploading 孤儿 -> pending
  T40 孤儿回收不碰本进程正在跑的任务
  T41 一堆历史任务不重复影响视频统计


测试只用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_task_lifecycle.py`，也可以 `pytest tests/test_task_lifecycle.py`。
"""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.config import Config                      # noqa: E402
from vidscribe.db import open_db                         # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db.importer import refresh_from_disk, reconcile  # noqa: E402
from vidscribe.db.schema import TASK_ACTIVE              # noqa: E402

MAIN_WINDOW = ROOT / "src" / "vidscribe" / "gui" / "main_window.py"


# ------------------------------------------------------------------ 测试夹具
def make_project(tmp_path: Path):
    """在临时目录里搭一个独立工程：库、输入、输出、AI 输出全在 tmp 下。"""
    for sub in ("database", "input", "output", "logs", "ai_out", "cache"):
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
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    # 安全阀：库文件必须在临时目录里，绝不能连到真实工程库上
    assert str(cfg.path("db_dir")).startswith(str(tmp_path))
    return cfg, db


def fake_video(cfg, name: str) -> Path:
    """造一个"够真"的视频文件：指纹靠文件内容算，所以每个文件字节必须不同。"""
    path = cfg.path("input_dir") / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def real_mp4(path: Path, frames: int = 6, size: int = 64, fps: int = 25) -> Path:
    """现场编一份最小的合法 mp4（无声）。

    成品这一种 artifact 登记前要过 `video_io.is_complete_video`（P1-4），所以"盘上有个
    非空文件"已经不够用了，必须是一份真封装完整的视频。不下载素材、不依赖 test.mp4。
    """
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        stream.options = {"crf": "30", "preset": "ultrafast"}
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((size, size, 3), (i * 9) % 256, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path



# ------------------------------------------------------------------ 1 幂等
def test_enqueue_is_idempotent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "idem.mp4"))
    first, created = db_repo.enqueue_ai_task(db, vid, mode="full")
    assert created is True

    for _ in range(10):  # 连点十次
        again, made = db_repo.enqueue_ai_task(db, vid, mode="full")
        assert again == first and made is False
    assert db_repo.queue_counts(db)["open"] == 1

    for state in TASK_ACTIVE:  # 跑着一半再点也一样
        db_repo.update_ai_task(db, first, status=state)
        again, made = db_repo.enqueue_ai_task(db, vid, mode="full")
        assert again == first and made is False, state

    try:  # 绕过 enqueue 硬插也得被唯一索引挡住
        db_repo.create_ai_task(db, vid, mode="full")
        raise AssertionError("唯一索引没挡住第二条没跑完的任务")
    except sqlite3.IntegrityError:
        pass

    other, made = db_repo.enqueue_ai_task(db, vid, mode="collect")
    assert made is True and other != first, "不同模式应该各自一条"

    db_repo.complete_ai_task(db, first)
    fresh, made = db_repo.enqueue_ai_task(db, vid, mode="full")
    assert made is True and fresh != first, "完成之后应该允许再排一条新的"
    assert len(db.all("SELECT id FROM ai_tasks WHERE video_id = ?", (vid,))) >= 3, "历史不能被覆盖"
    db.close()


# ------------------------------------------------------------------ 2 claim
def test_claim_is_atomic(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    for i in range(12):
        vid = db_repo.upsert_video(db, fake_video(cfg, "mw%02d.mp4" % i))
        db_repo.enqueue_ai_task(db, vid, mode="mw")

    grabbed: list[int] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        own = open_db(cfg)  # 每个 worker 自己的连接
        try:
            while True:
                task = db_repo.claim_next_ai_task(own, mode="mw", worker_id=name)
                if task is None:
                    return
                with lock:
                    grabbed.append(int(task["id"]))
                time.sleep(0.005)
        finally:
            own.close()

    threads = [threading.Thread(target=worker, args=("w%d" % i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(grabbed) == 12, "12 条任务应该全被领走：%d" % len(grabbed)
    assert len(set(grabbed)) == 12, "同一条任务被领了两次"
    counts = db_repo.queue_counts(db, mode="mw")
    assert counts["pending"] == 0 and counts["uploading"] == 12, counts
    db.close()


# ------------------------------------------------------------------ 3 心跳
def test_heartbeat_and_recovery(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "hb.mp4"))
    tid, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    claimed = db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    assert claimed is not None and claimed["status"] == "uploading"

    assert db_repo.touch_ai_task(db, tid) is True
    assert db_repo.recover_stale_ai_tasks(db, 30.0) == 0, "心跳新鲜的不该被捞回"

    with db.tx() as conn:  # 模拟被强杀：心跳停在很久以前
        conn.execute("UPDATE ai_tasks SET heartbeat_at = '2000-01-01T00:00:00' WHERE id = ?",
                     (tid,))
    assert db_repo.recover_stale_ai_tasks(db, 30.0) == 1
    row = db_repo.get_ai_task(db, tid)
    assert row["status"] == "pending"
    assert int(row["retry_count"] or 0) == 0, "崩溃退回不该吃重试额度"
    db.close()


# ------------------------------------------------------------------ 4 completed
def test_completed_cannot_be_touched(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "done.mp4"))
    tid, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.complete_ai_task(db, tid)
    assert db_repo.touch_ai_task(db, tid) is False, "已完成的任务不该被心跳弄活"
    assert db_repo.recover_stale_ai_tasks(db, 0.0) == 0, "已完成的任务不该被恢复逻辑碰"
    db.close()


# ------------------------------------------------------------------ 5 重试
def test_max_attempts(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    once = db_repo.upsert_video(db, fake_video(cfg, "once.mp4"))
    tid1, _ = db_repo.enqueue_ai_task(db, once, mode="full", max_attempts=1)
    assert db_repo.fail_or_requeue_ai_task(db, tid1, "boom") == "failed"

    twice = db_repo.upsert_video(db, fake_video(cfg, "twice.mp4"))
    tid2, _ = db_repo.enqueue_ai_task(db, twice, mode="full", max_attempts=2)
    assert db_repo.fail_or_requeue_ai_task(db, tid2, "boom") == "pending"
    assert db_repo.fail_or_requeue_ai_task(db, tid2, "boom again") == "failed"
    assert int(db_repo.get_ai_task(db, tid2)["retry_count"]) == 2
    db.close()


# ------------------------------------------------------------------ 6 外键
def test_ai_result_and_clip_links(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "res.mp4"))
    tid, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    payload = {"clip": {"start": 1.0, "end": 9.0, "score": 0.9, "type": "hook", "reason": "r"}}
    rid = db_repo.save_ai_result(db, vid, task_id=tid, raw_response="raw",
                                 json_data=payload, validated=True)
    specs = db_repo.clips_from_payload(payload)
    assert len(specs) == 1
    cid = db_repo.create_clip(db, vid, specs[0], ai_result_id=rid)

    res = db.one("SELECT * FROM ai_results WHERE id = ?", (rid,))
    clip = db.one("SELECT * FROM clips WHERE id = ?", (cid,))
    assert int(res["task_id"]) == tid and int(res["video_id"]) == vid
    assert int(clip["ai_result_id"]) == rid and int(clip["video_id"]) == vid
    assert clip["status"] == "planned" and clip["output_path"] is None

    # JSON 里没有可用 clip：结果照样留档，一条 clip 都不建
    bad = db_repo.save_ai_result(db, vid, task_id=tid, raw_response="no clip",
                                 json_data={"note": "nothing"})
    assert db_repo.clips_from_payload({"note": "nothing"}) == []
    assert db.one("SELECT id FROM ai_results WHERE id = ?", (bad,)) is not None

    # 任务失败也绝不删 AI 结果（AI 真的回过话，这是事实）
    db_repo.fail_or_requeue_ai_task(db, tid, "clip 建不出来")
    assert db_repo.get_ai_task(db, tid)["status"] == "failed"
    assert db_repo.get_ai_result(db, vid) is not None
    db.close()


# ------------------------------------------------------------------ 7 产物对账
def test_artifact_follows_disk(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "art.mp4")
    vid = db_repo.upsert_video(db, video)
    product = cfg.path("output_dir") / "art_高光时刻.mp4"
    product.write_bytes(b"x" * 2048)
    aid = db_repo.register_artifact(db, vid, "final_video", product)
    row = db.one("SELECT * FROM artifacts WHERE id = ?", (aid,))
    assert int(row["exists_on_disk"]) == 1 and int(row["size"]) == 2048

    product.unlink()
    changed = reconcile(cfg, db)
    assert changed["artifacts_gone"] >= 1
    assert int(db.one("SELECT exists_on_disk FROM artifacts WHERE id = ?",
                      (aid,))["exists_on_disk"]) == 0

    product.write_bytes(b"y" * 4096)
    changed = reconcile(cfg, db)
    assert changed["artifacts_back"] >= 1
    back = db.one("SELECT * FROM artifacts WHERE id = ?", (aid,))
    assert int(back["exists_on_disk"]) == 1 and int(back["size"]) == 4096
    db.close()


# ------------------------------------------------------------------ 8 成品存在性
def test_product_readiness(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "ready.mp4"))

    missing = cfg.path("output_dir") / "never_written.mp4"
    db_repo.register_artifact(db, vid, "final_video", missing)
    assert db_repo.has_artifact(db, vid, "final_video") is False, "渲染说成了但文件没落地不算成品"

    empty = cfg.path("output_dir") / "empty.mp4"
    empty.write_bytes(b"")
    db_repo.register_artifact(db, vid, "final_video", empty)
    assert db_repo.has_artifact(db, vid, "final_video") is False, "0 字节不算成品"

    real = cfg.path("output_dir") / "real.mp4"
    real.write_bytes(b"x" * 1024)
    db_repo.register_artifact(db, vid, "final_video", real)
    assert db_repo.artifact_path(db, vid, "final_video") == real

    # 四格统计、单个视频状态、批量状态必须同一个口径
    assert db_repo.video_state(db, vid)["clipped"] is True
    assert db_repo.states_for_videos(db, [vid])[vid]["clipped"] is True
    assert db_repo.get_statistics(db, [vid])["done"] == 1

    # clip 标了 rendered 但成品被删：只能算历史，不能继续显示"有成品"
    cid = db_repo.create_clip(db, vid, {"start": 0, "end": 5})
    db_repo.update_clip(db, cid, status="rendered", output_path=real)
    real.unlink()
    reconcile(cfg, db)
    assert db_repo.video_state(db, vid)["clipped"] is False
    assert db.one("SELECT status FROM clips WHERE id = ?", (cid,))["status"] == "rendered"
    db.close()


# ------------------------------------------------------------------ 9 reconcile 幂等
def test_reconcile_is_idempotent(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "again.mp4")
    vid = db_repo.upsert_video(db, video)
    text = video.with_suffix(".txt")
    text.write_text("merged", encoding="utf-8")
    db_repo.register_artifact(db, vid, "merged_txt", text)
    rid = db_repo.save_ai_result(db, vid, json_data={"clip": {"start": 0, "end": 3}})
    cid = db_repo.create_clip(db, vid, {"start": 0, "end": 3}, ai_result_id=rid)

    before = db_repo.counts(db)
    for _ in range(3):
        reconcile(cfg, db)
    after = db_repo.counts(db)
    assert after == before, "reconcile 反复跑不该改变任何表的行数：%s -> %s" % (before, after)
    assert db.one("SELECT id FROM ai_results WHERE id = ?", (rid,)) is not None
    assert db.one("SELECT id FROM clips WHERE id = ?", (cid,)) is not None
    db.close()


# ------------------------------------------------------------------ 10 提示词指纹
def test_prompt_fingerprint(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "prompt.mp4"))
    tid, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    assert db_repo.get_ai_task(db, tid)["prompt_hash"] is None, "入队时不该有指纹"

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A" * 100, encoding="utf-8")
    first = db_repo.note_task_prompt(db, tid, prompt)   # 相当于 dispatch 那一刻
    assert len(first["prompt_hash"]) == 64 and first["prompt_size"] == 100

    prompt.write_text("B" * 200, encoding="utf-8")      # 排队期间提示词被改了
    second = db_repo.note_task_prompt(db, tid, prompt)
    assert second["prompt_hash"] != first["prompt_hash"]
    assert second["prompt_size"] == 200

    task = db_repo.get_ai_task(db, tid)
    assert task["prompt_hash"] == second["prompt_hash"], "任务记的必须是最后真正发出去那份"
    db_repo.save_ai_result(db, vid, task_id=tid, json_data={"clip": {"start": 0, "end": 3}},
                           prompt_hash=task["prompt_hash"], prompt_path=task["prompt_path"],
                           prompt_size=task["prompt_size"])
    joined = db.one(
        """
        SELECT t.prompt_hash AS th, r.prompt_hash AS rh FROM ai_tasks t
          LEFT JOIN ai_results r ON r.task_id = t.id WHERE t.id = ?
        """, (tid,))
    assert joined["th"] == joined["rh"], "任务与结果两侧的指纹必须能对上"

    manual = db_repo.save_ai_result(db, vid, task_id=None,
                                    json_data={"clip": {"start": 0, "end": 3}},
                                    **db_repo.prompt_fingerprint(prompt))
    got = db.one("SELECT prompt_hash FROM ai_results WHERE id = ?", (manual,))["prompt_hash"]
    assert got == second["prompt_hash"], "手工单发（没有任务行）也要能追溯"
    db.close()


# --------------------------------------------- 11 P1-2 恢复前必须先对账
def _resume_call_order() -> list[str]:
    """从源码里取 _resume_auto_queue 依次调了哪些函数（不 import PyQt5）。"""
    tree = ast.parse(MAIN_WINDOW.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resume_auto_queue":
            names: list[str] = []
            for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                func = call.func
                if isinstance(func, ast.Attribute):
                    names.append((func.attr, getattr(call, "lineno", 0)))
            return [name for name, _ in sorted(names, key=lambda x: x[1])]
    raise AssertionError("main_window.py 里找不到 _resume_auto_queue")


def test_resume_reconciles_before_recovery(tmp_path: Path) -> None:
    order = _resume_call_order()
    assert "_sync_disk" in order, "开机自动恢复必须先跟磁盘对账（P1-2）"
    assert "recover_stale_ai_tasks" in order
    assert order.index("_sync_disk") < order.index("recover_stale_ai_tasks"), \
        "对账必须发生在捞回 stale 任务之前，否则已剪好的成品会被白剪一遍"

    # 行为侧：成品已经落盘但库里没登记（上次崩在 register_artifact 之前），
    # 对账之后必须认得出这个成品，队列才会跳过而不是再剪一遍。
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "crashed.mp4")
    vid = db_repo.upsert_video(db, video)
    tid, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    # 显式落 processing：模拟"上次卡在渲染那一步"（领取本身只落 uploading）
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-dead", status="processing")
    assert db_repo.get_ai_task(db, tid)["status"] == "processing"

    product = Path(str(cfg.bridge["ai_output_dir"])) / (video.stem + "_高光时刻.mp4")
    real_mp4(product)   # 一份真封装完整的成品；非空假 mp4 已经不算成品了（P1-4）
    assert db_repo.artifact_path(db, vid, "final_video") is None, "登记之前库里当然没有"

    refresh_from_disk(cfg, db, folders=[cfg.path("input_dir")], ai_out=product.parent)
    found = db_repo.artifact_path(db, vid, "final_video")
    assert found == product, "对账之后必须认出这个成品"
    assert db_repo.video_state(db, vid)["clipped"] is True

    # 这就是 _auto_step 用来跳过的那个判断：有成品 -> 直接 settle completed，不再渲染
    db_repo.complete_ai_task(db, tid)
    assert db_repo.get_ai_task(db, tid)["status"] == "completed"
    assert db_repo.queue_counts(db, mode="full")["open"] == 0
    db.close()


# ============================ Batch 7：状态机拆分（等 AI ≠ 在剪） ============================
def _task(cfg, db, name: str, mode: str = "full") -> tuple[int, int]:
    """一个视频 + 一条 pending 任务。返回 (video_id, task_id)。"""
    vid = db_repo.upsert_video(db, fake_video(cfg, name))
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode, max_attempts=1)
    return vid, task_id


def _status(db, task_id: int) -> str:
    return str(db_repo.get_ai_task(db, task_id)["status"])


# ------------------------------------------------------------------ T29
def test_claim_moves_pending_to_uploading(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, "t29.mp4")
    assert _status(db, task_id) == "pending"

    claimed = db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    assert claimed is not None and int(claimed["id"]) == task_id
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "uploading", "领到手是「正在提交给 AI」，不是「正在渲染」"
    assert row["worker_id"] == "w1"
    assert row["started_at"] and row["heartbeat_at"] and row["finished_at"] is None
    assert db_repo.claim_next_ai_task(db, mode="full", worker_id="w2") is None, "不许被领第二次"
    db.close()


# ------------------------------------------------------------------ T30
def test_uploading_moves_to_waiting(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, "t30.mp4")
    assert db_repo.mark_ai_task_waiting(db, task_id) is False, "还没领就不该能标「等 AI」"

    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    assert db_repo.mark_ai_task_waiting(db, task_id) is True
    assert _status(db, task_id) == "waiting"
    assert db_repo.mark_ai_task_waiting(db, task_id) is False, "重复标不该再改一次"
    assert _status(db, task_id) == "waiting"

    # completed 绝不能被 waiting 盖回去
    db_repo.complete_ai_task(db, task_id)
    assert db_repo.mark_ai_task_waiting(db, task_id) is False
    assert _status(db, task_id) == "completed"
    db.close()


# ------------------------------------------------------------------ T31
def test_waiting_moves_to_rendering(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, "t31.mp4")
    assert db_repo.mark_ai_task_rendering(db, task_id) is False, "pending 不许直接跳去渲染"

    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, task_id)
    assert db_repo.mark_ai_task_rendering(db, task_id) is True
    assert _status(db, task_id) == "processing", "processing 从现在起只表示「正在渲染」"

    # 脚本剪辑那一串不问 AI：uploading 也可以直接进渲染
    _vid2, script_id = _task(cfg, db, "t31_script.mp4", mode="script")
    db_repo.claim_next_ai_task(db, mode="script", worker_id="w1")
    assert _status(db, script_id) == "uploading"
    assert db_repo.mark_ai_task_rendering(db, script_id) is True
    assert _status(db, script_id) == "processing"

    # 已经结束的一律不许被拉回 processing
    for name, killer in (("done", db_repo.complete_ai_task),
                         ("cancel", db_repo.cancel_ai_task)):
        _v, tid = _task(cfg, db, f"t31_{name}.mp4")
        db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
        killer(db, tid)
        before = _status(db, tid)
        assert db_repo.mark_ai_task_rendering(db, tid) is False, f"{before} 不许倒退成 processing"
        assert _status(db, tid) == before
    db.close()


# ------------------------------------------------------------------ T32
def test_rendering_moves_to_completed(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, "t32.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, task_id)
    db_repo.mark_ai_task_rendering(db, task_id)

    db_repo.complete_ai_task(db, task_id)
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "completed" and row["finished_at"]
    assert row["worker_id"] is None and row["error"] is None
    assert db_repo.touch_ai_task(db, task_id) is False, "干完的任务不该被心跳弄活"
    db.close()


def _mw_calls(func_name: str) -> list[str]:
    """main_window 里某个方法按源码顺序调用了哪些名字（只看名字，不执行 GUI）。"""
    tree = ast.parse(MAIN_WINDOW.read_text(encoding="utf-8"))
    target = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert target is not None, f"main_window 里找不到 {func_name}"
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(target):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            out.append((node.lineno, node.col_offset, name))
    # ast.walk 是广度优先，不是源码顺序；要判断"谁在谁前面"必须自己按位置排
    return [name for _line, _col, name in sorted(out)]


# ------------------------------------------------------------------ T33
def test_broken_ai_json_never_reaches_rendering(tmp_path: Path) -> None:
    """AI 回来的东西解不开：这条只能失败/退回，绝不能变成「剪辑中」。"""
    cfg, db = make_project(tmp_path)
    vid, task_id = _task(cfg, db, "t33.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, task_id)
    db_repo.save_ai_result(db, vid, task_id=task_id, raw_response="AI 只回了一句话")
    with db.tx() as conn:      # 塞一段解不开的文本，绕过 _dumps
        conn.execute("UPDATE ai_results SET json_data = ? WHERE video_id = ?",
                     ("{不是 JSON", vid))

    assert db_repo.reusable_json_videos(db, [vid]) == set(), "解不开的 JSON 不算可复用"
    assert db_repo.fail_or_requeue_ai_task(db, task_id, "没拿到可用 JSON") == "failed"
    assert _status(db, task_id) == "failed"
    assert db_repo.mark_ai_task_rendering(db, task_id) is False, "failed 不许被拉进 processing"

    # 生产代码里这道闸门必须排在「标记剪辑中」和「起渲染」之前
    calls = _mw_calls("on_bridge_result")
    assert "clips_from_payload" in calls, "on_bridge_result 必须先看抠不抠得出片段"
    assert calls.index("clips_from_payload") < calls.index("_mark_auto_rendering")
    assert calls.index("clips_from_payload") < calls.index("run_highlight")
    db.close()


# ------------------------------------------------------------------ T34
def test_json_without_clip_never_reaches_rendering(tmp_path: Path) -> None:
    """JSON 合法但抠不出片段：同样不许进「剪辑中」。"""
    cfg, db = make_project(tmp_path)
    vid, task_id = _task(cfg, db, "t34.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, task_id)
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data={"note": "没有 clip"},
                           candidate_count=9, validated=True)

    row = db_repo.ai_result_for_task(db, task_id)
    assert row is not None and db_repo.clips_from_payload(json.loads(row["json_data"])) == []
    assert db_repo.reusable_json_videos(db, [vid]) == set(), "validated=True 也不算数"

    final = db_repo.fail_or_requeue_ai_task(db, task_id, "AI JSON 里没有可用片段")
    assert final in ("failed", "pending"), final
    assert _status(db, task_id) != "processing"
    db.close()


# ------------------------------------------------------------------ T35 / T36
def _heartbeat_survives(tmp_path: Path, state: str) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, f"hb_{state}.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, task_id)
    if state == "processing":
        db_repo.mark_ai_task_rendering(db, task_id)
    assert _status(db, task_id) == state

    with db.tx() as conn:      # 先把心跳压到很久以前
        conn.execute("UPDATE ai_tasks SET heartbeat_at = '2000-01-01T00:00:00' WHERE id = ?",
                     (task_id,))
    assert db_repo.touch_ai_task(db, task_id) is True, f"{state} 必须能刷心跳"
    fresh = db_repo.get_ai_task(db, task_id)
    assert str(fresh["heartbeat_at"]) > "2000-01-01T00:00:00"
    assert fresh["status"] == state, "刷心跳不许顺手改状态"
    assert db_repo.recover_stale_ai_tasks(db, 30.0) == 0, "心跳新鲜的不该被捞回"
    db.close()


def test_waiting_heartbeat_keeps_task_alive(tmp_path: Path) -> None:
    _heartbeat_survives(tmp_path, "waiting")


def test_rendering_heartbeat_keeps_task_alive(tmp_path: Path) -> None:
    _heartbeat_survives(tmp_path, "processing")


# ------------------------------------------------------------------ T37 / T38 / T39
def _orphan_goes_back(tmp_path: Path, state: str) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = _task(cfg, db, f"orphan_{state}.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-dead")
    if state in ("waiting", "processing"):
        db_repo.mark_ai_task_waiting(db, task_id)
    if state == "processing":
        db_repo.mark_ai_task_rendering(db, task_id)
    assert _status(db, task_id) == state
    with db.tx() as conn:      # 上次进程被强杀：心跳停在过去
        conn.execute("UPDATE ai_tasks SET heartbeat_at = '2000-01-01T00:00:00' WHERE id = ?",
                     (task_id,))

    assert db_repo.recover_orphaned_ai_tasks(db, "2026-01-01T00:00:00") == 1, state
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "pending", f"{state} 的孤儿必须退回 pending"
    assert int(row["retry_count"] or 0) == 0, "回收不算失败"
    assert row["worker_id"] is None and row["finished_at"] is None
    # 超时那条路（拿不到锁时走的）对三个状态一样管用
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-dead")
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET status = ?, heartbeat_at = '2000-01-01T00:00:00' "
                     "WHERE id = ?", (state, task_id))
    assert db_repo.recover_stale_ai_tasks(db, 30.0) == 1, state
    assert _status(db, task_id) == "pending"
    db.close()


def test_waiting_orphan_goes_back_to_pending(tmp_path: Path) -> None:
    _orphan_goes_back(tmp_path, "waiting")


def test_rendering_orphan_goes_back_to_pending(tmp_path: Path) -> None:
    _orphan_goes_back(tmp_path, "processing")


def test_uploading_orphan_goes_back_to_pending(tmp_path: Path) -> None:
    _orphan_goes_back(tmp_path, "uploading")


# ------------------------------------------------------------------ T40
def test_recovery_leaves_this_process_task_alone(tmp_path: Path) -> None:
    """拆出 waiting 之后，孤儿回收照旧只动"本次进程启动之前"的任务。"""
    cfg, db = make_project(tmp_path)
    _old_vid, old_id = _task(cfg, db, "t40_old.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-dead")
    db_repo.mark_ai_task_waiting(db, old_id)
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET heartbeat_at = '2000-01-01T00:00:00' WHERE id = ?",
                     (old_id,))

    started = "2025-01-01T00:00:00"      # 假装本次进程是这时候起来的
    _new_vid, new_id = _task(cfg, db, "t40_new.mp4")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-live")   # 心跳＝现在
    db_repo.mark_ai_task_waiting(db, new_id)

    assert db_repo.recover_orphaned_ai_tasks(db, started) == 1, "只该退回上次留下的那条"
    assert _status(db, old_id) == "pending"
    assert _status(db, new_id) == "waiting", "本进程正在等 AI 的活不许被抢走"
    db.close()


# ------------------------------------------------------------------ T41
def test_history_tasks_do_not_double_count(tmp_path: Path) -> None:
    """同一个视频攒了一堆历史任务：总览里它仍然只算一个视频、只落一个桶。"""
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t41.mp4"))
    for reason in ("failed", "cancelled"):
        task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full", max_attempts=1)
        db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
        if reason == "failed":
            db_repo.fail_or_requeue_ai_task(db, task_id, "boom")
        else:
            db_repo.cancel_ai_task(db, task_id, "人工停的")
    live, _ = db_repo.enqueue_ai_task(db, vid, mode="full", max_attempts=1)
    db_repo.claim_next_ai_task(db, mode="full", worker_id="w1")
    db_repo.mark_ai_task_waiting(db, live)

    st = db_repo.video_queue_statistics(db, [vid], mode="full")
    assert st["total"] == 1
    assert st["waiting_ai"] == 1, st
    buckets = ("no_json", "pending_render", "waiting_ai", "rendering",
               "done", "failed", "cancelled")
    assert sum(st[key] for key in buckets) == 1, f"只能落一个桶：{st}"
    assert st["failed"] == 0 and st["cancelled"] == 0, "历史任务不许再各算一次"
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_enqueue_is_idempotent,
    test_claim_is_atomic,
    test_heartbeat_and_recovery,
    test_completed_cannot_be_touched,
    test_max_attempts,
    test_ai_result_and_clip_links,
    test_artifact_follows_disk,
    test_product_readiness,
    test_reconcile_is_idempotent,
    test_prompt_fingerprint,
    test_resume_reconciles_before_recovery,
    test_claim_moves_pending_to_uploading,
    test_uploading_moves_to_waiting,
    test_waiting_moves_to_rendering,
    test_rendering_moves_to_completed,
    test_broken_ai_json_never_reaches_rendering,
    test_json_without_clip_never_reaches_rendering,
    test_waiting_heartbeat_keeps_task_alive,
    test_rendering_heartbeat_keeps_task_alive,
    test_waiting_orphan_goes_back_to_pending,
    test_rendering_orphan_goes_back_to_pending,
    test_uploading_orphan_goes_back_to_pending,
    test_recovery_leaves_this_process_task_alone,
    test_history_tasks_do_not_double_count,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="lifecycle_"))
        try:
            fn(work)
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001 - 测试脚本要把意外也报出来
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
