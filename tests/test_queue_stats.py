"""自动剪辑总览统计：按视频去重、每个视频只落一个桶（Phase 7 Batch 5）。

盯的是这件事：界面上的数字必须和自动剪辑的**实际**状态一致。
以前那四格是 `总任务 / 未剪辑 / 已获取 JSON / 成品`，其中"未剪辑"是 `total - done`
这种混合桶——把"还没问 AI"、"等 AI 回话"、"JSON 到手等渲染"、"渲染失败"、"被取消"
全塞在一个数字里。现在换成六个互斥桶 + 一个横切指标：

    done / rendering / waiting_ai / pending_render / failed / cancelled / no_json ← 互斥，和为 total
    json                                                               ← 横切，可与任何桶重叠

覆盖：
  T1  输入视频、无 JSON                  -> no_json
  T2  有效 JSON、无成品、无任务          -> pending_render
  T3  无效 JSON（坏 JSON / 非 dict / 无片段）-> 不算已获取 JSON
  T4  完整成品                           -> done
  T5  只有 .part 残片                    -> 不算成品
  T6  坏 MP4 占着成品名                  -> 不算成品
  T7  active 任务、无 JSON               -> waiting_ai（活任务盖过 no_json）
  T8  有效 JSON + failed 任务            -> failed
  T9  有效 JSON + cancelled 任务         -> cancelled
  T10 有效 JSON + 渲染中任务             -> rendering（不同时进 pending_render）
  T11 failed 之后又出了有效成品          -> done，失败清零
  T12 同一视频多条历史任务               -> 总视频仍是 1
  T13 同一视频多份 ai_results            -> 已获取 JSON 仍是 1
  T14 同一视频多份 final_video artifacts -> 已完成仍是 1
  T15 成品被删掉                         -> 回到 pending_render
  T16 AI_输入目录之外的视频              -> 完全不计入
  T17 旧成品 + 新渲染中任务              -> 已完成 1 / 剪辑中 0（成品优先）
  T18 互斥矩阵：每个视频恰好落一个桶，六桶之和 == 总视频
  T19 收取脚本口径（done_key='json'）
  T20 get_statistics 的 size 口径修好了（0 字节不算成品/脚本）
  T21 面板那几格的 key 和统计结果对得上（防 KeyError，也防"未剪辑"混合桶回来）

Batch 6 追加（统计口径收口）：
  T22 命令行总览和面板是同一份数字（同一个函数、同一个范围、同一个 done_key）
  T23 两条产物记录 updated_at 相同 -> artifact_path 稳定地选 id 最大的那条
  T24 AI_输入目录之外的视频不进 CLI 统计
  T25 input 不许把隔壁 input_old 一起捞进来（前缀带分隔符 + `_` 转义）
  T26 文件已经不在盘上的输入视频单独统计，不混进六个桶
  T27 一个视频堆满历史（多任务 + 多 JSON + 多成品）只算一次
  T28 混合样本 × 三种口径：六桶之和恒等于总视频，CLI 自检不触发

Batch 7 追加（状态机拆分）：
  T42 等待 AI（uploading/waiting）和剪辑中（processing）是两个互斥桶，GUI/CLI 都分得清



全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_queue_stats.py`，也可以 `pytest tests/test_queue_stats.py`。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.config import Config                      # noqa: E402
from vidscribe.db import open_db                         # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db.importer import reconcile, refresh_from_disk  # noqa: E402
from vidscribe.db.schema import TASK_ACTIVE              # noqa: E402

GOOD_JSON = {"clip": {"start": 4.0, "end": 13.0, "score": 0.87, "type": "hook", "reason": "r"}}
FINAL_TAIL = "_高光时刻.mp4"
BUCKETS = ("no_json", "pending_render", "waiting_ai", "rendering", "done", "failed", "cancelled")


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
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
    data["bridge"]["ai_job"] = "full"
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str, *, where: Path | None = None) -> Path:
    """一个"够真"的视频文件：指纹按内容算，所以每个文件字节必须不同。"""
    path = (where or cfg.path("input_dir")) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def real_mp4(path: Path, frames: int = 6, size: int = 64, fps: int = 25) -> Path:
    """现场编一份最小的合法 mp4（无声）。成品登记要过 is_complete_video（P1-4）。"""
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


def half_baked(path: Path) -> Path:
    """像"写到一半"的 mp4：有 ftyp、有一堆 mdat 数据，就是没有 moov。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
                     + b"\x00\x00\x04\x00mdat" + bytes(range(256)) * 4)
    return path


def ai_out(cfg) -> Path:
    return Path(str(cfg.bridge["ai_output_dir"]))


def sync(cfg, db) -> None:
    """走真实登记/对账链路：成品是不是有效由 importer 的闸门说话（Batch 4）。"""
    refresh_from_disk(cfg, db, folders=[cfg.path("input_dir")], ai_out=ai_out(cfg))


def scope(cfg, db) -> list[int]:
    """界面用的那份视频集合：还在盘上 + 在 AI_输入目录里。"""
    return [int(row["id"]) for row in db_repo.videos_under(db, cfg.path("input_dir"))]


def stats(cfg, db, *, mode: str = "full", done_key: str = "clipped") -> dict[str, int]:
    st = db_repo.video_queue_statistics(db, scope(cfg, db), mode=mode, done_key=done_key)
    assert sum(st[k] for k in BUCKETS) == st["total"], f"六个桶必须刚好把总视频分完：{st}"
    return st


def active_task(db, vid: int, mode: str = "full") -> int:
    """领一条任务。领到手是 uploading（Batch 7：领取 = 正在提交给 AI）。"""
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode)
    db_repo.claim_next_ai_task(db, mode=mode, worker_id="gui-test")
    assert db_repo.get_ai_task(db, task_id)["status"] in TASK_ACTIVE
    return task_id


def waiting_task(db, vid: int, mode: str = "full") -> int:
    """已经把东西交给 AI，正在等回话（uploading -> waiting）。"""
    task_id = active_task(db, vid, mode)
    assert db_repo.mark_ai_task_waiting(db, task_id) is True
    assert db_repo.get_ai_task(db, task_id)["status"] == "waiting"
    return task_id


def rendering_task(db, vid: int, mode: str = "full") -> int:
    """JSON 到手，正在渲染成品（-> processing）。"""
    task_id = waiting_task(db, vid, mode)
    assert db_repo.mark_ai_task_rendering(db, task_id) is True
    assert db_repo.get_ai_task(db, task_id)["status"] == "processing"
    return task_id


def failed_task(db, vid: int, mode: str = "full") -> int:
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode, max_attempts=1)
    db_repo.claim_next_ai_task(db, mode=mode, worker_id="gui-test")
    assert db_repo.fail_or_requeue_ai_task(db, task_id, "boom") == "failed"
    return task_id


def cancelled_task(db, vid: int, mode: str = "full") -> int:
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode)
    db_repo.cancel_ai_task(db, task_id, "人工停的")
    assert db_repo.get_ai_task(db, task_id)["status"] == "cancelled"
    return task_id


# ------------------------------------------------------------------ T1
def test_plain_video_has_no_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    fake_video(cfg, "t1.mp4")
    sync(cfg, db)

    st = stats(cfg, db)
    assert st["total"] == 1
    assert st["json"] == 0 and st["no_json"] == 1
    assert st["pending_render"] == 0 and st["waiting_ai"] == 0 and st["rendering"] == 0
    assert st["done"] == 0 and st["failed"] == 0 and st["cancelled"] == 0
    db.close()


# ------------------------------------------------------------------ T2
def test_valid_json_without_product_waits_for_render(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t2.mp4"))
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)

    st = stats(cfg, db)
    assert st["json"] == 1 and st["no_json"] == 0
    assert st["pending_render"] == 1, "JSON 到手、没成品、没在跑 —— 这才叫待剪辑"
    assert st["done"] == 0 and st["waiting_ai"] == 0 and st["rendering"] == 0
    db.close()


# ------------------------------------------------------------------ T3
def test_useless_json_does_not_count(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    broken = db_repo.upsert_video(db, fake_video(cfg, "t3_broken.mp4"))
    not_dict = db_repo.upsert_video(db, fake_video(cfg, "t3_str.mp4"))
    no_clip = db_repo.upsert_video(db, fake_video(cfg, "t3_empty.mp4"))
    # 库里直接塞一段解不开的文本，绕过 _dumps
    db_repo.save_ai_result(db, broken, raw_response="AI 只回了一句话")
    with db.tx() as conn:
        conn.execute("UPDATE ai_results SET json_data = ? WHERE video_id = ?",
                     ("{不是 JSON", broken))
    db_repo.save_ai_result(db, not_dict, json_data="就是个字符串", validated=True)
    db_repo.save_ai_result(db, no_clip, json_data={"note": "没有 clip"}, candidate_count=9,
                           validated=True)

    st = stats(cfg, db)
    assert st["total"] == 3
    assert st["json"] == 0, "解不开 / 不是 dict / 抠不出片段，一律不算已获取 JSON"
    assert st["no_json"] == 3
    # validated / candidate_count 是写入时的自我声明，不能当依据
    assert db_repo.reusable_json_videos(db, [broken, not_dict, no_clip]) == set()
    db.close()


# ------------------------------------------------------------------ T4
def test_complete_product_is_done(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t4.mp4")
    real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    sync(cfg, db)

    st = stats(cfg, db)
    assert st["total"] == 1 and st["done"] == 1
    assert st["no_json"] == 0, "有成品就落 done，不该同时算进未获取 JSON"
    assert st["pending_render"] == 0 and st["waiting_ai"] == 0 and st["rendering"] == 0
    db.close()


# ------------------------------------------------------------------ T5
def test_part_leftover_is_not_done(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t5.mp4")
    real_mp4(ai_out(cfg) / (f"{video.stem}{FINAL_TAIL}" + ".part"))
    sync(cfg, db)

    st = stats(cfg, db)
    assert st["done"] == 0, ".part 残片不是成品"
    assert st["no_json"] == 1
    db.close()


# ------------------------------------------------------------------ T6
def test_broken_mp4_is_not_done(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t6.mp4")
    bad = half_baked(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    assert bad.stat().st_size > 1000, "得是非空文件，才能证明不是靠体积挡住的"
    sync(cfg, db)

    st = stats(cfg, db)
    assert st["done"] == 0, "封装不完整的 mp4 不是成品（Batch 4 在登记入口就挡住了）"
    assert st["no_json"] == 1
    db.close()


# ------------------------------------------------------------------ T7
def test_active_task_beats_no_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t7.mp4"))
    active_task(db, vid)

    st = stats(cfg, db)
    assert st["waiting_ai"] == 1, "领到手（uploading）算「等待 AI」，不算「剪辑中」"
    assert st["rendering"] == 0
    assert st["no_json"] == 0 and st["pending_render"] == 0
    db.close()


# ------------------------------------------------------------------ T8
def test_failed_task_with_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t8.mp4"))
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    failed_task(db, vid)

    st = stats(cfg, db)
    assert st["failed"] == 1
    assert st["pending_render"] == 0 and st["done"] == 0 and st["rendering"] == 0
    assert st["json"] == 1
    db.close()


# ------------------------------------------------------------------ T9
def test_cancelled_task_with_json(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t9.mp4"))
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    cancelled_task(db, vid)

    st = stats(cfg, db)
    assert st["cancelled"] == 1
    assert st["failed"] == 0 and st["pending_render"] == 0
    db.close()


# ------------------------------------------------------------------ T10
def test_processing_with_json_is_not_pending_render(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t10.mp4"))
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    rendering_task(db, vid)

    st = stats(cfg, db)
    assert st["rendering"] == 1 and st["pending_render"] == 0
    assert st["waiting_ai"] == 0, "已经在渲染的不许再算「等待 AI」"
    assert st["json"] == 1, "已获取 JSON 是横切指标，剪辑中也照样显示"
    db.close()


# ------------------------------------------------------------------ T11
def test_product_after_failure_wins(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t11.mp4")
    vid = db_repo.upsert_video(db, video)
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    failed_task(db, vid)
    assert stats(cfg, db)["failed"] == 1

    real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")   # 后来重剪成功了
    sync(cfg, db)

    st = stats(cfg, db)
    assert st["done"] == 1 and st["failed"] == 0, "有成品就是完成，历史 failed 不该继续显示"
    db.close()


# ------------------------------------------------------------------ T12
def test_many_tasks_one_video(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t12.mp4"))
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    first, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-test")
    db_repo.complete_ai_task(db, first)
    second, made = db_repo.enqueue_ai_task(db, vid, mode="full")   # 完成之后又排了一条
    assert made is True and second != first
    assert len(db.all("SELECT id FROM ai_tasks WHERE video_id = ?", (vid,))) == 2

    st = stats(cfg, db)
    assert st["total"] == 1, "两条任务是同一个视频，总视频只能算 1"
    assert st["pending_render"] == 1, "新任务还是 pending（不算活任务），JSON 在手 -> 待剪辑"
    db.close()


# ------------------------------------------------------------------ T13
def test_many_ai_results_one_video(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid = db_repo.upsert_video(db, fake_video(cfg, "t13.mp4"))
    db_repo.save_ai_result(db, vid, json_data={"note": "第一次没抠出片段"})
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    assert len(db.all("SELECT id FROM ai_results WHERE video_id = ?", (vid,))) == 3

    st = stats(cfg, db)
    assert st["total"] == 1 and st["json"] == 1, "三份结果也只算一个视频拿到了 JSON"
    assert st["pending_render"] == 1
    db.close()


# ------------------------------------------------------------------ T14
def test_many_final_artifacts_one_video(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t14.mp4")
    real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    real_mp4(ai_out(cfg) / f"{video.stem}_手改过的名字.mp4", frames=8)
    sync(cfg, db)
    vid = db_repo.upsert_video(db, video)
    finals = db_repo.get_artifacts(db, vid, "final_video")
    assert len(finals) == 2, "两份成品都登记上了，才谈得上去重"

    st = stats(cfg, db)
    assert st["total"] == 1 and st["done"] == 1, "两份 artifacts 也只能算一个视频完成"
    db.close()


# ------------------------------------------------------------------ T15
def test_deleted_product_goes_back_to_pending(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t15.mp4")
    vid = db_repo.upsert_video(db, video)
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    product = real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    sync(cfg, db)
    assert stats(cfg, db)["done"] == 1

    product.unlink()
    reconcile(cfg, db)

    st = stats(cfg, db)
    assert st["done"] == 0, "成品被删了就不能继续算完成"
    assert st["pending_render"] == 1, "JSON 还在，回到待剪辑"
    db.close()


# ------------------------------------------------------------------ T16
def test_videos_outside_input_dir_are_ignored(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    fake_video(cfg, "inside.mp4")
    outside = fake_video(cfg, "outside.mp4", where=cfg.path("output_dir"))
    out_id = db_repo.upsert_video(db, outside)
    db_repo.save_ai_result(db, out_id, json_data=GOOD_JSON, validated=True)
    active_task(db, out_id)
    sync(cfg, db)

    ids = scope(cfg, db)
    assert out_id not in ids, "AI_输入目录之外的视频不进统计范围"
    st = stats(cfg, db)
    assert st["total"] == 1 and st["no_json"] == 1
    assert st["json"] == 0 and st["waiting_ai"] == 0 and st["rendering"] == 0
    db.close()


# ------------------------------------------------------------------ T17
def test_old_product_beats_new_processing_task(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t17.mp4")
    vid = db_repo.upsert_video(db, video)
    db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")     # 上一轮的成品还在
    sync(cfg, db)
    rendering_task(db, vid)                                  # 又排了一条，还进了渲染

    st = stats(cfg, db)
    assert st["done"] == 1, "成品优先"
    assert st["rendering"] == 0 and st["waiting_ai"] == 0, "同一个视频不许同时算进两个桶"
    assert st["pending_render"] == 0 and st["failed"] == 0
    db.close()


# ------------------------------------------------------------------ T18
def test_every_video_lands_in_exactly_one_bucket(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    made: dict[str, int] = {}

    plain = db_repo.upsert_video(db, fake_video(cfg, "m_plain.mp4"))
    made["no_json"] = plain

    waiting = db_repo.upsert_video(db, fake_video(cfg, "m_wait.mp4"))
    db_repo.save_ai_result(db, waiting, json_data=GOOD_JSON, validated=True)
    made["pending_render"] = waiting

    running = db_repo.upsert_video(db, fake_video(cfg, "m_run.mp4"))
    db_repo.save_ai_result(db, running, json_data=GOOD_JSON, validated=True)
    rendering_task(db, running)
    made["rendering"] = running

    asking = db_repo.upsert_video(db, fake_video(cfg, "m_ask.mp4"))
    waiting_task(db, asking)                 # 已提交，等 AI 回话；这时还没有 JSON
    made["waiting_ai"] = asking

    broke = db_repo.upsert_video(db, fake_video(cfg, "m_fail.mp4"))
    db_repo.save_ai_result(db, broke, json_data=GOOD_JSON, validated=True)
    failed_task(db, broke)
    made["failed"] = broke

    stopped = db_repo.upsert_video(db, fake_video(cfg, "m_cancel.mp4"))
    db_repo.save_ai_result(db, stopped, json_data=GOOD_JSON, validated=True)
    cancelled_task(db, stopped)
    made["cancelled"] = stopped

    finished_video = fake_video(cfg, "m_done.mp4")
    finished = db_repo.upsert_video(db, finished_video)
    db_repo.save_ai_result(db, finished, json_data=GOOD_JSON, validated=True)
    real_mp4(ai_out(cfg) / f"{finished_video.stem}{FINAL_TAIL}")
    made["done"] = finished
    sync(cfg, db)

    # 逐个视频：七个桶里必须恰好命中一个
    for bucket, vid in made.items():
        one = db_repo.video_queue_statistics(db, [vid], mode="full")
        hit = [key for key in BUCKETS if one[key]]
        assert hit == [bucket], f"{bucket} 那个视频落到了 {hit}：{one}"
        assert sum(one[key] for key in BUCKETS) == 1

    st = stats(cfg, db)
    assert st["total"] == len(made) == 7
    for bucket in BUCKETS:
        assert st[bucket] == 1, f"{bucket} 应该正好 1 个：{st}"
    assert st["json"] == 5, "七个里 m_plain 和 m_ask 没有可用 JSON"
    db.close()


# ------------------------------------------------------------------ T19
def test_collect_mode_counts_json_as_done(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    got = db_repo.upsert_video(db, fake_video(cfg, "t19_got.mp4"))
    db_repo.save_ai_result(db, got, json_data=GOOD_JSON, validated=True)
    db_repo.upsert_video(db, fake_video(cfg, "t19_none.mp4"))

    st = stats(cfg, db, mode="collect", done_key="json")
    assert st["total"] == 2
    assert st["done"] == 1, "收取脚本拿到 JSON 就算完事"
    assert st["pending_render"] == 0, "这一串没有渲染这一步"
    assert st["no_json"] == 1
    db.close()


# ------------------------------------------------------------------ T20
def test_get_statistics_needs_nonzero_size(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t20.mp4")
    vid = db_repo.upsert_video(db, video)
    empty_final = ai_out(cfg) / f"{video.stem}{FINAL_TAIL}"
    empty_final.write_bytes(b"")
    empty_script = ai_out(cfg) / f"{video.stem}_脚本.json"
    empty_script.write_bytes(b"")
    # 直接登记（绕过 importer 的闸门），只为了验证查询层的 size 口径
    db_repo.register_artifact(db, vid, "final_video", empty_final)
    db_repo.register_artifact(db, vid, "ai_script", empty_script)

    stat = db_repo.get_statistics(db, [vid])
    assert stat["done"] == 0, "0 字节不算成品，要和 artifact_path 同一个口径"
    assert stat["json"] == 0, "0 字节的脚本也不算拿到 JSON"
    assert db_repo.artifact_path(db, vid, "final_video") is None
    assert db_repo.states_for_videos(db, [vid])[vid]["clipped"] is False
    db.close()


# ------------------------------------------------------------------ T21
def test_gui_boxes_match_statistics_keys(tmp_path: Path) -> None:
    """面板头号数字的 key 必须都在统计结果里，否则刷新时直接 KeyError。"""
    import ast  # noqa: PLC0415

    source = (ROOT / "src" / "vidscribe" / "gui" / "ai_options.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    build = next((node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_build_headline"), None)
    assert build is not None, "ai_options 里找不到 _build_headline"
    keys = {node.elts[0].value for node in ast.walk(build)
            if isinstance(node, ast.Tuple) and len(node.elts) == 3
            and all(isinstance(el, ast.Constant) for el in node.elts)
            and isinstance(node.elts[0].value, str)}
    assert keys, "没解析出面板的统计 key"

    cfg, db = make_project(tmp_path)
    db_repo.upsert_video(db, fake_video(cfg, "t21.mp4"))
    st = stats(cfg, db)
    # made（盘上还在的成品 mp4）是面板自己数盘算出来的，不走统计函数。
    missing = sorted(keys - {"made"} - set(st))
    assert not missing, f"面板要显示但统计没给的字段：{missing}"
    assert "todo" not in keys, "「未剪辑」这种 total-done 混合桶不许再出现在界面上"
    db.close()


# ------------------------------------------------------------------ T22
def cli_overview(cfg, db) -> tuple[list[dict], list[str]]:
    """跑一遍命令行的自动剪辑总览，把它对统计函数的调用和打出来的行都录下来。

    录调用参数是为了证明"CLI 自己不算数"：范围、mode、done_key 全都得和面板一致。
    """
    import logging  # noqa: PLC0415

    from vidscribe import cli as cli_mod  # noqa: PLC0415

    calls: list[dict] = []
    real = db_repo.video_queue_statistics

    def spy(conn, video_ids, **kw):
        out = real(conn, video_ids, **kw)
        calls.append({"ids": sorted(int(v) for v in video_ids), "kw": kw, "out": out})
        return out

    lines: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    grab = Grab()
    cli_mod.logger.addHandler(grab)
    # 测试里没跑 setup_logging，日志级别默认还在 WARNING，info 会被直接丢掉
    old_level = cli_mod.logger.level
    cli_mod.logger.setLevel(logging.INFO)
    db_repo.video_queue_statistics = spy
    try:
        cli_mod._db_report_queue(cfg, db)
    finally:
        db_repo.video_queue_statistics = real
        cli_mod.logger.setLevel(old_level)
        cli_mod.logger.removeHandler(grab)
    return calls, lines


def test_cli_and_gui_report_the_same_numbers(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    finished_video = fake_video(cfg, "t22_done.mp4")
    finished = db_repo.upsert_video(db, finished_video)
    db_repo.save_ai_result(db, finished, json_data=GOOD_JSON, validated=True)
    real_mp4(ai_out(cfg) / f"{finished_video.stem}{FINAL_TAIL}")
    fake_video(cfg, "t22_plain.mp4")
    sync(cfg, db)

    gui = stats(cfg, db)
    calls, lines = cli_overview(cfg, db)
    assert len(calls) == 1, "CLI 只该问统计函数一次"
    assert calls[0]["ids"] == sorted(scope(cfg, db)), "CLI 的视频范围必须和面板一样"
    assert calls[0]["kw"] == {"mode": "full", "done_key": "clipped"}, calls[0]["kw"]
    assert calls[0]["out"] == gui, f"CLI 和界面必须是同一份数字：{calls[0]['out']} vs {gui}"
    assert gui["total"] == 2 and gui["done"] == 1 and gui["no_json"] == 1
    text = "\n".join(lines)
    assert "自动剪辑总览" in text
    assert "分桶合计" not in text, "对不上才会打这行，正常情况下不许出现"
    db.close()


# ------------------------------------------------------------------ T23
def test_same_updated_at_picks_biggest_id(tmp_path: Path) -> None:
    """两条产物记录时间戳一模一样时，必须稳定地选后登记的那条（id 大的）。"""
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t23.mp4")
    vid = db_repo.upsert_video(db, video)
    first = real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    second = real_mp4(ai_out(cfg) / f"{video.stem}_高光时刻_v2.mp4", frames=8)
    db_repo.register_artifact(db, vid, "final_video", first)
    db_repo.register_artifact(db, vid, "final_video", second)
    # now() 只到秒，同一秒登记两条就会撞成一样；这里直接压平，把"撞车"变成确定条件
    with db.tx() as conn:
        conn.execute("UPDATE artifacts SET updated_at = '2026-01-01 00:00:00' "
                     "WHERE video_id = ? AND type = 'final_video'", (vid,))
    rows = [r for r in db_repo.get_artifacts(db, vid, "final_video")]
    assert len(rows) == 2
    newest = max(rows, key=lambda r: int(r["id"]))

    got = db_repo.artifact_path(db, vid, "final_video")
    assert got is not None and str(got) == newest["path"], (
        f"时间戳相同就该认 id 最大的那条：拿到 {got}，期望 {newest['path']}")
    for _ in range(5):
        assert db_repo.artifact_path(db, vid, "final_video") == got, "多问几次结果必须一样"
    assert stats(cfg, db)["done"] == 1, "两条产物记录仍然只算一个视频完成"
    db.close()


# ------------------------------------------------------------------ T24
def test_cli_scope_excludes_videos_outside_input_dir(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    inside = db_repo.upsert_video(db, fake_video(cfg, "t24_in.mp4"))
    outside = db_repo.upsert_video(
        db, fake_video(cfg, "t24_out.mp4", where=tmp_path / "elsewhere"))
    db_repo.save_ai_result(db, outside, json_data=GOOD_JSON, validated=True)

    calls, _ = cli_overview(cfg, db)
    assert calls[0]["ids"] == [inside], "AI_输入目录之外的视频不许进 CLI 统计"
    assert calls[0]["out"]["total"] == 1 and calls[0]["out"]["json"] == 0
    db.close()


# ------------------------------------------------------------------ T25
def test_sibling_dir_with_same_prefix_does_not_bleed(tmp_path: Path) -> None:
    """`input` 不许把 `input_old` 一起捞进来——目录前缀要带分隔符，`_` 还要转义。"""
    cfg, db = make_project(tmp_path)
    inside = db_repo.upsert_video(db, fake_video(cfg, "t25_in.mp4"))
    sibling = tmp_path / (cfg.path("input_dir").name + "_old")
    old = db_repo.upsert_video(db, fake_video(cfg, "t25_old.mp4", where=sibling))

    ids = [int(r["id"]) for r in db_repo.videos_under(db, cfg.path("input_dir"))]
    assert ids == [inside], f"隔壁 {sibling.name} 的视频漏进来了：{ids}"
    assert [int(r["id"]) for r in db_repo.videos_under(db, sibling)] == [old]
    assert db_repo.missing_input_videos(db, cfg.path("input_dir")) == []
    calls, _ = cli_overview(cfg, db)
    assert calls[0]["ids"] == [inside]
    db.close()


# ------------------------------------------------------------------ T26
def test_missing_input_videos_are_counted_separately(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    gone_file = fake_video(cfg, "t26_gone.mp4")
    gone = db_repo.upsert_video(db, gone_file)
    db_repo.save_ai_result(db, gone, json_data=GOOD_JSON, validated=True)
    stay = db_repo.upsert_video(db, fake_video(cfg, "t26_stay.mp4"))
    assert stats(cfg, db)["total"] == 2

    gone_file.unlink()
    reconcile(cfg, db)

    missing = db_repo.missing_input_videos(db, cfg.path("input_dir"))
    assert [int(r["id"]) for r in missing] == [gone], "文件没了的视频要能单独查出来"
    st = stats(cfg, db)
    assert st["total"] == 1 and scope(cfg, db) == [stay], "查不到的视频不进分桶"
    _, lines = cli_overview(cfg, db)
    assert any("盘上找不着" in line for line in lines), "CLI 得把这 1 个说出来"
    db.close()


# ------------------------------------------------------------------ T27
def test_one_video_with_everything_is_counted_once(tmp_path: Path) -> None:
    """同一个视频身上堆满历史：多条任务 + 多份 JSON + 多份成品，只能算一个已完成。"""
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t27.mp4")
    vid = db_repo.upsert_video(db, video)
    for _ in range(3):
        db_repo.save_ai_result(db, vid, json_data=GOOD_JSON, validated=True)
    failed_task(db, vid)
    cancelled_task(db, vid)
    real_mp4(ai_out(cfg) / f"{video.stem}{FINAL_TAIL}")
    real_mp4(ai_out(cfg) / f"{video.stem}_高光时刻_2.mp4", frames=8)
    sync(cfg, db)
    active_task(db, vid)

    st = stats(cfg, db)
    assert st["total"] == 1 and st["json"] == 1 and st["done"] == 1
    assert st["rendering"] == 0 and st["failed"] == 0 and st["cancelled"] == 0
    assert st["pending_render"] == 0 and st["no_json"] == 0
    calls, _ = cli_overview(cfg, db)
    assert calls[0]["out"] == st
    db.close()


# ------------------------------------------------------------------ T28
def test_buckets_always_add_up(tmp_path: Path) -> None:
    """混合样本 + 两种口径：六个桶之和永远等于总视频，CLI 那条自检不许触发。"""
    cfg, db = make_project(tmp_path)
    db_repo.upsert_video(db, fake_video(cfg, "s_plain.mp4"))

    waiting = db_repo.upsert_video(db, fake_video(cfg, "s_wait.mp4"))
    db_repo.save_ai_result(db, waiting, json_data=GOOD_JSON, validated=True)

    running = db_repo.upsert_video(db, fake_video(cfg, "s_run.mp4"))
    active_task(db, running)

    broke = db_repo.upsert_video(db, fake_video(cfg, "s_fail.mp4"))
    db_repo.save_ai_result(db, broke, json_data=GOOD_JSON, validated=True)
    failed_task(db, broke)

    stopped = db_repo.upsert_video(db, fake_video(cfg, "s_cancel.mp4"))
    db_repo.save_ai_result(db, stopped, json_data=GOOD_JSON, validated=True)
    cancelled_task(db, stopped)

    finished_video = fake_video(cfg, "s_done.mp4")
    db_repo.upsert_video(db, finished_video)
    real_mp4(ai_out(cfg) / f"{finished_video.stem}{FINAL_TAIL}")
    half_baked(ai_out(cfg) / "s_wait_高光时刻.mp4")     # 半成品，不许算成品
    sync(cfg, db)

    ids = scope(cfg, db)
    assert len(ids) == 6
    for mode, done_key in (("full", "clipped"), ("collect", "json"), ("script", "clipped")):
        st = db_repo.video_queue_statistics(db, ids, mode=mode, done_key=done_key)
        assert sum(st[k] for k in BUCKETS) == st["total"] == 6, f"{mode}/{done_key}: {st}"
        assert st["json"] <= st["total"]
    _, lines = cli_overview(cfg, db)
    assert not any("分桶合计" in line for line in lines), "CLI 自检报警了，说明口径破了"
    db.close()


# ------------------------------------------------------------------ T42
def test_waiting_ai_and_rendering_are_two_buckets(tmp_path: Path) -> None:
    """「等待 AI」和「剪辑中」必须是两个数：GUI、CLI 都得分得清，七桶之和仍等于总视频。"""
    cfg, db = make_project(tmp_path)
    asking = db_repo.upsert_video(db, fake_video(cfg, "t42_ask.mp4"))
    waiting_task(db, asking)
    cutting = db_repo.upsert_video(db, fake_video(cfg, "t42_cut.mp4"))
    db_repo.save_ai_result(db, cutting, json_data=GOOD_JSON, validated=True)
    rendering_task(db, cutting)
    db_repo.upsert_video(db, fake_video(cfg, "t42_idle.mp4"))

    st = stats(cfg, db)
    assert st["waiting_ai"] == 1 and st["rendering"] == 1, st
    assert st["waiting_ai"] != st["total"] and "processing" not in st, "旧的混合桶不许留着"
    assert st["no_json"] == 1 and st["pending_render"] == 0
    assert st["json"] == 1, "横切指标照旧只看有没有能开剪的 JSON"

    # 逐个视频看：两条活任务分别落在自己的桶里，绝不互串
    assert db_repo.video_queue_statistics(db, [asking], mode="full")["waiting_ai"] == 1
    assert db_repo.video_queue_statistics(db, [asking], mode="full")["rendering"] == 0
    assert db_repo.video_queue_statistics(db, [cutting], mode="full")["rendering"] == 1
    assert db_repo.video_queue_statistics(db, [cutting], mode="full")["waiting_ai"] == 0

    calls, lines = cli_overview(cfg, db)
    assert calls[0]["out"] == st, "CLI 和界面还是同一份数字"
    text = "\n".join(lines)
    assert "等待 AI 1" in text and "剪辑中 1" in text, text
    assert not any("分桶合计" in line for line in lines), "七桶之和必须还等于总视频"
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_plain_video_has_no_json,
    test_valid_json_without_product_waits_for_render,
    test_useless_json_does_not_count,
    test_complete_product_is_done,
    test_part_leftover_is_not_done,
    test_broken_mp4_is_not_done,
    test_active_task_beats_no_json,
    test_failed_task_with_json,
    test_cancelled_task_with_json,
    test_processing_with_json_is_not_pending_render,
    test_product_after_failure_wins,
    test_many_tasks_one_video,
    test_many_ai_results_one_video,
    test_many_final_artifacts_one_video,
    test_deleted_product_goes_back_to_pending,
    test_videos_outside_input_dir_are_ignored,
    test_old_product_beats_new_processing_task,
    test_every_video_lands_in_exactly_one_bucket,
    test_collect_mode_counts_json_as_done,
    test_get_statistics_needs_nonzero_size,
    test_gui_boxes_match_statistics_keys,
    test_cli_and_gui_report_the_same_numbers,
    test_same_updated_at_picks_biggest_id,
    test_cli_scope_excludes_videos_outside_input_dir,
    test_sibling_dir_with_same_prefix_does_not_bleed,
    test_missing_input_videos_are_counted_separately,
    test_one_video_with_everything_is_counted_once,
    test_buckets_always_add_up,
    test_waiting_ai_and_rendering_are_two_buckets,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="qstats_"))
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
