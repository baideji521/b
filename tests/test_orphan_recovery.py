"""上次没退干净留下的 active 任务能否安全退回队列（Phase 7 Batch 3 / P1-3）。

盯的是这个场景：任务停在 processing、心跳还没超时，程序被关掉/崩掉；重启之后
队列必须能接着跑，而不是干等 `ai_task_timeout_minutes`（默认 30 分钟）。
安全底线同样重要：拿不到库目录的独占锁时**一条都不许动**，绝不能抢别人正在跑的活。

覆盖：
  T1  单实例 + 未超时孤儿 + 本任务已有 ai_results -> 回收并直接开剪，AI 调用 0
  T2  单实例 + 未超时孤儿 + 没有 ai_results       -> 回收后正常问 AI 一次
  T3  锁被另一个实例占着                          -> 回收 0，任务原样不动
  T4  同进程二次调用                              -> 不回收本进程刚领的任务
  T5  原来的超时捞回照旧能用，且不吃 retry_count
  T6  completed / failed / cancelled 一条都不动
  T7  回收不消耗 max_attempts（retry_count 不加）
  T8  回收后 worker_id 空、error 写明原因、finished_at 空
  T9  回收后唯一索引仍然成立（同 video + mode 只有一条 open）
  T10 full / collect / script 三种模式的孤儿都能回收，且不串 mode
  T11 两份配置指向同一个库 -> 争的是同一把锁，只有一个能拿到
  T12 锁不可用 -> 不抛异常，降级回超时捞回
  T13 auto_resume_queue=False -> 状态照样回收，但不自动开工
  T14 A1：正常关程序时手上那条任务退回 pending（completed 的碰不着）

功能测试直接调 `MainWindow` 上的真方法（绑到轻量替身上，不建窗口），
只有渲染 / 发 AI / 磁盘对账这些叶子调用换成计数替身。
全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_orphan_recovery.py`，也可以 `pytest tests/test_orphan_recovery.py`。
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
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 只导入模块，不建窗口

from vidscribe.config import Config                     # noqa: E402
from vidscribe.db import open_db                         # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db.lock import RuntimeLock, queue_lock_path   # noqa: E402
from vidscribe.db.schema import TASK_ACTIVE              # noqa: E402
from vidscribe.gui import main_window as mw              # noqa: E402

GOOD_JSON = {"clip": {"start": 4.0, "end": 13.0, "score": 0.87, "type": "hook", "reason": "r"}}
LONG_AGO = "2000-01-01T00:00:00"
FAR_AHEAD = "2999-01-01T00:00:00"


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path, *, db_dir: Path | None = None):
    for sub in ("database", "input", "output", "logs", "ai_out", "cache"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data.setdefault("paths", {}).update({
        "db_dir": str(db_dir or (tmp_path / "database")),
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
    assert str(cfg.path("db_dir")).startswith(str(db_dir or tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str) -> Path:
    path = cfg.path("input_dir") / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def orphan_task(cfg, db, name: str, mode: str = "full", *, heartbeat: str = LONG_AGO,
                merged_txt: bool = False):
    """造一条"上次崩在 processing"的任务：心跳停在过去，盘上没有成品。"""
    video = fake_video(cfg, name)
    vid = db_repo.upsert_video(db, video)
    if merged_txt:
        txt = video.with_suffix(".txt")
        txt.write_text("merged text for AI", encoding="utf-8")
        db_repo.register_artifact(db, vid, "merged_txt", txt)
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode)
    # 显式落 processing：模拟"崩在渲染那一步"。领取本身现在只落 uploading（Batch 7 拆状态）
    db_repo.claim_next_ai_task(db, mode=mode, worker_id="gui-dead", status="processing")
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET heartbeat_at = ? WHERE id = ?", (heartbeat, task_id))
    assert db_repo.get_ai_task(db, task_id)["status"] == "processing"
    return video, vid, task_id


class Win:
    """够跑恢复链路的替身：真方法绑在下面，叶子调用计数。"""

    _resume_auto_queue = mw.MainWindow._resume_auto_queue
    _reclaim_orphan_tasks = mw.MainWindow._reclaim_orphan_tasks
    _queue_lock_handle = mw.MainWindow._queue_lock_handle
    _release_auto_task_on_exit = mw.MainWindow._release_auto_task_on_exit
    _auto_step = mw.MainWindow._auto_step
    _resume_existing_ai_json = mw.MainWindow._resume_existing_ai_json
    _save_ai_result = mw.MainWindow._save_ai_result
    _auto_save_script = mw.MainWindow._auto_save_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _db_video_id = mw.MainWindow._db_video_id
    _register_artifact = mw.MainWindow._register_artifact
    _settle_auto_task = mw.MainWindow._settle_auto_task
    _mark_auto_rendering = mw.MainWindow._mark_auto_rendering   # 真写库：复用 JSON 就进 processing

    def __init__(self, cfg, db, started_at: str | None = None):
        self.cfg = cfg
        self._db_handle = db
        self._db_failed = False
        self._process_started_at = started_at or db_repo.now()
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
        self.speech = []
        self.timeline = []
        self.calls = {k: 0 for k in ("send_file_to_ai", "dispatch_ai", "run_highlight",
                                     "on_analyze", "_auto_after_analyze", "_sync_disk",
                                     "load_video", "finish")}
        self.logs: list[str] = []

    # --- 真库
    def _db(self):
        return self._db_handle

    def ai_dir(self, key):
        return Path(str(self.cfg.bridge.get(key)))

    def export_root(self):
        return self.cfg.path("output_dir")

    def _worker_id(self):
        return "gui-test"

    # --- 计数替身
    def append_log(self, message):
        self.logs.append(str(message))

    def _sync_disk(self):
        self.calls["_sync_disk"] += 1

    def load_video(self, path):
        self.calls["load_video"] += 1
        self.video_path = Path(path)

    def send_file_to_ai(self, path):
        self.calls["send_file_to_ai"] += 1
        return True

    def dispatch_ai(self, *a, **k):
        self.calls["dispatch_ai"] += 1

    def run_highlight(self, text, ai=False):
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


def counts(db):
    return (len(db.all("SELECT id FROM ai_results")), len(db.all("SELECT id FROM clips")))


# ------------------------------------------------------------------ T1
def test_orphan_with_ai_result_renders_without_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid, task_id = orphan_task(cfg, db, "t1.mp4", merged_txt=True)
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=GOOD_JSON, validated=True)
    before = counts(db)

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert win._auto_task_id == task_id, "回收之后必须还是原来那条任务"
    assert win.calls["send_file_to_ai"] == 0 and win.calls["dispatch_ai"] == 0, "不许再问 AI"
    assert win.calls["run_highlight"] == 1, "该直接拿库里那份结果开剪"
    assert counts(db) == before, "复用不能多插 ai_results / clips"
    assert json.loads(win.rendered) == GOOD_JSON
    db.close()


# ------------------------------------------------------------------ T2
def test_orphan_without_ai_result_asks_ai_once(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    orphan_task(cfg, db, "t2.mp4", merged_txt=True)

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert win.calls["send_file_to_ai"] == 1, "没有结果就该正常问一次 AI"
    assert win.calls["run_highlight"] == 0
    assert counts(db) == (0, 0)
    db.close()


# ------------------------------------------------------------------ T3
def test_lock_held_by_other_instance_recovers_nothing(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, _vid, task_id = orphan_task(cfg, db, "t3.mp4", merged_txt=True)

    other = RuntimeLock(queue_lock_path(db))       # 假装另一个实例正在用这个库
    assert other.acquire() is True

    win = Win(cfg, db)
    try:
        assert win._reclaim_orphan_tasks(db) == 0, "拿不到锁时一条都不许动"
        assert db_repo.get_ai_task(db, task_id)["status"] == "processing", "别人的活不能抢"
        assert any("拿不到队列锁" in line for line in win.logs), win.logs
    finally:
        other.release()
    db.close()


# ------------------------------------------------------------------ T4
def test_second_call_keeps_own_task(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    cfg.runtime["auto_resume_queue"] = False       # 只看回收，不牵扯开工
    _video, _vid, task_id = orphan_task(cfg, db, "t4.mp4", heartbeat=LONG_AGO)

    win = Win(cfg, db)
    assert win._reclaim_orphan_tasks(db) == 1
    assert db_repo.get_ai_task(db, task_id)["status"] == "pending"

    # 本进程启动之后才领的任务：心跳晚于 _process_started_at，第二次调用不许动它
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-test")
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET heartbeat_at = ? WHERE id = ?", (FAR_AHEAD, task_id))
    assert win._reclaim_orphan_tasks(db) == 0, "不能把自己刚领的活退回去"
    assert db_repo.get_ai_task(db, task_id)["status"] == "uploading"
    db.close()


# ------------------------------------------------------------------ T5
def test_timeout_recovery_still_works(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, _vid, task_id = orphan_task(cfg, db, "t5.mp4", heartbeat=LONG_AGO)
    assert db_repo.recover_stale_ai_tasks(db, 30.0) == 1
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "pending" and int(row["retry_count"]) == 0, "超时捞回不吃重试额度"
    db.close()


# ------------------------------------------------------------------ T6
def test_finished_states_are_untouched(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    frozen = {}
    for state in ("completed", "failed", "cancelled"):
        vid = db_repo.upsert_video(db, fake_video(cfg, "t6_%s.mp4" % state))
        task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
        with db.tx() as conn:
            conn.execute("UPDATE ai_tasks SET status = ?, heartbeat_at = ? WHERE id = ?",
                         (state, LONG_AGO, task_id))
        frozen[state] = task_id

    assert db_repo.recover_orphaned_ai_tasks(db, FAR_AHEAD) == 0, "已结束的任务一条都不能动"
    for state, task_id in frozen.items():
        assert db_repo.get_ai_task(db, task_id)["status"] == state
    db.close()


# ------------------------------------------------------------------ T7
def test_recovery_does_not_eat_attempts(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t7.mp4")
    vid = db_repo.upsert_video(db, video)
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full", max_attempts=1)
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-dead")
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET heartbeat_at = ? WHERE id = ?", (LONG_AGO, task_id))

    assert db_repo.recover_orphaned_ai_tasks(db, FAR_AHEAD) == 1
    row = db_repo.get_ai_task(db, task_id)
    assert int(row["retry_count"]) == 0, "回收不算失败，retry_count 不许加"
    assert int(row["max_attempts"]) == 1
    # 额度还在：回收之后仍然允许失败一次
    assert db_repo.fail_or_requeue_ai_task(db, task_id, "boom") == "failed"
    db.close()


# ------------------------------------------------------------------ T8
def test_recovered_row_is_clean(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, _vid, task_id = orphan_task(cfg, db, "t8.mp4")
    with db.tx() as conn:  # 崩溃前那条留下的 finished_at，回收后必须清掉
        conn.execute("UPDATE ai_tasks SET finished_at = ? WHERE id = ?", (LONG_AGO, task_id))

    assert db_repo.recover_orphaned_ai_tasks(db, FAR_AHEAD) == 1
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "pending"
    assert row["worker_id"] is None, "worker_id 必须清空"
    assert row["finished_at"] is None, "finished_at 必须清空"
    assert row["error"] and "退回等待" in str(row["error"]), row["error"]
    db.close()


# ------------------------------------------------------------------ T9
def test_unique_index_still_holds(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid, task_id = orphan_task(cfg, db, "t9.mp4")
    assert db_repo.recover_orphaned_ai_tasks(db, FAR_AHEAD) == 1

    again, made = db_repo.enqueue_ai_task(db, vid, mode="full")
    assert again == task_id and made is False, "回收之后应该接上原来那条，不是新建一条"
    assert db_repo.queue_counts(db, mode="full")["open"] == 1
    db.close()


# ------------------------------------------------------------------ T10
def test_all_modes_are_recovered(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    ids = {}
    for mode in ("full", "collect", "script"):
        _video, _vid, task_id = orphan_task(cfg, db, "t10_%s.mp4" % mode, mode=mode)
        ids[mode] = task_id

    assert db_repo.recover_orphaned_ai_tasks(db, FAR_AHEAD) == 3, "三种模式的孤儿都要回收"
    for mode, task_id in ids.items():
        row = db_repo.get_ai_task(db, task_id)
        assert row["status"] == "pending"
        assert row["mode"] == mode, "mode 不能被改串"
    db.close()


# ------------------------------------------------------------------ T11
def test_lock_path_follows_the_real_db_file(tmp_path: Path) -> None:
    shared = tmp_path / "shared_db"
    shared.mkdir(parents=True, exist_ok=True)
    cfg_a, db_a = make_project(tmp_path / "a", db_dir=shared)
    cfg_b, db_b = make_project(tmp_path / "b", db_dir=shared)

    # 期望值也要 resolve：Windows 的临时目录可能是 8.3 短名（ADMINI~1），
    # 产品代码按真实库文件 resolve 之后才是同一把锁
    expected = (shared.resolve() / "queue.lock")
    assert queue_lock_path(db_a) == queue_lock_path(db_b) == expected, \
        "锁路径必须由真实库文件派生，不看各自的 cache / 输入目录"
    first, second = RuntimeLock(queue_lock_path(db_a)), RuntimeLock(queue_lock_path(db_b))
    assert first.acquire() is True
    try:
        assert second.acquire() is False, "同一个库只允许一个实例拿到锁"
    finally:
        first.release()
    assert second.acquire() is True, "前一个放手之后应该能拿到"
    second.release()
    db_a.close()
    db_b.close()


# ------------------------------------------------------------------ T12
def test_lock_unavailable_degrades(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, _vid, fresh = orphan_task(cfg, db, "t12_fresh.mp4", heartbeat=FAR_AHEAD)
    _v2, _vid2, stale = orphan_task(cfg, db, "t12_stale.mp4", heartbeat=LONG_AGO)
    cfg.runtime["auto_resume_queue"] = False

    class DeadLock:  # 锁怎么都拿不到（目录不可写 / 网络盘锁语义异常）
        def __init__(self, path):
            self.path = path

        def acquire(self):
            return False

        def release(self):
            pass

    original = mw.RuntimeLock
    mw.RuntimeLock = DeadLock
    try:
        win = Win(cfg, db)
        win._resume_auto_queue()          # 不许抛异常
    finally:
        mw.RuntimeLock = original

    assert db_repo.get_ai_task(db, fresh)["status"] == "processing", "未超时的不许动"
    assert db_repo.get_ai_task(db, stale)["status"] == "pending", "超时捞回必须照旧生效"
    db.close()


# ------------------------------------------------------------------ T13
def test_recovery_runs_even_when_auto_resume_off(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    cfg.runtime["auto_resume_queue"] = False
    _video, _vid, task_id = orphan_task(cfg, db, "t13.mp4", merged_txt=True)

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert db_repo.get_ai_task(db, task_id)["status"] == "pending", "状态照样要修好"
    assert win.calls["send_file_to_ai"] == 0 and win.calls["run_highlight"] == 0, "但不许自动开工"
    assert win._auto_task_id is None
    assert any("auto_resume_queue" in line for line in win.logs), win.logs
    db.close()


# ------------------------------------------------------------------ T14
def test_close_releases_current_task(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t14.mp4")
    vid = db_repo.upsert_video(db, video)
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode="full")
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-test")

    win = Win(cfg, db)
    win._auto_task_id = task_id
    win._release_auto_task_on_exit()
    row = db_repo.get_ai_task(db, task_id)
    assert row["status"] == "pending", "正常关程序不该留 processing 孤儿"
    assert row["worker_id"] is None and row["finished_at"] is None
    assert int(row["retry_count"]) == 0
    assert win._auto_task_id is None

    # 已经干完的那条：关程序时碰不着
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-test")
    db_repo.complete_ai_task(db, task_id)
    win2 = Win(cfg, db)
    win2._auto_task_id = task_id
    win2._release_auto_task_on_exit()
    assert db_repo.get_ai_task(db, task_id)["status"] == "completed"
    assert win2._auto_task_id == task_id, "什么都没退回时不该假装退了"
    assert TASK_ACTIVE, "状态常量还在（防止误改 schema 定义）"
    db.close()


# --------------------------------------------------- AI 结果留档的真实性（手动那条路）
def test_error_reply_is_not_marked_validated(tmp_path: Path) -> None:
    """AI 回的是报错对象：照样留档，但不许标成「已校验」，也不许建 clip。"""
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "err.mp4")
    win = Win(cfg, db)
    win._auto_video = video                 # 手动单发：没有任务行，task_id 是空的
    win._save_ai_result({"error": "rules file not provided"}, '{"error": "..."}')

    vid = db_repo.find_video(db, video)["id"]
    row = db_repo.get_ai_result(db, vid)
    assert row is not None, "报错回复也要留档，方便追溯 AI 到底回了什么"
    assert int(row["validated"]) == 0, "抠不出片段的回复不算已校验"
    assert row["validation_error"], "得写清为什么不算数"
    assert row["task_id"] is None and row["raw_response"], (dict(row),)
    assert db_repo.get_clips(db, vid) == [] or len(db_repo.get_clips(db, vid)) == 0
    assert db_repo.reusable_json_videos(db, [vid]) == set(), "这份结果不能被当成可复用"
    db.close()


def test_overlay_evaluation_lands_in_clips(tmp_path: Path) -> None:
    """现行提示词把中文评价放在 clip.overlays.evaluation，落库时不能丢。"""
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "eva.mp4")
    evaluation = "情绪反差鲜明，信息完整"
    payload = {"video": video.name,
               "clip": {"start": 31.89, "end": 39.99, "duration": 8.1, "score": 92,
                        "type": "搞笑", "reason": "赌注揭晓后的反应最强",
                        "overlays": {"comment": {"time": 35.0, "text": "wow", "kind": "comment"},
                                     "evaluation": evaluation}}}
    win = Win(cfg, db)
    win._auto_video = video
    win._save_ai_result(payload, json.dumps(payload, ensure_ascii=False))

    vid = db_repo.find_video(db, video)["id"]
    row = db_repo.get_ai_result(db, vid)
    assert int(row["validated"]) == 1 and row["validation_error"] is None
    assert int(row["candidate_count"]) == 1 and float(row["winner_score"]) == 92.0
    clips = db_repo.get_clips(db, vid)
    assert len(clips) == 1, clips
    clip = clips[0]
    assert clip["evaluation"] == evaluation, "中文评价必须原样落库，不许变 NULL"
    assert clip["reason"] == "赌注揭晓后的反应最强" and clip["clip_type"] == "搞笑"
    assert float(clip["start_time"]) == 31.89 and float(clip["end_time"]) == 39.99
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_orphan_with_ai_result_renders_without_ai,
    test_orphan_without_ai_result_asks_ai_once,
    test_lock_held_by_other_instance_recovers_nothing,
    test_second_call_keeps_own_task,
    test_timeout_recovery_still_works,
    test_finished_states_are_untouched,
    test_recovery_does_not_eat_attempts,
    test_recovered_row_is_clean,
    test_unique_index_still_holds,
    test_all_modes_are_recovered,
    test_lock_path_follows_the_real_db_file,
    test_lock_unavailable_degrades,
    test_recovery_runs_even_when_auto_resume_off,
    test_close_releases_current_task,
    test_error_reply_is_not_marked_validated,
    test_overlay_evaluation_lands_in_clips,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="orphan_"))
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
