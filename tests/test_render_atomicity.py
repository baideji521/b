"""成品要么完整、要么不存在（Phase 7 Batch 4 / P1-4）。

盯的是这个场景：渲染写到一半程序崩了/被杀/断电，盘上留下一个非空但封装不完整的
mp4；下一次开程序 `_sync_disk()` 把它登记成 final_video，`_auto_done_file()` 就以为
这条任务干完了，于是**永远不会重剪**，而日志还会说"已经有成品，跳过"。

两道防线一起测：
  A 物理层：渲染全程只写 `<成品名>.part`，封装完整收尾之后才 `os.replace` 成成品；
  B 登记层：登记 final_video 之前必须过 `video_io.is_complete_video`。

覆盖：
  T1  1KB 垃圾数据冒充成品        -> 不登记 final_video
  T2  0 字节冒充成品              -> 不登记
  T3  只有 ftyp 头的残片          -> 不登记
  T4  PyAV 现场生成的合法 mp4     -> 照常登记（防止"为了防坏文件把好文件也拒了"）
  T5  <成品名>.part               -> importer 完全看不见
  T6  真实渲染 -> 原子提交：替换旧成品、不留 .part，且提交那一刻成品还是旧内容
  T7  os.replace 失败             -> 渲染算失败，成品不出现，残片留在 .part
  T8  孤儿任务 + .part 残片       -> 不当成品、不结算 completed，任务照常往下走
  T9  孤儿任务 + .part + 已有 AI 结果 -> 不问 AI，直接重新渲染（Batch 2 联动）
  T10 完整成品 + 已有 AI 结果     -> 正常识别成品、结算 completed，不重剪不重问

功能测试直接调 `MainWindow` 上的真方法（绑到轻量替身上，不建窗口），`_sync_disk`
用的是**真的**磁盘对账，只有渲染 / 发 AI 这些叶子调用换成计数替身。
全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_render_atomicity.py`，也可以 `pytest tests/test_render_atomicity.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 只导入模块，不建窗口

import av                                                   # noqa: E402
import numpy as np                                          # noqa: E402

from vidscribe.config import Config                          # noqa: E402
from vidscribe.db import open_db                              # noqa: E402
from vidscribe.db import importer as importer_mod             # noqa: E402
from vidscribe.db import repo as db_repo                      # noqa: E402
from vidscribe.db.importer import refresh_from_disk           # noqa: E402
from vidscribe.gui import main_window as mw                   # noqa: E402
from vidscribe.highlight import clip as clip_mod              # noqa: E402
from vidscribe.video_io import is_complete_video, probe_video  # noqa: E402

GOOD_JSON = {"clip": {"start": 0.0, "end": 0.32, "score": 0.9, "type": "hook", "reason": "r"}}
LONG_AGO = "2000-01-01T00:00:00"
FINAL_TAIL = "_高光时刻.mp4"


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


def real_mp4(path: Path, frames: int = 30, size: int = 64, fps: int = 25) -> Path:
    """现场编一份最小的合法 mp4（无声）。不下载素材、不依赖 test.mp4。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 显式给 format：路径可能叫 xxx.mp4.part，PyAV 猜不出容器
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        stream.options = {"crf": "30", "preset": "ultrafast"}
        for i in range(frames):
            arr = np.full((size, size, 3), (i * 7) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert path.stat().st_size > 0
    return path


def junk(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def half_baked(path: Path) -> Path:
    """像"写到一半"的 mp4：有 ftyp 头、有一堆 mdat 数据，就是没有 moov。"""
    return junk(path, b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
                      + b"\x00\x00\x04\x00mdat" + bytes(range(256)) * 4)


def source_video(cfg) -> Path:
    """输入目录里一份真视频（渲染要真读它）。"""
    return real_mp4(cfg.path("input_dir") / "src.mp4")


def fake_video(cfg, name: str) -> Path:
    """不需要真渲染的用例用它：库里有这个视频就够了。"""
    path = cfg.path("input_dir") / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


def spec_for(video: Path) -> clip_mod.HighlightSpec:
    return clip_mod.HighlightSpec(
        video_name=video.name, clip_start=0.0, clip_end=0.32, freeze_time=0.2,
        freeze_text="测试", freeze_overlays=[], other_overlays=[], raw={})


def orphan_task(cfg, db, video: Path, mode: str = "full", *, merged_txt: bool = False):
    """造一条"上次崩在 processing"的任务：心跳停在过去，盘上没有成品。"""
    vid = db_repo.upsert_video(db, video)
    if merged_txt:
        txt = video.with_suffix(".txt")
        txt.write_text("merged text for AI", encoding="utf-8")
        db_repo.register_artifact(db, vid, "merged_txt", txt)
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode)
    db_repo.claim_next_ai_task(db, mode=mode, worker_id="gui-dead")
    with db.tx() as conn:
        conn.execute("UPDATE ai_tasks SET heartbeat_at = ? WHERE id = ?", (LONG_AGO, task_id))
    assert db_repo.get_ai_task(db, task_id)["status"] == "processing"
    return vid, task_id


class Win:
    """够跑恢复链路的替身：真方法绑在下面，只有叶子调用计数。

    和 test_orphan_recovery 的 Win 差一处：`_sync_disk` 是**真的**，因为本轮要验的
    正是磁盘对账会不会把残片登记成成品。
    """

    _resume_auto_queue = mw.MainWindow._resume_auto_queue
    _reclaim_orphan_tasks = mw.MainWindow._reclaim_orphan_tasks
    _queue_lock_handle = mw.MainWindow._queue_lock_handle
    _sync_disk = mw.MainWindow._sync_disk
    _auto_step = mw.MainWindow._auto_step
    _resume_existing_ai_json = mw.MainWindow._resume_existing_ai_json
    _auto_save_script = mw.MainWindow._auto_save_script
    _auto_text_file = mw.MainWindow._auto_text_file
    _auto_script_file = mw.MainWindow._auto_script_file
    _auto_done_file = mw.MainWindow._auto_done_file
    _auto_product_ready = mw.MainWindow._auto_product_ready
    _db_video_id = mw.MainWindow._db_video_id
    _register_artifact = mw.MainWindow._register_artifact

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
        self.speech = []
        self.timeline = []
        self.settled: list[str] = []
        self.calls = {k: 0 for k in ("send_file_to_ai", "dispatch_ai", "run_highlight",
                                     "on_analyze", "_auto_after_analyze", "load_video", "finish")}
        self.logs: list[str] = []

    def _db(self):
        return self._db_handle

    def ai_dir(self, key):
        return Path(str(self.cfg.bridge.get(key)))

    def export_root(self):
        return self.cfg.path("output_dir")

    def _worker_id(self):
        return "gui-test"

    def _settle_auto_task(self, outcome, error=None):
        """真结算，顺手记一笔：本轮要证明"残片不会让任务被判 completed"。"""
        self.settled.append(str(outcome))
        return mw.MainWindow._settle_auto_task(self, outcome, error)


    # --- 计数替身
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


def sync_and_ask(cfg, db, video: Path):
    """走真实 importer 流程，然后问库里认不认这个成品。"""
    refresh_from_disk(cfg, db, folders=[cfg.path("input_dir")],
                      ai_out=Path(str(cfg.bridge.get("ai_output_dir"))))
    vid = db_repo.upsert_video(db, video)
    return db_repo.artifact_path(db, vid, "final_video")


# ------------------------------------------------------------------ T1
def test_half_baked_mp4_is_not_a_product(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t1.mp4")
    bad = half_baked(Path(str(cfg.bridge["ai_output_dir"])) / f"{video.stem}{FINAL_TAIL}")
    assert bad.stat().st_size > 1000, "残片必须是非空的，才能证明不是靠体积挡住的"

    assert is_complete_video(bad) is False
    assert sync_and_ask(cfg, db, video) is None, "封装不完整的残片不许登记成 final_video"
    db.close()


# ------------------------------------------------------------------ T2
def test_zero_byte_is_not_a_product(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t2.mp4")
    empty = junk(Path(str(cfg.bridge["ai_output_dir"])) / f"{video.stem}{FINAL_TAIL}", b"")

    assert empty.stat().st_size == 0
    assert is_complete_video(empty) is False
    assert sync_and_ask(cfg, db, video) is None
    db.close()


# ------------------------------------------------------------------ T3
def test_tiny_broken_mp4_is_not_a_product(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t3.mp4")
    stub = junk(Path(str(cfg.bridge["ai_output_dir"])) / f"{video.stem}{FINAL_TAIL}",
                b"\x00\x00\x00\x18ftypmp42")

    assert is_complete_video(stub) is False
    assert sync_and_ask(cfg, db, video) is None
    db.close()


# ------------------------------------------------------------------ T4
def test_real_mp4_is_registered(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t4.mp4")
    good = real_mp4(Path(str(cfg.bridge["ai_output_dir"])) / f"{video.stem}{FINAL_TAIL}")

    assert is_complete_video(good) is True, "正常成品必须放行，不能为了防坏文件把好文件拒了"
    assert sync_and_ask(cfg, db, video) == good
    db.close()


# ------------------------------------------------------------------ T5
def test_part_file_is_invisible_to_importer(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t5.mp4")
    ai_out = Path(str(cfg.bridge["ai_output_dir"]))
    # 故意让 .part 本身是一份**完整**视频：证明挡住它靠的是"不是成品名"，不是"内容坏"
    part = real_mp4(ai_out / (f"{video.stem}{FINAL_TAIL}" + clip_mod.PART_SUFFIX))
    assert part.is_file() and part.stat().st_size > 0

    assert sync_and_ask(cfg, db, video) is None, ".part 永远不是成品"
    vid = db_repo.upsert_video(db, video)
    kinds = [row["path"] for row in db_repo.get_artifacts(db, vid)]
    assert str(part) not in kinds, ".part 一条 artifacts 都不该留"
    assert importer_mod.PART_SUFFIX == clip_mod.PART_SUFFIX, "两边的 .part 后缀必须一致"
    db.close()


# ------------------------------------------------------------------ T6
def test_render_commits_atomically(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = source_video(cfg)
    out = Path(str(cfg.bridge["ai_output_dir"]))
    target = clip_mod.default_target(out, video)
    part = clip_mod.part_target(target)
    old_bytes = real_mp4(target, frames=6).read_bytes()   # 上一次的成品，必须被原子替换
    seen: list[tuple[bool, bool, int]] = []

    real_os = clip_mod.os

    class SpyOs:
        """在 os.replace 真正发生的那一刻看一眼盘上的样子。"""

        def replace(self, src, dst):
            seen.append((is_complete_video(Path(src)),
                         Path(dst).is_file(),
                         Path(dst).stat().st_size if Path(dst).is_file() else -1))
            return real_os.replace(src, dst)

    clip_mod.os = SpyOs()
    try:
        result = clip_mod.render_highlight(video, spec_for(video), target)
    finally:
        clip_mod.os = real_os

    assert len(seen) == 1, "提交只能发生一次"
    part_complete, target_there, target_size = seen[0]
    assert part_complete is True, "提交之前 .part 必须已经是完整视频"
    assert target_there is True and target_size == len(old_bytes), \
        "提交之前成品还应该是上一次那份（不许先删再写）"

    assert Path(result["output"]) == target
    assert target.is_file() and target.read_bytes() != old_bytes, "旧成品必须被换掉"
    assert is_complete_video(target) is True
    assert not part.exists(), "提交完了不该留 .part"
    assert probe_video(target).duration > 0

    vid = db_repo.upsert_video(db, video)
    assert sync_and_ask(cfg, db, video) == target, "真渲染出来的成品照样能登记"
    assert db_repo.artifact_path(db, vid, "final_video") == target
    db.close()


# ------------------------------------------------------------------ T7
def test_replace_failure_is_a_render_failure(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = source_video(cfg)
    out = Path(str(cfg.bridge["ai_output_dir"]))
    target = clip_mod.default_target(out, video)
    part = clip_mod.part_target(target)

    real_os = clip_mod.os

    class StuckOs:  # 成品正被播放器/资源管理器占着
        def replace(self, src, dst):
            raise PermissionError("目标被占用")

    clip_mod.os = StuckOs()
    try:
        try:
            clip_mod.render_highlight(video, spec_for(video), target)
        except RuntimeError as exc:
            assert "提交失败" in str(exc), exc
        else:
            raise AssertionError("提交失败必须抛出去，绝不能当成渲染成功")
    finally:
        clip_mod.os = real_os

    assert not target.exists(), "提交没成功，成品绝不能出现"
    assert part.is_file(), "残片留在 .part 里，下次重渲直接覆盖"
    assert sync_and_ask(cfg, db, video) is None, "失败的渲染不许留下 final_video"
    db.close()


# ------------------------------------------------------------------ T8
def test_orphan_with_part_leftover_is_re_rendered(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t8.mp4")
    vid, task_id = orphan_task(cfg, db, video, merged_txt=True)
    out = Path(str(cfg.bridge["ai_output_dir"]))
    # 上次崩在渲染中途：残片留在 .part，成品没出现
    part = real_mp4(out / (f"{video.stem}{FINAL_TAIL}" + clip_mod.PART_SUFFIX))

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert win._auto_task_id == task_id, "孤儿任务必须被捞回来接着跑"
    assert db_repo.artifact_path(db, vid, "final_video") is None, "残片不许变成成品"
    assert win.settled == [], "更不许直接结算 completed"
    assert db_repo.get_ai_task(db, task_id)["status"] == "processing", "是被本进程领走了，不是干完了"
    assert win.calls["send_file_to_ai"] == 1, "没有 AI 结果就照常问一次 AI"
    assert part.is_file(), "残片不用清，下次渲染会覆盖"
    assert not (out / f"{video.stem}{FINAL_TAIL}").exists()
    db.close()


# ------------------------------------------------------------------ T9
def test_orphan_with_part_and_ai_result_renders_without_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t9.mp4")
    vid, task_id = orphan_task(cfg, db, video, merged_txt=True)
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=GOOD_JSON, validated=True)
    out = Path(str(cfg.bridge["ai_output_dir"]))
    real_mp4(out / (f"{video.stem}{FINAL_TAIL}" + clip_mod.PART_SUFFIX))
    before = (len(db.all("SELECT id FROM ai_results")), len(db.all("SELECT id FROM clips")))

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert win.calls["send_file_to_ai"] == 0 and win.calls["dispatch_ai"] == 0, "不许再问 AI"
    assert win.calls["run_highlight"] == 1, "该拿库里那份结果直接重新渲染"
    assert json.loads(win.rendered) == GOOD_JSON
    assert db_repo.artifact_path(db, vid, "final_video") is None
    assert win.settled == []
    assert (len(db.all("SELECT id FROM ai_results")),
            len(db.all("SELECT id FROM clips"))) == before, "复用不能多插记录"
    db.close()


# ------------------------------------------------------------------ T10
def test_complete_product_short_circuits(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "t10.mp4")
    vid, task_id = orphan_task(cfg, db, video, merged_txt=True)
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=GOOD_JSON, validated=True)
    out = Path(str(cfg.bridge["ai_output_dir"]))
    good = real_mp4(out / f"{video.stem}{FINAL_TAIL}")   # 上次其实剪完了，只是没登记

    win = Win(cfg, db)
    win._resume_auto_queue()

    assert db_repo.artifact_path(db, vid, "final_video") == good, "完整成品必须被认出来"
    assert win.settled == ["completed"], "认出成品就该把这条任务结算掉"
    assert db_repo.get_ai_task(db, task_id)["status"] == "completed"
    assert win.calls["run_highlight"] == 0, "不许白剪第二遍"
    assert win.calls["send_file_to_ai"] == 0 and win.calls["dispatch_ai"] == 0, "更不许重问 AI"
    assert win.calls["finish"] == 1, "队列里没别的活了，正常收工"
    db.close()


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_half_baked_mp4_is_not_a_product,
    test_zero_byte_is_not_a_product,
    test_tiny_broken_mp4_is_not_a_product,
    test_real_mp4_is_registered,
    test_part_file_is_invisible_to_importer,
    test_render_commits_atomically,
    test_replace_failure_is_a_render_failure,
    test_orphan_with_part_leftover_is_re_rendered,
    test_orphan_with_part_and_ai_result_renders_without_ai,
    test_complete_product_short_circuits,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="atomic_"))
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
