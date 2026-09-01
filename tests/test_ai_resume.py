"""崩溃续跑不重复问 AI 的回归测试（Phase 7 Batch 2 / P1-1）。

盯的是这个场景：AI 已经回话、`ai_results` 已经落库，程序在 FFmpeg 之前被强关；
重启恢复之后不该再问一遍 AI（那是真金白银的配额），应该直接拿库里那份结果开剪。

覆盖：
  T1 这条任务没有 AI 结果            -> 照原样去问 AI
  T2 这条任务有有效结果              -> 复用，不问 AI
  T3 只有 ai_script 文件、没有结果    -> full 模式不认它，照原样去问 AI
  T4 结果里的 JSON 坏了              -> 不复用，回到问 AI
  T5 task 隔离：A 有结果、B 没有      -> B 必须去问 AI，不许捡 A 的
  T6 同一条任务多份结果              -> 取 id 最大的那份
  T7 复用不会再插一条 ai_results
  T8 collect / script 不走这条恢复逻辑
  T9 恢复分支的调用路径：ai_results -> run_highlight，不经过 dispatch_ai / send_file_to_ai

功能测试直接调 `MainWindow._resume_existing_ai_json`（真代码，绑到一个轻量替身上，
不建窗口）；"走哪条路"这种控制流用 AST 读 `_auto_step` 的源码来断言。
全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_ai_resume.py`，也可以 `pytest tests/test_ai_resume.py`。
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
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 只导入模块，不建窗口

from vidscribe.config import Config                    # noqa: E402
from vidscribe.db import open_db                        # noqa: E402
from vidscribe.db import repo as db_repo                # noqa: E402
from vidscribe.gui.main_window import MainWindow        # noqa: E402

MAIN_WINDOW_SRC = ROOT / "src" / "vidscribe" / "gui" / "main_window.py"
GOOD_JSON = {"clip": {"start": 3.0, "end": 12.5, "score": 0.88, "type": "hook", "reason": "r"}}


# ------------------------------------------------------------------ 夹具
class FakeWindow:
    """够 `_resume_existing_ai_json` 用的最小替身：一个库句柄、一个任务号、一份日志。"""

    def __init__(self, db, task_id: int | None) -> None:
        self._handle = db
        self._auto_task_id = task_id
        self.logs: list[str] = []

    def _db(self):
        return self._handle

    def append_log(self, message: str) -> None:
        self.logs.append(message)


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


def claimed_task(cfg, db, name: str, mode: str = "full"):
    """造一条正在跑的任务（跟崩溃前的状态一样）。"""
    vid = db_repo.upsert_video(db, fake_video(cfg, name))
    task_id, _ = db_repo.enqueue_ai_task(db, vid, mode=mode)
    db_repo.claim_next_ai_task(db, mode=mode, worker_id="gui-test")
    return vid, task_id


def resume(host: FakeWindow) -> str | None:
    """调真正的那段代码。"""
    return MainWindow._resume_existing_ai_json(host)


# ------------------------------------------------------------------ T1
def test_no_result_means_ask_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _vid, task_id = claimed_task(cfg, db, "fresh.mp4")
    assert resume(FakeWindow(db, task_id)) is None, "没有结果就该照原样去问 AI"
    assert resume(FakeWindow(db, None)) is None, "没有任务号时也不能凭空复用"
    assert resume(FakeWindow(None, task_id)) is None, "库打不开时按没有结果处理"
    db.close()


# ------------------------------------------------------------------ T2
def test_existing_result_is_reused(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "resume.mp4")
    db_repo.save_ai_result(db, vid, task_id=task_id, raw_response="raw",
                           json_data=GOOD_JSON, validated=True)
    got = resume(FakeWindow(db, task_id))
    assert got is not None, "这条任务已经有 AI 结果，必须复用"
    parsed = json.loads(got)
    assert db_repo.clips_from_payload(parsed)[0]["start"] == 3.0
    assert parsed == GOOD_JSON, "复用的必须就是库里那份，不能改内容"
    db.close()


# ------------------------------------------------------------------ T3
def test_ai_script_alone_is_not_evidence(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "scriptonly.mp4")
    # 用户手放在视频旁边的脚本也会被登记成 ai_script，它证明不了"这次 AI 回过话"
    script = cfg.path("input_dir") / "scriptonly_脚本.json"
    script.write_text(json.dumps(GOOD_JSON, ensure_ascii=False), encoding="utf-8")
    db_repo.register_artifact(db, vid, "ai_script", script)
    assert db_repo.artifact_path(db, vid, "ai_script") == script
    assert resume(FakeWindow(db, task_id)) is None, "full 模式不能拿 ai_script 当 AI 已完成的证据"
    db.close()


# ------------------------------------------------------------------ T4
def test_broken_json_falls_back_to_ai(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "broken.mp4")
    with db.tx() as conn:  # 直接塞一份坏的 json_data（模拟写坏 / 半截）
        conn.execute(
            "INSERT INTO ai_results(task_id, video_id, raw_response, json_data, validated, "
            "created_at) VALUES(?, ?, ?, ?, 0, datetime('now'))",
            (task_id, vid, "raw", '{"clip": {"start": 1'))
    host = FakeWindow(db, task_id)
    assert resume(host) is None, "坏 JSON 不能复用"
    assert any("解不开" in line for line in host.logs), "必须留一行明确日志：%s" % host.logs

    # 能解析但抠不出片段，也算不能用
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data={"note": "no clip"})
    host2 = FakeWindow(db, task_id)
    assert resume(host2) is None
    assert any("没有可用片段" in line for line in host2.logs), host2.logs
    db.close()


# ------------------------------------------------------------------ T5
def test_task_isolation(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_a = claimed_task(cfg, db, "iso.mp4")
    db_repo.save_ai_result(db, vid, task_id=task_a, json_data=GOOD_JSON, validated=True)
    db_repo.complete_ai_task(db, task_a)

    task_b, made = db_repo.enqueue_ai_task(db, vid, mode="full")  # 用户手动重跑
    assert made is True and task_b != task_a
    db_repo.claim_next_ai_task(db, mode="full", worker_id="gui-test")

    assert resume(FakeWindow(db, task_b)) is None, "新任务不许捡上一条任务的结果"
    assert resume(FakeWindow(db, task_a)) is not None, "老任务自己的结果还在"
    assert db_repo.get_ai_result(db, vid) is not None, "按 video 取当然有结果——正因如此不能用它"
    db.close()


# ------------------------------------------------------------------ T6
def test_latest_result_wins(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "many.mp4")
    old = {"clip": {"start": 1.0, "end": 2.0, "score": 0.1}}
    mid = {"clip": {"start": 5.0, "end": 6.0, "score": 0.5}}
    new = {"clip": {"start": 9.0, "end": 11.0, "score": 0.9}}
    for payload in (old, mid, new):
        db_repo.save_ai_result(db, vid, task_id=task_id, json_data=payload, validated=True)
    got = json.loads(resume(FakeWindow(db, task_id)) or "{}")
    assert got == new, "同一条任务有多份结果时要取 id 最大的那份：%s" % got
    db.close()


# ------------------------------------------------------------------ T7
def test_reuse_does_not_insert(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "noinsert.mp4")
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=GOOD_JSON, validated=True)
    before = db_repo.counts(db)
    for _ in range(3):
        assert resume(FakeWindow(db, task_id)) is not None
    after = db_repo.counts(db)
    assert after["ai_results"] == before["ai_results"] == 1, "复用不能再插一条结果"
    assert after["clips"] == before["clips"], "复用不能再建片段"
    db.close()


# --------------------------------------------------- AST 辅助（控制流断言）
def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MAIN_WINDOW_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("main_window.py 里找不到 %s" % name)


def _calls(node: ast.AST) -> list[str]:
    out: list[tuple[int, str]] = []
    for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
        if isinstance(call.func, ast.Attribute):
            out.append((getattr(call, "lineno", 0), call.func.attr))
    return [name for _, name in sorted(out)]


def _reusable_branch() -> ast.If:
    """`_auto_step` 里那个 `if reusable is not None:` 分支（库里已有可复用高光 JSON）。"""
    for node in ast.walk(_function("_auto_step")):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if isinstance(left, ast.Name) and left.id == "reusable":
            return node
    raise AssertionError("_auto_step 里没有「库里已有可复用高光 JSON」这个分支")


# ------------------------------------------------------------------ T8
def test_collect_and_script_untouched(tmp_path: Path) -> None:
    """恢复/复用只有一个入口，三种模式的完成判定全在库里。"""
    inner = _calls(_function("_reusable_highlight_json"))
    assert "_resume_existing_ai_json" in inner, "本任务自己的 AI 结果必须查"
    step = _calls(_function("_auto_step"))
    assert step.count("_resume_existing_ai_json") == 0, "_auto_step 不许自己再恢复一遍"
    assert step.count("_reusable_highlight_json") == 1, "「库里有没有可复用 JSON」只问一次"

    # 完成判定全查库：collect 看可复用高光 JSON，其余看还在盘上的 final_video
    done = _calls(_function("_auto_chain_done"))
    assert "reusable_json_videos" in done and "artifact_path" in done
    assert "_resume_existing_ai_json" not in done
    # _auto_done_file 只拿路径给日志用，不参与判定
    assert "artifact_path" in _calls(_function("_auto_done_file"))
    # 脚本剪辑那一串已经并回 _auto_step，不许再有第二个入口
    assert "_auto_clip_from_script" not in MAIN_WINDOW_SRC.read_text(encoding="utf-8"), \
        "旧的 _auto_clip_from_script 必须删干净，别留第二条剪辑路径"

    # collect 模式下这个 helper 根本不会被调到；就算被调，也只认 task_id，不看 mode
    cfg, db = make_project(tmp_path)
    vid, task_id = claimed_task(cfg, db, "collect.mp4", mode="collect")
    db_repo.save_ai_result(db, vid, task_id=task_id, json_data=GOOD_JSON, validated=True)
    counts = db_repo.queue_counts(db, mode="collect")
    assert counts["active"] == 1, counts
    db.close()


# ------------------------------------------------------------------ T9
def test_resume_path_goes_straight_to_render() -> None:
    branch = _calls(_reusable_branch())
    assert "run_highlight" in branch, "库里有可复用高光 JSON 之后必须直接开剪"
    for banned in ("dispatch_ai", "send_file_to_ai", "_save_ai_result", "create_clip",
                   "_auto_after_analyze", "on_analyze"):
        assert banned not in branch, "复用分支里不该出现 %s" % banned
    assert "_auto_save_script" in branch, "复用时补一份高光 JSON 留档（按设计）"

    # 复用分支必须排在"去问 AI"和"重跑分析"之前，否则照样会先发一遍
    step = _calls(_function("_auto_step"))
    assert step.index("_reusable_highlight_json") < step.index("_auto_text_file")
    assert step.index("_reusable_highlight_json") < step.index("send_file_to_ai")
    assert step.index("_reusable_highlight_json") < step.index("on_analyze")

    # _auto_save_script 只许写文件 + 登记产物，不许有别的副作用
    saver = _calls(_function("_auto_save_script"))
    for banned in ("save_ai_result", "_save_ai_result", "create_clip", "dispatch_ai",
                   "complete_ai_task", "fail_or_requeue_ai_task", "update_ai_task"):
        assert banned not in saver, "_auto_save_script 不该碰 %s" % banned
    assert "_register_artifact" in saver

    # 恢复查询只认 task_id，绝不退回按 video 取最新
    helper = _calls(_function("_resume_existing_ai_json"))
    assert "ai_result_for_task" in helper
    assert "get_ai_result" not in helper, "禁止退回 get_ai_result(video_id)"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_no_result_means_ask_ai,
    test_existing_result_is_reused,
    test_ai_script_alone_is_not_evidence,
    test_broken_json_falls_back_to_ai,
    test_task_isolation,
    test_latest_result_wins,
    test_reuse_does_not_insert,
    test_collect_and_script_untouched,
    test_resume_path_goes_straight_to_render,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="ai_resume_"))
        try:
            if fn.__code__.co_argcount:
                fn(work)
            else:
                fn()
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
