"""CLI `dance-montage` 子命令（技术指导第十九节 + 一期第四十二节）。

两件事：
  1. 新子命令能从头跑通一整条链（align → slice → materials → remix → stats → history）
  2. **原有子命令一个都没被挤掉**，参数解析仍然正常

所有 IO 都指向临时目录（用 `--config` 指一份临时配置），**绝不碰项目真实数据库**。
可以 `pytest tests/test_dance_cli.py`，也可以 `python tests/test_dance_cli.py`。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import delayed, make_song_file, make_source_video   # noqa: E402
from vidscribe import cli                                              # noqa: E402


class _Cfg:
    """只为造素材文件用的一个最小 cfg 替身（fixtures 只用到 dance 这个字典）。"""

    def __init__(self, mapping):
        self.dance = mapping


def _scene(work: Path) -> tuple[Path, Path]:
    """在临时目录里搭一份配置 + 一首歌 + 两个源视频，返回 `(配置路径, 输出目录)`。"""
    for sub in ("database", "input", "output", "logs", "cache",
                "dance_songs", "dance_materials", "dance_out"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data.setdefault("paths", {}).update({
        "db_dir": str(work / "database"), "cache_dir": str(work / "cache"),
        "output_dir": str(work / "output"), "input_dir": str(work / "input"),
        "video_dir": "", "log_dir": str(work / "logs"),
    })
    data["dance"] = {
        "song_dir": str(work / "dance_songs"), "source_dir": str(work / "input"),
        "material_dir": str(work / "dance_materials"), "output_dir": str(work / "dance_out"),
        "slice_duration": 2.0, "align_workers": 2,
        # 小画布 + 低帧率：这条测试要的是"命令通不通"，不是画质
        "canvas_width": 144, "canvas_height": 256, "canvas_fps": 24.0,
    }
    cfg_path = work / "config.json"
    cfg_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    stub = _Cfg(data["dance"])
    _, pcm = make_song_file(stub, "target.wav", bpm=120.0, duration=10.0)
    make_source_video(stub, "dancer0.mp4", pcm, tint=(210, 70, 120), fps=24.0)
    make_source_video(stub, "dancer1.mp4", delayed(pcm, 1.0), tint=(90, 180, 120), fps=25.0)
    return cfg_path, work / "dance_out"


def _run(cfg_path: Path, *argv: str) -> int:
    return cli.main(["--config", str(cfg_path), "dance-montage", *argv])


def test_existing_commands_still_parse() -> None:
    """加了新命令之后，原有子命令必须一个都没少、参数照旧。"""
    parser = cli.build_parser()
    actions = [a for a in parser._subparsers._group_actions if a.choices]  # noqa: SLF001
    names = set(actions[0].choices)
    for name in ("check", "download", "run", "translate", "cache", "db", "highlight",
                 "assets", "prm", "gui", "ai", "montage"):
        assert name in names, f"原有子命令 {name} 不见了"
    assert "dance-montage" in names, names
    # 原有命令的参数解析没被动过
    args = parser.parse_args(["run", "--limit", "3"])
    assert args.command == "run" and args.limit == 3
    args = parser.parse_args(["gui"])
    assert args.command == "gui"


def test_dance_help_lists_every_action() -> None:
    """新命令的动作清单要齐 —— 一期要求的四个必填参数也都在。"""
    parser = cli.build_parser()
    args = parser.parse_args(["dance-montage", "stats", "--song", "1"])
    assert args.action == "stats" and args.song == "1"
    for action in ("songs", "align", "slice", "materials", "recommend",
                   "remix", "stats", "history", "gui"):
        assert parser.parse_args(["dance-montage", action]).action == action
    full = parser.parse_args(["dance-montage", "remix", "--song", "2", "--slice", "1.5",
                             "--sources", "a.mp4", "b.mp4", "--versions", "3",
                             "--out", "x", "--seed", "7", "--no-recommend"])
    assert full.slice == 1.5 and full.versions == 3 and full.no_recommend is True
    assert full.sources == ["a.mp4", "b.mp4"]


def test_missing_song_is_a_usage_error(work: Path) -> None:
    """少了 --song 要返回 2（参数错），而不是抛栈或返回 0。"""
    cfg_path, _ = _scene(work)
    assert _run(cfg_path, "stats") == 2
    assert _run(cfg_path, "stats", "--song", "999") == 2


def test_full_chain_through_cli(work: Path) -> None:
    """对齐 → 切片 → 查素材 → 出片 → 统计 → 历史，全走命令行。"""
    cfg_path, out_dir = _scene(work)
    song = work / "dance_songs" / "target.wav"

    assert _run(cfg_path, "align", "--song", str(song),
                "--sources", str(work / "input")) == 0
    assert _run(cfg_path, "songs") == 0
    assert _run(cfg_path, "slice", "--song", "1", "--sources", str(work / "input"),
                "--slice", "2.0") == 0
    assert list((work / "dance_materials").glob("*.mp4")), "一条素材文件都没落盘"

    assert _run(cfg_path, "materials", "--song", "1") == 0
    assert _run(cfg_path, "materials", "--song", "1", "--preset", "never_used") == 0
    assert _run(cfg_path, "recommend", "--song", "1", "--seed", "42") == 0
    assert _run(cfg_path, "recommend", "--song", "1", "--verify", "1") == 0

    # 只出计划不渲染：应该很快，且不产出 mp4
    before = set(out_dir.glob("*.mp4"))
    assert _run(cfg_path, "remix", "--song", "1", "--plan-only", "--versions", "1") == 0
    assert set(out_dir.glob("*.mp4")) == before, "--plan-only 竟然渲染了"

    assert _run(cfg_path, "remix", "--song", "1", "--versions", "1", "--seed", "5") == 0
    made = sorted(out_dir.glob("*.mp4"))
    assert made, "remix 没产出成品"
    assert not list(out_dir.glob("*.mute.mp4")), "无声中间文件没删干净"

    assert _run(cfg_path, "stats", "--song", "1") == 0
    assert _run(cfg_path, "history", "--song", "1") == 0
    assert _run(cfg_path, "history", "--song", "1", "--recount") == 0
    print(f"  CLI 成品：{[p.name for p in made]}")


def test_manual_selection_through_cli(work: Path) -> None:
    """`--no-recommend --manual x.json` 纯手动出片。"""
    cfg_path, out_dir = _scene(work)
    song = work / "dance_songs" / "target.wav"
    assert _run(cfg_path, "slice", "--song", str(song),
                "--sources", str(work / "input" / "dancer0.mp4")) == 0

    from vidscribe.config import Config
    from vidscribe.db import open_db

    cfg = Config.load(work, cfg_path)
    db = open_db(cfg)
    try:
        rows = db.connect().execute(
            "SELECT segment_index, MIN(id) AS id FROM dance_materials "
            "WHERE target_song_id = 1 GROUP BY segment_index").fetchall()
        manual = {str(int(r["segment_index"])): int(r["id"]) for r in rows}
    finally:
        db.close()
    assert len(manual) >= 3, manual
    plan_file = work / "manual.json"
    plan_file.write_text(json.dumps(manual), encoding="utf-8")

    assert _run(cfg_path, "remix", "--song", "1", "--no-recommend",
                "--manual", str(plan_file)) == 0
    assert list(out_dir.glob("*.mp4")), "纯手动没出片"
    # 关了推荐却不给 manual → 明确报参数错
    assert _run(cfg_path, "remix", "--song", "1", "--no-recommend") != 0


TESTS = (
    test_existing_commands_still_parse,
    test_dance_help_lists_every_action,
    test_missing_song_is_a_usage_error,
    test_full_chain_through_cli,
    test_manual_selection_through_cli,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancecli_"))
        try:
            if fn.__code__.co_argcount:
                fn(work)
            else:
                fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001 - 意外也要报出来
            import traceback
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
            traceback.print_exc()
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
