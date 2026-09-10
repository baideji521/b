"""素材矩阵的拖拽规则：列内随便拖，跨段落一律拒绝。

真实的鼠标拖拽（QDrag）在离屏环境下跑不起来，所以测的是**规则本身**：
`accepts()` / `MaterialColumn.drop_payload()` / `TimelineSlot.drop_payload()`
——界面上的 dragEnter/drop 事件全都只是把 MIME 解出来交给这三个东西。

  X1  同段落：拖到最上面就是"这一格改用它"，⭐跟着走
  X2  跨段落：拒绝、什么都不改，而且要说清为什么
  X3  FINAL TIMELINE 的槽同一套规则
  X4  MIME 打包/解包：不是我们的东西、或者坏的，一律当没有
  X5  整个面板：picks() 只反映 FINAL TIMELINE 上摆着的那份

可以 `pytest tests/test_dance_matrix.py`，也可以 `python tests/test_dance_matrix.py`。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")       # 必须在 import PyQt5 之前

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt5.QtCore import QMimeData                            # noqa: E402
from PyQt5.QtWidgets import QApplication                      # noqa: E402

# 必须在 QApplication 之前 import `vidscribe.gui`（它钉 Qt 插件路径）
from vidscribe.gui.dance_montage import matrix_panel as mx     # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _rows(prefix: str, count: int, segment: int) -> list[dict]:
    """素材 id 从 1 起（库里是自增主键，0 不会出现）；每条素材属于一个视频（= 一行）。"""
    return [{"material_id": segment * 100 + i + 1, "video_id": i + 1,
             "video_name": f"{prefix}{i}.mp4", "label": f"{prefix}{i}.mp4",
             "segment_index": segment, "target_start": segment * 1.0,
             "target_end": segment * 1.0 + 1.0}
            for i in range(count)]


def _payload(material_id: int, segment: int, label: str = "x.mp4") -> dict:
    return {"material_id": material_id, "segment_index": segment, "label": label,
            "video_id": 1, "video_name": label}


# ================================================================== X1 ~ X2
def test_dragging_a_cell_up_sets_who_this_segment_uses() -> None:
    """X1：把某个视频在 S3 的格子拖到实时播放行 → 这一段就改用它。"""
    cell = mx.RealtimeCell(2, "S3")
    cell.set_material(_payload(201, 2, "dancer0.mp4"))
    picks: list[tuple[int, int]] = []
    cell.replaced.connect(lambda seg, mid: picks.append((seg, mid)))

    assert cell.material_id == 201
    assert cell.drop_payload(_payload(203, 2, "dancer2.mp4")) is True
    assert cell.material_id == 203
    assert "dancer2.mp4" in cell.body.text()
    assert picks and picks[-1] == (2, 203), picks


def test_crossing_segments_is_refused_and_changes_nothing() -> None:
    """X2：S4 的素材拖到 S1 那一格 → 拒绝、什么都不改、而且说清原因。"""
    cell = mx.RealtimeCell(0, "S1")
    cell.set_material(_payload(1, 0, "a0.mp4"))
    said: list[tuple[int, int]] = []
    cell.refused.connect(lambda target, came: said.append((target, came)))

    assert cell.drop_payload(_payload(303, 3, "e.mp4")) is False
    assert cell.material_id == 1, "跨段落居然把这一格改了"
    assert said and said[-1] == (0, 3), said
    assert mx.accepts(0, _payload(303, 3)) is False
    assert mx.accepts(3, _payload(303, 3)) is True


# ================================================================== X3 ~ X5
def test_the_matrix_has_one_column_per_segment() -> None:
    """X3：列数完全跟着 Segment 走 —— 4 段 4 列，加一刀就 5 列，取消就回 4 列。"""
    panel = mx.MatrixPanel()
    four = [{"index": i, "title": f"S{i + 1}", "span": (float(i), float(i + 1)),
             "materials": _rows("a", 2, i)} for i in range(4)]
    panel.load(four)
    assert len(panel.realtime) == 4, len(panel.realtime)
    assert len(panel.videos) == 2, panel.videos          # 行 = 视频
    assert len(panel.cells) == 8, len(panel.cells)       # 4 列 × 2 行

    panel.load(four + [{"index": 4, "title": "S5", "span": (4.0, 5.0),
                        "materials": _rows("a", 2, 4)}])
    assert len(panel.realtime) == 5, "切了一刀，矩阵没长出第五列"

    panel.load(four)
    assert len(panel.realtime) == 4, "取消切分之后列没减回去"


def test_mime_packing_ignores_anything_that_is_not_ours() -> None:
    """X4：MIME 打包/解包。别处拖来的文字、坏数据，一律当没有。"""
    payload = _payload(7, 1, "c.mp4")
    assert mx.unpack(mx.pack(payload)) == payload

    plain = QMimeData()
    plain.setText("随便一段文字")
    assert mx.unpack(plain) == {}
    broken = QMimeData()
    broken.setData(mx.MIME, b"{not json")
    assert mx.unpack(broken) == {}
    assert mx.accepts(1, {}) is False


def test_realtime_row_is_what_the_final_cut_reads() -> None:
    """X5：picks() 只看实时播放行；空着的段落就是空的，不会拿别的段顶上。"""
    panel = mx.MatrixPanel()
    panel.load([
        {"index": 0, "title": "S1", "span": (0.0, 1.0), "materials": _rows("a", 2, 0),
         "current": 1},
        {"index": 1, "title": "S2", "span": (1.0, 2.0), "materials": _rows("b", 3, 1),
         "current": 101},
        {"index": 2, "title": "S3", "span": (2.0, 3.0), "materials": []},
    ])
    seen: list[dict] = []
    panel.picks_changed.connect(seen.append)

    assert panel.picks() == {0: 1, 1: 101}, panel.picks()   # S3 没有候选，就是空的
    assert panel.current_payload(2) == {}, "S3 明明没素材"

    assert panel.choose(1, 103) is True
    assert panel.picks()[1] == 103, panel.picks()
    assert seen and seen[-1][1] == 103

    # 跨段落：picks 不许变，而且状态栏能收到一句人话
    words: list[str] = []
    panel.refused.connect(words.append)
    before = panel.picks()
    assert panel.realtime[0].drop_payload(_payload(103, 1)) is False
    assert panel.picks() == before
    assert words and "S2" in words[-1] and "S1" in words[-1], words
    assert panel.choose(0, 103) is False, "跨段落连 choose() 也不许"


def test_current_segment_follows_the_master_audio() -> None:
    """X6：主音频播到哪一段，就是哪一列在亮 —— 它只是显示，不改任何选择。"""
    panel = mx.MatrixPanel()
    panel.load([{"index": i, "title": f"S{i + 1}", "span": (float(i), float(i + 1)),
                 "materials": _rows("a", 1, i)} for i in range(3)])
    before = panel.picks()
    panel.set_current_segment(1)
    assert panel.current_segment == 1
    assert panel.realtime[1].styleSheet() != panel.realtime[0].styleSheet()
    assert panel.picks() == before, "只是高亮，居然把选择改了"


def test_the_panel_saves_and_restores_the_realtime_row() -> None:
    """X7：拖一下就落库（最终选择 + 候选顺序 + 人工流水账）；重开面板能恢复。"""
    from dance_fixtures import (
        fake_alignment,
        fake_material,
        fake_song,
        fake_video,
        make_project,
    )
    from vidscribe.dance import material_repository as repo

    work = Path(tempfile.mkdtemp(prefix="dancematrix_"))
    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=20.0)
        video_id = fake_video(db, "a.mp4")
        fake_alignment(db, song_id, video_id)
        other = fake_video(db, "b.mp4")
        fake_alignment(db, song_id, other, offset=2.0)
        first = fake_material(db, song_id, video_id, 0)
        second = fake_material(db, song_id, other, 0)

        rows = [{"material_id": first, "video_id": video_id, "video_name": "a.mp4",
                 "label": "a.mp4", "segment_index": 0},
                {"material_id": second, "video_id": other, "video_name": "b.mp4",
                 "label": "b.mp4", "segment_index": 0}]
        panel = mx.MatrixPanel(db=db, song_id=song_id)
        panel.load([{"index": 0, "title": "S1", "span": (0.0, 2.0),
                     "current": first, "materials": rows}])
        assert panel.picks() == {0: first}

        assert panel.choose(0, second) is True
        assert panel.picks() == {0: second}
        assert repo.final_selections(db, song_id) == {0: second}, "拖完没落库"
        assert repo.candidate_order(db, song_id, 0) == [first, second]
        logged = repo.manual_selections(db, song_id)
        assert logged and int(logged[0]["material_id"]) == second, "人工选择没记流水账"
        assert float(logged[0]["segment_end"]) == 2.0

        # 重新开一个面板：按库里那份摆回来（这就是"重开工程能恢复"）
        again = mx.MatrixPanel(db=db, song_id=song_id)
        ordered = repo.order_materials(repo.get_candidates(db, song_id, 0),
                                       repo.candidate_order(db, song_id, 0))
        again.load([{"index": 0, "title": "S1", "span": (0.0, 2.0), "current": second,
                     "materials": [
                         {"material_id": m.id, "video_id": int(m.source_video_id),
                          "video_name": Path(m.file_path).name,
                          "label": Path(m.file_path).name, "segment_index": 0}
                         for m in ordered]}])
        assert again.picks() == {0: second}, again.picks()

        # 撤销 → 回到上一份，而且库里也跟着回去
        assert again.undo() is False, "刚打开就有可撤销的东西？"
        panel.undo()
        assert panel.picks() == {0: first}, panel.picks()
        assert repo.final_selections(db, song_id) == {0: first}
        panel.redo()
        assert panel.picks() == {0: second}, panel.picks()
    finally:
        db.close()
        shutil.rmtree(work, ignore_errors=True)


def test_candidate_pools_are_isolated_per_segment() -> None:
    """X8：仓库按「歌 + 段落」隔离 —— 查 S1 只会拿到 S1 的候选，绝不串段。"""
    from dance_fixtures import (
        fake_alignment,
        fake_material,
        fake_song,
        fake_video,
        make_project,
    )
    from vidscribe.dance import material_repository as repo

    work = Path(tempfile.mkdtemp(prefix="dancepool_"))
    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=12.0)
        first = fake_video(db, "a.mp4")
        second = fake_video(db, "b.mp4")
        fake_alignment(db, song_id, first)
        fake_alignment(db, song_id, second, offset=1.0)
        made = {}
        for segment in (0, 1, 2):
            made[segment] = {fake_material(db, song_id, first, segment),
                             fake_material(db, song_id, second, segment)}

        for segment in (0, 1, 2):
            pool = repo.get_candidates(db, song_id, segment)
            assert {m.id for m in pool} == made[segment], segment
            assert all(int(m.segment_index) == segment for m in pool), "候选池串段了"
        assert repo.candidate_counts(db, song_id) == {0: 2, 1: 2, 2: 2}

        # 歌 / 音频两个名字分开存，片段能追溯到这两样
        info = repo.repository_summary(db, song_id)
        assert info["clip_count"] == 6, info
        assert info["audio_name"] and info["song_name"], info
        assert info["song_id"] == song_id
    finally:
        db.close()
        shutil.rmtree(work, ignore_errors=True)


TESTS = (

    test_dragging_a_cell_up_sets_who_this_segment_uses,
    test_crossing_segments_is_refused_and_changes_nothing,
    test_the_matrix_has_one_column_per_segment,
    test_mime_packing_ignores_anything_that_is_not_ours,
    test_realtime_row_is_what_the_final_cut_reads,
    test_current_segment_follows_the_master_audio,
    test_the_panel_saves_and_restores_the_realtime_row,
    test_candidate_pools_are_isolated_per_segment,
)



def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancematrix_"))
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
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


