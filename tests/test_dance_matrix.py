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
    """素材 id 从 1 起（库里是自增主键，0 不会出现）。"""
    return [{"material_id": segment * 100 + i + 1, "label": f"{prefix}{i}.mp4"}
            for i in range(count)]


def _payload(material_id: int, segment: int, label: str = "x.mp4") -> dict:
    return {"material_id": material_id, "segment_index": segment, "label": label}


# ================================================================== X1 ~ X2
def test_dragging_inside_one_column_changes_who_is_used() -> None:
    """X1：把第三张卡片拖到最上面 → 这一列的当前使用就换成它，⭐跟着走。"""
    column = mx.MaterialColumn(2, "S3")
    column.load(_rows("dancer", 3, 2))
    picks: list[tuple[int, int]] = []
    column.reordered.connect(lambda seg, mid: picks.append((seg, mid)))

    assert column.current_pick() == 201
    assert column.item(0).text().startswith("⭐")

    assert column.drop_payload(_payload(203, 2, "dancer2.mp4"), 0) is True
    assert column.current_pick() == 203, column.current_pick()
    assert column.count() == 3, "拖动是移动，不该多出一张卡片"
    assert column.item(0).text().startswith("⭐")
    assert not column.item(1).text().startswith("⭐"), "⭐留在了旧位置"
    assert picks and picks[-1] == (2, 203), picks


def test_crossing_segments_is_refused_and_changes_nothing() -> None:
    """X2：S4 的素材拖到 S1 的列 → 拒绝、一张卡片都不动、而且说清原因。"""
    column = mx.MaterialColumn(0, "S1")
    column.load(_rows("a", 2, 0))
    before = [p["material_id"] for p in column.payloads()]
    said: list[tuple[int, int]] = []
    column.refused.connect(lambda target, came: said.append((target, came)))

    assert column.drop_payload(_payload(303, 3, "e.mp4"), 0) is False
    assert [p["material_id"] for p in column.payloads()] == before
    assert column.count() == 2
    assert said and said[-1] == (0, 3), said
    assert mx.accepts(0, _payload(303, 3)) is False
    assert mx.accepts(3, _payload(303, 3)) is True


# ================================================================== X3 ~ X5
def test_timeline_slot_follows_the_same_rule() -> None:
    """X3：FINAL TIMELINE 的槽同一套规则 —— 同段落替换，跨段落纹丝不动。"""
    slot = mx.TimelineSlot(1, "S2")
    slot.set_material(101, "b1.mp4")
    changed: list[tuple[int, int]] = []
    slot.replaced.connect(lambda seg, mid: changed.append((seg, mid)))

    assert slot.drop_payload(_payload(102, 1, "b2.mp4")) is True
    assert slot.material_id == 102 and "b2.mp4" in slot.body.text()
    assert changed[-1] == (1, 102)

    assert slot.drop_payload(_payload(400, 4, "d.mp4")) is False
    assert slot.material_id == 102, "跨段落居然把槽改了"


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


def test_panel_picks_reflect_the_final_timeline() -> None:
    """X5：面板整体 —— picks() 只反映 FINAL TIMELINE 上摆着的那份。"""
    panel = mx.MatrixPanel()
    panel.load([
        {"index": 0, "title": "S1", "materials": _rows("a", 2, 0)},
        {"index": 1, "title": "S2", "materials": _rows("b", 3, 1)},
        {"index": 2, "title": "S3", "materials": []},
    ])
    seen: list[dict] = []
    panel.picks_changed.connect(seen.append)

    assert len(panel.columns) == 3 and len(panel.slots) == 3
    assert panel.picks() == {0: 1, 1: 101}, panel.picks()   # S3 没有候选，就是空的

    # 在 S2 那一列把第三张拖到最上面 → 槽和 picks 一起变
    panel.columns[1].drop_payload(_payload(103, 1, "b2.mp4"), 0)
    assert panel.picks()[1] == 103, panel.picks()
    assert seen and seen[-1][1] == 103

    # 跨段落：picks 不许变，而且状态栏能收到一句人话
    words: list[str] = []
    panel.refused.connect(words.append)
    before = panel.picks()
    assert panel.slots[0].drop_payload(_payload(103, 1)) is False
    assert panel.picks() == before
    assert words and "S2" in words[-1] and "S1" in words[-1], words


TESTS = (
    test_dragging_inside_one_column_changes_who_is_used,
    test_crossing_segments_is_refused_and_changes_nothing,
    test_timeline_slot_follows_the_same_rule,
    test_mime_packing_ignores_anything_that_is_not_ours,
    test_panel_picks_reflect_the_final_timeline,
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


