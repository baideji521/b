"""卡帧抖动（stutter）：只改帧序，不改时长。

盯的是这套效果唯一不许破的那条规矩 —— **帧数守恒**。时长一变，成片就前移、
音乐就漂，这和首尾补边界帧是同一条约束。

  E1  没有抖动点 = 恒等映射（调用方可以直接跳过）
  E2  窗口内按档保持，窗口一过立刻追回正确位置
  E3  帧数恒等于 count，一帧不多一帧不少
  E4  跨段落的点必须裁到段内，完全不相交的丢掉
  E5  太短 / 力度不合法的点一律不算
  E6  真渲一条素材：有抖动点，时长和没有时**完全一样**

可以 `pytest tests/test_dance_frame_effects.py`，也可以 `python tests/test_dance_frame_effects.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vidscribe.dance import frame_effects as fx      # noqa: E402


def test_no_points_means_no_change() -> None:
    """E1：没打点就是恒等映射。"""
    assert fx.frame_plan(5, 30.0, []) == [0, 1, 2, 3, 4]
    assert fx.frame_plan(5, 30.0, None) == [0, 1, 2, 3, 4]
    assert fx.parse([{"at": 1.0, "duration": 0.0}]) == []      # 时长 0 不算点


def test_window_holds_then_catches_up() -> None:
    """E2：窗口内每 2 帧一档；出了窗口立刻回到 k 本身（不许一直迟半拍）。"""
    # 10fps 便于手算：0.2s → 第 2 帧，窗口 0.4s → 4 帧（2,3,4,5）
    plan = fx.frame_plan(8, 10.0, [fx.Stutter(0.2, 0.4, hold=2)])
    assert plan == [0, 1, 2, 2, 4, 4, 6, 7], plan
    # hold=3：2,2,2 然后 5
    plan = fx.frame_plan(8, 10.0, [fx.Stutter(0.2, 0.4, hold=3)])
    assert plan == [0, 1, 2, 2, 2, 5, 6, 7], plan


def test_frame_count_never_changes() -> None:
    """E3：帧数守恒 —— 这是时长不漂的全部保证。"""
    for count in (1, 2, 7, 30, 121):
        points = [fx.Stutter(0.1, 0.5, 2), fx.Stutter(1.0, 0.3, 4)]
        plan = fx.frame_plan(count, 30.0, points)
        assert len(plan) == count, (count, len(plan))
        assert all(0 <= value < max(1, count) for value in plan), plan
        assert all(value <= index for index, value in enumerate(plan)), plan


def test_points_are_clipped_into_their_segment() -> None:
    """E4：跨段落的点裁到段内，不相交的丢掉。一个点只影响它所在那一段。"""
    points = [fx.Stutter(9.8, 0.6, 2),      # 跨 S1/S2 边界（10.0）
              fx.Stutter(3.0, 0.2, 2),      # 完全在别的段
              fx.Stutter(10.5, 0.3, 2)]     # 完全在 S2 里
    inside = fx.clip_to(points, 10.0, 12.0)
    assert [(round(p.at, 3), round(p.duration, 3)) for p in inside] == \
        [(10.0, 0.4), (10.5, 0.3)], inside


def test_silly_points_are_ignored() -> None:
    """E5：太短的、力度不合法的、根本不是字典的，一律不算。"""
    assert fx.parse([{"at": 1.0, "duration": 0.01}]) == []        # 比 MIN_SECONDS 还短
    assert fx.parse(["不是字典", None, 42]) == []
    only = fx.parse([{"at": 1.0, "duration": 0.3, "hold": 99}])
    assert only and only[0].hold == fx.MAX_HOLD, only            # 力度夹到上限
    only = fx.parse([{"at": 1.0, "duration": 0.3, "hold": 1}])
    assert only and only[0].hold == 2, only                      # hold<2 等于没抖，抬到 2
    # 排序：打点顺序随便，出来一定按时间
    order = fx.parse([{"at": 2.0, "duration": 0.2}, {"at": 0.5, "duration": 0.2}])
    assert [round(p.at, 3) for p in order] == [0.5, 2.0], order


def test_rendering_with_stutter_keeps_the_exact_duration() -> None:
    """E6：真渲一条素材 —— 有抖动点和没有，**帧数完全一样**。"""
    from dance_fixtures import song, write_video
    from vidscribe.dance import material_slice
    from vidscribe.dance.media_backend import Canvas, resolve
    from vidscribe.dance.types import SliceSpec

    work = Path(tempfile.mkdtemp(prefix="dancefx_"))
    try:
        source = work / "girl.mp4"
        write_video(source, song(duration=3.0), fps=15.0, width=120, height=160)

        engine = resolve("auto")
        canvas = Canvas(width=120, height=160, fps=15.0)
        spec = SliceSpec(segment_index=0, target_start=10.0, target_end=12.0,
                         source_start=0.5, source_end=2.5)

        plain = material_slice.render_material(source, spec, work / "plain.mp4",
                                               canvas=canvas, backend=engine)
        shaken = material_slice.render_material(
            source, spec, work / "shaken.mp4", canvas=canvas, backend=engine,
            stutters=[{"at": 10.6, "duration": 0.5, "hold": 3}])

        assert shaken["frames"] == plain["frames"], (shaken["frames"], plain["frames"])
        assert abs(shaken["duration"] - plain["duration"]) < 1e-6
        assert abs(shaken["duration"] - spec.duration) < 0.07, shaken["duration"]
        assert len(shaken["stutters"]) == 1, shaken["stutters"]
        # 段落之外的点不该落到这一段上
        away = material_slice.render_material(
            source, spec, work / "away.mp4", canvas=canvas, backend=engine,
            stutters=[{"at": 3.0, "duration": 0.5}])
        assert away["stutters"] == [], away["stutters"]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_pingpong_plays_forward_then_backward() -> None:
    """来回放（回旋镜）：窗口里正放到中点再倒放回来，帧数一样不变。"""
    # 10fps：窗口 0.2→1.0s = 8 帧，来回 1 趟
    plan = fx.frame_plan(12, 10.0, [fx.Stutter(0.2, 0.8, 1, "pingpong")])
    inside = plan[2:10]
    assert inside == [2, 3, 4, 5, 6, 5, 4, 3], inside     # 正放到中点再倒回来
    assert plan[:2] == [0, 1] and plan[10:] == [10, 11], plan
    assert len(plan) == 12

    # 来回 2 趟：一趟 4 帧（2 正 + 2 倒），走两遍
    plan = fx.frame_plan(12, 10.0, [fx.Stutter(0.2, 0.8, 2, "pingpong")])
    assert plan[2:10] == [2, 3, 4, 3, 2, 3, 4, 3], plan[2:10]

    # 来回放允许"来回 1 趟"；抖动那边 1 帧一档等于没抖，会被抬到 2（两套下限各管各的）
    assert fx.parse([{"at": 1.0, "duration": 0.5, "hold": 1,
                      "kind": "pingpong"}]), "来回 1 趟被判成不合法了"
    bumped = fx.parse([{"at": 1.0, "duration": 0.5, "hold": 1}])
    assert bumped and bumped[0].hold == 2 and bumped[0].kind == "stutter", bumped

    # 不认识的玩法退回抖动，绝不静默变成别的效果
    only = fx.parse([{"at": 1.0, "duration": 0.5, "kind": "什么鬼"}])
    assert only and only[0].kind == "stutter", only


def test_rewind_plays_backward_then_forward() -> None:
    """回放：窗口里**先倒放**回去，再正着放回来（倒带重看一遍那种感觉）。"""
    # 10fps：窗口 0.2→1.0s = 8 帧，来回 1 趟
    plan = fx.frame_plan(12, 10.0, [fx.Stutter(0.2, 0.8, 1, "rewind")])
    assert plan[2:10] == [6, 5, 4, 3, 2, 3, 4, 5], plan[2:10]   # 先倒回去再正着放
    assert plan[:2] == [0, 1] and plan[10:] == [10, 11], plan    # 窗口外一律恒等
    assert len(plan) == 12

    # 和「正放→倒放」是相反的相位，别把两种玩法混成一个
    other = fx.frame_plan(12, 10.0, [fx.Stutter(0.2, 0.8, 1, "pingpong")])
    assert other[2:10] != plan[2:10], (other[2:10], plan[2:10])

    # 趟数可以一路调到 30；窗口塞不下那么多趟时按帧数夹住，绝不越界、也不崩
    many = fx.frame_plan(12, 10.0, [fx.Stutter(0.2, 0.8, fx.MAX_TRIPS, "rewind")])
    assert len(many) == 12, len(many)
    assert all(2 <= value <= 9 for value in many[2:10]), many[2:10]
    assert many[2:10] != plan[2:10], "30 趟和 1 趟不该是同一份帧序"
    assert fx.MAX_TRIPS == 30, fx.MAX_TRIPS
    kept = fx.parse([{"at": 1.0, "duration": 0.5, "hold": 30, "kind": "rewind"}])
    assert kept and kept[0].hold == 30, kept        # 30 趟存得下、读得回



TESTS = (


    test_no_points_means_no_change,
    test_window_holds_then_catches_up,
    test_pingpong_plays_forward_then_backward,
    test_rewind_plays_backward_then_forward,
    test_frame_count_never_changes,
    test_points_are_clipped_into_their_segment,
    test_silly_points_are_ignored,
    test_rendering_with_stutter_keeps_the_exact_duration,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
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
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

