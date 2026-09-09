"""使用历史与编辑计划（技术指导第九、十四、十五节 + 第二十二节第 17~19 条）。

一条总原则：**事件流水是唯一事实来源，所有计数都必须能重算出来。**
所以这里反复做同一件事 —— 记事件、看计数、重算、比对。

  T1  四种计数各有各的口径：candidate ≠ use ≠ montage ≠ output，不许互相顶包
  T2  `render_failed` 一次都不许推 `output_count`
  T3  recount 能从流水把计数完全重建（改坏计数后重算必须修回来）
  T4  位置级使用统计：同一条素材在不同位置各算各的
  T5  编辑计划成片时间连续铺、引用 material_id、历史版本永不覆盖
  T6  validate 能抓出空计划 / 文件不在盘上 / 区间非法 / 位置填不满

素材全部直接塞库，不编码任何视频，所以这套测试是秒级的。
可以 `pytest tests/test_dance_history.py`，也可以 `python tests/test_dance_history.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import fake_library, make_project              # noqa: E402
from vidscribe.dance import history                                # noqa: E402
from vidscribe.dance import material_repository as repo            # noqa: E402
from vidscribe.dance import material_selection as selection        # noqa: E402
from vidscribe.dance import montage_timeline, statistics           # noqa: E402
from vidscribe.dance.types import DanceMontageClip, MaterialScore  # noqa: E402


def _clips(materials: dict, people, positions: int) -> list[DanceMontageClip]:
    """按"人物轮流"铺一版计划，成片时间连续。"""
    out = []
    for pos in range(positions):
        person = people[pos % len(people)]
        out.append(DanceMontageClip(
            order_index=pos, segment_index=pos, material_id=materials[(person, pos)],
            target_start=pos * 2.0, target_end=pos * 2.0 + 2.0,
            source_start=pos * 2.0, source_end=pos * 2.0 + 2.0,
            file_path=f"C:/fake/{person}_{pos}.mp4", person=person, source_video_id=1))
    return out


def test_four_counts_do_not_cover_for_each_other(work: Path) -> None:
    """candidate / use / montage / output 四个计数各管一件事，口径不许串。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A", "小B"))
        mid = materials[("小A", 0)]

        # 只是进了候选池：candidate 涨，其余全不动
        pool = selection.build_pool(db, song_id, 0)
        history.note_candidates(db, [pool])
        one = repo.get_material(db, mid)
        assert one.candidate_count == 1, one.candidate_count
        assert one.use_count == 0 and one.montage_count == 0 and one.output_count == 0, \
            "只是被考虑过，使用/出片次数就不该动"

        # 进了某一版编辑计划：use_count 才涨
        montage_id = repo.ensure_montage(db, target_song_id=song_id, name="t",
                                         slice_duration=2.0)
        clips = _clips(materials, ("小A", "小B"), 3)
        version_id = repo.save_version(db, montage_id=montage_id, version_index=1,
                                       signature="sig-1", strategy_id=None,
                                       recommendation_run_id=None, clips=clips,
                                       duration=6.0, repeat=None, timeline_json={})
        history.note_montage(db, version_id, montage_id, clips)
        two = repo.get_material(db, mid)
        assert two.use_count == 1, two.use_count
        assert two.montage_count == 1, two.montage_count
        assert two.output_count == 0, "还没渲染成功就涨了出片次数"
        assert two.candidate_count == 1, "记 montage 不该顺手改 candidate"
        assert two.last_used_at and two.first_used_at, "用过了却没记时间"

        # 渲染成功：output_count 才涨
        history.note_render(db, version_id, montage_id, clips, ok=True)
        three = repo.get_material(db, mid)
        assert three.output_count == 1, three.output_count
        assert three.use_count == 1, "渲染成功不该再推一次 use_count"
        assert three.last_output_at, "出片了却没记出片时间"
    finally:
        db.close()


def test_render_failure_never_touches_output_count(work: Path) -> None:
    """技术指导第九节的硬规矩：失败只留痕，绝不加出片次数。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A", "小B"))
        montage_id = repo.ensure_montage(db, target_song_id=song_id, name="t",
                                         slice_duration=2.0)
        clips = _clips(materials, ("小A", "小B"), 3)
        version_id = repo.save_version(db, montage_id=montage_id, version_index=1,
                                       signature="sig-1", strategy_id=None,
                                       recommendation_run_id=None, clips=clips,
                                       duration=6.0, repeat=None, timeline_json={})
        history.note_montage(db, version_id, montage_id, clips)
        for _ in range(3):
            history.note_render(db, version_id, montage_id, clips, ok=False,
                                detail={"error": "编码失败"})

        events = history.event_summary(db, song_id)
        assert events.get("render_failed", 0) == 3 * len(clips), events
        assert events.get("render_success", 0) == 0, events
        total = db.connect().execute(
            "SELECT COALESCE(SUM(output_count), 0) FROM dance_materials "
            "WHERE target_song_id = ?", (song_id,)).fetchone()[0]
        assert int(total) == 0, f"失败了 3 次却涨了 {total} 次出片"
        # 失败也要留痕，方便查"为什么这一版没出来"
        assert any(str(row["event"]) == "render_failed"
                   for row in history.material_events(db, clips[0].material_id))
    finally:
        db.close()


def test_counts_can_be_rebuilt_from_the_ledger(work: Path) -> None:
    """把计数手动改坏，recount 必须能按流水完全修回来。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A", "小B"))
        montage_id = repo.ensure_montage(db, target_song_id=song_id, name="t",
                                         slice_duration=2.0)
        clips = _clips(materials, ("小A", "小B"), 3)
        version_id = repo.save_version(db, montage_id=montage_id, version_index=1,
                                       signature="sig-1", strategy_id=None,
                                       recommendation_run_id=None, clips=clips,
                                       duration=6.0, repeat=None, timeline_json={})
        history.note_montage(db, version_id, montage_id, clips)
        history.note_render(db, version_id, montage_id, clips, ok=True)
        before = {c.material_id: repo.get_material(db, c.material_id).to_dict()
                  for c in clips}

        # 库被人手改花了
        db.connect().execute(
            "UPDATE dance_materials SET use_count=99, output_count=99, montage_count=99 "
            "WHERE target_song_id = ?", (song_id,))
        report = history.recount_song(db, song_id)
        assert report["changed"] >= len(clips), report
        for c in clips:
            now = repo.get_material(db, c.material_id)
            assert now.use_count == before[c.material_id]["use_count"], now.use_count
            assert now.output_count == before[c.material_id]["output_count"]
            assert now.montage_count == before[c.material_id]["montage_count"]
        # 再重算一次不该有任何变化（幂等）
        assert history.recount_song(db, song_id)["changed"] == 0
    finally:
        db.close()


def test_position_usage_is_per_position(work: Path) -> None:
    """同一条素材在不同位置各算各的 —— 一期第三十九节要的位置级统计。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A",))
        mid = materials[("小A", 0)]
        montage_id = repo.ensure_montage(db, target_song_id=song_id, name="t",
                                         slice_duration=2.0)
        # 同一条素材，一次放在位置 0，一次放在位置 2
        for index, pos in enumerate((0, 2), start=1):
            clip = DanceMontageClip(order_index=0, segment_index=pos, material_id=mid,
                                   target_start=0.0, target_end=2.0,
                                   source_start=0.0, source_end=2.0,
                                   file_path="C:/fake/x.mp4", person="小A",
                                   source_video_id=1)
            version_id = repo.save_version(db, montage_id=montage_id, version_index=index,
                                           signature=f"s{index}", strategy_id=None,
                                           recommendation_run_id=None, clips=[clip],
                                           duration=2.0, repeat=None, timeline_json={})
            history.note_montage(db, version_id, montage_id, [clip])

        usage = history.position_usage(db, song_id)
        assert usage.get((0, mid)) == 1, usage
        assert usage.get((2, mid)) == 1, usage
        assert (1, mid) not in usage, "位置 1 从没用过它，却被记了一笔"
        # 总使用次数是两次，但每个位置各一次 —— 这正是位置级统计存在的理由
        assert repo.get_material(db, mid).use_count == 2
    finally:
        db.close()


def test_timeline_is_continuous_and_references_materials(work: Path) -> None:
    """编辑计划：成片时间连续铺、引用 material_id、位置身份不丢。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=5, people=("小A", "小B"))
        from vidscribe.dance.types import FilterSpec

        lookup = {m.id: m for m in selection.find_materials(
            db, FilterSpec(target_song_id=song_id))}

        # 故意跳过位置 2：成片该连续铺，而不是留一段真空
        picks = [MaterialScore(material_id=materials[("小A", pos)], segment_index=pos,
                               final_score=1.0 - pos * 0.1, breakdown={"final_score": 1.0})
                 for pos in (0, 1, 3, 4)]
        context = montage_timeline.build_timeline(
            db, song_id, picks, lookup, song_path="C:/fake/song.wav", slice_duration=2.0)
        assert len(context.clips) == 4
        assert [c.target_start for c in context.clips] == [0.0, 2.0, 4.0, 6.0], \
            "成片时间没连续铺（空档会在成片里留真空）"
        assert [c.segment_index for c in context.clips] == [0, 1, 3, 4], \
            "素材原本对应的音乐位置丢了"
        assert abs(context.duration - 8.0) < 1e-6, context.duration
        assert context.signature, "没有组合签名，没法和历史版本比"
        assert all(c.material_id in lookup for c in context.clips)
        # 计划只报"文件不在盘上"这一条问题（假素材本来就没有文件）
        problems = montage_timeline.validate(context)
        assert problems and all("不在盘上" in p for p in problems), problems
        # 位置填不满要主动说出来
        short = montage_timeline.validate(context, expected_positions=5)
        assert any("5" in p and "短" in p for p in short), short
    finally:
        db.close()


def test_versions_are_never_overwritten(work: Path) -> None:
    """历史版本永不覆盖：存第二版是新增，第一版原样留着（技术指导第十四节）。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A", "小B"))
        from vidscribe.dance.types import FilterSpec

        lookup = {m.id: m for m in selection.find_materials(
            db, FilterSpec(target_song_id=song_id))}
        montage_id = repo.ensure_montage(db, target_song_id=song_id, name="t",
                                         slice_duration=2.0)

        def plan(people):
            picks = [MaterialScore(material_id=materials[(people[pos % len(people)], pos)],
                                   segment_index=pos, final_score=1.0, breakdown={})
                     for pos in range(3)]
            return montage_timeline.build_timeline(db, song_id, picks, lookup,
                                                   song_path="C:/fake/song.wav",
                                                   slice_duration=2.0,
                                                   montage_id=montage_id)

        first_id = montage_timeline.save(db, plan(("小A", "小B")))
        second_id = montage_timeline.save(db, plan(("小B", "小A")))
        assert first_id != second_id
        rows = repo.list_versions(db, montage_id)
        assert [int(r["version_index"]) for r in rows] == [1, 2] or \
               sorted(int(r["version_index"]) for r in rows) == [1, 2], \
            [dict(r) for r in rows]
        one = montage_timeline.load(db, first_id)
        two = montage_timeline.load(db, second_id)
        assert one is not None and two is not None
        assert one.signature != two.signature, "两版签名一样 —— 等于没换"
        assert [c.material_id for c in one.clips] != [c.material_id for c in two.clips]
        # 第一版的内容一个字都没被第二版改动
        assert len(one.clips) == 3 and one.version_index == 1
    finally:
        db.close()


def test_validate_catches_broken_plans(work: Path) -> None:
    """空计划、区间非法、时间不连续，三种坏计划都要在开始编码之前被拦下来。"""
    from vidscribe.dance.types import DanceMontageContext

    empty = DanceMontageContext(target_song_id=1)
    assert montage_timeline.validate(empty), "空计划竟然通过了校验"
    assert "一格都没有" in montage_timeline.validate(empty)[0]

    bad = DanceMontageContext(target_song_id=1, clips=[
        DanceMontageClip(order_index=0, segment_index=0, material_id=1,
                         target_start=2.0, target_end=1.0, source_start=0.0,
                         source_end=2.0, file_path="C:/fake/a.mp4"),
    ])
    assert any("非法" in p for p in montage_timeline.validate(bad)), \
        montage_timeline.validate(bad)

    gap = DanceMontageContext(target_song_id=1, clips=[
        DanceMontageClip(order_index=0, segment_index=0, material_id=1,
                         target_start=0.0, target_end=2.0, source_start=0.0,
                         source_end=2.0, file_path="C:/fake/a.mp4"),
        DanceMontageClip(order_index=1, segment_index=1, material_id=2,
                         target_start=5.0, target_end=7.0, source_start=0.0,
                         source_end=2.0, file_path="C:/fake/b.mp4"),
    ])
    assert any("不连续" in p for p in montage_timeline.validate(gap)), \
        montage_timeline.validate(gap)


def test_statistics_come_from_sql_not_guesses(work: Path) -> None:
    """统计面板的数字必须来自库，且和事件流水对得上。"""
    cfg, db = make_project(work)
    try:
        song_id, videos, materials = fake_library(db, positions=4, people=("小A", "小B"))
        overview = statistics.song_overview(db, song_id)
        assert int(overview["materials"]["total"]) == 8, overview["materials"]
        assert int(overview["materials"]["never_used"]) == 8
        assert int(overview["materials"]["sources"]) == 2
        assert int(overview["materials"]["positions"]) == 4

        coverage = statistics.position_coverage(db, song_id)
        assert [c["segment_index"] for c in coverage] == [0, 1, 2, 3], coverage
        assert all(c["materials"] == 2 and c["persons"] == 2 for c in coverage), coverage

        people = statistics.person_breakdown(db, song_id)
        assert {p["person"] for p in people} == {"小A", "小B"}, people
        assert all(p["materials"] == 4 for p in people), people
    finally:
        db.close()


TESTS = (
    test_four_counts_do_not_cover_for_each_other,
    test_render_failure_never_touches_output_count,
    test_counts_can_be_rebuilt_from_the_ledger,
    test_position_usage_is_per_position,
    test_timeline_is_continuous_and_references_materials,
    test_versions_are_never_overwritten,
    test_validate_catches_broken_plans,
    test_statistics_come_from_sql_not_guesses,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancehist_"))
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
