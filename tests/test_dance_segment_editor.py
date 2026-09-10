"""主音频编辑区的地基：人声活动分析 + 用户段落模板 + 音谱图缩略。

这一层的定位是**参考 + 编辑**，测试盯的就是这条边界：

  V1  人声段 / 空白段能分出来，中间那段空白被认成"停顿"
  V2  停顿推荐度：停得越久分越高；没有拍网格也不白扣分
  V3  导航：某一刻在唱没在唱、下一处/上一处停顿是哪个
  S1  等间隔起步模板铺满整首歌（末尾零头并进最后一段，不许丢）
  S2  切一刀 / 往前并 / 拖边界，每步都还是合法模板
  S3  拖不动的时候必须报错，不许静默 clamp 到别的位置
  S4  照人声停顿生成模板：切点落在停顿正中间；没有够用的停顿就报错
  S5  JSON 存读一模一样；存坏了抛 SegmentError 而不是返回半个模板
  S6  模板能摊成 TargetPosition（切片那一层只认这个类型）
  G1  音谱图缩略图：形状/类型固定，中频有内容时中间那几行确实更亮

可以 `pytest tests/test_dance_segment_editor.py`，
也可以 `python tests/test_dance_segment_editor.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import SR                                      # noqa: E402
from vidscribe.dance import dsp                                    # noqa: E402
from vidscribe.dance import segment_template as seg                # noqa: E402
from vidscribe.dance import vocal_activity as vocal                # noqa: E402
from vidscribe.dance.types import TargetPosition, VocalPause       # noqa: E402


def _bed(duration: float) -> np.ndarray:
    """伴奏垫底：60Hz 的"底鼓"+ 8kHz 的"镲"，刻意都不在人声带里。"""
    t = np.arange(int(duration * SR), dtype=np.float32) / SR
    kick = 0.5 * np.sin(2 * np.pi * 60.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0)
    hat = 0.08 * np.sin(2 * np.pi * 8000.0 * t)
    return (kick + hat).astype(np.float32)


def _voice(duration: float) -> np.ndarray:
    """"人声"：220Hz 基频 + 前三个谐波，全部落在 300~3400 的人声带附近。"""
    t = np.arange(int(duration * SR), dtype=np.float32) / SR
    out = np.zeros_like(t)
    for k, gain in ((1, 0.35), (2, 0.30), (3, 0.22), (4, 0.15)):
        out += gain * np.sin(2 * np.pi * 220.0 * k * t)
    return out.astype(np.float32)


def _sung(*, lead: float, first: float, gap: float, second: float) -> np.ndarray:
    """伴奏一直在，人声唱两句，中间空 `gap` 秒。"""
    total = lead + first + gap + second
    track = _bed(total)
    voice = np.zeros_like(track)
    at = int(lead * SR)
    voice[at:at + int(first * SR)] = _voice(first)
    at = int((lead + first + gap) * SR)
    voice[at:at + int(second * SR)] = _voice(second)
    return (track + voice).astype(np.float32)


# ================================================================== V1 ~ V3
def test_vocal_and_pause_are_separated() -> None:
    """V1：唱—空—唱 三段能分开，中间那段空白被认成停顿。"""
    pcm = _sung(lead=1.0, first=4.0, gap=1.5, second=4.0)
    activity = vocal.analyze_vocal_activity(pcm, SR)

    sung = [s for s in activity.spans if s.kind == "vocal"]
    assert len(sung) == 2, [s.to_dict() for s in activity.spans]
    assert abs(sung[0].start - 1.0) < 0.3, sung[0].to_dict()
    assert len(activity.pauses) == 1, [p.to_dict() for p in activity.pauses]

    pause = activity.pauses[0]
    assert abs(pause.duration - 1.5) < 0.4, pause.to_dict()
    # 开头那 1 秒纯伴奏不算"停顿"——那是还没开始唱，不是唱着唱着停了
    assert pause.start > 3.0, pause.to_dict()
    assert activity.version == vocal.VOCAL_VERSION


def test_pause_score_rewards_longer_and_survives_without_beats() -> None:
    """V2：停顿越长推荐度越高；没有拍网格时不白扣那 0.2 分。"""
    short = vocal.pause_score(0.35, 0.05, -1.0)
    long = vocal.pause_score(1.50, 0.05, -1.0)
    assert 0.0 <= short < long <= 1.0, (short, long)
    # 贴着拍点应该比差半拍的高
    on_beat = vocal.pause_score(1.0, 0.05, 0.0)
    off_beat = vocal.pause_score(1.0, 0.05, 0.5)
    assert on_beat > off_beat, (on_beat, off_beat)


def test_navigation_answers_where_am_i() -> None:
    """V3：状态栏那几句 —— 现在在唱吗、下一处停顿在哪、上一处在哪。"""
    pcm = _sung(lead=1.0, first=4.0, gap=1.5, second=4.0)
    activity = vocal.analyze_vocal_activity(pcm, SR)
    pause = activity.pauses[0]

    assert activity.speaking_at(3.0) is True
    assert activity.speaking_at(pause.middle) is False
    assert activity.next_pause(2.0) is pause
    assert activity.next_pause(pause.end + 0.1) is None      # 没有了就是没有
    assert activity.previous_pause(pause.end + 0.5) is pause
    assert 0.0 <= activity.at(3.0) <= 1.0


# ================================================================== S1 ~ S6
def test_uniform_template_covers_the_whole_song() -> None:
    """S1：等间隔模板必须铺满整首歌，末尾零头并进最后一段。"""
    plain = seg.uniform(30.0, 2.0)
    assert len(plain.spans) == 15
    assert plain.spans[0].start == 0.0 and plain.spans[-1].end == 30.0
    assert plain.spans[0].name == "S1" and plain.spans[-1].name == "S15"

    # 14.47 秒的歌 2 秒一格：7 整格 + 0.47 秒零头，零头太短 → 并进最后一段
    odd = seg.uniform(14.47, 2.0)
    assert len(odd.spans) == 7, [s.to_dict() for s in odd.spans]
    assert odd.spans[-1].end == 14.47
    assert abs(odd.spans[-1].duration - 2.47) < 1e-6, odd.spans[-1].to_dict()
    seg.validate(odd)                                        # 没缝、没重叠、够长


def test_split_merge_and_drag_keep_the_template_legal() -> None:
    """S2：切一刀 / 往前并 / 拖边界，每一步出来的都还是合法模板。"""
    base = seg.uniform(20.0, 5.0)                            # S1~S4
    cut = seg.split_at(base, 7.5)
    assert len(cut.spans) == 5 and 7.5 in cut.boundaries
    assert cut.source == "manual"                            # 用户动过就记 manual

    merged = seg.merge_at(cut, 2)                            # 把第 3 段并进第 2 段
    assert len(merged.spans) == 4
    seg.validate(merged)

    moved = seg.move_boundary(base, 0, 6.0)
    assert moved.boundaries[0] == 6.0
    assert moved.spans[0].duration == 6.0 and moved.spans[1].start == 6.0
    seg.validate(moved)


def test_illegal_edits_raise_instead_of_clamping() -> None:
    """S3：拖不过去、切不下去，必须报错。静默 clamp 会让素材和音乐悄悄错开。"""
    base = seg.uniform(20.0, 5.0)
    for bad in (5.2, 14.8):                                  # 离邻居太近
        try:

            seg.move_boundary(base, 1, bad)
        except seg.SegmentError as exc:
            assert "留" in str(exc) or "不在" in str(exc), exc
        else:
            raise AssertionError(f"拖到 {bad}s 居然成功了")
    try:
        seg.split_at(base, 5.1)                              # 离 S2 起点 0.1 秒
    except seg.SegmentError as exc:
        assert "太近" in str(exc), exc
    else:
        raise AssertionError("在边界旁边切了一刀还成功了")
    try:
        seg.merge_at(base, 0)                                # 第一段前面没东西
    except seg.SegmentError:
        pass
    else:
        raise AssertionError("第一段往前并居然成功了")


def test_template_from_pauses_cuts_in_the_middle_of_the_gap() -> None:
    """S4：照停顿生成模板 —— 切点落在停顿正中间；没有够用的停顿就报错。"""
    pauses = (VocalPause(index=0, start=4.0, end=5.0, score=0.8),
              VocalPause(index=1, start=11.5, end=12.5, score=0.4))
    built = seg.from_pauses(20.0, pauses)
    assert built.source == "pause"
    assert built.boundaries == (4.5, 12.0), built.boundaries

    # 门槛调高之后只剩一个停顿够用
    picky = seg.from_pauses(20.0, pauses, min_score=0.6)
    assert picky.boundaries == (4.5,), picky.boundaries
    try:
        seg.from_pauses(20.0, pauses, min_score=0.99)
    except seg.SegmentError as exc:
        assert "停顿" in str(exc), exc
    else:
        raise AssertionError("一个停顿都不够用还生成了模板")


def test_json_round_trip_and_a_broken_payload() -> None:
    """S5：存读一模一样；存坏了抛 SegmentError，不返回半个模板。"""
    original = seg.rename(seg.uniform(12.0, 3.0), 1, "副歌")
    again = seg.from_json(seg.to_json(original))
    assert again.boundaries == original.boundaries
    assert again.spans[1].label == "副歌" and again.spans[1].name == "副歌"
    assert again.duration == original.duration

    for broken in ('{"duration": 0}', '{"boundaries": [1]}', '{"duration": 10,'
                   ' "boundaries": [99]}'):
        try:
            seg.from_json(broken)
        except seg.SegmentError:
            pass
        else:
            raise AssertionError(f"{broken} 这种坏数据居然读出了模板")


def test_template_flattens_into_target_positions() -> None:
    """S6：模板摊成 TargetPosition —— 切片那一层只认这个类型，不用改。"""
    template = seg.uniform(9.0, 3.0)
    positions = seg.positions_of(template)
    assert all(isinstance(p, TargetPosition) for p in positions)
    assert [(p.index, p.start, p.end) for p in positions] == \
           [(0, 0.0, 3.0), (1, 3.0, 6.0), (2, 6.0, 9.0)]


def test_snap_pulls_to_the_nearest_reference_point() -> None:
    """吸附：够近就贴上去，不够近原样返回（拖动手感全靠这个）。"""
    beats = (2.0, 4.0, 6.0)
    assert seg.snap(4.05, beats) == 4.0
    assert seg.snap(4.50, beats) == 4.5
    assert seg.snap(4.50, ()) == 4.5


# ======================================================================= G1
def test_spectrogram_image_is_display_only_but_correct() -> None:
    """G1：音谱图缩略图形状/类型固定；中频有内容时中间那几行确实更亮。"""
    quiet = dsp.spectrogram_image(np.zeros(100, dtype=np.float32), SR, columns=40, rows=16)
    assert quiet.shape == (16, 40) and quiet.dtype == np.uint8
    assert quiet.max() == 0                                  # 太短就给全黑，不编内容

    image = dsp.spectrogram_image(_sung(lead=0.5, first=2.0, gap=1.0, second=2.0),
                                  SR, columns=120, rows=48)
    assert image.shape == (48, 120) and image.dtype == np.uint8
    # 行序：row 0 是最高频，最后一行是最低频。这段素材的内容全在低频
    # （60Hz 底鼓 + 220~880Hz 的"人声"），1.5k~5k 那一带是空的
    bottom = float(image[-8:].mean())
    middle = float(image[20:36].mean())
    assert bottom > middle, (bottom, middle)



# ================================================================== R1 ~ R3
def test_template_survives_a_restart(work: Path) -> None:
    """R1：模板存进库、读出来一模一样，而且一首歌同一时刻只有一份是活的。"""
    from dance_fixtures import fake_song, make_project
    from vidscribe.dance import material_repository as repo

    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=20.0)
        edited = seg.split_at(seg.uniform(20.0, 5.0, target_song_id=song_id), 7.5)
        repo.save_segment_template(db, edited)

        back = repo.active_segment_template(db, song_id)
        assert back is not None
        assert back.boundaries == edited.boundaries, (back.boundaries, edited.boundaries)
        assert back.duration == 20.0 and back.source == "manual"
        assert back.name == repo.DEFAULT_TEMPLATE_NAME

        other = repo.save_segment_template(db, seg.uniform(20.0, 4.0, target_song_id=song_id),
                                           name="四秒一段")
        rows = repo.list_segment_templates(db, song_id)
        assert len(rows) == 2
        assert sum(int(r["is_active"]) for r in rows) == 1, "同时有两份活的模板"
        assert repo.active_segment_template(db, song_id).name == "四秒一段"
        assert len(repo.get_segment_template(db, other).spans) == 5

    finally:
        db.close()


def test_saving_the_same_name_updates_instead_of_piling_up(work: Path) -> None:
    """R2：同名就是同一份（幂等）；没有模板时 active 返回 None，不许现编一份。"""
    from dance_fixtures import fake_song, make_project
    from vidscribe.dance import material_repository as repo

    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=12.0)
        assert repo.active_segment_template(db, song_id) is None, "空库居然给了一份模板"

        first = repo.save_segment_template(db, seg.uniform(12.0, 3.0, target_song_id=song_id))
        again = repo.save_segment_template(db, seg.uniform(12.0, 2.0, target_song_id=song_id))
        assert first == again, (first, again)
        assert len(repo.list_segment_templates(db, song_id)) == 1
        assert len(repo.active_segment_template(db, song_id).spans) == 6
    finally:
        db.close()


def test_an_illegal_template_never_reaches_the_database(work: Path) -> None:
    """R3：有缝的模板落库前就被 validate 拦住 —— 库里不许存在半份模板。"""
    from dance_fixtures import fake_song, make_project
    from vidscribe.dance import material_repository as repo
    from vidscribe.dance.types import SegmentSpan, SegmentTemplate

    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=10.0)
        holed = SegmentTemplate(target_song_id=song_id, duration=10.0, spans=(
            SegmentSpan(index=0, start=0.0, end=4.0),
            SegmentSpan(index=1, start=5.0, end=10.0),      # 4~5 秒是个缝
        ))
        try:
            repo.save_segment_template(db, holed)
        except seg.SegmentError as exc:
            assert "缝" in str(exc) or "重叠" in str(exc), exc
        else:
            raise AssertionError("有缝的模板居然存进去了")
        assert repo.list_segment_templates(db, song_id) == []
    finally:
        db.close()


# ================================================================== P1 ~ P2
def test_slicing_follows_the_template_when_there_is_one() -> None:
    """P1：给了模板就按模板切（每段长度可以不一样），没给还是等间隔 —— 老行为不变。"""
    from vidscribe.dance import align_probe, material_slice
    from vidscribe.dance.types import DanceAlignment

    # offset 取负数：source = target − offset = target + 1，整首歌都落在源里面
    align = DanceAlignment(offset=-1.0, confidence=0.9, method="hybrid",
                           waveform_offset=-1.0, waveform_confidence=0.9,
                           status="ok", source_duration=40.0, target_duration=20.0)
    template = seg.split_at(seg.uniform(20.0, 10.0, target_song_id=7), 6.0)   # 6 / 4 / 10
    plan = material_slice.plan_slices(
        align, song_duration=20.0, source_duration=40.0,
        positions=seg.positions_of(template), target_song_id=7)
    assert [round(s.duration, 3) for s in plan.specs] == [6.0, 4.0, 10.0]
    # source = target - offset，一格都不许 clamp
    assert abs(plan.specs[0].source_start - 1.0) < 1e-9
    assert abs(plan.specs[1].source_start - 7.0) < 1e-9

    # 不给 positions 就是老路：等间隔
    plain = material_slice.plan_slices(align, song_duration=20.0, source_duration=40.0,
                                       slice_duration=5.0)
    assert len(plain.specs) == 4
    # 探针和切片必须用同一份位置，否则界面说能用、切片却按另一把尺子
    rows = align_probe.probe_all(align, song_duration=20.0, source_duration=40.0,
                                 positions=seg.positions_of(template))
    assert [round(r.duration, 3) for r in rows] == [6.0, 4.0, 10.0]


def test_ingest_picks_up_the_saved_template(work: Path) -> None:
    """P2：入库切片会自己去库里拿那份模板；没有模板才回落成等间隔。"""
    from dance_fixtures import fake_song, make_project
    from vidscribe.dance import material_ingest as ingest
    from vidscribe.dance import material_repository as repo

    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=20.0)
        assert ingest._template_positions(db, song_id) is None      # noqa: SLF001

        repo.save_segment_template(
            db, seg.split_at(seg.uniform(20.0, 10.0, target_song_id=song_id), 6.0))
        picks = ingest._template_positions(db, song_id)              # noqa: SLF001
        assert picks is not None
        assert [(p.index, p.start, p.end) for p in picks] == \
               [(0, 0.0, 6.0), (1, 6.0, 10.0), (2, 10.0, 20.0)]
    finally:
        db.close()


# ================================================================== V4 ~ V5
def test_cut_zones_wrap_each_pause() -> None:
    """V4：可取区间围着停顿铺开，挨着的合成一带，推荐等级跟着分数走。"""
    activity = vocal.VocalActivity(
        duration=30.0, sample_rate=22050, hop_seconds=0.023,
        pauses=(VocalPause(index=0, start=5.0, end=5.6, score=0.8),
                VocalPause(index=1, start=6.0, end=6.4, score=0.4),
                VocalPause(index=2, start=20.0, end=21.0, score=0.3)))
    zones = vocal.cut_zones(activity, radius=0.5)
    assert len(zones) == 2, zones                     # 前两个挨着 → 合成一带
    first, second = zones
    assert first[0] == 4.5 and first[1] == 6.9, first
    assert first[3] == vocal.zone_level(0.8), first
    assert second[0] == 19.5 and second[1] == 21.5, second
    assert vocal.zone_level(0.8) != vocal.zone_level(0.3)
    # 门槛能把低分的滤掉；区间**不改分段**，它只是参考
    assert len(vocal.cut_zones(activity, radius=0.5, min_score=0.5)) == 1


def test_final_selection_is_checked_by_the_database_too(work: Path) -> None:
    """V5：最终选择、⭐标记、候选顺序都真的落库，而且跨段落的选择**库里也拒绝**。"""
    from dance_fixtures import fake_alignment, fake_material, fake_song, fake_video, make_project
    from vidscribe.dance import material_repository as repo

    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, duration=20.0)
        video_id = fake_video(db, "a.mp4")
        fake_alignment(db, song_id, video_id)
        first = fake_material(db, song_id, video_id, 0)
        second = fake_material(db, song_id, video_id, 1)

        repo.set_final_selection(db, song_id, 0, first)
        assert repo.final_selections(db, song_id) == {0: first}
        # 同一段再设一次就是覆盖，不会多出一行
        repo.set_final_selection(db, song_id, 0, first)
        assert len(repo.final_selections(db, song_id)) == 1

        try:
            repo.set_final_selection(db, song_id, 0, second)   # second 绑在 S2
        except repo.SelectionError as exc:
            assert "S2" in str(exc) and "S1" in str(exc), exc
        else:
            raise AssertionError("跨段落的最终选择居然写进库了")
        assert repo.final_selections(db, song_id) == {0: first}, "被拒绝的写入改动了库"

        # ⭐ 标记：加一次、再点一次就是取消
        repo.add_cut_mark(db, song_id, 7.5)
        assert [round(float(r["moment"]), 3) for r in repo.cut_marks(db, song_id)] == [7.5]
        assert repo.remove_cut_mark(db, song_id, 7.5) is True
        assert repo.cut_marks(db, song_id) == []

        # 候选顺序：存进去、读出来、按它重排
        repo.save_candidate_order(db, song_id, 0, [second, first])
        assert repo.candidate_order(db, song_id, 0) == [second, first]
        materials = repo.materials_at(db, song_id, 0)
        assert [m.id for m in repo.order_materials(materials, [first])] == [first]
    finally:
        db.close()


TESTS = (



    test_vocal_and_pause_are_separated,
    test_pause_score_rewards_longer_and_survives_without_beats,
    test_navigation_answers_where_am_i,
    test_uniform_template_covers_the_whole_song,
    test_split_merge_and_drag_keep_the_template_legal,
    test_illegal_edits_raise_instead_of_clamping,
    test_template_from_pauses_cuts_in_the_middle_of_the_gap,
    test_json_round_trip_and_a_broken_payload,
    test_template_flattens_into_target_positions,
    test_snap_pulls_to_the_nearest_reference_point,
    test_spectrogram_image_is_display_only_but_correct,
    test_template_survives_a_restart,
    test_saving_the_same_name_updates_instead_of_piling_up,
    test_an_illegal_template_never_reaches_the_database,
    test_slicing_follows_the_template_when_there_is_one,
    test_ingest_picks_up_the_saved_template,
    test_cut_zones_wrap_each_pause,
    test_final_selection_is_checked_by_the_database_too,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="danceseg_"))
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




