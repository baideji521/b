"""固定音乐位置切片（技术指导第七、八节 + 第二十二节第 6~9 条）。

盯的是这个项目最容易被写坏的一处：**时间换算只认一个口径**

    source_time = target_time - offset

以及那条硬规矩：**越界不做 clamp**。clamp 会悄悄产出一段和音乐错位的画面，
而错位是这类项目最难查的 bug —— 所以越界必须显式跳过并写下中文理由。

  T1  map_to_source 正负 offset 都对，且和 DanceAlignment.source_time 完全一致
  T2  越界（源开头不够 / 源结尾不够）抛 SourceRangeError，绝不 clamp
  T3  plan_slices 把越界位置记进 skipped 且带中文理由，其余位置一格不少
  T4  素材文件名带足身份信息（歌 / 源 / 位置 / 切片版本），换版本不撞名
  T5  真渲染：素材无声、时长准、画布归一，`.part` 不留残骸
  T6  重切（换 generation_version）不覆盖旧素材，旧的仍在库里可查

可以 `pytest tests/test_dance_material.py`，也可以 `python tests/test_dance_material.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import (                                     # noqa: E402
    delayed,
    make_project,
    make_song_file,
    make_source_video,
)
from vidscribe.dance import MATERIAL_GENERATION_VERSION          # noqa: E402
from vidscribe.dance import material_ingest as ingest            # noqa: E402
from vidscribe.dance import material_repository as repo          # noqa: E402
from vidscribe.dance import material_slice as slicer             # noqa: E402
from vidscribe.dance import media_backend as mb                  # noqa: E402
from vidscribe.dance.types import DanceAlignment                 # noqa: E402

CANVAS = mb.Canvas(width=144, height=256, fps=24.0)


def _alignment(offset: float, *, source_duration: float = 30.0) -> DanceAlignment:
    return DanceAlignment(offset=float(offset), confidence=0.9,
                          source_duration=float(source_duration), target_duration=30.0)


def test_mapping_matches_the_single_convention() -> None:
    """整个系统只有一个换算口径，这里把它和 DanceAlignment 对着钉死。"""
    for offset in (0.0, 1.5, -2.25, 3.9):

        spec = slicer.map_to_source(4.0, 6.0, offset, source_duration=100.0)
        assert abs(spec.source_start - (4.0 - offset)) < 1e-9, (offset, spec.source_start)
        assert abs(spec.source_end - (6.0 - offset)) < 1e-9
        assert abs(spec.duration - 2.0) < 1e-9
        # 和 DanceAlignment.source_time 必须逐位一致，否则两处口径会慢慢漂
        align = _alignment(offset, source_duration=100.0)
        assert spec.source_start == align.source_time(4.0)
        assert spec.source_end == align.source_time(6.0)
        # 反向也要能回来
        assert abs(align.target_time(spec.source_start) - 4.0) < 1e-9


def test_out_of_range_raises_instead_of_clamping() -> None:
    """源开头不够、源结尾不够，两种越界都必须抛，且理由是中文的。"""
    # offset=3 → 目标 0~2 秒映射到源 -3~-1 秒，源里根本没这段
    try:
        slicer.map_to_source(0.0, 2.0, 3.0, source_duration=30.0)
    except slicer.SourceRangeError as exc:
        assert "clamp" in str(exc) or "没有对应素材" in str(exc), str(exc)
    else:
        raise AssertionError("开头越界没抛 SourceRangeError（很可能被 clamp 了）")

    # 源只有 10 秒，目标 20~22 秒映射到源 20~22 秒，超出尾部
    try:
        slicer.map_to_source(20.0, 22.0, 0.0, source_duration=10.0)
    except slicer.SourceRangeError as exc:
        assert "10" in str(exc) or "越" in str(exc) or "不够" in str(exc), str(exc)
    else:
        raise AssertionError("尾部越界没抛 SourceRangeError")

    # 刚好贴边不算越界（浮点误差不能把合法的一格判死）
    spec = slicer.map_to_source(8.0, 10.0, 0.0, source_duration=10.0)
    assert abs(spec.source_end - 10.0) < 1e-9


def test_plan_records_skips_with_reasons() -> None:
    """offset=3 的源必然填不满前两个位置 —— 这两格要进 skipped 且写明原因。"""
    plan = slicer.plan_slices(_alignment(3.0, source_duration=12.0),
                              song_duration=20.0, source_duration=12.0,
                              slice_duration=2.0, source_video_id=7, target_song_id=3)
    covered = {spec.segment_index for spec in plan.specs}
    skipped = {index for index, _ in plan.skipped}
    assert covered | skipped == set(range(10)), "10 个位置必须每个都有交代"
    assert not (covered & skipped), "同一个位置不能既切了又跳了"
    assert 0 in skipped and 1 in skipped, f"offset=3 该跳过位置 0/1，实际跳了 {sorted(skipped)}"
    for index, reason in plan.skipped:
        assert reason.strip(), f"位置 {index} 被跳过却没写理由"
        assert any("\u4e00" <= ch <= "\u9fff" for ch in reason), f"理由不是中文：{reason}"
    # 每一格都严格满足口径
    for spec in plan.specs:
        assert abs(spec.source_start - (spec.target_start - 3.0)) < 1e-9
        assert spec.source_start >= -1e-6 and spec.source_end <= 12.0 + 1e-6
    assert plan.source_video_id == 7 and plan.target_song_id == 3
    assert plan.generation_version == MATERIAL_GENERATION_VERSION


def test_coverage_reports_the_truth() -> None:
    """coverage 报的是"这个源能铺满目标歌的多大比例"，不许四舍五入成好看的数字。"""
    plan = slicer.plan_slices(_alignment(0.0, source_duration=6.0),
                              song_duration=20.0, source_duration=6.0,
                              slice_duration=2.0, source_video_id=1, target_song_id=1)
    # 6 秒源只能填 0~2 / 2~4 / 4~6 三格 → 6/20
    assert len(plan.specs) == 3, [s.segment_index for s in plan.specs]
    assert len(plan.skipped) == 7, plan.skipped
    assert abs(slicer.coverage(plan, 20.0) - 0.3) < 1e-6, slicer.coverage(plan, 20.0)
    # 歌长为 0 是脏数据，不许除零崩掉
    assert slicer.coverage(plan, 0.0) == 0.0



def test_filename_carries_identity() -> None:
    """文件名要能一眼看出身份，且换切片版本不撞名（重切不覆盖旧素材）。"""
    spec = slicer.map_to_source(4.0, 6.0, 0.0, source_duration=30.0, segment_index=2)
    one = slicer.material_filename("dancer0", 5, spec, "slice-v1")
    two = slicer.material_filename("dancer0", 5, spec, "slice-v2")
    assert one != two, "换了切片版本还同名 → 重切会覆盖旧素材"
    for name in (one, two):
        assert name.endswith(".mp4"), name
        assert "5" in name and "2" in name, name
        assert "/" not in name and "\\" not in name, name


def test_render_produces_silent_normalized_material(work: Path) -> None:
    """真渲染一条素材：无声、时长准、尺寸归一到画布，`.part` 不留残骸。"""
    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    try:
        _, pcm = make_song_file(cfg, "t.wav", bpm=120.0, duration=8.0)
        source = make_source_video(cfg, "src.mp4", pcm, fps=30.0)
        spec = slicer.map_to_source(2.0, 4.0, 0.0, source_duration=8.0, segment_index=1)
        target = Path(cfg.dance_path("material_dir")) / "one.mp4"
        slicer.render_material(source, spec, target, canvas=CANVAS)

        assert target.is_file()
        meta = mb.resolve("auto").probe(target)
        assert meta.audio_streams == 0, "素材必须无声（目标歌才是唯一音轨）"
        assert meta.video_streams == 1
        assert (meta.width, meta.height) == (CANVAS.width, CANVAS.height), meta
        assert abs(meta.duration - 2.0) < 0.15, meta.duration
        assert not list(target.parent.glob("*" + slicer.PART_SUFFIX)), "留下了 .part 残骸"
    finally:
        db.close()


def test_regeneration_keeps_the_old_material(work: Path) -> None:
    """重切素材是**新增一个版本**，绝不物理删除旧素材（素材是长期资产）。"""
    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    try:
        song_path, pcm = make_song_file(cfg, "t.wav", bpm=120.0, duration=8.0)
        source = make_source_video(cfg, "src.mp4", pcm, fps=24.0)
        song = ingest.register_song(db, song_path)
        outcome = ingest.align_batch(db, song, [source], workers=1)[0]
        assert outcome.ok, outcome.error

        first = ingest.slice_and_register(db, song, outcome, slice_duration=2.0,
                                         material_dir=cfg.dance_path("material_dir"),
                                         canvas=CANVAS)
        assert first.ok and first.material_ids, first.error
        before = repo.materials_at(db, song.song_id, 0)
        assert len(before) == 1, before

        # 换切片版本重切：旧的必须还在（状态可以改成 regenerated，但行必须留着）
        repo.mark_regenerated(db, song.song_id, outcome.video_id,
                              keep_version="slice-v-next")

        old = repo.get_material(db, before[0].id)
        assert old is not None, "旧素材被物理删了 —— 素材是长期资产，只能改状态"
        assert old.status in ("ready", "regenerated"), old.status
        total = db.connect().execute(
            "SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ?",
            (song.song_id,)).fetchone()[0]
        assert int(total) == len(first.material_ids), "重切标记不该删行"
    finally:
        db.close()


def test_resume_only_redoes_what_is_missing(work: Path) -> None:
    """断点续跑：重跑一遍不重编已有素材，只补缺的那一条（技术指导第二十四节）。

    这是长期资产系统的硬要求 —— 100 个源视频跑到第 73 个崩了，重启之后
    只该补剩下的 27 个，而不是把前面 73 个重切一遍。
    靠三件事实现：对齐缓存命中、`render_plan(skip_existing=True)`、
    `upsert_material` 按 (歌, 源, 位置, 切片版本) 幂等。
    """
    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    try:
        song_path, pcm = make_song_file(cfg, "t.wav", bpm=120.0, duration=10.0)
        source = make_source_video(cfg, "src.mp4", pcm, fps=24.0)
        song = ingest.register_song(db, song_path)

        first_align = ingest.align_batch(db, song, [source], workers=1)[0]
        assert first_align.ok and not first_align.cached, "第一次不该命中缓存"
        first = ingest.slice_and_register(
            db, song, first_align, slice_duration=2.0,
            material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
        assert first.ok, first.error
        before = {m.id: Path(m.file_path).stat().st_mtime_ns
                  for m in repo.get_materials(db, first.material_ids).values()}
        assert len(before) == 5, before

        # 第二次：对齐命中缓存，素材文件全部复用，一条都不重编
        again_align = ingest.align_batch(db, song, [source], workers=1)[0]
        assert again_align.cached, "对齐没命中缓存 —— 会白算一遍"
        second = ingest.slice_and_register(
            db, song, again_align, slice_duration=2.0,
            material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
        assert second.ok, second.error
        assert sorted(second.material_ids) == sorted(first.material_ids), \
            "重跑产生了新的素材行 —— 幂等性坏了"
        after = {m.id: Path(m.file_path).stat().st_mtime_ns
                 for m in repo.get_materials(db, second.material_ids).values()}
        assert after == before, "素材文件被重编了一遍（skip_existing 没起作用）"
        total = db.connect().execute(
            "SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ?",
            (song.song_id,)).fetchone()[0]
        assert int(total) == 5, f"重跑之后库里变成 {total} 条"

        # 抽掉一条素材文件：只该补这一条，其余文件不动
        victim_id = sorted(before)[2]
        victim = Path(repo.get_material(db, victim_id).file_path)
        victim.unlink()
        third = ingest.slice_and_register(
            db, song, again_align, slice_duration=2.0,
            material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
        assert third.ok, third.error
        rebuilt = {m.id: Path(m.file_path).stat().st_mtime_ns
                   for m in repo.get_materials(db, third.material_ids).values()}
        assert victim.is_file(), "缺的那条没补回来"
        assert rebuilt[victim_id] != before[victim_id], "缺的那条没重渲"
        for material_id, stamp in before.items():
            if material_id != victim_id:
                assert rebuilt[material_id] == stamp, \
                    f"素材 #{material_id} 被无谓地重编了"
        print("  断点续跑：5 条素材复用 4 条、补回 1 条")
    finally:
        db.close()


TESTS = (
    test_mapping_matches_the_single_convention,
    test_out_of_range_raises_instead_of_clamping,
    test_plan_records_skips_with_reasons,
    test_coverage_reports_the_truth,
    test_filename_carries_identity,
    test_render_produces_silent_normalized_material,
    test_regeneration_keeps_the_old_material,
    test_resume_only_redoes_what_is_missing,
)



def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancemat_"))
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
