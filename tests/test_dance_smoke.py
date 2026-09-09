"""Smoke test：技术指导第二十三节那条最小闭环，端到端真跑一遍。

    10 秒目标歌 + 3 个短视频 + slice=2s
    → 5 个固定音乐位置
    → 每个源都对齐、都出素材
    → 候选池 → 推荐 → 组合搜索 → 编辑计划 → 渲染 → mux
    → 一个**能播的** MP4

验收项（技术指导逐条列的那些，一条都不许跳）：
    duration / video stream / audio stream / audio duration
    target song 是**唯一**音轨
    timeline 数量 / material usage / montage history / statistics

这个测试会真的编码视频，所以比其他测试慢（几十秒），但它是唯一能证明
"整条链真的通了"的东西。全部在临时目录里跑，**绝不碰项目真实数据库**。

可以 `pytest tests/test_dance_smoke.py`，也可以 `python tests/test_dance_smoke.py`。
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
from vidscribe.dance import history, material_ingest as ingest    # noqa: E402
from vidscribe.dance import material_repository as repo           # noqa: E402
from vidscribe.dance import media_backend as mb                   # noqa: E402
from vidscribe.dance import montage_render, montage_timeline      # noqa: E402
from vidscribe.dance import music_structure, statistics           # noqa: E402
from vidscribe.dance import strategy as strategy_mod              # noqa: E402

#: 小画布 + 低帧率：smoke test 要的是"链路通不通"，不是画质
CANVAS = mb.Canvas(width=144, height=256, fps=24.0)
SLICE = 2.0
SONG_SECONDS = 10.0


def _setup(work: Path):
    """搭最小场景：一首 10 秒目标歌 + 3 个源视频（人为错开 0 / 1.0 / 2.0 秒）。

    错开是关键：三个源的 offset 各不相同，所以它们能覆盖的音乐位置也不同 ——
    这样才能验证"越界的位置被正确跳过"而不是被 clamp。
    """
    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_path, pcm = make_song_file(cfg, "target.wav", bpm=120.0, duration=SONG_SECONDS)
    sources = []
    for index, offset in enumerate((0.0, 1.0, 2.0)):
        clip = delayed(pcm, offset) if offset else pcm
        sources.append(make_source_video(cfg, f"dancer{index}.mp4", clip,
                                        tint=(210 - 50 * index, 70 + 50 * index, 120),
                                        fps=24.0 + index))
    return cfg, db, song_path, sources


def test_smoke_end_to_end(work: Path) -> None:
    cfg, db, song_path, sources = _setup(work)
    try:
        # --- 1. 注册并分析目标歌 -----------------------------------------
        song = ingest.register_song(db, song_path)
        assert song.song_id > 0
        assert abs(song.duration - SONG_SECONDS) < 0.3, song.duration
        positions = music_structure.target_positions(song.duration, SLICE)
        assert len(positions) == 5, f"10 秒 / 2 秒该有 5 个位置，实际 {len(positions)}"

        # --- 2. 批量对齐 --------------------------------------------------
        outcomes = ingest.align_batch(db, song, sources, workers=3)
        assert len(outcomes) == 3
        for outcome, want in zip(outcomes, (0.0, 1.0, 2.0)):
            assert outcome.ok, f"{outcome.source_path.name} 对齐失败：{outcome.error}"
            assert abs(outcome.alignment.offset - want) < 0.08, \
                f"{outcome.source_path.name} offset {outcome.alignment.offset} != {want}"

        # 缓存复用：再跑一遍必须全部命中，一条都不重算
        again = ingest.align_batch(db, song, sources, workers=3)
        assert all(o.cached for o in again), "第二次对齐该全部命中缓存"

        # --- 3. 切片入库 --------------------------------------------------
        total = 0
        for outcome in outcomes:
            result = ingest.slice_and_register(
                db, song, outcome, slice_duration=SLICE,
                material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
            assert result.ok, f"{outcome.source_path.name} 切片失败：{result.error}"
            total += len(result.material_ids)
            # offset > 0 的源，开头那几个位置必然映射到负时间 → 必须被跳过而不是 clamp
            expect_skips = int(outcome.alignment.offset // SLICE + (
                1 if outcome.alignment.offset % SLICE > 1e-6 else 0))
            assert result.skipped >= expect_skips, \
                f"offset {outcome.alignment.offset} 该跳过至少 {expect_skips} 个位置"
        assert total >= 10, f"三个源该出十几条素材，实际 {total}"

        # 素材文件真的在盘上、真的是无声的
        for material in repo.materials_at(db, song.song_id, 4):
            assert Path(material.file_path).is_file(), material.file_path
            meta = mb.resolve("auto").probe(material.file_path)
            assert meta.audio_streams == 0, "素材必须无声（目标歌才是唯一音轨）"
            assert abs(meta.duration - SLICE) < 0.15, meta.duration
        print(f"  素材 {total} 条，位置覆盖 "
              f"{[p['segment_index'] for p in statistics.position_coverage(db, song.song_id)]}")

        # --- 4. 混剪：候选池 → 推荐 → 组合搜索 → 计划 → 渲染 -----------------
        strategy_mod.ensure_presets(db)
        lines: list[str] = []
        made = montage_render.remix(
            db, song.song_id, out_dir=cfg.dance_path("output_dir"),
            slice_duration=SLICE, versions=2, seed=20260909,
            canvas=CANVAS, name="smoke", on_log=lines.append)
        assert made, "一版都没出来"
        for line in lines:
            print(f"  {line}")

        # --- 5. 成品硬验收 ------------------------------------------------
        for version_id, result in made:
            assert result.ok, f"版本 #{version_id} 渲染失败：{result.error}"
            out = Path(result.output)
            assert out.is_file(), out
            assert out.stat().st_size > 1024, f"成品只有 {out.stat().st_size} 字节"
            assert result.video_streams == 1, result.video_streams
            assert result.audio_streams == 1, \
                f"音轨必须正好 1 条（目标歌），实际 {result.audio_streams}"
            want = result.clips * SLICE
            assert abs(result.duration - want) < 0.35, \
                f"成片 {result.duration:.3f}s，按 {result.clips} 格该是 {want:.1f}s"
            # 音轨长度跟画面长度对齐（mux 按画面时长截目标歌）
            assert abs(result.audio_duration - result.duration) < 0.5, \
                f"音轨 {result.audio_duration:.3f}s 和画面 {result.duration:.3f}s 差太多"
            # 中间的无声文件必须清掉，不能留在成品目录里
            assert not out.with_name(out.stem + montage_render.MUTE_SUFFIX).exists(), \
                "无声中间文件没删干净"

        # --- 6. 记账：timeline / usage / history / statistics ---------------
        versions = repo.versions_for_song(db, song.song_id)
        assert len(versions) == len(made), f"库里 {len(versions)} 版，实际渲了 {len(made)} 版"
        indexes = [int(v["version_index"]) for v in versions]
        assert len(set(indexes)) == len(indexes), f"版本号重复：{indexes}"

        for version_id, result in made:
            rows = repo.version_materials(db, version_id)
            assert len(rows) == result.clips, \
                f"版本 #{version_id} 库里 {len(rows)} 格、成片 {result.clips} 格"
            positions_used = [int(r["segment_index"]) for r in rows]
            assert positions_used == sorted(positions_used), "格子必须按音乐位置升序"
            assert len(set(positions_used)) == len(positions_used), "同一个位置不能填两格"
            assert str(repo.get_version(db, version_id)["render_status"]) == "rendered"

        events = history.event_summary(db, song.song_id)
        assert events.get("montage", 0) == sum(r.clips for _, r in made), events
        assert events.get("render_success", 0) == sum(r.clips for _, r in made), events
        assert events.get("render_failed", 0) == 0, events
        assert events.get("candidate", 0) > 0, "候选事件一条都没记"

        # 计数必须能从事件流水重算出来，且重算结果不变（技术指导第九节）
        again_counts = history.recount_song(db, song.song_id)
        assert again_counts.get("changed", 0) == 0, \
            f"重算改了 {again_counts.get('changed')} 条 —— 说明计数和事件不一致"

        overview = statistics.song_overview(db, song.song_id)
        assert int(overview["materials"]["total"]) == total, overview["materials"]
        assert int(overview["versions"]["rendered"]) == len(made), overview["versions"]
        used = db.connect().execute(
            "SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ? AND use_count > 0",
            (song.song_id,)).fetchone()[0]
        outputs = db.connect().execute(
            "SELECT COUNT(*) FROM dance_materials WHERE target_song_id = ? AND output_count > 0",
            (song.song_id,)).fetchone()[0]
        assert int(used) > 0, "没有任何素材的 use_count 被推上去"
        assert int(outputs) > 0, "没有任何素材的 output_count 被推上去"



        # --- 7. 历史版本可以读回来重新渲染 ----------------------------------
        first_id = made[0][0]
        loaded = montage_timeline.load(db, first_id)
        assert loaded is not None
        assert len(loaded.clips) == made[0][1].clips
        assert loaded.signature == repo.get_version(db, first_id)["signature"]
        assert not montage_timeline.validate(loaded), montage_timeline.validate(loaded)
        print(f"  版本 #{first_id} 读回成功：{len(loaded.clips)} 格"
              f"｜综合重复率 {loaded.repeat.overall_repeat:.3f}")
    finally:
        db.close()


def test_smoke_manual_selection(work: Path) -> None:
    """关掉智能推荐、纯手动指定每个位置用哪条素材，也必须能出片。

    技术指导第二十节的硬要求。这条路不经过推荐和组合搜索，所以它同时也是
    "渲染层真的不依赖推荐层"的证明。
    """
    cfg, db, song_path, sources = _setup(work)
    try:
        song = ingest.register_song(db, song_path)
        outcome = ingest.align_batch(db, song, sources[:1], workers=1)[0]
        assert outcome.ok, outcome.error
        sliced = ingest.slice_and_register(
            db, song, outcome, slice_duration=SLICE,
            material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
        assert sliced.ok, sliced.error

        manual = {}
        for pos in range(5):
            found = repo.materials_at(db, song.song_id, pos)
            if found:
                manual[pos] = found[0].id
        assert len(manual) >= 3, f"手动只凑到 {len(manual)} 个位置"

        made = montage_render.remix(
            db, song.song_id, out_dir=cfg.dance_path("output_dir"),
            slice_duration=SLICE, canvas=CANVAS, name="manual",
            recommend_enabled=False, manual=manual)
        assert len(made) == 1
        version_id, result = made[0]
        assert result.ok, result.error
        assert result.clips == len(manual)
        assert result.audio_streams == 1, result.audio_streams
        rows = repo.version_materials(db, version_id)
        assert {int(r["segment_index"]): int(r["material_id"]) for r in rows} == manual
        # 手动选择不许伪造推荐分
        assert all(float(r["selection_score"] or 0.0) == 0.0 for r in rows)
        assert history.event_summary(db, song.song_id).get("render_success", 0) == len(manual)
        print(f"  纯手动 {len(manual)} 格 → {Path(result.output).name}")
    finally:
        db.close()


def test_smoke_render_failure_does_not_count_output(work: Path) -> None:
    """素材文件被删掉之后渲染必须失败，且**出片次数一格都不许涨**（技术指导第九节）。"""
    cfg, db, song_path, sources = _setup(work)
    try:
        song = ingest.register_song(db, song_path)
        outcome = ingest.align_batch(db, song, sources[:1], workers=1)[0]
        sliced = ingest.slice_and_register(
            db, song, outcome, slice_duration=SLICE,
            material_dir=cfg.dance_path("material_dir"), canvas=CANVAS)
        assert sliced.ok, sliced.error

        manual = {}
        for pos in range(5):
            found = repo.materials_at(db, song.song_id, pos)
            if found:
                manual[pos] = found[0].id
        context = montage_timeline.from_manual(
            db, song.song_id, manual, song_path=str(song_path), slice_duration=SLICE)
        version_id = montage_timeline.save(db, context)

        # 抽掉一条素材文件：渲染必须在编码之前就报"不在盘上"
        victim = Path(context.clips[0].file_path)
        victim.unlink()
        result = montage_render.render(db, context, out_dir=cfg.dance_path("output_dir"),
                                      version_id=version_id, canvas=CANVAS,
                                      song_path=str(song_path))
        assert not result.ok
        assert "不在盘上" in (result.error or ""), result.error
        assert str(repo.get_version(db, version_id)["render_status"]) == "failed"

        events = history.event_summary(db, song.song_id)
        assert events.get("render_failed", 0) > 0, events
        assert events.get("render_success", 0) == 0, events
        outputs = db.connect().execute(
            "SELECT COALESCE(SUM(output_count), 0) FROM dance_materials WHERE target_song_id = ?",
            (song.song_id,)).fetchone()[0]
        assert int(outputs) == 0, f"渲染失败却涨了 {outputs} 次出片"
        print("  渲染失败已记账，出片次数仍为 0")
    finally:
        db.close()


TESTS = (
    test_smoke_end_to_end,
    test_smoke_manual_selection,
    test_smoke_render_failure_does_not_count_output,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancesmoke_"))
        try:
            fn(work)
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


