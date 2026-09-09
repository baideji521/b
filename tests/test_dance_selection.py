"""筛选 / 评分 / 组合搜索 / 推荐（技术指导第十~十三、十七节 + 第二十二节第 10~16 条）。

这一整条链一格视频都不碰，所以素材全部直接塞库（`fake_library`），几毫秒就跑完。
盯的是四件容易出错的事：

  T1  筛选全部走占位符（人物名和搜索词来自用户输入），一期方案 A~G 都要能用
  T2  静态分只看"素材 + 历史 + 音乐位置"，动态分才看邻居 —— 混了 beam search 就没法做
  T3  硬约束是**拒绝**不是扣分：连续同人、素材重复、人物/来源占比、相邻对历史
  T4  推荐可复现：同一个种子出同一份排名；换种子才可能变

可以 `pytest tests/test_dance_selection.py`，也可以 `python tests/test_dance_selection.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import (                                           # noqa: E402
    fake_library,
    fake_material,
    fake_song,
    fake_video,
    make_project,
)

from vidscribe.dance import combination_search as search               # noqa: E402
from vidscribe.dance import material_score as score_mod                # noqa: E402
from vidscribe.dance import material_selection as selection            # noqa: E402
from vidscribe.dance import recommendation, statistics                 # noqa: E402
from vidscribe.dance import strategy as strategy_mod                   # noqa: E402
from vidscribe.dance.types import FilterSpec                           # noqa: E402


def test_query_never_interpolates_user_input() -> None:
    """人物名和搜索词是用户输入，必须走占位符 —— 这里直接拿注入串当名字试。"""
    evil = "'; DROP TABLE dance_materials; --"
    sql, params = selection.build_query(
        FilterSpec(target_song_id=1, persons=(evil,), search=evil))
    assert evil not in sql, "用户输入被拼进 SQL 了"
    assert evil in params or f"%{evil}%" in params, params
    assert sql.count("?") == len(params), (sql.count("?"), len(params))


def test_long_unused_includes_never_used() -> None:
    """「超过 N 天没用」必须包含"从来没用过" —— 漏掉这一半是反直觉的经典坑。"""
    sql, _ = selection.build_query(FilterSpec(target_song_id=1, long_unused_days=7.0))
    assert "last_used_at IS NULL" in sql, sql


def test_presets_cover_the_seven_schemes() -> None:
    """一期列的 A~G 七套筛选方案要一个不少，且各自真的改了对应字段。"""
    assert len(selection.PRESETS) == 7, list(selection.PRESETS)
    assert selection.preset_spec("never_output").never_output is True
    assert selection.preset_spec("never_used").never_used is True
    assert selection.preset_spec("low_use").max_use_count == 2
    assert selection.preset_spec("long_unused").long_unused_days == 7.0
    assert selection.preset_spec("high_confidence").min_confidence == 0.90
    assert selection.preset_spec("exclude_recent").recently_used_days == 3.0
    # 指定人物的那套要能把人名传进去
    spec = selection.preset_spec("by_person", FilterSpec(persons=("小A",)))
    assert spec.persons == ("小A",), spec.persons


def test_filters_actually_filter(work: Path) -> None:
    """筛选条件要真的作用在库上，不是只改了 SQL 文本。"""
    cfg, db = make_project(work)
    try:
        song_id, videos, materials = fake_library(db, positions=3, people=("小A", "小B"))
        # 给小A 位置0 那条记上使用与出片
        target = materials[("小A", 0)]
        db.connect().execute(
            "UPDATE dance_materials SET use_count=5, output_count=2, "
            "last_used_at=datetime('now'), last_output_at=datetime('now') WHERE id=?",
            (target,))

        all_at_0 = selection.find_materials(db, FilterSpec(target_song_id=song_id,
                                                          segment_index=0))
        assert len(all_at_0) == 2, [m.id for m in all_at_0]

        never = selection.find_materials(db, FilterSpec(target_song_id=song_id,
                                                       segment_index=0, never_used=True))
        assert [m.id for m in never] == [materials[("小B", 0)]]

        by_person = selection.find_materials(db, FilterSpec(target_song_id=song_id,
                                                           persons=("小A",)))
        assert len(by_person) == 3 and all(m.person == "小A" for m in by_person)

        low_use = selection.find_materials(db, FilterSpec(target_song_id=song_id,
                                                         max_use_count=2))
        assert target not in [m.id for m in low_use]

        by_source = selection.find_materials(
            db, FilterSpec(target_song_id=song_id, source_video_ids=(videos["小B"],)))
        assert len(by_source) == 3 and all(m.source_video_id == videos["小B"]
                                           for m in by_source)
        # 停用的素材默认不出现（但行还在库里）
        db.connect().execute("UPDATE dance_materials SET status='disabled' WHERE id=?",
                            (materials[("小B", 1)],))
        assert len(selection.find_materials(db, FilterSpec(target_song_id=song_id,
                                                          segment_index=1))) == 1
        assert len(selection.find_materials(
            db, FilterSpec(target_song_id=song_id, segment_index=1,
                           statuses=("ready", "disabled")))) == 2
    finally:
        db.close()


def test_static_score_ignores_neighbours(work: Path) -> None:
    """静态分只能看素材自己 + 历史 + 音乐位置。看了邻居，beam search 就没法用了。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=3, people=("小A", "小B"))
        ctx = selection.score_context(db, song_id)
        pool = selection.build_pool(db, song_id, 0, context=ctx)
        assert len(pool.scored) == 2
        # 同一条素材，反复算必须完全一样（没有隐藏状态、没有随机）
        first = selection.build_pool(db, song_id, 0, context=ctx)
        assert [s.final_score for s in first.scored] == [s.final_score for s in pool.scored]
        # 排序是按分降序
        assert list(pool.scored) == sorted(pool.scored,
                                          key=lambda s: (-s.final_score, s.material_id))
        # breakdown 要能解释分数，且各项都记下来了
        top = pool.scored[0]
        assert top.breakdown, "没有评分明细，界面上没法解释为什么选它"
        assert "final_score" in top.breakdown
        assert score_mod.explain(top), "explain 该给出中文说明"

        # 出过片的素材在轮换策略下必须排到没出过片的后面
        db.connect().execute(
            "UPDATE dance_materials SET output_count=9, use_count=9 WHERE id=?",
            (materials[("小A", 0)],))
        ctx2 = selection.score_context(db, song_id)
        weights = recommendation.weights_for("rule_based",
                                            strategy_mod.preset("rotation").weights)
        rotated = selection.build_pool(db, song_id, 0, context=ctx2, weights=weights)
        assert rotated.scored[0].material_id == materials[("小B", 0)], \
            "轮换策略没把没出过片的素材排前面"
    finally:
        db.close()


def test_hard_constraints_reject_instead_of_penalize(work: Path) -> None:
    """硬约束必须**拒绝**：连续同人、同素材重复、人物占比。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=4, people=("小A", "小B"))
        ctx = selection.score_context(db, song_id)
        plan = strategy_mod.default_strategy()
        lookup = {m.id: m for m in selection.find_materials(
            db, FilterSpec(target_song_id=song_id))}

        a0 = lookup[materials[("小A", 0)]]
        a1 = lookup[materials[("小A", 1)]]
        b1 = lookup[materials[("小B", 1)]]
        # 连着同一个人 → 拒
        assert search.violates(a1, [a0], ctx, plan.constraints, 4), "连续同人没被拒"
        # 换个人 → 放行
        assert not search.violates(b1, [a0], ctx, plan.constraints, 4)
        # 同一条素材再来一次 → 拒
        assert search.violates(a0, [a0], ctx, plan.constraints, 4), "素材重复没被拒"
        # 人物占比：4 格里小A 已占 2 格（>45%），第 3 格还想要小A → 拒
        loose = dict(plan.constraints, max_consecutive_same_person=99)
        a2 = lookup[materials[("小A", 2)]]
        assert search.violates(a2, [a0, a1], ctx, loose, 4), "人物占比没被拒"
    finally:
        db.close()


def test_search_is_deterministic_and_bounded(work: Path) -> None:
    """同一个种子必须搜出同一版；预算必须封顶（不许全排列）。"""
    cfg, db = make_project(work)
    try:
        song_id, _, _ = fake_library(db, positions=6, people=("小A", "小B", "小C"))
        ctx = selection.score_context(db, song_id)
        plan = strategy_mod.default_strategy()
        pools = selection.build_pools(db, song_id, list(range(6)))
        assert all(pool.scored for pool in pools)

        one = search.search(pools, ctx, plan, seed=1234)
        two = search.search(pools, ctx, plan, seed=1234)
        assert [p.material_id for p in one.picks] == [p.material_id for p in two.picks], \
            "同一个种子搜出了不同的结果 —— 不可复现"
        assert one.nodes <= int(plan.search["max_nodes"]), (one.nodes, plan.search)
        assert len(one.picks) == 6, len(one.picks)
        # 硬约束在成品里也必须成立：不许连着同一个人
        people = [one.materials[i].person for i in range(len(one.picks))] \
            if isinstance(one.materials, list) else \
            [{m.id: m for m in one.materials}[p.material_id].person for p in one.picks]
        assert all(a != b for a, b in zip(people, people[1:])), people
        assert len(set(p.material_id for p in one.picks)) == len(one.picks), "素材重复了"
    finally:
        db.close()


def test_versions_differ_from_each_other(work: Path) -> None:
    """多版本混剪的意义就是"不一样"：第 2/3 版必须和第 1 版有明显差异。"""
    cfg, db = make_project(work)
    try:
        song_id, _, _ = fake_library(db, positions=6, people=("小A", "小B", "小C"))
        ctx = selection.score_context(db, song_id)
        plan = strategy_mod.default_strategy()
        pools = selection.build_pools(db, song_id, list(range(6)))
        found = search.search_versions(pools, ctx, plan, count=3, seed=7)
        assert len(found) >= 2, f"只搜出 {len(found)} 版"
        signatures = [score_mod.combination_signature(p.material_id for p in f.picks)
                      for f in found]
        assert len(set(signatures)) == len(signatures), "多个版本的组合签名一样 —— 等于没换"
        # 同一个种子重跑，整批版本都要一样
        again = search.search_versions(pools, ctx, plan, count=3, seed=7)
        assert [score_mod.combination_signature(p.material_id for p in f.picks)
                for f in again] == signatures
    finally:
        db.close()


def test_recommendation_is_reproducible(work: Path) -> None:
    """推荐必须可复现：种子落库，重跑同一个 run 得到同一份排名（技术指导第十七节）。"""
    cfg, db = make_project(work)
    try:
        song_id, _, _ = fake_library(db, positions=5, people=("小A", "小B", "小C"))
        strategy_mod.ensure_presets(db)
        plan = strategy_mod.resolve(db, 0)
        run = recommendation.recommend(db, song_id, list(range(5)), strategy=plan, seed=99)
        assert run.id > 0, "推荐记录没落库"
        assert run.random_seed == 99, run.random_seed
        assert run.items, "一条推荐都没有"
        assert run.algorithm_version, "没记算法版本 —— 换算法后没法区分历史"

        original, replay, same = recommendation.reproduce(db, run.id)
        assert same, "库没动过，同一个 run 却复现出了不同排名"
        one = [(i.segment_index, i.material_id, i.rank) for i in original.items]
        two = [(i.segment_index, i.material_id, i.rank) for i in replay.items]
        assert sorted(one) == sorted(two), (one[:5], two[:5])


        # 每个位置的排名要连续、从 1 开始
        for index in range(5):
            ranks = [i.rank for i in recommendation.items_for(run, index)]
            assert ranks == list(range(1, len(ranks) + 1)), (index, ranks)

        # 探索型策略带抖动，换种子应该能换出不同的头名（至少不崩）
        explore = strategy_mod.preset("exploration")
        first = recommendation.recommend(db, song_id, [0], strategy=explore, seed=1)
        second = recommendation.recommend(db, song_id, [0], strategy=explore, seed=2)
        assert first.random_seed != second.random_seed
        assert first.items and second.items
    finally:
        db.close()


def test_repeat_rates_are_honest(work: Path) -> None:
    """七个重复率的口径不能糊：版内重复算版内，历史重复算历史。"""
    cfg, db = make_project(work)
    try:
        song_id, _, materials = fake_library(db, positions=4, people=("小A", "小B"))
        from vidscribe.dance.types import DanceMontageClip

        def clip(order: int, pos: int, mid: int, person: str, video: int):
            return DanceMontageClip(order_index=order, segment_index=pos, material_id=mid,
                                   target_start=order * 2.0, target_end=order * 2.0 + 2.0,
                                   source_start=0.0, source_end=2.0,
                                   person=person, source_video_id=video)

        # 全不重复的一版
        clean = [clip(0, 0, 1, "小A", 1), clip(1, 1, 2, "小B", 2),
                 clip(2, 2, 3, "小A", 1), clip(3, 3, 4, "小B", 2)]
        stats = statistics.repeat_stats(clean)
        assert stats.material_repeat == 0.0, stats.material_repeat
        assert stats.detail["unique_materials"] == 4, stats.detail

        # 同一条素材用了两次 → 版内素材重复率不为 0
        dirty = [clip(0, 0, 1, "小A", 1), clip(1, 1, 1, "小A", 1),
                 clip(2, 2, 3, "小B", 2), clip(3, 3, 4, "小B", 2)]
        bad = statistics.repeat_stats(dirty)
        assert bad.material_repeat == 0.25, bad.material_repeat
        assert bad.detail["unique_materials"] == 3, bad.detail
        # 人物构成没变（都是两个人四格），人物重复率就不该变 —— 各项互不串味
        assert bad.person_repeat == stats.person_repeat, \
            (bad.person_repeat, stats.person_repeat)
        assert bad.overall_repeat > stats.overall_repeat, \
            (bad.overall_repeat, stats.overall_repeat)
        assert 0.0 <= bad.overall_repeat <= 1.0, bad.overall_repeat

        # 全是同一个人 → 人物重复率必须顶到最高
        solo = [clip(i, i, i + 10, "小A", 1) for i in range(4)]
        alone = statistics.repeat_stats(solo)
        assert alone.person_repeat > stats.person_repeat, alone.person_repeat
        assert alone.source_repeat > stats.source_repeat, alone.source_repeat
        assert alone.material_repeat == 0.0, "四条不同素材却报素材重复"
        assert statistics.describe(bad), "describe 要给出中文行"

    finally:
        db.close()


def test_candidate_pool_is_not_truncated_by_insertion_order(work: Path) -> None:
    """候选池的**预取**上限不许按入库顺序砍 —— 否则新切的素材在打分前就被扔掉。

    这是真踩过的坑：`sort="score_desc"` 时 SQL 排不了分（分要读历史），
    如果那一档的 ORDER BY 写成 `m.id ASC`，`LIMIT N` 就等于"库里最早的 N 条"，
    某个位置素材一多，从未使用的新素材永远进不了候选池。
    """
    cfg, db = make_project(work)
    try:
        song_id = fake_song(db, title="很多素材")
        # 12 条素材全在位置 0：前 6 条（id 小）已经用烂了，后 6 条（id 大）全新
        ids = []
        for index in range(12):
            video = fake_video(db, f"src{index:02d}")
            ids.append(fake_material(db, song_id, video, 0, person=f"人{index:02d}"))
        old, fresh = ids[:6], ids[6:]
        marks = ",".join("?" for _ in old)
        db.connect().execute(
            f"UPDATE dance_materials SET use_count=20, output_count=9, "
            f"last_used_at=datetime('now') WHERE id IN ({marks})", tuple(old))

        pool = selection.build_pool(
            db, song_id, 0, spec=FilterSpec(target_song_id=song_id, limit=6))
        got = [s.material_id for s in pool.scored]
        assert len(got) == 6, got
        assert set(got) == set(fresh), \
            f"预取只留下了 {got}，全新素材 {fresh} 被按 id 砍掉了"
        # SQL 的预取顺序必须体现"最该被考虑"，而不是 id
        sql, _params = selection.build_query(FilterSpec(target_song_id=song_id))
        assert "m.id ASC" != sql.split("ORDER BY")[1].split("LIMIT")[0].strip(), sql
        assert "output_count ASC" in sql and "use_count ASC" in sql, sql
        # 上限放开就全都在
        wide = selection.build_pool(
            db, song_id, 0, spec=FilterSpec(target_song_id=song_id, limit=500))
        assert len(wide.scored) == 12, len(wide.scored)
        assert selection.DEFAULT_POOL_LIMIT >= 500, selection.DEFAULT_POOL_LIMIT
    finally:
        db.close()


def test_history_heavy_material_ranks_below_fresh_one(work: Path) -> None:
    """用烂了的素材必须排在全新素材后面（默认权重下就该如此）。"""
    cfg, db = make_project(work)
    try:
        song_id = fake_song(db)
        one = fake_material(db, song_id, fake_video(db, "a"), 0, person="小A")
        two = fake_material(db, song_id, fake_video(db, "b"), 0, person="小B")
        db.connect().execute(
            "UPDATE dance_materials SET use_count=30, output_count=12, "
            "last_used_at=datetime('now'), last_output_at=datetime('now') WHERE id=?", (one,))
        pool = selection.build_pool(db, song_id, 0)
        ranked = [s.material_id for s in pool.scored]
        assert ranked[0] == two, f"全新素材没排在前面：{ranked}"
        scores = {s.material_id: s.final_score for s in pool.scored}
        assert scores[two] > scores[one], scores
        # 明细里要能看出扣分来自哪儿，而不是一个黑箱数字
        heavy = next(s for s in pool.scored if s.material_id == one)
        assert any(v < 0 for v in heavy.breakdown.values()), heavy.breakdown
    finally:
        db.close()


def test_soft_penalties_punish_same_person_and_same_source(work: Path) -> None:
    """同人 / 同源连着出现要**扣分**（软惩罚），这和硬约束是两码事。"""
    cfg, db = make_project(work)
    try:
        song_id, videos, materials = fake_library(db, positions=3, people=("小A", "小B"))
        ctx = selection.score_context(db, song_id)
        lookup = {m.id: m for m in selection.find_materials(
            db, FilterSpec(target_song_id=song_id))}
        a0 = lookup[materials[("小A", 0)]]
        a1 = lookup[materials[("小A", 1)]]
        b1 = lookup[materials[("小B", 1)]]

        same_delta, same_parts = score_mod.dynamic_delta(a1, ctx, [a0])
        diff_delta, diff_parts = score_mod.dynamic_delta(b1, ctx, [a0])
        assert same_parts["same_person_penalty"] < 0, same_parts
        assert diff_parts["same_person_penalty"] == 0.0, diff_parts
        assert same_delta < diff_delta, (same_delta, diff_delta)
        # 同源（这里同人即同源）也要体现在 source_diversity 上
        assert same_parts["source_diversity"] < diff_parts["source_diversity"], \
            (same_parts["source_diversity"], diff_parts["source_diversity"])
        # 第一格没有邻居，多样性项拿满分，不该无端扣分
        first_delta, first_parts = score_mod.dynamic_delta(a0, ctx, [])
        assert first_parts["same_person_penalty"] == 0.0, first_parts
        assert first_delta >= 0.0, first_delta
    finally:
        db.close()


TESTS = (
    test_query_never_interpolates_user_input,
    test_long_unused_includes_never_used,
    test_presets_cover_the_seven_schemes,
    test_filters_actually_filter,
    test_static_score_ignores_neighbours,
    test_candidate_pool_is_not_truncated_by_insertion_order,
    test_history_heavy_material_ranks_below_fresh_one,
    test_soft_penalties_punish_same_person_and_same_source,
    test_hard_constraints_reject_instead_of_penalize,
    test_search_is_deterministic_and_bounded,
    test_versions_differ_from_each_other,
    test_recommendation_is_reproducible,
    test_repeat_rates_are_honest,
)



def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancesel_"))
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
