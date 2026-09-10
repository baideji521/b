"""Dance 子系统的数据库迁移测试（技术指导第十八节 + 第二十二节第 20 条）。

盯的是两句话：
  1. **老库升上来之后，原有功能的行为必须一模一样** —— v11 只建新表，一个已有表都不碰。
  2. **新建库和升级库的表结构必须逐字相同** —— 历史上 v4 用 `ALTER TABLE ADD COLUMN`
     加不上 `REFERENCES`，导致两条路径外键不等价，这个坑不能再踩一次。

  T1  新建库直接到 v11，11 张 dance_* 表全在，integrity/foreign_key_check 干净
  T2  v10 老库升到 v11：dance_* 表建出来，且**原有表的 SQL 一个字符都没变**
  T3  两条路径（新建 / 升级）的 dance_* 建表语句逐字相同，索引集合也相同
  T4  升级是幂等的：重复 apply 不报错、版本不变
  T5  v11 里没有任何 DROP / ALTER 已有表的语句（静态检查，防止以后误加）
  T6  外键真的生效：删目标歌会级联删对齐与素材，删素材不会连累事件之外的东西
  T7  唯一约束真的生效：同一首歌同一个源同一个位置同一个切片版本只能有一条
  T8  db_admin.health_check 在 v11 库上通过（表/索引清单从 schema 现提取，不硬编码）

全部在临时目录里建库，**绝不碰项目真实数据库**。
可以 `pytest tests/test_dance_migration.py`，也可以 `python tests/test_dance_migration.py`。
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.db import admin as db_admin        # noqa: E402
from vidscribe.db import migrations, schema       # noqa: E402

#: v11 必须建出来的表，一张不少
DANCE_TABLE_NAMES = (
    "dance_target_songs",
    "dance_audio_alignments",
    "dance_materials",
    "dance_material_usage_events",
    "dance_montage_strategies",
    "dance_montages",
    "dance_montage_sources",
    "dance_montage_versions",

    "dance_montage_materials",
    "dance_recommendation_runs",
    "dance_recommendation_items",
    "dance_recommendation_feedback",
)


# ------------------------------------------------------------------ 夹具
def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def fresh_db(work: Path, name: str = "fresh.db") -> sqlite3.Connection:
    """全新库：`apply` 走 `_create_all`，一步到当前版本。"""
    conn = _open(work / name)
    migrations.apply(conn)
    return conn


def legacy_db(work: Path, name: str = "legacy.db") -> sqlite3.Connection:
    """造一个 v10 老库：建除 dance_* 之外的全部表，再把 user_version 钉成 10。"""
    conn = _open(work / name)
    later = (set(schema.DANCE_TABLES) | set(schema.DANCE_V12_TABLES)
             | set(schema.DANCE_V13_TABLES))
    for statement in schema.TABLES:
        if statement in later:
            continue                      # v11~v13 才有的舞蹈表，老库里当然没有
        conn.execute(statement)
    conn.execute("PRAGMA user_version=10")
    return conn



def _objects(conn: sqlite3.Connection, kind: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT name, COALESCE(sql, '') AS sql FROM sqlite_master WHERE type = ?", (kind,))
    return {row["name"]: row["sql"] for row in rows}


def _non_dance(conn: sqlite3.Connection, kind: str) -> dict[str, str]:
    """只留和舞蹈无关的对象。

    过滤条件必须按"名字里带 dance"来判，不能只挡 `idx_dance` 前缀：
    UNIQUE 约束会让 SQLite 自动生成 `sqlite_autoindex_dance_materials_1` 这类索引，
    漏掉它们会让"v11 没动已有索引"这条断言假失败。
    """
    return {name: sql for name, sql in _objects(conn, kind).items() if "dance" not in name}



# ------------------------------------------------------------------ T1
def test_fresh_db_lands_on_v11_with_all_dance_tables(work: Path) -> None:
    conn = fresh_db(work)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        assert version == schema.SCHEMA_VERSION, \
            f"新建库该是 v{schema.SCHEMA_VERSION}，实际 user_version={version}"

        meta = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
        assert meta and meta["value"] == str(schema.SCHEMA_VERSION), \
            "schema_meta 里的版本也要写对"


        tables = _objects(conn, "table")
        missing = [name for name in DANCE_TABLE_NAMES if name not in tables]
        assert not missing, f"少建了这些表：{missing}"

        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == [], "外键必须干净"

        indexes = [n for n in _objects(conn, "index") if n.startswith("idx_dance")]
        assert len(indexes) >= 18, f"dance 索引太少（{len(indexes)}），查询会退化成全表扫"
    finally:
        conn.close()


# ------------------------------------------------------------------ T2
def test_upgrade_from_v10_touches_no_existing_table(work: Path) -> None:
    legacy = legacy_db(work)
    try:
        before = _non_dance(legacy, "table")
        before_idx = _non_dance(legacy, "index")
        version = migrations.apply(legacy)
        assert version == schema.SCHEMA_VERSION, version


        after = _non_dance(legacy, "table")
        assert after == before, "v11 不许改动任何已有表的定义"
        after_idx = _non_dance(legacy, "index")
        assert after_idx == before_idx, "v11 不许改动任何已有索引"


        tables = _objects(legacy, "table")
        missing = [name for name in DANCE_TABLE_NAMES if name not in tables]
        assert not missing, f"升级后少了这些表：{missing}"
        assert legacy.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        legacy.close()


# ------------------------------------------------------------------ T3
def test_both_paths_produce_identical_schema(work: Path) -> None:
    """新建 vs 升级：dance_* 的建表语句必须逐字相同，索引集合也必须相同。"""
    fresh = fresh_db(work, "a.db")
    legacy = legacy_db(work, "b.db")
    try:
        migrations.apply(legacy)
        for kind, prefix in (("table", "dance_"), ("index", "idx_dance")):
            one = {n: s for n, s in _objects(fresh, kind).items() if n.startswith(prefix)}
            two = {n: s for n, s in _objects(legacy, kind).items() if n.startswith(prefix)}
            assert set(one) == set(two), \
                f"{kind} 集合不一致：只在新建库 {set(one) - set(two)}，只在升级库 {set(two) - set(one)}"
            for name in one:
                assert one[name] == two[name], f"{name} 的 SQL 两条路径不一致"
    finally:
        fresh.close()
        legacy.close()


# ------------------------------------------------------------------ T4
def test_apply_is_idempotent(work: Path) -> None:
    conn = fresh_db(work)
    try:
        again = migrations.apply(conn)
        assert again == schema.SCHEMA_VERSION, again
        third = migrations.apply(conn)
        assert third == schema.SCHEMA_VERSION, third

        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_newer_db_is_left_alone(work: Path) -> None:
    """库比程序新时只告警不动它 —— 别把用户用新版程序建的库降级砸坏。"""
    conn = fresh_db(work)
    try:
        conn.execute("PRAGMA user_version=99")
        assert migrations.apply(conn) == 99, "版本比程序新时必须原样返回，不许改结构"
    finally:
        conn.close()


# ------------------------------------------------------------------ T5
def test_v11_only_creates_never_drops() -> None:
    """静态检查 v11 的语句：只允许 CREATE，一条 DROP / ALTER 已有表都不许有。"""
    steps = migrations._STEPS[11]                                  # noqa: SLF001 - 就是要盯它
    assert steps, "v11 不能是空的"
    for statement in steps:
        head = " ".join(statement.strip().split()[:2]).upper()
        assert head.startswith("CREATE"), f"v11 只允许 CREATE，出现了：{head}"
        upper = statement.upper()
        assert " DROP " not in f" {upper} ", f"v11 不许出现 DROP：{statement[:80]}"
        assert "ALTER TABLE" not in upper, f"v11 不许 ALTER 已有表：{statement[:80]}"
        assert "DELETE FROM" not in upper, f"v11 不许删数据：{statement[:80]}"
        assert "UPDATE " not in upper, f"v11 不许改数据：{statement[:80]}"
    # 而且必须逐字等于 schema.DANCE_TABLES，保证两条路径不会走散
    assert steps == list(schema.DANCE_TABLES), "v11 必须直接复用 schema.DANCE_TABLES"


def test_v12_only_creates_never_drops() -> None:
    """v12（目标歌下架表）同样只许 CREATE，且必须直接复用 schema.DANCE_V12_TABLES。

    为什么不给 dance_target_songs 加一列：ADD COLUMN 会让"升级上来的表"和
    "新建库的表"的 SQL 文本不再逐字相同 —— 那正是 v4 走散的原因，
    上面 test_both_paths_produce_identical_schema 就是盯这个的。
    """
    steps = migrations._STEPS[12]                                  # noqa: SLF001 - 就是要盯它
    assert steps, "v12 不能是空的"
    for statement in steps:
        head = " ".join(statement.strip().split()[:2]).upper()
        assert head.startswith("CREATE"), f"v12 只允许 CREATE，出现了：{head}"
        upper = statement.upper()
        assert "ALTER TABLE" not in upper, f"v12 不许 ALTER 已有表：{statement[:80]}"
        assert " DROP " not in f" {upper} ", f"v12 不许出现 DROP：{statement[:80]}"
    assert steps == list(schema.DANCE_V12_TABLES), "v12 必须直接复用 schema.DANCE_V12_TABLES"
    # 下架表不能出现在 v11 那一批里，否则老库升级顺序会乱
    assert not (set(schema.DANCE_TABLES) & set(schema.DANCE_V12_TABLES))


def test_v13_only_creates_never_drops() -> None:
    """v13（用户段落模板表）同样只许 CREATE，且必须直接复用 schema.DANCE_V13_TABLES。

    段落模板是用户拖出来的资产，老库升上来是一张空表 = 还没人分过段，
    界面照旧按等间隔起步 —— 升级不会改变任何既有行为。
    """
    steps = migrations._STEPS[13]                                  # noqa: SLF001 - 就是要盯它
    assert steps, "v13 不能是空的"
    for statement in steps:
        head = " ".join(statement.strip().split()[:2]).upper()
        assert head.startswith("CREATE"), f"v13 只允许 CREATE，出现了：{head}"
        upper = statement.upper()
        assert "ALTER TABLE" not in upper, f"v13 不许 ALTER 已有表：{statement[:80]}"
        assert " DROP " not in f" {upper} ", f"v13 不许出现 DROP：{statement[:80]}"
    assert steps == list(schema.DANCE_V13_TABLES), "v13 必须直接复用 schema.DANCE_V13_TABLES"
    assert not (set(schema.DANCE_TABLES) & set(schema.DANCE_V13_TABLES))
    assert not (set(schema.DANCE_V12_TABLES) & set(schema.DANCE_V13_TABLES))



# ------------------------------------------------------------------ 造数据
NOW = "2026-01-01T00:00:00"


def _seed(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """插一个视频 + 一首目标歌 + 一条对齐，返回 `(video_id, song_id, alignment_id)`。"""
    conn.execute(
        "INSERT INTO videos(fingerprint, file_path, file_name, created_at, updated_at) "
        "VALUES('fp-v1', 'C:/v1.mp4', 'v1.mp4', ?, ?)", (NOW, NOW))
    video_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        "INSERT INTO dance_target_songs(fingerprint, file_path, file_name, duration, "
        "created_at, updated_at) VALUES('fp-s1', 'C:/s1.wav', 's1.wav', 10.0, ?, ?)",
        (NOW, NOW))
    song_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        "INSERT INTO dance_audio_alignments(source_video_id, target_song_id, cache_key, "
        "offset_seconds, confidence, status, created_at, updated_at) "
        "VALUES(?, ?, 'ck-1', 1.25, 0.9, 'ok', ?, ?)", (video_id, song_id, NOW, NOW))
    alignment_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    return video_id, song_id, alignment_id


def _material(conn: sqlite3.Connection, video_id: int, song_id: int, alignment_id: int,
              segment_index: int, version: str = "slice-v1") -> int:
    conn.execute(
        "INSERT INTO dance_materials(source_video_id, alignment_id, target_song_id, "
        "segment_index, target_start, target_end, source_start, source_end, duration, "
        "generation_version, created_at, updated_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, 2.0, ?, ?, ?)",
        (video_id, alignment_id, song_id, segment_index, segment_index * 2.0,
         segment_index * 2.0 + 2.0, segment_index * 2.0 - 1.25,
         segment_index * 2.0 + 0.75, version, NOW, NOW))
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


# ------------------------------------------------------------------ T6
def test_foreign_keys_cascade(work: Path) -> None:
    conn = fresh_db(work)
    try:
        video_id, song_id, alignment_id = _seed(conn)
        material_id = _material(conn, video_id, song_id, alignment_id, 0)
        conn.execute(
            "INSERT INTO dance_material_usage_events(material_id, event, created_at) "
            "VALUES(?, 'candidate', ?)", (material_id, NOW))
        assert conn.execute("SELECT COUNT(*) FROM dance_materials").fetchone()[0] == 1

        # 删目标歌 → 对齐和素材级联走掉，事件跟着素材一起走
        conn.execute("DELETE FROM dance_target_songs WHERE id=?", (song_id,))
        for table in ("dance_audio_alignments", "dance_materials",
                      "dance_material_usage_events"):
            left = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert left == 0, f"删目标歌后 {table} 该级联清空，实际还剩 {left} 行"
        # 视频本身不受影响：素材是派生物，源片不是
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_orphan_reference_is_refused(work: Path) -> None:
    """外键真的开着：往 dance_materials 塞一个不存在的 target_song_id 必须被拒。"""
    conn = fresh_db(work)
    try:
        video_id, song_id, alignment_id = _seed(conn)
        try:
            conn.execute(
                "INSERT INTO dance_materials(source_video_id, alignment_id, target_song_id, "
                "segment_index, target_start, target_end, source_start, source_end, "
                "duration, generation_version, created_at, updated_at) "
                "VALUES(?, ?, 9999, 0, 0.0, 2.0, 0.0, 2.0, 2.0, 'v', ?, ?)",
                (video_id, alignment_id, NOW, NOW))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("引用不存在的目标歌必须被外键拦住")
    finally:
        conn.close()


# ------------------------------------------------------------------ T7
def test_material_uniqueness(work: Path) -> None:
    """同一首歌 + 同一个源 + 同一个音乐位置 + 同一个切片版本 = 只能有一条。"""
    conn = fresh_db(work)
    try:
        video_id, song_id, alignment_id = _seed(conn)
        first = _material(conn, video_id, song_id, alignment_id, 3)
        assert first > 0
        try:
            _material(conn, video_id, song_id, alignment_id, 3)
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("重复的 (歌, 源, 位置, 切片版本) 必须被唯一约束拦住")
        # 换切片版本就是另一条素材（重切之后旧素材要留着，历史成品还引用它）
        second = _material(conn, video_id, song_id, alignment_id, 3, version="slice-v2")
        assert second != first
        assert conn.execute("SELECT COUNT(*) FROM dance_materials").fetchone()[0] == 2
        # 换位置也是另一条
        _material(conn, video_id, song_id, alignment_id, 4)
        assert conn.execute("SELECT COUNT(*) FROM dance_materials").fetchone()[0] == 3
    finally:
        conn.close()


def test_alignment_cache_key_is_unique(work: Path) -> None:
    """cache_key 唯一 = 同一套输入不会算两遍、也不会存出两条打架的结果。"""
    conn = fresh_db(work)
    try:
        video_id, song_id, _ = _seed(conn)
        try:
            conn.execute(
                "INSERT INTO dance_audio_alignments(source_video_id, target_song_id, "
                "cache_key, offset_seconds, created_at, updated_at) "
                "VALUES(?, ?, 'ck-1', 9.9, ?, ?)", (video_id, song_id, NOW, NOW))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("同一个 cache_key 必须被唯一约束拦住")
    finally:
        conn.close()


def test_version_index_is_unique_per_montage(work: Path) -> None:
    """一个混剪任务下版本号唯一 —— 历史版本靠它保住，不许被覆盖。"""
    conn = fresh_db(work)
    try:
        _, song_id, _ = _seed(conn)
        conn.execute("INSERT INTO dance_montages(target_song_id, name, created_at, updated_at)"
                     " VALUES(?, 'M1', ?, ?)", (song_id, NOW, NOW))
        montage_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute("INSERT INTO dance_montage_versions(montage_id, version_index, "
                     "created_at, updated_at) VALUES(?, 1, ?, ?)", (montage_id, NOW, NOW))
        try:
            conn.execute("INSERT INTO dance_montage_versions(montage_id, version_index, "
                         "created_at, updated_at) VALUES(?, 1, ?, ?)", (montage_id, NOW, NOW))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("同一任务下重复的 version_index 必须被拦住")
        conn.execute("INSERT INTO dance_montage_versions(montage_id, version_index, "
                     "created_at, updated_at) VALUES(?, 2, ?, ?)", (montage_id, NOW, NOW))
        assert conn.execute(
            "SELECT COUNT(*) FROM dance_montage_versions").fetchone()[0] == 2
    finally:
        conn.close()


def test_default_strategy_is_unique(work: Path) -> None:
    """默认策略只能有一份（部分唯一索引，软删的不占位）。"""
    conn = fresh_db(work)
    try:
        conn.execute("INSERT INTO dance_montage_strategies(name, is_default, created_at, "
                     "updated_at) VALUES('A', 1, ?, ?)", (NOW, NOW))
        try:
            conn.execute("INSERT INTO dance_montage_strategies(name, is_default, created_at,"
                         " updated_at) VALUES('B', 1, ?, ?)", (NOW, NOW))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("同时两份默认策略必须被拦住")
        # 软删掉 A 之后 B 就能当默认
        conn.execute("UPDATE dance_montage_strategies SET deleted_at=? WHERE name='A'", (NOW,))
        conn.execute("INSERT INTO dance_montage_strategies(name, is_default, created_at, "
                     "updated_at) VALUES('B', 1, ?, ?)", (NOW, NOW))
        live = conn.execute("SELECT name FROM dance_montage_strategies "
                            "WHERE is_default=1 AND deleted_at IS NULL").fetchall()
        assert [r["name"] for r in live] == ["B"], live
    finally:
        conn.close()


# ------------------------------------------------------------------ T8
def test_health_check_passes_on_v11(work: Path) -> None:
    """体检的表/索引清单是从 schema.py 现提取的，所以新表必须自动被纳入检查。"""
    from vidscribe.db.db import Database                       # noqa: PLC0415

    db = Database(work / "health.db")
    try:
        report = db_admin.health_check(db)
        assert report["ok"], f"v11 库该体检通过，问题：{report['problems']}"
        assert report["version"] == schema.SCHEMA_VERSION, report["version"]
        assert report["expected_version"] == schema.SCHEMA_VERSION, report["expected_version"]

        assert not report["missing_tables"], report["missing_tables"]
        assert not report["missing_indexes"], report["missing_indexes"]
        assert str(report["integrity"]).lower() == "ok", report["integrity"]

        # 体检的清单是从 schema.py 现提取的，所以新表新索引必须自动进检查范围
        assert "dance_materials" in db_admin.EXPECTED_TABLES, \
            "体检的表清单该自动包含 dance_materials"
        assert any(name.startswith("idx_dance") for name in db_admin.EXPECTED_INDEXES), \
            "体检的索引清单该自动包含 dance 索引"
    finally:
        db.close()




TESTS = (
    test_fresh_db_lands_on_v11_with_all_dance_tables,
    test_upgrade_from_v10_touches_no_existing_table,
    test_both_paths_produce_identical_schema,
    test_apply_is_idempotent,
    test_newer_db_is_left_alone,
    test_v11_only_creates_never_drops,
    test_v12_only_creates_never_drops,
    test_v13_only_creates_never_drops,


    test_foreign_keys_cascade,
    test_orphan_reference_is_refused,
    test_material_uniqueness,
    test_alignment_cache_key_is_unique,
    test_version_index_is_unique_per_montage,
    test_default_strategy_is_unique,
    test_health_check_passes_on_v11,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancemig_"))
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





