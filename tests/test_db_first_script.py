"""分析结果全部落库 + 剧本从数据库重建（schema v5）。

这一批盯的是一个具体缺口：**表情轨（meta.face.segments）以前只写在 timeline.json 和
cache/visual.json 里**，删掉 output/ 与 cache/ 之后剧本的 SECTION 3 就再也拿不回来，
而代码会"静默少一段"，下游 AI 只能理解成"这个视频没有情绪信息"。

现在的规矩：
  数据库是分析结果的唯一权威来源；output/ 与 cache/ 只是导出/临时/兼容文件。

覆盖：
  T1  expression_spans 存取无损：字段、顺序、raw_json 一模一样
  T2  v4 老库能一路升到 v7：老数据一行不动，新表新列到位，新列是 NULL
  T3  DB 重建的剧本和内存态导出的**逐行一致**（SECTION 1/2/3/4 全比）
  T4  删掉 output/ 和 cache/ 之后，只靠库还能生成完整剧本
  T5  表情三态不混：正常 / 全片无脸 / 历史缺失，且历史缺失绝不伪造时间轴
  T6  改了当前 GUI 配置也不影响旧视频重建：用的是当次分析存下来的 render_config
  T7  落库顺序：表情轨和渲染参数都在 finish（标 completed）之前写
  T8  译文没落库就明确报错，不拿原文冒充译文
  T9  发给扩展 AI 的附件只有 .txt（PRM + 完整剧本），没有 MP4

全部用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_db_first_script.py`，也可以 `pytest tests/test_db_first_script.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 只导入模块，不建窗口

from vidscribe.config import Config                      # noqa: E402
from vidscribe.db import migrations, open_db             # noqa: E402
from vidscribe.db import repo as db_repo                 # noqa: E402
from vidscribe.db import schema                          # noqa: E402
from vidscribe.events import VisualEvent                 # noqa: E402
from vidscribe.highlight import clip_engine               # noqa: E402
from vidscribe.timeline import exporters                 # noqa: E402
from vidscribe.timeline.engine import (                  # noqa: E402
    action_track,
    build_timeline,
    filter_timeline,
)


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
    for sub in ("database", "input", "output", "logs", "cache"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data.setdefault("paths", {}).update({
        "db_dir": str(tmp_path / "database"),
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "input_dir": str(tmp_path / "input"),
        "video_dir": "",
        "log_dir": str(tmp_path / "logs"),
    })
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def fake_video(cfg, name: str = "demo.mp4") -> Path:
    path = cfg.path("input_dir") / name
    path.write_bytes(name.encode("utf-8") + bytes(range(256)) * 8)
    return path


DURATION = 30.0
RENDER_CFG = {"min_overlap_seconds": 0.2, "importance_filter": "low",
              "confidence_filter": 0.0}


def visual_events() -> list[dict]:
    """三条画面事件：带 OCR、动作、场景、人脸覆盖后的情绪，重要度各不相同。"""
    return [
        VisualEvent(id=1, start=0.0, end=6.0, event="", description="一个人走进厨房",
                    confidence=0.82, importance="normal", timestamp_source="frame_based",
                    source_frames=[1, 2], ocr_text="早餐时间",
                    action="walking", scene="kitchen", subjects=["person"],
                    emotion="平静", emotion_en="neutral", emotion_confidence=0.41,
                    emotion_source="face").to_dict(),
        VisualEvent(id=2, start=6.0, end=14.0, event="", description="他打翻了杯子",
                    confidence=0.91, importance="high", timestamp_source="frame_based",
                    source_frames=[3], ocr_text=None,
                    action="dropping", scene="kitchen", subjects=["person", "cup"],
                    emotion="惊讶", emotion_en="surprise", emotion_confidence=0.88,
                    emotion_source="face").to_dict(),
        VisualEvent(id=3, start=14.0, end=22.0, event="", description="他弯腰去擦地",
                    confidence=0.66, importance="normal", timestamp_source="hybrid",
                    source_frames=[4, 5], ocr_text=None,
                    action="cleaning", scene="kitchen", subjects=["person"],
                    emotion="平静", emotion_en="neutral", emotion_confidence=0.35,
                    emotion_source="face").to_dict(),
    ]


def speech_segments() -> list[dict]:
    """两段语音，逐词时间戳齐全（SECTION 4 的数据源）。"""
    return [
        {"id": 1, "start": 1.0, "end": 4.4, "text": "我先去拿个杯子",
         "confidence": 0.93, "language": "zh", "original_text": "我先去拿个杯子",
         "original_language": "zh", "speaker": 1, "speaker_confidence": 0.7,
         "emotion": "平静", "emotion_en": "neutral", "emotion_intensity": 0.52,
         "words": [{"word": "我先", "start": 1.0, "end": 2.0, "probability": 0.9},
                   {"word": "去拿个", "start": 2.0, "end": 3.2, "probability": 0.88},
                   {"word": "杯子", "start": 3.2, "end": 4.4, "probability": 0.95}]},
        {"id": 2, "start": 15.0, "end": 18.6, "text": "全洒出来了",
         "confidence": 0.88, "language": "zh", "original_text": "全洒出来了",
         "original_language": "zh", "speaker": 1, "speaker_confidence": 0.7,
         "emotion": "生气", "emotion_en": "angry", "emotion_intensity": 0.81,
         "words": [{"word": "全洒", "start": 15.0, "end": 16.4, "probability": 0.86},
                   {"word": "出来了", "start": 16.4, "end": 18.6, "probability": 0.9}]},
    ]


def face_spans() -> list[dict]:
    """人脸表情轨：2fps 采样归并出来的独立时间轴，中间故意留一个没检到脸的缺口。"""
    return [
        {"start": 0.5, "end": 5.5, "emotion_en": "neutral", "confidence": 0.44, "samples": 10},
        {"start": 6.0, "end": 12.5, "emotion_en": "surprise", "confidence": 0.91, "samples": 13},
        {"start": 19.0, "end": 22.0, "emotion_en": "angry", "confidence": 0.77, "samples": 6},
    ]


def expression_lines(lines: list[str]) -> list[str]:
    """切出 SECTION 3 的正文行（只要带时间戳那些，图例和分隔线不算）。

    断言必须绑定到这一段：全文找子串的写法（`any("0.91" in line for line in all)`）
    在别的段落里也可能命中，置信度整段退化成 0.00 都能照样通过。
    """
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("SECTION 3 - "):
            inside = True
            continue
        if inside and line.startswith("SECTION "):
            break
        if inside and line.startswith("["):
            out.append(line)
    return out


def seed_analysis(cfg, db, *, video: Path | None = None, spans: list[dict] | None = None,
                  face_available: bool | None = True,
                  render_config: dict | None = None,
                  output_language: str = "zh",
                  segments: list[dict] | None = None) -> tuple[Path, int, int]:
    """照分析流程往库里写一次完整结果，返回（视频, video_id, analysis_id）。"""
    video = video or fake_video(cfg)
    vid = db_repo.upsert_video(db, video, info={"duration": DURATION,
                                                "width": 1080, "height": 1920, "fps": 30.0})
    analysis = db_repo.create_analysis(db, vid, db_repo.signature(cfg))
    db_repo.save_speech_segments(db, analysis,
                                 segments if segments is not None else speech_segments())
    db_repo.save_visual_events(db, analysis, visual_events())
    if spans is not None:
        db_repo.save_expression_spans(db, analysis, spans)
    if face_available is not None or render_config is not None:
        db_repo.note_render(db, analysis, output_language=output_language,
                            render_config=render_config or RENDER_CFG,
                            face_available=face_available)
    db_repo.finish_analysis(db, analysis, scene_count=3, speech_count=2,
                            output_dir=cfg.path("output_dir"))
    return video, vid, analysis


def memory_baseline(video_name: str, segments: list[dict], events: list[dict],
                    spans: list[dict], language: str = "zh") -> list[str]:
    """内存态那条路的剧本（GUI 走的就是这条）：

    刻意照 pipeline 写 timeline.json 的样子来——条目只保留写进 JSON 的那些键
    （action / scene / subjects 是不写的），这样"库里重建"和"读 timeline.json"
    两条路的差异如果泄漏到正文里，逐行比对立刻能看出来。
    """
    from vidscribe.events import SpeechEvent, SpeechWord

    visual = [VisualEvent(**e) for e in events]
    speech = [SpeechEvent(words=[SpeechWord(**w) for w in (s.get("words") or [])],
                          **{k: v for k, v in s.items() if k != "words"})
              for s in segments]
    entries = build_timeline(visual, speech, min_overlap=RENDER_CFG["min_overlap_seconds"])
    filtered = filter_timeline(entries, importance=RENDER_CFG["importance_filter"],
                               min_confidence=RENDER_CFG["confidence_filter"])
    json_like = [{
        "start": e["start"], "end": e["end"], "visual": e["visual"], "speech": e["speech"],
        "importance": e["importance"], "timestamp_source": e["timestamp_source"],
        "ocr_text": e["ocr_text"], "visual_event_id": e["visual_event_id"],
        "speech_event_ids": e["speech_event_ids"], "source_frames": e["source_frames"],
        "visual_confidence": e["visual_confidence"], "speech_confidence": e["speech_confidence"],
        "speech_speakers": e["speech_speakers"], "speech_emotion": e["speech_emotion"],
        "speech_emotion_en": e["speech_emotion_en"],
        "speech_emotion_intensity": e["speech_emotion_intensity"],
        "visual_emotion": e["visual_emotion"], "visual_emotion_en": e["visual_emotion_en"],
        "visual_emotion_confidence": e["visual_emotion_confidence"], "quality": e["quality"],
    } for e in filtered]
    lines, _ = exporters.merged_lines(
        video_name, segments, exporters.export_events(json_like), False, language,
        actions=action_track(visual), emotions=spans, duration=DURATION)
    return lines


# ------------------------------------------------------------------ T1
def test_expression_spans_roundtrip(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, _vid, analysis = seed_analysis(cfg, db, spans=face_spans())

    rows = db_repo.get_expression_spans(db, analysis)
    assert len(rows) == 3, "三段进去三段出来"
    assert [int(r["sequence"]) for r in rows] == [1, 2, 3], "顺序必须是存进去的顺序"
    for row, span in zip(rows, face_spans()):
        assert float(row["start_time"]) == span["start"]
        assert float(row["end_time"]) == span["end"]
        assert row["emotion_en"] == span["emotion_en"]
        assert abs(float(row["confidence"]) - span["confidence"]) < 1e-9
        assert int(row["samples"]) == span["samples"]
        assert json.loads(row["raw_json"]) == span, "raw_json 必须是原样，字段一个不少"

    # 重存就是替换，不叠加
    db_repo.save_expression_spans(db, analysis, face_spans()[:1])
    assert len(db_repo.get_expression_spans(db, analysis)) == 1
    db.close()


# ------------------------------------------------------------------ T2
def test_v4_upgrades_to_v7_without_losing_data(tmp_path: Path) -> None:
    def v4_statements() -> list[str]:
        """把 schema.TABLES 退回 v5 之前：没有 expression_spans，analysis_runs 没有三个新列，
        prm_profiles 还没有 v6 的 enabled 列和 v8 的 content 列、videos 还没有 v7 的
        blocked_language 列和 v10 的 no_audio 列（不然升级脚本的 ADD COLUMN 会撞重名）。"""
        out: list[str] = []
        for statement in schema.TABLES:
            if statement in schema.DANCE_TABLES or statement in schema.DANCE_V12_TABLES:
                continue                      # v11/v12 的舞蹈表，老库里当然没有

            if "expression_spans" in statement:

                continue
            if "CREATE TABLE IF NOT EXISTS prm_profiles" in statement:
                out.append("\n".join(line for line in statement.splitlines()
                                     if "enabled" not in line and "content" not in line))
                continue
            if "CREATE TABLE IF NOT EXISTS videos" in statement:
                out.append("\n".join(line for line in statement.splitlines()
                                     if "blocked_language" not in line
                                     and "no_audio" not in line
                                     and not line.strip().startswith("--")))
                continue
            if "CREATE TABLE IF NOT EXISTS analysis_runs" in statement:
                keep = [line for line in statement.splitlines()
                        if "output_language" not in line and "render_config" not in line
                        and "face_available" not in line
                        and not line.strip().startswith("--")]
                out.append("\n".join(keep).replace("created_at         TEXT    NOT NULL,",
                                                   "created_at         TEXT    NOT NULL"))
                continue
            out.append(statement)
        return out

    path = tmp_path / "database" / "v4.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    old.execute("BEGIN")
    for statement in v4_statements():
        old.execute(statement)
    old.execute("PRAGMA user_version=4")
    old.execute("INSERT INTO videos(fingerprint, file_path, file_name, file_size, duration,"
                " status, created_at, updated_at) VALUES('fp1','/x/a.mp4','a.mp4',10,5.0,"
                "'new','2026-01-01T00:00:00','2026-01-01T00:00:00')")
    old.execute("INSERT INTO analysis_runs(video_id, status, scene_count, speech_count,"
                " created_at) VALUES(1,'completed',7,9,'2026-01-01T00:00:00')")
    old.commit()

    names = lambda: [r[1] for r in old.execute("PRAGMA table_info(analysis_runs)")]
    assert "face_available" not in names(), "造出来的老库不该有 v5 的列"

    assert migrations.apply(old) == schema.SCHEMA_VERSION, "v4 能一路升到当前版本"
    assert int(old.execute("PRAGMA user_version").fetchone()[0]) == schema.SCHEMA_VERSION


    tables = {r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "expression_spans" in tables, "新表要有"
    prm_cols = [r[1] for r in old.execute("PRAGMA table_info(prm_profiles)")]
    assert "enabled" in prm_cols, "v6 的「PRM 使用状况」列要升上来"
    video_cols = [r[1] for r in old.execute("PRAGMA table_info(videos)")]
    assert "blocked_language" in video_cols, "v7 的「语言拦截」列要升上来"
    assert "no_audio" in video_cols, "v10 的「音轨预检」列要升上来"
    assert "content" in prm_cols, "v8 的「PRM 正文」列要升上来"
    for col in ("output_language", "render_config", "face_available"):
        assert col in names(), f"analysis_runs 缺列 {col}"

    video = old.execute("SELECT * FROM videos WHERE id = 1").fetchone()
    run = old.execute("SELECT * FROM analysis_runs WHERE id = 1").fetchone()
    assert video["file_name"] == "a.mp4" and float(video["duration"]) == 5.0, "老数据不许动"
    assert run["status"] == "completed" and int(run["scene_count"]) == 7, "老数据不许动"
    assert run["output_language"] is None and run["render_config"] is None, "新列该是 NULL"
    assert run["face_available"] is None, "历史分析的 face_available 必须是 NULL（=没存过）"
    assert old.execute("SELECT COUNT(*) FROM expression_spans").fetchone()[0] == 0
    old.close()


# ------------------------------------------------------------------ T3
def test_script_from_db_matches_memory_line_by_line(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid, _analysis = seed_analysis(cfg, db, spans=face_spans())

    payload = db_repo.script_inputs(db, vid)
    assert payload is not None, "库里有 completed 分析就必须取得到"
    assert payload["expression_state"] == schema.EXPRESSION_OK
    from_db, count = exporters.script_lines(payload)
    baseline = memory_baseline(video.name, speech_segments(), visual_events(), face_spans())

    assert from_db == baseline, "库里重建的剧本必须和内存态导出的逐行一致"
    assert count > 0

    # 四段结构和缺口标注都在
    text = "\n".join(from_db)
    for section in ("SECTION 1 - ", "SECTION 2 - ", "SECTION 3 - ", "SECTION 4 - "):
        assert section in text, f"缺 {section}"
    assert "没检到人脸" in text, "表情轨中间的缺口必须显式标出来"
    assert "没有说话" in text, "逐词之间的静音必须显式标出来"
    assert "早餐时间" in text, "OCR 要在 SECTION 1 里"
    db.close()


# ------------------------------------------------------------------ T4
def test_script_survives_deleting_output_and_cache(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video, vid, _analysis = seed_analysis(cfg, db, spans=face_spans())

    # 派生文件：分析当时会写这些，删掉之后一个都不许再依赖
    out_dir = cfg.path("output_dir") / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timeline.json").write_text(json.dumps({"expression_track": face_spans()}),
                                           encoding="utf-8")
    cache_dir = cfg.path("cache_dir") / "videos" / video.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "visual.json").write_text(json.dumps({"meta": {"face": {"segments": face_spans()}}}),
                                            encoding="utf-8")
    baseline, _ = exporters.script_lines(db_repo.script_inputs(db, vid))

    shutil.rmtree(cfg.path("output_dir"))
    shutil.rmtree(cfg.path("cache_dir"))
    assert not cfg.path("output_dir").exists() and not cfg.path("cache_dir").exists()

    target = tmp_path / "rebuilt.txt"
    count = exporters.write_script_txt(target, db_repo.script_inputs(db, vid))
    rebuilt = target.read_text(encoding="utf-8").splitlines()
    assert rebuilt == baseline, "删了 output/ 和 cache/ 之后，库里重建的剧本要一字不差"
    assert count > 0
    text = "\n".join(rebuilt)
    assert "SECTION 3 - " in text, "表情轨必须来自库，不是来自被删掉的文件"
    assert any("0.91" in line for line in rebuilt), "表情置信度是真数据，不是默认值"
    db.close()


# ---------------------------------------------------------------- T10/T11/T12
def test_legacy_intensity_key_still_exports_real_confidence(tmp_path: Path) -> None:
    """v9 改名之前落库的表情段：列里是真值，`raw_json` 里还是旧键 `intensity`。

    `ALTER TABLE ... RENAME COLUMN` 不会动 JSON 正文，所以只读 raw_json 重建时
    `confidence` 整个缺失，导出端一兜底就把真值印成 `0.00`——而 0.00 在 PRM 里的含义是
    "模型完全不确定"，等于主动给下游 AI 喂假证据。这条链路必须有测试守着。
    """
    cfg, db = make_project(tmp_path)
    _video, vid, analysis = seed_analysis(cfg, db, spans=face_spans())

    # 把库退回"改名前"的形状：列保留真值，raw_json 换回旧键
    with db.tx() as conn:
        rows = conn.execute("SELECT id, raw_json FROM expression_spans WHERE analysis_id = ?",
                            (analysis,)).fetchall()
        assert rows, "夹具本身要先把表情段写进去"
        for row in rows:
            span = json.loads(row[1])
            span["intensity"] = span.pop("confidence")
            conn.execute("UPDATE expression_spans SET raw_json = ? WHERE id = ?",
                         (json.dumps(span, ensure_ascii=False), row[0]))

    payload = db_repo.script_inputs(db, vid)
    assert payload is not None
    spans = payload["emotions"]
    assert [s.get("confidence") for s in spans] == [0.44, 0.91, 0.77], \
        "列是权威：raw_json 里是旧键，也必须还原出真实置信度"
    assert not any("intensity" in s for s in spans), "旧键不许继续往下游流"

    section3 = expression_lines(exporters.script_lines(payload)[0])
    assert len(section3) == 3, f"SECTION 3 该有三行：{section3}"
    assert not any(line.endswith("0.00") for line in section3), \
        f"置信度被兜底成 0.00 就是把真数据丢了：{section3}"
    assert any("0.91" in line for line in section3), "真值要原样印出来"
    db.close()


def test_saving_legacy_shaped_span_fills_the_confidence_column(tmp_path: Path) -> None:
    """旧形状的 span 落库不许把 confidence 列写成 NULL。

    `pipeline._annotate_face_emotion` 复用缓存的闸门只看有没有 segments、不看键名，
    所以旧缓存里的 `intensity` 形状完全可能一路走到落库。写成 NULL 比印成 0.00 更糟：
    那是把唯一一份好数据也丢了，只能重新分析。
    """
    cfg, db = make_project(tmp_path)
    _video, _vid, analysis = seed_analysis(cfg, db, spans=face_spans())

    legacy = [{"start": 1.0, "end": 2.5, "emotion_en": "happy", "intensity": 0.8, "samples": 3}]
    assert db_repo.save_expression_spans(db, analysis, legacy) == 1
    row = db.one("SELECT confidence, raw_json FROM expression_spans WHERE analysis_id = ?",
                 (analysis,))
    assert row["confidence"] == 0.8, "旧键也要填进 confidence 列，不能是 NULL"
    stored = json.loads(row["raw_json"])
    assert stored.get("confidence") == 0.8 and "intensity" not in stored, \
        "raw_json 也要归一化，别让旧形状继续扩散"
    assert legacy[0] == {"start": 1.0, "end": 2.5, "emotion_en": "happy",
                         "intensity": 0.8, "samples": 3}, "不许改调用方传进来的 dict"
    db.close()


def test_derived_files_with_legacy_keys_still_print_real_numbers(tmp_path: Path) -> None:
    """老 `output/*/timeline.json`：表情段是 `intensity`，条目情绪是 `visual_emotion_intensity`。

    这条路不经过数据库（GUI 直接把 timeline.json 透传给导出函数），所以列权威那招救不了它，
    只能靠读侧认旧名。仓库里现存 23 份旧表情轨 + 27 份旧条目情绪都走这条路。
    """
    entries = [{"start": 1.0, "end": 2.5, "visual": "她突然笑出来", "speech": "",
                "importance": "normal", "timestamp_source": "word_anchored",
                "visual_emotion": "开心", "visual_emotion_en": "happy",
                "visual_emotion_intensity": 0.83}]
    spans = [{"start": 1.0, "end": 2.5, "emotion_en": "happy", "intensity": 0.83, "samples": 4}]

    target = tmp_path / "timeline.txt"
    exporters.write_timeline_txt(target, "a.mp4", 5.0, "zh", entries,
                                 output_language="zh", emotions=spans)
    text = target.read_text(encoding="utf-8")
    expression = [ln for ln in text.splitlines() if ln.startswith("[") and "开心" in ln]
    assert expression, f"表情轨那行都没了：\n{text}"
    # 断言绑定到表情行本身：全文找 "0.00" 会被时间戳（00:10.00 这种）误伤
    assert expression[-1].endswith("开心 0.83"), f"旧键的置信度要照样印出来：{expression}"
    assert "（开心 0.83）" in text, "画面行的情绪数字不许消失"


def test_legacy_visual_emotion_key_survives_db_roundtrip(tmp_path: Path) -> None:
    """画面情绪置信度的旧名是 `emotion_intensity`：写侧要归一化，读侧也要认。

    visual_events 这张表**没有情绪置信度列**（`confidence` 列是事件置信度，
    filter_timeline 拿它当阈值），`raw_json` 是这个数的唯一副本。
    按 dataclass 字段名硬过滤重建（曾经的 `_rebuild`）会把旧键当"不认识的键"丢掉，
    于是 SECTION 1 画面行的情绪数字**整个消失**——不是印 0.00，是不显示，比 0.00 更难发现。
    """
    cfg, db = make_project(tmp_path)
    _video, vid, analysis = seed_analysis(cfg, db, spans=face_spans())

    # 写侧：旧形状喂进来，落库必须统一成新键
    legacy = []
    for event in visual_events():
        event = dict(event)
        event["emotion_intensity"] = event.pop("emotion_confidence")
        legacy.append(event)
    db_repo.save_visual_events(db, analysis, legacy)
    stored = [json.loads(row["raw_json"]) for row in db_repo.get_visual_events(db, analysis)]
    assert all("emotion_intensity" not in s for s in stored), "写侧要把旧键归一化掉"
    assert [s.get("emotion_confidence") for s in stored] == [0.41, 0.88, 0.35]

    # 读侧：把 raw_json 退回旧形状，模拟改名之前就落好的库
    with db.tx() as conn:
        rows = conn.execute("SELECT id, raw_json FROM visual_events WHERE analysis_id = ?",
                            (analysis,)).fetchall()
        assert rows
        for row in rows:
            data = json.loads(row[1])
            data["emotion_intensity"] = data.pop("emotion_confidence")
            conn.execute("UPDATE visual_events SET raw_json = ? WHERE id = ?",
                         (json.dumps(data, ensure_ascii=False), row[0]))

    lines, _ = exporters.script_lines(db_repo.script_inputs(db, vid))
    # 时间戳是右对齐补空格的（见 exporters._span），所以别按整段前缀匹配
    row = [ln for ln in lines if "6.00 -" in ln and "14.00]" in ln and "画面" in ln]
    assert row, f"SECTION 1 里该有这条画面行：\n{chr(10).join(lines[:20])}"
    # 显示名按 emotions 表现渲（中英都可能），这里只盯数字：数字消失才是那个 bug
    assert "0.88" in row[0], f"画面行的情绪置信度不许消失：{row[0]}"


# ------------------------------------------------------------------ T5
def test_expression_states_never_get_confused(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)

    ok_video, ok_vid, _ = seed_analysis(cfg, db, video=fake_video(cfg, "ok.mp4"),
                                        spans=face_spans(), face_available=True)
    none_video, none_vid, _ = seed_analysis(cfg, db, video=fake_video(cfg, "noface.mp4"),
                                            spans=[], face_available=False)
    # 历史分析：既没存表情轨，也没写过 face_available（v4 时代跑的）
    old_video, old_vid, old_analysis = seed_analysis(cfg, db,
                                                    video=fake_video(cfg, "legacy.mp4"),
                                                    spans=None, face_available=None)

    states = {p: db_repo.script_inputs(db, p)["expression_state"]
              for p in (ok_vid, none_vid, old_vid)}
    assert states[ok_vid] == schema.EXPRESSION_OK
    assert states[none_vid] == schema.EXPRESSION_NO_FACE, "跑过人脸模型但没脸 = no_face"
    assert states[old_vid] == schema.EXPRESSION_LEGACY_MISSING, "没存过 = legacy_missing"

    no_face_lines, _ = exporters.script_lines(db_repo.script_inputs(db, none_vid))
    legacy_lines, _ = exporters.script_lines(db_repo.script_inputs(db, old_vid))
    for lines, keyword in ((no_face_lines, "全片没有检测到有效人脸"),
                           (legacy_lines, "重新分析该视频")):
        text = "\n".join(lines)
        assert "SECTION 3 - " in text, "SECTION 3 不许静默消失"
        assert keyword in text, f"必须说清为什么是空的：缺「{keyword}」"
    # 两种空状态的说明不能是同一句
    assert "全片没有检测到有效人脸" not in "\n".join(legacy_lines)
    assert "重新分析该视频" not in "\n".join(no_face_lines)

    # 绝不伪造：空状态下 SECTION 3 里一条时间戳都不许有
    def section3(lines: list[str]) -> list[str]:
        body = "\n".join(lines).split("SECTION 3 - ")[1].split("SECTION 4")[0]
        return [ln for ln in body.splitlines() if ln.startswith("[")]

    assert section3(no_face_lines) == [], "no_face 不许编时间轴"
    assert section3(legacy_lines) == [], "legacy_missing 不许编时间轴"
    assert len(section3(exporters.script_lines(db_repo.script_inputs(db, ok_vid))[0])) == 3

    # 老记录没存过 output_language：语言由调用方兜底，不许一律当中文
    legacy_payload = db_repo.script_inputs(db, old_vid)
    assert legacy_payload["output_language"] is None
    english, _ = exporters.script_lines(legacy_payload, language="en")
    assert any("Expression track" in line for line in english), "兜底语言要生效"
    assert ok_video.exists() and none_video.exists() and old_video.exists()
    assert old_analysis > 0
    db.close()


# ------------------------------------------------------------------ T6
def test_rebuild_uses_the_render_config_of_that_analysis(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    # 当次分析：只留 high 及以上的画面条目
    strict = {"min_overlap_seconds": 0.2, "importance_filter": "high",
              "confidence_filter": 0.0}
    _video, vid, _analysis = seed_analysis(cfg, db, spans=face_spans(), render_config=strict)

    # 用户之后把 GUI 配置改成"全都要"——重建旧视频时必须无视它
    cfg.timeline["importance_filter"] = "low"
    cfg.timeline["confidence_filter"] = 0.9

    payload = db_repo.script_inputs(db, vid)
    assert payload["render_config"] == strict, "存的是当次那套参数"
    lines, _ = exporters.script_lines(payload)
    section1 = "\n".join(lines).split("SECTION 1 - ")[1].split("SECTION 2")[0]
    assert "他打翻了杯子" in section1, "high 的那条必须在"
    assert "一个人走进厨房" not in section1, "normal 的那条按当次参数就该被筛掉"
    assert "他弯腰去擦地" not in section1
    # 语音行永远不因重要度被丢
    assert "我先去拿个杯子" in section1
    db.close()


# ------------------------------------------------------------------ T7
def test_expression_is_saved_before_finish(tmp_path: Path) -> None:
    """落库顺序：表情轨 / 渲染参数写完才允许标 completed。"""
    from vidscribe import pipeline as pl

    source = Path(pl.__file__).read_text(encoding="utf-8")
    save_at = source.index("run.save_expression(")
    note_at = source.index("run.note_render(")
    finish_at = source.index("run.finish(")
    assert save_at < finish_at, "save_expression 必须在 finish 之前"
    assert note_at < finish_at, "note_render 必须在 finish 之前"

    cfg, db = make_project(tmp_path)
    video = fake_video(cfg, "order.mp4")
    vid = db_repo.upsert_video(db, video, info={"duration": DURATION})
    analysis = db_repo.create_analysis(db, vid, db_repo.signature(cfg))

    run = pl.DbRun.__new__(pl.DbRun)
    run.cfg, run.video_path, run.force = cfg, video, False
    run.db, run.video_id, run.analysis_id = db, vid, analysis
    run.reused, run.sig = False, {}

    calls: list[str] = []
    real_spans, real_render = db_repo.save_expression_spans, db_repo.note_render
    real_finish = db_repo.finish_analysis
    try:
        db_repo.save_expression_spans = lambda *a, **k: (calls.append("expression"),
                                                        real_spans(*a, **k))[1]
        db_repo.note_render = lambda *a, **k: (calls.append("render"), real_render(*a, **k))[1]
        db_repo.finish_analysis = lambda *a, **k: (calls.append("finish"), real_finish(*a, **k))[1]
        run.save_expression(face_spans())
        run.note_render(output_language="zh", render_config=RENDER_CFG, face_available=True)
        run.finish(visual_count=3, speech_count=2, out_dir=cfg.path("output_dir"))
    finally:
        db_repo.save_expression_spans, db_repo.note_render = real_spans, real_render
        db_repo.finish_analysis = real_finish

    assert calls == ["expression", "render", "finish"], f"顺序不对：{calls}"
    row = db.one("SELECT * FROM analysis_runs WHERE id = ?", (analysis,))
    assert row["status"] == "completed"
    assert int(row["face_available"]) == 1
    assert json.loads(row["render_config"]) == RENDER_CFG
    assert len(db_repo.get_expression_spans(db, analysis)) == 3, \
        "标了 completed 就必须已经有表情轨"
    db.close()


# ------------------------------------------------------------------ T8
def test_missing_translation_is_reported_not_faked(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    _video, vid, _analysis = seed_analysis(cfg, db, spans=face_spans())
    payload = db_repo.script_inputs(db, vid)

    try:
        exporters.script_lines(payload, translated=True)
    except ValueError as exc:
        assert "译文未落库" in str(exc), f"报错要说清是译文的问题：{exc}"
    else:
        raise AssertionError("库里没有译文却照样出了「译文」剧本——那是拿原文冒充的")

    # 库里真有译文就正常出译文
    translated = [dict(seg, text_translated=f"[EN] {seg['text']}") for seg in speech_segments()]
    _video2, vid2, _a2 = seed_analysis(cfg, db, video=fake_video(cfg, "tr.mp4"),
                                       spans=face_spans(), segments=translated)
    lines, _ = exporters.script_lines(db_repo.script_inputs(db, vid2), translated=True)
    assert any("[EN] 全洒出来了" in line for line in lines)
    db.close()


# ------------------------------------------------------------------ T9
def test_ai_attachments_are_two_txt_files(tmp_path: Path) -> None:
    """发给扩展 AI 的只有 PRM + 完整剧本，两个都是 .txt，MP4 绝不上传。"""
    from vidscribe.gui import main_window as mw

    cfg, db = make_project(tmp_path)
    video, vid, _analysis = seed_analysis(cfg, db, spans=face_spans())
    prm = tmp_path / "prm_en.txt"
    prm.write_text("rules", encoding="utf-8")
    script = tmp_path / f"{video.stem}.txt"
    exporters.write_script_txt(script, db_repo.script_inputs(db, vid))

    sent: list[dict] = []

    class Bridge:
        def __init__(self):
            self.tasks: list[dict] = []

        def submit(self, task_type, payload, files=None):
            self.tasks.append({"type": task_type, "payload": payload,
                               "files": [Path(f) for f in (files or [])]})
            sent.append(self.tasks[-1])
            return f"task-{len(self.tasks)}"

        def state(self):
            return {"extension_online": True}

    class Win:
        dispatch_ai = mw.MainWindow.dispatch_ai
        _ai_files_ok = mw.MainWindow._ai_files_ok      # 硬闸门用真的那一份，别绕过
        VIDEO_SUFFIXES = mw.MainWindow.VIDEO_SUFFIXES
        _note_prompt_use = staticmethod(lambda *a, **k: None)
        _mark_auto_waiting = staticmethod(lambda *a, **k: None)
        refresh_bridge_label = staticmethod(lambda *a, **k: None)

        def __init__(self):
            self.cfg = cfg
            self.cfg.bridge["mode"] = "extension"
            self.cfg.bridge["upload_mode"] = "auto"
            self.bridge = Bridge()
            self.video_path = video
            self._auto_video = video
            self._auto_task_id = None
            self._auto_job = "full"
            self.logs: list[str] = []

        def append_log(self, text):
            self.logs.append(text)

    win = Win()
    assert win.dispatch_ai(prm, script, 12, video=video) is True
    assert len(sent) == 1
    names = {f.name for f in sent[0]["files"]}
    assert names == {prm.name, script.name}, f"上传的应该只有这两份文本：{names}"
    assert all(f.suffix == ".txt" for f in sent[0]["files"]), "附件只能是 .txt"
    assert video.name not in names and not any(f.suffix == ".mp4" for f in sent[0]["files"]), \
        "MP4 绝不上传"
    assert sent[0]["payload"].get("video") == video.name, "任务里仍要标明是哪个 MP4 的"
    db.close()


def test_broken_sentences_join_and_pairs_are_offered(tmp_path: Path) -> None:
    """给 AI 看的剧本：断句拼回一行、噪声不进来、SECTION 5 给出算好的候选对。

    这三件事一起验：AI 反复把「条件说完」当成结果，错因在数据呈现——
    同一句话被 ASR 切成三条摆在三行上。
    """
    def word(start: float, end: float, text: str) -> dict:
        return {"word": text, "start": start, "end": end}

    segments = [
        # 一句话被静音切成两片：前一片不以句末标点收尾
        {"start": 10.0, "end": 10.4, "text": "If", "speaker": 1,
         "emotion_en": "disgusted", "emotion_intensity": 0.87,
         "words": [word(10.0, 10.4, "If")]},
        {"start": 13.0, "end": 14.0, "text": "this is green, I'm out.", "speaker": 1,
         "words": [word(13.0, 13.5, "this"), word(13.5, 14.0, "green,")]},
        # 短反应：结果句
        {"start": 14.4, "end": 14.8, "text": "Yellow.", "speaker": 2,
         "emotion_en": "happy", "emotion_intensity": 0.90,
         "words": [word(14.4, 14.8, "Yellow.")]},
    ]
    events = [{"start": 0.0, "end": 12.0, "description": "two people talk",
               "ocr_text": "MILAN", "importance": "normal"},
              {"start": 12.0, "end": 20.0, "description": "one reacts",
               "ocr_text": "MILAN", "importance": "normal"},
              {"start": 20.0, "end": 22.0, "description": "close up", "ocr_text": "ab"}]
    # 表情轨只在 14.4 那一段给 happy：音频的 disgusted 没有旁证，不许打
    emotions = [{"start": 14.0, "end": 15.0, "emotion_en": "happy", "confidence": 0.7}]

    lines, _ = exporters.merged_lines("v.mp4", segments, events, language="en",
                                     emotions=emotions, duration=22.0)
    text = "\n".join(lines)
    section1 = text.split("SECTION 1 - ")[1].split("SECTION 2")[0]

    # 拼回一行：不再有单独的 "If" 那一行，句内静音要注明
    assert "If this is green, I'm out." in section1, section1
    assert "Speech (speaker 1): If\n" not in section1, "被切断的半句不许单独成行"
    assert "inner silence" in section1, "句内静音必须标出来，否则下游以为一直在说"
    # 每句一个编号，画面行也各有编号：SECTION 4 / 5 靠它指认，不用数时间戳
    assert "S01  [" in section1 and "S02  [" in section1, section1
    assert "V01  [" in section1, section1
    # 音频情绪要双证据：disgusted 没有表情轨支持 -> 不打；happy 有 -> 打
    assert "disgusted" not in section1, "只有音频这一路的情绪是噪声，不许进来"
    assert "happy" in section1
    # OCR：重复的只打一次，过短的不打
    assert section1.count("MILAN") == 1, section1
    assert "On-screen text: ab" not in section1, "两个字的画面文字是误识别碎片，不许打"

    # SECTION 5：候选对存在，区间和 plan_from_span 算的一致，并引用句子编号
    section5 = text.split("SECTION 5 - ")[1]
    assert "setup_at 10.00 / result_at 14.40" in section5, section5
    assert "S01 -> S02" in section5, section5
    plan = clip_engine.plan_from_span(
        clip_engine.segments_from_payload(segments), 10.0, 14.4)
    assert plan is not None
    assert f"{plan[0]:.2f} - {plan[1]:.2f}" in section5, section5
    # 表头必须说清这是候选不是全集，否则 AI 只会在候选里挑
    assert "not the full set" in text
    # 客观数字摆在剧本里给 AI 当依据，而不是写进高光 JSON 替它做判断
    assert "result voiced" in section5 and "silence inside the clip" in section5, section5


TESTS = [
    test_expression_spans_roundtrip,
    test_v4_upgrades_to_v7_without_losing_data,
    test_script_from_db_matches_memory_line_by_line,
    test_script_survives_deleting_output_and_cache,
    test_legacy_intensity_key_still_exports_real_confidence,
    test_saving_legacy_shaped_span_fills_the_confidence_column,
    test_derived_files_with_legacy_keys_still_print_real_numbers,
    test_legacy_visual_emotion_key_survives_db_roundtrip,
    test_expression_states_never_get_confused,
    test_rebuild_uses_the_render_config_of_that_analysis,
    test_expression_is_saved_before_finish,
    test_missing_translation_is_reported_not_faked,
    test_ai_attachments_are_two_txt_files,
    test_broken_sentences_join_and_pairs_are_offered,
]


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dbfirst_"))
        try:
            fn(work)
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
