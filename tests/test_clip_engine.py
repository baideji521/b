"""剪辑决策引擎回归测试（Phase 7 Batch 8）。

盯的是一句话：**AI 说"这里精彩"，程序必须把这段精彩准确地剪出来。**

  T1  正常 8 秒高光原样通过
  T2  AI 起点落在句中 -> 回到整句起点
  T3  AI 结束落在句中 -> 补到整句说完
  T4  AI 结束越进下一句 -> 提前到下一句开口之前
  T5  AI 给 18 秒的普通片段 -> 收进 15 秒，且落在语义边界
  T6  AI 明确标了收尾 -> 允许超过 15 秒
  T7  多个高光重叠 / 完全重复 -> 不重复出片，留分高的
  T8  没有可剪片段 -> 不启动渲染
  T9  start/end 非法（文本、缺失、NaN、布尔）-> 安全拒绝
  T10 start >= end -> 安全拒绝
  T11 视频总时长不够 -> 安全收尾 / 整段拒绝
  T12 同一输入重复执行（含打乱顺序）-> ClipPlan 完全一致
  T13 中文 reason / evaluation -> 引擎一个字都不改
  T14 渲染失败 -> 成品路径不许出现
  T15 成品封装不完整 -> 不许登记 final_video
  T16 dry-run 报告把该说的都说了（原始、最终、时长、下一段说话、调整原因）

纯算时间的部分不碰磁盘、不碰数据库；只有 T14/T15 会真的写临时文件。
可以直接 `python tests/test_clip_engine.py`，也可以 `pytest tests/test_clip_engine.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.highlight import clip_engine as engine            # noqa: E402
from vidscribe.highlight.clip_engine import Segment, Word        # noqa: E402


# ------------------------------------------------------------------ 夹具
def seg(start: float, end: float, *words: tuple[float, float, str]) -> Segment:
    """一句话。给了词就用词，没给就按整句一个词处理（模拟没有逐词的段）。"""
    items = tuple(Word(w[0], w[1], w[2]) for w in words)
    text = "".join(w.text for w in items)
    return Segment(start, end, text or "整句", items)


def spoken(start: float, end: float, count: int = 4, tag: str = "词") -> Segment:
    """把 [start, end] 均分成 count 个词，方便造"句中间"的时间点。"""
    step = (end - start) / count
    words = tuple((round(start + i * step, 3), round(start + (i + 1) * step, 3),
                   f"{tag}{i}") for i in range(count))
    return seg(start, end, *words)


def one(payload, segments, **kw):
    """跑一次引擎，断言只出一条计划并返回它。"""
    result = engine.plan_clips(payload, segments, **kw)
    assert not result.rejected, result.rejected
    assert len(result.plans) == 1, result.plans
    return result.plans[0]


# ------------------------------------------------------------------ T1
def test_normal_eight_second_clip_passes_through(tmp_path: Path) -> None:
    segments = [spoken(10.0, 18.0, 8)]
    plan = one({"video": "a.mp4", "clip": {"start": 10.0, "end": 18.0, "score": 90,
                                           "type": "challenge", "reason": "最精彩的一下"}},
               segments)
    assert (plan.start, plan.end) == (10.0, 18.0)
    assert plan.duration == 8.0
    assert plan.capped is False and plan.is_ending is False
    assert plan.source_video == "a.mp4" and plan.score == 90
    assert len(plan.words) == 8, plan.words


# ------------------------------------------------------------------ T2
def test_start_inside_a_sentence_backs_off_to_sentence_start(tmp_path: Path) -> None:
    sentence = seg(20.20, 22.60,
                   (20.20, 20.51, "hello"), (20.51, 20.92, "everyone"),
                   (20.92, 21.31, "today"), (21.31, 21.62, "we"),
                   (21.62, 21.91, "are"), (21.91, 22.30, "doing"),
                   (22.30, 22.60, "this"))
    plan = one({"clip": {"start": 21.62, "end": 22.60}}, [sentence])
    assert plan.start == 20.20, "不许从「are doing this」这种半句话开始"
    assert plan.ai_start == 21.62, "原始 AI 起点必须留档"
    assert any("回溯到整句起点" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T3
def test_end_inside_a_sentence_extends_to_sentence_end(tmp_path: Path) -> None:
    segments = [spoken(20.0, 24.2, 6)]
    plan = one({"clip": {"start": 20.0, "end": 22.5}}, segments)
    assert plan.end == 24.2, "不许把最后一句切一半"
    assert any("整句说完" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T4
def test_end_never_crosses_into_next_speech(tmp_path: Path) -> None:
    segments = [spoken(20.0, 24.2, 6), spoken(25.1, 28.0, 5)]
    plan = one({"clip": {"start": 20.0, "end": 25.5}}, segments)
    assert plan.end == 24.2, "25.50 会把下一句带进来"
    assert plan.next_speech_start == 25.1
    assert plan.end < 25.1 - 0.05
    assert any("下一段说话之前" in note for note in plan.notes), plan.notes
    assert all(word.end <= plan.end for word in plan.words), "不许带上下一句的词"


# ------------------------------------------------------------------ T5
def test_eighteen_second_clip_is_trimmed_to_fifteen(tmp_path: Path) -> None:
    segments = [spoken(0.0, 5.0, 4), spoken(5.0, 10.0, 4),
                spoken(10.0, 14.0, 4), spoken(14.0, 19.0, 4)]
    plan = one({"clip": {"start": 0.0, "end": 18.0, "type": "challenge"}}, segments)
    assert plan.duration <= 15.0, plan
    assert plan.end == 14.0, "要收在 15 秒内最后一个整句结束点，而不是硬切 15.00"
    assert plan.capped is True
    assert any("整句结束点" in note for note in plan.notes), plan.notes


def test_hard_cap_only_when_no_boundary_exists(tmp_path: Path) -> None:
    """限额内一个语义边界都没有时才允许硬截断，而且要说明白。"""
    segments = [seg(0.0, 40.0)]          # 一整句 40 秒，中间没有词边界
    plan = one({"clip": {"start": 0.0, "end": 30.0}}, segments)
    assert plan.duration == 15.0 and plan.capped is True
    assert any("硬上限截断" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T6
def test_ending_clip_may_exceed_fifteen(tmp_path: Path) -> None:
    segments = [spoken(0.0, 5.0, 4), spoken(5.0, 10.0, 4),
                spoken(10.0, 14.0, 4), spoken(14.0, 19.0, 4)]
    plan = one({"clip": {"start": 0.0, "end": 18.0, "type": "ending",
                         "reason": "收尾，把结论说完"}}, segments)
    assert plan.is_ending is True and plan.capped is False
    assert plan.duration > 15.0, "收尾片段不许被 15 秒规则砍掉"
    assert plan.end == 19.0, "收尾也要说完整句"

    # 光是"时长超了"不算收尾
    plain = one({"clip": {"start": 0.0, "end": 18.0, "reason": "很精彩"}}, segments)
    assert plain.is_ending is False and plain.duration <= 15.0


# ------------------------------------------------------------------ T7
def test_overlapping_and_duplicate_clips_are_not_rendered_twice(tmp_path: Path) -> None:
    segments = [spoken(0.0, 6.0, 4), spoken(20.0, 26.0, 4)]
    payload = {"clips": [
        {"start": 0.0, "end": 6.0, "score": 70, "reason": "低分那条"},
        {"start": 1.0, "end": 5.0, "score": 95, "reason": "高分那条"},   # 与上面重叠
        {"start": 20.0, "end": 26.0, "score": 80, "reason": "另一段"},
        {"start": 20.0, "end": 26.0, "score": 80, "reason": "另一段"},   # 完全重复
    ]}
    result = engine.plan_clips(payload, segments)
    assert len(result.plans) == 2, [(p.start, p.end, p.score) for p in result.plans]
    first, second = result.plans
    assert first.score == 95, "重叠时留评分高的那条"
    assert (second.start, second.end) == (20.0, 26.0)
    spans = {(p.start, p.end) for p in result.plans}
    assert len(spans) == 2, "同一段内容不许出两次"


# ------------------------------------------------------------------ T8
def test_no_clip_means_no_render(tmp_path: Path) -> None:
    result = engine.plan_clips({"video": "a.mp4", "evaluation": "这个视频没有高光"}, [])
    assert result.plans == () and not result
    assert any("不启动渲染" in line for line in engine.describe_result(result))


# ------------------------------------------------------------------ T9
def test_invalid_times_are_rejected(tmp_path: Path) -> None:
    segments = [spoken(0.0, 10.0, 4)]
    bad = [{"start": "很早", "end": 5.0},
           {"start": 1.0},
           {"end": 5.0},
           {"start": True, "end": 5.0},
           {"start": float("nan"), "end": 5.0},
           {"start": float("inf"), "end": 5.0},
           {"start": -2.0, "end": 5.0}]
    for clip in bad:
        result = engine.plan_clips({"clips": [clip]}, segments)
        assert result.plans == (), (clip, result.plans)
        assert len(result.rejected) == 1, clip
        assert result.rejected[0][1], "拒绝必须给出中文原因"


# ------------------------------------------------------------------ T10
def test_start_not_before_end_is_rejected(tmp_path: Path) -> None:
    segments = [spoken(0.0, 10.0, 4)]
    for clip in ({"start": 5.0, "end": 5.0}, {"start": 8.0, "end": 3.0}):
        result = engine.plan_clips({"clip": clip}, segments)
        assert result.plans == ()
        assert "必须大于" in result.rejected[0][1]


# ------------------------------------------------------------------ T11
def test_short_video_is_handled_safely(tmp_path: Path) -> None:
    segments = [spoken(0.0, 10.0, 4)]
    plan = one({"clip": {"start": 0.0, "end": 30.0, "type": "ending"}}, segments,
               video_duration=12.0)
    assert plan.end <= 12.0, plan
    assert any("超出视频时长" in note for note in plan.notes), plan.notes

    late = engine.plan_clips({"clip": {"start": 20.0, "end": 25.0}}, segments,
                             video_duration=12.0)
    assert late.plans == () and "超出视频时长" in late.rejected[0][1]


# ------------------------------------------------------------------ T12
def test_same_input_gives_same_plan(tmp_path: Path) -> None:
    segments = [spoken(0.0, 6.0, 4), spoken(20.0, 26.0, 4)]
    payload = {"clips": [{"start": 20.0, "end": 26.0, "score": 80},
                         {"start": 0.0, "end": 6.0, "score": 90}]}
    first = engine.plan_clips(payload, segments)
    second = engine.plan_clips(payload, segments)
    assert first.plans == second.plans, "同一输入必须得到完全一致的计划"

    shuffled = {"clips": list(reversed(payload["clips"]))}
    assert engine.plan_clips(shuffled, segments).plans == first.plans, "顺序不该影响结果"
    assert [p.start for p in first.plans] == sorted(p.start for p in first.plans)


# ------------------------------------------------------------------ T13
def test_chinese_reason_is_untouched(tmp_path: Path) -> None:
    reason = "挑战失败的一瞬间表情最有戏，观众会停下来看"
    evaluation = "整体节奏偏慢，但这一段情绪最强"
    segments = [spoken(3.0, 9.0, 4)]
    plan = one({"video": "b.mp4",
                "clip": {"start": 3.0, "end": 9.0, "reason": reason,
                         "evaluation": evaluation, "type": "挑战"}}, segments)
    assert plan.reason == reason, "中文理由不许被改写或翻译"
    assert plan.type == "挑战"
    payload = engine.payload_for(plan)
    assert payload["clip"]["reason"] == reason
    assert payload["clip"]["evaluation"] == evaluation
    assert payload["clip"]["start"] == plan.start and payload["clip"]["end"] == plan.end
    assert payload["video"] == "b.mp4"


# ------------------------------------------------------------------ T14
def real_mp4(path: Path, seconds: float = 2.0, size: int = 64, fps: int = 25) -> Path:
    """现场编一份最小合法 mp4（无声），给渲染/登记这两个闸门当素材。"""
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(seconds * fps)))
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        stream.options = {"crf": "30", "preset": "ultrafast"}
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((size, size, 3), (i * 7) % 256, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def half_baked(path: Path) -> Path:
    """像"写到一半"的 mp4：有 ftyp、有 mdat，没有 moov。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
                     + b"\x00\x00\x04\x00mdat" + bytes(range(256)) * 4)
    return path


def test_render_failure_leaves_no_product(tmp_path: Path) -> None:
    from vidscribe.highlight import clip as clip_mod  # noqa: PLC0415

    broken = half_baked(tmp_path / "broken.mp4")
    target = tmp_path / "out" / "broken_高光时刻.mp4"
    payload = engine.payload_for(one({"clip": {"start": 0.0, "end": 1.0}},
                                     [spoken(0.0, 1.0, 2)]))
    payload["clip"]["overlays"] = {"comment": {"time": 1.0, "text": "测试", "kind": "comment"}}
    spec = clip_mod.parse_spec(payload)
    try:
        clip_mod.render_highlight(broken, spec, target, on_log=lambda line: None)
    except Exception:                      # noqa: BLE001 - 就是要它失败
        pass
    else:
        raise AssertionError("坏源视频不该渲染成功")
    assert not target.exists(), "渲染失败绝不能留下成品文件"


# ------------------------------------------------------------------ T15
def test_incomplete_mp4_is_never_registered(tmp_path: Path) -> None:
    from vidscribe.db import importer  # noqa: PLC0415
    from vidscribe.video_io import is_complete_video  # noqa: PLC0415

    bad = half_baked(tmp_path / "bad_高光时刻.mp4")
    good = real_mp4(tmp_path / "good_高光时刻.mp4")
    part = tmp_path / "mid_高光时刻.mp4.part"
    real_mp4(part.with_suffix(""))         # 先出一份完整的
    part.write_bytes((part.with_suffix("")).read_bytes())

    assert is_complete_video(bad) is False and is_complete_video(good) is True
    assert importer._ok_to_register("final_video", bad) is False
    assert importer._ok_to_register("final_video", good) is True
    assert part.name.endswith(importer.PART_SUFFIX), ".part 由后缀过滤挡在登记之外"


# ------------------------------------------------------------------ T16
def test_dry_run_report_explains_everything(tmp_path: Path) -> None:
    segments = [spoken(20.0, 24.2, 6), spoken(25.1, 28.0, 5)]
    result = engine.plan_clips({"video": "c.mp4",
                                "clip": {"start": 21.0, "end": 25.5, "score": 88,
                                         "type": "hook", "reason": "开头就抓人"}}, segments)
    text = "\n".join(engine.describe_result(result))
    for must in ("AI区间", "修正后", "时长", "下一段说话起点", "用到", "AI 理由", "调整"):
        assert must in text, (must, text)
    assert "21.00 → 25.50" in text and "20.00 → 24.20" in text, text
    assert text.count("[剪辑引擎]") >= 6


# ------------------------------------------------------------------ T17
def _call_names(func: str) -> list[str]:
    """按源码顺序列出某个函数体里调用到的名字（ast.walk 是广度优先，必须自己排序）。"""
    import ast

    source = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HighlightWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func:
                    target = item
    assert target is not None, "HighlightWorker.%s 不见了" % func
    found = []
    for node in ast.walk(target):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if name:
                found.append((node.lineno, node.col_offset, name))
    found.sort()
    return [name for _, _, name in found]


def test_gui_renders_through_the_engine(tmp_path: Path) -> None:
    names = _call_names("run")
    for must in ("_plan_result", "payload_for", "render_highlight", "is_complete_video"):
        assert must in names, (must, names)
    assert names.index("_plan_result") < names.index("render_highlight"), \
        "GUI 必须先跑引擎再渲染"
    assert names.index("payload_for") < names.index("render_highlight"), \
        "渲染用的必须是引擎修正后的区间"
    assert names.index("render_highlight") < names.index("is_complete_video"), \
        "渲染完要用 is_complete_video 校验成片"
    # 引擎那一步不能被跳过：没有可剪片段时抛错，不允许拿 AI 原区间硬剪
    plan_names = _call_names("_plan_result")
    assert "plan_clips" in plan_names and "segments_for_video" in plan_names, plan_names
    assert "describe_result" in plan_names, "GUI 也要打中文的引擎日志"


# ------------------------------------------------------------------ T18
def test_multi_clip_payload_can_still_locate_the_video(tmp_path: Path) -> None:
    """`clips: [...]` 这种多条写法：既有 parse_spec 认不了，得能折成第一条来定位视频。"""
    from vidscribe.highlight import parse_spec

    overlays = {"comment": {"time": 3.0, "text": "wow", "kind": "comment"}}
    payload = {"video": "m.mp4",
               "clips": [{"start": 2.0, "end": 6.0, "reason": "第一条", "overlays": overlays},
                         {"start": 9.0, "end": 12.0, "reason": "第二条", "overlays": overlays}]}
    try:
        parse_spec(payload)
        raise AssertionError("多条写法本来就该被 parse_spec 拒掉")
    except ValueError:
        pass

    alt = engine.first_clip_payload(payload)
    spec = parse_spec(alt)
    assert spec.video_name == "m.mp4" and spec.clip_start == 2.0, (alt, spec)
    assert alt["clip"]["reason"] == "第一条", "取第一条，文案原样"

    segments = [spoken(2.0, 6.0, 5), spoken(9.0, 12.0, 4)]
    result = engine.plan_clips(payload, segments)
    assert len(result.plans) == 2, [(p.start, p.end) for p in result.plans]
    assert [p.source_video for p in result.plans] == ["m.mp4", "m.mp4"]
    assert engine.first_clip_payload({"clips": []}) is None


# ------------------------------------------------------------------ 直接跑
TESTS = (

    test_normal_eight_second_clip_passes_through,
    test_start_inside_a_sentence_backs_off_to_sentence_start,
    test_end_inside_a_sentence_extends_to_sentence_end,
    test_end_never_crosses_into_next_speech,
    test_eighteen_second_clip_is_trimmed_to_fifteen,
    test_hard_cap_only_when_no_boundary_exists,
    test_ending_clip_may_exceed_fifteen,
    test_overlapping_and_duplicate_clips_are_not_rendered_twice,
    test_no_clip_means_no_render,
    test_invalid_times_are_rejected,
    test_start_not_before_end_is_rejected,
    test_short_video_is_handled_safely,
    test_same_input_gives_same_plan,
    test_chinese_reason_is_untouched,
    test_render_failure_leaves_no_product,
    test_incomplete_mp4_is_never_registered,
    test_dry_run_report_explains_everything,
    test_gui_renders_through_the_engine,
    test_multi_clip_payload_can_still_locate_the_video,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="clipeng_"))
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
