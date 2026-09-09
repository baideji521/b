"""剪辑决策引擎回归测试（Phase 7 Batch 8）。

盯的是一句话：**AI 说"这里精彩"，程序必须把这段精彩准确地剪出来。**

  T1  正常 8 秒高光原样通过
  T2  AI 起点落在句中 -> 回到整句起点
  T3  AI 结束落在句中 -> 补到整句说完
  T4  AI 结束越进下一句 -> 提前到下一句开口之前
  T5  AI 给 18 秒的普通片段 -> 不砍时长，按语义边界收（时长要求写在 PRM 里）
  T6  收尾片段和普通片段一个待遇 -> 都不按时长砍
  T7  segments 给了多段 -> 只剪第一段，并在 notes 里写明「多段拼接还没做」
  T8  没有可剪片段 -> 不启动渲染
  T9  sa/end 非法（文本、缺失、NaN、布尔）-> 一段都不出，validate 给中文原因
  T10 end <= sa -> 一段都不出，validate 给中文原因
  T11 视频总时长不够 -> 安全收尾 / 整段拒绝
  T12 同一输入重复执行 -> ClipPlan 完全一致；segments 顺序决定剪哪一段
  T13 中文 reason / t 描述块 -> 引擎一个字都不改
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

from vidscribe import ai_protocol                              # noqa: E402
from vidscribe.highlight import clip_engine as engine            # noqa: E402
from vidscribe.highlight import from_moments                     # noqa: E402
from vidscribe.highlight.clip_engine import Segment, Word        # noqa: E402

#: `hl(segments=...)` 的"没传"标记（None 表示"顶层连 segments 这个键都不要"）
_MISSING = object()



# ------------------------------------------------------------------ 夹具
def hl(sa: float | None = None, end: float | None = None, *, duration: float | None = None,
       video: str = "", score: float | None = None, kind: str = "", reason: str = "",
       more: tuple[tuple[float, float], ...] = (), segments: object = _MISSING,
       **extra: object) -> dict:
    """造一份**新协议**的高光 JSON（唯一认的写法，见 src/vidscribe/ai_protocol.py）。

    `sa` / `end` 是原视频时间，落在 `segments[0]`；`duration` 是成片时长，落在
    `timeline.duration`，`dst` 跟着写成 `[0.0, duration]`。
    `more` 里多给的段用来验证「多段拼接还没做，只剪第一段」；
    `segments` 可以整块换掉（传非法值 / 空列表 / 直接不给），用来测协议校验。
    `extra` 原样合到顶层（`o` / `s` / `w` / `a` / `e` / `t` 这些成片时间域的块）。
    """
    out: dict = {}
    if video:
        out["video"] = video
    span = None
    if duration is not None:
        span = round(duration, 3)
    elif sa is not None and end is not None:
        span = round(end - sa, 3)
    timeline: dict = {}
    if span is not None:
        timeline["duration"] = span
    if score is not None:
        timeline["score"] = score
    if kind:
        timeline["type"] = kind
    if reason:
        timeline["reason"] = reason
    out["timeline"] = timeline
    if segments is not _MISSING:
        if segments is not None:
            out["segments"] = segments
    else:
        first: dict = {"sa": sa, "end": end}
        if span is not None:
            first["dst"] = [0.0, span]
        out["segments"] = [first] + [
            {"sa": s, "end": e, "dst": [0.0, round(e - s, 3)]} for s, e in more]
    out.update(extra)
    return out


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
    plan = one(hl(10.0, 18.0, video="a.mp4", score=90, kind="challenge",
                  reason="最精彩的一下"), segments)
    assert (plan.start, plan.end) == (10.0, 18.0)
    assert plan.duration == 8.0
    assert plan.source_video == "a.mp4" and plan.score == 90
    assert len(plan.words) == 8, plan.words


# ------------------------------------------------------------------ T2
def test_start_inside_a_sentence_backs_off_to_sentence_start(tmp_path: Path) -> None:
    sentence = seg(20.20, 22.60,
                   (20.20, 20.51, "hello"), (20.51, 20.92, "everyone"),
                   (20.92, 21.31, "today"), (21.31, 21.62, "we"),
                   (21.62, 21.91, "are"), (21.91, 22.30, "doing"),
                   (22.30, 22.60, "this"))
    plan = one(hl(21.62, 22.60), [sentence])
    assert plan.start == 20.20, "不许从「are doing this」这种半句话开始"
    assert plan.ai_start == 21.62, "原始 AI 起点必须留档"
    assert any("回溯到整句起点" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T3
def test_end_inside_a_sentence_extends_to_sentence_end(tmp_path: Path) -> None:
    segments = [spoken(20.0, 24.2, 6)]
    plan = one(hl(20.0, 22.5), segments)
    assert plan.end == 24.2, "不许把最后一句切一半"
    assert any("整句说完" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T4
def test_end_never_crosses_into_next_speech(tmp_path: Path) -> None:
    segments = [spoken(20.0, 24.2, 6), spoken(25.1, 28.0, 5)]
    plan = one(hl(20.0, 25.5), segments)
    assert plan.end == 24.2, "25.50 会把下一句带进来"
    assert plan.next_speech_start == 25.1
    assert plan.end < 25.1 - 0.05
    assert any("下一段说话之前" in note for note in plan.notes), plan.notes
    assert all(word.end <= plan.end for word in plan.words), "不许带上下一句的词"


# ------------------------------------------------------------------ T5
def test_long_clip_is_not_cut_by_any_duration_rule(tmp_path: Path) -> None:
    """18 秒的普通片段不许被砍：多长是 PRM 里对 AI 提的要求，代码不再插手。"""
    segments = [spoken(0.0, 5.0, 4), spoken(5.0, 10.0, 4),
                spoken(10.0, 14.0, 4), spoken(14.0, 19.0, 4)]
    plan = one(hl(0.0, 18.0, kind="challenge"), segments)
    assert plan.duration > 15.0, f"不许再按 15 秒砍：{plan}"
    assert plan.end == 19.0, "AI 的 18.00 落在最后一句里，补到整句说完 19.00"
    assert not any("上限" in note for note in plan.notes), plan.notes


def test_one_long_sentence_keeps_its_own_boundary(tmp_path: Path) -> None:
    """一整句 40 秒也不硬切：只按语义边界收，不再有「找不到边界就截断」这条路。"""
    segments = [seg(0.0, 40.0)]          # 一整句 40 秒，中间没有词边界
    plan = one(hl(0.0, 30.0), segments)
    assert plan.duration == 40.0, f"这一句说完是 40.00：{plan}"
    assert any("整句说完" in note for note in plan.notes), plan.notes


# ------------------------------------------------------------------ T6
def test_ending_clip_is_treated_like_any_other(tmp_path: Path) -> None:
    """收尾片段不再需要特殊照顾：没有上限，谁都不砍，待遇一模一样。"""
    segments = [spoken(0.0, 5.0, 4), spoken(5.0, 10.0, 4),
                spoken(10.0, 14.0, 4), spoken(14.0, 19.0, 4)]
    ending = one(hl(0.0, 18.0, kind="ending", reason="收尾，把结论说完"), segments)
    plain = one(hl(0.0, 18.0, reason="很精彩"), segments)
    assert ending.end == plain.end == 19.0, (ending, plain)
    assert ending.duration == plain.duration, "标不标收尾都是同一个区间"


# ------------------------------------------------------------------ T7
def test_extra_segments_are_dropped_with_a_warning(tmp_path: Path) -> None:
    """新协议一份 JSON 只剪 `segments[0]`：多给的段丢掉，但必须在 notes 里说清楚。"""
    segments = [spoken(0.0, 6.0, 4), spoken(20.0, 26.0, 4)]
    payload = hl(0.0, 6.0, score=70, reason="第一段", more=((20.0, 26.0), (30.0, 33.0)))
    result = engine.plan_clips(payload, segments)
    assert not result.rejected, result.rejected
    assert len(result.plans) == 1, [(p.start, p.end) for p in result.plans]
    plan = result.plans[0]
    assert (plan.start, plan.end) == (0.0, 6.0), "只认第一段，后面的 20→26 / 30→33 都不剪"
    assert plan.reason == "第一段" and plan.score == 70
    assert any("多段拼接还没做" in note and "这次只剪第一段" in note for note in plan.notes), \
        plan.notes
    assert any("3 段 segments" in note for note in plan.notes), \
        "警告里要报清楚 JSON 一共给了几段：%s" % (plan.notes,)
    # 只有一段时不许乱报警告
    single = one(hl(0.0, 6.0, score=70), segments)
    assert not any("只剪第一段" in note for note in single.notes), single.notes
    # 丢了几段也从协议层直接看得出来
    assert ai_protocol.clips(payload)[0]["dropped_segments"] == 2
    assert len(ai_protocol.raw_segments(payload)) == 3



# ------------------------------------------------------------------ T8
def test_no_clip_means_no_render(tmp_path: Path) -> None:
    payload = hl(video="a.mp4", segments=None, t={"Speech text": "这个视频没有高光"})
    result = engine.plan_clips(payload, [])
    assert result.plans == () and not result
    assert any("不启动渲染" in line for line in engine.describe_result(result))
    why = ai_protocol.validate(payload)
    assert why and "缺少 segments" in why, why


# ------------------------------------------------------------------ T9
def test_invalid_times_are_rejected(tmp_path: Path) -> None:
    """sa / end 写坏了：一段都不许出，并且协议层能给出中文原因。"""
    segments = [spoken(0.0, 10.0, 4)]
    bad = [
        hl(segments=None),                                   # 连 segments 都没有
        hl(segments=[]),                                     # 空 segments
        hl(segments=[{"sa": "很早", "end": 5.0}]),            # sa 不是数字
        hl(segments=[{"sa": 1.0}]),                          # 缺 end
        hl(segments=[{"end": 5.0}]),                         # 缺 sa
        hl(segments=[{"sa": True, "end": 5.0}]),             # 布尔不算数字
        hl(segments=[{"sa": float("nan"), "end": 5.0}]),     # NaN
        hl(segments=[{"sa": float("inf"), "end": 5.0}]),     # inf
    ]
    for payload in bad:
        result = engine.plan_clips(payload, segments)
        assert result.plans == (), (payload, result.plans)
        why = ai_protocol.validate(payload)
        assert why, "非法协议必须给出中文原因：%s" % (payload,)
        assert "segments" in why, why
        assert any("不启动渲染" in line for line in engine.describe_result(result))

    # sa 是负数：区间本身合法，所以能进引擎，由引擎明确拒掉并给中文原因
    negative = hl(-2.0, 5.0)
    assert ai_protocol.validate(negative) is None, "负数区间在协议层算「写法合法」"
    result = engine.plan_clips(negative, segments)
    assert result.plans == ()
    assert len(result.rejected) == 1, result.rejected
    assert "时间不能是负数" in result.rejected[0][1], result.rejected


# ------------------------------------------------------------------ T10
def test_start_not_before_end_is_rejected(tmp_path: Path) -> None:
    """end <= sa 在协议层就不成立：既剪不出片段，parse_spec 也要抛中文错。"""
    from vidscribe.highlight import clip as clip_mod  # noqa: PLC0415

    segments = [spoken(0.0, 10.0, 4)]
    for payload in (hl(5.0, 5.0), hl(8.0, 3.0)):
        result = engine.plan_clips(payload, segments)
        assert result.plans == (), result
        # 剪不出来必须给中文原因，不能只剩一句「没有可剪的片段」
        assert len(result.rejected) == 1, result.rejected
        assert "不是有效区间" in result.rejected[0][1], result.rejected
        why = ai_protocol.validate(payload)
        assert why and "不是有效区间" in why, why
        try:
            clip_mod.parse_spec(payload)
        except ValueError as exc:
            assert "不是有效区间" in str(exc), exc
        else:
            raise AssertionError("end <= sa 必须被 parse_spec 拒掉")



# ------------------------------------------------------------------ T11
def test_short_video_is_handled_safely(tmp_path: Path) -> None:
    segments = [spoken(0.0, 10.0, 4)]
    plan = one(hl(0.0, 30.0, kind="ending"), segments, video_duration=12.0)
    assert plan.end <= 12.0, plan
    assert any("超出视频时长" in note for note in plan.notes), plan.notes

    late = engine.plan_clips(hl(20.0, 25.0), segments, video_duration=12.0)
    assert late.plans == () and "超出视频时长" in late.rejected[0][1]


# ------------------------------------------------------------------ T12
def test_same_input_gives_same_plan(tmp_path: Path) -> None:
    segments = [spoken(0.0, 6.0, 4), spoken(20.0, 26.0, 4)]
    payload = hl(0.0, 6.0, score=90, more=((20.0, 26.0),))
    first = engine.plan_clips(payload, segments)
    second = engine.plan_clips(payload, segments)
    assert first.plans == second.plans, "同一输入必须得到完全一致的计划"
    assert len(first.plans) == 1 and (first.plans[0].start, first.plans[0].end) == (0.0, 6.0)

    # 新协议里"第一段"是位置决定的：把两段换个位置，剪的就是另一段
    swapped = hl(20.0, 26.0, score=90, more=((0.0, 6.0),))
    other = engine.plan_clips(swapped, segments)
    assert len(other.plans) == 1
    assert (other.plans[0].start, other.plans[0].end) == (20.0, 26.0), \
        "segments[0] 换了，剪的那一段就得跟着换"
    assert engine.plan_clips(swapped, segments).plans == other.plans, "重跑仍要完全一致"


# ------------------------------------------------------------------ T13
def test_chinese_reason_is_untouched(tmp_path: Path) -> None:
    reason = "挑战失败的一瞬间表情最有戏，观众会停下来看"
    trim = {"Scene": "室内房间", "Action": "查看结果并绝望吐槽",
            "Speech text": "整体节奏偏慢，但这一段情绪最强"}
    overlays = [[0.0, "word", "2087?"], [2.79, "comment", "真的假的"], [5.5, "emoji", "💀"]]
    segments = [spoken(3.0, 9.0, 4)]
    plan = one(hl(3.0, 9.0, video="b.mp4", reason=reason, kind="挑战",
                  t=trim, o=overlays), segments)
    assert plan.reason == reason, "中文理由不许被改写或翻译"
    assert plan.type == "挑战"
    payload = engine.payload_for(plan)
    assert payload["timeline"]["reason"] == reason
    assert payload["timeline"]["type"] == "挑战"
    assert payload["t"] == trim, "t 那块描述原样带走"
    assert payload["o"] == overlays, "o 那些成片时间域的贴纸原样带走"
    assert payload["segments"][0]["sa"] == plan.start
    assert payload["segments"][0]["end"] == plan.end
    assert payload["segments"][0]["dst"] == [0.0, payload["timeline"]["duration"]], \
        "dst 按成片重算成 [0, duration]，不写末帧冻结"
    assert payload["video"] == "b.mp4"
    assert "clip" not in payload, "老协议的 clip 块一个都不许再出现"



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
    payload = engine.payload_for(one(hl(0.0, 1.0), [spoken(0.0, 1.0, 2)]))
    payload["o"] = [[0.5, "comment", "测试"]]
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
    result = engine.plan_clips(hl(21.0, 25.5, video="c.mp4", score=88,
                                  kind="hook", reason="开头就抓人"), segments)
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
def test_multi_segment_payload_folds_down_to_the_first(tmp_path: Path) -> None:
    """多段 segments：定位视频照旧，但只剪第一段；老协议会被自动升级成新形状。"""
    from vidscribe import ai_protocol
    from vidscribe.highlight import parse_spec

    overlays = [[3.0, "comment", "wow"]]
    payload = hl(2.0, 6.0, video="m.mp4", reason="第一条",
                 more=((9.0, 12.0),), o=overlays)

    # 老协议不再判死，而是由 `ai_protocol.payload_of` 升级成新形状：clip.start / clip.end
    # 落进 segments[0].sa / .end，所以下游只有一套形状要管
    old = {"video": "m.mp4", "clip": {"start": 2.0, "end": 6.0}}
    upgraded = ai_protocol.payload_of(old)
    assert upgraded["segments"][0]["sa"] == 2.0 and upgraded["segments"][0]["end"] == 6.0
    legacy = parse_spec(old)
    assert (legacy.clip_start, legacy.clip_end) == (2.0, 6.0), "升级过的老 JSON 照旧能剪"
    # 真正判死的是「连 segments 都抠不出来」的 JSON
    try:
        parse_spec({"video": "m.mp4"})
        raise AssertionError("抠不出 segments 的 JSON 必须被拒掉")
    except ValueError as exc:
        assert "segments" in str(exc), exc


    spec = parse_spec(payload)
    assert spec.video_name == "m.mp4" and spec.clip_start == 2.0, (payload, spec)
    assert spec.clip_end == 6.0, "多余的 9→12 不许参与定位"

    alt = engine.first_clip_payload(payload)
    assert len(alt["segments"]) == 1, "折下来的那份只留第一段：%s" % (alt,)
    assert alt["segments"][0]["sa"] == 2.0 and alt["segments"][0]["end"] == 6.0
    assert alt["timeline"]["reason"] == "第一条", "取第一段，文案原样"
    assert alt["o"] == overlays
    assert parse_spec(alt).clip_start == 2.0

    segments = [spoken(2.0, 6.0, 5), spoken(9.0, 12.0, 4)]
    result = engine.plan_clips(payload, segments)
    assert len(result.plans) == 1, [(p.start, p.end) for p in result.plans]
    assert result.plans[0].source_video == "m.mp4"
    assert any("只剪第一段" in note for note in result.plans[0].notes), result.plans[0].notes
    assert engine.first_clip_payload(hl(segments=[])) is None
    folded = engine.first_clip_payload({"clip": {"start": 1.0, "end": 2.0}})
    assert folded is not None and folded["segments"][0]["sa"] == 1.0, \
        "老协议先升级再折：折出来的是新形状的第一段"
    assert engine.first_clip_payload({"video": "m.mp4"}) is None, \
        "连区间都没有的 JSON 折不出片段"



# ------------------------------------------------------------------ T20
def test_tracks_are_retimed_to_the_real_cut(tmp_path: Path) -> None:
    """加减秒数一改，成片时长和 o/s/w/a/e 的时间都得跟着重算，越界的裁掉。"""
    payload = hl(48.55, 55.37, duration=6.82,
                 o=[[0.0, "word", "1000?"], [2.66, "emoji", "X"]],
                 s=[[0.0, 2.4, "第一句"], [2.6, 6.82, "第二句"]],
                 w=[[0.0, 0.4, "第"], [6.5, 6.82, "词"]],
                 a=[[0.0, 6.82, "展示", "室内"]],
                 e=[[0.0, 1.45, "neutral", 0.43]])
    clip = ai_protocol.clips(payload)[0]

    # 起始 -0.55（提前起剪）、结束 +2.0（多留一点）：播放段 = (55.37+2) - (48.55-0.55)
    play = ai_protocol.play_seconds(clip, startframe=-0.55, freeze=2.0)
    assert play == 9.37, play
    # 成片总长 = 播放段 + 末帧冻结 2 秒
    assert ai_protocol.final_duration(play, freeze_tail=2.0) == 11.37
    # 不冻帧就只有播放段
    assert ai_protocol.final_duration(play) == 9.37

    tracks = ai_protocol.retime_tracks(clip, startframe=-0.55, freeze=2.0, span=play)
    # 提前 0.55 秒起剪 → 所有事件整体后移 0.55 秒；文字一个字不改
    assert tracks["o"] == [[0.55, "word", "1000?"], [3.21, "emoji", "X"]], tracks["o"]
    assert tracks["s"] == [[0.55, 2.95, "第一句"], [3.15, 7.37, "第二句"]], tracks["s"]
    assert tracks["w"] == [[0.55, 0.95, "第"], [7.05, 7.37, "词"]], tracks["w"]
    assert tracks["a"] == [[0.55, 7.37, "展示", "室内"]], tracks["a"]
    assert tracks["e"] == [[0.55, 2.0, "neutral", 0.43]], tracks["e"]

    # 结束往回收 5 秒：落到播放段外面的记录要被裁掉 / 丢掉
    short = ai_protocol.play_seconds(clip, startframe=0.0, freeze=-5.0)
    assert short == 1.82, short
    cut = ai_protocol.retime_tracks(clip, startframe=0.0, freeze=-5.0, span=short)
    assert cut["o"] == [[0.0, "word", "1000?"]], "2.66 秒那条已经在成片外面了"
    assert cut["s"] == [[0.0, 1.82, "第一句"]], cut["s"]
    assert cut["w"] == [[0.0, 0.4, "第"]], cut["w"]


# ------------------------------------------------------------------ T21
def test_silent_gaps_are_measured_not_guessed(tmp_path: Path) -> None:
    """原声空隙：程序按逐词算，成片坐标从 0 起，短于 0.5 秒的不报，含首尾。"""
    # 10 → 20 这段素材：10.0-11.0 静音、11.0-12.0 说话、12.0-14.5 静音、
    # 14.5-15.0 说话、15.0-15.5 静音（正好够 0.5，塞得进一句短旁白）、
    # 15.5-16.0 说话、16.0-20.0 静音
    segments = [seg(11.0, 12.0, (11.0, 11.5, "你"), (11.5, 12.0, "好")),
                seg(14.5, 15.0, (14.5, 15.0, "嗯")),
                seg(15.5, 16.0, (15.5, 16.0, "对"))]
    gaps = engine.silent_gaps(segments, 10.0, 20.0)
    assert gaps == [[0.0, 1.0], [2.0, 4.5], [5.0, 5.5], [6.0, 10.0]], gaps

    # 越出素材范围的语音要裁进来，不许把整句时长算成占用
    assert engine.silent_gaps([seg(5.0, 11.0, (5.0, 11.0, "长"))], 10.0, 20.0) \
        == [[1.0, 10.0]]
    # 全程满语音 → 空数组（调用方据此写 "gaps": []，第二轮就知道这条不插配音）
    assert engine.silent_gaps([seg(10.0, 20.0, (10.0, 20.0, "满"))], 10.0, 20.0) == []
    # 没有逐词的句子退回整句区间：宁可少报一个空隙，也不谎报静音
    assert engine.silent_gaps([Segment(12.0, 18.0, "没有逐词")], 10.0, 20.0) \
        == [[0.0, 2.0], [8.0, 10.0]]
    # 没有语音数据 / 区间非法 → 空数组，不猜
    assert engine.silent_gaps((), 10.0, 20.0) == [[0.0, 10.0]], "没句子就是整段静音"
    assert engine.silent_gaps(segments, 20.0, 10.0) == []
    # 阈值可调，但默认就是 0.5（第二轮插一句最短旁白的下限）：
    # 0.4 秒那种缝只有放宽阈值才会出现
    tight = [seg(11.0, 12.0, (11.0, 12.0, "话")), seg(12.4, 13.0, (12.4, 13.0, "接"))]
    assert engine.silent_gaps(tight, 11.0, 13.0) == [], "0.4 秒的缝默认不报"
    loose = engine.silent_gaps(tight, 11.0, 13.0, floor=0.3)
    assert [1.0, 1.4] in loose, loose


def test_moment_list_survives_chat_noise(tmp_path: Path) -> None:
    """结果清单 → 方案：``` 围栏和 `[cite: 1]` 不许吃掉整行，脏行也得进来。"""
    # 三句话：铺垫 / 结果 / 下一件事，中间留静音
    segments = [spoken(10.0, 12.0, 4, "铺垫"), spoken(13.0, 15.0, 4, "结果"),
                spoken(30.0, 32.0, 4, "别的")]
    text = ('```json\n'
            '{"setup_at": 10.5, "result_at": 14.0, "score": 90, "word": "冲击"}'
            ' [cite: 1, 2]\n'
            '{"setup_at": 30.2, "result_at": 31.5, "score": 80, "word": "第二条"}\n'
            '```\n')
    rows = from_moments.rows_from_text(text)
    assert len(rows) == 2, f"围栏/引用标记吃掉了行：{rows}"
    assert rows[0]["word"] == "冲击" and rows[1]["word"] == "第二条"

    made, logs = from_moments.payloads_from_rows(segments, rows, video_name="a.mp4",
                                                min_sec=1.0)
    assert len(made) == 2, logs
    payload, speech, row = made[0]
    clip = payload["clip"]
    # 区间落在句边界上，不是 AI 给的那两个点
    assert (clip["start"], clip["end"]) == (10.0, 15.0), clip
    # Speech text 一律留空等翻译；源语言原文单独带出来
    assert clip["Trimclip"]["Speech text"] == ""
    assert speech, "区间里明明有语音，原文不该是空的"
    # 清单原行原样带回来 —— 入库时它就是 raw_json（AI 原话）
    assert row["setup_at"] == 10.5 and row["result_at"] == 14.0
    # 挂字：word 钉在结果时刻，comment 往前错开，都不越界；
    # 没给 emoji 就不造空轨（emoji 现在写在 word 里面）
    assert "emoji" not in clip["overlays"], clip["overlays"]
    word_at = clip["overlays"]["word"]["time"]
    assert clip["start"] <= word_at <= clip["end"], clip["overlays"]

    # word 里带 emoji 的写法原样存着，不拆
    merged = from_moments.payload_from_moment(
        segments, {"setup_at": 10.5, "result_at": 14.0, "word": "WHAT?! 🐶"},
        video_name="a.mp4", min_sec=1.0)
    assert merged is not None
    assert merged[0]["clip"]["overlays"]["word"]["text"] == "WHAT?! 🐶"
    # 老清单单独给 emoji 的写法照旧认
    old = from_moments.payload_from_moment(
        segments, {"setup_at": 10.5, "result_at": 14.0, "word": "WHAT?!", "emoji": "🐶"},
        video_name="a.mp4", min_sec=1.0)
    assert old is not None
    assert old[0]["clip"]["overlays"]["emoji"]["text"] == "🐶"


    # 同一件事被指两遍 → 后来者按区间重叠丢掉，不出两份重复方案
    twice, _ = from_moments.payloads_from_rows(segments, [rows[0], dict(rows[0])],
                                               video_name="a.mp4", min_sec=1.0)
    assert len(twice) == 1

    # 一行都解不出来时返回空列表，调用方据此报错而不是入库一份空方案
    assert from_moments.rows_from_text("这只是一段说明文字，没有 JSON") == []


def test_silence_does_not_eat_the_length_budget(tmp_path: Path) -> None:
    """时长考核看发声、不看跨度：中间干等再久，整件事也要完整留下来。"""
    # 条件句发声 3 秒，中间空 8 秒（等结果），结果句 0.5 秒 —— 跨度 11.5s，发声 3.5s
    setup = seg(10.0, 13.0, (10.0, 11.5, "如果"), (11.5, 13.0, "是绿的"))
    result = seg(21.0, 21.5, (21.0, 21.5, "黄的"))
    segments = [setup, result]
    assert engine.voiced_between(segments, 10.0, 21.5) == 3.5

    start, end, notes = engine.plan_from_span(segments, 10.0, 21.0)
    # 按跨度考核（旧口径 max_sec=10）这里会把条件句砍掉，起点变成 21.0
    assert (start, end) == (10.0, 21.5), (start, end, notes)
    assert not any("去掉最前面" in one for one in notes), notes

    # 跨度硬上限仍然拦得住极端：发声一样少，但中间空到 30 秒
    far = seg(45.0, 45.5, (45.0, 45.5, "黄的"))
    start2, end2, notes2 = engine.plan_from_span([setup, far], 10.0, 45.0)
    assert start2 == 45.0, (start2, end2, notes2)
    assert any("去掉最前面" in one for one in notes2), notes2

    # 发声超上限才砍：三句各发声 4 秒，连起来 12 秒 > 10
    talky = [spoken(0.0, 4.0, 4, "甲"), spoken(4.5, 8.5, 4, "乙"), spoken(9.0, 13.0, 4, "丙")]
    start3, _, notes3 = engine.plan_from_span(talky, 0.0, 9.0, max_voiced=10.0)
    assert start3 > 0.0, notes3
    assert any("发声" in one for one in notes3), notes3


def test_scores_stay_separate_and_metrics_are_computed(tmp_path: Path) -> None:
    """四维评分不合成、越界钳回 0-5；程序不代算任何评分维度。"""
    segments = [spoken(10.0, 12.0, 4, "铺垫"), spoken(13.0, 15.0, 4, "结果")]
    row = {"setup_at": 10.0, "result_at": 13.0, "surprise": 9, "standalone": -2,
           "emotion_power": 4, "caption": 2.6}
    made = from_moments.payload_from_moment(segments, row, video_name="a.mp4", min_sec=1.0)
    assert made is not None
    clip = made[0]["clip"]
    # 钳进 0-5，四维各自独立留着（没有任何合成分）
    assert clip["scores"] == {"surprise": 5, "standalone": 0,
                              "emotion_power": 4, "caption": 3}, clip["scores"]
    # 没给老写法的 score 就一个字都不写：`score: 0` 会被下游当成"判了 0 分"
    assert "score" not in clip
    # 程序不往产物里塞自己算的指标：客观数字只摆在剧本里给 AI 看
    assert "metrics" not in clip
    assert not hasattr(from_moments, "metrics_of")

    # 一维都没给也不报错，只是 scores 为空；老写法的 score 照旧留着
    plain = from_moments.payload_from_moment(segments, {"setup_at": 10.0, "result_at": 13.0,
                                                       "score": 88},
                                            video_name="a.mp4", min_sec=1.0)
    assert plain is not None and plain[0]["clip"]["scores"] == {}
    assert plain[0]["clip"]["score"] == 88
    # 升级到新协议后 scores 挂在 timeline 上，提取数据那一步才拿得到
    up = ai_protocol.payload_of(made[0])
    assert up["timeline"]["scores"] == clip["scores"]
    assert "metrics" not in up["timeline"]


def test_quiet_video_pulls_span_back_from_result_at(work: Path) -> None:
    """整段几乎没语音的视频：不吸句边界，从 result_at 反推，语音原文留住。

    实测那条：视频 21.73s，全片只识别出一句 14.84-15.20「Let us go home!」，
    AI 给 setup_at=10 / result_at=14.84。吸句边界只能算出 0.36s，够不上 3s，
    这一行整条被丢掉，这个视频永远出不来成品。
    """
    quiet = [seg(14.84, 15.2, (14.84, 15.2, "Let us go home!"))]
    row = {"setup_at": 10.0, "result_at": 14.84, "word": "GO HOME"}
    made = from_moments.payload_from_moment(quiet, row, video_name="a.mp4", duration=21.73)
    assert made is not None, "几乎没语音的视频也得算得出区间"
    clip = made[0]["clip"]
    # 起点就是 AI 给的 setup_at（不吸句边界），终点延到结果点那句话说完
    assert (clip["start"], clip["end"]) == (10.0, 15.2), clip
    # 区间内的语音原文留住：入库当方案备注，翻译填 Speech text 也靠它
    assert "Let us go home!" in made[1], made[1]

    # 两点之间不足 3 秒：从终点往前反推补足，终点不越过片子结尾
    short = from_moments.payload_from_moment(
        [], {"setup_at": 4.0, "result_at": 4.5}, video_name="a.mp4", duration=4.6)
    assert short is not None
    assert (short[0]["clip"]["start"], short[0]["clip"]["end"]) == (1.5, 4.5), short[0]["clip"]

    # 有话说的视频一个字都不变：照旧吸句边界，不走反推那条
    talky = [spoken(10.0, 12.0, 4, "铺垫"), spoken(13.0, 15.0, 4, "结果")]
    normal = from_moments.payload_from_moment(talky, {"setup_at": 10.0, "result_at": 13.0},
                                              video_name="a.mp4", duration=30.0)
    assert normal is not None
    assert not any("不吸句边界" in one for one in normal[2]), normal[2]
    assert normal[0]["clip"]["end"] - normal[0]["clip"]["start"] >= 3.0, normal[0]["clip"]




def test_target_trims_only_as_much_as_needed(work: Path) -> None:
    """成品时长上限：没超一刀不剪，超了只剪到刚好压进来，压不进去就剪到最紧。"""
    # 两句话各 2 秒 / 3 秒，中间空 6 秒：跨度 11 秒，发声 5 秒
    segments = (spoken(0.0, 2.0), spoken(8.0, 11.0))

    # 一、没超上限 -> 一刀不剪
    assert engine.trim_to_target(segments, 0.0, 11.0, target=12.0,
                                 min_keep=0.3) == [(0.0, 11.0)]

    # 二、超了 -> 剪到刚好压进来，而且**静音尽量多留**
    spans = engine.trim_to_target(segments, 0.0, 11.0, target=9.0, min_keep=0.3)
    total = round(sum(hi - lo for lo, hi in spans), 2)
    assert len(spans) == 2, spans
    assert total <= 9.0 + 0.01, "压不进目标：%s -> %.2f" % (spans, total)
    assert total >= 8.5, "剪过头了，静音本该尽量留：%s -> %.2f" % (spans, total)

    # 三、目标比发声还短 -> 怎么剪都压不进去，这时按 min_keep 剪到最紧。
    # **不许退回更松的档**：那样「目标越紧、成品反而越长」，时长就不单调了
    hard = engine.trim_to_target(segments, 0.0, 11.0, target=4.0, min_keep=0.3)
    tight = engine.trim_plan(segments, 0.0, 11.0, keep=0.3)
    assert hard == tight, "压不进去就该剪到最紧：%s vs %s" % (hard, tight)
    loose = engine.trim_to_target(segments, 0.0, 11.0, target=9.0, min_keep=0.3)
    assert sum(hi - lo for lo, hi in hard) <= sum(hi - lo for lo, hi in loose) + 0.01, \
        "目标越紧，成品不该反而越长"

    # 四、统一入口：给了 target 走目标模式，没给就按固定 keep
    assert engine.trim_for(segments, 0.0, 11.0, keep=0.3, target=9.0) == spans
    assert engine.trim_for(segments, 0.0, 11.0, keep=0.3) == tight
    assert engine.trim_for(segments, 0.0, 11.0, keep=0.0) == [(0.0, 11.0)]


# ------------------------------------------------------------------ 直接跑
TESTS = (


    test_normal_eight_second_clip_passes_through,
    test_target_trims_only_as_much_as_needed,

    test_start_inside_a_sentence_backs_off_to_sentence_start,
    test_end_inside_a_sentence_extends_to_sentence_end,
    test_end_never_crosses_into_next_speech,
    test_long_clip_is_not_cut_by_any_duration_rule,
    test_one_long_sentence_keeps_its_own_boundary,
    test_ending_clip_is_treated_like_any_other,
    test_extra_segments_are_dropped_with_a_warning,
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
    test_multi_segment_payload_folds_down_to_the_first,
    test_tracks_are_retimed_to_the_real_cut,
    test_silent_gaps_are_measured_not_guessed,
    test_moment_list_survives_chat_noise,
    test_silence_does_not_eat_the_length_budget,
    test_scores_stay_separate_and_metrics_are_computed,
    test_quiet_video_pulls_span_back_from_result_at,
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
