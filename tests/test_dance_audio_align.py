"""音乐对齐回归测试（技术指导第二十二节 1~8 条）。

盯的是一句话：**offset 算错，后面整条流水线产出的全部素材都是错位的，而且事后无从追查。**

  T1  相同音频 → offset ≈ 0
  T2  人为延迟 1 / 3.25 / 7.5 秒 → 都能找回来（`source_time = target_time - offset`）
  T3  源比目标早开始（负 offset）→ 同样找得回来
  T4  waveform 与 chroma 一致时置信度高；完全不相干的音频 → 置信度低且判 rejected
  T5  多窗口 offset 稳定（agreement=1，max_deviation≈0）
  T6  边缘 offset 被拒绝（offset 顶到源时长边缘 → rejected）
  T7  没有音轨 / 文件不存在 → 抛 AudioReadError，**不返回假的 offset=0**
  T8  人工修正必须留痕，且不许静默覆盖（无理由直接 ValueError）
  T9  peak_ratio_confidence 的四种边界（空数组 / primary≈0 / secondary≈0 / guard 超范围）
  T10 window_plan 覆盖开头/中间/结尾，素材过短时退化成单窗口

纯算的部分不碰磁盘；T7 会真的写一个无音轨文件。
可以 `pytest tests/test_dance_audio_align.py`，也可以 `python tests/test_dance_audio_align.py`。
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

from dance_fixtures import SR, delayed, padded, song, write_wav     # noqa: E402
from vidscribe.dance import alignment_validation as validate        # noqa: E402
from vidscribe.dance import audio_align as align                    # noqa: E402
from vidscribe.dance.audio_fingerprint import (                     # noqa: E402
    AudioReadError,
    extract_analysis_audio,
    peak_ratio_confidence,
)

#: 允许的 offset 误差（秒）。波形互相关是采样级精度，0.05 已经很宽松了
TOLERANCE = 0.05


def _target(duration: float = 40.0) -> np.ndarray:
    return song(bpm=120.0, duration=duration, seed=0)


# ------------------------------------------------------------------ T1
def test_same_audio_offset_is_zero() -> None:
    target = _target()
    result = align.align_arrays(target, target.copy())
    assert abs(result.offset) <= TOLERANCE, f"相同音频 offset 应该 ≈0，实际 {result.offset}"
    assert result.status == "ok", f"相同音频该判 ok，实际 {result.status}：{result.notes}"
    assert result.confidence >= 0.6, f"相同音频置信度该很高，实际 {result.confidence}"
    assert result.window_count >= 3, "40 秒素材该切出多个验证窗口"


# ------------------------------------------------------------------ T2
def test_artificial_delay_is_recovered() -> None:
    target = _target()
    for want in (1.0, 3.25, 7.5):
        source = delayed(target, want)
        result = align.align_arrays(target, source)
        assert abs(result.offset - want) <= TOLERANCE, \
            f"延迟 {want}s 应算出 offset≈{want}，实际 {result.offset}"
        # 口径自检：目标歌第 10 秒对应源视频第 10-offset 秒，两个方向必须严格互逆
        assert abs(result.source_time(10.0) - (10.0 - result.offset)) <= 1e-6, \
            f"source_time 口径不对：{result.source_time(10.0)} vs {10.0 - result.offset}"
        assert abs(result.target_time(result.source_time(10.0)) - 10.0) <= 1e-6, \
            "target_time 必须是 source_time 的逆运算"



# ------------------------------------------------------------------ T3
def test_negative_offset_is_recovered() -> None:
    """源比目标早开始（前面多了一段静音）→ offset 为负。"""
    target = _target()
    source = padded(target, 2.0)
    result = align.align_arrays(target, source)
    assert abs(result.offset + 2.0) <= TOLERANCE, \
        f"源提前 2s 应算出 offset≈-2.0，实际 {result.offset}"


# ------------------------------------------------------------------ T4
def test_unrelated_audio_is_rejected() -> None:
    target = _target()
    noise = (np.random.RandomState(7).randn(int(20 * SR)) * 0.2).astype(np.float32)
    result = align.align_arrays(target, noise)
    assert result.status == "rejected", f"不相干音频该判 rejected，实际 {result.status}"
    assert result.confidence < validate.REJECT_CONFIDENCE, \
        f"不相干音频置信度该低于 {validate.REJECT_CONFIDENCE}，实际 {result.confidence}"
    assert result.notes, "拒绝必须给出中文原因"


def test_two_methods_agree_raises_confidence() -> None:
    """两法一致 → 置信度明显高于两法矛盾。"""
    target = _target()
    good = align.align_arrays(target, delayed(target, 3.0))
    noise = (np.random.RandomState(11).randn(int(20 * SR)) * 0.2).astype(np.float32)
    bad = align.align_arrays(target, noise)
    assert good.confidence > bad.confidence + 0.25, \
        f"一致时({good.confidence})该明显高于矛盾时({bad.confidence})"
    assert good.chroma_offset is not None, "40 秒素材该算得出 chroma offset"


# ------------------------------------------------------------------ T5
def test_multi_window_offsets_are_stable() -> None:
    target = _target(60.0)
    result = align.align_arrays(target, delayed(target, 4.0))
    assert result.window_count >= 4, f"60 秒素材该有多个窗口，实际 {result.window_count}"
    assert result.agreement >= 0.99, f"同一份音频各窗口该完全一致，实际 {result.agreement}"
    assert result.max_deviation <= validate.OFFSET_TOLERANCE, \
        f"最大偏差 {result.max_deviation} 超过容差"
    offsets = [w.offset for w in result.windows]
    assert max(offsets) - min(offsets) <= validate.OFFSET_TOLERANCE, offsets


def test_window_disagreement_lowers_confidence() -> None:
    """前半段是目标歌、后半段换成别的歌 → 各窗口 offset 打架，置信度必须降下来。"""
    target = _target(60.0)
    other = song(bpm=95.0, duration=30.0, seed=5)
    frankenstein = np.concatenate([delayed(target, 3.0)[:int(25 * SR)], other])
    result = align.align_arrays(target, frankenstein)
    clean = align.align_arrays(target, delayed(target, 3.0))
    assert result.confidence < clean.confidence, \
        f"拼接素材({result.confidence})该低于干净素材({clean.confidence})"


# ------------------------------------------------------------------ T6
def test_edge_offset_is_rejected() -> None:
    """offset 顶到源时长边缘就是伪峰，必须拒绝（不是"精度差"，是"对错了"）。"""
    status, reasons = validate.decide_status(
        confidence=0.9, max_deviation=0.0, offset=9.6, source_duration=10.0,
        methods_agree=1.0)
    assert status == "rejected", f"边缘 offset 该被拒，实际 {status}"
    assert any("边缘" in r for r in reasons), reasons
    # 正常范围内的同样置信度必须放行
    ok, _ = validate.decide_status(0.9, 0.0, 2.0, 10.0, 1.0)
    assert ok == "ok", ok


def test_huge_deviation_is_rejected() -> None:
    status, reasons = validate.decide_status(0.95, 1.2, 1.0, 60.0, 1.0)
    assert status == "rejected", status
    assert any("偏差" in r for r in reasons), reasons


# ------------------------------------------------------------------ T7
def test_missing_audio_raises_instead_of_faking_zero(work: Path) -> None:
    missing = work / "nope.wav"
    try:
        extract_analysis_audio(missing)
    except AudioReadError as exc:
        assert "不存在" in str(exc), str(exc)
    else:
        raise AssertionError("文件不存在必须抛 AudioReadError，不能返回空数组")

    silent = work / "novoice.bin"
    silent.write_bytes(b"not a media file at all" * 64)
    try:
        extract_analysis_audio(silent)
    except AudioReadError:
        pass
    else:
        raise AssertionError("解不开的文件必须抛 AudioReadError")


def test_wav_roundtrip_keeps_offset(work: Path) -> None:
    """真的落盘再读回来，offset 依然对得上（验证 extract_analysis_audio 的裁切口径）。"""
    target = _target(30.0)
    target_file = write_wav(work / "target.wav", target)
    source_file = write_wav(work / "source.wav", delayed(target, 2.5))
    result = align.find_alignment(str(source_file), str(target_file))
    assert abs(result.offset - 2.5) <= TOLERANCE, result.offset
    # 带 start 参数读：第 0 个采样必须严格对应 start 那一刻
    whole = extract_analysis_audio(target_file)
    piece = extract_analysis_audio(target_file, start=5.0, duration=1.0)
    expect = whole[int(5.0 * SR):int(6.0 * SR)]
    assert piece.size == expect.size, (piece.size, expect.size)
    assert float(np.max(np.abs(piece - expect))) < 1e-3, "按 start 裁切的样本必须对齐"


# ------------------------------------------------------------------ T8
def test_manual_override_keeps_a_trail() -> None:
    target = _target(30.0)
    result = align.align_arrays(target, delayed(target, 3.0))
    fixed = validate.manual_override(result, 3.5, "第一拍在起手式之后", operator="tester")
    assert fixed.offset == 3.5, fixed.offset
    assert fixed.status == "manual", fixed.status
    assert abs(fixed.original_offset - result.offset) < 1e-9, "算法原值必须留着"
    assert fixed.original_confidence == result.confidence, "原置信度必须留着"
    assert fixed.manual_reason == "第一拍在起手式之后"
    assert fixed.manual_operator == "tester"
    assert fixed.manual_at, "必须记操作时间"
    assert fixed.algorithm_version == result.algorithm_version, "算法版本必须带着走"
    assert fixed.manual, "manual 标记必须为真"
    assert any("人工修正" in note for note in fixed.notes), fixed.notes
    # 原对象一个字段都不许被改（frozen dataclass + replace 的语义）
    assert result.status != "manual" and result.manual_offset is None

    # 二次修正：original_offset 仍是**算法**原值，不能被上一次人工值顶掉
    again = validate.manual_override(fixed, 3.8, "再往后半拍")
    assert abs(again.original_offset - result.offset) < 1e-9, again.original_offset
    assert again.offset == 3.8


def test_manual_override_without_reason_is_refused() -> None:
    target = _target(20.0)
    result = align.align_arrays(target, delayed(target, 1.0))
    for bad in ("", "   ", "\n"):
        try:
            validate.manual_override(result, 2.0, bad)
        except ValueError as exc:
            assert "理由" in str(exc), str(exc)
        else:
            raise AssertionError("没有理由的人工修正必须被拒（禁止静默覆盖）")


# ------------------------------------------------------------------ T9
def test_peak_ratio_confidence_edges() -> None:
    assert peak_ratio_confidence(np.zeros(0), 0, 5) == 0.0, "空数组该返回 0"
    assert peak_ratio_confidence(np.array([1.0, 2.0]), 9, 1) == 0.0, "下标越界该返回 0"
    assert peak_ratio_confidence(np.array([1.0, 2.0]), -1, 1) == 0.0, "负下标该返回 0"
    assert peak_ratio_confidence(np.zeros(10), 5, 1) == 0.0, "主峰≈0 该返回 0"
    # 次峰≈0：护栏外一片平地，主峰独一无二
    lonely = np.zeros(21)
    lonely[10] = 5.0
    assert peak_ratio_confidence(lonely, 10, 2) == 1.0, "次峰≈0 该返回 1"
    # guard 大到盖住整条：只有一个峰可比，不许报错
    assert peak_ratio_confidence(lonely, 10, 999) == 1.0, "guard 超范围该返回 1 而不是抛异常"
    # 主次一样高 → 0（完全无法区分）
    twins = np.zeros(41)
    twins[10] = twins[30] = 4.0
    assert peak_ratio_confidence(twins, 10, 2) == 0.0, "主次等高该返回 0"
    # ratio=2 → 0.5，ratio=5 → 0.8（映射 1 - 1/ratio）
    pair = np.zeros(41)
    pair[10], pair[30] = 4.0, 2.0
    assert abs(peak_ratio_confidence(pair, 10, 2) - 0.5) < 1e-6
    pair[30] = 0.8
    assert abs(peak_ratio_confidence(pair, 10, 2) - 0.8) < 1e-6


# ------------------------------------------------------------------ T10
def test_window_plan_covers_head_middle_tail() -> None:
    plan = validate.window_plan(120.0, window_seconds=20.0, count=5)
    assert len(plan) == 5, plan
    assert plan[0][0] == 0.0, "必须有开头窗口"
    assert abs(plan[-1][0] - 100.0) < 1e-6, "最后一个窗口必须贴着结尾"
    starts = [start for start, _ in plan]
    assert starts == sorted(starts), "窗口起点必须递增"
    assert all(start + length <= 120.0 + 1e-6 for start, length in plan), "窗口不许越出素材"
    middle = starts[len(starts) // 2]
    assert 30.0 < middle < 70.0, f"中间窗口该落在素材中段，实际 {middle}"

    # 素材太短 → 退化成一个整段窗口，且明确只有一个
    short = validate.window_plan(8.0, window_seconds=20.0, count=5)
    assert short == [(0.0, 8.0)], short
    assert validate.window_plan(0.0) == [], "零时长该返回空计划"


def test_short_source_cannot_reach_top_confidence() -> None:
    """只有一个窗口时拿不到"多窗口一致"的证据，置信度天然上不了顶 —— 这是有意的。"""
    single = validate.combine_confidence(1.0, 1.0, 1.0, 1.0, window_count=1)
    many = validate.combine_confidence(1.0, 1.0, 1.0, 1.0, window_count=5)
    assert single < many, (single, many)
    assert abs(many - 1.0) < 1e-6, many


def test_edge_limit_is_direction_aware() -> None:
    """边缘伪峰的上限要**分方向**，否则会误杀"短源对在长歌后段"这种正常情况。

    口径 `source_time = target_time - offset`：
      offset > 0 → 源出现在歌里更靠后 → 上限看**歌**多长
      offset < 0 → 源比歌先开始       → 上限看**源**多长

    20 秒的源合法地对在 200 秒歌的第 19 秒上（offset=+19）：
    如果两边都拿源时长比，19 ≥ 20×0.95 就被判成伪峰 —— 这是真会发生的误杀。
    """
    ok, reasons = validate.decide_status(0.8, 0.01, 19.0, 20.0, 1.0,
                                        target_duration=200.0)
    assert ok == "ok", (ok, reasons)
    # 同样的 offset，但歌本身只有 20 秒 → 这才是真的顶到边缘
    bad, reasons = validate.decide_status(0.8, 0.01, 19.0, 20.0, 1.0,
                                         target_duration=20.0)
    assert bad == "rejected", (bad, reasons)
    assert any("边缘" in r for r in reasons), reasons
    # 负方向仍然按源时长判：源只有 20 秒，却说它比歌早开始 19.5 秒
    negative, reasons = validate.decide_status(0.8, 0.01, -19.5, 20.0, 1.0,
                                              target_duration=200.0)
    assert negative == "rejected", (negative, reasons)
    # 不知道歌多长时退回原来的行为（拿源时长兜底），不能因为少传一个参数就放行一切
    fallback, _ = validate.decide_status(0.8, 0.01, 19.0, 20.0, 1.0)
    assert fallback == "rejected", fallback


TESTS = (
    test_same_audio_offset_is_zero,

    test_artificial_delay_is_recovered,
    test_negative_offset_is_recovered,
    test_unrelated_audio_is_rejected,
    test_two_methods_agree_raises_confidence,
    test_multi_window_offsets_are_stable,
    test_window_disagreement_lowers_confidence,
    test_edge_offset_is_rejected,
    test_edge_limit_is_direction_aware,

    test_huge_deviation_is_rejected,
    test_missing_audio_raises_instead_of_faking_zero,
    test_wav_roundtrip_keeps_offset,
    test_manual_override_keeps_a_trail,
    test_manual_override_without_reason_is_refused,
    test_peak_ratio_confidence_edges,
    test_window_plan_covers_head_middle_tail,
    test_short_source_cannot_reach_top_confidence,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancealign_"))
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




