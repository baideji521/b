"""帧采样 / 降级阶梯回归测试。

覆盖三个已经真实发生过的 Bug：
1. plan_frame_indices(info, start, end, fps, min_frames, max_frames) 参数被调用方写反，
   导致改配置里的 max_frames 没有任何效果（帧数永远被 min/max 顶死）。
2. sample_frames 在 seek 之后读 CAP_PROP_POS_MSEC，拿到的是垃圾值，
   所有帧时间戳都塌到 0.0x 秒 —— 整条“真实帧时间戳”地基是坏的。
3. VisualParams.degrade() 的 max_new_tokens=max(384, x*0.75)，512 只能降到 384，
   256/192/128 永远测不到。

不依赖 pytest：直接 `python tests/test_frame_sampling.py` 即可运行。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe import video_io                                      # noqa: E402
from vidscribe.video_io import plan_frame_indices, probe_video, sample_frames  # noqa: E402
from vidscribe.visual.qwen_vl import VisualParams                  # noqa: E402

VIDEO = ROOT / "test.mp4"


def test_signature_order() -> None:
    """函数签名必须是 (info, start, end, fps, min_frames, max_frames)。"""
    names = list(inspect.signature(plan_frame_indices).parameters)
    assert names == ["info", "start", "end", "fps", "min_frames", "max_frames"], names


def test_call_sites_use_min_then_max() -> None:
    """仓库里所有调用点都必须按 min_frames, max_frames 的顺序传参。"""
    bad: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "plan_frame_indices(" not in line or "def plan_frame_indices" in line:
                continue
            if "params.max_frames, params.min_frames" in line:
                bad.append(f"{path}:{lineno}")
    assert not bad, f"参数顺序写反的调用点: {bad}"


def test_max_frames_takes_effect() -> None:
    """改 max_frames 必须真的改变实际帧数（6/8/12/16 各不相同）。"""
    info = probe_video(VIDEO)
    actual: dict[int, int] = {}
    for want in (6, 8, 12, 16):
        # fps 给足，让 max_frames 成为唯一约束
        idx = plan_frame_indices(info, 0.0, 10.0, 4.0, 4, want)
        batch = sample_frames(info, idx, 112 * 32 * 32)
        actual[want] = len(batch)
        assert len(idx) == want, f"max_frames={want} 规划出 {len(idx)} 帧"
        assert len(batch) == want, f"max_frames={want} 实际采到 {len(batch)} 帧"
    assert len(set(actual.values())) == 4, f"max_frames 未生效: {actual}"


def test_timestamps_are_real() -> None:
    """帧时间戳必须单调递增，并铺满窗口（回归 POS_MSEC 塌到 0 的 Bug）。"""
    info = probe_video(VIDEO)
    for start, end in ((0.0, 10.0), (10.0, 20.0), (30.0, 40.0)):
        idx = plan_frame_indices(info, start, end, 1.0, 4, 8)
        batch = sample_frames(info, idx, 112 * 32 * 32)
        ts = batch.timestamps
        assert ts == sorted(ts), f"时间戳非单调: {ts}"
        assert ts[-1] - ts[0] > (end - start) * 0.6, f"时间戳没铺开: {ts}"
        for t, i in zip(ts, batch.frame_indices):
            expect = i / info.fps
            assert abs(t - expect) < 0.15, f"帧 {i} 时间戳 {t} 与 {expect:.3f} 不符"
        assert 0.4 < batch.sample_fps < 2.0, f"sample_fps 异常: {batch.sample_fps}"


def test_degrade_reaches_128() -> None:
    """降级阶梯必须能真的走到 max_new_tokens=128，并且每一步可追溯。"""
    params = VisualParams(fps=1.5, max_frames=16, min_frames=6, max_pixels_tokens=112,
                          total_pixels_tokens=2048, max_new_tokens=512)
    seen_tokens = {params.max_new_tokens}
    seen_frames = {params.max_frames}
    cur = params
    for _ in range(20):
        if not cur.can_degrade():
            break
        nxt = cur.degrade(reason="unit_test")
        assert nxt.degrade_level == cur.degrade_level + 1
        assert len(nxt.degrade_history) == nxt.degrade_level
        assert nxt.max_frames <= cur.max_frames
        assert nxt.max_pixels_tokens <= cur.max_pixels_tokens
        assert nxt.max_new_tokens <= cur.max_new_tokens
        seen_tokens.add(nxt.max_new_tokens)
        seen_frames.add(nxt.max_frames)
        cur = nxt
    assert not cur.can_degrade(), "降级阶梯没有收敛"
    assert {512, 384, 256, 192, 128} <= seen_tokens, f"生成长度阶梯不完整: {sorted(seen_tokens)}"
    assert cur.max_new_tokens == 128, cur.max_new_tokens
    assert 6 in seen_frames, f"帧数没降到下限: {sorted(seen_frames)}"
    # 原对象不能被静默修改
    assert params.max_new_tokens == 512 and params.max_frames == 16


def test_degrade_does_not_touch_defaults() -> None:
    """默认参数下 to_dict 不应出现降级字段，避免污染正常配置记录。"""
    params = VisualParams(fps=1.5, max_frames=16, min_frames=6, max_pixels_tokens=112,
                          total_pixels_tokens=2048, max_new_tokens=512)
    assert "degrade_level" not in params.to_dict()
    assert "degrade_level" in params.degrade().to_dict()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        needs_video = fn.__name__ in ("test_max_frames_takes_effect", "test_timestamps_are_real")
        if needs_video and not VIDEO.is_file():
            print(f"SKIP {fn.__name__}（缺少 {VIDEO.name}）")
            continue
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
