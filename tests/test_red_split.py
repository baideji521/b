"""按片尾红屏切素材（Phase 17）。

每个高光成品最后都固定接 1 秒纯红（highlight/clip.py 的 RED_TAIL_SECONDS）。
把成品拼成一条长视频后，那几段红屏就是素材之间的分界线。这里测的就是这件事：
红屏找得准、切出来的素材边界对，而且红屏本身不算进任何一段。

覆盖：
  T1 彩色 1s + 红 1s + 彩色 1s + 红 1s -> 两段素材，边界落在红屏两侧
  T2 通篇没有红屏                      -> 整片算一段
  T3 按素材分段导出                    -> 每段一份完整的合并导出，时间戳还是绝对秒

可以直接 `python tests/test_red_split.py`，也可以 `pytest tests/test_red_split.py`。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import av                                                      # noqa: E402
import numpy as np                                             # noqa: E402

from vidscribe.timeline import exporters                        # noqa: E402
from vidscribe.timeline import red_split                        # noqa: E402

FPS = 10
SIZE = (64, 48)  # (宽, 高)，小一点让测试快


def make_video(path: Path, blocks: list[tuple[str, float]]) -> float:
    """按 [(颜色, 秒数), ...] 生成一段 mp4，返回总时长。

    颜色只有两种："red" = 纯红（要被当成分界），"color" = 带纹理的非红画面。
    """
    width, height = SIZE
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, FPS)
    stream.options = {"crf": "18"}
    index = 0
    for kind, seconds in blocks:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        if kind == "red":
            rgb[:, :] = (255, 0, 0)
        else:
            # 非红画面故意做成有纹理的蓝绿渐变：既不满足"红够亮"，也不满足"整屏一致"
            ramp = np.linspace(0, 255, width, dtype=np.uint8)
            rgb[:, :, 1] = ramp[None, :]
            rgb[:, :, 2] = 255 - ramp[None, :]
        for _ in range(int(round(seconds * FPS))):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, FPS)
            container.mux(stream.encode(frame))
            index += 1
    container.mux(stream.encode(None))
    container.close()
    return index / FPS


def near(value: float, target: float, tol: float = 0.25) -> bool:
    return abs(value - target) <= tol


def test_red_frames_split_the_merge(work: Path) -> None:
    video = work / "merged.mp4"
    duration = make_video(video, [("color", 1.0), ("red", 1.0),
                                  ("color", 1.0), ("red", 1.0)])
    spans = red_split.segments_of(video, duration)
    assert len(spans) == 2, f"应该切出 2 段素材，实际 {len(spans)}：{spans}"
    assert near(spans[0][0], 0.0), f"第 1 段该从 0 开始：{spans[0]}"
    assert near(spans[0][1], 1.0), f"第 1 段该在红屏开始处收住：{spans[0]}"
    assert near(spans[1][0], 2.0), f"第 2 段该从第 1 段红屏结束处开始：{spans[1]}"
    assert near(spans[1][1], 3.0), f"第 2 段该在第 2 段红屏开始处收住：{spans[1]}"
    # 红屏不属于任何一段：1.0-2.0 和 3.0-4.0 都不在任何区间里
    for begin, finish in spans:
        assert not (begin < 1.5 < finish), f"红屏被算进素材了：{spans}"
        assert not (begin < 3.5 < finish), f"红屏被算进素材了：{spans}"


def test_video_without_red_is_one_piece(work: Path) -> None:
    video = work / "plain.mp4"
    duration = make_video(video, [("color", 2.0)])
    spans = red_split.segments_of(video, duration)
    assert len(spans) == 1, f"没有红屏就该整片算一段，实际 {spans}"
    assert near(spans[0][0], 0.0) and near(spans[0][1], duration), f"{spans}"


def test_grouped_export_keeps_absolute_seconds(work: Path) -> None:
    spans = [(0.0, 1.0), (2.0, 3.0)]
    segments = [
        {"start": 0.2, "end": 0.8, "text": "第一段说话",
         "words": [{"start": 0.2, "end": 0.5, "word": "第一段"},
                   {"start": 2.2, "end": 2.5, "word": "串段了"}]},
        {"start": 2.2, "end": 2.9, "text": "第二段说话", "words": []},
    ]
    events = [{"start": 0.1, "end": 0.6, "description": "画面一", "event": "", "importance": ""},
              {"start": 2.1, "end": 2.6, "description": "画面二", "event": "", "importance": ""}]
    lines, total = exporters.grouped_merged_lines(
        "merged.mp4", spans, segments, events, duration=4.0)
    text = "\n".join(lines)
    assert "素材 1 / 2" in text and "素材 2 / 2" in text, "缺少分段块头"
    assert "共 2 段素材" in text, "缺少素材清单"
    assert total == 4, f"两段各 2 条，共 4 条，实际 {total}"
    # 时间戳保持合并视频的绝对秒：第二段的内容仍然写成 2.xx，不重排到 0.xx
    assert "[2.20 - 2.90]" in text, "第二段的语音时间戳被改过"
    first_block = text.split("素材 2 / 2")[0]
    assert "串段了" not in first_block, "第一段里出现了属于第二段的词"

    path = work / "pieces.txt"
    written = exporters.write_grouped_merged_txt(
        path, "merged.mp4", spans, segments, events, duration=4.0)
    assert written == total
    assert path.read_text(encoding="utf-8").strip(), "导出文件是空的"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_red_frames_split_the_merge,
    test_video_without_red_is_one_piece,
    test_grouped_export_keeps_absolute_seconds,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="redsplit_"))
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
