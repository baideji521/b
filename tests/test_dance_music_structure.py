"""目标歌音乐分析（技术指导第五节 + 第二十二节第 3/4/5 条）。

盯四件事：

  T1  节拍：BPM 落在真值附近，拍点间隔稳定，拍数和时长对得上
  T2  固定音乐位置：位置只由 (歌长, 切片时长) 决定，和任何源视频无关
  T3  结构分段：段落连续无缝、不重叠、覆盖整首歌，类型在白名单里
  T4  分析结果全部可 JSON 序列化（要落库、要给界面）

**不用 librosa**，所以顺手也验一遍手写 DSP 在多个速度上都不掉八度。
全部用合成音乐，不依赖仓库里有没有真歌。

可以 `pytest tests/test_dance_music_structure.py`，也可以直接
`python tests/test_dance_music_structure.py`。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dance_fixtures import SR, song, write_wav                  # noqa: E402
from vidscribe.dance import dsp, music_structure as ms          # noqa: E402


def test_tempo_survives_every_speed() -> None:
    """90 / 120 / 128 / 150 BPM 都不能掉八度或翻倍。

    这是手写 DSP 最容易错的地方：自相关的短 lag 天然占便宜（重叠多），
    不做去锥形修正就会一路往 60 BPM 掉。
    """
    for bpm in (90.0, 120.0, 128.0, 150.0):
        pcm = song(bpm=bpm, duration=24.0)
        grid = ms.detect_target_beat_grid(pcm, SR)
        assert abs(grid.bpm - bpm) / bpm < 0.06, f"{bpm} BPM 被判成 {grid.bpm:.2f}"
        assert grid.source in ("beat_track", "onset_fallback", "grid"), grid.source
        assert grid.source == "beat_track", f"{bpm} BPM 走了兜底 {grid.source}"



def test_beat_grid_is_regular_and_covers_the_song() -> None:
    """拍点必须递增、间隔接近一拍、总数和时长对得上。"""
    import numpy as np

    duration = 20.0
    grid = ms.detect_target_beat_grid(song(bpm=120.0, duration=duration), SR)
    beats = np.asarray(grid.beats, dtype=float)
    assert beats.size >= 30, f"20 秒 120BPM 该有 40 拍上下，实际 {beats.size}"
    assert np.all(np.diff(beats) > 0), "拍点必须严格递增"
    gaps = np.diff(beats)
    period = 60.0 / grid.bpm
    assert abs(float(np.median(gaps)) - period) < 0.08, \
        f"拍间隔中位数 {np.median(gaps):.3f}s，一拍该是 {period:.3f}s"
    assert float(np.std(gaps)) < 0.12, f"拍点抖得太厉害：std {np.std(gaps):.3f}"
    assert beats[-1] <= duration + 0.5
    assert grid.downbeats, "重拍一个都没标"
    assert set(grid.downbeats) <= set(grid.beats), "重拍必须是拍点的子集"


def test_positions_depend_only_on_song_and_slice() -> None:
    """固定音乐位置是**目标歌的属性** —— 换源视频、换素材，位置一格都不该变。

    这是整个项目和普通随机混剪的分水岭（技术指导第七节），所以这条测试
    直接把口径钉死：位置连续铺、不重叠、最后一个不越界。
    """
    positions = ms.target_positions(10.0, 2.0)
    assert [p.index for p in positions] == [0, 1, 2, 3, 4]
    assert [p.start for p in positions] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert all(abs(p.end - p.start - 2.0) < 1e-9 for p in positions)
    assert positions[-1].end <= 10.0 + 1e-9

    # 装不满的尾巴不许硬凑：11 秒 / 2 秒还是 5 格，剩下 1 秒丢掉
    assert len(ms.target_positions(11.0, 2.0)) == 5
    # 歌比一格还短 → 一个位置都没有，而不是造一个残格
    assert ms.target_positions(1.5, 2.0) == ()
    # 切片时长变了，位置跟着变，但仍然只由这两个数决定
    assert len(ms.target_positions(10.0, 2.5)) == 4
    assert len(ms.target_positions(10.0, 1.0)) == 10


def test_positions_carry_music_meaning() -> None:
    """位置上要带音乐信息（段落 / 能量），否则评分层没法"看音乐挑素材"。"""
    analysis = ms.analyze_song(song(bpm=128.0, duration=32.0), SR)
    positions = ms.target_positions(float(analysis["duration"]), 2.0)
    sections = analysis["sections"]
    features = analysis["features"]
    assert sections, "一段都没分出来"
    for pos in positions:
        middle = (pos.start + pos.end) / 2.0
        section = ms.section_at(sections, middle)
        assert section is not None, f"位置 {pos.index} 落在了段落之外"
        assert section.type in (
            "intro", "verse", "prechorus", "chorus", "bridge",
            "breakdown", "drop", "outro", "unknown"), section.type
        # 能量 / 冲击 / 亮度都要能按区间问出来，且归一在 0~1
        for value in (features.energy_at(pos.start, pos.end),
                      features.impact_at(pos.start, pos.end),
                      features.brightness_at(pos.start, pos.end)):
            assert 0.0 <= value <= 1.0, value


def test_sections_tile_the_song_without_gaps() -> None:
    """段落必须首尾相接铺满整首歌：不重叠、不留缝。"""
    analysis = ms.analyze_song(song(bpm=120.0, duration=40.0), SR)
    sections = analysis["sections"]
    duration = float(analysis["duration"])
    assert sections[0].start <= 0.05, sections[0].start
    assert abs(sections[-1].end - duration) < 0.5, (sections[-1].end, duration)
    for previous, current in zip(sections, sections[1:]):
        assert abs(current.start - previous.end) < 1e-6, \
            f"段落之间有缝：{previous.end} → {current.start}"
        assert current.end > current.start
    for section in sections[:-1]:
        assert section.duration >= ms.MIN_SECTION_SECONDS - 1e-6, f"段落太短：{section}"
    # 段落序号必须连续，界面上是按它排的
    assert [s.index for s in sections] == list(range(len(sections)))


def test_rhythm_bands_are_separated() -> None:
    """四条频带各自出击点：kick 在低频、hihat 在高频，不能糊成同一条。"""
    rhythm = ms.analyze_rhythm_bands(song(bpm=120.0, duration=24.0), SR)
    hits = {"kick": rhythm.kick_hits, "bass": rhythm.bass_hits,
            "clap": rhythm.clap_hits, "hihat": rhythm.hihat_hits}
    assert set(hits) == set(ms.BANDS), set(hits)

    assert rhythm.kick_hits, "kick 一个击点都没检出来"
    assert rhythm.hihat_hits, "hihat 一个击点都没检出来"
    # 合成歌里每拍都有 kick，24 秒 120BPM ≈ 48 拍
    assert 20 <= len(rhythm.kick_hits) <= 80, len(rhythm.kick_hits)
    for name, series in hits.items():
        assert all(b > a for a, b in zip(series, series[1:])), f"{name} 的击点没排序"
        assert all(0.0 <= h <= 24.5 for h in series), f"{name} 击点越界"
    # 逐帧曲线长度必须和帧时间轴一致，否则界面画出来会错位
    for track in (rhythm.kick, rhythm.bass, rhythm.clap, rhythm.hihat):
        assert len(track) == len(rhythm.frame_times), (len(track), len(rhythm.frame_times))
    assert rhythm.dominant_at(0.0, 2.0), "dominant_at 该说出一个主导频带"


def test_analysis_is_json_serializable(work: Path) -> None:
    """落库的那四份 JSON 必须真能 dumps —— numpy 标量会在这里露馅。"""
    path = write_wav(work / "s.wav", song(bpm=120.0, duration=20.0))
    analysis = ms.analyze_song(path, SR)
    for key in ("duration", "sample_rate", "bpm", "beat_count", "analysis_version",
                "beats_json", "sections_json", "features_json", "rhythm_json"):
        assert key in analysis, f"分析结果缺 {key}"
    payload = {key: value for key, value in analysis.items()
               if key.endswith("_json") or isinstance(value, (int, float, str))}
    text = json.dumps(payload, ensure_ascii=False)
    assert len(text) > 200
    back = json.loads(text)
    assert abs(back["bpm"] - analysis["bpm"]) < 1e-9
    assert back["analysis_version"] == ms.ANALYSIS_VERSION
    assert back["beats_json"]["beats"], "落库的拍点是空的"
    assert back["sections_json"], "落库的段落是空的"




def test_analyze_song_accepts_path_and_array(work: Path) -> None:
    """同一段音频，走文件和走数组必须得到同样的结论。"""
    pcm = song(bpm=120.0, duration=18.0)
    path = write_wav(work / "same.wav", pcm)
    from_array = ms.analyze_song(pcm, SR)
    from_file = ms.analyze_song(path, SR)
    assert abs(from_array["bpm"] - from_file["bpm"]) < 0.6, \
        (from_array["bpm"], from_file["bpm"])
    assert abs(from_array["duration"] - from_file["duration"]) < 0.05


def test_silence_falls_back_instead_of_crashing() -> None:
    """全静音也得给出一个可用的网格（走兜底），而不是抛异常或返回空。"""
    import numpy as np

    grid = ms.detect_target_beat_grid(np.zeros(int(SR * 8), dtype="float32"), SR)
    assert grid.beats, "静音也该有兜底网格"
    assert grid.source in ("onset_fallback", "grid"), grid.source
    assert grid.confidence <= 0.5, f"静音的置信度不该高：{grid.confidence}"
    assert dsp.BPM_MIN <= grid.bpm <= dsp.BPM_MAX
    # 太短的音频（不够两个 FFT 窗）也走同一条兜底路
    tiny = ms.detect_target_beat_grid(np.zeros(1000, dtype="float32"), SR)
    assert tiny.beats and tiny.source == "grid" and tiny.confidence == 0.0



TESTS = (
    test_tempo_survives_every_speed,
    test_beat_grid_is_regular_and_covers_the_song,
    test_positions_depend_only_on_song_and_slice,
    test_positions_carry_music_meaning,
    test_sections_tile_the_song_without_gaps,
    test_rhythm_bands_are_separated,
    test_analysis_is_json_serializable,
    test_analyze_song_accepts_path_and_array,
    test_silence_falls_back_instead_of_crashing,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancemusic_"))
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
