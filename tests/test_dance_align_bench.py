"""音频对齐 / 卡点测试台（技术指导第三十五节的十条）。

盯的是这条链路上最容易出错、又最不容易从数字上看出来的一段：

  T1  target 10 + offset 3 → source 7
  T2  target 10→12 + offset 3 → source 7→9
  T3  源只有 8 秒时 7→9 必须判**不可用**（不许 clamp）
  T4  负 offset（源比目标先开始）
  T5  手动 offset 覆盖算法值，且算法原值必须还在
  T6  2 秒固定位置的生成
  T7  48 秒的歌 / 2 秒一格 = 24 格
  T8  对齐 worker 真跑一遍真实媒体，给出 offset / confidence / status
  T9  文件不存在时界面不崩，而且要说清原因
  T10 低置信度 / 失败时界面要如实显示状态

外加两条补窟窿的：
  播放"只播这一格"的边界（播到 source_end 自动停 / 循环）
  「人工修正偏移…」按钮真的点得下去（列名是 offset_seconds，不是 offset）

判定逻辑一律走后端（`align_probe` → `material_slice`），这个文件只负责**问对问题**。
可以 `pytest tests/test_dance_align_bench.py`，也可以 `python tests/test_dance_align_bench.py`。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")       # 必须在 import PyQt5 之前

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt5.QtWidgets import QApplication, QMessageBox          # noqa: E402

from vidscribe.dance import align_probe                        # noqa: E402
from vidscribe.dance import alignment_validation as validate    # noqa: E402
from vidscribe.dance import material_slice                      # noqa: E402
from vidscribe.dance.music_structure import target_positions    # noqa: E402
from vidscribe.dance.types import DanceAlignment                # noqa: E402
# **必须在 QApplication 之前 import `vidscribe.gui`**：它的 __init__ 会把
# QT_QPA_PLATFORM_PLUGIN_PATH 钉到 PyQt5 自带的 plugins 目录。这台机器上系统里
# 还有别的 Qt5，不钉的话 QApplication 构造直接 0xC0000409 崩掉（连堆栈都没有）。
from vidscribe.gui.dance_montage.align_bench import AlignBenchPanel   # noqa: E402
from vidscribe.gui.dance_montage.alignment_panel import AlignmentPanel  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _alignment(offset: float, *, source: float = 30.0, target: float = 48.0,
               confidence: float = 0.9, status: str = "ok") -> DanceAlignment:
    """一份"算好的"对齐结果。不跑 DSP —— 映射那几条规则和信号无关。"""
    return DanceAlignment(offset=float(offset), confidence=confidence, method="hybrid",
                          waveform_offset=float(offset), waveform_confidence=confidence,
                          chroma_offset=float(offset), chroma_confidence=confidence - 0.05,
                          window_count=5, max_deviation=0.01, agreement=1.0,
                          status=status, algorithm_version="test",
                          source_duration=float(source), target_duration=float(target))


class FakePlayer:
    """替掉 `FramePlayer` 用的假播放器。

    为什么不用真的：真播放器要 `import cv2`，而 cv2 会改写
    `QT_QPA_PLATFORM_PLUGIN_PATH`；更要紧的是"播到区间末尾自动停"这件事
    靠真解码去测就变成了在测机器快慢。这里要测的是**逻辑**：
    seek 到哪、什么时候 pause、循环时回到哪。
    """

    def __init__(self, duration: float = 30.0) -> None:
        self.calls: list[tuple] = []
        self._duration = float(duration)
        self._position = 0.0
        self._playing = False

    def open(self, path) -> bool:
        self.calls.append(("open", str(path)))
        return True

    def close_video(self) -> None:
        self.calls.append(("close",))

    def seek(self, seconds: float) -> None:
        self._position = float(seconds)
        self.calls.append(("seek", round(float(seconds), 3)))

    def play(self) -> None:
        self._playing = True
        self.calls.append(("play",))

    def pause(self) -> None:
        self._playing = False
        self.calls.append(("pause",))

    def is_playing(self) -> bool:
        return self._playing

    def duration(self) -> float:
        return self._duration

    def position(self) -> float:
        return self._position

    def set_audio_file(self, path) -> bool:
        self.calls.append(("audio", None if path is None else str(path)))
        self.audio = None if path is None else str(path)
        return path is not None

    def set_audio_enabled(self, enabled: bool) -> None:
        self.calls.append(("audio_on", bool(enabled)))
        self.audio_on = bool(enabled)

    def seeks(self) -> list[float]:
        return [value for name, value in
                ((c[0], c[1] if len(c) > 1 else None) for c in self.calls)
                if name == "seek"]


def _panel(work: Path, *, player: FakePlayer | None = None):
    """搭一个临时项目 + 测试台面板。面板不碰库也能用（对齐本身不需要库）。"""
    from dance_fixtures import make_project

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    panel = AlignBenchPanel(cfg)
    panel.player = player or FakePlayer()
    panel.db = db
    return panel, cfg, db


# ==================================================================== T1 ~ T7
def test_target_to_source_is_a_single_subtraction() -> None:
    """T1：target 10 + offset 3 → source 7。全系统只有这一个换算。"""
    align = _alignment(3.0)
    assert align.source_time(10.0) == 7.0
    assert align_probe.map_moment(align, 10.0) == 7.0
    # 反向也要对得上，否则时间轴上的播放头会画错位置
    assert align.target_time(7.0) == 10.0


def test_span_maps_to_span() -> None:
    """T2：target 10→12 + offset 3 → source 7→9，长度不变。"""
    row = align_probe.probe_one(_alignment(3.0), 10.0, 2.0, 30.0)
    assert row.ok, row.reason
    assert (row.source_start, row.source_end) == (7.0, 9.0)
    assert row.target_start == 10.0 and row.target_end == 12.0
    assert row.duration == 2.0


def test_out_of_range_span_is_refused_not_clamped() -> None:
    """T3：源只有 8 秒，7→9 越界。**必须判不可用，且不许把 9 掐成 8。**"""
    row = align_probe.probe_one(_alignment(3.0, source=8.0), 10.0, 2.0, 8.0)
    assert not row.ok
    assert "超过源时长" in row.reason, row.reason
    assert row.source_end == 9.0, "越界的区间被 clamp 了 —— 素材会和音乐位置错开"
    # 严格入口该直接抛，两边判定必须一致
    try:
        material_slice.map_to_source(10.0, 12.0, 3.0, 8.0)
    except material_slice.SourceRangeError:
        pass
    else:
        raise AssertionError("map_to_source 放过了越界区间")


def test_negative_offset_means_source_starts_first() -> None:
    """T4：offset 为负 = 源比目标歌先开始，所以源时间要往后推。"""
    align = _alignment(-1.827, source=40.0)
    assert align.source_time(0.0) == 1.827
    row = align_probe.probe_one(align, 0.0, 2.0, 40.0)
    assert row.ok, row.reason
    assert (row.source_start, row.source_end) == (1.827, 3.827)
    # 负 offset 时开头这一格是**能用**的（不像正 offset 会越到源前面去）
    first = align_probe.probe_all(align, song_duration=10.0, source_duration=40.0,
                                 slice_duration=2.0)[0]
    assert first.ok, first.reason


def test_manual_offset_overrides_the_algorithm_and_keeps_the_original() -> None:
    """T5：人工 offset 说话，但算法原值必须还在，理由必须有。"""
    auto = _alignment(3.274)
    fixed = validate.manual_override(auto, 3.250, "算法差了一帧，按鼓点对的")
    assert fixed.offset == 3.25
    assert fixed.manual and fixed.manual_offset == 3.25
    assert fixed.original_offset == 3.274, "算法原值被顶掉了"
    assert fixed.status == "manual"
    assert "算法差了一帧" in fixed.manual_reason
    # 换算立刻跟着走人工值
    assert fixed.source_time(10.0) == 6.75
    assert align_probe.probe_one(fixed, 10.0, 2.0, 30.0).source_start == 6.75
    # 没理由不许改
    try:
        validate.manual_override(auto, 3.25, "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("没写理由也让改了 —— 这就是静默覆盖")


def test_fixed_two_second_positions() -> None:
    """T6：2 秒一格的位置是**目标歌的属性**，与源无关。"""
    positions = target_positions(10.0, 2.0)
    assert [(p.index, p.start, p.end) for p in positions] == [
        (0, 0.0, 2.0), (1, 2.0, 4.0), (2, 4.0, 6.0), (3, 6.0, 8.0), (4, 8.0, 10.0)]
    # 末尾不足一整格的丢掉
    assert len(target_positions(9.5, 2.0)) == 4


def test_a_48_second_song_has_24_positions() -> None:
    """T7：48 秒 / 2 秒 = 24 格；offset 3.274 时前两格越界，其余可用。"""
    align = _alignment(3.274, source=30.0, target=48.0)
    rows = align_probe.probe_all(align, song_duration=48.0, source_duration=30.0,
                                slice_duration=2.0)
    assert len(rows) == 24, len(rows)
    assert not rows[0].ok and not rows[1].ok, "开头两格该越到源视频前面去"
    assert rows[2].ok, rows[2].reason
    assert (rows[2].source_start, rows[2].source_end) == (0.726, 2.726)
    # 源只有 30 秒，target 33.274 之后就没画面了 → 尾巴几格也不可用
    assert not rows[-1].ok
    usable, total, ratio = align_probe.coverage_of(rows)
    assert total == 24 and usable == sum(1 for r in rows if r.ok)
    assert 0.0 < ratio < 1.0, ratio
    print(f"  48s/2s：可用 {usable}/{total}，覆盖率 {ratio * 100:.2f}%")


def test_rejected_alignment_offers_no_usable_position() -> None:
    """被判 rejected 时一格都不该"看起来能用" —— 界面和切片必须同一个结论。"""
    align = _alignment(3.0, status="rejected", confidence=0.1)
    rows = align_probe.probe_all(align, song_duration=48.0, source_duration=30.0,
                                slice_duration=2.0)
    assert len(rows) == 24 and not any(row.ok for row in rows)
    assert "rejected" in rows[0].reason
    assert align_probe.coverage_of(rows)[2] == 0.0
    assert any(line.startswith("✗") for line in align_probe.diagnose(align, rows))


# ==================================================================== 界面
def test_panel_shows_everything_the_operator_needs(work: Path) -> None:
    """把一份对齐结果喂进面板：八个字段、时间映射、卡点表、覆盖率都得真的出现。"""
    panel, _cfg, db = _panel(work)
    try:
        panel._payload = {"song_duration": 48.0, "song_id": 0,          # noqa: SLF001
                          "source_duration": 30.0}
        panel._adopt({"path": "", "name": "girl01.mp4",                 # noqa: SLF001
                      "alignment": _alignment(3.274)})

        assert panel.out["offset"].text() == "+3.274 s", panel.out["offset"].text()
        assert "OK" in panel.out["status"].text(), panel.out["status"].text()
        assert panel.out["final"].text() == "0.900", panel.out["final"].text()
        assert "0.850" in panel.out["chroma"].text(), panel.out["chroma"].text()
        assert "5 个" in panel.out["windows"].text(), panel.out["windows"].text()
        assert panel.out["source_duration"].text() == "30.00 s"
        assert panel.out["target_duration"].text() == "48.00 s"
        assert "AUTO" in panel.origin.text(), panel.origin.text()

        # 时间映射必须把方向写出来，不能让人猜
        text = panel.mapping.text()
        assert "target" in text and "source" in text and "target − offset" in text, text

        assert panel.positions.rowCount() == 24, panel.positions.rowCount()
        assert "覆盖率" in panel.coverage.text(), panel.coverage.text()
        assert panel.positions.item(2, 4).text() == "可用"
        assert panel.positions.item(0, 4).text() == "越界"
        assert "0.726 → 2.726" in panel.positions.item(2, 2).text(), \
            panel.positions.item(2, 2).text()
        assert panel.diagnosis.toPlainText().strip(), "诊断是空的"
    finally:
        db.close()


def test_probe_and_play_only_touch_that_one_span(work: Path) -> None:
    """输 10 秒 + 2 秒 → 源 6.726→8.726；播放只播这一段，播完自动停。

    这是第六步的验收：播放器 seek 到 source_start，到 source_end 停，
    **不重新编码任何东西**。
    """
    player = FakePlayer(duration=30.0)
    panel, _cfg, db = _panel(work, player=player)
    try:
        panel._payload = {"song_duration": 48.0, "source_duration": 30.0}   # noqa: SLF001
        panel._adopt({"path": "", "name": "girl01.mp4",                     # noqa: SLF001
                      "alignment": _alignment(3.274)})
        panel.at.setValue(10.0)
        panel.slice_seconds.setValue(2.0)
        panel._probe_span()                                                 # noqa: SLF001

        assert "6.726" in panel.probe_text.text(), panel.probe_text.text()
        assert "8.726" in panel.probe_text.text(), panel.probe_text.text()
        assert round(panel._play_from, 3) == 6.726                          # noqa: SLF001
        assert round(panel._play_to, 3) == 8.726                            # noqa: SLF001

        panel._play_span()                                                  # noqa: SLF001
        assert ("play",) in player.calls, player.calls
        assert 6.726 in player.seeks(), player.seeks()

        panel._on_position(7.5)                                             # noqa: SLF001
        assert player.is_playing(), "才播到一半就停了"
        panel._on_position(8.726)                                           # noqa: SLF001
        assert not player.is_playing(), "播过了区间末尾却没停 —— 会一路播到片尾"

        # 循环模式：到末尾回到起点接着播
        panel.btn_loop.setChecked(True)
        panel._play_span()                                                  # noqa: SLF001
        panel._on_position(8.73)                                            # noqa: SLF001
        assert player.is_playing(), "循环模式下不该停"
        assert player.seeks()[-1] == 6.726, player.seeks()

        # 越界的格子不给播
        panel.at.setValue(0.0)
        panel._probe_span()                                                 # noqa: SLF001
        assert panel._play_to == panel._play_from == 0.0                    # noqa: SLF001
        assert "越界" in panel.probe_text.text(), panel.probe_text.text()
        assert "播不了" in panel.play_hint.text(), panel.play_hint.text()
    finally:
        db.close()


def test_manual_offset_in_the_panel_needs_a_reason(work: Path) -> None:
    """T5 的界面一侧：改 offset 必须弹理由框；没理由就不生效，来源要写 MANUAL。"""
    from PyQt5.QtWidgets import QInputDialog

    panel, _cfg, db = _panel(work)
    try:
        panel._payload = {"song_duration": 48.0, "source_duration": 30.0}   # noqa: SLF001
        panel._adopt({"path": "", "name": "girl01.mp4",                     # noqa: SLF001
                      "alignment": _alignment(3.274)})
        panel.manual.setValue(3.25)

        original_text = QInputDialog.getText
        original_warning = QMessageBox.warning
        warned: list[int] = []
        QMessageBox.warning = lambda *a, **k: warned.append(1)  # type: ignore[assignment]
        try:
            QInputDialog.getText = staticmethod(  # type: ignore[assignment]
                lambda *a, **k: ("", False))
            panel._apply_manual()                                           # noqa: SLF001
            assert panel._alignment.offset == 3.274, "取消了还是把 offset 改了"  # noqa: SLF001
            assert warned, "没理由却没有任何提示"

            QInputDialog.getText = staticmethod(  # type: ignore[assignment]
                lambda *a, **k: ("按鼓点手动对的", True))
            panel._apply_manual()                                           # noqa: SLF001
        finally:
            QInputDialog.getText = original_text                # type: ignore[assignment]
            QMessageBox.warning = original_warning              # type: ignore[assignment]

        assert panel._alignment.offset == 3.25                              # noqa: SLF001
        assert "MANUAL" in panel.origin.text(), panel.origin.text()
        assert "+3.274" in panel.origin.text(), "算法原值没显示出来"
        assert panel.out["offset"].text() == "+3.250 s"
        # 卡点表跟着人工值重算
        assert "0.750 → 2.750" in panel.positions.item(2, 2).text(), \
            panel.positions.item(2, 2).text()

        panel._restore_auto()                                               # noqa: SLF001
        assert panel._alignment.offset == 3.274                             # noqa: SLF001
        assert "AUTO" in panel.origin.text(), panel.origin.text()
    finally:
        db.close()


def test_missing_files_do_not_crash_the_panel(work: Path) -> None:
    """T9：源视频/目标歌不存在时不许崩，而且要拦在开跑之前。"""
    panel, _cfg, db = _panel(work)
    try:
        warned: list[str] = []
        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(  # type: ignore[assignment]
            lambda _p, title, text, *a, **k: warned.append(str(title)))
        try:
            panel.source.setText(str(work / "根本没有.mp4"))
            panel.target.setText(str(work / "也没有.wav"))
            panel.start()                       # 不该起线程，也不该抛
        finally:
            QMessageBox.warning = original      # type: ignore[assignment]
        assert warned and "文件不在盘上" in warned[0], warned
        assert panel.worker is None, "文件都不存在却还是起了线程"

        # 什么都没填也要拦住
        panel.source.clear()
        panel.target.clear()
        warned.clear()
        QMessageBox.warning = staticmethod(  # type: ignore[assignment]
            lambda _p, title, text, *a, **k: warned.append(str(title)))
        try:
            panel.start()
        finally:
            QMessageBox.warning = original      # type: ignore[assignment]
        assert warned, "空输入没有任何提示"
    finally:
        db.close()


def test_failure_is_reported_not_swallowed(work: Path) -> None:
    """T10：一条都算不出来时，界面上必须写清原因，而不是留一片"—"。"""
    panel, _cfg, db = _panel(work)
    try:
        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: None)  # type: ignore[assignment]
        try:
            panel._finished(False, "对齐完成 0／1", {                       # noqa: SLF001
                "song_duration": 48.0,
                "results": [{"path": "x.mp4", "name": "x.mp4", "alignment": None,
                             "error": "没有音轨", "cached": False}]})
        finally:
            QMessageBox.warning = original      # type: ignore[assignment]
        assert panel.out["status"].text() == "失败", panel.out["status"].text()
        assert "没有音轨" in panel.diagnosis.toPlainText(), panel.diagnosis.toPlainText()
        assert panel.positions.rowCount() == 0
        assert panel.batch.rowCount() == 1, "批量表里也该留下这条失败记录"
        assert "失败" in panel.batch.item(0, 3).text(), panel.batch.item(0, 3).text()
    finally:
        db.close()


def test_low_confidence_is_shown_as_low(work: Path) -> None:
    """T10 的另一半：置信度低要如实显示 LOW，并给出该查什么。"""
    panel, _cfg, db = _panel(work)
    try:
        panel._payload = {"song_duration": 48.0, "source_duration": 30.0}   # noqa: SLF001
        panel._adopt({"path": "", "name": "iffy.mp4",                       # noqa: SLF001
                      "alignment": _alignment(3.0, confidence=0.42,
                                              status="low_confidence")})
        assert "LOW" in panel.out["status"].text(), panel.out["status"].text()
        text = panel.diagnosis.toPlainText()
        assert "建议" in text and "变速" in text, text
    finally:
        db.close()


# ==================================================================== 真实媒体
def test_worker_aligns_real_media_without_writing_anything(work: Path) -> None:
    """T8 + 第十八节：真视频真音频跑一遍，给出 offset/置信度/结论，**且一条都不入库**。

    源视频用"目标歌截掉开头 1 秒"合成，所以正确答案是 offset ≈ +1.0
    （口径：source_time = target_time − offset）。
    """
    from dance_fixtures import delayed, make_project, make_song_file, make_source_video

    from vidscribe.gui.dance_montage.align_worker import DanceAlignWorker

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_path, pcm = make_song_file(cfg, "target.wav", bpm=120.0, duration=16.0)
    source = make_source_video(cfg, "girl01.mp4", delayed(pcm, 1.0), fps=24.0)
    db.close()

    logs: list[str] = []
    finished: list[tuple] = []
    worker = DanceAlignWorker(cfg, {"song": str(song_path), "sources": [str(source)],
                                    "workers": 1, "envelope": True})
    worker.log.connect(logs.append)
    worker.done.connect(lambda ok, msg, data: finished.append((ok, msg, data)))
    worker.run()

    assert finished, "done 一次都没发"
    ok, message, payload = finished[0]
    assert ok, f"{message}\n" + "\n".join(logs[-10:])
    row = payload["results"][0]
    align = row["alignment"]
    assert align is not None, row["error"]
    assert abs(align.offset - 1.0) < 0.05, f"offset {align.offset}（该在 +1.0 附近）"
    assert align.confidence > 0.0 and align.status in ("ok", "low_confidence"), align.status
    assert align.window_count >= 1
    assert payload["song_duration"] > 15.0
    assert len(payload["target_envelope"]) == 600, len(payload["target_envelope"])
    assert len(payload["source_envelope"]) == 600
    assert payload["fingerprint"], "没带目标歌指纹，后面就没法落库"
    print(f"  真实媒体：offset {align.offset:+.3f}s｜置信度 {align.confidence:.3f}"
          f"｜{align.status}")

    # ---- 测试台不许留痕：目标歌该登记（位置尺子要用它），对齐和视频一条都不许有
    from vidscribe.db import open_db

    check = open_db(cfg)
    try:
        conn = check.connect()
        assert conn.execute("SELECT COUNT(*) FROM dance_target_songs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM dance_audio_alignments"
                            ).fetchone()[0] == 0, "测试台把对齐写进库了"
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0, \
            "测试台把源视频登记进 videos 了"
        assert conn.execute("SELECT COUNT(*) FROM dance_materials").fetchone()[0] == 0

        # ---- 点了「保存对齐结果」才落库，而且落完能被正式流程当缓存命中
        from vidscribe.dance import material_ingest as ingest

        song = ingest.TargetSong(
            song_id=int(payload["song_id"]), path=Path(payload["song_path"]),
            fingerprint=str(payload["fingerprint"]),
            duration=float(payload["song_duration"]), bpm=float(payload["bpm"]),
            sample_rate=int(payload["sample_rate"]))
        outcome = ingest.persist_alignment(check, song, source, align)
        assert not outcome.error, outcome.error
        assert outcome.alignment_id > 0
        assert conn.execute("SELECT COUNT(*) FROM dance_audio_alignments"
                            ).fetchone()[0] == 1
        song.pcm = None
        again = ingest.align_batch(check, song, [source], workers=1, persist=False)
        assert again[0].cached, "保存之后重跑没有命中缓存 —— 缓存键两边不一致"
        assert abs(again[0].alignment.offset - align.offset) < 1e-6
    finally:
        check.close()


def test_the_old_alignment_panel_can_still_fix_an_offset(work: Path) -> None:
    """回归：「人工修正偏移…」按钮点得下去。

    列名是 `offset_seconds`（`OFFSET` 是 SQL 关键字，建表时避开了它）。
    表格那一格改对过一次，但这个按钮里还留着 `row["offset"]` ——
    一点就 IndexError。所以这里直接把按钮点一遍。
    """
    from PyQt5.QtWidgets import QInputDialog

    from dance_fixtures import fake_alignment, fake_library, make_project

    cfg, db = make_project(work)
    song_id, videos, _materials = fake_library(db, positions=2, people=("小A",))
    align_id = fake_alignment(db, song_id, list(videos.values())[0], offset=1.5)

    panel = AlignmentPanel()
    panel.refresh(db, song_id)
    assert panel.table.rowCount() == 1
    panel.table.selectRow(0)

    original_double = QInputDialog.getDouble
    original_text = QInputDialog.getText
    original_info = QMessageBox.information
    QInputDialog.getDouble = staticmethod(  # type: ignore[assignment]
        lambda *a, **k: (1.75, True))
    QInputDialog.getText = staticmethod(  # type: ignore[assignment]
        lambda *a, **k: ("按鼓点重对的", True))
    QMessageBox.information = staticmethod(lambda *a, **k: None)  # type: ignore[assignment]
    try:
        panel._fix_offset()                                     # noqa: SLF001
    finally:
        QInputDialog.getDouble = original_double    # type: ignore[assignment]
        QInputDialog.getText = original_text        # type: ignore[assignment]
        QMessageBox.information = original_info     # type: ignore[assignment]

    row = db.connect().execute(
        "SELECT offset_seconds, manual_offset, original_offset, manual_reason "
        "FROM dance_audio_alignments WHERE id = ?", (align_id,)).fetchone()
    assert abs(float(row["offset_seconds"]) - 1.75) < 1e-6, dict(row)
    assert abs(float(row["manual_offset"]) - 1.75) < 1e-6
    assert abs(float(row["original_offset"]) - 1.5) < 1e-6, "算法原值没留住"
    assert row["manual_reason"] == "按鼓点重对的"
    assert panel.table.item(0, 1).text() == "+1.750", panel.table.item(0, 1).text()
    db.close()


def test_preview_audio_is_extracted_in_the_background(work: Path) -> None:
    """「带声音」那一勾：音轨真解得出来（wav），第二次走缓存，面板拿到之后真挂上去。

    听源视频的原声就是**听对齐**：offset 求对了的话，源视频这一段的音乐
    和目标歌这一格的音乐是同一段。
    """
    from dance_fixtures import make_project, make_song_file, make_source_video

    from vidscribe.gui.dance_montage.align_worker import ClipJobWorker

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    _song_path, pcm = make_song_file(cfg, "target.wav", duration=6.0)
    source = make_source_video(cfg, "girl01.mp4", pcm, fps=24.0)
    db.close()

    job = {"kind": "audio", "source": str(source),
           "cache_dir": str(cfg.path("cache_dir"))}
    finished: list[tuple] = []
    worker = ClipJobWorker(cfg, job)
    worker.done.connect(lambda ok, msg: finished.append((ok, msg)))
    worker.run()

    assert finished, "done 一次都没发"
    ok, message = finished[0]
    assert ok, message
    wav = Path(message)
    assert wav.is_file() and wav.suffix == ".wav", message
    assert wav.stat().st_size > 2048, wav.stat().st_size

    logs: list[str] = []
    again = ClipJobWorker(cfg, job)
    again.log.connect(logs.append)
    again.done.connect(lambda ok2, msg2: finished.append((ok2, msg2)))
    again.run()
    assert finished[1][0] and finished[1][1] == message
    assert any("复用缓存" in line for line in logs), logs

    # 面板那一侧：拿到 wav 就挂给播放器，并按勾选开声音
    player = FakePlayer()
    panel, _cfg2, db2 = _panel(work, player=player)
    try:
        panel.chk_sound.blockSignals(True)
        panel.chk_sound.setChecked(True)
        panel.chk_sound.blockSignals(False)
        panel._audio_done(True, str(wav))                                # noqa: SLF001
        assert ("audio", str(wav)) in player.calls, player.calls
        assert ("audio_on", True) in player.calls, player.calls
        assert panel.chk_sound.isChecked()

        # 解不出来时：取消勾选并说清楚，别让人以为静音是对齐问题
        panel._audio_done(False, "这个视频里没有能用的音轨")             # noqa: SLF001
        assert not panel.chk_sound.isChecked()
        assert "音轨解不出来" in panel.play_hint.text(), panel.play_hint.text()
    finally:
        db2.close()
    print(f"  预览音轨：{wav.name}｜{wav.stat().st_size // 1024} KB")


def test_the_layout_is_actually_usable(work: Path) -> None:
    """布局回归：控件不许是默认小尺寸，页面在小窗口下要能滚而不是被压扁。

    第一版就是"按钮小、什么都挤在一列"，卡点表只剩两行 —— 那样等于没做。
    所以把这几条钉住：主按钮 ≥44、次按钮 ≥34、输入框 ≥30、表格行高 ≥26、
    整页外面有滚动区。
    """
    from PyQt5.QtWidgets import QScrollArea

    panel, _cfg, db = _panel(work)
    try:
        assert panel.btn_start.minimumHeight() >= 44, panel.btn_start.minimumHeight()
        assert panel.btn_play.minimumHeight() >= 44, panel.btn_play.minimumHeight()
        for button in (panel.btn_stop, panel.btn_reset, panel.btn_manual, panel.btn_auto,
                       panel.btn_save, panel.btn_ingest, panel.btn_export):
            assert button.minimumHeight() >= 34, (button.text(), button.minimumHeight())
        for field in (panel.source, panel.target, panel.at, panel.slice_seconds,
                      panel.manual):
            assert field.minimumHeight() >= 30, field.minimumHeight()
        assert panel.positions.verticalHeader().defaultSectionSize() >= 26
        assert panel.findChild(QScrollArea) is not None, "整页没有滚动区，小窗口会被压扁"
        # 越界原因在表格里只放短句，完整那句进 tooltip
        from vidscribe.gui.dance_montage.align_bench import _short_reason

        long_reason = ("位置 #0（目标 0.000→2.000）映射到源 -3.274s，"
                       "早于源视频开头 —— 不做 clamp，这一段没有对应素材")
        assert _short_reason(long_reason) == "源里还没开始"
        assert len(_short_reason("其它什么原因 —— 后面一长串解释")) <= 24
    finally:
        db.close()


def test_cli_and_gui_share_one_backend() -> None:
    """CLI 的 `align-test` 和界面走的是同一套函数，不许各写一份算法。"""
    from vidscribe import cli
    from vidscribe.gui.dance_montage import align_bench, align_worker

    parser = cli.build_parser()
    args = parser.parse_args(["dance-montage", "align-test", "--song", "1",
                              "--sources", "a.mp4", "--at", "10"])
    assert args.action == "align-test" and args.at == 10.0
    assert callable(getattr(cli, "_dance_align_test", None)), "CLI 那条动作没有实现"

    # 两边都只从 align_probe / align_batch 拿结论
    source = Path(align_bench.__file__).read_text(encoding="utf-8")
    assert "align_probe" in source
    assert "target_time -" not in source and "- offset" not in source, \
        "界面里自己减 offset 了 —— 换算只能走 DanceAlignment.source_time()"
    worker_source = Path(align_worker.__file__).read_text(encoding="utf-8")
    assert "align_batch" in worker_source
    assert "persist=False" in worker_source, "测试台的 worker 没有关掉落库"


def test_the_bench_tab_is_wired_into_the_window(work: Path) -> None:
    """页面要真的挂在 Dance 主窗口上，而且不是空壳。"""
    from dance_fixtures import fake_library, make_project

    from vidscribe.gui.dance_montage import main_page

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_id, _videos, _materials = fake_library(db, positions=3, people=("小A",))
    db.close()

    window = main_page.DanceMontageWindow(cfg)
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert "对齐/卡点测试" in titles, titles
        assert window.bench is not None
        window._song_id = song_id                      # noqa: SLF001
        window.reload()                                # 刷一遍不许抛
        assert window.bench.target.text() == str(song_id), window.bench.target.text()
        # 测试台确认后的入库请求要接到正式流水线上
        assert window.bench.receivers(window.bench.ingest_requested) >= 1
        assert window.bench.receivers(window.bench.changed) >= 1

        # 切到测试台时左边那栏要收起来：三栏工作台挤在 900 像素里没法用
        window.tabs.setCurrentWidget(window.bench)
        assert window.split.sizes()[0] == 0, window.split.sizes()
        window.tabs.setCurrentIndex(0)
        assert window.split.sizes()[0] > 0, "切回混剪之后左栏没有恢复"
    finally:
        window.close()


TESTS = (
    test_target_to_source_is_a_single_subtraction,
    test_span_maps_to_span,
    test_out_of_range_span_is_refused_not_clamped,
    test_negative_offset_means_source_starts_first,
    test_manual_offset_overrides_the_algorithm_and_keeps_the_original,
    test_fixed_two_second_positions,
    test_a_48_second_song_has_24_positions,
    test_rejected_alignment_offers_no_usable_position,
    test_panel_shows_everything_the_operator_needs,
    test_probe_and_play_only_touch_that_one_span,
    test_manual_offset_in_the_panel_needs_a_reason,
    test_missing_files_do_not_crash_the_panel,
    test_failure_is_reported_not_swallowed,
    test_low_confidence_is_shown_as_low,
    test_preview_audio_is_extracted_in_the_background,
    test_the_layout_is_actually_usable,
    test_cli_and_gui_share_one_backend,
    test_the_bench_tab_is_wired_into_the_window,
    test_the_old_alignment_panel_can_still_fix_an_offset,
    test_worker_aligns_real_media_without_writing_anything,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancebench_"))
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






