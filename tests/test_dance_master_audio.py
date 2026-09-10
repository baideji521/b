"""MASTER AUDIO 编辑区的界面回归（段落模板 + 人声导航 + 时间轴交互）。

盯的是这条分工别被写坏：

  M1  分析只摆参考信息，**不会**自动改分段
  M2  「等间隔起步」铺满整首歌；「照停顿分」切在停顿正中间
  M3  ✂ 在这里分段会先吸附到最近的参考点（拍点/停顿中心）
  M4  拖边界拖不过去时状态栏说清原因，模板一个字都不变（不许静默 clamp）
  M5  撤销能一步一步退回去
  M6  点人声导航一行 → 发出 seek，播放头跟着走
  M7  「保存这份分段」真的落库；没登记过歌就说清楚而不是假装存了
  M8  控件不许是默认小尺寸（这一页要能真用）

Qt 一律 offscreen；重活（解码/FFT）不在这里跑，模板和活动直接喂进去。
可以 `pytest tests/test_dance_master_audio.py`，也可以 `python tests/test_dance_master_audio.py`。
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

from PyQt5.QtWidgets import QApplication, QMessageBox        # noqa: E402

from vidscribe.dance import segment_template as seg          # noqa: E402
from vidscribe.dance import vocal_activity as vocal          # noqa: E402
from vidscribe.dance.types import VocalPause, VocalSpan      # noqa: E402
# 必须在 QApplication 之前 import `vidscribe.gui`（它钉 Qt 插件路径），理由同 bench 那个文件
from vidscribe.gui.dance_montage.master_audio import MasterAudioPanel   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _activity(duration: float = 20.0):
    """一份"算好的"人声活动：唱 0~4、停 4~5、唱 5~11、停 11~12.5、唱到底。"""
    spans = (VocalSpan(index=0, start=0.0, end=4.0, kind="vocal", strength=0.8),
             VocalSpan(index=1, start=4.0, end=5.0, kind="pause", strength=0.05),
             VocalSpan(index=2, start=5.0, end=11.0, kind="vocal", strength=0.75),
             VocalSpan(index=3, start=11.0, end=12.5, kind="pause", strength=0.03),
             VocalSpan(index=4, start=12.5, end=duration, kind="vocal", strength=0.7))
    pauses = (VocalPause(index=0, start=4.0, end=5.0, score=0.8, nearest_beat=4.5),
              VocalPause(index=1, start=11.0, end=12.5, score=0.4, nearest_beat=11.75))
    return vocal.VocalActivity(duration=duration, sample_rate=22050, hop_seconds=0.023,
                               frame_times=(0.0, 1.0), strength=(0.9, 0.1),
                               threshold=0.55, spans=spans, pauses=pauses,
                               version=vocal.VOCAL_VERSION)


def _panel(work: Path, *, duration: float = 20.0, with_db: bool = True):
    """搭一个分析完的面板：时长/拍点/人声都填好，但**故意不给它分段**。"""
    from dance_fixtures import make_project

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    panel = MasterAudioPanel(cfg, db if with_db else None)
    panel._duration = duration                                    # noqa: SLF001
    panel._beats = [i * 0.5 for i in range(int(duration / 0.5))]   # noqa: SLF001
    panel._activity = _activity(duration)                         # noqa: SLF001
    panel.timeline.set_song(duration, [0.5] * 100)
    panel.timeline.set_beats(panel._beats)                        # noqa: SLF001
    panel.timeline.set_vocal(panel._activity.spans,               # noqa: SLF001
                            panel._activity.pauses)               # noqa: SLF001
    panel._fill_navigator()                                       # noqa: SLF001
    return panel, cfg, db


# ================================================================== M1 ~ M3
def test_analysis_never_touches_the_segments() -> None:
    """M1：分析只摆参考信息 —— 人声/停顿都在了，分段仍然是空的。"""
    work = Path(tempfile.mkdtemp(prefix="dancemaster_"))
    panel, _cfg, db = _panel(work)
    try:
        assert panel.nav.count() == 2, panel.nav.count()
        assert panel.template is None, "分析居然自己把段落切了"
        assert panel.timeline._spans == []                        # noqa: SLF001
    finally:
        db.close()
        shutil.rmtree(work, ignore_errors=True)


def test_uniform_and_pause_starters(work: Path) -> None:
    """M2：等间隔铺满整首歌；照停顿分切在停顿正中间。"""
    panel, _cfg, db = _panel(work)
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                     # noqa: SLF001
        assert len(panel.template.spans) == 4
        assert panel.template.spans[-1].end == 20.0
        assert panel.template.source == "uniform"

        panel.min_score.setValue(0.0)
        panel._make_by_pause()                                    # noqa: SLF001
        assert panel.template.source == "pause"
        assert panel.template.boundaries == (4.5, 11.75), panel.template.boundaries

        # 门槛提到 0.6 就只剩第一个停顿够格
        panel.min_score.setValue(0.6)
        panel._make_by_pause()                                    # noqa: SLF001
        assert panel.template.boundaries == (4.5,), panel.template.boundaries
    finally:
        db.close()


def test_split_snaps_to_the_nearest_reference_point(work: Path) -> None:
    """M3：✂ 分段先吸附到最近的参考点，切出来的边界是整齐的。"""
    panel, _cfg, db = _panel(work)
    try:
        panel.step.setValue(10.0)
        panel._make_uniform()                                     # noqa: SLF001
        panel._moved_to(6.04)                                     # noqa: SLF001 - 拍点在 6.0
        panel._split_here()                                       # noqa: SLF001
        assert 6.0 in panel.template.boundaries, panel.template.boundaries
    finally:
        db.close()


# ================================================================== M4 ~ M6
def test_dragging_too_far_says_why_and_changes_nothing(work: Path) -> None:
    """M4：拖不过去 → 状态栏说清原因，模板一个字都不变。"""
    panel, _cfg, db = _panel(work)
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                     # noqa: SLF001
        before = panel.template.boundaries
        panel._drag_commit(1, 5.2)          # 想把 10.0 拖到 5.2，离 5.0 只剩 0.2 秒
        assert panel.template.boundaries == before, panel.template.boundaries
        assert "做不了" in panel.status.text(), panel.status.text()
        assert str(seg.MIN_SEGMENT_SECONDS) in panel.status.text(), panel.status.text()

        # 拖到合法位置就该成功，而且状态栏报出落点
        panel._drag_commit(1, 12.0)                               # noqa: SLF001
        assert 12.0 in panel.template.boundaries, panel.template.boundaries
    finally:
        db.close()


def test_undo_walks_back_step_by_step(work: Path) -> None:
    """M5：撤销一步一步退。"""
    panel, _cfg, db = _panel(work)
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                     # noqa: SLF001
        first = panel.template.boundaries
        panel._moved_to(7.5)                                      # noqa: SLF001
        panel._split_here()                                       # noqa: SLF001
        assert len(panel.template.boundaries) == len(first) + 1
        assert panel.btn_undo.isEnabled()

        panel._undo_once()                                        # noqa: SLF001
        assert panel.template.boundaries == first, panel.template.boundaries
        assert not panel.btn_undo.isEnabled(), "撤销栈空了按钮还亮着"
    finally:
        db.close()


def test_clicking_the_navigator_seeks(work: Path) -> None:
    """M6：点导航一行 → 发 seek。默认落在**停顿开始**，也能改成中心/结束。"""
    panel, _cfg, db = _panel(work)
    seen: list[float] = []
    panel.seek_requested.connect(seen.append)
    try:
        panel.nav.setCurrentRow(0)
        assert seen and abs(seen[-1] - 4.0) < 1e-6, seen        # 停顿 4.0→5.0 的开始
        assert abs(panel._at - 4.0) < 1e-6                      # noqa: SLF001

        panel.anchor.setCurrentIndex(1)                         # 跳到停顿中心
        panel.nav.setCurrentRow(1)
        assert abs(seen[-1] - 11.75) < 1e-6, seen               # 11.0→12.5 的中心

        panel.anchor.setCurrentIndex(0)
        panel._jump(1)                                          # noqa: SLF001 - 没有更后面的了
        assert "没有下一处停顿" in panel.status.text(), panel.status.text()
    finally:
        db.close()


# ================================================================== M7 ~ M8
def test_saving_needs_a_registered_song_and_then_really_saves(work: Path) -> None:
    """M7：没登记过歌就说清楚（并且**不存**）；登记过就真的落库、重开还在。"""
    from dance_fixtures import fake_song
    from vidscribe.dance import material_repository as repo

    panel, _cfg, db = _panel(work)
    warned: list[str] = []
    original = QMessageBox.warning
    QMessageBox.warning = staticmethod(                       # type: ignore[assignment]
        lambda *args, **kwargs: warned.append(str(args[2]) if len(args) > 2 else ""))
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                     # noqa: SLF001
        assert panel.save_template() == 0, "没有歌 id 居然存成功了"
        assert warned and "分析主音频" in warned[-1], warned

        panel._song_id = fake_song(db, duration=20.0)              # noqa: SLF001
        panel._make_uniform()                                     # noqa: SLF001
        template_id = panel.save_template()
        assert template_id > 0
        back = repo.active_segment_template(db, panel._song_id)     # noqa: SLF001
        assert back is not None and len(back.spans) == 4
        assert "已保存" in panel.status.text(), panel.status.text()
    finally:
        QMessageBox.warning = original                        # type: ignore[assignment]
        db.close()


def test_the_page_is_actually_usable(work: Path) -> None:
    """M8：控件不许是默认小尺寸；时间轴该有的横带都在，而且能被点。"""
    panel, _cfg, db = _panel(work)
    try:
        assert panel.btn_analyze.minimumHeight() >= 40
        for button in (panel.btn_prev, panel.btn_next, panel.btn_split, panel.btn_merge,
                       panel.btn_save, panel.btn_undo):
            assert button.minimumHeight() >= 34, (button.text(), button.minimumHeight())
        for field in (panel.path, panel.step, panel.min_score):
            assert field.minimumHeight() >= 30, field.minimumHeight()
        assert panel.nav.minimumWidth() >= 200
        # 时间轴：五条带首尾相接，谁也不压谁
        panel.timeline.resize(1000, 260)
        lanes = panel.timeline._lanes()                           # noqa: SLF001
        order = ["tick", "spectrum", "wave", "vocal", "segment"]
        for upper, lower in zip(order, order[1:]):
            assert lanes[upper].bottom() <= lanes[lower].top() + 0.01, (upper, lower)
        assert lanes["segment"].bottom() <= 260
    finally:
        db.close()


# ================================================================= M9 ~ M11
def test_zoom_and_scroll_share_one_axis(work: Path) -> None:
    """M9：缩放/滚动只改可见窗口，所有横带用的还是同一条时间轴。"""
    panel, _cfg, db = _panel(work, duration=60.0)
    try:
        timeline = panel.timeline
        timeline.resize(1000, 300)
        assert timeline.visible_span() == (0.0, 60.0)          # 默认看全曲

        timeline.set_view(10.0, 20.0)
        assert timeline.visible_span() == (10.0, 30.0)
        # 同一时刻在所有带里都是同一个 x：坐标只有一处换算
        x = timeline._x_of(20.0)                                # noqa: SLF001
        assert abs(timeline._time_of(x) - 20.0) < 1e-6          # noqa: SLF001

        timeline.zoom(0.5, 20.0)                                # 放大到 10 秒
        left, right = timeline.visible_span()
        assert abs((right - left) - 10.0) < 1e-6, (left, right)

        timeline.scroll_to(55.0)                                # 越界要被夹回来
        left, right = timeline.visible_span()
        assert right <= 60.0 + 1e-6 and left >= 0.0, (left, right)

        # 自动跟随：播放头跑到窗口外面就把窗口挪过去
        timeline.set_view(0.0, 10.0)
        timeline.ensure_visible(42.0)
        left, right = timeline.visible_span()
        assert left <= 42.0 <= right, (left, right)

        # 滚动条跟着窗口走（两边不许各说各话）
        panel._view_changed(*timeline.view())                   # noqa: SLF001
        assert panel.scroll.isEnabled()
        assert abs(panel.scroll.value() / 1000.0 - timeline.view()[0]) < 0.01
    finally:
        db.close()


def test_dragging_the_track_moves_the_window_not_the_zoom(work: Path) -> None:
    """M13：鼠标抓着音轨挪 —— 只搬可见窗口，缩放倍率和分段一个都不动。"""
    from PyQt5.QtCore import QEvent, QPoint, Qt
    from PyQt5.QtGui import QMouseEvent

    panel, _cfg, db = _panel(work, duration=60.0)
    try:
        timeline = panel.timeline
        timeline.resize(1000, 300)
        timeline.set_view(20.0, 20.0)
        span_before = timeline.view()[1]

        timeline.pan_by(100)          # 往右拽 100 像素 = 看更早的地方
        start, span = timeline.view()
        assert span == span_before, "挪一下把缩放也改了"
        assert abs(start - 18.0) < 1e-6, start

        timeline.pan_by(-100)
        assert abs(timeline.view()[0] - 20.0) < 1e-6, timeline.view()

        # 中键真的拖一遍：左键还得留给"点一下定位"，所以挪动只认中键/Alt+左键
        moved: list[float] = []
        timeline.seeked.connect(moved.append)
        press = QMouseEvent(QEvent.MouseButtonPress, QPoint(500, 150),
                            Qt.MiddleButton, Qt.MiddleButton, Qt.NoModifier)
        timeline.mousePressEvent(press)
        drag = QMouseEvent(QEvent.MouseMove, QPoint(600, 150),
                           Qt.NoButton, Qt.MiddleButton, Qt.NoModifier)
        timeline.mouseMoveEvent(drag)
        release = QMouseEvent(QEvent.MouseButtonRelease, QPoint(600, 150),
                              Qt.MiddleButton, Qt.NoButton, Qt.NoModifier)
        timeline.mouseReleaseEvent(release)
        assert timeline.view()[0] < 20.0, timeline.view()
        assert not moved, "中键拖动不该顺手改播放位置"

        # 全曲视图没得可挪
        timeline.set_view(0.0, 0.0)
        timeline.pan_by(300)
        assert timeline.view() == (0.0, 0.0)
    finally:
        db.close()


def test_playback_speed_only_changes_playback(work: Path) -> None:
    """M14：加减速换的是播放器速率，时间轴和分段的秒数一点没动。"""
    panel, _cfg, db = _panel(work, duration=20.0)
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                   # noqa: SLF001
        before = panel.template.boundaries
        assert abs(panel.player.playbackRate() - 1.0) < 1e-6

        index = panel.speeds.findData(0.5)
        assert index >= 0, "没有 0.5× 这一档"
        panel.speeds.setCurrentIndex(index)
        assert abs(panel.player.playbackRate() - 0.5) < 1e-6
        assert panel.template.boundaries == before, "慢放把分段改了"

        panel.speeds.setCurrentIndex(panel.speeds.findData(2.0))
        assert abs(panel.player.playbackRate() - 2.0) < 1e-6

        # 记得住：state/restore 走一圈还是这一档
        saved = panel.state()
        panel.speeds.setCurrentIndex(panel.speeds.findData(1.0))
        panel.restore(saved)
        assert abs(panel.player.playbackRate() - 2.0) < 1e-6
    finally:
        db.close()



def test_marks_are_recorded_but_change_nothing(work: Path) -> None:
    """M10：⭐ 标记落库、再点一次取消；**它一个字都不改分段**。"""
    from dance_fixtures import fake_song
    from vidscribe.dance import material_repository as repo

    panel, _cfg, db = _panel(work)
    try:
        panel._song_id = fake_song(db, duration=20.0)           # noqa: SLF001
        panel.step.setValue(5.0)
        panel._make_uniform()                                   # noqa: SLF001
        before = panel.template.boundaries

        panel._moved_to(6.4)                                    # noqa: SLF001
        panel.toggle_mark()
        assert [round(float(r["moment"]), 3) for r in
                repo.cut_marks(db, panel._song_id)] == [6.4]    # noqa: SLF001
        assert panel.marks == [6.4], panel.marks
        assert panel.template.boundaries == before, "标记居然改了分段"

        panel.toggle_mark()                                     # 同一处再点 = 取消
        assert repo.cut_marks(db, panel._song_id) == []         # noqa: SLF001
        assert panel.marks == []
    finally:
        db.close()


def test_playing_one_segment_stops_at_its_end(work: Path) -> None:
    """M11：「播放当前段」到段尾自动停；重做能把撤销掉的那一步做回来。"""
    panel, _cfg, db = _panel(work)
    try:
        panel.step.setValue(5.0)
        panel._make_uniform()                                   # noqa: SLF001
        panel._moved_to(7.0)                                    # noqa: SLF001 - 落在 S2
        panel.play_current_segment()
        assert panel._stop_at == 10.0, panel._stop_at           # noqa: SLF001
        assert "S2" in panel.status.text(), panel.status.text()

        panel._position_changed(10_200)                         # noqa: SLF001 - 播过段尾
        assert panel._stop_at is None, "过了段尾没有停"          # noqa: SLF001

        # 撤销 → 重做
        first = panel.template.boundaries
        panel._moved_to(12.5)                                   # noqa: SLF001
        panel._split_here()                                     # noqa: SLF001
        cut = panel.template.boundaries
        panel.undo()
        assert panel.template.boundaries == first
        panel.redo()
        assert panel.template.boundaries == cut, panel.template.boundaries
    finally:
        db.close()


def test_the_song_picker_really_opens(work: Path) -> None:
    """M12：点「选主音频…」必须真的弹出对话框并把路径填回去。

    这条是补窟窿的：`dialogs.open_file` 的签名是 (parent, title, folder, filters, key)，
    我漏了 folder，于是一点按钮就 TypeError。签名对不对，只有真调一次才知道。
    """
    from PyQt5.QtWidgets import QFileDialog

    panel, _cfg, db = _panel(work)
    seen: list[tuple] = []
    original = QFileDialog.getOpenFileName

    def fake(_parent, title, folder, filters, options=0):
        seen.append((title, folder, filters))
        return (str(work / "song.mp3"), "")

    QFileDialog.getOpenFileName = staticmethod(fake)
    try:
        panel._pick()                                           # noqa: SLF001
    finally:
        QFileDialog.getOpenFileName = original

    try:
        assert len(seen) == 1, seen
        title, folder, filters = seen[0]
        assert "主音频" in title, title
        assert Path(folder).is_dir(), f"起始目录不存在：{folder}"
        assert "mp3" in filters, filters
        assert panel.path.text().endswith("song.mp3"), panel.path.text()
    finally:
        db.close()


TESTS = (


    test_analysis_never_touches_the_segments,
    test_uniform_and_pause_starters,
    test_split_snaps_to_the_nearest_reference_point,
    test_dragging_too_far_says_why_and_changes_nothing,
    test_undo_walks_back_step_by_step,
    test_clicking_the_navigator_seeks,
    test_saving_needs_a_registered_song_and_then_really_saves,
    test_the_page_is_actually_usable,
    test_zoom_and_scroll_share_one_axis,
    test_marks_are_recorded_but_change_nothing,
    test_playing_one_segment_stops_at_its_end,
    test_the_song_picker_really_opens,
    test_dragging_the_track_moves_the_window_not_the_zoom,
    test_playback_speed_only_changes_playback,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancemaster_"))
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



