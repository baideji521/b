"""AI_卡点舞 独立界面（技术指导第二十节 + 一期第二十四节）。

界面测试跑在 `QT_QPA_PLATFORM=offscreen` 下，不需要真屏幕。盯的是：

  T1  窗口能建起来，四个区域的面板都在
  T2  面板能按真实素材库铺出内容（不是空壳）
  T3  切片时长预设 1.0/1.5/2.0/2.5/3.0/自定义 齐全，取值正确
  T4  关掉智能推荐 + 没有手动选择时，点开始必须被拦住（而不是跑出个空片）
  T5  候选池的手动钉选能传到混剪面板
  T6  九个阶段的名字和一期钉的一模一样
  T7  `DanceMontageWorker` 和 `AnalyzeWorker` 是两个毫无关系的类

**主线程绝不 import cv2**（会改写 QT_QPA_PLATFORM_PLUGIN_PATH 把 QApplication 搞崩），
所以这里只碰界面和库，重活那条路交给 smoke test。

可以 `pytest tests/test_dance_gui.py`，也可以 `python tests/test_dance_gui.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")       # 必须在 import PyQt5 之前

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt5.QtWidgets import QApplication                      # noqa: E402

from dance_fixtures import fake_library, make_project         # noqa: E402
from vidscribe.gui.dance_montage import main_page             # noqa: E402
from vidscribe.gui.dance_montage.remix_panel import SLICE_PRESETS   # noqa: E402
from vidscribe.gui.dance_montage.worker import STAGES, DanceMontageWorker  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _window(work: Path):
    """搭一个临时项目 + 一个方阵素材库，然后把窗口开起来。"""
    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_id, _videos, materials = fake_library(db, positions=5,
                                               people=("小A", "小B", "小C"))
    db.close()                       # 窗口自己开自己的连接，别让两份连接抢
    window = main_page.DanceMontageWindow(cfg)
    return window, song_id, materials


def test_window_has_all_four_regions(work: Path) -> None:
    """一期第二十四节的四个区域，面板一个都不能少。"""
    window, _song_id, _m = _window(work)
    try:
        assert window.remix is not None            # ① 输入与操作
        assert window.filters is not None and window.library is not None   # ② 素材资产
        assert window.candidates is not None and window.recommend is not None  # ③ 选择与推荐
        assert window.history is not None and window.statistics is not None    # ④ 历史与统计
        assert window.alignment is not None
        assert window.bench is not None             # 音频对齐 / 卡点测试台
        assert window.master is not None            # 主音频编辑区（段落模板）
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == ["素材资产", "主音频/分段", "音频对齐", "对齐/卡点测试",
                          "选择与推荐", "素材矩阵/成片", "历史与统计"], titles
        assert "卡点舞" in window.windowTitle()
        # 小窗口也要能用（一期第十五节）
        assert window.minimumWidth() <= 1000 and window.minimumHeight() <= 640
    finally:
        window.close()


def test_panels_show_real_library(work: Path) -> None:
    """面板要真的按库里的素材铺内容，不是空壳。"""
    window, song_id, _m = _window(work)
    try:
        window._song_id = song_id                  # noqa: SLF001 - 测试里直接切歌
        window.reload()
        assert window.library.table.rowCount() == 15, window.library.table.rowCount()
        assert "15 条素材" in window.library.summary.text(), window.library.summary.text()
        assert window.candidates.positions.count() == 5, window.candidates.positions.count()
        assert window.candidates.table.rowCount() == 3, "每个位置该有 3 个人的素材"
        assert window.statistics.positions.rowCount() == 5
        assert window.statistics.persons.rowCount() == 3
        assert "素材 15 条" in window.statistics.overview.text()
        # 停用一条之后默认就不显示了（但库里还在）
        window.library.table.selectRow(0)
        assert window.library._ids(), "选中行取不到 id"      # noqa: SLF001
    finally:
        window.close()


def test_filter_presets_reach_the_query(work: Path) -> None:
    """七套方案要能选、能真的改查询结果。"""
    window, song_id, materials = _window(work)
    try:
        window._song_id = song_id                  # noqa: SLF001
        window.reload()
        assert window.filters.preset.count() == 8, "七套方案 + 一个「不用预设」"
        # 把一条素材标成用过，再筛"从未使用"，就该少一条
        window.db.connect().execute(
            "UPDATE dance_materials SET use_count = 3, last_used_at = datetime('now') "
            "WHERE id = ?", (materials[("小A", 0)],))
        index = window.filters.preset.findData("never_used")
        assert index > 0
        window.filters.preset.setCurrentIndex(index)
        window._reload_materials()                 # noqa: SLF001
        assert window.library.table.rowCount() == 14, window.library.table.rowCount()
        # 排序方式换成"使用次数少→多"也不该报错
        window.filters.sort.setCurrentIndex(1)
        window._reload_materials()                 # noqa: SLF001
        assert window.library.table.rowCount() == 14
    finally:
        window.close()


def test_slice_presets(work: Path) -> None:
    """1.0/1.5/2.0/2.5/3.0/自定义 —— 一期第二十二节的原话。"""
    window, _song_id, _m = _window(work)
    try:
        values = [v for v, _label in SLICE_PRESETS]
        assert values[:5] == [1.0, 1.5, 2.0, 2.5, 3.0], values
        assert values[-1] < 0, "最后一档该是自定义"
        panel = window.remix
        assert panel.slice_duration() == 2.0, "默认该是 2.0 秒"
        panel.slice_preset.setCurrentIndex(0)
        assert panel.slice_duration() == 1.0
        panel.slice_preset.setCurrentIndex(len(values) - 1)
        assert panel.slice_custom.isEnabled(), "选了自定义却没打开输入框"
        panel.slice_custom.setValue(1.75)
        assert panel.slice_duration() == 1.75
    finally:
        window.close()


def test_manual_selection_flows_to_remix(work: Path) -> None:
    """候选池里钉的选择，要能传到混剪面板（关推荐时就靠它出片）。"""
    window, song_id, materials = _window(work)
    try:
        window._song_id = song_id                  # noqa: SLF001
        window.reload()
        window.candidates.positions.setCurrentRow(0)
        window.candidates.table.selectRow(0)
        window.candidates._pick()                  # noqa: SLF001 - 等价于双击
        assert len(window.candidates.manual()) == 1, window.candidates.manual()
        assert "1 格" in window.remix.manual_hint.text(), window.remix.manual_hint.text()
        # 一键填满所有格
        window.candidates._fill_from_top()         # noqa: SLF001
        assert len(window.candidates.manual()) == 5, window.candidates.manual()
        assert window.remix.job()["manual"] == window.candidates.manual()
        window.candidates.clear_manual()
        assert window.candidates.manual() == {}
    finally:
        window.close()


def test_start_refuses_empty_manual_when_recommend_off(work: Path) -> None:
    """关了智能推荐又没手动选 → 必须拦住，不许跑出个空片。"""
    window, song_id, _m = _window(work)
    try:
        window._song_id = song_id                  # noqa: SLF001
        window.reload()
        window.remix.recommend.setChecked(False)
        window.remix.song.setText(str(song_id))
        blocked = {"count": 0}
        from PyQt5.QtWidgets import QMessageBox

        original = QMessageBox.warning
        QMessageBox.warning = lambda *a, **k: blocked.__setitem__(  # type: ignore[assignment]
            "count", blocked["count"] + 1)
        try:
            window.start(window.remix.job())
        finally:
            QMessageBox.warning = original         # type: ignore[assignment]
        assert blocked["count"] == 1, "该弹一次警告"
        assert window.worker is None, "被拦住了却还是起了线程"
    finally:
        window.close()


def test_worker_is_isolated_from_analyze_worker() -> None:
    """九个阶段名字要和一期钉的一致；工人类和 AnalyzeWorker 毫无关系。"""
    assert STAGES == ("扫描素材", "音频提取", "音乐对齐", "生成切片", "建立素材库",
                      "准备混剪", "渲染", "封装", "完成"), STAGES
    from vidscribe.gui import main_window

    analyze = getattr(main_window, "AnalyzeWorker", None)
    assert analyze is not None, "主界面的 AnalyzeWorker 不见了 —— 原功能被动了"
    assert DanceMontageWorker is not analyze
    assert not issubclass(DanceMontageWorker, analyze)
    assert not issubclass(analyze, DanceMontageWorker)
    # 信号也各自独立，名字撞了不要紧，对象不能是同一个
    assert set(DanceMontageWorker.__dict__) & {"log", "stage", "progress", "done"}


def test_worker_runs_the_whole_job(work: Path) -> None:
    """工人线程整条路真跑一遍：九个阶段都走到，最后真出一个能播的成品。

    这里**直接调 `run()`**（不 `start()`）：同一个线程里跑完，不用起 Qt 事件循环，
    信号照样发得出来（PyQt 的直连槽是同步调用）。
    """
    from dance_fixtures import delayed, make_project, make_song_file, make_source_video

    cfg, db = make_project(work)
    cfg.dance.update({"canvas_width": 144, "canvas_height": 256, "canvas_fps": 24.0})
    cfg.ensure_dance_dirs()
    song_path, pcm = make_song_file(cfg, "target.wav", bpm=120.0, duration=10.0)
    make_source_video(cfg, "d0.mp4", pcm, tint=(200, 80, 90), fps=24.0)
    make_source_video(cfg, "d1.mp4", delayed(pcm, 1.0), tint=(80, 190, 120), fps=25.0)
    db.close()

    stages: list[str] = []
    logs: list[str] = []
    finished: list[tuple] = []
    worker = DanceMontageWorker(cfg, {
        "song": str(song_path), "sources": [str(cfg.dance_path("source_dir"))],
        "slice_duration": 2.0, "workers": 2, "versions": 1, "seed": 4242,
        "do_slice": True, "do_remix": True, "render": True, "recommend": True,
    })
    worker.stage.connect(lambda name, _i, _t: stages.append(name))
    worker.log.connect(logs.append)
    worker.done.connect(lambda ok, msg, data: finished.append((ok, msg, data)))
    worker.run()

    assert finished, "done 信号一次都没发"
    ok, message, data = finished[0]
    assert ok, f"{message}\n" + "\n".join(logs[-12:])
    assert stages == list(STAGES), stages          # 九个阶段一个不少、顺序不乱
    assert int(data["materials"]) >= 8, data
    assert data["versions"] and data["outputs"], data
    out = Path(str(data["outputs"][0]))
    assert out.is_file() and out.stat().st_size > 1024, out
    assert any("音轨 1 条" in line for line in logs), "没确认目标歌是唯一音轨"
    print(f"  工人线程出片：{out.name}｜素材 {data['materials']} 条")


def test_worker_stops_cooperatively(work: Path) -> None:
    """点了停止要在阶段边界停下来，而不是把文件写坏或者装作没听见。"""
    from dance_fixtures import make_project, make_song_file, make_source_video

    cfg, db = make_project(work)
    cfg.dance.update({"canvas_width": 144, "canvas_height": 256, "canvas_fps": 24.0})
    cfg.ensure_dance_dirs()
    song_path, pcm = make_song_file(cfg, "target.wav", bpm=120.0, duration=10.0)
    make_source_video(cfg, "d0.mp4", pcm, fps=24.0)
    db.close()

    finished: list[tuple] = []
    worker = DanceMontageWorker(cfg, {
        "song": str(song_path), "sources": [str(cfg.dance_path("source_dir"))],
        "slice_duration": 2.0, "workers": 1, "do_slice": True, "do_remix": True,
    })
    worker.done.connect(lambda ok, msg, data: finished.append((ok, msg, data)))
    worker.stop()                                  # 一开始就按下停止
    worker.run()

    assert finished, "停了也得发 done"
    _ok, message, data = finished[0]
    assert "停止" in message, message
    assert not data.get("outputs"), "已经喊停了却还是出了片"


def test_file_pickers_use_the_system_dialog_and_remember_the_folder(work: Path) -> None:
    """选文件走**系统对话框**，而且每个用途各记一个"上次去过的目录"。

    这里曾经反过来钉着"必须用 Qt 自绘"，理由是怀疑原生对话框把界面拖死了 ——
    后来查清楚那次卡死是对齐面板读了不存在的列名抛 IndexError，跟对话框无关。
    系统对话框认得快速访问/最近使用/网盘，找素材顺手得多，所以改回默认用它，
    但 `dance.native_dialogs = false` 的退路仍然要真的管用。
    """
    from PyQt5.QtWidgets import QFileDialog

    from vidscribe.gui.dance_montage import dialogs

    window, _song_id, _m = _window(work)
    try:
        panel = window.remix
        seen: list[tuple] = []

        def fake(_parent, title, folder, filters, options=0):
            seen.append((title, folder, int(options)))
            return ("", "")

        original = QFileDialog.getOpenFileName
        QFileDialog.getOpenFileName = staticmethod(fake)
        try:
            panel._pick_song()                     # noqa: SLF001
        finally:
            QFileDialog.getOpenFileName = original

        assert len(seen) == 1, seen
        _title, folder, options = seen[0]
        assert not options & int(QFileDialog.DontUseNativeDialog), \
            "默认该用系统自带的对话框"
        assert Path(folder).is_dir(), f"起始目录不存在：{folder}"

        # 退路：配置关掉之后真的换成 Qt 自绘
        dialogs.configure(False)
        try:
            assert int(dialogs.options()) & int(QFileDialog.DontUseNativeDialog)
        finally:
            dialogs.configure(True)

        # 记忆：选过一次之后，同一个用途下次就从那个目录开始
        store: dict = {}
        dialogs.install_memory(store, None)
        try:
            picked = work / "girl01.mp4"
            picked.write_bytes(b"x")
            dialogs.remember("dance.source", picked)
            assert store["dance.source"] == str(work)
            assert Path(dialogs.start_dir("", "dance.source")) == work
            # 没记过的用途照旧退回默认目录 / 主目录
            assert Path(dialogs.start_dir("Z:/根本没有这个盘/x", "dance.song")) == Path.home()
            assert Path(dialogs.start_dir("")) == Path.home()
            assert Path(dialogs.start_dir(work)) == work
        finally:
            dialogs.install_memory(window.state.setdefault("dirs", {}),
                                   window.save_settings)
    finally:
        window.close()



def test_settings_survive_a_restart(work: Path) -> None:
    """关掉再开，界面该长回上次的样子：窗口大小、分栏比例、当前标签、各输入框。

    存在 `gui_settings.json` 里（和主界面同一个文件，各占一个键）。
    """
    from vidscribe.gui.dance_montage import dialogs

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_id, _videos, _materials = fake_library(db, positions=4, people=("小A",))
    db.close()

    first = main_page.DanceMontageWindow(cfg)
    try:
        first.resize(1320, 880)
        first.split.setSizes([480, 840])
        first.remix.song.setText(str(song_id))
        first.remix.slice_preset.setCurrentIndex(1)          # 1.5 秒
        first.remix.person.setText("小A")
        first.remix.recommend.setChecked(False)
        first.bench.source.setText(str(work / "girl01.mp4"))
        first.bench.at.setValue(12.5)
        first.bench.slice_seconds.setValue(1.5)
        # 只测"记不记得住"，所以别触发槽：真勾选会去解音轨，那是另一码事
        first.bench.chk_sound.blockSignals(True)
        first.bench.chk_sound.setChecked(True)
        first.bench.chk_sound.blockSignals(False)
        first.bench.btn_loop.setChecked(True)
        first.tabs.setCurrentIndex(4)       # 「选择与推荐」——它不会把左栏收起来
        dialogs.remember("dance.source", work / "girl01.mp4")
    finally:
        first.close()                                       # closeEvent 里落盘

    saved = json.loads((Path(cfg.root) / "gui_settings.json").read_text(encoding="utf-8"))
    assert "dance_window" in saved, saved.keys()

    second = main_page.DanceMontageWindow(cfg)
    try:
        assert second.remix.song.text() == str(song_id)
        assert second.remix.slice_duration() == 1.5, second.remix.slice_duration()
        assert second.remix.person.text() == "小A"
        assert second.remix.recommend.isChecked() is False
        assert second.bench.source.text().endswith("girl01.mp4")
        assert abs(second.bench.at.value() - 12.5) < 1e-6
        assert abs(second.bench.slice_seconds.value() - 1.5) < 1e-6
        assert second.bench.chk_sound.isChecked(), "「带声音」的勾选没记住"
        assert second.bench.btn_loop.isChecked()
        assert second.tabs.currentIndex() == 4, second.tabs.currentIndex()
        # 分栏比例：离屏窗口没有真实宽度，Qt 会把 setSizes 缩放掉，所以这里比的是
        # 「存下来的那份和套回来的那份一致」，而不是当初写进去的字面值
        assert second.split.sizes() == saved["dance_window"]["split"], second.split.sizes()
        assert second.width() == 1320 and second.height() == 880, second.size()
        # 上次选文件去过的目录也记住了
        assert Path(dialogs.start_dir("", "dance.source")) == work
        # 当前目标歌也跟着切回来了（输入框里是库里的 id）
        assert second._song_id == song_id                   # noqa: SLF001
    finally:
        second.close()


def test_launch_installs_an_excepthook() -> None:
    """槽里抛异常必须能看见：pythonw 下没有控制台，不装钩子就是"啪一下没了"。"""
    import sys

    from vidscribe.gui import dance_montage

    original = sys.excepthook
    try:
        dance_montage._install_excepthook()        # noqa: SLF001
        assert sys.excepthook is not original, "excepthook 没装上"
        assert sys.excepthook.__name__ == "on_error", sys.excepthook
    finally:
        sys.excepthook = original


def test_every_panel_fills_its_table_with_real_rows(work: Path) -> None:
    """**每一张表都必须真的有行**，且填表代码真的跑到。

    这条测试是补窟窿的：原来的夹具只造素材，不造对齐记录和混剪版本，
    于是"对齐面板"和"历史面板"那两段填表循环一次都没执行过 ——
    界面里把 `offset_seconds` 写成 `offset`（`OFFSET` 是 SQL 关键字，建表时避开了它），
    测试全绿，一开界面就 IndexError。所以这里坚持：造行 → 刷面板 → 断言行数 > 0。
    """
    from dance_fixtures import fake_alignment, fake_library, fake_version, make_project

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    song_id, videos, materials = fake_library(db, positions=4, people=("小A", "小B"))
    for video_id in videos.values():
        fake_alignment(db, song_id, video_id, offset=1.5)
    picked = [materials[("小A", 0)], materials[("小B", 1)], materials[("小A", 2)]]
    fake_version(db, song_id, picked, rendered=True)
    db.close()

    window = main_page.DanceMontageWindow(cfg)
    try:
        window._song_id = song_id                  # noqa: SLF001
        window.reload()                            # 出错的话这里就抛了

        assert window.alignment.table.rowCount() == 2, window.alignment.table.rowCount()
        offset_cell = window.alignment.table.item(0, 1)
        assert offset_cell is not None and offset_cell.text() == "+1.500", \
            offset_cell.text() if offset_cell else "偏移那一格是空的"
        assert "共 2 条对齐" in window.alignment.summary.text(), \
            window.alignment.summary.text()

        assert window.history.versions.rowCount() == 1, window.history.versions.rowCount()
        assert window.history.events.rowCount() >= len(picked), \
            window.history.events.rowCount()
        rendered = window.history.versions.item(0, 4)
        assert rendered is not None and rendered.text() == "rendered", rendered.text()

        assert window.library.table.rowCount() == 8, window.library.table.rowCount()
        assert window.candidates.positions.count() == 4
        assert window.statistics.positions.rowCount() == 4
        assert window.recommend.strategy.count() >= 1, "策略下拉是空的"

        # 每个面板的详情按钮也得能点（它们同样会碰列名）
        window.history.versions.selectRow(0)
        assert window.history._selected_version() > 0                    # noqa: SLF001
        window.alignment.table.selectRow(0)
        assert window.alignment._selected() is not None                  # noqa: SLF001
    finally:
        window.close()


TESTS = (
    test_window_has_all_four_regions,
    test_panels_show_real_library,
    test_every_panel_fills_its_table_with_real_rows,
    test_filter_presets_reach_the_query,
    test_slice_presets,
    test_manual_selection_flows_to_remix,
    test_start_refuses_empty_manual_when_recommend_off,
    test_file_pickers_use_the_system_dialog_and_remember_the_folder,
    test_settings_survive_a_restart,
    test_launch_installs_an_excepthook,
    test_worker_is_isolated_from_analyze_worker,
    test_worker_runs_the_whole_job,
    test_worker_stops_cooperatively,
)





def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="dancegui_"))
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
