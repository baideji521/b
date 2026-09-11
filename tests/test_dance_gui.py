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

from PyQt5.QtCore import Qt                                   # noqa: E402
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


def test_the_master_audio_drives_the_realtime_row(work: Path) -> None:
    """主音频位置 → 当前 Segment → 矩阵那一列 → 实时画面；**没素材就不播**。"""
    from vidscribe.dance import segment_template as seg

    window, song_id, _m = _window(work)
    try:
        window._song_id = song_id                          # noqa: SLF001
        window.reload()
        # 12 秒的歌，3 秒一段 → S1~S4；主音频那条时间轴是唯一权威
        window.master._adopt(seg.uniform(12.0, 3.0), remember=False)   # noqa: SLF001
        window.master._duration = 12.0                     # noqa: SLF001

        window._master_moved(4.0)                          # noqa: SLF001
        assert window.matrix.current_segment == 1, window.matrix.current_segment
        window._master_moved(9.5)                          # noqa: SLF001
        assert window.matrix.current_segment == 3

        # 这一段没素材：画面保持空，绝不拿别的段顶上
        window._sync_live({}, 9.5)                         # noqa: SLF001
        assert window._live_material == 0                  # noqa: SLF001
        assert "没有素材" in window.live_note.text(), window.live_note.text()
        assert not window.live.is_playing()

        # 片段仓库那一栏说得出音频名 / 歌名 / 已生成片段
        text = window.repo_note.text()
        assert "音频名称" in text and "已生成片段" in text, text
    finally:
        window.close()


def test_segment_frames_are_predecoded_into_memory(work: Path) -> None:
    """片段预解码：**播放/定位/逐帧全部不再解码**，所以才不卡、才能核卡点。

    证明方式够狠：`preload` 之后直接把 cv2 的读写头 `release()` 掉。如果播放
    还偷偷在解码，后面的 seek / 逐帧一定拿不到画面；能照样出帧就说明帧真在内存里。

    这里会 import cv2（`FramePlayer` 自己延迟 import 的）。它改写
    QT_QPA_PLATFORM_PLUGIN_PATH 只影响**之后**创建的 QApplication，
    而本文件顶上那个 QApplication 早就建好了 —— 和真实程序里的先后顺序一致。
    """
    from dance_fixtures import song, write_video
    from vidscribe.gui.player import CACHE_EDGE, FramePlayer

    clip = write_video(work / "cache.mp4", song(duration=2.0), fps=24.0,
                       width=1080, height=1440)      # 3:4：素材基本都是这个比例
    player = FramePlayer()
    try:
        assert player.open(clip), "测试素材解不开"
        assert not player.is_cached(), "还没 preload 就说自己在放内存"
        assert player.preload(0.5, 1.5), "1 秒的片段应该缓存得下"
        begin, end = player.cached_span()
        assert abs(begin - 0.5) < 0.05 and abs(end - 1.5) < 0.1, (begin, end)
        # 缓存的帧被缩到闸门以内，否则 1080×1440 几秒就能吃掉几百 MB
        assert max(player._image.width(), player._image.height()) <= CACHE_EDGE  # noqa: SLF001

        player._cap.release()                                    # noqa: SLF001
        player.seek(1.0)
        assert abs(player.position() - 1.0) < 0.05, player.position()
        first = player.position()
        player.step_frame(1)                                     # 逐帧：往后一帧
        assert abs(player.position() - first - 1 / 24.0) < 0.01, player.position()
        player.step_frame(-1)                                    # 再回来，回到原处
        assert abs(player.position() - first) < 0.01, player.position()
        assert not player.is_playing(), "逐帧必须先停住"

        # 越界不炸也不乱跳：贴到最近的一端
        player.seek(0.0)
        assert abs(player.position() - begin) < 0.05, player.position()
    finally:
        player.close_video()
        assert not player.is_cached(), "关掉视频要把缓存放掉，不然内存只涨不降"


def test_live_window_covers_both_kinds_of_material(work: Path) -> None:
    """预解码的窗口用一条公式覆盖两种素材：库里的片段文件 / 内存里的实时格子。"""
    window, _song_id, _m = _window(work)
    try:
        # ① 库里的片段：文件本身就是这一段，base = target_start → 0 起
        begin, end = window._live_window(                        # noqa: SLF001
            {"target_start": 6.0, "target_end": 9.0})
        assert (round(begin, 3), round(end, 3)) == (0.0, 3.0), (begin, end)
        # ② 实时格子：放的是整条源视频，base = offset → source_start 起
        begin, end = window._live_window(                        # noqa: SLF001
            {"target_start": 6.0, "target_end": 9.0, "seek_base": 2.0})
        assert (round(begin, 3), round(end, 3)) == (4.0, 7.0), (begin, end)
        # ③ 缺时间信息就别猜，返回 None（外面自动退回流式播放）
        assert window._live_window({"path": "x.mp4"}) is None    # noqa: SLF001
    finally:
        window.close()


def test_window_has_all_four_regions(work: Path) -> None:


    """一期第二十四节的四个区域，面板一个都不能少。"""
    window, _song_id, _m = _window(work)
    try:
        assert window.remix is not None            # ① 输入与操作
        assert window.filters is not None and window.library is not None   # ② 素材资产
        assert window.candidates is not None and window.recommend is not None  # ③ 选择与推荐
        assert window.history is not None and window.statistics is not None    # ④ 历史与统计
        assert window.alignment is not None
        assert window.bench is not None             # 源视频对齐 / 切片入库（在编排台里）
        assert window.master is not None            # 主音频编辑区（段落模板）
        assert window.matrix is not None            # 素材矩阵（就在编排台里）
        # 一页到底：没有 Tab，编排台就是整个窗口的正面
        assert not hasattr(window, "tabs"), "编排台不该再被塞进 Tab 里"
        assert window.centralWidget().isAncestorOf(window.studio)
        # 音频对齐是随时能点到的右侧侧栏（默认收着，不挡编排台）
        # 窗口自己没 show，子控件 isVisible() 恒 False，所以看"有没有被显式收起来"
        assert window.dock_align.widget() is window.alignment
        assert window.dock_align.isHidden() is True
        window.btn_dock_align.click()
        assert window.dock_align.isHidden() is False
        window.btn_dock_align.click()
        assert window.dock_align.isHidden() is True
        # 输入与操作在左侧侧栏里
        assert window.dock_remix.widget().isAncestorOf(window.remix)
        # 其余三组走弹窗，点了才建，建完面板还是原来那几个实例
        assets = window._open_panel("assets")            # noqa: SLF001 - 测试里直接开
        assert assets.isAncestorOf(window.filters) and assets.isAncestorOf(window.library)
        assert window._open_panel("assets") is assets    # noqa: SLF001 - 只建一次
        choose = window._open_panel("choose")            # noqa: SLF001
        assert choose.isAncestorOf(window.candidates) and choose.isAncestorOf(window.recommend)
        review = window._open_panel("review")            # noqa: SLF001
        assert review.isAncestorOf(window.history) and review.isAncestorOf(window.statistics)
        for window_ in (assets, choose, review):
            window_.close()
        # 编排台一页走完：一行工具栏 → 左半边视频+音谱 / 右半边素材矩阵
        assert window.studio_split.widget(0) is window.bench
        assert window.studio_split.widget(1) is window.body_split
        stage = window.body_split.widget(0)
        assert stage.isAncestorOf(window.live), "视频画面不在音谱那一块里"
        assert stage.isAncestorOf(window.master)
        # 「视频位置」那条覆盖带已经删掉了（用户不要）
        assert not hasattr(window, "coverage")
        # 「起步参数」那一排（N 秒一段 / 等间隔起步 / 停顿推荐度 / 照人声停顿分 /
        # 保存这份分段）在界面上收起来了；控件还在，参数和 Ctrl+S 那条路不受影响
        assert window.master._segment_tools.isHidden()    # noqa: SLF001
        assert window.master.step is not None and window.master.btn_save is not None
        # 上边视频、下边音谱：拿它们在同一个布局里的纵坐标比一下
        assert window.stage_split.widget(0).isAncestorOf(window.live)
        assert window.stage_split.widget(1).isAncestorOf(window.master)
        # 右半边三块：视频列表 / 实时播放（自己单独一行）/ 段落候选矩阵
        assert window.body_split.widget(1) is window.right_split
        assert window.right_split.count() == 3, window.right_split.count()
        assert window.right_split.widget(0) is window.video_list
        assert window.right_split.widget(1) is window.matrix.realtime_box
        assert window.right_split.widget(2) is window.matrix
        # 实时播放那一块已经从矩阵里搬出来了，不该还在矩阵内部
        assert not window.matrix.isAncestorOf(window.matrix.realtime_box)
        assert window.body_split.orientation() == Qt.Horizontal
        assert window.right_split.orientation() == Qt.Vertical


        for button in (window.btn_preview_final, window.btn_save_all,
                       window.btn_export_final):
            assert button.minimumHeight() >= 38, button.text()
        assert "卡点舞" in window.windowTitle()
        # 小窗口也要能用（一期第十五节）
        assert window.minimumWidth() <= 1000 and window.minimumHeight() <= 640
    finally:
        window.close()


def test_video_and_spectrum_take_only_the_left_half(work: Path) -> None:
    """视频 + 音谱同一块、视频在上音谱在下，**而且只占左半边**，右半边留给视频列表。

    这条必须量真实像素：只看控件树的话，"列表在右边"和"列表被挤成一条缝"
    长得一模一样。主音频那两排按钮以前是 QHBoxLayout，最小宽度顶到 1300 上下，
    左半边怎么拖都缩不下来 —— 换成会折行的 `_FlowLayout` 才真的对半分。
    """
    window, _song_id, _m = _window(work)
    try:
        window.resize(1600, 1000)
        window.show()
        _app.processEvents()

        stage = window.body_split.widget(0)
        stage_x = stage.mapTo(window, stage.rect().topLeft()).x()
        right = window.video_list
        right_x = right.mapTo(window, right.rect().topLeft()).x()
        video_y = window.live.mapTo(window, window.live.rect().topLeft()).y()
        audio_y = window.master.mapTo(window, window.master.rect().topLeft()).y()

        assert video_y < audio_y, "视频得在音谱上边"
        assert right_x >= stage_x + stage.width() - 8, "视频列表没在右半边"
        share = stage.width() / max(1, window.width())
        assert 0.3 <= share <= 0.7, f"左半边占了 {share:.0%}，不是半边"
        # 左半边还得真能拖窄：最小宽度别再被那排按钮顶住
        assert stage.minimumSizeHint().width() <= 700, stage.minimumSizeHint().width()
    finally:
        window.close()


def test_the_right_list_shows_folder_videos_and_alignment_results(work: Path) -> None:
    """右侧列表：选完文件夹先列文件，对齐算完再填 Offset / 置信度，**全程不落库**。"""
    from types import SimpleNamespace

    window, _song_id, _m = _window(work)
    try:
        before = _library_counts(window)

        window.bench.folder_scanned.emit([str(work / "girl01.mp4"),
                                          str(work / "girl02.mp4")])
        rows = window.video_list.rows()
        assert [r["视频"] for r in rows] == ["girl01.mp4", "girl02.mp4"], rows
        assert all(r["状态"] == "未对齐" for r in rows), rows
        assert all(r["Offset"] == "—" for r in rows), rows
        assert "2 个视频" in window.video_list.hint.text(), window.video_list.hint.text()

        window.bench.results_ready.emit([
            {"path": str(work / "girl01.mp4"), "name": "girl01.mp4",
             "alignment": SimpleNamespace(offset=3.201, confidence=0.981,
                                          status="ok", source_duration=62.4)},
            {"path": str(work / "girl02.mp4"), "name": "girl02.mp4",
             "alignment": None, "error": "音轨解不开"},
        ])
        rows = {r["视频"]: r for r in window.video_list.rows()}
        assert rows["girl01.mp4"]["Offset"] == "+3.201", rows
        assert rows["girl01.mp4"]["置信度"] == "0.981", rows
        assert rows["girl01.mp4"]["时长"] == "62.40s", rows
        assert rows["girl01.mp4"]["状态"] == "✅ 可用", rows
        assert "失败" in rows["girl02.mp4"]["状态"], rows
        # 对齐只是算给人看：库里的素材/对齐条数一条都没变
        assert _library_counts(window) == before, (_library_counts(window), before)
        # 字号比全局 12px 小一号
        assert "font-size:11px" in window.video_list.table.styleSheet()
    finally:
        window.close()


def test_unaligned_rows_disappear_once_alignment_ran(work: Path) -> None:
    """对齐跑完，还挂着「未对齐」的行必须清掉；路径写法不同也要落到同一行。"""
    from types import SimpleNamespace

    window, _song_id, _m = _window(work)
    try:
        window.bench.folder_scanned.emit([str(work / "a.mp4"), str(work / "b.mp4")])
        assert len(window.video_list.rows()) == 2
        # 结果里的路径故意换个写法（斜杠 + 大小写）：还是同一个文件，不许另开一行
        odd = str(work / "a.mp4").replace("\\", "/").upper()
        window.bench.results_ready.emit([
            {"path": odd, "name": "a.mp4",
             "alignment": SimpleNamespace(offset=1.5, confidence=0.9,
                                         status="ok", source_duration=30.0)}])
        rows = window.video_list.rows()
        assert len(rows) == 1, rows            # b.mp4 那行「未对齐」被清掉了
        assert rows[0]["Offset"] == "+1.500", rows
        assert all(r["状态"] != "未对齐" for r in rows), rows
    finally:
        window.close()


def test_right_click_can_copy_paste_and_really_delete(work: Path) -> None:
    """右键：全选 / 复制 / 粘贴 / 删除。**删除是真删本地文件**，删完清干净。"""
    from PyQt5.QtWidgets import QMessageBox

    files = []
    for name in ("v1.mp4", "v2.mp4"):
        path = work / name
        path.write_bytes(b"not really a video")     # 右键这几个动作不解码
        files.append(str(path))

    window, _song_id, _m = _window(work)
    try:
        panel = window.video_list
        window.bench.folder_scanned.emit(files)
        panel.table.selectAll()
        assert sorted(panel.selected_paths()) == sorted(files), panel.selected_paths()

        assert panel.copy_selected() == panel.selected_paths()
        panel.drop_files(files)                     # 假装列表被清空了
        assert panel.rows() == []
        assert sorted(panel.paste_files()) == sorted(files), "粘贴没把文件加回来"
        assert len(panel.rows()) == 2

        # 删除：确认框答"是"，文件真的从磁盘上消失
        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try:
            panel.table.selectAll()
            gone = panel.delete_selected()
        finally:
            QMessageBox.warning = original
        assert sorted(gone) == sorted(files), gone
        assert all(not Path(p).exists() for p in files), "文件还在，没真删"
        assert panel.rows() == [], panel.rows()
        # 批量清单也跟着清了，否则下一轮对齐会去算不存在的文件
        assert window.bench.more.count() == 0, window.bench.more.count()
    finally:
        window.close()


def test_frames_are_decoded_off_the_gui_thread(work: Path) -> None:
    """预取线程真的在后台解片段，播放器拿现成的装上（GUI 线程一帧都不解）。"""
    import time

    from dance_fixtures import song, write_video
    from vidscribe.gui.player import FramePlayer, FramePrefetcher

    clip = write_video(work / "warm.mp4", song(duration=3.0), fps=24.0,
                       width=540, height=960)
    got: list[tuple] = []
    pump = FramePrefetcher()
    pump.ready.connect(lambda *args: got.append(args))
    try:
        pump.request(str(clip), 0.5, 1.5)
        deadline = time.monotonic() + 30.0
        while not got and time.monotonic() < deadline:
            _app.processEvents()
            time.sleep(0.02)
        assert got, "后台线程没把帧送回来"
        path, begin, end, bundle = got[0]
        assert (round(begin, 3), round(end, 3)) == (0.5, 1.5), (begin, end)
        assert len(bundle[0]) >= 20, len(bundle[0])       # 1 秒 @24fps
    finally:
        pump.shutdown()

    player = FramePlayer()
    try:
        assert player.open(clip)
        assert player.adopt_cache(path, begin, end, bundle) is True
        assert player.is_cached(), "预取的帧没装上"
        assert abs(player.cached_span()[0] - 0.5) < 0.05, player.cached_span()
        # 换了别的文件就不许收：不然画面和素材对不上
        assert player.adopt_cache(str(work / "other.mp4"), begin, end, bundle) is False
    finally:
        player.close_video()


def test_deleting_a_video_releases_the_file_first(work: Path) -> None:
    """右键删除：**先松开占用再删**。播放器还开着的话 Windows 会锁住这个文件。"""
    from PyQt5.QtWidgets import QMessageBox

    from dance_fixtures import song, write_video

    clip = write_video(work / "locked.mp4", song(duration=1.5), fps=24.0,
                       width=320, height=240)
    window, _song_id, _m = _window(work)
    try:
        window.bench.folder_scanned.emit([str(clip)])
        assert window.live.open(str(clip)), "先让播放器占着这个文件"
        assert window.live.holds(str(clip))

        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try:
            window.video_list.table.selectAll()
            gone = window.video_list.delete_selected()
        finally:
            QMessageBox.warning = original

        assert gone == [str(clip)], gone
        assert not clip.exists(), "占用没松开，文件删不掉"
        assert window.live.path() == "", "播放器还attach着已经删掉的文件"
    finally:
        window.close()


def test_changing_the_head_room_refills_the_first_column(work: Path) -> None:
    """改「首缺≤」必须**立刻**让 S1 那一列长出格子。

    第一版漏了这条连线：数值改了但没人重铺矩阵，S1 一直空着，看起来就像
    这个输入框根本没接上。
    """
    from types import SimpleNamespace

    from vidscribe.dance import segment_template as seg

    window, _song_id, _m = _window(work)
    try:
        window.master._adopt(seg.uniform(60.0, 12.0), remember=False)   # noqa: SLF001
        window.master._duration = 60.0                                  # noqa: SLF001
        window.bench.head_room.setValue(0.0)
        window.bench.tail_room.setValue(0.0)
        # offset +1、源只有 58 秒 → S1 开头缺 1 秒、S5 结尾缺 1 秒
        window._align_rows_ready([{                                     # noqa: SLF001
            "path": str(work / "v.mp4"), "name": "v.mp4",
            "alignment": SimpleNamespace(offset=1.0, confidence=0.9,
                                         status="ok", source_duration=58.0)}])
        assert (1, 1) in window.matrix.cells, "中间那几段本来就该有格子"
        assert (1, 0) not in window.matrix.cells, "余量 0 时 S1 该是空的"
        assert (1, 4) not in window.matrix.cells, "余量 0 时 S5 该是空的"

        window.bench.head_room.setValue(2.0)      # 只改这一个：S1 应该立刻出现
        assert (1, 0) in window.matrix.cells, "改了首缺余量，S1 还是空的"
        assert (1, 4) not in window.matrix.cells, "尾缺余量还是 0，S5 不该动"
        window.bench.tail_room.setValue(2.0)
        assert (1, 4) in window.matrix.cells, "改了尾缺余量，S5 还是空的"
        # 补的量如实写在格子上，别让人以为整段都有画面
        assert "补 1.00s" in window.matrix.cells[(1, 0)].payload["detail"], \
            window.matrix.cells[(1, 0)].payload["detail"]
    finally:
        window.close()


def test_slice_export_lists_only_the_realtime_row_in_order(work: Path) -> None:
    """✂ 切片导出：**只导实时播放行选中的那几段**，按段落顺序编号，带首尾补帧信息。"""
    from types import SimpleNamespace

    from vidscribe.dance import segment_template as seg

    clip = work / "v.mp4"
    clip.write_bytes(b"not a real video")      # 这一步只算清单，不解码
    window, _song_id, _m = _window(work)
    try:
        window.master._adopt(seg.uniform(60.0, 12.0), remember=False)   # noqa: SLF001
        window.master._duration = 60.0                                  # noqa: SLF001
        window.bench.head_room.setValue(2.0)
        window.bench.tail_room.setValue(2.0)
        window._align_rows_ready([{                                     # noqa: SLF001
            "path": str(clip), "name": "v.mp4",
            "alignment": SimpleNamespace(offset=1.0, confidence=0.9,
                                         status="ok", source_duration=58.0)}])
        # 一段都没选时不许导
        assert window._slice_items()[0] == []                           # noqa: SLF001

        assert window.matrix.choose(0, -1001) is True                   # S1（开头缺 1s）
        assert window.matrix.choose(2, -1003) is True                   # S3（完整）
        items, skipped = window._slice_items()                          # noqa: SLF001
        assert skipped == [], skipped
        assert [i["name"] for i in items] == ["01_S01_v_mp4.mp4", "02_S03_v_mp4.mp4"], items
        assert [i["segment_index"] for i in items] == [0, 2], items
        # S1 首尾补帧：开头缺 1 秒、结尾不缺
        assert (items[0]["head_pad"], items[0]["tail_pad"]) == (1.0, 0.0), items[0]
        assert (items[0]["source_start"], items[0]["source_end"]) == (0.0, 11.0), items[0]
        # S3 完整：一点都不用补
        assert (items[1]["head_pad"], items[1]["tail_pad"]) == (0.0, 0.0), items[1]
        # 每一条的"源那截 + 补的" 必须正好等于段落长度 12 秒
        for item in items:
            span = item["source_end"] - item["source_start"]
            assert abs(span + item["head_pad"] + item["tail_pad"] - 12.0) < 1e-6, item
    finally:
        window.close()


def _library_counts(window) -> tuple[int, int]:
    """(素材数, 对齐数)。用来钉住"对齐不入库"。"""
    cursor = window.db.execute("SELECT COUNT(*) FROM dance_materials")
    materials = int(cursor.fetchone()[0])
    cursor = window.db.execute("SELECT COUNT(*) FROM dance_audio_alignments")
    return materials, int(cursor.fetchone()[0])


def test_the_right_matrix_is_live_and_follows_the_cuts(work: Path) -> None:
    """右侧矩阵**入库之前就有内容**，列数跟着音谱上的分段走，越界的格子留空。

    每一格按对齐那层唯一的换算现算：`源时间 = 目标时间 − offset`。
    整段落不进源视频就不给格子（不偷偷 clamp —— 那会让画面和音乐错开）。
    """
    from types import SimpleNamespace

    window, _song_id, _m = _window(work)
    try:
        window.master._duration = 12.0            # noqa: SLF001 - 假装主音频解过了
        window.remix.slice_preset.setCurrentIndex(2)   # 2.0 秒一段 → 6 段
        window.bench.results_ready.emit([
            # 整条 60 秒，offset 3：0→12s 这一段整段都在它里面
            {"path": str(work / "girl01.mp4"), "name": "girl01.mp4",
             "alignment": SimpleNamespace(offset=3.0, confidence=0.9, status="ok",
                                          source_duration=60.0)},
            # 只有 5 秒，offset 0：后面几段落在视频外面，那些格子必须空着
            {"path": str(work / "girl02.mp4"), "name": "girl02.mp4",
             "alignment": SimpleNamespace(offset=0.0, confidence=0.8, status="ok",
                                          source_duration=5.0)},
        ])

        assert len(window.matrix.segments) == 6, len(window.matrix.segments)
        assert window.matrix.db is None, "实时格子不该挂着库连接（免得写进去）"
        # 第一段 0→2s：长视频 offset=3 会算出源 −3→−1（负的）→ 拒绝，不给格子；
        # 短视频 offset=0 算出源 0→2，在 5 秒之内 → 有格子
        first = window.matrix.segments[0]["materials"]
        assert [row["video_name"] for row in first] == ["girl02.mp4"], first
        assert first[0]["source_start"] == 0.0 and first[0]["seek_base"] == 0.0, first
        assert int(first[0]["material_id"]) < 0, "内存格子的 id 该是负数"
        # 中间那段 4→6s：长视频算出源 1→3，落在里面 → 有格子（源时间 = 目标 − offset）
        middle = {row["video_name"]: row for row in window.matrix.segments[2]["materials"]}
        assert abs(middle["girl01.mp4"]["source_start"] - 1.0) < 1e-9, middle
        assert abs(middle["girl01.mp4"]["seek_base"] - 3.0) < 1e-9, middle
        # 最后一段（10→12s）：短视频进不去，只剩长视频那一格
        last = window.matrix.segments[-1]["materials"]
        assert [row["video_name"] for row in last] == ["girl01.mp4"], last

        # 切一刀就多一列：这里直接改分段数（模拟 ✂ 切分之后 template 变了）
        window.remix.slice_preset.setCurrentIndex(0)   # 1.0 秒一段 → 12 段
        window._reload_matrix()                        # noqa: SLF001
        assert len(window.matrix.segments) == 12, len(window.matrix.segments)
    finally:
        window.close()


def test_live_cells_can_be_clicked_and_dragged_into_the_realtime_row(work: Path) -> None:
    """内存里的实时格子**照样能点、能拖进「实时播放」行**。

    这条是补一个真窟窿：之前只检查了生成出来的格子数据，没有真去点/拖一下，
    结果两处"只认正数 id / 少了 segment_index"的判定把实时格子全拒了 ——
    界面上表现就是"下边的片段拉不到实时播放里边"。
    """
    from types import SimpleNamespace

    from PyQt5.QtGui import QDragEnterEvent
    from PyQt5.QtCore import QPoint

    from vidscribe.gui.dance_montage import matrix_panel as mx

    window, _song_id, _m = _window(work)
    try:
        window.master._duration = 12.0            # noqa: SLF001
        window.remix.slice_preset.setCurrentIndex(2)      # 2 秒一段
        window.bench.results_ready.emit([
            {"path": str(work / f"{name}.mp4"), "name": f"{name}.mp4",
             "alignment": SimpleNamespace(offset=0.0, confidence=0.9, status="ok",
                                          source_duration=60.0)}
            for name in ("girl01", "girl02")
        ])
        first = window.matrix.segments[0]["materials"]
        assert len(first) == 2, first
        cells = [window.matrix.cells[(row["video_id"], 0)] for row in first]
        target = window.matrix.realtime[0]

        # ① 点一下就选上，而且这一格变绿（"被采用"要看得见）
        window.matrix._cell_clicked(dict(cells[0].payload))    # noqa: SLF001
        assert target.material_id == cells[0].material_id, target.material_id
        assert window.matrix.picks()[0] == cells[0].material_id
        assert cells[0].is_chosen() is True
        assert cells[1].is_chosen() is False

        # ② 同一列另一格拖上来就换人（业务层的落格判定），颜色也跟着换
        assert target.drop_payload(dict(cells[1].payload)) is True
        assert target.material_id == cells[1].material_id
        assert cells[1].is_chosen() is True
        assert cells[0].is_chosen() is False, "被替换掉的那一格该回到原色"

        # ③ 真的拖拽事件也认（dragEnter 必须 accept，否则光标是禁止符号、松手没反应）
        # mime 必须**先存进变量**：内联传进 QDragEnterEvent 的话 Python 这边引用计数
        # 立刻归零，Qt 那边还握着指针 —— 进程直接 access violation 崩掉
        data = mx.pack(dict(cells[0].payload))
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, data,
                                Qt.LeftButton, Qt.NoModifier)
        target.dragEnterEvent(enter)
        assert enter.isAccepted(), "实时格子被拒了，界面上就是「拖不进去」"

        # ④ 跨列照旧拒绝：素材和音乐位置是绑死的（这一格还是 ② 摆上去的那条）
        other = window.matrix.cells[(first[0]["video_id"], 2)]
        assert target.drop_payload(dict(other.payload)) is False
        assert target.material_id == cells[1].material_id
    finally:
        window.close()


def test_playback_broadcasts_the_position_so_the_page_follows(work: Path) -> None:
    """主音频**一边播一边广播位置**，右边矩阵和左边画面才跟得上。

    以前位置只在手动拖动时广播（`_moved_to`），播放回调 `_position_changed`
    只更新自己那一块 —— 于是一按播放，矩阵的当前列不走、左边画面也不动。
    """
    window, _song_id, _m = _window(work)
    try:
        seen: list[float] = []
        window.master.seek_requested.connect(seen.append)
        window.master._position_changed(1500)          # noqa: SLF001 - 假装播到 1.5s
        assert seen and abs(seen[-1] - 1.5) < 1e-6, seen
        window.master._position_changed(3250)          # noqa: SLF001
        assert abs(seen[-1] - 3.25) < 1e-6, seen
    finally:
        window.close()


def test_the_left_player_follows_the_master_audio(work: Path) -> None:
    """实时播放行摆了谁，主音频走到哪儿，左边画面就播那一段的那个位置。

    这条把整条链一次走完：位置广播 → 当前是第几段 → 那一格的素材 → 打开+定位。
    两个真 bug 曾经卡在这条链上：
      · 「当前是第几段」只问 `master.template`，没切过分段时永远是 −1；
      · `FramePlayer.position` 是方法不是属性，`float(方法)` 抛 TypeError，
        而这是在槽里 —— PyQt 直接把进程干掉，界面上就是"按了播放没反应"。
    """
    from types import SimpleNamespace

    window, _song_id, _m = _window(work)
    try:
        window.master._duration = 12.0            # noqa: SLF001
        window.remix.slice_preset.setCurrentIndex(2)      # 2 秒一段 → 6 段
        window.bench.results_ready.emit([
            {"path": str(work / "girl01.mp4"), "name": "girl01.mp4",
             "alignment": SimpleNamespace(offset=0.0, confidence=0.9, status="ok",
                                          source_duration=60.0)},
        ])
        cell = window.matrix.cells[(1, 1)]        # 视频 1 在 S2（2→4s）那一格
        window.matrix._cell_clicked(dict(cell.payload))   # noqa: SLF001
        assert window.matrix.picks() == {1: cell.material_id}

        calls: list[str] = []
        window.live.open = lambda path: (calls.append(f"open:{Path(path).name}"), True)[1]
        window.live.seek = lambda seconds: calls.append(f"seek:{seconds:.3f}")
        window.live.play = lambda: calls.append("play")
        window.live.pause = lambda: calls.append("pause")
        window.live.close_video = lambda: calls.append("close")
        window.live.position = lambda: 0.0
        window.live.set_audio_enabled = lambda _on: None

        window.master.seek_requested.emit(3.0)   # 主音频走到 3.0s，落在 S2
        assert window.matrix.current_segment == 1, window.matrix.current_segment
        assert window._live_material == cell.material_id   # noqa: SLF001
        assert "open:girl01.mp4" in calls, calls
        # 段落内位置 = 3.0 − seek_base(0.0)；offset 只减一次
        assert "seek:3.000" in calls, calls
        assert "素材" in window.live_note.text()

        # 走到没选素材的那一段 → 画面清空，不拿别的段顶替
        window.master.seek_requested.emit(9.0)
        assert window._live_material == 0          # noqa: SLF001
        assert "close" in calls, calls
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
        first.studio_split.setSizes([120, 900])
        first.body_split.setSizes([700, 820])
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
        first.dock_align.setVisible(True)      # 侧栏开着，下次开窗口该还是开着
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
        assert second.dock_align.isHidden() is False, "侧栏开着的状态没记住"
        # 分栏比例：离屏窗口没有真实尺寸，Qt 会按控件大小重新缩放 setSizes，
        # 两次开窗口的可用尺寸还可能不一样。所以这里只钉住"每一块都还在、比例大致一致"，
        # 不比字面值 —— 比字面值测的是 Qt 的缩放实现，不是我们存没存对
        live = second.studio_split.sizes()
        kept = saved["dance_window"]["studio_split"]
        assert len(live) == 2 and all(v >= 0 for v in live), live
        assert sum(live) > 0 and sum(kept) > 0, (live, kept)
        assert abs(live[-1] / sum(live) - kept[-1] / sum(kept)) < 0.12, (live, kept)
        # 左半边（视频+音谱）和右半边（矩阵）的宽度比例也记住了
        live = second.body_split.sizes()
        kept = saved["dance_window"]["body_split"]
        assert len(live) == 2 and sum(live) > 0, live
        assert abs(live[0] / sum(live) - kept[0] / sum(kept)) < 0.12, (live, kept)
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


def test_the_main_window_cannot_wipe_what_the_studio_just_saved(work: Path) -> None:
    """主界面和卡点舞共用一个 `gui_settings.json`，**各自只写自己名下的键**。

    以前两边都是启动时读整份、退出时写整份：主界面那份快照是开卡点舞之前读的，
    它一落盘就把卡点舞后写的 `dance_window` 按老快照抹回去 ——
    选了「手动切分」关掉窗口，再开就又变成「等间切分」，就是这么丢的。
    """
    from types import SimpleNamespace

    from vidscribe.gui import main_window as mw
    from vidscribe.gui import settings as gui_settings

    cfg, db = make_project(work)
    cfg.ensure_dance_dirs()
    db.close()

    first = main_page.DanceMontageWindow(cfg)
    try:
        assert first.master.mode_key() == "uniform"
    finally:
        first.close()
    # 主界面就是在这一刻开的：它手里这份快照里，分段方式还是"等间切分"
    stale = gui_settings.load(cfg)
    assert stale["dance_window"]["master"]["mode"] == "uniform"

    second = main_page.DanceMontageWindow(cfg)
    try:
        second.master.mode.setCurrentIndex(second.master.mode.findData("manual"))
    finally:
        second.close()
    assert gui_settings.load(cfg)["dance_window"]["master"]["mode"] == "manual"

    # 主界面这才退出，用的还是那份老快照
    class _Box:
        def __init__(self, value: object = 0) -> None:
            self._v = value

        def currentData(self) -> object:
            return self._v

        def currentIndex(self) -> int:
            return 0

        def value(self) -> float:
            return 0.5

        def isChecked(self) -> bool:
            return False

    rect = SimpleNamespace(x=lambda: 0, y=lambda: 0,
                           width=lambda: 800, height=lambda: 600)
    fake = SimpleNamespace(
        cfg=cfg, settings=stale, _loading_settings=False,
        isMaximized=lambda: False, geometry=lambda: rect, normalGeometry=lambda: rect,
        cmb_model=_Box("qwen"), cmb_speaker=_Box("ecapa"), cmb_importance=_Box(),
        spin_conf=_Box(), chk_sound=_Box(), chk_auto_ai=_Box(),
        chk_auto_translate=_Box(), chk_emotion_audio=_Box(), chk_emotion_visual=_Box(),
        export_dir=None, video_path=None,
        table=SimpleNamespace(columnCount=lambda: 0, columnWidth=lambda _c: 0,
                              verticalHeader=lambda: SimpleNamespace(
                                  defaultSectionSize=lambda: 24)),
        _highlight_offsets=(), _bridge_token="", _splitters=lambda: (),
    )
    mw.MainWindow.save_settings(fake)

    saved = gui_settings.load(cfg)
    assert saved["dance_window"]["master"]["mode"] == "manual", \
        "主界面落盘把卡点舞刚存的分段方式抹掉了"
    assert saved["visual_model"] == "qwen", "主界面自己那几个键还是要存下来"

    third = main_page.DanceMontageWindow(cfg)
    try:
        assert third.master.mode_key() == "manual", "重开界面又回到等间切分了"
    finally:
        third.close()


def test_the_video_list_numbers_every_row_from_one(work: Path) -> None:
    """视频列表第一列是**屏幕上的序号**：1..N 连号，换排序 / 删行之后重新数。

    序号不是身份 —— 行的真身还是那格里存的全路径，所以排序之后
    对齐结果照旧落在自己那一行上。
    """
    from vidscribe.gui.dance_montage import video_list as vl

    window, _song_id, _m = _window(work)
    try:
        panel = window.video_list
        assert vl.COLUMNS[0] == "#", vl.COLUMNS
        window.bench.folder_scanned.emit([str(work / "girl03.mp4"),
                                          str(work / "girl01.mp4"),
                                          str(work / "girl02.mp4")])
        rows = panel.rows()
        assert [r["#"] for r in rows] == ["1", "2", "3"], rows
        # 默认按文件名升序，序号跟着屏幕顺序走
        assert [r["视频"] for r in rows] == ["girl01.mp4", "girl02.mp4", "girl03.mp4"], rows

        panel.table.sortItems(vl.COL_NAME, Qt.DescendingOrder)
        rows = panel.rows()
        assert [r["视频"] for r in rows] == ["girl03.mp4", "girl02.mp4", "girl01.mp4"], rows
        assert [r["#"] for r in rows] == ["1", "2", "3"], "换了排序序号没重新数"

        panel.drop_files([str(work / "girl02.mp4")])
        rows = panel.rows()
        assert [r["#"] for r in rows] == ["1", "2"], rows
        assert [r["视频"] for r in rows] == ["girl03.mp4", "girl01.mp4"], rows
    finally:
        window.close()


TESTS = (


    test_the_master_audio_drives_the_realtime_row,
    test_segment_frames_are_predecoded_into_memory,
    test_live_window_covers_both_kinds_of_material,
    test_unaligned_rows_disappear_once_alignment_ran,
    test_the_video_list_numbers_every_row_from_one,
    test_right_click_can_copy_paste_and_really_delete,
    test_changing_the_head_room_refills_the_first_column,
    test_slice_export_lists_only_the_realtime_row_in_order,
    test_frames_are_decoded_off_the_gui_thread,
    test_deleting_a_video_releases_the_file_first,
    test_window_has_all_four_regions,
    test_panels_show_real_library,
    test_every_panel_fills_its_table_with_real_rows,
    test_filter_presets_reach_the_query,
    test_slice_presets,
    test_manual_selection_flows_to_remix,
    test_start_refuses_empty_manual_when_recommend_off,
    test_file_pickers_use_the_system_dialog_and_remember_the_folder,
    test_settings_survive_a_restart,
    test_the_main_window_cannot_wipe_what_the_studio_just_saved,
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
