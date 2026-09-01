"""视频资产中心 UX 守卫（Phase 15）。

Phase 15 收的是「看不懂 / 被遮盖 / 没有右键」这三件事，所以这一组测试盯的是
**信息架构、可操作性、以及小窗口下不许有控件跑出窗口**：

  T1   资产中心能正常打开
  T2   非模态
  T3   视频列表存在（第一层索引）
  T4   当前视频标题存在，而且是右侧最大的字
  T5   高光 JSON 区存在
  T6   成品区存在
  T7   唯一 Primary 是「直接剪辑」
  T8   右键视频有菜单
  T9   右键高光 JSON 有菜单，且按状态给动作
  T10  右键成品有菜单
  T11  右键 PRM 有菜单，默认那份不再显示「设为默认」
  T12  双击 JSON = 查看
  T13  双击成品 = 打开成品
  T14  血缘里的高光 JSON 节点能定位 JSON 行
  T15  血缘里的成品节点能定位成品行；PRM 节点能切到 PRM 页
  T16  当前 JSON 有明显的 ★
  T17  三层区间三行全部可见（多段也不许被截）
  T18  小窗口 / 放大字体（DPI 代理）下没有控件跑出窗口
  T19  主界面只会有一个资产中心实例
  T20  GUI 里一句裸 SQL 都没有

真的建 Qt 控件（offscreen），用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_asset_center_ux.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, QRect, Qt  # noqa: E402
from PyQt5.QtGui import QFontMetrics  # noqa: E402
from PyQt5.QtWidgets import (QApplication, QDialog, QGroupBox, QMenu,  # noqa: E402
                             QMessageBox, QPushButton, QWidget)

from vidscribe.db import assets as db_assets  # noqa: E402
from vidscribe.gui import assets_dialog as ad  # noqa: E402

from test_highlight_assets import make_project, video_row  # noqa: E402

GUI_DIR = ROOT / "src" / "vidscribe" / "gui"
PANEL = (GUI_DIR / "assets_dialog.py").read_text(encoding="utf-8")

APP: QApplication | None = None
MENUS: list[QMenu] = []


def app() -> QApplication:
    global APP
    if APP is None:
        APP = QApplication.instance() or QApplication(sys.argv[:1])
    return APP


def quiet() -> None:
    """弹窗自动点掉 + 右键菜单不真的阻塞，只把菜单记下来给断言看。"""
    ad.QMessageBox.information = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMenu.exec_ = lambda self, *a, **k: MENUS.append(self)
    # PRM 的「新增 / 修改」弹窗是模态的：offscreen 下默认按「取消」，不许阻塞
    ad.PrmEditDialog.exec_ = lambda self: QDialog.Rejected



def payload(start: float = 8.23, end: float = 23.49, score: float = 0.91,
            video: str = "demo.mp4") -> str:
    return json.dumps({"video": video,
                       "clip": {"start": start, "end": end, "score": score,
                                "type": "hook", "reason": "很炸", "evaluation": "好笑"}})


class FakeWindow(QWidget):
    """替身主界面：只提供 `render_asset`，不做分析、不碰 AI。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def render_asset(self, asset_id, prm_id=None):
        self.calls.append((int(asset_id), prm_id))
        return True


def center(tmp_path: Path, *, videos: int = 1, assets: int = 2, products: int = 1):
    """临时库 + 真的建一个资产中心，可选顺手造几个成品。"""
    app()
    quiet()
    MENUS.clear()
    cfg, db = make_project(tmp_path)
    made = []
    for index in range(videos):
        video, vid = video_row(cfg, db, f"ux_{index}.mp4")
        ids = [db_assets.create_asset(db, vid, payload(video=video.name),
                                      name=f"JSON {n}", source_type="ai",
                                      provider="Gemini" if n == 0 else None,
                                      model="Gemini Flash" if n == 0 else None)
               for n in range(assets)]
        made.append((video, vid, ids))
    prm_path = cfg.root / "prm" / "rules.txt"
    prm_path.parent.mkdir(parents=True, exist_ok=True)
    prm_path.write_text("剪辑规则", encoding="utf-8")
    prm_id = db_assets.create_prm(db, "PRM V1", str(prm_path), language="zh", version="v1")
    db_assets.set_default_prm(db, prm_id)
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(products):
        target = out_dir / f"done_{index}.mp4"
        target.write_bytes(b"mp4")
        db_assets.record_product(db, made[0][1], target,
                                 specs=[{"start": 8.23, "end": 19.39, "duration": 11.16}],
                                 asset_id=made[0][2][-1], prm_id=prm_id)
    window = FakeWindow()
    view = ad.AssetCenter(cfg, window, log=lambda _text: None)
    view.reload()
    view.select_video(made[0][1])
    return cfg, db, made, prm_id, window, view


def texts(menu: QMenu) -> list[str]:
    return [a.text() for a in menu.actions() if a.text()]


def col(table, row: int, index: int) -> str:
    item = table.item(row, index)
    return "" if item is None else item.text()


def group_title(widget: QWidget) -> str:
    """往上找最近的 GroupBox 标题（层级标号就写在这些标题里）。"""
    node = widget.parent()
    while node is not None:
        if isinstance(node, QGroupBox):
            return node.title()
        node = node.parent()
    return ""


def right_click(table, row: int, column: int = 1) -> QPoint:
    """算出某一行在表格视口里的坐标 —— 右键必须落在这一行上，不能落到别行。"""
    item = table.item(row, column)
    assert item is not None, f"第 {row} 行没有内容，右键无从下手"
    return table.visualItemRect(item).center()


def asset_rows(table) -> dict[int, int]:
    """JSON 表：`高光 JSON id → 行号`（id 藏在第二列的 UserRole 里）。"""
    out: dict[int, int] = {}
    for line in range(table.rowCount()):
        item = table.item(line, 1)
        if item is not None and item.data(Qt.UserRole) is not None:
            out[int(item.data(Qt.UserRole))] = line
    return out


# ------------------------------------------------------------------ T1 / T2
def test_center_opens(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    assert view.isWindow(), "资产中心必须是顶层窗口"
    assert view.tabs.count() == 2, "两页结构变了"
    assert "①" in view.lbl_steps.text() and "右键" in view.lbl_tip.text(), \
        "顶部要有层级提示和右键提示"


def test_center_is_not_modal(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    assert not view.isModal(), "资产中心不能是模态"


# ------------------------------------------------------------------ T3～T6
def test_video_list_exists(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, videos=3)
    assert view.tbl_videos.rowCount() == 3, "视频列表要把视频都列出来"
    assert "①" in group_title(view.tbl_videos), "视频库要标成第一层"


def test_current_video_title_exists(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path)
    page = view.videos
    assert made[0][0].name in page.lbl_video.text(), "当前视频标题要写清是哪个视频"
    # 字号改成用 QSS 定（Phase 16：主题里的 `QWidget{font-size:12px}` 会盖掉 pointSize），
    # 所以按真实渲染高度比大小，断言本身不变：当前视频必须是右侧最大的字
    assert QFontMetrics(page.lbl_video.font()).height() > \
        QFontMetrics(page.lbl_state.font()).height(), "当前视频必须是右侧最大的字"
    assert "②" in page.lbl_step.text(), "当前视频要标成第二层"


def test_json_area_exists(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path)
    assert view.tbl_assets.rowCount() == len(made[0][2]), "JSON 区要跟着视频列出 JSON"
    assert "③" in group_title(view.tbl_assets), "JSON 要标成第三层"


def test_product_area_exists(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path, products=2)
    assert view.tbl_products.rowCount() == 2, "成品区要把成品列出来"
    assert "④" in group_title(view.tbl_products), "成品要标成第四层"
    assert view.videos.tree_lineage is not None, "血缘树要在"


# ------------------------------------------------------------------ T7
def test_only_one_primary(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    bold = [b.text() for b in page.findChildren(QPushButton) if b.font().bold()]
    assert bold == ["直接剪辑"], f"唯一 Primary 必须是「直接剪辑」：{bold}"
    # 一级按钮全在「③ 高光 JSON」弹窗里（视频页自己一个按钮都不摆）
    page.on_open_video()
    app().processEvents()
    shown = {b.text() for b in page.findChildren(QPushButton)
             if b.window() is page.dlg_json and b.isVisibleTo(page.dlg_json)}
    assert shown == {"直接剪辑", "查看", "更多 ▾"}, f"一级操作最多三个：{shown}"
    assert not [b.text() for b in page.findChildren(QPushButton) if b.window() is view], \
        "视频页上不该再摆按钮，全部收进右键菜单和弹窗"
    page.dlg_json.close()


# ------------------------------------------------------------------ T8
def test_video_context_menu(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    MENUS.clear()
    view.videos.on_video_menu(QPoint(5, 5))
    assert MENUS, "右键视频必须弹菜单"
    items = " ｜ ".join(texts(MENUS[-1]))
    for needed in ("播放视频",
                   "只看这个视频的高光", "只看这个视频的成品", "看全部视频",
                   "打开所在文件夹", "复制视频路径", "从库里删除",
                   "删除该视频（包含本地文件）"):
        assert needed in items, f"右键视频缺 {needed}：{items}"
    # 「查看高光 JSON」「查看成品与血缘」改成点那一行的 JSON / 成品格子，不再进菜单
    for gone in ("重新分析", "直接剪辑", "查看高光 JSON", "查看成品与血缘"):
        assert gone not in items, f"右键视频不该还有「{gone}」：{items}"


# ------------------------------------------------------------------ T9
def test_json_context_menu(tmp_path: Path) -> None:
    _cfg, db, made, _prm, _win, view = center(tmp_path, assets=2)
    page = view.videos
    _video, vid, ids = made[0]
    db_assets.set_current_asset(db, ids[0])
    page.refresh_assets()

    rows = asset_rows(page.tbl_assets)
    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[0]]))   # 当前 JSON
    items = texts(MENUS[-1])
    assert "★ 当前 JSON" in items and "设为当前" not in items, f"当前 JSON 菜单不对：{items}"
    for needed in ("查看", "直接剪辑", "复制 JSON", "导入 JSON…", "显示原文",
                   "复制原文", "删除"):
        assert needed in items, f"右键 JSON 缺 {needed}：{items}"

    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[1]]))   # 普通 JSON
    items = texts(MENUS[-1])
    assert "设为当前" in items and "删除" in items and "恢复" not in items, \
        f"普通 JSON 菜单不对：{items}"

    db_assets.delete_asset(db, ids[1])        # 已删除的 JSON：只给「恢复」
    page.act_deleted.setChecked(True)
    page.chk_deleted.setChecked(True)
    page.refresh_assets()
    rows = asset_rows(page.tbl_assets)
    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[1]]))
    items = texts(MENUS[-1])
    assert "恢复" in items and "删除" not in items, f"已删除 JSON 菜单不对：{items}"


# ------------------------------------------------------------------ T10
def test_product_context_menu(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    page.tbl_products.selectRow(0)
    MENUS.clear()
    page.on_product_menu(QPoint(5, 5))
    assert MENUS, "右键成品必须弹菜单"
    items = texts(MENUS[-1])
    for needed in ("打开成品", "打开所在文件夹", "查看来源 JSON", "查看 PRM",
                   "查看完整血缘", "复制血缘", "复制文件路径"):
        assert needed in items, f"右键成品缺 {needed}：{items}"
    source = [a for a in MENUS[-1].actions() if a.text() == "查看来源 JSON"][0]
    prm = [a for a in MENUS[-1].actions() if a.text() == "查看 PRM"][0]
    assert source.isEnabled() and prm.isEnabled(), "这个成品记了来源 JSON 和 PRM，不该灰着"


def test_product_menu_navigates(tmp_path: Path) -> None:
    """右键「查看来源 JSON」跳 JSON 行；「查看 PRM」切到 PRM 页并选中那一份。"""
    _cfg, _db, made, prm_id, _win, view = center(tmp_path, assets=2)
    page = view.videos
    _video, _vid, ids = made[0]
    page.select_asset(ids[0])
    page.tbl_products.selectRow(0)
    page.on_show_source_json()
    assert view.selected_asset() == ids[-1], "查看来源 JSON 要跳到成品对应的那一份"
    page.on_show_prm()
    assert view.tabs.currentIndex() == 1, "查看 PRM 要切到 PRM 页"
    assert view.prm_panel.selected() == prm_id, "查看 PRM 要选中对应那一份"


# ------------------------------------------------------------------ T11
def test_prm_context_menu(tmp_path: Path) -> None:
    _cfg, db, _made, prm_id, _win, view = center(tmp_path)
    prm = view.prm_panel
    prm.reload()
    prm.select(prm_id)
    MENUS.clear()
    prm.on_menu(QPoint(5, 5))
    assert MENUS, "右键 PRM 必须弹菜单"
    items = texts(MENUS[-1])
    for needed in ("修改", "新增", "复制", "删除", "复制提示词正文", "打开提示词文件"):
        assert needed in items, f"右键 PRM 缺 {needed}：{items}"
    assert "★ 默认 PRM" in items and "设为默认" not in items, \
        f"默认那份不该再显示「设为默认」：{items}"
    db_assets.delete_prm(db, prm_id)
    prm.chk_all.setChecked(True)
    prm.reload()
    prm.select(prm_id)
    MENUS.clear()
    prm.on_menu(QPoint(5, 5))
    items = texts(MENUS[-1])
    assert "恢复" in items and "删除" not in items, f"已删除的 PRM 该显示恢复：{items}"


# ------------------------------------------------------------------ T12 / T13
def test_double_click_json_is_view(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2)
    page = view.videos
    _video, _vid, ids = made[0]
    page.json_panel.clear()
    assert page.json_panel.asset_id is None
    page.tbl_assets.selectRow(0)
    index = page.tbl_assets.model().index(0, 1)
    page.tbl_assets.doubleClicked.emit(index)
    assert page.json_panel.asset_id in ids, "双击 JSON 必须等价于「查看」"


def test_double_click_product_opens(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    opened: list[int] = []
    page.on_open_product = lambda: opened.append(1)      # 别真的调起系统播放器
    page.tbl_products.doubleClicked.disconnect()
    page.tbl_products.doubleClicked.connect(lambda _=None: page.on_open_product())
    page.tbl_products.selectRow(0)
    page.tbl_products.doubleClicked.emit(page.tbl_products.model().index(0, 1))
    assert opened, "双击成品必须等价于「打开成品」"
    assert "doubleClicked.connect(lambda _=None: self.on_open_product())" in PANEL, \
        "成品双击必须连到 on_open_product"


# ------------------------------------------------------------------ T14 / T15
def test_lineage_locates_json_and_product(tmp_path: Path) -> None:
    _cfg, _db, made, prm_id, _win, view = center(tmp_path, assets=2)
    page = view.videos
    _video, _vid, ids = made[0]
    page.tbl_products.selectRow(0)
    page.refresh_lineage()
    root = view.tree_lineage.topLevelItem(0)
    nodes = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    assert {"高光 JSON", "Clip Engine", "PRM", "实际成品"} <= set(nodes), \
        f"血缘树缺环节：{list(nodes)}"

    page.select_asset(ids[0])
    page.on_lineage_clicked(nodes["高光 JSON"], 0)
    assert view.selected_asset() == ids[-1], "点血缘里的 JSON 要定位到那一行"

    page.tbl_products.selectRow(0)
    root = view.tree_lineage.topLevelItem(0)
    nodes = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    page.on_lineage_clicked(nodes["实际成品"], 0)
    assert page.selected_product() is not None, "点血缘里的成品要选中成品行"

    root = view.tree_lineage.topLevelItem(0)
    nodes = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    page.on_lineage_clicked(nodes["PRM"], 0)
    assert view.tabs.currentIndex() == 1 and view.prm_panel.selected() == prm_id, \
        "点血缘里的 PRM 要切到 PRM 页并选中它"


# ------------------------------------------------------------------ T16
def test_current_json_is_starred(tmp_path: Path) -> None:
    _cfg, db, made, _prm, _win, view = center(tmp_path, assets=3)
    _video, vid, ids = made[0]
    db_assets.set_current_asset(db, ids[1])
    view.refresh_assets()
    marks = {col(view.tbl_assets, line, 1): col(view.tbl_assets, line, 0)
             for line in range(view.tbl_assets.rowCount())}
    starred = [name for name, mark in marks.items() if "★" in mark]
    assert starred == [f"高光 JSON #{ids[1]}"], f"当前 JSON 的 ★ 不对：{marks}"
    assert "★" in view.lbl_current.text(), "标题栏也要说清当前是哪一份"
    # 「成品」列在 Phase 16 里挪到了第 9 列（中间插了 名称 / AI / 模型），断言不变
    made_col = [col(view.tbl_assets, line, 9) for line in range(view.tbl_assets.rowCount())]
    assert any("✓ 已生成" in text for text in made_col) and \
        any("未剪辑" in text for text in made_col), f"剪没剪过要一眼看出来：{made_col}"


# ------------------------------------------------------------------ T17
def test_layer_grid_always_visible(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=1)
    page = view.videos
    page.select_asset(made[0][2][0])
    grid = view.json_panel.tbl_layers
    assert grid.rowCount() == 3, f"三层区间要三行：{grid.rowCount()}"
    layers = [col(grid, line, 0) for line in range(3)]
    assert layers == ["AI 原始", "Clip Engine", "实际渲染"], f"层级不对：{layers}"
    assert grid.maximumHeight() >= grid.rowCount() * 24, "网格高度必须放得下所有层级"
    assert col(grid, 2, 4) in ("还没剪", "✓ 一致", "⚠ 不一致"), "实际渲染那一格要给结论"
    assert "Engine" in view.json_panel.lbl_engine.text(), "结论 / 原因要写出来"


# ------------------------------------------------------------------ T18
def test_nothing_overflows_small_windows(tmp_path: Path) -> None:
    """各档小屏尺寸 + 放大字体（DPI 代理）下，关键控件都还完整落在窗口里。"""
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2)
    page = view.videos
    application = app()
    view.show()
    application.processEvents()
    watched = (("视频表", page.tbl_videos),)
    # 详情控件现在住在两个弹窗里，单独按弹窗量（主窗口只管视频列表）
    page.on_open_video()
    page.on_focus_products()
    application.processEvents()
    popups = ((page.dlg_json, (("JSON 表", page.tbl_assets),
                               ("三层区间", view.json_panel.tbl_layers),
                               ("直接剪辑", page.btn_render), ("查看", page.btn_view),
                               ("更多", page.btn_more), ("当前视频", page.lbl_video))),
              (page.dlg_products, (("成品表", page.tbl_products),
                                   ("血缘树", page.tree_lineage))))
    base = application.font()
    try:
        for scale in (1.0, 1.25, 1.5):          # 125% / 150% DPI 的代理：字体放大
            font = application.font()
            font.setPointSizeF(max(6.0, base.pointSizeF() * scale))
            application.setFont(font)
            application.processEvents()
            for width, height in ((1000, 620), (1180, 760), (1280, 720),
                                  (1366, 768), (1920, 1080)):
                hint = view.minimumSizeHint()   # Qt 自己算的下限，比它还小就不是布局的错
                view.resize(max(width, hint.width()), max(height, hint.height()))
                application.processEvents()
                shown = f"{view.width()}×{view.height()} @{scale:g}"
                for name, widget in watched:
                    spot = widget.mapTo(view, QPoint(0, 0))
                    rect = QRect(spot, widget.size())
                    assert widget.isVisibleTo(view), f"{shown}：{name} 看不见了"
                    assert view.rect().contains(rect), \
                        f"{shown}：{name} 跑出窗口 {rect} 不在 {view.rect()} 里"
                    assert widget.height() > 8 and widget.width() > 8, \
                        f"{shown}：{name} 被压成一条线"
                for dlg, items in popups:
                    dlg.resize(max(900, dlg.minimumSizeHint().width()),
                               max(560, dlg.minimumSizeHint().height()))
                    application.processEvents()
                    seen = f"{dlg.width()}×{dlg.height()} @{scale:g}"
                    for name, widget in items:
                        rect = QRect(widget.mapTo(dlg, QPoint(0, 0)), widget.size())
                        assert widget.isVisibleTo(dlg), f"{seen}：{name} 看不见了"
                        assert dlg.rect().contains(rect), \
                            f"{seen}：{name} 跑出弹窗 {rect} 不在 {dlg.rect()} 里"
                        assert widget.height() > 8 and widget.width() > 8, \
                            f"{seen}：{name} 被压成一条线"
    finally:
        application.setFont(base)
        application.processEvents()
        view.resize(1240, 800)
        page.dlg_json.close()
        page.dlg_products.close()
        view.close()                    # 别把窗口留给后面的测试（单例那条会数窗口）


def test_minimum_size_fits_small_screens(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    app().processEvents()
    hint = view.minimumSizeHint()
    assert view.minimumWidth() <= 1280 and view.minimumHeight() <= 720, \
        f"写死的最小尺寸放不进 1280×720：{view.minimumWidth()}×{view.minimumHeight()}"
    assert hint.width() <= 1280 and hint.height() <= 720, \
        f"布局自身的下限放不进 1280×720：{hint.width()}×{hint.height()}"
    for split, name in ((view.videos.split_assets, "JSON 表 / 详情"),
                        (view.videos.split_products, "成品表 / 血缘")):
        assert not split.childrenCollapsible(), f"{name} 不许被拖成 0"


# ------------------------------------------------------------------ T19
def test_only_one_asset_center(tmp_path: Path) -> None:
    """主界面上的资产中心是单例：点两次拿到的是同一个窗口。"""
    from vidscribe.gui.main_window import MainWindow  # noqa: PLC0415

    app()
    quiet()
    for widget in QApplication.topLevelWidgets():      # 前面的测试留下的窗口先收掉
        if isinstance(widget, ad.AssetCenter):
            widget.close()
    cfg, _db = make_project(tmp_path)
    window = MainWindow(cfg)
    try:
        window.on_asset_center()
        first = window.asset_center
        window.on_asset_center()
        assert window.asset_center is first, "资产中心必须是单例"
        opened = [w for w in QApplication.topLevelWidgets()
                  if isinstance(w, ad.AssetCenter) and w.isVisible()]
        assert len(opened) == 1, f"同时开着 {len(opened)} 个资产中心"
    finally:
        if window.asset_center is not None:
            window.asset_center.close()
        window.close()


# ------------------------------------------------------------------ T20
def test_gui_has_no_sql(tmp_path: Path) -> None:
    for path in sorted(GUI_DIR.glob("*.py")):
        upper = path.read_text(encoding="utf-8").upper()
        for word in ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM"):
            assert word not in upper, f"{path.name} 里还有裸 SQL：{word}"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_center_opens,
    test_center_is_not_modal,
    test_video_list_exists,
    test_current_video_title_exists,
    test_json_area_exists,
    test_product_area_exists,
    test_only_one_primary,
    test_video_context_menu,
    test_json_context_menu,
    test_product_context_menu,
    test_product_menu_navigates,
    test_prm_context_menu,
    test_double_click_json_is_view,
    test_double_click_product_opens,
    test_lineage_locates_json_and_product,
    test_current_json_is_starred,
    test_layer_grid_always_visible,
    test_nothing_overflows_small_windows,
    test_minimum_size_fits_small_screens,
    test_only_one_asset_center,
    test_gui_has_no_sql,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="uxcenter_"))
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
