"""视频资产中心最终 UX 守卫（Phase 16）。

Phase 16 收的是「一眼看懂 + 操作找得到 + 小窗口不遮挡 + 不许回归」，所以这一组测试
真的建 Qt 控件（offscreen）、真的套上程序自己的主题（`theme.apply`，9pt），
用临时目录里的临时库，**绝不碰项目真实数据库**。

  T1   六档窗口尺寸下布局不越界
  T2   当前视频标题明显存在（右侧最大字号）
  T3   一级 Primary 只有「直接剪辑」
  T4   JSON 一级按钮最多「直接剪辑 / 查看 / 更多 ▾」
  T5   右键视频菜单完整
  T6   右键高光 JSON 菜单完整
  T7   右键成品菜单完整
  T8   右键 PRM 菜单完整
  T9   双击语义正确（视频 / JSON / 成品 / PRM）
  T10  血缘 JSON 节点可定位 JSON
  T11  血缘成品节点可定位成品
  T12  血缘 PRM 节点可定位 PRM（切页 + 选中）
  T13  JSON=有 + 成品=无 可以组合筛选
  T14  GUI 里没有裸 SQL
  T15  资产中心全程 AI task = 0
  T16  JSON → 直接剪辑仍走 MainWindow.render_asset
  T17  5000 视频列表不产生 N+1
  T18  20 份 JSON 不产生逐行 SQL
  T19  60 个成品不产生逐行 SQL
  T20  刷新后当前视频不丢
  T21  刷新后当前 JSON 不丢
  T22  PRM 默认状态正确（★ + 菜单不再给「设为默认」）
  T23  删除 / 恢复正确（软删，成品不动）
  T24  三层区间始终显示 AI 原始 / Clip Engine / 实际渲染
  T25  三层区间最终结论正确（✓ / ⚠ / 还没剪，且写出原因）
  T26  顶部工作流提示 ①②③④ + 「当前：…」状态行
  T27  空状态都有人话（视频 / JSON / 成品 / 血缘 / PRM）
  T28  操作结果有顶部反馈，不只写日志
  T29  PRM 列表不逐行查成品（Phase 16 新收的 N+1）

可以直接 `python tests/test_asset_center_final_ux.py`。
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
from vidscribe.db import repo as db_repo  # noqa: E402
from vidscribe.db.db import Database  # noqa: E402
from vidscribe.gui import assets_dialog as ad  # noqa: E402
from vidscribe.gui import theme  # noqa: E402

from test_highlight_assets import make_project, video_row  # noqa: E402

GUI_DIR = ROOT / "src" / "vidscribe" / "gui"
PANEL = (GUI_DIR / "assets_dialog.py").read_text(encoding="utf-8")
AI_PANEL = (GUI_DIR / "ai_options.py").read_text(encoding="utf-8")
MAIN_WINDOW = (GUI_DIR / "main_window.py").read_text(encoding="utf-8")

# Phase 16 §14 要真的测这六档
SIZES = ((1000, 620), (1024, 680), (1152, 720), (1280, 720), (1280, 800), (1600, 900))

APP: QApplication | None = None
MENUS: list[QMenu] = []


def app() -> QApplication:
    """真程序怎么起，测试就怎么起：Fusion + QSS + 9pt。"""
    global APP
    if APP is None:
        APP = QApplication.instance() or QApplication(sys.argv[:1])
        theme.apply(APP)
    return APP


def quiet() -> None:
    """弹窗自动点掉 + 右键菜单不阻塞，只把菜单记下来给断言看。

    「直接剪辑」那个 RenderDialog 是模态的，offscreen 下会永远等下去：
    直接替换成「按当前选择开剪，然后返回 Accepted」。
    """
    ad.QMessageBox.information = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMenu.exec_ = lambda self, *a, **k: MENUS.append(self)
    ad.RenderDialog.exec_ = lambda self: (self.on_start(), QDialog.Accepted)[1]


class FakeWindow(QWidget):
    """替身主界面：只提供 `render_asset`，不做分析、不碰 AI。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def render_asset(self, asset_id, prm_id=None):
        self.calls.append((int(asset_id), prm_id))
        return True


def payload(video: str = "demo.mp4", start: float = 8.23, end: float = 23.49) -> str:
    return json.dumps({"video": video,
                       "clip": {"start": start, "end": end, "score": 0.91,
                                "type": "hook", "reason": "很炸", "evaluation": "好笑"}})


def center(tmp_path: Path, *, videos: int = 1, assets: int = 2, products: int = 1):
    """临时库 + 真的建一个资产中心（顺手造几个成品）。"""
    app()
    quiet()
    MENUS.clear()
    cfg, db = make_project(tmp_path)
    made = []
    for index in range(videos):
        video, vid = video_row(cfg, db, f"final_{index}.mp4")
        ids = [db_assets.create_asset(db, vid, payload(video=video.name),
                                      name=f"方案 {n}", source_type="ai",
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
    app().processEvents()
    return cfg, db, made, prm_id, window, view


def bulk_videos(db, count: int, start: int = 1000) -> None:
    """直接灌视频行（测试自己可以写 SQL，界面不许）—— 5000 条不写 5000 个文件。"""
    stamp = "2026-08-30T00:00:00"
    rows = [(f"fp_{n}", f"C:/videos/bulk_{n}.mp4", f"bulk_{n}.mp4", 1234, 120.0,
             stamp, stamp) for n in range(start, start + count)]
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO videos (fingerprint, file_path, file_name, file_size, duration,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def sql_count(func) -> int:
    """数一次调用里到底发了几条 SQL（包装 Database 的四个出口）。"""
    counted = {"n": 0}
    original = {name: getattr(Database, name)
                for name in ("all", "one", "value", "execute")}

    def wrap(name):
        def inner(self, *a, **k):
            counted["n"] += 1
            return original[name](self, *a, **k)
        return inner

    for name in original:
        setattr(Database, name, wrap(name))
    try:
        func()
    finally:
        for name, restore in original.items():
            setattr(Database, name, restore)
    return counted["n"]


def ai_counts(db) -> tuple[int, int]:
    return (int(db.value("SELECT COUNT(*) FROM ai_tasks", default=0) or 0),
            int(db.value("SELECT COUNT(*) FROM ai_results", default=0) or 0))


def texts(menu: QMenu) -> list[str]:
    return [a.text() for a in menu.actions() if a.text()]


def col(table, row: int, index: int) -> str:
    item = table.item(row, index)
    return "" if item is None else item.text()


def group_title(widget: QWidget) -> str:
    node = widget.parent()
    while node is not None:
        if isinstance(node, QGroupBox):
            return node.title()
        node = node.parent()
    return ""


def right_click(table, row: int, column: int = 1) -> QPoint:
    """右键必须落在这一行上（处理函数会按坐标重选行）。"""
    item = table.item(row, column)
    assert item is not None, f"第 {row} 行没有内容，右键无从下手"
    return table.visualItemRect(item).center()


def asset_rows(table) -> dict[int, int]:
    out: dict[int, int] = {}
    for line in range(table.rowCount()):
        item = table.item(line, 1)
        if item is not None and item.data(Qt.UserRole) is not None:
            out[int(item.data(Qt.UserRole))] = line
    return out


def lineage_nodes(tree) -> dict[str, object]:
    out: dict[str, object] = {}

    def walk(item):
        out[item.text(0)] = item
        for index in range(item.childCount()):
            walk(item.child(index))

    for index in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(index))
    return out


# ------------------------------------------------------------------ T1
def test_layout_fits_six_window_sizes(tmp_path: Path) -> None:
    """六档尺寸：布局下限放得进去，而且没有控件跑出窗口 / 被压成一条线。"""
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2, products=1)
    page = view.videos
    page.select_asset(made[0][2][-1])
    application = app()
    application.processEvents()
    watched = (("视频表", page.tbl_videos), ("JSON 表", page.tbl_assets),
               ("三层区间", view.json_panel.tbl_layers),
               ("结论条", view.json_panel.box_engine),
               ("成品表", page.tbl_products), ("血缘树", page.tree_lineage),
               ("当前视频", page.lbl_video), ("直接剪辑", page.btn_render),
               ("查看", page.btn_view), ("更多", page.btn_more),
               ("工作流提示", view.lbl_steps), ("当前状态", view.lbl_now))
    hint = view.minimumSizeHint()
    assert hint.width() <= 1000 and hint.height() <= 620, \
        f"布局下限放不进最小的一档 1000×620：{hint.width()}×{hint.height()}"
    view.show()
    try:
        for width, height in SIZES:
            view.resize(width, height)
            application.processEvents()
            shown = f"{view.width()}×{view.height()}"
            assert view.width() <= width and view.height() <= height, \
                f"{width}×{height} 这一档窗口被撑大到 {shown}"
            for name, widget in watched:
                spot = widget.mapTo(view, QPoint(0, 0))
                rect = QRect(spot, widget.size())
                assert widget.isVisibleTo(view), f"{shown}：{name} 看不见了"
                assert view.rect().contains(rect), \
                    f"{shown}：{name} 跑出窗口 {rect} 不在 {view.rect()} 里"
                assert widget.height() > 8 and widget.width() > 8, \
                    f"{shown}：{name} 被压成一条线"
    finally:
        view.resize(1240, 800)
        view.close()


# ------------------------------------------------------------------ T2
def test_current_video_is_the_hero(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path)
    page = view.videos
    assert made[0][0].name in page.lbl_video.text(), "当前视频标题要写清是哪个视频"
    # 主题 QSS 里有 `QWidget { font-size: 12px }`，只比 pointSize 会被它骗过去，
    # 所以按真实渲染高度比：当前视频必须比正文和区块标题都大
    hero = QFontMetrics(page.lbl_video.font()).height()
    body = QFontMetrics(page.lbl_state.font()).height()
    section = QFontMetrics(page.lbl_current.font()).height()
    assert hero > body and hero > section, \
        f"当前视频必须是右侧最大的字：标题 {hero}px / 正文 {body}px / 区块 {section}px"
    assert page.box_current.frameShape() != page.box_current.NoFrame, \
        "当前视频要独立成块，和普通 GroupBox 区分开"
    assert "②" in page.lbl_step.text(), "当前视频是第二层"


# ------------------------------------------------------------------ T3
def test_only_one_primary(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    bold = [b.text() for b in view.videos.findChildren(QPushButton) if b.font().bold()]
    assert bold == ["直接剪辑"], f"唯一 Primary 必须是「直接剪辑」：{bold}"


# ------------------------------------------------------------------ T4
def test_json_area_has_three_buttons(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    shown = [b.text() for b in page.findChildren(QPushButton) if b.isVisibleTo(view)]
    assert set(shown) == {"直接剪辑", "查看", "更多 ▾"}, f"一级按钮没收干净：{shown}"
    assert len(shown) <= 3, f"一级按钮不许超过三个：{shown}"
    for name in ("直接剪辑", "查看", "更多 ▾"):
        widget = {"直接剪辑": page.btn_render, "查看": page.btn_view,
                  "更多 ▾": page.btn_more}[name]
        assert "③" in group_title(widget), f"{name} 要摆在「③ 高光 JSON」这一区里"


# ------------------------------------------------------------------ T5
def test_video_menu_is_complete(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    MENUS.clear()
    page.on_video_menu(right_click(page.tbl_videos, 0))
    assert MENUS, "右键视频没弹菜单"
    items = " ｜ ".join(texts(MENUS[-1]))
    for needed in ("打开视频", "查看高光 JSON", "查看成品", "只看这个视频的高光",
                   "只看这个视频的成品", "重新分析", "打开所在文件夹", "复制视频路径"):
        assert needed in items, f"视频右键少了「{needed}」：{items}"


# ------------------------------------------------------------------ T6
def test_json_menu_is_complete(tmp_path: Path) -> None:
    _cfg, db, made, _prm, _win, view = center(tmp_path, assets=3)
    page = view.videos
    _video, vid, ids = made[0]
    db_assets.set_current_asset(db, ids[0])
    page.refresh_assets()
    rows = asset_rows(page.tbl_assets)
    page.select_asset(ids[1])
    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[1]]))
    items = " ｜ ".join(texts(MENUS[-1]))
    for needed in ("查看", "编辑", "直接剪辑", "设为当前", "复制 JSON",
                   "显示原文", "复制原文", "删除", "导入 JSON…"):
        assert needed in items, f"JSON 右键少了「{needed}」：{items}"
    # 当前那一份不给「设为当前」，只给一条灰的 ★；软删的给「恢复」
    page.select_asset(ids[0])
    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[0]]))
    current_items = texts(MENUS[-1])
    assert "★ 当前 JSON" in current_items and "设为当前" not in current_items, \
        f"当前 JSON 的菜单不对：{current_items}"
    db_assets.delete_asset(db, ids[2])
    page.chk_deleted.setChecked(True)
    page.refresh_assets()
    rows = asset_rows(page.tbl_assets)
    page.select_asset(ids[2])
    MENUS.clear()
    page.on_asset_menu(right_click(page.tbl_assets, rows[ids[2]]))
    deleted_items = texts(MENUS[-1])
    assert "恢复" in deleted_items and "删除" not in deleted_items, \
        f"已删除 JSON 的菜单要给「恢复」：{deleted_items}"


# ------------------------------------------------------------------ T7
def test_product_menu_is_complete(tmp_path: Path) -> None:
    _cfg, _db, _made, _prm, _win, view = center(tmp_path)
    page = view.videos
    page.tbl_products.selectRow(0)
    MENUS.clear()
    page.on_product_menu(right_click(page.tbl_products, 0))
    items = " ｜ ".join(texts(MENUS[-1]))
    for needed in ("打开成品", "打开所在文件夹", "查看来源 JSON", "查看 PRM",
                   "查看完整血缘", "复制血缘", "复制文件路径"):
        assert needed in items, f"成品右键少了「{needed}」：{items}"


# ------------------------------------------------------------------ T8
def test_prm_menu_is_complete(tmp_path: Path) -> None:
    _cfg, db, _made, prm_id, _win, view = center(tmp_path)
    prm = view.prm_panel
    other = db_assets.create_prm(db, "PRM V2", "prm/rules.txt")
    prm.reload()
    prm.select(other)
    MENUS.clear()
    prm.on_menu(right_click(prm.table, prm.table.currentRow()))
    items = " ｜ ".join(texts(MENUS[-1]))
    for needed in ("查看 / 编辑", "复制", "设为默认", "删除",
                   "复制提示词正文", "打开提示词文件"):
        assert needed in items, f"PRM 右键少了「{needed}」：{items}"
    prm.select(prm_id)
    MENUS.clear()
    prm.on_menu(right_click(prm.table, prm.table.currentRow()))
    default_items = texts(MENUS[-1])
    assert "★ 默认 PRM" in default_items and "设为默认" not in default_items, \
        f"默认那一份不该再显示「设为默认」：{default_items}"


# ------------------------------------------------------------------ T9
def test_double_click_semantics(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2)
    page = view.videos
    # 视频：进入 / 刷新当前视频工作区（焦点落到它的 JSON 表）
    assert "self.tbl_videos.doubleClicked.connect(lambda _=None: self.on_open_video())" \
        in PANEL, "双击视频要进当前视频工作区"
    page.on_open_video()
    assert page.selected_asset() is not None, "双击视频后应当停在它的高光 JSON 上"
    # JSON：查看
    assert "self.tbl_assets.doubleClicked.connect(lambda _=None: self.on_view())" in PANEL, \
        "双击 JSON 必须是「查看」"
    # 成品：打开成品
    assert "self.tbl_products.doubleClicked.connect(lambda _=None: self.on_open_product())" \
        in PANEL, "双击成品必须是「打开成品」"
    # PRM：编辑
    assert "self.table.doubleClicked.connect(lambda _=None: self.on_double_click())" in PANEL, \
        "双击 PRM 必须进编辑"
    view.prm_panel.on_double_click()
    assert view.prm_panel.edit_name.hasFocus() or view.prm_panel.edit_name.text(), \
        "双击 PRM 要能直接改档案"
    # 双击不许触发危险动作
    assert "doubleClicked.connect(lambda _=None: self.on_delete" not in PANEL, \
        "双击绝不能删东西"


# ------------------------------------------------------------------ T10 / T11 / T12
def test_lineage_locates_json_product_and_prm(tmp_path: Path) -> None:
    _cfg, _db, made, prm_id, _win, view = center(tmp_path, assets=2, products=1)
    page = view.videos
    page.tbl_products.selectRow(0)
    page.refresh_lineage()
    nodes = lineage_nodes(page.tree_lineage)
    assert {"高光 JSON", "Clip Engine", "PRM", "实际成品"} <= set(nodes), \
        f"血缘树缺节点：{sorted(nodes)}"
    # T10：点「高光 JSON」定位 JSON 行
    page.select_asset(made[0][2][0])
    page.on_lineage_clicked(lineage_nodes(page.tree_lineage)["高光 JSON"], 0)
    assert page.selected_asset() == made[0][2][-1], "血缘的 JSON 节点没定位到来源 JSON"
    # T11：点「实际成品」定位成品行
    page.tbl_products.clearSelection()
    page.refresh_lineage()
    page.tbl_products.selectRow(0)
    page.refresh_lineage()
    page.on_lineage_clicked(lineage_nodes(page.tree_lineage)["实际成品"], 0)
    assert page.selected_product() is not None, "血缘的成品节点没定位到成品行"
    # T12：点「PRM」切页并选中
    view.tabs.setCurrentIndex(0)
    page.on_lineage_clicked(lineage_nodes(page.tree_lineage)["PRM"], 0)
    assert view.tabs.currentIndex() == 1, "点 PRM 节点要切到 PRM 页"
    assert view.prm_panel.selected() == prm_id, "PRM 页要停在这个成品用的那一份上"


# ------------------------------------------------------------------ T13
def test_filters_combine_json_and_product(tmp_path: Path) -> None:
    """JSON=有 + 成品=无：只剩「有 JSON 还没剪」的视频。"""
    _cfg, db, made, prm_id, _win, view = center(tmp_path, videos=1, assets=1, products=1)
    cfg = _cfg
    page = view.videos
    plain, plain_id = video_row(cfg, db, "no_json.mp4")
    waiting, waiting_id = video_row(cfg, db, "waiting.mp4")
    db_assets.create_asset(db, waiting_id, payload(video=waiting.name), name="方案 A")
    page.cmb_json.setCurrentIndex(page.cmb_json.findData("has"))
    page.cmb_product.setCurrentIndex(page.cmb_product.findData("none"))
    page.reload()
    names = [col(page.tbl_videos, line, 1) for line in range(page.tbl_videos.rowCount())]
    assert names == [waiting.name], f"「有 JSON + 无成品」筛出来的不对：{names}"
    assert plain.name not in names, "没有 JSON 的视频不该出现"
    assert made[0][0].name not in names, "已经剪出成品的视频不该出现"


# ------------------------------------------------------------------ T14
def test_gui_has_no_sql(tmp_path: Path) -> None:
    for name, text in (("assets_dialog.py", PANEL), ("ai_options.py", AI_PANEL),
                       ("main_window.py", MAIN_WINDOW)):
        upper = text.upper()
        for word in ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM",
                     "CONN.EXECUTE(", "CURSOR.EXECUTE("):
            assert word not in upper, f"{name} 里还有裸 SQL / 直连游标：{word}"


# ------------------------------------------------------------------ T15
def test_asset_center_never_calls_ai(tmp_path: Path) -> None:
    """打开 / 刷新 / 搜索 / 筛选 / 查看 / 复制 / 删除 / 恢复 / PRM CRUD / 直接剪辑：AI 全程 0。"""
    _cfg, db, made, prm_id, window, view = center(tmp_path, assets=2)
    page = view.videos
    before = ai_counts(db)
    page.reload()
    page.edit_search.setText("final")
    page.reload()
    page.cmb_status.setCurrentIndex(1)
    page.reload()
    page.cmb_status.setCurrentIndex(0)
    page.edit_search.setText("")
    page.reload()
    page.select_video(made[0][1])
    page.select_asset(made[0][2][0])
    page.on_view()
    page.on_copy()
    page.on_set_current()
    page.on_delete()
    page.chk_deleted.setChecked(True)
    page.refresh_assets()
    page.on_restore()
    view.prm_panel.edit_name.setText("PRM V3")
    view.prm_panel.edit_file.setText("prm/rules.txt")
    view.prm_panel.on_add()
    view.prm_panel.on_copy()
    view.prm_panel.on_default()
    view.prm_panel.on_delete()
    view.prm_panel.on_restore()
    page.select_asset(made[0][2][-1])
    page.on_render()
    page.tbl_products.selectRow(0)
    page.refresh_lineage()
    after = ai_counts(db)
    assert before == after == (0, 0), f"资产中心动了 AI：ai_tasks/ai_results {before} → {after}"
    for banned in ("enqueue_ai_task", "dispatch_ai", "send_to_ai", "ask_ai"):
        assert banned not in PANEL, f"资产中心里出现了 AI 入口：{banned}"


# ------------------------------------------------------------------ T16
def test_render_goes_through_main_window(tmp_path: Path) -> None:
    _cfg, _db, made, prm_id, window, view = center(tmp_path, assets=2)
    page = view.videos
    page.select_asset(made[0][2][0])
    page.on_render()
    assert len(window.calls) == 1 and window.calls[0][0] == made[0][2][0], \
        f"直接剪辑必须打到 MainWindow.render_asset：{window.calls}"
    assert window.calls[0][1] in (None, prm_id), \
        f"带过去的 PRM 不对：{window.calls}"
    assert "render_highlight" not in PANEL, "资产中心不许自己渲染"
    assert "plan_clips(" not in PANEL, \
        "资产中心不许自己算区间（区间只能来自 db/assets.py 的读取接口）"


# ------------------------------------------------------------------ T17
def test_5000_videos_is_not_n_plus_one(tmp_path: Path) -> None:
    _cfg, db, _made, _prm, _win, view = center(tmp_path, assets=1, products=1)
    page = view.videos
    bulk_videos(db, 5000)
    page.cmb_page.setCurrentIndex(page.cmb_page.findData(5000))
    count = sql_count(page.reload)
    assert page.tbl_videos.rowCount() >= 5000, \
        f"5000 条视频没全列出来：{page.tbl_videos.rowCount()}"
    assert count <= 20, f"5000 视频列表退化成 N+1：{count} 条 SQL"


# ------------------------------------------------------------------ T18 / T19
def test_20_json_and_60_products_stay_flat(tmp_path: Path) -> None:
    _cfg, db, made, prm_id, _win, view = center(tmp_path, assets=20, products=0)
    page = view.videos
    _video, vid, ids = made[0]
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, asset_id in enumerate(ids):
        for copy in range(3):
            target = out_dir / f"p_{index}_{copy}.mp4"
            target.write_bytes(b"mp4")
            db_assets.record_product(db, vid, target,
                                     specs=[{"start": 1.0, "end": 5.0, "duration": 4.0}],
                                     asset_id=asset_id, prm_id=prm_id)
    page.reload()
    page.select_video(vid)
    json_sql = sql_count(page.refresh_assets)
    product_sql = sql_count(page.refresh_products)
    assert page.tbl_assets.rowCount() == 20, f"20 份 JSON 都要在：{page.tbl_assets.rowCount()}"
    assert page.tbl_products.rowCount() == 60, \
        f"60 个成品都要在：{page.tbl_products.rowCount()}"
    assert json_sql < 10, f"JSON 表逐行查库了：{json_sql} 条 SQL"
    assert product_sql < 10, f"成品表逐行查库了：{product_sql} 条 SQL"


# ------------------------------------------------------------------ T20 / T21
def test_refresh_keeps_current_video_and_json(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, videos=3, assets=3)
    page = view.videos
    _video, vid, ids = made[2]
    page.select_video(vid)
    page.select_asset(ids[1])
    assert page.current_video_id() == vid and page.selected_asset() == ids[1]
    page.reload()
    assert page.current_video_id() == vid, "刷新后当前视频丢了"
    assert page.selected_asset() == ids[1], "刷新后当前 JSON 丢了"
    view.reload()
    assert page.current_video_id() == vid, "整窗刷新后当前视频丢了"
    assert page.selected_asset() == ids[1], "整窗刷新后当前 JSON 丢了"


# ------------------------------------------------------------------ T22
def test_prm_default_state(tmp_path: Path) -> None:
    _cfg, db, _made, prm_id, _win, view = center(tmp_path)
    prm = view.prm_panel
    other = db_assets.create_prm(db, "PRM V2", "prm/rules.txt")
    prm.reload()
    marks = {}
    for line in range(prm.table.rowCount()):
        marks[int(col(prm.table, line, 0))] = col(prm.table, line, 5)
    assert marks[prm_id] == "★", f"默认 PRM 要打星：{marks}"
    assert marks[other] == "", f"非默认那份不该有星：{marks}"
    db_assets.set_default_prm(db, other)
    prm.reload()
    again = {int(col(prm.table, line, 0)): col(prm.table, line, 5)
             for line in range(prm.table.rowCount())}
    assert again[other] == "★" and again[prm_id] == "", f"换默认之后星没跟着走：{again}"


# ------------------------------------------------------------------ T23
def test_delete_and_restore(tmp_path: Path) -> None:
    _cfg, db, made, _prm, _win, view = center(tmp_path, assets=3, products=1)
    page = view.videos
    _video, vid, ids = made[0]
    doomed = ids[0]
    page.select_asset(doomed)
    page.on_delete()
    assert db_assets.get_asset(db, doomed)["deleted_at"], "删除必须是软删（有 deleted_at）"
    assert doomed not in asset_rows(page.tbl_assets), "删掉的 JSON 默认不该还在列表里"
    assert len(db_assets.list_products(db, vid)) == 1, "删 JSON 不许动成品"
    page.chk_deleted.setChecked(True)
    page.refresh_assets()
    assert doomed in asset_rows(page.tbl_assets), "勾上「含已删除」要能看见它"
    page.select_asset(doomed)
    page.on_restore()
    assert db_assets.get_asset(db, doomed)["deleted_at"] is None, "恢复没生效"


# ------------------------------------------------------------------ T24
def test_three_layers_always_shown(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=1, products=1)
    page = view.videos
    grid = view.json_panel.tbl_layers
    page.select_asset(made[0][2][0])
    layers = [col(grid, line, 0) for line in range(3)]
    assert layers == ["AI 原始", "Clip Engine", "实际渲染"], f"三层不对：{layers}"
    # 没选 JSON 的空态也要摆着三行，不许变成一张白表
    view.json_panel.clear("请先选择一个视频")
    empty = [col(grid, line, 0) for line in range(grid.rowCount())]
    assert empty == ["AI 原始", "Clip Engine", "实际渲染"], f"空态三层不见了：{empty}"


# ------------------------------------------------------------------ T25
def test_layer_conclusion_is_correct(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2, products=1)
    page = view.videos
    grid = view.json_panel.tbl_layers
    # 剪过的那一份：结论要么一致要么不一致，而且原因写在结论后面
    page.select_asset(made[0][2][-1])
    verdict = col(grid, 2, 4)
    assert verdict in ("✓ 一致", "⚠ 不一致"), f"剪过的那份要给一致性结论：{verdict}"
    lines = [line for line in view.json_panel.lbl_engine.text().splitlines() if line]
    assert lines and lines[0][0] in "✓⚠○", f"结论必须写在第一行：{lines[:1]}"
    assert any("原因" in line or "上限" in line or "时间戳" in line for line in lines[1:]) \
        or len(lines) == 1, f"原因要紧跟结论：{lines}"
    # 没剪过的那一份：明说还没剪
    page.select_asset(made[0][2][0])
    assert col(grid, 2, 4) == "还没剪", f"没剪过的那份要说「还没剪」：{col(grid, 2, 4)}"
    assert "还没剪过" in view.json_panel.lbl_engine.text(), \
        f"结论条要说清还没剪：{view.json_panel.lbl_engine.text()}"


# ------------------------------------------------------------------ T26
def test_workflow_hint_and_focus_line(tmp_path: Path) -> None:
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2)
    steps = view.lbl_steps.text()
    for mark in ("①", "②", "③", "④"):
        assert mark in steps, f"顶部工作流少了 {mark}：{steps}"
    assert "选视频" in steps and "直接剪辑" in steps and "成品" in steps, \
        f"顶部提示要写清工作流：{steps}"
    view.videos.select_asset(made[0][2][0])
    now = view.lbl_now.text()
    assert now.startswith("当前："), f"状态行格式不对：{now}"
    assert made[0][0].name in now and "高光 JSON #" in now and "成品" in now, \
        f"状态行要说清视频 / JSON / 成品：{now}"


# ------------------------------------------------------------------ T27
def test_empty_states_speak_chinese(tmp_path: Path) -> None:
    """没视频 / 没 JSON / 没成品 / 没血缘 / 没 PRM 都得有人话，不许只有空白或「—」。"""
    cfg, db, _made, _prm, _win, view = center(tmp_path, videos=1, assets=0, products=0)
    page = view.videos
    assert "当前视频暂无高光 JSON" in page.lbl_current.text(), \
        f"没 JSON 要说清：{page.lbl_current.text()}"
    assert "当前视频还没有成品" in page.lbl_product_head.text(), \
        f"没成品要说清：{page.lbl_product_head.text()}"
    assert page.tree_lineage.topLevelItemCount() >= 1, "血缘区不许是一片空白"
    # 一个视频都没有的空库：标题和状态行都要说「请选择一个视频」，PRM 页要给下一步
    empty = Path(tempfile.mkdtemp())
    try:
        cfg2, _db2 = make_project(empty)
        other = ad.AssetCenter(cfg2, None, log=lambda _t: None)
        try:
            other.reload()
            assert other.videos.tbl_videos.rowCount() == 0
            assert other.videos.lbl_video.text() == "请选择一个视频", \
                f"没选视频要说清：{other.videos.lbl_video.text()}"
            assert other.lbl_now.text() == "当前：请选择一个视频", \
                f"状态行不对：{other.lbl_now.text()}"
            assert other.prm_panel.table.rowCount() == 0
            assert other.prm_panel.lbl_empty.isVisibleTo(other.prm_panel), \
                "一份 PRM 都没有时要给出下一步"
            assert "暂无 PRM" in other.prm_panel.lbl_empty.text()
        finally:
            other.close()
    finally:
        shutil.rmtree(empty, ignore_errors=True)


# ------------------------------------------------------------------ T28
def test_actions_report_back(tmp_path: Path) -> None:
    """操作结果写在顶部，不只写日志。"""
    _cfg, _db, made, _prm, _win, view = center(tmp_path, assets=2)
    page = view.videos
    page.select_asset(made[0][2][0])
    page.on_copy()
    assert "已复制" in view.lbl_flash.text(), f"复制没反馈：{view.lbl_flash.text()}"
    page.on_set_current()
    assert "已设为当前" in view.lbl_flash.text(), f"设为当前没反馈：{view.lbl_flash.text()}"
    doomed = page.selected_asset()          # 复制之后选中的可能已经是新那份
    page.on_delete()
    assert "回收" in view.lbl_flash.text(), f"删除没反馈：{view.lbl_flash.text()}"
    page.chk_deleted.setChecked(True)
    page.refresh_assets()
    page.select_asset(doomed)
    page.on_restore()
    assert "已恢复" in view.lbl_flash.text(), f"恢复没反馈：{view.lbl_flash.text()}"
    view.prm_panel.select(None)
    view.prm_panel.on_default()
    assert view.lbl_flash.text(), "PRM 操作也要有反馈"


# ------------------------------------------------------------------ T29
def test_prm_list_is_not_n_plus_one(tmp_path: Path) -> None:
    """40 份 PRM：PRM 页刷新也不许逐行查成品。"""
    _cfg, db, _made, _prm, _win, view = center(tmp_path)
    for index in range(40):
        db_assets.create_prm(db, f"PRM {index}", "prm/rules.txt")
    count = sql_count(view.prm_panel.reload)
    assert view.prm_panel.table.rowCount() == 41, \
        f"41 份 PRM 都要在：{view.prm_panel.table.rowCount()}"
    assert count < 10, f"PRM 列表退化成 N+1：{count} 条 SQL"


# ------------------------------------------------------------------ 直接跑
def main() -> int:
    tests = dict((name, obj) for name, obj in globals().items()
                 if name.startswith("test_") and callable(obj))
    order = [
        "test_layout_fits_six_window_sizes", "test_current_video_is_the_hero",
        "test_only_one_primary", "test_json_area_has_three_buttons",
        "test_video_menu_is_complete", "test_json_menu_is_complete",
        "test_product_menu_is_complete", "test_prm_menu_is_complete",
        "test_double_click_semantics", "test_lineage_locates_json_product_and_prm",
        "test_filters_combine_json_and_product", "test_gui_has_no_sql",
        "test_asset_center_never_calls_ai", "test_render_goes_through_main_window",
        "test_5000_videos_is_not_n_plus_one", "test_20_json_and_60_products_stay_flat",
        "test_refresh_keeps_current_video_and_json", "test_prm_default_state",
        "test_delete_and_restore", "test_three_layers_always_shown",
        "test_layer_conclusion_is_correct", "test_workflow_hint_and_focus_line",
        "test_empty_states_speak_chinese", "test_actions_report_back",
        "test_prm_list_is_not_n_plus_one",
    ]
    ordered = [(name, tests[name]) for name in order if name in tests]
    ordered += [(name, func) for name, func in sorted(tests.items()) if name not in order]
    good = 0
    for name, func in ordered:
        work = Path(tempfile.mkdtemp(prefix="p16_"))
        try:
            func(work)
            print(f"PASS {name}")
            good += 1
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        finally:
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, ad.AssetCenter):
                    widget.close()
            shutil.rmtree(work, ignore_errors=True)
    print(f"\n{good}/{len(ordered)} 通过")
    return 0 if good == len(ordered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
