"""视频资产中心 GUI 结构守卫（Phase 13）。

Phase 13 只收口 GUI，所以这一组测试盯的是**信息架构和交互路径**别再退化回
「弹窗套娃 + 数据库后台」那套：

  T_GUI_01  资产中心里不存在 JsonDialog（JSON 详情必须是页内面板）
  T_GUI_02  资产中心里不存在 PrmDialog（PRM 必须是独立 Tab）
  T_GUI_03  资产中心只有一个顶层窗口，子面板都不是窗口
  T_GUI_04  视频列表是第一层索引（不是视频下拉框）
  T_GUI_05  高光 JSON 是视频下面的二级资产，跟着选中的视频走
  T_GUI_06  成品能追溯到高光 JSON（成品表 + 血缘树）
  T_GUI_07  「直接剪辑」走的是 MainWindow.render_asset
  T_GUI_08  「直接剪辑」一次 AI 都不调
  T_GUI_09  PRM 是独立 Tab
  T_GUI_10  AI 面板不再承担 JSON / PRM 的增删改
  T_GUI_11  raw_json 不允许被原地修改（编辑只能另存成新的高光 JSON）
  T_GUI_12  当前 JSON 有明确的 ★ 标记

真的建 Qt 控件（offscreen），用临时目录里的临时库，**绝不碰项目真实数据库**。
可以直接 `python tests/test_gui_assets_center.py`。
"""

from __future__ import annotations

import ast
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

from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,   # noqa: E402
                             QMessageBox, QPushButton, QWidget)

from vidscribe.db import assets as db_assets                      # noqa: E402
from vidscribe.db.db import Database                              # noqa: E402
from vidscribe.gui import assets_dialog as ad                     # noqa: E402

from test_highlight_assets import make_project, video_row         # noqa: E402

PANEL = (ROOT / "src" / "vidscribe" / "gui" / "assets_dialog.py").read_text(encoding="utf-8")
AI_PANEL = (ROOT / "src" / "vidscribe" / "gui" / "ai_options.py").read_text(encoding="utf-8")
MAIN_WINDOW = (ROOT / "src" / "vidscribe" / "gui" / "main_window.py").read_text(encoding="utf-8")

APP: QApplication | None = None


def app() -> QApplication:
    global APP
    if APP is None:
        APP = QApplication.instance() or QApplication(sys.argv[:1])
    return APP


def ai_payload(start: float = 8.23, end: float = 23.49, score: float = 0.91,
               video: str = "demo.mp4") -> dict:
    return {"video": video,
            "clip": {"start": start, "end": end, "score": score,
                     "type": "hook", "reason": "很炸", "evaluation": "好笑"}}


class FakeWindow(QWidget):
    """替身主界面：只提供 `render_asset`，用来验证剪辑请求确实打到它身上。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def render_asset(self, asset_id, prm_id=None):
        self.calls.append((int(asset_id), prm_id))
        return True


def quiet() -> None:
    """提示类弹窗自动点掉：测试不能卡在 QMessageBox 上。"""
    ad.QMessageBox.information = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    ad.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)


def center(tmp_path: Path, *, videos: int = 1, assets: int = 2):
    """建一份临时库 + 真的建一个资产中心窗口。"""
    app()
    quiet()
    cfg, db = make_project(tmp_path)
    made = []
    for index in range(videos):
        video, vid = video_row(cfg, db, f"gui_{index}.mp4")
        ids = [db_assets.create_asset(db, vid, json.dumps(ai_payload(video=video.name)),
                                      name=f"JSON {n}", source_type="ai",
                                      provider="Gemini" if n == 0 else None,
                                      model="Gemini Flash" if n == 0 else None)
               for n in range(assets)]
        made.append((video, vid, ids))
    prm_path = cfg.root / "prm" / "rules.txt"
    prm_path.parent.mkdir(parents=True, exist_ok=True)
    prm_path.write_text("剪辑规则", encoding="utf-8")
    db_assets.create_prm(db, "PRM V1", str(prm_path), language="zh", version="v1")
    window = FakeWindow()
    view = ad.AssetCenter(cfg, window, log=lambda _text: None)
    return cfg, db, made, window, view



def col(table, row: int, index: int) -> str:
    item = table.item(row, index)
    return "" if item is None else item.text()


# ------------------------------------------------------------------ T_GUI_01/02
def test_no_json_dialog(tmp_path: Path) -> None:
    assert "class JsonDialog" not in PANEL, "JSON 详情必须是页内面板，不能再有 JsonDialog"
    assert "class JsonPanel" in PANEL, "页内 JSON 面板不在了"


def test_no_prm_dialog(tmp_path: Path) -> None:
    assert "class PrmDialog(" not in PANEL, "PRM 必须是独立 Tab，不能再有 PrmDialog"
    assert "class PrmPanel" in PANEL, "PRM 页不在了"


# ------------------------------------------------------------------ T_GUI_03
def test_only_one_top_level_window(tmp_path: Path) -> None:
    _cfg, _db, _made, _window, view = center(tmp_path)
    assert view.isWindow(), "资产中心自己必须是顶层窗口"
    assert not view.isModal(), "资产中心不能是模态"
    for name, child in (("视频页", view.videos), ("PRM 页", view.prm_panel),
                        ("JSON 面板", view.json_panel)):
        assert not child.isWindow(), f"{name} 必须是页内面板，不能是独立窗口"
    opened = [w for w in QApplication.topLevelWidgets()
              if isinstance(w, QDialog) and w.isVisible()]
    assert not opened, f"打开资产中心不该顺带弹对话框：{opened}"


# ------------------------------------------------------------------ T_GUI_04
def test_video_list_is_the_first_index(tmp_path: Path) -> None:
    _cfg, _db, made, _window, view = center(tmp_path, videos=3)
    assert view.tbl_videos.rowCount() == 3, "视频列表得把三个视频都列出来"
    names = {col(view.tbl_videos, line, 1) for line in range(3)}
    assert names == {v.name for v, _vid, _ids in made}, f"列表里的视频名不对：{names}"
    # 视频不许再塞进下拉框
    for box in view.videos.findChildren(QComboBox):
        values = [str(box.itemText(i)) for i in range(box.count())]
        assert not any(name in " ".join(values) for name in names), \
            f"视频名跑进下拉框了：{values}"
    view.videos.edit_search.setText("gui_1")
    view.videos.reload()
    assert view.tbl_videos.rowCount() == 1, "搜索没生效"


# ------------------------------------------------------------------ T_GUI_05
def test_json_is_second_level(tmp_path: Path) -> None:
    _cfg, _db, made, _window, view = center(tmp_path, videos=2, assets=2)
    for _video, vid, ids in made:
        view.select_video(vid)
        assert view.tbl_assets.rowCount() == len(ids), "JSON 表要跟着选中的视频走"
        shown = {col(view.tbl_assets, line, 1) for line in range(view.tbl_assets.rowCount())}
        assert shown == {f"高光 JSON #{i}" for i in ids}, f"JSON 名称不对：{shown}"
        assert view.selected_asset() in ids, "选了视频就得自动选中它的一份 JSON"


# ------------------------------------------------------------------ T_GUI_06
def test_product_traces_back_to_json(tmp_path: Path) -> None:
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    asset_id = ids[0]
    prm_id = int(db_assets.list_prms(db)[0]["id"])

    out = tmp_path / "output" / "done.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    db_assets.record_product(db, vid, out,
                             specs=[{"start": 8.23, "end": 19.39, "duration": 11.16}],
                             asset_id=asset_id, prm_id=prm_id)
    view.videos.reload()
    view.select_video(vid)
    assert view.tbl_products.rowCount() == 1, "成品没列出来"
    assert col(view.tbl_products, 0, 2) == f"高光 JSON #{asset_id}", "成品得写清来源 JSON"
    assert col(view.tbl_products, 0, 3) == "PRM V1", "成品得写清用的哪版 PRM"
    assert "8.23 → 19.39" in col(view.tbl_products, 0, 5), "成品得显示实际渲染区间"
    view.tbl_products.selectRow(0)
    view.refresh_lineage()
    root = view.tree_lineage.topLevelItem(0)
    labels = [root.child(i).text(0) for i in range(root.childCount())]
    assert "高光 JSON" in labels and "Clip Engine" in labels and "PRM" in labels, \
        f"血缘树缺环节：{labels}"


# ------------------------------------------------------------------ T_GUI_07/08
def test_render_goes_through_main_window(tmp_path: Path) -> None:
    _cfg, db, made, window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    view.select_video(vid)

    view.videos.select_asset(ids[0])
    ad.RenderDialog.exec_ = lambda self: (self.on_start(), QDialog.Accepted)[1]
    view.on_render()
    assert window.calls == [(ids[0], None)] or window.calls == [(ids[0], 1)], \
        f"直接剪辑必须调 MainWindow.render_asset，实际={window.calls}"


def test_render_never_calls_ai(tmp_path: Path) -> None:
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    view.select_video(vid)
    view.videos.select_asset(ids[0])
    ad.RenderDialog.exec_ = lambda self: (self.on_start(), QDialog.Accepted)[1]
    view.on_render()
    assert db.value("SELECT COUNT(*) FROM ai_tasks", (), 0) == 0, "这条路不许排 AI 任务"
    assert db.value("SELECT COUNT(*) FROM ai_results", (), 0) == 0, "这条路不许写 AI 结果"
    source = ast.parse(PANEL)
    for node in ast.walk(source):
        if isinstance(node, ast.ClassDef) and node.name == "RenderDialog":
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert not {"enqueue_ai_task", "send_to_ai", "ask_ai"} & names, \
                "剪辑确认窗口不许碰 AI"


# ------------------------------------------------------------------ T_GUI_09
def test_prm_is_its_own_tab(tmp_path: Path) -> None:
    _cfg, _db, _made, _window, view = center(tmp_path)
    titles = [view.tabs.tabText(i) for i in range(view.tabs.count())]
    assert titles == ["视频资产", "PRM 管理"], f"两页结构变了：{titles}"
    view.show_prm_page()
    assert view.tabs.currentIndex() == 1, "切不到 PRM 页"
    assert hasattr(view.prm_panel, "view_text"), "PRM 页得能改提示词正文"
    prm = view.prm_panel
    prm.reload()
    assert prm.table.rowCount() == 1, "PRM 列表没把档案列出来"
    assert prm.table.item(0, 1).text() == "PRM V1", "PRM 名字没显示"
    assert prm.table.isColumnHidden(0), "主键列该藏起来"
    prm.table.selectRow(0)
    assert prm.view_text.toPlainText() == "剪辑规则", "选中 PRM 要能看到正文"
    prm.edit_search.setText("不存在的名字")
    assert prm.table.rowCount() == 0, "PRM 搜索没生效"
    prm.edit_search.clear()
    assert prm.table.rowCount() == 1, "清空搜索要回到全部"



# ------------------------------------------------------------------ T_GUI_10
def test_ai_panel_has_no_asset_crud(tmp_path: Path) -> None:
    tree = ast.parse(AI_PANEL)
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    forbidden = {"create_prm", "update_prm", "delete_prm", "copy_prm", "restore_prm",
                 "set_default_prm", "create_asset", "edit_asset", "delete_asset",
                 "restore_asset", "copy_asset", "set_current_asset"}
    assert not forbidden & called, f"AI 面板不该再做资产增删改：{forbidden & called}"
    assert "list_prms" in called, "AI 面板还是要能读 PRM 列表（选哪一版发给 AI）"
    assert "视频资产中心" in AI_PANEL, "AI 面板要保留资产中心入口"
    for gone in ("高光方案…", "JSON 管理…", "PRM 管理…"):
        assert gone not in AI_PANEL, f"AI 面板里还留着重复入口：{gone}"


# ------------------------------------------------------------------ T_GUI_11
def test_raw_json_is_never_edited_in_place(tmp_path: Path) -> None:
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    asset_id = ids[0]
    view.select_video(vid)
    view.videos.select_asset(asset_id)
    panel = view.json_panel
    panel.on_edit()
    changed = json.loads(panel.view.toPlainText())
    changed["clip"]["score"] = 0.5
    panel.view.setPlainText(json.dumps(changed, ensure_ascii=False))
    panel.on_save()
    raw = json.loads(db.value("SELECT raw_json FROM highlight_assets WHERE id = ?",
                              (asset_id,), "{}"))
    assert raw["clip"]["score"] == 0.91, "raw_json 被改了，这是资产原则的红线"
    rows = db_assets.list_assets(db, vid)
    assert len(rows) == 2, "编辑必须另存成新的高光 JSON"
    fresh = [r for r in rows if int(r["id"]) != asset_id][0]
    assert int(fresh["parent_id"]) == asset_id, "新 JSON 得记得自己是复制自哪一份"


# ------------------------------------------------------------------ T_GUI_12
def test_current_json_is_marked(tmp_path: Path) -> None:
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=3)
    _video, vid, ids = made[0]
    db_assets.set_current_asset(db, ids[1])
    view.select_video(vid)
    view.refresh_assets()
    marks = {col(view.tbl_assets, line, 1): col(view.tbl_assets, line, 0)
             for line in range(view.tbl_assets.rowCount())}
    starred = [name for name, mark in marks.items() if "★" in mark]
    assert starred == [f"高光 JSON #{ids[1]}"], f"当前 JSON 的 ★ 标记不对：{marks}"
    assert "★" in view.lbl_current.text(), "标题栏也得说清当前是哪一份"


# ------------------------------------------------------------------ T_GUI_13
def test_sort_follows_the_order_box(tmp_path: Path) -> None:
    """排序只认「排序」下拉：表格不许自己按隐藏的 ID 列再排一遍。"""
    _cfg, db, made, _window, view = center(tmp_path, videos=3, assets=1)
    seconds = (10.0, 300.0, 100.0)
    with db.tx():
        for value, (_video, vid, _ids) in zip(seconds, made):
            db.execute("UPDATE videos SET duration = ? WHERE id = ?", (value, vid))
    names = [video.name for video, _vid, _ids in made]
    page = view.videos
    page.cmb_order.setCurrentIndex(page.cmb_order.findData("duration"))
    page.reload()
    seen = [col(view.tbl_videos, line, 1) for line in range(view.tbl_videos.rowCount())]
    assert seen == [names[1], names[2], names[0]], f"「视频时长」排序没生效：{seen}"
    view.tbl_videos.horizontalHeader().sectionClicked.emit(1)   # 点「视频」表头
    assert str(page.cmb_order.currentData()) == "name", "点表头应该切「排序」下拉，而不是另排一套"
    seen = [col(view.tbl_videos, line, 1) for line in range(view.tbl_videos.rowCount())]
    assert seen == sorted(names), f"按视频名称排序没生效：{seen}"


# ------------------------------------------------------------------ T_GUI_14
def test_filters_combine_json_and_product(tmp_path: Path) -> None:
    """场景 A：有 JSON、没成品 —— 三个下拉一起生效，一步筛出来。"""
    _cfg, db, made, _window, view = center(tmp_path, videos=3, assets=1)
    with_product = made[0]
    out = tmp_path / "output" / "a.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    db_assets.record_product(db, with_product[1], out,
                             specs=[{"start": 1.0, "end": 5.0, "duration": 4.0}],
                             asset_id=with_product[2][0])
    db.execute("DELETE FROM highlight_assets WHERE video_id = ?", (made[2][1],))
    page = view.videos
    page.cmb_json.setCurrentIndex(page.cmb_json.findData("has"))
    page.cmb_product.setCurrentIndex(page.cmb_product.findData("none"))
    page.reload()
    names = [col(view.tbl_videos, line, 1) for line in range(view.tbl_videos.rowCount())]
    assert names == [made[1][0].name], f"「有 JSON + 无成品」只该剩一个视频：{names}"
    assert "chk_has_json" not in PANEL, "「只看有高光」那个勾选框应该被 JSON 下拉取代"


# ------------------------------------------------------------------ T_GUI_15
def test_lists_are_not_n_plus_one(tmp_path: Path) -> None:
    """20 份 JSON × 3 个成品：刷新只准发个位数 SQL。"""
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=20)
    _video, vid, ids = made[0]
    prm_id = int(db_assets.list_prms(db)[0]["id"])
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, asset_id in enumerate(ids[:20]):
        for copy in range(3):
            target = out_dir / f"p_{index}_{copy}.mp4"
            target.write_bytes(b"mp4")
            db_assets.record_product(db, vid, target,
                                     specs=[{"start": 1.0, "end": 5.0, "duration": 4.0}],
                                     asset_id=asset_id, prm_id=prm_id)
    view.videos.reload()
    view.select_video(vid)
    counted = {"n": 0}
    original = {name: getattr(Database, name) for name in ("all", "one", "value", "execute")}

    def wrap(name):
        def inner(self, *a, **k):
            counted["n"] += 1
            return original[name](self, *a, **k)
        return inner

    for name in original:
        setattr(Database, name, wrap(name))
    try:
        counted["n"] = 0
        view.videos.refresh_assets()
        json_sql = counted["n"]
        counted["n"] = 0
        view.videos.refresh_products()
        product_sql = counted["n"]
    finally:
        for name, func in original.items():
            setattr(Database, name, func)
    assert view.tbl_assets.rowCount() == 20, "20 份 JSON 都要在"
    assert view.tbl_products.rowCount() == 60, "60 个成品都要在"
    assert json_sql < 10, f"JSON 表还是 N+1：{json_sql} 条 SQL"
    assert product_sql < 10, f"成品表还是 N+1：{product_sql} 条 SQL"


# ------------------------------------------------------------------ T_GUI_16
def test_gui_has_no_sql(tmp_path: Path) -> None:
    """界面里一句 SQL 都不许有：查库全部走 db/assets.py。"""
    for name, text in (("assets_dialog.py", PANEL), ("ai_options.py", AI_PANEL),
                       ("main_window.py", MAIN_WINDOW)):
        upper = text.upper()
        for word in ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM"):
            assert word not in upper, f"{name} 里还有裸 SQL：{word}"


# ------------------------------------------------------------------ T_GUI_17
def test_only_one_primary_button(tmp_path: Path) -> None:
    """高频动作只剩「直接剪辑（唯一加粗）」和「查看」，其余进「更多 ▾」。"""
    _cfg, _db, _made, _window, view = center(tmp_path)
    texts = [b.text() for b in view.videos.findChildren(QPushButton) if b.isVisibleTo(view)]
    assert set(texts) == {"查看原视频", "直接剪辑", "打开成品", "查看", "更多 ▾"}, \
        f"视频页按钮没收干净：{texts}"
    bold = [b.text() for b in view.videos.findChildren(QPushButton) if b.font().bold()]
    assert bold == ["直接剪辑"], f"主动作必须只有一个加粗按钮：{bold}"
    menu = view.videos.btn_more.menu()
    actions = [a.text() for a in menu.actions() if a.text()]
    for needed in ("编辑（保存会新建一份）", "复制这份 JSON", "设为当前 JSON",
                   "导入现成 JSON…", "删除（软删）", "恢复已删除",
                   "显示 JSON 原文", "复制 JSON 原文", "复制血缘"):
        assert needed in actions, f"「更多」里少了 {needed}：{actions}"


# ------------------------------------------------------------------ T_GUI_18
def test_layers_are_a_grid(tmp_path: Path) -> None:
    """三层区间是固定网格：AI 原始 / Clip Engine / 实际渲染，最右一格结论。"""
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    view.select_video(vid)
    view.videos.select_asset(ids[0])
    grid = view.json_panel.tbl_layers
    assert view.json_panel.LAYER_HEADERS == ("层级", "起点", "终点", "时长", "结论")
    assert grid.rowCount() == 3, f"三层区间应该正好三行：{grid.rowCount()}"
    layers = [col(grid, line, 0) for line in range(3)]
    assert layers == ["AI 原始", "Clip Engine", "实际渲染"], f"层级不对：{layers}"
    assert col(grid, 0, 1) == "8.23" and col(grid, 0, 2) == "23.49", "AI 原始区间不对"
    assert col(grid, 2, 4) in ("还没剪", "✓ 一致", "⚠ 不一致"), "实际渲染那一格要给结论"


# ------------------------------------------------------------------ T_GUI_19
def test_lineage_click_locates_rows(tmp_path: Path) -> None:
    """点血缘节点能定位：高光 JSON → JSON 表那一行，实际成品 → 成品表那一行。"""
    _cfg, db, made, _window, view = center(tmp_path, videos=1, assets=2)
    _video, vid, ids = made[0]
    prm_id = int(db_assets.list_prms(db)[0]["id"])
    out = tmp_path / "output" / "trace.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    db_assets.record_product(db, vid, out,
                             specs=[{"start": 8.23, "end": 19.39, "duration": 11.16}],
                             asset_id=ids[1], prm_id=prm_id)
    view.videos.reload()
    view.select_video(vid)
    view.videos.select_asset(ids[0])
    view.tbl_products.selectRow(0)
    view.refresh_lineage()
    root = view.tree_lineage.topLevelItem(0)
    nodes = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    view.videos.on_lineage_clicked(nodes["高光 JSON"], 0)
    assert view.selected_asset() == ids[1], "点血缘里的 JSON 应该跳到 JSON 表那一行"
    view.videos.on_lineage_clicked(nodes["实际成品"], 0)
    assert view.videos.selected_product() is not None, "点血缘里的成品应该选中成品行"


# ------------------------------------------------------------------ T_GUI_20
def test_edit_buttons_only_in_edit_mode(tmp_path: Path) -> None:
    """平时看不到「保存为新 JSON / 取消编辑」，进了编辑态才出现。"""
    _cfg, _db, made, _window, view = center(tmp_path, videos=1, assets=1)
    _video, vid, ids = made[0]
    view.select_video(vid)
    view.videos.select_asset(ids[0])
    panel = view.json_panel
    assert not panel.btn_save.isVisible() and not panel.btn_cancel.isVisible(), \
        "普通状态不该显示编辑按钮"
    view.videos.on_edit_json()
    assert panel.btn_save.isVisibleTo(panel) and panel.btn_cancel.isVisibleTo(panel), \
        "编辑态要出现「保存为新 JSON」「取消编辑」"
    assert panel.raw_visible(), "编辑态要把原文展开"
    panel.on_cancel_edit()
    assert not panel.btn_save.isVisibleTo(panel), "取消编辑后按钮要收回去"


# ------------------------------------------------------------------ 直接跑
TESTS = (
    test_no_json_dialog,
    test_no_prm_dialog,
    test_only_one_top_level_window,
    test_video_list_is_the_first_index,
    test_json_is_second_level,
    test_product_traces_back_to_json,
    test_render_goes_through_main_window,
    test_render_never_calls_ai,
    test_prm_is_its_own_tab,
    test_ai_panel_has_no_asset_crud,
    test_raw_json_is_never_edited_in_place,
    test_current_json_is_marked,
    test_sort_follows_the_order_box,
    test_filters_combine_json_and_product,
    test_lists_are_not_n_plus_one,
    test_gui_has_no_sql,
    test_only_one_primary_button,
    test_layers_are_a_grid,
    test_lineage_click_locates_rows,
    test_edit_buttons_only_in_edit_mode,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="guicenter_"))
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
