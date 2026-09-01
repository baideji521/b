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
  T_GUI_13  PRM 正文以数据库为准：选中导入老文件、改名 / 存正文只动库、新建不落盘
  T_GUI_14  「导入文件…」当场读进编辑框但不写库，点「新建」才登记
  T_GUI_15  改名有自己的提交按钮；撞名被挡；改名不会丢掉刚导入的正文

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
from vidscribe.db import repo as db_repo                          # noqa: E402
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
    # PRM 的「新增 / 修改」弹窗是模态的：默认按「取消」，需要走保存的用例自己脚本化
    ad.PrmEditDialog.exec_ = lambda self: QDialog.Rejected



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
    names = {col(view.tbl_videos, line, 2) for line in range(3)}
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
    assert hasattr(ad, "PrmEditDialog"), "PRM 正文得有地方改（新增 / 修改弹窗）"
    prm = view.prm_panel
    prm.reload()
    assert prm.table.rowCount() == 1, "PRM 列表没把档案列出来"
    assert prm.table.item(0, 1).text() == "PRM V1", "PRM 名字没显示"
    assert prm.table.isColumnHidden(0), "主键列该藏起来"
    prm.table.selectRow(0)
    dlg = prm._dialog(db_assets.get_prm(prm._handle(), prm.selected()))
    assert dlg.view_text.toPlainText() == "剪辑规则", "「修改」弹窗要带出库里的正文"
    dlg.reject()

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
                 "set_default_prm", "set_prm_enabled", "create_asset", "edit_asset",
                 "delete_asset", "restore_asset", "copy_asset", "set_current_asset"}
    assert not forbidden & called, f"AI 面板不该再做资产增删改：{forbidden & called}"
    assert "enabled_prms" in called, "AI 面板要能读「现在会发哪几份 PRM」（只读，不许改）"
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
    seen = [col(view.tbl_videos, line, 2) for line in range(view.tbl_videos.rowCount())]
    assert seen == [names[1], names[2], names[0]], f"「视频时长」排序没生效：{seen}"
    view.tbl_videos.horizontalHeader().sectionClicked.emit(2)   # 点「视频」表头
    assert str(page.cmb_order.currentData()) == "name", "点表头应该切「排序」下拉，而不是另排一套"
    seen = [col(view.tbl_videos, line, 2) for line in range(view.tbl_videos.rowCount())]
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
    names = [col(view.tbl_videos, line, 2) for line in range(view.tbl_videos.rowCount())]
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
    page = view.videos
    # 三个按钮都在「③ 高光 JSON」弹窗里；视频页自己一个按钮都不摆
    assert not [b for b in page.findChildren(QPushButton) if b.window() is view], \
        "视频页上不该再摆按钮"
    texts = [b.text() for b in page.findChildren(QPushButton)
             if b.window() is page.dlg_json and b.isVisibleTo(page.dlg_json)]
    assert set(texts) == {"直接剪辑", "查看", "更多 ▾"}, \
        f"视频页按钮没收干净：{texts}"
    bold = [b.text() for b in view.videos.findChildren(QPushButton) if b.font().bold()]
    assert bold == ["直接剪辑"], f"主动作必须只有一个加粗按钮：{bold}"
    menu = view.videos.btn_more.menu()
    actions = [a.text() for a in menu.actions() if a.text()]
    for needed in ("编辑（保存会新建一份）", "复制这份 JSON", "设为当前 JSON",
                   "导入现成 JSON…", "删除（软删）", "恢复已删除",
                   "查看原视频", "打开成品",
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


# ------------------------------------------------------------------ T_GUI_21
def test_center_dirs_are_its_own_config_keys(tmp_path: Path) -> None:
    """资产中心自己的两个目录写 `assets` 一节，绝不碰 AI 面板和主界面那几个键。"""
    cfg, _db, _made, _window, view = center(tmp_path)
    source = tmp_path / "中心_原始"
    product = tmp_path / "中心_成品"
    for path in (source, product):
        path.mkdir()
    before = json.loads((cfg.root / "config.json").read_text(encoding="utf-8"))
    view.edit_assets_in.setText(str(source))
    view.edit_assets_out.setText(str(product))
    view._save_dirs()
    data = json.loads((cfg.root / "config.json").read_text(encoding="utf-8"))
    assert data["assets"]["input_dir"] == str(source)
    assert data["assets"]["output_dir"] == str(product)
    assert (data.get("bridge", {}).get("ai_input_dir")
            == before.get("bridge", {}).get("ai_input_dir")), "不许动 AI_输入目录"
    assert (data.get("bridge", {}).get("ai_output_dir")
            == before.get("bridge", {}).get("ai_output_dir")), "不许动 AI_输出目录"
    assert data["paths"]["input_dir"] == before["paths"]["input_dir"], "不许动主界面导入目录"
    assert view.assets_dir("input_dir") == source
    assert view.assets_dir("output_dir") == product


# ------------------------------------------------------------------ T_GUI_22
def test_center_scan_registers_its_own_videos(tmp_path: Path) -> None:
    """「扫描目录」按中心自己的输入目录登记原始视频，登记完列表里就能看到。"""
    cfg, db, _made, _window, view = center(tmp_path)
    source = tmp_path / "中心_原始"
    source.mkdir()
    (source / "独立素材.mp4").write_bytes(b"m" * 4096)
    view.edit_assets_in.setText(str(source))
    view._save_dirs()
    view.on_scan_dirs()
    names = [str(row["file_name"]) for row in db.all("SELECT file_name FROM videos")]
    assert "独立素材.mp4" in names, f"扫描没把中心自己目录里的视频登记进库：{names}"
    listed = [str(item["file_name"]) for item in view.videos._rows]
    assert "独立素材.mp4" in listed, f"登记完列表里就该看得到：{listed}"


# ------------------------------------------------------------------ T_GUI_23
def test_video_row_shows_its_folder(tmp_path: Path) -> None:
    """「目录」列在「视频」右边、「时长」左边，写这个文件真正所在的目录。"""
    _cfg, _db, made, _window, view = center(tmp_path)
    page = view.videos
    video, _vid, _ids = made[0]
    assert page.VIDEO_HEADERS[page.NAME_COLUMN] == "视频"
    assert page.VIDEO_HEADERS[page.DIR_COLUMN] == "目录"
    assert page.DIR_COLUMN == page.NAME_COLUMN + 1, "目录列必须紧挨着视频名"
    assert page.VIDEO_HEADERS[page.DIR_COLUMN + 1] == "时长", "目录列右边就是时长"
    cell = page.tbl_videos.item(0, page.DIR_COLUMN)
    # Windows 上临时目录会以 8.3 短名（ADMINI~1）出现，按真实路径比
    assert Path(cell.text()).resolve() == video.parent.resolve(), f"目录列写错了：{cell.text()}"
    assert Path(cell.toolTip()).resolve() == video.resolve(), "悬停要能看到完整路径"


# ------------------------------------------------------------------ T_GUI_24
def test_json_and_product_cells_open_the_popups(tmp_path: Path) -> None:
    """点「JSON」格子开③弹窗，点「成品」格子开④弹窗（右键里不再有这两项）。"""
    _cfg, db, made, _window, view = center(tmp_path, assets=1)
    page = view.videos
    _video, vid, _ids = made[0]
    product = tmp_path / "成品.mp4"
    product.write_bytes(b"p" * 4096)
    db_repo.register_artifact(db, vid, "final_video", product)
    page.reload()
    assert page.VIDEO_HEADERS[page.JSON_COLUMN] == "JSON"
    assert page.VIDEO_HEADERS[page.PRODUCT_COLUMN] == "成品"
    page._on_cell_clicked(page.tbl_videos.item(0, page.JSON_COLUMN))
    app().processEvents()
    assert page.dlg_json.isVisible(), "点 JSON 格子要开出③高光 JSON 弹窗"
    page.dlg_json.close()
    page._on_cell_clicked(page.tbl_videos.item(0, page.PRODUCT_COLUMN))
    app().processEvents()
    assert page.dlg_products.isVisible(), "点成品格子要开出④成品与血缘弹窗"
    page.dlg_products.close()
    # 点视频名那一格什么都不该发生（免得手一滑就弹窗）
    page._on_cell_clicked(page.tbl_videos.item(0, page.NAME_COLUMN))
    assert not page.dlg_json.isVisible() and not page.dlg_products.isVisible()


# ------------------------------------------------------------------ T_GUI_26
def test_products_follow_the_selected_video_after_reload(tmp_path: Path) -> None:
    """列表重画后选中行换了视频，成品区必须跟着换 —— 否则「打开成品」开的是别人的成品。

    Qt 的坑：`setRowCount(0)` 之后重新插行，选中的**行号**还在原地，
    itemSelectionChanged 不会再发一次，详情区（成品表 / 血缘）就留在上一个视频上。
    """
    _cfg, db, made, _window, view = center(tmp_path, videos=2, assets=1)
    page = view.videos
    prm_id = int(db_assets.list_prms(db)[0]["id"])
    outs = {}
    for _video, vid, ids in made:
        out = tmp_path / "output" / f"成品_{vid}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        db_assets.record_product(db, vid, out,
                                 specs=[{"start": 1.0, "end": 4.0, "duration": 3.0}],
                                 asset_id=ids[0], prm_id=prm_id)
        outs[vid] = out
    page.reload()
    first = page.current_video_id()
    assert first is not None
    page.tbl_videos.selectRow(0)
    first = page.current_video_id()

    # 搜另一个视频：重画之后第 0 行换成了它，行号却没变
    other = next(vid for _v, vid, _ids in made if vid != first)
    page.edit_search.setText(str(next(v.name for v, vid, _i in made if vid == other)))
    page.reload()
    assert page.tbl_videos.rowCount() == 1
    assert page.current_video_id() == other, "搜完只剩这一个视频，它就是当前视频"

    shown = {info["path"] for info in (page._product_rows or [])}
    assert shown == {str(outs[other])}, f"成品区还是上一个视频的：{shown}"
    page.tbl_products.selectRow(0)
    info = page._product_info(page.selected_product())
    assert info is not None and info["path"] == str(outs[other]), \
        f"「打开成品」会开错文件：{info and info['path']}"


# ------------------------------------------------------------------ T_GUI_25
def test_dir_filter_is_saved_and_survives_clear(tmp_path: Path) -> None:
    """原视频目录筛选：只看这个子目录、存进全局配置，而且「清掉筛选」不动它。"""
    cfg, db, made, _window, view = center(tmp_path)
    page = view.videos
    inside, _vid, _ids = made[0]
    other_dir = tmp_path / "别的子目录"
    other_dir.mkdir()
    other = other_dir / "别处.mp4"
    other.write_bytes(b"o" * 4096)
    db_repo.upsert_video(db, other)
    page.reload()
    names = {str(row["file_name"]) for row in page._rows}
    assert {inside.name, "别处.mp4"} <= names, f"两个目录的视频都该在：{names}"

    index = next((i for i in range(page.cmb_video_dir.count())
                  if page.cmb_video_dir.itemData(i)
                  and Path(str(page.cmb_video_dir.itemData(i))).resolve()
                  == other_dir.resolve()), -1)
    assert index > 0, f"目录下拉里应该有 {other_dir}：" \
                      f"{[page.cmb_video_dir.itemData(i) for i in range(page.cmb_video_dir.count())]}"
    picked = str(page.cmb_video_dir.itemData(index))
    page.cmb_video_dir.setCurrentIndex(index)     # 走和用户点选一样的路（选完即存 + 刷新）
    names = {str(row["file_name"]) for row in page._rows}
    assert names == {"别处.mp4"}, f"选了目录就只看这个目录：{names}"
    data = json.loads((cfg.root / "config.json").read_text(encoding="utf-8"))
    assert data["assets"]["filter_video_dir"] == picked, "目录筛选要落进全局配置"

    page.on_clear_filters()
    assert str(page.cmb_video_dir.currentData()) == picked, \
        "「清掉筛选」不许把手动挑的目录作用域清掉"
    names = {str(row["file_name"]) for row in page._rows}
    assert names == {"别处.mp4"}, f"清筛选之后目录作用域还在：{names}"


def scripted_prm_dialog(prm, *, name=None, source=None, text=None, pick=None,
                        accepted: bool = True):
    """把 PRM 的「新增 / 修改」弹窗脚本化：不真弹窗，按参数填好字段再返回结果。

    走的是真实的 `PrmEditDialog`（真控件、真 on_pick、真 payload），只替换 `exec_`，
    这样「弹窗 → 保存 → 写库」整条路都被测到，测试又不会卡在模态窗口上。
    """
    original = prm._dialog

    def make(row=None):
        dlg = original(row)
        if pick is not None:
            ad.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(pick), ""))
            dlg.on_pick()
        if name is not None:
            dlg.edit_name.setText(name)
        if source is not None:
            dlg.edit_file.setText(source)
        if text is not None:
            dlg.view_text.setPlainText(text)
        dlg.exec_ = lambda: (QDialog.Accepted if accepted else QDialog.Rejected)
        return dlg

    prm._dialog = make
    return prm


# ------------------------------------------------------------------ T_GUI_13
def test_prm_text_is_edited_against_the_database(tmp_path: Path) -> None:
    """PRM 正文以库为准：选中就把老文件导进库，改名 / 改正文都只动库，磁盘文件不碰。"""
    cfg, db, _made, _window, view = center(tmp_path)
    prm = view.prm_panel
    prm.reload()
    prm.table.selectRow(0)
    prm_id = prm.selected()
    assert prm_id is not None, "选不中 PRM"
    source = cfg.root / "prm" / "rules.txt"

    # 选中就顺手把老库里「正文还在文件里」的那份导进库
    assert str(db_assets.get_prm(db, prm_id)["content"]) == "剪辑规则", \
        "选中之后正文还没进库"
    assert "正文待导入" not in col(prm.table, 0, 4), \
        f"正文进库之后状态列不该再催导入：{col(prm.table, 0, 4)}"
    assert "rules.txt" in prm.table.item(0, 1).toolTip(), \
        f"来源文件挂在名称的提示里：{prm.table.item(0, 1).toolTip()}"

    # 改正文：一次写回库，来源文件一个字都不动
    scripted_prm_dialog(prm, text="新的剪辑规则")
    prm.on_modify()
    assert str(db_assets.get_prm(db, prm_id)["content"]) == "新的剪辑规则", "正文没写进库"
    assert source.read_text(encoding="utf-8") == "剪辑规则", "保存正文不该回写来源文件"

    # 改名也只动库，正文跟着这份档案不变（名称 + 正文一次写完，不存在半途丢失）
    scripted_prm_dialog(prm, name="PRM V2")
    prm.on_modify()
    row = db_assets.get_prm(db, prm_id)
    assert str(row["name"]) == "PRM V2", "改名没生效"
    assert str(row["content"]) == "新的剪辑规则", "改名不该动正文"
    assert col(prm.table, 0, 1) == "PRM V2", "表里的名字没跟着改"

    # 新增：只填名称 + 正文，连来源文件都不用给
    before = {int(r["id"]) for r in db_assets.list_prms(db)}
    scripted_prm_dialog(prm, name="手写规则", source="", text="只在库里的正文")
    prm.on_new()
    fresh = [r for r in db_assets.list_prms(db) if int(r["id"]) not in before]
    assert len(fresh) == 1, f"新增了 {len(fresh)} 份，应该正好 1 份"
    new = fresh[0]
    assert str(new["name"]) == "手写规则", "新增的名字不对"
    assert str(new["content"]) == "只在库里的正文", "新增的正文没进库"
    assert str(new["filename"]) == "手写规则.txt", f"来源文件该按名字兜底：{new['filename']}"
    assert not (cfg.root / "手写规则.txt").exists(), "新增 PRM 不该往磁盘写文件"
    assert prm.selected() == int(new["id"]), "新增之后该选中它"


# ------------------------------------------------------------------ T_GUI_14
def test_importing_a_file_fills_the_editor_without_writing_the_db(tmp_path: Path) -> None:
    """「导入文件…」当场把正文读进弹窗（库还不动），点「保存」才登记成一份新 PRM。"""
    cfg, db, _made, _window, view = center(tmp_path)
    prm = view.prm_panel
    prm.reload()
    prm.table.selectRow(0)
    old_id = prm.selected()
    old_text = str(db_assets.get_prm(db, old_id)["content"] or "")

    source = cfg.root / "prm" / "另一套.txt"
    source.write_text("另一套剪辑规则", encoding="utf-8")
    picked = ad.QFileDialog.getOpenFileName
    ad.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(source), ""))
    try:
        # 弹窗里导入：正文当场读进来，名称留空按文件名兜底，库一个字都不动
        dlg = prm._dialog()
        dlg.on_pick()
        assert dlg.view_text.toPlainText() == "另一套剪辑规则", "导入没把正文读进弹窗"
        assert dlg.edit_file.text() == str(source), "来源文件框没填上"
        assert dlg.edit_name.text() == "另一套", "名称留空时该按文件名兜底"
        assert str(db_assets.get_prm(db, old_id)["content"] or "") == old_text, \
            "导入只读进弹窗，不许动库里选中的那份"
        assert len(db_assets.list_prms(db)) == 1, "导入不许自己偷偷新建一份"

        # 空文件：明确拒绝，不许把空正文带进弹窗
        empty = cfg.root / "prm" / "空的.txt"
        empty.write_text("   \n", encoding="utf-8")
        ad.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(empty), ""))
        blank = prm._dialog()
        blank.on_pick()
        assert blank.view_text.toPlainText() == "", "空文件不该被导入"
        assert blank.edit_file.text() == "", "空文件连来源都不该记"
        # 正文空着点「保存」也必须被挡住
        blank.edit_name.setText("空规则")
        blank.accept()
        assert blank.result() != QDialog.Accepted, "正文空着不许保存"

        before = {int(r["id"]) for r in db_assets.list_prms(db)}
        scripted_prm_dialog(prm, pick=source)
        prm.on_new()
        fresh = [r for r in db_assets.list_prms(db) if int(r["id"]) not in before]
        assert len(fresh) == 1, f"点「保存」该正好多出 1 份，实际 {len(fresh)}"
        assert str(fresh[0]["name"]) == "另一套", "新增的名字不对"
        assert str(fresh[0]["content"]) == "另一套剪辑规则", "导入的正文没跟着进库"
    finally:
        ad.QFileDialog.getOpenFileName = picked


# ------------------------------------------------------------------ T_GUI_15
def test_renaming_a_prm_has_its_own_button_and_keeps_the_text(tmp_path: Path) -> None:
    """改名走「修改」弹窗：名称 + 正文一次写回；撞名被挡；取消不写库。"""
    cfg, db, _made, _window, view = center(tmp_path)
    prm = view.prm_panel
    prm.reload()
    prm.table.selectRow(0)
    prm_id = prm.selected()

    # 一级动作里不许再出现「先存这个再存那个」那套按钮
    texts = [b.text() for b in prm.findChildren(QPushButton)]
    for gone in ("保存档案", "保存名称等信息", "保存正文", "保存修改"):
        assert gone not in texts, f"同一个动作不要摆两个按钮：还有「{gone}」"
    assert "修改" in texts and "新增" in texts, f"清单页要有「修改 / 新增」：{texts}"

    scripted_prm_dialog(prm, name="规则甲")
    prm.on_modify()
    assert str(db_assets.get_prm(db, prm_id)["name"]) == "规则甲", "改名没写进库"
    assert str(db_assets.get_prm(db, prm_id)["content"]) == "剪辑规则", "改名不该动正文"

    # 撞名：库里名字唯一，撞了要说清楚而不是静悄悄失败
    second = db_assets.create_prm(db, "规则乙", "乙.txt", content="乙的正文")
    prm.reload()
    prm.select(second)
    scripted_prm_dialog(prm, name="规则甲")
    prm.on_modify()
    assert str(db_assets.get_prm(db, second)["name"]) == "规则乙", "撞名不该改成功"
    assert str(db_assets.get_prm(db, second)["content"]) == "乙的正文", "撞名不该动正文"

    # 取消：弹窗里改了半天，点取消就一个字都不许写库
    prm.select(second)
    scripted_prm_dialog(prm, name="规则丁", text="丁的正文", accepted=False)
    prm.on_modify()
    row = db_assets.get_prm(db, second)
    assert str(row["name"]) == "规则乙" and str(row["content"]) == "乙的正文", \
        "点取消不该写库"

    # 导入正文 → 顺手改个名 → 保存：名称和正文一起进库，不会互相顶掉
    source = cfg.root / "prm" / "丙.txt"
    source.write_text("丙的正文", encoding="utf-8")
    picked = ad.QFileDialog.getOpenFileName
    try:
        prm.select(second)
        scripted_prm_dialog(prm, pick=source, name="规则丙")
        prm.on_modify()
    finally:
        ad.QFileDialog.getOpenFileName = picked
    row = db_assets.get_prm(db, second)
    assert str(row["name"]) == "规则丙", "改名没生效"
    assert str(row["content"]) == "丙的正文", "改名把刚导入的正文丢了"
    assert str(row["filename"]) == str(source), "来源文件该记成刚导入那个"



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
    test_center_dirs_are_its_own_config_keys,
    test_center_scan_registers_its_own_videos,
    test_video_row_shows_its_folder,
    test_json_and_product_cells_open_the_popups,
    test_products_follow_the_selected_video_after_reload,
    test_prm_text_is_edited_against_the_database,
    test_importing_a_file_fills_the_editor_without_writing_the_db,
    test_renaming_a_prm_has_its_own_button_and_keeps_the_text,
    test_dir_filter_is_saved_and_survives_clear,
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
