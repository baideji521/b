"""AI 选项对话框：AI 走哪条路、用哪个模型、目录放哪儿、扩展怎么跑，都在这儿改。

改完直接写回 config.json（只写涉及的键，文件里其它内容不动），下次「发送_AI」立刻生效；
只有端口要重启 GUI 才换得过去，因为 Bridge 服务在启动时就绑好了。
输出目录跟第一行的「导出目录…」是同一个设置，改哪边都算数——存在 gui_settings.json 里。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]


class DropDirEdit(QLineEdit):
    """能直接把文件夹拖进来的路径输入框。

    拖文件进来也认——取它所在的文件夹，省得你先打开一层再拖。
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("把文件夹拖进来，或点右边「浏览…」")

    @staticmethod
    def _folder_from(event) -> str | None:
        data = event.mimeData()
        if not data.hasUrls():
            return None
        for url in data.urls():
            local = url.toLocalFile()
            if not local:
                continue
            path = Path(local)
            if path.is_dir():
                return str(path)
            if path.exists():  # 拖进来的是文件：用它所在的目录
                return str(path.parent)
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 的命名
        if self._folder_from(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._folder_from(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        folder = self._folder_from(event)
        if folder:
            self.setText(folder)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


def _open_dir(widget, raw: str) -> None:
    """在文件管理器里打开这个目录；不存在就问一句要不要建。"""
    text = (raw or "").strip()
    if not text:
        QMessageBox.information(widget, "AI 选项", "先填个目录")
        return
    path = Path(text)
    if not path.is_dir():
        ok = QMessageBox.question(widget, "AI 选项", f"目录还不存在：\n{path}\n\n要现在建吗？",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ok != QMessageBox.Yes:
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(widget, "AI 选项", f"建不了：{exc}")
            return
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - 打开自己选的目录
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class AiOptionsDialog(QDialog):
    """bridge 那一节配置 + 输入/输出目录的编辑器。点「保存」才落盘。"""

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self._window = parent  # 用来同步导出目录（跟第一行那个按钮共用一个设置）
        self.setWindowTitle("AI 选项")
        self.setMinimumWidth(560)
        bridge = cfg.bridge

        # --- 目录 ---
        self.edit_input = DropDirEdit(str(cfg.path("input_dir")))
        self.edit_output = DropDirEdit(self._current_export_dir())

        # --- AI 怎么跑 ---
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("接口直连（不开浏览器，要 API key）", "api")
        self.cmb_mode.addItem("网页版扩展（用浏览器里的 Gemini）", "extension")
        idx = self.cmb_mode.findData(str(bridge.get("mode") or "api"))
        self.cmb_mode.setCurrentIndex(max(0, idx))

        self.edit_key = QLineEdit(str(bridge.get("api_key") or ""))
        self.edit_key.setEchoMode(QLineEdit.Password)
        self.edit_key.setPlaceholderText("留空则读环境变量 GEMINI_API_KEY")
        self.edit_key.setToolTip("去 https://aistudio.google.com/apikey 领；"
                                 "不想写进仓库就设环境变量")

        self.cmb_model = QComboBox()
        self.cmb_model.setEditable(True)
        self.cmb_model.addItems(MODELS)
        self.cmb_model.setCurrentText(str(bridge.get("api_model") or MODELS[0]))

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(30, 3600)
        self.spin_timeout.setSuffix(" 秒")
        self.spin_timeout.setValue(int(float(bridge.get("api_timeout") or 300)))

        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(int(bridge.get("port") or 5998))
        self.spin_port.setToolTip("Bridge 监听端口，扩展选项页要填同一个；改完要重开 GUI")

        self.cmb_upload = QComboBox()
        self.cmb_upload.addItem("自动拖文件", "auto")
        self.cmb_upload.addItem("我自己选文件", "manual")
        idx = self.cmb_upload.findData(str(bridge.get("upload_mode") or "auto"))
        self.cmb_upload.setCurrentIndex(max(0, idx))

        self.chk_side = QCheckBox("Gemini 放到不抢焦点的小窗口")
        self.chk_side.setChecked(bool(bridge.get("side_window", True)))
        self.chk_side.setToolTip("后台标签页会被浏览器冻结，什么都干不了；独立窗口照常渲染")
        self.chk_focus = QCheckBox("允许浏览器跳到前台")
        self.chk_focus.setChecked(bool(bridge.get("focus_browser", False)))
        self.chk_clip = QCheckBox("拿到 JSON 就自动开剪")
        self.chk_clip.setChecked(bool(bridge.get("auto_clip", True)))

        hint = QLabel("接口直连纯后台跑，失败原因明确；网页版扩展要开着浏览器，"
                      "而且窗口被完全盖住时页面会被冻结。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("输入目录", self._dir_row(self.edit_input, "选择输入目录"))
        form.addRow("输出目录", self._dir_row(self.edit_output, "选择输出目录"))
        form.addRow("走哪条路", self.cmb_mode)
        form.addRow("API key", self.edit_key)
        form.addRow("模型", self.cmb_model)
        form.addRow("超时", self.spin_timeout)
        form.addRow("Bridge 端口", self.spin_port)
        form.addRow("扩展上传方式", self.cmb_upload)
        form.addRow(self.chk_side)
        form.addRow(self.chk_focus)
        form.addRow(self.chk_clip)
        form.addRow(hint)
        form.addRow(buttons)
        self.cmb_mode.currentIndexChanged.connect(self.sync_enabled)
        self.sync_enabled()

    # ------------------------------------------------------------- 组件
    def _dir_row(self, edit: DropDirEdit, title: str):
        """一行：路径框 + 浏览 + 打开。返回装好的容器控件。"""
        from PyQt5.QtWidgets import QWidget  # noqa: PLC0415

        browse = QPushButton("浏览…")
        browse.clicked.connect(lambda: self._browse(edit, title))
        opener = QPushButton("打开")
        opener.clicked.connect(lambda: _open_dir(self, edit.text()))
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        row.addWidget(browse)
        row.addWidget(opener)
        return box

    def _browse(self, edit: DropDirEdit, title: str) -> None:
        start = edit.text().strip() or str(self.cfg.root)
        chosen = QFileDialog.getExistingDirectory(self, title, start)
        if chosen:
            edit.setText(chosen)

    def _current_export_dir(self) -> str:
        getter = getattr(self._window, "export_root", None)
        if callable(getter):
            return str(getter())
        return str(self.cfg.path("output_dir"))

    def sync_enabled(self) -> None:
        """按选的路子灰掉用不上的项，免得改了半天不生效还以为坏了。"""
        api = self.cmb_mode.currentData() == "api"
        for w in (self.edit_key, self.cmb_model, self.spin_timeout):
            w.setEnabled(api)
        for w in (self.cmb_upload, self.chk_side, self.chk_focus):
            w.setEnabled(not api)

    # ------------------------------------------------------------- 保存
    def save(self) -> None:
        old_port = int(self.cfg.bridge.get("port") or 5998)
        patch: dict[str, Any] = {"bridge": {
            "mode": self.cmb_mode.currentData(),
            "api_key": self.edit_key.text().strip(),
            "api_model": self.cmb_model.currentText().strip(),
            "api_timeout": int(self.spin_timeout.value()),
            "port": int(self.spin_port.value()),
            "upload_mode": self.cmb_upload.currentData(),
            "side_window": self.chk_side.isChecked(),
            "focus_browser": self.chk_focus.isChecked(),
            "auto_clip": self.chk_clip.isChecked(),
        }}
        indir = self.edit_input.text().strip()
        if indir:
            patch["paths"] = {"input_dir": indir}
        try:
            path = self.cfg.save_patch(patch)
        except OSError as exc:
            QMessageBox.warning(self, "AI 选项", f"写 config.json 失败：{exc}")
            return

        # 输出目录存在 gui_settings.json 里，跟第一行的「导出目录…」共用
        outdir = self.edit_output.text().strip()
        apply_export = getattr(self._window, "apply_export_dir", None)
        if outdir and callable(apply_export):
            apply_export(Path(outdir))

        if self._log:
            mode = "接口直连" if patch["bridge"]["mode"] == "api" else "网页版扩展"
            self._log(f"[AI 选项] 已保存到 {path}：{mode}，模型 "
                      f"{patch['bridge']['api_model']}，端口 {patch['bridge']['port']}")
            self._log(f"[AI 选项] 输入目录 {indir or '（没改）'}；输出目录 {outdir or '（没改）'}")
        if int(self.spin_port.value()) != old_port:
            QMessageBox.information(self, "AI 选项",
                                    "端口改了，要重开 GUI 才会换过去；"
                                    "扩展选项页里的地址也要跟着改。")
        self.accept()
