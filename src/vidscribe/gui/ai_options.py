"""AI 选项对话框：AI 走哪条路、用哪个模型、扩展怎么跑，都在这儿改。

改完直接写回 config.json（只写这几个键，文件里其它内容不动），下次「发给 AI」立刻生效；
只有端口要重启 GUI 才换得过去，因为 Bridge 服务在启动时就绑好了。
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
)

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]


class AiOptionsDialog(QDialog):
    """bridge 那一节配置的编辑器。点「保存」才落盘。"""

    def __init__(self, cfg: Any, parent=None, log=None):
        super().__init__(parent)
        self.cfg = cfg
        self._log = log
        self.setWindowTitle("AI 选项")
        self.setMinimumWidth(460)
        bridge = cfg.bridge

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

    def sync_enabled(self) -> None:
        """按选的路子灰掉用不上的项，免得改了半天不生效还以为坏了。"""
        api = self.cmb_mode.currentData() == "api"
        for w in (self.edit_key, self.cmb_model, self.spin_timeout):
            w.setEnabled(api)
        for w in (self.cmb_upload, self.chk_side, self.chk_focus):
            w.setEnabled(not api)

    def save(self) -> None:
        old_port = int(self.cfg.bridge.get("port") or 5998)
        patch = {"bridge": {
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
        try:
            path = self.cfg.save_patch(patch)
        except OSError as exc:
            QMessageBox.warning(self, "AI 选项", f"写 config.json 失败：{exc}")
            return
        if self._log:
            mode = "接口直连" if patch["bridge"]["mode"] == "api" else "网页版扩展"
            self._log(f"[AI 选项] 已保存到 {path}：{mode}，模型 "
                      f"{patch['bridge']['api_model']}，端口 {patch['bridge']['port']}")
        if int(self.spin_port.value()) != old_port:
            QMessageBox.information(self, "AI 选项",
                                    "端口改了，要重开 GUI 才会换过去；"
                                    "扩展选项页里的地址也要跟着改。")
        self.accept()
