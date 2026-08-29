"""AI 选项对话框：找哪家 AI、走哪条路、用哪个模型、目录放哪儿、扩展怎么跑，都在这儿改。

改完直接写回 config.json（只写涉及的键，文件里其它内容不动），下次「发送_AI」立刻生效；
只有端口要重启 GUI 才换得过去，因为 Bridge 服务在启动时就绑好了。
AI_输入目录 / AI_输出目录 只归 AI 用：跟界面第一行的「导入文件」「导出目录…」互不相干，
留空就按老规矩来（合并导出落 cache/，AI 自动剪的成品落导出目录）。

提供方（Gemini / DeepSeek）各有自己的 key 和模型，切换时这两栏会跟着换，互不覆盖。
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

from ..bridge import providers


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
        self._window = parent

        self.setWindowTitle("AI 选项")
        self.setMinimumWidth(560)
        bridge = cfg.bridge

        # --- 目录（只归 AI 用，跟界面的导入/导出目录各走各的）---
        self.edit_input = DropDirEdit(str(bridge.get("ai_input_dir") or ""))
        self.edit_output = DropDirEdit(str(bridge.get("ai_output_dir") or ""))
        self.edit_input.setToolTip("发给 AI 的合并 txt 放这儿。留空＝放 cache/ 并在任务结束后删掉")
        self.edit_output.setToolTip("AI 自动剪的高光成品放这儿。留空＝放界面上选的导出目录")


        # --- 找哪家 AI ---
        self.cmb_provider = QComboBox()
        for name, spec in providers.PROVIDERS.items():
            self.cmb_provider.addItem(spec["label"], name)
        self._provider = providers.normalize(bridge.get("provider"))
        self.cmb_provider.setCurrentIndex(max(0, self.cmb_provider.findData(self._provider)))
        # 每家的 key / 模型各存一份，切来切去不会互相覆盖，保存时一起落盘
        self._draft = {name: {"api_key": str(providers.node(bridge, name).get("api_key") or ""),
                              "api_model": str(providers.settings(bridge, name)["api_model"])}
                       for name in providers.PROVIDERS}

        # --- AI 怎么跑 ---
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("接口直连（不开浏览器，要 API key）", "api")
        self.cmb_mode.addItem("网页版扩展（用浏览器里的对话页）", "extension")
        idx = self.cmb_mode.findData(str(bridge.get("mode") or "api"))
        self.cmb_mode.setCurrentIndex(max(0, idx))

        self.edit_key = QLineEdit()
        self.edit_key.setEchoMode(QLineEdit.Password)

        self.cmb_model = QComboBox()
        self.cmb_model.setEditable(True)

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
        self.cmb_upload.addItem("只看我操作（扩展不动手）", "observe")
        self.cmb_upload.setToolTip("只看我操作：页面打开后扩展一个键都不点，只把你碰过的"
                                   "元素记进日志，用来查它该点哪里")
        idx = self.cmb_upload.findData(str(bridge.get("upload_mode") or "auto"))
        self.cmb_upload.setCurrentIndex(max(0, idx))

        self.chk_side = QCheckBox("对话页放到不抢焦点的小窗口")
        self.chk_side.setChecked(bool(bridge.get("side_window", True)))
        self.chk_side.setToolTip("后台标签页会被浏览器冻结，什么都干不了；独立窗口照常渲染")
        self.chk_focus = QCheckBox("允许浏览器跳到前台")
        self.chk_focus.setChecked(bool(bridge.get("focus_browser", False)))
        self.chk_clip = QCheckBox("拿到 JSON 就自动开剪")
        self.chk_clip.setChecked(bool(bridge.get("auto_clip", True)))

        self.hint = QLabel()
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("AI_输入目录", self._dir_row(self.edit_input, "选择 AI_输入目录"))
        form.addRow("AI_输出目录", self._dir_row(self.edit_output, "选择 AI_输出目录"))

        form.addRow("找哪家 AI", self.cmb_provider)
        form.addRow("走哪条路", self.cmb_mode)
        form.addRow("API key", self.edit_key)
        form.addRow("模型", self.cmb_model)
        form.addRow("超时", self.spin_timeout)
        form.addRow("Bridge 端口", self.spin_port)
        form.addRow("扩展上传方式", self.cmb_upload)
        form.addRow(self.chk_side)
        form.addRow(self.chk_focus)
        form.addRow(self.chk_clip)
        form.addRow(self.hint)
        form.addRow(buttons)
        self.cmb_mode.currentIndexChanged.connect(self.sync_enabled)
        self.cmb_provider.currentIndexChanged.connect(self.on_provider_changed)
        self.load_provider(self._provider)
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

    # --------------------------------------------------------- 提供方切换

    def on_provider_changed(self) -> None:
        """换提供方：先把当前这家的 key / 模型收进草稿，再摊开新那家的。"""
        self.stash_provider()
        self._provider = providers.normalize(self.cmb_provider.currentData())
        self.load_provider(self._provider)
        self.sync_enabled()

    def stash_provider(self) -> None:
        draft = self._draft.setdefault(self._provider, {})
        draft["api_key"] = self.edit_key.text().strip()
        draft["api_model"] = self.cmb_model.currentText().strip()

    def load_provider(self, name: str) -> None:
        spec = providers.PROVIDERS[name]
        draft = self._draft.get(name, {})
        self.edit_key.setText(str(draft.get("api_key") or ""))
        self.edit_key.setPlaceholderText(f"留空则读环境变量 {spec['key_env']}")
        self.edit_key.setToolTip(f"去 {spec['key_page']} 领；不想写进仓库就设环境变量")
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.addItems(spec["models"])
        self.cmb_model.setCurrentText(str(draft.get("api_model") or spec["models"][0]))
        self.cmb_model.blockSignals(False)
        self.hint.setText(
            f"接口直连纯后台跑，失败原因明确（{spec['label']} key 去 {spec['key_page']} 领）；"
            f"网页版扩展要开着浏览器上 {spec['ai_url']}，而且窗口被完全盖住时页面会被冻结。")

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
        self.stash_provider()
        bridge: dict[str, Any] = {
            "mode": self.cmb_mode.currentData(),
            "provider": self._provider,
            "api_timeout": int(self.spin_timeout.value()),
            "port": int(self.spin_port.value()),
            "upload_mode": self.cmb_upload.currentData(),
            "side_window": self.chk_side.isChecked(),
            "focus_browser": self.chk_focus.isChecked(),
            "auto_clip": self.chk_clip.isChecked(),
            # AI 专属目录：只写进 bridge，不碰 paths.input_dir 也不碰导出目录
            "ai_input_dir": self.edit_input.text().strip(),
            "ai_output_dir": self.edit_output.text().strip(),
        }

        # 每家的 key / 模型写回各自那一节（Gemini 是 bridge 下的老键，DeepSeek 在 bridge.deepseek）
        for name, draft in self._draft.items():
            section = providers.section_for(name)
            if section:
                bridge.setdefault(section, {}).update(draft)
            else:
                bridge.update(draft)
        patch: dict[str, Any] = {"bridge": bridge}
        try:
            path = self.cfg.save_patch(patch)
        except OSError as exc:
            QMessageBox.warning(self, "AI 选项", f"写 config.json 失败：{exc}")
            return

        if self._log:
            mode = "接口直连" if bridge["mode"] == "api" else "网页版扩展"
            spec = providers.PROVIDERS[self._provider]
            self._log(f"[AI 选项] 已保存到 {path}：{mode}，{spec['label']} "
                      f"{self._draft[self._provider]['api_model']}，端口 {bridge['port']}")
            self._log(f"[AI 选项] AI_输入目录 {bridge['ai_input_dir'] or '（留空，用 cache/）'}；"
                      f"AI_输出目录 {bridge['ai_output_dir'] or '（留空，用导出目录）'}")

        if int(self.spin_port.value()) != old_port:
            QMessageBox.information(self, "AI 选项",
                                    "端口改了，要重开 GUI 才会换过去；"
                                    "扩展选项页里的地址也要跟着改。")
        self.accept()
