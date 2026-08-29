"""PyQt5 GUI：左侧视频播放器，右侧事件时间轴，底部语音文本 + 日志。

点击时间轴条目或语音行 -> 播放器跳到对应真实秒数。
数据全部来自 output/<视频名>/ 下的 JSON，GUI 不做任何时间推算。

三个列表（事件时间轴 / 语音 / 日志）都支持右键：全选、复制、编辑、删除、清空。
编辑和删除会即时写回对应的 JSON，原文另有 original_text 兜底，重新分析可复原。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,

    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..constants import VIDEO_SUFFIXES
from ..db import open_db
from ..db import repo as db_repo
from ..db.importer import refresh_from_disk
from ..emotions import display_name as emotion_display
from . import flow
from .flow import FlowLayout
from ..progress import parse as parse_progress
from ..speech.sentences import split_sentences
from ..translate import needs_translation

from ..timeline.exporters import (
    fmt_time,
    multi_speaker,
    speaker_tag,
    txt_words,
    words_of,
    write_capcut_srt,
    write_events_txt,
    write_json,
    write_merged_txt,
    write_speech_txt,
    write_words_txt,
)

from . import theme
from . import settings as gui_settings
from .player import FramePlayer

IMPORTANCE_COLOR = {
    "low": QColor("#8a8a8a"),
    "normal": QColor("#e6e6e6"),
    "high": QColor("#ffb454"),
    "critical": QColor("#ff5f5f"),
}
PLAYING_COLOR = QColor(theme.PLAYING)   # 正在播放的字幕
NORMAL_TEXT_COLOR = QColor(theme.TEXT)  # 播过去恢复的原色


def _emotion_cell(emotion_en: Any, intensity: Any, language: str, stored: Any = None) -> str:
    """情绪单元格文本：按当前显示语言现渲，没判到就显示 -。

    切「翻译」看译文时显示名要跟着译文语言变，所以不能直接用 JSON 里存的显示名。
    老结果没有英文标签时用存的显示名反查（emotions.display_name 认中英两种写法）。
    """
    name = emotion_display(emotion_en, stored, language)
    if not name:
        return "-"
    if isinstance(intensity, (int, float)):
        return f"{name} {float(intensity):.2f}"
    return str(name)


class AnalyzeWorker(QThread):
    """用子进程跑分析流水线 / 翻译。

    刻意不在 GUI 进程里 import torch：torch 会占住显存，分开进程更稳，
    而且日志/进度可以实时回传。
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(dict)
    done = pyqtSignal(bool, str)

    def __init__(self, root: Path, argv: list[str], label: str = "分析"):
        super().__init__()
        self.root = root
        self.argv = argv
        self.label = label
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        python = self.root / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            python = Path(sys.executable)
        cmd = [str(python), str(self.root / "run.py"), *self.argv]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["VIDSCRIBE_PROGRESS"] = "json"  # 让子进程输出机器可读的进度行
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        env.pop("QT_PLUGIN_PATH", None)

        self.log.emit("$ " + " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                line = line.rstrip()
                payload = parse_progress(line)
                if payload is not None:
                    self.progress.emit(payload)  # 进度行不进日志面板，免得刷屏
                    continue
                self.log.emit(line)
            code = self._proc.wait()
        except Exception as exc:
            self.log.emit(f"[错误] {type(exc).__name__}: {exc}")
            self.done.emit(False, str(exc))
            return
        self.done.emit(code == 0, f"退出码 {code}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


class HighlightDialog(QDialog):
    """剪辑高光的输入窗：上面三个加减秒数（默认 0.00），下面粘贴 AI JSON。

    起剪点 = clip.start + 起始加减
    冻帧点 = clip.end   + 结束加减
    片尾   = 冻帧点     + 文本加减（冻帧+字幕这段的时长）
    overlay.time 不参与计算，只从 overlay 里取字幕文本。
    值会存进 gui_settings.json，下次打开自动带回来。
    """


    

    offsetsChanged = pyqtSignal(float, float, float)
    sfxChanged = pyqtSignal(str, float)

    def __init__(self, parent, text: str, offsets: tuple[float, float, float],
                 peaks: list[dict] | None = None,
                 sfx: tuple[str, float] = ("", -6.0),
                 sfx_categories: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("剪辑高光")
        self.resize(720, 560)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self.spin_start = self._spin(offsets[0])
        self.spin_end = self._spin(offsets[1])
        self.spin_text = self._spin(offsets[2])
        self.spin_text.setMinimum(0.0)   # 片尾 = 冻帧点 + 本值，负数会让片尾跑到冻帧点之前


        for row, (label, spin, tip) in enumerate((
            ("起始 加减秒数", self.spin_start, "起剪点 = clip.start + 本值；负数提前起剪"),
            ("结束 加减秒数", self.spin_end, "冻帧点 = clip.end + 本值；正数晚一点冻结"),
            ("文本 加减秒数", self.spin_text, "冻帧+字幕这段的时长；0 只留一帧，不能填负数"),

        )):
            spin.setToolTip(tip)
            # 改一下就往外抛一次，交给主窗口存盘，取消也不会丢
            spin.valueChanged.connect(self._emit_offsets)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(QLabel(tip), row, 2)

        # 冻帧音效：冻帧段原本是纯静音，这里选混哪一类（自动 = 按冻帧点的表情查配置映射）
        self.combo_sfx = QComboBox()
        self.combo_sfx.addItem("自动（按表情）", "")
        self.combo_sfx.addItem("不加音效", "none")
        for name in (sfx_categories or []):
            self.combo_sfx.addItem(name, name)
        index = self.combo_sfx.findData(sfx[0])
        self.combo_sfx.setCurrentIndex(index if index >= 0 else 0)
        self.combo_sfx.currentIndexChanged.connect(self._emit_sfx)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setDecimals(1)
        self.spin_gain.setRange(-40.0, 6.0)
        self.spin_gain.setSingleStep(1.0)
        self.spin_gain.setSuffix(" dB")
        self.spin_gain.setValue(float(sfx[1]))
        self.spin_gain.valueChanged.connect(self._emit_sfx)

        row = grid.rowCount()
        grid.addWidget(QLabel("冻帧音效"), row, 0)
        sfx_row = QHBoxLayout()
        sfx_row.addWidget(self.combo_sfx, 1)
        sfx_row.addWidget(self.spin_gain)
        grid.addLayout(sfx_row, row, 1)
        hint = ("混在冻帧那一刻（原本是静音）；音效库为空就跑 tools/fetch_sfx.py"
                if sfx_categories else "音效库为空，先跑 tools/fetch_sfx.py 下载 CC0 音效")
        grid.addWidget(QLabel(hint), row, 2)
        grid.setColumnStretch(2, 1)

        self.edit = QPlainTextEdit(text)
        self.edit.setPlaceholderText("在这里粘贴 AI JSON")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("开始剪辑")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        if peaks:
            # 语音情绪最强的几句：情绪爆点通常就是想冻住的那一瞬间，这里只给参考不自动改 JSON
            # 这个对话框的文案本来就是中文，情绪也统一按中文显示名渲，别混英文标签
            tips = "　".join(
                f"{p.get('freeze_at', 0):.2f}s "
                f"{emotion_display(p.get('emotion_en'), p.get('emotion'), 'zh') or '?'}"
                f"({float(p.get('intensity') or 0):.2f})" for p in peaks[:5])
            hint = QLabel(f"情绪高光候选（冻帧点参考）：{tips}")
            hint.setWordWrap(True)
            hint.setToolTip("来自语音情绪识别，按情绪强度排序；时间是该句说完的时刻")
            layout.addWidget(hint)
        layout.addWidget(self.edit, 1)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-600.0, 600.0)
        spin.setSingleStep(0.05)
        spin.setSuffix(" 秒")
        spin.setValue(float(value))
        return spin

    def payload(self) -> str:
        return self.edit.toPlainText()

    def _emit_offsets(self, *_args) -> None:
        self.offsetsChanged.emit(*self.offsets())

    def _emit_sfx(self, *_args) -> None:
        self.sfxChanged.emit(*self.sfx())

    def sfx(self) -> tuple[str, float]:
        """(类别, 增益dB)。类别空串 = 自动按表情选，"none" = 不加音效。"""
        return (str(self.combo_sfx.currentData() or ""), round(self.spin_gain.value(), 1))

    def offsets(self) -> tuple[float, float, float]:
        return (round(self.spin_start.value(), 2), round(self.spin_end.value(), 2),
                round(self.spin_text.value(), 2))



class AiApiWorker(QThread):
    """直接调 AI 接口问高光 JSON。不开浏览器、不用扩展，纯后台一次请求。

    走哪家（Gemini / DeepSeek）由 provider 决定，实现在 bridge/providers.py 里分发。
    网页版那条路依赖 DOM 和窗口可见性，太脆；这条只有网络会失败，失败原因明确。
    """

    log = pyqtSignal(str)
    done = pyqtSignal(str, str)  # (回答正文, 错误)

    def __init__(self, provider: str, api_key: str, model: str, prompt_text: str,
                 merged_text: str, message: str, timeout: float, base_url: str = ""):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.prompt_text = prompt_text
        self.merged_text = merged_text
        self.message = message
        self.timeout = timeout
        self.base_url = base_url

    def run(self) -> None:
        from ..bridge import providers  # noqa: PLC0415

        try:
            text = providers.ask(self.provider, self.api_key, self.prompt_text,
                                 self.merged_text, self.message, self.model,
                                 self.timeout, self.base_url)
        except providers.AiError as exc:
            self.done.emit("", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 网络层什么都可能抛，别让线程静默死掉
            self.done.emit("", f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(text, "")


class HighlightWorker(QThread):
    """剪辑高光：按 AI JSON 起剪 -> 到冻帧点抓帧冻结 -> 冻帧特效 + 逐字字幕 -> 片尾收尾。


    渲染是纯 CPU 的解码/画图/编码，放线程里跑，日志逐行发回界面。
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)  # (已完成帧, 总帧, 阶段)
    done = pyqtSignal(bool, str)

    def __init__(self, cfg, payload_text: str, fallback: Path | None,
                 export_dir: Path | None, offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 sfx: tuple[str, float] = ("", -6.0), video_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.payload_text = payload_text
        self.fallback = fallback
        self.export_dir = export_dir
        self.offsets = offsets
        self.sfx = sfx
        # AI 自动剪辑走这条：只落 <视频名>_高光时刻.mp4，不写同名 .json
        self.video_only = video_only
        self.output: Path | None = None


    def _sfx_plan(self, video: Path, freeze_time: float):
        """按冻帧点的表情挑一条音效。表情来自该视频的 timeline.json，读不到就走兜底类别。"""
        from ..highlight import plan as plan_sfx  # noqa: PLC0415

        category, gain_db = self.sfx
        cfg_sfx = dict(self.cfg.highlight.get("sfx") or {})
        cfg_sfx["gain_db"] = gain_db
        root = Path(cfg_sfx.get("dir") or "assets/sfx")
        if not root.is_absolute():
            root = self.cfg.root / root

        timeline = None
        path = self.cfg.path("output_dir") / video.stem / "timeline.json"
        if path.is_file():
            try:
                timeline = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - 读不到就当没有表情数据
                self.log.emit(f"[SFX] timeline.json 读取失败，改用兜底类别：{exc}")
        else:
            self.log.emit(f"[SFX] 找不到 {path}，没有表情数据，改用兜底类别")

        chosen = plan_sfx(cfg_sfx, root, timeline, freeze_time,
                          key=f"{video.stem}|{freeze_time:.2f}", category=category)
        if chosen.path is None:
            self.log.emit(f"[SFX] 不混音效：{chosen.reason}")
        return chosen

    def run(self) -> None:
        try:
            from ..highlight import (  # noqa: PLC0415 - 重依赖（av/cv2/PIL）延迟导入
                default_target, parse_spec, render_highlight, resolve_video,
            )
            from ..timeline.exporters import write_json  # noqa: PLC0415

            spec = parse_spec(json.loads(self.payload_text))
            start_delta, end_delta, text_delta = self.offsets
            self.log.emit(f"[OFFSET] JSON 原始 clip.start={spec.clip_start:.2f} "
                          f"clip.end={spec.clip_end:.2f}")
            spec = spec.shifted(start_delta, end_delta, text_delta)
            self.log.emit(f"[OFFSET] 起始{start_delta:+.2f} / 结束{end_delta:+.2f} / "
                          f"文本{text_delta:+.2f} 秒 → 起剪={spec.clip_start:.2f} "
                          f"冻帧={spec.freeze_time:.2f} 片尾={spec.clip_end:.2f}")

            video = resolve_video(spec, self.cfg.path("output_dir"),
                                  self.cfg.path("input_dir"), self.fallback)
            # 和文本导出共用「导出目录」；没设过就退回该视频的结果目录
            directory = (self.export_dir if self.export_dir and self.export_dir.is_dir()
                         else self.cfg.path("output_dir") / video.stem)
            target = default_target(directory, video)
            result = render_highlight(video, spec, target, on_log=self.log.emit,
                                      on_progress=self.progress.emit,
                                      sfx=self._sfx_plan(video, spec.freeze_time))
            if not self.video_only:
                write_json(target.with_suffix(".json"),
                           {"spec": spec.raw,
                            "offsets": {"start": start_delta, "end": end_delta,
                                        "text": text_delta},

                            "result": result})

        except Exception as exc:
            self.log.emit(f"[错误] {type(exc).__name__}: {exc}")
            self.done.emit(False, f"{type(exc).__name__}: {exc}")
            return
        self.output = target
        self.done.emit(True, str(target))


class AudioWorker(QThread):
    """把音轨解成 wav（PyAV），供播放器出声。耗时很短但仍放线程里，避免卡界面。"""

    done = pyqtSignal(str, str)  # (wav 路径或空, 错误说明)

    def __init__(self, video: Path, target: Path):
        super().__init__()
        self.video = video
        self.target = target

    def run(self) -> None:
        try:
            from ..audio import extract_wav  # noqa: PLC0415

            path = extract_wav(self.video, self.target)
        except Exception as exc:
            self.done.emit("", f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(str(path) if path else "", "" if path else "这个视频没有可用音轨")


class BridgeEvents(QObject):
    """把 Bridge 的 HTTP 线程事件搬到 GUI 线程：跨线程 emit 走队列连接是安全的。"""

    event = pyqtSignal(str, object)


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, video: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.video_path: Path | None = None
        self.timeline: list[dict[str, Any]] = []
        self.speech: list[dict[str, Any]] = []
        self.timeline_doc: dict[str, Any] = {}
        self.speech_doc: dict[str, Any] = {}
        self.worker: AnalyzeWorker | None = None
        self.audio_worker: AudioWorker | None = None
        self.clip_worker: HighlightWorker | None = None
        self.ai_worker: AiApiWorker | None = None
        # 浏览器扩展对接（Bridge）：GUI 起服务，扩展轮询领任务去驱动网页版 AI
        self.bridge = None
        self._bridge_events: BridgeEvents | None = None
        self._bridge_token = ""
        # 发给扩展的临时文件（合并导出），任务结束就删
        self._bridge_temp_files: list[Path] = []
        self._last_highlight_json = ""
        # 「自动剪辑」的队列在数据库里（ai_tasks），这里只留跑当前这一条要用的东西：
        # 关掉程序再开，没跑完的任务还在，能接着跑
        self._auto_task_id: int | None = None
        self._auto_active = False
        self._auto_done = 0            # 只用来显示进度，状态不看它
        self._auto_total = 0
        self._auto_video: Path | None = None
        self._auto_job = ""
        # 状态都从数据库读，句柄按需打开；打不开就退回「什么都没有」，界面照样能用
        self._db_handle = None
        self._db_failed = False
        # 最近一次真正发出去的提示词指纹（hash/path/size），只做追溯记录，不参与上传
        self._last_prompt: dict[str, Any] | None = None
        # AI 面板（第二主界面）：非模态，只开一个
        self.ai_panel = None


        # 剪辑高光的三个加减秒数（起始 / 结束 / 文本），从设置里带回来

        self._highlight_offsets = (0.0, 0.0, 0.0)
        # 冻帧音效：(类别, 增益dB)，类别空串 = 自动按表情选
        self._highlight_sfx = ("", float((cfg.highlight.get("sfx") or {}).get("gain_db", -6.0)))
        self.show_translated = False
        self._translate_request: dict[str, str] = {}
        self._translate_result: Path | None = None
        # 启动时先把上次的设置读出来，_build_ui 之后再套到控件上
        self.settings = gui_settings.load(cfg)
        self._loading_settings = True
        saved_export = self.settings.get("export_dir")
        self.export_dir: Path | None = Path(saved_export) if saved_export else None
        self._columns_user_sized = False
        self._suppress_column_signal = False
        self._playing_speech = -1  # 当前标绿的那句字幕在列表里的行号
        self._playing_row = -1     # 当前标绿的画面事件在表格里的行号
        # 拖分隔条、拉列宽会连着触发几十次，攒一下再写设置文件
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_settings)

        # 当前这条自动剪辑任务的心跳：Qwen / Whisper / FFmpeg / 等 AI 回话都可能比
        # ai_task_timeout_minutes 还久，只要一直在刷心跳，恢复逻辑就不会把它当死任务捞走。
        # 刷得比超时快得多（取超时的四分之一，10 秒到 60 秒之间）
        timeout_ms = float(cfg.runtime.get("ai_task_timeout_minutes", 30) or 30) * 60_000
        self._hb_timer = QTimer(self)
        self._hb_timer.setInterval(int(max(10_000, min(60_000, timeout_ms / 4))))
        self._hb_timer.timeout.connect(self._touch_auto_task)
        self._hb_timer.start()


        self.setWindowTitle(theme.APP_TITLE)
        self.setWindowIcon(theme.app_icon())
        self.resize(1500, 900)
        self.setAcceptDrops(True)  # 支持把视频文件直接拖进窗口
        self._build_ui()
        self.apply_settings()
        self.check_cache()
        self.start_bridge()   # apply_settings 之后才有令牌，起服务要排在它后面



        if video:
            self.load_video(video)
        # 上次没跑完的自动剪辑任务：等窗口先显示出来，再捞回来接着跑
        QTimer.singleShot(0, self._resume_auto_queue)


    # --------------------------------------------------------------- 设置
    def _splitters(self) -> tuple[tuple[QSplitter, str], ...]:
        """三块可拖的分区：左右（播放器/时间轴）、底部（语音/日志）、上下。"""
        return ((self.split_main, "split_main"),
                (self.split_bottom, "split_bottom"),
                (self.split_vertical, "split_vertical"))

    def schedule_save(self, *_args) -> None:
        """拖动分隔条或列宽时连续触发，攒 400ms 只写一次设置文件。"""
        if self._loading_settings:
            return
        self._save_timer.start(400)

    def apply_settings(self) -> None:
        """把上次退出时的参数套回控件。坏值一律忽略，不让设置文件把界面弄崩。"""
        s = self.settings
        model = s.get("visual_model")
        if model:
            idx = self.cmb_model.findData(model)
            if idx >= 0:
                self.cmb_model.setCurrentIndex(idx)
        speaker = s.get("speaker_model")
        if speaker:
            idx = self.cmb_speaker.findData(speaker)
            if idx >= 0:
                self.cmb_speaker.setCurrentIndex(idx)
        index = s.get("importance_index")
        if isinstance(index, int) and 0 <= index < self.cmb_importance.count():
            self.cmb_importance.setCurrentIndex(index)
        conf = s.get("confidence")
        if isinstance(conf, (int, float)):
            self.spin_conf.setValue(float(conf))
        if isinstance(s.get("auto_translate"), bool):
            self.chk_auto_translate.setChecked(bool(s["auto_translate"]))
        if isinstance(s.get("emotion_audio"), bool):
            self.chk_emotion_audio.setChecked(bool(s["emotion_audio"]))
        if isinstance(s.get("emotion_visual"), bool):
            self.chk_emotion_visual.setChecked(bool(s["emotion_visual"]))
        if isinstance(s.get("play_sound"), bool):
            # 这时还没打开视频，prepare_audio 会自己跳过；等 load_video 再解音轨
            self.chk_sound.setChecked(bool(s["play_sound"]))
        for splitter, key in self._splitters():
            sizes = s.get(key)
            if (isinstance(sizes, list) and len(sizes) == splitter.count()
                    and all(isinstance(v, int) and v >= 0 for v in sizes) and sum(sizes) > 0):
                splitter.setSizes(sizes)
        widths = s.get("timeline_columns")
        # 存过的列宽只在“看起来正常”时才用：画面列窄于 160px 的状态没法看，宁可重新自适应
        if (isinstance(widths, list) and len(widths) == self.table.columnCount()
                and all(isinstance(v, int) and v >= 60 for v in widths) and widths[2] >= 160):
            self._set_widths(widths, mark_user=True)
        row_height = s.get("timeline_row_height")
        if isinstance(row_height, int) and row_height > 0:
            self.set_row_height(row_height)
        offsets = s.get("highlight_offsets")
        if (isinstance(offsets, list) and len(offsets) == 3
                and all(isinstance(v, (int, float)) for v in offsets)):
            self._highlight_offsets = tuple(round(float(v), 2) for v in offsets)
        saved_sfx = s.get("highlight_sfx")
        if (isinstance(saved_sfx, list) and len(saved_sfx) == 2
                and isinstance(saved_sfx[0], str) and isinstance(saved_sfx[1], (int, float))):
            self._highlight_sfx = (saved_sfx[0], round(float(saved_sfx[1]), 1))
        token = s.get("bridge_token")
        # 配对令牌存在设置里，重启后扩展不用重新配对
        if isinstance(token, str) and token.strip():
            self._bridge_token = token.strip()
        geo = s.get("window")
        if isinstance(geo, list) and len(geo) == 4 and all(isinstance(v, int) for v in geo):
            self.setGeometry(*geo)
        if s.get("maximized"):
            self.showMaximized()
        self._loading_settings = False
        self.refresh_export_hint()

    def save_settings(self) -> None:
        if self._loading_settings:  # 启动套用设置时别把默认值写回去
            return
        # 最大化时 geometry() 是全屏尺寸，存 normalGeometry 才能还原成还原后的窗口
        rect = self.normalGeometry() if self.isMaximized() else self.geometry()
        if rect.width() <= 0 or rect.height() <= 0:
            rect = self.geometry()
        self.settings.update({
            "visual_model": self.cmb_model.currentData(),
            "speaker_model": self.cmb_speaker.currentData(),
            "importance_index": self.cmb_importance.currentIndex(),
            "confidence": round(float(self.spin_conf.value()), 3),
            "play_sound": bool(self.chk_sound.isChecked()),
            "auto_ai": bool(self.chk_auto_ai.isChecked()),
            "auto_translate": bool(self.chk_auto_translate.isChecked()),
            "emotion_audio": bool(self.chk_emotion_audio.isChecked()),
            "emotion_visual": bool(self.chk_emotion_visual.isChecked()),
            "export_dir": str(self.export_dir) if self.export_dir else None,
            "last_video_dir": str(self.video_path.parent) if self.video_path else
                              self.settings.get("last_video_dir"),
            "window": [rect.x(), rect.y(), rect.width(), rect.height()],
            "maximized": bool(self.isMaximized()),
            "timeline_columns": [self.table.columnWidth(c)
                                 for c in range(self.table.columnCount())],
            "timeline_row_height": self.table.verticalHeader().defaultSectionSize(),
            "highlight_offsets": list(self._highlight_offsets),
            "highlight_sfx": list(self._highlight_sfx),
            "bridge_token": self._bridge_token,
        })
        for splitter, key in self._splitters():
            self.settings[key] = splitter.sizes()
        gui_settings.save(self.cfg, self.settings)


    # ------------------------------------------------------------- 高级选项
    def on_advanced(self) -> None:
        """高级选项对话框：视觉模型、声纹、过滤、自动翻译、两路情绪。

        里面放的就是原来主界面上那几个控件本身（第一次打开时 reparent 进来），
        所以已有的信号连接和存盘逻辑一行都不用改，勾选状态照旧实时写 gui_settings.json。
        """
        if self._advanced_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("高级选项")
            form = QFormLayout(dialog)
            form.addRow("视觉模型", self.cmb_model)
            form.addRow("声纹", self.cmb_speaker)
            form.addRow("重要性", self.cmb_importance)
            form.addRow("画面事件", self.spin_conf)
            form.addRow(self.chk_auto_translate)
            form.addRow(self.chk_emotion_audio)
            form.addRow(self.chk_emotion_visual)
            btn_cache = QPushButton("缓存管理…")
            btn_cache.setToolTip("看每份缓存占多少、多久没动过，自己挑着删。缓存不再自动清理")
            btn_cache.clicked.connect(self.on_cache_manager)
            form.addRow("缓存", btn_cache)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            self._advanced_dialog = dialog
        self._advanced_dialog.show()
        self._advanced_dialog.raise_()

    def on_cache_manager(self) -> None:
        """缓存管理：列出每份缓存，勾选删除。开软件不再自动删任何东西。"""
        from .cache_dialog import CacheDialog  # noqa: PLC0415

        CacheDialog(self.cfg, self, log=self.append_log).exec_()

    def on_ai_options(self) -> None:
        """开 AI 面板（第二主界面）：模式、目录、任务表、自动剪辑都在那儿。

        非模态而且只开一个：再点一次是把它拎到前面，不会又叠一个出来。
        """
        from .ai_options import AiPanel  # noqa: PLC0415

        if self.ai_panel is None:
            self.ai_panel = AiPanel(self.cfg, self, log=self.append_log)
            self.ai_panel.finished.connect(lambda *_: self.on_ai_panel_closed())
        self.ai_panel.set_running(self.auto_running())
        self.refresh_bridge_label()  # 让面板右上角的连接状态立刻跟上
        self.ai_panel.show()

        self.ai_panel.raise_()
        self.ai_panel.activateWindow()

    def on_ai_api(self) -> None:
        """AI 接口设置：找哪家 AI、走接口还是网页版扩展、key、模型、端口、上传方式。"""
        from .ai_options import AiApiDialog  # noqa: PLC0415

        dialog = AiApiDialog(self.cfg, self, log=self.append_log)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_bridge_label()

    def on_ai_panel_closed(self) -> None:
        self.ai_panel = None
        self.refresh_bridge_label()



    # ------------------------------------------------------------- 导出目录
    def export_root(self) -> Path:
        """导出文件默认放哪儿：用户选过就用它，没选过就用当前视频的结果目录。"""
        if self.export_dir is not None and self.export_dir.is_dir():
            return self.export_dir
        out = self.output_dir()
        return out if out is not None else self.cfg.path("output_dir")

    def refresh_export_hint(self) -> None:
        self.btn_export_dir.setToolTip(f"导出文件放到：{self.export_root()}")

    def on_pick_export_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择导出目录", str(self.export_root()))
        if not chosen:
            return
        self.apply_export_dir(Path(chosen))

    def apply_export_dir(self, path: Path) -> None:
        """设导出目录。第一行的「导出目录…」用这个（AI 那两个目录另存，互不影响）。"""
        self.export_dir = Path(path)
        self.save_settings()
        self.refresh_export_hint()
        self.append_log(f"[导出目录] {self.export_dir}")
        self.statusBar().showMessage(f"导出目录：{self.export_dir}")

    # ------------------------------------------------------- AI 自己的目录
    def ai_dir(self, key: str) -> Path | None:
        """读 bridge.ai_input_dir / ai_output_dir。留空或建不出来就返回 None（按老规矩走）。"""
        raw = str(self.cfg.bridge.get(key) or "").strip()
        if not raw:
            return None
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.append_log(f"[AI 目录] {key} 建不了：{path} {exc}；这次按默认目录走")
            return None
        return path


    # --------------------------------------------------------------- 缓存
    def cache_dir_for_video(self) -> Path:
        """当前视频的缓存目录（cache/videos/<视频标识>/），没打开视频时用 _gui。"""
        from .. import cache as cache_mod  # noqa: PLC0415

        root = self.cfg.path("cache_dir")
        if self.video_path is None:
            path = cache_mod.videos_root(root) / "_gui"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return cache_mod.video_dir_in(root, self.video_path)



    def check_cache(self) -> None:
        """开软件只看一眼缓存现状，一个字节都不删。

        清理都是手动的：「高级选项 -> 缓存管理」里自己挑，或者跑 `python run.py cache --clean`。
        """
        from .. import cache as cache_mod  # noqa: PLC0415

        max_age = float(self.cfg.runtime.get("cache_max_age_days", 3))
        try:
            info = cache_mod.scan_on_start(self.cfg, max_age_days=max_age)
        except Exception as exc:  # 缓存检查失败不该挡着用软件
            self.append_log(f"缓存检查失败（不影响使用）：{type(exc).__name__}: {exc}")
            return
        self.append_log(cache_mod.summary_line(info))


    # ------------------------------------------------------------------ UI
    @staticmethod
    def _section(text: str) -> QLabel:
        """面板小标题：统一走 QSS 里的 role=section 样式。"""
        label = QLabel(text)
        label.setProperty("role", "section")
        return label

    def _build_ui(self) -> None:
        # 工具栏用自动换行布局：窗口拖窄时按钮往下折行，不再把最小宽度顶死（见 gui/flow.py）
        top = FlowLayout(spacing=8)
        top.setContentsMargins(2, 2, 2, 6)
        self.btn_open = QPushButton("打开视频")
        self.btn_open.clicked.connect(self.on_open)
        self.btn_analyze = QPushButton("分析当前视频")
        self.btn_analyze.setProperty("role", "primary")
        self.btn_analyze.clicked.connect(lambda: self.on_analyze(False))
        self.btn_reanalyze = QPushButton("重新分析（忽略缓存）")
        self.btn_reanalyze.clicked.connect(lambda: self.on_analyze(True))
        self.btn_outdir = QPushButton("打开导出目录")
        self.btn_outdir.setToolTip("打开导出文件所在的目录（分析结果目录由 config.json 固定，界面不改它）")
        self.btn_outdir.clicked.connect(self.on_open_outdir)

        # 分析结束时模型还在显存里，顺手翻译只花解码时间；单独点"翻译"要重新加载模型（约 15s）
        self.chk_auto_translate = QCheckBox("分析后自动翻译")
        self.chk_auto_translate.setChecked(True)
        self.chk_auto_translate.setToolTip("分析结束时模型还在显存里，这时候翻译最快（省掉约 15s 模型加载）")
        self.chk_auto_translate.toggled.connect(self.save_settings)

        # 两路情绪各自一个开关：勾选状态实时存盘，分析时透传成 CLI 参数
        ecfg = self.cfg.speech.get("emotion", {})
        self.chk_emotion_audio = QCheckBox("音频情绪")
        self.chk_emotion_audio.setChecked(bool(ecfg.get("enabled", True)))
        self.chk_emotion_audio.setToolTip(
            "听声音判情绪（emotion2vec+），按语音的每句话逐段判。"
            "要额外加载一个约 1.8GB 的模型，实测多花十几秒"
        )
        self.chk_emotion_audio.toggled.connect(self.save_settings)

        self.chk_emotion_visual = QCheckBox("画面情绪")
        self.chk_emotion_visual.setChecked(bool(self.cfg.visual.get("emotion_enabled", True)))
        self.chk_emotion_visual.setToolTip(
            "看画面判情绪（表情/姿态），由视觉模型在描述画面的同一次推理里顺便给出，"
            "不额外加载模型；改这个开关后要重新分析才会生效"
        )
        self.chk_emotion_visual.toggled.connect(self.save_settings)



        self.cmb_importance = QComboBox()
        self.cmb_importance.addItems(["全部", "normal 以上", "high 以上", "仅 critical"])
        self.cmb_importance.currentIndexChanged.connect(self.refresh_timeline_table)
        self.cmb_importance.currentIndexChanged.connect(self.save_settings)

        # 视觉模型切换：只影响下一次分析，不动已有结果
        from ..visual.factory import known_models  # noqa: PLC0415

        self.visual_models = known_models(self.cfg.visual)
        self.cmb_model = QComboBox()
        for entry in self.visual_models:
            self.cmb_model.addItem(f"{entry['label']}  [{entry['backend']}]", entry["model_id"])
        current = str(self.cfg.visual.get("model_id") or "")
        idx = self.cmb_model.findData(current)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)
        self.cmb_model.setToolTip("切换视觉模型，切换后点“重新分析”生效")
        # 下拉框默认按最长条目算最小宽度（实测 863px），窗口就被顶死拉不窄了；限宽 + 文字省略
        self.cmb_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.cmb_model.setMinimumContentsLength(16)
        self.cmb_model.setMaximumWidth(260)
        self.cmb_model.currentIndexChanged.connect(self.save_settings)

        # 声纹模型（说话人分离）：只有英文/中文两个选择，默认英文。
        # 「英文」用的是中英双语那份 cam++ —— 3D-Speaker 的纯英文 VoxCeleb 权重 funasr 加载不了。
        scfg = self.cfg.speech.get("speaker", {})
        self.cmb_speaker = QComboBox()
        entries = scfg.get("models") or [
            {"label": "英文", "model_id": "iic/speech_campplus_sv_zh_en_16k-common_advanced"},
            {"label": "中文", "model_id": "iic/speech_campplus_sv_zh-cn_16k-common"},
        ]
        for entry in entries:
            self.cmb_speaker.addItem(str(entry.get("label") or entry.get("model_id")),
                                     str(entry.get("model_id")))
        idx = self.cmb_speaker.findData(str(scfg.get("model_id") or ""))
        if idx >= 0:
            self.cmb_speaker.setCurrentIndex(idx)
        self.cmb_speaker.setToolTip(
            "给每句语音标「这是谁说的」用的声纹模型，按素材语言选。"
            "人数由声纹自己定，不固定几人；分不出来时判成 1 人而不是给假答案。"
            "切换后要重新分析才生效"
        )
        self.cmb_speaker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.cmb_speaker.setMinimumContentsLength(4)
        self.cmb_speaker.setMaximumWidth(120)
        self.cmb_speaker.currentIndexChanged.connect(self.save_settings)


        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.0)
        self.spin_conf.setPrefix("置信度≥ ")
        self.spin_conf.valueChanged.connect(self.refresh_timeline_table)
        self.spin_conf.valueChanged.connect(self.save_settings)

        # 不常改的都收进「高级选项」对话框，主界面只留每天要点的
        self.btn_advanced = QPushButton("高级选项")
        self.btn_advanced.setToolTip("视觉模型、声纹、重要性/置信度过滤、分析后自动翻译、两路情绪开关")
        self.btn_advanced.clicked.connect(self.on_advanced)
        self._advanced_dialog: QDialog | None = None

        for w in (self.btn_open, self.btn_analyze, self.btn_reanalyze):
            top.addWidget(w)

        # 导出行：三个文本导出挪到播放器右键里了，这里只留剪辑
        export_row = FlowLayout(spacing=8)
        export_row.setContentsMargins(2, 0, 2, 4)
        self.btn_export_dir = QPushButton("导出目录…")
        self.btn_export_dir.clicked.connect(self.on_pick_export_dir)
        self.btn_highlight = QPushButton("剪辑高光")
        self.btn_highlight.setToolTip("粘贴 AI JSON：clip.start 起剪，clip.end 抓帧冻结，"
                                     "冻帧上做特效 + 逐字字幕，片尾由「文本 加减秒数」决定；"
                                     "输出到导出目录，文件名带 _高光时刻")

        self.btn_highlight.clicked.connect(self.on_highlight)
        # 选目录和打开目录挨着放在第一行：先选，再打开
        top.addWidget(self.btn_export_dir)
        top.addWidget(self.btn_outdir)

        # --- AI 对接（浏览器扩展 Bridge）---
        # 合并导出 + 提示词交给扩展，扩展在浏览器里问网页版 AI，回传的 JSON 直接进剪辑高光
        self.lbl_bridge = QLabel("未启动")
        self.lbl_bridge.setProperty("role", "pill")  # 药丸样式，见 theme.QSS
        self.lbl_bridge.setAlignment(Qt.AlignCenter)
        # 最长的状态文案是「:65535 配对窗口 120s」，按它留够宽度，别让字被切掉
        self.lbl_bridge.setMinimumWidth(
            self.lbl_bridge.fontMetrics().horizontalAdvance(":65535 配对窗口 120s") + 32)
        self.lbl_bridge.setToolTip("本机 Bridge 服务状态。扩展轮询它领任务")

        self.btn_bridge_pair = QPushButton("配对扩展")
        self.btn_bridge_pair.setToolTip("打开 120 秒配对窗口，扩展会自动把令牌领走；"
                                       "扩展选项页里的地址要填这里显示的端口")
        self.btn_bridge_pair.clicked.connect(self.on_bridge_pair)
        # 令牌摆在明面上：扩展选项页里要填它，配对领不到时可以手抄
        self.lbl_token_tag = QLabel("Token:")
        self.edit_token = QLineEdit()
        self.edit_token.setReadOnly(True)
        # 令牌是 32 个字符的 urlsafe 串，按字宽算够整串显示，不用横向拖
        self.edit_token.setMinimumWidth(
            self.edit_token.fontMetrics().horizontalAdvance("M" * 34) + 16)

        self.edit_token.setPlaceholderText("Bridge 没启动")
        self.edit_token.setToolTip("这台机器 Bridge 的令牌。点「配对扩展」扩展会自动领；"
                                  "领不到就复制它，粘到扩展选项页里")
        self.btn_token_copy = QPushButton("复制 Token")
        self.btn_token_copy.setToolTip("把令牌复制到剪贴板")
        self.btn_token_copy.clicked.connect(self.on_copy_token)

        self.btn_ai_options = QPushButton("AI 面板")
        self.btn_ai_options.setToolTip("第二主界面：干哪一串、AI 专属目录、任务统计和任务表，"
                                      "自动剪辑也在那儿点")
        self.btn_ai_options.clicked.connect(self.on_ai_options)
        self.btn_ai_api = QPushButton("AI接口")
        self.btn_ai_api.setToolTip("找哪家 AI、走接口直连还是网页版扩展、API key、模型、"
                                   "超时、Bridge 端口、扩展上传方式")
        self.btn_ai_api.clicked.connect(self.on_ai_api)

        # 「自动」勾上：分析一跑完就自己把文本发给 AI，拿到 JSON 再按 auto_clip 开剪，
        # 整条链路不用你点。状态存 gui_settings.json
        self.chk_auto_ai = QCheckBox("自动")
        self.chk_auto_ai.setToolTip("分析完自动发给 AI（回来的 JSON 照旧按「拿到 JSON 就自动开剪」处理）")
        self.chk_auto_ai.setChecked(bool(self.settings.get("auto_ai", False)))
        self.chk_auto_ai.toggled.connect(lambda *_: self.save_settings())
        self.btn_bridge_send = QPushButton("发送_AI")
        self.btn_bridge_send.setToolTip("把合并导出的文本 + 高光筛选提示词发给 AI，"
                                        "回来的 JSON 直接开剪。走哪条路在「AI 选项」里选")
        self.btn_bridge_send.clicked.connect(self.on_bridge_send)
        self.btn_bridge_stop = QPushButton("停止_AI")
        self.btn_bridge_stop.setToolTip("取消正在跑的 AI 任务（自动剪辑排着的队也一并中止）")
        self.btn_bridge_stop.clicked.connect(self.on_bridge_stop)


        # 第二行按用起来的顺序排：发给 AI -> 停止 -> 剪辑高光。
        # 端口状态和配对按钮是扩展那边的杂事，钉到最右边，跟第一行的「高级选项」对齐
        export_row.addWidget(self.btn_highlight)



        # --- 左侧播放器（OpenCV 逐帧渲染画面，音轨走 QMediaPlayer）---
        self.player = FramePlayer()
        self.player.setMinimumSize(240, 180)  # 给小一点，左侧画面才能被真正拖窄
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.stateChanged.connect(
            lambda playing: self.btn_play.setText("暂停" if playing else "播放")
        )
        self.player.audioFailed.connect(self.on_audio_failed)
        # 右键菜单：添加 / 从播放器移除 / 彻底删除
        self.player.setContextMenuPolicy(Qt.CustomContextMenu)
        self.player.customContextMenuRequested.connect(self.on_player_menu)

        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(lambda ms: self.player.seek(ms / 1000.0))
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setMinimumWidth(110)  # 别占太宽，否则左侧整块拖不窄
        self.chk_sound = QCheckBox("播放声音")
        self.chk_sound.setToolTip("勾选后预览带声音（首次勾选会先从视频里解出音轨）")
        self.chk_sound.toggled.connect(self.on_sound_toggled)
        self.chk_sound.toggled.connect(self.save_settings)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.lbl_time)
        controls.addWidget(self.chk_sound)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.player, 1)
        left_layout.addLayout(controls)

        # --- 右侧时间轴 ---
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["时间", "重要性", "画面", "时间来源", "语音情绪", "画面情绪"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.table.viewport().setAcceptDrops(False)
        self.table.verticalHeader().setVisible(False)
        # 行高可调 + 自动折行：行拉高后「画面」长文本会换行显示，不再只剩一行省略号
        self.table.setWordWrap(True)
        self._default_row_height = self.table.verticalHeader().defaultSectionSize()
        # 画面列拉宽到超出可视区时，横向按像素滚，而不是一整列一整列地跳
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # 每列都能拖，宽度存进设置
        header.setStretchLastSection(False)
        header.setSectionsMovable(True)  # 列顺序也能拖
        header.sectionResized.connect(self.on_section_resized)
        # 表头右键：不想拖那 4px 边界的话，直接从菜单里选列宽方案
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.on_header_menu)
        self.table.cellClicked.connect(self.on_timeline_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_timeline_menu)
        # Ctrl+滚轮 调画面列宽，Ctrl+Shift+滚轮 调行高：不用去掐表头那 4px 边界
        self.table.viewport().installEventFilter(self)
        header.installEventFilter(self)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._section("事件时间轴（点击跳转，右键更多操作）"))
        right_layout.addWidget(self.table, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([760, 740])
        self.split_main = split

        # --- 底部语音 + 日志 ---
        self.speech_list = QListWidget()
        self.speech_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.speech_list.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.speech_list.viewport().setAcceptDrops(False)
        self.speech_list.itemClicked.connect(self.on_speech_clicked)
        self.speech_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.speech_list.customContextMenuRequested.connect(self.on_speech_menu)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))
        # 文本框默认会把拖进来的文件路径当文本插入，关掉它，让拖拽事件冒泡到窗口
        self.log_view.setAcceptDrops(False)
        self.log_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_view.customContextMenuRequested.connect(self.on_log_menu)

        # 进度：阶段名 + 明细 + 总进度条（数据来自子进程的 @@PROGRESS 行）
        self.lbl_stage = QLabel("空闲")
        self.lbl_stage.setMinimumWidth(120)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.lbl_stage)
        progress_row.addWidget(self.progress_bar, 1)

        bottom = QSplitter(Qt.Horizontal)
        speech_box = QWidget()
        sb = QVBoxLayout(speech_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(self._section("语音（点击跳转，右键更多操作）"))
        sb.addWidget(self.speech_list)
        log_box = QWidget()
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.addWidget(self._section("运行日志（右键更多操作）"))
        lb.addLayout(progress_row)
        lb.addWidget(self.log_view)
        bottom.addWidget(speech_box)
        bottom.addWidget(log_box)
        bottom.setSizes([900, 600])
        self.split_bottom = bottom

        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(split)
        vertical.addWidget(bottom)
        vertical.setSizes([600, 280])
        self.split_vertical = vertical
        for splitter, _key in self._splitters():
            splitter.setHandleWidth(8)              # 分隔条给宽一点，好抓
            splitter.setChildrenCollapsible(False)  # 不许拖到 0 宽，免得面板消失找不回来
            splitter.splitterMoved.connect(self.schedule_save)
            # 面板宽度变了，没手动调过列宽的话顺手重新自适应一次
            splitter.splitterMoved.connect(lambda *_: self.autosize_columns())

        central = QWidget()
        layout = QVBoxLayout(central)
        # 两行右侧各自紧凑排列（各控件保持自然宽度，右端都贴着窗口右边）：
        # 第一行 自动 | 发送_AI | 停止_AI | AI接口 | AI 面板
        # 第二行 状态药丸 | Token: | 令牌 | 复制 Token | 配对扩展 | 高级选项
        first_row = QHBoxLayout()
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.addWidget(flow.wrap(top), 1)
        first_row.addWidget(self.chk_auto_ai, 0, Qt.AlignVCenter)
        first_row.addWidget(self.btn_bridge_send, 0, Qt.AlignTop)
        first_row.addWidget(self.btn_bridge_stop, 0, Qt.AlignTop)
        first_row.addWidget(self.btn_ai_api, 0, Qt.AlignTop)
        first_row.addWidget(self.btn_ai_options, 0, Qt.AlignTop)
        layout.addLayout(first_row)
        second_row = QHBoxLayout()
        second_row.setContentsMargins(0, 0, 0, 0)
        second_row.addWidget(flow.wrap(export_row), 1)
        second_row.addWidget(self.lbl_bridge, 0, Qt.AlignVCenter)
        second_row.addWidget(self.lbl_token_tag, 0, Qt.AlignVCenter)
        second_row.addWidget(self.edit_token, 0, Qt.AlignVCenter)
        second_row.addWidget(self.btn_token_copy, 0, Qt.AlignTop)
        second_row.addWidget(self.btn_bridge_pair, 0, Qt.AlignTop)
        second_row.addWidget(self.btn_advanced, 0, Qt.AlignTop)
        layout.addLayout(second_row)
        layout.addWidget(vertical, 1)


        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("把视频拖进窗口，或点左上角“打开视频”")

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(250)
        self._follow_timer.timeout.connect(self.highlight_current)
        self._follow_timer.start()

    # --------------------------------------------------------------- 数据加载
    def output_dir(self) -> Path | None:
        if self.video_path is None:
            return None
        return self.cfg.path("output_dir") / self.video_path.stem

    def load_video(self, video: Path) -> None:
        self.video_path = Path(video).resolve()
        if not self.player.open(self.video_path):
            self.append_log(f"[播放器] 无法解码 {self.video_path.name}")
        self.setWindowTitle(f"{theme.APP_TITLE} — {self.video_path.name}")
        # 换视频后重新准备音轨：勾着声音就直接解，没勾就等勾选时再解
        self.player.set_audio_file(None)
        self.chk_sound.setEnabled(True)
        if self.chk_sound.isChecked():
            self.prepare_audio()
        self.load_results()
        self.save_settings()  # 记住这次是从哪个目录打开的
        self.refresh_export_hint()

    def load_results(self) -> None:
        out = self.output_dir()
        self.timeline, self.speech = [], []
        self.timeline_doc, self.speech_doc = {}, {}
        self.show_translated = False
        if out is None:
            return
        timeline_file = out / "timeline.json"
        speech_file = out / "speech_events.json"
        visual_file = out / "visual_events.json"
        used_model = ""
        if visual_file.is_file():
            try:
                with open(visual_file, "r", encoding="utf-8") as fh:
                    vmeta = json.load(fh).get("meta") or {}
                if vmeta.get("model_id"):
                    used_model = f"，视觉模型 {str(vmeta['model_id']).split('/')[-1]}"
            except Exception:
                used_model = ""
        if timeline_file.is_file():
            try:
                with open(timeline_file, "r", encoding="utf-8") as fh:
                    self.timeline_doc = json.load(fh)
                self.timeline = self.timeline_doc.get("timeline", [])
                self.statusBar().showMessage(
                    f"{self.video_path.name}：{len(self.timeline)} 条时间轴，"
                    f"音频语言 {self.timeline_doc.get('original_language') or self.timeline_doc.get('language') or '无语音'}"
                    f" -> 输出语言 {self.timeline_doc.get('output_language') or '-'}{used_model}"
                )
            except Exception as exc:
                self.append_log(f"[警告] 读取 timeline.json 失败: {exc}")
        else:
            self.statusBar().showMessage(f"{self.video_path.name}：还没有分析结果，点击“分析当前视频”")
        if speech_file.is_file():
            try:
                with open(speech_file, "r", encoding="utf-8") as fh:
                    self.speech_doc = json.load(fh)
                segments = self.speech_doc.get("segments", [])
                # 老结果是"一段多句"的：读进来就按标点切成一句一行，不用重跑分析。
                # 只切显示用的这份，不回写文件；要落盘走「保存语音结果」。
                self.speech = split_sentences(segments)
                if len(self.speech) != len(segments):
                    self.append_log(f"[断句] 已有结果按标点重排：{len(segments)} 段 -> "
                                    f"{len(self.speech)} 行（译文需重新翻译）")
            except Exception as exc:
                self.append_log(f"[警告] 读取 speech_events.json 失败: {exc}")

        self.refresh_timeline_table()
        self.refresh_speech_list()

    def has_translation(self) -> bool:
        """界面上有没有任何译文：决定「翻译」是重新跑模型还是只切换显示。"""
        if any(s.get("text_translated") for s in self.speech):
            return True
        return any(e.get("visual_translated") for e in self.timeline)

    def speech_display(self, seg: dict[str, Any]) -> str:
        if self.show_translated and seg.get("text_translated"):
            return str(seg["text_translated"])
        return str(seg.get("text") or "")

    def visual_display(self, entry: dict[str, Any]) -> str:
        if self.show_translated and entry.get("visual_translated"):
            return str(entry["visual_translated"])
        return str(entry.get("visual") or "")

    def refresh_timeline_table(self) -> None:
        min_rank = {0: -1, 1: 1, 2: 2, 3: 3}[self.cmb_importance.currentIndex()]
        rank = {"low": 0, "normal": 1, "high": 2, "critical": 3}
        min_conf = float(self.spin_conf.value())

        rows = []
        for entry in self.timeline:
            if entry.get("visual"):
                if rank.get(entry.get("importance", "normal"), 1) < min_rank:
                    continue
                if (entry.get("visual_confidence") or 0.0) < min_conf:
                    continue
            elif min_rank > 1 or min_conf > 0:
                continue  # 过滤模式下不显示纯语音条目
            rows.append(entry)

        self._rows = rows
        self._playing_row = -1  # 表格重建了，上一次标绿的行号作废
        self.table.setRowCount(len(rows))
        for i, entry in enumerate(rows):
            time_text = f"{fmt_time(entry['start'])} - {fmt_time(entry['end'])}"
            if entry.get("visual"):
                visual = self.visual_display(entry)
            else:
                speech = self.speech_display({"text": entry.get("speech"),
                                              "text_translated": entry.get("speech_translated")})
                visual = f"（无画面事件）{speech}"[:60]
            # 情绪显示名跟当前视图语言走（看译文就出译文语言），旧结果没有英文标签时兜底用存的显示名
            lang = self.export_language()
            speech_emotion = _emotion_cell(entry.get("speech_emotion_en"),
                                           entry.get("speech_emotion_intensity"), lang,
                                           entry.get("speech_emotion") or entry.get("emotion"))
            visual_emotion = _emotion_cell(entry.get("visual_emotion_en"),
                                           entry.get("visual_emotion_intensity"), lang,
                                           entry.get("visual_emotion"))
            cells = [
                time_text,
                entry.get("importance", "-") if entry.get("visual") else "-",
                visual,
                entry.get("timestamp_source", "-"),
                speech_emotion,
                visual_emotion,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, float(entry["start"]))
                if col == 2 and entry.get("visual"):
                    color = IMPORTANCE_COLOR.get(entry.get("importance", "normal"))
                    if color:
                        item.setForeground(QBrush(color))
                self.table.setItem(i, col, item)
        self.autosize_columns()

    def autosize_columns(self) -> None:
        """按内容 + 剩余宽度排列宽。用户自己拖过或菜单调过之后就不再自动动它。"""
        if self._columns_user_sized or not self.table.rowCount():
            return
        self.fit_columns(mark_user=False)

    def _set_widths(self, widths: list[int], mark_user: bool) -> None:
        """统一改列宽：改的过程里屏蔽 sectionResized，免得被当成用户手动拖动。"""
        self._suppress_column_signal = True
        try:
            for col, width in enumerate(widths):
                self.table.setColumnWidth(col, max(40, width))
        finally:
            self._suppress_column_signal = False
        if mark_user:
            self._columns_user_sized = True
        self.schedule_save()

    def fit_columns(self, mark_user: bool = True) -> None:
        """按内容自适应，"画面"列吃掉剩下的宽度。"""
        self._suppress_column_signal = True
        try:
            self.table.resizeColumnsToContents()
        finally:
            self._suppress_column_signal = False
        header = self.table.horizontalHeader()
        widths = [header.sectionSize(c) for c in range(self.table.columnCount())]
        rest = self.table.viewport().width() - sum(widths) + widths[2] - 4
        widths[2] = max(200, rest)
        self._set_widths(widths, mark_user)

    def step_visual_column(self, delta: int) -> None:
        """「画面」列按固定步长加宽/变窄，不用去掐表头那 4px 边界。"""
        widths = [self.table.columnWidth(c) for c in range(self.table.columnCount())]
        widths[2] = max(120, widths[2] + delta)
        self._set_widths(widths, mark_user=True)

    def set_row_height(self, height: int) -> None:
        """统一改行高：行高归 verticalHeader 管，表格重建后依然生效。"""
        height = min(400, max(self._default_row_height, height))
        self.table.verticalHeader().setDefaultSectionSize(height)
        self.schedule_save()

    def step_row_height(self, delta: int) -> None:
        """行高按固定步长增减，配合自动折行让长文本多显示几行。"""
        self.set_row_height(self.table.verticalHeader().defaultSectionSize() + delta)

    def reset_row_height(self) -> None:
        self.set_row_height(self._default_row_height)

    def eventFilter(self, obj, event):
        """Ctrl+滚轮 = 画面列宽，Ctrl+Shift+滚轮 = 行高。表格区和表头上都生效。"""
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                if event.modifiers() & Qt.ShiftModifier:
                    self.step_row_height(8 if delta > 0 else -8)
                else:
                    self.step_visual_column(40 if delta > 0 else -40)
            return True  # 吃掉事件，别让表格顺带滚动
        return super().eventFilter(obj, event)

    def _add_size_actions(self, menu: QMenu) -> None:
        """列宽/行高调节项：表头右键和表格右键挂的是同一份。"""
        menu.addAction("画面列加宽 (+80)　Ctrl+滚轮↑", lambda: self.step_visual_column(80))
        menu.addAction("画面列变窄 (-80)　Ctrl+滚轮↓", lambda: self.step_visual_column(-80))
        menu.addSeparator()
        menu.addAction("行高增高 (+16)　Ctrl+Shift+滚轮↑", lambda: self.step_row_height(16))
        menu.addAction("行高降低 (-16)　Ctrl+Shift+滚轮↓", lambda: self.step_row_height(-16))
        menu.addSeparator()
        menu.addAction("按内容自适应列宽", self.fit_columns)
        menu.addAction("平均分配列宽", self.spread_columns)
        menu.addAction("恢复默认列宽", self.reset_columns)
        menu.addAction("恢复默认行高", self.reset_row_height)



    def spread_columns(self) -> None:
        """平均分配列宽。"""
        count = self.table.columnCount()
        each = max(60, (self.table.viewport().width() - 4) // count)
        self._set_widths([each] * count, mark_user=True)

    def reset_columns(self) -> None:
        """回到默认列宽，并允许之后再次自动排版。"""
        rest = max(200, self.table.viewport().width() - 130 - 90 - 110 - 4)
        self._set_widths([130, 90, rest, 110], mark_user=False)
        self._columns_user_sized = False

    def on_section_resized(self, *_args) -> None:
        if self._suppress_column_signal:
            return
        self._columns_user_sized = True  # 用户拖过就别再自动重排
        self.schedule_save()

    def on_header_menu(self, pos) -> None:
        menu = QMenu(self)
        self._add_size_actions(menu)
        menu.exec_(self.table.horizontalHeader().mapToGlobal(pos))


    def refresh_speech_list(self) -> None:
        self.speech_list.clear()
        self._playing_speech = -1  # 列表重建了，上一次标绿的行号作废
        # 判出 2 人以上才在每行标说话人，单人素材不加这个前缀
        multi = multi_speaker(self.speech)
        for seg in self.speech:
            conf = seg.get("confidence")
            suffix = f"  (conf {conf:.2f})" if isinstance(conf, (int, float)) else ""
            emotion = emotion_display(seg.get("emotion_en"), seg.get("emotion"),
                                      self.export_language())
            intensity = seg.get("emotion_intensity")
            if emotion:
                suffix += f"  [{emotion}"
                suffix += f" {intensity:.2f}]" if isinstance(intensity, (int, float)) else "]"
            text = self.speech_display(seg)
            who = speaker_tag(seg.get("speaker"), self.export_language()) if multi else ""
            item = QListWidgetItem(
                f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}]{who} {text}{suffix}")
            item.setData(Qt.UserRole, float(seg["start"]))
            self.speech_list.addItem(item)

    # ------------------------------------------------------------------ 落盘
    def save_speech(self) -> None:
        out = self.output_dir()
        if out is None or not self.speech_doc:
            return
        self.speech_doc["segments"] = self.speech
        write_json(out / "speech_events.json", self.speech_doc)
        self.append_log(f"[保存] speech_events.json（{len(self.speech)} 段）")

    def save_timeline(self) -> None:
        out = self.output_dir()
        if out is None or not self.timeline_doc:
            return
        self.timeline_doc["timeline"] = self.timeline
        counts = self.timeline_doc.get("counts")
        if isinstance(counts, dict):
            counts["timeline_entries"] = len(self.timeline)
        write_json(out / "timeline.json", self.timeline_doc)
        self.append_log(f"[保存] timeline.json（{len(self.timeline)} 条）")

    # ------------------------------------------------------------------ 交互
    def input_root(self) -> Path:
        """打开视频对话框从哪儿开。

        你在「AI 选项」里明确指了输入目录（不是默认的 input/）就听你的；
        没指过才按老规矩：当前视频所在目录 -> 上次用过的目录 -> input/。
        """
        configured = self.cfg.path("input_dir")
        if configured.is_dir() and configured != (self.cfg.root / "input"):
            return configured
        if self.video_path is not None and self.video_path.parent.is_dir():
            return self.video_path.parent
        last = self.settings.get("last_video_dir")
        if last and Path(last).is_dir():
            return Path(last)
        return configured if configured.is_dir() else self.cfg.root

    def on_open(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(VIDEO_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(self.input_root()),
                                              f"视频文件 ({patterns})")
        if path:
            self.load_video(Path(path))

    # --------------------------------------------------- 播放器右键：添加 / 删除
    def on_player_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("添加视频…")
        menu.addSeparator()
        # 三个文本导出从上面的按钮行挪到这儿，界面清爽一点
        act_speech = menu.addAction("导出语音文本（SRT 剪映可用 / txt）")
        act_events = menu.addAction("导出事件文本（SRT 剪映可用 / txt）")
        act_merged = menu.addAction("合并导出（画面 + 语音按时间穿插，txt）")
        menu.addSeparator()
        act_close = menu.addAction("删除（从播放器移除）")
        act_purge = menu.addAction("彻底删除（不可恢复）")
        loaded = self.video_path is not None
        for act in (act_speech, act_events, act_merged, act_close, act_purge):
            act.setEnabled(loaded)
        chosen = menu.exec_(self.player.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_add:
            self.on_open()
        elif chosen is act_speech:
            self.export_text("speech")
        elif chosen is act_events:
            self.export_text("events")
        elif chosen is act_merged:
            self.export_text("merged")
        elif chosen is act_close:
            self.close_current_video()
        elif chosen is act_purge:
            self.delete_current_video_file()

    def close_current_video(self) -> None:
        """只从界面上撤下来：磁盘上的视频和已有分析结果都不动。"""
        if self.video_path is None:
            return
        name = self.video_path.name
        self.player.close_video()
        self.video_path = None
        self.timeline, self.speech = [], []
        self.timeline_doc, self.speech_doc = {}, {}
        self.show_translated = False
        self.chk_sound.blockSignals(True)
        self.chk_sound.setChecked(False)
        self.chk_sound.blockSignals(False)
        self.slider.setValue(0)
        self.lbl_time.setText("00:00.00 / 00:00.00")
        self.setWindowTitle(theme.APP_TITLE)
        self.refresh_timeline_table()
        self.refresh_speech_list()
        self.refresh_export_hint()
        self.statusBar().showMessage("已从播放器移除，文件还在原处")
        self.append_log(f"[播放器] 移除 {name}（文件未删除）")

    def delete_current_video_file(self) -> None:
        """把视频文件本身删掉。不进回收站，删完就没了，所以要二次确认。"""
        if self.video_path is None:
            return
        target = self.video_path
        answer = QMessageBox.warning(
            self, "彻底删除",
            f"把这个文件从磁盘上删除，不进回收站、无法恢复：\n\n{target}\n\n确定删除？",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        # 先松开文件：播放器还占着句柄时 Windows 上删不掉
        self.close_current_video()
        try:
            target.unlink()
        except OSError as exc:
            QMessageBox.critical(self, "删除失败", f"{target}\n\n{exc}")
            self.append_log(f"[删除] 失败：{target} —— {exc}")
            return
        self.append_log(f"[删除] 已彻底删除 {target}")
        self.statusBar().showMessage(f"已删除 {target.name}（分析结果仍保留在输出目录）")


    # ------------------------------------------------------------- 拖拽打开
    @staticmethod
    def _videos_in(mime) -> list[Path]:
        """从拖拽数据里挑出视频文件（按扩展名，不猜内容）。"""
        if not mime.hasUrls():
            return []
        found = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found.append(path)
        return found

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._videos_in(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        videos = self._videos_in(event.mimeData())
        if not videos:
            event.ignore()
            return
        event.acceptProposedAction()
        if len(videos) > 1:
            # 界面一次只放一个视频；多选时取第一个，其余提示走命令行批处理
            self.append_log(f"[拖入] 收到 {len(videos)} 个视频，先打开 {videos[0].name}；"
                            f"批量处理请用命令行 run")
        self.load_video(videos[0])

    def busy(self) -> bool:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "已有任务在运行")
            return True
        if self.clip_worker is not None and self.clip_worker.isRunning():
            QMessageBox.information(self, "提示", "高光剪辑还在渲染，等它跑完")
            return True
        return False

    def on_highlight(self) -> None:
        """剪辑高光：粘贴 AI JSON，起剪 / 冻帧 / 收尾三个时间照 JSON 执行，另可再加减秒数。"""
        if self.busy():
            return
        dialog = HighlightDialog(self, self._last_highlight_json, self._highlight_offsets,
                                 (self.speech_doc or {}).get("emotion_peaks"),
                                 self._highlight_sfx, self.sfx_categories())
        dialog.offsetsChanged.connect(self.on_highlight_offsets_changed)
        dialog.sfxChanged.connect(self.on_highlight_sfx_changed)
        if dialog.exec_() != QDialog.Accepted:
            return
        text = dialog.payload()
        self._highlight_offsets = dialog.offsets()
        self._highlight_sfx = dialog.sfx()
        self.schedule_save()  # 加减秒数存进 gui_settings.json，下次打开自动带回来
        if not text.strip():
            return
        self.run_highlight(text)

    def run_highlight(self, text: str, ai: bool = False) -> None:
        """按 JSON 直接起渲染。手动走对话框和 AI 自动回填都汇到这里。

        用的是界面上「剪辑高光」那套配置（加减秒数、音效类别/增益）。
        AI 自动那条只出一个成品：<视频名>_高光时刻.mp4，落在「AI_输出目录」；
        那栏留空才退回界面上选的「导出目录」。
        """
        self._last_highlight_json = text
        self.btn_highlight.setEnabled(False)
        self.set_progress(0.0)
        self.lbl_stage.setText("剪辑高光｜准备中")

        self.statusBar().showMessage("正在渲染高光片段…")
        directory = self.export_dir
        ai_out = self.ai_dir("ai_output_dir") if ai else None
        if ai_out is not None:
            directory = ai_out
        self.clip_worker = HighlightWorker(self.cfg, text, self.video_path, directory,
                                           self._highlight_offsets, self._highlight_sfx,
                                           video_only=ai)
        if ai:
            where = ai_out if ai_out is not None else self.export_root()
            label = "AI_输出目录" if ai_out is not None else "导出目录"
            self.append_log(f"[剪辑高光] 输出到{label}：{where}"
                            f"（加减秒数 {self._highlight_offsets}，音效 {self._highlight_sfx[0] or '自动'}）")


        self.clip_worker.log.connect(self.append_log)
        self.clip_worker.progress.connect(self.on_highlight_progress)
        self.clip_worker.done.connect(self.on_highlight_done)
        self.clip_worker.start()

    def set_progress(self, ratio: float, text: str | None = None) -> None:
        """刷进度条。跑满时整条变绿（theme 里的 `QProgressBar[done="true"]`）。

        Qt 的样式表按属性选择器命中之后不会自己重绘，setProperty 后必须
        unpolish/polish 一次，否则颜色不变。
        """
        ratio = min(1.0, max(0.0, float(ratio)))
        self.progress_bar.setValue(int(round(ratio * 1000)))
        self.progress_bar.setFormat(text if text is not None else f"{ratio * 100:.1f}%")
        done = "true" if ratio >= 1.0 else "false"
        if self.progress_bar.property("done") != done:
            self.progress_bar.setProperty("done", done)
            self.progress_bar.style().unpolish(self.progress_bar)
            self.progress_bar.style().polish(self.progress_bar)

    def on_highlight_progress(self, done: int, total: int, stage: str) -> None:

        """按已写入帧数推进进度条（渲染在子线程里跑，信号回主线程刷界面）。"""
        total = max(total, 1)
        ratio = min(1.0, max(0.0, done / total))
        self.set_progress(ratio)
        self.lbl_stage.setText(f"剪辑高光｜{stage} {done}/{total} 帧")


    def on_highlight_offsets_changed(self, start: float, end: float, text: float) -> None:

        """对话框里一改加减秒数就存盘（400ms 防抖），点取消也留着。"""
        self._highlight_offsets = (round(start, 2), round(end, 2), round(text, 2))

        self.schedule_save()

    def sfx_categories(self) -> list[str]:
        """音效库里现有的类别目录名，给对话框的下拉用；库不在就返回空列表。"""
        from ..highlight import library  # noqa: PLC0415 - 只在开对话框时才用

        root = Path((self.cfg.highlight.get("sfx") or {}).get("dir") or "assets/sfx")
        if not root.is_absolute():
            root = self.cfg.root / root
        return list(library(root))

    def on_highlight_sfx_changed(self, category: str, gain_db: float) -> None:
        """音效类别 / 音量一改就存盘，和加减秒数一样点取消也留着。"""
        self._highlight_sfx = (category, round(float(gain_db), 1))
        self.schedule_save()

    # ------------------------------------------------------- AI 对接（Bridge）
    def start_bridge(self) -> None:
        """起本机 Bridge 服务，供浏览器扩展轮询领任务。端口被占就往后顺延。"""
        cfg = self.cfg.bridge
        if not cfg.get("enabled", True) or self.bridge is not None:
            return
        from ..bridge import BridgeServer  # noqa: PLC0415 - 只有 GUI 用得上

        self._bridge_events = BridgeEvents()
        self._bridge_events.event.connect(self.on_bridge_event)
        emit = self._bridge_events.event.emit
        server = BridgeServer(port=int(cfg.get("port") or 5998),
                              fallbacks=int(cfg.get("port_fallbacks") or 0),
                              token=self._bridge_token,
                              on_event=lambda kind, data: emit(kind, data))
        try:
            server.start()
        except OSError as exc:
            self.append_log(f"[AI 对接] Bridge 起不来：{exc}")
            self.set_bridge_pill("端口被占", "off")
            return
        self.bridge = server
        if self._bridge_token != server.token:
            self._bridge_token = server.token
            self.schedule_save()
        self.append_log(f"[AI 对接] Bridge 监听 {server.url}"
                        f"（扩展选项页填这个地址，然后点「配对扩展」）")
        self._bridge_timer = QTimer(self)
        self._bridge_timer.setInterval(2000)
        self._bridge_timer.timeout.connect(self.refresh_bridge_label)
        self._bridge_timer.start()
        self.refresh_bridge_label()

    def stop_bridge(self) -> None:
        if self.bridge is None:
            return
        self.bridge.stop()
        self.bridge = None

    def refresh_bridge_label(self) -> None:
        """刷状态药丸：文字 + 颜色（绿=通了，琥珀=忙/等配对，红=没通）。顺手刷令牌框。"""
        self.edit_token.setText(self.bridge.token if self.bridge is not None else "")
        if self.bridge is None:
            self.set_bridge_pill("未启动", "off")
            return

        state = self.bridge.state()
        port = state["url"].rsplit(":", 1)[-1]
        task = state["task"]
        if task:
            text, mood = f":{port} 任务中 {task['stage'] or task['status']}", "busy"
        elif state["pair_window_left"] > 0:
            text, mood = f":{port} 配对窗口 {state['pair_window_left']:.0f}s", "busy"
        elif state["extension_online"]:
            text, mood = f":{port} 扩展在线", "ok"
        elif state["paired_at"]:
            text, mood = f":{port} 扩展离线", "off"
        else:
            text, mood = f":{port} 等待配对", "busy"
        from ..bridge import providers  # noqa: PLC0415

        spec = providers.settings(self.cfg.bridge)
        if str(self.cfg.bridge.get("mode") or "api") == "api":
            # 接口直连根本不用扩展，这时候显示扩展在不在线只会让人误会。
            # 提供方和模型名在「AI 选项」里看得到，不用挤在药丸上
            text, mood = "接口直连", "ok"
        self.set_bridge_pill(text, mood)
        self.lbl_bridge.setToolTip(f"Bridge {state['url']}\n"
                                   f"当前 AI：{spec['label']}（{spec['api_model']}）\n"
                                   f"扩展选项页的地址填这个，令牌点「配对扩展」自动领取\n"
                                   f"走哪条路、找哪家 AI、端口、模型都在「AI 选项」里改")

    def set_bridge_pill(self, text: str, mood: str) -> None:
        """药丸文字 + 配色。改了 state 属性必须 unpolish/polish，否则颜色不变。"""
        if self.ai_panel is not None:  # AI 面板右上角那个跟着一起变
            self.ai_panel.set_connection(text, mood)

        self.lbl_bridge.setText(text)
        if self.lbl_bridge.property("state") != mood:
            self.lbl_bridge.setProperty("state", mood)
            self.lbl_bridge.style().unpolish(self.lbl_bridge)
            self.lbl_bridge.style().polish(self.lbl_bridge)

    def on_copy_token(self) -> None:
        """把 Bridge 令牌复制到剪贴板，扩展选项页里手填时用。"""
        if self.bridge is None:
            self.start_bridge()
        if self.bridge is None:
            QMessageBox.warning(self, "AI 对接", "Bridge 没有启动，端口可能被占用")
            return
        QApplication.clipboard().setText(self.bridge.token)
        self.edit_token.setText(self.bridge.token)
        self.append_log("[AI 对接] 令牌已复制到剪贴板")
        self.statusBar().showMessage("令牌已复制到剪贴板", 5000)

    def on_bridge_pair(self) -> None:

        """开一个 120 秒配对窗口：扩展轮询到就自动把令牌领走，不用手抄。"""
        if self.bridge is None:
            self.start_bridge()
        if self.bridge is None:
            QMessageBox.warning(self, "AI 对接", "Bridge 没有启动，端口可能被占用")
            return
        self.bridge.open_pair_window()
        self.append_log(f"[AI 对接] 配对窗口已开（120 秒）。扩展地址：{self.bridge.url}")
        self.refresh_bridge_label()

    def on_bridge_send(self) -> None:
        """把 prm/prm_en.txt 和当前视频的合并文本发给 AI 要高光 JSON。

        两条路，由 bridge.mode 决定：
        - api（默认）：Python 直接调 AI 接口（Gemini 或 DeepSeek）。纯后台，不开浏览器。
        - extension：老路子，扩展去驱动网页版对话页。要开着浏览器且窗口不能被冻结。
        合并文本写进 AI_输入目录（<视频名>.txt，留着）；那栏留空就落 cache/，任务结束删掉。
        """

        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开一个视频")
            return
        if not self.speech and not self.timeline:
            QMessageBox.information(self, "提示", "还没有分析结果，先跑一次分析")
            return
        prompt_path = self.resolve_prompt_file()
        if prompt_path is None:
            QMessageBox.warning(self, "AI 对接",
                                "找不到高光筛选提示词。放一份 prm_en.txt 到 prm/ 或项目根目录")
            return
        merged_path, count = self.write_ai_text()
        self.dispatch_ai(prompt_path, merged_path, count)

    def write_ai_text(self) -> tuple[Path, int]:
        """把当前结果写成一份给 AI 看的合并 txt，返回（路径，时间线条数）。

        设了 AI_输入目录就写 <视频名>.txt 留在那儿（下次「自动剪辑」看见它就不再分析）；
        没设就落 cache/ 的 <视频名>_merged.txt，任务结束删掉。
        """
        ai_in = self.ai_dir("ai_input_dir")
        target_dir = ai_in or self.cfg.path("cache_dir")
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = self.video_path.stem if self.video_path is not None else "video"
        merged_path = target_dir / (f"{stem}.txt" if ai_in else f"{stem}_merged.txt")
        count = write_merged_txt(
            merged_path, self.video_path.name, self.speech, self._events_for_export(),
            self.show_translated, self.export_language(),
            actions=self.timeline_doc.get("action_track"),
            emotions=self.timeline_doc.get("expression_track"),
            duration=float(self.timeline_doc.get("duration") or 0.0))
        # 自己指定了 AI_输入目录就当归档留着，别偷偷删用户自己的目录
        self._bridge_temp_files = [] if ai_in else [merged_path]
        if ai_in:
            # 落在 cache 的那份任务结束就删了，不进库；归档的这份才登记
            self._register_artifact(self.video_path, "merged_txt", merged_path)
            self.append_log(f"[AI 对接] 合并 txt 写到 AI_输入目录：{merged_path}")
        return merged_path, count

    def send_file_to_ai(self, text_path: Path) -> bool:
        """把一份现成的 txt 直接发给 AI，不重新导出（「有同名 .txt 就不再分析」那条路）。"""
        prompt_path = self.resolve_prompt_file()
        if prompt_path is None:
            self.append_log("[AI 对接] 找不到高光筛选提示词（prm_en.txt），发不出去")
            return False
        self._bridge_temp_files = []  # 是用户自己的文件，别删
        self.append_log(f"[AI 对接] 用现成的 {text_path.name}，不再重新导出")
        self.dispatch_ai(prompt_path, text_path, 0)
        return True

    def _note_prompt_use(self, prompt_path: Path) -> None:
        """记下这一次真正要发出去的提示词是哪一版（只写库，不动上传内容）。

        必须在真要发的这一刻算：一批任务可能排很久，中间 prm_en.txt 被改过，
        入队时算的指纹就不是实际发出去的那一版了。
        记录失败绝不能拦住发送——AI 剪辑照跑，只是这条少一份审计信息。
        """
        self._last_prompt = None
        try:
            info = db_repo.prompt_fingerprint(prompt_path)
        except OSError as exc:
            self.append_log(f"[AI 对接] 提示词指纹算不出来（不影响发送）：{exc}")
            return
        self._last_prompt = info
        db = self._db()
        if db is None or self._auto_task_id is None:
            return
        try:
            db_repo.note_task_prompt(db, self._auto_task_id, prompt_path)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] 任务 #{self._auto_task_id} 的提示词指纹写不进去：{exc}")

    def dispatch_ai(self, prompt_path: Path, merged_path: Path, count: int) -> None:
        """两个附件都齐了，按 bridge.mode 决定走接口还是走扩展。"""
        self._note_prompt_use(prompt_path)   # 只记账：发什么、怎么发都没变
        cfg = self.cfg.bridge
        if str(cfg.get("mode") or "api") == "api":
            self.send_via_api(prompt_path, merged_path, count)
            return

        if self.bridge is None:
            QMessageBox.warning(self, "AI 对接", "Bridge 没有启动")
            return
        from ..bridge import providers  # noqa: PLC0415

        # 网页版开哪家由 bridge.provider 定：gemini -> gemini.google.com，
        # deepseek -> chat.deepseek.com；扩展也按网址挑对应的页面选择器
        spec = providers.settings(cfg)
        task_id = self.bridge.submit(

            str(cfg.get("task_type") or "gemini_json"),
            {"url": spec["ai_url"],
             "provider": spec["provider"],
             "video": self.video_path.name if self.video_path else merged_path.stem,

             "message": str(cfg.get("message") or ""),
             "upload_mode": str(cfg.get("upload_mode") or "manual"),
             "focus_browser": bool(cfg.get("focus_browser", False)),
             # 后台标签页会被浏览器冻结（不排版、不跑定时器），所以默认把网页版 AI
             # 挪进一个不抢焦点的小窗口，页面照常渲染又挡不着你干活
             "side_window": bool(cfg.get("side_window", True)),


             "expect": "json"},
            files=[prompt_path, merged_path])
        state = self.bridge.state()
        self.append_log(f"[AI 对接] 已入队 {task_id}（{spec['label']} 网页版）：上传 "
                        f"{prompt_path.name} + {merged_path.name}（时间线 {count} 条）"
                        + ("，等扩展领取" if state["extension_online"]
                           else "；扩展当前离线，先确认扩展已装好并配对"))
        mode = str(cfg.get("upload_mode") or "manual")
        if mode == "manual":
            self.append_log(f"[AI 对接] 半自动模式：{spec['label']} 打开后请自己把这两个文件"
                            "选进去，挂好之后扩展会自动发送并回传")
            self.append_log(f"[AI 对接] 文件 1：{prompt_path}")
            self.append_log(f"[AI 对接] 文件 2：{merged_path}")
        elif mode == "observe":
            self.append_log(f"[AI 对接] 观察模式：{spec['label']} 打开后扩展一个键都不点，"
                            "你自己走一遍（挂文件、发送），日志里会记下你碰过的元素；"
                            "看完点「停止 AI」结束")
            self.append_log(f"[AI 对接] 文件 1：{prompt_path}")
            self.append_log(f"[AI 对接] 文件 2：{merged_path}")
        self.refresh_bridge_label()

    def api_key(self) -> str:
        """当前提供方的 API key：先看配置里写死的，为空就读环境变量。

        环境变量优先给不想把 key 写进仓库的情况用（Gemini 是 GEMINI_API_KEY，
        DeepSeek 是 DEEPSEEK_API_KEY，都能在配置里改名字）。
        """
        from ..bridge import providers  # noqa: PLC0415

        return str(providers.settings(self.cfg.bridge)["api_key"])

    def send_via_api(self, prompt_path: Path, merged_path: Path, count: int) -> None:
        """直接调 AI 接口（Gemini 或 DeepSeek）。不开浏览器，一次请求拿回 JSON。"""
        if self.ai_worker is not None and self.ai_worker.isRunning():
            QMessageBox.information(self, "AI 接口", "上一次请求还没回来")
            return
        from ..bridge import providers  # noqa: PLC0415

        cfg = self.cfg.bridge
        spec = providers.settings(cfg)
        key = str(spec["api_key"])
        if not key:
            self.clean_bridge_temp()
            QMessageBox.warning(self, "AI 接口",
                                f"没有 {spec['label']} 的 API key。去 {spec['key_page']} "
                                f"领一个，填到 AI 选项里的 API key，"
                                f"或设环境变量 {spec['api_key_env']}")
            return
        model = str(spec["api_model"])
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            merged_text = merged_path.read_text(encoding="utf-8")
        except OSError as exc:
            self.clean_bridge_temp()
            QMessageBox.warning(self, "AI 接口", f"读文件失败：{exc}")
            return
        self.btn_bridge_send.setEnabled(False)
        self.lbl_stage.setText(f"AI 接口｜{model} 处理中")
        self.append_log(f"[AI 接口] {spec['label']} {model}：提示词 {len(prompt_text)} 字 + "
                        f"合并文本 {len(merged_text)} 字（时间线 {count} 条），等回答…")
        self.ai_worker = AiApiWorker(str(spec["provider"]), key, model, prompt_text,
                                     merged_text, str(cfg.get("message") or ""),
                                     float(spec["api_timeout"]), str(spec["base_url"]))
        self.ai_worker.log.connect(self.append_log)
        self.ai_worker.done.connect(self.on_api_done)
        self.ai_worker.start()

    def on_api_done(self, text: str, error: str) -> None:
        """接口回来了：走和扩展回传完全一样的后续（抠 JSON -> 自动剪辑）。"""
        self.btn_bridge_send.setEnabled(True)
        if error:
            self.clean_bridge_temp()
            self.lbl_stage.setText("失败")
            self.append_log(f"[AI 接口] 失败：{error}")
            if self.auto_running():
                self._auto_advance("failed", f"AI 接口失败：{error}")
                return
            QMessageBox.warning(self, "AI 接口", error)
            return

        from ..bridge import providers  # noqa: PLC0415

        self.append_log(f"[AI 接口] 收到 {len(text)} 字")
        self.on_bridge_result({"json": providers.extract_json(text), "text": text})

    def resolve_prompt_file(self) -> Path | None:
        """找高光筛选提示词。按 config 里的路径、AI_输入目录、prm/、项目根、包内副本依次找。

        这份文件被挪过好几次位置，找不到就返回 None，由调用方提示，别让任务默默少一个附件。
        """
        candidates = []
        configured = str(self.cfg.bridge.get("prompt_file") or "").strip()
        if configured:
            path = Path(configured)
            candidates.append(path if path.is_absolute() else self.cfg.root / path)
        ai_in = str(self.cfg.bridge.get("ai_input_dir") or "").strip()
        if ai_in:
            candidates.append(Path(ai_in) / "prm_en.txt")
        candidates += [self.cfg.root / "prm" / "prm_en.txt",

                       self.cfg.root / "prm_en.txt",
                       Path(__file__).resolve().parents[1] / "prm_en.txt"]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def on_bridge_stop(self) -> None:
        if self.auto_running():
            self._auto_finish("已中止，剩下的不跑了", cancel=True)
        if self.bridge is None:
            return

        self.bridge.cancel()
        self.clean_bridge_temp()
        self.append_log("[AI 对接] 已请求取消当前任务")
        self.refresh_bridge_label()

    def clean_bridge_temp(self) -> None:
        """删掉临时的合并导出。配置里 keep_merged_file=true 就留着。"""
        if self.cfg.bridge.get("keep_merged_file"):
            self._bridge_temp_files = []
            return
        for path in getattr(self, "_bridge_temp_files", []):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self.append_log(f"[AI 对接] 临时文件删不掉：{path.name} {exc}")
        self._bridge_temp_files = []

    # -------------------------------------------------------- 自动剪辑（批量）
    VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".m4v",
                      ".ts", ".mpg", ".mpeg", ".wmv")

    def auto_busy(self) -> str:
        """有活在跑就返回一句人话，闲着返回空串。跟 busy() 同样的判断，只是不弹窗。"""
        if self.worker is not None and self.worker.isRunning():
            return "分析还在跑"
        if self.clip_worker is not None and self.clip_worker.isRunning():
            return "高光剪辑还在渲染"
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return "上一次 AI 请求还没回来"
        return ""

    def on_auto_clip(self) -> None:
        """扫 AI_输入目录里的视频，挨个跑「AI 面板 - 自动剪辑干什么」选的那一串。

        剪辑成片 / 收取脚本：有同名 .txt 就不再分析，直接把它发给 AI；没有 .txt 但缓存里
        有上次的分析结果，就照缓存导出 <视频名>.txt，也不重跑分析；两样都没有才分析。
        回来的 JSON 按主界面高光配置剪，成品落 AI_输出目录。
        脚本剪辑：跳过 AI，直接用现成的脚本 JSON 开剪。

        """
        if self.auto_running():
            QMessageBox.information(self, "自动剪辑", "已经在跑了，要停就点「停止_AI」")
            return
        reason = self.auto_busy()
        if reason:
            QMessageBox.information(self, "自动剪辑", f"{reason}，等它跑完再点")
            return
        ai_in = self.ai_dir("ai_input_dir")
        if ai_in is None:
            QMessageBox.information(self, "自动剪辑", "先在 AI 面板里设好 AI_输入目录")
            return
        videos = sorted(p for p in ai_in.iterdir()
                        if p.is_file() and p.suffix.lower() in self.VIDEO_SUFFIXES)
        if not videos:
            QMessageBox.information(self, "自动剪辑", f"{ai_in} 里没有视频")
            return
        db = self._db()
        if db is None:
            QMessageBox.warning(self, "自动剪辑",
                                "数据库打不开，自动剪辑的队列就在数据库里，先解决数据库再跑")
            return
        job = str(self.cfg.bridge.get("ai_job") or "full")
        labels = {"full": "剪辑成片", "collect": "收取脚本", "script": "脚本剪辑"}
        self._auto_job = job
        self._set_auto_state(False)
        self._set_auto_step("", "")
        self._set_auto_progress(0)
        self._sync_disk()  # 手动丢进目录的 TXT/JSON/成品先进库，后面每一步只查库
        created, reused, already = self._enqueue_auto_tasks(videos, job)
        counts = db_repo.queue_counts(db, mode=job)
        if not counts["open"]:
            self._set_auto_state(True, "闲着")
            self.append_log(f"[自动剪辑] {labels.get(job, job)}：{len(videos)} 个视频都已经有成品，没活可干")
            return
        self._auto_active = True
        self._auto_done = 0
        self._auto_total = counts["open"]
        self.append_log(
            f"[自动剪辑] {labels.get(job, job)}：队列里 {counts['open']} 条待办"
            f"（新建 {created}，接上原有 {reused}"
            + (f"，{already} 个已有成品不排队" if already else "") + "）")
        self._auto_step()

    def _worker_id(self) -> str:
        """谁在跑这条任务。以后多开一个进程也能看出是谁占的。"""
        return f"gui-{os.getpid()}"

    def _enqueue_auto_tasks(self, videos: list[Path], job: str) -> tuple[int, int, int]:
        """把这一批视频排进数据库队列，返回（新建、复用、已有成品跳过）。

        幂等：同一个视频 + 同一种模式已经有没跑完的任务就复用那条，
        所以连点五次「自动剪辑」也不会多出四条重复任务。
        """
        db = self._db()
        if db is None:
            return 0, 0, 0
        created = reused = already = 0
        for video in videos:
            if self._auto_done_file(video) is not None:
                already += 1
                continue
            vid = self._db_video_id(video, create=True)
            if vid is None:
                self.append_log(f"[自动剪辑] {video.name} 登记不进数据库，跳过")
                continue
            try:
                _task_id, is_new = db_repo.enqueue_ai_task(
                    db, vid, mode=job,
                    provider=str(self.cfg.bridge.get("provider") or "") or None,
                    model=str(self.cfg.bridge.get("api_model") or "") or None,
                    prompt_version=str(self.cfg.bridge.get("task_type") or "") or None,
                    max_attempts=1)
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"[自动剪辑] {video.name} 入队失败：{exc}")
                continue
            if is_new:
                created += 1
            else:
                reused += 1
        return created, reused, already

    def _auto_step(self) -> None:
        """从数据库队列里领下一条任务。后面几步靠各自的完成回调推进，这儿只负责起头。"""
        db = self._db()
        if db is None:
            self._auto_finish("数据库不可用，队列停下")
            return
        while True:
            task = db_repo.claim_next_ai_task(db, mode=self._auto_job,
                                              worker_id=self._worker_id())
            if task is None:
                self._auto_finish("全部跑完")
                return
            video = Path(str(task["file_path"]))
            self._auto_task_id = int(task["id"])
            self._auto_video = video
            self._auto_done += 1
            total = max(self._auto_total, self._auto_done)
            self.append_log(f"[自动剪辑] ({self._auto_done}/{total}) 任务 #{task['id']} {video.name}")
            self._set_auto_state(False, f"跑着 {self._auto_done}/{total}")
            if not video.is_file():
                self.append_log(f"[自动剪辑] {video.name} 已经不在盘上，这条记 failed")
                self._settle_auto_task("failed", "视频不在盘上了")
                self._auto_video = None
                continue
            done = self._auto_done_file(video)
            if done is not None:
                self.append_log(f"[自动剪辑] AI_输出目录里已经有 {done.name}，这个跳过")
                self._settle_auto_task("completed")
                self._auto_video = None
                continue
            break
        self.load_video(video)

        if self._auto_job == "script":
            self._set_auto_step(video.stem, "剪辑")
            self._auto_clip_from_script(video)
            return
        text_file = self._auto_text_file(video)
        if text_file is not None:
            self.append_log(f"[自动剪辑] 已有 {text_file.name}，不再分析，直接发 AI")
            self._set_auto_step(video.stem, "发送")
            if not self.send_file_to_ai(text_file):
                self._auto_advance("failed", "发不出去（缺提示词或读文件失败）")
            return
        # 没有 txt，但缓存里有上次分析的结果（load_video 刚读过 output/<视频名>/），
        # 那就直接照缓存导出合并 txt，不用再跑一遍分析
        if self.speech or self.timeline:
            self.append_log(f"[自动剪辑] 缓存里有 {video.stem} 的分析结果，直接导出 TXT，不重跑分析")
            self._set_auto_step(video.stem, "导出")
            self._auto_after_analyze()
            return
        self.append_log(f"[自动剪辑] 没有 {video.stem}.txt 也没缓存，先按主界面配置分析")
        self._set_auto_step(video.stem, "分析")
        self.on_analyze(False)

    def _settle_auto_task(self, outcome: str, error: str | None = None) -> str:
        """给当前这条任务落状态。返回数据库里最终的状态（可能被退回 pending 重试）。"""
        db = self._db()
        task_id = self._auto_task_id
        self._auto_task_id = None
        if db is None or task_id is None:
            return outcome
        try:
            if outcome == "completed":
                db_repo.complete_ai_task(db, task_id)
                return "completed"
            if outcome == "cancelled":
                db_repo.cancel_ai_task(db, task_id, error)
                return "cancelled"
            final = db_repo.fail_or_requeue_ai_task(db, task_id, error or "没说原因")
            if final == "pending":
                self.append_log(f"[自动剪辑] 任务 #{task_id} 失败，退回队列等下一轮")
            else:
                self.append_log(f"[自动剪辑] 任务 #{task_id} 记为 failed：{error or ''}")
            return final
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] 任务 #{task_id} 状态写不进去：{exc}")
            return outcome

    def _touch_auto_task(self) -> None:
        """给当前这条任务刷心跳。没有在跑的任务就什么都不做。"""
        task_id = self._auto_task_id
        if task_id is None:
            return
        db = self._db()
        if db is None:
            return
        try:
            db_repo.touch_ai_task(db, task_id)
        except Exception:  # noqa: BLE001, S110
            # 刷心跳失败不该打断正在跑的活，也别每分钟往日志面板刷一行
            pass

    def _resume_auto_queue(self) -> None:
        """开程序时把上次没跑完的任务捞回来接着跑（强关、崩溃都算）。

        processing/uploading/waiting 且心跳超时的退回 pending，然后照常一条条领。
        """
        db = self._db()
        if db is None:
            return
        timeout = float(self.cfg.runtime.get("ai_task_timeout_minutes", 30) or 30)
        try:
            recovered = db_repo.recover_stale_ai_tasks(db, timeout)
            job = str(self.cfg.bridge.get("ai_job") or "full")
            counts = db_repo.queue_counts(db, mode=job)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[自动剪辑] 队列恢复失败：{exc}")
            return
        if recovered:
            self.append_log(f"[自动剪辑] 上次有 {recovered} 条任务没跑完，已退回等待")
        if not counts["pending"]:
            if counts["active"]:
                self.append_log(f"[自动剪辑] 还有 {counts['active']} 条卡在跑着但没超时，先不动它们")
            return
        if not self.cfg.runtime.get("auto_resume_queue", True):
            self.append_log(f"[自动剪辑] 队列里还有 {counts['pending']} 条待办"
                            f"（auto_resume_queue 关着，点「自动剪辑」再继续）")
            return
        if self.auto_running() or self.auto_busy():
            return
        self._auto_job = job
        self._auto_active = True
        self._auto_done = 0
        self._auto_total = counts["pending"]
        self._set_auto_state(False)
        self._set_auto_progress(0)
        self.append_log(f"[自动剪辑] 接着上次跑：队列里还有 {counts['pending']} 条")
        self._auto_step()




    def _auto_clip_from_script(self, video: Path) -> None:
        """脚本剪辑：读 AI_输入目录里现成的脚本 JSON，直接开剪。"""
        script = self._auto_script_file(video)
        if script is None:
            self.append_log(f"[自动剪辑] {video.stem} 旁边没有脚本 JSON，跳过")
            self._auto_advance("failed", "没有脚本 JSON")
            return
        try:
            text = script.read_text(encoding="utf-8")
        except OSError as exc:
            self.append_log(f"[自动剪辑] 脚本读不了：{exc}，跳过")
            self._auto_advance("failed", f"脚本读不了：{exc}")
            return
        self.append_log(f"[自动剪辑] 按现成脚本剪：{script.name}")
        self._last_highlight_json = text
        self.run_highlight(text, ai=True)

    def _db(self):
        """数据库句柄。打不开就记一句日志，之后所有状态查询都当「没有」。"""
        if self._db_handle is None and not self._db_failed:
            try:
                self._db_handle = open_db(self.cfg)
            except Exception as exc:  # noqa: BLE001 - 界面不能因为库打不开就崩
                self._db_failed = True
                self.append_log(f"[数据库] 打不开，状态判断退回空：{exc}")
        return self._db_handle

    def _db_video_id(self, video: Path, *, create: bool = False) -> int | None:
        """视频在库里的 id。create=True 时没有就登记一条（产出文件时才需要）。"""
        db = self._db()
        if db is None:
            return None
        row = db_repo.find_video(db, video)
        if row is not None:
            return int(row["id"])
        if not create:
            return None
        try:
            return db_repo.upsert_video(db, video)
        except Exception as exc:  # noqa: BLE001 - 视频读不了也不该让界面挂掉
            self.append_log(f"[数据库] 登记视频失败：{exc}")
            return None

    def _register_artifact(self, video: Path | None, kind: str, path: Path) -> None:
        """刚生成的文件立刻进库，免得下一步还得靠扫目录才看得见。"""
        if video is None:
            return
        db = self._db()
        if db is None:
            return
        vid = self._db_video_id(video, create=True)
        if vid is None:
            return
        try:
            db_repo.register_artifact(db, vid, kind, path)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] 登记 {kind} 失败：{exc}")

    def _sync_disk(self) -> None:
        """开跑前跟磁盘对一次账：手动丢进目录的视频/TXT/JSON/成品都登记进库。

        磁盘扫描只在这儿（以及 AI 面板刷新）发生，后面每一步的状态判断都只查库。
        """
        db = self._db()
        if db is None:
            return
        folders = [p for p in (self.ai_dir("ai_input_dir"),) if p is not None]
        try:
            refresh_from_disk(self.cfg, db, folders=folders,
                              ai_out=self.ai_dir("ai_output_dir") or self.export_root())
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] 对账失败，状态可能不准：{exc}")

    def _auto_product_ready(self) -> Path | None:
        """数据库里确认这条任务的成品真的登记上了，才允许把任务算 completed。

        剪辑成片 / 脚本剪辑看 final_video，收取脚本看落在 AI_输出目录的 ai_script。
        渲染线程说"成了"不算数——文件没落地、或者没登记进 artifacts，就是没完成。
        """
        if self._auto_video is None:
            return None
        return self._auto_done_file(self._auto_video)

    def _register_final_video(self, output: str) -> None:
        """成品刚出炉：登记 final_video，并把这个视频的 clip 标成已渲染。"""
        video = self._auto_video or self.video_path
        if video is None or not output:
            return
        target = Path(output)
        if not target.is_file():
            return
        self._register_artifact(video, "final_video", target)
        db = self._db()
        vid = self._db_video_id(video)
        if db is None or vid is None:
            return
        try:
            for clip in db_repo.get_clips(db, vid):
                if clip["status"] != "rendered":
                    db_repo.update_clip(db, int(clip["id"]), status="rendered",
                                        output_path=target)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] 标记片段已渲染失败：{exc}")

    def _save_ai_result(self, parsed: dict, raw_text: str = "") -> None:
        """AI 回的 JSON 进库，挂在当前这条任务下面（ai_results.task_id 指回 ai_tasks.id）。

        手工单发（没有队列任务）时 task_id 是空的，结果照样留档，不会挂到别人身上；
        提示词指纹这三列两条路都写，所以手工发的结果也能回答用的是哪一版提示词。
        """
        video = self._auto_video or self.video_path
        if video is None:
            return
        db = self._db()
        if db is None:
            return
        vid = self._db_video_id(video, create=True)
        if vid is None:
            return
        prompt = self._last_prompt or {}
        try:
            clips = db_repo.clips_from_payload(parsed)
            result_id = db_repo.save_ai_result(
                db, vid, task_id=self._auto_task_id, raw_response=raw_text or None,
                json_data=parsed, candidate_count=len(clips) or None, validated=True,
                prompt_hash=prompt.get("prompt_hash"), prompt_path=prompt.get("prompt_path"),
                prompt_size=prompt.get("prompt_size"))
            for spec in clips:
                db_repo.create_clip(db, vid, spec, ai_result_id=result_id)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"[数据库] AI 结果存不进去：{exc}")

    def _auto_text_file(self, video: Path) -> Path | None:
        """给 AI 看的合并文本。查 artifacts.merged_txt，路径由库里给。"""
        db = self._db()
        vid = self._db_video_id(video)
        if db is None or vid is None:
            return None
        return db_repo.artifact_path(db, vid, "merged_txt")

    def _auto_script_file(self, video: Path) -> Path | None:
        """脚本 JSON。查 artifacts.ai_script（视频旁边和 AI_输出目录都登记在这一类）。"""
        db = self._db()
        vid = self._db_video_id(video)
        if db is None or vid is None:
            return None
        return db_repo.artifact_path(db, vid, "ai_script")

    def _auto_done_file(self, video: Path) -> Path | None:
        """这个视频算干完了没有——干完了就返回成品路径，用来跳过。

        收取脚本看 ai_script（脚本已经存下来了），剪辑成片 / 脚本剪辑看 final_video。
        判断全部来自数据库，不再 is_file() / st_size；真实路径存在 artifacts 里。
        """
        db = self._db()
        vid = self._db_video_id(video)
        if db is None or vid is None:
            return None
        if self._auto_job == "collect":
            out = self.ai_dir("ai_output_dir") or self.export_root()
            hit = db_repo.artifact_path(db, vid, "ai_script")
            # 收取脚本只认存到 AI_输出目录的那份，视频旁边自带的脚本不算干完
            if hit is not None and hit.parent == out:
                return hit
            return None
        return db_repo.artifact_path(db, vid, "final_video")



    def _auto_after_analyze(self) -> None:
        """分析跑完了（自动剪辑那一串里）：生成 <视频名>.txt，接着发给 AI。"""
        if self._auto_video is None:
            return
        if not self.speech and not self.timeline:
            self.append_log("[自动剪辑] 分析完了却没读到结果，跳过这个")
            self._auto_advance("failed", "分析完了没读到结果")
            return
        prompt_path = self.resolve_prompt_file()
        if prompt_path is None:
            self.append_log("[自动剪辑] 找不到 prm_en.txt，整串停下")
            self._auto_finish("缺提示词，已停", cancel=True)
            return
        merged_path, count = self.write_ai_text()
        self._set_auto_step(self._auto_video.stem, "发送")
        self.dispatch_ai(prompt_path, merged_path, count)

    def _auto_save_script(self) -> bool:

        """把 AI 回的 JSON 存进 AI_输出目录，当脚本留档（任务表的 JSON 列就看它）。

        存下来并登记进库才返回 True——「收取脚本」这一串就靠它判断算不算干完。
        """
        out = self.ai_dir("ai_output_dir") or self.export_root()
        stem = self._auto_video.stem if self._auto_video is not None else "script"
        target = out / f"{stem}_脚本.json"
        try:
            target.write_text(self._last_highlight_json, encoding="utf-8")
        except OSError as exc:
            self.append_log(f"[自动剪辑] 脚本存不下来：{exc}")
            return False
        self.append_log(f"[自动剪辑] 脚本已存：{target}")
        self._register_artifact(self._auto_video, "ai_script", target)
        return True

    # ---------------------------------------------------- 面板上的状态回显
    def _set_auto_step(self, stem: str, step: str) -> None:
        """告诉 AI 面板现在在处理谁、走到哪一步（分析 / 导出 / 发送 / 剪辑）。"""
        if self.ai_panel is not None:
            self.ai_panel.set_active(stem, step)

    def _set_auto_progress(self, done: int) -> None:
        if self.ai_panel is not None:
            self.ai_panel.set_queue_progress(done, max(self._auto_total, done))


    def _auto_advance(self, outcome: str = "completed", error: str | None = None) -> None:
        """当前这条落个状态，然后排队叫下一条。

        outcome：completed（这个视频这一串走完了）/ failed（哪一步砸了，error 写进
        last_error）/ cancelled（人工停的）。失败的任务由数据库按 max_attempts 决定
        是退回 pending 还是定格 failed，这里不自己决定重试。
        """
        if self._auto_video is None:
            return
        self._settle_auto_task(outcome, error)
        self._auto_video = None
        self._set_auto_step("", "")
        self._set_auto_progress(self._auto_done)
        QTimer.singleShot(0, self._auto_step)  # 让当前回调先返回，别在信号里套信号

    def _auto_finish(self, why: str, *, cancel: bool = False) -> None:
        """整批收工。cancel=True（人工停 / 缺提示词停）时把没跑完的标 cancelled。"""
        if cancel:
            self._settle_auto_task("cancelled", why)
            db = self._db()
            if db is not None:
                try:
                    left = db_repo.cancel_open_ai_tasks(db, mode=self._auto_job)
                except Exception as exc:  # noqa: BLE001
                    self.append_log(f"[数据库] 取消剩余任务失败：{exc}")
                else:
                    if left:
                        self.append_log(f"[自动剪辑] 还没跑的 {left} 条标成已取消")
        self._auto_task_id = None
        self._auto_active = False
        self._auto_video = None
        self._auto_total = 0
        self._auto_done = 0
        self._set_auto_state(True, "闲着")
        self._set_auto_step("", "")
        self.append_log(f"[自动剪辑] {why}")


    def _set_auto_state(self, idle: bool, state: str = "") -> None:
        """AI 面板上的「自动剪辑」按钮和状态字。面板没开着就什么都不用做。"""
        if self.ai_panel is not None:
            self.ai_panel.set_running(not idle, state)


    def auto_running(self) -> bool:
        """这一批还在跑没有。当前有领到的任务，或者这一批还没收工，都算在跑。"""
        return self._auto_video is not None or self._auto_active


    def on_bridge_event(self, kind: str, data: object) -> None:
        """Bridge 的 HTTP 线程事件（已经过 BridgeEvents 搬到 GUI 线程）。"""

        info = data if isinstance(data, dict) else {}
        if kind == "paired":
            self.append_log("[AI 对接] 扩展已配对，令牌已领取")
        elif kind == "claimed":
            self.append_log(f"[AI 对接] 扩展领走任务 {info.get('task_id', '')}")
        elif kind == "progress":
            self.append_log(f"[AI 对接] {info.get('stage', '')} {info.get('message', '')}".rstrip())
        elif kind == "result":
            self.on_bridge_result(info)
        self.refresh_bridge_label()

    def on_bridge_result(self, info: dict) -> None:
        """AI 回来了：解析出 JSON 就直接按它剪，解析不出只记日志不猜。"""
        self.clean_bridge_temp()
        parsed = info.get("json")
        if not isinstance(parsed, dict):
            reason = info.get("error") or "回答里没有可解析的 JSON"
            self.append_log(f"[AI 对接] 任务失败：{reason}")
            text = str(info.get("text") or "")
            if text:
                self.append_log(f"[AI 对接] AI 原文前 200 字：{text[:200]}")
            if self.auto_running():  # 批量里不弹窗拦着，记一笔接着下一个
                self._auto_advance("failed", f"没拿到可用 JSON：{reason}")
                return
            QMessageBox.warning(self, "AI 对接", f"没拿到可用 JSON：{reason}")
            return
        self._last_highlight_json = json.dumps(parsed, ensure_ascii=False, indent=2)
        clip = parsed.get("clip") if isinstance(parsed.get("clip"), dict) else parsed
        self.append_log(f"[AI 对接] 收到 JSON：clip.start={clip.get('start')} "
                        f"clip.end={clip.get('end')}")
        self._save_ai_result(parsed, str(info.get("text") or ""))
        idle = ((self.clip_worker is None or not self.clip_worker.isRunning())
                and (self.worker is None or not self.worker.isRunning()))
        if self.auto_running():
            saved = self._auto_save_script()  # 不管哪一串都留档，任务表的 JSON 列就看这个
            if self._auto_job == "collect":  # 收取脚本：只存不剪，存下来才算干完
                if saved and self._auto_product_ready() is not None:
                    self._auto_advance("completed")
                else:
                    self._auto_advance("failed", "脚本没存进 AI_输出目录")
                return
            self.append_log("[自动剪辑] 拿到 JSON，按主界面高光配置开剪")
            if self._auto_video is not None:
                self._set_auto_step(self._auto_video.stem, "剪辑")
            self.run_highlight(self._last_highlight_json, ai=True)
            return

        if self.cfg.bridge.get("auto_clip", True) and idle:

            self.append_log("[AI 对接] 直接按这份 JSON 开始剪辑（bridge.auto_clip）")
            self.run_highlight(self._last_highlight_json, ai=True)

            return
        self.statusBar().showMessage("AI 结果已收到，剪辑高光对话框已带上 JSON", 10000)
        if idle:
            self.on_highlight()


    def on_highlight_done(self, ok: bool, message: str) -> None:
        self.btn_highlight.setEnabled(True)
        if ok:
            self.set_progress(1.0, "100%")
            self.lbl_stage.setText("完成")

            self.statusBar().showMessage(f"高光片段已生成：{message}", 10000)
            self.append_log(f"[剪辑高光] 已生成 {message}")
            self._register_final_video(message)
        else:
            self.lbl_stage.setText("失败")
            self.append_log(f"[剪辑高光] 失败：{message}")
            if not self.auto_running():
                QMessageBox.warning(self, "剪辑高光失败", message)
        if self.auto_running():
            if self.ai_panel is not None:
                self.ai_panel.refresh_tasks()
            # 剪辑砸了就是砸了；就算渲染线程说成了，成品没登记进库也不算完成
            if not ok:
                self._auto_advance("failed", f"剪辑失败：{message}")
                return
            if self._auto_product_ready() is None:
                self._auto_advance("failed", f"渲染说成了但成品没落地/没登记：{message}")
                return
            self._auto_advance("completed")




    def start_worker(self, argv: list[str], label: str) -> None:
        self.btn_analyze.setEnabled(False)
        self.btn_reanalyze.setEnabled(False)
        self.set_progress(0.0)
        self.lbl_stage.setText(f"{label}｜启动子进程")

        self.worker = AnalyzeWorker(self.cfg.root, argv, label)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_worker_done)
        self.worker.start()

    def _emotion_argv(self) -> list[str]:
        """两个情绪开关 + 声纹模型透传给 run：显式给参数，不让子进程去猜 config.json。"""
        argv = [
            "--audio-emotion" if self.chk_emotion_audio.isChecked() else "--no-audio-emotion",
            "--visual-emotion" if self.chk_emotion_visual.isChecked() else "--no-visual-emotion",
        ]
        speaker = self.cmb_speaker.currentData()
        if speaker:
            argv += ["--speaker-model", str(speaker)]
        return argv

    def on_analyze(self, force: bool) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开一个视频")
            return
        if self.busy():
            return
        model_id = self.cmb_model.currentData()
        argv = ["run", str(self.video_path)]
        if force:
            argv.append("--force")
        if model_id:
            argv += ["--visual-model", model_id]
        argv += self._emotion_argv()
        auto = self.chk_auto_translate.isChecked()
        if auto:
            argv.append("--translate")
        self.append_log(f"开始分析 {self.video_path.name}（force={force}，视觉模型={model_id}，"
                        f"音频情绪={'开' if self.chk_emotion_audio.isChecked() else '关'}，"
                        f"画面情绪={'开' if self.chk_emotion_visual.isChecked() else '关'}，"
                        f"声纹={self.cmb_speaker.currentText()}，"
                        f"顺手翻译={'是' if auto else '否'}）…")
        self.start_worker(argv, "分析")

    def source_language(self) -> str | None:
        """界面上这批文本的语言：优先用 whisper 判定的音频语言。"""
        for key in ("language", "original_language"):
            value = self.speech_doc.get(key) or self.timeline_doc.get(key)
            if value:
                return str(value)
        return None

    def export_language(self) -> str:
        """导出文本该用哪种语言的表头/标签：跟当前显示的内容一致。

        看译文时用译文语言，看原文时用输出语言（= 原始音频语言），
        这样整份文件不会出现"中文表头 + 英文正文"这种混排。
        """
        if self.show_translated:
            for doc in (self.speech_doc, self.timeline_doc):
                meta = doc.get("translation") or {}
                if meta.get("target_language"):
                    return str(meta["target_language"])
            for seg in self.speech:
                if seg.get("translated_language"):
                    return str(seg["translated_language"])
        value = self.timeline_doc.get("output_language") or self.source_language()
        return str(value or "zh")


    def pending_translations(self) -> list[dict[str, str]]:
        """界面上"还没有译文"的行。原文被编辑过的也算，旧译文视为失效。"""
        items: list[dict[str, str]] = []
        for i, seg in enumerate(self.speech):
            text = str(seg.get("text") or "")
            if needs_translation(text, seg.get("text_translated"), seg.get("text_translated_from")):
                items.append({"key": f"s{i}", "text": text})
        for i, entry in enumerate(self.timeline):
            if not entry.get("visual"):
                continue
            text = str(entry.get("visual") or "")
            if needs_translation(text, entry.get("visual_translated"),
                                 entry.get("visual_translated_from")):
                items.append({"key": f"v{i}", "text": text})
        return items

    def on_translate(self) -> None:
        if self.show_translated:  # 回译：切回原文，不跑模型
            self.set_translated_view(False)
            self.append_log("已切回原文")
            return

        pending = self.pending_translations()
        if not pending:
            if self.has_translation():  # 全都译过了，直接切换显示
                self.set_translated_view(True)
                self.append_log("已切到译文（所有条目都有译文，没有重新翻译）")
            else:
                QMessageBox.information(self, "提示", "界面上没有可翻译的文本")
            return
        if self.busy():
            return

        out = self.output_dir()
        work = self.cache_dir_for_video()
        request_file = work / "translate_request.json"
        result_file = work / "translate_result.json"
        self._translate_request = {row["key"]: row["text"] for row in pending}
        self._translate_result = result_file
        write_json(request_file, {"source": self.source_language(), "items": pending,
                                  "output_dir": str(out) if out else None})
        argv = ["translate", "--items", str(request_file), "--result", str(result_file)]
        model_id = self.cmb_model.currentData()
        if model_id:
            argv += ["--visual-model", model_id]
        self.append_log(f"翻译界面上还没有译文的 {len(pending)} 行（纯文本，不重新分析视频）…")
        self.start_worker(argv, "翻译")

    def set_translated_view(self, translated: bool) -> None:
        self.show_translated = bool(translated)
        self.refresh_timeline_table()
        self.refresh_speech_list()

    def apply_translation_result(self) -> bool:
        """把子进程的翻译结果贴回界面当前的数据，然后落盘。"""
        result_file = getattr(self, "_translate_result", None)
        if result_file is None or not Path(result_file).is_file():
            self.append_log("[翻译] 找不到结果文件")
            return False
        try:
            with open(result_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.append_log(f"[翻译] 结果文件读不出来：{exc}")
            return False

        got = data.get("translations") or {}
        target = data.get("target_language")
        asked = getattr(self, "_translate_request", {})
        speech_hit = event_hit = skipped = 0
        for key, text in got.items():
            if not text:
                continue
            try:
                index = int(key[1:])
            except ValueError:
                continue
            # 翻译期间用户可能删/改过行，原文对不上就跳过，不贴错位置
            if key.startswith("s") and index < len(self.speech):
                seg = self.speech[index]
                if asked.get(key, str(seg.get("text") or "")) != str(seg.get("text") or ""):
                    skipped += 1
                    continue
                seg["text_translated"] = text
                seg["text_translated_from"] = str(seg.get("text") or "")
                seg["translated_language"] = target
                speech_hit += 1
            elif key.startswith("v") and index < len(self.timeline):
                entry = self.timeline[index]
                if asked.get(key, str(entry.get("visual") or "")) != str(entry.get("visual") or ""):
                    skipped += 1
                    continue
                entry["visual_translated"] = text
                entry["visual_translated_from"] = str(entry.get("visual") or "")
                event_hit += 1
            else:
                skipped += 1

        # 时间轴里的语音条目是多段拼接的，用段 id 取译文重拼，不再翻译一次拼接串
        by_id = {int(s["id"]): s["text_translated"] for s in self.speech
                 if isinstance(s.get("id"), int) and s.get("text_translated")}
        for entry in self.timeline:
            ids = [i for i in (entry.get("speech_event_ids") or []) if isinstance(i, int)]
            parts = [by_id[i] for i in ids if i in by_id]
            if parts:
                entry["speech_translated"] = " ".join(parts)

        if speech_hit:
            self.save_speech()
        if event_hit or speech_hit:
            self.save_timeline()
        failed = len(data.get("failed") or [])
        self.append_log(f"[翻译] 语音 {speech_hit} 行、画面 {event_hit} 条写入译文"
                        + (f"，失败 {failed} 行" if failed else "")
                        + (f"，跳过 {skipped} 行（翻译期间内容被改过）" if skipped else ""))
        return bool(speech_hit or event_hit)

    def on_progress(self, payload: dict) -> None:
        overall = float(payload.get("overall") or 0.0)
        self.set_progress(overall)
        stage = payload.get("stage_label") or payload.get("stage") or ""
        detail = payload.get("detail") or ""
        self.lbl_stage.setText(f"{stage}｜{detail}" if detail else stage)


    def on_worker_done(self, ok: bool, message: str) -> None:
        for btn in (self.btn_analyze, self.btn_reanalyze):
            btn.setEnabled(True)
        label = self.worker.label if self.worker is not None else "任务"
        if ok:
            self.set_progress(1.0, "100%")
            self.lbl_stage.setText("完成")

            self.append_log(f"{label}完成（{message}）")
            if label == "翻译":
                # 翻译只补文本，不重读磁盘：界面上的手动编辑要保住
                if self.apply_translation_result():
                    self.set_translated_view(True)
            else:
                self.append_log("重新加载结果")
                self.load_results()
                if label == "分析":
                    self._drop_preview_audio()
                if self._auto_video is not None and label == "分析":
                    # 自动剪辑那一串：分析完就生成 <视频名>.txt 接着发 AI
                    self._auto_after_analyze()
                # 「自动」勾着就顺手把文本发给 AI，回来的 JSON 再按 auto_clip 开剪
                elif label == "分析" and self.chk_auto_ai.isChecked():
                    self.append_log("[自动] 分析完了，直接发给 AI")
                    self.on_bridge_send()
        else:
            self.lbl_stage.setText("失败")
            self.append_log(f"{label}失败：{message}")
            if self._auto_video is not None and label == "分析":
                self.append_log("[自动剪辑] 这个分析没成，跳过")
                self._auto_advance("failed", f"分析失败：{message}")
                return
            QMessageBox.warning(self, f"{label}失败", f"{message}\n详细日志见 logs/ 目录")


    # ------------------------------------------------------------------ 声音
    def _drop_preview_audio(self) -> None:
        """开了「分析完就删预览音轨」就把这个视频的 wav 删掉，cache 里只剩 json。"""
        if not self.cfg.runtime.get("drop_preview_audio", False) or self.video_path is None:
            return
        from .. import cache as cache_mod  # noqa: PLC0415

        result = cache_mod.drop_preview_audio(self.cfg, self.video_path)
        if result["removed"]:
            self.append_log(f"[缓存] 删掉预览音轨，腾出 {result['freed_text']}")

    def prepare_audio(self) -> None:
        if self.video_path is None:
            return
        if self.audio_worker is not None and self.audio_worker.isRunning():
            return
        from ..audio import wav_path  # noqa: PLC0415

        target = wav_path(self.cfg.path("cache_dir"), self.video_path)
        self.statusBar().showMessage("正在从视频里解出音轨…")
        self.audio_worker = AudioWorker(self.video_path, target)
        self.audio_worker.done.connect(self.on_audio_ready)
        self.audio_worker.start()

    def on_audio_ready(self, path: str, error: str) -> None:
        if not path:
            self.chk_sound.blockSignals(True)
            self.chk_sound.setChecked(False)
            self.chk_sound.blockSignals(False)
            self.chk_sound.setEnabled(False)
            self.append_log(f"[声音] 不可用：{error}")
            self.statusBar().showMessage(f"声音不可用：{error}")
            return
        if not self.player.set_audio_file(path):
            self.chk_sound.setChecked(False)
            self.append_log("[声音] Qt 无法加载这个音轨")
            return
        self.player.set_audio_enabled(self.chk_sound.isChecked())
        self.append_log(f"[声音] 音轨就绪：{path}")
        self.statusBar().showMessage("声音已就绪")

    def on_sound_toggled(self, checked: bool) -> None:
        if checked and not self.player.audio_available():
            self.prepare_audio()
            return
        self.player.set_audio_enabled(checked)

    def on_audio_failed(self, message: str) -> None:
        self.chk_sound.blockSignals(True)
        self.chk_sound.setChecked(False)
        self.chk_sound.blockSignals(False)
        self.append_log(f"[声音] 播放失败：{message}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.save_settings()  # 退出时把界面参数存下来，下次启动自动加载
        self.stop_bridge()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.audio_worker is not None and self.audio_worker.isRunning():
            self.audio_worker.wait(3000)
        if self.clip_worker is not None and self.clip_worker.isRunning():
            self.clip_worker.wait(5000)  # 渲染没有中断点，给它一点时间收尾
        self.player.close_video()
        super().closeEvent(event)

    def on_open_outdir(self) -> None:
        out = self.export_root()
        if not out.is_dir():
            QMessageBox.information(self, "提示", f"目录还不存在：{out}")
            return
        if os.name == "nt":
            os.startfile(str(out))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(out)])

    def on_timeline_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        self.seek(float(item.data(Qt.UserRole)))

    def on_speech_clicked(self, item: QListWidgetItem) -> None:
        self.seek(float(item.data(Qt.UserRole)))

    def seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        if not self.player.is_playing():
            self.player.play()

    def toggle_play(self) -> None:
        self.player.toggle()

    def on_position_changed(self, seconds: float) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(int(round(seconds * 1000)))
        self.lbl_time.setText(f"{fmt_time(seconds)} / {fmt_time(self.player.duration())}")

    def on_duration_changed(self, seconds: float) -> None:
        self.slider.setRange(0, max(int(round(seconds * 1000)), 0))

    def highlight_current(self) -> None:
        seconds = self.player.position()
        rows = getattr(self, "_rows", [])
        active_row = -1
        for i, entry in enumerate(rows):
            if entry["start"] - 0.01 <= seconds <= entry["end"] + 0.01:
                active_row = i
                break
        if active_row != self._playing_row:  # 换行才重画，别每 250ms 刷整张表
            self._paint_row(self._playing_row, playing=False)
            self._paint_row(active_row, playing=True)
            self._playing_row = active_row
            if active_row >= 0:
                if self.table.currentRow() != active_row and not self.table.hasFocus():
                    self.table.selectRow(active_row)
                self.table.scrollToItem(self.table.item(active_row, 0))
        active = -1
        for i, seg in enumerate(self.speech[:self.speech_list.count()]):
            if seg["start"] - 0.01 <= seconds <= seg["end"] + 0.01:
                active = i
                break
        if active == self._playing_speech:  # 每 250ms 跑一次，没换句就别重画
            return
        self._paint_speech(self._playing_speech, playing=False)
        self._paint_speech(active, playing=True)
        self._playing_speech = active
        if active >= 0:
            self.speech_list.scrollToItem(self.speech_list.item(active))

    def _paint_speech(self, row: int, playing: bool) -> None:
        """正在播放的那句标绿加粗，播过去就恢复原色。"""
        item = self.speech_list.item(row) if row >= 0 else None
        if item is None:
            return
        font = item.font()
        font.setBold(playing)
        item.setFont(font)
        item.setForeground(QBrush(PLAYING_COLOR if playing else NORMAL_TEXT_COLOR))

    def _paint_row(self, row: int, playing: bool) -> None:
        """正在播放的画面事件整行标绿加粗，播过去恢复原来的重要性配色。"""
        if row < 0 or row >= self.table.rowCount():
            return
        rows = getattr(self, "_rows", [])
        entry = rows[row] if row < len(rows) else {}
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item is None:
                continue
            font = item.font()
            font.setBold(playing)
            item.setFont(font)
            if playing:
                item.setForeground(QBrush(PLAYING_COLOR))
                continue
            color = None
            if col == 2 and entry.get("visual"):
                color = IMPORTANCE_COLOR.get(entry.get("importance", "normal"))
            item.setForeground(QBrush(color or NORMAL_TEXT_COLOR))

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        # AI 面板开着就把 AI 相关的行也贴过去，跑批量时不用来回切窗口
        if self.ai_panel is not None and text.lstrip().startswith(("[自动剪辑]", "[AI")):
            self.ai_panel.append_log(text)


    # ------------------------------------------------------- 右键菜单：语音
    def selected_speech(self) -> list[int]:
        return sorted(self.speech_list.row(i) for i in self.speech_list.selectedItems())

    def on_speech_menu(self, pos) -> None:
        menu = QMenu(self)
        act_all = menu.addAction("全选")
        act_copy = menu.addAction("复制")
        act_edit = menu.addAction("编辑")
        act_del = menu.addAction("删除")
        act_clear = menu.addAction("清空")
        menu.addSeparator()
        act_tr = menu.addAction("回译（切回原文）" if self.show_translated else "翻译")
        menu.addSeparator()
        act_txt = menu.addAction("导出（SRT 剪映可用 / txt）")
        act_words = menu.addAction("逐词导出（一个词一个时间戳）")
        chosen = menu.exec_(self.speech_list.mapToGlobal(pos))

        if chosen is None:
            return
        if chosen is act_all:
            self.speech_list.selectAll()
        elif chosen is act_copy:
            self.copy_lines([self.speech_list.item(i).text() for i in self.selected_speech()]
                            or [self.speech_list.item(i).text() for i in range(self.speech_list.count())])
        elif chosen is act_edit:
            self.edit_speech()
        elif chosen is act_del:
            self.delete_speech()
        elif chosen is act_clear:
            self.clear_speech()
        elif chosen is act_tr:
            self.on_translate()
        elif chosen is act_txt:
            self.export_text("speech")
        elif chosen is act_words:
            self.export_words()


    def edit_speech(self) -> None:
        rows = self.selected_speech()
        if len(rows) != 1:
            QMessageBox.information(self, "提示", "编辑请只选中一行")
            return
        seg = self.speech[rows[0]]
        field = "text_translated" if (self.show_translated and seg.get("text_translated")) else "text"
        current = str(seg.get(field) or "")
        text, ok = QInputDialog.getMultiLineText(self, "编辑语音文本", "内容：", current)
        if not ok:
            return
        seg[field] = text.strip()
        self.save_speech()
        self.refresh_speech_list()

    def delete_speech(self) -> None:
        rows = self.selected_speech()
        if not rows:
            return
        if QMessageBox.question(self, "删除", f"删除选中的 {len(rows)} 段语音？") != QMessageBox.Yes:
            return
        for row in reversed(rows):
            if 0 <= row < len(self.speech):
                self.speech.pop(row)
        self.save_speech()
        self.refresh_speech_list()

    def clear_speech(self) -> None:
        if not self.speech:
            return
        if QMessageBox.question(self, "清空", "清空全部语音段？") != QMessageBox.Yes:
            return
        self.speech = []
        self.save_speech()
        self.refresh_speech_list()

    # --------------------------------------------------- 右键菜单：事件时间轴
    def selected_entries(self) -> list[dict[str, Any]]:
        rows = getattr(self, "_rows", [])
        picked = sorted({i.row() for i in self.table.selectedIndexes()})
        return [rows[r] for r in picked if 0 <= r < len(rows)]

    def on_timeline_menu(self, pos) -> None:
        menu = QMenu(self)
        act_all = menu.addAction("全选")
        act_copy = menu.addAction("复制")
        act_edit = menu.addAction("编辑")
        act_del = menu.addAction("删除")
        act_clear = menu.addAction("清空")
        menu.addSeparator()
        act_tr = menu.addAction("回译（切回原文）" if self.show_translated else "翻译")
        menu.addSeparator()
        act_txt = menu.addAction("导出（SRT 剪映可用 / txt）")
        menu.addSeparator()
        self._add_size_actions(menu.addMenu("画面列宽 / 行高"))
        chosen = menu.exec_(self.table.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_all:
            self.table.selectAll()
        elif chosen is act_copy:
            entries = self.selected_entries() or getattr(self, "_rows", [])
            self.copy_lines([
                f"[{fmt_time(e['start'])} - {fmt_time(e['end'])}] "
                f"{self.visual_display(e) or e.get('speech') or ''}" for e in entries
            ])
        elif chosen is act_edit:
            self.edit_entry()
        elif chosen is act_del:
            self.delete_entries()
        elif chosen is act_clear:
            self.clear_entries()
        elif chosen is act_tr:
            self.on_translate()
        elif chosen is act_txt:
            self.export_text("events")

    def edit_entry(self) -> None:
        entries = self.selected_entries()
        if len(entries) != 1:
            QMessageBox.information(self, "提示", "编辑请只选中一行")
            return
        entry = entries[0]
        if entry.get("visual"):
            field = "visual_translated" if (self.show_translated and entry.get("visual_translated")) else "visual"
            title = "编辑画面事件"
        else:
            field = "speech_translated" if (self.show_translated and entry.get("speech_translated")) else "speech"
            title = "编辑语音条目"
        text, ok = QInputDialog.getMultiLineText(self, title, "内容：", str(entry.get(field) or ""))
        if not ok:
            return
        entry[field] = text.strip()
        self.save_timeline()
        self.refresh_timeline_table()

    def delete_entries(self) -> None:
        entries = self.selected_entries()
        if not entries:
            return
        if QMessageBox.question(self, "删除", f"删除选中的 {len(entries)} 条时间轴条目？") != QMessageBox.Yes:
            return
        keep = [e for e in self.timeline if not any(e is x for x in entries)]
        self.timeline = keep
        self.timeline_doc["timeline"] = self.timeline
        self.save_timeline()
        self.refresh_timeline_table()

    def clear_entries(self) -> None:
        if not self.timeline:
            return
        if QMessageBox.question(self, "清空", "清空全部时间轴条目？") != QMessageBox.Yes:
            return
        self.timeline = []
        self.save_timeline()
        self.refresh_timeline_table()

    # ------------------------------------------------------- 右键菜单：日志
    def on_log_menu(self, pos) -> None:
        menu = QMenu(self)
        act_all = menu.addAction("全选")
        act_copy = menu.addAction("复制")
        act_edit = menu.addAction("只读" if not self.log_view.isReadOnly() else "编辑")
        act_del = menu.addAction("删除（选中部分）")
        act_clear = menu.addAction("清空")
        chosen = menu.exec_(self.log_view.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_all:
            self.log_view.selectAll()
        elif chosen is act_copy:
            self.log_view.copy()
            if not self.log_view.textCursor().hasSelection():
                self.copy_lines([self.log_view.toPlainText()])
        elif chosen is act_edit:
            self.log_view.setReadOnly(not self.log_view.isReadOnly())
            self.statusBar().showMessage("日志已切换为只读" if self.log_view.isReadOnly() else "日志可编辑")
        elif chosen is act_del:
            cursor = self.log_view.textCursor()
            if cursor.hasSelection():
                read_only = self.log_view.isReadOnly()
                self.log_view.setReadOnly(False)
                cursor.removeSelectedText()
                self.log_view.setReadOnly(read_only)
        elif chosen is act_clear:
            self.log_view.clear()

    # ------------------------------------------------------------------ 复制
    def copy_lines(self, lines: list[str]) -> None:
        text = "\n".join(l for l in lines if l)
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"已复制 {len(text.splitlines())} 行")

    # ------------------------------------------------------------------ 导出
    def _events_for_export(self) -> list[dict[str, Any]]:
        """把时间轴条目整理成导出用的事件结构（画面条目才算事件）。"""
        out = []
        for e in self.timeline:
            if not e.get("visual"):
                continue
            out.append({
                "start": e["start"], "end": e["end"],
                "description": e.get("visual"),
                "description_translated": e.get("visual_translated"),
                "event": "", "importance": e.get("importance") or "",
                "ocr_text": e.get("ocr_text"),
                # 结构化事实：动作轨要靠 action 归并（老结果里 timeline.json 没有轨时的兜底）
                "action": e.get("action"),
                "scene": e.get("scene"),
                "subjects": e.get("subjects") or [],
                # 画面事件只带画面情绪，语音情绪由语音段自己带，导出时不会串行
                "emotion": e.get("visual_emotion"),
                "emotion_en": e.get("visual_emotion_en"),
                "emotion_intensity": e.get("visual_emotion_intensity"),
            })
        return out

    def _ask_path(self, default_name: str, filter_text: str) -> Path | None:
        out = self.export_root()
        path, _ = QFileDialog.getSaveFileName(self, "导出到", str(out / default_name), filter_text)
        return Path(path) if path else None

    def remember_export_dir(self, path: Path) -> None:
        """记住这次导出去了哪儿，下次对话框直接开在那里。"""
        if path.parent.is_dir():
            self.export_dir = path.parent
            self.save_settings()
            self.refresh_export_hint()


    def export_text(self, kind: str) -> None:
        """导出语音 / 事件 / 合并。

        语音和事件默认存成剪映可用的 SRT（对话框里可以改存成 .txt）；
        合并导出是给人读的时间线，只出 txt。
        """
        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开一个视频")
            return
        stem = self.video_path.stem
        lang = self.export_language()
        w = txt_words(lang)
        suffix = f"_{w['translated_file']}" if self.show_translated else ""
        kind_word = {"speech": w["speech_file"], "events": w["events_file"],
                     "merged": w["merged_file"]}[kind]
        srt_ok = kind in ("speech", "events")
        ext = ".srt" if srt_ok else ".txt"
        filters = "字幕文件 (*.srt);;文本文件 (*.txt)" if srt_ok else "文本文件 (*.txt)"
        path = self._ask_path(f"{stem}_{kind_word}{suffix}{ext}", filters)
        if path is None:
            return
        if srt_ok and path.suffix.lower() == ".srt":
            self._write_srt(path, kind)
            return
        try:
            if kind == "speech":
                count = write_speech_txt(path, self.video_path.name, self.speech,
                                         self.show_translated, lang)
            elif kind == "events":
                count = write_events_txt(path, self.video_path.name, self._events_for_export(),
                                        self.show_translated, lang)
            else:
                # 动作轨/表情轨来自 timeline.json；老结果里没有就让导出层从事件现算动作轨
                count = write_merged_txt(path, self.video_path.name, self.speech,
                                        self._events_for_export(), self.show_translated, lang,
                                        actions=self.timeline_doc.get("action_track"),
                                        emotions=self.timeline_doc.get("expression_track"),
                                        duration=float(self.timeline_doc.get("duration") or 0.0))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.append_log(f"[导出] {path}（{count} 条）")
        self.statusBar().showMessage(f"已导出 {count} 条到 {path.name}")
        self.remember_export_dir(path)

    def export_words(self) -> None:
        """逐词导出：一个词一条，时间用 whisper 的 word_timestamps。

        单独一份，不动句级那份——句级带译文和情绪，逐词只有原文（逐词翻译是词表，
        情绪模型也要求至少 0.3s 音频，多数词达不到）。
        SRT 走 `write_capcut_srt`，所以时间轴是标准化过的：升序、不重叠、不零长。
        """
        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开一个视频")
            return
        items = words_of(self.speech)
        if not items:
            QMessageBox.information(self, "提示", "没有可导出的词——这份结果没有词级时间戳")
            return
        lang = self.export_language()
        name = txt_words(lang)["words_file"]
        path = self._ask_path(f"{self.video_path.stem}_{name}.srt",
                              "字幕文件 (*.srt);;文本文件 (*.txt)")
        if path is None:
            return
        try:
            if path.suffix.lower() == ".txt":
                count = write_words_txt(path, self.video_path.name, self.speech, lang)
            else:
                count = write_capcut_srt(path, items)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.append_log(f"[逐词导出] {path}（{count} 个词，一词一个时间戳）")
        self.statusBar().showMessage(f"已导出 {count} 个词到 {path.name}")
        self.remember_export_dir(path)

    def _write_srt(self, path: Path, kind: str) -> None:

        """写剪映可用的 SRT：UTF-8 无 BOM、序号连续、时间不重叠、不留空块。"""
        if kind == "speech":
            items = [(float(s["start"]), float(s["end"]), self.speech_display(s)) for s in self.speech]
        else:
            items = [(float(e["start"]), float(e["end"]), self.visual_display(e))
                     for e in self.timeline if e.get("visual")]
        if not items:
            QMessageBox.information(self, "提示", "没有可导出的内容")
            return
        try:
            count = write_capcut_srt(path, items)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.append_log(f"[导出] {path}（{count} 个字幕块，剪映可用：UTF-8 无 BOM、时间已去重叠）")
        self.statusBar().showMessage(f"已导出 {count} 个字幕块到 {path.name}")
        self.remember_export_dir(path)



def launch(cfg: Config, video: str | Path | None = None, panel_only: bool = False,
           auto: bool = False) -> int:
    """起 GUI。panel_only=True 就只开 AI 面板（主界面在后面备着，不显示）。

    面板要用主界面的分析、渲染和 Bridge，所以 MainWindow 照样建，只是不 show；
    关掉面板就退出进程。auto=True 是开起来直接跑一遍自动剪辑，不用手点。
    """
    app = QApplication(sys.argv[:1])
    theme.apply(app)

    # PyQt5 里槽函数抛出的异常会直接 abort 整个进程——界面「闪退」，什么都看不到。
    # 接住它：弹窗给出类型、消息和调用栈，界面继续开着。
    def on_error(kind, value, trace) -> None:
        import traceback  # noqa: PLC0415

        detail = "".join(traceback.format_exception(kind, value, trace))
        print(detail, file=sys.stderr)
        QMessageBox.critical(None, "出错了", f"{kind.__name__}: {value}\n\n{detail[-1500:]}")

    sys.excepthook = on_error
    window = MainWindow(cfg, Path(video) if video else None)
    if not panel_only:
        window.show()
        return app.exec_()

    window.on_ai_options()
    panel = window.ai_panel
    if panel is None:  # 理论上不会，保险起见退回整个主界面
        window.show()
        return app.exec_()
    panel.finished.connect(lambda *_: app.quit())
    panel.set_standalone()  # 独立跑就是正经窗口：任务栏、最小化、最大化都有
    panel.show()
    panel.raise_()
    panel.activateWindow()
    if window.bridge is None:
        # 端口被别人占着（多半是 GUI 已经开着）：这时候扩展连不上，说清楚别让人干等
        port = int(cfg.bridge.get("port") or 5998)
        panel.append_log(f"[AI 对接] Bridge 没起来（{port} 端口被占），扩展连不上")
        QMessageBox.warning(panel, "AI 面板",
                            f"Bridge 起不来：{port} 端口被占用了。\n\n"
                            "多半是主界面（GUI）已经开着——同一个端口只能一个进程用。\n"
                            "要么关掉主界面再开这个面板，要么直接用主界面里的「AI 面板」按钮。\n\n"
                            "接口直连（不用扩展）那条路不受影响，照样能跑。")
    if auto:

        QTimer.singleShot(0, window.on_auto_clip)
    return app.exec_()


