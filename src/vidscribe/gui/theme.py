"""灰色主题 + 应用图标。

图标不落地成图片文件：用 QPainter 现画多尺寸 QPixmap，
这样打包/迁移时不用带资源文件，也不引入新依赖。

配色只用中性灰 + 一个琥珀色强调色：
- 界面主体是灰阶，长时间盯着不刺眼；
- 强调色只出现在"当前选中/进度/正在播放"这类需要一眼找到的地方。
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)

APP_TITLE = "VidScribe · 视频事件与语音时间轴"

# --- 调色板 ---------------------------------------------------------------
BG = "#26292d"          # 窗口底
PANEL = "#2f3338"       # 面板/输入框
PANEL_ALT = "#383d43"   # 悬停、表头
LINE = "#454b52"        # 边框、分隔线
TEXT = "#e7e9ec"
TEXT_DIM = "#98a0a8"
ACCENT = "#d9a441"      # 琥珀：选中、进度、播放位置
ACCENT_DIM = "#8a6a2c"
VIDEO_BG = "#1b1d20"    # 播放器背板

QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
}}
QMainWindow, QDialog {{ background: {BG}; }}

QLabel {{ background: transparent; color: {TEXT}; }}
QLabel[role="section"] {{ color: {TEXT_DIM}; font-weight: 600; padding: 2px 0 4px 2px; }}
QLabel[role="hint"] {{ color: {TEXT_DIM}; }}

QPushButton {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {PANEL_ALT}; border-color: #5a616a; }}
QPushButton:pressed {{ background: #23262a; }}
QPushButton:disabled {{ color: #6c737a; background: #2a2d31; border-color: #3a3f45; }}
QPushButton[role="primary"] {{
    background: {ACCENT}; border: 1px solid {ACCENT}; color: #241c07; font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: #e5b558; }}
QPushButton[role="primary"]:pressed {{ background: #c08f33; }}
QPushButton[role="primary"]:disabled {{ background: {ACCENT_DIM}; border-color: {ACCENT_DIM}; color: #3a3222; }}

QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: #241c07;
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: #5a616a; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {LINE};
    selection-background-color: {PANEL_ALT};
    selection-color: {TEXT};
    outline: none;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background: {PANEL_ALT}; border: none; width: 14px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{ background: #454b52; }}

QTableWidget, QListWidget, QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
    alternate-background-color: #33373c;
    selection-background-color: {PANEL_ALT};
    selection-color: {TEXT};
    outline: none;
}}
QTableWidget {{ gridline-color: #3d4249; }}
QTableWidget::item {{ padding: 4px 6px; border: none; }}
QTableWidget::item:selected, QListWidget::item:selected {{
    background: {PANEL_ALT};
    color: {ACCENT};
}}
QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:hover, QTableWidget::item:hover {{ background: #343a40; }}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {LINE};
    border-bottom: 1px solid {LINE};
    padding: 6px;
    font-weight: 600;
}}
QHeaderView::section:last {{ border-right: none; }}
QTableCornerButton::section {{ background: {PANEL_ALT}; border: none; }}

QProgressBar {{
    background: #23262a;
    border: 1px solid {LINE};
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_DIM}, stop:1 {ACCENT});
    border-radius: 6px;
}}

QSlider::groove:horizontal {{
    background: #23262a; height: 5px; border-radius: 3px; border: 1px solid #3a3f45;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: #f0f2f4; border: 1px solid #1c1e21;
    width: 12px; margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {PANEL_ALT}; }}

QStatusBar {{ background: #222528; color: {TEXT_DIM}; border-top: 1px solid {LINE}; }}
QStatusBar::item {{ border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #4a5057; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #5a6169; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #4a5057; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #5a6169; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: #1c1e21; color: {TEXT};
    border: 1px solid {LINE}; padding: 4px 6px;
}}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; }}
QMenu::item:selected {{ background: {PANEL_ALT}; color: {ACCENT}; }}
"""


def palette() -> QPalette:
    """QSS 之外的兜底：原生绘制的控件（工具提示、禁用态）也走灰色。"""
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL_ALT))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANEL))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#241c07"))
    pal.setColor(QPalette.ToolTipBase, QColor("#1c1e21"))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#6c737a"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6c737a"))
    return pal


def _draw_icon(size: int) -> QPixmap:
    """胶片格 + 声波：一眼看出是"画面 + 语音"。小尺寸下只保留大色块。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 64.0

    # 圆角底：深灰渐变
    body = QRectF(2 * s, 2 * s, 60 * s, 60 * s)
    grad = QLinearGradient(body.topLeft(), body.bottomRight())
    grad.setColorAt(0.0, QColor("#4a5157"))
    grad.setColorAt(1.0, QColor("#22252a"))
    path = QPainterPath()
    path.addRoundedRect(body, 14 * s, 14 * s)
    p.fillPath(path, QBrush(grad))
    p.setPen(QPen(QColor("#151719"), max(1.0, 1.4 * s)))
    p.drawPath(path)

    # 胶片竖条 + 齿孔
    strip = QRectF(11 * s, 13 * s, 12 * s, 38 * s)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#12141600" if size < 20 else "#141618"))
    p.drawRoundedRect(strip, 2.5 * s, 2.5 * s)
    if size >= 24:
        p.setBrush(QColor("#c8ced4"))
        hole_h = 5.2 * s
        for i in range(4):
            top = strip.top() + 3.2 * s + i * (hole_h + 3.4 * s)
            p.drawRoundedRect(QRectF(strip.left() + 2.6 * s, top, 6.8 * s, hole_h), 1.2 * s, 1.2 * s)

    # 声波：围绕同一中线的三根琥珀色竖条（居中才像波形，不像柱状图）
    p.setBrush(QColor(ACCENT))
    bars = ((29, 19, 24), (37, 13, 36), (45, 23, 16)) if size >= 24 else ((30, 20, 22), (42, 14, 34))
    for x, y, h in bars:
        p.drawRoundedRect(QRectF(x * s, y * s, 6 * s, h * s), 3 * s, 3 * s)
    if size >= 32:
        p.setPen(QPen(QColor("#e8ecef"), 1.8 * s, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(int(29 * s), int(54 * s), int(51 * s), int(54 * s))
    p.end()
    return pix


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_draw_icon(size))
    return icon


def apply(app) -> None:
    """给 QApplication 套主题 + 图标；GUI 入口只需要调这一个函数。"""
    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setStyleSheet(QSS)
    app.setWindowIcon(app_icon())
    # 只调字号，不硬指定字族：字族由 QSS 里的候选链决定，
    # 系统缺字时交给 Qt 自己替换，避免出现空白文字。
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)


def icon_size() -> QSize:
    return QSize(18, 18)
