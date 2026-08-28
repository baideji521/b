"""把 GUI 的应用图标导出成 PNG，同步给浏览器扩展 AI_剪辑师_好帮手。

图标只有一份源：src/vidscribe/gui/theme.py 的 _draw_icon（纯代码画的，没有素材文件）。
改了那边的画法，跑一次这个脚本，GUI 和扩展的图标就一起更新：

    .venv\\Scripts\\python.exe tools\\make_icons.py

不接受尺寸参数：扩展 manifest 里引用的就是这几个尺寸，多生成也没人用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 扩展目录名跟着产品名走，改名了就改这里
EXTENSION_ICONS = ROOT / "AI_剪辑师_好帮手" / "icons"
SIZES = (16, 24, 32, 48, 128)


def main() -> int:
    # 无界面环境也要能画，用 offscreen 后端
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT / "src"))
    from PyQt5.QtWidgets import QApplication  # noqa: PLC0415 - 必须在设置后端之后导入

    from vidscribe.gui import theme  # noqa: PLC0415

    app = QApplication([])  # QPixmap 需要一个 QApplication 存在
    EXTENSION_ICONS.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        target = EXTENSION_ICONS / f"active-{size}.png"
        if not theme._draw_icon(size).save(str(target), "PNG"):
            print(f"[失败] 写不出 {target}")
            return 1
        print(f"[生成] {target.relative_to(ROOT)}  {target.stat().st_size} B")
    print(f"[完成] 图标已与 {theme.APP_TITLE} 同步（源：gui/theme.py _draw_icon）")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
