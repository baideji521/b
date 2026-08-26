"""项目根目录入口，免去手工设置 PYTHONPATH。

    python run.py check
    python run.py download
    python run.py run test.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vidscribe.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
