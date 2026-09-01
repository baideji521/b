"""pytest 适配层：这批测试原本是「直接 python tests/xxx.py」跑的独立脚本。

它们的 `main()` 自己造临时目录传进去，参数名有 `work` / `_tmp` 两种；
用 pytest 收集时这些参数会被当成 fixture 找不着，整份文件直接 ERROR。
这里把两个名字都补成「一个干净的临时目录」，脚本入口和 pytest 两条路都能跑，
测试内容一个字没改。

另外 test_frame_sampling.py 里有两个用例需要仓库根目录的真实 test.mp4
（脚本入口里本来就会 SKIP）：文件不在时这里也 SKIP，而不是报 FileNotFoundError。
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 需要真实视频的用例：没有 test.mp4 就跳过（跟各脚本 main() 里的判断一致）
NEEDS_REAL_VIDEO = {
    "test_frame_sampling.py": ("test_max_frames_takes_effect", "test_timestamps_are_real"),
}


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    """脚本里叫 work 的那个临时工作目录。"""
    return tmp_path


@pytest.fixture()
def _tmp(tmp_path: Path) -> Path:
    """脚本里叫 _tmp 的那个临时工作目录。"""
    return tmp_path


def pytest_runtest_setup(item: pytest.Item) -> None:
    names = NEEDS_REAL_VIDEO.get(Path(str(item.fspath)).name, ())
    if item.name in names and not (ROOT / "test.mp4").is_file():
        pytest.skip("缺少仓库根目录的真实 test.mp4，跳过真实解码用例")
