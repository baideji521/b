"""下载 CC0 音效库并按高光剪辑的用途归类到 assets/sfx/<类别>/。

素材来自 kenney.nl 的音频包，全部是 CC0（公共领域，免费商用、无需署名）。
音效文件不进版本库（.gitignore 里排除了 assets/sfx/），需要时重跑这个脚本即可：

    .venv\\Scripts\\python.exe tools\\fetch_sfx.py

类别就是 highlight.sfx.emotion_map 里用的那几个名字，改类别要同步改 config.json。
自己另外找的音效直接丢进对应类别目录也能用，扫描按目录走，不认文件名。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFX_DIR = ROOT / "assets" / "sfx"

# kenney.nl 的直链（"Continue without donating" 那个），路径里带内容哈希，改版会失效；
# 失效时去 https://kenney.nl/assets/category:Audio 对应包页面重新取一次
PACKS: dict[str, str] = {
    "impact": "https://kenney.nl/media/pages/assets/impact-sounds/"
              "87b4ddecda-1677589768/kenney_impact-sounds.zip",
    "interface": "https://kenney.nl/media/pages/assets/interface-sounds/"
                 "fa43c1dd4d-1677589452/kenney_interface-sounds.zip",
    "ui": "https://kenney.nl/media/pages/assets/ui-audio/"
          "490d233f68-1677590494/kenney_ui-audio.zip",
    "jingles": "https://kenney.nl/media/pages/assets/music-jingles/"
               "f37e530b9e-1677590399/kenney_music-jingles.zip",
    "digital": "https://kenney.nl/media/pages/assets/digital-audio/"
               "216eac4753-1677590265/kenney_digital-audio.zip",
    "scifi": "https://kenney.nl/media/pages/assets/sci-fi-sounds/"
             "6b296f9ecf-1677589334/kenney_sci-fi-sounds.zip",
    "casino": "https://kenney.nl/media/pages/assets/casino-audio/"
              "2472606a04-1721639069/kenney_casino-audio.zip",
}

# 类别 -> (包名, 文件名前缀) 白名单。一个包里几百个文件，只挑短视频用得上的。
CATEGORIES: dict[str, tuple[tuple[str, str], ...]] = {
    # 冻帧那一下的重击，配 Zoom Punch
    "punch": (
        ("impact", "impactPunch_heavy"), ("impact", "impactPunch_medium"),
        ("impact", "impactMetal_heavy"), ("impact", "impactBell_heavy"),
        ("scifi", "lowFrequency_explosion"), ("scifi", "explosionCrunch"),
    ),
    # 逐字弹出、轻点，短促干净
    "pop": (
        ("interface", "click_"), ("interface", "pluck_"), ("interface", "select_"),
        ("interface", "drop_"), ("ui", "click"), ("ui", "rollover"),
    ),
    # 揭晓 / 确认 / 答对
    "ding": (
        ("interface", "confirmation_"), ("interface", "bong_"), ("interface", "glass_"),
        ("interface", "question_"), ("jingles", "jingles_HIT"), ("digital", "pepSound"),
    ),
    # 搞笑、滑稽、吐槽
    "funny": (
        ("jingles", "jingles_PIZZI"), ("jingles", "jingles_SAX"),
        ("jingles", "jingles_STEEL"), ("casino", "dice-shake"), ("casino", "card-fan"),
    ),
    # 悬念上扬，铺在揭晓之前
    "riser": (
        ("digital", "phaserUp"), ("digital", "powerUp"), ("digital", "threeTone"),
        ("digital", "zapThreeToneUp"), ("digital", "highUp"),
    ),
    # 转场、扫过
    "whoosh": (
        ("interface", "maximize_"), ("interface", "minimize_"),
        ("scifi", "forceField_"), ("scifi", "thrusterFire_"),
    ),
    # 翻车、尴尬、失败
    "fail": (
        ("interface", "error_"), ("interface", "glitch_"),
        ("digital", "phaserDown"), ("digital", "lowDown"), ("digital", "highDown"),
    ),
}

AUDIO_SUFFIXES = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
# 直连 kenney.nl 不带 UA 会被挡
HEADERS = {"User-Agent": "Mozilla/5.0 (VidScribe fetch_sfx)"}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS)  # noqa: S310 - 常量直链
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        target.write_bytes(response.read())


def collect(pack_dir: Path) -> dict[str, Path]:
    """包里的音频文件：文件名 -> 路径（同名只留一个，包内本来就不重名）。"""
    found: dict[str, Path] = {}
    for path in pack_dir.rglob("*"):
        if path.suffix.lower() in AUDIO_SUFFIXES and path.name.lower() != "preview.ogg":
            found.setdefault(path.name, path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并归类 CC0 音效库")
    parser.add_argument("--keep-temp", action="store_true",
                       help="保留下载的 zip 和解压目录（默认跑完就删）")
    args = parser.parse_args()

    temp = SFX_DIR / "_packs"
    files: dict[str, dict[str, Path]] = {}
    for pack, url in PACKS.items():
        zip_path = temp / f"{pack}.zip"
        print(f"[下载] {pack} <- {url}")
        try:
            download(url, zip_path)
        except Exception as exc:  # noqa: BLE001 - 单个包失败不影响其他包
            print(f"[跳过] {pack} 下载失败：{exc}")
            continue
        pack_dir = temp / pack
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(pack_dir)
        files[pack] = collect(pack_dir)
        print(f"[解压] {pack} 共 {len(files[pack])} 个音频文件")

    if not files:
        print("一个包都没下下来，检查网络或直链是否失效")
        return 1

    total = 0
    for category, rules in CATEGORIES.items():
        target_dir = SFX_DIR / category
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for pack, prefix in rules:
            for name, path in sorted(files.get(pack, {}).items()):
                if not name.lower().startswith(prefix.lower()):
                    continue
                shutil.copy2(path, target_dir / f"{pack}_{name}")
                count += 1
        print(f"[归类] {category:8s} {count:4d} 个")
        total += count

    if not args.keep_temp:
        shutil.rmtree(temp, ignore_errors=True)
    print(f"[完成] {SFX_DIR} 共 {total} 个音效（CC0 / kenney.nl，免费商用无需署名）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
