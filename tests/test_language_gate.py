"""语言拦截：只跑 speech.allowed_languages 里的语言，其它语言当场停下并标记。

盯的是这一批的红线：
  T1  allowed_languages 两种写法都认（字符串 / 列表），大小写和 en-US 这种写法都归一
  T2  留空 = 谁都跑（老配置不许被新功能拦住）
  T3  语言不在名单里 → 抛 LanguageNotAllowed，而且带着判定出来的语言码
  T4  语言预检失败（检测不出来）不拦：宁可多跑一条，也不许误杀
  T5  拦在**完整识别之前**：_language_and_prompt 就抛，transcribe 那几分钟根本不花
  T6  标记落在 videos.blocked_language 上，只记语言码；新视频默认没标记
  T7  子进程打的 `[语言拦截] language=xx` 被 AnalyzeWorker 认出来（GUI 靠它弹窗 / 跳过）
  T8  pipeline 先标记再记 failed 再往外抛；cli 把这条记成 SKIP_LANGUAGE，不是失败

全部用临时目录里的临时库，**绝不碰项目真实数据库**，也不加载任何模型。
可以直接 `python tests/test_language_gate.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication              # noqa: E402

from vidscribe.config import Config                   # noqa: E402
from vidscribe.db import open_db                      # noqa: E402
from vidscribe.db import repo as db_repo              # noqa: E402
from vidscribe.gui import main_window as mw           # noqa: E402
from vidscribe.speech.whisper_asr import (            # noqa: E402
    LanguageNotAllowed,
    WhisperASR,
)

CLI_SRC = (ROOT / "src" / "vidscribe" / "cli.py").read_text(encoding="utf-8")
PIPELINE_SRC = (ROOT / "src" / "vidscribe" / "pipeline.py").read_text(encoding="utf-8")

_APP = None


def app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


# ------------------------------------------------------------------ 夹具
def make_project(tmp_path: Path):
    for sub in ("database", "input", "output", "logs", "cache"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data.setdefault("paths", {}).update({
        "db_dir": str(tmp_path / "database"),
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "input_dir": str(tmp_path / "input"),
        "video_dir": "",
        "log_dir": str(tmp_path / "logs"),
    })
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def asr(**speech) -> WhisperASR:
    """只造对象不加载模型：这一批只测语言判定，不测识别。"""
    cfg = {"model_size": "large-v3"}
    cfg.update(speech)
    return WhisperASR(cfg)


class ExplodingModel:
    """预检一碰就炸的模型替身：用来验证"检测失败不拦"。"""

    def detect_language(self, audio=None):
        raise RuntimeError("音频解不开")


# ------------------------------------------------------------------ T1-T2 名单
def test_allowlist_reads_both_shapes(_tmp: Path) -> None:
    assert asr(allowed_languages=["en", "zh"])._allowed_languages() == ("en", "zh")
    assert asr(allowed_languages="EN")._allowed_languages() == ("en",)
    assert asr(allowed_languages=["EN-US", " zh "])._allowed_languages() == ("en", "zh")
    assert asr(allowed_languages=["en", ""])._allowed_languages() == ("en",), "空串不算一项"


def test_empty_allowlist_runs_everything(_tmp: Path) -> None:
    engine = asr(allowed_languages=[])
    assert engine._allowed_languages() == ()
    engine._gate_language("id", 0.93, engine._allowed_languages())   # 不许抛
    assert asr()._allowed_languages() == (), "没配这一项就是不拦"


# ------------------------------------------------------------------ T3-T4 拦不拦
def test_allowed_language_passes(_tmp: Path) -> None:
    engine = asr(allowed_languages=["en", "zh"])
    for code in ("en", "zh", "EN", "zh-cn"):
        engine._gate_language(code, 0.9, ("en", "zh"))


def test_other_language_is_stopped(_tmp: Path) -> None:
    engine = asr(allowed_languages=["en", "zh"])
    try:
        engine._gate_language("id", 0.93, ("en", "zh"))
    except LanguageNotAllowed as exc:
        assert exc.language == "id", f"得带上判定出来的语言：{exc.language}"
        assert exc.probability == 0.93
    else:
        raise AssertionError("印尼语必须被拦下来")


def test_detection_failure_never_blocks(_tmp: Path) -> None:
    """检测不出来（None）就不拦：宁可多跑一条，也不许把好素材误杀。"""
    engine = asr(allowed_languages=["en", "zh"])
    engine._gate_language(None, None, ("en", "zh"))
    engine.model = ExplodingModel()
    assert engine._detect_language(None) == (None, None), "预检炸了要安静降级"


# ------------------------------------------------------------------ T5 拦在识别之前
def test_gate_happens_before_transcribe(_tmp: Path) -> None:
    """语言判定 + 拦截都在 _language_and_prompt 里，transcribe 的几分钟一秒都不花。"""
    engine = asr(allowed_languages=["en", "zh"], language="id",
                 initial_prompt={"en": "hi", "zh": "你好"})
    engine.model = ExplodingModel()      # 模型一碰就炸：证明这条路不碰模型
    try:
        engine._language_and_prompt(None, True)
    except LanguageNotAllowed as exc:
        assert exc.language == "id"
    else:
        raise AssertionError("语言不符必须在挑 prompt / 识别之前就停下")
    ok = asr(allowed_languages=["en", "zh"], language="en",
             initial_prompt={"en": "hi", "zh": "你好"})
    assert ok._language_and_prompt(None, True) == ("en", "hi"), "允许的语言照常挑同语言 prompt"


# ------------------------------------------------------------------ T6 标记
def test_mark_lives_on_the_video_row(tmp_path: Path) -> None:
    cfg, db = make_project(tmp_path)
    video = cfg.path("input_dir") / "a.mp4"
    video.write_bytes(b"a" * 512)
    vid = db_repo.upsert_video(db, video)
    assert db_repo.blocked_language(db, vid) is None, "新视频默认没被拦过"
    db_repo.set_blocked_language(db, vid, "ID-latn")
    assert db_repo.blocked_language(db, vid) == "id", "只记语言码，大小写和地区后缀都归一"
    db_repo.set_blocked_language(db, vid, None)
    assert db_repo.blocked_language(db, vid) is None, "置空＝没标记"
    assert db_repo.blocked_language(db, 9999) is None, "查不存在的视频不许炸"


# ------------------------------------------------------------------ T7 子进程 → GUI
def test_worker_picks_up_the_language_line(tmp_path: Path) -> None:
    """子进程打的那一行就是 GUI 的唯一依据，格式变了这条会红。"""
    app()
    lines = ["$ 开始",
             "[语言拦截] language=id video=" + str(tmp_path / "带 空格 的.mp4"),
             "分析结束"]
    worker = mw.AnalyzeWorker(ROOT, ["highlight"], label="分析")
    real = mw.subprocess
    mw.subprocess = FakeSubprocess(lines)
    try:
        worker.run()
    finally:
        mw.subprocess = real
    assert worker.blocked_language == "id", f"没认出语言码：{worker.blocked_language}"

    plain = mw.AnalyzeWorker(ROOT, ["highlight"], label="分析")
    mw.subprocess = FakeSubprocess(["一切正常", "分析结束"])
    try:
        plain.run()
    finally:
        mw.subprocess = real
    assert plain.blocked_language is None, "没有那一行就不许标语言"


class FakeProc:
    def __init__(self, lines: list[str]):
        self.stdout = iter(line + "\n" for line in lines)

    def wait(self) -> int:
        return 1        # 语言拦截那次子进程是非零退出

    def poll(self):
        return 1


class FakeSubprocess:
    """只够 AnalyzeWorker.run 用的 subprocess 替身，绝不真起进程。"""

    PIPE = -1
    STDOUT = -2

    def __init__(self, lines: list[str]):
        self._lines = lines

    def Popen(self, *_args, **_kwargs):   # noqa: N802 - 名字要跟真模块一样
        return FakeProc(self._lines)


# ------------------------------------------------------------------ T8 落库顺序 / 不算失败
def test_pipeline_marks_before_it_fails_and_reraises(_tmp: Path) -> None:
    body = PIPELINE_SRC.split("except LanguageNotAllowed as exc:", 1)[1][:400]
    mark = body.index("run.block_language(exc.language)")
    fail = body.index("run.fail(exc)")
    assert mark < fail, "标记要在记 failed 之前落库"
    assert "raise" in body[fail:], "标记完还得往外抛，不能咽掉"
    assert "except LanguageNotAllowed:\n                    raise" in PIPELINE_SRC, \
        "语音识别那一步不许把语言拦截当成普通识别失败咽掉"


def test_cli_reports_skip_not_failure(_tmp: Path) -> None:
    assert '[语言拦截] language={exc.language} video={video}' in CLI_SRC, \
        "GUI 靠这一行认语言，格式不许改"
    assert '"status": "SKIP_LANGUAGE"' in CLI_SRC, "语言不符是跳过，不是失败"


# ------------------------------------------------------------------ 跑
TESTS = (
    test_allowlist_reads_both_shapes,
    test_empty_allowlist_runs_everything,
    test_allowed_language_passes,
    test_other_language_is_stopped,
    test_detection_failure_never_blocks,
    test_gate_happens_before_transcribe,
    test_mark_lives_on_the_video_row,
    test_worker_picks_up_the_language_line,
    test_pipeline_marks_before_it_fails_and_reraises,
    test_cli_reports_skip_not_failure,
)


def main() -> int:
    failed = 0
    for fn in TESTS:
        work = Path(tempfile.mkdtemp(prefix="langgate_"))
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
