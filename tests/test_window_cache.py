"""窗口缓存（visual_windows.json）回归测试。

盯的是 Phase 6 修掉的那个坑：窗口 key 只带了模型/语言/情绪开关，改 fps、改分辨率预算、
关掉人脸表情（会改 prompt）之后，旧窗口仍然被当成命中复用，Qwen 一次都不重跑。
现在窗口缓存自带 `visual_config_hash`，对不上就整份不命中，但旧文件一个字节都不动。

顺带覆盖原子写：崩在写盘中途时，盘上要么是上一份完整缓存，要么是新的完整缓存。

不依赖 pytest：直接 `python tests/test_window_cache.py` 即可运行。
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.checkpoint import WINDOW_CACHE_VERSION, Checkpoint  # noqa: E402
from vidscribe.config import DEFAULTS  # noqa: E402
from vidscribe.pipeline import _WINDOW_HASH_KEYS, visual_config_hash  # noqa: E402

BASE = copy.deepcopy(DEFAULTS["visual"])
# 改了也不该让长视频重跑的：都在窗口循环之后才用，或者会被 OOM 自动改
UNRELATED = {
    "batch_size": 1, "dedup_similarity": 0.5, "merge_similarity": 0.5,
    "min_event_seconds": 1.5, "max_event_seconds": 30.0, "window_seconds": 20.0,
    "window_overlap_seconds": 5.0, "long_video_threshold": 60.0,
    "models": [], "fallback_model_ids": ["x/y"],
}
# 改了必须整份不命中的：每一项都真的改变 Qwen 的窗口输入或窗口产物
HASHED = {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct", "backend": "minicpm", "dtype": "float16",
    "attn_implementation": "eager", "quantization": "int4", "frame_source": "official",
    "fps": 1.5, "max_frames": 16, "min_frames": 2, "max_pixels_tokens": 224,
    "total_pixels_tokens": 4096, "max_new_tokens": 384, "emotion_enabled": False,
    "snap_tolerance_seconds": 2.5, "scene_detect": False, "scene_threshold": 0.7,
    "scene_sample_fps": 1.0,
}


def key(idx: int) -> str:
    return "Qwen3-VL-4B-Instruct|en|em1|%d:%.3f-%.3f" % (idx, idx * 15.0, idx * 15.0 + 15.0)


def window(idx: int) -> dict:
    return {"events": [{"i": idx}], "meta": {"frames": 8}}


def test_hash_field_selection() -> None:
    base = visual_config_hash(BASE, backend="qwen3vl")
    assert len(base) == 64, "visual_config_hash 必须是 SHA256 全 64 位"
    assert visual_config_hash(BASE, backend="qwen3vl") == base, "同一份配置必须稳定"
    for field, value in UNRELATED.items():
        cfg = copy.deepcopy(BASE)
        cfg[field] = value
        assert visual_config_hash(cfg, backend="qwen3vl") == base, f"{field} 不该影响窗口复用"
    assert set(HASHED) == set(_WINDOW_HASH_KEYS), "这份清单要跟 _WINDOW_HASH_KEYS 一致"
    for field, value in HASHED.items():
        cfg = copy.deepcopy(BASE)
        cfg[field] = value
        got = visual_config_hash(cfg, backend=value if field == "backend" else "qwen3vl")
        assert got != base, f"{field} 改了必须整份不命中"
    # 人脸表情开关会改 user prompt（视觉模型还判不判情绪），必须进哈希
    cfg = copy.deepcopy(BASE)
    cfg["face_emotion"] = dict(cfg["face_emotion"])
    cfg["face_emotion"]["enabled"] = not cfg["face_emotion"]["enabled"]
    assert visual_config_hash(cfg, backend="qwen3vl") != base
    # 但人脸检测的采样参数是独立阶段的事，不该拖累窗口复用
    cfg = copy.deepcopy(BASE)
    cfg["face_emotion"] = dict(cfg["face_emotion"])
    cfg["face_emotion"]["sample_fps"] = 9.0
    assert visual_config_hash(cfg, backend="qwen3vl") == base
    print("PASS 字段取舍")


def test_v2_hit_and_legacy(work: Path) -> None:
    video = work / "clip.mp4"
    video.write_bytes(b"window-cache-test" * 32)
    hash_a = visual_config_hash(BASE, backend="qwen3vl")
    cfg_b = copy.deepcopy(BASE)
    cfg_b["max_frames"] = 12
    hash_b = visual_config_hash(cfg_b, backend="qwen3vl")

    def ckpt() -> Checkpoint:
        return Checkpoint(work / "cache", video)

    ckpt().save_window_cache({key(0): window(0), key(1): window(1)}, hash_a)
    raw = json.loads(ckpt().window_cache_file().read_text(encoding="utf-8"))
    assert raw["version"] == WINDOW_CACHE_VERSION
    assert raw["visual_config_hash"] == hash_a
    assert len(ckpt().load_window_cache(hash_a)) == 2, "同配置必须照常续跑"
    assert ckpt().load_window_cache(hash_b) == {}, "换了配置必须整份不命中"
    assert ckpt().window_cache_file().is_file(), "不命中不等于删文件"

    # v1 裸 dict：读得出来，但没记自己是哪套配置产的，不能猜
    legacy = {key(0): window(0), key(1): window(1), key(2): window(2)}
    ckpt().window_cache_file().write_text(json.dumps(legacy), encoding="utf-8")
    before = ckpt().window_cache_file().read_bytes()
    assert ckpt().load_window_cache(hash_a) == {}, "legacy 不许命中"
    assert ckpt().window_cache_file().read_bytes() == before, "legacy 文件不许被动"
    print("PASS v2 命中 / legacy 不命中")


def test_resume(work: Path) -> None:
    video = work / "long.mp4"
    video.write_bytes(b"resume-test" * 64)
    hash_a = visual_config_hash(BASE, backend="qwen3vl")

    def ckpt() -> Checkpoint:
        return Checkpoint(work / "cache2", video)

    ckpt().save_window_cache({key(i): window(i) for i in range(30)}, hash_a)
    resumed = ckpt().load_window_cache(hash_a)
    assert [i for i in range(100) if key(i) in resumed] == list(range(30)), "跑过的 30 个要全中"
    assert key(30) not in resumed, "没跑的不能凭空命中"
    print("PASS 长视频续跑")


def test_atomic_write(work: Path) -> None:
    video = work / "atomic.mp4"
    video.write_bytes(b"atomic-test" * 64)
    hash_a = visual_config_hash(BASE, backend="qwen3vl")
    ck = Checkpoint(work / "cache3", video)
    ck.save_window_cache({key(i): window(i) for i in range(5)}, hash_a)
    good = ck.window_cache_file().read_bytes()

    orig = json.dump

    def boom(payload, fh, **kwargs):  # 写了一半就崩
        fh.write('{"version": 2, "windows": {"half')
        raise RuntimeError("simulated crash")

    json.dump = boom
    try:
        ck.save_window_cache({key(0): window(0)}, hash_a)
        raise AssertionError("这里本该抛异常")
    except RuntimeError:
        pass
    finally:
        json.dump = orig
    assert ck.window_cache_file().read_bytes() == good, "崩在写盘中途不能毁掉上一份缓存"
    assert len(Checkpoint(work / "cache3", video).load_window_cache(hash_a)) == 5
    ck.window_cache_file().with_name(ck.window_cache_file().name + ".tmp").unlink(missing_ok=True)
    print("PASS 原子写")


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="window_cache_test_"))
    try:
        test_hash_field_selection()
        test_v2_hit_and_legacy(work)
        test_resume(work)
        test_atomic_write(work)
        print("全部通过")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
