"""舞蹈子系统测试共用夹具：造合成音乐、合成视频、临时项目。

这些测试**绝不碰项目真实数据库**，所有库都建在 pytest 给的临时目录里。
造素材一律用合成信号，不依赖仓库里有没有 test.mp4。

可以 `pytest tests/test_dance_*.py`，也可以 `python tests/test_dance_*.py`。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.dance import dsp  # noqa: E402

SR = dsp.DEFAULT_SR


def song(bpm: float = 120.0, duration: float = 30.0, seed: int = 0,
         sample_rate: int = SR) -> np.ndarray:
    """造一段有节拍、有和声走向的假音乐（mono float32）。

    三层叠加：持续的低音垫 + 每拍一个衰减 kick + 每拍一小段噪声 hihat，
    再加一条 16 段不重复的和弦进行 —— 和弦不重复很重要，否则 chroma 相关会周期性
    出现同样高的峰，测试就不是在测算法而是在测运气。
    """
    total = int(sample_rate * duration)
    t = np.arange(total) / sample_rate
    rs = np.random.RandomState(seed)
    x = (0.06 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    period = 60.0 / float(bpm)
    kick = (np.exp(-np.arange(2500) / 320.0)
            * np.sin(2 * np.pi * 70 * np.arange(2500) / sample_rate)).astype(np.float32)
    for beat in range(int(duration / period)):
        start = int(beat * period * sample_rate)
        room = min(kick.size, total - start)
        if room > 0:
            x[start:start + room] += kick[:room]
        hat = (np.exp(-np.arange(700) / 70.0) * rs.randn(700)).astype(np.float32) * 0.22
        room = min(hat.size, total - start)
        if room > 0:
            x[start:start + room] += hat[:room]
    chords = (261.6, 293.7, 329.6, 392.0, 349.2, 246.9, 220.0, 311.1,
              277.2, 415.3, 466.2, 523.3, 196.0, 174.6, 164.8, 146.8)
    for index, second in enumerate(range(0, int(duration), 2)):
        start = int(second * sample_rate)
        room = min(int(2 * sample_rate), total - start)
        if room <= 0:
            break
        freq = chords[index % len(chords)]
        x[start:start + room] += (0.06 * np.sin(
            2 * np.pi * freq * np.arange(room) / sample_rate)).astype(np.float32)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SR) -> Path:
    """把 mono float32 写成 16bit PCM wav。用标准库 `wave`，不依赖 PyAV 编码器。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    frames = (data * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(frames)
    return path


def write_video(path: Path, pcm: np.ndarray, *, fps: float = 24.0,
                width: int = 128, height: int = 128, sample_rate: int = SR,
                tint: tuple[int, int, int] = (200, 80, 80)) -> Path:
    """造一个带音轨的小 mp4：画面是随时间变化的纯色块 + 一个移动方块。

    画面刻意做成"每一帧都不同"（色相随帧号走、方块位置随帧号走），这样渲染出来的
    成片可以肉眼/程序验证"第 N 段确实来自源视频的第 M 帧"，而不是全绿糊一片。
    """
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    duration = data.size / float(sample_rate)
    total_frames = max(1, int(round(duration * fps)))
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=int(round(fps)))
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "28", "preset": "ultrafast"}
        audio = container.add_stream("aac", rate=int(sample_rate))
        audio.codec_context.layout = "mono"
        for index in range(total_frames):
            frame = _paint(index, total_frames, width, height, tint)
            picture = av.VideoFrame.from_ndarray(frame, format="rgb24")
            picture.pts = index
            for packet in video.encode(picture):
                container.mux(packet)
        _mux_audio(container, audio, data, sample_rate)
        for packet in video.encode():
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return path


def _paint(index: int, total: int, width: int, height: int,
           tint: tuple[int, int, int]) -> np.ndarray:
    """第 index 帧的画面：底色随进度渐变，再画一个随帧号横移的白方块。"""
    ratio = index / float(max(1, total - 1))
    frame = np.empty((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        frame[:, :, channel] = int(np.clip(tint[channel] * (0.35 + 0.65 * ratio), 0, 255))
    size = max(4, width // 8)
    left = int(ratio * (width - size))
    top = height // 2 - size // 2
    frame[top:top + size, left:left + size] = 255
    return frame


def _mux_audio(container, stream, data: np.ndarray, sample_rate: int) -> None:
    """按 AAC 的 frame_size 分块喂音频，pts 按累计样本数排（同 highlight/clip.py 的写法）。"""
    from fractions import Fraction

    import av

    size = int(stream.frame_size or 1024)
    fed = 0
    for begin in range(0, data.size, size):
        block = np.ascontiguousarray(data[begin:begin + size].reshape(1, -1))
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
        frame.sample_rate = int(sample_rate)
        frame.time_base = Fraction(1, int(sample_rate))
        frame.pts = fed
        fed += frame.samples
        for packet in stream.encode(frame):
            container.mux(packet)


# ------------------------------------------------------------------ 临时项目
def make_project(tmp_path: Path):
    """在临时目录里搭一个完整项目（配置 + 库），返回 `(cfg, db)`。

    刻意从仓库的 config.json 起步再整段覆盖 paths / dance 目录：这样默认值走的是
    真实配置那一套，而落盘位置全在临时目录里 —— 绝不碰项目真实数据库。
    """
    import json

    from vidscribe.config import Config
    from vidscribe.db import open_db

    for sub in ("database", "input", "output", "logs", "cache",
                "dance_songs", "dance_materials", "dance_out"):
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
    data["dance"] = {
        "song_dir": str(tmp_path / "dance_songs"),
        "source_dir": str(tmp_path / "input"),
        "material_dir": str(tmp_path / "dance_materials"),
        "output_dir": str(tmp_path / "dance_out"),
        "slice_duration": 2.0,
        "align_workers": 2,
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cfg = Config.load(tmp_path, cfg_file)
    cfg.ensure_dirs()
    db = open_db(cfg)
    assert str(cfg.path("db_dir")).startswith(str(tmp_path)), "测试库必须在临时目录里"
    return cfg, db


def make_song_file(cfg, name: str = "target.wav", *, bpm: float = 120.0,
                   duration: float = 20.0, seed: int = 0) -> tuple[Path, np.ndarray]:
    """写一首目标歌到 dance.song_dir，返回 `(路径, pcm)`。"""
    pcm = song(bpm=bpm, duration=duration, seed=seed)
    path = Path(cfg.dance["song_dir"]) / name
    write_wav(path, pcm)
    return path, pcm


def make_source_video(cfg, name: str, pcm: np.ndarray, *, tint=(200, 80, 80),
                      fps: float = 24.0) -> Path:
    """写一个源舞蹈视频到 dance.source_dir。"""
    path = Path(cfg.dance["source_dir"]) / name
    return write_video(path, pcm, tint=tint, fps=fps)


def delayed(pcm: np.ndarray, seconds: float, sample_rate: int = SR) -> np.ndarray:
    """截掉开头 `seconds` 秒 —— 得到的音频满足 `source_time = target_time - seconds`。"""
    return np.ascontiguousarray(pcm[int(round(seconds * sample_rate)):])


def padded(pcm: np.ndarray, seconds: float, sample_rate: int = SR) -> np.ndarray:
    """在开头补 `seconds` 秒静音 —— offset 为负的情形（源比目标早开始）。"""
    pad = np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)
    return np.concatenate([pad, np.asarray(pcm, dtype=np.float32)])


__all__ = [
    "ROOT", "SR", "song", "write_wav", "write_video",
    "make_project", "make_song_file", "make_source_video", "delayed", "padded",
    "fake_song", "fake_video", "fake_material", "fake_library",
    "fake_alignment", "fake_version",
]



# ------------------------------------------------- 纯逻辑测试用的假素材库
# 评分 / 筛选 / 组合搜索 / 推荐 / 统计这几层根本不碰视频文件，给它们真编码素材
# 是纯浪费（每条素材几百毫秒）。所以这里直接往库里塞行，几毫秒就能造出上百条素材。
# 代价是这些行的 file_path 指向的文件不存在 —— 所以**只能**给不渲染的测试用。
def fake_song(db, *, title: str = "假歌", duration: float = 20.0,
              fingerprint: str = "fp-song") -> int:
    """插一首目标歌，返回 song_id。"""
    from vidscribe.dance import material_repository as repo

    return repo.upsert_song(db, fingerprint=fingerprint,
                            file_path=f"C:/fake/{title}.wav", file_name=f"{title}.wav",
                            title=title, duration=float(duration), sample_rate=SR)


def fake_video(db, name: str, *, duration: float = 60.0) -> int:
    """插一条 videos 行（dance_materials.source_video_id 有外键，必须真存在）。"""
    from vidscribe.dance import material_repository as repo

    stamp = repo.now()
    cur = db.connect().execute(
        "INSERT OR IGNORE INTO videos (fingerprint, file_path, file_name, duration, "
        "width, height, fps, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1080, 1920, 30.0, 'new', ?, ?)",
        (f"fp-{name}", f"C:/fake/{name}.mp4", f"{name}.mp4", float(duration), stamp, stamp))
    if cur.lastrowid:
        return int(cur.lastrowid)
    row = db.connect().execute("SELECT id FROM videos WHERE fingerprint = ?",
                              (f"fp-{name}",)).fetchone()
    return int(row["id"])


def fake_material(db, song_id: int, video_id: int, segment_index: int, *,
                  person: str = "", slice_duration: float = 2.0, offset: float = 0.0,
                  confidence: float = 0.9, quality: float = 0.8,
                  source_group: str = "") -> int:
    """插一条素材。`target_start` 按位置算，`source_start` 严格走全系统口径。"""
    from vidscribe.dance import MATERIAL_GENERATION_VERSION
    from vidscribe.dance import material_repository as repo

    start = round(segment_index * slice_duration, 6)
    end = round(start + slice_duration, 6)
    return repo.upsert_material(
        db, source_video_id=video_id, alignment_id=None, target_song_id=song_id,
        segment_index=int(segment_index), target_start=start, target_end=end,
        source_start=round(start - offset, 6), source_end=round(end - offset, 6),
        duration=float(slice_duration), generation_version=MATERIAL_GENERATION_VERSION,
        file_path=f"C:/fake/materials/s{song_id}_v{video_id}_p{segment_index}.mp4",
        file_hash=f"h{song_id}-{video_id}-{segment_index}",
        alignment_confidence=float(confidence), person=person,
        source_group=source_group, quality=float(quality))


def fake_alignment(db, song_id: int, video_id: int, *, offset: float = 1.5,
                   confidence: float = 0.82, status: str = "ok") -> int:
    """插一条**真的**对齐记录，走生产用的 `repo.save_alignment`。

    为什么一定要走生产入口而不是手写 INSERT：列名只能有一处真相。
    界面上曾经把 `offset_seconds` 写成 `offset`（`OFFSET` 是 SQL 关键字，
    建表时刻意避开了），而当时的夹具根本没有对齐行，界面那段填表循环一次都没跑过，
    测试全绿、一开界面就抛 IndexError。造行必须走真实路径，测试才盯得住。
    """
    from vidscribe.dance import ALIGNMENT_ALGORITHM_VERSION
    from vidscribe.dance import material_repository as repo
    from vidscribe.dance.types import DanceAlignment

    alignment = DanceAlignment(
        offset=float(offset), confidence=float(confidence), method="hybrid",
        waveform_offset=float(offset), waveform_confidence=float(confidence),
        chroma_offset=float(offset), chroma_confidence=float(confidence) - 0.1,
        window_count=5, max_deviation=0.02, agreement=0.9, status=status,
        algorithm_version=ALIGNMENT_ALGORITHM_VERSION,
        source_duration=60.0, target_duration=20.0, sample_rate=SR)
    return repo.save_alignment(db, source_video_id=video_id, target_song_id=song_id,
                               cache_key=f"ck-{song_id}-{video_id}", alignment=alignment,
                               config_hash="cfg-test")


def fake_version(db, song_id: int, material_ids, *, rendered: bool = True) -> int:
    """插一版**真的**混剪版本 + 对应的使用事件，返回 version_id。

    和 `fake_alignment` 同一个道理：历史面板那张表也得有行才跑得到填表代码。
    """
    from vidscribe.dance import history
    from vidscribe.dance import material_repository as repo
    from vidscribe.dance.types import DanceMontageClip

    clips = []
    for order, material_id in enumerate(material_ids):
        material = repo.get_material(db, int(material_id))
        clips.append(DanceMontageClip(
            order_index=order, segment_index=int(material.segment_index),
            material_id=int(material_id),
            target_start=order * 2.0, target_end=order * 2.0 + 2.0,
            source_start=float(material.source_start), source_end=float(material.source_end),
            file_path=str(material.file_path), person=str(material.person or ""),
            source_video_id=int(material.source_video_id)))
    montage_id = repo.ensure_montage(db, target_song_id=song_id, name="夹具混剪",
                                     slice_duration=2.0)
    version_id = repo.save_version(
        db, montage_id=montage_id, version_index=repo.next_version_index(db, montage_id),
        signature="-".join(str(int(m)) for m in sorted(material_ids)),
        strategy_id=None, recommendation_run_id=None, clips=clips,
        duration=2.0 * len(clips), repeat=None, timeline_json={})
    history.note_montage(db, version_id, montage_id, clips)
    if rendered:
        repo.set_render_status(db, version_id, "rendered",
                              output_path="C:/fake/out/夹具混剪_v01.mp4")
        history.note_render(db, version_id, montage_id, clips, ok=True)
    return version_id


def fake_library(db, *, positions: int = 5, people=("小A", "小B", "小C"),

                 slice_duration: float = 2.0, title: str = "假歌"):
    """造一个"每个人在每个位置都有一条素材"的方阵库。

    返回 `(song_id, {人名: video_id}, {(人名, 位置): material_id})`。
    方阵是有意的：约束测试要能确定"换个人"永远有得换，否则测出来的失败
    分不清是约束写错了还是素材不够。
    """
    song_id = fake_song(db, title=title, duration=positions * slice_duration + 1.0)
    videos = {person: fake_video(db, f"src-{index}")
              for index, person in enumerate(people)}
    materials: dict[tuple[str, int], int] = {}
    for person, video_id in videos.items():
        for pos in range(int(positions)):
            materials[(person, pos)] = fake_material(
                db, song_id, video_id, pos, person=person,
                slice_duration=slice_duration, source_group=person)
    return song_id, videos, materials



