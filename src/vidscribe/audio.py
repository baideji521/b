"""把视频音轨解码成 wav，只给 GUI 预览播放用。

ASR 不需要这一步（faster-whisper 直接读视频文件），所以这是 GUI 专属的：
这台机器上 Qt 的 WMF 后端解不了 H.264 视频（见 gui/player.py 的说明），
但能正常播放 PCM WAV，所以"预览带声音"的做法是先把音轨解出来再交给 Qt。

用 PyAV（faster-whisper 的依赖）而不是外部 ffmpeg.exe，免去额外安装。
"""

from __future__ import annotations

from pathlib import Path

from .cache import video_dir_in
from .logging_setup import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 44100
_WAV_HEADER_BYTES = 44


def wav_path(cache_root: Path, video: Path) -> Path:
    return video_dir_in(cache_root, video) / "preview_audio.wav"


def _resample(resampler, frame):
    """PyAV 各版本 resample() 有的返回单帧、有的返回列表，统一成列表。"""
    out = resampler.resample(frame)
    if out is None:
        return []
    return out if isinstance(out, list) else [out]


def extract_wav(video: str | Path, target: str | Path, force: bool = False) -> Path | None:
    """解码音轨为 16bit 立体声 wav。没有音轨或解码失败返回 None（调用方据此禁用声音）。"""
    video = Path(video)
    target = Path(target)
    if not force and target.is_file() and target.stat().st_size > _WAV_HEADER_BYTES:
        try:
            if target.stat().st_mtime >= video.stat().st_mtime:
                return target
        except OSError:
            pass

    try:
        import av  # noqa: PLC0415
    except Exception as exc:  # PyAV 缺失时不让 GUI 崩，只是没有声音
        logger.warning("PyAV 不可用，无法提取预览音轨：%s", exc)
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".part.wav")
    try:
        with av.open(str(video)) as src:
            if not src.streams.audio:
                logger.info("%s 没有音轨，预览无声音", video.name)
                return None
            in_stream = src.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="stereo", rate=SAMPLE_RATE)
            with av.open(str(tmp), mode="w", format="wav") as dst:
                out_stream = dst.add_stream("pcm_s16le", rate=SAMPLE_RATE, layout="stereo")
                for frame in src.decode(in_stream):
                    for chunk in _resample(resampler, frame):
                        chunk.pts = None
                        for packet in out_stream.encode(chunk):
                            dst.mux(packet)
                for chunk in _resample(resampler, None):  # flush 重采样器
                    chunk.pts = None
                    for packet in out_stream.encode(chunk):
                        dst.mux(packet)
                for packet in out_stream.encode(None):  # flush 编码器
                    dst.mux(packet)
        tmp.replace(target)
        logger.info("预览音轨已生成：%s", target)
        return target
    except Exception as exc:
        logger.warning("提取预览音轨失败：%s: %s", type(exc).__name__, str(exc)[:200])
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def slice_wav(source: str | Path, target: str | Path, start_seconds: float) -> Path | None:
    """从 wav 的某一秒开始切出剩余部分，写成新的 wav。

    播放器只能"从头播"（winsound 没有定位接口），所以每次 play/seek 都先切一段
    再播。49s 立体声 16bit 只有 8MB 级别，切一次是毫秒量级，比引入新依赖划算。
    """
    import wave  # noqa: PLC0415

    source, target = Path(source), Path(target)
    if not source.is_file():
        return None
    try:
        with wave.open(str(source), "rb") as src:
            rate = src.getframerate()
            total = src.getnframes()
            offset = max(0, min(int(round(max(start_seconds, 0.0) * rate)), total))
            src.setpos(offset)
            frames = src.readframes(total - offset)
            params = src.getparams()
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as dst:
            dst.setnchannels(params.nchannels)
            dst.setsampwidth(params.sampwidth)
            dst.setframerate(params.framerate)
            dst.writeframes(frames)
        return target
    except Exception as exc:
        logger.warning("切分预览音轨失败：%s: %s", type(exc).__name__, str(exc)[:200])
        return None

