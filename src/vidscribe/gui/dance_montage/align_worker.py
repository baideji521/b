"""「音频对齐 / 卡点测试」的后台工人线程。**只算不写**。

和 `DanceMontageWorker` 分开是有意的：那个是"出片"的九阶段流水线，这个是研发验收台。
两者共用后端（`material_ingest.align_batch`），但职责不能混 ——
测试台一旦顺手把素材切了、把计数加了，用户就再也不敢拿它试东西了。

所以这里的铁律（技术指导第十八节）：

    点「开始对齐」= 解码 + 互相关 + 验证，然后把结果摆出来
    **不**登记素材、**不**切片、**不**动 use_count / output_count、**不**写历史

唯一会落库的是**目标歌本身**（`register_song`）：位置尺子是目标歌的属性，
不分析它就没有"第几格"可谈；而它按指纹幂等，同一首歌反复选也只有一行。
对齐结果要不要留下，由用户点「保存对齐结果」决定（走 `ingest.persist_alignment`）。

`av` / `numpy` 这些重模块只在 `run()` 里 import，库连接也在本线程自己开 ——
和 `DanceMontageWorker` 同一套规矩，原因见那边的注释。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal

from ...logging_setup import get_logger

logger = get_logger("dance.gui.align")

#: 波形缩略图的桶数。600 个点铺在一千来像素宽的控件上，肉眼已经看不出台阶
ENVELOPE_BUCKETS = 600


class DanceAlignWorker(QThread):
    """把 `[源视频…]` 对到一首目标歌上，结果**不落库**地交回界面。

    `job` 的键：

        song      目标歌文件路径，或库里的歌 id
        sources   源舞蹈视频路径列表（一个就是单视频模式，多个就是批量）
        workers   并发数（交给 `align_batch`，它自己封顶 6 个）
        force     True = 忽略对齐缓存重算
        envelope  True = 顺带算波形缩略图（只在单视频时有意义）
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(bool, str, object)

    def __init__(self, cfg, job: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.job = dict(job)
        self._stop = False

    def stop(self) -> None:
        """协作式停止：当前这个源算完就不再往下走。"""
        self._stop = True
        self.log.emit("[停止] 已收到停止请求，当前这一步做完就停")

    def _say(self, line: str) -> None:
        self.log.emit(line)

    # ---------------------------------------------------------------- 主体
    def run(self) -> None:
        payload: dict[str, Any] = {"results": [], "song_id": 0}
        try:
            from ...db import open_db  # noqa: PLC0415
            from ...dance import material_ingest as ingest  # noqa: PLC0415

            sources = [Path(p) for p in (self.job.get("sources") or [])]
            if not sources:
                return self._finish(False, "还没选源舞蹈视频", payload)

            db = open_db(self.cfg)
            try:
                song = self._song(db, ingest)
                payload.update({"song_id": song.song_id, "song_path": str(song.path),
                                "song_duration": float(song.duration),
                                "bpm": float(song.bpm),
                                "fingerprint": str(song.fingerprint or ""),
                                "sample_rate": int(song.sample_rate)})
                self._say(f"[目标歌] {song.path.name}｜{song.duration:.2f}s"
                          f"｜BPM {song.bpm:.1f}")
                if self._stop:
                    return self._finish(False, "已停止（目标歌分析完）", payload)

                outcomes = ingest.align_batch(
                    db, song, sources,
                    workers=int(self.job.get("workers")
                                or self.cfg.dance["align_workers"]),
                    force=bool(self.job.get("force")),
                    persist=False,                     # ← 测试台绝不落库
                    on_log=self._say,
                    on_progress=lambda a, b, t: self.progress.emit(a, b, t))
                payload["results"] = [self._row(o) for o in outcomes]

                if self.job.get("envelope") and len(sources) == 1:
                    self._envelopes(payload, song, sources[0])

                good = [o for o in outcomes if o.ok]
                return self._finish(bool(good),
                                    f"对齐完成 {len(good)}／{len(outcomes)}",
                                    payload)
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 - 后台线程抛出去就没人接了
            import traceback  # noqa: PLC0415

            logger.error("对齐测试失败：%s", exc)
            logger.error(traceback.format_exc())
            self._say(f"[错误] {type(exc).__name__}: {exc}")
            self._finish(False, f"失败：{exc}", payload)

    # ---------------------------------------------------------------- 辅助
    def _finish(self, ok: bool, message: str, payload: dict[str, Any]) -> None:
        self.done.emit(bool(ok), str(message), payload)

    def _song(self, db, ingest):
        """目标歌：给 id 读库，给路径就登记 + 分析（按指纹幂等，不会造重复行）。

        库里读出来的那条**也要现解一次 PCM**：`align_batch` 要靠它做"目标歌只解一次"，
        而画波形也要它。少了这一步，两边都得各自再解一遍。
        """
        from ...dance import material_repository as repo  # noqa: PLC0415
        from ...dance.audio_fingerprint import extract_analysis_audio  # noqa: PLC0415

        text = str(self.job.get("song") or "").strip()
        if not text:
            raise ValueError("没有选目标歌")
        if text.isdigit():
            row = repo.get_song(db, int(text))
            if row is None:
                raise ValueError(f"库里没有目标歌 #{text}")
            path = Path(row["file_path"])
            if not path.is_file():
                raise ValueError(f"目标歌 #{text} 的文件不在盘上了：{path}")
            sample_rate = int(row["sample_rate"] or 0) or None
            song = ingest.TargetSong(
                song_id=int(row["id"]), path=path,
                fingerprint=str(row["fingerprint"] or ""),
                duration=float(row["duration"] or 0.0), bpm=float(row["bpm"] or 0.0),
                sample_rate=sample_rate or ingest.DEFAULT_SR)
            song.pcm = extract_analysis_audio(path, sample_rate=song.sample_rate)
            return song
        path = Path(text)
        if not path.is_file():
            raise ValueError(f"目标歌文件不在盘上：{text}")
        return ingest.register_song(db, path, on_log=self._say)

    @staticmethod
    def _row(outcome) -> dict[str, Any]:
        """一个源视频的结果。`alignment` 直接给对象 —— 同进程内传引用，不做序列化。"""
        return {"path": str(outcome.source_path), "name": outcome.source_path.name,
                "alignment": outcome.alignment, "error": outcome.error,
                "cached": bool(outcome.cached)}

    def _envelopes(self, payload: dict[str, Any], song, source: Path) -> None:
        """算两条波形缩略图，供界面画"两边是怎么错开的"。

        只在单视频模式下算：批量时几十条包络画不下，也没人看。
        算失败**不影响主结果** —— 图是辅助，数字才是结论。
        """
        from ...dance import dsp  # noqa: PLC0415
        from ...dance.audio_fingerprint import extract_analysis_audio  # noqa: PLC0415

        try:
            target_pcm = song.pcm
            source_pcm = extract_analysis_audio(source, sample_rate=song.sample_rate)
            payload["target_envelope"] = [float(v) for v in
                                          dsp.envelope(target_pcm, ENVELOPE_BUCKETS)]
            payload["source_envelope"] = [float(v) for v in
                                          dsp.envelope(source_pcm, ENVELOPE_BUCKETS)]
            payload["source_duration"] = round(source_pcm.size / float(song.sample_rate), 3)
            payload["envelope_of"] = str(source)
            self._say(f"[波形] 目标 {len(payload['target_envelope'])} 点"
                      f"｜源 {len(payload['source_envelope'])} 点")
        except Exception as exc:  # noqa: BLE001 - 画不出图不该让整次测试失败
            logger.debug("波形缩略图算不出来：%s", exc)
            self._say(f"[波形] 缩略图算不出来（{type(exc).__name__}），只看数字也一样能验收")


class ClipJobWorker(QThread):
    """后台的两件杂活：解一条预览音轨、导出一个测试片段。

    都放在线程里是同一个理由：两件事都要 `import av` 并且真的在解码/编码，
    主线程干这个会把界面冻住好几秒。

    `job["kind"]`：
        `"audio"`  把源视频的音轨解成 wav（`winsound` 只认 PCM wav），
                   落在 cache 目录，同一个视频只解一次
        其它       导出 `start → end` 这一段成一个独立文件（默认）

    刻意**不**做"顺手把整格素材都导出来"这种事：这里只是给人工细看用的一次性副本，
    正式素材一律由 `material_slice.render_plan` 在切片流程里产出。
    """

    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, cfg, job: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.job = dict(job)

    def run(self) -> None:
        try:
            if str(self.job.get("kind") or "") == "audio":
                self.done.emit(True, self._extract_audio())
                return
            self.done.emit(True, self._export_clip())
        except Exception as exc:  # noqa: BLE001 - 后台线程抛出去就没人接了
            import traceback  # noqa: PLC0415

            logger.error("后台杂活失败：%s", exc)
            logger.error(traceback.format_exc())
            self.done.emit(False, f"{type(exc).__name__}: {exc}")

    def _extract_audio(self) -> str:
        """源视频音轨 → wav。复用主项目那套（`audio.wav_path` + `extract_wav`）。"""
        from ...audio import extract_wav, wav_path  # noqa: PLC0415

        source = Path(str(self.job["source"]))
        cache = Path(str(self.job.get("cache_dir") or "."))
        cache.mkdir(parents=True, exist_ok=True)
        target = wav_path(cache, source)
        if target.is_file():
            self.log.emit(f"[声音] 复用缓存音轨 {target.name}")
            return str(target)
        self.log.emit(f"[声音] 解音轨 {source.name} → {target.name}")
        made = extract_wav(source, target)
        if made is None or not Path(made).is_file():
            raise RuntimeError("这个视频里没有能用的音轨")
        return str(made)

    def _export_clip(self) -> str:
        from ...dance import material_slice, media_backend  # noqa: PLC0415
        from ...dance.types import SliceSpec  # noqa: PLC0415

        start = float(self.job["start"])
        end = float(self.job["end"])
        target = Path(str(self.job["target"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        spec = SliceSpec(segment_index=0, target_start=0.0, target_end=end - start,
                         source_start=start, source_end=end)
        canvas = media_backend.Canvas(
            width=int(self.cfg.dance["canvas_width"]),
            height=int(self.cfg.dance["canvas_height"]),
            fps=float(self.cfg.dance["canvas_fps"]))
        self.log.emit(f"[导出] {start:.3f}s → {end:.3f}s → {target.name}")
        material_slice.render_material(
            str(self.job["source"]), spec, target, canvas=canvas,
            backend=media_backend.resolve(str(self.cfg.dance["media_backend"])),
            on_log=self.log.emit)
        return str(target)



__all__ = ["ENVELOPE_BUCKETS", "DanceAlignWorker", "ClipJobWorker"]
