"""AI_卡点舞 的后台工人线程。**和 `AnalyzeWorker` 完全隔离**，一行代码都不共用。

技术指导第二十节的硬要求：界面上所有重活都在这里跑，主线程只画界面。
另外两条也在这里落实：

  - **库连接在本线程自己开**（`Database` 是每线程一条连接），绝不把主线程的连接带进来
  - `av` / `cv2` 这些重模块只在 `run()` 里 import：主线程碰 cv2 会改写
    `QT_QPA_PLATFORM_PLUGIN_PATH`，QApplication 会直接崩（0xC0000409）

九个阶段的名字是一期第四十节钉死的，界面上照这个显示，不许自己发明措辞。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal

from ...logging_setup import get_logger

logger = get_logger("dance.gui.worker")

#: 九个阶段。顺序即进度顺序，界面按下标算百分比
STAGES = ("扫描素材", "音频提取", "音乐对齐", "生成切片", "建立素材库",
          "准备混剪", "渲染", "封装", "完成")


class DanceMontageWorker(QThread):
    """跑一次完整的卡点舞流程：对齐 → 切片 → 素材库 → 推荐 → 混剪 → 渲染。

    `job` 里的键都是可选的，缺了就用配置里的默认值：

        song            目标歌文件路径，或库里的歌 id
        sources         源视频列表（目录或文件）
        slice_duration  每格时长
        workers         对齐并发数
        do_align        只对齐不切片（界面上"只对齐"按钮）
        do_slice        对齐完接着切片
        do_remix        切完接着出片
        versions        出几个版本
        seed            随机种子（0 = 现取一个并落库）
        strategy_id     策略 id
        preset          筛选方案名
        recommend       False = 关掉智能推荐，走 manual
        manual          {位置: 素材id}
        render          False = 只出计划不渲染
    """

    log = pyqtSignal(str)
    stage = pyqtSignal(str, int, int)          # 阶段名, 第几个(0起), 总数
    progress = pyqtSignal(int, int, str)       # 当前, 总数, 说明
    done = pyqtSignal(bool, str, object)       # 成功?, 一句话结论, 结果字典

    def __init__(self, cfg, job: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.job = dict(job)
        self._stop = False

    # ---------------------------------------------------------------- 控制
    def stop(self) -> None:
        """请求停止。**协作式**：跑到下一个阶段边界才真停，不会把文件写坏。"""
        self._stop = True
        self.log.emit("[停止] 已收到停止请求，会在当前这一步做完之后停下")

    def _stopped(self) -> bool:
        return bool(self._stop)

    def _say(self, line: str) -> None:
        self.log.emit(line)

    def _stage(self, name: str) -> None:
        self.stage.emit(name, STAGES.index(name), len(STAGES))

    # ---------------------------------------------------------------- 主体
    def run(self) -> None:                     # noqa: C901 - 九个阶段串一条线，拆开更难读
        result: dict[str, Any] = {"materials": 0, "versions": [], "outputs": []}
        try:
            from ...db import open_db
            from ...dance import material_ingest as ingest
            from ...dance import media_backend, montage_render
            from ...dance import material_selection as selection
            from ...dance import strategy as strategy_mod

            db = open_db(self.cfg)
            try:
                self._stage("扫描素材")
                sources = self._sources()
                song_arg = self.job.get("song") or ""
                if not song_arg:
                    raise ValueError("没有选目标歌")
                self._say(f"[扫描] 源视频 {len(sources)} 个｜目标歌 {song_arg}")

                self._stage("音频提取")
                song = self._song(db, ingest, song_arg)
                result["song_id"] = song.song_id
                self._say(f"[目标歌] {song.path.name}｜{song.duration:.2f}s｜BPM {song.bpm:.1f}")
                if self._stopped():
                    return self._finish(False, "已停止（分析完目标歌）", result)

                slice_duration = float(self.job.get("slice_duration")
                                       or self.cfg.dance["slice_duration"])
                # 画布跟素材走（默认）：3:4 的源出 3:4 的成片，不裁边。
                # 源视频尺寸不一致时按多数派归一，见 media_backend.canvas_for
                canvas = media_backend.resolve_canvas(self.cfg, sources)
                self._say(f"[画布] {canvas.width}×{canvas.height} @ {canvas.fps:g}fps")
                backend = media_backend.resolve(str(self.cfg.dance["media_backend"]))
                strategy_mod.ensure_presets(db)

                outcomes = []
                if sources:
                    self._stage("音乐对齐")
                    outcomes = ingest.align_batch(
                        db, song, sources,
                        workers=int(self.job.get("workers")
                                    or self.cfg.dance["align_workers"]),
                        force=bool(self.job.get("force")),
                        on_log=self._say,
                        on_progress=lambda a, b, t: self.progress.emit(a, b, t))
                    ok = [o for o in outcomes if o.ok]
                    result["aligned"] = len(ok)
                    result["align_failed"] = len(outcomes) - len(ok)
                    self._say(f"[对齐] 成功 {len(ok)}／{len(outcomes)}")
                if self._stopped():
                    return self._finish(True, "已停止（对齐完成，素材未切）", result)

                if self.job.get("do_slice", True) and outcomes:
                    self._stage("生成切片")
                    for index, outcome in enumerate(outcomes, 1):
                        if self._stopped():
                            self._say("[停止] 剩下的源不再切片")
                            break
                        if not outcome.ok:
                            self._say(f"[切片] 跳过 {outcome.source_path.name}："
                                      f"对齐失败（{outcome.error}）")
                            continue
                        self.progress.emit(index, len(outcomes), outcome.source_path.name)
                        sliced = ingest.slice_and_register(
                            db, song, outcome, slice_duration=slice_duration,
                            material_dir=self.cfg.dance_path("material_dir"),
                            canvas=canvas, backend=backend,
                            person=str(self.job.get("person") or ""),
                            min_confidence=float(self.cfg.dance["min_confidence"]),
                            head_room=float(self.job.get(
                                "head_room", self.cfg.dance.get("slice_head_room", 0.0))),
                            tail_room=float(self.job.get(
                                "tail_room", self.cfg.dance.get("slice_tail_room", 0.0))),
                            on_log=self._say)
                        result["materials"] += len(sliced.material_ids)
                    self._stage("建立素材库")
                    self._say(f"[素材库] 本次新增 {result['materials']} 条素材")

                if not self.job.get("do_remix", True):
                    return self._finish(True, f"素材已入库（{result['materials']} 条）", result)
                if self._stopped():
                    return self._finish(True, "已停止（素材已入库，未出片）", result)

                self._stage("准备混剪")
                spec = (selection.preset_spec(str(self.job["preset"]))
                        if self.job.get("preset") else None)
                self._stage("渲染")
                made = montage_render.remix(
                    db, song.song_id,
                    out_dir=(self.job.get("out_dir") or self.cfg.dance_path("output_dir")),
                    slice_duration=slice_duration,
                    versions=int(self.job.get("versions") or 0),
                    strategy_id=int(self.job.get("strategy_id") or 0),
                    seed=int(self.job.get("seed") or 0),
                    spec=spec, canvas=canvas, backend=backend,
                    name=str(self.job.get("name") or ""),
                    recommend_enabled=bool(self.job.get("recommend", True)),
                    manual=self.job.get("manual"),
                    render_video=bool(self.job.get("render", True)),
                    pool_size=int(self.job.get("pool_size")
                                  or self.cfg.dance["candidate_pool_size"]),
                    on_log=self._say,

                    on_progress=lambda a, b, t: self.progress.emit(a, b, t))
                self._stage("封装")
                for version_id, render in made:
                    result["versions"].append(version_id)
                    if render.output:
                        result["outputs"].append(render.output)
                    for line in montage_render.describe(render):
                        self._say(line)
                self._stage("完成")
                good = [r for _v, r in made if r.ok]
                if not self.job.get("render", True):
                    return self._finish(True, f"已出 {len(made)} 版编辑计划（未渲染）", result)
                return self._finish(bool(good),
                                    f"出片 {len(good)}／{len(made)} 版" if made else "一版都没出来",
                                    result)
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 - 后台线程里抛出去就没人接了
            import traceback

            logger.error("卡点舞流程失败：%s", exc)
            logger.error(traceback.format_exc())
            self._say(f"[错误] {type(exc).__name__}: {exc}")
            self._finish(False, f"失败：{exc}", result)

    # ---------------------------------------------------------------- 辅助
    def _finish(self, ok: bool, message: str, result: dict[str, Any]) -> None:
        self.done.emit(bool(ok), str(message), result)

    def _song(self, db, ingest, text: str):
        """目标歌：给 id 就读库，给路径就登记 + 分析（命中指纹自动复用）。"""
        from ...dance import material_repository as repo

        if str(text).isdigit():
            row = repo.get_song(db, int(text))
            if row is None:
                raise ValueError(f"库里没有目标歌 #{text}")
            return ingest.TargetSong(
                song_id=int(row["id"]), path=Path(row["file_path"]),
                fingerprint=str(row["fingerprint"] or ""),
                duration=float(row["duration"] or 0.0), bpm=float(row["bpm"] or 0.0),
                sample_rate=int(row["sample_rate"] or 0))
        path = Path(text)
        if not path.is_file():
            raise ValueError(f"目标歌文件不在盘上：{text}")
        return ingest.register_song(db, path, on_log=self._say)

    def _sources(self) -> list[Path]:
        """源视频：目录就递归扫，文件就直接用。"""
        out: list[Path] = []
        raw = self.job.get("sources") or []
        if isinstance(raw, (str, Path)):
            raw = [raw]
        for text in raw:
            path = Path(text)
            if path.is_dir():
                out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in
                                  (".mp4", ".mov", ".mkv", ".avi", ".flv")))
            elif path.is_file():
                out.append(path)
            else:
                self._say(f"[扫描] 跳过不存在的源：{path}")
        return out


__all__ = ["STAGES", "DanceMontageWorker"]
