"""说话人分离：给每句标上"这是谁说的"。

**主力路径在 `diarization.py`**（pyannote 分段 + onnx 声纹 + 层次聚类），人数和归属都由它定。
下面这套"按句取声纹 + 谱聚类"只是兜底（缺 sherpa-onnx 或下不到 onnx 模型时用）。

**为什么按句聚类不够**：ASR 的句子边界不是说话人边界。实测那条已知 2 人的视频，一句
6~11 秒里两个人都在说，句级声纹是混音，聚类怎么调都判成 1 人。相关实测数据：

- 声纹本身很能分：拿 cam++ 自带参考音频，同一人 cos=0.694/0.794，不同人 -0.084/-0.219
- 把 k=2 直接告诉谱聚类，窗级准确率 100%；但人数得自己定，funasr 的 `ClusterBackend`
  在几十个窗上会自估成 5 人（`merge_thr=0.78` 比同一人的真实相似度还高，过分裂合不回去）
- 自己用余弦轮廓系数挑 k 在参考音频上成立（k=2 得 +0.437/+0.610，单人视频 0.243），
  但真实对话素材只有 0.109~0.32，跟单人素材分不开 —— 所以才换成 diarization

**声纹取样：按句为主，滑窗兜底。** 句子少于 `min_sentences`（默认 12，十几秒的短片）时退回
滑窗：funasr 的 `ClusterBackend.forward` 里有 `if X.shape[0] < 20: return zeros`
（少于 20 段全判成同一个人），短素材按句切一定撞这条线。
"""


from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_setup import get_logger
from ..video_io import VideoInfo
from .diarization import Diarizer, assign_sentences, drop_tiny
from .emotion import SAMPLE_RATE, _decode_mono16k
from .sentences import split_on_turns


logger = get_logger(__name__)


def _windows(segments: list[dict[str, Any]], window: float, hop: float,
             audio_seconds: float) -> list[tuple[float, float]]:
    """只在有语音的区间上滑窗，静音段不取——静音的声纹是噪声，会把聚类带歪。"""
    spans: list[tuple[float, float]] = []
    for seg in segments:
        start = max(0.0, float(seg.get("start") or 0.0))
        end = min(audio_seconds, float(seg.get("end") or 0.0))
        if end - start <= 0:
            continue
        if spans and start - spans[-1][1] <= hop:   # 紧挨着的句子并成一段连续语音
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    out: list[tuple[float, float]] = []
    for start, end in spans:
        if end - start < window:
            # 比一个窗还短的语音段：整段当一个窗，别丢掉（短句正是需要判人的地方）
            out.append((start, end))
            continue
        cursor = start
        while cursor + window <= end + 1e-6:
            out.append((cursor, cursor + window))
            cursor += hop
    return out


MIN_CLIP_SECONDS = 0.35   # 比这还短的音频，声纹模型给不出可靠向量


def _sentence_spans(segments: list[dict[str, Any]], audio_seconds: float,
                    min_seconds: float = MIN_CLIP_SECONDS) -> list[tuple[int, float, float]]:
    """按句取样：返回 (句子下标, 起, 止)。太短的句子不取声纹，后面按时间就近继承。"""
    out: list[tuple[int, float, float]] = []
    for i, seg in enumerate(segments):
        start = max(0.0, float(seg.get("start") or 0.0))
        end = min(audio_seconds, float(seg.get("end") or 0.0))
        if end - start >= min_seconds:
            out.append((i, start, end))
    return out


def _fold_small(labels: np.ndarray, unit: np.ndarray, min_share: float) -> np.ndarray:
    """占比不到 min_share 的簇折回最近的大簇。背景音乐、笑声这种零星窗口会自己抱成小团。"""
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        if ids.size <= 1:
            return labels
        share = counts / counts.sum()
        if share.min() >= min_share:
            return labels
        drop = ids[int(np.argmin(share))]
        keep = ids[ids != drop]
        centers = np.stack([unit[labels == i].mean(0) for i in keep])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        for idx in np.where(labels == drop)[0]:
            labels[idx] = keep[int(np.argmax(centers @ unit[idx]))]


def cluster_embeddings(embeddings: np.ndarray, max_speakers: int = 6,
                       min_silhouette: float = 0.25,
                       min_share: float = 0.05) -> tuple[np.ndarray, float]:
    """声纹矩阵 -> (每个窗的说话人标签, 轮廓系数)。人数自己定，见模块开头的实测数据。"""
    from funasr.models.campplus.cluster_backend import SpectralCluster  # noqa: PLC0415
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    total = embeddings.shape[0]
    if total < 8:      # 窗口太少，谈不上聚类
        return np.zeros(total, dtype=int), -1.0

    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    best: tuple[float, np.ndarray] | None = None
    for k in range(2, max(3, min(int(max_speakers), total // 4)) + 1):
        spectral = SpectralCluster(min_num_spks=1, max_num_spks=15, pval=0.022)
        labels = np.asarray(spectral(embeddings.copy(), oracle_num=k), dtype=int)
        if np.unique(labels).size < 2:
            continue
        score = float(silhouette_score(unit, labels, metric="cosine"))
        if best is None or score > best[0]:
            best = (score, labels)

    if best is None or best[0] < min_silhouette:
        # 分不出来就老实说一个人。单人视频实测所有 k 的轮廓系数都只有 +0.10 上下，
        # 双人素材是 +0.35~+0.50，界线在 0.25 附近很干净。
        return np.zeros(total, dtype=int), (best[0] if best else -1.0)
    return _fold_small(best[1].copy(), unit, float(min_share)), best[0]


RAW_CAMPPLUS: dict[str, dict[str, Any]] = {
    # 纯英文声纹（3D-Speaker VoxCeleb 权重）。funasr 的 AutoModel 认不了这个仓库
    # （configuration.json 是 modelscope pipeline 格式，报 "is not registered"），
    # 但权重本身就是 funasr 里已注册的 CAMPPlus 结构，自己建模型 + load_state_dict 就能用，
    # 比装 modelscope pipeline 那一串依赖（datasets/addict…）干净得多。
    # 拿仓库自带示例音频验证过：同一人 +0.794，不同人 -0.089 / -0.219
    # （中文那份 cam++ 是 +0.694 / -0.084，英文素材上这份区分度更好）。
    "iic/speech_campplus_sv_en_voxceleb_16k": {
        "weight": "campplus_voxceleb.bin", "embedding_size": 512, "feat_dim": 80,
    },
}


def _write_speakers(segments: list[dict[str, Any]],
                    picks: list[tuple[int, float]]) -> int:
    """把 (簇号, 置信度) 写回句子，簇号按首次开口时间重编成 1、2、3…

    重跑同一条视频编号稳定，「说话人1」永远是先开口的那个。返回人数。
    """
    order: list[int] = []
    for cluster_id, _ in picks:
        if cluster_id not in order:
            order.append(cluster_id)
    rename = {cluster_id: i + 1 for i, cluster_id in enumerate(order)}
    for seg, (cluster_id, confidence) in zip(segments, picks):
        seg["speaker"] = rename[cluster_id]
        seg["speaker_confidence"] = confidence
    return len(rename)


class SpeakerTagger:

    """按句标注说话人。模型只加载一次，多视频复用。"""

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model = None
        # 声纹模型只有两个选择（界面「声纹」下拉，走 --speaker-model 传进来）：
        # - 中文：cam++ zh-cn（192 维）—— 默认。在官方标注素材上全中（4 人判 4、
        #   三条 2 人判 2），英文素材上也比下面那份准
        # - 英文：3D-Speaker 的 VoxCeleb cam++（512 维），实测更差，留着备用
        self.model_id: str = str(self.cfg.get("model_id")
                                 or "iic/speech_campplus_sv_zh-cn_16k-common")
        self.device: str = "cpu"
        self.load_seconds = 0.0
        self.model_path: str | None = None
        # 主力路径：pyannote 分段 + onnx 声纹（见 diarization.py）。
        # 句子边界不等于说话人边界，所以人数和归属都由它定；它跑不起来（缺网、缺 sherpa-onnx）
        # 才退回下面那套"按句取声纹 + 谱聚类"。
        self.diarizer = Diarizer(self.cfg.get("diarization", {}), self.model_id, model_dir)


    @property
    def raw_spec(self) -> dict[str, Any] | None:
        """当前模型是否要走"裸权重"加载（见 RAW_CAMPPLUS）。"""
        return RAW_CAMPPLUS.get(self.model_id)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def load(self) -> None:
        if self.model is not None:
            return
        if self.raw_spec is not None:
            self._load_raw(self.raw_spec)
            return
        import torch  # noqa: PLC0415
        from funasr import AutoModel  # noqa: PLC0415

        want = self.cfg.get("device", "auto")
        if want == "auto":
            want = "cuda:0" if torch.cuda.is_available() else "cpu"

        target = self.model_id
        if self.model_dir:
            from ..mirrors import resolve_model  # noqa: PLC0415

            # cam++ 和 emotion2vec 一样是权重 + config.yaml，走同一套 kind
            target = resolve_model(self.model_id, Path(self.model_dir), self.mirrors,
                                   kind="emotion")

        errors: list[str] = []
        for device in ([want, "cpu"] if want != "cpu" else ["cpu"]):
            started = time.perf_counter()
            try:
                logger.info("加载声纹模型 %s (device=%s)", target, device)
                self.model = AutoModel(model=target, device=device, disable_update=True,
                                       disable_log=True, disable_pbar=True, hub="ms")
            except Exception as exc:  # noqa: BLE001 - 换设备再试，全失败才放弃
                errors.append(f"{device}: {str(exc)[:160]}")
                logger.warning("声纹模型加载失败 %s@%s：%s", self.model_id, device, str(exc)[:160])
                continue
            self.device, self.model_path = device, target
            self.load_seconds = round(time.perf_counter() - started, 2)
            logger.info("声纹模型就绪：%s / %s，耗时 %.1fs", self.model_id, device,
                        self.load_seconds)
            return
        raise RuntimeError("声纹模型加载失败: " + "; ".join(errors))

    def _load_raw(self, spec: dict[str, Any]) -> None:
        """裸权重加载：自己建 CAMPPlus + load_state_dict，特征也自己算（kaldi fbank + CMN）。"""
        import torch  # noqa: PLC0415
        from funasr.models.campplus.model import CAMPPlus  # noqa: PLC0415

        want = self.cfg.get("device", "auto")
        if want == "auto":
            want = "cuda:0" if torch.cuda.is_available() else "cpu"

        target = self.model_id
        if self.model_dir:
            from ..mirrors import resolve_model  # noqa: PLC0415

            target = resolve_model(self.model_id, Path(self.model_dir), self.mirrors,
                                   kind="speaker")
        weight = Path(target) / str(spec["weight"])
        if not weight.is_file():
            # resolve 失败会返回原始 repo id，这时让 modelscope 自己下到它的默认缓存
            from modelscope import snapshot_download  # noqa: PLC0415

            target = snapshot_download(self.model_id)
            weight = Path(target) / str(spec["weight"])

        started = time.perf_counter()
        model = CAMPPlus(feat_dim=int(spec.get("feat_dim", 80)),
                         embedding_size=int(spec["embedding_size"]))
        state = torch.load(str(weight), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        errors: list[str] = []
        for device in ([want, "cpu"] if want != "cpu" else ["cpu"]):
            try:
                self.model = model.to(device)
            except Exception as exc:  # noqa: BLE001 - 显存不够就退 CPU
                errors.append(f"{device}: {str(exc)[:160]}")
                continue
            self.device, self.model_path = device, str(weight)
            self.load_seconds = round(time.perf_counter() - started, 2)
            logger.info("声纹模型就绪（裸权重）：%s / %s，耗时 %.1fs", self.model_id, device,
                        self.load_seconds)
            return
        raise RuntimeError("声纹模型加载失败: " + "; ".join(errors))


    def unload(self) -> None:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self.diarizer.unload()
        if self.model is None:
            return
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("已释放声纹模型显存")

    def _embed(self, clips: list[np.ndarray]) -> np.ndarray:
        """一批窗口 -> 声纹矩阵 (N, dim)。逐条跑：批量接口对变长输入会补零，短窗容易脏。"""
        if self.raw_spec is not None:
            return self._embed_raw(clips)
        rows: list[np.ndarray] = []
        for clip in clips:
            out = self.model.generate(input=clip, fs=SAMPLE_RATE)
            emb = out[0]["spk_embedding"] if out else None
            if emb is None:
                raise RuntimeError("声纹模型没有返回 spk_embedding")
            arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
            rows.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        return np.stack(rows) if rows else np.zeros((0, 192), dtype=np.float32)

    def _embed_raw(self, clips: list[np.ndarray]) -> np.ndarray:
        """裸权重路径的特征+前向：kaldi fbank(80) + 逐句均值归一（3D-Speaker 的做法）。"""
        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415

        spec = self.raw_spec or {}
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for clip in clips:
                wav = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
                # fbank 期望 16bit 量级的波形，解码出来的是 [-1,1]，这里放大回去
                feat = torchaudio.compliance.kaldi.fbank(
                    wav.unsqueeze(0) * (1 << 15),
                    num_mel_bins=int(spec.get("feat_dim", 80)),
                    sample_frequency=SAMPLE_RATE, dither=0.0,
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                emb = self.model(feat.unsqueeze(0).to(self.device))
                rows.append(emb.squeeze().float().cpu().numpy())
        if not rows:
            return np.zeros((0, int(spec.get("embedding_size", 512))), dtype=np.float32)
        return np.stack(rows)

    def annotate(self, info: VideoInfo, segments: list[dict[str, Any]],
                 language: str | None = None) -> dict[str, Any]:
        """给每句补 `speaker` / `speaker_confidence`，返回统计（写进 speech_events.json 的 meta）。

        `language` 只是记在 meta 里备查（whisper 判出的音频语言），不参与挑模型——
        用哪个声纹模型由界面上的「声纹」下拉决定。

        `speaker` 从 1 开始编号，按首次出现的时间排序——这样"说话人1"总是先开口的那个，
        重跑同一条视频编号稳定。

        `speaker_confidence` 的含义随取样方式不同：
        - 按句取样：该句声纹到自己簇心的余弦相似度，减去到最近的其他簇心的相似度（余弦间隔）。
          越大越有把握；太短没取到声纹、靠邻句继承的句子记 0.0，表示"没测过"。
        - 滑窗兜底：该句内窗口投票的占比。
        """
        if not segments:
            return {"available": False, "reason": "no_speech_segments"}
        if not info.has_audio:
            return {"available": False, "reason": "no_audio_stream"}

        started = time.perf_counter()
        audio = _decode_mono16k(Path(info.path))
        if audio.size == 0:
            return {"available": False, "reason": "decode_failed"}
        audio_seconds = audio.size / SAMPLE_RATE

        diarized = self._annotate_by_diarization(segments, audio, language, started)
        if diarized is not None:
            return diarized

        window = float(self.cfg.get("window_seconds", 1.0))

        hop = float(self.cfg.get("hop_seconds", 0.25))
        min_sentences = int(self.cfg.get("min_sentences", 12))

        sentences = _sentence_spans(segments, audio_seconds)
        if len(sentences) >= min_sentences:
            mode = "sentence"
            spans = [(a, b) for _, a, b in sentences]
            # 按句取样的向量类内更紧，轮廓系数量纲和滑窗不同，阈值必须分开定。实测：
            # 参考音频切句（真 2 人）+0.437 / +0.610，真实单人视频 0.243，双胞胎 0.114
            # —— 最高负例和最低正例之间有 0.19 的空隙，界线取中间的 0.33。
            min_silhouette = float(self.cfg.get("min_silhouette_sentence", 0.33))
        else:
            # 句子太少（十几秒的短片），按句聚类不可靠，退回滑窗取样
            mode = "window"
            spans = [(a, b) for a, b in _windows(segments, window, hop, audio_seconds)
                     if b - a >= MIN_CLIP_SECONDS]
            # 滑窗的标定数据：参考音频 +0.458，单人视频 +0.105
            min_silhouette = float(self.cfg.get("min_silhouette", 0.25))
        if len(spans) < 2:
            return {"available": False, "reason": "audio_too_short", "samples": len(spans)}

        self.load()
        clips = [np.ascontiguousarray(audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)])
                 for a, b in spans]
        try:
            embeddings = self._embed(clips)
        except Exception as exc:  # noqa: BLE001 - 声纹是增强项，失败不能连累转写
            logger.warning("声纹提取失败：%s", str(exc)[:200])
            return {"available": False, "reason": f"embed_failed: {str(exc)[:160]}"}

        labels, silhouette = cluster_embeddings(
            embeddings,
            max_speakers=int(self.cfg.get("max_speakers", 6)),
            min_silhouette=min_silhouette,
            min_share=float(self.cfg.get("min_cluster_share", 0.05)),
        )
        if labels.size != len(spans):
            return {"available": False, "reason": "cluster_size_mismatch"}

        if mode == "sentence":
            picks = self._assign_by_sentence(segments, sentences, embeddings, labels)
        else:
            picks = self._assign_by_window(segments, spans, labels)

        # 按首次出现时间给簇重新编号，1 号是先开口的那个
        speakers = _write_speakers(segments, picks)

        elapsed = round(time.perf_counter() - started, 2)

        logger.info("说话人分离完成：%d 人 / %d 句（%s取样 %d 个声纹，轮廓系数 %+.3f），耗时 %.1fs",
                    speakers, len(segments), "按句" if mode == "sentence" else "滑窗",
                    len(spans), silhouette, elapsed)
        meta: dict[str, Any] = {
            "available": True,
            "model": {"id": self.model_id, "device": self.device, "path": self.model_path},
            "language": language,
            "speakers": speakers,
            "sampling": mode,
            "samples": len(spans),
            "silhouette": round(silhouette, 3),
            "min_silhouette": min_silhouette,
            "load_seconds": self.load_seconds,
            "elapsed_seconds": elapsed,
        }
        if mode == "window":
            meta["window_seconds"] = window
            meta["hop_seconds"] = hop
        return meta

    def _annotate_by_diarization(self, segments: list[dict[str, Any]], audio: np.ndarray,
                                 language: str | None,
                                 started: float) -> dict[str, Any] | None:
        """主力路径：pyannote 分段 + onnx 声纹聚类。跑不起来返回 None，让调用方退回老路径。"""
        if not self.diarizer.available:
            logger.info("%s 没有对应的 onnx 声纹，改用按句聚类", self.model_id)
            return None
        try:
            turns = self.diarizer.run(audio)
        except Exception as exc:  # noqa: BLE001 - 缺网/缺 sherpa-onnx 都退回老路径
            logger.warning("说话人分离（sherpa-onnx）不可用：%s", str(exc)[:200])
            return None
        if not turns:
            logger.warning("分段模型没切出任何说话片段，改用按句聚类")
            return None

        raw_speakers = len({spk for _, _, spk in turns})
        diar_cfg = self.cfg.get("diarization", {}) or {}
        turns = drop_tiny(turns,
                          min_seconds=float(diar_cfg.get("min_speaker_seconds", 2.0)),
                          min_share=float(diar_cfg.get("min_speaker_share", 0.10)))
        # 先按切换点把跨了两个人的句子切开，再归属：whisper 按停顿分句，一问一答之间
        # 常常没停顿，整句只能挑一个标签，后半句就被标错人。
        before = len(segments)
        segments[:] = split_on_turns(
            segments, [t for turn in turns for t in (turn[0], turn[1])])
        turn_splits = len(segments) - before
        picks = assign_sentences(segments, turns)
        speakers = _write_speakers(segments, picks)

        elapsed = round(time.perf_counter() - started, 2)
        logger.info("说话人分离完成：%d 人 / %d 句（分段 %d 段，聚类前 %d 人，阈值 %.2f，"
                    "按切换点切开 %d 句），耗时 %.1fs",
                    speakers, len(segments), len(turns), raw_speakers,
                    self.diarizer.threshold, turn_splits, elapsed)
        return {
            "available": True,
            "model": {"id": self.model_id, "device": "cpu",
                      "path": str(self.diarizer.root)},
            "language": language,
            "speakers": speakers,
            "sampling": "diarization",
            "turns": len(turns),
            "turn_splits": turn_splits,
            "raw_speakers": raw_speakers,
            "threshold": self.diarizer.threshold,
            "load_seconds": self.diarizer.load_seconds,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _assign_by_sentence(segments: list[dict[str, Any]],
                            sentences: list[tuple[int, float, float]],


                            embeddings: np.ndarray,
                            labels: np.ndarray) -> list[tuple[int, float]]:
        """按句取样时的归属：声纹对应哪句是确定的，只需要给太短的句子找个邻居继承。"""
        unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = np.unique(labels)
        centers = np.stack([unit[labels == i].mean(0) for i in ids])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        picks: list[tuple[int, float] | None] = [None] * len(segments)
        for row, (seg_index, _, _) in enumerate(sentences):
            sims = centers @ unit[row]
            own = int(np.where(ids == labels[row])[0][0])
            if ids.size == 1:
                # 只有一个人时没有"跟别人的差距"可算，就报这条声音离簇心多近（0~1），
                # 含义是"这条声音有多典型"，不要跟多人时的余弦间隔混着理解。
                score = float(sims[own])
            else:
                others = np.delete(sims, own)
                score = float(sims[own] - others.max())
            picks[seg_index] = (int(labels[row]), round(max(0.0, score), 3))

        # 没取到声纹的短句：跟时间上最近的、判过的句子同一个人
        measured = [i for i, p in enumerate(picks) if p is not None]
        for i, pick in enumerate(picks):
            if pick is not None:
                continue
            nearest = min(measured, key=lambda j: abs(j - i))
            picks[i] = (picks[nearest][0], 0.0)
        return [p for p in picks if p is not None]

    @staticmethod
    def _assign_by_window(segments: list[dict[str, Any]], spans: list[tuple[float, float]],
                          labels: np.ndarray) -> list[tuple[int, float]]:
        """滑窗取样时的归属：窗口中心落在句子里就算这句一票，一票都没有就取最近的窗口。"""
        centers = [(a + b) / 2.0 for a, b in spans]
        picks: list[tuple[int, float]] = []
        for seg in segments:
            start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
            counter: Counter = Counter()
            for center, label in zip(centers, labels):
                if start <= center < end:
                    counter[int(label)] += 1
            if not counter:
                nearest = min(range(len(centers)),
                              key=lambda i: abs(centers[i] - (start + end) / 2.0))
                counter[int(labels[nearest])] += 1
            top, hits = counter.most_common(1)[0]
            picks.append((top, round(hits / (sum(counter.values()) or 1), 3)))
        return picks
