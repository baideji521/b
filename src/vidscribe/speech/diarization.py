"""说话人分离（sherpa-onnx：pyannote 分段 + cam++ 声纹 + FastClustering）。

**为什么换掉"按句取声纹 + 谱聚类"那套**：句子边界不是说话人边界。ASR 切出来的一句
经常横跨两个人（实测那条双人视频，一句 6~11 秒里两个人都在说），这种句子的声纹是混音，
聚类怎么调都白搭。pyannote 的分段模型是直接按"谁在说"切的，还能标重叠说话，
切出来的每段基本是单人，声纹才干净。

**人数怎么定**：`FastClusteringConfig(num_clusters=-1, threshold=T)` 按余弦距离做层次
合并，阈值 T 决定人数。实测那条已知 2 人的视频（双胞胎，同一副嗓子，最难的一档）：

- 英文 VoxCeleb cam++：T=0.7 → 6 人，0.9 → 4 人，**1.1 → 2 人（25.9s / 93.3s，
  跟强制 k=2 的结果完全一致）**，1.3 → 1 人
- 中文 cam++ zh-cn：T=0.7 → 7 人，**0.9 → 2 人**，1.1 → 1 人

所以阈值跟着声纹模型走（见 `EMBEDDINGS`），不是一个全局常数。
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000

# pyannote/segmentation-3.0 的 onnx 导出（sherpa-onnx 用的那份）。
SEGMENTATION = {
    "file": "segmentation.onnx",
    "urls": [
        "https://www.modelscope.cn/models/pengzhendong/sherpa-onnx-pyannote-segmentation-3-0/resolve/master/model.onnx",
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
    ],
}

# 声纹（onnx 版）+ 各自的聚类阈值。key 是 config 里 speech.speaker.model_id。
EMBEDDINGS: dict[str, dict[str, Any]] = {
    "iic/speech_campplus_sv_en_voxceleb_16k": {
        "file": "campplus_en_voxceleb.onnx",
        "threshold": 1.1,
        "urls": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
            "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        ],
    },
    "iic/speech_campplus_sv_zh-cn_16k-common": {
        "file": "campplus_zh_cn_common.onnx",
        "threshold": 0.9,
        "urls": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
            "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        ],
    },
}


def _fetch(target: Path, urls: list[str]) -> Path:
    """下模型：两个源轮着试。这两个文件都是单个 onnx，没有配套文件，直接 HTTP 拉就行
    （huggingface_hub 在这台机器上拉 LFS 会 LocalEntryNotFoundError，所以不走它）。"""
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    errors: list[str] = []
    for url in urls:
        try:
            logger.info("下载说话人分离模型：%s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 - 换下一个源
            errors.append(f"{url}: {str(exc)[:120]}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("模型下载失败: " + "; ".join(errors))


class Diarizer:
    """sherpa-onnx 的离线说话人分离。模型只加载一次，多视频复用。"""

    def __init__(self, cfg: dict[str, Any], model_id: str, model_dir: str | Path | None = None):
        self.cfg = cfg or {}
        self.model_id = model_id
        self.root = Path(model_dir or "models") / "sherpa-diarization"
        self.spec = EMBEDDINGS.get(model_id)
        self.threshold = float(self.cfg.get("threshold")
                               or (self.spec or {}).get("threshold", 1.1))
        self.sd = None
        self.load_seconds = 0.0

    @property
    def available(self) -> bool:
        return self.spec is not None

    def load(self) -> None:
        if self.sd is not None:
            return
        if self.spec is None:
            raise RuntimeError(f"没有 {self.model_id} 对应的 onnx 声纹模型")
        import sherpa_onnx  # noqa: PLC0415

        started = time.perf_counter()
        seg = _fetch(self.root / str(SEGMENTATION["file"]), list(SEGMENTATION["urls"]))
        emb = _fetch(self.root / str(self.spec["file"]), list(self.spec["urls"]))
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg)),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1,
                                                        threshold=self.threshold),
            min_duration_on=float(self.cfg.get("min_duration_on", 0.3)),
            min_duration_off=float(self.cfg.get("min_duration_off", 0.5)),
        )
        if not config.validate():
            raise RuntimeError("sherpa-onnx 分离配置不合法（模型文件对不上）")
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        self.load_seconds = round(time.perf_counter() - started, 2)
        logger.info("说话人分离模型就绪：%s（阈值 %.2f），耗时 %.1fs",
                    self.spec["file"], self.threshold, self.load_seconds)

    def unload(self) -> None:
        self.sd = None

    def run(self, audio: np.ndarray) -> list[tuple[float, float, int]]:
        """16k 单声道波形 -> [(起, 止, 说话人簇号)]，按时间排好序。"""
        self.load()
        assert self.sd is not None
        if int(self.sd.sample_rate) != SAMPLE_RATE:
            raise RuntimeError(f"分段模型要求 {self.sd.sample_rate}Hz")
        result = self.sd.process(np.ascontiguousarray(audio, dtype=np.float32))
        return [(float(s.start), float(s.end), int(s.speaker))
                for s in result.sort_by_start_time()]


def drop_tiny(turns: list[tuple[float, float, int]], min_seconds: float = 2.0,
              min_share: float = 0.05) -> list[tuple[float, float, int]]:
    """说话时长太短的簇并回时间上相邻的簇。

    过度切分的残渣一般只有一两秒（笑声、背景人声、"嗯"），留着会让人数虚高。
    """
    if not turns:
        return turns
    total = sum(end - start for start, end, _ in turns)
    spoken: dict[int, float] = {}
    for start, end, spk in turns:
        spoken[spk] = spoken.get(spk, 0.0) + (end - start)
    floor = max(min_seconds, min_share * total)
    keep = {spk for spk, seconds in spoken.items() if seconds >= floor}
    if not keep or len(keep) == len(spoken):
        return turns
    out: list[tuple[float, float, int]] = []
    for i, (start, end, spk) in enumerate(turns):
        if spk in keep:
            out.append((start, end, spk))
            continue
        # 找时间上最近的、被保留的那一段，继承它的说话人
        best: tuple[float, int] | None = None
        for j, (s2, e2, spk2) in enumerate(turns):
            if i == j or spk2 not in keep:
                continue
            gap = 0.0 if s2 < end and start < e2 else min(abs(start - e2), abs(s2 - end))
            if best is None or gap < best[0]:
                best = (gap, spk2)
        if best is not None:
            out.append((start, end, best[1]))
    return out


def assign_sentences(segments: list[dict[str, Any]],
                     turns: list[tuple[float, float, int]]) -> list[tuple[int, float]]:
    """把分离结果映射到 ASR 的句子上：[(簇号, 置信度)]，跟 segments 一一对应。

    置信度 = 判给这个人的重叠时长 / 这句里所有重叠时长之和。一句横跨两个人时会明显偏低，
    正好当"这句可能不止一个人在说"的提示。整句都没重叠（分段模型认为是静音/音乐）时
    取时间最近的一段，置信度记 0。
    """
    picks: list[tuple[int, float]] = []
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        overlap: dict[int, float] = {}
        for t_start, t_end, spk in turns:
            shared = min(end, t_end) - max(start, t_start)
            if shared > 0:
                overlap[spk] = overlap.get(spk, 0.0) + shared
        if overlap:
            total = sum(overlap.values())
            spk = max(overlap, key=lambda k: overlap[k])
            picks.append((spk, round(overlap[spk] / total, 3)))
            continue
        middle = (start + end) / 2.0
        nearest = min(turns, key=lambda t: abs((t[0] + t[1]) / 2.0 - middle))
        picks.append((nearest[2], 0.0))
    return picks
