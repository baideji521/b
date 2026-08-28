"""画面表情：YuNet 人脸检测 + HSEmotion（AffectNet 8 类）onnx 分类。

**为什么不让视觉模型判表情**：`max_pixels_tokens=112` 下 810×1080 的竖屏被缩到约
256×342，人脸只剩 70~85 像素；而且视觉模型一个 15 秒窗口只看 8 帧，表情这种几百毫秒
就变的东西根本采不到。实测同一条素材直接在原始帧上检测，人脸有 139~250 像素，
2 fps 采样 174 秒的片子拿到 348 个样本，CPU 上 20.5 秒跑完（视觉阶段本身要 81 秒）。

**为什么不用 MediaPipe**：OpenCV 4.11 自带 `cv2.FaceDetectorYN`（YuNet，232KB onnx），
检测这一步就够了，省掉 mediapipe 那 60MB 依赖。表情分类用 HSEmotion 的
`enet_b0_8_best_afew.onnx`（16MB，AffectNet 8 类），走已经装好的 onnxruntime。

输出只覆盖 `VisualEvent` 的情绪字段（`emotion_en` / `emotion_intensity` /
`emotion_source="face"`），事件的时间轴和描述都不动——判不出人脸的事件保留视觉模型给的
情绪，`emotion_source` 记 `model`。
"""

from __future__ import annotations

import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..events import VisualEvent
from ..logging_setup import get_logger
from ..video_io import VideoInfo

logger = get_logger(__name__)

# enet_b0_8 的输出顺序（AffectNet 8 类），右边是本项目内部标签（见 emotions.EMOTION_ZH）
AFFECTNET: tuple[tuple[str, str], ...] = (
    ("anger", "angry"),
    ("contempt", "contempt"),
    ("disgust", "disgusted"),
    ("fear", "fearful"),
    ("happiness", "happy"),
    ("neutral", "neutral"),
    ("sadness", "sad"),
    ("surprise", "surprised"),
)
LABELS: tuple[str, ...] = tuple(internal for _, internal in AFFECTNET)

# ImageNet 归一化（HSEmotion 训练时的预处理）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FILES: dict[str, dict[str, Any]] = {
    "detector": {
        "file": "yunet.onnx",
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://hf-mirror.com/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    "emotion": {
        "file": "enet_b0_8.onnx",
        "urls": [
            "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
        ],
    },
}


def _fetch(target: Path, urls: list[str]) -> Path:
    """下模型：按 Content-Length 校验完整性。

    实测这两个源会中途断流（第一次拿到 15874801 / 应为 16039595 字节，
    onnxruntime 直接报 Protobuf parsing failed），所以长度不对就重试。
    """
    if target.is_file() and target.stat().st_size > 100_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        for attempt in range(3):
            try:
                logger.info("下载表情模型：%s", url)
                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    expect = int(resp.headers.get("Content-Length") or 0)
                    data = resp.read()
                if expect and len(data) != expect:
                    errors.append(f"{url}: 只收到 {len(data)}/{expect} 字节")
                    continue
                target.write_bytes(data)
                return target
            except Exception as exc:  # noqa: BLE001 - 换源/重试
                errors.append(f"{url}#{attempt}: {str(exc)[:100]}")
    raise RuntimeError("表情模型下载失败: " + "; ".join(errors[-3:]))


def _ascii_path(path: Path) -> str:
    """OpenCV 的 dnn 读不了带非 ASCII 字符的路径（Windows 上 std::ifstream 用窄字符），
    项目目录带中文时会报 "Can't read ONNX file"。这种情况把模型拷到系统临时目录再喂给它。

    onnxruntime 没这个问题（Windows 上走宽字符 API），所以只有检测器需要这层处理。
    """
    text = str(path)
    if text.isascii():
        return text
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    target = Path(tempfile.gettempdir()) / "vidscribe-onnx" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != path.stat().st_size:
        shutil.copyfile(path, target)
        logger.info("模型路径含非 ASCII 字符，已拷到 %s 供 OpenCV 读取", target)
    return str(target)


class FaceEmotion:
    """按固定帧率扫全片，给每个采样点的人脸打表情标签。模型只加载一次，多视频复用。"""

    def __init__(self, cfg: dict[str, Any], model_dir: str | Path | None = None):
        self.cfg = cfg or {}
        self.root = Path(model_dir or "models") / "face-emotion"
        self.detector = None
        self.session = None
        self.load_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def load(self) -> None:
        if self.session is not None:
            return
        import cv2  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        started = time.perf_counter()
        det_path = _fetch(self.root / str(FILES["detector"]["file"]),
                          list(FILES["detector"]["urls"]))
        emo_path = _fetch(self.root / str(FILES["emotion"]["file"]),
                          list(FILES["emotion"]["urls"]))
        # 输入尺寸每帧都要按实际缩放后的画面重设，这里先给个占位值
        self.detector = cv2.FaceDetectorYN.create(
            _ascii_path(det_path), "", (320, 320),
            float(self.cfg.get("detect_score", 0.6)), 0.3, 5000,
        )
        self.session = ort.InferenceSession(str(emo_path), providers=["CPUExecutionProvider"])
        self.load_seconds = round(time.perf_counter() - started, 2)
        logger.info("表情模型就绪：YuNet + HSEmotion(8类)，耗时 %.1fs", self.load_seconds)

    def unload(self) -> None:
        self.detector = None
        self.session = None

    def scan(self, info: VideoInfo, on_progress=None) -> list[dict[str, Any]]:
        """按 sample_fps 扫全片，返回 [{time, faces:[{emotion_en, score, width}]}]。

        只保留最大的 max_faces 张脸（按面积），太小的脸（`min_face_px`）直接丢——
        背景里的路人脸判出来的表情是噪声。
        """
        import cv2  # noqa: PLC0415

        self.load()
        assert self.session is not None and self.detector is not None
        sample_fps = float(self.cfg.get("sample_fps", 2.0))
        detect_long = int(self.cfg.get("detect_size", 640))
        min_face = int(self.cfg.get("min_face_px", 60))
        max_faces = int(self.cfg.get("max_faces", 2))

        cap = cv2.VideoCapture(info.path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"无法打开视频: {info.path}")
        fps = info.fps if info.fps > 0 else cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / max(sample_fps, 0.1))))
        scale = min(1.0, detect_long / max(info.width, info.height, 1))
        dw, dh = max(1, int(info.width * scale)), max(1, int(info.height * scale))
        self.detector.setInputSize((dw, dh))

        samples: list[dict[str, Any]] = []
        total = float(info.total_frames or 0)
        report_every = max(step, int(fps * 5))
        frame_no = 0
        try:
            while True:
                if not cap.grab():
                    break
                if frame_no % step == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        faces = self._faces(frame, scale, dw, dh, min_face, max_faces)
                        samples.append({"time": round(frame_no / fps, 3), "faces": faces})
                if on_progress is not None and total > 0 and frame_no % report_every == 0:
                    on_progress(min(1.0, frame_no / total))
                frame_no += 1
        finally:
            cap.release()
        if on_progress is not None:
            on_progress(1.0)
        with_face = sum(1 for s in samples if s["faces"])
        logger.info("表情扫描完成：%d 个采样点，其中 %d 个检出人脸（%.1f fps 采样）",
                    len(samples), with_face, sample_fps)
        return samples

    def _faces(self, frame, scale: float, dw: int, dh: int,
               min_face: int, max_faces: int) -> list[dict[str, Any]]:
        import cv2  # noqa: PLC0415

        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        _, found = self.detector.detect(small)
        if found is None or len(found) == 0:
            return []
        crops: list[np.ndarray] = []
        widths: list[int] = []
        for row in sorted(found, key=lambda f: -float(f[2]) * float(f[3]))[:max_faces]:
            x, y, fw, fh = (int(round(float(v) / scale)) for v in row[:4])
            x, y = max(0, x), max(0, y)
            if fw < min_face or fh < min_face:
                continue
            crop = frame[y:y + fh, x:x + fw]
            if crop.size == 0:
                continue
            img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            crops.append(((img - _MEAN) / _STD).transpose(2, 0, 1))
            widths.append(fw)
        if not crops:
            return []
        logits = self.session.run(None, {"input": np.stack(crops).astype(np.float32)})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        out: list[dict[str, Any]] = []
        for i in range(len(crops)):
            top = int(probs[i].argmax())
            out.append({"emotion_en": LABELS[top], "score": round(float(probs[i][top]), 3),
                        "width": widths[i]})
        return out


def aggregate(samples: list[dict[str, Any]], start: float, end: float,
              min_score: float = 0.35) -> dict[str, Any] | None:
    """一段时间里的采样点 -> 一个表情标签。

    每个采样点只取最大那张脸（画面主体），按置信度加权投票；强度取胜出标签的平均置信度。
    这一段里一张脸都没检出，或者所有置信度都低于 `min_score`，返回 None（保留视觉模型的判断）。
    """
    weights: dict[str, float] = defaultdict(float)
    scores: dict[str, list[float]] = defaultdict(list)
    used = 0
    for sample in samples:
        t = float(sample.get("time") or 0.0)
        if t < start or t > end:
            continue
        faces = sample.get("faces") or []
        if not faces:
            continue
        face = max(faces, key=lambda f: int(f.get("width") or 0))
        score = float(face.get("score") or 0.0)
        label = str(face.get("emotion_en") or "")
        if not label or score < min_score:
            continue
        weights[label] += score
        scores[label].append(score)
        used += 1
    if not weights:
        return None
    top = max(weights, key=lambda k: weights[k])
    return {
        "emotion_en": top,
        "intensity": round(sum(scores[top]) / len(scores[top]), 3),
        "samples": used,
        "share": round(len(scores[top]) / used, 3),
    }


def annotate(events: list[VisualEvent], samples: list[dict[str, Any]],
             min_score: float = 0.35) -> dict[str, Any]:
    """用人脸表情覆盖事件的情绪字段，返回统计（写进 visual meta）。

    `emotion` 显示名不在这儿定：pipeline 统一按 output_language 用 emotion_en 渲染。
    """
    replaced = 0
    kept = 0
    for ev in events:
        agreed = aggregate(samples, ev.start, ev.end, min_score)
        if agreed is None:
            ev.emotion_source = "model" if ev.emotion_en else None
            kept += 1
            continue
        ev.emotion_en = agreed["emotion_en"]
        ev.emotion_intensity = agreed["intensity"]
        ev.emotion_source = "face"
        replaced += 1
    with_face = sum(1 for s in samples if s.get("faces"))
    logger.info("画面表情：%d 个事件用人脸模型判定，%d 个沿用视觉模型（采样 %d 点 / 检出 %d 点）",
                replaced, kept, len(samples), with_face)
    return {
        "available": True,
        "events_from_face": replaced,
        "events_from_model": kept,
        "samples": len(samples),
        "samples_with_face": with_face,
        "min_score": min_score,
    }
