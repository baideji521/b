"""标点恢复（FunASR / ct-punc），并把标点贴回词级时间戳。

whisper 在无标点素材上经常整段吐全小写零标点（实测某条 TikTok：标点 0 个、大写 0 个），
`initial_prompt` 能救一部分但不保证。工业界的常规做法是 ASR 之后再过一个专门的标点模型，
这里用 FunASR 的 `ct-punc`（`iic/punc_ct-transformer_cn-en-common-vocab471067-large`，中英双语，
英文还会补首字母大写）。funasr 这个依赖项目里已经在用（emotion2vec 走的就是它）。

关键点是**标点必须回贴到词上**，不能只改整段 text：断句靠的是 `words[].word` 的收尾字符 +
词间停顿，时间戳全部来自 whisper 的 word_timestamps。所以流程是

    words -> 拼成无标点纯文本 -> ct-punc -> 按字符对齐回每个 word -> sentences.split_sentences

对齐用"逐字符消费"：把模型输出里的非标点字符和原词序列的字符一个个对上，遇到标点就贴到
当前那个词的尾巴上。这样即使模型改了空格、大小写或者吞掉个别字符，也不会把整段错位——
对不上就原样返回，宁可不加标点也不能错位。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

logger = get_logger(__name__)

# ct-punc 会吐的标点（中英）。贴回词尾时只认这些，别的字符一律当正文
_PUNCT = set("。，、？！；：,.?!;:")
# 判断一段文本是否"已经有标点"，有就不用再跑模型
_SENTENCE_MARKS = set("。，、？！；：,.?!;:…")


def has_punctuation(text: str, min_ratio: float = 0.002) -> bool:
    """文本里标点是否够用。

    不是"有一个就算"：whisper 偶尔会在开头点一个逗号然后整段光秃秃。按字符数算个比例，
    低于 min_ratio 视为没标点（1681 字的实测样本里 0 个标点，正常转写在 5% 上下）。
    """
    body = str(text or "")
    if not body:
        return True
    marks = sum(1 for ch in body if ch in _SENTENCE_MARKS)
    return marks >= max(1, int(len(body) * min_ratio))


def _plain(words: list[dict[str, Any]]) -> str:
    """词序列拼成喂给模型的纯文本：去掉已有标点，中文不加空格，英文按空格分词。"""
    pieces: list[str] = []
    for word in words:
        token = str(word.get("word") or "").strip()
        token = "".join(ch for ch in token if ch not in _PUNCT)
        if token:
            pieces.append(token)
    return " ".join(pieces)


def attach(words: list[dict[str, Any]], punctuated: str) -> int:
    """把 `punctuated` 里的标点贴回 words[].word，返回贴上的标点个数。

    逐字符对齐：输出里的正文字符必须和原词序列（忽略空白、忽略原有标点）一一对上；
    对不上就整段放弃（返回 0，words 不动），避免错位污染时间轴。
    """
    # 原词序列展平成 (字符, 词下标)，忽略空白和原有标点
    flat: list[tuple[str, int]] = []
    for index, word in enumerate(words):
        for ch in str(word.get("word") or ""):
            if ch.isspace() or ch in _PUNCT:
                continue
            flat.append((ch, index))

    tail: dict[int, str] = {}      # 词下标 -> 要追加的标点
    cursor = 0
    for ch in punctuated:
        if ch.isspace():
            continue
        if ch in _PUNCT:
            if cursor > 0:         # 标点属于前一个字符所在的词
                tail[flat[cursor - 1][1]] = ch
            continue
        if cursor >= len(flat):
            break
        # 模型会改大小写（英文首字母），比较时统一小写
        if flat[cursor][0].lower() != ch.lower():
            logger.warning("标点回贴对齐失败：第 %d 个字符 %r != %r，放弃本段标点",
                           cursor, flat[cursor][0], ch)
            return 0
        # 大小写按模型的来：它把句首字母改大写，这本身就是断句要用的信号
        flat[cursor] = (ch, flat[cursor][1])
        cursor += 1

    if cursor < len(flat):
        logger.warning("标点回贴只覆盖 %d/%d 个字符，放弃本段标点", cursor, len(flat))
        return 0

    # 按词重建 word 文本：正文用模型的大小写，尾巴补标点，前导空格保持 whisper 原样
    rebuilt: list[list[str]] = [[] for _ in words]
    for ch, index in flat:
        rebuilt[index].append(ch)
    added = 0
    for index, word in enumerate(words):
        body = "".join(rebuilt[index])
        if not body:
            continue
        lead = " " if str(word.get("word") or "").startswith(" ") else ""
        mark = tail.get(index, "")
        if mark:
            added += 1
        word["word"] = lead + body + mark
    return added


class Punctuator:
    """ct-punc 标点恢复。模型只加载一次，多视频复用。"""

    def __init__(self, cfg: dict[str, Any], model_dir: str | None = None,
                 mirrors: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.model_dir = model_dir
        self.mirrors = mirrors or {}
        self.model = None
        self.model_id: str = str(self.cfg.get("model_id")
                                 or "iic/punc_ct-transformer_cn-en-common-vocab471067-large")
        self.device: str = "cpu"
        self.load_seconds = 0.0
        self.model_path: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def load(self) -> None:
        if self.model is not None:
            return
        import torch  # noqa: PLC0415
        from funasr import AutoModel  # noqa: PLC0415

        want = self.cfg.get("device", "auto")
        if want == "auto":
            want = "cuda:0" if torch.cuda.is_available() else "cpu"

        target = self.model_id
        if self.model_dir:
            from ..mirrors import resolve_model  # noqa: PLC0415

            # ct-punc 和 emotion2vec 一样是 model.pt + config.yaml，走同一套 kind
            target = resolve_model(self.model_id, Path(self.model_dir), self.mirrors,
                                   kind="emotion")

        errors: list[str] = []
        for device in ([want, "cpu"] if want != "cpu" else ["cpu"]):
            started = time.perf_counter()
            try:
                logger.info("加载标点模型 %s (device=%s)", target, device)
                self.model = AutoModel(model=target, device=device, disable_update=True,
                                       disable_log=True, disable_pbar=True, hub="ms")
            except Exception as exc:  # noqa: BLE001 - 换设备再试，全失败才放弃
                errors.append(f"{device}: {str(exc)[:160]}")
                logger.warning("标点模型加载失败 %s@%s：%s", self.model_id, device, str(exc)[:160])
                continue
            self.device, self.model_path = device, target
            self.load_seconds = round(time.perf_counter() - started, 2)
            logger.info("标点模型就绪：%s / %s，耗时 %.1fs", self.model_id, device,
                        self.load_seconds)
            return
        raise RuntimeError("标点模型加载失败: " + "; ".join(errors))

    def unload(self) -> None:
        """和 WhisperASR / EmotionRecognizer 一样显式回收，别占着显存等视觉模型。"""
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self.model is None:
            return
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("已释放标点模型显存")

    def _generate(self, text: str) -> str:
        out = self.model.generate(input=text)
        if isinstance(out, list) and out:
            return str(out[0].get("text") or "")
        return ""

    def restore(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """给缺标点的段补标点：标点贴到 words 上，同时刷新该段的 text。

        逐段跑（不整条视频拼一起）：段内的词序列是连续语音，模型上下文够用；跨段拼起来
        一旦某处对齐失败会连累全片。返回统计，写进 speech_events.json 的 meta。
        """
        todo = [seg for seg in segments
                if (seg.get("words") or []) and not has_punctuation(seg.get("text") or "")]
        if not todo:
            return {"available": False, "reason": "already_punctuated"}

        started = time.perf_counter()
        self.load()
        done = failed = marks = 0
        for seg in todo:
            words = seg["words"]
            plain = _plain(words)
            if not plain:
                continue
            try:
                punctuated = self._generate(plain)
            except Exception as exc:  # noqa: BLE001 - 单段失败不影响其它段
                logger.warning("标点恢复失败（%.2fs 起）：%s", seg.get("start", 0.0), str(exc)[:160])
                failed += 1
                continue
            added = attach(words, punctuated)
            if not added:
                failed += 1
                continue
            text = "".join(str(w.get("word") or "") for w in words).strip()
            seg["text"] = text
            seg["original_text"] = text
            seg["punctuation_restored"] = True
            done += 1
            marks += added

        elapsed = round(time.perf_counter() - started, 2)
        logger.info("标点恢复完成：%d/%d 段（失败 %d），补了 %d 个标点，耗时 %.1fs",
                    done, len(todo), failed, marks, elapsed)
        return {
            "available": done > 0,
            "model": {"id": self.model_id, "device": self.device, "path": self.model_path},
            "restored_segments": done,
            "failed_segments": failed,
            "marks": marks,
            "load_seconds": self.load_seconds,
            "elapsed_seconds": elapsed,
        }
