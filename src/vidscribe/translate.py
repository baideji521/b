"""中英互译：复用已经下载的视觉模型做**纯文本**翻译，不额外下载翻译模型。

规则（来自需求）：**英文 -> 中文，中文 -> 英文**；其他语言默认译成中文。

三个要点：
1. 只翻译传进来的文本行，**完全不碰视频**，不解码任何一帧（模型只做文本生成）。
2. **增量**：已经有译文、且原文没被改过的条目会跳过，重复点"翻译"只补缺的那几条。
3. 原文永不被覆盖：译文写进 `*_translated`，同时记下 `*_translated_from`（当时的原文），
   原文后来被编辑过就算译文失效，下次会重新翻译。所以"回译"只是切回原字段，无损。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .language import normalize_code, scripts_in
from .logging_setup import get_logger
from .timeline.exporters import write_json

logger = get_logger(__name__)

BATCH_LINES = 16

# 中英互译；其他语言统一译成中文
_PAIR = {"zh": "en", "en": "zh"}



def target_language(source: str | None) -> str:
    return _PAIR.get(normalize_code(source) or "", "zh")


def guess_language(texts: list[str]) -> str | None:
    """没有 whisper 语言时，按书写系统粗判：出现汉字算中文，否则算英文。"""
    joined = " ".join(t for t in texts if t)[:2000]
    if not joined.strip():
        return None
    found = scripts_in(joined)
    if "cjk" in found or "kana" in found:
        return "zh"
    if "latin" in found:
        return "en"
    return None


def needs_translation(text: str | None, translated: str | None,
                      translated_from: str | None) -> bool:
    """这条要不要翻译：没原文不用，没译文要，原文改过了也要（旧译文失效）。"""
    body = str(text or "").strip()
    if not body:
        return False
    if not str(translated or "").strip():
        return True
    return translated_from is not None and str(translated_from).strip() != body


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def translate_lines(analyzer: Any, texts: list[str], target: str,
                    on_progress: Callable[[int, int], None] | None = None) -> list[str | None]:
    """批量翻译。返回与输入等长的列表，失败位置为 None（调用方保留原文）。

    优先走后端的 `translate_lines`（一行一条序列、整批一次解码，快约 10 倍），
    后端没有这个入口时退回 `rewrite_texts`（单提示词 + JSON 数组）。
    """
    results: list[str | None] = []
    total = len(texts)
    done = 0
    batched = callable(getattr(analyzer, "translate_lines", None))
    for chunk in _chunks(texts, BATCH_LINES):
        if batched:
            out = analyzer.translate_lines(chunk, target, max_new_tokens=128)
        else:
            budget = min(2048, 128 * len(chunk) + 96)
            try:
                out = analyzer.rewrite_texts(chunk, target, max_new_tokens=budget)
            except Exception as exc:
                logger.warning("翻译调用失败：%s: %s", type(exc).__name__, str(exc)[:200])
                out = [None] * len(chunk)
        if len(out) != len(chunk):  # 后端返回长度不符时按位置补齐，不让下标错位
            out = (list(out) + [None] * len(chunk))[:len(chunk)]
        results.extend(out)
        done += len(chunk)
        if on_progress is not None:
            on_progress(done, total)
    return results



def _pick_model(cfg: Any, model_id: str | None) -> tuple[str | None, str | None]:
    """选一个能做纯文本翻译的模型。返回 (model_id, 失败原因)。"""
    from .visual.factory import backend_for, known_models  # noqa: PLC0415

    picked = model_id or str(cfg.visual.get("model_id") or "")
    if backend_for(cfg.visual, picked) != "minicpm46":
        return picked, None
    # MiniCPM-V 4.6 的 worker 没有纯文本入口（rewrite_texts 原样返回），
    # 用它翻译会把原文当译文写进去，所以自动换成一个能翻译的后端。
    fallback = next((m["model_id"] for m in known_models(cfg.visual)
                     if m["backend"] != "minicpm46"), None)
    if fallback is None:
        return None, "当前视觉后端不支持纯文本翻译，且没有可替代的模型"
    logger.info("%s 不支持纯文本翻译，改用 %s", picked, fallback)
    return fallback, None


def translate_items(cfg: Any, items: list[dict[str, Any]], source: str | None = None,
                    model_id: str | None = None,
                    on_progress: Callable[[int, int], None] | None = None,
                    analyzer: Any = None) -> dict[str, Any]:
    """翻译一批文本行。

    items: `[{"key": 任意标识, "text": 原文}]`，调用方自己决定这些行从哪来
    （GUI 传的就是界面上当前显示、且还没有译文的那些行）。
    analyzer: 已经在显存里的后端；传进来就直接复用，省掉一次约 15s 的模型加载
    （pipeline 分析完顺手翻译走这条路）。
    返回 `{"ok", "target_language", "translations": {key: 译文}, "failed": [key]}`。
    """
    rows = [{"key": str(it.get("key")), "text": str(it.get("text") or "")}
            for it in items if str(it.get("text") or "").strip()]
    if not rows:
        return {"ok": False, "reason": "no_text", "detail": "没有需要翻译的文本",
                "translations": {}, "failed": []}

    source = normalize_code(source) or guess_language([r["text"] for r in rows])
    target = target_language(source)

    from .visual.factory import backend_for, create_analyzer  # noqa: PLC0415

    # 复用条件：确实是能做纯文本翻译的后端（MiniCPM-V 4.6 的 worker 会原样返回原文）
    reuse = analyzer is not None and backend_for(cfg.visual, analyzer.model_id) != "minicpm46"
    if reuse:
        picked, error = analyzer.model_id, None
    else:
        picked, error = _pick_model(cfg, model_id)
    if picked is None:
        return {"ok": False, "reason": "backend_cannot_translate", "detail": error,
                "translations": {}, "failed": [r["key"] for r in rows]}

    logger.info("纯文本翻译：%s -> %s，共 %d 行，引擎 %s%s（不解码视频）",
                source or "auto", target, len(rows), picked,
                "，复用已加载的模型" if reuse else "")

    if not reuse:
        analyzer = create_analyzer(cfg.visual, str(cfg.path("model_dir")), cfg.mirrors,
                                   model_id=picked)
    started = time.perf_counter()
    try:
        analyzer.load()  # 已经加载过的话是空操作
        out = translate_lines(analyzer, [r["text"] for r in rows], target, on_progress=on_progress)
    finally:
        if not reuse:  # 复用的模型归调用方管，这里不能卸载
            try:
                analyzer.unload()
            except Exception:
                pass
    elapsed = round(time.perf_counter() - started, 2)


    translations: dict[str, str] = {}
    failed: list[str] = []
    for row, text in zip(rows, out):
        if text and str(text).strip():
            translations[row["key"]] = str(text).strip()
        else:
            failed.append(row["key"])
    logger.info("翻译完成：成功 %d / 失败 %d，耗时 %.1fs", len(translations), len(failed), elapsed)
    return {
        "ok": bool(translations),
        "reason": None if translations else "all_failed",
        "source_language": source,
        "target_language": target,
        "engine": "visual_model_text",
        "model_id": getattr(analyzer, "model_id", picked),
        "elapsed_seconds": elapsed,
        "translations": translations,
        "failed": failed,
    }


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("读取 %s 失败：%s", path.name, exc)
        return None


def translate_output(cfg: Any, out_dir: str | Path, model_id: str | None = None,
                     retranslate: bool = False,
                     on_progress: Callable[[int, int], None] | None = None,
                     analyzer: Any = None) -> dict[str, Any]:
    """命令行入口：翻译一个输出目录里还没有译文的语音段与画面事件。

    GUI 不走这条路（它按界面上显示的内容调 translate_items），这里是给
    `run.py translate <目录>` 和 pipeline 的"分析完顺手翻译"用的：
    增量、原文不覆盖；analyzer 传进来就复用显存里的模型。
    """

    out_dir = Path(out_dir)
    speech_doc = _read(out_dir / "speech_events.json")
    visual_doc = _read(out_dir / "visual_events.json")
    timeline_doc = _read(out_dir / "timeline.json")
    if speech_doc is None and visual_doc is None:
        return {"ok": False, "reason": "no_results", "detail": f"{out_dir} 下没有分析结果"}

    segments: list[dict[str, Any]] = list((speech_doc or {}).get("segments") or [])
    events: list[dict[str, Any]] = list((visual_doc or {}).get("events") or [])

    items: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        text = str(seg.get("text") or "")
        if retranslate or needs_translation(text, seg.get("text_translated"),
                                           seg.get("text_translated_from")):
            items.append({"key": f"s{i}", "text": text})
    for i, ev in enumerate(events):
        text = str(ev.get("description") or "")
        if retranslate or needs_translation(text, ev.get("description_translated"),
                                            ev.get("description_translated_from")):
            items.append({"key": f"v{i}", "text": text})
    if not items:
        return {"ok": True, "reason": "already_translated", "speech_translated": 0,
                "event_translated": 0, "detail": "所有条目都已经有译文，无需重复翻译"}

    source = (speech_doc or {}).get("language") or (timeline_doc or {}).get("original_language")
    result = translate_items(cfg, items, source=source, model_id=model_id,
                             on_progress=on_progress, analyzer=analyzer)
    if not result.get("ok"):
        return result

    got: dict[str, str] = result["translations"]
    target = result["target_language"]
    meta = {k: result[k] for k in
            ("source_language", "target_language", "engine", "model_id", "elapsed_seconds")}
    meta["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    speech_ok = 0
    by_id: dict[int, str] = {}
    for i, seg in enumerate(segments):
        text = got.get(f"s{i}")
        if not text:
            continue
        seg["text_translated"] = text
        seg["text_translated_from"] = str(seg.get("text") or "")
        seg["translated_language"] = target
        speech_ok += 1
        if isinstance(seg.get("id"), int):
            by_id[int(seg["id"])] = text
    event_ok = 0
    for i, ev in enumerate(events):
        text = got.get(f"v{i}")
        if not text:
            continue
        ev["description_translated"] = text
        ev["description_translated_from"] = str(ev.get("description") or "")
        ev["translated_language"] = target
        event_ok += 1

    if speech_doc is not None and speech_ok:
        speech_doc["segments"] = segments
        speech_doc["translation"] = {**meta, "translated": speech_ok, "total": len(segments)}
        write_json(out_dir / "speech_events.json", speech_doc)
    if visual_doc is not None and event_ok:
        visual_doc["events"] = events
        visual_doc["translation"] = {**meta, "translated": event_ok, "total": len(events)}
        write_json(out_dir / "visual_events.json", visual_doc)

    if timeline_doc is not None and (speech_ok or event_ok):
        event_map = {str(e.get("description") or ""): e.get("description_translated")
                     for e in events if e.get("description_translated")}
        for entry in timeline_doc.get("timeline") or []:
            visual = str(entry.get("visual") or "")
            if visual and event_map.get(visual):
                entry["visual_translated"] = event_map[visual]
            # 语音条目是多段拼接的，用段 id 取译文重新拼，不再翻译一次拼接串
            ids = [i for i in (entry.get("speech_event_ids") or []) if isinstance(i, int)]
            parts = [by_id[i] for i in ids if i in by_id]
            if parts:
                entry["speech_translated"] = " ".join(parts)
        timeline_doc["translation"] = {**meta, "speech_translated": speech_ok,
                                      "visual_translated": event_ok}
        write_json(out_dir / "timeline.json", timeline_doc)

    return {
        "ok": True,
        "target_language": target,
        "source_language": result.get("source_language"),
        "speech_translated": speech_ok,
        "speech_total": len(segments),
        "event_translated": event_ok,
        "event_total": len(events),
        "failed": result.get("failed") or [],
        "elapsed_seconds": result.get("elapsed_seconds"),
    }
