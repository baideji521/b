"""视觉后端工厂：按 model_id / backend 选择 Qwen3-VL 或 MiniCPM-V。

两个后端对外契约完全一致（load / unload / set_output_language /
analyze_window / analyze_windows / rewrite_texts / model_id / load_seconds），
Pipeline 只认这套接口，不关心具体模型。
"""

from __future__ import annotations

from typing import Any

BACKENDS = ("qwen3vl", "minicpm")


def resolve_backend(model_id: str, backend: str | None = None) -> str:
    """backend 为空或 auto 时按 model_id 猜；猜不出来默认 qwen3vl。"""
    if backend and backend.lower() not in ("auto", ""):
        code = backend.lower()
        if code not in BACKENDS:
            raise ValueError(f"未知视觉后端: {backend}（可选 {', '.join(BACKENDS)}）")
        return code
    name = (model_id or "").lower()
    if "minicpm" in name:
        return "minicpm"
    return "qwen3vl"


def known_models(vcfg: dict[str, Any]) -> list[dict[str, str]]:
    """配置里的可切换模型清单，保证当前 model_id 一定在列表里。"""
    items: list[dict[str, str]] = []
    for entry in vcfg.get("models") or []:
        mid = str(entry.get("model_id") or "").strip()
        if not mid:
            continue
        items.append({
            "label": str(entry.get("label") or mid),
            "model_id": mid,
            "backend": resolve_backend(mid, entry.get("backend")),
        })
    current = str(vcfg.get("model_id") or "").strip()
    if current and not any(i["model_id"] == current for i in items):
        items.insert(0, {
            "label": current,
            "model_id": current,
            "backend": resolve_backend(current, vcfg.get("backend")),
        })
    return items


def backend_for(vcfg: dict[str, Any], model_id: str) -> str:
    """优先用 models 清单里声明的 backend，其次用全局 backend/自动判断。"""
    for entry in vcfg.get("models") or []:
        if str(entry.get("model_id") or "") == model_id and entry.get("backend"):
            return resolve_backend(model_id, entry["backend"])
    return resolve_backend(model_id, vcfg.get("backend"))


def create_analyzer(vcfg: dict[str, Any], model_dir: str | None = None,
                    mirrors: dict[str, Any] | None = None,
                    model_id: str | None = None) -> Any:
    """创建分析器实例（不加载权重）。"""
    target = model_id or str(vcfg.get("model_id") or "")
    backend = backend_for(vcfg, target)
    cfg = dict(vcfg)
    cfg["model_id"] = target
    if backend == "minicpm":
        from .minicpm import MiniCPMAnalyzer  # noqa: PLC0415

        return MiniCPMAnalyzer(cfg, model_dir, mirrors)
    from .qwen_vl import QwenVLAnalyzer  # noqa: PLC0415

    return QwenVLAnalyzer(cfg, model_dir, mirrors)
