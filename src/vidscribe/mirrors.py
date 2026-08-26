"""模型下载源：优先国内镜像，全部指向官方模型仓库。

顺序（可在 config.json 的 mirrors.model_sources 里调整）：
1. modelscope  —— 阿里云 ModelScope，Qwen 官方账号，国内直连最快
2. hf_mirror   —— hf-mirror.com，HuggingFace 官方仓库的国内镜像
3. hf          —— huggingface.co 官方源（兜底）

镜像只用于加速下载，repo id 始终是官方仓库，不替换成任何第三方/破解模型。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .logging_setup import get_logger

logger = get_logger(__name__)

HF_OFFICIAL = "https://huggingface.co"
VISUAL_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "*.model", "*.md"]
WHISPER_ALLOW = None  # faster-whisper 仓库很小，整仓下载


def _cache_file(model_dir: Path) -> Path:
    return model_dir / "resolved_models.json"


def _load_cache(model_dir: Path) -> dict[str, str]:
    path = _cache_file(model_dir)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_cache(model_dir: Path, cache: dict[str, str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(_cache_file(model_dir), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def _looks_complete(path: Path, kind: str) -> bool:
    if not path.is_dir():
        return False
    if kind == "whisper":
        return (path / "model.bin").is_file()
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def _hf_download(repo_id: str, cache_dir: Path, endpoint: str | None,
                 allow_patterns: Iterable[str] | None) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    # 用 endpoint 参数而不是改环境变量：huggingface_hub 在 import 时就固定了 ENDPOINT，
    # 导入之后再改 HF_ENDPOINT 是无效的。
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        endpoint=endpoint or HF_OFFICIAL,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )


def _modelscope_download(repo_id: str, cache_dir: Path,
                         allow_patterns: Iterable[str] | None) -> str:
    from modelscope import snapshot_download  # noqa: PLC0415

    kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    return snapshot_download(repo_id, **kwargs)


def resolve_model(repo_id: str, model_dir: Path, mirrors: dict[str, Any],
                  kind: str = "visual", force: bool = False) -> str:
    """返回可直接给 from_pretrained / WhisperModel 使用的本地目录；失败则返回原 repo id。"""
    model_dir = Path(model_dir)
    cache = _load_cache(model_dir)
    key = f"{kind}:{repo_id}"
    if not force and key in cache and _looks_complete(Path(cache[key]), kind):
        logger.info("使用已下载的模型: %s -> %s", repo_id, cache[key])
        return cache[key]

    sources = mirrors.get("whisper_sources" if kind == "whisper" else "model_sources") \
        or ["modelscope", "hf_mirror", "hf"]
    ms_map = mirrors.get("whisper_modelscope_map" if kind == "whisper" else "modelscope_map") or {}
    endpoint = mirrors.get("hf_endpoint") or "https://hf-mirror.com"
    allow = WHISPER_ALLOW if kind == "whisper" else VISUAL_ALLOW

    errors: list[str] = []
    for source in sources:
        try:
            if source == "modelscope":
                ms_id = ms_map.get(repo_id)
                if not ms_id:
                    continue
                logger.info("从 ModelScope（国内）下载 %s ...", ms_id)
                path = _modelscope_download(ms_id, model_dir / "modelscope", allow)
            elif source == "hf_mirror":
                logger.info("从 %s（国内镜像）下载 %s ...", endpoint, repo_id)
                path = _hf_download(repo_id, model_dir / "huggingface", endpoint, allow)
            else:
                logger.info("从 huggingface.co 官方源下载 %s ...", repo_id)
                path = _hf_download(repo_id, model_dir / "huggingface", HF_OFFICIAL, allow)
        except Exception as exc:
            errors.append(f"{source}: {str(exc)[:180]}")
            logger.warning("下载源 %s 失败：%s", source, str(exc)[:180])
            continue

        if _looks_complete(Path(path), kind):
            cache[key] = str(Path(path).resolve())
            _save_cache(model_dir, cache)
            logger.info("模型就绪（来源=%s）: %s", source, path)
            return cache[key]
        errors.append(f"{source}: 下载内容不完整 ({path})")

    logger.error("所有下载源均失败 %s: %s", repo_id, "; ".join(errors))
    return repo_id


def whisper_repo_id(size: str) -> str:
    """faster-whisper 官方 CTranslate2 权重仓库（Systran 官方账号）。"""
    if "/" in size or Path(size).is_dir():
        return size
    return f"Systran/faster-whisper-{size}"


def apply_pip_env(mirrors: dict[str, Any]) -> None:
    """让子进程里的 pip 也默认走国内镜像。"""
    pypi = mirrors.get("pypi") or []
    if pypi:
        os.environ.setdefault("PIP_INDEX_URL", pypi[0])
        if len(pypi) > 1:
            os.environ.setdefault("PIP_EXTRA_INDEX_URL", " ".join(pypi[1:]))
