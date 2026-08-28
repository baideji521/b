"""AI 提供方注册表：Gemini / DeepSeek 两条路的差异全收在这儿。

上层（GUI、任务下发）只认提供方的名字，key 从哪读、模型叫什么、网页版开哪个网址、
接口直连调哪个函数，都问这个模块。加第三家就在 PROVIDERS 里加一条 + 写个 ask。

配置读法：
- gemini 用 bridge 里的老键（api_key / api_key_env / api_model），保证旧 config.json 照旧能跑；
- 其它提供方各占一个子节，比如 bridge.deepseek.api_key。
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PROVIDER = "gemini"

PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {
        "label": "Gemini",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "key_env": "GEMINI_API_KEY",
        "key_page": "https://aistudio.google.com/apikey",
        "ai_url": "https://gemini.google.com/app",
        # 老配置把 Gemini 的键平铺在 bridge 下，不搬，免得升级把 key 丢了
        "section": "",
        "base_url": "",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_env": "DEEPSEEK_API_KEY",
        "key_page": "https://platform.deepseek.com/api_keys",
        "ai_url": "https://chat.deepseek.com/",
        "section": "deepseek",
        "base_url": "https://api.deepseek.com",
    },
}


class AiError(RuntimeError):
    """接口直连失败。两家的具体异常都会被包成这个抛出来。"""


def normalize(name: str | None) -> str:
    """把配置里的提供方名字收敛到已知的那几个，不认识就退回默认。"""
    key = str(name or "").strip().lower()
    return key if key in PROVIDERS else DEFAULT_PROVIDER


def node(bridge: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """这家提供方的配置节原样返回（不读环境变量，给编辑器用）。"""
    provider = normalize(name if name is not None else bridge.get("provider"))
    section = PROVIDERS[provider]["section"]
    return dict(bridge.get(section) or {}) if section else dict(bridge)


def settings(bridge: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """把某个提供方要用的东西一次算清楚：key、模型、网址、超时。

    key 先看配置里写死的，为空才读环境变量——不想把 key 写进仓库的走环境变量。
    """
    provider = normalize(name if name is not None else bridge.get("provider"))
    spec = PROVIDERS[provider]
    section = spec["section"]
    node = dict(bridge.get(section) or {}) if section else dict(bridge)

    key = str(node.get("api_key") or "").strip()
    env_name = str(node.get("api_key_env") or spec["key_env"])
    if not key:
        key = os.environ.get(env_name, "").strip()
    timeout = node.get("api_timeout")
    if timeout is None:
        timeout = bridge.get("api_timeout")
    # 网页版网址：这家自己那一节写了就用它；gemini 还认 bridge.ai_url 这个老键
    ai_url = str(node.get("ai_url") or "").strip()
    if not ai_url and not section:
        ai_url = str(bridge.get("ai_url") or "").strip()
    return {
        "provider": provider,
        "label": spec["label"],
        "models": list(spec["models"]),
        "api_key": key,
        "api_key_env": env_name,
        "api_model": str(node.get("api_model") or spec["models"][0]),
        "api_timeout": float(timeout or 300.0),
        "base_url": str(node.get("base_url") or spec["base_url"]),
        "ai_url": ai_url or spec["ai_url"],
        "key_page": spec["key_page"],
    }


def section_for(name: str) -> str:
    """这家提供方的键该写到 bridge 下的哪一节：空字符串表示直接平铺在 bridge 里。"""
    return str(PROVIDERS[normalize(name)]["section"])


def ask(name: str, api_key: str, prompt_text: str, merged_text: str, message: str = "",
        model: str = "", timeout: float = 300.0, base_url: str = "") -> str:
    """按提供方分发到对应的接口实现，失败统一抛 AiError。"""
    provider = normalize(name)
    if provider == "deepseek":
        from . import deepseek_api  # noqa: PLC0415

        try:
            return deepseek_api.ask(api_key, prompt_text, merged_text, message,
                                    model or deepseek_api.DEFAULT_MODEL, timeout,
                                    base_url or deepseek_api.DEFAULT_BASE_URL)
        except deepseek_api.DeepSeekError as exc:
            raise AiError(str(exc)) from exc

    from . import gemini_api  # noqa: PLC0415

    try:
        return gemini_api.ask(api_key, prompt_text, merged_text, message,
                              model or gemini_api.DEFAULT_MODEL, timeout)
    except gemini_api.GeminiError as exc:
        raise AiError(str(exc)) from exc


def extract_json(text: str) -> dict | None:
    """抠 JSON 的逻辑两家共用，别复制第二份。"""
    from . import gemini_api  # noqa: PLC0415

    return gemini_api.extract_json(text)
