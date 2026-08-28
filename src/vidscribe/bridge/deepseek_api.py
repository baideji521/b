"""直接调 DeepSeek API 拿高光 JSON——不开浏览器、不用扩展。

接口是 OpenAI 那一套（POST /chat/completions，Bearer 鉴权），所以这里手写请求就够，
不引 openai 包。和 gemini_api 一样只用标准库 urllib。

JSON 模式：response_format={"type":"json_object"}。deepseek-reasoner 不吃这个参数，
所以只给非 reasoner 模型带上；抠 JSON 那步照样兜一层（见 gemini_api.extract_json）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekError(RuntimeError):
    """调用失败：网络、鉴权、余额不足、模型名写错都归这里。"""


def ask(api_key: str, prompt_text: str, merged_text: str, message: str = "",
        model: str = DEFAULT_MODEL, timeout: float = 300.0,
        base_url: str = DEFAULT_BASE_URL) -> str:
    """把提示词和合并文本发给 DeepSeek，返回回答正文。

    提示词当 system，合并文本 + 那句话当 user——等价于网页版挂两个 txt 附件，
    服务端反正也是当纯文本读。
    """
    if not api_key:
        raise DeepSeekError("没有 API key。去 https://platform.deepseek.com/api_keys 领一个，"
                            "填到 config.json 的 bridge.deepseek.api_key，"
                            "或设环境变量 DEEPSEEK_API_KEY")
    user_parts = [merged_text]
    if message.strip():
        user_parts.append(message.strip())
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        # 高光筛选要的是确定性，不要它每次换一套答案
        "temperature": 0.2,
        "stream": False,
    }
    if "reasoner" not in model:
        body["response_format"] = {"type": "json_object"}
    endpoint = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/chat/completions"
    request = urllib.request.Request(  # noqa: S310 - 地址来自配置里的 https 基址
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        hint = "（402 是账户余额不足）" if exc.code == 402 else ""
        raise DeepSeekError(f"HTTP {exc.code}{hint}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise DeepSeekError(f"连不上（{exc.reason}）") from exc
    except json.JSONDecodeError as exc:
        raise DeepSeekError(f"返回的不是 JSON：{exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        detail = str(payload.get("error") or payload)[:300]
        raise DeepSeekError(f"没有返回内容：{detail}")
    text = str((choices[0].get("message") or {}).get("content") or "").strip()
    if not text:
        finish = choices[0].get("finish_reason") or ""
        raise DeepSeekError(f"回答是空的{f'（finish_reason={finish}）' if finish else ''}")
    return text
