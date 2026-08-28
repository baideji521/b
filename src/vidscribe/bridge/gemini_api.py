"""直接调 Gemini API 拿高光 JSON——不开浏览器、不用扩展。

网页版那条路（扩展控制 gemini.google.com）毛病太多：上传控件不稳定、窗口被盖住页面
就被浏览器冻结、发送和读回答全靠猜 DOM。这里换成官方 HTTP 接口，一次请求一个回答，
纯后台，可重试，出错有明确的状态码。

只用标准库 urllib，不引第三方依赖。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiError(RuntimeError):
    """调用失败：网络、鉴权、配额、模型名写错都归这里。"""


def ask(api_key: str, prompt_text: str, merged_text: str, message: str = "",
        model: str = DEFAULT_MODEL, timeout: float = 300.0) -> str:
    """把提示词和合并文本发给 Gemini，返回回答正文。

    提示词和正文当成两段 text 发过去，效果等价于网页版挂两个 txt 附件——
    附件在服务端也是被当文本读的，没必要走 File API 多绕一趟。
    """
    if not api_key:
        raise GeminiError("没有 API key。去 https://aistudio.google.com/apikey 领一个，"
                          "填到 config.json 的 bridge.api_key，或设环境变量 GEMINI_API_KEY")
    parts = [{"text": prompt_text}, {"text": merged_text}]
    if message.strip():
        parts.append({"text": message.strip()})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        # 高光筛选要的是确定性，不要它每次换一套答案
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(  # noqa: S310 - 地址是写死的 https 常量
        ENDPOINT.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise GeminiError(f"HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise GeminiError(f"连不上（{exc.reason}）。这个接口在国内要走代理") from exc
    except json.JSONDecodeError as exc:
        raise GeminiError(f"返回的不是 JSON：{exc}") from exc

    candidates = payload.get("candidates") or []
    if not candidates:
        # 被安全策略拦下时没有 candidates，原因在 promptFeedback 里
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError(f"没有返回内容{f'（{reason}）' if reason else ''}")
    chunks = [str(p.get("text") or "")
              for p in (candidates[0].get("content") or {}).get("parts") or []]
    text = "".join(chunks).strip()
    if not text:
        finish = candidates[0].get("finishReason") or ""
        raise GeminiError(f"回答是空的{f'（finishReason={finish}）' if finish else ''}")
    return text


def extract_json(text: str) -> dict | None:
    """从回答里抠 JSON：先 ```json 围栏，再退回第一个配平的 {...}。

    responseMimeType 已经要求纯 JSON，但模型偶尔还是会加围栏，所以照样兜一层。
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence = text.find("```")
    if fence >= 0:
        rest = text[fence + 3:]
        if rest[:4].lower() == "json":
            rest = rest[4:]
        end = rest.find("```")
        try:
            parsed = json.loads(rest[:end if end >= 0 else None].strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None
