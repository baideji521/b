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
    docs = extract_json_list(text)
    return docs[0] if docs else None


def extract_json_list(text: str) -> list[dict]:
    """从回答里抠出**所有** JSON 对象，向下兼容：只回一份就一份，回多份全带走。

    认四种形状：整段纯 JSON（responseMimeType 的正常情况）、```json 围栏（一块或多块）、
    顶层数组（[ {...}, {...} ]，一个元素一份方案）、正文里挨着的几个配平 {...}。
    AI 有时会一口气给几套高光方案——以前只拿第一份，剩下全扔了；
    现在全带回去，入库时各存一行、方案名自动排开，谁也不覆盖谁。
    完全相同的两份只留第一份（一个字不差的重复入库没有意义）。
    """
    if not text:
        return []
    out: list[dict] = []

    def _push(value: object) -> None:
        if isinstance(value, dict):
            if value not in out:
                out.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item not in out:
                    out.append(item)

    try:
        _push(json.loads(text))
        if out:
            return out
    except json.JSONDecodeError:
        pass

    fence = text.find("```")
    while fence >= 0:
        body_start = fence + 3
        if text[body_start:body_start + 4].lower() == "json":
            body_start += 4
        close = text.find("```", body_start)
        body = text[body_start:close if close >= 0 else len(text)].strip()
        try:
            _push(json.loads(body))
        except json.JSONDecodeError:
            pass
        fence = text.find("```", close + 3) if close >= 0 else -1
    if out:
        return out

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = escape = False
        end = -1
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
                    end = i
                    break
        if end < 0:
            break
        try:
            _push(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            pass
        start = text.find("{", end + 1)
    return out
