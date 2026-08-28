// 好帮手唯一的活儿：从 AI_剪辑师 领任务 → 把两个 txt 上传到 Gemini → 等回答 →
// 把 JSON 回传给 AI_剪辑师。
//
//   GET  /v1/ai/next?types=gemini_json   领任务（文件清单 + 要说的那句话）
//   GET  /v1/ai/file?task_id=..&index=N  取 txt 本体（prm_en.txt / *_merged.txt）
//     → gemini.google.com 已经开着就用那个窗口，没开才新建，等输入框出现
//     → 把两个 txt 依次「拖」进页面（模仿手动拖放，拖不成再退到粘贴 / file 控件），
//       每个都等页面认账


//     → 输入那句话并发送
//     → 等回答停止增长
//     → 从回答里抠出 JSON
//   POST /v1/ai/progress                 回报阶段，响应里带 cancelled 就中断
//   POST /v1/ai/result                   回传原文 + 解析出的 JSON
//
// 底线：解析全在这里做，拿不到就如实报错，绝不伪造成功、绝不猜一个 JSON 出来。

import { loadBridgeConfig, trimEndpoint, autoPair, handleUnauthorized } from "./bridge-client.js";

const LOG_PREFIX = "[AI剪辑师好帮手]";

export const AI_POLL_ALARM = "AI剪辑师好帮手-ai-poll";
export const AI_TASK_TYPES = "gemini_json";

const POLL_IDLE_MS = 2000;
const POLL_BACKOFF_MS = 10000;
const BRIDGE_TIMEOUT_MS = 30000;
// 页面就绪 / 附件挂上 / 回答完成的等待上限
const READY_TIMEOUT_MS = 60000;
// 塞文件的方式，按顺序试：模仿手动拖放 -> 粘贴 -> 塞 file 控件
// 塞文件的顺序：一次拖两个 -> 逐个拖 -> 粘贴 -> 塞 file 控件（见 handleAiTask 里的 plans）

// 拖进去之后等这一个出卡片的时间；手动也就一秒多，超了就换下一种方式
const ATTACH_VERIFY_MS = 5000;
// 都挂上之后再确认一下没在转圈，很短；到点就直接按回车
const SETTLE_TIMEOUT_MS = 3000;


// 半自动模式等你手动把文件选进去的上限
const MANUAL_TIMEOUT_MS = 600000;
// 发出去这么久还没有新回答，就再点一次发送（后台标签页第一次可能没吃进去）
const RESEND_AFTER_MS = 12000;



const ANSWER_TIMEOUT_MS = 600000;

// 回答连续这么久没有变化就算写完了
const STABLE_MS = 3000;
const POLL_ANSWER_MS = 1500;

let polling = false;
let busy = false;
let lastPollState = "";

const log = (...args) => console.info(LOG_PREFIX, ...args);
const warn = (...args) => console.warn(LOG_PREFIX, ...args);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function notePollState(state) {
  if (state !== lastPollState) {
    lastPollState = state;
    log("轮询状态：" + state);
  }
}

// ---------------------------------------------------------------------------
// 与 AI_剪辑师 的 Bridge 通信
// ---------------------------------------------------------------------------

async function bridgeFetch(path, { method = "GET", body = null, timeoutMs = BRIDGE_TIMEOUT_MS } = {}) {
  const config = await loadBridgeConfig();
  const endpoint = trimEndpoint(config.endpoint);
  const token = typeof config.token === "string" ? config.token.trim() : "";
  if (!endpoint) return { ok: false, reason: "missing-endpoint" };
  if (!token) return { ok: false, reason: "missing-token" };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const init = { method, headers: { Authorization: `Bearer ${token}` }, signal: controller.signal };
    if (body !== null) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const response = await fetch(`${endpoint}${path}`, init);
    if (response.status === 401 || response.status === 403) {
      await handleUnauthorized();
      return { ok: false, reason: "unauthorized", status: response.status };
    }
    return { ok: response.ok, status: response.status, response };
  } catch (error) {
    return { ok: false, reason: "fetch-failed", message: error?.message ?? String(error) };
  } finally {
    clearTimeout(timer);
  }
}

async function bridgeJson(path, options) {
  const result = await bridgeFetch(path, options);
  if (!result.ok) return result;
  const body = await result.response.json().catch(() => null);
  return { ok: true, body };
}

async function reportProgress(taskId, stage, message = "") {
  const result = await bridgeJson("/v1/ai/progress", {
    method: "POST",
    body: { task_id: taskId, stage, message },
  });
  // 顺带回传 cancelled，用来在 GUI 点「停止 AI」后及时中断
  return Boolean(result.ok && result.body?.cancelled);
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  // 一次转太多会爆栈，分块拼
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** 把任务里登记的 txt 全下来，转成能塞进页面的 base64。 */
async function downloadTaskFiles(task) {
  const files = Array.isArray(task.files) ? task.files : [];
  const payloads = [];
  for (const item of files) {
    const result = await bridgeFetch(item.url, { timeoutMs: BRIDGE_TIMEOUT_MS });
    if (!result.ok) {
      throw new Error(`取文件失败：${item.name}（${result.reason || result.status}）`);
    }
    const buffer = await result.response.arrayBuffer();
    payloads.push({
      name: item.name,
      mime: "text/plain",
      b64: toBase64(buffer),
      size: buffer.byteLength,
    });
  }
  return payloads;
}

// ---------------------------------------------------------------------------
// 注入到 Gemini 页面里执行的函数（必须自包含，不能引用模块作用域的东西）
// ---------------------------------------------------------------------------

/** 找输入框。Gemini 是 Quill 富文本，退而求其次找任意 contenteditable。 */
function pageProbeEditor() {
  const selectors = [
    "div.ql-editor[contenteditable='true']",
    "rich-textarea div[contenteditable='true']",
    "[contenteditable='true']",
    "textarea",
  ];
  for (const selector of selectors) {
    if (document.querySelector(selector)) return { ok: true, selector };
  }
  return { ok: false, selector: null };
}

/**
 * 把 txt 塞进页面，默认走「模仿手动拖进去」——页面里能手动拖，就用同一条路。
 * mode: drop（拖放，默认）/ paste（粘贴）/ input（塞 file 控件，兜底）。
 */
function pageAttachFiles(payloads, mode) {
  const transfer = new DataTransfer();
  for (const item of payloads) {
    const binary = atob(item.b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    transfer.items.add(new File([bytes], item.name, { type: item.mime || "text/plain" }));
  }
  try {
    transfer.dropEffect = "copy";
    transfer.effectAllowed = "all";
  } catch {}

  const editor = document.querySelector(
    "div.ql-editor[contenteditable='true'], rich-textarea div[contenteditable='true'], "
    + "[contenteditable='true'], textarea"
  );

  if (mode === "drop" || !mode) {
    const root = document.documentElement;
    if (!document.body && !root) return { ok: false, error: "页面还没渲染出可拖放的区域" };
    // 手动往页面任何一处拖都能进，所以直接砸整页：body / html / document，
    // 再顺带砸一遍输入框那几层，谁监听谁收
    const targets = [];
    for (let node = editor; node && targets.length < 3; node = node.parentElement) {
      targets.push(node);
    }
    if (document.body) targets.push(document.body);
    if (root) targets.push(root);
    targets.push(document);
    if (!targets.length) return { ok: false, error: "找不到拖放目标" };

    const box = document.body || root;
    const rect = box.getBoundingClientRect();
    const point = { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
    let fired = 0;
    for (const target of targets) {
      for (const type of ["dragenter", "dragover", "drop"]) {
        try {
          target.dispatchEvent(new DragEvent(type, {
            bubbles: true, cancelable: true, composed: true, dataTransfer: transfer, ...point,
          }));
          fired += 1;
        } catch {}
      }
    }
    return fired
      ? { ok: true, via: "drop", targets: targets.length, count: transfer.files.length }
      : { ok: false, error: "拖放事件派发失败" };
  }


  if (mode === "paste") {
    if (!editor) return { ok: false, error: "找不到输入框，没法模拟粘贴" };
    editor.focus();
    try {
      editor.dispatchEvent(new ClipboardEvent("paste", {
        bubbles: true, cancelable: true, composed: true, clipboardData: transfer,
      }));
      return { ok: true, via: "paste", count: transfer.files.length };
    } catch (error) {
      return { ok: false, error: `模拟粘贴失败：${error?.message || error}` };
    }
  }

  // input 兜底：连 shadow root 一起翻，Angular Material 常把控件藏在里面
  const inputs = [];
  const walk = (root) => {
    if (!root?.querySelectorAll) return;
    inputs.push(...root.querySelectorAll("input[type='file']"));
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document);
  // 优先挑没限制 accept 或明确收 text 的那个
  const input =
    inputs.find((el) => {
      const accept = (el.getAttribute("accept") || "").toLowerCase();
      return !accept || accept.includes("text") || accept.includes("*/*") || accept.includes(".txt");
    }) || inputs[0];
  if (!input) return { ok: false, error: "页面上找不到文件上传控件" };
  input.files = transfer.files;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
  return { ok: true, via: "input", count: input.files.length,
           accept: input.getAttribute("accept") || "" };
}


/** 点一下「+ / 上传文件」把上传控件催出来。返回点了什么、现在有几个 file input。 */
function pageOpenUploadMenu() {
  const wanted = /添加照片和文件|上传文件|添加文件|attach|upload|add files/i;
  // 云端硬盘那条会弹 Google Drive 选择器，纯挡路；发送/录音也别碰
  const skip = /云端|硬盘|drive|发送|send|停止|stop|麦克风|mic|语音|录音|图片生成|制作/i;
  const clicked = [];
  const nodes = Array.from(
    document.querySelectorAll("button, [role='button'], [role='menuitem']")
  );
  for (const node of nodes) {
    const label = `${node.getAttribute("aria-label") || ""} `
      + `${node.getAttribute("mattooltip") || ""} ${node.textContent || ""}`;
    if (!wanted.test(label) || skip.test(label) || node.disabled) continue;
    try {
      node.click();
      clicked.push(label.trim().replace(/\s+/g, " ").slice(0, 24));
    } catch {}
    if (clicked.length >= 2) break;
  }
  return { clicked, inputs: document.querySelectorAll("input[type='file']").length };
}


/**
 * 关掉挡在前面的浮层：那个「+」菜单开着会盖住输入框，回车也发不出去。
 * 按 Esc + 点一下遮罩，都是手动关它的办法。
 */
function pageCloseOverlays() {
  const key = {
    key: "Escape", code: "Escape", keyCode: 27, which: 27,
    bubbles: true, cancelable: true, composed: true,
  };
  document.dispatchEvent(new KeyboardEvent("keydown", key));
  document.dispatchEvent(new KeyboardEvent("keyup", key));
  let backdrops = 0;
  for (const el of document.querySelectorAll(
    ".cdk-overlay-backdrop, .mat-mdc-menu-backdrop, [class*='overlay-backdrop']"
  )) {
    try {
      el.click();
      backdrops += 1;
    } catch {}
  }
  const menus = document.querySelectorAll("[role='menu'], .mat-mdc-menu-panel").length;
  return { backdrops, menus };
}



/**
 * 数页面上这个文件出现了几次。Gemini 的附件卡片只写主名（prm_en.txt 显示成 prm_en，
 * 后缀是单独的 TXT 角标），名字太长还会截断，所以按「全名 → 去后缀 → 名字前缀」
 * 三级放宽去数，哪一级数到就用哪一级。复用的窗口里可能有上一轮的同名附件，
 * 所以只看「有没有」会误判，必须比塞之前后的次数。
 */
function pageCountAttachment(name) {
  // 后台标签页可能不做布局，innerText 会是空的，退回 textContent
  const body = document.body;
  const text = body ? body.innerText || body.textContent || "" : "";

  const count = (needle) => {
    if (!needle) return 0;
    let hit = 0;
    for (let i = text.indexOf(needle); i >= 0; i = text.indexOf(needle, i + needle.length)) hit += 1;
    return hit;
  };
  const stem = name.replace(/\.[^.]+$/, "");
  // 卡片上的名字会被截断成「2026082619....」，所以前缀要短一点才认得出
  const needles = [name, stem, stem.slice(0, 10), stem.slice(0, 8)];
  let matched = 0;
  let used = "";
  for (const needle of needles) {
    matched = count(needle);
    if (matched > 0) {
      used = needle;
      break;
    }
  }
  // 兜底信号：卡片左上角那个类型角标（TXT），数它有几个
  const ext = (name.match(/\.([^.]+)$/) || ["", ""])[1].toUpperCase();
  const chips = ext ? count(ext) : 0;
  const failed = /上传失败|上传出错|failed to upload|unsupported file/i.test(text);
  return { count: matched, used, chips, failed };
}




/** 附件是不是都加载完了：还在转圈 / 还写着「上传中」就不算完，这时候按回车会白发。 */
function pageUploadSettled() {
  const body = document.body;
  const text = body ? body.innerText || body.textContent || "" : "";

  const pending = /上传中|正在上传|处理中|uploading|processing/i.test(text);
  const spinner = document.querySelectorAll(
    "mat-progress-bar, mat-spinner, mat-progress-spinner, [role='progressbar']"
  ).length;
  return { settled: !pending && spinner === 0, pending, spinner };
}


/**
 * 附件挂稳之后发送。
 *
 * 两个坑都在「页面没有键盘焦点」上：
 * 1. execCommand("insertText") 要求文档有焦点，后台窗口里直接返回 false，
 *    所以没焦点时改成写 DOM + 派发 InputEvent，Angular 照样收得到。
 * 2. 回车事件在没焦点的页面常被忽略，所以优先点发送按钮（点击不需要焦点）。
 * 按钮是 Angular 按输入内容动态启用的，第一次拿到可能还是 disabled，所以要重试几次。
 * 返回值带上诊断字段，日志里能看出到底卡在哪一步。
 */
async function pageSendMessage(selector, text) {
  const editor = document.querySelector(selector);
  if (!editor) return { ok: false, error: "输入框不见了" };
  const nap = (ms) => new Promise((done) => setTimeout(done, ms));
  const focused = document.hasFocus();
  const message = typeof text === "string" ? text : "";
  const content = () => (editor.tagName === "TEXTAREA" ? editor.value : editor.textContent || "");

  if (message) {
    editor.focus();
    if (editor.tagName === "TEXTAREA") {
      editor.value = message;
      editor.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: false }));
    } else {
      let typed = false;
      if (focused) {
        try {
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(editor);
          selection.removeAllRanges();
          selection.addRange(range);
          typed = document.execCommand("insertText", false, message);
        } catch {}
      }
      if (!typed || !content().includes(message.slice(0, 12))) {
        // 富文本控件靠 input 事件同步自己的模型，光改 DOM 不发事件按钮不会亮
        editor.textContent = message;
        editor.dispatchEvent(new InputEvent("beforeinput", {
          bubbles: true, cancelable: true, inputType: "insertText", data: message,
        }));
        editor.dispatchEvent(new InputEvent("input", {
          bubbles: true, cancelable: false, inputType: "insertText", data: message,
        }));
      }
    }
  }

  const findSend = () => {
    const buttons = Array.from(document.querySelectorAll(
      "button.send-button, button, [role='button']"
    ));
    return buttons.find((b) => {
      const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""} `
        + `${b.className || ""} ${b.querySelector("mat-icon")?.getAttribute("fonticon") || ""}`;
      if (!/发送|send|提交|submit/i.test(label)) return false;
      return !/停止|stop|取消|cancel|录音|mic/i.test(label);
    }) || null;
  };

  let button = null;
  let disabled = false;
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    button = findSend();
    if (button) {
      disabled = button.disabled || button.getAttribute("aria-disabled") === "true";
      if (!disabled) {
        button.click();
        await nap(300);
        return {
          ok: true, sent: "button", typed: message.length, focused,
          editorLen: content().trim().length, attempts: attempt,
        };
      }
      // 按钮还灰着：再补一次 input 事件催 Angular 重算状态
      editor.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: false }));
    }
    await nap(250);
  }

  const key = {
    key: "Enter", code: "Enter", keyCode: 13, which: 13,
    bubbles: true, cancelable: true, composed: true,
  };
  editor.dispatchEvent(new KeyboardEvent("keydown", key));
  editor.dispatchEvent(new KeyboardEvent("keypress", key));
  editor.dispatchEvent(new KeyboardEvent("keyup", key));
  await nap(300);
  return {
    ok: true, sent: "enter", typed: message.length, focused,
    editorLen: content().trim().length,
    button: button ? (disabled ? "灰着" : "没点上") : "没找到",
  };
}



/** 读最后一条回答的纯文本，并判断是否还在写。 */
function pageReadAnswer() {
  const groups = [
    "message-content.model-response-text",
    ".model-response-text",
    "model-response",
    ".markdown",
  ];
  let nodes = [];
  let used = "";
  for (const selector of groups) {
    nodes = Array.from(document.querySelectorAll(selector));
    if (nodes.length) {
      used = selector;
      break;
    }
  }
  const last = nodes[nodes.length - 1] || null;
  // 后台标签页不做布局时 innerText 是空的，退回 textContent
  const text = last ? (last.innerText || last.textContent || "").trim() : "";

  // 还在生成时页面上有「停止」按钮
  const streaming = Array.from(document.querySelectorAll("button")).some((b) => {
    const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""}`;
    return /stop|停止/i.test(label);
  });
  // 一个块都没有时，报一下整页文本长度：0 说明页面被冻结/还没渲染，不是选择器写错
  const bodyLen = (document.body?.textContent || "").length;
  return { text, streaming, blocks: nodes.length, used, bodyLen };
}


// ---------------------------------------------------------------------------
// 标签页操作
// ---------------------------------------------------------------------------

async function runInTab(tabId, fn, args = []) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId },
    func: fn,
    args,
    world: "MAIN",
  });
  return result?.result ?? null;
}

/**
 * 把 Gemini 挪到一个自己的小窗口里，不抢焦点。
 *
 * 关键在于「后台标签页」和「不在最前面的窗口」是两码事：藏在别的标签页后面的标签页
 * 会被 Chrome 冻结——不排版、不跑定时器，拖放和读回答全废；而独立窗口里的活动标签页
 * 照常渲染，只是没有键盘焦点（所以发送要点按钮，不能靠回车）。
 * 窗口开得小、贴右下角，挡不着你干活。
 */
async function moveToSideWindow(tabId) {
  const anchor = await chrome.windows.getLastFocused({}).catch(() => null);
  const bounds = { width: 560, height: 620 };
  if (anchor && typeof anchor.left === "number" && typeof anchor.width === "number") {
    bounds.left = Math.max(0, anchor.left + anchor.width - bounds.width - 24);
    bounds.top = Math.max(0, (anchor.top || 0) + Math.max(0, (anchor.height || 900) - bounds.height - 48));
  }
  const win = await chrome.windows.create({ tabId, focused: false, ...bounds }).catch((err) => {
    log("挪窗口失败，先凑合用原来那个", err?.message || err);
    return null;
  });
  return win?.id ?? null;
}

/**
 * 先看 gemini.google.com 是不是已经开着：开着就直接用，绝不动它的 URL，
 * 也不等页面加载（省掉重新加载那几秒）。没开才新建一个。
 *
 * sideWindow=true（默认）时保证它是自己窗口里的活动标签页——不然一被别的标签页盖住，
 * 页面就被冻结，上传上去也发不出去、回答也读不出来。
 * 返回 { tabId, created, ready }，created=false 的标签页是用户自己的，事后不许关。
 */
async function ensureGeminiTab(url, { focus = false, sideWindow = true } = {}) {
  const opened = await chrome.tabs.query({ url: "*://gemini.google.com/*" }).catch(() => []);
  const existing = Array.isArray(opened) ? opened.find((tab) => typeof tab.id === "number") : null;
  if (existing) {
    log("Gemini 已经开着，直接用", existing.id, existing.status || "");
    if (focus) {
      await chrome.tabs.update(existing.id, { active: true }).catch(() => {});
      if (typeof existing.windowId === "number") {
        await chrome.windows.update(existing.windowId, { focused: true }).catch(() => {});
      }
    } else if (sideWindow && !existing.active) {
      // 被别的标签页盖着 = 冻结状态，拽出来单开一个窗口，不抢焦点
      log("它被别的标签页盖着，挪到单独的小窗口");
      await moveToSideWindow(existing.id);
    }
    return { tabId: existing.id, created: false, ready: existing.status === "complete" };
  }
  if (!focus && sideWindow) {
    log("Gemini 没打开，开个不抢焦点的小窗口", url);
    const win = await chrome.windows.create({ url, focused: false, width: 560, height: 620 })
      .catch(() => null);
    const tab = win?.tabs?.[0];
    if (tab && typeof tab.id === "number") {
      return { tabId: tab.id, created: true, ready: false };
    }
  }
  log("Gemini 没打开，新建标签页", url, focus ? "" : "（后台打开）");
  const tab = await chrome.tabs.create({ url, active: focus });
  return { tabId: tab.id, created: true, ready: false };
}



async function waitForTabComplete(tabId, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    if (!tab) return false;
    if (tab.status === "complete") return true;
    if (Date.now() > deadline) return false;
    await sleep(400);
  }
}

/** 从 AI 回答里抠 JSON：先 ```json 围栏，再退回第一个配平的 {...}。 */
export function extractJson(text) {
  if (!text) return null;
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      const parsed = JSON.parse(fenced[1].trim());
      if (parsed && typeof parsed === "object") return parsed;
    } catch {}
  }
  for (let start = text.indexOf("{"); start >= 0; start = text.indexOf("{", start + 1)) {
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (inString) {
        if (escape) escape = false;
        else if (ch === "\\") escape = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          try {
            const parsed = JSON.parse(text.slice(start, i + 1));
            if (parsed && typeof parsed === "object") return parsed;
          } catch {}
          break;
        }
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// 任务执行
// ---------------------------------------------------------------------------

async function handleAiTask(task) {
  const taskId = task.task_id;
  const url = task.url || "https://gemini.google.com/app";
  const message = String(task.message || "Reply with the JSON object only.");
  // manual = 半自动：文件你自己选进去，剩下的（发送、等回答、抠 JSON、回传）扩展来
  const uploadMode = String(task.upload_mode || "manual");
  // 默认死活不抢焦点；要它自己跳到前台就在 config.json 里把 bridge.focus_browser 打开
  const focusBrowser = Boolean(task.focus_browser);
  // 独立小窗口：不抢焦点但保证页面在渲染（后台标签页会被冻结，什么都干不了）
  const sideWindow = task.side_window === undefined ? true : Boolean(task.side_window);



  let tabId = null;
  let createdTab = false;

  const finish = async (payload) => {
    // 只关自己开的标签页；用户本来就开着的 Gemini 窗口一律留着
    if (payload.status === "completed" && tabId && createdTab) {
      await chrome.tabs.remove(tabId).catch(() => {});
    }
    await bridgeJson("/v1/ai/result", { method: "POST", body: { task_id: taskId, ...payload } });
  };

  const cancelled = async (stage, text) => {
    if (await reportProgress(taskId, stage, text)) {
      await finish({ status: "failed", error: "任务已被用户停止" });
      return true;
    }
    return false;
  };

  try {
    const fileList = Array.isArray(task.files) ? task.files : [];
    if (!fileList.length) {
      return finish({ status: "failed", error: "任务里没有要上传的文件" });
    }
    // 半自动模式文件由你自己选，扩展不用把内容取过来
    let payloads = [];
    if (uploadMode === "auto") {
      if (await cancelled("downloading", "取文件")) return;
      payloads = await downloadTaskFiles(task);
      log("取到文件", payloads.map((p) => `${p.name} ${p.size}B`).join(", "));
    } else {
      log("半自动模式：等你手动选文件", fileList.map((f) => f.name).join(", "));
    }


    if (await cancelled("opening", `打开 ${url}`)) return;
    const target = await ensureGeminiTab(url, { focus: focusBrowser, sideWindow });


    tabId = target.tabId;
    createdTab = target.created;
    // 已经开着而且加载完的，直接干，不等；只有新开的或还在转的才等页面 complete
    if (!target.ready) await waitForTabComplete(tabId, READY_TIMEOUT_MS);


    // 新开的页面 Angular 还要渲染一会儿；复用的页面通常第一次探测就命中
    let editor = null;
    const readyDeadline = Date.now() + READY_TIMEOUT_MS;
    while (Date.now() < readyDeadline) {
      editor = await runInTab(tabId, pageProbeEditor).catch(() => null);
      if (editor?.ok) break;
      if (await cancelled("waiting_editor", "等输入框出现")) return;
      await sleep(400);
    }

    if (!editor?.ok) {
      return finish({ status: "failed", error: "找不到输入框，可能没登录或页面结构变了" });
    }

    // 记下每个文件名现在在页面上出现几次，之后靠「多了一次」判断挂没挂上。
    // 两个文件是一起拖进去的，所以只要认出 prm_en 那个就说明都进去了，
    // 第二个名字太长会被截断成「2026082619....」，本来就认不准，不拿它当门槛。
    const names = (payloads.length ? payloads : fileList).map((f) => f.name);
    const probe = names.slice(0, 1);

    const baselines = {};
    let chipsBase = 0;
    for (const name of names) {
      const seen = await runInTab(tabId, pageCountAttachment, [name]).catch(() => null);
      baselines[name] = Number(seen?.count || 0);
      chipsBase = Math.max(chipsBase, Number(seen?.chips || 0));
    }



    let autoDone = false;
    if (uploadMode === "auto") {
      // 等某几个文件出卡片；出来就返回 true
      const waitCards = async (wanted, ms) => {
        const deadline = Date.now() + ms;
        let left = wanted.slice();
        while (Date.now() < deadline) {
          const still = [];
          let chipsNow = 0;
          for (const name of left) {
            const check = await runInTab(tabId, pageCountAttachment, [name]).catch(() => null);
            if (check?.failed) throw new Error(`页面提示上传失败（${name}）`);
            chipsNow = Math.max(chipsNow, Number(check?.chips || 0));
            if (Number(check?.count || 0) > baselines[name]) log("已挂上", name, `匹配=${check?.used || ""}`);
            else still.push(name);
          }
          left = still;
          if (!left.length) return true;
          // 名字被截断得认不出时，用类型角标数兜底：多出来的角标数够了就算挂上
          if (chipsNow - chipsBase >= wanted.length) {
            log("按类型角标认账", `角标 ${chipsBase} -> ${chipsNow}`);
            return true;
          }
          await sleep(300);
        }
        log("还没出卡片", left.join("、"));
        return false;
      };


      // 你实测两个文件一起往页面任何一处拖就行，所以先一次拖两个；
      // 不行再一个一个拖，最后才退到粘贴 / file 控件
      const plans = [
        { mode: "drop", batch: true },
        { mode: "drop", batch: false },
        { mode: "paste", batch: true },
        { mode: "input", batch: true },
      ];
      for (let p = 0; p < plans.length; p += 1) {
        const plan = plans[p];
        // 卡片可能晚一点才冒出来，升级到下一种方式之前先复查一遍，别白折腾
        if (p > 0 && await waitCards(probe, 600)) {
          autoDone = true;
          break;
        }
        if (plan.mode === "input") {

          const menu = await runInTab(tabId, pageOpenUploadMenu).catch(() => null);
          log("催上传控件", JSON.stringify(menu || {}));
          await sleep(300);
        }
        try {
          if (plan.batch) {
            const attached = await runInTab(tabId, pageAttachFiles, [payloads, plan.mode]);
            if (!attached?.ok) {
              log("塞入失败", plan.mode, attached?.error || "未知原因");
              continue;
            }
            autoDone = await waitCards(probe, ATTACH_VERIFY_MS);
          } else {
            autoDone = true;
            for (const item of payloads) {
              const attached = await runInTab(tabId, pageAttachFiles, [[item], plan.mode]);
              if (!attached?.ok) {
                log("塞入失败", item.name, plan.mode, attached?.error || "未知原因");
                autoDone = false;
                break;
              }
              await sleep(1200);
            }
            if (autoDone) autoDone = await waitCards(probe, ATTACH_VERIFY_MS);
          }

        } catch (error) {
          return finish({ status: "failed",
                          error: `${error?.message || error}（可能不收 txt 或文件太大）` });
        }
        if (autoDone) {
          log(`${payloads.length} 个文件塞完了`, `方式=${plan.mode}`, plan.batch ? "一次拖" : "逐个拖");
          break;
        }

        log("换下一种方式", plan.mode, plan.batch ? "一次拖" : "逐个拖");
      }
    }



    if (!autoDone) {

      // 半自动：要你亲手选文件。只有开了 focus_browser 才把窗口拉到前台，
      // 否则安静等着，你自己切到 Gemini 标签页就行（AI_剪辑师 的日志里有文件路径）
      if (focusBrowser) {
        await chrome.tabs.update(tabId, { active: true }).catch(() => {});
        const tab = await chrome.tabs.get(tabId).catch(() => null);
        if (typeof tab?.windowId === "number") {
          await chrome.windows.update(tab.windowId, { focused: true }).catch(() => {});
        }
      }

      const menu = await runInTab(tabId, pageOpenUploadMenu).catch(() => null);
      log("等你手动选文件", JSON.stringify(menu || {}));

      const manualDeadline = Date.now() + MANUAL_TIMEOUT_MS;
      for (;;) {
        const missing = [];
        let chipsNow = 0;
        // 一起选的，认出第一个就够；第二个名字会被截断，认不准
        for (const name of probe) {
          const check = await runInTab(tabId, pageCountAttachment, [name]).catch(() => null);
          chipsNow = Math.max(chipsNow, Number(check?.chips || 0));
          if (Number(check?.count || 0) <= baselines[name]) missing.push(name);
        }
        if (!missing.length) {
          log("文件已经在页面上了");
          break;
        }
        // 名字截断认不出时用类型角标兜底，别把已经传好的又叫你传一遍
        if (chipsNow - chipsBase >= names.length) {
          log("按类型角标认账", `角标 ${chipsBase} -> ${chipsNow}`);
          break;
        }


        if (Date.now() > manualDeadline) {
          return finish({ status: "failed",
                          error: `等手动选文件超时，还差：${missing.join("、")}` });
        }
        if (await cancelled("waiting_manual", `请手动选这些文件：${missing.join("、")}`)) return;
        await sleep(2000);
      }
    }

    if (await cancelled("sending", "等附件加载完")) return;
    // 「+」菜单这类浮层会盖住输入框，回车也发不出去，先关掉
    const closed = await runInTab(tabId, pageCloseOverlays).catch(() => null);
    if (closed?.backdrops || closed?.menus) log("关掉浮层", JSON.stringify(closed));
    await sleep(300);


    // 附件还在转圈就按回车会白发一条，等页面彻底安静下来
    const settleDeadline = Date.now() + SETTLE_TIMEOUT_MS;
    for (;;) {
      const settle = await runInTab(tabId, pageUploadSettled).catch(() => null);
      if (settle?.settled) {
        log("附件加载完成");
        break;
      }
      if (Date.now() > settleDeadline) {
        log("等加载超时，仍然按回车试一次", JSON.stringify(settle || {}));
        break;
      }
      if (await cancelled("sending", "附件还在加载")) return;
      await sleep(500);
    }


    // 复用的窗口里可能已经有旧回答，先记下条数，别把旧的当成这次的结果
    const before = await runInTab(tabId, pageReadAnswer).catch(() => null);
    const baselineBlocks = Number(before?.blocks || 0);
    const sent = await runInTab(tabId, pageSendMessage, [editor.selector, message]);


    if (!sent?.ok) {
      return finish({ status: "failed", error: `发送失败：${sent?.error || "未知原因"}` });
    }
    log("已发送", sent.sent, JSON.stringify({
      typed: sent.typed, editorLen: sent.editorLen, focused: sent.focused,
      attempts: sent.attempts, button: sent.button,
    }));

    // 等回答：文本连续 STABLE_MS 不变且没有「停止」按钮就算写完
    let text = "";
    let stableSince = 0;
    let resent = false;
    const sentAt = Date.now();
    const answerDeadline = Date.now() + ANSWER_TIMEOUT_MS;
    while (Date.now() < answerDeadline) {
      await sleep(POLL_ANSWER_MS);
      const snapshot = await runInTab(tabId, pageReadAnswer).catch(() => null);
      // 新回答还没冒出来（条数没涨）就继续等，别读到窗口里原有的旧回答
      if (Number(snapshot?.blocks || 0) <= baselineBlocks) {
        // 后台标签页里第一次发送有可能没吃进去，等一会儿没反应就再点一次发送
        if (!resent && Date.now() - sentAt > RESEND_AFTER_MS) {
          resent = true;
          const again = await runInTab(tabId, pageSendMessage, [editor.selector, ""]).catch(() => null);
          log("没反应，再发一次", again?.sent || "失败", JSON.stringify({
            blocks: snapshot?.blocks, baseline: baselineBlocks,
            bodyLen: snapshot?.bodyLen, used: snapshot?.used,
          }));
        }
        if (await cancelled("waiting_answer", "等新回答出现")) return;
        continue;
      }

      const current = snapshot?.text || "";

      if (current && current === text && !snapshot?.streaming) {
        if (!stableSince) stableSince = Date.now();
        if (Date.now() - stableSince >= STABLE_MS) break;
      } else {
        stableSince = 0;
        text = current;
      }
      if (await cancelled("waiting_answer", `已收到 ${current.length} 字`)) return;
    }

    if (!text) {
      // 把最后一眼的现场情况带上：blocks=0 而 bodyLen>0 说明选择器没命中，
      // bodyLen=0 说明这个标签页压根没渲染（被冻结了）
      const last = await runInTab(tabId, pageReadAnswer).catch(() => null);
      const detail = JSON.stringify({
        blocks: last?.blocks, baseline: baselineBlocks,
        bodyLen: last?.bodyLen, used: last?.used, streaming: last?.streaming,
      });
      log("读不到回答", detail);
      return finish({ status: "failed", error: `AI 没有产出可读的回答 ${detail}` });
    }
    const parsed = extractJson(text);
    await reportProgress(taskId, "reporting", parsed ? "回传 JSON" : "回答里没有 JSON，原文回传");
    return finish({
      status: parsed ? "completed" : "failed",
      text,
      json: parsed,
      error: parsed ? "" : "回答里没有可解析的 JSON",
    });
  } catch (error) {
    const detail = error?.message ?? String(error);
    warn("任务执行异常", error);
    return finish({ status: "failed", error: `扩展异常：${detail}` });
  }
}

// ---------------------------------------------------------------------------
// 轮询循环
// ---------------------------------------------------------------------------

async function pollOnce() {
  if (busy) return POLL_IDLE_MS;

  const config = await loadBridgeConfig();
  if (!config.token) {
    notePollState("未配对：在 AI_剪辑师 里点「配对扩展」");
    await autoPair().catch(() => null);
    return POLL_IDLE_MS;
  }

  const response = await bridgeJson(`/v1/ai/next?types=${AI_TASK_TYPES}`, { timeoutMs: 8000 });
  if (!response.ok) {
    notePollState(`AI_剪辑师 不可达/出错：${response.reason || ""} ${response.status || ""}`);
    return POLL_BACKOFF_MS;
  }

  const task = response.body?.task;
  if (!task) {
    notePollState("已连上 AI_剪辑师，暂无任务");
    return POLL_IDLE_MS;
  }

  busy = true;
  notePollState("执行任务中");
  log("领取任务", task.task_id, task.video || "");
  try {
    await handleAiTask(task);
  } finally {
    busy = false;
    lastPollState = "";
  }
  return POLL_IDLE_MS;
}

async function pollLoop() {
  if (polling) return;
  polling = true;
  try {
    for (;;) {
      let delay = POLL_IDLE_MS;
      try {
        delay = await pollOnce();
      } catch (error) {
        warn("轮询异常", error);
        delay = POLL_BACKOFF_MS;
      }
      await sleep(delay);
    }
  } finally {
    polling = false;
  }
}

/** 启动轮询。alarm 只是 service worker 被回收后的唤醒兜底（最小周期 1 分钟）。 */
export function startAiPolling() {
  log(`任务轮询已启动（间隔 ${POLL_IDLE_MS}ms）`);
  try {
    chrome.alarms.create(AI_POLL_ALARM, { periodInMinutes: 1 });
  } catch (error) {
    warn("创建轮询闹钟失败", error);
  }
  pollLoop();
}

/** alarm 唤醒后重新拉起轮询循环 */
export function resumeAiPolling() {
  pollLoop();
}

/** 给 popup 用的状态快照 */
export function pollingStatus() {
  return { polling, busy, state: lastPollState };
}
