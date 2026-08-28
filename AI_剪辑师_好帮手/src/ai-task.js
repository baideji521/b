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
const ATTACH_MODES = ["drop", "paste", "input"];
// 每种方式塞完之后等页面认账的时间，超了就换下一种
const ATTACH_VERIFY_MS = 40000;
// 附件挂上之后等它加载完（转圈停掉）的上限，之后才按回车
const SETTLE_TIMEOUT_MS = 60000;
// 半自动模式等你手动把文件选进去的上限
const MANUAL_TIMEOUT_MS = 600000;


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
    if (!editor && !document.body) return { ok: false, error: "页面还没渲染出可拖放的区域" };
    // 手动拖文件时事件落在输入框那一片，从输入框往上逐层派发，谁监听谁收
    const targets = [];
    for (let node = editor; node && targets.length < 4; node = node.parentElement) {
      targets.push(node);
    }
    if (document.body) targets.push(document.body);
    if (!targets.length) return { ok: false, error: "找不到拖放目标" };

    const rect = (editor || document.body).getBoundingClientRect();
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
  const wanted = /上传|添加文件|添加图片|上传文件|attach|upload|add files/i;
  const skip = /发送|send|停止|stop|麦克风|mic|语音|录音/i;
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
 * 数页面上这个文件名出现了几次。复用的窗口里可能有上一轮的同名附件，
 * 所以只看「有没有」会误判，必须比上传前后的次数。
 */
function pageCountAttachment(name) {
  const text = document.body ? document.body.innerText || "" : "";
  let count = 0;
  for (let idx = text.indexOf(name); idx >= 0; idx = text.indexOf(name, idx + name.length)) count += 1;
  const failed = /上传失败|上传出错|failed to upload|unsupported file/i.test(text);
  return { count, failed };
}


/** 附件是不是都加载完了：还在转圈 / 还写着「上传中」就不算完，这时候按回车会白发。 */
function pageUploadSettled() {
  const text = document.body ? document.body.innerText || "" : "";
  const pending = /上传中|正在上传|处理中|uploading|processing/i.test(text);
  const spinner = document.querySelectorAll(
    "mat-progress-bar, mat-spinner, mat-progress-spinner, [role='progressbar']"
  ).length;
  return { settled: !pending && spinner === 0, pending, spinner };
}


/**
 * 附件挂稳之后发送：有那句话就先填进去，然后模仿按回车——手动就是这么发的，

 * 不去猜哪个是发送按钮（按钮 aria-label 一变就点错，回车不会）。
 */
function pageSendMessage(selector, text) {
  const editor = document.querySelector(selector);
  if (!editor) return { ok: false, error: "输入框不见了" };
  editor.focus();

  const message = typeof text === "string" ? text : "";
  if (message) {
    if (editor.tagName === "TEXTAREA") {
      editor.value = message;
      editor.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: false }));
    } else {
      // Quill 这类富文本要靠 beforeinput/input 才认，直接改 innerText 模型不更新，
      // 所以优先用 execCommand 走浏览器自己的插入路径
      let typed = false;
      try {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(editor);
        selection.removeAllRanges();
        selection.addRange(range);
        typed = document.execCommand("insertText", false, message);
      } catch {}
      if (!typed) {
        editor.innerText = message;
        editor.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: false }));
      }
    }
  }

  const key = {
    key: "Enter", code: "Enter", keyCode: 13, which: 13,
    bubbles: true, cancelable: true, composed: true,
  };
  editor.dispatchEvent(new KeyboardEvent("keydown", key));
  editor.dispatchEvent(new KeyboardEvent("keypress", key));
  editor.dispatchEvent(new KeyboardEvent("keyup", key));
  return { ok: true, sent: "enter", typed: message.length };
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
  for (const selector of groups) {
    nodes = Array.from(document.querySelectorAll(selector));
    if (nodes.length) break;
  }
  const last = nodes[nodes.length - 1] || null;
  const text = last ? (last.innerText || "").trim() : "";
  // 还在生成时页面上有「停止」按钮
  const streaming = Array.from(document.querySelectorAll("button")).some((b) => {
    const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""}`;
    return /stop|停止/i.test(label);
  });
  return { text, streaming, blocks: nodes.length };
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
 * 先看 gemini.google.com 是不是已经开着：开着就直接用那个窗口，没开才新建一个。
 * 返回 { tabId, created }，created=false 的标签页是用户自己的，事后不许关。
 */
async function ensureGeminiTab(url) {
  const opened = await chrome.tabs.query({ url: "*://gemini.google.com/*" }).catch(() => []);
  const existing = Array.isArray(opened) ? opened.find((tab) => typeof tab.id === "number") : null;
  if (existing) {
    log("复用已打开的 Gemini 窗口", existing.id);
    await chrome.tabs.update(existing.id, { active: true }).catch(() => {});
    if (typeof existing.windowId === "number") {
      await chrome.windows.update(existing.windowId, { focused: true }).catch(() => {});
    }
    return { tabId: existing.id, created: false };
  }
  log("Gemini 没打开，新建标签页", url);
  const tab = await chrome.tabs.create({ url, active: true });
  return { tabId: tab.id, created: true };
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
    const target = await ensureGeminiTab(url);
    tabId = target.tabId;
    createdTab = target.created;
    await waitForTabComplete(tabId, READY_TIMEOUT_MS);


    // 页面 complete 之后 Angular 还要再渲染一会儿，等输入框出现
    let editor = null;
    const readyDeadline = Date.now() + READY_TIMEOUT_MS;
    while (Date.now() < readyDeadline) {
      editor = await runInTab(tabId, pageProbeEditor).catch(() => null);
      if (editor?.ok) break;
      if (await cancelled("waiting_editor", "等输入框出现")) return;
      await sleep(1000);
    }
    if (!editor?.ok) {
      return finish({ status: "failed", error: "找不到输入框，可能没登录或页面结构变了" });
    }

    // 先记下每个文件名现在在页面上出现几次，之后靠「多了一次」判断挂没挂上
    const names = (payloads.length ? payloads : fileList).map((f) => f.name);

    const baselines = {};
    for (const name of names) {
      const seen = await runInTab(tabId, pageCountAttachment, [name]).catch(() => null);
      baselines[name] = Number(seen?.count || 0);
    }

    let autoDone = false;
    if (uploadMode === "auto") {
      autoDone = true;
      // 两个 txt 依次塞：先模仿手动拖，拖不成退到粘贴，最后才去塞 file 控件
      for (let i = 0; i < payloads.length && autoDone; i += 1) {
        const item = payloads[i];
        const label = `${i + 1}/${payloads.length} ${item.name}`;
        if (await cancelled("uploading", `上传 ${label}`)) return;
        const baseline = baselines[item.name];

        let attachedOk = false;
        let usedVia = "";
        let lastError = "";
        for (const mode of ATTACH_MODES) {
          if (mode === "input") {
            // 控件平时不在 DOM 里，先点一下「+ / 上传文件」把它催出来
            const menu = await runInTab(tabId, pageOpenUploadMenu).catch(() => null);
            log("催上传控件", JSON.stringify(menu || {}));
            await sleep(800);
          }
          const attached = await runInTab(tabId, pageAttachFiles, [[item], mode]).catch(() => null);
          if (!attached?.ok) {
            lastError = `${mode}：${attached?.error || "未知原因"}`;
            log("塞入失败", label, lastError);
            continue;
          }
          log("已塞入，等页面认账", label, `方式=${mode}`);

          const modeDeadline = Date.now() + ATTACH_VERIFY_MS;
          while (Date.now() < modeDeadline) {
            const check = await runInTab(tabId, pageCountAttachment, [item.name]).catch(() => null);
            if (check?.failed) {
              return finish({ status: "failed",
                              error: `页面提示上传失败（${label}，可能不收 txt 或文件太大）` });
            }
            if (Number(check?.count || 0) > baseline) {
              attachedOk = true;
              usedVia = mode;
              break;
            }
            if (await cancelled("uploading", `等 ${label} 挂上（${mode}）`)) return;
            await sleep(1500);
          }
          if (attachedOk) break;
          lastError = `${mode}：塞进去了但页面没挂上`;
          log("没挂上，换下一种方式", label, mode);
        }
        if (attachedOk) {
          log("附件已挂上", label, `方式=${usedVia}`);
        } else {
          // 自动塞不进去不算失败，改成等你手动选
          log("自动上传没成，转半自动", label, lastError);
          autoDone = false;
        }
      }
    }

    if (!autoDone) {
      // 半自动：把 Gemini 窗口拉到前台、顺手点开上传菜单，等你自己把文件选进去
      await chrome.tabs.update(tabId, { active: true }).catch(() => {});
      const tab = await chrome.tabs.get(tabId).catch(() => null);
      if (typeof tab?.windowId === "number") {
        await chrome.windows.update(tab.windowId, { focused: true }).catch(() => {});
      }
      const menu = await runInTab(tabId, pageOpenUploadMenu).catch(() => null);
      log("等你手动选文件", JSON.stringify(menu || {}));

      const manualDeadline = Date.now() + MANUAL_TIMEOUT_MS;
      for (;;) {
        const missing = [];
        for (const name of names) {
          const check = await runInTab(tabId, pageCountAttachment, [name]).catch(() => null);
          if (Number(check?.count || 0) <= baselines[name]) missing.push(name);
        }
        if (!missing.length) {
          log("两个文件都挂上了");
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
      await sleep(1000);
    }

    // 复用的窗口里可能已经有旧回答，先记下条数，别把旧的当成这次的结果
    const before = await runInTab(tabId, pageReadAnswer).catch(() => null);
    const baselineBlocks = Number(before?.blocks || 0);
    const sent = await runInTab(tabId, pageSendMessage, [editor.selector, message]);


    if (!sent?.ok) {
      return finish({ status: "failed", error: `发送失败：${sent?.error || "未知原因"}` });
    }
    log("已发送", sent.sent);

    // 等回答：文本连续 STABLE_MS 不变且没有「停止」按钮就算写完
    let text = "";
    let stableSince = 0;
    const answerDeadline = Date.now() + ANSWER_TIMEOUT_MS;
    while (Date.now() < answerDeadline) {
      await sleep(POLL_ANSWER_MS);
      const snapshot = await runInTab(tabId, pageReadAnswer).catch(() => null);
      // 新回答还没冒出来（条数没涨）就继续等，别读到窗口里原有的旧回答
      if (Number(snapshot?.blocks || 0) <= baselineBlocks) {
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
      return finish({ status: "failed", error: "AI 没有产出可读的回答" });
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
