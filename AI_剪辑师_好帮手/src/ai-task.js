// 好帮手唯一的活儿：从 AI_剪辑师 领任务 → 把两个 txt 上传到网页版 AI → 等回答 →
// 把 JSON 回传给 AI_剪辑师。支持 Gemini 和 DeepSeek 两家，选哪家由任务里的
// provider / 网址决定，页面差异都在下面的 SITES 里。
//
//   GET  /v1/ai/next?types=gemini_json   领任务（文件清单 + 要说的那句话）
//   GET  /v1/ai/file?task_id=..&index=N  取 txt 本体（prm_en.txt / *_merged.txt）
//     → 这家的对话页已经开着就用那个窗口，没开才新建，等输入框出现
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
// ---------------------------------------------------------------------------
// 站点档案：Gemini 和 DeepSeek 的页面长得不一样，差异全收在这儿
//
// 注入到页面里的函数不能引用模块作用域，所以这些选择器都是当参数传进去的。
// 加第三家网页版 AI，就在这里再加一条 + 在 manifest 里加 host_permissions。
// ---------------------------------------------------------------------------

const SITES = {
  gemini: {
    label: "Gemini",
    url: "https://gemini.google.com/app",
    match: "*://gemini.google.com/*",
    hosts: ["gemini.google.com"],
    editors: [
      "div.ql-editor[contenteditable='true']",
      "rich-textarea div[contenteditable='true']",
      "[contenteditable='true']",
      "textarea",
    ],
    answers: [
      "message-content.model-response-text",
      ".model-response-text",
      "model-response",
      ".markdown",
    ],
    sends: ["button.send-button", "button[aria-label*='发送']", "button[aria-label*='Send']",
            "button[mattooltip*='发送']"],
    spinners: "mat-progress-bar, mat-spinner, mat-progress-spinner, [role='progressbar']",
    // Gemini 只有一处收 drop，一口气砸整页反而最稳（实测过的老路子，不会重复收）
    dropAll: true,
    upload: "添加照片和文件|上传文件|添加文件|attach|upload|add files",
    // 云端硬盘那条会弹 Google Drive 选择器，纯挡路；发送/录音也别碰
    uploadSkip: "云端|硬盘|drive|发送|send|停止|stop|麦克风|mic|语音|录音|图片生成|制作",
  },
  deepseek: {
    label: "DeepSeek",
    url: "https://chat.deepseek.com/",
    match: "*://chat.deepseek.com/*",
    hosts: ["chat.deepseek.com"],
    // DeepSeek 的输入框是普通 textarea（id 一直是 chat-input），比富文本好对付
    editors: ["textarea#chat-input", "textarea[placeholder]", "textarea", "[contenteditable='true']"],
    // 回答块挂 ds-markdown；类名带哈希的那些一律用前缀匹配兜住
    answers: [".ds-markdown", "[class*='ds-markdown']", "[class*='_md_']", "[class*='markdown']"],
    sends: ["div[role='button'][aria-disabled]", "button[type='submit']"],
    spinners: "[role='progressbar'], [class*='loading'], [class*='uploading']",
    // DeepSeek 输入框那几层和 body 各挂了一个 drop 监听，砸多了会被收好几遍
    dropAll: false,
    upload: "上传附件|添加附件|上传文件|attach|upload",
    uploadSkip: "深度思考|联网搜索|发送|send|停止|stop|语音|录音|新对话|new chat",
  },
};

/** 按任务给的 provider / 网址挑站点档案，认不出就当 Gemini。 */
function siteFor(url, provider) {
  const name = String(provider || "").toLowerCase();
  if (SITES[name]) return SITES[name];
  const text = String(url || "");
  for (const site of Object.values(SITES)) {
    if (site.hosts.some((host) => text.includes(host))) return site;
  }
  return SITES.gemini;
}

// ---------------------------------------------------------------------------
// 注入到对话页里执行的函数（必须自包含，不能引用模块作用域的东西）
// ---------------------------------------------------------------------------

/** 找输入框。选择器按站点档案给的顺序试，越靠前越准。 */
function pageProbeEditor(selectors) {
  for (const selector of selectors) {
    if (document.querySelector(selector)) return { ok: true, selector };
  }
  return { ok: false, selector: null };
}

/**
 * 把 txt 塞进页面，默认走「模仿手动拖进去」——页面里能手动拖，就用同一条路。
 * mode: drop（拖放，默认）/ paste（粘贴）/ input（塞 file 控件，兜底）。
 */
function pageAttachFiles(payloads, mode, editorSelector, targetIndex) {
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

  const editor = document.querySelector(editorSelector || "[contenteditable='true'], textarea");

  if (mode === "drop" || !mode) {
    const root = document.documentElement;
    if (!document.body && !root) return { ok: false, error: "页面还没渲染出可拖放的区域" };
    // 一次只砸一个目标！有的站点（DeepSeek）输入框那几层和 body 各挂了一个 drop 监听，
    // 一口气全砸等于把文件传 N 遍——页面上就会冒出一堆重复附件。
    // 所以这里按 targetIndex 只派发一个，成没成由外面数卡片来判断，不行才换下一个。
    const targets = [];
    for (let node = editor; node && targets.length < 3; node = node.parentElement) {
      targets.push(node);
    }
    if (document.body) targets.push(document.body);
    if (root) targets.push(root);
    targets.push(document);
    if (!targets.length) return { ok: false, error: "找不到拖放目标" };
    const index = Number.isInteger(targetIndex) ? targetIndex : 0;
    // index < 0 表示「整页都砸一遍」：只有一处收 drop 的站点（Gemini）这样最稳
    const chosen = index < 0 ? targets : [targets[index]];
    if (!chosen[0]) {
      return { ok: false, error: "拖放目标试完了", targets: targets.length };
    }

    const box = document.body || root;
    const rect = box.getBoundingClientRect();
    const point = { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
    let fired = 0;
    for (const target of chosen) {
      for (const type of ["dragenter", "dragover", "drop"]) {
        try {
          target.dispatchEvent(new DragEvent(type, {
            bubbles: true, cancelable: true, composed: true, dataTransfer: transfer, ...point,
          }));
          fired += 1;
        } catch {}
      }
    }
    const one = chosen[0];
    const where = index < 0 ? `整页${targets.length}处`
      : (one === document ? "document"
        : `${one.tagName || ""}${one.className ? `.${String(one.className).slice(0, 20)}` : ""}`);
    return fired
      ? { ok: true, via: "drop", target: index, targets: targets.length, where,
          count: transfer.files.length }
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
function pageOpenUploadMenu(wantedSource, skipSource) {
  const wanted = new RegExp(wantedSource || "attach|upload", "i");
  const skip = new RegExp(skipSource || "发送|send|停止|stop", "i");
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
  // DeepSeek 这类站点把全名放在 title / aria-label 里，页面上只显示截断后的短名，
  // 光数正文会漏；漏判的代价是又拖一遍（页面上多出重复附件），所以顺带数一遍属性
  if (!matched) {
    let attrs = 0;
    for (const el of document.querySelectorAll("[title], [aria-label], [alt]")) {
      const label = `${el.getAttribute("title") || ""} ${el.getAttribute("aria-label") || ""} `
        + `${el.getAttribute("alt") || ""}`;
      if (label.includes(name) || label.includes(stem)) attrs += 1;
    }
    if (attrs) {
      matched = attrs;
      used = "属性";
    }
  }
  // 兜底信号：卡片左上角那个类型角标（TXT），数它有几个
  const ext = (name.match(/\.([^.]+)$/) || ["", ""])[1].toUpperCase();
  const chips = ext ? count(ext) : 0;
  const failed = /上传失败|上传出错|failed to upload|unsupported file/i.test(text);
  return { count: matched, used, chips, failed };
}




/** 附件是不是都加载完了：还在转圈 / 还写着「上传中」就不算完，这时候按回车会白发。 */
function pageUploadSettled(spinnerSelector) {
  const body = document.body;
  const text = body ? body.innerText || body.textContent || "" : "";

  const pending = /上传中|正在上传|处理中|uploading|processing/i.test(text);
  const spinner = document.querySelectorAll(
    spinnerSelector || "[role='progressbar']"
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
async function pageSendMessage(selector, text, sendSelectors) {
  const editor = document.querySelector(selector);
  if (!editor) return { ok: false, error: "输入框不见了" };
  const nap = (ms) => new Promise((done) => setTimeout(done, ms));
  const focused = document.hasFocus();
  const message = typeof text === "string" ? text : "";
  const content = () => (editor.tagName === "TEXTAREA" ? editor.value : editor.textContent || "");
  const landed = () => content().includes(message.slice(0, 12));
  let how = "";

  if (message) {
    editor.focus();
    if (editor.tagName === "TEXTAREA") {
      // 纯 textarea（DeepSeek）最省心：改 value + 发一个 input 就行，不需要焦点
      editor.value = message;
      editor.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: false }));
      how = "value";
    } else {
      if (focused) {
        try {
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(editor);
          selection.removeAllRanges();
          selection.addRange(range);
          if (document.execCommand("insertText", false, message) && landed()) how = "execCommand";
        } catch {}
      }
      // Quill（Gemini）这类富文本自己维护一份模型，光改 DOM 它不认，发送按钮就一直灰着。
      // 粘贴事件它必须处理，而且不像 execCommand 那样要求文档有焦点，所以优先走这条。
      if (!how) {
        try {
          const transfer = new DataTransfer();
          transfer.setData("text/plain", message);
          editor.dispatchEvent(new ClipboardEvent("paste", {
            bubbles: true, cancelable: true, clipboardData: transfer,
          }));
          if (landed()) how = "paste";
        } catch {}
      }
      if (!how) {
        editor.textContent = message;
        editor.dispatchEvent(new InputEvent("beforeinput", {
          bubbles: true, cancelable: true, inputType: "insertText", data: message,
        }));
        editor.dispatchEvent(new InputEvent("input", {
          bubbles: true, cancelable: false, inputType: "insertText", data: message,
        }));
        how = landed() ? "dom" : "没打进去";
      }
    }
  }

  const enabled = (el) => !(el.disabled || el.getAttribute("aria-disabled") === "true");
  const findSend = () => {
    const buttons = Array.from(document.querySelectorAll(
      "button.send-button, button, [role='button']"
    ));
    const labelled = buttons.find((b) => {
      const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""} `
        + `${b.className || ""} ${b.querySelector("mat-icon")?.getAttribute("fonticon") || ""}`;
      if (!/发送|send|提交|submit/i.test(label)) return false;
      return !/停止|stop|取消|cancel|录音|mic/i.test(label);
    });
    if (labelled) return labelled;
    // 站点档案给的选择器：DeepSeek 那种按钮没有文字标签，只能按结构找
    for (const selector of sendSelectors || []) {
      const hit = Array.from(document.querySelectorAll(selector)).filter(enabled).pop();
      if (hit) return hit;
    }
    // 最后兜底：从输入框往上找几层，取容器里最后一个「只有图标的」按钮——发送键就是这种。
    // 必须挑得很死：Gemini 输入框旁边还蹲着图片生成、视频、Canvas、深度研究这些工具键，
    // 以及首页那几张建议卡片。误点一下就变成它替你出题（问出来一堆「可爱宠物照片建议」），
    // 附件白挂、回答里当然没有 JSON。
    const trap = /停止|stop|取消|cancel|录音|mic|语音|附件|attach|upload|图片|image|图像|生成|generate|创建|create|视频|video|canvas|研究|research|学习|guided|建议|试试|suggest|分享|share|复制|copy|设置|setting|菜单|menu|登录|account/i;
    for (let node = editor, up = 0; node && up < 5; node = node.parentElement, up += 1) {
      const near = Array.from(node.querySelectorAll("button, [role='button']")).filter((b) => {
        const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""} `
          + `${b.className || ""}`;
        // 图标键自己没什么文字；有一长串文字的是工具条或建议卡片，绝对不能点
        const own = (b.innerText || b.textContent || "").trim();
        return enabled(b) && own.length <= 4 && !trap.test(label) && !trap.test(own);
      });
      if (near.length) return near[near.length - 1];
    }
    return null;
  };

  let button = null;
  let disabled = false;
  // 点了哪个键：误点工具键这种事只有把它记下来才查得出
  const nameOf = (el) => (el
    ? `${el.tagName}[${el.getAttribute("aria-label") || el.getAttribute("mattooltip")
      || (el.innerText || "").trim().slice(0, 10) || String(el.className || "").slice(0, 20)}]`
    : "无");
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    button = findSend();
    if (button) {
      disabled = !enabled(button);
      if (!disabled) {
        const clicked = nameOf(button);
        button.click();
        await nap(300);
        return {
          ok: true, sent: "button", typed: message.length, focused, how, clicked,
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
    ok: true, sent: "enter", typed: message.length, focused, how,
    editorLen: content().trim().length,
    button: button ? (disabled ? "灰着" : "没点上") : "没找到",
  };
}



/** 读最后一条回答的纯文本，并判断是否还在写。选择器按站点档案给的顺序试。 */
function pageReadAnswer(groups, editorSelector) {
  let nodes = [];
  let used = "";
  for (const selector of groups) {
    nodes = Array.from(document.querySelectorAll(selector));
    if (nodes.length) {
      used = selector;
      break;
    }
  }
  // 选择器全落空时的通用兜底（站点改版、类名带哈希都会这样）：挑「自己有一大段文字、
  // 子元素里没有同样长文字」的叶子块，文档里最后那个就是最新的回答。
  if (!nodes.length) {
    const leafs = [];
    for (const el of document.querySelectorAll("div, section, article, p, pre, li, td")) {
      const own = (el.innerText || el.textContent || "").trim();
      if (own.length < 40) continue;
      const sameInChild = Array.from(el.children).some(
        (c) => (c.innerText || c.textContent || "").trim().length >= own.length * 0.9
      );
      if (!sameInChild) leafs.push(el);
    }
    nodes = leafs;
    used = leafs.length ? "通用兜底" : "";
  }
  const last = nodes[nodes.length - 1] || null;
  // 后台标签页不做布局时 innerText 是空的，退回 textContent
  const text = last ? (last.innerText || last.textContent || "").trim() : "";

  // 还在生成时页面上有「停止」按钮
  const streaming = Array.from(document.querySelectorAll("button, [role='button']")).some((b) => {
    const label = `${b.getAttribute("aria-label") || ""} ${b.getAttribute("mattooltip") || ""}`;
    return /stop|停止/i.test(label);
  });
  // 一个块都没有时，报一下整页文本长度：0 说明页面被冻结/还没渲染，不是选择器写错
  const pageText = document.body ? document.body.innerText || document.body.textContent || "" : "";
  const bodyLen = pageText.length;
  // 命中的是哪个块：万一读错地方，日志里能直接看出来该改哪个选择器
  const hint = last
    ? `${last.tagName}.${String(last.className || "").slice(0, 40)}`
    : "";
  // 输入框里还有没有字：还留着说明那句话压根没发出去，空了就是已经发走了。
  // 「要不要补发一次」只看这个，别拿「没等到回答」当理由——那样会重复发。
  const box = editorSelector ? document.querySelector(editorSelector) : null;
  const editorLen = box
    ? (box.tagName === "TEXTAREA" ? box.value : box.textContent || "").trim().length
    : -1;

  // 最后一道保险，也是最不挑站点的一道：我们要的就是一个 JSON 对象，
  // 那就直接在整页文字里找最长的、能解析的 {...}。选择器猜错、类名改版都不影响。
  // 页面上不会有别的大 JSON——发过去的是附件，聊天气泡里只有那句短指令。
  let json = "";
  let tried = 0;
  for (let start = pageText.indexOf("{"); start >= 0 && tried < 200;
       start = pageText.indexOf("{", start + 1)) {
    tried += 1;
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < pageText.length; i += 1) {
      const ch = pageText[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          const slice = pageText.slice(start, i + 1);
          if (slice.length > json.length) {
            for (const candidate of [slice, slice.replace(/[\r\n]+/g, " ")]) {
              try {
                const parsed = JSON.parse(candidate);
                if (parsed && typeof parsed === "object") {
                  json = candidate;
                  break;
                }
              } catch {}
            }
          }
          break;
        }
      }
    }
  }
  return { text, streaming, blocks: nodes.length, used, bodyLen, hint,
           json, jsonLen: json.length, editorLen };
}


// ---------------------------------------------------------------------------
// 标签页操作
// ---------------------------------------------------------------------------

async function runInTab(tabId, fn, args = []) {
  // executeScript 的 args 必须可序列化：混进一个 undefined 会让整次调用报
  // 「Value is unserializable」，还看不出是哪个参数的问题，所以统一换成 null
  const safe = args.map((item) => (item === undefined ? null : item));
  const [result] = await chrome.scripting.executeScript({
    target: { tabId },
    func: fn,
    args: safe,
    world: "MAIN",
  });
  return result?.result ?? null;
}

/**
 * 把对话页挪到一个自己的小窗口里，不抢焦点。
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
 * 先看这家的对话页是不是已经开着：开着就直接用，绝不动它的 URL，
 * 也不等页面加载（省掉重新加载那几秒）。没开才新建一个。
 *
 * sideWindow=true（默认）时保证它是自己窗口里的活动标签页——不然一被别的标签页盖住，
 * 页面就被冻结，上传上去也发不出去、回答也读不出来。
 * 返回 { tabId, created, ready }，created=false 的标签页是用户自己的，事后不许关。
 */
async function ensureAiTab(url, site, { focus = false, sideWindow = true } = {}) {
  const label = site.label;
  const opened = await chrome.tabs.query({ url: site.match }).catch(() => []);
  const existing = Array.isArray(opened) ? opened.find((tab) => typeof tab.id === "number") : null;
  if (existing) {
    log(`${label} 已经开着，直接用`, existing.id, existing.status || "");
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
    log(`${label} 没打开，开个不抢焦点的小窗口`, url);
    const win = await chrome.windows.create({ url, focused: false, width: 560, height: 620 })
      .catch(() => null);
    const tab = win?.tabs?.[0];
    if (tab && typeof tab.id === "number") {
      return { tabId: tab.id, created: true, ready: false };
    }
  }
  log(`${label} 没打开，新建标签页`, url, focus ? "" : "（后台打开）");
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
  // 站点档案决定选择器和要开哪个网址：AI_剪辑师 会把 provider 一起发过来，
  // 老版本没这个字段就按网址认（认不出当 Gemini）
  const site = siteFor(task.url, task.provider);
  const url = task.url || site.url;
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
    const target = await ensureAiTab(url, site, { focus: focusBrowser, sideWindow });


    tabId = target.tabId;
    createdTab = target.created;
    // 已经开着而且加载完的，直接干，不等；只有新开的或还在转的才等页面 complete
    if (!target.ready) await waitForTabComplete(tabId, READY_TIMEOUT_MS);


    // 新开的页面框架还要渲染一会儿；复用的页面通常第一次探测就命中
    let editor = null;
    const readyDeadline = Date.now() + READY_TIMEOUT_MS;
    while (Date.now() < readyDeadline) {
      editor = await runInTab(tabId, pageProbeEditor, [site.editors]).catch(() => null);
      if (editor?.ok) break;
      if (await cancelled("waiting_editor", "等输入框出现")) return;
      await sleep(400);
    }

    if (!editor?.ok) {
      return finish({ status: "failed",
                      error: `${site.label} 页面找不到输入框，可能没登录或页面结构变了` });
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


      // 拖放怎么砸看站点：Gemini 只有一处收，砸整页最稳（target=-1）；
      // DeepSeek 那几层各挂一个监听，只能一次砸一个，砸完数卡片，没进去才换下一个目标。
      // 注意每个 plan 都必须带上 target 这个数字——executeScript 的 args 里出现
      // undefined 会直接报「Value is unserializable」。
      const plans = [];
      if (site.dropAll) {
        plans.push({ mode: "drop", batch: true, target: -1 });
      } else {
        for (let t = 0; t < 6; t += 1) plans.push({ mode: "drop", batch: true, target: t });
      }
      plans.push({ mode: "drop", batch: false, target: site.dropAll ? -1 : 0 });
      plans.push({ mode: "paste", batch: true, target: 0 });
      plans.push({ mode: "input", batch: true, target: 0 });
      for (let p = 0; p < plans.length; p += 1) {
        const plan = plans[p];
        // 卡片可能晚一点才冒出来，换下一种方式之前先复查一遍，别白塞第二遍
        if (p > 0 && await waitCards(probe, 900)) {
          autoDone = true;
          break;
        }
        if (plan.mode === "input") {

          const menu = await runInTab(tabId, pageOpenUploadMenu,
                                     [site.upload, site.uploadSkip]).catch(() => null);
          log("催上传控件", JSON.stringify(menu || {}));
          await sleep(300);
        }
        try {
          if (plan.batch) {
            const attached = await runInTab(tabId, pageAttachFiles,
                                           [payloads, plan.mode, editor.selector, plan.target]);
            if (!attached?.ok) {
              log("塞入失败", plan.mode, attached?.error || "未知原因");
              // 拖放目标试完了就别再往下试同类方案
              if (attached?.error === "拖放目标试完了") {
                while (p + 1 < plans.length && plans[p + 1].mode === "drop") p += 1;
              }
              continue;
            }
            if (plan.mode === "drop") log("拖了一次", `目标${attached.target}=${attached.where}`);
            autoDone = await waitCards(probe, ATTACH_VERIFY_MS);
          } else {
            autoDone = true;
            for (const item of payloads) {
              const attached = await runInTab(tabId, pageAttachFiles,
                                             [[item], plan.mode, editor.selector, plan.target]);
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

      const menu = await runInTab(tabId, pageOpenUploadMenu,
                                 [site.upload, site.uploadSkip]).catch(() => null);
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
      const settle = await runInTab(tabId, pageUploadSettled, [site.spinners]).catch(() => null);
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


    // 复用的窗口里可能已经有旧回答，先记下条数、最后那块的文字、整页扫到的 JSON，
    // 别把上一轮的结果当成这次的
    const before = await runInTab(tabId, pageReadAnswer,
                                 [site.answers, editor.selector]).catch(() => null);
    const baselineBlocks = Number(before?.blocks || 0);
    const baselineText = String(before?.text || "");
    const baselineJson = String(before?.json || "");
    const sent = await runInTab(tabId, pageSendMessage,
                               [editor.selector, message, site.sends]);


    if (!sent?.ok) {
      return finish({ status: "failed", error: `发送失败：${sent?.error || "未知原因"}` });
    }
    const sentInfo = `发送=${sent.sent} 打字=${sent.how || "无"} 长度=${sent.typed} `
      + `焦点=${sent.focused ? "有" : "无"} 点了=${sent.clicked || sent.button || "无"}`;
    log("已发送", sent.sent, JSON.stringify({
      typed: sent.typed, how: sent.how, clicked: sent.clicked, editorLen: sent.editorLen,
      focused: sent.focused, attempts: sent.attempts, button: sent.button,
    }));

    // 等回答：文本连续 STABLE_MS 不变且没有「停止」按钮就算写完
    let text = "";
    let stableSince = 0;
    let resent = false;
    let polls = 0;
    const sentAt = Date.now();
    const answerDeadline = Date.now() + ANSWER_TIMEOUT_MS;
    while (Date.now() < answerDeadline) {
      await sleep(POLL_ANSWER_MS);
      polls += 1;
      const snapshot = await runInTab(tabId, pageReadAnswer,
                                     [site.answers, editor.selector]).catch(() => null);
      const shot = String(snapshot?.text || "");
      const shotJson = String(snapshot?.json || "");
      // 现场情况：卡住时全靠这几个数判断是选择器没命中还是页面根本没渲染。
      // 顺带塞进进度消息里，AI_剪辑师 的日志能直接看到，不用去翻扩展控制台。
      const diag = `blocks=${snapshot?.blocks ?? "?"}/${baselineBlocks} `
        + `选择器=${snapshot?.used || "无"} 块=${snapshot?.hint || "无"} `
        + `文字=${shot.length} JSON=${shotJson.length} 整页=${snapshot?.bodyLen ?? "?"} `
        + `输入框=${snapshot?.editorLen ?? "?"}`;
      // 新回答算不算冒出来，三个信号任一成立：回答块变多了；最后那块文字变了（且不是
      // 我们刚发的那句）；整页扫出来的 JSON 跟发之前不一样。第三个最不挑站点，
      // 选择器猜错、类名改版都还能救回来。
      const fresh = Number(snapshot?.blocks || 0) > baselineBlocks
        || (shot && shot !== baselineText && !shot.includes(message.slice(0, 20)))
        || (shotJson && shotJson !== baselineJson);
      if (!fresh) {
        // 补发只在「输入框里那句话还留着」时做——那才说明真没发出去。
        // 拿「没等到回答」当理由会误判成没发送，页面上就会多出一条重复提问。
        if (!resent && Date.now() - sentAt > RESEND_AFTER_MS
            && Number(snapshot?.editorLen || 0) > 0) {
          resent = true;
          const again = await runInTab(tabId, pageSendMessage,
                                     [editor.selector, "", site.sends]).catch(() => null);
          log("那句话还在输入框里，补发一次", again?.sent || "失败", diag);
        }
        if (polls % 10 === 0) log("还没等到新回答", diag);
        if (await cancelled("waiting_answer", `等新回答出现｜${sentInfo}｜${diag}`)) return;
        continue;
      }

      // 选择器读到的那块优先；读不到（或读到的没 JSON）就用整页扫出来的 JSON
      const current = shot.includes("{") || !shotJson ? shot : shotJson;

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
      const last = await runInTab(tabId, pageReadAnswer,
                                 [site.answers, editor.selector]).catch(() => null);
      if (last?.json && last.json !== baselineJson) text = String(last.json);
      const detail = JSON.stringify({
        blocks: last?.blocks, baseline: baselineBlocks,
        bodyLen: last?.bodyLen, used: last?.used, hint: last?.hint,
        jsonLen: last?.jsonLen, streaming: last?.streaming,
      });
      if (!text) {
        log("读不到回答", detail);
        return finish({ status: "failed", error: `AI 没有产出可读的回答 ${detail}` });
      }
      log("超时前从整页 JSON 里救回来了", detail);
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
