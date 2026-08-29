// AI_剪辑师_好帮手 的 service worker。
//
// 只做三件事：
//   1. 自动配对（在 AI_剪辑师 里点「配对扩展」后，这边把令牌领回来）
//   2. 轮询 AI_剪辑师 的任务队列（见 ai-task.js）
//   3. 给 popup 提供状态 / 手动配对入口
//
// 没有下载、没有嗅探、没有 cookie 采集——那些都砍掉了。

import {
  autoPair,
  loadBridgeConfig,
  checkBridgeHealth,
  discoverBridgeEndpoint,
  saveBridgeConfig,
  saveBridgePort,
  saveCloseTab,
  DEFAULT_PORT,
  AUTOPAIR_ALARM_NAME,
} from "./bridge-client.js";

import { startAiPolling, resumeAiPolling, pollingStatus, AI_POLL_ALARM } from "./ai-task.js";

const LOG_PREFIX = "[AI剪辑师好帮手]";

function log(...args) {
  console.info(LOG_PREFIX, ...args);
}

async function tryAutoPair(reason) {
  const result = await autoPair().catch((error) => ({ ok: false, reason: String(error) }));
  if (result?.ok && result.reason === "paired") {
    log(`配对成功（${reason}）：${result.endpoint}`);
    // 配对完就不用再定时探了
    try {
      chrome.alarms.clear(AUTOPAIR_ALARM_NAME);
    } catch {}
  }
  return result;
}

if (chrome.alarms?.onAlarm) {
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm?.name === AI_POLL_ALARM) resumeAiPolling();
    if (alarm?.name === AUTOPAIR_ALARM_NAME) tryAutoPair("闹钟");
  });
}

chrome.runtime.onInstalled.addListener(() => {
  log("已安装/更新");
  try {
    chrome.alarms.create(AUTOPAIR_ALARM_NAME, { periodInMinutes: 1 });
  } catch {}
  tryAutoPair("安装");
});

chrome.runtime.onStartup?.addListener(() => {
  tryAutoPair("浏览器启动");
});

// popup 的问答接口
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "status") {
    (async () => {
      const config = await loadBridgeConfig();
      const health = await checkBridgeHealth(config.endpoint);
      sendResponse({
        endpoint: config.endpoint,
        port: config.port,
        manual: config.manual,
        default_port: DEFAULT_PORT,
        paired: Boolean(config.token),
        close_tab: config.closeTab !== false,
        health,
        polling: pollingStatus(),
      });
    })();
    return true;
  }
  if (message?.type === "setPort") {
    (async () => {
      const result = await saveBridgePort(message.port);
      if (result.ok) log(`端口已改为 ${result.port}（手填，不再自动探测）`);
      sendResponse(result);
    })();
    return true;
  }
  if (message?.type === "setCloseTab") {
    (async () => {
      const result = await saveCloseTab(message.value);
      if (result.ok) {
        log(result.closeTab ? "拿到数据后关掉自己开的标签页" : "拿到数据后留着标签页");
      }
      sendResponse(result);
    })();
    return true;
  }
  if (message?.type === "pair") {
    (async () => {
      const config = await loadBridgeConfig();
      // 手填过端口就守着它；没手填才去端口段里找一个活的
      if (!config.manual) {
        const found = await discoverBridgeEndpoint();
        if (found?.endpoint && found.endpoint !== config.endpoint) {
          await saveBridgeConfig({ endpoint: found.endpoint, token: config.token });
        }
      }
      sendResponse(await tryAutoPair("popup"));
    })();
    return true;
  }

  return false;
});

startAiPolling();
