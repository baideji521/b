// popup：显示连不连得上、配没配对、在干什么，可以自己填端口（默认 5998），外加重新配对。

const endpointEl = document.getElementById("endpoint");
const pairedEl = document.getElementById("paired");
const stateEl = document.getElementById("state");
const pairButton = document.getElementById("pair");
const portInput = document.getElementById("port");
const savePortButton = document.getElementById("savePort");
const portHintEl = document.getElementById("portHint");

// 用户正在输入时不要被定时刷新覆盖掉
let editingPort = false;

function ask(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => resolve(response ?? null));
  });
}

function paint(status) {
  if (!status) {
    endpointEl.textContent = "后台没响应";
    endpointEl.className = "bad";
    return;
  }
  const healthy = Boolean(status.health?.ok);
  endpointEl.textContent = status.endpoint || "未设置";
  endpointEl.className = healthy ? "ok" : "bad";
  if (!healthy) {
    endpointEl.textContent += "（连不上，AI_剪辑师 开着吗）";
  }
  pairedEl.textContent = status.paired ? "已配对" : "未配对";
  pairedEl.className = status.paired ? "ok" : "bad";
  stateEl.textContent = status.polling?.busy
    ? "执行任务中"
    : status.polling?.state || (status.polling?.polling ? "轮询中" : "未启动");
  if (!editingPort && document.activeElement !== portInput) {
    portInput.value = status.port ?? status.default_port ?? 5998;
  }
  portHintEl.textContent = status.manual
    ? `当前端口 ${status.port} 由你指定，不会自动改；默认 ${status.default_port}。`
    : `默认 ${status.default_port}，连不上时会在 ${status.default_port}-${status.default_port + 9} 里自动找。`;
}

async function refresh() {
  paint(await ask({ type: "status" }));
}

portInput.addEventListener("focus", () => {
  editingPort = true;
});
portInput.addEventListener("blur", () => {
  editingPort = false;
});
portInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") savePortButton.click();
});

savePortButton.addEventListener("click", async () => {
  const result = await ask({ type: "setPort", port: Number(portInput.value) });
  if (!result?.ok) {
    portHintEl.textContent = "端口不对：填 1-65535 的整数";
    return;
  }
  editingPort = false;
  portHintEl.textContent = result.token_kept
    ? `已保存 ${result.port}`
    : `已保存 ${result.port}，端口变了，需要重新配对`;
  await refresh();
});

pairButton.addEventListener("click", async () => {
  pairButton.disabled = true;
  pairButton.textContent = "配对中…";
  const result = await ask({ type: "pair" });
  pairButton.textContent = result?.ok ? "配对成功" : "配对失败：先在 AI_剪辑师 里点配对";
  await refresh();
  setTimeout(() => {
    pairButton.disabled = false;
    pairButton.textContent = "重新配对";
  }, 2000);
});

refresh();
setInterval(refresh, 2000);

