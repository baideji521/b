// popup：只显示连不连得上、配没配对、在干什么，加一个重新配对按钮。

const endpointEl = document.getElementById("endpoint");
const pairedEl = document.getElementById("paired");
const stateEl = document.getElementById("state");
const pairButton = document.getElementById("pair");

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
}

async function refresh() {
  paint(await ask({ type: "status" }));
}

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
