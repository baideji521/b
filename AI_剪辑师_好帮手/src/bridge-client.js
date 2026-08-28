// 与 AI_剪辑师 的本机 Bridge 通信：地址发现、配对、令牌存取。
//
// 契约：
//   - AI_剪辑师（GUI）起一个只监听 127.0.0.1 的 HTTP 服务，默认 5998，被占用往后顺延。
//   - GUI 里点「配对扩展」会开一个约 120 秒的单次窗口，期间 GET /v1/pair 能领到令牌。
//   - 令牌存在 chrome.storage.local，之后每个请求带 Bearer。
//   - 令牌失效（GUI 换了令牌）时服务端回 401，这里清掉旧令牌并重新武装自动配对，
//     否则这个浏览器会永久连不上。

export const STORAGE_KEY_ENDPOINT = "bridge_endpoint";
export const STORAGE_KEY_TOKEN = "bridge_token";
// 用户在 popup 里手填过端口就置 true：之后一律用这个端口，不再自动探测顺延端口段。
export const STORAGE_KEY_MANUAL = "bridge_endpoint_manual";

// 端点迁移标记。默认端口改过几次（47720 段 -> 5999 -> 5998），而浏览器存储里
// 的旧地址优先于 DEFAULT_ENDPOINT，所以换默认端口时必须主动清一次。
const STORAGE_KEY_ENDPOINT_MIGRATED = "bridge_endpoint_migration";
const ENDPOINT_MIGRATION_ID = "port-5998";
// 历史默认端口。只清这些，用户自己填的非默认端口不动。
const LEGACY_DEFAULT_PORTS = new Set([
  "47720", "47721", "47722", "47723", "47724",
  "47725", "47726", "47727", "47728", "47729",
  "59999", "5999",
]);
// background.js 里那个低频自动配对闹钟的名字。401 恢复路径要用，导出避免写两遍。
export const AUTOPAIR_ALARM_NAME = "AI剪辑师好帮手-autopair";
const HEALTH_TIMEOUT_MS = 1500;
// AI_剪辑师 的 Bridge 默认监听 5998（见 config.json 的 bridge.port）
export const DEFAULT_PORT = 5998;
const DEFAULT_ENDPOINT = `http://127.0.0.1:${DEFAULT_PORT}`;
// 没手填端口时自动探测的范围：5998 起往后 9 个
export const DEFAULT_PORT_RANGE = [
  5998, 5999, 6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007,
];

export function trimEndpoint(value) {

  if (typeof value !== "string") return "";
  let trimmed = value.trim();
  if (!trimmed) return "";
  while (trimmed.endsWith("/")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed;
}

/** 从端点里抠出端口号，抠不到就给默认端口。 */
export function portOf(endpoint) {
  try {
    const port = Number(new URL(trimEndpoint(endpoint)).port);
    return Number.isFinite(port) && port > 0 ? port : DEFAULT_PORT;
  } catch {
    return DEFAULT_PORT;
  }
}

// 判断某个端点是否指向历史默认端口。用户手填的自定义端口不在名单里，不会被误清。

function isLegacyDefaultEndpoint(value) {
  const trimmed = trimEndpoint(value);
  if (!trimmed) return false;
  try {
    return LEGACY_DEFAULT_PORTS.has(new URL(trimmed).port);
  } catch {
    return false;
  }
}

export async function loadBridgeConfig({ storage = globalThis.chrome?.storage?.local } = {}) {
  if (!storage?.get) {
    return { endpoint: "", token: "", port: DEFAULT_PORT, manual: false };
  }
  return new Promise((resolve) => {
    storage.get(
      [STORAGE_KEY_ENDPOINT, STORAGE_KEY_TOKEN, STORAGE_KEY_ENDPOINT_MIGRATED,
       STORAGE_KEY_MANUAL],
      (items) => {
        let stored = trimEndpoint(items?.[STORAGE_KEY_ENDPOINT]);
        const token =
          typeof items?.[STORAGE_KEY_TOKEN] === "string" ? items[STORAGE_KEY_TOKEN].trim() : "";
        const manual = Boolean(items?.[STORAGE_KEY_MANUAL]);

        // 一次性迁移：存储里留着旧默认端口时清掉，让 DEFAULT_ENDPOINT 生效。
        const migrated = items?.[STORAGE_KEY_ENDPOINT_MIGRATED];
        if (migrated !== ENDPOINT_MIGRATION_ID) {
          if (isLegacyDefaultEndpoint(stored)) stored = "";
          try {
            storage.set({
              [STORAGE_KEY_ENDPOINT]: stored,
              [STORAGE_KEY_ENDPOINT_MIGRATED]: ENDPOINT_MIGRATION_ID,
            });
          } catch {}
        }

        const endpoint = stored || DEFAULT_ENDPOINT;
        resolve({ endpoint, token, port: portOf(endpoint), manual });
      }
    );
  });
}

export async function saveBridgeConfig(
  { endpoint = "", token = "", manual = null },
  { storage = globalThis.chrome?.storage?.local } = {}
) {
  if (!storage?.set) return false;
  const payload = {
    [STORAGE_KEY_ENDPOINT]: trimEndpoint(endpoint),
    [STORAGE_KEY_TOKEN]: typeof token === "string" ? token.trim() : "",
    // 用户显式保存过的地址就是最终答案，打上标记避免之后被迁移逻辑清掉
    [STORAGE_KEY_ENDPOINT_MIGRATED]: ENDPOINT_MIGRATION_ID,
  };
  // manual 给 null 表示别动这个标记（自动发现改地址时不该把手填状态抹掉）
  if (manual !== null) payload[STORAGE_KEY_MANUAL] = Boolean(manual);
  return new Promise((resolve) => {
    storage.set(payload, () => resolve(true));
  });
}

/**
 * popup 保存端口：只认 1-65535，存成 http://127.0.0.1:<port> 并锁定手填状态。
 * 端口变了就把旧令牌丢掉——换端口通常就是换了另一个 AI_剪辑师 实例。
 */
export async function saveBridgePort(port, { storage = globalThis.chrome?.storage?.local } = {}) {
  const value = Number(port);
  if (!Number.isInteger(value) || value < 1 || value > 65535) {
    return { ok: false, reason: "bad-port" };
  }
  const current = await loadBridgeConfig({ storage });
  const endpoint = `http://127.0.0.1:${value}`;
  const token = portOf(current.endpoint) === value ? current.token : "";
  await saveBridgeConfig({ endpoint, token, manual: true }, { storage });
  return { ok: true, endpoint, port: value, token_kept: Boolean(token) };
}


// 忘掉令牌但保留地址（地址通常还是对的）
export async function clearStoredToken({ storage = globalThis.chrome?.storage?.local } = {}) {
  if (!storage?.set) return false;
  return new Promise((resolve) => {
    storage.set({ [STORAGE_KEY_TOKEN]: "" }, () => resolve(true));
  });
}

// 401/403 = 令牌过期（GUI 那边换过）。清掉并重新武装自动配对闹钟。
export async function handleUnauthorized(storage = globalThis.chrome?.storage?.local) {
  try {
    await clearStoredToken({ storage });
  } catch {}
  try {
    const alarms = globalThis.chrome?.alarms;
    if (alarms?.create) {
      const existing = alarms.get ? await alarms.get(AUTOPAIR_ALARM_NAME) : null;
      if (!existing) {
        alarms.create(AUTOPAIR_ALARM_NAME, { periodInMinutes: 1 });
      }
    }
  } catch {}
}

function withTimeout(promise, ms, controller) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      try {
        controller?.abort();
      } catch {}
      reject(new Error(`连接 AI_剪辑师 超时（${ms}ms）`));
    }, ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

// 并行探测端口段，返回第一个 /v1/health 回 ok 的地址
export async function discoverBridgeEndpoint({
  fetchImpl = typeof fetch !== "undefined" ? fetch : null,
  ports = DEFAULT_PORT_RANGE,
  host = "127.0.0.1",
  timeoutMs = HEALTH_TIMEOUT_MS,
} = {}) {
  if (!fetchImpl) return null;
  const probes = ports.map(async (port) => {
    const endpoint = `http://${host}:${port}`;
    const result = await checkBridgeHealth(endpoint, { fetchImpl, timeoutMs });
    return result.ok ? { endpoint, version: result.version ?? null } : null;
  });

  const settled = await Promise.allSettled(probes);
  for (const result of settled) {
    if (result.status === "fulfilled" && result.value) {
      return result.value;
    }
  }
  return null;
}

export async function checkBridgeHealth(
  endpoint,
  { fetchImpl = (typeof fetch !== "undefined" ? fetch : null), timeoutMs = HEALTH_TIMEOUT_MS } = {}
) {
  if (!fetchImpl) return { ok: false, reason: "no-fetch" };
  const target = trimEndpoint(endpoint);
  if (!target) return { ok: false, reason: "no-endpoint" };

  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  try {
    const response = await withTimeout(
      fetchImpl(`${target}/v1/health`, { method: "GET", signal: controller?.signal }),
      timeoutMs,
      controller
    );
    if (!response.ok) {
      return { ok: false, reason: "http-error", status: response.status };
    }
    const body = await response.json().catch(() => null);
    return { ok: Boolean(body?.ok), version: body?.version ?? null, app: body?.app ?? null };
  } catch (error) {
    return { ok: false, reason: "fetch-failed", message: error?.message ?? String(error) };
  }
}

// 一次性自动配对：没有令牌时先发现地址，再 GET /v1/pair。
// 该接口只在 GUI 点过「配对扩展」的窗口期内给令牌，所以这一步把
// 「找令牌、复制、粘贴」变成了在 GUI 里点一下。
export async function autoPair({
  fetchImpl = typeof fetch !== "undefined" ? fetch : null,
  storage = globalThis.chrome?.storage?.local,
  timeoutMs = HEALTH_TIMEOUT_MS,
} = {}) {
  if (!fetchImpl) return { ok: false, reason: "no-fetch" };
  const current = await loadBridgeConfig({ storage });
  if (current.token) return { ok: true, reason: "already-paired" };

  let endpoint = current.endpoint;
  // 手填过端口就死守这个端口，不去探测端口段（不然自己指定的端口会被自动发现覆盖）
  if (!current.manual) {
    const discovered = await discoverBridgeEndpoint({ fetchImpl, timeoutMs });
    if (discovered?.endpoint) endpoint = discovered.endpoint;
  }
  endpoint = trimEndpoint(endpoint);

  if (!endpoint) return { ok: false, reason: "no-endpoint" };

  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  try {
    const response = await withTimeout(
      fetchImpl(`${endpoint}/v1/pair`, { method: "GET", signal: controller?.signal }),
      timeoutMs,
      controller
    );
    if (!response.ok) return { ok: false, reason: "window-closed" };
    const parsed = await response.json().catch(() => null);
    const token = typeof parsed?.token === "string" ? parsed.token.trim() : "";
    if (!parsed?.ok || !token) return { ok: false, reason: "no-token" };
    await saveBridgeConfig({ endpoint, token }, { storage });
    return { ok: true, reason: "paired", endpoint };
  } catch (error) {
    return { ok: false, reason: "error", message: error?.message ?? String(error) };
  }
}
