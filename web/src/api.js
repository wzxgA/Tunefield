// fetch 封装（WebSocket 订阅在 F2 事件流交付后补充）

async function request(path, options = {}) {
  // multipart 上传（FormData）不手动设 Content-Type：交给浏览器附带 boundary；
  // 其余 JSON 请求显式声明 application/json。
  const isForm = options.body instanceof FormData;
  const headers = isForm
    ? { ...(options.headers || {}) }
    : { "Content-Type": "application/json", ...(options.headers || {}) };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => "")}`);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  get: (path) => request(path),
  post: (path, body) =>
    request(path, { method: "POST", body: JSON.stringify(body ?? {}) }),
  del: (path) => request(path, { method: "DELETE" }),
  // multipart 上传（浏览器自动带 boundary，不手动设 Content-Type）
  upload: (path, formData) =>
    request(path, { method: "POST", body: formData }),
};
