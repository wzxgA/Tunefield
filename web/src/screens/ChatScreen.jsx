import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

// T10 · 对话屏:Ollama 预检 → 导入平台 GGUF → 选模型 → 消息流 + 采样参数
// 转发链路:前端 → POST /v1/chat/completions(OpenAI 兼容)→ Ollama
export default function ChatScreen() {
  const [status, setStatus] = useState(null);
  const [ggufs, setGgufs] = useState([]);       // 平台 GGUF 清单(可导入)
  const [ollamaModels, setOllamaModels] = useState([]); // 已导入(tunefield- 前缀)
  const [selected, setSelected] = useState("");
  const [messages, setMessages] = useState([]); // {role, content}
  const [input, setInput] = useState("");
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.9);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [importingId, setImportingId] = useState("");
  const [removing, setRemoving] = useState("");
  const logRef = useRef(null);

  const load = useCallback(async () => {
    try {
      setStatus(await api.get("/api/chat/status"));
    } catch {
      setStatus(null);
    }
    try {
      setGgufs(await api.get("/api/models"));
    } catch {
      /* ignore */
    }
    try {
      const r = await api.get("/api/chat/models");
      setOllamaModels(r.models || []);
      setSelected((cur) => cur || (r.models?.[0]?.name ?? ""));
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const importModel = async (m) => {
    setError("");
    setImportingId(m.id);
    try {
      await api.post(`/api/models/${m.id}/import`, {});
      const r = await api.get("/api/chat/models");
      setOllamaModels(r.models || []);
      setSelected(m.ollama_name);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setImportingId("");
    }
  };

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!selected) {
      setError("请先导入并选择一个模型");
      return;
    }
    const history = [...messages, { role: "user", content: text }];
    setMessages(history);
    setInput("");
    setSending(true);
    setError("");
    try {
      const resp = await api.post("/v1/chat/completions", {
        model: selected,
        messages: history.map((m) => ({ role: m.role, content: m.content })),
        temperature: Number(temperature) || 0.8,
        top_p: Number(topP) || 0.9,
        stream: false,
      });
      const content = resp?.choices?.[0]?.message?.content ?? "(空回复)";
      setMessages([...history, { role: "assistant", content }]);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSending(false);
    }
  };

  // 移除已导入 Ollama 的模型（不影响平台 GGUF 文件与下载）
  const removeModel = async (name) => {
    if (!window.confirm(`移除已导入的 ${name}？\n（仅从 Ollama 删除，平台 GGUF 文件保留）`)) {
      return;
    }
    setError("");
    setRemoving(name);
    try {
      await api.del(`/api/chat/models/${encodeURIComponent(name)}`);
      const r = await api.get("/api/chat/models");
      const next = r.models || [];
      setOllamaModels(next);
      setSelected((cur) => (cur === name ? (next[0]?.name ?? "") : cur));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRemoving("");
    }
  };

  const importedNames = new Set(ollamaModels.map((m) => m.name));
  const banner = (() => {
    if (!status) return null;
    if (!status.installed)
      return (
        <div className="chat-banner chat-err">
          未安装 Ollama —— 对话功能不可用(不影响训练/导出)。安装:{status.hint}
        </div>
      );
    if (!status.running)
      return (
        <div className="chat-banner chat-warn">
          Ollama 已安装但服务未运行 —— 请启动 Ollama(托盘图标或 `ollama serve`)后重试。
        </div>
      );
    return <div className="chat-banner chat-ok">Ollama 已连接({status.version}) · {status.host}</div>;
  })();

  return (
    <section className="screen chat-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">模型对话</h2>
          <p className="screen-hint">导入已量化的领域模型,直接对话试用;参数可调。</p>
        </div>
        <span className={`ws-dot${status?.running ? " ws-on" : ""}`} title="Ollama 连接状态" />
      </div>

      {banner}

      <div className="chat-models">
        <select className="input" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {ollamaModels.length === 0 && <option value="">(尚无已导入模型)</option>}
          {ollamaModels.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
        <div className="chat-params">
          <label>temp <input className="input chat-num" type="number" step="0.1" min="0" max="2" value={temperature} onChange={(e) => setTemperature(e.target.value)} /></label>
          <label>top_p <input className="input chat-num" type="number" step="0.05" min="0" max="1" value={topP} onChange={(e) => setTopP(e.target.value)} /></label>
        </div>
      </div>

      {/* 始终展示已导入模型列表：每项均提供「移除」按钮，与平台 GGUF 列表无关 */}
      {ollamaModels.length > 0 && (
        <div className="chat-imported">
          <span className="pf-label">已导入 Ollama（可移除）：</span>
          {ollamaModels.map((m) => (
            <span key={m.name} className="chat-import-item">
              <span className="pf-rule">{m.name}</span>
              <button
                type="button"
                className="btn-inspect"
                disabled={removing === m.name}
                onClick={() => removeModel(m.name)}
              >
                {removing === m.name ? "移除中…" : "移除"}
              </button>
            </span>
          ))}
        </div>
      )}

      {ggufs.length > 0 && (
        <div className="chat-import">
          <span className="pf-label">平台 GGUF(可导入):</span>
          {ggufs.map((m) => (
            <span key={m.id} className="chat-import-item">
              <span className="pf-rule">{m.slug}-{m.version} · {m.quant}</span>
              {importedNames.has(m.ollama_name) ? (
                <span className="chat-import-actions">
                  <span className="pf-clean-none">已导入</span>
                  <button
                    type="button"
                    className="btn-inspect"
                    disabled={removing === m.ollama_name}
                    onClick={() => removeModel(m.ollama_name)}
                  >
                    {removing === m.ollama_name ? "移除中…" : "移除"}
                  </button>
                </span>
              ) : (
                <button type="button" className="btn-inspect" disabled={importingId === m.id} onClick={() => importModel(m)}>
                  {importingId === m.id ? "导入中…" : "导入"}
                </button>
              )}
            </span>
          ))}
        </div>
      )}

      {error && <p className="form-error">{error}</p>}

      <div className="chat-log" ref={logRef}>
        {messages.length === 0 && (
          <div className="job-empty">选择模型后发消息开始对话</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <span className="chat-role">{m.role === "user" ? "你" : "模型"}</span>
            <pre className="chat-bubble">{m.content}</pre>
          </div>
        ))}
        {sending && <div className="chat-msg assistant"><span className="chat-role">模型</span><pre className="chat-bubble">生成中…</pre></div>}
      </div>

      <div className="chat-input">
        <input
          className="input"
          value={input}
          placeholder={selected ? `向 ${selected} 发送消息…` : "先选择/导入模型"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
          }}
          disabled={sending}
        />
        <button className="btn-primary" onClick={send} disabled={sending || !selected}>
          {sending ? "生成中…" : "发送"}
        </button>
      </div>
    </section>
  );
}
