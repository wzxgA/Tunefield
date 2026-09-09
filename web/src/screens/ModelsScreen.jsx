import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

// 模型屏（T9/T10 API 驱动）：GGUF 量化产物清单 + 一键导入 Ollama + 指纹信息。
// 数据源：GET /api/models（adapters+quantized 联表）；已导入态来自 GET /api/chat/models。

const fmtSize = (bytes) => {
  const n = Number(bytes) || 0;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
};

export default function ModelsScreen() {
  const [models, setModels] = useState([]);
  const [imported, setImported] = useState(new Set()); // ollama_name 集合
  const [importing, setImporting] = useState(""); // 导入中的 model id
  const [notice, setNotice] = useState("");
  const [noticeKind, setNoticeKind] = useState("");

  const load = useCallback(async () => {
    try {
      const [rows, ollamaModels] = await Promise.all([
        api.get("/api/models"),
        api.get("/api/chat/models").catch(() => ({ models: [] })),
      ]);
      setModels(Array.isArray(rows) ? rows : []);
      setImported(
        new Set(
          (ollamaModels?.models || []).map((m) => m.name || m.model || "").filter(Boolean),
        ),
      );
    } catch {
      /* 后端未就绪保持空态 */
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onImport = async (m) => {
    setImporting(m.id);
    setNotice("");
    try {
      await api.post(`/api/models/${m.id}/import`, {});
      setNotice(`已导入 Ollama：「${m.ollama_name}」· 可在对话屏直接使用`);
      setNoticeKind("ok");
      await load();
    } catch (e) {
      setNotice(`导入失败：${String(e.message || e)}`);
      setNoticeKind("err");
    } finally {
      setImporting("");
    }
  };

  return (
    <section className="screen models-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">模型库</h2>
          <p className="screen-hint">
            训练产物的 GGUF 量化清单（LoRA 合并或预训练完整权重 → convert → quantize）。
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={load}>
          刷新清单
        </button>
      </div>

      {notice && <p className={`notice n-${noticeKind}`}>{notice}</p>}

      <div className="ds-block">
        <div className="ds-title">
          GGUF 产物
          <span className="ds-count">{models.length} 个</span>
        </div>
        {models.length === 0 ? (
          <div className="job-empty">
            暂无产物 · 完成训练后导出，或用一键端到端自动产出
          </div>
        ) : (
          <div className="ds-list">
            {[...models].reverse().map((m) => {
              const ok = imported.has(m.ollama_name);
              return (
                <div key={m.id} className="ds-item">
                  <div className="ds-line1">
                    <span className="ds-name">{m.domain || m.slug}</span>
                    <span className="ds-actions">
                      <span className="badge m-quant">{m.quant}</span>
                      <span className={`badge ${ok ? "st-built" : "st-ingested"}`}>
                        {ok ? "已导入" : "未导入"}
                      </span>
                      <button
                        type="button"
                        className="btn-inspect"
                        disabled={ok || importing === m.id}
                        onClick={() => onImport(m)}
                        title={ok ? "已导入 Ollama" : `导入为 ${m.ollama_name}`}
                      >
                        {importing === m.id ? "导入中…" : ok ? "已导入" : "导入 Ollama"}
                      </button>
                    </span>
                  </div>
                  <div className="ds-line2">
                    <span className="ds-id">{m.ollama_name}</span>
                    <span className="ds-fp">
                      {fmtSize(m.size_bytes)} · job {String(m.job_id).slice(0, 8)}
                      {m.created_at ? ` · ${String(m.created_at).slice(0, 16).replace("T", " ")}` : ""}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
