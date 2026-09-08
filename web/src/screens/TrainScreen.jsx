import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { useEvents } from "../hooks/useEvents";
import LossChart from "../components/LossChart";

const STATUS_LABEL = {
  queued: "排队中",
  pending_gpu: "等待 GPU",
  running: "训练中",
  done: "已完成",
  failed: "失败",
};

// 后端 loss_json 形如 "[1,0.9],[2,0.7]"
function parseLoss(lossJson) {
  if (!lossJson) return [];
  try {
    return JSON.parse(`[${lossJson}]`).map(([step, value]) => ({ step, value }));
  } catch {
    return [];
  }
}

export default function TrainScreen() {
  const [jobs, setJobs] = useState([]);
  const [lossMap, setLossMap] = useState({});
  const [selectedId, setSelectedId] = useState(null);
  const [domain, setDomain] = useState("demo");
  const [kind, setKind] = useState("finetune");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");

  // 初始：拉取任务列表 + 历史 loss（刷新后恢复现场）
  const load = useCallback(async () => {
    try {
      const list = await api.get("/api/jobs");
      setJobs(list);
      const lm = {};
      for (const j of list) lm[j.id] = parseLoss(j.loss_json);
      setLossMap(lm);
      if (list.length > 0) {
        setSelectedId((prev) => prev ?? list[list.length - 1].id);
      }
    } catch {
      /* 后端未就绪时保持空态 */
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  // 事件订阅：job.created / job.status / job.loss 实时更新，无需轮询
  const onEvent = useCallback((msg) => {
    const { type, data } = msg;
    if (type === "job.created") {
      setJobs((js) =>
        js.some((j) => j.id === data.id)
          ? js
          : [...js, { ...data, progress: 0, loss_json: null }]
      );
      setSelectedId((cur) => cur ?? data.id);
    } else if (type === "job.status") {
      setJobs((js) =>
        js.map((j) =>
          j.id === data.id
            ? {
                ...j,
                status: data.status,
                progress: data.progress ?? j.progress,
                error: data.error ?? j.error,
              }
            : j
        )
      );
    } else if (type === "job.loss") {
      setLossMap((m) => ({
        ...m,
        [data.id]: [...(m[data.id] || []), { step: data.step, value: data.value }],
      }));
    }
  }, []);
  const connected = useEvents(onEvent);

  const selected = useMemo(
    () => jobs.find((j) => j.id === selectedId) ?? null,
    [jobs, selectedId]
  );
  const points = lossMap[selectedId] ?? [];

  const createJob = async () => {
    setCreating(true);
    setCreateError("");
    try {
      await api.post("/api/jobs", { domain: domain || "demo", kind });
    } catch (e) {
      setCreateError(String(e.message || e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <section className="screen train-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">训练任务</h2>
          <p className="screen-hint">创建任务后，状态与 loss 曲线经事件流实时推送，无需刷新。</p>
        </div>
        <span
          className={`ws-dot${connected ? " ws-on" : ""}`}
          title={connected ? "事件流已连接" : "事件流未连接（自动重试中）"}
        />
      </div>

      <div className="train-form">
        <input
          className="input"
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
          placeholder="领域名称"
        />
        <select className="input" value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="finetune">微调（引擎A）</option>
          <option value="pretrain">从零预训练（引擎B）</option>
        </select>
        <button className="btn-primary" onClick={createJob} disabled={creating}>
          {creating ? "创建中…" : "创建任务"}
        </button>
      </div>
      {createError && <p className="form-error">{createError}</p>}

      <div className="train-layout">
        <div className="job-list">
          {jobs.length === 0 && <div className="job-empty">暂无任务 · 创建一个试试</div>}
          {[...jobs].reverse().map((j) => (
            <button
              key={j.id}
              type="button"
              className={`job-item${j.id === selectedId ? " job-active" : ""}`}
              onClick={() => setSelectedId(j.id)}
            >
              <div className="job-line1">
                <span className="job-domain">{j.domain}</span>
                <span className={`badge st-${j.status}`}>
                  {STATUS_LABEL[j.status] || j.status}
                </span>
              </div>
              <div className="job-line2">
                <span className="job-id">{j.id.slice(0, 10)}</span>
                <span className="job-kind">{j.kind === "pretrain" ? "预训练" : "微调"}</span>
              </div>
              {(j.status === "running" || j.status === "queued") && (
                <div className="job-progress">
                  <div
                    className="job-progress-fill"
                    style={{ width: `${Math.round((j.progress ?? 0) * 100)}%` }}
                  />
                </div>
              )}
            </button>
          ))}
        </div>

        <div className="job-detail">
          {!selected ? (
            <div className="job-empty">选择左侧任务查看详情</div>
          ) : (
            <>
              <div className="detail-head">
                <span className="detail-domain">{selected.domain}</span>
                <span className={`badge st-${selected.status}`}>
                  {STATUS_LABEL[selected.status] || selected.status}
                </span>
              </div>
              <div className="detail-meta">
                <span>任务 {selected.id}</span>
                <span>{selected.kind === "pretrain" ? "从零预训练" : "微调"}</span>
              </div>
              {selected.error && <p className="detail-error">{selected.error}</p>}
              {selected.status === "running" && (
                <div className="job-progress big">
                  <div
                    className="job-progress-fill"
                    style={{ width: `${Math.round((selected.progress ?? 0) * 100)}%` }}
                  />
                </div>
              )}
              <div className="loss-panel">
                <div className="loss-title">
                  Loss 曲线
                  {points.length > 0 && <span className="loss-count">{points.length} 个采样点</span>}
                </div>
                {points.length === 0 ? (
                  <div className="job-empty">等待 loss 采样…</div>
                ) : (
                  <LossChart points={points} />
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
