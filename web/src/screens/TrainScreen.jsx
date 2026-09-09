import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useEvents } from "../hooks/useEvents";
import LossChart from "../components/LossChart";

// T7：训练任务可选「已构建」数据集；任务运行中实时推送 loss 采样与日志行
// （日志来自 engine 子进程逐行监控 → job.log 事件）
// T11：顶部「一键端到端」——选任意已上传数据集，自动走
// 构建→训练→导出→导入 Ollama，阶段时间线经 run.created/status/stage 事件驱动

const STATUS_LABEL = {
  queued: "排队中",
  pending_gpu: "等待 GPU",
  running: "训练中",
  done: "已完成",
  failed: "失败",
};
const RUN_STATUS_LABEL = {
  queued: "排队中",
  pending_gpu: "等待 GPU",
  running: "运行中",
  done: "已跑通",
  failed: "失败",
};

// 阶段展示顺序与文案（后端 key → 中文）
const PHASE_LABEL = { build: "构建", train: "训练", export: "导出", import: "导入" };
const STAGE_LABEL = {
  ok: "完成",
  failed: "失败",
  skipped: "跳过",
  running: "进行中",
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

// 把 stage 事件/响应合并进 run.timeline（按 key 去重置换）
function upsertPhase(timeline, phase) {
  const rest = (timeline || []).filter((p) => p.key !== phase.key);
  return [...rest, phase];
}

function fmtDuration(s) {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m${Math.round(s % 60)}s`;
}

export default function TrainScreen() {
  const [jobs, setJobs] = useState([]);
  const [datasets, setDatasets] = useState([]); // 已构建数据集（可训练）
  const [allDatasets, setAllDatasets] = useState([]); // 所有数据集（可一键端到端）
  const [lossMap, setLossMap] = useState({});
  const [logs, setLogs] = useState({}); // {jobId: [line,...]}
  const [selectedId, setSelectedId] = useState(null);
  const [datasetSel, setDatasetSel] = useState("");
  const [kind, setKind] = useState("finetune");
  const [epochs, setEpochs] = useState(3);
  const [rec, setRec] = useState(null); // T8 推荐配置
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [models, setModels] = useState([]); // T9 GGUF 清单
  const [exporting, setExporting] = useState(false);

  // T11 端到端 run
  const [runs, setRuns] = useState([]);
  const [runDetailLog, setRunDetailLog] = useState({}); // {runId: [line,...]}
  const [runSel, setRunSel] = useState(null);
  const [runDatasetSel, setRunDatasetSel] = useState("");
  const [runEpochs, setRunEpochs] = useState(3);
  const [runCreating, setRunCreating] = useState(false);
  const [runError, setRunError] = useState("");
  const logRef = useRef(null);

  // 初始：拉取任务列表 + 历史 loss + 已构建数据集（刷新后恢复现场）
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
    try {
      const ds = await api.get("/api/datasets");
      setAllDatasets(ds || []);
      const built = (ds || []).filter((d) => d.status === "built");
      setDatasets(built);
      if (built.length > 0) setDatasetSel((cur) => cur || built[0].id);
      if (ds?.length > 0) setRunDatasetSel((cur) => cur || ds[ds.length - 1].id);
    } catch {
      /* ignore */
    }
    try {
      setRuns(await api.get("/api/runs"));
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  // T8：切换数据集 → 拉取推荐配置（显存档位/基座/轮次/学习率…）
  useEffect(() => {
    if (!datasetSel) {
      setRec(null);
      return;
    }
    let alive = true;
    api
      .get(`/api/train/preview?dataset_id=${datasetSel}`)
      .then((r) => {
        if (!alive) return;
        setRec(r.recommend || null);
        if (r.recommend?.epochs) setEpochs(r.recommend.epochs);
      })
      .catch(() => alive && setRec(null));
    return () => {
      alive = false;
    };
  }, [datasetSel]);

  // 选中任务时拉取其日志尾部（进程内缓冲，供刷新后恢复展示）
  const openJob = useCallback(
    (id) => {
      setSelectedId(id);
      api
        .get(`/api/jobs/${id}`)
        .then((detail) => {
          if (detail && Array.isArray(detail.log)) {
            setLogs((m) => ({ ...m, [id]: detail.log }));
          }
        })
        .catch(() => {});
    },
    []
  );

  // T9：GGUF 清单 + 导出
  const loadModels = useCallback(async () => {
    try {
      setModels(await api.get("/api/models"));
    } catch {
      /* 后端未就绪保持空态 */
    }
  }, []);
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const onExport = async (jobId) => {
    setExporting(true);
    try {
      await api.post(`/api/jobs/${jobId}/export`, {});
      await loadModels();
    } catch (e) {
      setCreateError(`导出失败：${String(e.message || e)}`);
    } finally {
      setExporting(false);
    }
  };

  // T11：一键端到端
  const createRun = async () => {
    const dataset = allDatasets.find((d) => d.id === runDatasetSel);
    if (!dataset) {
      setRunError("请先上传一个数据集");
      return;
    }
    setRunCreating(true);
    setRunError("");
    try {
      await api.post("/api/runs", {
        dataset_id: dataset.id,
        epochs: Number(runEpochs) || undefined,
      });
    } catch (e) {
      setRunError(String(e.message || e));
    } finally {
      setRunCreating(false);
    }
  };

  const openRun = useCallback((id) => {
    setRunSel(id);
    if (runDetailLog[id]) return; // 已有进程内日志，直接显示
    api
      .get(`/api/runs/${id}`)
      .then((detail) => {
        if (detail && Array.isArray(detail.log)) {
          setRunDetailLog((m) => ({ ...m, [id]: detail.log }));
        }
      })
      .catch(() => {});
  }, [runDetailLog]);

  const fmtTime = (iso) => (iso ? iso.slice(11, 19) : "…");

  // 事件订阅：job.* 与 run.* 实时更新，无需轮询
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
    } else if (type === "job.log") {
      // T7：引擎日志行实时增量
      setLogs((m) => {
        const lines = [...(m[data.id] || []), data.line].slice(-800);
        return { ...m, [data.id]: lines };
      });
    } else if (type === "run.created") {
      setRuns((rs) =>
        rs.some((r) => r.id === data.id)
          ? rs
          : [{ ...data, progress: 0, timeline: [] }, ...rs]
      );
    } else if (type === "run.status") {
      setRuns((rs) =>
        rs.map((r) =>
          r.id === data.id
            ? {
                ...r,
                status: data.status,
                progress: data.progress ?? r.progress,
                error: data.error ?? r.error,
              }
            : r
        )
      );
    } else if (type === "run.stage") {
      // T11：阶段事件即推进时间线（含失败/跳过），无需轮询
      const { id, ...phase } = data;
      setRuns((rs) =>
        rs.map((r) =>
          r.id === id ? { ...r, timeline: upsertPhase(r.timeline, phase) } : r
        )
      );
    }
  }, []);
  const connected = useEvents(onEvent);
  // 日志面板自动滚到底部
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [selectedId, logs]);

  const selected = useMemo(
    () => jobs.find((j) => j.id === selectedId) ?? null,
    [jobs, selectedId]
  );
  const points = lossMap[selectedId] ?? [];

  const createJob = async () => {
    setCreating(true);
    setCreateError("");
    const dataset = datasets.find((d) => d.id === datasetSel);
    if (!dataset) {
      setCreateError("请先上传并「构建」一个数据集，再创建训练任务");
      setCreating(false);
      return;
    }
    try {
      await api.post("/api/jobs", {
        dataset_id: dataset.id,
        domain: dataset.name,
        kind,
        overrides: { epochs: Number(epochs) || 3 },
      });
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
          <p className="screen-hint">
            选择已构建数据集创建任务；或对任意上传数据一键「端到端」自动完成构建 → 微调 → 导出 → 导入对话。
          </p>
        </div>
        <span
          className={`ws-dot${connected ? " ws-on" : ""}`}
          title={connected ? "事件流已连接" : "事件流未连接（自动重试中）"}
        />
      </div>

      {/* ===== T11 一键端到端 + 阶段时间线 ===== */}
      <div className="run-panel">
        <div className="run-form">
          <select
            className="input"
            value={runDatasetSel}
            onChange={(e) => setRunDatasetSel(e.target.value)}
            title="端到端数据：任意已上传数据集（未构建也会自动构建）"
          >
            {allDatasets.length === 0 && <option value="">（暂无数据集 · 请先到「数据」页上传）</option>}
            {allDatasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}（{d.status === "built" ? "已构建" : "未构建"}）
              </option>
            ))}
          </select>
          <input
            className="input epochs-input"
            type="number"
            min={1}
            max={20}
            value={runEpochs}
            onChange={(e) => setRunEpochs(e.target.value)}
            title="训练轮数（epochs）"
          />
          <button className="btn-primary" onClick={createRun} disabled={runCreating}>
            {runCreating ? "创建中…" : "一键端到端"}
          </button>
          {runError && <span className="form-error inline">{runError}</span>}
        </div>
        <div className="run-list">
          {runs.length === 0 && (
            <div className="job-empty">还没有端到端运行 · 选数据集后点「一键端到端」</div>
          )}
          {runs.map((r) => {
            const tl = r.timeline || [];
            const expanded = r.id === runSel;
            return (
              <div key={r.id} className={`run-item${expanded ? " run-open" : ""}`}>
                <button type="button" className="run-head" onClick={() => openRun(r.id)}>
                  <span className="job-domain">{r.domain}</span>
                  <span className={`badge st-${r.status}`}>
                    {RUN_STATUS_LABEL[r.status] || r.status}
                  </span>
                  {r.dataset_name && <span className="run-ds">{r.dataset_name}</span>}
                  {tl.length > 0 && (
                    <span className="run-phases">
                      {Object.keys(PHASE_LABEL).map((k) => {
                        const p = tl.find((x) => x.key === k);
                        return (
                          <span
                            key={k}
                            className={`run-phase st-p-${p ? p.status : "pending"}`}
                            title={p ? `${PHASE_LABEL[k]}：${STAGE_LABEL[p.status] || p.status}` : `${PHASE_LABEL[k]}：等待`}
                          >
                            {PHASE_LABEL[k]}
                          </span>
                        );
                      })}
                    </span>
                  )}
                </button>
                {expanded && (
                  <div className="run-detail">
                    {r.error && <p className="pf-error">{r.error}</p>}
                    {tl.length === 0 && <div className="job-empty">等待阶段推进…</div>}
                    {tl.map((p) => (
                      <div key={p.key} className="tl-row">
                        <span className="tl-stage">
                          <span className={`run-phase st-p-${p.status}`}>{PHASE_LABEL[p.key] || p.key}</span>
                          <span className="tl-status">{STAGE_LABEL[p.status] || p.status}</span>
                        </span>
                        <span className="tl-time">
                          {fmtTime(p.started_at)}–{fmtTime(p.finished_at)} · {fmtDuration(p.duration_s)}
                        </span>
                        {p.detail && <span className="tl-detail">{p.detail}</span>}
                      </div>
                    ))}
                    {runDetailLog[r.id]?.length > 0 && (
                      <pre className="job-log run-log">{runDetailLog[r.id].join("\n")}</pre>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ===== 普通训练任务创建 ===== */}
      <div className="train-form">
        <select
          className="input"
          value={datasetSel}
          onChange={(e) => setDatasetSel(e.target.value)}
          title="选择已构建的数据集"
        >
          {datasets.length === 0 && <option value="">（暂无已构建数据集）</option>}
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
        <select className="input" value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="finetune">微调（引擎A）</option>
          <option value="pretrain" disabled>预训练（引擎B · T13 接入）</option>
        </select>
        <input
          className="input epochs-input"
          type="number"
          min={1}
          max={20}
          value={epochs}
          onChange={(e) => setEpochs(e.target.value)}
          title="训练轮数（epochs）"
        />
        <button className="btn-primary" onClick={createJob} disabled={creating}>
          {creating ? "创建中…" : "创建任务"}
        </button>
      </div>
      {createError && <p className="form-error">{createError}</p>}

      {rec && (
        <div className="rec-box">
          <div className="rec-head">
            <span className="pf-label">推荐配置</span>
            <span className="pf-rule">
              {rec._meta?.tier || "-"} · 语料 {rec._meta?.corpus_mb ?? "?"}MB
            </span>
          </div>
          <div className="rec-chips">
            <span className="pf-rule">基座 {rec.base_model}</span>
            <span className="pf-rule">轮次 {rec.epochs}</span>
            <span className="pf-rule">lr {rec.learning_rate}</span>
            <span className="pf-rule">cutoff {rec.cutoff_len}</span>
            <span className="pf-rule">{rec.quantization_bit}bit QLoRA</span>
            <span className="pf-rule">LoRA {rec.lora_rank}/{rec.lora_alpha}</span>
          </div>
        </div>
      )}

      <div className="train-layout">
        <div className="job-list">
          {jobs.length === 0 && <div className="job-empty">暂无任务 · 创建一个试试</div>}
          {[...jobs].reverse().map((j) => (
            <button
              key={j.id}
              type="button"
              className={`job-item${j.id === selectedId ? " job-active" : ""}`}
              onClick={() => openJob(j.id)}
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
              {selected.status === "done" && (
                <div className="gg-panel">
                  <div className="loss-title">
                    量化导出
                    <button
                      type="button"
                      className="btn-ghost"
                      disabled={exporting}
                      onClick={() => onExport(selected.id)}
                    >
                      {exporting ? "导出中…" : "导出 GGUF"}
                    </button>
                  </div>
                  {models.filter((m) => m.job_id === selected.id).length === 0 ? (
                    <div className="job-empty">尚无 GGUF 产物 · 点「导出 GGUF」生成</div>
                  ) : (
                    <div className="gg-list">
                      {models
                        .filter((m) => m.job_id === selected.id)
                        .map((m) => (
                          <div key={m.id} className="gg-row">
                            <span className="pf-rule">{m.quant}</span>
                            <span className="gg-size">
                              {(m.size_bytes / 1024 / 1024).toFixed(1)} MB
                            </span>
                            <a className="gg-dl" href={`/api/models/${m.id}/download`}>
                              下载
                            </a>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}
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
              <div className="log-panel">
                <div className="loss-title">
                  训练日志
                  <span className="loss-count">{(logs[selected.id] || []).length} 行</span>
                </div>
                <pre ref={logRef} className="job-log">
                  {(logs[selected.id] || []).join("\n") || "暂无日志（引擎尚未输出）…"}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
