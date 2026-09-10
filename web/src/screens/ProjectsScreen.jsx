import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

// 编排项目页：列**编排项目**（项目 = 命名 + 绑定数据集 + 画布配置的组合实体），
// 不是数据集列表——数据集是素材，上传多少个与这里无关。
// 新建：命名 + 选数据集；点项目卡 → 进入整屏画布（画布参数持久化在项目上）。

const samplesOf = (dataset) => {
  try {
    const report = JSON.parse(dataset?.stats_json || "null");
    const n = report?.summary?.sample_count;
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
};

export default function ProjectsScreen({ onEnter, onUpload }) {
  const [projects, setProjects] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [creating, setCreating] = useState(false); // 展开新建表单
  const [name, setName] = useState("");
  const [dsId, setDsId] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null); // {kind, text}
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const [ps, ds] = await Promise.all([
        api.get("/api/projects"),
        api.get("/api/datasets").catch(() => []),
      ]);
      setProjects(Array.isArray(ps) ? ps : []);
      setDatasets(Array.isArray(ds) ? ds : []);
    } catch {
      /* 后端未就绪保持空态 */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onCreate = async () => {
    if (!name.trim()) {
      setNotice({ kind: "err", text: "请填写项目名称" });
      return;
    }
    if (!dsId) {
      setNotice({ kind: "err", text: "请选择要编排的数据集" });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      const proj = await api.post("/api/projects", { name: name.trim(), dataset_id: dsId });
      setCreating(false);
      setName("");
      setNotice({ kind: "ok", text: `已创建项目「${proj.name}」` });
      await load();
      onEnter(proj); // 创建即进入画布
    } catch (e) {
      setNotice({ kind: "err", text: `创建失败：${String(e.message || e)}` });
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (p) => {
    if (!window.confirm(`删除项目「${p.name}」？\n数据集与已有运行记录不受影响。`)) return;
    try {
      await api.del(`/api/projects/${p.id}`);
      await load();
    } catch (e) {
      setNotice({ kind: "err", text: `删除失败：${String(e.message || e)}` });
    }
  };

  return (
    <section className="screen projects-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">编排项目</h2>
          <p className="screen-hint">
            项目 = 一份素材数据集 + 一套可持久化的画布配置；同一数据集可建多个项目对比参数。
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={() => setCreating((v) => !v)}>
          {creating ? "收起新建" : "＋ 新建项目"}
        </button>
      </div>

      {notice && <p className={`notice n-${notice.kind}`}>{notice}</p>}

      {creating && (
        <div className="proj-create glass-card">
          <input
            className="input"
            placeholder="项目名称（如：青蛙文档 · 长块对比）"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <select
            className="input"
            value={dsId}
            onChange={(e) => setDsId(e.target.value)}
          >
            <option value="">选择数据集…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
                {d.status === "built" ? "（已构建）" : "（未构建）"}
              </option>
            ))}
          </select>
          <button type="button" className="btn-primary" onClick={onCreate} disabled={busy}>
            {busy ? "创建中…" : "创建并进入画布"}
          </button>
        </div>
      )}

      {loaded && projects.length === 0 ? (
        <div className="placeholder-card">
          {datasets.length === 0
            ? "还没有项目 · 先到「数据」上传一批文件，再回到这里新建编排项目"
            : "还没有编排项目 · 点「＋ 新建项目」，选一份已上传的数据集开始"}
        </div>
      ) : (
        <div className="proj-grid">
          {projects.map((p) => {
            const samples = samplesOf(p.dataset);
            const built = p.dataset?.status === "built";
            return (
              <div key={p.id} className="pcard pcard-static">
                <div className="pcard-head" onClick={() => onEnter(p)} role="button" tabIndex={0}>
                  <span className="pi-ic">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="5.5" cy="6" r="2.4" />
                      <circle cx="18.5" cy="6" r="2.4" />
                      <circle cx="12" cy="18" r="2.6" />
                      <path d="M7.5 7.2 10.4 15.6M16.5 7.2l-3 8.4M7.9 6h8.2" />
                    </svg>
                  </span>
                  <b>{p.name}</b>
                  <small>
                    素材：{p.dataset?.name || "（数据集已删除）"}
                    {samples != null ? ` · ${samples} 样本` : ""}
                  </small>
                  <span className="pi-meta">
                    <span className={`badge ${built ? "st-built" : "st-ingested"}`}>
                      {built ? "可运行" : "未构建"}
                    </span>
                    {p.config ? <span className="badge m-quant">已存配置</span> : null}
                    <span className="pi-go">进入编排 →</span>
                  </span>
                </div>
                <button
                  type="button"
                  className="btn-danger pcard-del"
                  onClick={() => onDelete(p)}
                  title="删除项目（不影响数据集）"
                >
                  删除
                </button>
              </div>
            );
          })}
          <button type="button" className="pcard new" onClick={() => setCreating(true)}>
            ＋ 新建编排项目
          </button>
        </div>
      )}

      <div className="wire" style={{ marginTop: 14 }}>
        画布执行序由平台编排固定（构建 → 微调 → 导出 → 导入）；画布内的调参会保存到项目，
        下次进入自动恢复。
      </div>
    </section>
  );
}
