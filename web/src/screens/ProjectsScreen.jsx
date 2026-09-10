import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

// 编排项目页：列**编排项目**（项目 = 名称 + 画布配置的组合实体）。
// 新建 = 弹窗填名称即可；数据集进画布后由「数据源节点」的下拉绑定。

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
  const [creating, setCreating] = useState(false); // 新建弹窗
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null); // {kind, text}
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const ps = await api.get("/api/projects");
      setProjects(Array.isArray(ps) ? ps : []);
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
    setBusy(true);
    setNotice(null);
    try {
      const proj = await api.post("/api/projects", { name: name.trim() });
      setCreating(false);
      setName("");
      await load();
      onEnter(proj); // 创建即进入画布，数据集在画布里由数据源节点绑定
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
            项目 = 一套可持久化的画布配置；创建后进画布，数据集由数据源节点绑定。
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={() => setCreating(true)}>
          ＋ 新建项目
        </button>
      </div>

      {notice && <p className={`notice n-${notice.kind}`}>{notice}</p>}

      {loaded && projects.length === 0 ? (
        <div className="placeholder-card">
          还没有编排项目 · 点「＋ 新建项目」，命名后进画布开始编排
          <br />
          还没有数据集？先到「数据」上传文件。
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
                    {p.dataset
                      ? `素材：${p.dataset.name}${samples != null ? ` · ${samples} 样本` : ""}`
                      : "未绑定数据集 · 进画布后在数据源节点绑定"}
                  </small>
                  <span className="pi-meta">
                    {!p.dataset ? (
                      <span className="badge">未绑定</span>
                    ) : (
                      <span className={`badge ${built ? "st-built" : "st-ingested"}`}>
                        {built ? "可运行" : "未构建"}
                      </span>
                    )}
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

      {/* 新建项目弹窗：只填名称；数据集进画布后绑定 */}
      {creating && (
        <div className="modal-mask" onClick={(e) => e.target === e.currentTarget && setCreating(false)}>
          <div className="modal-box">
            <h3>新建编排项目</h3>
            <input
              className="input modal-input"
              placeholder="项目名称（如：青蛙文档 · 长块对比）"
              value={name}
              autoFocus
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onCreate()}
            />
            <p className="modal-hint">
              创建后进入画布；数据集在画布的「数据源」节点下拉绑定，参数随时可保存到项目。
            </p>
            <div className="modal-actions">
              <button type="button" className="ws-btn" onClick={() => setCreating(false)}>
                取消
              </button>
              <button type="button" className="ws-btn primary" onClick={onCreate} disabled={busy}>
                {busy ? "创建中…" : "创建并进入画布"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
