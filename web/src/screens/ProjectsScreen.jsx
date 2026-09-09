import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

// 编排项目选择页（W3）：真实数据集卡片 → 进入整屏编排画布（W4 挂载）。
// 原型形态见 yuanxing/index-3.html 的 view-projects：玻璃卡片网格 + 状态徽章。

const samplesOf = (d) => {
  try {
    const report = JSON.parse(d.stats_json || "null");
    const n = report?.summary?.sample_count;
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
};

export default function ProjectsScreen({ onEnter, onNew }) {
  const [datasets, setDatasets] = useState([]);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const rows = await api.get("/api/datasets");
      setDatasets(Array.isArray(rows) ? rows : []);
    } catch {
      /* 后端未就绪保持空态 */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <section className="screen projects-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">编排项目</h2>
          <p className="screen-hint">
            选择数据集进入整屏编排画布：数据源 → 解析/清洗/切片 → 训练 → 导出 → 对话。
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={onNew}>
          ＋ 新建项目
        </button>
      </div>

      {loaded && datasets.length === 0 ? (
        <div className="placeholder-card">
          暂无项目 · 先到「数据」上传一批文件，再回到这里进入编排
        </div>
      ) : (
        <div className="proj-grid">
          {[...datasets].reverse().map((d) => {
            const samples = samplesOf(d);
            const built = d.status === "built";
            return (
              <button
                key={d.id}
                type="button"
                className="pcard"
                onClick={() => onEnter(d)}
              >
                <span className="pi-ic">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="5.5" cy="6" r="2.4" />
                    <circle cx="18.5" cy="6" r="2.4" />
                    <circle cx="12" cy="18" r="2.6" />
                    <path d="M7.5 7.2 10.4 15.6M16.5 7.2l-3 8.4M7.9 6h8.2" />
                  </svg>
                </span>
                <b>{d.name}</b>
                <small>
                  {samples != null ? `${samples} 样本 · ` : ""}
                  指纹 {String(d.content_hash || "").slice(0, 8)}…
                </small>
                <span className="pi-meta">
                  <span className={`badge ${built ? "st-built" : "st-ingested"}`}>
                    {built ? "可编排" : "待构建"}
                  </span>
                  <span className="pi-go">进入编排 →</span>
                </span>
              </button>
            );
          })}
          <button type="button" className="pcard new" onClick={onNew}>
            ＋ 新建编排项目
          </button>
        </div>
      )}

      <div className="wire" style={{ marginTop: 14 }}>
        画布执行序由平台编排固定（构建 → 微调 → 导出 → 导入），画布用于流程可视化、
        参数入口与运行状态跟踪。
      </div>
    </section>
  );
}
