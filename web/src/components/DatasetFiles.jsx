import { useEffect, useState } from "react";
import { api } from "../api";

// T2 · 数据集逐文件解析状态 + 中间文本预览
// 点击数据集卡片「解析预览」后加载 /api/datasets/{id}/files：
// 每个文件显示 kind/language/状态/字符数，展开可看抽取出的中间文本。
const STATUS_LABEL = {
  ok: "已解析",
  empty: "无内容",
  error: "解析失败",
  unsupported: "不支持",
};

// 与后端 cleaning.RULE_LABELS 对应的展示名
const RULE_LABELS = {
  invisible: "控制字符",
  space: "全角空格",
  fullwidth: "全半角",
  html: "HTML",
  page: "页码行",
  boiler: "页眉页脚",
};

export default function DatasetFiles({ datasetId }) {
  const [state, setState] = useState({
    loading: true,
    error: "",
    files: [],
  });
  const [open, setOpen] = useState({}); // {fileName: bool}

  useEffect(() => {
    let alive = true;
    api
      .get(`/api/datasets/${datasetId}/files`)
      .then((res) =>
        alive && setState({ loading: false, error: "", files: res.files || [] })
      )
      .catch((e) =>
        alive &&
        setState({
          loading: false,
          error: String(e.message || e),
          files: [],
        })
      );
    return () => {
      alive = false;
    };
  }, [datasetId]);

  if (state.loading) {
    return <div className="ds-files-loading">正在解析文件…</div>;
  }
  if (state.error) {
    return <div className="pf-error">解析结果加载失败：{state.error}</div>;
  }
  if (state.files.length === 0) {
    return <div className="ds-files-loading">暂无文件（raw 落盘为空）</div>;
  }

  const count = (s) => state.files.filter((f) => f.status === s).length;
  const bad = count("error") + count("empty");
  const skipped = count("unsupported");

  return (
    <div className="ds-files">
      <div className="pf-sum">
        <span>{state.files.length} 个文件</span>
        <span className="pf-ok">{count("ok")} 解析成功</span>
        {bad > 0 && <span className="pf-err">{bad} 需关注</span>}
        {skipped > 0 && <span className="pf-warn">{skipped} 不支持</span>}
      </div>
      <div className="pf-list">
        {state.files.map((f) => (
          <div key={f.name} className={`pf-file${f.status !== "ok" ? " pf-file-bad" : ""}`}>
            <button
              type="button"
              className="pf-head"
              onClick={() => setOpen((o) => ({ ...o, [f.name]: !o[f.name] }))}
            >
              <span className="pf-name" title={f.name}>{f.name}</span>
              <span className="pf-tags">
                {f.kind === "code" ? (
                  <span className="pf-lang">{f.language}</span>
                ) : (
                  <span className="pf-kind">{f.kind}</span>
                )}
                <span className="pf-count">{f.chars}字 · {f.blocks}段</span>
                <span className={`pf-status s-${f.status}`}>
                  {STATUS_LABEL[f.status]}
                </span>
              </span>
            </button>
            {f.status !== "ok" && f.error && (
              <div className="pf-error">{f.error}</div>
            )}
            {open[f.name] && (
              <div className="pf-detail">
                {f.preview && (
                  <>
                    <div className="pf-label">中间文本（解析后）</div>
                    <pre className="pf-preview">{f.preview}</pre>
                  </>
                )}
                {f.clean && (
                  <div className="pf-clean">
                    <div className="pf-clean-head">
                      <span className="pf-label">清洗</span>
                      {f.clean.removed_chars > 0 ? (
                        <>
                          <span className="pf-clean-diff">
                            移除 {f.clean.removed_chars} 字
                          </span>
                          {Object.entries(f.clean.rules).map(([k, n]) => (
                            <span key={k} className="pf-rule">
                              {RULE_LABELS[k] || k} ×{n}
                            </span>
                          ))}
                        </>
                      ) : (
                        <span className="pf-clean-none">无需清洗</span>
                      )}
                    </div>
                    {f.clean.removed_chars > 0 && f.clean.preview && (
                      <pre className="pf-preview">{f.clean.preview}</pre>
                    )}
                  </div>
                )}
                {f.chunks && <ChunkSummary chunks={f.chunks} />}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// T4 · 切片预览：块数 / P50·P90 / 分桶直方图 / 少量样本块
function ChunkSummary({ chunks }) {
  const { count, stats, buckets, params, samples } = chunks;
  if (!count) return null;
  const maxCount = Math.max(...buckets.map((b) => b.count), 1);
  const bucketLabel = (b) => (b.max === null ? "∞" : String(b.max));

  return (
    <div className="pf-chunks">
      <div className="pf-clean-head">
        <span className="pf-label">切片</span>
        <span className="pf-clean-diff">
          {count} 块 · P50 {stats.p50} / P90 {stats.p90} tok
        </span>
        <span className="pf-rule">块长 {params.size} · 重叠 {params.overlap}</span>
        {stats.max > params.size && (
          <span className="pf-warn">含超长块（函数/句子不切）</span>
        )}
      </div>
      <div className="ch-hist" title="块长分布（近似 token）">
        {buckets.map((b, i) => (
          <div key={i} className="ch-col" title={`${b.min}–${b.max ?? "∞"}: ${b.count} 块`}>
            <div className="ch-bar">
              <div
                className="ch-bar-fill"
                style={{ height: `${Math.max((b.count / maxCount) * 100, b.count ? 8 : 2)}%` }}
              />
            </div>
            <span className="ch-col-label">{bucketLabel(b)}</span>
          </div>
        ))}
      </div>
      {samples.length > 0 && (
        <ul className="ch-samples">
          {samples.map((s) => (
            <li key={s.index} className="ch-sample" title={s.text}>
              <span className="ch-s-idx">#{s.index}</span>
              <span className="ch-s-tok">~{s.tokens} tok</span>
              {s.label && <span className="ch-s-label">{s.label}</span>}
              {s.oversized && <span className="pf-warn">超长</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
