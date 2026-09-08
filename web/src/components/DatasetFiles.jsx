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
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
