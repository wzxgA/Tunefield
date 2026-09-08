import { useEffect, useState } from "react";
import { api } from "../api";

// T6 · 数据集质检报告卡：样本数/语料/重复率/低质/长度 + 阈值告警与建议
const LEVEL_META = {
  ok: { text: "可训练", cls: "s-ok" },
  warn: { text: "有告警", cls: "rp-warn" },
  error: { text: "数据不足，暂不可训练", cls: "s-error" },
};

export default function DatasetReport({ datasetId, tick }) {
  const [state, setState] = useState({ loading: true, report: null });

  useEffect(() => {
    let alive = true;
    setState({ loading: true, report: null });
    api
      .get(`/api/datasets/${datasetId}/report`)
      .then((res) =>
        alive && setState({ loading: false, report: res.report || null })
      )
      .catch(() => alive && setState({ loading: false, report: null }));
    return () => {
      alive = false;
    };
  }, [datasetId, tick]);

  if (state.loading) return <div className="rp-note">读取质检报告…</div>;
  const report = state.report;
  if (!report)
    return (
      <div className="rp-note">尚未构建 · 点数据集右侧「构建」产出 JSONL 与质检报告</div>
    );

  const s = report.summary;
  const level = LEVEL_META[report.overall] || LEVEL_META.warn;
  const issues = report.checks.filter((c) => c.level !== "info");
  const files = s.files || {};

  return (
    <div className="rp-card">
      <div className="rp-head">
        <span className="pf-label">质检报告</span>
        <span className={`pf-status ${level.cls}`}>{level.text}</span>
      </div>
      <div className="rp-sum">
        <span className="rp-chip">样本 {s.sample_count} 条</span>
        <span className="rp-chip">语料 {s.corpus_mb} MB</span>
        <span className="rp-chip">块 {s.block_count}</span>
        <span className="rp-chip">
          重复率 {(s.dup_rate * 100).toFixed(1)}%
        </span>
        <span className="rp-chip">
          低质 {(s.lowq_rate * 100).toFixed(1)}%
        </span>
        <span className="rp-chip">
          长度 P50 {s.length.p50} / P90 {s.length.p90}
        </span>
        {(files.error || 0) + (files.unsupported || 0) > 0 && (
          <span className="rp-chip rp-warn">
            解析失败 {files.error || 0} · 不支持 {files.unsupported || 0}
          </span>
        )}
      </div>
      {issues.length > 0 && (
        <ul className="rp-checks">
          {issues.map((c, i) => (
            <li key={i} className={`rp-check rp-${c.level}`}>
              <b>{c.metric} {c.value}</b> —— {c.advice}
            </li>
          ))}
        </ul>
      )}
      {report.lowq_examples && report.lowq_examples.length > 0 && (
        <details className="rp-examples">
          <summary>低质块样例（{report.lowq_examples.length} 条）</summary>
          {report.lowq_examples.map((t, i) => (
            <pre key={i} className="rp-example">{t}</pre>
          ))}
        </details>
      )}
    </div>
  );
}
