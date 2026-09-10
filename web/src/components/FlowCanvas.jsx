import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useEvents } from "../hooks/useEvents.js";

// Flow 画布（W4，依 yuanxing/index-3.html 原型平移为 React）：
// 节点 = pipeline 阶段；交互 = 拖空白平移 / 滚轮缩放 / 拖节点 / 右端口→左端口连线 /
// 点选删除 / 适应视图 / 自动布局(拓扑排序) / 节点库点击添加 / 选中节点参数编辑。
// 执行序由后端固定编排；本画布是流程可视化 + 参数入口 + 状态跟踪（运行状态 W5 接入）。

const NODE_W = 176;
const NODE_H = 96; // 固定高：端口位置 = (x, y + H/2)，连线纯 state 计算零滞后

const ICONS = {
  data: (
    <>
      <ellipse cx="12" cy="5.6" rx="7.4" ry="2.9" />
      <path d="M4.6 5.6v12.4c0 1.6 3.3 2.9 7.4 2.9s7.4-1.3 7.4-2.9V5.6" />
      <path d="M4.6 11.8c0 1.6 3.3 2.9 7.4 2.9s7.4-1.3 7.4-2.9" />
    </>
  ),
  parse: (
    <>
      <path d="M4 6h11M4 12h6M4 18h13" />
      <circle cx="18" cy="6" r="1.7" />
      <circle cx="14" cy="12" r="1.7" />
      <circle cx="20" cy="18" r="1.7" />
    </>
  ),
  clean: (
    <>
      <circle cx="10.5" cy="10.5" r="6.4" />
      <path d="M15 15 20.5 20.5" />
    </>
  ),
  split: (
    <>
      <path d="M4 12h6.5m3 0h6.5" />
      <path d="M10.5 8.5 14 12l-3.5 3.5" />
    </>
  ),
  text: <path d="M4 7V5h16v2M12 5v14M8 19h8" />,
  check: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12.2l2.7 2.7 5.5-6" />
    </>
  ),
  train: <path d="M2.5 12h3.2l2.4-6.2 3.6 12.4 2.6-6.2h7.2" />,
  export: (
    <>
      <path d="M12 2.6 20 7.4v9.2l-8 4.8-8-4.8V7.4z" />
      <path d="M4 7.4l8 4.8 8-4.8" />
      <path d="M12 12.2V21" />
    </>
  ),
  chat: (
    <>
      <path d="M20.8 11.9a7.9 7.9 0 0 1-11.6 6.9L4.2 20.4l1.6-4.8a7.9 7.9 0 1 1 15-3.7z" />
      <path d="M8.4 10.2h7.2M8.4 13.4h4.6" />
    </>
  ),
};

// 阶段参数 schema：def 即真实默认值；split/text/train/export 的 key 与
// POST /api/runs overrides 对齐（W5 直接读取生成请求）
// params = 可配置参数（输入框/下拉，运行时生效或暂存）；info = 后端内置逻辑的只读说明
// （渲染为徽章而非输入框，避免"能改但没用"的伪配置）
const STAGES = [
  {
    key: "data", label: "数据源", sub: "DATA",
    params: [{ k: "dataset", label: "绑定数据集", type: "dataset" }],
    info: ["运行时读取该数据集原始文件", "格式自动识别：txt/md/pdf/docx/代码"],
  },
  {
    key: "parse", label: "解析", sub: "PARSE", params: [],
    info: ["txt / md / pdf / docx / 代码", "按扩展名自动路由解析器", "逐文件容错，坏文件不中断"],
  },
  {
    key: "clean", label: "清洗", sub: "CLEAN", params: [],
    info: ["控制字符/乱码", "特殊空格", "全半角", "HTML 残留", "页码行", "页眉页脚", "代码文件只做字符归一"],
  },
  {
    key: "split", label: "切片", sub: "CHUNK",
    params: [{ k: "chunk_size", label: "块长 (tok)", def: "768" }, { k: "overlap", label: "重叠 (tok)", def: "96" }],
    info: ["代码按函数/类边界不切半"],
  },
  {
    key: "text", label: "指令化", sub: "TEXT",
    params: [{ k: "template", label: "模板", def: "continuation" }],
    info: ["continuation 续写 / qa 问答"],
  },
  {
    key: "check", label: "质检", sub: "CHECK", params: [],
    info: ["样本数 / 重复率", "长度分布 P50·P90", "低质占比", "结论与建议入报告"],
  },
  {
    key: "train", label: "训练", sub: "TRAIN",
    params: [{ k: "engine", label: "引擎 (A/B)", def: "A" }, { k: "epochs", label: "轮次", def: "3" }],
    info: ["A 微调 QLoRA / B 从零预训练", "显存不足自动降级"],
  },
  {
    key: "export", label: "导出", sub: "EXPORT",
    params: [{ k: "quants", label: "量化档位", def: "q4_k_m,q8" }],
    info: ["LoRA 合并或完整权重 → GGUF"],
  },
  {
    key: "chat", label: "对话", sub: "CHAT", params: [],
    info: ["产物导入 Ollama", "对话屏直接选用"],
  },
];

const stageOf = (k) => STAGES.find((t) => t.key === k);

const SEED_CHAIN = ["data", "parse", "clean", "split", "train", "export"];

// 节点 → 端到端 run 阶段（engine/flow.py 的 timeline key）映射
const PHASE_OF = {
  data: "build",
  parse: "build",
  clean: "build",
  split: "build",
  text: "build",
  check: "build",
  train: "train",
  export: "export",
  chat: "import",
};

// run 行规范化：timeline_json（create_run 返回字符串）→ timeline 数组
function normalizeRun(r) {
  if (!r) return null;
  let tl = r.timeline;
  if (typeof tl === "string") {
    try {
      tl = JSON.parse(tl);
    } catch {
      tl = [];
    }
  }
  return { ...r, timeline: Array.isArray(tl) ? tl : [] };
}

let seq = 0;

function mkNode(key, x, y, datasetId) {
  const t = stageOf(key);
  seq += 1;
  const meta = Object.fromEntries(t.params.map((p) => [p.k, p.def]));
  if (key === "data" && datasetId) meta.dataset = datasetId;
  return { id: `n${seq}`, key, x, y, meta };
}

export default function FlowCanvas({ dataset, onDatasetChange }) {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [view, setView] = useState({ x: 60, y: 60, z: 1 });
  const [selNode, setSelNode] = useState(null);
  const [selEdge, setSelEdge] = useState(null); // "from>to"
  const [linking, setLinking] = useState(null); // {from, mx, my, hover}
  const [, bump] = useState(0); // 拖动后强制重算连线
  const [run, setRun] = useState(null); // 当前/最近端到端 run（W5）
  const [history, setHistory] = useState([]);
  const [launching, setLaunching] = useState(false);
  const [notice, setNotice] = useState(""); // {kind, text}
  const [datasets, setDatasets] = useState([]); // 数据源节点可绑定的数据集清单

  useEffect(() => {
    api
      .get("/api/datasets")
      .then((rows) => setDatasets(Array.isArray(rows) ? rows : []))
      .catch(() => {});
  }, []);

  const dsName = useCallback(
    (id) => datasets.find((d) => d.id === id)?.name || String(id || "").slice(0, 8),
    [datasets],
  );

  /* ---------- run 加载 / 发起（W5） ---------- */
  const loadRuns = useCallback(async () => {
    if (!dataset) return;
    try {
      const all = await api.get("/api/runs");
      const mine = (Array.isArray(all) ? all : [])
        .filter((r) => r.dataset_id === dataset.id)
        .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
      setHistory(mine.slice(0, 5));
      setRun(normalizeRun(mine[0] || null));
    } catch {
      /* 后端未就绪保持空态 */
    }
  }, [dataset]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  // 从画布节点参数生成 overrides（缺失的节点用后端默认）
  const readOverrides = useCallback(() => {
    const byKey = Object.fromEntries(nodes.map((n) => [n.key, n.meta]));
    const num = (v) => {
      const n = parseInt(v, 10);
      return Number.isFinite(n) ? n : undefined;
    };
    return {
      ...(byKey.split?.chunk_size ? { chunk_size: num(byKey.split.chunk_size) } : {}),
      ...(byKey.split?.overlap ? { overlap: num(byKey.split.overlap) } : {}),
      ...(byKey.text?.template ? { template: byKey.text.template } : {}),
      ...(byKey.train?.epochs ? { epochs: num(byKey.train.epochs) } : {}),
    };
  }, [nodes]);

  const onRun = useCallback(async () => {
    if (!dataset || launching) return;
    setLaunching(true);
    setNotice(null);
    try {
      const created = await api.post("/api/runs", {
        dataset_id: dataset.id,
        auto_import: true,
        ...readOverrides(),
      });
      setRun(normalizeRun(created));
      setNotice({ kind: "ok", text: `已发起端到端 run ${String(created.id).slice(0, 8)}…` });
      loadRuns();
    } catch (e) {
      setNotice({ kind: "err", text: `发起失败：${String(e.message || e)}` });
    } finally {
      setLaunching(false);
    }
  }, [dataset, launching, readOverrides, loadRuns]);

  // 事件流：run.stage 点亮阶段、run.status 推进状态（终态回读权威数据）
  useEvents((msg) => {
    const d = msg.data || {};
    if (!run || d.id !== run.id) return;
    if (msg.type === "run.stage") {
      setRun((r) => {
        if (!r || d.id !== r.id) return r;
        const tl = [...r.timeline];
        const i = tl.findIndex((p) => p.key === d.key);
        const ph = {
          key: d.key,
          label: d.label,
          status: d.status,
          started_at: d.started_at,
          finished_at: d.finished_at ?? null,
          duration_s: d.duration_s ?? null,
          detail: d.detail ?? null,
        };
        if (i >= 0) tl[i] = { ...tl[i], ...ph };
        else tl.push(ph);
        return { ...r, timeline: tl };
      });
    } else if (msg.type === "run.status") {
      setRun((r) => (r && d.id === r.id ? { ...r, status: d.status, progress: d.progress, error: d.error } : r));
      if (d.status === "done" || d.status === "failed") loadRuns();
    }
  });

  // 节点四态：由 run.timeline 的阶段状态聚合而来
  const nodeState = useCallback(
    (n) => {
      if (!run) return "idle";
      const ph = run.timeline.find((p) => p.key === PHASE_OF[n.key]);
      if (!ph) return run.status === "running" ? "idle" : "idle";
      return ph.status || "idle"; // ok | failed | skipped | running
    },
    [run],
  );

  const canvasRef = useRef(null);
  const dragRef = useRef(null); // {mode:'pan'|'node'|'link', ...}

  const z = view.z;
  const toVp = useCallback(
    (cx, cy) => {
      const r = canvasRef.current.getBoundingClientRect();
      return { x: (cx - r.left - view.x) / z, y: (cy - r.top - view.y) / z };
    },
    [view, z],
  );

  const portPos = useCallback(
    (nodeId, side) => {
      const n = nodes.find((x) => x.id === nodeId);
      if (!n) return null;
      return side === "out"
        ? { x: n.x + NODE_W, y: n.y + NODE_H / 2 }
        : { x: n.x, y: n.y + NODE_H / 2 };
    },
    [nodes],
  );

  const pathOf = useCallback(
    (a, b) => {
      const dx = Math.max(24, Math.abs(b.x - a.x) * 0.5);
      return `M${a.x} ${a.y} C${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
    },
    [],
  );

  /* ---------- 选择 / 增删 / 连线 ---------- */
  const addEdge = useCallback((from, to) => {
    if (!from || !to || from === to) return;
    setEdges((es) =>
      es.some((e) => e.from === from && e.to === to) ? es : [...es, { from, to }],
    );
  }, []);

  const addNode = useCallback(
    (key, vx, vy) => {
      const n = mkNode(key, Math.round(vx), Math.round(vy), dataset?.id);
      setNodes((ns) => [...ns, n]);
      setSelNode(n.id);
      setSelEdge(null);
      return n.id;
    },
    [dataset],
  );

  const deleteSel = useCallback(() => {
    if (selNode) {
      setEdges((es) => es.filter((e) => e.from !== selNode && e.to !== selNode));
      setNodes((ns) => ns.filter((n) => n.id !== selNode));
      setSelNode(null);
    } else if (selEdge) {
      const [f, t] = selEdge.split(">");
      setEdges((es) => es.filter((e) => !(e.from === f && e.to === t)));
      setSelEdge(null);
    }
  }, [selNode, selEdge]);

  /* ---------- 视图工具 ---------- */
  const zoomAt = useCallback((cx, cy, factor) => {
    setView((v) => {
      const nz = Math.min(2.2, Math.max(0.25, v.z * factor));
      const r = canvasRef.current.getBoundingClientRect();
      const mx = cx - r.left - v.x;
      const my = cy - r.top - v.y;
      return { z: nz, x: cx - r.left - mx * (nz / v.z), y: cy - r.top - my * (nz / v.z) };
    });
  }, []);

  const zoomCenter = useCallback(
    (factor) => {
      const r = canvasRef.current.getBoundingClientRect();
      zoomAt(r.left + r.width / 2, r.top + r.height / 2, factor);
    },
    [zoomAt],
  );

  const fitView = useCallback(() => {
    const ns = nodes;
    const r = canvasRef.current.getBoundingClientRect();
    if (!ns.length || !r.width) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    ns.forEach((n) => {
      x0 = Math.min(x0, n.x);
      y0 = Math.min(y0, n.y);
      x1 = Math.max(x1, n.x + NODE_W);
      y1 = Math.max(y1, n.y + NODE_H);
    });
    const w = x1 - x0;
    const h = y1 - y0;
    const nz = Math.min(1.4, Math.max(0.25, Math.min(r.width / (w + 90), r.height / (h + 110))));
    setView({
      z: nz,
      x: r.width / 2 - (x0 + w / 2) * nz,
      y: r.height / 2 - (y0 + h / 2) * nz,
    });
  }, [nodes]);

  const autoLayout = useCallback(() => {
    if (!nodes.length) return;
    const indeg = Object.fromEntries(nodes.map((n) => [n.id, 0]));
    edges.forEach((e) => {
      if (indeg[e.to] !== undefined) indeg[e.to] += 1;
    });
    const queue = nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      edges
        .filter((e) => e.from === id)
        .forEach((e) => {
          if (indeg[e.to] !== undefined) {
            indeg[e.to] -= 1;
            if (indeg[e.to] === 0) queue.push(e.to);
          }
        });
    }
    nodes.forEach((n) => {
      if (!order.includes(n.id)) order.push(n.id); // 防环漏排
    });
    const pos = Object.fromEntries(order.map((id, i) => [id, { x: 30 + i * (NODE_W + 30), y: 180 }]));
    setNodes((ns) => ns.map((n) => ({ ...n, ...pos[n.id] })));
    setSelNode(null);
    setSelEdge(null);
    requestAnimationFrame(() => fitRef.current());
  }, [nodes, edges]);

  // fitView 闭包引用最新 nodes；自动布局后延迟调用用 ref 转发
  const fitRef = useRef(fitView);
  fitRef.current = fitView;

  const seed = useCallback(() => {
    const ns = [];
    const es = [];
    let prev = null;
    SEED_CHAIN.forEach((key, i) => {
      const n = mkNode(key, 40 + i * (NODE_W + 50), 150 + (i % 2 ? 46 : 0), dataset?.id);
      ns.push(n);
      if (prev) es.push({ from: prev, to: n.id });
      prev = n.id;
    });
    setNodes(ns);
    setEdges(es);
    setSelNode(null);
    setSelEdge(null);
    requestAnimationFrame(() => fitRef.current());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset?.id]);

  useEffect(() => {
    seed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---------- 指针交互 ---------- */
  const nearestInPort = useCallback(
    (pt) => {
      let best = null;
      let bd = 26;
      nodes.forEach((n) => {
        const c = { x: n.x, y: n.y + NODE_H / 2 };
        const d = Math.hypot(c.x - pt.x, c.y - pt.y);
        if (d < bd) {
          bd = d;
          best = { id: n.id, c };
        }
      });
      return best;
    },
    [nodes],
  );

  const onCanvasPointerDown = (e) => {
    const t = e.target;
    if (t === canvasRef.current || t.classList?.contains("fvp") || t.classList?.contains("flinks")) {
      dragRef.current = { mode: "pan", sx: e.clientX, sy: e.clientY, px: view.x, py: view.y };
      setSelNode(null);
      setSelEdge(null);
    }
  };

  const onNodePointerDown = (e, n) => {
    e.stopPropagation();
    const port = e.target.closest?.(".fport");
    if (port?.classList.contains("out")) {
      e.preventDefault();
      const p = toVp(e.clientX, e.clientY);
      setLinking({ from: n.id, mx: p.x, my: p.y, hover: null });
      dragRef.current = { mode: "link" };
      return;
    }
    setSelNode(n.id);
    setSelEdge(null);
    e.preventDefault();
    dragRef.current = { mode: "node", id: n.id, ox: n.x, oy: n.y, sx: e.clientX, sy: e.clientY };
  };

  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      if (!d) return;
      if (d.mode === "pan") {
        setView((v) => ({ ...v, x: d.px + (e.clientX - d.sx), y: d.py + (e.clientY - d.sy) }));
      } else if (d.mode === "node") {
        const nx = d.ox + (e.clientX - d.sx) / z;
        const ny = d.oy + (e.clientY - d.sy) / z;
        setNodes((ns) => ns.map((n) => (n.id === d.id ? { ...n, x: nx, y: ny } : n)));
      } else if (d.mode === "link") {
        const p = toVp(e.clientX, e.clientY);
        const hover = nearestInPort(p);
        setLinking((l) => (l ? { ...l, mx: p.x, my: p.y, hover } : l));
      }
    };
    const onUp = () => {
      const d = dragRef.current;
      if (d?.mode === "link") {
        setLinking((l) => {
          if (l?.hover && l.hover.id !== l.from) addEdge(l.from, l.hover.id);
          return null;
        });
      }
      dragRef.current = null;
      bump((x) => x + 1);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [z, toVp, nearestInPort, addEdge]);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && !["INPUT", "TEXTAREA"].includes(e.target.tagName)) {
        deleteSel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleteSel]);

  /* ---------- 渲染 ---------- */
  const sel = nodes.find((n) => n.id === selNode) || null;
  const selDef = sel ? stageOf(sel.key) : null;

  const setMeta = (k, v) => {
    setNodes((ns) => ns.map((n) => (n.id === selNode ? { ...n, meta: { ...n.meta, [k]: v } } : n)));
  };

  // 数据源节点切换绑定数据集：节点 meta + 画布项目（crumb/标题/运行目标）同步
  const onDatasetSelect = (id) => {
    setMeta("dataset", id);
    const ds = datasets.find((d) => d.id === id);
    if (ds) onDatasetChange?.(ds);
  };

  const linkPath = linking
    ? pathOf(
        portPos(linking.from, "out") || { x: 0, y: 0 },
        linking.hover ? linking.hover.c : { x: linking.mx, y: linking.my },
      )
    : null;

  return (
    <div className="flow-root">
      <div className="flow-tools">
        <button
          type="button"
          className="ws-btn primary"
          onClick={onRun}
          disabled={launching || run?.status === "running"}
          title="以画布参数发起端到端 run（构建 → 训练 → 导出 → 导入）"
        >
          {run?.status === "running" ? "运行中…" : launching ? "发起中…" : "运行流水线"}
        </button>
        {run && (
          <span className="pill" title={run.error || undefined}>
            <span className={`dot ${run.status === "done" ? "ok" : run.status === "failed" ? "off" : ""}`} />
            run {String(run.id).slice(0, 6)} · {Math.round((Number(run.progress) || 0) * 100)}% · {run.status}
          </span>
        )}
        {notice && <span className={`notice n-${notice.kind} flow-notice`}>{notice.text}</span>}
        <span className="top-spacer" />
        <span className="ctl" title="视图缩放">
          <button type="button" className="zbtn" onClick={() => zoomCenter(0.82)} title="缩小">−</button>
          <span className="zpct">{Math.round(z * 100)}%</span>
          <button type="button" className="zbtn" onClick={() => zoomCenter(1.22)} title="放大">＋</button>
          <span className="sep" />
          <button type="button" className="zbtn wide" onClick={fitView} title="缩放并平移，让所有节点完整进入画布">适应视图</button>
          <button type="button" className="zbtn wide" onClick={autoLayout} title="按连线依赖从左到右自动排布节点">自动布局</button>
          <button type="button" className="zbtn wide" onClick={deleteSel} title="删除选中的节点或连线（Delete）">删除</button>
        </span>
        <span className="top-spacer" />
        <span className="flow-stat">
          节点 {nodes.length} · 连线 {edges.length}
        </span>
      </div>

      <div className="flow-body">
        <div
          className="flow-canvas2"
          ref={canvasRef}
          onPointerDown={onCanvasPointerDown}
          onWheel={(e) => {
            e.preventDefault();
            zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 0.89);
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const key = e.dataTransfer.getData("text/plain");
            if (stageOf(key)) {
              const p = toVp(e.clientX, e.clientY);
              addNode(key, p.x - NODE_W / 2, p.y - NODE_H / 2);
            }
          }}
        >
          <div
            className="fvp"
            style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${z})` }}
          >
            <svg className="flinksvg">
              {edges.map((e) => {
                const a = portPos(e.from, "out");
                const b = portPos(e.to, "in");
                if (!a || !b) return null;
                const d = pathOf(a, b);
                const s = selEdge === `${e.from}>${e.to}`;
                return (
                  <g key={`${e.from}>${e.to}`}>
                    <path
                      className="fhit"
                      d={d}
                      onPointerDown={(ev) => {
                        ev.stopPropagation();
                        setSelEdge(`${e.from}>${e.to}`);
                        setSelNode(null);
                      }}
                    />
                    <path className={`fline${s ? " sel" : ""}`} d={d} />
                  </g>
                );
              })}
              {linking && linkPath && <path className="ftmp" d={linkPath} />}
            </svg>

            {nodes.map((n) => {
              const t = stageOf(n.key);
              const st = nodeState(n);
              const ph = run?.timeline.find((p) => p.key === PHASE_OF[n.key]);
              const stLabel = { running: "运行中", ok: "完成", failed: "失败", skipped: "跳过" }[st];
              return (
                <div
                  key={n.id}
                  className={`fnode${selNode === n.id ? " sel" : ""}${st !== "idle" ? ` st-${st}` : ""}`}
                  style={{ left: n.x, top: n.y }}
                  onPointerDown={(e) => onNodePointerDown(e, n)}
                  title={ph?.detail || undefined}
                >
                  {st !== "idle" && (
                    <span className={`fstate ${st}`}>
                      {stLabel}
                      {ph?.duration_s != null ? ` ${ph.duration_s}s` : ""}
                    </span>
                  )}
                  <div className="fhead">
                    <span className="fic">
                      <svg viewBox="0 0 24 24">{ICONS[t.key]}</svg>
                    </span>
                    <span className="fttl">
                      {t.label}
                      <small>{t.sub}</small>
                    </span>
                  </div>
                  <div className="fbody">
                    {t.params.map((p) => (
                      <div key={p.k} className="fr">
                        <span>{p.label}</span>
                        <b>
                          {n.key === "data" && p.k === "dataset"
                            ? dsName(n.meta.dataset)
                            : n.meta[p.k]}
                        </b>
                      </div>
                    ))}
                    {t.params.length === 0 && t.info.length > 0 && (
                      <div className="fr">
                        <span className="finfo-line">{t.info.slice(0, 2).join(" · ")}</span>
                      </div>
                    )}
                  </div>
                  <span className="fport in" />
                  <span className="fport out" />
                </div>
              );
            })}
          </div>
        </div>

        <aside className="flow-pane">
          <div className="box">
            <h5>节点库</h5>
            <ul className="flow-lib">
              {STAGES.map((t) => (
                <li
                  key={t.key}
                  draggable
                  onDragStart={(e) => e.dataTransfer.setData("text/plain", t.key)}
                  onClick={() => addNode(t.key, 420 + Math.random() * 120 - 60, 220 + Math.random() * 120 - 60)}
                >
                  <span className="fic">
                    <svg viewBox="0 0 24 24">{ICONS[t.key]}</svg>
                  </span>
                  <span className="fl-ttl">{t.label}</span>
                  <span className="fl-add">＋</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="box">
            <h5>选中节点参数</h5>
            {!sel ? (
              <div className="f-empty">点击画布上的节点查看与编辑参数</div>
            ) : (
              <>
                <div className="fprop">
                  类型 <span className="v">{selDef.sub}</span>
                </div>
                {selDef.params.map((p) =>
                  sel.key === "data" && p.k === "dataset" ? (
                    <label key={p.k} className="fprop fedit">
                      {p.label}
                      <select
                        className="input finput"
                        value={sel.meta.dataset || ""}
                        onChange={(e) => onDatasetSelect(e.target.value)}
                      >
                        {(datasets.length ? datasets : dataset ? [dataset] : []).map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : (
                    <label key={p.k} className="fprop fedit">
                      {p.label}
                      <input
                        className="input finput"
                        value={sel.meta[p.k] ?? ""}
                        onChange={(e) => setMeta(p.k, e.target.value)}
                      />
                    </label>
                  ),
                )}
                {selDef.info.length > 0 && (
                  <>
                    <div className="fchips">
                      {selDef.info.map((line) => (
                        <span key={line} className="fchip">
                          {line}
                        </span>
                      ))}
                    </div>
                    <div className="f-empty" style={{ marginTop: 6 }}>
                      由后端内置执行，暂不可调
                    </div>
                  </>
                )}
                <div className="fprop">
                  坐标 <span className="v">{Math.round(sel.x)}, {Math.round(sel.y)}</span>
                </div>
              </>
            )}
          </div>
          <div className="box">
            <h5>运行历史</h5>
            {history.length === 0 ? (
              <div className="f-empty">暂无 run · 点「运行流水线」发起端到端</div>
            ) : (
              <ul className="rh-list">
                {history.map((r) => (
                  <li key={r.id}>
                    <span className={`rh-st st-${r.status}`}>
                      {{ done: "完成", failed: "失败", running: "运行中", queued: "排队中", pending_gpu: "等GPU" }[r.status] || r.status}
                    </span>
                    <span className="rh-id">{String(r.id).slice(0, 6)}</span>
                    <span className="rh-p">{Math.round((Number(r.progress) || 0) * 100)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="box">
            <h5>说明</h5>
            <div className="f-empty">
              拖空白平移 · 滚轮缩放 · 拖标题移动节点 · 右端口拖到左端口连线 ·
              Delete 删除；运行时节点按阶段点亮（构建→训练→导出→导入）
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
