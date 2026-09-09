import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { useEvents } from "./hooks/useEvents.js";
import ChatScreen from "./screens/ChatScreen.jsx";
import TrainScreen from "./screens/TrainScreen.jsx";
import UploadScreen from "./screens/UploadScreen.jsx";

// W1 工作站壳（依 yuanxing/index-3.html 原型）：top / rail / main / side / bar 五区。
// 导航五项（无总览）；projects/models 两屏分别由 W3/W2 交付，当前占位。
// 整屏编排（flow-full）由 W4 画布挂载时启用，CSS 作用域已在 global.css 就位。

const NAV = [
  {
    id: "projects",
    label: "编排",
    icon: (
      <>
        <circle cx="5.5" cy="6" r="2.4" />
        <circle cx="18.5" cy="6" r="2.4" />
        <circle cx="12" cy="18" r="2.6" />
        <path d="M7.5 7.2 10.4 15.6M16.5 7.2l-3 8.4M7.9 6h8.2" />
      </>
    ),
  },
  {
    id: "data",
    label: "数据",
    icon: (
      <>
        <ellipse cx="12" cy="5.6" rx="7.4" ry="2.9" />
        <path d="M4.6 5.6v12.4c0 1.6 3.3 2.9 7.4 2.9s7.4-1.3 7.4-2.9V5.6" />
        <path d="M4.6 11.8c0 1.6 3.3 2.9 7.4 2.9s7.4-1.3 7.4-2.9" />
      </>
    ),
  },
  {
    id: "train",
    label: "训练",
    icon: <path d="M2.5 12h3.2l2.4-6.2 3.6 12.4 2.6-6.2h7.2" />,
  },
  {
    id: "models",
    label: "模型",
    icon: (
      <>
        <path d="M12 2.6 20 7.4v9.2l-8 4.8-8-4.8V7.4z" />
        <path d="M4 7.4l8 4.8 8-4.8" />
        <path d="M12 12.2V21" />
      </>
    ),
  },
  {
    id: "chat",
    label: "对话",
    icon: (
      <>
        <path d="M20.8 11.9a7.9 7.9 0 0 1-11.6 6.9L4.2 20.4l1.6-4.8a7.9 7.9 0 1 1 15-3.7z" />
        <path d="M8.4 10.2h7.2M8.4 13.4h4.6" />
      </>
    ),
  },
];

const CRUMBS = {
  projects: "Workbench / 编排项目",
  data: "Workbench / 数据",
  train: "Workbench / 训练",
  models: "Workbench / 模型",
  chat: "Workbench / 对话",
};

// 视为"运行中"的任务状态（与队列恢复语义一致）
const ACTIVE_STATES = new Set([
  "queued",
  "pending_gpu",
  "running",
  "training",
  "exporting",
  "parsing",
]);

const EVENT_LABELS = {
  "job.created": "新任务",
  "job.status": "任务状态",
  "job.log": "日志",
  "run.created": "新端到端",
  "run.stage": "阶段推进",
  "run.status": "端到端状态",
};

const clockNow = () => new Date().toTimeString().slice(0, 8);

export default function App() {
  const [screen, setScreen] = useState("data"); // 默认 = 数据（无总览）
  const [sys, setSys] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [modelCount, setModelCount] = useState(0);
  const [activity, setActivity] = useState([]);
  const [clock, setClock] = useState(clockNow());

  const loadSys = useCallback(() => {
    api.get("/api/system/status").then(setSys).catch(() => {});
  }, []);

  const loadCounts = useCallback(() => {
    api
      .get("/api/jobs")
      .then((rows) => setJobs(Array.isArray(rows) ? rows : []))
      .catch(() => {});
    api
      .get("/api/models")
      .then((rows) => setModelCount(Array.isArray(rows) ? rows.length : 0))
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadSys();
    loadCounts();
    const timer = setInterval(() => setClock(clockNow()), 1000);
    return () => clearInterval(timer);
  }, [loadSys, loadCounts]);

  // 事件流：驱动"最近活动"，状态类事件顺带刷新计数
  const connected = useEvents((msg) => {
    const d = msg.data || {};
    const label = EVENT_LABELS[msg.type] || msg.type;
    const text = [label, d.domain, d.status, d.line]
      .filter(Boolean)
      .join(" · ")
      .slice(0, 72);
    setActivity((prev) =>
      [{ t: clockNow().slice(0, 5), text }, ...prev].slice(0, 9),
    );
    if (msg.type.endsWith(".status") || msg.type.endsWith(".created")) {
      loadCounts();
    }
  });

  const active = jobs.find((j) => ACTIVE_STATES.has(j.status));
  const runningCount = jobs.filter((j) => ACTIVE_STATES.has(j.status)).length;
  const activePct = Math.round((Number(active?.progress) || 0) * 100);
  const gpu = sys?.gpu?.vram_gb;
  const ollamaOk = Boolean(sys?.ollama?.running);

  return (
    <div className="ws">
      <header className="ws-top">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="5.2" strokeDasharray="3.4 4" />
            <circle cx="12" cy="12" r="1.6" fill="#fff" stroke="none" />
          </svg>
          <span className="brand-name">TUNEFIELD</span>
        </div>
        <span className="crumb">{CRUMBS[screen]}</span>
        <span className="top-spacer" />
        <span className="pill">
          <span className={`dot${gpu ? " ok" : ""}`} />
          GPU {gpu ? `${gpu} GB` : "检测中"}
        </span>
        <span className="pill">
          <span className={`dot ${ollamaOk ? "ok" : "off"}`} />
          Ollama {ollamaOk ? "已连接" : "离线"}
        </span>
        <button
          type="button"
          className="ws-btn primary"
          onClick={() => setScreen("train")}
        >
          一键端到端
        </button>
      </header>

      <nav className="rail">
        {NAV.map((n) => (
          <button
            key={n.id}
            type="button"
            className={`nav${screen === n.id ? " on" : ""}`}
            onClick={() => setScreen(n.id)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              {n.icon}
            </svg>
            <span>{n.label}</span>
          </button>
        ))}
        <span className="rail-foot">TUNEFIELD</span>
      </nav>

      <main className="ws-main" key={screen}>
        <div className="fade-in">
          {screen === "data" && <UploadScreen />}
          {screen === "train" && <TrainScreen />}
          {screen === "chat" && <ChatScreen />}
          {screen === "projects" && (
            <div className="placeholder-card">
              编排项目选择页 · W3 交付：真实数据集卡片 → 进入整屏编排画布
              <br />
              （交互原型预览：yuanxing/index-3.html）
            </div>
          )}
          {screen === "models" && (
            <div className="placeholder-card">
              模型库 · W2 交付：GGUF 清单 / 一键导入 Ollama / 指纹反查
            </div>
          )}
        </div>
      </main>

      <aside className="side">
        <div className="box">
          <h5>当前任务</h5>
          {active ? (
            <div className="ring-row">
              <div className="ring" style={{ "--p": activePct }}>
                <b>{activePct}%</b>
              </div>
              <div>
                <div className="side-strong">
                  {active.domain} · {active.kind}
                </div>
                <div className="side-sub">{active.status}</div>
              </div>
            </div>
          ) : (
            <div className="side-sub">空闲 · 无运行中的任务</div>
          )}
        </div>

        <div className="box">
          <h5>资源</h5>
          <div className="kpi">
            <span className="lbl">GPU</span>
            <div className="bar">
              <i style={{ width: gpu ? "60%" : "0%" }} />
            </div>
            <span className="val">{gpu ? `${gpu}G` : "—"}</span>
          </div>
          <div className="kpi">
            <span className="lbl">队列</span>
            <div className="bar">
              <i style={{ width: `${Math.min(100, runningCount * 25)}%` }} />
            </div>
            <span className="val">{runningCount} 任务</span>
          </div>
        </div>

        <div className="box">
          <h5>阶段</h5>
          <div className="chips">
            {["解析", "清洗", "切片", "指令化", "质检"].map((s) => (
              <span key={s} className="chip">
                {s}
              </span>
            ))}
          </div>
        </div>

        <div className="box">
          <h5>最近活动</h5>
          {activity.length === 0 ? (
            <div className="empty">
              暂无事件 · {connected ? "已连接事件流" : "事件流连接中…"}
            </div>
          ) : (
            <ul className="activity">
              {activity.map((a, i) => (
                <li key={`${a.t}-${i}`}>
                  <span className="t">{a.t}</span>
                  <span>{a.text}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <footer className="ws-bar">
        <span>Jobs {jobs.length}</span>
        <span className="sep" />
        <span>Running {runningCount}</span>
        <span className="sep" />
        <span>Models {modelCount}</span>
        <span className="right">
          <span>GPU {gpu ? `${gpu} GB` : "—"}</span>
          <span className="sep" />
          <span>Ollama {ollamaOk ? "已连接" : "离线"}</span>
          <span className="sep" />
          <span>{clock}</span>
        </span>
      </footer>
    </div>
  );
}
