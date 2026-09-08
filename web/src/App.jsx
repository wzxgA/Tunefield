import { useState } from "react";
import UploadScreen from "./screens/UploadScreen.jsx";
import TrainScreen from "./screens/TrainScreen.jsx";
import ChatScreen from "./screens/ChatScreen.jsx";

// 三屏顶层状态切换（无路由库）：screen ∈ upload | train | chat
const SCREENS = [
  { id: "upload", label: "数据" },
  { id: "train", label: "训练" },
  { id: "chat", label: "对话" },
];

export default function App() {
  const [screen, setScreen] = useState("upload");

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-name">Tunefield</span>
          <span className="brand-sub">领域专属小模型训练平台</span>
        </div>
        <nav className="tabs">
          {SCREENS.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`tab${screen === s.id ? " tab-active" : ""}`}
              onClick={() => setScreen(s.id)}
            >
              {s.label}
            </button>
          ))}
        </nav>
      </header>
      {/* key 触发重挂载，配合 CSS 动画实现三屏切换的淡入/位移 */}
      <main className="app-body" key={screen}>
        {screen === "upload" && <UploadScreen />}
        {screen === "train" && <TrainScreen />}
        {screen === "chat" && <ChatScreen />}
      </main>
    </div>
  );
}
