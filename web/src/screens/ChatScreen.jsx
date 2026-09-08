// 对话屏（F0 占位）：ModelSelect / 消息流 / 采样参数面板由 T10 交付
export default function ChatScreen() {
  return (
    <section className="screen">
      <h2 className="screen-title">模型对话</h2>
      <p className="screen-hint">选择已量化的领域模型，直接对话试用。</p>
      <div className="placeholder-card">
        占位 · T10 对话验证（模型下拉切换 / 消息流 / 采样参数面板）将在此呈现。
      </div>
    </section>
  );
}
