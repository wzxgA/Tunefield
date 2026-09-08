// 训练屏（F0 占位）：任务列表 / 阶段进度 / LossChart / 配置覆盖由 T7、T8 交付
export default function TrainScreen() {
  return (
    <section className="screen">
      <h2 className="screen-title">训练任务</h2>
      <p className="screen-hint">创建训练任务，实时观察阶段进度与 loss 曲线。</p>
      <div className="placeholder-card">
        占位 · T7 训练引擎（loss 曲线 / 阶段进度 / 日志流，经 WebSocket 实时推送）与 T8 配置推荐器（推荐值预览 / 覆盖）将在此呈现。
      </div>
    </section>
  );
}
