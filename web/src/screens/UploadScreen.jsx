// 上传屏（F0 占位）：FileDrop / 解析进度 / 质检报告卡片由 T1–T6 对应任务交付
export default function UploadScreen() {
  return (
    <section className="screen">
      <h2 className="screen-title">数据上传</h2>
      <p className="screen-hint">拖拽文件或挂载本地目录，平台将自动解析并产出质检报告。</p>
      <div className="placeholder-card">
        占位 · T1 通用接入（文件清单 / 哈希去重命中）与后续解析、清洗、切片、指令化、质检的逐步骤产物将在此呈现。
      </div>
    </section>
  );
}
