import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import DatasetFiles from "../components/DatasetFiles.jsx";
import DatasetReport from "../components/DatasetReport.jsx";

// 数据上传屏（T1 接入 + T2 解析预览 + T6 构建/质检）：多文件上传 + 哈希去重 +
// dataset 清单；点开可看质检报告与逐文件解析/清洗/切片/样本
export default function UploadScreen() {
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]); // File[]
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(""); // "上传成功 / 去重命中 / 错误"
  const [noticeKind, setNoticeKind] = useState(""); // ok | dup | err
  const [inspect, setInspect] = useState({}); // {datasetId: bool} 展开解析预览
  const [building, setBuilding] = useState({}); // {datasetId: bool}
  const [deletingDs, setDeletingDs] = useState(""); // 删除中的数据集 id
  const [builtTick, setBuiltTick] = useState(0); // 构建完成信号 → 刷新报告

  const loadDatasets = useCallback(async () => {
    try {
      const list = await api.get("/api/datasets");
      setDatasets(list);
    } catch {
      /* 后端未就绪保持空态 */
    }
  }, []);
  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const onPick = (e) => setFiles(Array.from(e.target.files || []));

  const toggleInspect = (datasetId) =>
    setInspect((s) => ({ ...s, [datasetId]: !s[datasetId] }));

  // 删除数据集（级联：任务/导出产物/原始文件，不可恢复）
  const onDelete = async (datasetId) => {
    const ds = datasets.find((x) => x.id === datasetId);
    if (
      !window.confirm(
        `删除数据集「${ds?.name ?? datasetId}」？\n将同时删除其全部训练任务、导出产物与原始文件，且不可恢复。`
      )
    ) {
      return;
    }
    setDeletingDs(datasetId);
    setNotice("");
    try {
      await api.del(`/api/datasets/${datasetId}`);
      await loadDatasets();
      setInspect((s) => {
        const next = { ...s };
        delete next[datasetId];
        return next;
      });
      setNotice("已删除数据集");
      setNoticeKind("ok");
    } catch (e) {
      setNotice(`删除失败：${String(e.message || e)}`);
      setNoticeKind("err");
    } finally {
      setDeletingDs("");
    }
  };

  // T6：全管线构建（解析→清洗→切片→指令化→质检 + JSONL 落盘）
  const onBuild = async (datasetId) => {
    setBuilding((s) => ({ ...s, [datasetId]: true }));
    setNotice("");
    try {
      await api.post(`/api/datasets/${datasetId}/build`, {});
      await loadDatasets();
      setBuiltTick((t) => t + 1);
      setNotice("构建完成：已产出 JSONL 与质检报告");
      setNoticeKind("ok");
    } catch (e) {
      setNotice(`构建失败：${String(e.message || e)}`);
      setNoticeKind("err");
    } finally {
      setBuilding((s) => ({ ...s, [datasetId]: false }));
    }
  };

  const onUpload = async () => {
    if (!name.trim()) {
      setNotice("请填写领域名称");
      setNoticeKind("err");
      return;
    }
    if (files.length === 0) {
      setNotice("请选择文件");
      setNoticeKind("err");
      return;
    }
    setUploading(true);
    setNotice("");
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      files.forEach((f) => fd.append("files", f));
      const res = await api.upload("/api/datasets", fd);
      setNotice(
        res.deduped
          ? `去重命中，复用既有数据集「${res.dataset.name}」`
          : `接入成功「${res.dataset.name}」`
      );
      setNoticeKind(res.deduped ? "dup" : "ok");
      await loadDatasets();
    } catch (e) {
      setNotice(String(e.message || e));
      setNoticeKind("err");
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="screen upload-screen">
      <div className="train-head">
        <div>
          <h2 className="screen-title">数据上传</h2>
          <p className="screen-hint">上传文件，平台将去重落盘并登记数据集（重复内容秒级复用）。</p>
        </div>
      </div>

      <div className="upload-form">
        <input
          className="input"
          placeholder="领域名称"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <label className="btn-ghost file-pick">
          选择文件
          <input type="file" multiple onChange={onPick} hidden />
        </label>
        <button className="btn-primary" onClick={onUpload} disabled={uploading}>
          {uploading ? "上传中…" : "上传接入"}
        </button>
      </div>

      {files.length > 0 && (
        <div className="file-picked">
          {files.map((f) => (
            <span key={f.name + f.lastModified} className="file-chip">{f.name}</span>
          ))}
        </div>
      )}

      {notice && (
        <p className={`notice n-${noticeKind}`}>{notice}</p>
      )}

      <div className="ds-block">
        <div className="ds-title">
          已接入数据集
          <span className="ds-count">{datasets.length} 个</span>
        </div>
        {datasets.length === 0 ? (
          <div className="job-empty">暂无数据集 · 先上传一批文件</div>
        ) : (
          <div className="ds-list">
            {[...datasets].reverse().map((d) => (
              <div key={d.id} className="ds-item">
                <div className="ds-line1">
                  <span className="ds-name">{d.name}</span>
                  <span className="ds-actions">
                    <span className={`badge ${d.status === "built" ? "st-built" : "st-ingested"}`}>
                      {d.status === "built" ? "已构建" : "已接入"}
                    </span>
                    <button
                      type="button"
                      className="btn-inspect"
                      disabled={building[d.id]}
                      onClick={() => onBuild(d.id)}
                    >
                      {building[d.id] ? "构建中…" : d.status === "built" ? "重建" : "构建"}
                    </button>
                    <button
                      type="button"
                      className="btn-inspect"
                      onClick={() => toggleInspect(d.id)}
                    >
                      {inspect[d.id] ? "收起" : "预览/报告"}
                    </button>
                    <button
                      type="button"
                      className="btn-danger"
                      disabled={deletingDs === d.id || building[d.id]}
                      onClick={() => onDelete(d.id)}
                      title="删除数据集及其全部训练产物"
                    >
                      {deletingDs === d.id ? "删除中…" : "删除"}
                    </button>
                  </span>
                </div>
                <div className="ds-line2">
                  <span className="ds-id">{d.id.slice(0, 10)}</span>
                  <span className="ds-fp">指纹 {d.content_hash.slice(0, 10)}…</span>
                </div>
                {inspect[d.id] && (
                  <>
                    <DatasetReport datasetId={d.id} tick={builtTick} />
                    <DatasetFiles datasetId={d.id} />
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}