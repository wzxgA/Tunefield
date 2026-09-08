import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import DatasetFiles from "../components/DatasetFiles.jsx";

// 数据上传屏（T1 接入 + T2 解析预览）：多文件上传 + 哈希去重 +
// 已接入 dataset 清单；点开数据集可逐文件查看解析状态与中间文本预览
export default function UploadScreen() {
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]); // File[]
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(""); // "上传成功 / 去重命中 / 错误"
  const [noticeKind, setNoticeKind] = useState(""); // ok | dup | err
  const [inspect, setInspect] = useState({}); // {datasetId: bool} 展开解析预览

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
                    <span className="badge st-ingested">已接入</span>
                    <button
                      type="button"
                      className="btn-inspect"
                      onClick={() => toggleInspect(d.id)}
                    >
                      {inspect[d.id] ? "收起解析" : "解析预览"}
                    </button>
                  </span>
                </div>
                <div className="ds-line2">
                  <span className="ds-id">{d.id.slice(0, 10)}</span>
                  <span className="ds-fp">指纹 {d.content_hash.slice(0, 10)}…</span>
                </div>
                {inspect[d.id] && <DatasetFiles datasetId={d.id} />}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}