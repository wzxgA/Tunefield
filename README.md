# Tunefield

领域专属小模型训练平台：上传一批领域数据（纯文本 / pdf / docx / 代码），一条链跑通
**接入 → 解析 → 清洗 → 切片 → 指令化 → 质检 → 微调（LoRA）/ 从零预训练 → GGUF 量化导出 → 导入 Ollama → 对话验证**。

- **Web 工作站**：暗色银琉璃界面，五入口（编排 / 数据 / 训练 / 模型 / 对话）；
- **整屏编排画布**：选数据集进入，节点即 pipeline 阶段，可拖拽连线、改参数、一键运行，运行时按阶段实时点亮；
- **CUDA OOM 自动降级**：训练显存不足时自动依次缩短序列 → 降低量化位宽 → 降基座档位，不中断任务；
- **CLI 全命令**：`ingest / build / train / export / chat / run`，与 Web 共享同一套管线与引擎。

## 环境要求

- Python 3.11+（包管理用 [uv](https://docs.astral.sh/uv/)）
- Node.js 18+（仅前端构建）
- 训练需 NVIDIA GPU（8GB 起步，推荐器按显存自动选档）；无 GPU 时管线与质检仍可用
- 对话验证需本机安装并运行 [Ollama](https://ollama.com/)

## 启动

```bash
# 1. 安装依赖（自动创建虚拟环境）
uv sync

# 2. 构建前端（产物输出到 web/dist，由后端托管）
npm --prefix web install
npm --prefix web run build

# 3. 启动平台
uv run tunefield serve          # 默认 http://127.0.0.1:8000
uv run tunefield serve --port 8747
```

浏览器打开 `http://127.0.0.1:8000` 即是工作站。

## 常用操作

**Web**：「数据」上传文件并构建 → 「编排」选数据集进画布 → 调参数（块长/模板/轮次）→ 运行流水线，节点逐步点亮 → 「模型」一键导入 Ollama → 「对话」试用。

**CLI**：

```bash
uv run tunefield run <数据目录> --name 我的领域        # 一键端到端
uv run tunefield run <数据目录> --name 我的领域 --no-import   # 不自动导入 Ollama
uv run tunefield train <数据集> --dry-run              # 预览推荐训练配置
uv run tunefield chat <模型名或 gguf 路径>             # 与产物对话
```

## 目录结构

```
tunefield/            # 后端包
  pipeline/           # 接入 / 解析 / 清洗 / 切片 / 指令化 / 质检
  engine/             # 引擎调度：A 微调(LLaMA-Factory QLoRA)、B 从零预训练、导出、降级链、端到端编排
  serve/              # FastAPI、内嵌队列、SQLite、事件流、Ollama 集成
  assets/             # 产物登记（LoRA / GGUF / 指纹）
web/                  # React 前端（Vite），构建产物由后端托管
data/                 # 运行时数据（raw / datasets / adapters / gguf / db）
plans/                # 设计与计划文档
tests/                # pytest（123 用例）
yuanxing/             # UI 原型（index-3 为界面定稿参考）
```

## 测试

```bash
uv run pytest -q
```
