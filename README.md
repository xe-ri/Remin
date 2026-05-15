# 基于 RAG 的智能交通系统文献在线分析工具

Remin 是一个面向智能交通系统领域的学术文献在线分析工具，围绕“智能交通主题搜索 -> 基于摘要相关性、领域相关性与时效性推荐文献 -> 用户选择后下载全文/上传 PDF -> RAG 文献分析问答与综述生成”构建完整闭环。

## 当前能力

- 面向智能交通系统领域进行论文检索，自动加入交通流预测、交通信号控制、车路协同、V2X、自动驾驶等领域约束词
- 按 `摘要相关性 + 智能交通领域相关性 + 时效性` 推荐论文
- 默认使用 `arXiv API` 获取在线预印本检索结果，失败时回退到智能交通方向演示数据
- 支持在用户勾选论文后再下载开放获取全文并参与分析
- 支持上传 PDF，解析全文并写入 ChromaDB
- 基于证据检索生成更审慎、更学术化的智能交通文献分析问答与综述生成结果
- 支持对 `Qwen3-8B` 进行智能交通领域学术化 LoRA 微调，以增强文献分析问答与综述生成时的严谨性与溯源能力
- 微调模型不可用时，自动回退到证据式回答

## 主要目录

- `app/`: FastAPI 接口、智能交通文献检索排序、RAG 生成、向量库读写
- `data_pipeline/`: PDF 解析与文本切片
- `frontend/`: Next.js 前端界面
- `training/`: 智能交通领域微调与评估脚本
- `storage/`: ChromaDB 和上传文件缓存
- `models/remin_adapter/`: 本地完整模型或 LoRA adapter 目录

> GitHub 仓库默认只保留代码、配置示例和目录占位文件，不提交 `.env`、PDF、ChromaDB 数据、依赖目录、模型权重和生成训练数据。

## 环境准备

建议环境：

- Python 3.10+
- Node.js 20+

复制环境变量示例文件：

```bash
cp .env.example .env
cp frontend/.env.local.example frontend/.env.local
```

安装后端依赖：

```bash
pip install -r requirements.txt
```

安装前端依赖：

```bash
cd frontend
npm install
```

## 后端启动

```bash
uvicorn app.main:app --reload
```

默认接口：

- `POST /search`
- `POST /upload`
- `POST /chat`

可选环境变量：

- `MODEL_PATH`
  可选，默认是 `models/remin_adapter`，用于指定本地完整模型或 LoRA adapter 目录。
- `LOCAL_BASE_MODEL_PATH` / `BASE_MODEL_PATH`
  当 `models/remin_adapter` 中保存的是 LoRA adapter 时，可用这两个变量指定本地基座模型目录，避免运行时再访问 Hugging Face。
- `EMBEDDING_MODEL_NAME`
  可选，默认是 `sentence-transformers/all-MiniLM-L6-v2`，用于指定向量检索使用的嵌入模型。
- `LOCAL_EMBEDDING_MODEL_PATH`
  可选，用于指定本地嵌入模型目录。设置后，ChromaDB 检索阶段会优先从本地加载嵌入模型，不再依赖联网访问 Hugging Face。

## 前端启动

```bash
cd frontend
npm run dev
```

前端默认访问 `http://127.0.0.1:8000` 作为后端服务地址。

## 智能交通领域检索逻辑

搜索阶段不会直接下载全文，而是先基于 arXiv 元数据和摘要进行推荐；若接口不可用，则回退到演示数据。系统会在用户输入主题后自动加入智能交通领域约束词，例如：

- `intelligent transportation system`
- `traffic flow prediction`
- `traffic signal control`
- `traffic state estimation`
- `connected vehicles`
- `V2X`
- `autonomous driving`
- `spatio-temporal traffic forecasting`
- `graph neural network traffic`
- `reinforcement learning traffic control`

推荐阶段综合计算：

- 用户主题与摘要的相关性
- 论文是否属于智能交通系统领域
- 论文发表时间所体现的时效性

只有当用户选择推荐文献或上传 PDF 后，系统才会进入全文解析、切片入库与 RAG 文献分析阶段。系统既支持围绕具体问题进行文献分析问答，也支持基于所选材料生成结构化综述。

## 微调现有大模型

先准备智能交通领域训练数据：

```bash
python training/datasets/synth_gen.py
```

生成的数据默认会采用更贴近线上回答的格式：

- `核心结论`
- `依据说明`
- `局限性`

并使用证据约束式中文问答模板，便于 Qwen3-8B 学到更适合智能交通系统文献分析问答和综述生成的回答风格。

再基于现有模型做 LoRA 微调：

```bash
python training/train_lora.py --base-model unsloth/Qwen3-8B-bnb-4bit
```

如果你已经有本地模型目录，也可以直接传本地路径：

```bash
python training/train_lora.py --base-model C:/your_models/your_base_model
```

如果你已经把基座模型下载到了本地，运行问答服务时也建议一并设置：

```bash
set LOCAL_BASE_MODEL_PATH=C:/your_models/Qwen3-8B-bnb-4bit
```

这样当前项目会优先从本地加载基座模型，再挂载 `models/remin_adapter` 中的 LoRA adapter。

如果你还希望让 RAG 检索完全离线运行，也建议把嵌入模型下载到本地，并设置：

```bash
set LOCAL_EMBEDDING_MODEL_PATH=C:/your_models/all-MiniLM-L6-v2
```

这样向量检索阶段也会优先使用本地嵌入模型。

微调输出默认保存在：

```bash
models/remin_adapter
```

当前默认训练方案：

- 基座模型：`Qwen3-8B`
- 微调方式：LoRA / PEFT
- 目标：强化智能交通系统领域的文献分析问答、证据约束回答、综述生成时的严谨性与可溯源性
- 推荐训练特征：更长上下文、结构化回答模板、显式保留“证据不足”表达
