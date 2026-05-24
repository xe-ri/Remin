# 基于检索增强生成的智能交通文献分析工具

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
- 支持通过 `/chat` 的 `mode=direct_llm` 生成不接入检索证据的直接回答，用于论文中的 `RAG vs 直接 LLM` 对比实验
- 支持对超长文档、空 PDF、无有效文本、全文下载失败等异常场景返回结构化告警信息

## 主要目录

- `app/`: FastAPI 接口、智能交通文献检索排序、RAG 生成、向量库读写
- `data_pipeline/`: PDF 解析与文本切片
- `frontend/`: Next.js 前端界面
- `training/`: 智能交通领域微调与评估脚本
- `evaluation/`: 检索、接口响应时间和问答质量测试脚本
- `storage/`: ChromaDB 和上传文件缓存，本地运行时自动产生
- `models/remin_adapter/`: LoRA adapter 输出目录，仓库中仅保留说明文件

> GitHub 仓库默认只保留代码、配置示例和目录占位文件，不提交 `.env`、论文正文、PDF、ChromaDB 数据、依赖目录、模型权重和生成训练数据。

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
- `DIRECT_LLM_MODEL_PATH`
  可选，用于指定 `/chat` 的 `mode=direct_llm` 直接回答基线模型。该路径可以是未微调的基座模型目录，不要求使用 LoRA adapter。
- `LOCAL_BASE_MODEL_PATH` / `BASE_MODEL_PATH`
  当 `models/remin_adapter` 中保存的是 LoRA adapter 时，可用这两个变量指定本地基座模型目录，避免运行时再访问 Hugging Face；若未设置 `DIRECT_LLM_MODEL_PATH`，直接 LLM 基线也会尝试使用这里配置的基座模型。
- `EMBEDDING_MODEL_NAME`
  可选，默认是 `sentence-transformers/all-MiniLM-L6-v2`，用于指定向量检索使用的嵌入模型。
- `LOCAL_EMBEDDING_MODEL_PATH`
  可选，用于指定本地嵌入模型目录。设置后，ChromaDB 检索阶段会优先从本地加载嵌入模型，不再依赖联网访问 Hugging Face。
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`
  可选，用于配置文本切片长度和重叠长度，默认分别为 `800` 和 `100`。
- `RAG_TOP_K`
  可选，用于配置问答阶段的默认向量召回数量，默认是 `10`。
- `MAX_UPLOAD_BYTES`
  可选，用于限制上传 PDF 大小，默认约 `20MB`。
- `MAX_DOCUMENT_CHARACTERS` / `MAX_CHUNKS_PER_DOCUMENT`
  可选，用于限制超长文档入库时的最大字符数和最大切片数，避免异常大文件影响系统稳定性。

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

## 论文 5.3 测试脚本

项目提供 `evaluation/` 目录，用于支撑毕业论文 5.3 节的数据收集，不影响系统主流程：

- `evaluation/ranking_benchmark.py`
  对比关键词检索与本文系统检索排序的 Top-3 命中率、领域相关率、平均相关等级和 nDCG
- `evaluation/api_benchmark.py`
  统计 `/search`、`/upload`、`/chat` 三个接口的响应时间
- `evaluation/answer_benchmark.py`
  批量对比 `RAG 回答` 与 `直接 LLM 回答` 的结构完整性、证据意识和高风险术语情况
- `evaluation/paper53_summary.py`
  汇总检索、问答、接口响应时间和参数敏感性结果，生成论文 5.3 可引用的表格文本

具体配置样例位于 `evaluation/cases/`，运行结果默认输出到 `evaluation/results/`。

常用运行命令：

```bash
python evaluation/ranking_benchmark.py
python evaluation/api_benchmark.py
python evaluation/answer_benchmark.py
python evaluation/paper53_summary.py
```
