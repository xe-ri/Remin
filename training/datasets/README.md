# Datasets

这个目录用于准备 Remin 的智能交通领域指令微调数据。

## 文件说明

- `source_papers.jsonl`
  原始论文元数据输入文件，每行一篇论文。
- `export_openalex.py`
  从 OpenAlex 批量导出智能交通系统领域论文元数据到 `source_papers.jsonl`。
- `clean_source_papers.py`
  对采集得到的论文元数据进行自动清洗，输出更适合微调的 `source_papers.cleaned.jsonl`。
- `synth_gen.py`
  将原始元数据转换为适合 SFT/LoRA 的 `train.jsonl`。
- `train.jsonl`
  运行生成脚本后得到的训练集文件。

## `source_papers.jsonl` 格式

每行是一个 JSON 对象，建议包含以下字段：

```json
{
  "paperId": "https://openalex.org/W1234567890",
  "title": "论文标题",
  "abstract": "论文摘要",
  "authors": ["作者A", "作者B"],
  "year": 2024,
  "keywords": ["traffic flow prediction", "intelligent transportation systems"],
  "citationCount": 12,
  "topic": "traffic flow prediction",
  "source": "openalex",
  "openalex_id": "https://openalex.org/W1234567890",
  "doi": "https://doi.org/10.xxxx/xxxx",
  "evidence_chunks": []
}
```

最少必填字段：

- `title`
- `abstract`

可选字段：

- `paperId`
- `authors`
- `year`
- `keywords`
- `citationCount`
- `topic`
- `source`
- `openalex_id`
- `doi`
- `evidence_chunks`

## 从 OpenAlex 导出智能交通论文

可选环境变量：

- `OPENALEX_API_KEY`
- `OPENALEX_MAILTO`

基础用法：

```bash
python training/datasets/export_openalex.py --overwrite
```

这会按默认的智能交通主题列表抓取论文并写入 `training/datasets/source_papers.jsonl`。

如果要指定主题：

```bash
python training/datasets/export_openalex.py \
  --topic "traffic flow prediction" \
  --topic "traffic signal control" \
  --topic "connected vehicles" \
  --papers-per-topic 40 \
  --overwrite
```

如果要在现有文件基础上继续追加：

```bash
python training/datasets/export_openalex.py --topic "V2X communication"
```

## 清洗原始语料

建议在生成训练集前，先对 `source_papers.jsonl` 做一轮自动清洗：

```bash
python training/datasets/clean_source_papers.py
```

默认会输出：

```bash
training/datasets/source_papers.cleaned.jsonl
```

如果要调整过滤强度：

```bash
python training/datasets/clean_source_papers.py --min-abstract-words 80 --min-domain-hits 3
```

## 生成训练集

在训练环境中运行：

```bash
python training/datasets/synth_gen.py --input training/datasets/source_papers.cleaned.jsonl
```

如果要指定输入输出路径：

```bash
python training/datasets/synth_gen.py --input training/datasets/source_papers.cleaned.jsonl --output training/datasets/train.jsonl
```

## 建议

- 优先收集与你目标场景相关的智能交通论文，例如交通流预测、信号控制、V2X、自动驾驶、车辆路径规划等。
- 先用 `export_openalex.py` 扩充 `source_papers.jsonl`，再人工抽检和补充高质量 `evidence_chunks`。
- 正式微调前，建议先运行 `clean_source_papers.py` 做自动去重和领域过滤。
- 尽量保证摘要质量，不要混入机器翻译错误或严重残缺文本。
- 训练前建议人工抽检部分 `train.jsonl`，确认回答风格符合“严谨、克制、证据优先”的目标。
