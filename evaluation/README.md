# Evaluation Scripts

这一目录用于支撑论文 5.3 节的数据收集，不修改系统主业务逻辑。

## 文件说明

- `ranking_benchmark.py`
  对比关键词检索与本文系统检索排序的 Top-3 命中率、领域相关率、平均相关等级和 nDCG。
- `api_benchmark.py`
  对 `/search`、`/upload`、`/chat` 三个核心接口做计时统计。
- `answer_benchmark.py`
  对 `RAG 回答` 与 `直接 LLM 回答` 的结构完整性、证据意识和高风险术语进行轻量对比评估。
- `paper53_summary.py`
  汇总检索、问答、接口响应时间和参数敏感性结果，生成论文 5.3 可引用的表格文本。
- `cases/`
  存放样例配置文件，按需复制修改。
- `results/`
  存放脚本运行后输出的 JSON 结果。

## 运行方式

### 1. 推荐排序测试

```bash
python evaluation/ranking_benchmark.py --cases evaluation/cases/ranking_cases.sample.json
```

### 2. 接口性能测试

先启动后端服务：

```bash
uvicorn app.main:app --reload
```

再执行：

```bash
python evaluation/api_benchmark.py --config evaluation/cases/api_cases.sample.json
```

### 3. 回答质量测试

```bash
python evaluation/answer_benchmark.py --cases evaluation/cases/answer_cases.sample.json
```

### 4. 论文表格汇总

```bash
python evaluation/paper53_summary.py
```

## 输出结果

默认输出到：

- `evaluation/results/ranking_results.json`
- `evaluation/results/api_results.json`
- `evaluation/results/answer_results.json`

这些结果可直接整理为论文 5.3 节中的表格。
