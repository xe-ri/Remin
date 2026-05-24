import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


RESULT_DIR = Path("evaluation/results")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return "待补"


def print_ranking_table(ranking: Dict[str, Any]) -> None:
    summary = ranking.get("summary", {})
    if not summary:
        print("表5.2：尚未生成 ranking_results.json")
        return

    print("表5.2  检索效果对比表")
    print("方法\tTop-K命中率\t领域相关率\t平均相关等级\tnDCG")
    keyword_hit_rate = (
        summary.get("avg_keyword_hits", 0) / max(summary.get("top_k", 1), 1)
    )
    joint_hit_rate = (
        summary.get("avg_joint_hits", 0) / max(summary.get("top_k", 1), 1)
    )
    print(
        "关键词检索\t"
        f"{format_number(keyword_hit_rate)}\t"
        f"{format_number(summary.get('avg_keyword_domain_rate'))}\t"
        f"{format_number(summary.get('avg_keyword_relevance_grade'))}\t"
        f"{format_number(summary.get('avg_keyword_ndcg'))}"
    )
    print(
        "本文系统检索排序\t"
        f"{format_number(joint_hit_rate)}\t"
        f"{format_number(summary.get('avg_joint_domain_rate'))}\t"
        f"{format_number(summary.get('avg_joint_relevance_grade'))}\t"
        f"{format_number(summary.get('avg_joint_ndcg'))}"
    )
    print()


def evidence_citation_counts(answer: Dict[str, Any], mode: str) -> List[int]:
    return [
        int(item.get(mode, {}).get("evaluation", {}).get("citation_count", 0))
        for item in answer.get("results", [])
    ]


def print_answer_table(answer: Dict[str, Any], api: Dict[str, Any]) -> None:
    summary = answer.get("summary", {})
    if not summary:
        print("表5.3：尚未生成 answer_results.json")
        return

    api_summary = api.get("summary", {})
    chat_time = api_summary.get("chat", {}).get("avg_seconds")
    rag_citations = evidence_citation_counts(answer, "rag")
    direct_citations = evidence_citation_counts(answer, "direct_llm")
    avg_rag_citations = mean(rag_citations) if rag_citations else 0
    avg_direct_citations = mean(direct_citations) if direct_citations else 0

    print("表5.3  问答质量与响应时间对比表")
    print("方法\t回答质量得分\t平均证据引用数\t证据意识\t平均响应时间/s")
    print(
        "直接LLM回答\t"
        f"{format_number(summary.get('avg_direct_score'))}\t"
        f"{format_number(avg_direct_citations)}\t"
        f"{summary.get('direct_evidence_awareness_count', 0)}/{summary.get('direct_available_case_count', 0)}\t"
        f"{format_number(chat_time)}"
    )
    print(
        "本文RAG问答\t"
        f"{format_number(summary.get('avg_rag_score'))}\t"
        f"{format_number(avg_rag_citations)}\t"
        f"{summary.get('rag_evidence_awareness_count', 0)}/{summary.get('case_count', 0)}\t"
        f"{format_number(chat_time)}"
    )
    print()


def print_api_summary(api: Dict[str, Any]) -> None:
    summary = api.get("summary", {})
    if not summary:
        print("接口响应时间：尚未生成 api_results.json")
        return

    print("接口响应时间汇总")
    print("接口\t平均耗时/s\t最小耗时/s\t最大耗时/s")
    for name in ("search", "upload", "chat"):
        item = summary.get(name, {})
        print(
            f"{name}\t"
            f"{format_number(item.get('avg_seconds'))}\t"
            f"{format_number(item.get('min_seconds'))}\t"
            f"{format_number(item.get('max_seconds'))}"
        )
    print()


def print_parameter_table() -> None:
    configs = [
        ("400", "50", "5"),
        ("800", "100", "10"),
        ("1200", "200", "15"),
    ]
    rows = []
    for chunk_size, overlap, top_k in configs:
        suffix = f"{chunk_size}_{overlap}_{top_k}"
        api = load_json(RESULT_DIR / f"api_results_{suffix}.json")
        answer = load_json(RESULT_DIR / f"answer_results_{suffix}.json")
        if not api or not answer:
            rows.append((chunk_size, overlap, top_k, "待补", "待补", "待补"))
            continue

        rows.append(
            (
                chunk_size,
                overlap,
                top_k,
                format_number(answer.get("summary", {}).get("avg_rag_score")),
                format_number(api.get("summary", {}).get("chat", {}).get("avg_seconds")),
                format_number(api.get("upload_runs", [{}])[0].get("chunk_count")),
            )
        )

    print("表5.5  参数敏感性分析表")
    print("文本块长度\t重叠长度\t检索数量\t回答相关性\t平均响应时间/s\t上传切片数")
    for row in rows:
        print("\t".join(row))
    print()


def main() -> None:
    ranking = load_json(RESULT_DIR / "ranking_results.json")
    answer = load_json(RESULT_DIR / "answer_results.json")
    api = load_json(RESULT_DIR / "api_results.json")

    print_ranking_table(ranking)
    print_answer_table(answer, api)
    print_api_summary(api)
    print_parameter_table()


if __name__ == "__main__":
    main()
