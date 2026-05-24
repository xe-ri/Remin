import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark answer structure and evidence awareness for thesis chapter 6."
    )
    parser.add_argument(
        "--cases",
        default="evaluation/cases/answer_cases.sample.json",
        help="JSON file containing answer benchmark cases.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/answer_results.json",
        help="Output JSON file for answer benchmark results.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Answer benchmark cases JSON must be a list.")
    return data


def resolve_paper_ids(case: dict) -> List[str]:
    from app.search_engine import fetch_and_sort_papers

    if "paper_ids" in case:
        return [str(item) for item in case["paper_ids"]]

    keyword = str(case["search_keyword"]).strip()
    limit = int(case.get("limit", 8))
    papers, _ = fetch_and_sort_papers(keyword, limit)
    paper_ids = []
    for index in case.get("selected_paper_indexes", [1]):
        position = int(index) - 1
        if 0 <= position < len(papers):
            paper_ids.append(str(papers[position]["paperId"]))
    return paper_ids


def build_context(query: str, source_ids: List[str]) -> str:
    from app.database import query_vector_db
    from app.rag_service import _build_retrieval_query

    retrieval_query = _build_retrieval_query(query)
    result = query_vector_db(retrieval_query, n_results=10, source_ids=source_ids)
    documents = result.get("documents", [[]])[0]
    return "\n".join(documents)


def main():
    args = parse_args()

    from app.main import _ingest_selected_papers
    from app.rag_service import generate_answer_bundle
    from training.evaluate import evaluate_academic_answer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(Path(args.cases))

    results = []
    rag_scores = []
    direct_scores = []

    for case in cases:
        query = str(case["query"]).strip()
        paper_ids = resolve_paper_ids(case)
        ingested_sources = _ingest_selected_papers(paper_ids) if paper_ids else []
        uploaded_source_ids = [str(item) for item in case.get("uploaded_source_ids", [])]
        source_ids = ingested_sources + uploaded_source_ids

        if not source_ids:
            raise ValueError(f"Case has no usable sources: {case}")

        context = build_context(query, source_ids)
        rag_bundle = generate_answer_bundle(query, source_ids=source_ids, mode="rag")
        rag_evaluation = evaluate_academic_answer(rag_bundle.answer, context)
        rag_scores.append(rag_evaluation["score"])

        direct_bundle = generate_answer_bundle(query, source_ids=source_ids, mode="direct_llm")
        direct_context = "" if direct_bundle.answer_mode == "direct_llm" else context
        direct_evaluation = evaluate_academic_answer(direct_bundle.answer, direct_context)
        direct_available = direct_bundle.answer_mode == "direct_llm"
        if direct_available:
            direct_scores.append(direct_evaluation["score"])

        results.append(
            {
                "query": query,
                "source_ids": source_ids,
                "rag": {
                    "answer_mode": rag_bundle.answer_mode,
                    "answer_preview": rag_bundle.answer[:400],
                    "warnings": rag_bundle.warnings,
                    "evaluation": rag_evaluation,
                },
                "direct_llm": {
                    "available": direct_available,
                    "answer_mode": direct_bundle.answer_mode,
                    "answer_preview": direct_bundle.answer[:400],
                    "warnings": direct_bundle.warnings,
                    "evaluation": direct_evaluation,
                },
            }
        )

    payload = {
        "summary": {
            "case_count": len(results),
            "avg_rag_score": round(mean(rag_scores), 4) if rag_scores else 0,
            "avg_direct_score": round(mean(direct_scores), 4) if direct_scores else 0,
            "rag_strict_pass_count": sum(
                1 for item in results if item["rag"]["evaluation"]["label"] == "较严谨"
            ),
            "direct_strict_pass_count": sum(
                1
                for item in results
                if item["direct_llm"]["available"]
                and item["direct_llm"]["evaluation"]["label"] == "较严谨"
            ),
            "rag_evidence_awareness_count": sum(
                1 for item in results if item["rag"]["evaluation"]["evidence_awareness"]
            ),
            "direct_evidence_awareness_count": sum(
                1
                for item in results
                if item["direct_llm"]["available"]
                and item["direct_llm"]["evaluation"]["evidence_awareness"]
            ),
            "direct_available_case_count": len(direct_scores),
            "direct_unavailable_case_count": len(results) - len(direct_scores),
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Answer benchmark finished. Results saved to: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
