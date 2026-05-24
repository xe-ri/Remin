import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.search_engine import fetch_and_sort_papers, _normalize_paper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ranking quality for thesis chapter 6."
    )
    parser.add_argument(
        "--cases",
        default="evaluation/cases/ranking_cases.sample.json",
        help="JSON file containing ranking benchmark cases.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k cutoff used for hit counting.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/ranking_results.json",
        help="Output JSON file for benchmark results.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Ranking cases JSON must be a list.")
    return data


def is_relevant(paper: dict, relevant_keywords: List[str]) -> bool:
    if paper.get("benchmark_relevant") is False:
        return False
    searchable = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return any(keyword.lower() in searchable for keyword in relevant_keywords)


def relevance_grade(paper: dict, relevant_keywords: List[str]) -> float:
    if paper.get("benchmark_relevant") is False:
        return 0.0
    searchable = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    keyword_hits = sum(1 for keyword in relevant_keywords if keyword.lower() in searchable)
    keyword_score = keyword_hits / max(len(relevant_keywords), 1)
    domain_score = float(paper.get("domain_score", 0) or 0)
    recency_score = float(paper.get("recency_score", 0) or 0)
    return round(keyword_score * 0.65 + domain_score * 0.25 + recency_score * 0.1, 4)


def sort_similarity_only(papers: List[dict]) -> List[dict]:
    return sorted(
        papers,
        key=lambda item: (
            item.get("similarity_score", 0),
            item.get("year") or 0,
        ),
        reverse=True,
    )


def sort_joint_score(papers: List[dict]) -> List[dict]:
    return sorted(
        papers,
        key=lambda item: (
            item.get("remin_score", 0),
            item.get("domain_score", 0),
            item.get("similarity_score", 0),
            item.get("year") or 0,
        ),
        reverse=True,
    )


def discounted_cumulative_gain(grades: List[float]) -> float:
    return sum(grade / math.log2(index + 2) for index, grade in enumerate(grades))


def evaluate_top_k(papers: List[dict], relevant_keywords: List[str], top_k: int) -> Dict[str, object]:
    ranked = papers[:top_k]
    evaluated_count = len(ranked)
    hits = [paper for paper in ranked if is_relevant(paper, relevant_keywords)]
    grades = [relevance_grade(paper, relevant_keywords) for paper in ranked]
    ideal_grades = sorted(
        [relevance_grade(paper, relevant_keywords) for paper in papers],
        reverse=True,
    )[:top_k]
    dcg = discounted_cumulative_gain(grades)
    ideal_dcg = discounted_cumulative_gain(ideal_grades)
    domain_hits = [
        paper for paper in ranked
        if paper.get("domain_tags") or (paper.get("domain_score", 0) >= 0.35)
    ]
    first_hit_rank = next(
        (index for index, paper in enumerate(ranked, start=1) if is_relevant(paper, relevant_keywords)),
        None,
    )
    return {
        "requested_top_k": top_k,
        "evaluated_count": evaluated_count,
        "top_k_hits": len(hits),
        "hit_rate": round(len(hits) / max(evaluated_count, 1), 4),
        "domain_relevance_rate": round(len(domain_hits) / max(evaluated_count, 1), 4),
        "avg_relevance_grade": round(mean(grades), 4) if grades else 0,
        "mrr": round(1 / first_hit_rank, 4) if first_hit_rank else 0,
        "ndcg": round(dcg / ideal_dcg, 4) if ideal_dcg else 0,
        "first_hit_rank": first_hit_rank,
        "top_titles": [paper.get("title", "") for paper in ranked],
        "top_relevance_grades": grades,
    }


def build_distractor_papers(case: dict) -> List[dict]:
    distractors = []
    for index, paper in enumerate(case.get("distractors", []), start=1):
        normalized = _normalize_paper(
            {
                "paperId": paper.get("paperId") or f"distractor-{case['keyword']}-{index}",
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "year": int(paper.get("year", 2024)),
                "authors": paper.get("authors", ["Synthetic Distractor"]),
                "externalIds": {},
                "openAccessPdf": None,
                "url": None,
            },
            str(case["keyword"]),
        )
        normalized["benchmark_relevant"] = False
        distractors.append(normalized)
    return distractors


def main():
    args = parse_args()
    cases_path = Path(args.cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    results = []

    for case in cases:
        keyword = str(case["keyword"]).strip()
        relevant_keywords = [str(item).strip() for item in case["relevant_keywords"]]
        limit = int(case.get("limit", max(args.top_k * 2, 8)))

        papers, result_source = fetch_and_sort_papers(keyword, limit)
        for paper in papers:
            paper.setdefault("benchmark_relevant", is_relevant(paper, relevant_keywords))
        distractors = build_distractor_papers(case)
        papers = papers + distractors
        similarity_ranked = sort_similarity_only(papers)
        joint_ranked = sort_joint_score(papers)

        similarity_metrics = evaluate_top_k(similarity_ranked, relevant_keywords, args.top_k)
        joint_metrics = evaluate_top_k(joint_ranked, relevant_keywords, args.top_k)

        results.append(
            {
                "keyword": keyword,
                "result_source": result_source,
                "distractor_count": len(distractors),
                "relevant_keywords": relevant_keywords,
                "keyword_baseline": similarity_metrics,
                "joint_score": joint_metrics,
                "improvement_hits": joint_metrics["top_k_hits"] - similarity_metrics["top_k_hits"],
            }
        )

    summary = {
        "case_count": len(results),
        "top_k": args.top_k,
        "avg_keyword_hits": round(mean(item["keyword_baseline"]["top_k_hits"] for item in results), 4)
        if results
        else 0,
        "avg_joint_hits": round(mean(item["joint_score"]["top_k_hits"] for item in results), 4)
        if results
        else 0,
        "avg_keyword_domain_rate": round(mean(item["keyword_baseline"]["domain_relevance_rate"] for item in results), 4)
        if results
        else 0,
        "avg_joint_domain_rate": round(mean(item["joint_score"]["domain_relevance_rate"] for item in results), 4)
        if results
        else 0,
        "avg_keyword_relevance_grade": round(mean(item["keyword_baseline"]["avg_relevance_grade"] for item in results), 4)
        if results
        else 0,
        "avg_joint_relevance_grade": round(mean(item["joint_score"]["avg_relevance_grade"] for item in results), 4)
        if results
        else 0,
        "avg_keyword_mrr": round(mean(item["keyword_baseline"]["mrr"] for item in results), 4)
        if results
        else 0,
        "avg_joint_mrr": round(mean(item["joint_score"]["mrr"] for item in results), 4)
        if results
        else 0,
        "avg_keyword_ndcg": round(mean(item["keyword_baseline"]["ndcg"] for item in results), 4)
        if results
        else 0,
        "avg_joint_ndcg": round(mean(item["joint_score"]["ndcg"] for item in results), 4)
        if results
        else 0,
        "avg_hit_improvement": round(mean(item["improvement_hits"] for item in results), 4)
        if results
        else 0,
    }

    payload = {
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Ranking benchmark finished. Results saved to: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
