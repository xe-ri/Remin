import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_INPUT = Path("training/datasets/source_papers.jsonl")
DEFAULT_OUTPUT = Path("training/datasets/source_papers.cleaned.jsonl")

ITS_REQUIRED_HINTS = {
    "traffic",
    "transportation",
    "vehicle",
    "vehicles",
    "vehicular",
    "road",
    "roads",
    "routing",
    "driving",
    "autonomous",
    "mobility",
    "congestion",
    "signal",
    "v2x",
    "vanet",
    "trajectory",
    "iot",
    "intelligent transportation",
}

GENERIC_NOISY_KEYWORDS = {
    "computer science",
    "artificial intelligence",
    "the internet",
    "wireless",
    "visualization",
    "graph",
    "software",
    "computer network",
}


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff\-]+", (text or "").lower()))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_keywords(paper: Dict) -> List[str]:
    keywords = paper.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    return [item.strip() for item in keywords if item and item.strip()]


def _paper_search_blob(paper: Dict) -> str:
    parts = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("topic", ""),
        " ".join(_normalize_keywords(paper)),
    ]
    return _normalize_text(" ".join(parts)).lower()


def _is_its_related(paper: Dict, min_hits: int) -> bool:
    blob = _paper_search_blob(paper)
    hits = 0
    for hint in ITS_REQUIRED_HINTS:
        if hint in blob:
            hits += 1
    return hits >= min_hits


def _clean_keywords(paper: Dict) -> List[str]:
    cleaned: List[str] = []
    seen: Set[str] = set()
    for item in _normalize_keywords(paper):
        lowered = item.lower()
        if lowered in GENERIC_NOISY_KEYWORDS:
            continue
        if lowered not in seen:
            seen.add(lowered)
            cleaned.append(item)
    return cleaned[:8]


def _should_keep(paper: Dict, min_abstract_words: int, min_domain_hits: int) -> Tuple[bool, str]:
    title = _normalize_text(paper.get("title", ""))
    abstract = _normalize_text(paper.get("abstract", ""))

    if not title:
        return False, "missing_title"
    if not abstract:
        return False, "missing_abstract"

    abstract_words = len(abstract.split())
    if abstract_words < min_abstract_words:
        return False, "abstract_too_short"

    if "暂无摘要" in abstract:
        return False, "placeholder_abstract"

    if not _is_its_related(paper, min_domain_hits):
        return False, "not_its_related"

    return True, "kept"


def clean_papers(input_path: Path, output_path: Path, min_abstract_words: int, min_domain_hits: int) -> Dict[str, int]:
    stats = {
        "input": 0,
        "kept": 0,
        "removed_duplicate": 0,
        "missing_title": 0,
        "missing_abstract": 0,
        "abstract_too_short": 0,
        "placeholder_abstract": 0,
        "not_its_related": 0,
    }

    seen_ids: Set[str] = set()
    seen_dois: Set[str] = set()
    seen_title_year: Set[Tuple[str, str]] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue

            stats["input"] += 1
            paper = json.loads(line)

            paper_id = _normalize_text(str(paper.get("paperId", "")))
            doi = _normalize_text(str(paper.get("doi", ""))).lower()
            title = _normalize_text(paper.get("title", "")).lower()
            year = str(paper.get("year", "")).strip()

            duplicate = False
            if paper_id and paper_id in seen_ids:
                duplicate = True
            if doi and doi in seen_dois:
                duplicate = True
            if title and (title, year) in seen_title_year:
                duplicate = True

            if duplicate:
                stats["removed_duplicate"] += 1
                continue

            keep, reason = _should_keep(paper, min_abstract_words, min_domain_hits)
            if not keep:
                stats[reason] += 1
                continue

            paper["title"] = _normalize_text(paper.get("title", ""))
            paper["abstract"] = _normalize_text(paper.get("abstract", ""))
            paper["keywords"] = _clean_keywords(paper)
            paper["authors"] = [
                _normalize_text(author.get("name") if isinstance(author, dict) else str(author))
                for author in (paper.get("authors") or [])
                if _normalize_text(author.get("name") if isinstance(author, dict) else str(author))
            ]
            paper["evidence_chunks"] = paper.get("evidence_chunks") or []

            if paper_id:
                seen_ids.add(paper_id)
            if doi:
                seen_dois.add(doi)
            if title:
                seen_title_year.add((title, year))

            dst.write(json.dumps(paper, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean intelligent transportation source papers before SFT generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入原始 source_papers.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出清洗后的 JSONL")
    parser.add_argument("--min-abstract-words", type=int, default=60, help="摘要最少单词数")
    parser.add_argument("--min-domain-hits", type=int, default=2, help="至少命中多少个 ITS 领域提示词")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"未找到输入文件: {args.input}")

    stats = clean_papers(
        input_path=args.input,
        output_path=args.output,
        min_abstract_words=args.min_abstract_words,
        min_domain_hits=args.min_domain_hits,
    )

    print(
        "清洗完成 -> {output}\n"
        "输入: {input_count} | 保留: {kept} | 去重: {duplicates} | 过滤(缺标题/缺摘要/过短/占位/非ITS): "
        "{missing_title}/{missing_abstract}/{short_abs}/{placeholder}/{not_its}".format(
            output=args.output,
            input_count=stats["input"],
            kept=stats["kept"],
            duplicates=stats["removed_duplicate"],
            missing_title=stats["missing_title"],
            missing_abstract=stats["missing_abstract"],
            short_abs=stats["abstract_too_short"],
            placeholder=stats["placeholder_abstract"],
            not_its=stats["not_its_related"],
        )
    )


if __name__ == "__main__":
    main()
