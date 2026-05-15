import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_OUTPUT = Path("training/datasets/source_papers.jsonl")
DEFAULT_TOPICS = [
    "traffic flow prediction",
    "traffic signal control",
    "traffic state estimation",
    "trajectory prediction",
    "vehicle routing",
    "connected vehicles",
    "V2X communication",
    "autonomous driving",
    "traffic congestion analysis",
    "transportation graph learning",
]
DOMAIN_HINT = "intelligent transportation systems traffic transportation vehicle"
TOPIC_QUERY_EXPANSIONS = {
    "traffic flow prediction": [
        "traffic flow prediction",
        "traffic forecasting",
        "traffic speed prediction",
    ],
    "traffic signal control": [
        "traffic signal control",
        "traffic light control",
        "adaptive traffic signal control",
    ],
    "traffic state estimation": [
        "traffic state estimation",
        "traffic state prediction",
        "traffic condition estimation",
    ],
    "trajectory prediction": [
        "trajectory prediction",
        "vehicle trajectory prediction",
        "traffic trajectory forecasting",
    ],
    "vehicle routing": [
        "vehicle routing",
        "vehicle routing problem",
        "route planning transportation",
    ],
    "connected vehicles": [
        "connected vehicles",
        "vehicular networks",
        "cooperative vehicles",
    ],
    "V2X communication": [
        "V2X communication",
        "vehicle to everything",
        "vehicular communication",
    ],
    "autonomous driving": [
        "autonomous driving",
        "self-driving vehicles",
        "intelligent vehicles",
    ],
    "traffic congestion analysis": [
        "traffic congestion analysis",
        "traffic congestion prediction",
        "urban congestion",
    ],
    "transportation graph learning": [
        "transportation graph learning",
        "graph neural network traffic",
        "graph learning transportation",
    ],
}


def _decode_abstract(inverted_index: Optional[dict]) -> str:
    if not inverted_index:
        return ""

    words: List[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            words.append((position, token))

    words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in words)


def _normalize_authors(authorships: List[dict]) -> List[str]:
    return [
        (authorship.get("author") or {}).get("display_name", "").strip()
        for authorship in authorships
        if (authorship.get("author") or {}).get("display_name")
    ]


def _normalize_keywords(work: dict, fallback_topic: str) -> List[str]:
    keywords: List[str] = []

    primary_topic = (work.get("primary_topic") or {}).get("display_name")
    if primary_topic:
        keywords.append(primary_topic)

    for concept in work.get("concepts", [])[:8]:
        name = (concept.get("display_name") or "").strip()
        if name:
            keywords.append(name)

    keywords.append(fallback_topic)

    deduped: List[str] = []
    seen: Set[str] = set()
    for item in keywords:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(item)
    return deduped[:8]


def _normalize_work(work: dict, topic: str) -> Optional[Dict]:
    title = (work.get("display_name") or "").strip()
    abstract = _decode_abstract(work.get("abstract_inverted_index")).strip()
    if not title or not abstract:
        return None

    ids = work.get("ids") or {}
    paper_id = work.get("id") or ids.get("doi") or title

    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": _normalize_authors(work.get("authorships", [])),
        "year": work.get("publication_year"),
        "keywords": _normalize_keywords(work, topic),
        "citationCount": work.get("cited_by_count", 0),
        "topic": topic,
        "source": "openalex",
        "openalex_id": work.get("id"),
        "doi": ids.get("doi"),
        "evidence_chunks": [],
    }


def _build_params(query: str, per_page: int, cursor: str = "*") -> dict:
    params = {
        "search": query,
        "per-page": per_page,
        "cursor": cursor,
        "select": ",".join(
            [
                "id",
                "ids",
                "display_name",
                "authorships",
                "publication_year",
                "cited_by_count",
                "abstract_inverted_index",
                "concepts",
                "primary_topic",
            ]
        ),
    }

    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key

    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto

    return params


def _build_search_queries(topic: str) -> List[str]:
    expansions = TOPIC_QUERY_EXPANSIONS.get(topic, [topic])
    candidates = [
        f"{item} {DOMAIN_HINT}" for item in expansions
    ] + [
        topic,
        f"{topic} traffic",
        f"{topic} transportation",
    ]

    seen: Set[str] = set()
    queries: List[str] = []
    for item in candidates:
        normalized = item.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(item.strip())
    return queries


def _iter_query_works(query: str, max_results: int, per_page: int, sleep_seconds: float) -> Iterable[dict]:
    collected = 0
    cursor = "*"

    while collected < max_results:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=_build_params(query, min(per_page, max_results - collected), cursor),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break

        for work in results:
            yield work
            collected += 1
            if collected >= max_results:
                break

        meta = payload.get("meta") or {}
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _iter_topic_works(topic: str, papers_per_topic: int, per_page: int, sleep_seconds: float) -> Iterable[dict]:
    yielded_ids: Set[str] = set()
    collected = 0

    # Each topic can fall back across several query variants, which helps avoid
    # very small first-round exports caused by a single over-narrow query.
    for query in _build_search_queries(topic):
        if collected >= papers_per_topic:
            break

        remaining = papers_per_topic - collected
        query_budget = max(remaining * 2, per_page)
        for work in _iter_query_works(query, query_budget, per_page, sleep_seconds):
            work_id = str(work.get("id") or (work.get("ids") or {}).get("doi") or "").strip()
            if not work_id or work_id in yielded_ids:
                continue

            yielded_ids.add(work_id)
            yield work
            collected += 1
            if collected >= papers_per_topic:
                break


def _load_existing_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()

    seen: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(paper.get("paperId") or paper.get("openalex_id") or paper.get("title") or "").strip()
            if paper_id:
                seen.add(paper_id)
    return seen


def export_openalex_papers(
    output_path: Path,
    topics: List[str],
    papers_per_topic: int,
    per_page: int,
    sleep_seconds: float,
    overwrite: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set() if overwrite else _load_existing_ids(output_path)
    mode = "w" if overwrite else "a"
    written = 0

    with output_path.open(mode, encoding="utf-8") as file:
        for topic in topics:
            for work in _iter_topic_works(topic, papers_per_topic, per_page, sleep_seconds):
                normalized = _normalize_work(work, topic)
                if not normalized:
                    continue

                paper_id = str(normalized.get("paperId", "")).strip()
                if not paper_id or paper_id in existing_ids:
                    continue

                existing_ids.add(paper_id)
                file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1

    return written


def _parse_topics(raw_topics: List[str]) -> List[str]:
    topics: List[str] = []
    for item in raw_topics:
        parts = [part.strip() for part in re.split(r"[,\n]+", item) if part.strip()]
        topics.extend(parts)
    return topics or DEFAULT_TOPICS


def main():
    parser = argparse.ArgumentParser(
        description="Export intelligent transportation papers from OpenAlex to source_papers.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSONL 路径。",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="智能交通子主题，可重复传入；不传则使用默认主题列表。",
    )
    parser.add_argument(
        "--papers-per-topic",
        type=int,
        default=30,
        help="每个主题抓取的论文数量。",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=25,
        help="每次请求拉取的论文数量。",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="请求之间的休眠时间，避免过快访问接口。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出文件；默认采用追加写入并按 paperId 去重。",
    )
    args = parser.parse_args()

    topics = _parse_topics(args.topic)
    count = export_openalex_papers(
        output_path=args.output,
        topics=topics,
        papers_per_topic=args.papers_per_topic,
        per_page=args.per_page,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
    )
    print(f"已写入 {count} 篇智能交通领域论文 -> {args.output}")


if __name__ == "__main__":
    main()
