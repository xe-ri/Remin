import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

from app.its_domain import (
    build_domain_hint,
    classify_its_topics,
    expand_query_with_domain_terms,
    matched_domain_terms,
    tokenize_its_text,
)

ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
SEARCH_CACHE: Dict[str, dict] = {}
SEARCH_RESULT_CACHE: Dict[str, tuple[float, List[dict], str]] = {}
SEARCH_RESULT_CACHE_TTL_SECONDS = 30 * 60
ARXIV_RATE_LIMIT_COOLDOWN_SECONDS = 15 * 60
ARXIV_RATE_LIMITED_UNTIL = 0.0
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 35
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
}
FALLBACK_PAPERS = [
    {
        "paperId": "demo-its-1",
        "title": "Spatio-Temporal Graph Neural Networks for Traffic Flow Prediction",
        "abstract": "This paper studies traffic flow prediction in intelligent transportation systems using spatio-temporal graph neural networks. It models road network topology, temporal dependencies, and congestion propagation to improve short-term traffic forecasting.",
        "year": 2024,
        "authors": [{"name": "Demo ITS Author A"}, {"name": "Demo ITS Author B"}],
        "externalIds": {"ArXiv": "demo-its-1"},
        "url": None,
        "openAccessPdf": None,
    },
    {
        "paperId": "demo-its-2",
        "title": "Deep Reinforcement Learning for Adaptive Traffic Signal Control",
        "abstract": "The study proposes a reinforcement learning framework for adaptive traffic signal control. It optimizes intersection signal timing based on traffic demand, queue length, and network-level mobility efficiency in urban transportation systems.",
        "year": 2023,
        "authors": [{"name": "Demo ITS Author C"}],
        "externalIds": {"ArXiv": "demo-its-2"},
        "url": None,
        "openAccessPdf": None,
    },
    {
        "paperId": "demo-its-3",
        "title": "Connected Vehicles and V2X Communication for Intelligent Transportation",
        "abstract": "This work reviews connected vehicle technologies and V2X communication in intelligent transportation systems. It discusses vehicle-infrastructure cooperation, safety applications, traffic efficiency, and deployment challenges.",
        "year": 2024,
        "authors": [{"name": "Demo ITS Author D"}, {"name": "Demo ITS Author E"}],
        "externalIds": {"ArXiv": "demo-its-3"},
        "url": None,
        "openAccessPdf": None,
    },
]


class ArxivRateLimitError(RuntimeError):
    pass


def _tokenize(text: str) -> List[str]:
    return [token for token in tokenize_its_text(text) if re.fullmatch(r"[A-Za-z0-9]+", token)]


def _calculate_relevance_score(paper: dict, user_query: str) -> float:
    query_tokens = set(_tokenize(" ".join(expand_query_with_domain_terms(user_query))))
    if not query_tokens:
        return 0.0

    abstract_tokens = set(_tokenize(paper.get("abstract", "")))
    abstract_overlap = len(query_tokens & abstract_tokens) / len(query_tokens)
    return min(1.0, abstract_overlap)


def _calculate_domain_score(paper: dict) -> float:
    searchable_text = " ".join(
        [
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("keywords", [])),
        ]
    )
    tags = classify_its_topics(searchable_text)
    matched_terms = matched_domain_terms(searchable_text)
    tokens = set(tokenize_its_text(searchable_text))
    if not tokens and not matched_terms:
        return 0.0

    generic_hits = len(tokens & {"traffic", "transportation", "vehicle", "mobility", "road", "signal"})
    return min(1.0, len(matched_terms) / 5 + len(tags) * 0.18 + generic_hits * 0.05)


def _calculate_recency_score(paper: dict) -> float:
    year = paper.get("year")
    if not isinstance(year, int):
        return 0.0
    return min(1.0, max(0, year - 2018) / 8)


def calculate_custom_score(paper: dict, user_query: str) -> dict:
    relevance_score = _calculate_relevance_score(paper, user_query)
    domain_score = _calculate_domain_score(paper)
    recency_score = _calculate_recency_score(paper)
    remin_score = (
        relevance_score * 0.55
        + domain_score * 0.35
        + recency_score * 0.1
    )

    return {
        "similarity_score": round(relevance_score, 4),
        "domain_score": round(domain_score, 4),
        "recency_score": round(recency_score, 4),
        "remin_score": round(remin_score, 4),
    }


def _normalize_paper(paper: dict, user_query: str) -> dict:
    searchable_text = " ".join(
        [paper.get("title", ""), paper.get("abstract", ""), " ".join(paper.get("keywords", []))]
    )
    normalized = {
        "paperId": paper.get("paperId") or paper.get("id") or paper.get("title"),
        "title": paper.get("title", ""),
        "abstract": (paper.get("abstract") or "暂无摘要").strip(),
        "year": paper.get("year"),
        "authors": paper.get("authors", []),
        "url": paper.get("url"),
        "openAccessPdf": paper.get("openAccessPdf"),
        "externalIds": paper.get("externalIds", {}),
        "keywords": paper.get("keywords", []),
        "domain_tags": classify_its_topics(searchable_text),
        "matched_domain_terms": matched_domain_terms(searchable_text),
    }
    normalized.update(calculate_custom_score(normalized, user_query))
    return normalized


def _build_arxiv_query(keyword: str) -> str:
    expansion_terms = expand_query_with_domain_terms(keyword, limit=6)
    field_terms = [f'all:"{term}"' for term in expansion_terms]
    if not field_terms:
        field_terms = [f'all:"{keyword}"']
    joined_terms = " OR ".join(field_terms[:4])
    return f"({joined_terms}) AND (all:\"traffic\" OR all:\"transportation\" OR all:\"vehicle\" OR all:\"vehicular\" OR all:\"V2X\" OR all:\"autonomous driving\" OR all:\"traffic signal\")"


def _build_arxiv_query_candidates(keyword: str) -> List[str]:
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    expansion_terms = expand_query_with_domain_terms(keyword, limit=6)
    token_query = " AND ".join(f'all:"{token}"' for token in _tokenize(keyword))
    expansion_query = " OR ".join(f'all:"{term}"' for term in expansion_terms if term)
    candidates = [
        _build_arxiv_query(keyword),
        f'all:"{keyword}"',
    ]
    if expansion_query:
        candidates.append(f"({expansion_query}) AND (all:traffic OR all:transportation OR all:vehicle)")
    if token_query:
        candidates.append(
            f"({token_query}) AND (all:traffic OR all:transportation OR all:vehicle)"
        )

    # Keep order while removing duplicates. arXiv can be slow, so we try simpler
    # searches before giving up and falling back to demo data.
    return list(dict.fromkeys(candidates))


def _request_with_retries(url: str, params: dict) -> requests.Response:
    global ARXIV_RATE_LIMITED_UNTIL
    last_error: Exception | None = None

    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "ReminITSRAG/1.0"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                ARXIV_RATE_LIMITED_UNTIL = time.time() + ARXIV_RATE_LIMIT_COOLDOWN_SECONDS
                retry_after = response.headers.get("Retry-After")
                message = "arXiv 接口请求过于频繁，已被临时限流"
                if retry_after:
                    message += f"，建议 {retry_after} 秒后重试"
                raise ArxivRateLimitError(message)
            response.raise_for_status()
            return response
        except ArxivRateLimitError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < REQUEST_MAX_RETRIES - 1:
                time.sleep(REQUEST_BACKOFF_SECONDS * (2**attempt))
                continue
            raise

    raise RuntimeError(f"外部学术接口请求失败: {last_error}")


def _request_arxiv(params: dict) -> str:
    return _request_with_retries(ARXIV_QUERY_URL, params).text


def _search_arxiv(keyword: str, limit: int) -> List[dict]:
    last_error: Exception | None = None
    response_text = ""
    papers: List[dict] = []
    for search_query in _build_arxiv_query_candidates(keyword):
        try:
            response_text = _request_arxiv(
                {
                    "search_query": search_query,
                    "start": 0,
                    "max_results": max(limit * 2, limit),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }
            )
            papers = _parse_arxiv_response(response_text, keyword)
            if papers:
                break
            print(f"arXiv 查询无结果，准备尝试备用查询: {search_query}")
        except ArxivRateLimitError:
            raise
        except Exception as exc:
            last_error = exc
            print(f"arXiv 查询失败，准备尝试备用查询: {search_query} | {exc}")

    if not response_text:
        raise RuntimeError(f"arXiv 所有查询均失败: {last_error}")

    return papers


def _parse_arxiv_response(response_text: str, keyword: str) -> List[dict]:
    root = ET.fromstring(response_text)
    papers: List[dict] = []
    for entry in root.findall("atom:entry", ARXIV_NAMESPACES):
        entry_id = (
            entry.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACES) or ""
        ).strip()
        title = re.sub(
            r"\s+",
            " ",
            entry.findtext("atom:title", default="", namespaces=ARXIV_NAMESPACES),
        ).strip()
        abstract = re.sub(
            r"\s+",
            " ",
            entry.findtext("atom:summary", default="", namespaces=ARXIV_NAMESPACES),
        ).strip()
        published = entry.findtext(
            "atom:published", default="", namespaces=ARXIV_NAMESPACES
        )
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (author.findtext("atom:name", default="", namespaces=ARXIV_NAMESPACES) or "").strip()
            for author in entry.findall("atom:author", ARXIV_NAMESPACES)
        ]
        arxiv_id = entry_id.rsplit("/", 1)[-1]

        papers.append(
            _normalize_paper(
                {
                    "paperId": f"arxiv:{arxiv_id}",
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                    "externalIds": {"ArXiv": arxiv_id},
                    "openAccessPdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "url": entry_id,
                },
                keyword,
            )
        )

    return papers


def _fetch_arxiv_papers_by_ids(paper_ids: List[str]) -> List[dict]:
    arxiv_ids = [
        paper_id.removeprefix("arxiv:")
        for paper_id in paper_ids
        if paper_id.startswith("arxiv:")
    ]
    if not arxiv_ids:
        return []

    response_text = _request_arxiv(
        {
            "id_list": ",".join(arxiv_ids),
            "max_results": len(arxiv_ids),
        }
    )
    return _parse_arxiv_response(response_text, " ".join(arxiv_ids))


def fetch_and_sort_papers(keyword: str, limit: int = 20) -> tuple[List[dict], str]:
    cache_key = f"{keyword.strip().lower()}::{limit}"
    cached = SEARCH_RESULT_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < SEARCH_RESULT_CACHE_TTL_SECONDS:
        cached_papers, cached_source = cached[1], cached[2]
        for paper in cached_papers:
            SEARCH_CACHE[paper["paperId"]] = paper
        return cached_papers, cached_source

    if time.time() < ARXIV_RATE_LIMITED_UNTIL:
        remaining_seconds = int(ARXIV_RATE_LIMITED_UNTIL - time.time())
        print(f"arXiv 仍处于限流冷却期，{remaining_seconds} 秒内直接使用离线数据")
        papers = []
        result_source = "fallback"
    else:
        try:
            papers = _search_arxiv(keyword, limit)
            result_source = "arxiv"
        except ArxivRateLimitError as arxiv_exc:
            print(f"获取 arXiv 数据失败: {arxiv_exc}")
            papers = []
            result_source = "fallback"
        except Exception as arxiv_exc:
            print(f"获取 arXiv 数据失败: {arxiv_exc}")
            papers = []
            result_source = "fallback"

    if not papers:
        papers = [
            _normalize_paper(
                {
                    **paper,
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                },
                keyword,
            )
            for paper in FALLBACK_PAPERS[:limit]
        ]
        result_source = "fallback"

    sorted_papers = sorted(
        papers,
        key=lambda item: (
            item["remin_score"],
            item.get("domain_score", 0),
            item["similarity_score"],
            item.get("year") or 0,
        ),
        reverse=True,
    )[:limit]

    for paper in sorted_papers:
        paper["domain_hint"] = build_domain_hint(keyword)
        SEARCH_CACHE[paper["paperId"]] = paper

    SEARCH_RESULT_CACHE[cache_key] = (time.time(), sorted_papers, result_source)
    return sorted_papers, result_source


def get_cached_papers(paper_ids: List[str]) -> List[dict]:
    papers = [SEARCH_CACHE[paper_id] for paper_id in paper_ids if paper_id in SEARCH_CACHE]
    found_ids = {paper["paperId"] for paper in papers}
    missing_ids = [paper_id for paper_id in paper_ids if paper_id not in found_ids]

    fallback_by_id = {
        paper["paperId"]: _normalize_paper(
            {
                **paper,
                "authors": [author.get("name", "") for author in paper.get("authors", [])],
            },
            paper["title"],
        )
        for paper in FALLBACK_PAPERS
    }
    for paper_id in missing_ids:
        if paper_id in fallback_by_id:
            paper = fallback_by_id[paper_id]
            SEARCH_CACHE[paper_id] = paper
            papers.append(paper)

    missing_arxiv_ids = [
        paper_id
        for paper_id in paper_ids
        if paper_id not in {paper["paperId"] for paper in papers}
        and paper_id.startswith("arxiv:")
    ]
    if missing_arxiv_ids:
        try:
            fetched_papers = _fetch_arxiv_papers_by_ids(missing_arxiv_ids)
            for paper in fetched_papers:
                SEARCH_CACHE[paper["paperId"]] = paper
            papers.extend(fetched_papers)
        except Exception as exc:
            print(f"按 paperId 恢复 arXiv 论文失败: {exc}")

    paper_by_id = {paper["paperId"]: paper for paper in papers}
    return [paper_by_id[paper_id] for paper_id in paper_ids if paper_id in paper_by_id]
