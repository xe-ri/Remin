import argparse
import json
import mimetypes
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark /search, /upload and /chat endpoints for thesis chapter 6."
    )
    parser.add_argument(
        "--config",
        default="evaluation/cases/api_cases.sample.json",
        help="JSON config file for API benchmark cases.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI service.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/api_results.json",
        help="Output JSON file for API benchmark results.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("API benchmark config must be a JSON object.")
    return data


def timed_request(session: requests.Session, method: str, url: str, **kwargs):
    start = time.perf_counter()
    response = session.request(method, url, timeout=120, **kwargs)
    duration = time.perf_counter() - start
    response.raise_for_status()
    return response, duration


def summarize_durations(durations: List[float]) -> Dict[str, float]:
    if not durations:
        return {"count": 0, "avg_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0}
    return {
        "count": len(durations),
        "avg_seconds": round(mean(durations), 4),
        "min_seconds": round(min(durations), 4),
        "max_seconds": round(max(durations), 4),
    }


def main():
    args = parse_args()
    config = load_config(Path(args.config))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    search_runs = []
    upload_runs = []
    chat_runs = []
    skipped_runs = []

    for case in config.get("search_cases", []):
        payload = {
            "keyword": case["keyword"],
            "limit": int(case.get("limit", 8)),
        }
        response, duration = timed_request(
            session,
            "POST",
            f"{args.base_url}/search",
            json=payload,
        )
        body = response.json()
        search_runs.append(
            {
                "keyword": payload["keyword"],
                "limit": payload["limit"],
                "duration_seconds": round(duration, 4),
                "result_count": len(body.get("results", [])),
                "result_source": body.get("result_source"),
            }
        )

    uploaded_source_ids = []
    for file_path_str in config.get("upload_files", []):
        file_path = Path(file_path_str)
        if not file_path.exists():
            skipped_runs.append(
                {
                    "type": "upload",
                    "target": str(file_path),
                    "reason": "file_not_found",
                }
            )
            continue
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/pdf"
        with file_path.open("rb") as handle:
            files = {"file": (file_path.name, handle, mime_type)}
            data = {"title": file_path.stem}
            response, duration = timed_request(
                session,
                "POST",
                f"{args.base_url}/upload",
                files=files,
                data=data,
            )
        body = response.json()
        uploaded_source_ids.append(body["source_id"])
        upload_runs.append(
            {
                "file": str(file_path),
                "duration_seconds": round(duration, 4),
                "source_id": body.get("source_id"),
                "chunk_count": body.get("chunk_count"),
            }
        )

    for case in config.get("chat_cases", []):
        paper_ids = []
        if case.get("keyword"):
            search_payload = {
                "keyword": case["keyword"],
                "limit": int(case.get("limit", 8)),
            }
            response, _ = timed_request(
                session,
                "POST",
                f"{args.base_url}/search",
                json=search_payload,
            )
            search_results = response.json().get("results", [])
            for index in case.get("paper_indexes", [1]):
                position = int(index) - 1
                if 0 <= position < len(search_results):
                    paper_ids.append(search_results[position]["paperId"])

        selected_uploaded_ids = uploaded_source_ids if case.get("use_uploaded_sources") else []
        if case.get("use_uploaded_sources") and not selected_uploaded_ids:
            skipped_runs.append(
                {
                    "type": "chat",
                    "target": case["query"],
                    "reason": "no_uploaded_sources",
                }
            )
            continue
        chat_payload = {
            "query": case["query"],
            "paper_ids": paper_ids,
            "uploaded_source_ids": selected_uploaded_ids,
            "mode": case.get("mode", "rag"),
        }
        response, duration = timed_request(
            session,
            "POST",
            f"{args.base_url}/chat",
            json=chat_payload,
        )
        body = response.json()
        chat_runs.append(
            {
                "query": case["query"],
                "duration_seconds": round(duration, 4),
                "source_count": len(body.get("sources", [])),
                "answer_length": len(body.get("answer", "")),
            }
        )

    payload = {
        "summary": {
            "search": summarize_durations([item["duration_seconds"] for item in search_runs]),
            "upload": summarize_durations([item["duration_seconds"] for item in upload_runs]),
            "chat": summarize_durations([item["duration_seconds"] for item in chat_runs]),
        },
        "search_runs": search_runs,
        "upload_runs": upload_runs,
        "chat_runs": chat_runs,
        "skipped_runs": skipped_runs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"API benchmark finished. Results saved to: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
