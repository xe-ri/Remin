from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


def _normalize_pdf_url(pdf_url: str) -> str:
    if not pdf_url:
        return pdf_url

    parsed = urlparse(pdf_url)
    if "arxiv.org" in parsed.netloc and "/abs/" in parsed.path:
        arxiv_id = parsed.path.rsplit("/", 1)[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return pdf_url


def resolve_pdf_url(paper: dict) -> Optional[str]:
    open_access_pdf = paper.get("openAccessPdf")
    if isinstance(open_access_pdf, dict):
        pdf_url = (open_access_pdf.get("url") or "").strip()
        if pdf_url:
            return _normalize_pdf_url(pdf_url)

    if isinstance(open_access_pdf, str) and open_access_pdf.strip():
        return _normalize_pdf_url(open_access_pdf.strip())

    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return None


def download_pdf(pdf_url: str, save_path: Path, timeout: int = 30) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()
    save_path.write_bytes(response.content)
    return save_path
