import re

import fitz  # PyMuPDF


def _is_layout_noise_line(line: str) -> bool:
    """Filter chart axes, page numbers, and other low-semantic PDF artifacts."""
    normalized = re.sub(r"\s+", " ", line or "").strip()
    if not normalized:
        return True

    if re.fullmatch(r"[\d\s.,:;()\-+]+", normalized):
        return True

    tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    if not tokens:
        return True

    numeric_count = sum(token.isdigit() for token in tokens)
    alpha_count = sum(any(char.isalpha() for char in token) for token in tokens)
    numeric_ratio = numeric_count / max(len(tokens), 1)
    lower_line = normalized.lower()

    axis_terms = (
        "time of a day",
        "traffic flow volume",
        "ground truth",
        "x-axis",
        "y-axis",
    )
    if numeric_count >= 4 and any(term in lower_line for term in axis_terms):
        return True
    if numeric_count >= 6 and numeric_ratio >= 0.35:
        return True
    if numeric_count >= 3 and alpha_count <= 3 and len(normalized) < 120:
        return True

    return False


def clean_pdf_text(pdf_path):
    """Parse a PDF and remove obvious layout noise before text chunking."""
    doc = fitz.open(pdf_path)
    cleaned_lines = []

    for page in doc:
        page_text = page.get_text("text")
        for line in page_text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not _is_layout_noise_line(line):
                cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"-\n(?=[A-Za-z])", "", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()
