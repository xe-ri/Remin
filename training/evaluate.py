import re
from typing import Dict


def evaluate_academic_answer(model_output: str, context: str) -> Dict[str, object]:
    """
    轻量评估：
    1. 是否包含结构化回答段落
    2. 是否出现证据不足提示
    3. 是否存在明显脱离上下文的高风险术语
    """
    required_sections = ["核心结论", "依据说明", "局限性"]
    missing_sections = [section for section in required_sections if section not in model_output]

    context_lower = context.lower()
    hallucination_terms = ["significant improvement", "state-of-the-art", "p-value", "benchmark score"]
    unsupported_terms = [
        term for term in hallucination_terms
        if term in model_output.lower() and term not in context_lower
    ]

    citation_like = len(re.findall(r"\[证据\d+\]", model_output))
    evidence_awareness = "证据不足" in model_output or citation_like > 0

    score = 100
    score -= len(missing_sections) * 20
    score -= len(unsupported_terms) * 15
    if not evidence_awareness:
        score -= 15
    score = max(score, 0)

    return {
        "score": score,
        "missing_sections": missing_sections,
        "unsupported_terms": unsupported_terms,
        "citation_count": citation_like,
        "evidence_awareness": evidence_awareness,
        "label": "较严谨" if score >= 75 else "需改进",
    }
