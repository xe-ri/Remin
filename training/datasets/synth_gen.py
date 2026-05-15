import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INPUT = Path("training/datasets/source_papers.jsonl")
DEFAULT_OUTPUT = Path("training/datasets/train.jsonl")

SYSTEM_PROMPT = (
    "你是 Remin 学术助手。回答必须基于给定文献证据，风格正式、审慎，"
    "不得编造实验细节、数据或结论；若证据不足，必须明确写出“证据不足”。"
)


def load_source_papers(input_path: Path) -> List[Dict]:
    papers = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def normalize_keywords(paper: Dict) -> List[str]:
    keywords = paper.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    return keywords[:8]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part.strip()]


def clean_clause(text: str, limit: int = 180) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip(" ,;，；") + "…"


def infer_topic_phrase(paper: Dict) -> str:
    topic = (paper.get("topic") or "").strip()
    if topic:
        return topic

    keywords = normalize_keywords(paper)
    if keywords:
        preferred = [item for item in keywords if len(item) <= 40 and "computer science" not in item.lower()]
        if preferred:
            return preferred[0]

    title = (paper.get("title") or "该研究").strip()
    return title


def get_evidence_lines(paper: Dict, limit: int = 2) -> List[str]:
    chunks = paper.get("evidence_chunks") or []
    if isinstance(chunks, str):
        chunks = [chunks]

    evidence = [clean_clause(chunk, 160) for chunk in chunks if chunk and str(chunk).strip()]
    if evidence:
        return evidence[:limit]

    abstract = (paper.get("abstract") or "").strip()
    sentences = [clean_clause(sentence, 160) for sentence in split_sentences(abstract)]
    return sentences[:limit] if sentences else [clean_clause(abstract, 160)]


def infer_method_summary(paper: Dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    for sentence in split_sentences(abstract):
        if re.search(r"\b(propose|present|develop|design|introduce|use|based on|framework|model)\b", sentence, re.I):
            return clean_clause(sentence, 180)

    topic_phrase = infer_topic_phrase(paper)
    return f"该研究围绕 {topic_phrase} 构建方法框架，但仅凭当前摘要仍难以还原完整技术细节。"


def infer_contribution_summary(paper: Dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    for sentence in split_sentences(abstract):
        if re.search(r"\b(show|demonstrate|improve|outperform|effective|accuracy|efficient|survey|review)\b", sentence, re.I):
            return clean_clause(sentence, 180)

    topic_phrase = infer_topic_phrase(paper)
    return f"从当前摘要看，该研究的贡献主要体现在围绕 {topic_phrase} 提供了系统化分析、方法改进或应用验证。"


def infer_scope_summary(paper: Dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    topic_phrase = infer_topic_phrase(paper)
    if not abstract:
        return f"该文献主要讨论 {topic_phrase} 相关问题。"

    first_sentence = split_sentences(abstract)
    if first_sentence:
        return clean_clause(first_sentence[0], 180)

    return f"该文献主要讨论 {topic_phrase} 相关问题。"


def build_context(paper: Dict) -> str:
    title = paper.get("title", "未知标题")
    abstract = paper.get("abstract", "").strip() or "暂无摘要"
    authors = paper.get("authors") or []
    if authors and isinstance(authors[0], dict):
        authors = [author.get("name", "") for author in authors]
    authors_text = ", ".join([author for author in authors if author]) or "未知作者"
    keyword_text = ", ".join(normalize_keywords(paper)) or "未提供关键词"
    year = paper.get("year", "未知年份")
    citation_count = paper.get("citationCount", 0)

    return "\n".join(
        [
            f"标题: {title}",
            f"作者: {authors_text}",
            f"年份: {year}",
            f"关键词: {keyword_text}",
            f"引用量: {citation_count}",
            f"摘要: {abstract}",
        ]
    )


def format_sample(question: str, context: str, response: str) -> Dict[str, str]:
    # 用接近 Qwen 指令跟随的聊天格式，尽量贴近线上 RAG 提示词。
    combined_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n文献证据：\n{context}\n\n用户问题：\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return {"text": combined_text}


def format_answer(conclusion: str, evidence: str, limitation: str) -> str:
    return (
        "核心结论：\n"
        f"{conclusion}\n\n"
        "依据说明：\n"
        f"{evidence}\n\n"
        "局限性：\n"
        f"{limitation}"
    )


def build_task_samples(paper: Dict) -> Iterable[Dict[str, str]]:
    context = build_context(paper)
    abstract = paper.get("abstract", "").strip() or "暂无摘要"
    title = paper.get("title", "该研究")
    topic_phrase = infer_topic_phrase(paper)
    evidence_lines = get_evidence_lines(paper)
    evidence_text = "\n".join(
        f"{line} [证据{index}]"
        for index, line in enumerate(evidence_lines, start=1)
    )
    method_summary = infer_method_summary(paper)
    contribution_summary = infer_contribution_summary(paper)
    scope_summary = infer_scope_summary(paper)

    responses = [
        (
            "请概括该文献主要研究什么内容。",
            format_answer(
                conclusion=(
                    f"该文献主要围绕 {topic_phrase} 展开，重点讨论相关任务的研究目标、应用场景与关键挑战。"
                ),
                evidence=evidence_text,
                limitation="当前判断仅依据摘要，尚不能替代对全文引言和结论部分的正式核对。",
            ),
        ),
        (
            "请总结该研究的方法路径，并保持学术表达。",
            format_answer(
                conclusion=f"从当前证据看，该研究的方法路径主要体现在：{method_summary}",
                evidence=evidence_text,
                limitation="摘要通常不会完整提供参数设置、训练细节或全部实验流程，因此方法总结只能视为摘要层面的概括。",
            ),
        ),
        (
            "请概括该文献可能的学术贡献。",
            format_answer(
                conclusion=(
                    f"基于当前摘要，本文的学术贡献主要可概括为：{contribution_summary}"
                ),
                evidence=evidence_text,
                limitation="贡献归纳目前仅依据摘要文本，不能替代对实验结果、对比基线和消融研究的正式验证。",
            ),
        ),
        (
            "请指出该研究的局限性。如果证据不足，请明确说明。",
            format_answer(
                conclusion="就当前证据而言，能够直接确认的局限性信息有限，许多结论仍需保留审慎态度。",
                evidence="如果摘要未直接说明样本规模、外部有效性、鲁棒性或对比基线，则只能判断为“证据不足”。",
                limitation="当前最主要的限制就是证据不足，正式的局限性分析需要结合全文讨论部分。",
            ),
        ),
        (
            "如果用户追问摘要之外的实验设置、数据集参数或显著性结果，应如何回答？",
            format_answer(
                conclusion="在仅有标题、关键词和摘要的情况下，不应直接确认摘要之外的实验细节或定量结论。",
                evidence="现有证据只覆盖摘要层信息，未提供完整实验设置、数据集划分或统计显著性分析。",
                limitation="更严谨的处理方式是明确说明证据不足，并建议进一步查阅论文全文的方法、实验与附录部分。",
            ),
        ),
        (
            "请将该文献整理成综述式回答。",
            format_answer(
                conclusion=(
                    f"从综述角度看，该文献聚焦于 {topic_phrase}，其核心内容可概括为：{scope_summary}"
                ),
                evidence=evidence_text,
                limitation="由于当前仅使用摘要，综述结论仍属于摘要级归纳，不能替代基于全文的系统综述。",
            ),
        ),
    ]

    for question, response in responses:
        yield format_sample(question, context, response)


def build_pairwise_samples(papers: List[Dict]) -> Iterable[Dict[str, str]]:
    if len(papers) < 2:
        return []

    samples = []
    shuffled = papers[:]
    random.shuffle(shuffled)

    for first, second in zip(shuffled[::2], shuffled[1::2]):
        first_context = build_context(first)
        second_context = build_context(second)
        first_topic = infer_topic_phrase(first)
        second_topic = infer_topic_phrase(second)
        first_method = infer_method_summary(first)
        second_method = infer_method_summary(second)
        first_contribution = infer_contribution_summary(first)
        second_contribution = infer_contribution_summary(second)
        response = format_answer(
            conclusion=(
                f"两篇文献都属于智能交通相关研究，但关注点并不完全相同："
                f"文献一更偏向 {first_topic}，文献二更偏向 {second_topic}。"
            ),
            evidence=(
                f"文献一的方法线索为：{first_method} [证据1]\n"
                f"文献一的贡献线索为：{first_contribution} [证据2]\n"
                f"文献二的方法线索为：{second_method} [证据3]\n"
                f"文献二的贡献线索为：{second_contribution} [证据4]"
            ),
            limitation="基于摘要可以做初步对比，但还不能替代对全文方法、实验设计和结论部分的系统比较。",
        )
        samples.append(
            format_sample(
                "请比较两篇文献的研究主题、方法和可能贡献，若信息不足请明确指出。",
                f"文献一:\n{first_context}\n\n文献二:\n{second_context}",
                response,
            )
        )

    return samples


def generate_synthetic_dataset(input_path: Path, output_path: Path, seed: int = 42) -> int:
    random.seed(seed)
    papers = load_source_papers(input_path)

    dataset: List[Dict[str, str]] = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        dataset.extend(build_task_samples(paper))

    dataset.extend(build_pairwise_samples(papers))
    random.shuffle(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for entry in dataset:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Generate Remin SFT dataset from academic metadata.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL with paper metadata.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL for SFT training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"未找到源数据文件: {args.input}。请先准备 source_papers.jsonl，"
            "每行至少包含 title、abstract，可选 authors/year/keywords/citationCount。"
        )

    sample_count = generate_synthetic_dataset(args.input, args.output, args.seed)
    print(f"已生成 {sample_count} 条微调样本 -> {args.output}")


if __name__ == "__main__":
    main()
