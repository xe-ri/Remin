import json
import os
from functools import lru_cache
import importlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from app.config import SETTINGS
from app.database import query_vector_db
from app.its_domain import build_domain_hint, classify_its_topics, expand_query_with_domain_terms

MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/remin_adapter"))
BASE_MODEL_ENV_KEYS = ("LOCAL_BASE_MODEL_PATH", "BASE_MODEL_PATH")
DIRECT_LLM_ENV_KEYS = ("DIRECT_LLM_MODEL_PATH", "LOCAL_BASE_MODEL_PATH", "BASE_MODEL_PATH")
METHOD_PATTERNS = [
    (
        re.compile(r"\b(large language model|llm|prompt|language model|foundation model)\b", re.I),
        "大语言模型与提示工程",
        "利用语言模型的语义理解、推理或提示构造能力，将交通流预测问题转化为可解释的文本/时序建模任务。",
    ),
    (
        re.compile(r"\b(graph neural network|graph convolution|gcn|tgcn|spatio[- ]temporal graph|road network|topolog)\b", re.I),
        "图神经网络/图卷积时空建模",
        "把道路路网表示为图结构，同时建模路段之间的空间依赖和交通状态随时间变化的动态关系。",
    ),
    (
        re.compile(r"\b(transformer|attention|self-attention|temporal attention)\b", re.I),
        "Transformer 或注意力机制",
        "通过注意力机制捕捉不同时间片、路段或特征之间的重要依赖关系，用于提升长期或复杂交通模式的表达能力。",
    ),
    (
        re.compile(r"\b(lstm|gru|recurrent|rnn|temporal convolution|tcn)\b", re.I),
        "循环网络/时序神经网络",
        "重点建模交通流的时间连续性和历史观测对未来状态的影响。",
    ),
    (
        re.compile(r"\b(tensor|fusion|multi-segment|latent factor|matrix factorization|multi[- ]view)\b", re.I),
        "多源融合、张量或潜因子建模",
        "融合多路段、多特征或多视角交通信息，并用张量分解、潜因子或融合结构提取隐含交通模式。",
    ),
    (
        re.compile(r"\b(reinforcement learning|q-learning|policy|agent|markov)\b", re.I),
        "强化学习",
        "将交通控制或决策建模为交互式优化问题，通过奖励函数学习信号控制、路径选择或系统调度策略。",
    ),
    (
        re.compile(r"\b(bayesian|probabilistic|uncertainty|gaussian)\b", re.I),
        "概率建模与不确定性估计",
        "在预测过程中显式考虑随机性和不确定性，使结果更适合风险评估或可靠性分析。",
    ),
]
REVIEW_INTENTS = ("综述", "总结", "概括", "发展现状", "研究现状", "review", "survey", "summary", "state of")
THEME_PATTERNS = [
    (
        re.compile(r"\b(v2x|vehicle-to-everything|vehicular|connected vehicle|internet of vehicles|vanet|communication)\b", re.I),
        "车联网与 V2X 协同通信",
        "文献主要围绕车辆、道路基础设施和交通系统之间的信息交互展开，说明 V2X 已成为智能交通协同感知与协同控制的重要支撑。",
    ),
    (
        re.compile(r"\b(performance evaluation|simulation|scenario|sumo|large[- ]scale|traffic data|evaluation)\b", re.I),
        "仿真验证与性能评估",
        "部分研究侧重通过交通场景、仿真平台或大规模交通数据评估通信方案在真实交通环境中的适用性。",
    ),
    (
        re.compile(r"\b(survey|taxonomy|review|classification|overview)\b", re.I),
        "综述分类与体系化梳理",
        "部分文献以综述或分类框架为主，尝试对 V2X、车联网和智能网联车辆通信技术进行系统整理。",
    ),
    (
        re.compile(r"\b(5g|cellular|dsrc|c-v2x|lte|d2d|network)\b", re.I),
        "通信网络演进",
        "相关研究关注 DSRC、蜂窝网络、5G 或 C-V2X 等通信技术路线对车联网发展的影响。",
    ),
    (
        re.compile(r"\b(security|attack|intrusion|privacy|reliability|safety)\b", re.I),
        "安全性与可靠性问题",
        "部分研究将通信安全、入侵检测、可靠性或道路安全作为智能交通通信系统的重要问题。",
    ),
    (
        re.compile(r"\b(traffic signal|traffic control|traffic flow|congestion|intersection)\b", re.I),
        "交通运行与控制应用",
        "文献将通信能力与交通流预测、信号控制、拥堵缓解或交叉口协同等具体交通任务结合起来。",
    ),
]
ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "to", "of", "for", "in", "on", "at", "by", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "their", "his", "her", "they", "them",
    "we", "our", "you", "your", "he", "she", "not", "can", "could", "may", "might", "will",
    "would", "should", "do", "does", "did", "have", "has", "had", "using", "used", "use",
    "based", "through", "into", "about", "over", "under", "between", "within", "across",
    "paper", "study", "work", "result", "results", "approach", "method", "methods",
    "introduction", "conclusion", "analysis", "system", "systems",
    "traffic", "flow", "volume", "time", "day", "ground", "truth", "xtp",
    "llm", "llms",
}
CHINESE_STOPWORDS = {
    "的", "了", "和", "与", "及", "并", "并且", "或", "而", "但", "如果", "因此", "所以", "以及",
    "对于", "通过", "进行", "主要", "相关", "研究", "文献", "方法", "结果", "分析", "系统", "问题",
    "我们", "他们", "可以", "能够", "一种", "这个", "这些", "该", "其中", "由于", "基于", "围绕",
    "主要表明", "摘要", "标题", "作者", "年份", "引用量"
}


@dataclass
class AnswerBundle:
    answer: str
    answer_mode: str
    retrieval_count: int = 0
    evidence_topics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    used_fallback: bool = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", (text or "").lower())


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text or "")
    return [
        part.strip()
        for part in parts
        if part and len(part.strip()) > 20 and not _looks_like_layout_noise(part)
    ]


def _looks_like_layout_noise(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return True

    tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    if not tokens:
        return True

    numeric_count = sum(token.isdigit() for token in tokens)
    alpha_count = sum(any(char.isalpha() for char in token) for token in tokens)
    numeric_ratio = numeric_count / max(len(tokens), 1)
    lower_text = normalized.lower()
    axis_terms = (
        "time of a day",
        "traffic flow volume",
        "ground truth",
        "x-axis",
        "y-axis",
    )

    if numeric_count >= 6 and any(term in lower_text for term in axis_terms):
        return True
    if numeric_count >= 10 and numeric_ratio >= 0.25:
        return True
    if numeric_count >= 5 and alpha_count <= 6 and len(normalized) < 180:
        return True
    if re.fullmatch(r"[\d\s.,:;()\-+A-Za-z]*", normalized) and numeric_count >= 8:
        return True

    return False


def _filter_retrieved_context(
    documents: List[str],
    metadatas: List[dict],
) -> Tuple[List[str], List[dict]]:
    filtered_documents = []
    filtered_metadatas = []

    for document, metadata in zip(documents, metadatas):
        if _looks_like_layout_noise(document):
            continue
        sentences = _split_sentences(document)
        if not sentences:
            continue
        filtered_documents.append(" ".join(sentences[:6]))
        filtered_metadatas.append(metadata)

    return filtered_documents, filtered_metadatas


def _clean_for_analysis(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"(标题|作者|年份|引用量|摘要)\s*[:：]", "", text)
    text = re.sub(r"\b(?:\d+\s+){5,}\d+\b", "", text)
    return text


def _extract_keywords(text: str, limit: int = 4) -> List[str]:
    tokens = [
        token for token in _tokenize(text)
        if len(token) > 1
        and token not in ENGLISH_STOPWORDS
        and token not in CHINESE_STOPWORDS
        and not token.isdigit()
    ]
    counter = Counter(tokens)
    return [token for token, _ in counter.most_common(limit)]


def _humanize_keyword(token: str) -> str:
    mapping = {
        "llm": "大语言模型",
        "llms": "大语言模型",
        "prompt": "提示工程",
        "language": "语言模型",
        "models": "模型",
        "model": "模型",
        "graph": "图结构",
        "convolutional": "图卷积",
        "tensor": "张量建模",
        "fusion": "融合建模",
        "latent": "潜因子",
        "factor": "潜因子",
        "prediction": "预测",
        "forecasting": "预测",
        "vehicle": "车辆",
        "vehicular": "车联网",
        "communication": "通信",
        "v2x": "V2X",
    }
    return mapping.get(token.lower(), token)


def _infer_methods(title: str, text: str) -> List[Tuple[str, str]]:
    searchable_text = f"{title} {text}"
    methods = [
        (label, explanation)
        for pattern, label, explanation in METHOD_PATTERNS
        if pattern.search(searchable_text)
    ]
    if methods:
        return methods[:3]

    return [("证据不足", "当前检索片段不足以可靠判断该文献采用的具体方法。")]


def _is_review_query(user_query: str) -> bool:
    normalized = (user_query or "").lower()
    return any(intent in normalized for intent in REVIEW_INTENTS)


def _infer_themes(text: str) -> List[Tuple[str, str]]:
    themes = [
        (label, explanation)
        for pattern, label, explanation in THEME_PATTERNS
        if pattern.search(text)
    ]
    return themes[:4]


def _infer_task_focus(title: str, text: str) -> str:
    searchable_text = f"{title} {text}".lower()
    if "traffic flow" in searchable_text or "traffic volume" in searchable_text:
        return "交通流/交通量预测"
    if "v2x" in searchable_text or "vehicular" in searchable_text or "communication" in searchable_text:
        return "车联网/V2X 通信与交通系统协同"
    if "traffic signal" in searchable_text or "signal control" in searchable_text:
        return "交通信号控制"
    if "trajectory" in searchable_text:
        return "车辆轨迹或出行行为建模"
    return "智能交通系统相关问题"


def _build_evidence_hint(title: str, source: str, text: str, methods: List[Tuple[str, str]]) -> str:
    method_labels = "、".join(label for label, _ in methods)
    keywords = [
        _humanize_keyword(token)
        for token in _extract_keywords(f"{title} {text}", limit=6)
    ]
    keyword_text = "、".join(dict.fromkeys(keywords)) or "未提取到稳定关键词"
    return f"[{title} | {source}] 可支持的方法判断：{method_labels}；证据关键词：{keyword_text}。"


def _build_context_block(documents: List[str], metadatas: List[dict]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)
        blocks.append(f"[证据{index}] 来源={title} | ID={source}\n{document}")
    return "\n\n".join(blocks)


def _build_topic_guidance(user_query: str, metadatas: List[dict]) -> str:
    topic_labels = classify_its_topics(user_query)
    for metadata in metadatas:
        searchable = " ".join(
            [
                str(metadata.get("title", "")),
                str(metadata.get("source", "")),
            ]
        )
        for label in classify_its_topics(searchable):
            if label not in topic_labels:
                topic_labels.append(label)
    if not topic_labels:
        topic_labels = [build_domain_hint(user_query)]
    return "、".join(topic_labels[:4])


def _rank_evidence_sentences(
    user_query: str,
    documents: List[str],
    metadatas: List[dict],
) -> List[Tuple[float, int, str, str, str]]:
    query_tokens = set(_tokenize(user_query))
    ranked_sentences: List[Tuple[float, int, str, str, str]] = []

    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)

        for sentence in _split_sentences(document):
            sentence_tokens = set(_tokenize(sentence))
            overlap = len(query_tokens & sentence_tokens) if query_tokens else 0
            density_bonus = min(len(sentence) / 200, 1)
            score = overlap * 2 + density_bonus
            ranked_sentences.append((score, index, title, source, sentence))

    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    return ranked_sentences


def _group_documents_by_source(documents: List[str], metadatas: List[dict]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for document, metadata in zip(documents, metadatas):
        if _looks_like_layout_noise(document):
            continue
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)
        item = grouped.setdefault(
            source,
            {
                "source": source,
                "title": title,
                "documents": [],
            },
        )
        item["documents"].append(document)
    return list(grouped.values())


def _summarize_source(user_query: str, title: str, source: str, documents: List[str]) -> Tuple[str, str]:
    ranked_sentences = _rank_evidence_sentences(
        user_query,
        documents,
        [{"source": source, "title": title} for _ in documents],
    )
    if not ranked_sentences:
        return (
            f"{title} 与当前问题相关，但现有片段不足以支持更细致分析。",
            f"[{title} | {source}] 未能提取到足够清晰的相关句子。",
        )

    top_sentences = [
        _clean_for_analysis(sentence)
        for _, _, _, _, sentence in ranked_sentences
        if not _looks_like_layout_noise(sentence)
    ][:6]
    if not top_sentences:
        return (
            f"{title} 与当前问题相关，但检索片段主要是图表或版面噪声，证据不足以生成可靠综述。",
            f"[{title} | {source}] 检索片段噪声较高，未采用为实质证据。",
        )

    joined_text = " ".join(top_sentences)
    methods = _infer_methods(title, joined_text)
    task_focus = _infer_task_focus(title, joined_text)
    method_labels = "、".join(label for label, _ in methods)
    method_explanations = "；".join(explanation for _, explanation in methods[:2])

    conclusion = (
        f"《{title}》主要面向{task_focus}，采用的方法可归纳为{method_labels}。"
        f"具体来说，{method_explanations}"
    )
    evidence = _build_evidence_hint(title, source, joined_text, methods)
    return conclusion, evidence


def _compose_direct_answer(
    user_query: str,
    documents: List[str],
    metadatas: List[dict],
) -> Tuple[str, str]:
    grouped_sources = _group_documents_by_source(documents, metadatas)
    if not grouped_sources:
        return (
            "当前检索片段不足以直接回答该问题，现有证据只能支持非常有限的归纳。",
            "未能从检索结果中提取出足够清晰的句子级证据。",
        )

    answer_lines = []
    evidence_lines = []
    method_counter: Counter[str] = Counter()

    for index, item in enumerate(grouped_sources[:3], start=1):
        conclusion, evidence = _summarize_source(
            user_query,
            str(item["title"]),
            str(item["source"]),
            list(item["documents"]),
        )
        for label, _ in _infer_methods(str(item["title"]), " ".join(item["documents"])):
            method_counter[label] += 1
        answer_lines.append(f"{index}. {conclusion} [证据{index}]")
        evidence_lines.append(f"{index}. {evidence}")

    common_methods = "、".join(
        label
        for label, _ in method_counter.most_common(5)
        if label != "证据不足"
    ) or "若干智能交通建模方法"
    overview = (
        f"综合当前选中的文献来看，它们不是在简单讨论若干英文关键词，"
        f"而是主要围绕{common_methods}来解决智能交通中的预测、通信或控制问题。"
        "下面按文献归纳其方法，不直接照搬原文句子。"
    )
    direct_answer = overview + "\n" + "\n".join(answer_lines)
    evidence_summary = "\n".join(evidence_lines)
    return direct_answer, evidence_summary


def _compose_review_answer(
    user_query: str,
    documents: List[str],
    metadatas: List[dict],
) -> Tuple[str, str]:
    ranked_sentences = _rank_evidence_sentences(user_query, documents, metadatas)
    useful_sentences = [
        (index, title, source, _clean_for_analysis(sentence))
        for _, index, title, source, sentence in ranked_sentences
        if not _looks_like_layout_noise(sentence)
    ][:8]
    combined_text = " ".join([sentence for _, _, _, sentence in useful_sentences])
    themes = _infer_themes(" ".join([combined_text] + [str(meta.get("title", "")) for meta in metadatas]))

    if not useful_sentences or not themes:
        return (
            "当前检索到的片段不足以可靠总结该方向的发展现状。现有证据可能只包含标题、关键词或噪声片段，"
            "尚不能支持对研究脉络、技术路线和发展趋势作出稳定归纳。",
            "未能从当前检索结果中提取出足够清晰的句子级证据。",
        )

    candidate_sentences = [
        (candidate_id, evidence_index, title, source, sentence)
        for candidate_id, (evidence_index, title, source, sentence) in enumerate(useful_sentences, start=1)
    ]

    theme_matches: Dict[str, List[int]] = {}
    for theme, _ in themes:
        matched_ids = []
        for candidate_id, _, title, _, sentence in candidate_sentences:
            searchable = f"{title} {sentence}"
            if any(pattern.search(searchable) and label == theme for pattern, label, _ in THEME_PATTERNS):
                matched_ids.append(candidate_id)
        theme_matches[theme] = list(dict.fromkeys(matched_ids))[:2]

    cited_candidate_ids = {
        candidate_id
        for matched_ids in theme_matches.values()
        for candidate_id in matched_ids
    }
    selected_candidates = [
        item for item in candidate_sentences if item[0] in cited_candidate_ids
    ][:6]
    if len(selected_candidates) < 4:
        selected_ids = {item[0] for item in selected_candidates}
        for item in candidate_sentences:
            if item[0] in selected_ids:
                continue
            selected_candidates.append(item)
            selected_ids.add(item[0])
            if len(selected_candidates) >= min(4, len(candidate_sentences)):
                break

    final_evidence_ids = {
        candidate_id: final_id
        for final_id, (candidate_id, _, _, _, _) in enumerate(selected_candidates, start=1)
    }

    theme_lines = []
    for order, (theme, explanation) in enumerate(themes, start=1):
        citation_ids = [
            final_evidence_ids[candidate_id]
            for candidate_id in theme_matches.get(theme, [])
            if candidate_id in final_evidence_ids
        ]
        citation = " ".join(f"[证据{item}]" for item in citation_ids) or "[证据不足]"
        theme_lines.append(f"{order}. {theme}：{explanation}{citation}")

    source_titles = []
    for metadata in metadatas:
        title = str(metadata.get("title", metadata.get("source", "未知文献")))
        if title not in source_titles:
            source_titles.append(title)

    overview = (
        "从当前检索到的文献证据看，相关研究已经从单一通信概念讨论，逐步扩展到车联网协同、"
        "仿真评估、技术体系梳理以及交通应用场景结合等方向。需要注意的是，以下结论只依据当前已检索片段，"
        "不扩展到证据之外的标准、产业部署或实验结果。"
    )
    direct_answer = (
        f"{overview}\n"
        + "\n".join(theme_lines)
        + "\n总体来看，当前证据支持将该方向理解为智能交通系统中连接车辆、道路与交通管理任务的关键技术基础，"
        "但对于统一标准成熟度、真实道路部署规模和定量性能优劣，现有片段仍不足以作出强结论。"
    )

    evidence_lines = []
    for final_id, (_, _, title, source, sentence) in enumerate(selected_candidates, start=1):
        evidence_lines.append(f"[证据{final_id}] {title}（{source}）：{sentence[:180]}")
    if source_titles:
        evidence_lines.append("涉及文献：" + "；".join(source_titles[:5]))

    return direct_answer, "\n".join(evidence_lines)


def _fallback_academic_answer(user_query: str, documents: List[str], metadatas: List[dict]) -> str:
    if _is_review_query(user_query):
        direct_answer, evidence_summary = _compose_review_answer(user_query, documents, metadatas)
    else:
        direct_answer, evidence_summary = _compose_direct_answer(user_query, documents, metadatas)
    return (
        "基于当前检索到的文献证据，以下是更审慎的学术性回答。\n\n"
        f"研究问题：{user_query}\n\n"
        "核心结论：\n"
        f"{direct_answer}\n\n"
        "依据说明：\n"
        f"{evidence_summary}\n\n"
        "局限性：\n"
        "以上回答仅依据当前检索到的片段生成，若问题涉及更完整的方法细节、实验设置或定量结论，仍需进一步核对论文全文。"
    )


def _build_retrieval_query(user_query: str) -> str:
    normalized = (user_query or "").strip().lower()
    expansion_terms = expand_query_with_domain_terms(user_query, limit=8)
    expansion_text = " ".join(expansion_terms)
    if any(intent in normalized for intent in REVIEW_INTENTS):
        return (
            f"{expansion_text} "
            "title abstract introduction proposed method framework contribution "
            "experiment result conclusion intelligent transportation v2x vehicular communication"
        )
    return f"{user_query} {expansion_text}".strip()


def _has_file(path: Path, *names: str) -> bool:
    return any((path / name).exists() for name in names)


def _read_adapter_config(path: Path) -> Dict[str, object]:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        return {}

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_base_model_reference(adapter_dir: Path) -> str:
    for env_key in BASE_MODEL_ENV_KEYS:
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            return env_value

    adapter_config = _read_adapter_config(adapter_dir)
    base_model_name = str(adapter_config.get("base_model_name_or_path", "")).strip()
    if base_model_name:
        return base_model_name

    raise FileNotFoundError(
        "未在 adapter_config.json 中找到基座模型信息，也没有设置 LOCAL_BASE_MODEL_PATH / BASE_MODEL_PATH。"
    )


def _resolve_direct_model_reference() -> str:
    for env_key in DIRECT_LLM_ENV_KEYS:
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            return env_value

    if _is_full_model_directory(MODEL_DIR):
        return str(MODEL_DIR)

    if _is_adapter_directory(MODEL_DIR):
        return _resolve_base_model_reference(MODEL_DIR)

    raise FileNotFoundError(
        "未找到可用于直接 LLM 基线的模型。请设置 DIRECT_LLM_MODEL_PATH，"
        "或设置 LOCAL_BASE_MODEL_PATH / BASE_MODEL_PATH 指向基座模型目录。"
    )


def _is_existing_local_path(reference: str) -> bool:
    try:
        return Path(reference).exists()
    except OSError:
        return False


def _is_adapter_directory(path: Path) -> bool:
    return path.is_dir() and _has_file(path, "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def _is_full_model_directory(path: Path) -> bool:
    return path.is_dir() and _has_file(
        path,
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "generation_config.json",
    )


def _load_plain_generator(model_reference: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    local_model = _is_existing_local_path(model_reference)
    model = AutoModelForCausalLM.from_pretrained(
        model_reference,
        local_files_only=local_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        local_files_only=local_model,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def _get_auto_peft_model_class():
    try:
        peft_module = importlib.import_module("peft")
        return getattr(peft_module, "AutoPeftModelForCausalLM", None)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_generator():
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"未找到模型路径: {MODEL_DIR}")

    if _is_adapter_directory(MODEL_DIR):
        peft_module = importlib.import_module("peft")
        peft_model_class = getattr(peft_module, "PeftModel", None)
        if peft_model_class is None:
            raise RuntimeError("当前环境未正确安装 peft，无法加载 LoRA adapter")

        base_model_reference = _resolve_base_model_reference(MODEL_DIR)
        local_base_model = _is_existing_local_path(base_model_reference)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_reference,
            local_files_only=local_base_model,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_reference,
            local_files_only=local_base_model,
        )
        model = peft_model_class.from_pretrained(
            base_model,
            MODEL_DIR,
            local_files_only=True,
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    if _is_full_model_directory(MODEL_DIR):
        return _load_plain_generator(str(MODEL_DIR))

    raise FileNotFoundError(
        "models/remin_adapter 不是有效的模型目录。请放入完整模型文件，"
        "或 LoRA adapter 文件（如 adapter_config.json）。如需本地加载基座模型，可设置 LOCAL_BASE_MODEL_PATH 或 BASE_MODEL_PATH。"
    )


@lru_cache(maxsize=1)
def _load_direct_generator():
    return _load_plain_generator(_resolve_direct_model_reference())


def _run_direct_llm_baseline(user_query: str) -> AnswerBundle:
    generator = _load_direct_generator()
    direct_prompt = f"""你是智能交通领域学术助手。请直接回答用户问题，不使用外部检索证据。

要求：
1. 回答风格正式、学术。
2. 不要虚构具体实验数据、论文标题或定量结论。
3. 如果无法确认，请明确说明“依据模型已有知识，无法保证与最新文献完全一致”。
4. 输出结构为：核心结论、依据说明、局限性。

用户问题：
{user_query}
"""
    response = generator(
        direct_prompt,
        max_new_tokens=SETTINGS.answer_max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return AnswerBundle(
        answer=response[0]["generated_text"].strip(),
        answer_mode="direct_llm",
        warnings=["该回答未使用外部文献检索证据，仅适合作为论文中的对比基线。"],
    )


def generate_answer_bundle(
    user_query: str,
    source_ids: Optional[List[str]] = None,
    mode: str = "rag",
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> AnswerBundle:
    normalized_mode = (mode or "rag").strip().lower()
    if normalized_mode == "direct_llm":
        try:
            return _run_direct_llm_baseline(user_query)
        except Exception as exc:
            return AnswerBundle(
                answer=(
                    "当前环境未能成功加载基座模型，因此无法生成“直接 LLM 回答”基线结果。"
                    "如需进行论文中的基线对比，请先配置可用的大模型权重。"
                ),
                answer_mode="direct_llm_unavailable",
                warnings=[f"直接回答基线不可用: {exc}"],
                used_fallback=True,
            )

    effective_top_k = top_k or SETTINGS.retrieval_top_k
    retrieval_query = _build_retrieval_query(user_query)
    context_data = query_vector_db(
        retrieval_query,
        n_results=effective_top_k,
        source_ids=source_ids,
        collection_name=collection_name,
    )
    documents = context_data.get("documents", [[]])[0]
    metadatas = context_data.get("metadatas", [[]])[0]
    documents, metadatas = _filter_retrieved_context(documents, metadatas)
    evidence_topics = classify_its_topics(
        " ".join([user_query] + [str(metadata.get("title", "")) for metadata in metadatas])
    )
    warnings: List[str] = []

    if not documents:
        return AnswerBundle(
            answer=(
                "当前知识库中没有足够清晰的正文证据来回答该问题。"
                "可能原因是检索到的片段主要来自图表、坐标轴或版面噪声；"
                "请重新勾选文献入库，或上传更清晰的 PDF 后再试。"
            ),
            answer_mode="rag_no_evidence",
            retrieval_count=0,
            evidence_topics=evidence_topics,
            warnings=["未检索到可用于生成的有效证据片段。"],
        )

    context_text = _build_context_block(documents, metadatas)
    topic_guidance = _build_topic_guidance(user_query, metadatas)
    prompt = f"""你是 Remin 学术助手。请严格依据给定证据回答问题。

要求：
1. 先直接回答用户问题，不要只写空泛免责声明。
2. 只能使用证据中的信息，不得编造不存在的实验、数据或结论。
3. 回答风格需正式、学术、审慎。
4. 若证据不足，必须明确说明“证据不足”。
5. 在关键结论后附上证据编号，例如 [证据1]。
6. 应综合多条证据形成归纳性回答，不要简单罗列或逐句复制原文。
7. 如果用户要求“总结/综述/发展现状”，请按“总体判断、主要研究方向、存在问题或趋势、局限性”组织回答。
8. 不要说“只能从若干关键词推断”，除非确实没有句子级证据；证据不足时应直接说明不足，不要强行归纳。
9. 当前问题所属的智能交通主题优先理解为：{topic_guidance}。

证据材料：
{context_text}

用户问题：
{user_query}

请输出：
- 核心结论
- 依据说明
- 局限性
"""

    try:
        generator = _load_generator()
        response = generator(
            prompt,
            max_new_tokens=SETTINGS.answer_max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return AnswerBundle(
            answer=response[0]["generated_text"].strip(),
            answer_mode="rag",
            retrieval_count=len(documents),
            evidence_topics=evidence_topics,
            warnings=warnings,
        )
    except Exception as exc:
        print(f"微调模型加载或生成失败，改用证据式回答: {exc}")
        warnings.append(f"模型生成失败，已切换为证据式回退回答: {exc}")
        return AnswerBundle(
            answer=_fallback_academic_answer(user_query, documents, metadatas),
            answer_mode="rag_evidence_fallback",
            retrieval_count=len(documents),
            evidence_topics=evidence_topics,
            warnings=warnings,
            used_fallback=True,
        )


def generate_academic_summary(user_query: str, source_ids: Optional[List[str]] = None) -> str:
    return generate_answer_bundle(user_query, source_ids=source_ids).answer
