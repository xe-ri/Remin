import re
from typing import Dict, List


ITS_DOMAIN_CATALOG: Dict[str, Dict[str, List[str] | str]] = {
    "交通流预测": {
        "aliases": [
            "traffic flow prediction",
            "traffic forecasting",
            "traffic speed prediction",
            "spatio-temporal traffic forecasting",
            "交通流预测",
            "交通速度预测",
            "交通状态预测",
        ]
    },
    "交通信号控制": {
        "aliases": [
            "traffic signal control",
            "signal timing",
            "intersection control",
            "adaptive traffic signal",
            "交通信号控制",
            "信号配时",
            "路口控制",
        ]
    },
    "车联网与V2X": {
        "aliases": [
            "v2x",
            "vehicle-to-everything",
            "connected vehicle",
            "vehicular communication",
            "internet of vehicles",
            "c-v2x",
            "vanet",
            "车联网",
            "车路协同",
            "车辆通信",
        ]
    },
    "自动驾驶": {
        "aliases": [
            "autonomous driving",
            "automated driving",
            "self-driving",
            "intelligent driving",
            "自动驾驶",
            "智能驾驶",
        ]
    },
    "交通安全与事件检测": {
        "aliases": [
            "traffic safety",
            "incident detection",
            "crash prediction",
            "risk assessment",
            "交通安全",
            "事件检测",
            "事故预测",
            "风险评估",
        ]
    },
    "出行需求与路径规划": {
        "aliases": [
            "travel demand",
            "route planning",
            "vehicle routing",
            "trajectory prediction",
            "mobility analysis",
            "出行需求",
            "路径规划",
            "车辆路径",
            "轨迹预测",
        ]
    },
}

GENERIC_ITS_TERMS = [
    "intelligent transportation system",
    "traffic",
    "transportation",
    "vehicle",
    "mobility",
]


def tokenize_its_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", (text or "").lower())


def classify_its_topics(text: str, limit: int = 4) -> List[str]:
    searchable = (text or "").lower()
    labels = []
    for label, config in ITS_DOMAIN_CATALOG.items():
        aliases = [str(item).lower() for item in config.get("aliases", [])]
        if any(alias in searchable for alias in aliases):
            labels.append(label)
    return labels[:limit]


def matched_domain_terms(text: str, limit: int = 8) -> List[str]:
    searchable = (text or "").lower()
    matched = []
    for config in ITS_DOMAIN_CATALOG.values():
        for alias in config.get("aliases", []):
            alias = str(alias).lower()
            if alias in searchable and alias not in matched:
                matched.append(alias)
    return matched[:limit]


def expand_query_with_domain_terms(query: str, limit: int = 8) -> List[str]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    expansions = [normalized_query]
    query_lower = normalized_query.lower()

    for _, config in ITS_DOMAIN_CATALOG.items():
        aliases = [str(item) for item in config.get("aliases", [])]
        alias_matches = [alias for alias in aliases if alias.lower() in query_lower]
        if alias_matches:
            for alias in aliases[:3]:
                if alias not in expansions:
                    expansions.append(alias)

    if len(expansions) == 1:
        expansions.extend(GENERIC_ITS_TERMS[:3])

    expansions.extend(term for term in GENERIC_ITS_TERMS if term not in expansions)
    return list(dict.fromkeys(expansions))[:limit]


def build_domain_hint(query: str) -> str:
    topics = classify_its_topics(query)
    if topics:
        return "、".join(topics)
    return "智能交通通用主题"
