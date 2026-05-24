from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, List, Optional

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./storage/chroma_db")
COLLECTION_NAME = "remin_papers"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "").strip()

client = chromadb.PersistentClient(path=CHROMA_PATH)


@lru_cache(maxsize=1)
def _get_embedding_function():
    model_reference = LOCAL_EMBEDDING_MODEL_PATH or EMBEDDING_MODEL_NAME
    if LOCAL_EMBEDDING_MODEL_PATH and not Path(LOCAL_EMBEDDING_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"未找到本地嵌入模型目录: {LOCAL_EMBEDDING_MODEL_PATH}"
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_reference
    )


def _resolve_collection_name(collection_name: Optional[str] = None) -> str:
    return (collection_name or COLLECTION_NAME).strip() or COLLECTION_NAME


def _get_collection(collection_name: Optional[str] = None):
    return client.get_or_create_collection(
        name=_resolve_collection_name(collection_name),
        embedding_function=_get_embedding_function(),
    )


def save_to_vector_db(
    source_id: str,
    text_chunks: List[str],
    source_type: str = "paper",
    title: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> int:
    collection = _get_collection(collection_name)
    try:
        collection.delete(where={"source": source_id})
    except Exception as exc:
        print(f"清理旧文本块失败，将继续 upsert: {source_id} | {exc}")

    ids = [f"{source_id}_{index}" for index in range(len(text_chunks))]
    metadatas = [
        {
            "source": source_id,
            "source_type": source_type,
            "title": title or source_id,
            "chunk_index": index,
        }
        for index, _ in enumerate(text_chunks)
    ]

    collection.upsert(documents=text_chunks, ids=ids, metadatas=metadatas)
    print(f"文献 {source_id} 已入库，共 {len(text_chunks)} 个文本块")
    return len(text_chunks)


def query_vector_db(
    query: str,
    n_results: int = 5,
    source_ids: Optional[Iterable[str]] = None,
    collection_name: Optional[str] = None,
):
    try:
        collection = client.get_collection(
            name=_resolve_collection_name(collection_name),
            embedding_function=_get_embedding_function(),
        )
    except NotFoundError:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
    except Exception as exc:
        print(f"加载向量集合失败: {exc}")
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    where = None
    source_ids = list(source_ids or [])
    if len(source_ids) == 1:
        where = {"source": source_ids[0]}
    elif len(source_ids) > 1:
        where = {"source": {"$in": source_ids}}

    try:
        return collection.query(query_texts=[query], n_results=n_results, where=where)
    except Exception as exc:
        print(f"向量检索失败: {exc}")
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}


def count_vector_records(
    source_ids: Optional[Iterable[str]] = None,
    collection_name: Optional[str] = None,
) -> int:
    try:
        collection = client.get_collection(
            name=_resolve_collection_name(collection_name),
            embedding_function=_get_embedding_function(),
        )
    except Exception:
        return 0

    source_ids = list(source_ids or [])
    if not source_ids:
        try:
            return int(collection.count())
        except Exception:
            return 0

    total = 0
    for source_id in source_ids:
        try:
            result = collection.get(where={"source": source_id}, include=[])
            total += len(result.get("ids", []))
        except Exception:
            continue
    return total
