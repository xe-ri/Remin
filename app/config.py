import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _read_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        return default

    return max(parsed, minimum)


@dataclass(frozen=True)
class SystemSettings:
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    answer_max_new_tokens: int
    max_upload_bytes: int
    max_document_characters: int
    max_chunks_per_document: int


SETTINGS = SystemSettings(
    chunk_size=_read_int_env("RAG_CHUNK_SIZE", 800, 100),
    chunk_overlap=_read_int_env("RAG_CHUNK_OVERLAP", 100, 0),
    retrieval_top_k=_read_int_env("RAG_TOP_K", 10, 1),
    answer_max_new_tokens=_read_int_env("RAG_ANSWER_MAX_NEW_TOKENS", 512, 64),
    max_upload_bytes=_read_int_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024, 1024),
    max_document_characters=_read_int_env("MAX_DOCUMENT_CHARACTERS", 300000, 2000),
    max_chunks_per_document=_read_int_env("MAX_CHUNKS_PER_DOCUMENT", 500, 10),
)
