from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.database import save_to_vector_db
from app.rag_service import generate_academic_summary
from app.search_engine import fetch_and_sort_papers, get_cached_papers
from data_pipeline.fetch_fulltext import download_pdf, resolve_pdf_url
from data_pipeline.pdf_parser import clean_pdf_text
from data_pipeline.text_processor import TextProcessor

load_dotenv()

app = FastAPI(title="Remin Academic AI System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("storage/uploads")
PDF_CACHE_DIR = Path("storage/pdf_cache")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
text_processor = TextProcessor()


class SearchRequest(BaseModel):
    keyword: str
    limit: int = 10


class ChatRequest(BaseModel):
    query: str
    paper_ids: List[str] = Field(default_factory=list)
    uploaded_source_ids: List[str] = Field(default_factory=list)


def _ingest_selected_papers(paper_ids: List[str]) -> List[str]:
    ingested_sources = []
    for paper in get_cached_papers(paper_ids):
        source_text = None
        abstract = (paper.get("abstract") or "").strip()
        paper_context = "\n".join(
            [
                f"标题: {paper.get('title', '')}",
                f"作者: {', '.join(paper.get('authors', []))}",
                f"年份: {paper.get('year')}",
                f"摘要: {abstract or '暂无摘要'}",
            ]
        )
        pdf_url = resolve_pdf_url(paper)
        if pdf_url:
            pdf_path = PDF_CACHE_DIR / f"{paper['paperId']}.pdf"
            try:
                if not pdf_path.exists():
                    download_pdf(pdf_url, pdf_path)
                pdf_text = clean_pdf_text(str(pdf_path))
                source_text = f"{paper_context}\n\n正文摘录:\n{pdf_text}"
            except Exception as exc:
                print(f"下载或解析全文失败，回退到摘要: {paper['paperId']} | {exc}")

        if not source_text:
            if not abstract or abstract == "暂无摘要":
                continue
            source_text = paper_context

        chunks = text_processor.split_text(source_text)
        save_to_vector_db(
            source_id=paper["paperId"],
            text_chunks=chunks,
            source_type="recommended_paper",
            title=paper.get("title"),
        )
        ingested_sources.append(paper["paperId"])

    return ingested_sources


@app.get("/")
def read_root():
    return {"message": "Welcome to Remin Academic API"}


@app.post("/search")
async def search_papers(req: SearchRequest):
    results, result_source = fetch_and_sort_papers(req.keyword, req.limit)
    if not results:
        raise HTTPException(status_code=404, detail="未找到相关论文")
    return {"results": results, "result_source": result_source}


@app.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="当前仅支持 PDF 文件上传")

    source_id = f"upload_{uuid4().hex}"
    file_path = UPLOAD_DIR / f"{source_id}.pdf"

    content = await file.read()
    file_path.write_bytes(content)

    try:
        cleaned_text = clean_pdf_text(str(file_path))
        chunks = text_processor.split_text(cleaned_text)
        if not chunks:
            raise ValueError("PDF 未解析出有效文本")

        save_to_vector_db(
            source_id=source_id,
            text_chunks=chunks,
            source_type="uploaded_pdf",
            title=title or file.filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件解析或入库失败: {exc}") from exc

    return {
        "source_id": source_id,
        "filename": file.filename,
        "title": title or file.filename,
        "chunk_count": len(chunks),
    }


@app.post("/chat")
async def chat_with_papers(req: ChatRequest):
    selected_sources = []
    if req.paper_ids:
        selected_sources.extend(_ingest_selected_papers(req.paper_ids))
    if req.uploaded_source_ids:
        selected_sources.extend(req.uploaded_source_ids)

    if not selected_sources:
        raise HTTPException(
            status_code=400,
            detail="请先选择推荐文献，或先上传 PDF 文件后再提问",
        )

    answer = generate_academic_summary(req.query, source_ids=selected_sources)
    return {
        "answer": answer,
        "sources": selected_sources,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
