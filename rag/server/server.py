import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# Minimal RAG pieces (LangChain-like + FAISS)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

# For generation via Ollama
from langchain_community.llms import Ollama

APP_DIR = Path(__file__).resolve().parents[1]
CLIENT_DIR = APP_DIR / "client"
TEMPLATES_DIR = CLIENT_DIR / "templates"
STATIC_DIR = CLIENT_DIR / "static"
STORES_DIR = APP_DIR / "stores"
STORES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Chat App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static and index
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Index not found</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


class BuildResponse(BaseModel):
    store_id: str
    chunks: int
    embedding_model: str

class ChatRequest(BaseModel):
    store_id: Optional[str] = None
    model: str
    question: str
    k: int = 4

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


SUPPORTED_EXTS = {".pdf", ".txt", ".csv", ".xlsx"}


def _detect_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(str(path))
    if ext == ".xlsx":
        return UnstructuredExcelLoader(str(path))
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


def _load_docs(tmp_path: Path) -> List[Document]:
    loader = _detect_loader(tmp_path)
    return loader.load()


def _get_store_path(store_id: str) -> Path:
    return STORES_DIR / store_id


def _get_meta_path(store_id: str) -> Path:
    return STORES_DIR / f"{store_id}.meta.json"


@app.get("/api/stores")
def list_stores():
    items = []
    for p in STORES_DIR.glob("*"):
        if p.is_dir():
            meta = _get_meta_path(p.name)
            embedding_model = None
            if meta.exists():
                try:
                    import json
                    embedding_model = json.loads(meta.read_text()).get("embedding_model")
                except Exception:
                    embedding_model = None
            items.append({"id": p.name, "embedding_model": embedding_model})
    return {"stores": items}


@app.post("/api/build", response_model=BuildResponse)
async def build_store(store_id: str = Form(...), file: UploadFile = File(...), embedding_model: str = Form("nomic-embed-text")):
    if not store_id:
        raise HTTPException(status_code=400, detail="store_id required")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Extension not supported. Use one of: {', '.join(sorted(SUPPORTED_EXTS))}")

    tmp_path = STORES_DIR / f"tmp_{file.filename}"
    tmp_path.write_bytes(await file.read())

    try:
        docs = _load_docs(tmp_path)
        if not docs:
            raise HTTPException(status_code=400, detail="No documents could be loaded")

        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model=embedding_model)
        vs_path = _get_store_path(store_id)

        if vs_path.exists():
            # Overwrite existing store
            import shutil
            shutil.rmtree(vs_path)

        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(vs_path))

        # Save simple meta
        import json
        _get_meta_path(store_id).write_text(json.dumps({"embedding_model": embedding_model}), encoding="utf-8")

        return BuildResponse(store_id=store_id, chunks=len(docs), embedding_model=embedding_model)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass


@app.get("/api/models")
def list_models():
    # Be robust across Ollama versions and OS
    import subprocess, json, shutil

    models = []
    errors = []

    if not shutil.which("ollama"):
        # Ollama not available on PATH
        return {"models": [], "error": "ollama not found on PATH. Install/start Ollama and pull models."}

    # 1) Newer: list --format json (array)
    try:
        out = subprocess.check_output(["ollama", "list", "--format", "json"], text=True, stderr=subprocess.STDOUT)
        data = json.loads(out)
        for m in data:
            name = (m.get("name") or m.get("model") or "").strip()
            if name:
                models.append(name)
    except Exception as e:
        errors.append(f"--format json failed: {e}")

    # 2) Some versions: list --json (ndjson lines)
    if not models:
        try:
            out = subprocess.check_output(["ollama", "list", "--json"], text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    name = (obj.get("name") or obj.get("model") or "").strip()
                    if name:
                        models.append(name)
                except Exception as ie:
                    errors.append(f"ndjson parse error: {ie}")
        except Exception as e:
            errors.append(f"--json failed: {e}")

    # 3) Fallback: parse plain text table
    if not models:
        try:
            out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
            lines = [l for l in out.splitlines() if l.strip()]
            # skip header if present
            start = 1 if lines and lines[0].lower().startswith("name") else 0
            for line in lines[start:]:
                parts = line.split()
                if not parts:
                    continue
                # Expected columns: NAME TAG SIZE ...
                name = parts[0]
                tag = parts[1] if len(parts) > 1 else ""
                full = f"{name}:{tag}" if tag and tag != "-" else name
                if full:
                    models.append(full)
        except Exception as e:
            errors.append(f"plain parse failed: {e}")

    # de-duplicate
    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    if models:
        return {"models": models}

    # Final fallback list with context
    return {"models": ["llama3:latest", "qwen2:7b", "phi3:mini"], "warning": "Using fallback models; Ollama list failed.", "errors": errors}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Validate model & question
    if not req.model:
        raise HTTPException(status_code=400, detail="Model is required")
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")

    use_retrieval = bool(req.store_id)
    context = ""
    sources: List[dict] = []

    if use_retrieval:
        store_id = (req.store_id or "").strip()
        store_path = _get_store_path(store_id)
        meta_path = _get_meta_path(store_id)

        if not store_path.exists():
            raise HTTPException(status_code=404, detail="Store not found")
        if not meta_path.exists():
            raise HTTPException(status_code=500, detail="Store meta missing")

        import json
        meta = json.loads(meta_path.read_text())
        embedding_model = meta.get("embedding_model", "nomic-embed-text")

        try:
            embeddings = OllamaEmbeddings(model=embedding_model)
            vs = FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)
            docs_and_scores = vs.similarity_search_with_score(req.question, k=req.k)
            context = "\n\n".join([d.page_content for d, _ in docs_and_scores])
            for d, score in docs_and_scores:
                src = d.metadata.copy() if isinstance(d.metadata, dict) else {}
                src.update({"snippet": d.page_content[:500], "score": score})
                sources.append(src)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Generate via Ollama (with or without context)
    try:
        llm = Ollama(model=req.model)
        if context.strip():
            prompt = f"""
You are a concise assistant. Answer using only the provided context. If the context is insufficient, say you don't know.

Question: {req.question}

Context:
{context}
""".strip()
        else:
            prompt = f"""
You are a concise assistant. Answer the user question briefly and accurately.

Question: {req.question}
""".strip()
        answer = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed (is Ollama running and model pulled?): {e}")

    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/search")
def search(store_id: str, q: str, k: int = 8):
    store_path = _get_store_path(store_id)
    meta_path = _get_meta_path(store_id)
    if not store_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail="Store not found")

    import json
    embedding_model = json.loads(meta_path.read_text()).get("embedding_model", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embedding_model)
    vs = FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = vs.similarity_search_with_score(q, k=k)

    return {
        "results": [
            {
                "content": d.page_content,
                "metadata": d.metadata,
                "score": score,
            }
            for d, score in docs_and_scores
        ]
    }
