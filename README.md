# RAG Chat Web App

Single-user RAG web application with:
- Multiple RAG stores (build from uploaded PDF/TXT/CSV/XLSX)
- Per-session generation model selection (Ollama)
- Source preview
- Semantic search inside selected store
- Export chat as text
- Error reveal button (only shows when errors occur)

## Quick Start

1) Install dependencies

```bash
pip install -U fastapi uvicorn langchain langchain-community faiss-cpu pypdf "unstructured[xlsx]" ollama python-multipart pandas
```

2) Start the server

```bash
uvicorn rag.server.server:app --reload --port 8000
```

3) Open the web UI

- http://localhost:8000

## Create a Store
1. Expand "Create RAG Store"
2. Provide store id and upload a file
3. Optionally change the embedding model (default: nomic-embed-text)
4. Click Build
5. Select the store from dropdown

## Chat
1. Pick an Ollama model (from the Model dropdown)
2. Pick a store
3. Ask a question
4. View sources (expand the cards)
5. Export transcript with the Export button

## Semantic Search
- Use the top search box to find chunks in the currently selected store.

## Folder Layout
```
rag/
  server/server.py       # FastAPI app + RAG endpoints
  client/
    templates/index.html # Web UI
    static/{app.js,styles.css}
  stores/                # Auto-created. FAISS index per store + <store_id>.meta.json
```

## Notes
- Only the generation model is selectable per session (embedding model is stored with the RAG store meta)
- No authentication (single user)
- Chat history not persisted after reload
- Error button appears only when an error occurs
- Requires Ollama running locally with desired models pulled (e.g., `ollama pull llama3`)

## Troubleshooting

- Models list empty or chat errors
  - Ensure Ollama is installed and running.
  - Pull models: `ollama pull llama3`; `ollama pull nomic-embed-text`.
  - Verify: `ollama list` shows your models.
  - Restart the server after installing models.

- Chat returns Generation failed
  - Check server logs; error usually indicates Ollama not running or model missing.
  - Try a simpler model name that exists in `ollama list`.

- Build store fails on uploads
  - Make sure `python-multipart` is installed.
  - For CSV/XLSX parsing, ensure `pandas` and `unstructured[xlsx]` are installed.

