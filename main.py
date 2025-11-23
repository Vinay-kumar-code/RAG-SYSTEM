import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround for OpenMP duplicate runtime on Windows
import tempfile # To handle uploaded files temporarily
import requests
import json
import pickle
from pathlib import Path
import hashlib
import logging
import time
import traceback
import uuid

# LangChain components
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- App Configuration ---

# Configure the Streamlit page
st.set_page_config(page_title="Doc Q&A with Ollama", layout="wide")
# Disable file watcher to avoid torch.classes inspection crash on some Windows/PyTorch setups
try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass
st.title("📄 Document Query Assistant (using Ollama)")

# --- Constants ---
NUM_CHUNKS_TO_RETRIEVE = 3
VECTOR_STORES_DIR = "vector_stores"

# Create directory for storing vector stores
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)

# --- Logging Configuration ---
LOG_DIR = os.path.join(os.getcwd(), "logs")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    pass
log_file_path = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("rag_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        # If file logger fails, continue with stream only
        pass

logger.info("Application started")

def _new_req_id():
    """Return a short correlation id for log lines."""
    return uuid.uuid4().hex[:8]

# --- Helper Functions ---

def get_ollama_models():
    """Fetch available Ollama models from the local Ollama instance."""
    req = _new_req_id()
    logger.info(f"[{req}] Fetching Ollama models from http://localhost:11434/api/tags")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        logger.info(f"[{req}] Ollama tags status={response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            logger.info(f"[{req}] Found {len(models)} models: {models}")
            return models
        else:
            logger.warning(f"[{req}] Could not connect to Ollama API. status={response.status_code}")
            st.warning("Could not connect to Ollama API. Make sure Ollama is running.")
            return []
    except Exception as e:
        logger.exception(f"[{req}] Error fetching Ollama models: {e}")
        st.warning(f"Error fetching Ollama models: {e}")
        return []

def compute_file_id(file_bytes: bytes, filename: str) -> str:
    """Compute a stable ID for an uploaded file using content + filename."""
    h = hashlib.md5()
    h.update(file_bytes)
    h.update(filename.encode("utf-8"))
    return h.hexdigest()

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_saved_vector_stores():
    """Get list of previously saved FAISS vector store directories."""
    req = _new_req_id()
    logger.info(f"[{req}] Scanning vector stores in '{VECTOR_STORES_DIR}'")
    stores: list[str] = []
    if os.path.exists(VECTOR_STORES_DIR):
        for name in os.listdir(VECTOR_STORES_DIR):
            dir_path = os.path.join(VECTOR_STORES_DIR, name)
            if os.path.isdir(dir_path):
                faiss_idx = os.path.join(dir_path, "index.faiss")
                pkl_idx = os.path.join(dir_path, "index.pkl")
                if os.path.exists(faiss_idx) and os.path.exists(pkl_idx):
                    stores.append(name)
    logger.info(f"[{req}] Found stores: {stores}")
    return sorted(stores)

def save_vector_store(vectorstore, filename):
    """Save vector store to disk as a FAISS folder under VECTOR_STORES_DIR/filename."""
    req = _new_req_id()
    logger.info(f"[{req}] Saving vector store '{filename}'")
    try:
        save_dir = os.path.join(VECTOR_STORES_DIR, filename)
        vectorstore.save_local(save_dir)
        logger.info(f"[{req}] Saved vector store to {save_dir}")
        st.success(f"Vector store saved as '{filename}'")
        return True
    except Exception as e:
        logger.exception(f"[{req}] Error saving vector store: {e}")
        st.error(f"Error saving vector store: {e}")
        return False

def load_vector_store(filename, embeddings):
    """Load vector store from disk."""
    req = _new_req_id()
    logger.info(f"[{req}] Loading vector store '{filename}'")
    try:
        filepath = os.path.join(VECTOR_STORES_DIR, filename)
        vectorstore = FAISS.load_local(filepath, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"[{req}] Loaded vector store from {filepath}")
        return vectorstore
    except Exception as e:
        logger.exception(f"[{req}] Error loading vector store '{filename}': {e}")
        st.error(f"Error loading vector store '{filename}': {e}")
        return None

# --- Ollama Model Configuration ---
st.sidebar.header("🔧 Ollama Configuration")

# Get available models
available_models = get_ollama_models()

if available_models:
    # LLM Model Selection
    llm_models = [model for model in available_models if not any(embed_keyword in model.lower() 
                 for embed_keyword in ['embed', 'embedding', 'nomic'])]
    
    if llm_models:
        default_llm = llm_models[0]
        # Try to find gemma model as default
        for model in llm_models:
            if 'gemma' in model.lower():
                default_llm = model
                break
        
        OLLAMA_LLM_MODEL_TAG = st.sidebar.selectbox(
            "Select Ollama LLM Model",
            options=llm_models,
            index=llm_models.index(default_llm) if default_llm in llm_models else 0
        )
    else:
        st.sidebar.warning("No LLM models found in Ollama")
        OLLAMA_LLM_MODEL_TAG = st.sidebar.text_input(
            "Enter Ollama LLM Model Tag",
            value="gemma2:2b"
        )
    
    # Embedding Model Selection
    embed_models = [model for model in available_models if any(embed_keyword in model.lower() 
                   for embed_keyword in ['embed', 'embedding', 'nomic'])]
    
    if embed_models:
        default_embed = embed_models[0]
        # Try to find nomic-embed as default
        for model in embed_models:
            if 'nomic' in model.lower():
                default_embed = model
                break
        
        OLLAMA_EMBED_MODEL_TAG = st.sidebar.selectbox(
            "Select Ollama Embedding Model",
            options=embed_models,
            index=embed_models.index(default_embed) if default_embed in embed_models else 0
        )
    else:
        st.sidebar.warning("No embedding models found in Ollama")
        OLLAMA_EMBED_MODEL_TAG = st.sidebar.text_input(
            "Enter Ollama Embedding Model Tag",
            value="nomic-embed-text:latest"
        )

else:
    st.sidebar.warning("⚠️ Could not fetch Ollama models. Using manual input.")
    OLLAMA_LLM_MODEL_TAG = st.sidebar.text_input(
        "Enter Ollama LLM Model Tag",
        value="gemma2:2b"
    )
    OLLAMA_EMBED_MODEL_TAG = st.sidebar.text_input(
        "Enter Ollama Embedding Model Tag",
        value="nomic-embed-text:latest"
    )

# Refresh models button
if st.sidebar.button("🔄 Refresh Models"):
    st.rerun()

# --- Vector Store Management ---
st.sidebar.header("💾 Vector Store Management")

# Load existing vector store
saved_stores = get_saved_vector_stores()
if saved_stores:
    selected_store = st.sidebar.selectbox(
        "Load Previously Created Vector Store",
        options=[""] + saved_stores,
        format_func=lambda x: "Select a vector store..." if x == "" else x
    )

    if selected_store and st.sidebar.button("📂 Load Selected Store"):
        st.session_state.load_existing_store = selected_store
        st.rerun()
else:
    st.sidebar.info("No saved vector stores found. Create one by processing a document and enabling 'Save vector store'.")

# --- Caching Functions ---
# Cache the Ollama LLM and Embeddings models to avoid re-initializing
@st.cache_resource
def get_ollama_llm(_model_tag):
    """Initializes and returns the Ollama LLM."""
    req = _new_req_id()
    t0 = time.time()
    logger.info(f"[{req}] Initializing Ollama LLM model='{_model_tag}'")
    try:
        llm = Ollama(model=_model_tag)
        dt = (time.time() - t0) * 1000
        logger.info(f"[{req}] LLM ready (took {dt:.1f} ms)")
        return llm
    except Exception as e:
        logger.exception(f"[{req}] Error initializing Ollama LLM ({_model_tag}): {e}")
        st.error(f"Error initializing Ollama LLM ({_model_tag}): {e}", icon="🚨")
        st.warning(f"Ensure Ollama is running and model '{_model_tag}' is installed ('ollama pull {_model_tag}').")
        return None

@st.cache_resource
def get_ollama_embeddings(_model_tag):
    """Initializes and returns the Ollama Embeddings."""
    req = _new_req_id()
    t0 = time.time()
    logger.info(f"[{req}] Initializing Ollama Embeddings model='{_model_tag}'")
    try:
        embeddings = OllamaEmbeddings(model=_model_tag)
        dt = (time.time() - t0) * 1000
        logger.info(f"[{req}] Embeddings ready (took {dt:.1f} ms)")
        return embeddings
    except Exception as e:
        logger.exception(f"[{req}] Error initializing Ollama Embeddings ({_model_tag}): {e}")
        st.error(f"Error initializing Ollama Embeddings ({_model_tag}): {e}", icon="🚨")
        st.warning(f"Ensure Ollama is running and model '{_model_tag}' is installed ('ollama pull {_model_tag}').")
        return None

# --- RAG Processing Functions ---

def load_document(uploaded_file):
    """Loads content from uploaded file based on extension."""
    req = _new_req_id()
    logger.info(f"[{req}] Loading uploaded file name='{uploaded_file.name}' size={len(uploaded_file.getvalue())} bytes")
    docs = []

    # Use context manager for proper cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_extension = os.path.splitext(temp_path)[1].lower()
        logger.info(f"[{req}] Detected extension {file_extension}; temp path={temp_path}")

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(temp_path, mode="elements")
            elif file_extension == ".txt":
                loader = TextLoader(temp_path, encoding='utf-8')
            else:
                st.warning(f"Unsupported file extension '{file_extension}'. Attempting to load as text.", icon="⚠️")
                loader = TextLoader(temp_path, encoding='utf-8', autodetect_encoding=True)

            docs = loader.load()
            if not docs:
                raise ValueError("Document is empty or could not be loaded.")
            logger.info(f"[{req}] Loaded {len(docs)} document sections/pages/rows")

        except ImportError as e:
            logger.exception(f"[{req}] Import error while loading: {e}")
            st.error(f"Import Error: {e}. Library missing for {file_extension} files.", icon="🚨")
            st.info("Install required libraries: `pip install pypdf 'unstructured[xlsx]'`")
            docs = None
        except Exception as e:
            logger.exception(f"[{req}] Error loading document: {e}")
            st.error(f"Error loading document: {e}", icon="🚨")
            docs = None

    return docs

# Cache the vector store creation based on file content and embedding model
# Use file ID and model tag as part of the cache key implicitly
@st.cache_resource(show_spinner="Processing Document and Creating Vector Store...")
def create_vector_store(_docs, _embeddings, save_name=None):
    """Splits docs, creates embeddings, and builds the FAISS vector store."""
    req = _new_req_id()
    logger.info(f"[{req}] Creating vector store; docs={len(_docs) if _docs else 0} save_name={save_name}")
    if not _docs:
        st.error("Cannot create vector store: No documents loaded.", icon="🚫")
        logger.warning(f"[{req}] No documents to index")
        return None
    if not _embeddings:
        st.error("Cannot create vector store: Embeddings model not available.", icon="🚫")
        logger.warning(f"[{req}] No embeddings instance")
        return None

    try:
        # Use larger chunk size for better performance
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(_docs)
        logger.info(f"[{req}] Split into {len(texts)} chunks")
        if not texts:
            st.error("No text chunks generated after splitting. Check document content.", icon="⚠️")
            logger.warning(f"[{req}] No chunks generated")
            return None

        vectorstore = FAISS.from_documents(texts, _embeddings)
        logger.info(f"[{req}] Vector store built")

        if save_name:
            save_vector_store(vectorstore, save_name)

        return vectorstore
    except Exception as e:
        logger.exception(f"[{req}] Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}", icon="🚨")
        return None

def get_rag_chain(_llm, _vectorstore):
    """Creates the RetrievalQA chain."""
    req = _new_req_id()
    logger.info(f"[{req}] Creating RAG chain")
    if not _llm or not _vectorstore:
        logger.warning(f"[{req}] Missing llm={bool(_llm)} vectorstore={bool(_vectorstore)}")
        return None

    retriever = _vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})

    prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context ONLY to answer the question.
If the context doesn't contain the answer, just say that you don't know based on the provided document.
Do not make up an answer or use external knowledge. Keep the answer relavant.

Context: {context}

Question: {question}

Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    print("Creating the RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    logger.info(f"[{req}] RAG chain ready")
    return qa_chain

# --- Streamlit UI Elements ---

# Check if we should load an existing vector store
if "load_existing_store" in st.session_state and st.session_state.load_existing_store:
    store_name = st.session_state.load_existing_store
    st.info(f"Loading vector store: {store_name}", icon="📂")

    # Load models (cached)
    llm = get_ollama_llm(OLLAMA_LLM_MODEL_TAG)
    embeddings = get_ollama_embeddings(OLLAMA_EMBED_MODEL_TAG)

    if llm and embeddings:
        vectorstore = load_vector_store(store_name, embeddings)
        if vectorstore:
            st.session_state.rag_chain = get_rag_chain(llm, vectorstore)
            st.session_state.current_file_id = f"loaded_{store_name}"
            st.session_state.messages = []  # Clear chat history
            st.success(f"Vector store '{store_name}' loaded successfully! Ready for questions!", icon="✅")
        else:
            st.error(f"Failed to load vector store '{store_name}'", icon="❌")
    else:
        st.error("Failed to initialize Ollama models. Cannot load vector store.", icon="❌")

    # Clear the load flag
    del st.session_state.load_existing_store

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your document (.txt, .pdf, .xlsx, .xls)",
    type=["txt", "pdf", "xlsx", "xls"]
)

# Option to save vector store when processing new document
save_vector_store_option = st.checkbox("💾 Save vector store for future use", value=False)
vector_store_name = ""
if save_vector_store_option:
    vector_store_name = st.text_input(
        "Vector store name",
        value=(uploaded_file.name.split('.')[0] if uploaded_file else ""),
        placeholder="Enter a name for the vector store"
    ).strip()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize RAG chain in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None


# Process uploaded file only if it's new or hasn't been processed
if uploaded_file is not None:
    req = _new_req_id()
    logger.info(f"[{req}] File uploaded: name='{uploaded_file.name}' size={len(uploaded_file.getvalue())}")
    # Compute a stable file id from content + name
    file_bytes = uploaded_file.getvalue()
    file_id = compute_file_id(file_bytes, uploaded_file.name)
    logger.info(f"[{req}] Computed file_id={file_id}")
    if st.session_state.current_file_id != file_id:
        logger.info(f"[{req}] New file detected. Starting processing")
        st.info(f"Processing uploaded file: {uploaded_file.name}", icon="⏳")
        st.session_state.messages = []
        st.session_state.rag_chain = None

        # Load models (cached)
        llm = get_ollama_llm(OLLAMA_LLM_MODEL_TAG)
        embeddings = get_ollama_embeddings(OLLAMA_EMBED_MODEL_TAG)

        if llm and embeddings:
            docs = load_document(uploaded_file)
            if docs:
                save_name = vector_store_name if (save_vector_store_option and vector_store_name) else None
                vectorstore = create_vector_store(docs, embeddings, save_name)
                if vectorstore:
                    st.session_state.rag_chain = get_rag_chain(llm, vectorstore)
                    st.session_state.current_file_id = file_id
                    success_msg = "Document processed successfully. Ready for questions!"
                    if save_name:
                        success_msg += f" Vector store saved as '{save_name}'."
                    logger.info(f"[{req}] Processing done. chain ready. save_name={save_name}")
                    st.success(success_msg, icon="✅")
                else:
                    logger.warning(f"[{req}] Vector store creation failed")
                    st.error("Failed to create vector store from the document.", icon="❌")
            else:
                logger.warning(f"[{req}] Document load returned no docs")
                st.error("Failed to load the document.", icon="❌")
        else:
            logger.warning(f"[{req}] Model initialization failed llm={bool(llm)} embeddings={bool(embeddings)}")
            st.error("Failed to initialize Ollama models. Cannot process document.", icon="❌")

elif st.session_state.rag_chain is None:
    st.info("Please upload a document or load a previously saved vector store to begin.")

# Display current status
if st.session_state.rag_chain is not None:
    with st.expander("ℹ️ Current Session Info"):
        st.write(f"**LLM Model:** {OLLAMA_LLM_MODEL_TAG}")
        st.write(f"**Embedding Model:** {OLLAMA_EMBED_MODEL_TAG}")
        if hasattr(st.session_state, 'current_file_id') and st.session_state.current_file_id:
            if st.session_state.current_file_id.startswith('loaded_'):
                store_name = st.session_state.current_file_id.replace('loaded_', '')
                st.write(f"**Source:** Loaded vector store '{store_name}'")
            else:
                st.write(f"**Source:** Uploaded document")
        st.write(f"**Chunks to retrieve:** {NUM_CHUNKS_TO_RETRIEVE}")

# Vector Store Management Section
if saved_stores:
    with st.expander("🗂️ Manage Saved Vector Stores"):
        st.write("**Available Vector Stores:**")
        for i, store in enumerate(saved_stores):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {store}")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    try:
                        store_path = os.path.join(VECTOR_STORES_DIR, store)
                        # FAISS saves as directory with multiple files
                        import shutil
                        if os.path.isdir(store_path):
                            shutil.rmtree(store_path)
                        # Clear cache to reflect deletion
                        get_saved_vector_stores.clear()
                        st.success(f"Deleted vector store '{store}'")
                        st.rerun()
                    except PermissionError as e:
                        logger.exception(f"Permission error deleting vector store '{store}': {e}")
                        st.error(f"Permission error: Cannot delete '{store}'. Ensure it's not in use.")
                    except Exception as e:
                        logger.exception(f"Error deleting vector store '{store}': {e}")
                        st.error(f"Error deleting vector store: {e}")


# React to user input
if prompt := st.chat_input("Ask a question about the document..."):
    req = _new_req_id()
    logger.info(f"[{req}] User question: {prompt}")
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.rag_chain is not None:
        try:
            with st.spinner("Thinking..."):
                t0 = time.time()
                result = st.session_state.rag_chain.invoke({"query": prompt})
                dt = (time.time() - t0) * 1000
                response = result["result"]
                logger.info(f"[{req}] Answered in {dt:.1f} ms; tokens=NA")
        except Exception as e:
            logger.exception(f"[{req}] Error during query: {e}")
            response = f"An error occurred during query processing: {e}"
            st.error(response, icon="🚨")
    elif uploaded_file is None:
        response = "Please upload a document first."
        logger.info(f"[{req}] No document loaded; prompted user")
        st.warning(response, icon="⚠️")
    else:
        response = "The document is still processing or failed to process. Please check status messages above."
        logger.info(f"[{req}] Chain not ready when user asked question")
        st.warning(response, icon="⚠️")

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

