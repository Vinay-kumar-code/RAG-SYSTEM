import streamlit as st
import os
import tempfile # To handle uploaded files temporarily

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
st.title("📄 Document Query Assistant (using Ollama)")

# --- Ollama Model Configuration ---
# Sidebar for model selection (optional, but good practice)
st.sidebar.header("Ollama Configuration")
OLLAMA_LLM_MODEL_TAG = st.sidebar.text_input(
    "Enter Ollama LLM Model Tag",
    value="gemma3:1b" # Default value
)
OLLAMA_EMBED_MODEL_TAG = st.sidebar.text_input(
    "Enter Ollama Embedding Model Tag",
    value="nomic-embed-text:latest" # Default value
)

# --- Constants ---
NUM_CHUNKS_TO_RETRIEVE = 3

# --- Caching Functions ---
# Cache the Ollama LLM and Embeddings models to avoid re-initializing
@st.cache_resource
def get_ollama_llm(_model_tag):
    """Initializes and returns the Ollama LLM."""
    print(f"Initializing Ollama LLM with model '{_model_tag}'...")
    try:
        llm = Ollama(model=_model_tag)
        # Quick verification
        llm.invoke("Hello!")
        print("Ollama LLM initialized and connection verified.")
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama LLM ({_model_tag}): {e}", icon="🚨")
        st.warning(f"Ensure Ollama is running and model '{_model_tag}' is installed ('ollama pull {_model_tag}').")
        return None

@st.cache_resource
def get_ollama_embeddings(_model_tag):
    """Initializes and returns the Ollama Embeddings."""
    print(f"Initializing Ollama Embeddings with model '{_model_tag}'...")
    try:
        embeddings = OllamaEmbeddings(model=_model_tag)
        # Quick verification
        _ = embeddings.embed_query("Test embedding.")
        print("Ollama Embeddings initialized successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error initializing Ollama Embeddings ({_model_tag}): {e}", icon="🚨")
        st.warning(f"Ensure Ollama is running and model '{_model_tag}' is installed ('ollama pull {_model_tag}').")
        return None

# --- RAG Processing Functions ---

def load_document(uploaded_file):
    """Loads content from uploaded file based on extension."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, uploaded_file.name)

    # Save the uploaded file temporarily
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    file_extension = os.path.splitext(temp_path)[1].lower()
    print(f"Loading document from '{temp_path}' (type: {file_extension})...")

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
        print(f"Loaded {len(docs)} document section(s)/page(s)/row(s).")

    except ImportError as e:
        st.error(f"Import Error: {e}. Library missing for {file_extension} files.", icon="🚨")
        st.info("Install required libraries: `pip install pypdf 'unstructured[xlsx]'`")
        docs = None # Indicate failure
    except Exception as e:
        st.error(f"Error loading document: {e}", icon="🚨")
        docs = None # Indicate failure
    finally:
        # Clean up the temporary directory
        temp_dir.cleanup()

    return docs

# Cache the vector store creation based on file content and embedding model
# Use file ID and model tag as part of the cache key implicitly
@st.cache_resource(show_spinner="Processing Document and Creating Vector Store...")
def create_vector_store(_docs, _embeddings):
    """Splits docs, creates embeddings, and builds the FAISS vector store."""
    if not _docs:
        st.error("Cannot create vector store: No documents loaded.", icon="🚫")
        return None
    if not _embeddings:
        st.error("Cannot create vector store: Embeddings model not available.", icon="🚫")
        return None

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(_docs)
    print(f"Split into {len(texts)} chunks.")
    if not texts:
        st.error("No text chunks generated after splitting. Check document content.", icon="⚠️")
        return None

    try:
        print("Creating vector store (FAISS)...")
        vectorstore = FAISS.from_documents(texts, _embeddings)
        print("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}", icon="🚨")
        return None

def get_rag_chain(_llm, _vectorstore):
    """Creates the RetrievalQA chain."""
    if not _llm or not _vectorstore:
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
    print("RAG chain created successfully.")
    return qa_chain

# --- Streamlit UI Elements ---

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your document (.txt, .pdf, .xlsx, .xls)",
    type=["txt", "pdf", "xlsx", "xls"]
)

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
    # Check if it's a new file
    file_id = uploaded_file.file_id
    if st.session_state.current_file_id != file_id:
        st.info(f"Processing uploaded file: {uploaded_file.name}", icon="⏳")
        st.session_state.messages = [] # Clear chat history for new file
        st.session_state.rag_chain = None # Reset chain

        # Load models (cached)
        llm = get_ollama_llm(OLLAMA_LLM_MODEL_TAG)
        embeddings = get_ollama_embeddings(OLLAMA_EMBED_MODEL_TAG)

        if llm and embeddings:
            # Load and process document
            docs = load_document(uploaded_file)
            if docs:
                # Create vector store (cached based on docs)
                vectorstore = create_vector_store(docs, embeddings)
                if vectorstore:
                    # Create RAG chain
                    st.session_state.rag_chain = get_rag_chain(llm, vectorstore)
                    st.session_state.current_file_id = file_id # Mark file as processed
                    st.success("Document processed successfully. Ready for questions!", icon="✅")
                else:
                    st.error("Failed to create vector store from the document.", icon="❌")
            else:
                 st.error("Failed to load the document.", icon="❌")
        else:
            st.error("Failed to initialize Ollama models. Cannot process document.", icon="❌")

elif st.session_state.rag_chain is None:
    st.info("Please upload a document to begin.")


# React to user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if RAG chain is ready
    if st.session_state.rag_chain is not None:
        # Generate response
        try:
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({"query": prompt})
                response = result["result"]
                # Optionally display source documents (can be noisy)
                # sources = result.get("source_documents", [])
                # if sources:
                #    response += "\n\n**Sources:**\n"
                #    for i, doc in enumerate(sources):
                #        response += f"\n*Source {i+1}:*\n```\n{doc.page_content[:150]}...\n```"

        except Exception as e:
            response = f"An error occurred during query processing: {e}"
            st.error(response, icon="🚨")
    elif uploaded_file is None:
         response = "Please upload a document first."
         st.warning(response, icon="⚠️")
    else:
         response = "The document is still processing or failed to process. Please check status messages above."
         st.warning(response, icon="⚠️")


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

