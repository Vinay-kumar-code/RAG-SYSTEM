import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

OLLAMA_LLM_MODEL_TAG = "gemma3:1b"  #You can select any installed Ollama Model
OLLAMA_EMBED_MODEL_TAG = "nomic-embed-text:latest" #You can select any installed embedding model from ollama


c = input("Enter the path to your document (Supported File formats: txt, pdf, xls, xlsx): ")

DOCUMENT_PATH = c

NUM_CHUNKS_TO_RETRIEVE = 3

print("Initializing components...")
try:
    print(f"Initializing Ollama Embeddings with model '{OLLAMA_EMBED_MODEL_TAG}'...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL_TAG)
    _ = embeddings.embed_query("Test embedding.")
    print("Ollama Embeddings initialized successfully.")
except Exception as e:
    print(f"Error initializing Ollama Embeddings: {e}")
    print(f"Ensure Ollama is running and model '{OLLAMA_EMBED_MODEL_TAG}' is installed.")
    exit()

try:
    print(f"Initializing Ollama LLM with model '{OLLAMA_LLM_MODEL_TAG}'...")
    llm = Ollama(model=OLLAMA_LLM_MODEL_TAG)
    print("Verifying Ollama LLM connection...")
    llm.invoke("Hello!")
    print("Ollama LLM initialized and connection verified.")
except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    print(f"Ensure Ollama is running and model '{OLLAMA_LLM_MODEL_TAG}' is installed.")
    exit()


try:
    
    file_path = DOCUMENT_PATH
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified path does not exist: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    print(f"Loading document from '{file_path}' (type: {file_extension})...")

    if file_extension == ".pdf":
        
        loader = PyPDFLoader(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        
        if not os.path.isfile(file_path):
             raise FileNotFoundError(f"The path is not a valid file: {file_path}")
        print(f"Warning: Unsupported file extension '{file_extension}'. Attempting to load as text.")
        loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)

    docs_raw = loader.load() 

    if not docs_raw:
        raise ValueError("Document is empty or could not be loaded.")

    docs = docs_raw
    print(f"Loaded {len(docs)} document section(s)/page(s)/row(s).")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except ImportError as e:
    print(f"Import Error: {e}. Make sure you have installed the required libraries.")
    print("For PDF: pip install pypdf")
    print("For Excel: pip install \"unstructured[xlsx]\"")
    exit()
except Exception as e:
    print(f"An error occurred loading the document: {e}")
    exit()

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)
print(f"Split into {len(texts)} chunks.")
if not texts:
    print("Error: No text chunks were generated. Check document content and splitter settings.")
    exit()


try:
    print("Creating vector store (FAISS) using Ollama embeddings... This might take a moment.")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully.")
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit()


retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})
print(f"Retriever created to fetch top {NUM_CHUNKS_TO_RETRIEVE} chunks.")


prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer from the provided context, just say that you don't know.
Keep relevant to the question.

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


print("Creating the RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)
print("RAG chain created successfully.")



print("\n--- Ready to Query ---")
print("Enter 'exit' or 'bye' to quit.")

while True:
    query = input("\nEnter your query: ")
    if query.lower() in ["exit", "bye"]:
        print("Exiting...")
        break
    if query.strip():
        print(f"\nProcessing query: '{query}'")
        try:
            result = qa_chain.invoke({"query": query})
            print("\n--- Answer ---")
            print(result["result"])
            
        except Exception as e:
            print(f"\nAn error occurred during the RAG query: {e}")
    else:
        print("Please enter a valid query.")

print("\n--- RAG session ended ---")
