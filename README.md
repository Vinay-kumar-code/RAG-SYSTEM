# 📄 Document Query Assistant (RAG System with Ollama)

A powerful Retrieval-Augmented Generation (RAG) system built with Streamlit and Ollama for intelligent document querying. Upload documents and ask questions to get contextual answers powered by local AI models.

## 🌟 Features

- **Multi-format Document Support**: Upload and process `.txt`, `.pdf`, `.xlsx`, and `.xls` files
- **Local AI Processing**: Uses Ollama for both LLM and embedding models (no API keys required)
- **Auto-detect Ollama Models**: Automatically fetches and displays available Ollama models for selection
- **Vector Store Management**: Save and load previously created vector stores for reuse
- **Interactive Chat Interface**: Clean, user-friendly chat interface for document Q&A
- **Smart Document Processing**: Automatic text splitting and vector store creation
- **Persistent Chat History**: Maintains conversation context within sessions
- **Model Selection Interface**: Easy model selection through dropdown menus
- **Vector Store Persistence**: Save vector stores to avoid reprocessing documents
- **Error Handling**: Comprehensive error handling with user-friendly messages

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **Vector Database**: FAISS
- **AI Models**: Ollama (local models)
- **Document Processing**: PyPDF, UnstructuredExcelLoader
- **Embeddings**: Ollama Embeddings

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama installed and running**
   - Install from [https://ollama.ai/](https://ollama.ai/)
   - Ensure Ollama service is running

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies
```bash
pip install streamlit langchain langchain-community faiss-cpu pypdf unstructured[xlsx] openpyxl
```

### 3. Set Up Ollama Models
```bash
# Install models (the app will auto-detect these)
ollama pull gemma2:2b          # or any LLM model you prefer
ollama pull llama3.1:8b        # alternative LLM model
ollama pull nomic-embed-text:latest    # embedding model
ollama pull mxbai-embed-large:latest   # alternative embedding model
```

### 4. Run the Application
```bash
streamlit run RAG-system-GUI.py
```

### 5. Access the App
Open your browser and navigate to `http://localhost:8501`

## 🎯 Usage

1. **Model Selection**
   - The app automatically detects and displays available Ollama models
   - Use the sidebar to select your preferred LLM and embedding models
   - Click "🔄 Refresh Models" to update the model list

2. **Document Processing**
   - Upload a document using the file uploader
   - Optionally check "💾 Save vector store for future use" and provide a name
   - Wait for processing confirmation

3. **Vector Store Management**
   - Load previously saved vector stores from the sidebar
   - Manage (view/delete) existing vector stores in the expandable section
   - Avoid reprocessing the same documents by reusing saved vector stores

4. **Ask Questions**
   - Type your question in the chat input
   - Get contextual answers based on your document content
   - Continue the conversation with follow-up questions

5. **Session Information**
   - View current session details in the "ℹ️ Current Session Info" section
   - See which models and data sources are currently active

## ⚙️ Configuration

### Model Configuration
The app automatically detects available Ollama models. You can:
- Select models from dropdown menus in the sidebar
- Refresh the model list using the "🔄 Refresh Models" button
- Fall back to manual input if auto-detection fails

### Vector Store Settings
```python
VECTOR_STORES_DIR = "vector_stores"  # Directory for saved vector stores
NUM_CHUNKS_TO_RETRIEVE = 3          # Number of document chunks to retrieve
chunk_size = 500                    # Size of text chunks
chunk_overlap = 50                  # Overlap between chunks
```

### Ollama Connection
The app connects to Ollama at `http://localhost:11434` by default. Ensure Ollama is running on this address.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │───▶│   Text Splitter  │───▶│   Embeddings    │
│   Loader        │    │   (Chunks)       │    │   (Ollama)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   User Query    │───▶│   Retrieval QA   │◀────────────┘
│                 │    │   Chain          │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   FAISS Vector   │
                       │   Store          │
                       └──────────────────┘
```

## 🚀 New Features

### 🎯 Model Auto-Detection
- Automatically fetches available Ollama models via API
- Separates LLM models from embedding models
- Provides fallback manual input if connection fails

### 💾 Vector Store Management
- **Save Vector Stores**: Persist processed documents for future use
- **Load Existing Stores**: Quickly access previously processed documents
- **Store Management**: View and delete saved vector stores
- **Avoid Reprocessing**: Save time by reusing vector stores

### 🔧 Enhanced UI
- **Dropdown Model Selection**: Easy model switching
- **Session Information**: View current configuration and data source
- **Store Management Panel**: Organize your saved vector stores
- **Status Indicators**: Clear feedback on operations

```python
# Save a vector store when processing a document
save_vector_store_option = st.checkbox("💾 Save vector store for future use")

# Load from existing store
selected_store = st.selectbox("Load Previously Created Vector Store", saved_stores)
```

## 🚀 Advanced Usage

### Custom Prompt Template
The system uses a carefully crafted prompt template for accurate responses:

```python
prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context ONLY to answer the question.
If the context doesn't contain the answer, just say that you don't know based on the provided document.
Do not make up an answer or use external knowledge. Keep the answer relevant.

Context: {context}
Question: {question}
Answer:"""
```

### Caching
The application implements intelligent caching for:
- Ollama model initialization
- Vector store creation
- Document processing

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Ensure Ollama is running and model is installed
   ```
   **Solution**: Start Ollama service and pull required models

2. **Import Error for Document Types**
   ```
   ImportError: Library missing for .pdf files
   ```
   **Solution**: Install additional dependencies
   ```bash
   pip install pypdf 'unstructured[xlsx]'
   ```

3. **Memory Issues with Large Documents**
   - Reduce `chunk_size` parameter
   - Process smaller documents
   - Ensure sufficient RAM available

### Performance Optimization

- Use smaller, efficient models for faster responses
- Adjust chunk size based on document type
- Consider using GPU acceleration with Ollama

## 📝 File Structure

```
├── RAG-system-GUI.py          # Main application file
├── README.md                  # This file
└── requirements.txt           # Dependencies (optional)
```

## 🔧 Dependencies

Create a `requirements.txt` file:
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.10
faiss-cpu>=1.7.4
pypdf>=3.17.0
unstructured[xlsx]>=0.11.0
openpyxl>=3.1.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI model serving
- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Ensure Ollama is properly installed and running

---

**Made with ❤️ by [Vinay Kumar Balisetti]**
