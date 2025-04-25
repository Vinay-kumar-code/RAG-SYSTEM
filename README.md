# 🧠 Ollama RAG QA System

A Retrieval-Augmented Generation (RAG) based Question Answering system powered by Ollama, FAISS, and LangChain. This tool enables users to query their own documents using local LLMs and embedding models, with support for `.txt`, `.pdf`, `.xls`, and `.xlsx` files.

---

## 🚀 Features

- 🔍 Load and parse documents from `.txt`, `.pdf`, `.xls`, or `.xlsx` formats
- 🧠 Uses Ollama's local models for embeddings and LLM inference
- ✂️ Splits documents into manageable chunks for vector storage
- 🔎 FAISS-based vector search for fast and relevant retrieval
- 🤖 Prompt-based QA with context-aware answering
- 📂 Returns top-k relevant document chunks with answers
- 💻 Fully local — no external API calls required

---

## 📦 Requirements

Install the required packages:

```bash
pip install langchain faiss-cpu pypdf "unstructured[xlsx]"
```

Make sure you have **Ollama** installed and running.

You should also install the required models:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

---

## 📁 Supported Document Types

- `.txt`
- `.pdf` (requires `pypdf`)
- `.xls` / `.xlsx` (requires `unstructured[xlsx]`)

---

## 🛠️ How It Works

1. User provides a document path via input.
2. The script:
   - Loads the document using an appropriate loader.
   - Splits the text into smaller chunks.
   - Creates embeddings using Ollama.
   - Stores them in a FAISS vector store.
   - Initializes an LLM to answer queries.
3. Users can interactively ask questions about their document.

---

## 💬 Usage

```bash
python rag_qa.py
```

You’ll be prompted to enter the path to your document. Once loaded and processed, you can enter questions based on its content.

To exit, type `exit` or `bye`.

---

## 📸 Example Interaction

```text
Enter the path to your document (Supported File formats: txt, pdf, xls, xlsx): ./example.pdf
...

--- Ready to Query ---
Enter 'exit' or 'bye' to quit.

Enter your query: What is the main topic of the first section?

--- Answer ---
The first section discusses ...
```

---

## ⚠️ Troubleshooting

- Ensure Ollama is running before starting the script.
- Verify the models (`gemma3:1b`, `nomic-embed-text:latest`) are installed.
- For Excel support, install the extended unstructured package:
  ```bash
  pip install "unstructured[xlsx]"
  ```

---

## 📚 Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [PyPDF](https://pypi.org/project/pypdf/)
---

## 📄 License

This project is licensed under the GNU General Public License.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome! Please open an issue to discuss changes or improvements.

