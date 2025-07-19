# RAG Tutorial

A simple Retrieval-Augmented Generation (RAG) system using LangChain, Ollama, and ChromaDB for document-based Q&A.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally ([Ollama installation guide](https://ollama.com/download))
- **PDF files** in the `data/` directory for ingestion

---

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Tornshorts/RAG-Tutorial.git
   cd rag1
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Start Ollama:**
   ```sh
   ollama serve
   ```

---

## Project Structure

```
rag tuitorial/
│
├── data/                    # Place your PDF files here
├── main.py                  # Main entry point for the system
├── ask_question.py          # Q&A loop
├── load_data.py             # Loads and splits documents, populates ChromaDB
├── get_embedding_function.py# Embedding function for vector DB
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Usage Instructions

### 1. Prepare Data

- Place your PDF files in the `data/` folder.

### 2. (Optional) Reset Database

- To clear the ChromaDB before loading new data:
  ```sh
  python load_data.py --reset
  ```

### 3. Load Data

- Run the main script to ingest documents and start the Q&A session:
  ```sh
  python main.py
  ```

### 4. Ask Questions

- Type your questions at the prompt.
- Type `exit` to quit.

---

## Example Session

```
🚀 Starting RAG System...

📚 Loading and processing documents...
➕ Adding 3 new documents...

🤖 Starting Q&A session...
🎯 Ready! Ask your questions (type 'exit' to quit):

Ask a question (or type 'exit'): What is the summary of document X?
🧠 Answer: [Model-generated answer]

Ask a question (or type 'exit'): exit
👋 Goodbye!
```

---

## Configuration Options

- **Model Selection:**  
  Change the model in `ask_question.py` or `main.py`:

  ```python
  llm = OllamaLLM(model="llama3.2:latest")
  ```

  Replace `"llama3.2:latest"` with any model available in your Ollama instance.

- **Embedding Model:**  
  Change the embedding model in `get_embedding_function.py`:

  ```python
  OllamaEmbeddings(model="nomic-embed-text:latest")
  ```

- **ChromaDB Path:**  
  Change the database path in scripts:

  ```python
  CHROMA_PATH = "chroma"
  ```

- **Base URL for Ollama:**  
  If Ollama runs on a different host/port, set `base_url` in `get_embedding_function.py`:
  ```python
  OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
  ```

---

## Troubleshooting

- **Ollama not running:**  
  Make sure `ollama serve` is running before starting the scripts.

- **No documents found:**  
  Ensure your PDFs are in the `data/` directory.

- **Dependency issues:**  
  Run `pip install -r requirements.txt` to install all required packages.

---

##
