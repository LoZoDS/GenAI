# 🧒 Early Childhood Development Milestones — RAG Indexing Pipeline

A data ingestion pipeline that scrapes and processes developmental milestone content from **UNICEF** and the **CDC**, chunks it into a structured knowledge base, and saves it as a JSON file ready for use in a Retrieval-Augmented Generation (RAG) system.

---

## 📋 Overview

This pipeline:
1. **Scrapes** UNICEF milestone pages (2 months → 2 years) using Selenium
2. **Extracts** milestone text from a CDC PDF checklist using PyMuPDF
3. **Cleans** both sources — removes navigation noise, URLs, bullet symbols, and boilerplate
4. **Chunks** all text using LangChain's `RecursiveCharacterTextSplitter`
5. **Outputs** a single `cdev_knowledge_base.json` with text chunks and metadata

---

## 🔧 Configuration

Key parameters at the top of `main.py`:

| Parameter       | Default | Description                              |
|----------------|---------|------------------------------------------|
| `CHUNK_SIZE`    | `500`   | Max characters per chunk                 |
| `CHUNK_OVERLAP` | `100`   | Overlapping characters between chunks   |
| `SLEEP`         | `3`     | Seconds to wait after page load          |

---

## 🗂️ Project Structure

```
.
├── app.py                                  # Streamlit chat UI
├── chat_service.py                         # RAG chain service
├── child_development_indexing_pipeline.py  # Main pipeline script
├── cdc-milestone-checklists-ltsae-english-508.pdf  # CDC Data Source
├── cdev_knowledge_base.json               # Output knowledge base (generated)
├── prompts.py                             # Prompt templates
├── rag_indexing.py                        # Embedding pipeline
├── rag_retrieval_chain.py                 # Retrieval chain
├── RAGAS_evaluation.py                    # Evaluation script
├── requirements.txt                       # Dependencies
├── safety.py                              # Safety filters
├── test_chat_service.py                   # Tests
├── sources/                               # Scraped UNICEF pages (generated)
│   ├── your-babys-developmental-milestones-2-months.txt
│   ├── your-babys-developmental-milestones-4-months.txt
│   ├── your-babys-developmental-milestones-6-months.txt
│   ├── your-babys-developmental-milestones-9-months.txt
│   ├── your-toddlers-developmental-milestones-1-year.txt
│   ├── your-toddlers-developmental-milestones-18-months.txt
│   └── your-toddlers-developmental-milestones-2-years.txt
└── README.md
```

---

## 📚 Data Sources

| Source | Ages Covered | Format |
|--------|-------------|--------|
| [UNICEF Parenting](https://www.unicef.org/parenting/child-development) | 2 months – 2 years | Web (scraped) |
| [CDC Act Early](https://www.cdc.gov/ncbddd/actearly) | 2 months – 5 years | PDF |

---

## 🧰 Dependencies

- `selenium`
- `beautifulsoup4` 
- `pymupdf` (`fitz`) 
- `langchain-text-splitters`
---

## 🚀 Running the Chatbot

The chatbot can be run locally in two steps: first the vector database must be created, and then the chat interface can be started.

### 1. Install dependencies

Make sure all required Python packages are installed:

```bash
py -m pip install -r sources/requirements.txt
```

If needed, install additional packages used in the chatbot interface and RAG pipeline:

```bash
py -m pip install streamlit langchain-huggingface langchain-chroma langchain-core langchain transformers sentence-transformers chromadb
```

### 2. Build the vector database

Before running the chatbot, the knowledge base must be indexed into the local Chroma vector database:

```bash
py rag_indexing.py
```

This reads the cdev_knowledge_base.json file, generates embeddings, and stores them in the chroma_db folder.

### 3. Test the backend (optional)
check whether the retrieval and response pipeline works correctly, run

```bash
py test_chat_service.py
```

This sends a sample question through the chatbot backend and prints the answer, sources, and status in the terminal.

### 4. Start the chatbot UI
Start the chatbot UI

```python
py -m streamlit run app.py
```

After that, open the local URL shown in the terminal, usually:
http://localhost:8501

---
## 💬 Chatbot Features
The chatbot interface includes:

- free-text questions about early childhood development
- suggested prompt buttons for common example questions
- a “Start new chat” function
- chat renaming for easier conversation management
- retrieved source display for each answer
- fallback responses for sensitive, diagnostic, urgent, or out-of-scope questions
---
## ⚠️ Disclaimer
This chatbot provides general information about early childhood development based on the available knowledge base. It does not diagnose developmental conditions and does not replace professional medical or developmental assessment.
