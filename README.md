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
├── child_development_indexing_pipeline.py                          # Main pipeline script
├── cdc-milestone-checklists-ltsae-english-508.pdf                  # CDC Data Source
├── cdev_knowledge_base.json                                        # Output knowledge base (generated)
├── your-babys-developmental-milestones-2-months.txt                # Scraped UNICEF pages (generated)
├── your-babys-developmental-milestones-4-months.txt
├── your-babys-developmental-milestones-6-months.txt
├── your-babys-developmental-milestones-9-months.txt
├── your-toddlers-developmental-milestones-1-year.txt
├── your-toddlers-developmental-milestones-2-years.txt
├── your-toddlers-developmental-milestones-18-months.txt
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

- `selenium` — browser automation for UNICEF scraping
- `beautifulsoup4` — HTML parsing and text extraction
- `pymupdf` (`fitz`) — CDC PDF text extraction
- `langchain-text-splitters` — recursive text chunking

---
