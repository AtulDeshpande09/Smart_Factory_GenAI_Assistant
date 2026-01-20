# Industrial RAG System for Machine Maintenance Analysis

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to analyze structured industrial data such as machine maintenance logs, incident reports, production reports, and SOPs (Standard Operating Procedures).

Unlike generic chatbot-style RAG demos, this system focuses on **data correctness, auditability, and retrieval reliability**. The design prioritizes retrieval validation before generation and avoids unnecessary complexity.

---

## Key Features

* **Structured Data Ingestion** from JSONL files
* **Metadata-Preserving Vector Storage** using ChromaDB
* **Retrieval-First Design** (retrieval verified independently)
* **Strict Hallucination Control** via prompt constraints
* **Deterministic & Auditable Outputs**
* **Terminal-Based Workflow** (no notebooks required)

---

## Dataset Structure

The dataset is split into four JSONL files, each representing a distinct document type:

* `logs.jsonl` — Machine maintenance logs
* `incident.jsonl` — Incident and anomaly descriptions
* `report.jsonl` — Daily production reports
* `sop.jsonl` — Standard Operating Procedures

Each record follows this structure:

```json
{
  "id": "unique-id",
  "document_type": "maintenance_log | incident | production_report | sop",
  "text": "RAG-optimized natural language content",
  "metadata": {
    "machine_id": "optional",
    "date": "optional",
    "severity": "optional",
    "system": "optional"
  }
}
```

---

## System Architecture

```
JSONL Files
   ↓
Custom Loader (preserves metadata)
   ↓
SentenceTransformer Embeddings
   ↓
Chroma Vector Database
   ↓
Similarity Search (Retrieval)
   ↓
Prompt Construction
   ↓
Local LLM (Generation)
```

Retrieval can be executed independently without invoking the language model.

---

## Technologies Used

* **Python 3.10+**
* **LangChain** (document abstraction & prompt templates)
* **ChromaDB** (vector storage)
* **SentenceTransformers** (embeddings)
* **Hugging Face Transformers** (local LLM)
* **PyTorch**

---

## Usage

### 1. Ingest Data

```bash
python db.py
```

This loads all JSONL files, embeds documents, and persists vectors in the `chroma/` directory.

---

### 2. Retrieval-Only Testing

```bash
python test.py
```

This script performs similarity search and saves results to a text file for inspection.

---

### 3. Retrieval + Generation

```bash
python main.py
```

The system retrieves relevant documents and generates an answer **strictly grounded in retrieved context**.

If no answer is present in the data, the system responds with:

```
No root cause found in available documents.
```

---

## Design Decisions

### Why RAG instead of Fine-Tuning?

* Data is frequently updated
* Requires traceability and explainability
* Lower cost and easier iteration

### Why No Reranking?

* Small, clean dataset
* Strong embeddings
* Metadata-driven retrieval sufficient

### Why No Chain-of-Thought (yet)?

* Avoids masking retrieval errors
* Keeps outputs concise and auditable

