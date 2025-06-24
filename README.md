# Chinese Wikipedia Document Retrieval and Stance Classification System

A modular system for document retrieval, key sentence extraction, and sentence-level stance classification for user-provided statements, built on a local Chinese Wikipedia corpus.
**Built with LangChain, supporting classical (BM25) and neural (BERT-based) methods, and designed for easy benchmarking, extension, and research.**

---

## 📚 Project Overview

- **Input:** A user statement (e.g., "The Han Dynasty lasted over 400 years.")
- **Output:**
  - Top relevant documents (with retrieval scores)
  - Key sentences from those documents (with scores)
  - Stance label (*supportive*, *non-supportive*, or *neutral*) for each key sentence

---

## 🗂️ System Architecture
```
User Statement
│
├──► Document Retrieval
│   ├── BM25, Dense, Hybrid → `retrievers/`
│   └── Evaluation (NDCG@n) → `evaluation/`
│
├──► Sentence Retrieval (from top documents)
│   ├── Sentence-level extraction → `pipeline.py`, `utils/`
│   └── Evaluation (F1) → `evaluation/`
│
├──► Sentence Classification
│   ├── Supportive / Non-supportive / Neutral → `rerankers/`
│   └── Evaluation (F1) → `evaluation/`
│
├──► (Optional) Search Agent
│   ├── Iterative retrieval/classification if confidence is low
│   └── (Under development or in `scripts/`)
│
├──► Experiments & Analysis
│   └── Notebooks → `notebooks/`, `visualize_result.ipynb`
│
└──► Entry Point
    └── Main script → `main.py`
```
---

## 🚧 TODO List

### 1. **Document Retrieval Pipeline**
   - [x] Implement BM25 retriever
   - [x] Implement text embedding retriever (e.g., BERT/SBERT/BGE/QWEN3)
   - [x] (Optional) Implement hybrid retriever
   - [x] Compare retrievers on same set of queries

### 2. **Document Retrieval Evaluation**
   - [x] Prepare/query gold standard relevance data for statements
   - [x] Implement NDCG@n and HitRate@n
   - [x] Log and visualize retrieval performance

### 3. **Sentence Retrieval Pipeline**
   - [X] Extract most relevant sentences from top documents (similarity, keyword overlap, etc.)
   - [X] (Optional) Experiment with LLM-based sentence scoring

### 4. **Sentence Retrieval Evaluation**
   - [ ] Prepare gold standard (statements with relevant key sentences labeled)
   - [ ] Compute F1 for sentence retrieval

### 5. **Sentence Classification Pipeline**
   - [ ] Implement rule-based classifier (supportive/non-supportive/neutral)
   - [ ] Upgrade to ML/LLM-based classifier (zero-shot or fine-tuned)
   - [ ] Standardize output: label, confidence, etc.

### 6. **Sentence Classification Evaluation**
   - [ ] Prepare gold standard labeled data
   - [ ] Compute F1, confusion matrix, example outputs

### 7. **(Optional) Search Agent / Iterative Pipeline**
   - [ ] Build an agent that checks classification/retrieval confidence
   - [ ] If confidence is low, reformulate query and repeat retrieval/sentence extraction up to N times
   - [ ] Document agent stopping criteria and performance

### 8. **Full Pipeline Evaluation**
   - [ ] Define a strict “success”:
        1. Document retrieval hits at least one relevant document
        2. Sentence retrieval finds at least one key sentence
        3. Sentence classification is correct
   - [ ] Compute F1 and other metrics for end-to-end pipeline

---

## 📝 Data

- **Corpus:** Local files from Chinese Wikipedia (e.g., .txt, .json, or .csv format; details TBD)
- **Gold Standards:**
  - Statement–relevant document pairs (for document retrieval)
  - Statement–key sentence pairs (for sentence retrieval)
  - Labeled sentences (supportive/non-supportive/neutral) for stance classification
  - For strict pipeline evaluation: all three criteria must be met for a prediction to count as correct

---

## 🧪 Evaluation Metrics

- **Document Retrieval:** NDCG@n
- **Sentence Retrieval:** F1
- **Sentence Classification:** F1, confusion matrix
- **Full Pipeline:** Strict F1 (all three steps must be correct for a success)

---

## 🚀 Getting Started

> _Instructions will be added as code is developed._

---

## 🧪 Example Usage

> _To be filled in after initial demo is ready._

---

## 📊 Results

> _Track performance of each module_

---

## 📅 Project Log

- [X] Day 1: Project initialization, README, pipeline design
- [x] Day 2: Qwen Retriever implementation, evaluation
- [x] Day 3: Qwen Reranker and modular evaluation (Retriever+Reranker). Qwen Retriever with FAISS vector database. Set pre-commit.
- [x] Day 4: Finish Sentence Retrieval, Adjust retriever for better modularity.
- [ ] Day 5: Optimize code with DeepSeed support, Sentence Retrieval Evaluation and Sentence Reranker

---

## 💡 Feedback and Questions

- Suggestions for data format, evaluation, or module design are welcome!
- Currently using a local Chinese Wikipedia corpus; please note if you want to adapt for a different domain or language.
