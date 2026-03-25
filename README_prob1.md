# Problem 1 — Learning Word Embeddings from IIT Jodhpur Data

## Overview

The notebook trains and evaluates Word2Vec models on a domain-specific corpus built from IIT Jodhpur web sources. It covers four tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Corpus ingestion, preprocessing, word cloud generation |
| **Task 2** | Training CBOW and Skip-gram models across 3 hyperparameter configs (6 models total) |
| **Task 3** | Semantic evaluation — top-5 nearest neighbors + word analogy experiments |
| **Task 4** | 2D visualisation of embeddings using PCA and t-SNE |

---

## Prerequisites

### Platform
The notebook is designed to run on **Google Colab**. It uses `google.colab.files.upload()` for corpus ingestion, so it will **not** work directly as a local Jupyter notebook without modifying that one cell.

### Python Version
Python 3.8 or higher (Colab default satisfies this).

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `numpy` | ≥ 1.21 | Pre-installed on Colab |
| `matplotlib` | ≥ 3.4 | Pre-installed on Colab |
| `scikit-learn` | ≥ 0.24 | Pre-installed on Colab |
| `pandas` | ≥ 1.3 | Pre-installed on Colab |
| `wordcloud` | ≥ 1.8 | **Installed by the notebook** (`pip install wordcloud`) |

---

## Setup

### Step 1 — Open the Notebook in Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com), click **File → Upload notebook**, and upload `Prob1_Word2Vec.ipynb`.

Alternatively, if the notebook is in Google Drive: **File → Open notebook → Google Drive**.

### Step 2 — Prepare Your Corpus File

You must supply a plain-text file named **exactly** `corpus_raw.txt`.

### Step 3 — Select Runtime (Optional but Recommended)

In Colab: **Runtime → Change runtime type → Hardware accelerator: None** (CPU is sufficient; GPU offers no benefit for this NumPy implementation).

---

## How to Run

### Quick Start — Run All Cells

Go to **Runtime → Run all** (`Ctrl+F9`).

The notebook will pause at the 'File Picker' cell and display a file upload dialog. Upload your `corpus_raw.txt` file there and execution will resume automatically.

---

## Outputs

After a full run, the following files will be present in the Colab session storage:

| File | Description |
|------|-------------|
| `iitj_clean_corpus.txt` | Tokenised corpus, one sentence per line |
| `w2v_cbow_best.npz` | Config B CBOW model weights (NumPy archive) |
| `w2v_skipgram_best.npz` | Config B Skip-gram model weights (NumPy archive) |

---

## Hyperparameter Configurations

Three configurations are trained for both CBOW and Skip-gram:

| Config | Embedding Dim | Window Size | Negative Samples | Notes |
|--------|:---:|:---:|:---:|-------|
| **A** | 50 | 3 | 5 | Fast; lower capacity |
| **B** | 100 | 5 | 10 | **Best overall** — selected as final model |
| **C** | 300 | 9 | 20 | Highest capacity; slowest to train |

**Why Config B is best:** It achieves the highest composite score across four evaluation metrics — intra-cluster semantic coherence, analogy Mean Reciprocal Rank, average top-1 neighbor similarity, and probe word vocabulary coverage — striking the optimal balance between embedding expressiveness and corpus size constraints (~82K tokens).
