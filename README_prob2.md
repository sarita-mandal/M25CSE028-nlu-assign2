# Problem 2 - Character-Level Name Generation using RNN Variants

## Overview
This notebook trains and compares three character-level recurrent neural
architectures for generating full names:

| Model | Architecture |
|---|---|
| Vanilla RNN | Embedding → RNN → Dropout → Linear |
| Bidirectional LSTM (BiLSTM) | Embedding → BiLSTM + Forward LSTM → Linear |
| RNN + Bahdanau Attention | Embedding → RNN → Attention → Linear |

---

## Requirements

### Python packages

```bash
pip install torch numpy matplotlib
```

> **Python ≥ 3.8** and **PyTorch ≥ 1.12** are recommended.  
> No GPU is required — the notebook auto-detects CUDA and falls back to CPU.

### Running environment

The notebook is **Google Colab ready** and also runs in any standard
Jupyter environment (JupyterLab, VS Code, local Jupyter Notebook).

---

## How to Run

### Option A — Google Colab (recommended)

1. Upload `Prob2_Name_Generation.ipynb` to
   [colab.research.google.com](https://colab.research.google.com).
2. Optionally enable a GPU runtime:  
   **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all** (or press `Ctrl+F9`).

### Option B — Local Jupyter

```bash
# 1. Install dependencies
pip install torch numpy matplotlib jupyter

# 2. Launch Jupyter
jupyter notebook RNN_FullName_Generation_base_modified.ipynb

# 3. Inside the notebook: Kernel → Restart & Run All
```
---

## Outputs

After a full run the following files will be written to the working directory:

```
TrainingNames.txt       — 1001 deduplicated training names
rnn_generated.txt       — 500 names from Vanilla RNN
lstm_generated.txt      — 500 names from Bidirectional LSTM
attn_generated.txt      — 500 names from RNN + Attention
```

Plots are displayed inline inside the notebook (loss curves, bar charts,
length distribution histogram).

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding size | 64 |
| Hidden size | 256 |
| Layers | 2 |
| Dropout | 0.30 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 80 |
| LR schedule | StepLR (step=20, γ=0.5) |
| Generation temperature | 0.8 |
| Names generated per model | 500 |
