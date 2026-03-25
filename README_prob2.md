# Character-Level Full Name Generation using RNN Variants

**CSL 7640 — Natural Language Understanding | Assignment-2, Problem 2**

This notebook trains and compares three character-level recurrent neural
architectures for generating Indian full names (first name + surname):

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

1. Upload `RNN_FullName_Generation_base_modified.ipynb` to
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

## Notebook Structure

Each cell is labelled (`CELL 1`, `CELL 2`, …) and must be run **in order**.

| Cell | Description |
|------|-------------|
| **CELL 1** | Imports (`torch`, `numpy`, `matplotlib`, etc.) and random seed setup. Sets `device` to CUDA or CPU. |
| **CELL 2** | Defines the raw list of 1 000 Indian full names. De-duplicates and saves `TrainingNames.txt`. |
| **CELL 3** | Builds the character vocabulary (size ≈ 30). Defines encoding/decoding helpers and performs the 90/10 train/validation split. |
| **CELL 4** | Defines `VanillaRNN` — includes `count_parameters()` and `model_size_mb()` utilities. |
| **CELL 5** | Defines `BidirectionalLSTM` with a paired forward-only inference head (`step()`). |
| **CELL 6** | Defines `BahdanauAttention` and `RNNWithAttention`. |
| **CELL 7** | Sets hyperparameters, instantiates all three models, and prints a parameter/size summary table. |
| **CELL 8** | Defines training utilities: `make_batches`, `train_model`, `train_blstm`, `compute_val_loss`. |
| **CELL 9** | **Trains all three models** for 80 epochs. Prints Vanilla RNN parameter count and model size (MB) from the saved best checkpoint after training. |
| **CELL 10** | Plots training and validation loss curves for all models. |
| **CELL 11** | Defines `generate_full_name` and `generate_names_batch`. |
| **CELL 12** | Generates 500 names per model (temperature = 0.8) and saves them to `rnn_generated.txt`, `lstm_generated.txt`, `attn_generated.txt`. |
| **CELL 13** | Defines evaluation metrics: novelty rate, diversity, average length, bigram entropy, regional consistency. Runs evaluation and prints results. |
| **CELL 14** | Plots bar charts comparing all five metrics across the three models. |
| **CELL 15** | Prints 20 random sample names from each model. |
| **CELL 15b** | Plots name-length distribution (training vs. all generated sets). |

---

## Outputs

After a full run the following files will be written to the working directory:

```
TrainingNames.txt       — 990 deduplicated training names
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

To experiment, edit the constants at the top of **CELL 7** before running.

---

## Expected Runtime

| Device | Approximate time |
|--------|-----------------|
| CPU (modern laptop) | 15–25 minutes |
| GPU (Colab T4) | 3–5 minutes |

---

## Reproducibility

A fixed random seed (`SEED = 42`) is applied to `torch`, `numpy`, and
`random` at the start of CELL 1. Re-running from scratch will produce
the same model weights and generated names.
