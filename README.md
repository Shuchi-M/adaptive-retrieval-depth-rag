# Adaptive Retrieval Depth Optimization in RAG

**COMPSCI 651: Optimization for Machine Learning : Final Project (Spring 2026)**  
Shuchi Mishra

---

## Overview

This repository contains the full experiment notebook for the paper:

> *Adaptive Retrieval Depth Optimization in Retrieval-Augmented Generation*

Standard RAG systems fix retrieval depth `k` as a hyperparameter, applying the same number of retrieved passages to every query regardless of its information needs. This project treats `k` as a **per-query optimization variable** and proposes a **confidence-based adaptive stopping rule** that requires no model fine-tuning and no task-specific supervision.

The stopping rule is simple: retrieve passages incrementally from `k = 1` upward, and halt when the marginal gain in the generator's answer confidence (mean token log-probability) falls below a threshold `τ`:

```
Stop if: conf(k) − conf(k−1) < τ
```

The cost-aware objective that guides evaluation is:

```
L(k) = E(k) + λ · C(k)
```

where `E(k)` is the error rate at depth `k`, `C(k) = avg_k / max_k` is the normalized retrieval cost, and `λ ≥ 0` controls the accuracy–efficiency tradeoff.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Project_final_workflow_2.ipynb` | Complete experiment notebook — all models, datasets, plots, and results |
| `README.md` | This file |

---

## Reproducing Results

All experiments were run on a **T4 GPU** (Google Colab). Expected runtime: ~45–60 minutes for both models.

### Notebook Structure

| Cell | What it does |
|------|-------------|
| 0 | Install dependencies |
| 1–2 | Imports and device setup (CUDA / CPU) |
| 3–4 | Toy corpus (25 passages, 20 questions) and evaluation utilities |
| 5–6 | Generation and confidence scoring functions; toy BM25 index |
| 7 | Load SQuAD (100 questions) and TriviaQA (100 questions) |
| 8 | Build retrieval corpora (100 passages + 10 toy distractors each) |
| 9 | `run_full_evaluation()` - fixed-k baselines, confidence scoring, noise analysis, adaptive threshold sweep, cost-aware objective — for one model |
| 10 | Run evaluation for both `flan-t5-small` and `flan-t5-base` |
| 11 | Cross-model comparison summary table |
| 12 | Generate all plots (noise rate, confidence signal, Pareto frontier, k-distribution) |
| 13 | Complete final results summary |

### Key implementation detail : confidence scoring

For encoder-decoder models (T5 family), naive confidence scoring using `generate()` outputs directly produces incorrect log-probabilities. This notebook uses the correct approach:

```python
decoder_input_ids = model._shift_right(generated_ids)
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    decoder_input_ids=decoder_input_ids
)
# score each generated token against its decoder context
```

---

## Key Results

### Fixed-k vs. Adaptive (Contains-Gold accuracy, n=100 per cell)

| Model | Dataset | Fixed k=1 | Fixed k=2 | Fixed k=3 | Fixed k=5 | **Adaptive** | avg\_k |
|-------|---------|-----------|-----------|-----------|-----------|-------------|--------|
| flan-t5-small | SQuAD | 0.32 | 0.28 | 0.27 | 0.25 | **0.35** (τ=0.03) | 1.19 |
| flan-t5-small | TriviaQA | 0.18 | 0.17 | 0.16 | 0.16 | **0.19** (τ=0.10) | 1.24 |
| flan-t5-base | SQuAD | 0.40 | 0.46 | 0.44 | 0.44 | **0.46** (τ=0.00) | 1.60 |
| flan-t5-base | TriviaQA | 0.24 | 0.22 | 0.22 | 0.23 | 0.22 (τ=0.10) | 1.21 |

### Retrieval recall and noise

| Dataset | Recall @1 | Recall @3 | Recall @5 | Noise rate (k=1) | Noise rate (k=5) |
|---------|-----------|-----------|-----------|-----------------|-----------------|
| SQuAD | 0.74 | 0.89 | 0.94 | 26.0% | 74.4% |
| TriviaQA | 0.84 | 0.88 | 0.90 | 16.0% | 61.2% |

### Summary

- **flan-t5-small / SQuAD**: Adaptive is Pareto dominant : higher CG (0.35 vs 0.32) at lower cost (avg\_k = 1.19)
- **flan-t5-base / SQuAD**: Adaptive matches Fixed k=2 accuracy (CG = 0.46) at avg\_k = 1.60 vs. 2.00; Pareto dominant at all λ > 0
- **flan-t5-small / TriviaQA**: Directional cost-efficiency gain (+0.01 CG, reduced avg\_k)
- **flan-t5-base / TriviaQA**: Failure case : confidence signal is inverted (increases with k), undermining the stopping rule

---

## Setup

### Dependencies

```
torch
transformers
datasets
rank_bm25
pandas
numpy
matplotlib
```

### Install

```bash
pip install datasets transformers rank_bm25 pandas numpy matplotlib
```

---

## Experimental Configuration

| Setting | Value |
|---------|-------|
| Retriever | BM25Okapi (`rank_bm25`) |
| Generators | `google/flan-t5-small` (77M), `google/flan-t5-base` (248M) |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=20` |
| Passage truncation | 45 words per passage (512-token input limit) |
| Eval datasets | SQuAD validation, TriviaQA rc.wikipedia |
| Questions per dataset | 100 unique-passage questions |
| Corpus size | 110 passages (100 dataset + 10 toy distractors) |
| k values evaluated | {1, 2, 3, 5} |
| τ sweep | {−0.05, −0.02, 0.00, 0.01, 0.03, 0.05, 0.10} |
| λ values | {0.0, 0.25, 0.5, 1.0} |
| Hardware | T4 GPU (Google Colab) |


---

## Related Work

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Jiang et al. (2023). *Active Retrieval Augmented Generation (FLARE).* EMNLP.
- Jeong et al. (2024). *Adaptive-RAG.* NAACL.
- Asai et al. (2023). *Self-RAG.* arXiv:2310.11511.
- Shi et al. (2023). *Large language models can be easily distracted by irrelevant context.* ICML.
