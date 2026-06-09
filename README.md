# Endodontic Outcome Prediction — Part 2: Transformer-Based & Classical ML with Text Features

This repository contains the text-based and hybrid experiments from my research on predicting endodontic treatment outcomes from clinical data. Two complementary approaches are implemented:

1. **Transformer fine-tuning** — DistilBERT, RoBERTa, and Danish-BERT fine-tuned end-to-end on clinical text
2. **Classical ML with token embeddings** — frozen DistilBERT embeddings concatenated with tabular features, evaluated with Logistic Regression, Naive Bayes, Random Forest, and XGBoost

> **Data availability:** Patient-level data are not included in this repository in accordance with data privacy requirements. See [Data](#data) below for the expected file layout.



## Notebook 1 — Transformer Fine-Tuning

`transformer_text_models.ipynb`

Five model configurations are compared across two input strategies: translated prognosis notes only vs. prognosis notes combined with signs and symptoms, with an additional Danish-BERT run on the original untranslated Danish text.

| Model | Input text | Checkpoint |
|---|---|---|
| DistilBERT | Translated prognosis note | `distilbert/distilbert-base-uncased` |
| RoBERTa | Translated prognosis note | `roberta-base` |
| DistilBERT (all texts) | Prognosis + signs & symptoms | `distilbert/distilbert-base-uncased` |
| RoBERTa (all texts) | Prognosis + signs & symptoms | `roberta-base` |
| Danish-BERT | Original untranslated Danish text | `Maltehb/danish-bert-botxo` |

All models are fine-tuned with **class-weighted focal loss** (γ = 2, full inverse-frequency weighting) to address class imbalance. Decision thresholds are selected on the validation set by maximising minority-class F1.

**Sections**
1. Setup & Configuration
2. Data Preparation & Train/Val/Test Split
3. Dataset Loading & Tokenisation
4. Shared Utilities (FocalLoss, FocalTrainer, threshold selection)
5–9. One section per model experiment
10. ROC Curve Visualisation
11. Bootstrap Confidence Intervals (2 000-sample, 95 % CI)
12. Interpretability: Attention & Gradient Saliency

---

## Notebook 2 — Classical ML with Token Embeddings

`ml_token_embeddings.ipynb`

Frozen DistilBERT embeddings (mean-pooled from the prognosis text) are concatenated with encoded tabular clinical features to form a single feature matrix. Four classifiers are then evaluated using repeated nested cross-validation with threshold optimisation.

| Classifier | Hyperparameter search |
|---|---|
| Logistic Regression | C ∈ {0.0001, 0.01, 0.1, 1} |
| Gaussian Naive Bayes | — |
| Random Forest | max\_depth, min\_samples\_split, min\_samples\_leaf |
| XGBoost | max\_depth, learning\_rate, n\_estimators |

Evaluation: 5 outer folds × 5 inner folds × 5 trials, with `TunedThresholdClassifierCV` applied on each training fold. Metrics: ROC-AUC, PR-AUC, macro-F1, precision, recall (mean ± std).

**Sections**
1. Dependencies
2. Configuration & Utilities
3. Data Loading & Cleaning
4. Feature Engineering (tabular encoding + Chi² importance)
5. Text Embeddings (DistilBERT mean pooling)
6. Combined Feature Matrix
7. Model Evaluation — Nested CV
8. ROC Curve Visualisation

---

## Data
Not avaialbe publically due to ethical aspects.
```

## Requirements

```
torch>=2.0
transformers>=4.40
datasets>=2.18
scikit-learn>=1.4
imbalanced-learn>=0.12
xgboost>=2.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
nltk>=3.8
evaluate>=0.4
tqdm
```

A CUDA-capable GPU is strongly recommended for Notebook 1 (transformer fine-tuning). Notebook 2 uses frozen embeddings and runs on CPU, though GPU speeds up the embedding extraction step.
