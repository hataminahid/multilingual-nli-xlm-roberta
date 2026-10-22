#  Multilingual Natural Language Inference with XLM-RoBERTa

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hataminahid/multilingual-nli-xlm-roberta/blob/main/notebooks/multilingual_nli_xlm_roberta.ipynb)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-ee4c2c)
![Task](https://img.shields.io/badge/task-NLI%20(3--class)-green)


Given a **premise** and a **hypothesis** in one of **15 languages**, predict whether the hypothesis is
**entailed by**, **neutral to**, or **contradicts** the premise.
Dataset: Kaggle [*Contradictory, My Dear Watson*](https://www.kaggle.com/competitions/contradictory-my-dear-watson) (12,120 training / 5,195 test pairs).

The project compares a **fine-tuned multilingual transformer (XLM-RoBERTa-base)** against **classical ML models on hand-crafted lexical features**.

| Model | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC |
|---|--:|--:|--:|--:|
| **XLM-RoBERTa-base** | **0.696** | **0.695** | **0.695** | **0.859** |
| Random Forest | 0.480 | 0.466 | 0.453 | 0.641 |
| XGBoost | 0.463 | 0.459 | 0.451 | 0.639 |
| Logistic Regression | 0.460 | 0.446 | 0.432 | 0.621 |
| SVM (LinearSVC) | 0.467 | 0.445 | 0.428 | 0.621 |

![Model comparison](images/07_final_model_comparison.png)

---

##  Repository structure

```
multilingual-nli-xlm-roberta/
├── notebooks/
│   └── multilingual_nli_xlm_roberta.ipynb   # full pipeline (Phase I → III)
├── data/                                    # put Kaggle CSVs here (see data/README.md)
├── outputs/                                 # submission.csv & saved model are written here (git-ignored model)
├── images/                                  # figures used in this README
├── requirements.txt
└── README.md
```

##  Getting started

### Google Colab (recommended — needs a GPU)
1. Click the **Open in Colab** badge, then *Runtime → Change runtime type → T4 GPU*.
2. Run the setup cell; upload the Kaggle zip when asked.
3. Run all cells. (Fine-tuning with 3-fold CV over 3 configs takes a while on a T4.)

### Local

```bash
git clone https://github.com/hataminahid/multilingual-nli-xlm-roberta.git
cd multilingual-nli-xlm-roberta
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download train.csv, test.csv, sample_submission.csv from Kaggle into data/
jupyter notebook notebooks/multilingual_nli_xlm_roberta.ipynb
```

--

##  Methodology

### Phase I — EDA & feature engineering
- No missing values, no duplicate rows / pairs / IDs. Labels are well balanced (4176 / 3880 / 4064); English dominates the languages (6,870 of 12,120 rows), the other 14 languages have roughly 340–410 each.
- Premises average 18 words, hypotheses 9 words (right-skewed).
- **Preprocessing:** Unicode NFC normalisation, zero-width character removal, whitespace cleaning; lowercasing **only for Latin-script languages**. Negation words are deliberately kept.
- **Hand-crafted features (7):** `word_overlap`, `jaccard`, `tfidf_cos` (per-language TF-IDF cosine similarity), `len_diff`, `neg_p`, `neg_h`, `neg_diff`.

![Labels](images/01_label_distribution.png)
![Languages](images/02_language_distribution.png)
![Features](images/03_key_features_by_label.png)

Entailment pairs show higher lexical overlap, but the features are highly inter-correlated (> 0.84) and only weakly related to the label, i.e. surface overlap alone is not enough.

### Phase II — XLM-RoBERTa fine-tuning
- `xlm-roberta-base`, max length 128, AdamW (weight decay 0.01), linear schedule with 10 % warm-up, gradient clipping 1.0, plain cross-entropy (classes are balanced), seed 42.
- **Stratified 3-fold CV** over a small grid:

| lr | batch | epochs | mean macro-F1 (3 folds) |
|--:|--:|--:|--:|
| **2e-5** | 16 | **3** | **0.696 ± 0.007** |
| 3e-5 | 16 | 3 | 0.580 ± 0.161 |
| 2e-5 | 16 | 4 | 0.548 ± 0.216 |

  (The two weaker settings are unstable — in some folds training did not converge.)
- Final model: best config retrained on the full training set; test predictions are written to `outputs/submission.csv`.
- Evaluation on the last CV fold (4,040 held-out pairs): accuracy 0.70, macro-F1 0.70, macro ROC-AUC 0.859. Per class F1: Entailment 0.73, Neutral 0.65, Contradiction 0.71.

![Confusion matrix](images/04_xlmr_confusion_matrix.png)
![ROC](images/05_xlmr_roc_curves.png)

### Phase III — Traditional ML baselines
Logistic Regression, Random Forest, LinearSVC (calibrated) and XGBoost on the 7 features above, tuned with `RandomizedSearchCV` (5-fold, macro-F1); metrics come from 5-fold out-of-fold predictions. All stay around 0.43–0.45 macro-F1, with a strong tendency to over-predict *Neutral* and under-detect *Contradiction* (recall ≈ 0.22–0.31). Word overlap is the most important feature (~24 %).

![RF importance](images/06_rf_feature_importance.png)

**Conclusion:** lexical-similarity features cannot capture the semantic relation between sentences; a multilingual transformer improves macro-F1 from ≈ 0.45 to ≈ 0.70.

---

##  Notes & limitations
- The XLM-R row in the comparison table is measured on **one held-out CV fold** (model retrained on the other two folds), while the classical models use 5-fold out-of-fold predictions on all training data. The CV mean (0.696 ± 0.007) is the more stable estimate. Hyper-parameters were selected on the same folds, so the numbers can be slightly optimistic.
- No official Kaggle leaderboard score is reported here.
- English is over-represented in training, so performance on other languages may differ (per-language metrics were not computed).
- The trained model (~1 GB) is not stored in the repo; it is saved to `outputs/xlmr_nli_model/` when you run the notebook.

##  Tech stack
Python · PyTorch · Hugging Face Transformers · scikit-learn · XGBoost · pandas · NumPy · matplotlib · seaborn · missingno

##  License & data
Add a license of your choice (e.g. MIT) as `LICENSE`. The data belongs to Kaggle / the competition organisers — follow the competition rules.
