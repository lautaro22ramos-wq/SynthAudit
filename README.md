# SynthAudit

A synthetic financial transaction dataset generator with injected fraud patterns, used to train and evaluate a machine learning fraud detection model.

---

## Objective

Most public fraud datasets are either too small, heavily anonymized, or stripped of the behavioral context that makes fraud detection interesting. This project solves that by generating a realistic dataset from scratch — with full control over the fraud patterns injected — and using it to train a classifier that can detect those patterns.

The goal is to demonstrate an end-to-end Risk Analytics workflow: from data design to model evaluation.

---

## Project Structure
SynthAudit//
├── data/
│   ├── raw/               # Generated transaction data (not tracked by Git)
│   └── processed/         # Preprocessed features for modeling
├── notebooks/
│   ├── 01_eda_and_patterns.ipynb       # Exploratory analysis and fraud visualization
│   └── 02_fraud_detection_model.ipynb  # Model training and evaluation
├── outputs/
│   └── figures/           # All generated charts
├── references/            # Academic references
├── src/
│   └── data_generator.py  # Synthetic data generation script
├── .gitignore
└── README.md

---

## Why Synthetic Data

This is the central question of the project. Generating synthetic data instead of using a public dataset was not just a practical decision — it was the primary research interest. The goal was to test whether artificially generated data, designed with explicit behavioral assumptions, can effectively train a machine learning model to detect patterns that mirror real-world fraud.

This matters beyond this project. If synthetic data can reliably feed a fraud detection model, the same methodology can be extended to train specialized bots capable of performing automated risk analysis on demand — without depending on sensitive or restricted real-world datasets. This project is a first validation of that pipeline.

The immediate tradeoff is that synthetic patterns are structurally cleaner than real-world data, which explains the near-perfect model scores. That is expected and documented. The more important result is that the pipeline works: data designed with intent produces a model that learns with intent.

---

## Fraud Patterns

Three patterns were injected, each reflecting a documented real-world fraud behavior:

**Unusual hour transactions** — Fraudulent activity concentrated between 1 AM and 4 AM, paired with elevated amounts and high-risk country codes. Legitimate cardholders rarely transact at those hours, which makes time of day a meaningful signal.

**Threshold avoidance** — Amounts set just below common detection thresholds ($499, $999, $4,999). This mimics a technique known as structuring, where fraudsters deliberately keep transactions under automated alert limits to avoid triggering reviews.

**New merchant spikes** — Large transactions from merchant entities that do not appear in the legitimate transaction history. Unfamiliar merchants with no prior relationship are a recognized risk signal in card fraud.

The dataset was kept at 5% fraud rate — consistent with industry benchmarks for card-present fraud — to preserve the class imbalance that makes detection genuinely difficult.

---

## Methodology

**Data generation** — `src/data_generator.py` generates 950 legitimate and 50 fraudulent transactions. Legitimate transactions follow realistic hour distributions, amount ranges per merchant category, and a 90% domestic / 10% international country split. Fraud transactions are generated separately per pattern and merged with the legitimate set before shuffling.

**Exploratory analysis** — `notebooks/01_eda_and_patterns.ipynb` visualizes each fraud pattern against the legitimate baseline. The goal is to confirm that the injected patterns are visible in the data before attempting to model them.

**Feature engineering** — Categorical variables (`merchant`, `category`, `country`) are encoded numerically. The `transaction_id`, `timestamp`, and `fraud_type` columns are dropped — the model should not have access to the fraud label or any derived identifier.

**Model selection** — Random Forest was chosen over simpler models because it handles mixed feature types without scaling, provides native feature importance scores, and is interpretable enough to justify decisions in a risk context. `class_weight="balanced"` was set explicitly to prevent the model from ignoring the minority fraud class due to imbalance.

**Evaluation** — The model is assessed on a held-out 20% test set using Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC. Recall on the fraud class was prioritized over Precision — in fraud detection, missing a real fraud (false negative) is more costly than raising a false alarm.

---

## Results

| Metric | Legit | Fraud |
|---|---|---|
| Precision | 1.00 | 0.91 |
| Recall | 0.99 | 1.00 |
| F1-Score | 1.00 | 0.95 |

ROC-AUC: 1.0000

The model detected 100% of fraudulent transactions in the test set, with a single false positive. The near-perfect ROC-AUC reflects the synthetic nature of the data — the injected patterns are structurally distinct enough that the model learns them without difficulty. In a production environment with real transaction data, scores would be lower due to noise, concept drift, and overlapping behavior between fraud and legitimate activity. Feature importance analysis confirmed that `amount` was the dominant signal, followed by `country` and `hour` — consistent with the patterns that were designed into the data.

---

## Stack

- Python 3.14
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (RandomForestClassifier)
- Jupyter Notebook

---

## References

Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. https://doi.org/10.1023/A:1010933404324 — full paper included in `/references`

---

## Author

Lautaro Ramos
GitHub: https://github.com/lautaro22ramos-wq
LinkedIn: https://www.linkedin.com/in/lr22/
