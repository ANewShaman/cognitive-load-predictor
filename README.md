# Cognitive Load Predictor

Predicts a developer's cognitive load score (0–100) from behavioral and lifestyle signals using machine learning.
Try it here: https://anewshaman.github.io/cognitive-load-predictor/

---

## What it does

Given observable developer metrics — stress level, sleep hours, bugs found/fixed, hours coding, coffee intake, AI usage — the model predicts how mentally overloaded a developer is at any given time.

The prediction maps to four risk zones:

| Zone | Score | Avg Task Success Rate |
|------|-------|----------------------|
| Low | 0–30 | 87% |
| Medium | 30–50 | 72% |
| High | 50–70 | 51% |
| Critical | 70+ | 34% |

---

## Dataset

**AI Developer Performance Dataset** — 1000 developer records  
Source: [kaggle.com/datasets/tahirmohd/ai-dataset](https://www.kaggle.com/datasets/tahirmohd/ai-dataset)

> **Note:** Correlation analysis revealed this dataset is synthetically generated. `Stress_Level` correlates at r=0.969 with `Cognitive_Load` while all behavioral metrics fall below r=0.06. The ML pipeline is correctly implemented and would generalise to richer real-world data.

---

## Key Findings

- **Best engineered feature:** `Cognitive_Strain = Stress_Level × (1 / Sleep_Hours)` — r=0.820
- **Best model:** Gradient Boosting (tuned via GridSearchCV)
- **Test R²:** 0.9271 | **CV R² (5-fold):** 0.9346 | **Overfit gap:** 0.007
- **97%** of predictions fall within ±10 points on a 0–100 scale
- Permutation importance confirmed `Stress_Level` as dominant signal

---

## Project Structure
```
cognitive-load-predictor/
├── cognitive_load_predictor.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Run
```bash
git clone https://github.com/YOUR_USERNAME/cognitive-load-predictor.git
cd cognitive-load-predictor

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Download `AI_Developer_Performance_Extended_1000.csv` from Kaggle and place it in the root folder, then open `cognitive_load_predictor.ipynb`.

---

## Models Compared

Linear Regression · Ridge · Lasso · Decision Tree · Random Forest · **Gradient Boosting** · XGBoost

---

## Limitations

- Synthetic dataset — model learns `Cognitive_Load ≈ f(Stress_Level)`
- No individual baseline — breaks down for neurodiverse users
- Real deployment needs physiological signals (EEG, HRV, GSR)
- Keystroke/sleep tracking requires explicit user consent

---

## Tech Stack

Python · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · scipy
