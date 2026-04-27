# Phase 2 — Tuned Models, SHAP, Ethics Audit

## Goal

Build real, tuned models and evaluate them head-to-head. Explain predictions with SHAP and audit for bias.

## What was done

- **Tuned Logistic Regression** via `GridSearchCV` over regularization strength `C`. Used `TimeSeriesSplit` for cross-validation so validation folds always come after training folds (no leakage).
- **Tuned Random Forest** via grid search over `n_estimators`, `max_depth`, `min_samples_leaf`.
- **Head-to-head evaluation** of both models — ROC-AUC, F1, precision, recall, and top-24 recall per test season.
- **SHAP feature attribution** — explains why the Random Forest predicts what it predicts. Outputs a beeswarm summary plot and mean |SHAP| bar chart.
- **Ethics audit** — checked whether predictions are biased by position, conference (East/West), or market size.

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9695 | 0.5596 | 0.400 | 0.931 |
| Random Forest | 0.9684 | 0.5776 | 0.443 | 0.830 |

- **Top-24 global recall: ~60% (LR) / 59% (RF)**
- Ethics audit showed small deltas (<0.01) across position, conference, and market size — model does not appear systematically biased by those factors.

## Key finding from SHAP

`pts_per_game`, `mp_per_game`, and `ws` are the top features. Efficiency stats (`fg_percent`, `ft_percent`) are near the bottom.

## Outputs

| File | What it shows |
|---|---|
| `roc_comparison.png` | Side-by-side ROC curves for LR and RF |
| `confusion_comparison.png` | Side-by-side confusion matrices with F1 scores |
| `top24_recall.png` | Per-season top-24 recall bar chart |
| `shap_summary.png` | SHAP beeswarm showing feature impact distribution |
| `shap_importance.png` | Mean |SHAP| bar chart |

## Run

```bash
../venv/bin/python phase2.py
```
